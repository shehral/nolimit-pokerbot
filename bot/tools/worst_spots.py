"""Where do our slowest decisions spend their CPU? One process, our bot in 4 of 8 seats, every act() timed in thread
CPU time with a per-component breakdown (normalize, tracker sync, exact bubble factors, the rest); the N slowest
decisions are printed with their spot (street, players live, BFs started, guard trips) and saved as JSON.

Instrumentation is monkeypatched onto the bot's modules from outside (timers add about 0.2 us per wrapped call);
nothing in bot/ changes. Fields: "mixed" (chart_tag, icm_pro, pushfold_maniac, station: bot/tests/latency.py's),
an archetype key (4 copies), or a zoo mix name (seats drawn per table).

    .venv/bin/python bot/tools/worst_spots.py --field station --version 2 --decisions 200000 [--top 20]
                                              [--params "GUARD_NS=150000"] [--json bot/out/worst_station.json]
"""
import argparse
import ast
import heapq
import json
import os
import random
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import Table                  # noqa: E402

TT = time.thread_time_ns


def opponents(field, tables_seed):
    from sim.zoo import fields as F
    if field == "mixed":
        return lambda t: [F.factory("chart_tag"), F.factory("icm_pro"), F.factory("pushfold_maniac"),
                          F.factory("station")]
    if field in F.ARCHETYPES:
        return lambda t: [F.factory(field)] * 4
    if field in F.MIXES:
        mix = F.MIXES[field]
        keys = sorted(mix)

        def draw(t):
            rng = random.Random(f"worst:{tables_seed}:{field}:{t}")
            return [F.factory(k) for k in rng.choices(keys, [mix[k] for k in keys], k=4)]
        return draw
    raise SystemExit(f"unknown field {field!r}")


class Clock:
    """Accumulates thread-CPU per component for the decision in progress."""

    def __init__(self):
        self.cur = {}

    def wrap(self, owner, name, key):
        orig = getattr(owner, name)
        cur = self.cur

        def w(*a, **k):
            c0 = TT()
            try:
                return orig(*a, **k)
            finally:
                cur[key] = cur.get(key, 0) + TT() - c0
                cur[key + "_n"] = cur.get(key + "_n", 0) + 1
        setattr(owner, name, w)
        return orig


def parse_params(text):
    out = {}
    for item in [x for x in text.split(",") if x.strip()]:
        k, _, v = item.partition("=")
        out[k.strip()] = ast.literal_eval(v.strip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="mixed")
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--decisions", type=int, default=200_000)
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--params", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--deep", action="store_true", help="also time preflop/postflop/icm_jam/typed_jam/call_jam")
    a = ap.parse_args(argv)
    from bot import shell as SH
    B, AD = SH.brain, SH.adapter
    clock = Clock()
    clock.wrap(AD, "normalize", "normalize")
    clock.wrap(B.HandTracker, "sync_from_view", "sync")
    clock.wrap(B, "bubble_factor", "bf")                    # Brain.bf looks bubble_factor up as a module global
    clock.wrap(AD, "to_action", "to_action")
    clock.wrap(AD, "is_legal", "is_legal")
    if a.deep:                                             # nested timers (parts overlap: icm_jam includes bf)
        for name in ("preflop", "postflop", "icm_jam", "typed_jam", "call_jam", "open_range"):
            clock.wrap(B.Brain, name, name)
    params = parse_params(a.params)
    draw = opponents(a.field, a.seed)
    heap = []                                              # (cpu_ns, i, record), min-heap of the slowest
    hist = {}
    n_dec = 0
    seq = 0
    bots = []

    def make(seat, cfg):
        b = SH.make(seat, cfg, version=a.version, params=params)
        orig = b.act

        def act(state, _o=orig, _b=b):
            nonlocal n_dec, seq
            clock.cur.clear()
            br = _b._brain
            trips0 = br.guard_trips if br is not None else 0
            r0 = resource.getrusage(resource.RUSAGE_SELF)    # outside the timed window
            c0 = TT()
            w0 = time.perf_counter_ns()
            out = _o(state)
            c = TT() - c0
            w = time.perf_counter_ns() - w0
            r1 = resource.getrusage(resource.RUSAGE_SELF)
            n_dec += 1
            bucket = c // 10_000                           # 10-us bins
            hist[bucket] = hist.get(bucket, 0) + 1
            if len(heap) < a.top or c > heap[0][0]:
                cur = dict(clock.cur)
                rec = {"cpu_us": c / 1e3, "wall_us": w / 1e3, "street": state.get("street"),
                       "n_live": sum(1 for s in state.get("stacks", ()) if s > 0),
                       "in_hand": sum(1 for x in state.get("active", ()) if x),
                       "to_call": state.get("to_call"), "action": out.get("action"),
                       "guard_trips": (br.guard_trips - trips0) if br is not None else 0,
                       "parts_us": {k: (v / 1e3 if not k.endswith("_n") else v) for k, v in cur.items()},
                       "loadavg": os.getloadavg()[0], "hand_no": state.get("hand_no"),
                       # process-wide (one thread here): kernel time, page faults, involuntary context switches
                       "sys_us": round((r1.ru_stime - r0.ru_stime) * 1e6, 1),
                       "user_us": round((r1.ru_utime - r0.ru_utime) * 1e6, 1),
                       "minflt": r1.ru_minflt - r0.ru_minflt, "majflt": r1.ru_majflt - r0.ru_majflt,
                       "nivcsw": r1.ru_nivcsw - r0.ru_nivcsw}
                rec["parts_us"]["rest"] = round(c / 1e3 - sum(v / 1e3 for k, v in cur.items()
                                                              if k in ("normalize", "sync", "bf", "to_action",
                                                                       "is_legal")), 1)
                seq += 1
                item = (c, seq, rec)
                if len(heap) < a.top:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
            return out
        b.act = act
        bots.append(b)
        return b

    t_start = time.time()
    load0 = os.getloadavg()
    tables = 0
    while n_dec < a.decisions:
        opp = draw(tables)
        facs = [make, opp[0], make, opp[1], make, opp[2], make, opp[3]]
        rot = tables % 8
        facs = facs[rot:] + facs[:rot]
        Table(facs, a.payouts, a.schedule, seed=f"worst:{a.seed}:{tables}", timing="wall",
              time_limit_ms=None).play()
        tables += 1
    worst = [r for _, _, r in sorted(heap, reverse=True)]
    tot = sum(hist.values())
    over = {f">{t}us": sum(c for b, c in hist.items() if b * 10 >= t) for t in (100, 150, 200, 250, 500, 1000)}
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{stamp} field {a.field} v{a.version} {a.payouts} {a.schedule} params {params or '-'}: {tot:,} decisions, "
          f"{tables} tables, {time.time() - t_start:.0f} s, load {load0[0]:.1f}->{os.getloadavg()[0]:.1f}; "
          f"cpu over {over}")
    for r in worst:
        p = r["parts_us"]
        print(f"  cpu {r['cpu_us']:7.1f} wall {r['wall_us']:7.1f} {r['street']:<7} live {r['n_live']} in {r['in_hand']}"
              f" act {r['action']:<6} bf {p.get('bf_n', 0)} ({p.get('bf', 0):.1f}) norm {p.get('normalize', 0):.1f}"
              f" sync {p.get('sync', 0):.1f} rest {p['rest']:.1f} trips {r['guard_trips']} load {r['loadavg']:.1f}"
              f" | sys {r['sys_us']:.0f} flt {r['minflt']}/{r['majflt']} ivcsw {r['nivcsw']}"
              + ("" if not a.deep else " | " + " ".join(f"{k} {p[k]:.1f}" for k in
                                                        ("preflop", "icm_jam", "typed_jam", "call_jam", "postflop")
                                                        if k in p)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"stamp": stamp, "args": vars(a), "decisions": tot, "tables": tables, "over": over,
                       "hist_10us": {str(k * 10): v for k, v in sorted(hist.items())}, "worst": worst,
                       "loadavg_start": load0, "loadavg_end": os.getloadavg(), "python": sys.version.split()[0]},
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
