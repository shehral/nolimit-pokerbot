"""Latency measurement of our bot's act() (SPEC s3: >= 200,000 decisions; target p99.9 <= 250 us on this Mac).

Single process (no worker pool), real kit tables: our bot in 4 seats, 4 opponents. Default field "mixed": cheap
zoo/kit opponents (chart_tag, icm_pro, pushfold_maniac, station) so the spots are varied (jams, calls, multiway
postflop); --field station / icm_pro (4 copies of one archetype) or T6_like / STRONG / T1_like (4 seats drawn per
table from that zoo mix, deterministic by table index). Wall clock (time.perf_counter_ns, what the rules limit) and
thread CPU time per decision come from the kit engine's 0.1-us histograms. The bot's own shell does gc.collect() +
gc.freeze() at import; the collector stays ON (the default production setting).

--rusage wraps our act() in getrusage() calls (outside our own thread-CPU window, inside the kit's, so the kit's
numbers rise by about 1-2 us) and splits our decisions into clean ones and ones with an involuntary context switch
(preempted: the OS took the core mid-decision; CPU time then includes cache refill and kernel time). "cpu max clean"
is the bot's own worst case; the overall max also carries the machine's load.

    .venv/bin/python bot/tests/latency.py [--decisions 200000] [--versions 0,1,2] [--gc default|disable]
                                          [--field mixed] [--params "GUARD_NS=200000"] [--rusage]
                                          [--json bot/out/latency.json]
"""
import argparse
import ast
import gc
import json
import os
import random
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import SeatStats, Table       # noqa: E402
from sim.kit.runner import load_factory           # noqa: E402


def opponents(field="mixed"):
    """field -> function(table index) -> 4 opponent factories."""
    try:
        from sim.zoo import fields as F
    except Exception:                               # zoo missing: kit fixtures
        from sim.kit.example_agent import ExampleAgent
        from sim.kit.fixtures import CheckCallAgent, PassiveRandomAgent, RandomAgent
        return lambda t: [ExampleAgent, CheckCallAgent, PassiveRandomAgent, RandomAgent]
    if field == "mixed":
        opp = [F.factory("chart_tag"), F.factory("icm_pro"), F.factory("pushfold_maniac"), F.factory("station")]
        return lambda t: opp
    if field in F.ARCHETYPES:
        opp = [F.factory(field)] * 4
        return lambda t: opp
    if field in F.MIXES:
        mix = F.MIXES[field]
        keys = sorted(mix)

        def draw(t):
            rng = random.Random(f"lat-field:{field}:{t}")
            return [F.factory(k) for k in rng.choices(keys, [mix[k] for k in keys], k=4)]
        return draw
    raise SystemExit(f"unknown field {field!r}")


def parse_params(text):
    out = {}
    for item in [x for x in (text or "").split(",") if x.strip()]:
        k, _, v = item.partition("=")
        out[k.strip()] = ast.literal_eval(v.strip())
    return out


def measure(version, decisions, payouts="TOP3", schedule="TURBO", seed=0, gc_mode="default", field="mixed",
            params=None, rusage=False):
    fac = load_factory(f"bot/shell.py:BotV{version}")
    shell = sys.modules[fac.__module__]
    draw = opponents(field)
    ours = SeatStats()
    bots = []
    TT = time.thread_time_ns
    split = {"clean": 0, "preempted": 0, "cpu_max_clean_us": 0.0, "cpu_max_preempted_us": 0.0,
             "over_200_clean": 0, "over_200_preempted": 0, "sys_max_us": 0.0}
    gru = resource.getrusage
    SELF = resource.RUSAGE_SELF

    def make(seat, cfg):
        if params:
            cfg = dict(cfg, bot_params=dict(params))
        b = fac(seat, cfg)
        bots.append(b)
        if rusage:
            orig = b.act

            def act(state, _o=orig):
                # the wrapper's own allocations (rusage structs) must not trigger a collection inside the kit's
                # window after our act() re-enabled the collector (that would be a measurement artifact)
                g = gc.isenabled()
                if g:
                    gc.disable()
                r0 = gru(SELF)
                c0 = TT()
                out = _o(state)
                c = (TT() - c0) / 1e3
                r1 = gru(SELF)
                if g:
                    gc.enable()
                if r1.ru_nivcsw == r0.ru_nivcsw:
                    split["clean"] += 1
                    if c > split["cpu_max_clean_us"]:
                        split["cpu_max_clean_us"] = c
                    if c > 200.0:
                        split["over_200_clean"] += 1
                else:
                    split["preempted"] += 1
                    if c > split["cpu_max_preempted_us"]:
                        split["cpu_max_preempted_us"] = c
                    if c > 200.0:
                        split["over_200_preempted"] += 1
                ds = (r1.ru_stime - r0.ru_stime) * 1e6
                if ds > split["sys_max_us"]:
                    split["sys_max_us"] = ds
                return out
            b.act = act
        return b
    was = gc.isenabled()
    if gc_mode == "disable":
        gc.disable()
    t_start = time.perf_counter()
    load0 = os.getloadavg()
    run_started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    tables = 0
    guard = errors = illegal_fixed = bf = 0
    try:
        while ours.decisions < decisions:
            opp = draw(tables)
            facs = [make, opp[0], make, opp[1], make, opp[2], make, opp[3]]
            rot = tables % 8
            facs = facs[rot:] + facs[:rot]
            bots.clear()
            res = Table(facs, payouts, schedule, seed=f"lat:{seed}:{tables}", timing="wall",
                        time_limit_ms=None).play()
            for s in range(8):
                if facs[s] is make:
                    ours.merge(res.stats[s])
            for b in bots:
                errors += b.errors
                illegal_fixed += b.illegal_fixed
                if b._brain is not None:
                    guard += b._brain.guard_trips
                    bf += b._brain.bf_computed
            tables += 1
    finally:
        if gc_mode == "disable" and was:
            gc.enable()
    s = ours.summary()
    over_250_wall = sum(c for b, c in ours.lat_hist.items() if b >= 2500)      # 0.1-us bins
    over_250_cpu = sum(c for b, c in ours.cpu_hist.items() if b >= 2500)
    over_200_cpu = sum(c for b, c in ours.cpu_hist.items() if b >= 2000)
    over_1ms_wall = sum(c for b, c in ours.lat_hist.items() if b > 10000)
    over_1ms_cpu = sum(c for b, c in ours.cpu_hist.items() if b > 10000)
    return {
        "version": version, "decisions": s["decisions"], "tables": tables, "payouts": payouts, "schedule": schedule,
        "field": field, "params": dict(params or {}), "started": run_started,
        "gc_mode": gc_mode, "wall_us": s["latency_us"], "cpu_us": s["cpu_us"],
        "over_1ms": {"wall": over_1ms_wall, "cpu": over_1ms_cpu,
                     "wall_minus_cpu": max(0, over_1ms_wall - over_1ms_cpu)},
        "over_250us": {"wall": over_250_wall, "cpu": over_250_cpu}, "over_200us_cpu": over_200_cpu,
        "rusage_split": split if rusage else None,
        "shell_exceptions": errors, "shell_illegal_fixed": illegal_fixed, "guard_trips": guard,
        "bf_computed": bf, "kit_exceptions": s["exceptions"], "kit_illegal": s["illegal"],
        "hist_wall": s["latency_hist_us"], "elapsed_s": time.perf_counter() - t_start,
        "loadavg_start": list(load0), "loadavg_end": list(os.getloadavg()), "python": sys.version.split()[0],
        "warmup": dict(shell.WARMUP_INFO),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=200_000)
    ap.add_argument("--versions", default="0,1,2")
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--gc", default="default", choices=["default", "disable"])
    ap.add_argument("--field", default="mixed")
    ap.add_argument("--params", default="", help='PARAMS overrides for our bot, e.g. "GUARD_NS=200000"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rusage", action="store_true", help="split decisions by involuntary context switches")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    params = parse_params(a.params)
    out = []
    for v in [int(x) for x in a.versions.split(",")]:
        r = measure(v, a.decisions, a.payouts, a.schedule, seed=a.seed, gc_mode=a.gc, field=a.field, params=params,
                    rusage=a.rusage)
        w, c = r["wall_us"], r["cpu_us"]
        line = (f"{r['started']} py{r['python']} v{v} field {a.field} {a.payouts} {a.schedule} "
                f"params {params or '-'}: {r['decisions']:,} decisions in {r['tables']} tables "
                f"({r['elapsed_s']:.1f} s), gc {a.gc}: "
                f"wall p50 {w['p50']:.1f} p99 {w['p99']:.1f} p99.9 {w['p99.9']:.1f} max {w['max']:.1f} us | "
                f"cpu p50 {c['p50']:.1f} p99 {c['p99']:.1f} p99.9 {c['p99.9']:.1f} max {c['max']:.1f} | "
                f">200us cpu {r['over_200us_cpu']} | >250us wall {r['over_250us']['wall']} cpu {r['over_250us']['cpu']}"
                f" | >1ms {r['over_1ms']} | guard trips {r['guard_trips']} of {r['bf_computed']} BFs | shell exc "
                f"{r['shell_exceptions']} illegal-fixed {r['shell_illegal_fixed']} | "
                f"load {r['loadavg_start'][0]:.1f}->{r['loadavg_end'][0]:.1f}")
        sp = r["rusage_split"]
        if sp:
            line += (f" | clean {sp['clean']:,} max {sp['cpu_max_clean_us']:.1f} (>200: {sp['over_200_clean']}), "
                     f"preempted {sp['preempted']:,} max {sp['cpu_max_preempted_us']:.1f} "
                     f"(>200: {sp['over_200_preempted']}), sys max {sp['sys_max_us']:.0f}")
        print(line, flush=True)
        out.append(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
