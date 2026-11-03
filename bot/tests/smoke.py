"""Smoke run of our bot (SPEC s3): many tables, every error type counted, nothing masked.

STRICT mode (NOLIMIT_BOT_STRICT=1, set below before the bot is imported): the shell lets exceptions and illegal
results through to the engine, so the kit's own counters show every error the shell would otherwise hide.
Enforcement is on thread CPU time (timing="cpu"), so a decision counts as a timeout only when the bot itself used
more than 1 ms; the kit also records wall-clock overruns and how many of them were preemption (wall over, CPU under)
on this shared laptop. Payout/schedule rotate in four equal parts: TOP3/TURBO, LINEAR/FAST, WTA/SLOW, TOP4/TURBO.

Configurations (our decisions are counted over every seat we hold):
  example     each of v0, v1, v2 in one seat + 7 kit example agents
  self        8 copies of v2; and a mixed table of 3 v0 + 3 v1 + 2 v2
  fields      v2 + 7 seats drawn per table from the zoo mixes T1_like, T6_like, STRONG
  archetypes  v2 + 7 copies of each zoo archetype

    .venv/bin/python bot/tests/smoke.py [--tables 2000] [--arch-tables 200] [--workers 8] [--json bot/out/smoke.json]
"""
import argparse
import functools
import json
import os
import sys
import time

os.environ["NOLIMIT_BOT_STRICT"] = "1"           # must precede any import of the bot
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.runner import run                   # noqa: E402
from bot.tools.ab import mixed_opponent          # noqa: E402  (package path: picklable for the worker pool)

PARTS = (("TOP3", "TURBO"), ("LINEAR", "FAST"), ("WTA", "SLOW"), ("TOP4", "TURBO"))
OURS = {f"v{v}": f"bot/shell.py:BotV{v}" for v in (0, 1, 2)}
EXAMPLE = "sim/kit/example_agent.py:ExampleAgent"


def configs(arch_tables, tables):
    from sim.zoo import fields as F
    out = []
    for v in ("v0", "v1", "v2"):
        out.append((f"example:{v}", [(v, OURS[v])] + [("example", EXAMPLE)] * 7, tables))
    out.append(("self:v2x8", [("v2", OURS["v2"])] * 8, tables))
    out.append(("self:mixed", [("v0", OURS["v0"])] * 3 + [("v1", OURS["v1"])] * 3 + [("v2", OURS["v2"])] * 2,
                tables))
    for mix in ("T1_like", "T6_like", "STRONG"):
        out.append((f"field:{mix}", [("v2", OURS["v2"])] +
                    [(mix, functools.partial(mixed_opponent, mix_name=mix))] * 7, tables))
    for arch in F.ARCHETYPES:
        out.append((f"arch:{arch}", [("v2", OURS["v2"])] + [(arch, F.factory(arch))] * 7, arch_tables))
    return out


def ours_names(entrants):
    return {name for name, _ in entrants if name in OURS}


def run_config(label, entrants, n_tables, workers, seed):
    mine = ours_names(entrants)
    agg = {"label": label, "tables": 0, "decisions": 0, "exceptions": 0, "illegal": 0, "timeouts": 0,
           "wall_over_1ms": 0, "cpu_over_1ms": 0, "preempted": 0, "observe_exceptions": 0, "init_failures": 0,
           "wall_max_us": 0.0, "cpu_max_us": 0.0, "examples": {}, "parts": []}
    per = n_tables // len(PARTS)
    for i, (pay, sched) in enumerate(PARTS):
        n = per if i < len(PARTS) - 1 else n_tables - per * (len(PARTS) - 1)
        res = run(entrants, n, seed=seed + i, payouts=pay, schedule=sched, workers=workers, timing="cpu")
        agg["tables"] += n
        part = {"payouts": pay, "schedule": sched, "tables": n}
        for e in res.entrants:
            if e.name not in mine:
                continue
            s = e.stats.summary()
            agg["decisions"] += s["decisions"]
            agg["exceptions"] += s["exceptions"]
            agg["illegal"] += s["illegal"]
            agg["timeouts"] += s["timeouts"]
            ol = s["over_limit"]
            agg["wall_over_1ms"] += ol.get("wall", 0)
            agg["cpu_over_1ms"] += ol.get("cpu", 0)
            agg["preempted"] += ol.get("preempted", 0)
            agg["observe_exceptions"] += s["observe"]["exceptions"]
            agg["init_failures"] += s["init"].get("exceptions", 0)
            agg["wall_max_us"] = max(agg["wall_max_us"], s["latency_us"]["max"])
            agg["cpu_max_us"] = max(agg["cpu_max_us"], s["cpu_us"]["max"])
            for k, ex in (s.get("examples") or {}).items():
                agg["examples"].setdefault(k, []).extend(ex[:2])
        agg["parts"].append(part)
    agg["clean"] = (agg["exceptions"] == 0 and agg["illegal"] == 0 and agg["timeouts"] == 0
                    and agg["observe_exceptions"] == 0 and agg["init_failures"] == 0)
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=2000)
    ap.add_argument("--arch-tables", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1106)
    ap.add_argument("--only", default="", help="comma-separated label prefixes, e.g. example,self")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    t_start = time.time()
    load0 = os.getloadavg()
    rows = []
    for label, ents, n in configs(a.arch_tables, a.tables):
        if a.only and not any(label.startswith(p) for p in a.only.split(",")):
            continue
        t0 = time.time()
        r = run_config(label, ents, n, a.workers, a.seed)
        r["secs"] = time.time() - t0
        rows.append(r)
        print(f"{label:<22} tables {r['tables']:>5}  our decisions {r['decisions']:>8,}  exc {r['exceptions']}  "
              f"illegal {r['illegal']}  timeouts(cpu) {r['timeouts']}  | wall>1ms {r['wall_over_1ms']} "
              f"(preempted {r['preempted']})  cpu>1ms {r['cpu_over_1ms']}  cpu max {r['cpu_max_us']:.0f} us  "
              f"| observe exc {r['observe_exceptions']}  {'CLEAN' if r['clean'] else 'NOT CLEAN'}  "
              f"({r['secs']:.0f} s)", flush=True)
        if not r["clean"]:
            for k, ex in r["examples"].items():
                for x in ex[:3]:
                    print(f"     {k}: {x}")
    tot = {k: sum(r[k] for r in rows) for k in ("tables", "decisions", "exceptions", "illegal", "timeouts",
                                                "wall_over_1ms", "preempted", "cpu_over_1ms",
                                                "observe_exceptions")}
    print(f"TOTAL tables {tot['tables']:,}  our decisions {tot['decisions']:,}  exceptions {tot['exceptions']}  "
          f"illegal {tot['illegal']}  timeouts(cpu) {tot['timeouts']}  wall>1ms {tot['wall_over_1ms']} "
          f"(preempted {tot['preempted']})  cpu>1ms {tot['cpu_over_1ms']}  observe exc {tot['observe_exceptions']}"
          f"  | {time.time() - t_start:.0f} s, load {load0[0]:.1f} -> {os.getloadavg()[0]:.1f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"strict": True, "timing": "cpu", "workers": a.workers, "seed": a.seed, "parts": PARTS,
                       "loadavg_start": load0, "loadavg_end": os.getloadavg(), "total": tot, "rows": rows,
                       "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))}, f, indent=1)
    return 0 if all(r["clean"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
