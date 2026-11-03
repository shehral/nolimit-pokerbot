"""Latency measurement of our bot's act() (SPEC s3: >= 200,000 decisions; target p99.9 <= 250 us on this Mac).

Single process (no worker pool), real kit tables: our bot in 4 seats, 4 cheap zoo/kit opponents (chart_tag, icm_pro,
pushfold_maniac, station) so the spots are varied (jams, calls, multiway postflop). Wall clock (time.perf_counter_ns,
what the rules limit) and thread CPU time per decision come from the kit engine's 0.1-us histograms. The bot's own
shell does gc.collect() + gc.freeze() at import; the collector stays ON (the default production setting).

    .venv/bin/python bot/tests/latency.py [--decisions 200000] [--versions 0,1,2] [--gc default|disable]
                                          [--json bot/out/latency.json]
"""
import argparse
import gc
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import SeatStats, Table       # noqa: E402
from sim.kit.runner import load_factory           # noqa: E402


def opponents():
    try:
        from sim.zoo import fields as F
        return [F.factory("chart_tag"), F.factory("icm_pro"), F.factory("pushfold_maniac"), F.factory("station")]
    except Exception:                               # zoo missing: kit fixtures
        from sim.kit.example_agent import ExampleAgent
        from sim.kit.fixtures import CheckCallAgent, PassiveRandomAgent, RandomAgent
        return [ExampleAgent, CheckCallAgent, PassiveRandomAgent, RandomAgent]


def measure(version, decisions, payouts="TOP3", schedule="TURBO", seed=0, gc_mode="default"):
    fac = load_factory(f"bot/shell.py:BotV{version}")
    shell = sys.modules[fac.__module__]
    opp = opponents()
    ours = SeatStats()
    bots = []

    def make(seat, cfg):
        b = fac(seat, cfg)
        bots.append(b)
        return b
    was = gc.isenabled()
    if gc_mode == "disable":
        gc.disable()
    t_start = time.perf_counter()
    load0 = os.getloadavg()
    tables = 0
    guard = errors = illegal_fixed = bf = 0
    try:
        while ours.decisions < decisions:
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
    over_1ms_wall = sum(c for b, c in ours.lat_hist.items() if b > 10000)
    over_1ms_cpu = sum(c for b, c in ours.cpu_hist.items() if b > 10000)
    return {
        "version": version, "decisions": s["decisions"], "tables": tables, "payouts": payouts, "schedule": schedule,
        "gc_mode": gc_mode, "wall_us": s["latency_us"], "cpu_us": s["cpu_us"],
        "over_1ms": {"wall": over_1ms_wall, "cpu": over_1ms_cpu,
                     "wall_minus_cpu": max(0, over_1ms_wall - over_1ms_cpu)},
        "over_250us": {"wall": over_250_wall, "cpu": over_250_cpu},
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
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = []
    for v in [int(x) for x in a.versions.split(",")]:
        r = measure(v, a.decisions, a.payouts, a.schedule, gc_mode=a.gc)
        w, c = r["wall_us"], r["cpu_us"]
        print(f"v{v}: {r['decisions']:,} decisions in {r['tables']} tables ({r['elapsed_s']:.1f} s), gc {a.gc}: "
              f"wall p50 {w['p50']:.1f} p99 {w['p99']:.1f} p99.9 {w['p99.9']:.1f} max {w['max']:.1f} us | "
              f"cpu p50 {c['p50']:.1f} p99.9 {c['p99.9']:.1f} max {c['max']:.1f} | >250us wall "
              f"{r['over_250us']['wall']} cpu {r['over_250us']['cpu']} | >1ms {r['over_1ms']} | guard trips "
              f"{r['guard_trips']} | shell exc {r['shell_exceptions']} illegal-fixed {r['shell_illegal_fixed']} | "
              f"load {r['loadavg_start'][0]:.1f}->{r['loadavg_end'][0]:.1f}", flush=True)
        out.append(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
