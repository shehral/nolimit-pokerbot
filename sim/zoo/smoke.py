"""Smoke test of the whole zoo with the kit runner: every archetype plays >= 300 recorded tables in MIXED tables.

All 12 archetypes are entered once with seating="event": each iteration shuffles them into 2 tables of 8 (4 seats are
random filler copies that play but are not recorded), so every archetype faces random mixes of the others. Wall-clock
enforcement (faithful to rules s3.7), 6 workers. Writes sim/zoo/out/smoke.json and prints a markdown table.

    .venv/bin/python -m sim.zoo.smoke [tables] [workers] [time_scale]
"""
import json
import os
import sys
import time

from sim.kit.runner import run
from sim.zoo import fields as F


def main(tables=800, workers=6, time_scale=1.0, seed=2026, out="sim/zoo/out/smoke.json"):
    ents = [(k, F.factory(k)) for k in F.ARCHETYPES]
    t0 = time.time()
    res = run(ents, n_tables=tables, seed=seed, workers=workers, timing="wall", time_scale=time_scale)
    rows = []
    for e in res.entrants:
        s = e.stats.summary()
        d = max(1, s["decisions"])
        rows.append({"archetype": e.name, "tables": s["tables"], "decisions": s["decisions"],
                     "timeout_rate": s["timeouts"] / d, "exception_rate": s["exceptions"] / d,
                     "illegal_rate": s["illegal"] / d, "timeouts": s["timeouts"], "exceptions": s["exceptions"],
                     "illegal": s["illegal"], "cpu_over": s["over_limit"]["cpu"], "preempted": s["over_limit"]["preempted"],
                     "p50_us": s["latency_us"]["p50"], "p99_us": s["latency_us"]["p99"],
                     "p99.9_us": s["latency_us"]["p99.9"], "max_us": s["latency_us"]["max"],
                     "cpu_p50_us": s["cpu_us"]["p50"], "mean_points": e.mean, "rel": e.rel()["rel"], "rel_lo": e.rel()["lo"], "rel_hi": e.rel()["hi"],
                     "vpip": s["vpip"], "pfr": s["pfr"]})
    meta = {"tables_run": tables, "seed": seed, "workers": workers, "time_scale": time_scale,
            "seconds": round(time.time() - t0, 1), "load_avg": list(os.getloadavg())}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(out, "w"), indent=1)
    print(f"{'archetype':16}{'tables':>7}{'decisions':>10}{'timeout':>9}{'exc':>8}{'illegal':>9}{'p50us':>9}{'p99us':>9}"
          f"{'cpu_over':>9}{'rel':>7}")
    for r in rows:
        print(f"{r['archetype']:16}{r['tables']:>7}{r['decisions']:>10}{r['timeout_rate']:>9.2%}{r['exception_rate']:>8.2%}"
              f"{r['illegal_rate']:>9.2%}{r['p50_us']:>9.1f}{r['p99_us']:>9.1f}{r['cpu_over']:>9}"
              f"{(r['rel'] or 0):>7.2f}")
    print(meta)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 800, int(a[1]) if len(a) > 1 else 6, float(a[2]) if len(a) > 2 else 1.0)
