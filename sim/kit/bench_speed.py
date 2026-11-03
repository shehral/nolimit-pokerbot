"""Measure kit throughput (tables per second) with cheap bots, single core and multiprocess.

    .venv/bin/python -m sim.kit.bench_speed            # default sizes, writes sim/kit/out/speed.json
    .venv/bin/python -m sim.kit.bench_speed --quick

Fields (all cheap, microsecond-scale bots): example x8, check/call station x8 (the slow case: every table runs
200 hands with ~4,900 decisions), passive-random x8, random x8 (fast: all-ins end tables in ~15 hands), and a
mixed field. Numbers include pool start-up (spawn) and result merging; the per-decision figure is wall time per
decision for the whole run (engine + bots) divided by decisions per worker.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.kit.runner import default_workers, run  # noqa: E402

FIELDS = {
    "example_x8": [("example", "sim.kit.example_agent:ExampleAgent")] * 8,
    "station_x8": [("station", "sim.kit.fixtures:CheckCallAgent")] * 8,
    "passive_x8": [("passive", "sim.kit.fixtures:PassiveRandomAgent")] * 8,
    "random_x8": [("random", "sim.kit.fixtures:RandomAgent")] * 8,
    "mixed": [("example", "sim.kit.example_agent:ExampleAgent"), ("station", "sim.kit.fixtures:CheckCallAgent"),
              ("passive", "sim.kit.fixtures:PassiveRandomAgent"), ("random", "sim.kit.fixtures:RandomAgent"),
              ("allin", "sim.kit.fixtures:AllInAgent"), ("checkfold", "sim.kit.fixtures:CheckFoldAgent"),
              ("example", "sim.kit.example_agent:ExampleAgent"), ("passive", "sim.kit.fixtures:PassiveRandomAgent")],
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "speed.json"))
    a = ap.parse_args(argv)
    w = a.workers or default_workers()
    rows = []
    for name, ents in FIELDS.items():
        fast = name == "random_x8"
        n1 = (300 if fast else 60) if a.quick else (2000 if fast else 200)
        nw = (3000 if fast else 600) if a.quick else (30000 if fast else 4000)
        for workers, n in ((1, n1), (w, nw)):
            res = run(ents, n, seed=123, workers=workers)
            m = res.meta
            dec = sum(e.stats.decisions for e in res.entrants)
            row = {"field": name, "workers": workers, "tables": n, "elapsed_s": round(m["elapsed_s"], 3),
                   "tables_per_s": round(m["tables_per_s"], 1), "hands_per_table": round(m["mean_hands_per_table"], 1),
                   "decisions_per_table": round(dec / n, 1),
                   "us_per_decision_per_worker": round(m["elapsed_s"] * workers / dec * 1e6, 2) if dec else None,
                   "errors": sum(e.stats.timeouts + e.stats.exceptions + e.stats.illegal for e in res.entrants),
                   "timeouts": sum(e.stats.timeouts for e in res.entrants),
                   "timeouts_cpu_over": sum(e.stats.cpu_over for e in res.entrants),
                   "timeouts_preempted": sum(e.stats.preempted for e in res.entrants),
                   "loadavg_start": m["loadavg_start"][0], "loadavg_end": m["loadavg_end"][0],
                   "act_p99_9_us_max_over_entrants": max(e.stats.latency_us(0.999) for e in res.entrants),
                   "act_max_us": max(e.stats.lat_max_ns for e in res.entrants) / 1000.0,
                   "act_cpu_max_us": max(e.stats.cpu_max_ns for e in res.entrants) / 1000.0}
            rows.append(row)
            print(f"{name:12s} workers={workers:2d} tables={n:6d} {row['tables_per_s']:8.1f} tables/s  "
                  f"{row['hands_per_table']:6.1f} hands/t  {row['decisions_per_table']:7.1f} dec/t  "
                  f"{row['us_per_decision_per_worker']} us/dec/worker  errors={row['errors']} "
                  f"(t/o {row['timeouts']}: cpu {row['timeouts_cpu_over']}, os-noise {row['timeouts_preempted']}) "
                  f"act p99.9 {row['act_p99_9_us_max_over_entrants']:.1f} us max {row['act_max_us']:.0f} us "
                  f"(cpu max {row['act_cpu_max_us']:.0f} us) load {row['loadavg_start']:.1f}->{row['loadavg_end']:.1f}",
                  flush=True)
    out = {"measured": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "cpu_count": os.cpu_count(), "workers": w,
           "python": sys.version.split()[0], "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
