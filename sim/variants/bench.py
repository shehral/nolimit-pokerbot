"""Throughput of the variants versus the plain kit, example agents x8.

    .venv/bin/python -m sim.variants.bench [--quick]      writes sim/variants/out/speed.json
Slowdown = variant us per hand / plain-kit us per hand (table lengths differ, so tables/s alone misleads), same agent class
(VariantExample) and worker count.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from sim.kit.runner import run as kit_run

from .agents import EXAMPLES
from .runner import run_variant
from .spec import NAMES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args(argv)
    rows = []
    for workers, n in ((1, 100 if a.quick else 300), (a.workers, 600 if a.quick else 3000)):
        spec = ("ex", "sim.variants.agents:VariantExample")
        t0 = time.perf_counter()
        kr = kit_run([spec] * 8, n, seed=5, workers=workers)
        base = n / (time.perf_counter() - t0)
        rows.append({"variant": "kit (plain Table)", "workers": workers, "tables": n, "tables_per_s": round(base, 1),
                     "slowdown_vs_kit": 1.0, "hands_per_table": round(kr.meta["mean_hands_per_table"], 1),
                     "us_per_hand_per_worker": round(workers / (base * kr.meta["mean_hands_per_table"]) * 1e6, 1)})
        base_us = rows[-1]["us_per_hand_per_worker"]
        print(rows[-1], flush=True)
        for v in NAMES:
            ex = f"sim.variants.agents:{EXAMPLES[v].__name__}"
            r = run_variant(v, [("ex", ex)] * 8, n, seed=5, workers=workers)
            s = r.stats["ex"]
            rows.append({"variant": v, "workers": workers, "tables": n, "tables_per_s": round(r.tables_per_s, 1),
                         "slowdown_vs_kit": round(r.elapsed * workers / r.hands * 1e6 / base_us, 2),
                         "hands_per_table": round(r.hands / n, 1),
                         "errors": s.exceptions + s.illegal, "timeouts": s.timeouts, "timeouts_preempted": s.preempted,
                         "timeouts_cpu_over": s.cpu_over,
                         "us_per_hand_per_worker": round(r.elapsed * workers / r.hands * 1e6, 1)})
            print(rows[-1], flush=True)
    out = {"measured": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "cpu_count": os.cpu_count(),
           "python": sys.version.split()[0], "loadavg": os.getloadavg(), "rows": rows}
    p = Path(__file__).resolve().parent / "out" / "speed.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
