"""Quick quality check of our bot (NOT the benchmark; sim/bench is). Runs our bot (one seat) against 7 copies of
each zoo archetype, and against sampled realistic fields, and prints the relative score (points / 12.5) with a 95%
CI, plus our error counters. CPU-time enforcement (laptop noise excluded), TOP3 + TURBO unless told otherwise.

    .venv/bin/python bot/tools/quick_bench.py [--tables 300] [--slow-tables 80] [--versions 0,1,2]
        [--payouts TOP3] [--schedule TURBO] [--workers 8] [--json bot/out/quick_bench.json] [--only chart_tag,...]
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.runner import run                     # noqa: E402
from sim.zoo import fields as F                    # noqa: E402

SLOW = set()      # near-limit archetypes get CPU-time enforcement; none are published


def mixed_opponent(seat, config, mix_name):
    """A seat filled per table by an archetype drawn from the mix (deterministic from the table's agent seed, so
    every version faces the same opponents at the same table)."""
    mix = F.MIXES[mix_name]
    keys = sorted(mix)
    rng = random.Random(config["seed"] * 7 + 12345)
    arch = rng.choices(keys, [mix[k] for k in keys], k=1)[0]
    return F.factory(arch)(seat, config)


def field_entrants(ours_spec, mix_name):
    """Our bot + 7 seats each drawn from the mix per table (our bot plays every table)."""
    import functools
    return [("ours", ours_spec)] + [("mix", functools.partial(mixed_opponent, mix_name=mix_name))] * 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=300)
    ap.add_argument("--slow-tables", type=int, default=80)
    ap.add_argument("--versions", default="0,1,2")
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default="")
    ap.add_argument("--fields", default="T1_like,T6_like,STRONG")
    ap.add_argument("--field-tables", type=int, default=200)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    versions = [int(x) for x in a.versions.split(",") if x != ""]
    archs = [k for k in F.ARCHETYPES if not a.only or k in a.only.split(",")]
    out = {"meta": {"payouts": a.payouts, "schedule": a.schedule, "timing": "cpu", "seed": a.seed,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}, "h2h": {}, "fields": {}}
    print(f"payouts {a.payouts} schedule {a.schedule}; rel = our points / 12.5 [95% CI]; err = exc/illegal/timeouts")
    for arch in archs:
        n = a.slow_tables if arch in SLOW else a.tables
        row = []
        for ver in versions:
            ents = [("ours", f"bot/shell.py:BotV{ver}")] + [(arch, F.factory(arch))] * 7
            t0 = time.time()
            res = run(ents, n, seed=a.seed, payouts=a.payouts, schedule=a.schedule, workers=a.workers,
                      timing="cpu")
            e = res.entrants[0]
            r = e.rel()
            s = e.stats.summary()
            opp = [x for x in res.entrants[1:]]
            opp_pts = sum(sum(x.points) for x in opp) / max(1, sum(len(x.points) for x in opp))
            out["h2h"].setdefault(arch, {})[f"v{ver}"] = {"rel": r, "n": n, "errors": s["errors"],
                                                         "exceptions": s["exceptions"], "illegal": s["illegal"],
                                                         "timeouts": s["timeouts"], "opp_mean_points": opp_pts,
                                                         "secs": time.time() - t0}
            row.append(f"v{ver} {r['rel']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] err {s['exceptions']}/{s['illegal']}/"
                       f"{s['timeouts']}")
        print(f"{arch:<16} n={n:<4} " + " | ".join(row), flush=True)
    for fname in [x for x in a.fields.split(",") if x]:
        row = []
        for ver in versions:
            ents = field_entrants(f"bot/shell.py:BotV{ver}", fname)
            res = run(ents, a.field_tables, seed=a.seed, payouts=a.payouts, schedule=a.schedule,
                      workers=a.workers, timing="cpu", seating="event")
            e = res.entrants[0]
            r = e.rel()
            s = e.stats.summary()
            out["fields"].setdefault(fname, {})[f"v{ver}"] = {"rel": r, "n": len(e.points), "errors": s["errors"]}
            row.append(f"v{ver} {r['rel']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] err {s['errors']}")
        print(f"field {fname:<10} " + " | ".join(row), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
