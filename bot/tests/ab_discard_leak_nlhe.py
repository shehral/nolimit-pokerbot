"""Plain-NLHE inertness cells for the revealed-discard features (DISCARD_LEAK notes): sim/bench/common.run_cell
with v2 (shipped) vs v2:LEAK_PENALTY=True vs v2:USE_REVEALED=True on the same paired tables.

    .venv/bin/python bot/tests/ab_discard_leak_nlhe.py --fields AI_MIX,STRONG,T6_like --tables 5000 --seed 501

Expected: identical points on every table (no discard request and no discard event in hold'em) up to the noise
floor that clocks put in: our own BF guard (GUARD_NS) and the AI bots that stop their Monte Carlo on a clock make a
table replay differently for different occupants (two AI bots stop their Monte Carlo on a clock). --placebo adds an occupant with
the shipped config under another name (v2:LEAK_LAMBDA=20.0, the default) to measure that floor; --base GUARD_NS=
1000000000000000 removes our own clock (zoo fields are then exactly reproducible). Log lines carry the real clock
and the load average; JSON per field.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def log(msg, fh):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')} load {' '.join(f'{x:.2f}' for x in os.getloadavg())}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="AI_MIX,STRONG,T6_like")
    ap.add_argument("--tables", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=501)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default="", help="params added to every occupant, e.g. GUARD_NS=1000000000000000")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args(argv)
    from sim.bench.common import run_cell
    occ = ["v2", "v2:LEAK_PENALTY=True", "v2:USE_REVEALED=True"] + (["v2:LEAK_LAMBDA=20.0"] if a.placebo else [])
    if a.base:
        occ = [o + ("," if ":" in o else ":") + a.base for o in occ]
    tag = "_det" if a.base else ""
    fh = open(a.log, "a")
    for f in a.fields.split(","):
        log(f"NLHE cell {f}: {occ}, {a.tables} paired tables, seed {a.seed}, TOP3 TURBO, workers {a.workers}", fh)
        c = run_cell(f, occ, a.tables, seed=a.seed, workers=a.workers, verbose=False)
        pts = {o: c["occ"][o]["points"] for o in occ}
        for o in occ[1:]:
            same = sum(1 for x, y in zip(pts[o], pts[occ[0]]) if x == y)
            d = c["pairs"][f"{o} - {occ[0]}"]
            log(f"  {f} {o} - base: {d['diff_rel']:+.4f} [{d['lo_rel']:+.4f}, {d['hi_rel']:+.4f}] rel; "
                f"identical tables {same} of {len(pts[o])}; "
                f"rel base {c['occ'][occ[0]]['rel']['rel']:.4f}; shell {c['occ'][o]['shell']}", fh)
        with open(os.path.join(a.out, f"nlhe_{f}_seed{a.seed}{tag}.json"), "w") as g:
            json.dump({k: v for k, v in c.items() if k != "occ"} | {"rel": {o: c["occ"][o]["rel"] for o in occ},
                      "identical": {o: sum(1 for x, y in zip(pts[o], pts[occ[0]]) if x == y) for o in occ[1:]},
                      "shell": {o: c["occ"][o]["shell"] for o in occ}, "stats": {o: c["occ"][o]["stats"] for o in occ}},
                      g, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
