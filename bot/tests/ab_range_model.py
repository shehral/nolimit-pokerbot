"""Paired A/B of the range model (RANGE_MODEL notes) through sim/bench/common.run_cell, the same machinery as
the AI-field bench and the tuning log (TOP3 TURBO, AI bots at their 1.5x timeout rates, zoo near-limit bots at
SANDBOX_P), one JSON per cell. Not a pytest file.

    .venv/bin/python bot/tests/ab_range_model.py --fields AI_MIX,STRONG,T6_like --tables 5000 --seed 4101 \
        --workers 4 --outdir <dir>
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.bench import common as C          # noqa: E402
from sim.field_ai import fields as AF       # noqa: E402

BASE = "v2"
CAND = "v2:RANGE_MODEL=True"


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="AI_MIX,STRONG,T6_like")
    ap.add_argument("--tables", type=int, default=5000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--cand", default=CAND, help="';'-separated candidate occupants")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cands = [c for c in a.cand.split(";") if c]
    occ = [BASE] + cands
    pool = C.make_pool(a.workers)
    try:
        for fld in a.fields.split(","):
            ai = fld in AF.MIXES or AF.is_ai(fld)
            sp = AF.sandbox_p("1.5") if ai else None
            print(f"{stamp()} load {os.getloadavg()[0]:.2f} | start {fld} seed {a.seed} {a.tables} tables", flush=True)
            out = C.run_cell(field=fld, occupants=occ, n_tables=a.tables, seed=a.seed, payouts=a.payouts,
                             schedule=a.schedule, context=BASE, pool=pool, sandbox_p=sp, verbose=False,
                             label=f"range model {fld} seed {a.seed}")
            out["finished"] = stamp()
            out["command"] = " ".join([sys.executable, "bot/tests/ab_range_model.py"] + sys.argv[1:])
            path = os.path.join(a.outdir, f"rm_{fld}_{a.payouts}_s{a.seed}.json")
            with open(path, "w") as f:
                json.dump(out, f, separators=(",", ":"), default=str)
            line = [f"{stamp()} load {out['loadavg_start'][0]:.2f}->{out['loadavg_end'][0]:.2f} | {fld} seed {a.seed}"
                    f" {a.tables} tables, {out['elapsed_s']:.0f} s, workers {out['workers']}"]
            for o in occ:
                r = out["occ"][o]["rel"]
                st = out["occ"][o]["stats"]
                sh = out["occ"][o]["shell"]
                line.append(f"   {o:<26} rel {r['rel']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  dec {st['decisions']}"
                            f" exc {st['exceptions']} ill {st['illegal']} cpu>1ms {st['over_1ms_cpu']}"
                            f" masked {sh['errors_masked']} fallbacks {sh['fallbacks_v0']}")
            for c in cands:
                p = out["pairs"][f"{c} - {BASE}"]
                line.append(f"   {c} - {BASE}: {p['diff_rel']:+.3f} [{p['lo_rel']:+.3f}, {p['hi_rel']:+.3f}]"
                            f" (corr {p['corr']:.2f}, identical tables {p['identical_tables']})")
            print("\n".join(line), flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
