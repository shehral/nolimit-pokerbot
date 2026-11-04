"""Paired A/B of the postflop engine: v2 (POSTFLOP_ENGINE off, the shipped default) vs v2:POSTFLOP_ENGINE=True on
identical tables, through sim/bench/common.run_cell (the bench's pairing, production-mode shell counters, AI-bot
timeout model at time_scale 1.5), TOP3 TURBO.

    .venv/bin/python bot/tests/ab_postflop_engine.py --cells AI_MIX:5000,STRONG:5000,T6_like:5000 --seed 3101
    .venv/bin/python bot/tests/ab_postflop_engine.py --cells ai_quant:3000,ai_gto_fan:3000 --seed 3103
    .venv/bin/python bot/tests/ab_postflop_engine.py --arms "v2;v2:POSTFLOP_ENGINE=True;v2:POSTFLOP_ENGINE=True,..."

Writes bot/out/postflop_engine/ab_<field>_<seed><tag>.json (the full cell) and appends one line per arm to
bot/out/postflop_engine/ab_log.txt, stamped with the real clock and the load average.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
ROOT = os.path.dirname(BOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.bench import common as C           # noqa: E402

OUT = os.path.join(BOT, "out", "postflop_engine")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')} load {' '.join('%.1f' % x for x in os.getloadavg())}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ab_log.txt"), "a") as f:
        f.write(line + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", required=True, help="field:tables,...")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default="v2;v2:POSTFLOP_ENGINE=True", help="';'-separated occupants, first = base")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    from sim.field_ai import fields as AF
    sp = AF.sandbox_p("1.5")
    arms = [x for x in a.arms.split(";") if x]
    pool = C.make_pool(a.workers)
    try:
        for item in a.cells.split(","):
            field, _, n = item.partition(":")
            n = int(n)
            name = f"ab_{field}_{a.seed}{a.tag}.json"
            log(f"start {field} {n} tables seed {a.seed} arms {arms} workers {a.workers}")
            cpu0 = time.process_time()
            out = C.run_cell(field, arms, n, seed=a.seed, payouts="TOP3", schedule="TURBO", pool=pool, context=arms[0],
                             sandbox_p=sp, label=f"postflop engine {field} seed {a.seed}", verbose=False)
            out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            out["command"] = " ".join([os.path.relpath(__file__, ROOT)] + (argv if argv is not None else sys.argv[1:]))
            os.makedirs(OUT, exist_ok=True)
            path = os.path.join(OUT, name)
            import json
            with open(path + ".tmp", "w") as f:
                json.dump(out, f, separators=(",", ":"), default=str)
            os.replace(path + ".tmp", path)
            for o in arms:
                r = out["occ"][o]["rel"]
                st = out["occ"][o]["stats"]
                sh = out["occ"][o]["shell"]
                log(f"  {field} seed {a.seed} {o:<48} rel {r['rel']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] dec "
                    f"{st['decisions']} exc {st['exceptions']} ill {st['illegal']} cpu>1ms {st['over_1ms_cpu']} "
                    f"masked {sh['errors_masked']} fallbacks {sh['fallbacks_v0']}")
            for k, p in out["pairs"].items():
                log(f"  {field} seed {a.seed} n {n} DIFF {k}: {p['diff_rel']:+.4f} [{p['lo_rel']:+.4f}, "
                    f"{p['hi_rel']:+.4f}] rel (corr {p['corr']:.2f}, identical tables {p['identical_tables']}) "
                    f"elapsed {out['elapsed_s']:.0f} s, load {out['loadavg_start'][0]:.1f}->"
                    f"{out['loadavg_end'][0]:.1f}, parent cpu {time.process_time() - cpu0:.0f} s")
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
