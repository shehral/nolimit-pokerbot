"""Paired A/B: variant-aware postflop class (brain._POST, POSTFLOP_VARIANT notes) vs the hold'em approximation
(hand_class / post_class, NOLIMIT_POSTFLOP_VARIANT=0), on a sim/variants rehearsal variant with its rebuilt tables.

    .venv/bin/python bot/tests/ab_postflop_variant.py shortdeck --tables 1000 --seed 1106 --workers 4
    .venv/bin/python bot/tests/ab_postflop_variant.py toss --tables 1000

Both arms: our bot (bot.shell:Bot) in 1 seat vs 7 of the variant's example agents, STRICT mode, timeouts on thread
CPU time at 1 ms, the SAME seed (same seats and decks), the same swapped tables (bot/out/drill/<variant>/
variant_tables.py). Each arm runs in a fresh process (drill.py --bench), sequentially, with --workers processes.
Writes bot/out/postflop_variant/<variant>_ab.json: per-arm errors and CPU/wall latency of act(), the paired
difference (variant - holdem) in points per table with a 95% interval, and a class-difference diagnostic (share of
our postflop decisions whose class changes, from an in-process replay of --diag-tables tables).
Every log line starts with the real clock.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
ROOT = os.path.dirname(BOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PY = sys.executable
OUT = os.path.join(BOT, "out", "postflop_variant")


def log(msg, fh=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} load {os.getloadavg()[0]:.1f}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return {"mean": round(m, 3), "lo": round(m - h, 3), "hi": round(m + h, 3), "n": n,
            "identical_tables": sum(1 for x in d if x == 0)}


def arm(variant, tables, seed, workers, tables_path, on, out_json):
    env = dict(os.environ, NOLIMIT_TABLES=tables_path, NOLIMIT_BOT_STRICT="1", NOLIMIT_TABLES_QUIET="1",
               NOLIMIT_POSTFLOP_VARIANT="1" if on else "0")
    cmd = [PY, os.path.join(BOT, "tools", "drill.py"), "--bench", variant, "--tables", str(tables), "--seed",
           str(seed), "--workers", str(workers), "--bench-out", out_json]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    with open(out_json) as f:
        return json.load(f)


DIAG_CODE = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from bot import brain
from sim.variants.runner import run_variant
from bot.tests.ab_postflop_variant import EXAMPLE
cnt = {"n": 0, "diff": 0, "up": 0, "down": 0, "pairs": {}}
orig = brain.post_class_variant
def wrapped(hole, board, boards=None, street=None):
    c = orig(hole, board, boards, street)
    if len(hole) == 2 and not boards:
        h = brain.hand_class(hole[0], hole[1], board)
    else:
        h = brain.post_class(hole, board, boards)
    cnt["n"] += 1
    if c != h:
        cnt["diff"] += 1
        cnt["up" if c > h else "down"] += 1
        k = f"{h}->{c}"
        cnt["pairs"][k] = cnt["pairs"].get(k, 0) + 1
    return c
brain.post_class_variant = wrapped
v, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ents = [("me", "bot.shell:Bot")] + [("ex", "sim.variants.agents:" + EXAMPLE[v])] * 7
run_variant(v, ents, n, seed=seed, workers=1, time_limit_ms=None, timing="cpu")
print(json.dumps(cnt))
"""

EXAMPLE = {"plain": "VariantExample", "toss": "TossExample", "shortdeck": "ShortDeckExample",
           "bounty": "BountyExample", "doubleboard": "DoubleBoardExample", "antes": "AnteExample"}


def diag(variant, tables, seed, tables_path):
    env = dict(os.environ, NOLIMIT_TABLES=tables_path, NOLIMIT_TABLES_QUIET="1", NOLIMIT_POSTFLOP_VARIANT="1")
    p = subprocess.run([PY, "-c", DIAG_CODE, variant, str(tables), str(seed)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--tables", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1106)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--diag-tables", type=int, default=200)
    ap.add_argument("--tables-path", default=None)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    tp = a.tables_path or os.path.join(BOT, "out", "drill", a.variant, "variant_tables.py")
    fh = open(os.path.join(OUT, f"{a.variant}_ab.log"), "a")
    log(f"A/B {a.variant}: {a.tables} paired tables, seed {a.seed}, workers {a.workers}, tables {tp}", fh)
    res = {"variant": a.variant, "tables": a.tables, "seed": a.seed, "workers": a.workers, "tables_path": tp,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "loadavg_start": list(os.getloadavg())}
    rows = {}
    for name, on in (("holdem", False), ("variant", True)):
        t0 = time.time()
        r = arm(a.variant, a.tables, a.seed, a.workers, tp, on, os.path.join(OUT, f"{a.variant}_{name}.json"))
        rows[name] = r
        log(f"  arm {name}: {r['decisions']:,} decisions, exc/illegal/timeouts {r['exceptions']}/{r['illegal']}/"
            f"{r['timeouts']}, over {r.get('over_limit')}, mean pts {r['mean_points']:.3f}, CPU us {r['cpu_us']}, "
            f"wall us p50/p99.9/max {r['latency_us']['p50']}/{r['latency_us']['p99.9']}/{r['latency_us']['max']}, "
            f"tables {r['tables_in_play'][:60]}, {time.time() - t0:.1f} s, load {r['loadavg']}", fh)
        res[name] = {k: v for k, v in r.items() if k != "points"}
    d = paired(rows["variant"]["points"], rows["holdem"]["points"])
    res["paired_diff_pts"] = d
    log(f"  paired variant - holdem: {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] pts/table over {d['n']} "
        f"tables ({d['identical_tables']} tables with identical points)", fh)
    if a.diag_tables:
        dg = diag(a.variant, a.diag_tables, a.seed, tp)
        dg["share"] = round(dg["diff"] / dg["n"], 4) if dg["n"] else 0.0
        res["class_diff"] = dg
        log(f"  class diagnostic ({a.diag_tables} tables in-process): {dg['diff']} of {dg['n']} postflop classes "
            f"differ ({100 * dg['share']:.2f}%), up {dg['up']}, down {dg['down']}, {dg['pairs']}", fh)
    res["loadavg_end"] = list(os.getloadavg())
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUT, f"{a.variant}_ab.json"), "w") as f:
        json.dump(res, f, indent=1)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
