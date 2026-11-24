"""Print the headline tables of docs/BENCHMARK.md from the benchmark JSON (default: results/sim/).

    .venv/bin/python sim/bench/summarize.py                 # reads results/sim/*.json
    .venv/bin/python sim/bench/summarize.py sim/bench/out   # reads a fresh run

Numbers are printed exactly as stored by sim/bench/common.run_cell (relative score = points per table / 12.5,
95% confidence intervals; paired differences are computed table by table on identical tables).
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIELDS = ["T1_like", "T6_like", "STRONG", "AI_ONLY", "AI_MIX"]


def _load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_TOP3_TURBO.json"))):
        with open(p) as f:
            c = json.load(f)
        out[c["field"]] = (os.path.relpath(p, ROOT), c)
    return out


def _rel(c, o):
    r = c["occ"][o]["rel"]
    return f"{r['rel']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def _pair(c, a, b):
    p = c["pairs"].get(f"{a} - {b}")
    if p is None:
        p = c["pairs"][f"{b} - {a}"]
        return f"{-p['diff_rel']:+.3f} [{-p['hi_rel']:+.3f}, {-p['lo_rel']:+.3f}]"
    return f"{p['diff_rel']:+.3f} [{p['lo_rel']:+.3f}, {p['hi_rel']:+.3f}]"


def main(argv):
    d = argv[1] if len(argv) > 1 else os.path.join(ROOT, "results", "sim")
    cells = _load(d)
    fields = [f for f in FIELDS if f in cells]
    print("| field | tables | seed | v0 | v1 | v2 | v2 - v0 (paired) | file |")
    print("|---|---|---|---|---|---|---|---|")
    for f in fields:
        path, c = cells[f]
        print(f"| {f} | {c['n_tables']:,} | {c['seed']} | {_rel(c, 'v0')} | {_rel(c, 'v1')} | {_rel(c, 'v2')} | "
              f"{_pair(c, 'v2', 'v0')} | `{path}` |")
    print()
    print("| field | icm_pro in our seat | chart_tag in our seat | lag_aggro in our seat | v2 - icm_pro (paired) |")
    print("|---|---|---|---|---|")
    for f in fields:
        _, c = cells[f]
        print(f"| {f} | {_rel(c, 'icm_pro')} | {_rel(c, 'chart_tag')} | {_rel(c, 'lag_aggro')} | "
              f"{_pair(c, 'v2', 'icm_pro')} |")
    print()
    print("| field | our decisions (v0 + v1 + v2) | exceptions | illegal | CPU over 1 ms | masked by the shell |")
    print("|---|---|---|---|---|---|")
    for f in fields:
        _, c = cells[f]
        st = [c["occ"][o]["stats"] for o in ("v0", "v1", "v2")]
        sh = [c["occ"][o]["shell"] or {} for o in ("v0", "v1", "v2")]
        print(f"| {f} | {sum(s['decisions'] for s in st):,} | {sum(s['exceptions'] for s in st)} | "
              f"{sum(s['illegal'] for s in st)} | {sum(s['over_1ms_cpu'] for s in st)} | "
              f"{sum(x.get('errors_masked', 0) for x in sh)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
