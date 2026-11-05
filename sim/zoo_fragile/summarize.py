"""Markdown rows from probe_bench cells: induced opponent errors (engine-counted and masked) and our paired score.

    .venv/bin/python sim/zoo_fragile/summarize.py sim/zoo_fragile/out/frag_*_s501_n2000.json
"""
import json
import sys


def row(path):
    c = json.load(open(path))
    occs = c["occupants"]
    base = c["occ"][occs[0]]
    out = []
    for o in occs[1:]:
        d = c["occ"][o]
        p = c["comparisons"][o]
        s = p["score"]
        e = p["opp_errors_per_1k_hands_diff"]
        m = p.get("opp_masked_per_1k_hands_diff", [0, 0, 0])
        out.append(
            f"| {c['field']} | {c['seed']} | {c['n_tables']:,} | `{o}` | "
            f"{base['opp_errors_per_1k_hands']:.2f} | {d['opp_errors_per_1k_hands']:.2f} | "
            f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}] | "
            f"{base.get('opp_masked_per_1k_hands', 0):.2f} / {d.get('opp_masked_per_1k_hands', 0):.2f} "
            f"({m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]) | "
            f"{base['rel']['rel']:.3f} | {s['diff_rel']:+.3f} [{s['lo_rel']:+.3f}, {s['hi_rel']:+.3f}] | "
            f"{base['ours_errors_engine']}+{base['ours_errors_masked']} / {d['ours_errors_engine']}+{d['ours_errors_masked']} | "
            f"{c['loadavg_start'][0]:.0f}-{c['loadavg_end'][0]:.0f} | {c['finished']} |")
    return out


if __name__ == "__main__":
    print("| field | seed | tables | occupant | opp errors / 1k hands OFF | ON | ON - OFF (paired) | masked / 1k hands "
          "OFF / ON (diff) | v2 rel | score ON - OFF (paired) | our errors OFF / ON | load | finished |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sys.argv[1:]:
        for r in row(p):
            print(r)
