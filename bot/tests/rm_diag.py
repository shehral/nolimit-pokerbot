"""Range-model diagnostic (not a test): the end-of-table posterior ratios our v2 assigns to 7 copies of an opponent.

    .venv/bin/python bot/tests/rm_diag.py ai_default_ml ai_poker_pro icm_pro --tables 60
"""
import argparse
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.bench import common as C          # noqa: E402
from sim.field_ai import fields as AF       # noqa: E402
from sim.kit.engine import Table            # noqa: E402


def run(opp_name, n, params):
    from bot import shell
    made = []

    def ours(seat, cfg):
        b = shell.make(seat, cfg, version=2, params=params)
        made.append((seat, b))
        return b
    opp = C.factory_for(opp_name, sandbox_p=AF.sandbox_p("1.5"))
    rows = {"jam": [], "open": [], "3bet": [], "call": [], "n": []}
    for t in range(n):
        made.clear()
        facs = [ours if s == 0 else opp for s in range(8)]
        Table(facs, "TOP3", "TURBO", seed=f"rmdiag:{t}", timing="cpu", time_limit_ms=None).play()
        for seat, b in made:
            br = b._brain
            if br.tr.rm_pend:
                br.tr.rm_commit()
            for j in range(8):
                if j == seat:
                    continue
                st = br.stats[j]
                rows["jam"].append(br.rm_jam_ratio(j, False))
                rows["3bet"].append(br.rm_jam_ratio(j, True))
                rows["open"].append(br.rm_open_ratio(j))
                rows["call"].append(br.rm_call_ratio(j))
                rows["n"].append((st.rm_jam_n, st.rm_open_n, st.rm_3b_n, st.rm_cj_n))
    return rows


def q(x):
    x = sorted(x)
    return f"p10 {x[len(x) // 10]:.2f} / med {x[len(x) // 2]:.2f} / p90 {x[9 * len(x) // 10]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opponents", nargs="+")
    ap.add_argument("--tables", type=int, default=60)
    a = ap.parse_args()
    for o in a.opponents:
        rows = run(o, a.tables, {"RANGE_MODEL": True})
        ns = list(zip(*rows["n"]))
        print(f"| {o} | {a.tables} | " + " | ".join(q(rows[k]) for k in ("jam", "open", "3bet", "call")) +
              " | " + " / ".join(f"{statistics.median(c):.0f}" for c in ns) + " |", flush=True)


if __name__ == "__main__":
    main()
