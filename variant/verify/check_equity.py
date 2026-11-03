"""Independent cross-checks of built EQ_RANDOM values (not using variant/mc.py).

toss: heads-up direct simulation of the rehearsal rule (3 dealt, 2-card flop, each player keeps spec.keep(dealt,
      flop) and tosses the third card to the board, turn, river; best 5 of 2 kept + 8 board via evaluate_any).
doubleboard / antes / bounty: their EQ_RANDOM should equal plain's within noise (same card game per board).

    .venv/bin/python -m variant.verify.check_equity variant/verify/out/tables_toss.py [n_per_class]
"""
import json
import random
import sys

from variant.build_tables import load_tables
from variant.handspace import HandSpace
from variant.verify import specs as S


def toss_direct(spec, hs, cls, n, rng):
    deck = list(spec.deck)
    mem = [h for h, c in zip(hs.universe, hs.univ_class) if c == cls]
    tot = 0.0
    for _ in range(n):
        hero = rng.choice(mem)
        rest = [c for c in deck if c not in hero]
        rng.shuffle(rest)
        vil, flop, tr = rest[:3], rest[3:5], rest[5:7]
        kh = spec.keep(tuple(hero), tuple(flop))
        kv = spec.keep(tuple(vil), tuple(flop))
        board = list(flop) + [c for c in hero if c not in kh] + [c for c in vil if c not in kv] + list(tr)
        a = spec.evaluator(list(kh) + board)
        b = spec.evaluator(list(kv) + board)
        tot += 1.0 if a > b else 0.5 if a == b else 0.0
    return tot / n


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = argv[0]
    n = int(argv[1]) if len(argv) > 1 else 20000
    T = load_tables(path)
    name = T.META["spec"]
    rng = random.Random(3)
    out = {"path": path, "spec": name}
    if name in ("toss", "timed_toss"):
        spec = S.toss()[0]
        hs = HandSpace(spec)
        labs = T.EQ_LABELS.split()
        ev = T.EQ_RANDOM
        order = sorted(range(len(labs)), key=lambda i: -ev[i])
        picks = [order[0], order[len(order) // 4], order[len(order) // 2], order[3 * len(order) // 4], order[-1]]
        rows = []
        for i in picks:
            d = toss_direct(spec, hs, hs.label_index[labs[i]], n, rng)
            se = (0.25 / n) ** 0.5
            rows.append({"label": labs[i], "built": round(ev[i], 4), "direct": round(d, 4),
                         "diff": round(ev[i] - d, 4), "direct_se": round(se, 4)})
        out["rows"] = rows
        out["max_abs_diff"] = max(abs(r["diff"]) for r in rows)
    else:
        P = load_tables(argv[2]) if len(argv) > 2 else load_tables(path.replace(f"tables_{name}", "tables_plain"))
        pl = dict(zip(P.EQ_LABELS.split(), P.EQ_RANDOM))
        diffs = [abs(e - pl[l]) for l, e in zip(T.EQ_LABELS.split(), T.EQ_RANDOM)]
        out["vs_plain_eq_random_max_abs_diff"] = round(max(diffs), 4)
        out["vs_plain_eq_random_mean_abs_diff"] = round(sum(diffs) / len(diffs), 4)
        pt = dict(zip(P.EQ_LABELS.split(), P.EQ_TABLE))
        cd = [abs(a - b) for l, row in zip(T.EQ_LABELS.split(), T.EQ_TABLE) for a, b in zip(row, pt[l])]
        out["vs_plain_eq_table_max_abs_diff"] = round(max(cd), 4)
        out["vs_plain_eq_table_mean_abs_diff"] = round(sum(cd) / len(cd), 4)
        out["push_order_same"] = T.PUSH_ORDER == P.PUSH_ORDER
        out["jam_fit"], out["plain_jam_fit"] = T.JAM_FIT, P.JAM_FIT
        out["call_fit"], out["plain_call_fit"] = T.CALL_FIT, P.CALL_FIT
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
