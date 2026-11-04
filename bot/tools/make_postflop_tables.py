"""Generate bot/postflop_tables.py: the postflop engine's equity-vs-range tables (POSTFLOP_ENGINE notes).

Offline Monte Carlo with the kit's 7-card evaluator (sim/kit/cards.py). One sample: a street (flop, turn, river in
turn), our hand from the top HERO_PCT of the push order, a random board of that street; then, for each opponent
width (top 12% / 28% / 55% of the push order), PER_WIDTH opponent hands drawn from that range without our cards or
the board's, each with its own random runout to the river; our result (1 / 0.5 / 0) is added to the cell
(street, texture, our class, width) under the opponent's stratum (postflop.stratum of its own class on the board).
The classifiers are bot/postflop.py's own, so the generator and the lookups cannot disagree.

Sparse cells are shrunk: each stratum equity toward the same stratum's equity pooled over textures (same street,
class, width), which is itself shrunk toward the pool over draws (same street, made class, width); frequencies
the same way. Values are stored as bytes (x 255), base64 encoded, in a pure-data module.

    .venv/bin/python bot/tools/make_postflop_tables.py --boards 400000 --workers 4 --seed 1
"""
import argparse
import base64
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
ROOT = os.path.dirname(BOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HERO_PCT = 0.50
WIDTHS = (0.12, 0.28, 0.55)
PER_WIDTH = 3
K_SHRINK = 25.0


def _ranges():
    from bot import bot_tables as T
    order = T.PUSH_ORDER.split()
    ranks = "23456789TJQKA"

    def combos_of(lab):
        a, b = ranks.index(lab[0]), ranks.index(lab[1])
        out = []
        if a == b:
            for s1 in range(4):
                for s2 in range(s1 + 1, 4):
                    out.append((a * 4 + s1, b * 4 + s2))
        elif lab[2] == "s":
            for s in range(4):
                out.append((a * 4 + s, b * 4 + s))
        else:
            for s1 in range(4):
                for s2 in range(4):
                    if s1 != s2:
                        out.append((a * 4 + s1, b * 4 + s2))
        return out

    def top(frac):
        out = []
        for lab in order:
            if len(out) >= frac * 1326:
                break
            out.extend(combos_of(lab))
        return out
    return top(HERO_PCT), [top(w) for w in WIDTHS]


def work(task):
    seed, n_boards = task
    from bot import postflop as PF
    from sim.kit.cards import evaluate
    hero_r, opp_r = _ranges()
    rng = random.Random(seed)
    NS, NT, NC, NW = PF.N_STREETS, PF.N_TEX, PF.N_CLS, PF.N_WIDTH
    n_cells = NS * NT * NC * NW
    cnt = [0] * (n_cells * 4)
    eqs = [0.0] * (n_cells * 4)
    feat, tex_of, strat = PF.hand_feat, PF.texture, PF.stratum
    rr = rng.random
    rnd = rng.randrange
    for it in range(n_boards):
        st = it % 3
        nb = 3 + st
        h = hero_r[int(rr() * len(hero_r))]
        used = {h[0], h[1]}
        board = []
        while len(board) < nb:
            c = rnd(52)
            if c not in used:
                used.add(c)
                board.append(c)
        tex = tex_of(board)
        made, draw = feat(h[0], h[1], board)
        cls = made * PF.N_DRAW + draw
        base_cell = ((st * NT + tex) * NC + cls) * NW
        for w in range(NW):
            rg = opp_r[w]
            L = len(rg)
            idx = (base_cell + w) * 4
            k = 0
            tries = 0
            while k < PER_WIDTH and tries < 50:
                tries += 1
                o = rg[int(rr() * L)]
                if o[0] in used or o[1] in used:
                    continue
                k += 1
                om, od = feat(o[0], o[1], board)
                s = strat(om, od)
                full = list(board)
                u2 = set(used)
                u2.add(o[0])
                u2.add(o[1])
                while len(full) < 5:
                    c = rnd(52)
                    if c not in u2:
                        u2.add(c)
                        full.append(c)
                a = evaluate(full + [h[0], h[1]])
                b = evaluate(full + [o[0], o[1]])
                cnt[idx + s] += 1
                eqs[idx + s] += 1.0 if a > b else (0.5 if a == b else 0.0)
    return cnt, eqs


def shrink(cnt, eqs):
    """Hierarchical shrinkage -> list of 7-tuples (f0, f1, f2, e0..e3) per cell."""
    from bot import postflop as PF
    NS, NT, NM, ND, NW = PF.N_STREETS, PF.N_TEX, PF.N_MADE, PF.N_DRAW, PF.N_WIDTH
    NC = NM * ND

    def ci(st, tex, cls, w):
        return ((st * NT + tex) * NC + cls) * NW + w
    # pools over textures (st, cls, w) and over textures + draws (st, made, w)
    p1c = {}
    p1e = {}
    p2c = {}
    p2e = {}
    for st in range(NS):
        for tex in range(NT):
            for cls in range(NC):
                m = cls // ND
                for w in range(NW):
                    i = ci(st, tex, cls, w) * 4
                    a = p1c.setdefault((st, cls, w), [0, 0, 0, 0])
                    b = p1e.setdefault((st, cls, w), [0.0, 0.0, 0.0, 0.0])
                    a2 = p2c.setdefault((st, m, w), [0, 0, 0, 0])
                    b2 = p2e.setdefault((st, m, w), [0.0, 0.0, 0.0, 0.0])
                    for s in range(4):
                        a[s] += cnt[i + s]
                        b[s] += eqs[i + s]
                        a2[s] += cnt[i + s]
                        b2[s] += eqs[i + s]
    K = K_SHRINK

    def est(c, e, prior_f, prior_e):
        n = sum(c)
        f = [(c[s] + K * prior_f[s]) / (n + K) for s in range(4)]
        q = [(e[s] + K * prior_e[s]) / (c[s] + K) for s in range(4)]
        return f, q
    uni_f, uni_e = [0.25] * 4, [0.5] * 4
    g = {}
    for key, c in p2c.items():
        g[key] = est(c, p2e[key], uni_f, uni_e)
    p = {}
    for key, c in p1c.items():
        st, cls, w = key
        gf, ge = g[(st, cls // ND, w)]
        p[key] = est(c, p1e[key], gf, ge)
    out = []
    n_low = 0
    for st in range(NS):
        for tex in range(NT):
            for cls in range(NC):
                for w in range(NW):
                    i = ci(st, tex, cls, w) * 4
                    c = cnt[i:i + 4]
                    if sum(c) < 50:
                        n_low += 1
                    pf, pe = p[(st, cls, w)]
                    f, q = est(c, eqs[i:i + 4], pf, pe)
                    out.append((f[0], f[1], f[2], q[0], q[1], q[2], q[3]))
    return out, n_low


def encode(rows):
    b = bytearray()
    for r in rows:
        f0, f1, f2 = r[0], r[1], r[2]
        q0, q1, q2 = int(round(f0 * 255)), int(round(f1 * 255)), int(round(f2 * 255))
        while q0 + q1 + q2 > 255:                     # keep f3 = 1 - f0 - f1 - f2 non-negative after rounding
            j = max(range(3), key=lambda t: (q0, q1, q2)[t])
            if j == 0:
                q0 -= 1
            elif j == 1:
                q1 -= 1
            else:
                q2 -= 1
        b.extend((q0, q1, q2))
        for x in r[3:]:
            b.append(max(0, min(255, int(round(x * 255)))))
    return bytes(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boards", type=int, default=400000, help="hero boards in total (x3 widths x PER_WIDTH opps)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(BOT, "postflop_tables.py"))
    a = ap.parse_args(argv)
    from bot import postflop as PF
    t0 = time.time()
    cpu0 = time.process_time()
    chunks = max(a.workers * 4, 1)
    per = a.boards // chunks
    tasks = [(a.seed * 1000003 + i, per) for i in range(chunks)]
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(work, tasks)
    else:
        res = [work(t) for t in tasks]
    n = len(res[0][0])
    cnt = [0] * n
    eqs = [0.0] * n
    for c, e in res:
        for i in range(n):
            if c[i]:
                cnt[i] += c[i]
                eqs[i] += e[i]
    rows, n_low = shrink(cnt, eqs)
    data = encode(rows)
    b64 = base64.b64encode(data).decode("ascii")
    total = sum(cnt)
    info = (f"{per * chunks} boards, {total} opponent samples, seed {a.seed}, widths {WIDTHS}, hero top "
            f"{HERO_PCT:.0%}, K {K_SHRINK:g}, {n_low} of {len(rows)} keys under 50 samples (shrunk), generated "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines = [b64[i:i + 100] for i in range(0, len(b64), 100)]
    with open(a.out + ".tmp", "w") as f:
        f.write('"""GENERATED by bot/tools/make_postflop_tables.py; do not edit. Pure data for bot/postflop.py.\n\n')
        f.write("DATA_B64: base64 of one byte per value (x / 255), 7 values per key (f0, f1, f2, e0, e1, e2, e3),\n")
        f.write("keys in LAYOUT order (street, texture, class, width). See POSTFLOP_ENGINE notes.\n")
        f.write('"""\n')
        f.write(f"LAYOUT = ({PF.N_STREETS}, {PF.N_TEX}, {PF.N_CLS}, {PF.N_WIDTH}, {PF.N_K})\n")
        f.write(f"WIDTHS = {WIDTHS!r}\n")
        f.write(f"INFO = {info!r}\n")
        f.write("DATA_B64 = (\n")
        for ln in lines:
            f.write(f'    "{ln}"\n')
        f.write(")\n")
    os.replace(a.out + ".tmp", a.out)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} load {os.getloadavg()[0]:.1f}] wrote {a.out}: "
          f"{os.path.getsize(a.out)} bytes; {info}; wall {time.time() - t0:.0f} s, parent cpu "
          f"{time.process_time() - cpu0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
