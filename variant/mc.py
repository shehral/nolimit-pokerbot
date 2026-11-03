"""Monte Carlo engines for build_tables (standard library; run inside multiprocessing workers).

Separable games (a player's showdown value depends only on its own dealt cards and the board): for each sampled board
the engine ranks EVERY dealt hand once (one evaluator call per hand), sorts them, and sweeps in value order. Card
removal is exact by inclusion-exclusion over the hero's cards (counters per card, per card pair, ...), so one board
gives every hero hand's result against every villain hand compatible with it. Two flavours:

  range_engine   K weight vectors over classes (equity vs random = one all-ones vector; vs the top-J grid = K
                 nested ranges). Accumulates, per hero class and range, 2 x wins + ties (num2) and the weighted
                 number of compatible villain hands (den) over all (hero, villain, board) triples: the pooled
                 estimator, identical in expectation to the bot's table (sum_j W_ij E_ij r_j / sum_j W_ij r_j).
  matrix_engine  the class-vs-class matrix (num2 only; the denominator is exact and board-free: see build_tables).

Non-separable games (toss to board, hi/lo, double board, custom share_fn): rollout_engine samples hero, villain and
board explicitly and calls spec.shares().
"""
import random
import time

_W = {}          # worker state: spec, hand space, universe arrays (set by init_worker or inherited through fork)


MAX_HANDS = 3000     # hands ranked per board when a board leaves more valid dealt hands (3 dealt: 16,215)


def init_worker(spec, hs, seed_base=0, max_hands=MAX_HANDS):
    """Install worker globals. Under fork this runs in the parent before the pool starts (inherited); under spawn it
    is the pool initializer (spec and hs must then be picklable)."""
    _W.clear()
    _W["spec"] = spec
    _W["hs"] = hs
    if hs.universe is None:
        return
    U = hs.universe
    deck = hs.deck
    M = max(deck) + 1
    masks = []
    for h in U:
        m = 0
        for c in h:
            m |= 1 << c
        masks.append(m)
    n = hs.n
    # subset ids for inclusion-exclusion: singles = card int; pairs, triples ... get ids above M
    sub_id = {}
    nxt = M
    subs = []
    for h in U:
        lst = []
        for size in range(1, n):
            sign = -1 if size % 2 else 1
            for t in _subsets(h, size):
                if size == 1:
                    sid = t[0]
                else:
                    sid = sub_id.get(t)
                    if sid is None:
                        sid = nxt
                        sub_id[t] = sid
                        nxt += 1
                lst.append((sign, sid))
        subs.append(tuple(lst))
    # what each hand plays at showdown when the keep is made before the board (precomputed)
    play = None
    if spec.must_use is None and (not spec.hole_kept or spec.keep_street == 0):
        play = [list(spec.keep(h)) if spec.hole_kept else list(h) for h in U]
    _W.update(U=U, UC=hs.univ_class, masks=masks, subs=subs, n_ids=nxt, M=M, play=play, max_hands=max_hands,
              full_sign=(-1) ** n, deck=list(deck), nb=sum(spec.board_schedule))


def _subsets(h, size):
    from itertools import combinations
    return combinations(h, size)


def _board_values(rng):
    """Sample a board; return (valid hand indices sorted by value, their values, board)."""
    deck = _W["deck"]
    nb = _W["nb"]
    board = rng.sample(deck, nb)
    bm = 0
    for c in board:
        bm |= 1 << c
    masks = _W["masks"]
    valid = [u for u, m in enumerate(masks) if not m & bm]
    mh = _W.get("max_hands")
    if mh and len(valid) > mh:
        # same random subset for heroes and villains: every compatible pair is kept with the same probability,
        # so the pooled estimators stay consistent; more boards per CPU second, less board-driven variance
        valid = rng.sample(valid, mh)
    play = _W["play"]
    spec = _W["spec"]
    if play is not None:
        ev = spec.evaluator
        if spec.must_use is None:
            vals = [ev(play[u] + board) for u in valid]
        else:
            vals = [spec.hand_value(play[u], board) for u in valid]
    else:
        U = _W["U"]
        vals = [spec.value(U[u], board) for u in valid]
    order = sorted(range(len(valid)), key=vals.__getitem__)
    return [valid[i] for i in order], [vals[i] for i in order], board


def _groups(vals):
    """(start, end) runs of equal values in a sorted list."""
    L = len(vals)
    i = 0
    out = []
    while i < L:
        j = i + 1
        v = vals[i]
        while j < L and vals[j] == v:
            j += 1
        out.append((i, j))
        i = j
    return out


# ===================================================================== range engine
def range_task(args):
    """args = (seed, max_boards, deadline, wv) with wv[class] = K-tuple of weights. Returns (num2, den, boards)."""
    seed, max_boards, deadline, wv = args
    if len(wv[0]) == 1:
        return _range_task1(args)
    rng = random.Random(seed)
    UC, subs, n_ids, full_sign = _W["UC"], _W["subs"], _W["n_ids"], _W["full_sign"]
    C = len(wv)
    K = len(wv[0])
    zero = [0.0] * K
    num2 = [list(zero) for _ in range(C)]
    den = [list(zero) for _ in range(C)]
    boards = 0
    while boards < max_boards and (boards == 0 or time.time() < deadline):
        hands, vals, _ = _board_values(rng)
        below0 = list(zero)
        below = [None] * n_ids          # lazily allocated K-lists
        for (i, j) in _groups(vals):
            if j - i == 1:
                u = hands[i]
                cls = UC[u]
                acc = below0
                for sign, sid in subs[u]:
                    b = below[sid]
                    if b is not None:
                        if sign < 0:
                            acc = [x - y for x, y in zip(acc, b)]
                        else:
                            acc = [x + y for x, y in zip(acc, b)]
                r = num2[cls]
                num2[cls] = [x + y + y for x, y in zip(r, acc)]
                w = wv[cls]
                below0 = [x + y for x, y in zip(below0, w)]
                for _, sid in subs[u]:
                    b = below[sid]
                    below[sid] = list(w) if b is None else [x + y for x, y in zip(b, w)]
            else:
                grp0 = list(zero)
                grp = {}
                members = hands[i:j]
                for u in members:
                    w = wv[UC[u]]
                    grp0 = [x + y for x, y in zip(grp0, w)]
                    for _, sid in subs[u]:
                        g = grp.get(sid)
                        grp[sid] = list(w) if g is None else [x + y for x, y in zip(g, w)]
                for u in members:
                    cls = UC[u]
                    acc = below0
                    tie = grp0
                    for sign, sid in subs[u]:
                        b = below[sid]
                        g = grp.get(sid)
                        if sign < 0:
                            if b is not None:
                                acc = [x - y for x, y in zip(acc, b)]
                            tie = [x - y for x, y in zip(tie, g)]
                        else:
                            if b is not None:
                                acc = [x + y for x, y in zip(acc, b)]
                            tie = [x + y for x, y in zip(tie, g)]
                    w = wv[cls]
                    if full_sign < 0:
                        tie = [x - y for x, y in zip(tie, w)]
                    else:
                        tie = [x + y for x, y in zip(tie, w)]
                    r = num2[cls]
                    num2[cls] = [x + a + a + t for x, a, t in zip(r, acc, tie)]
                below0 = [x + y for x, y in zip(below0, grp0)]
                for sid, g in grp.items():
                    b = below[sid]
                    below[sid] = g if b is None else [x + y for x, y in zip(b, g)]
        # denominators: compatible villain weight per hero = total - sum over proper subsets + (-1)^n own weight
        tot0 = below0
        for u in hands:
            cls = UC[u]
            acc = tot0
            for sign, sid in subs[u]:
                b = below[sid]
                if b is not None:
                    if sign < 0:
                        acc = [x - y for x, y in zip(acc, b)]
                    else:
                        acc = [x + y for x, y in zip(acc, b)]
            w = wv[cls]
            r = den[cls]
            if full_sign < 0:
                den[cls] = [x + a - y for x, a, y in zip(r, acc, w)]
            else:
                den[cls] = [x + a + y for x, a, y in zip(r, acc, w)]
        boards += 1
    return num2, den, boards


def _range_task1(args):
    """range_task for a single range (equity vs random and the like) with scalar counters: same results."""
    seed, max_boards, deadline, wv = args
    rng = random.Random(seed)
    UC, subs, n_ids, full_sign = _W["UC"], _W["subs"], _W["n_ids"], _W["full_sign"]
    C = len(wv)
    w1 = [x[0] for x in wv]
    num2 = [0.0] * C
    den = [0.0] * C
    boards = 0
    while boards < max_boards and (boards == 0 or time.time() < deadline):
        hands, vals, _ = _board_values(rng)
        below0 = 0.0
        below = [0.0] * n_ids
        for (i, j) in _groups(vals):
            if j - i == 1:
                u = hands[i]
                cls = UC[u]
                acc = below0
                for sign, sid in subs[u]:
                    acc += sign * below[sid]
                num2[cls] += acc + acc
                w = w1[cls]
                below0 += w
                for _, sid in subs[u]:
                    below[sid] += w
            else:
                members = hands[i:j]
                grp0 = 0.0
                grp = {}
                for u in members:
                    w = w1[UC[u]]
                    grp0 += w
                    for _, sid in subs[u]:
                        grp[sid] = grp.get(sid, 0.0) + w
                for u in members:
                    cls = UC[u]
                    acc = below0
                    tie = grp0
                    for sign, sid in subs[u]:
                        acc += sign * below[sid]
                        tie += sign * grp[sid]
                    tie += full_sign * w1[cls]
                    num2[cls] += acc + acc + tie
                below0 += grp0
                for sid, g in grp.items():
                    below[sid] += g
        for u in hands:
            cls = UC[u]
            acc = below0
            for sign, sid in subs[u]:
                acc += sign * below[sid]
            den[cls] += acc + full_sign * w1[cls]
        boards += 1
    return [[x] for x in num2], [[x] for x in den], boards


# ===================================================================== matrix engine
def matrix_task(args):
    """args = (seed, max_boards, deadline). Returns (num2 as C x C list, boards): num2[i][j] = sum over boards and
    compatible (a in i, b in j) of 2 [v_a > v_b] + [v_a == v_b]."""
    seed, max_boards, deadline = args
    rng = random.Random(seed)
    UC, subs, n_ids, full_sign = _W["UC"], _W["subs"], _W["n_ids"], _W["full_sign"]
    C = _W["hs"].n_classes
    num2 = [[0] * C for _ in range(C)]
    boards = 0
    while boards < max_boards and (boards == 0 or time.time() < deadline):
        hands, vals, _ = _board_values(rng)
        below0 = [0] * C
        below = [None] * n_ids
        for (i, j) in _groups(vals):
            if j - i == 1:
                u = hands[i]
                cls = UC[u]
                acc = below0
                for sign, sid in subs[u]:
                    b = below[sid]
                    if b is not None:
                        if sign < 0:
                            acc = [x - y for x, y in zip(acc, b)]
                        else:
                            acc = [x + y for x, y in zip(acc, b)]
                r = num2[cls]
                num2[cls] = [x + y + y for x, y in zip(r, acc)]
                below0[cls] += 1
                for _, sid in subs[u]:
                    b = below[sid]
                    if b is None:
                        b = below[sid] = [0] * C
                    b[cls] += 1
            else:
                members = hands[i:j]
                grp0 = [0] * C
                grp = {}
                for u in members:
                    cls = UC[u]
                    grp0[cls] += 1
                    for _, sid in subs[u]:
                        g = grp.get(sid)
                        if g is None:
                            g = grp[sid] = [0] * C
                        g[cls] += 1
                for u in members:
                    cls = UC[u]
                    acc = below0
                    tie = list(grp0)
                    for sign, sid in subs[u]:
                        b = below[sid]
                        g = grp[sid]
                        if sign < 0:
                            if b is not None:
                                acc = [x - y for x, y in zip(acc, b)]
                            tie = [x - y for x, y in zip(tie, g)]
                        else:
                            if b is not None:
                                acc = [x + y for x, y in zip(acc, b)]
                            tie = [x + y for x, y in zip(tie, g)]
                    tie[cls] += full_sign
                    r = num2[cls]
                    num2[cls] = [x + a + a + t for x, a, t in zip(r, acc, tie)]
                below0 = [x + y for x, y in zip(below0, grp0)]
                for sid, g in grp.items():
                    b = below[sid]
                    below[sid] = g if b is None else [x + y for x, y in zip(b, g)]
        boards += 1
    return num2, boards


# ===================================================================== rollout engine (non-separable games)
def rollout_task(args):
    """args = (seed, max_rounds, deadline, wv). One round = for every class, one hero hand vs one villain per
    range. Returns (num2, den, rounds) with num2 = 2 x the sum of hero pot shares (ties split), the same
    scale as range_task."""
    import bisect
    seed, max_rounds, deadline, wv = args
    rng = random.Random(seed)
    spec, hs = _W["spec"], _W["hs"]
    C = len(wv)
    K = len(wv[0])
    deck = list(hs.deck)
    nb = spec.board_cards
    if "members" not in _W:
        if hs.universe is not None:
            mem = [[] for _ in range(C)]
            for h, c in zip(hs.universe, hs.univ_class):
                mem[c].append(h)
        else:
            mem = None
        _W["members"] = mem
    mem = _W["members"]
    cum = []
    for r in range(K):
        acc, lst = 0.0, []
        for j in range(C):
            acc += wv[j][r] * hs.combos[j]
            lst.append(acc)
        cum.append(lst)
    num = [[0.0] * K for _ in range(C)]
    den = [[0] * K for _ in range(C)]
    rounds = 0

    def draw_hero(i):
        if mem is not None:
            return mem[i][rng.randrange(len(mem[i]))]
        for _ in range(100000):
            h = tuple(sorted(rng.sample(deck, hs.n)))
            if hs.class_of(h) == i:
                return h
        return None

    while rounds < max_rounds and (rounds == 0 or time.time() < deadline):
        for i in range(C):
            hero = draw_hero(i)
            if hero is None:
                continue
            hset = set(hero)
            for r in range(K):
                lst = cum[r]
                if lst[-1] <= 0:
                    continue
                vil = None
                for _ in range(200):
                    j = bisect.bisect_right(lst, rng.random() * lst[-1])
                    j = min(j, C - 1)
                    if mem is not None:
                        cand = mem[j][rng.randrange(len(mem[j]))]
                    else:
                        cand = tuple(sorted(rng.sample(deck, hs.n)))
                        if hs.class_of(cand) != j:
                            continue
                    if not hset.intersection(cand):
                        vil = cand
                        break
                if vil is None:
                    continue
                rest = [c for c in deck if c not in hset and c not in vil]
                board = rng.sample(rest, nb)
                num[i][r] += 2.0 * spec.shares([hero, vil], board)[0]
                den[i][r] += 1
        rounds += 1
    return num, den, rounds
