"""Engines: the board sweep equals brute-force pair enumeration on fixed boards (2 and 3 hole cards); the rollout
engine (independent code path) agrees with the sweep within sampling error."""
import random
import time

import pytest

from variant import mc
from variant.handspace import HandSpace
from variant.spec import VariantSpec, nlhe_spec


class FixedBoard(random.Random):
    def __init__(self, boards):
        super().__init__(0)
        self.boards = list(boards)

    def sample(self, population, k):
        return list(self.boards.pop(0))


def brute(hs, spec, board, wv):
    """num2[class][K], den[class][K] by enumerating every compatible (hero, villain) pair on one board."""
    C, K = hs.n_classes, len(wv[0])
    num2 = [[0.0] * K for _ in range(C)]
    den = [[0.0] * K for _ in range(C)]
    bs = set(board)
    hands = [h for h in hs.universe if not bs.intersection(h)]
    val = {h: spec.value(h, board) for h in hands}
    cls = {h: hs.class_of(h) for h in hands}
    for a in hands:
        sa = set(a)
        ca = cls[a]
        for b in hands:
            if sa.intersection(b):
                continue
            w = wv[cls[b]]
            r = 2 if val[a] > val[b] else (1 if val[a] == val[b] else 0)
            for g in range(K):
                num2[ca][g] += r * w[g]
                den[ca][g] += w[g]
    return num2, den


def _check(spec, boards):
    hs = HandSpace(spec)
    mc.init_worker(spec, hs)
    rng = random.Random(4)
    wv3 = [(1.0, rng.random(), float(rng.random() < 0.3)) for _ in range(hs.n_classes)]
    wv1 = [(w[1],) for w in wv3]                         # K = 1 takes the scalar fast path
    for wv in (wv3, wv1):
        K = len(wv[0])
        for bd in boards:
            orig = mc.random.Random
            mc.random.Random = lambda seed, _b=bd: FixedBoard([_b])
            try:
                n2, d2, nb = mc.range_task((0, 1, time.time() + 1e6, wv))
            finally:
                mc.random.Random = orig
            bn, bdn = brute(hs, spec, bd, wv)
            for i in range(hs.n_classes):
                for g in range(K):
                    assert abs(n2[i][g] - bn[i][g]) < 1e-6 and abs(d2[i][g] - bdn[i][g]) < 1e-6, (K, i, g)
    # matrix engine on the same boards: num2 with one-hot weights equals the brute force with one-hot weights
    bd = boards[0]
    orig = mc.random.Random
    mc.random.Random = lambda seed, _b=bd: FixedBoard([_b])
    try:
        m2, _ = mc.matrix_task((0, 1, time.time() + 1e6))
    finally:
        mc.random.Random = orig
    onehot = [tuple(1.0 if j == i else 0.0 for j in range(hs.n_classes)) for i in range(hs.n_classes)]
    bn, _ = brute(hs, spec, bd, onehot)
    for i in range(hs.n_classes):
        for j in range(hs.n_classes):
            assert abs(m2[i][j] - bn[i][j]) < 1e-6


def test_sweep_equals_brute_force_two_cards():
    spec = VariantSpec(name="tiny2", removed_ranks="23456789")        # T..A, 20 cards, 2 hole cards
    cd = spec.codec
    _check(spec, [cd.ints("Ts Jh Qd Kc As".split()),
                  cd.ints("Ah Ad Ac Ks Kd".split()), cd.ints("Th Jh Qh Kh Ts".split())])


def test_sweep_equals_brute_force_three_cards_keep_two():
    spec = VariantSpec(name="tiny3", removed_ranks="23456789", hole_dealt=3, hole_kept=2)
    cd = spec.codec
    _check(spec, [cd.ints("Ts Jh Qd Kc As".split()), cd.ints("Ah Ad Kc Ks Td".split())])


def test_rollout_agrees_with_sweep_nlhe():
    spec = nlhe_spec()
    hs = HandSpace(spec)
    mc.init_worker(spec, hs)
    wv = [(1.0,)] * hs.n_classes
    n2, d2, nb = mc.range_task((1, 4000, time.time() + 60, wv))
    rn, rd, rounds = mc.rollout_task((2, 300, time.time() + 60, wv))
    for lab in ("AA", "KK", "AKs", "T9s", "72o", "22"):
        i = hs.label_index[lab]
        a = n2[i][0] / (2 * d2[i][0])
        b = rn[i][0] / (2 * rd[i][0])
        assert abs(a - b) < 4 * (0.25 / rd[i][0]) ** 0.5 + 0.01, (lab, a, b)
