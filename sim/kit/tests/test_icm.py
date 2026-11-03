"""Exact ICM against brute force (n <= 6) and the two exact implementations against each other (n <= 8)."""
import random

import pytest

from sim.kit.icm import icm_brute, icm_equities, make_icm
from sim.kit.payouts import LINEAR, PAYOUTS, TOP3, WTA, get_payouts


@pytest.mark.parametrize("pname", sorted(PAYOUTS))
def test_vs_bruteforce_small_n(pname):
    rng = random.Random(11)
    pays = PAYOUTS[pname]
    icm1 = make_icm(pays)
    for _ in range(200):
        n = rng.randint(2, 6)
        stacks = [rng.choice([0, rng.randint(1, 5000)]) if rng.random() < 0.2 else rng.randint(1, 5000)
                  for _ in range(n)]
        if sum(1 for s in stacks if s > 0) == 0:
            stacks[0] = 100
        ref = icm_brute(stacks, pays)
        dp = icm_equities(stacks, pays)
        for i in range(n):
            assert abs(dp[i] - ref[i]) < 1e-9
            assert abs(icm1(stacks, i) - ref[i]) < 1e-7
        k = sum(1 for s in stacks if s > 0)
        assert abs(sum(dp) - sum(get_payouts(pays)[:k])) < 1e-9


def test_eight_players_dp_vs_exponential_race():
    rng = random.Random(3)
    for pays in (TOP3, LINEAR, WTA, (35.0, 25.0, 15.0, 10.0, 7.0, 5.0, 2.0, 1.0)):
        icm1 = make_icm(pays)
        for _ in range(50):
            stacks = [rng.randint(1, 8000) for _ in range(8)]
            dp = icm_equities(stacks, pays)
            for i in range(8):
                assert abs(icm1(stacks, i) - dp[i]) < 1e-7
            assert abs(sum(dp) - sum(pays)) < 1e-9


def test_known_values():
    eq = icm_equities([1000] * 8, TOP3)
    assert all(abs(x - 12.5) < 1e-12 for x in eq)
    # heads-up: the loser gets 2nd place money for sure, the rest is chip-proportional
    eq = icm_equities([3000, 1000], TOP3)
    assert abs(eq[0] - (30 + 20 * 0.75)) < 1e-12 and abs(eq[1] - (30 + 20 * 0.25)) < 1e-12
    assert icm_equities([0, 500, 0], TOP3) == [0.0, 50.0, 0.0]
    eq = icm_equities([5000, 3000, 2000], WTA)       # winner-take-all ICM = chip share
    assert all(abs(eq[i] - 100 * s / 10000) < 1e-12 for i, s in enumerate([5000, 3000, 2000]))
