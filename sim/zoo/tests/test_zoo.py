"""Zoo tests: every archetype plays legal, reproducible poker; error rates match the design; mixes are well formed."""
import pickle
import random

import pytest

from sim.kit.runner import run
from sim.zoo import fields as F
from sim.zoo._common import chen, label, post_class, CARD_INT, has_draw, equity_vs_random

KEYS = list(F.ARCHETYPES)


def _run(key, tables=40, **kw):
    ents = [(key, F.factory(key))] + [("ex", "sim/kit/example_agent.py:ExampleAgent")] * 7
    return run(ents, n_tables=tables, seed=11, workers=1, time_limit_ms=None, **kw)


@pytest.mark.parametrize("key", KEYS)
def test_archetype_is_legal(key):
    res = _run(key)
    s = res.entrants[0].stats.summary()
    assert s["decisions"] > 200
    if key == "buggy":
        assert s["exceptions"] > 0 and s["illegal"] > 0
    else:
        assert s["exceptions"] == 0 and s["illegal"] == 0, s["examples"]


def test_buggy_rates():
    s = run([("b", F.factory("buggy"))] * 8, n_tables=60, seed=3, workers=1,
            time_limit_ms=None).entrants[0].stats.summary()
    # illegal draws that are not illegal in context (none: amount 0 / min-1 / max+1 are always out of range) so exact
    assert abs(s["exceptions"] / s["decisions"] - 0.10) < 0.02
    assert abs(s["illegal"] / s["decisions"] - 0.05) < 0.02


def test_reproducible_given_seed():
    a = _run("chart_tag", 20).entrants[0].points
    b = _run("chart_tag", 20).entrants[0].points
    assert a == b


def test_pickle_variants():
    for _, fac, _ in F.jittered_variants(20, 3):
        pickle.dumps(fac)


def test_mixes():
    for name, m in F.MIXES.items():
        assert sum(m.values()) == 100, name
    assert 0.25 <= F.broken_share(F.T1_like) <= 0.35
    assert 0.05 <= F.broken_share(F.T6_like) <= 0.10
    assert F.STRONG["icm_pro"] + F.STRONG["chart_tag"] >= 80
    assert sum(F.allocate(F.T1_like, 37).values()) == 37


def test_variants_distinct_and_in_range():
    v = F.jittered_variants(20, 5)
    assert len(v) == 20 and len({n for n, _, _ in v}) == 20
    for _, _, spec in v:
        cls = F.agent_class(spec["archetype"])
        for k, val in spec["params"].items():
            lo, hi = cls.JITTER[k]
            assert lo - 1e-3 <= val <= hi + 1e-3


def test_hand_helpers():
    assert chen(["Ah", "Ad"]) == 20 and chen(["7h", "2d"]) == -1 and chen(["Ah", "Kh"]) == 12
    assert label(["Ah", "Kh"]) == "AKs" and label(["Td", "Th"]) == "TT" and label(["2c", "9d"]) == "92o"
    i = lambda *cs: [CARD_INT[c] for c in cs]
    assert post_class(i("Ah", "Ad"), i("2c", "7d", "9s")) == 2
    assert post_class(i("Ah", "9d"), i("2c", "7c", "9s")) == 2
    assert post_class(i("Ah", "7d"), i("2c", "7c", "9s")) == 1
    assert post_class(i("Ah", "7d"), i("2c", "9c", "Ks")) == 0
    assert post_class(i("Ah", "Kd"), i("As", "Kc", "9s")) == 3
    assert has_draw(i("Ah", "Kh"), i("2h", "7h", "9s"))
    eq = equity_vs_random(i("Ah", "Ad"), [], 400, random.Random(1))
    assert 0.8 < eq < 0.9
