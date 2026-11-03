"""HandSpace: NLHE reproduces the bot's 169 classes byte for byte; 3-keep-2 and short deck enumerate correctly."""
import random

import pytest

from bot import bot_tables as T
from bot import brain
from variant.handspace import HandSpace
from variant.spec import nlhe_spec, short_deck_spec, three_keep_two_spec, VariantSpec


@pytest.fixture(scope="module")
def nlhe():
    return HandSpace(nlhe_spec())


def test_nlhe_169_in_bot_order(nlhe):
    assert nlhe.n_classes == 169
    assert nlhe.labels == T.EQ_LABELS.split() == brain.LABELS
    assert nlhe.total_combos == 1326 and sum(nlhe.combos) == 1326
    assert nlhe.combos == [brain.combos(l) for l in nlhe.labels]


def test_nlhe_class_of_matches_brain(nlhe):
    for a in range(52):
        for b in range(52):
            if a != b:
                assert nlhe.class_of((a, b)) == brain.CLASS_OF[a * 52 + b]
    assert nlhe.label_of((50, 51)) == "AA" and nlhe.label_of((51, 47)) == "AKs" and nlhe.label_of((0, 21)) == "72o"


def test_nlhe_percentiles_match_brain(nlhe):
    assert nlhe.percentiles(T.PUSH_ORDER.split()) == brain.PUSH_PCT
    assert nlhe.percentiles(T.CALL_ORDER.split()) == brain.CALL_PCT


def _invariance(hs, trials=300, seed=3):
    rng = random.Random(seed)
    for _ in range(trials):
        h = rng.sample(hs.deck, hs.n)
        ci = hs.class_of(h)
        rng.shuffle(h)
        assert hs.class_of(h) == ci
        p = list(range(4))
        rng.shuffle(p)
        mapped = [c - c % 4 + p[c % 4] for c in h]
        assert hs.class_of(mapped) == ci


def test_three_keep_two():
    hs = HandSpace(three_keep_two_spec())
    assert hs.n_classes == 1755                 # 3-card suit-isomorphism classes of a 52-card deck
    assert hs.total_combos == 22100 and sum(hs.combos) == 22100
    cd = hs.spec.codec
    lab = lambda s: hs.label_of(cd.ints(s.split()))
    assert lab("As Ah Ks") == "AA|Ka"           # keep the aces; the king shares a suit with one of them
    assert lab("As Ah Kd") == "AA|Kc"
    assert lab("As Ah Ad") == "AA|Ac"
    assert lab("Ks Qs 2s") == "KQs|2a"
    assert lab("As Kh 7d") == "AKo|7c"
    trips = [l for l in hs.labels if l[0] == l[1] and len(l) == 5 and l[3] == l[0]]
    assert len(trips) == 13 and all(hs.combos[hs.label_index[t]] == 4 for t in trips)
    assert all(c > 0 for c in hs.combos)
    _invariance(hs)


def test_short_deck():
    hs = HandSpace(short_deck_spec())
    assert len(hs.deck) == 36
    assert hs.n_classes == 81 and hs.total_combos == 630 and sum(hs.combos) == 630
    assert hs.labels[:3] == ["AA", "AKs", "AKo"] and hs.labels[-1] == "66"
    assert not any(ch in l for l in hs.labels for ch in "2345")
    assert sorted(set(hs.combos)) == [4, 6, 12]
    _invariance(hs)


def test_percentiles_and_top_range():
    hs = HandSpace(short_deck_spec())
    pct = hs.percentiles(list(range(hs.n_classes)))
    assert 0 < min(pct) and max(pct) < 1
    r = hs.top_range(list(range(hs.n_classes)), 0.1)
    assert abs(sum(w * c for w, c in zip(r, hs.combos)) / hs.total_combos - 0.1) < 1e-9


def test_sampled_universe_summary_modes():
    spec = VariantSpec(name="five_keep_two", hole_dealt=5, hole_kept=2)
    hs = HandSpace(spec, discard_info="none", sample_n=4000)
    assert not hs.enumerated and hs.n_classes <= 169
    assert abs(sum(hs.combos) - hs.total_combos) < 1e-6 * hs.total_combos
    h = random.Random(1).sample(hs.deck, 5)
    assert 0 <= hs.class_of_safe(h) < hs.n_classes
