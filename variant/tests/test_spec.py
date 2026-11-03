"""VariantSpec, spec_from_kit and the evaluator probe; evaluators against independent brute force."""
import random
from itertools import combinations

from sim.kit.cards import evaluate as kit_evaluate
from variant.cards import (STANDARD_ORDER, STRAIGHT, evaluate52, evaluate_short, make_ranked, STD_CODEC)
from variant.spec import (RequestType, VariantSpec, nlhe_spec, probe_evaluator, short_deck_spec, spec_from_kit,
                          three_keep_two_spec, toss_spec)


def test_evaluate52_is_the_kit_evaluator():
    rng = random.Random(5)
    for _ in range(20000):
        h = rng.sample(range(52), rng.choice((5, 6, 7)))
        assert evaluate52(h) == kit_evaluate(h)
        assert make_ranked(STRAIGHT, STANDARD_ORDER)(h) == evaluate52(h)


def _brute_short5(cards):
    """Independent 36-card 5-card ranking: categories with flush above full house; A-6-7-8-9 straight."""
    rs = sorted((c >> 2 for c in cards), reverse=True)
    flush = len(set(c & 3 for c in cards)) == 1
    cnt = {r: rs.count(r) for r in rs}
    shape = sorted(cnt.values(), reverse=True)
    uniq = sorted(cnt, reverse=True)
    straight = None
    if len(uniq) == 5 and uniq[0] - uniq[4] == 4:
        straight = uniq[0]
    if uniq == [12, 7, 6, 5, 4]:
        straight = 7
    by = sorted(cnt, key=lambda r: (cnt[r], r), reverse=True)
    if straight is not None and flush:
        return (8, straight)
    if shape[0] == 4:
        return (7, by)
    if flush:
        return (6, uniq)
    if shape[:2] == [3, 2]:
        return (5, by)
    if straight is not None:
        return (4, straight)
    if shape[0] == 3:
        return (3, by)
    if shape[:2] == [2, 2]:
        return (2, by)
    if shape[0] == 2:
        return (1, by)
    return (0, uniq)


def test_short_deck_evaluator_vs_brute_force():
    deck = list(short_deck_spec().deck)
    rng = random.Random(9)
    for _ in range(3000):
        a, b = rng.sample(deck, 7), rng.sample(deck, 7)
        ba = max(_brute_short5(list(c)) for c in combinations(a, 5))
        bb = max(_brute_short5(list(c)) for c in combinations(b, 5))
        ea, eb = evaluate_short(a), evaluate_short(b)
        assert (ba > bb) == (ea > eb) and (ba == bb) == (ea == eb)


def test_probe_nlhe_and_short():
    rep = probe_evaluator(nlhe_spec())
    assert rep["ok"] and rep["accepts"][7] is True
    assert rep["category_order"][-1] == "straight_flush"
    assert rep["flush_beats_full_house"] is False and rep["wheel_is_straight"] is True
    rs = probe_evaluator(short_deck_spec())
    assert rs["ok"] and rs["flush_beats_full_house"] is True and rs["a6789_is_straight"] is True


def test_probe_flags_a_bad_evaluator():
    def five_only(cards):
        if len(cards) != 5:
            raise ValueError("5 cards only")
        return evaluate52(cards)
    spec, rep = spec_from_kit(five_only)
    assert not rep["ok"] and any("fails on 7" in e for e in rep["errors"])


def test_spec_from_kit_nlhe_worked_example():
    spec, rep = spec_from_kit(kit_evaluate, name="nlhe")
    assert rep["ok"] and len(spec.deck) == 52 and spec.n_kept == 2 and spec.board_cards == 5 and spec.separable
    assert spec.summary().startswith("nlhe: deck 52 cards")


def test_keep_and_shares():
    s3 = three_keep_two_spec()
    cd = s3.codec
    assert sorted(s3.keep(tuple(cd.ints("As Ah 7d".split())))) == sorted(cd.ints("As Ah".split()))
    t = toss_spec()
    assert not t.separable
    hands = [tuple(cd.ints("As Ah 7d".split())), tuple(cd.ints("Ks Kh 2c".split()))]
    board = cd.ints("Ac 9d 3s Td".split())
    sh = t.shares(hands, board)
    assert abs(sum(sh) - 1.0) < 1e-9 and sh[0] == 1.0


def test_request_type_validation():
    RequestType("bid", "BidAction", response="int")
    try:
        RequestType("dance")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind accepted")


def test_evaluate_upto9_vs_brute_force():
    from variant.cards import evaluate_upto9
    rng = random.Random(13)
    brute = lambda h: max(evaluate52(list(c)) for c in combinations(h, 5))
    for _ in range(1500):
        n = rng.choice((5, 6, 7, 8, 9))
        a, b = rng.sample(range(52), n), rng.sample(range(52), n)
        assert (brute(a) > brute(b)) == (evaluate_upto9(a) > evaluate_upto9(b))
        assert (brute(a) == brute(b)) == (evaluate_upto9(a) == evaluate_upto9(b))
    two_quads = STD_CODEC.ints("Ac Ad Ah As Kc Kd Kh Ks".split())
    assert evaluate_upto9(two_quads) > evaluate_upto9(STD_CODEC.ints("Kc Kd Kh Ks Ac Ad Qh Qs".split()))
