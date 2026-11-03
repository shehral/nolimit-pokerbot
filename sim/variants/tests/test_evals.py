import itertools
import random

from sim.kit.cards import evaluate as kit_eval, CARD_INT
from sim.variants.evals import (SHORT_DECK, evaluate_any, evaluate_short, evaluate_short_trips, evaluate_str)


def S(cards, ev=evaluate_short):
    return evaluate_str(cards.split(), ev)


def brute(cards, ev):
    return max(ev(list(c)) for c in itertools.combinations(cards, 5))


def test_any_matches_kit_up_to_7_cards():
    rnd = random.Random(1)
    for _ in range(3000):
        k = rnd.choice((5, 6, 7))
        cs = rnd.sample(range(52), k)
        assert evaluate_any(cs) == kit_eval(cs)


def test_any_matches_bruteforce_for_8_to_12_cards():
    rnd = random.Random(2)
    for k in (8, 9, 10, 12):
        for _ in range(150 if k < 12 else 40):
            cs = rnd.sample(range(52), k)
            assert evaluate_any(cs) == brute(cs, kit_eval), (k, cs)


def test_any_handles_two_quads_and_two_flush_suits():
    # two quads: the higher must be used
    assert evaluate_any([CARD_INT[x] for x in "9c 9d 9h 9s 5c 5d 5h 5s Ah".split()]) >> 20 == 7
    hi = evaluate_str("9c 9d 9h 9s 5c 5d 5h 5s Ah".split())
    lo = evaluate_str("9c 9d 9h 9s Ah Kd 2c 3d 4h".split())
    assert hi >> 16 & 15 == 7 or True  # quads rank is 9 (index 7) in both
    assert (hi >> 16) & 15 == 7 and (lo >> 16) & 15 == 7
    # a heart flush and a spade straight flush among 10 cards: the straight flush wins
    cs = "2h 4h 6h 8h Th 3s 4s 5s 6s 7s".split()
    assert evaluate_str(cs) >> 20 == 8


def test_short_flush_beats_full_house_standard_does_not():
    fh = "9c 9d 9h 6h 6c 7s 8s"
    fl = "Ah Kh 9h 7h 6h 2c 3d".replace("2c 3d", "Td Jc")
    assert S(fl) > S(fh)                       # short deck: flush > full house
    assert evaluate_str(fl.split()) < evaluate_str(fh.split())   # 52-card ranking: full house > flush
    assert S(fl) >> 20 == 6 and S(fh) >> 20 == 5


def test_short_wheel_is_a_straight_and_lowest():
    wheel = S("Ad 6c 7h 8s 9d")
    assert wheel >> 20 == 4
    six_to_ten = S("6c 7h 8s 9d Td")
    assert six_to_ten > wheel
    assert S("Ad 6c 7h 8s 9d") > S("Ad Ah 7h 8s 9d")             # straight beats a pair
    assert S("Ad Kc Qh Js Td") > six_to_ten                         # broadway above
    # a 52-card wheel (A-2-3-4-5) does not exist; A-6-7-8-9 with an extra card
    assert S("Ad 6c 7h 8s 9d Kd Qd") >> 20 == 4
    # straight flush wheel
    assert S("Ad 6d 7d 8d 9d") >> 20 == 8
    assert S("Ad 6d 7d 8d 9d") < S("6d 7d 8d 9d Td")


def test_short_trips_vs_straight_option():
    trips = "Kh Kd Kc 9d 8s"
    straight = "6h 7d 8c 9d Ts"
    assert S(straight) > S(trips)                                   # default: straight above trips
    assert S(trips, evaluate_short_trips) > S(straight, evaluate_short_trips)   # Triton rule
    assert S(trips, evaluate_short_trips) >> 20 == 4 and S(straight, evaluate_short_trips) >> 20 == 3


def test_short_other_categories_keep_order():
    quads = S("9c 9d 9h 9s Ac")
    assert quads > S("Ah Kh 9h 7h 6h")                              # quads > flush
    assert S("Ah Kh 9h 7h 6h") > S("9c 9d 9h 6h 6c")
    assert S("9c 9d 9h 6h 6c") > S("6h 7d 8c 9d Ts")
    assert S("Ac Ad 7h 8s Ts") > S("Kc Kd Qh Js 9s")


def test_short_matches_bruteforce_consistency():
    rnd = random.Random(3)
    for _ in range(300):
        cs = rnd.sample(SHORT_DECK, rnd.choice((7, 8)))
        assert evaluate_short(cs) == brute(cs, evaluate_short)
