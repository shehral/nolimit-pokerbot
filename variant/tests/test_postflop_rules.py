"""Variant ranking probe and the variant-aware postflop classifier (variant/bridge.py, POSTFLOP_VARIANT notes),
checked by brute force against itertools-based reference evaluators written independently here.

* short deck: evaluate_short (variant/cards) orders random 7-card hands exactly like a reference that scores every
  5-card subset with the short-deck rules (flush > full house, A-6-7-8-9 the lowest straight, no 2-5 in the deck);
  probe_ranking recovers those rules (and the Triton trips > straight flip) from the evaluators alone
* SPEC_PARAMS carry the ranking and spec_from_params rebuilds an evaluator with it; old params keep the deck guess
* must-use (Omaha 2 + 3): the classifier's class agrees with the reference's best legal 5-card category
* keep k of n: the classifier is the best kept subset, discards joining the board when they go there (itertools)
* any combination of 3 hole cards: made-hand properties against the reference over every 5-card subset
"""
import random
from itertools import combinations

from variant.bridge import class_any, class_must, postflop_classifier, rules_from_ranking
from variant.cards import STRAIGHTS_SHORT, evaluate52, evaluate_short, make_ranked, STRAIGHT_SHORT, \
    SHORT_DECK_ORDER
from variant.spec import (VariantSpec, is_standard_ranking, nlhe_spec, omaha_spec, probe_ranking, short_deck_spec,
                          spec_from_params, three_keep_two_spec, toss_spec)

SHORT = tuple(c for c in range(52) if c >> 2 >= 4)
HIGH, PAIR, TWO, TRIPS, STRAIGHT, FLUSH, FULL, QUADS, SFLUSH = range(9)   # reference category ids (standard names)


# ------------------------------------------------------------------ independent reference evaluators
def ref5(cards, short=False):
    """(category rank, tiebreak ranks) of exactly 5 cards; categories ranked by the game's order."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    st = -1
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            st = uniq[0]
        elif not short and uniq == [12, 3, 2, 1, 0]:
            st = 3
        elif short and uniq == [12, 7, 6, 5, 4]:
            st = 7
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [g[1] for g in groups]
    tb = [g[0] for g in groups]
    if st >= 0 and flush:
        cat, tb = SFLUSH, [st]
    elif shape[0] == 4:
        cat = QUADS
    elif shape[:2] == [3, 2]:
        cat = FULL
    elif flush:
        cat, tb = FLUSH, ranks
    elif st >= 0:
        cat, tb = STRAIGHT, [st]
    elif shape[0] == 3:
        cat = TRIPS
    elif shape[:2] == [2, 2]:
        cat = TWO
    elif shape[0] == 2:
        cat = PAIR
    else:
        cat = HIGH
    order = (HIGH, PAIR, TWO, TRIPS, STRAIGHT, FULL, FLUSH, QUADS, SFLUSH) if short else \
        (HIGH, PAIR, TWO, TRIPS, STRAIGHT, FLUSH, FULL, QUADS, SFLUSH)
    return (order.index(cat), tuple(tb)), cat


def ref_best(cards, short=False):
    return max((ref5(c, short) for c in combinations(cards, 5)), key=lambda x: x[0])


def ref_omaha(hole, board):
    return max((ref5(h + b) for h in combinations(hole, 2) for b in combinations(board, 3)), key=lambda x: x[0])


def _sign(x):
    return (x > 0) - (x < 0)


# ------------------------------------------------------------------ short-deck ranking
def test_short_deck_ranking_brute_force():
    rng = random.Random(2026)
    from sim.variants.evals import evaluate_short as engine_short
    for _ in range(2500):
        a, b = rng.sample(SHORT, 7), rng.sample(SHORT, 7)
        ra, rb = ref_best(a, True)[0], ref_best(b, True)[0]
        want = _sign((ra > rb) - (ra < rb))
        assert _sign(evaluate_short(a) - evaluate_short(b)) == want, (a, b)
        assert _sign(engine_short(a) - engine_short(b)) == want, (a, b)
        # the category the evaluator reports is the reference's (score >> 20 = position in the short-deck order)
        assert evaluate_short(a) >> 20 == ra[0]


def test_short_deck_rules_explicit():
    c = lambda s: [((("23456789TJQKA".index(x[0])) << 2) | "cdhs".index(x[1])) for x in s.split()]
    assert evaluate_short(c("As Js 8s 7s 6s")) > evaluate_short(c("Ks Kh Kd Qc Qs"))       # flush > full house
    assert evaluate52(c("As Js 8s 7s 6s")) < evaluate52(c("Ks Kh Kd Qc Qs"))               # not in hold'em
    wheel = evaluate_short(c("As 6h 7d 8c 9s"))
    assert wheel >> 20 == SHORT_DECK_ORDER.index(4)                                         # A6789 is a straight
    assert wheel < evaluate_short(c("6s 7h 8d 9c Ts"))                                       # ... the lowest one
    assert wheel > evaluate_short(c("Ks Kh Kd 9c 7s"))                                       # above trips
    assert all(x >> 2 >= 4 for x in short_deck_spec().deck) and len(short_deck_spec().deck) == 36   # no 2-5


def test_probe_ranking_recovers_rules():
    from sim.kit.cards import evaluate as kit
    from sim.variants.evals import evaluate_any, evaluate_short as eng_short, evaluate_short_trips
    for ev, deck in ((kit, range(52)), (evaluate_any, range(52)), (evaluate52, range(52))):
        rk = probe_ranking(ev, tuple(deck))
        assert is_standard_ranking(rk) and len(rk["straights"]) == 10
    for ev in (evaluate_short, eng_short):
        rk = probe_ranking(ev, SHORT)
        assert rk["flush_over_full_house"] and not rk["trips_over_straight"]
        assert tuple(rk["straights"]) == STRAIGHTS_SHORT and (1 << 12 | 0b11110000) in rk["straights"]
    rk = probe_ranking(evaluate_short_trips, SHORT)
    assert rk["flush_over_full_house"] and rk["trips_over_straight"]
    assert probe_ranking(lambda cs: 1 / 0, SHORT) is None                 # never raises


def test_spec_params_carry_the_ranking():
    p = short_deck_spec().spec_params()
    assert p["ranking"]["flush_over_full_house"]
    sp = spec_from_params(p)
    rng = random.Random(5)
    for _ in range(500):
        a, b = rng.sample(SHORT, 7), rng.sample(SHORT, 7)
        assert _sign(sp.evaluator(a) - sp.evaluator(b)) == _sign(evaluate_short(a) - evaluate_short(b))
    # an evaluator the deck guess would get wrong (Triton): the ranking entry wins
    from sim.variants.evals import evaluate_short_trips
    tri = VariantSpec(name="triton", removed_ranks="2345", evaluator=evaluate_short_trips).spec_params()
    assert tri["ranking"]["trips_over_straight"]
    assert probe_ranking(spec_from_params(tri).evaluator, SHORT)["trips_over_straight"]
    # old tables modules (no "ranking"): the deck guess, as before
    p.pop("ranking")
    assert probe_ranking(spec_from_params(p).evaluator, SHORT)["flush_over_full_house"]
    assert spec_from_params(nlhe_spec().spec_params()).evaluator is evaluate52


# ------------------------------------------------------------------ the classifier
def test_routing_none_for_plain_holdem_specs():
    assert postflop_classifier(nlhe_spec()) is None
    assert postflop_classifier(VariantSpec(name="db", n_boards=2)) is None
    assert postflop_classifier(VariantSpec(name="ante", ante_bb=0.125)) is None
    for s in (short_deck_spec(), omaha_spec(), toss_spec(), three_keep_two_spec(keep_street=2)):
        assert postflop_classifier(s) is not None, s.name


def test_short_deck_classes():
    f = postflop_classifier(short_deck_spec())
    c = lambda s: [((("23456789TJQKA".index(x[0])) << 2) | "cdhs".index(x[1])) for x in s.split()]
    # the A-6-7-8-9 straight is a straight, and the deck's lowest one: class 2 (measured, POSTFLOP_VARIANT notes s4)
    assert f(c("As 6h"), c("7d 8c 9s"), 1) == 2
    assert f(c("As 6h"), c("7d 8c 9s"), 3) == 2                 # not a draw: made (river, no cards to come)
    assert class_any(c("As 6h"), c("7d 8c 9s"), rules_from_ranking(None), False) == 0   # hold'em reading: air
    assert f(c("9s 9h"), c("9d Ac 6s 7h 8d"), 3) == 3           # a set inside it still counts as 3
    assert f(c("Ts 6h"), c("7d 8c 9s"), 1) == 3                 # 6-T: a real straight, 3
    assert f(c("6s 7h"), c("8d 9c Ks"), 1) == 1                 # 6789: open-ended (A or T)
    assert f(c("6s 7h"), c("8d 9c Ks"), 3) == 0                 # no draws on the river
    assert f(c("Ah 7h"), c("8d 9c Ks"), 1) == 0                 # A789: one straight only, a gutshot
    assert f(c("Ts 9h"), c("6d 7c 8s"), 1) == 3                 # 6-T straight, as in hold'em
    assert f(c("Ad 9h"), c("6d 7c 8s"), 1) == 2                 # A6789 with the ace and the nine
    R = f.rules
    assert R.weak_heights == {7} and rules_from_ranking(None).weak_heights == frozenset()
    om = postflop_classifier(VariantSpec(name="omaha_short", hole_dealt=4, must_use=(2, 3), removed_ranks="2345",
                                         evaluator=evaluate_short))
    assert om(c("As 9h Kd Kc"), c("6d 7c 8s"), 1) == 2          # Omaha short deck: the same reading
    assert om(c("Ts 9h Kd Qc"), c("6d 7c 8s"), 1) == 3
    assert f(c("Ks Qs"), c("Ts 7s 6s"), 1) == 3                 # flush with hole cards (flush > full house)


def test_must_use_vs_reference():
    R = rules_from_ranking(None)
    rng = random.Random(11)
    f = postflop_classifier(omaha_spec())
    seen = set()
    for _ in range(3000):
        cs = rng.sample(range(52), 9)
        hole, board = cs[:4], cs[4:]
        (rank, _), cat = ref_omaha(hole, board)
        cls = f(hole, board, 3)
        assert cls == max(class_must(s, board, R, 2, 3, False) for s in combinations(hole, 2))
        if cat in (STRAIGHT, QUADS, SFLUSH):
            assert cls == 3, (hole, board, cat)
        if cat == FLUSH:
            assert cls >= 1, (hole, board)
        if cat == FULL:
            assert cls in (2, 3), (hole, board)
        if cat <= PAIR:
            assert cls <= 2, (hole, board, cat)
        if cat == HIGH:
            assert cls == 0, (hole, board)
        seen.add(cat)
    assert {HIGH, PAIR, TWO, TRIPS, STRAIGHT, FLUSH, FULL} <= seen
    # the rule matters: one hole card + four board cards of a suit is a hold'em flush, not an Omaha one
    c = lambda s: [((("23456789TJQKA".index(x[0])) << 2) | "cdhs".index(x[1])) for x in s.split()]
    hole, board = c("As Kd 7c 2h"), c("Qs Js 8s 3s 4d")
    assert ref_omaha(hole, board)[1] != FLUSH and f(hole, board, 3) < 3
    assert class_any(hole, board, R) == 3


def test_keep_rule_vs_itertools():
    R = rules_from_ranking(None)
    rng = random.Random(12)
    muck = postflop_classifier(three_keep_two_spec(keep_street=2))          # keep after the turn: 3 cards on flop
    tb = postflop_classifier(toss_spec())                                    # discards go to the board
    for _ in range(3000):
        cs = rng.sample(range(52), 3 + 3)
        hole, board = cs[:3], cs[3:]
        want = max(class_any(list(s), board, R, True) for s in combinations(hole, 2))
        assert muck(hole, board, 1) == want
        want2 = max(class_any(list(s), board + [x for x in hole if x not in s], R, True)
                    for s in combinations(hole, 2))
        assert tb(hole, board, 1) == want2
        assert tb(hole[:2], board, 1) == class_any(hole[:2], board, R, True)   # after the toss: the kept 2


def test_any_combination_three_hole_cards():
    R = rules_from_ranking(None)
    rng = random.Random(13)
    n_st = 0
    for _ in range(4000):
        cs = rng.sample(range(52), 8)
        hole, board = cs[:3], cs[3:]
        cls = class_any(hole, board, R, False)
        (rank, tb), cat = ref_best(hole + board)
        (brank, btb), bcat = ref5(board)
        if cat <= PAIR:
            assert cls <= 2, (hole, board)
        if cat == HIGH:
            assert cls == 0, (hole, board)
        if (rank, tb) <= (brank, btb) and bcat < STRAIGHT:
            # the hole cards do not improve the board: never strong. (Inherited hand_class reading, kept for
            # identity: on a board that is itself a straight or better, a set or two pair still counts as 3.)
            assert cls <= 2, (hole, board)
        flush_possible = any(sum(1 for x in hole + board if x & 3 == s) >= 5 for s in range(4))
        if cat == STRAIGHT and not flush_possible and (bcat != STRAIGHT or tb[0] > btb[0]):
            n_st += 1
            assert cls == 3, (hole, board)          # a straight the hole cards make
    assert n_st > 20
