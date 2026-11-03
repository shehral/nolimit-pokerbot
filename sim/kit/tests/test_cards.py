"""Evaluator: known hands, and agreement with an independent brute-force evaluator on random 7-card hands."""
import itertools
import random

from sim.kit.cards import CARD_INT, CARD_STR, category, evaluate, evaluate_str


def ref5(cards):
    """Independent 5-card evaluator returning a comparable tuple (category, tiebreak ranks)."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_hi = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_hi = 3
    counts = sorted(((ranks.count(r), r) for r in uniq), reverse=True)
    if straight_hi is not None and flush:
        return (8, straight_hi)
    if counts[0][0] == 4:
        return (7, counts[0][1], counts[1][1])
    if counts[0][0] == 3 and counts[1][0] == 2:
        return (6, counts[0][1], counts[1][1])
    if flush:
        return (5,) + tuple(ranks)
    if straight_hi is not None:
        return (4, straight_hi)
    if counts[0][0] == 3:
        return (3, counts[0][1]) + tuple(r for _, r in counts[1:])
    if counts[0][0] == 2 and counts[1][0] == 2:
        return (2, counts[0][1], counts[1][1], counts[2][1])
    if counts[0][0] == 2:
        return (1, counts[0][1]) + tuple(r for _, r in counts[1:])
    return (0,) + tuple(ranks)


def ref7(cards):
    return max(ref5(c) for c in itertools.combinations(cards, 5))


def test_card_strings_roundtrip():
    assert len(set(CARD_STR)) == 52
    assert CARD_STR[CARD_INT["Ah"]] == "Ah" and CARD_INT["2c"] == 0 and CARD_INT["As"] == 51


def test_known_categories():
    assert category(["Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"]) == 8
    assert category(["Ah", "2h", "3h", "4h", "5h"]) == 8          # steel wheel
    assert category(["Ac", "Ad", "Ah", "As", "Kd"]) == 7
    assert category(["Ac", "Ad", "Ah", "Ks", "Kd"]) == 6
    assert category(["2h", "7h", "9h", "Jh", "Kh"]) == 5
    assert category(["Ac", "2d", "3h", "4s", "5d"]) == 4          # wheel
    assert category(["7c", "7d", "7h", "As", "Kd"]) == 3
    assert category(["7c", "7d", "8h", "8s", "Kd"]) == 2
    assert category(["7c", "7d", "8h", "9s", "Kd"]) == 1
    assert category(["2c", "7d", "8h", "9s", "Kd"]) == 0
    assert evaluate_str(["Ac", "2d", "3h", "4s", "5d"]) < evaluate_str(["2c", "3d", "4h", "5s", "6d"])


def test_matches_bruteforce_on_random_7card_hands():
    rng = random.Random(7)
    hands = [rng.sample(range(52), 7) for _ in range(2500)]
    ours = [evaluate(h) for h in hands]
    refs = [ref7(h) for h in hands]
    for a, b in zip(ours, refs):
        assert a >> 20 == b[0]
    for i in range(0, len(hands) - 1):
        j = i + 1
        assert (ours[i] > ours[j]) == (refs[i] > refs[j])
        assert (ours[i] == ours[j]) == (refs[i] == refs[j])
