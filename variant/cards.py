"""Card codec and reference evaluators for the variant layer. Pure standard library.

Card ints use the kit numbering by default: c = rank * 4 + suit, rank 0 = deuce ... 12 = ace, suit 0..3 = c, d, h, s
(so rank = c >> 2, suit = c & 3, exactly sim/kit/cards.py and bot/adapter.py). A short deck keeps these ints and just
removes cards, so any kit-numbered evaluator works unchanged. Exotic decks (more than 4 suits, more than 13 ranks) use
a Codec with a larger suit slot: c = rank * slot + suit.

Evaluators: callable(cards: list[int]) -> comparable, higher is better, equal = tie.
  evaluate52      the kit evaluator, copied unchanged from sim/kit/cards.py (exact for 5 to 7 cards)
  make_ranked(...)  the same algorithm with a configurable straight table and category order (short deck: flush above
                  full house, A-6-7-8-9 straight); exact for 5 to 9 cards, up to 4 suits (evaluate_short,
                  evaluate_upto9 = standard ranking for the 8-card hands of a toss variant)
  best_of_five(ev5) wraps any exact 5-card evaluator into an any-N-cards evaluator (max over 5-subsets; slow)
"""
from itertools import combinations

STD_RANKS = "23456789TJQKA"
STD_SUITS = "cdhs"


class Codec:
    """Card string <-> int. rank_chars low to high; suit_chars in suit-index order."""

    def __init__(self, rank_chars=STD_RANKS, suit_chars=STD_SUITS):
        self.rank_chars = rank_chars
        self.suit_chars = suit_chars
        self.slot = max(4, len(suit_chars))
        self._r = {ch: i for i, ch in enumerate(rank_chars)}
        self._r.update({ch.lower(): i for i, ch in enumerate(rank_chars)})
        if "T" in self._r:
            self._r["10"] = self._r["T"]
        self._s = {ch: i for i, ch in enumerate(suit_chars)}
        self._s.update({ch.upper(): i for i, ch in enumerate(suit_chars)})
        if suit_chars == STD_SUITS:
            self._s.update({"♣": 0, "♦": 1, "♥": 2, "♠": 3})

    def rank(self, c):
        return c // self.slot

    def suit(self, c):
        return c % self.slot

    def make(self, r, s):
        return r * self.slot + s

    def to_int(self, card):
        if isinstance(card, int) and not isinstance(card, bool):
            return card
        s = str(card).strip()
        return self.make(self._r[s[:-1]], self._s[s[-1]])

    def to_str(self, c):
        return self.rank_chars[self.rank(c)] + self.suit_chars[self.suit(c)]

    def ints(self, cards):
        return [self.to_int(c) for c in cards]


STD_CODEC = Codec()


# ---------------------------------------------------------------- kit evaluator (copied from sim/kit/cards.py)
def _straight_table(lowest_top, wheel_mask, wheel_top):
    t = [-1] * 8192
    for m in range(8192):
        h = -1
        for r in range(12, lowest_top - 1, -1):
            if (m >> (r - 4)) & 31 == 31:
                h = r
                break
        if h < 0 and wheel_mask and (m & wheel_mask) == wheel_mask:
            h = wheel_top
        t[m] = h
    return t


STRAIGHT = _straight_table(4, 0x100F, 3)              # 52 cards: 6-high .. A-high, wheel A2345 = 5-high (3)
STRAIGHT_SHORT = _straight_table(8, 0x10F0, 7)        # 36 cards (6..A): T-high .. A-high, A6789 = 9-high (7)
POP = [bin(m).count("1") for m in range(8192)]


def _topk(m, k):
    out = 0
    n = 0
    r = 12
    while r >= 0 and n < k:
        if m >> r & 1:
            out = (out << 4) | r
            n += 1
        r -= 1
    while n < k:
        out <<= 4
        n += 1
    return out


TOP5 = [_topk(m, 5) for m in range(8192)]
TOP3 = [_topk(m, 3) for m in range(8192)]
TOP2 = [_topk(m, 2) for m in range(8192)]
TOP1 = [_topk(m, 1) for m in range(8192)]


def evaluate52(cards):
    """Kit evaluator (sim/kit/cards.py, unchanged): 5..7 card ints, higher is better, score >> 20 = category."""
    s0 = s1 = s2 = s3 = 0
    cnt = [0] * 13
    for c in cards:
        r = c >> 2
        cnt[r] += 1
        su = c & 3
        if su == 0:
            s0 |= 1 << r
        elif su == 1:
            s1 |= 1 << r
        elif su == 2:
            s2 |= 1 << r
        else:
            s3 |= 1 << r
    allm = s0 | s1 | s2 | s3
    fm = 0
    for sm in (s0, s1, s2, s3):
        if POP[sm] >= 5:
            fm = sm
            break
    if fm:
        h = STRAIGHT[fm]
        if h >= 0:
            return (8 << 20) | h
    q = t1 = t2 = p1 = p2 = -1
    for r in range(12, -1, -1):
        c = cnt[r]
        if c == 4:
            q = r
        elif c == 3:
            if t1 < 0:
                t1 = r
            elif t2 < 0:
                t2 = r
        elif c == 2:
            if p1 < 0:
                p1 = r
            elif p2 < 0:
                p2 = r
    if q >= 0:
        return (7 << 20) | (q << 16) | (TOP1[allm & ~(1 << q)] << 12)
    if t1 >= 0 and (t2 >= 0 or p1 >= 0):
        return (6 << 20) | (t1 << 16) | ((t2 if t2 > p1 else p1) << 12)
    if fm:
        return (5 << 20) | TOP5[fm]
    st = STRAIGHT[allm]
    if st >= 0:
        return (4 << 20) | st
    if t1 >= 0:
        return (3 << 20) | (t1 << 16) | (TOP2[allm & ~(1 << t1)] << 8)
    if p2 >= 0:
        return (2 << 20) | (p1 << 16) | (p2 << 12) | (TOP1[allm & ~(1 << p1) & ~(1 << p2)] << 8)
    if p1 >= 0:
        return (1 << 20) | (p1 << 16) | (TOP3[allm & ~(1 << p1)] << 4)
    return TOP5[allm]


# standard category ids used by make_ranked (before re-ordering)
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT_C, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
STANDARD_ORDER = (HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT_C, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH)
SHORT_DECK_ORDER = (HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT_C, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH)


def make_ranked(straight=STRAIGHT, order=STANDARD_ORDER):
    """Evaluator with a custom straight table and category ranking. order lists the standard category ids from worst
    to best; the score's category (score >> 20) is the POSITION in that list. Exact for 5..9 cards (up to 9 cards
    only one suit can hold a flush), <= 4 suits."""
    rank_of_cat = [0] * 9
    for pos, cat in enumerate(order):
        rank_of_cat[cat] = pos
    R = rank_of_cat

    def evaluate(cards):
        sm = [0, 0, 0, 0]
        cnt = [0] * 13
        for c in cards:
            r = c >> 2
            cnt[r] += 1
            sm[c & 3] |= 1 << r
        allm = sm[0] | sm[1] | sm[2] | sm[3]
        fm = 0
        for m in sm:
            if POP[m] >= 5:
                fm = m
                break
        sf = straight[fm] if fm else -1
        q = t1 = t2 = p1 = p2 = -1
        for r in range(12, -1, -1):
            c = cnt[r]
            if c == 4:
                if q < 0:
                    q = r
            elif c == 3:
                if t1 < 0:
                    t1 = r
                elif t2 < 0:
                    t2 = r
            elif c == 2:
                if p1 < 0:
                    p1 = r
                elif p2 < 0:
                    p2 = r
        cand = []
        if sf >= 0:
            cand.append((R[STRAIGHT_FLUSH] << 20) | sf)
        if q >= 0:
            cand.append((R[QUADS] << 20) | (q << 16) | (TOP1[allm & ~(1 << q)] << 12))
        if t1 >= 0 and (t2 >= 0 or p1 >= 0):
            cand.append((R[FULL_HOUSE] << 20) | (t1 << 16) | ((t2 if t2 > p1 else p1) << 12))
        if fm:
            cand.append((R[FLUSH] << 20) | TOP5[fm])
        st = straight[allm]
        if st >= 0:
            cand.append((R[STRAIGHT_C] << 20) | st)
        if t1 >= 0:
            cand.append((R[TRIPS] << 20) | (t1 << 16) | (TOP2[allm & ~(1 << t1)] << 8))
        if p2 >= 0:
            cand.append((R[TWO_PAIR] << 20) | (p1 << 16) | (p2 << 12) | (TOP1[allm & ~(1 << p1) & ~(1 << p2)] << 8))
        if p1 >= 0:
            cand.append((R[PAIR] << 20) | (p1 << 16) | (TOP3[allm & ~(1 << p1)] << 4))
        cand.append((R[HIGH] << 20) | TOP5[allm])
        return max(cand)

    return evaluate


evaluate_short = make_ranked(STRAIGHT_SHORT, SHORT_DECK_ORDER)
evaluate_upto9 = make_ranked(STRAIGHT, STANDARD_ORDER)    # standard ranking, exact for 5..9 cards (toss boards)


def best_of_five(ev5):
    """Any-N-card evaluator from an exact 5-card evaluator (max over all 5-card subsets). Slow: use offline only."""
    def evaluate(cards):
        if len(cards) <= 5:
            return ev5(list(cards))
        return max(ev5(list(c)) for c in combinations(cards, 5))
    return evaluate
