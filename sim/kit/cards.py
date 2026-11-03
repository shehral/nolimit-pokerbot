"""Cards and the 7-card hand evaluator of the mock starter kit.

Card ints 0..51: rank = c >> 2 (0 = deuce ... 12 = ace), suit = c & 3 (0 = clubs, 1 = diamonds, 2 = hearts,
3 = spades). Card strings: rank char + suit char, e.g. "Ah", "Td", "2c".

The evaluator is ported unchanged from an earlier research script (not published) (tested there; re-tested in
tests/test_cards.py against an independent brute-force 5-card evaluator). evaluate() takes 5 to 7 card ints and
returns an int score: higher is better, equal scores tie, and score >> 20 is the hand category (0 high card ...
8 straight flush). Pure standard library, about 4 us per 7-card call on an M4.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD_STR = tuple(RANKS[c >> 2] + SUITS[c & 3] for c in range(52))
CARD_INT = {s: i for i, s in enumerate(CARD_STR)}
CATEGORY_NAMES = ("high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads",
                  "straight flush")


def card_str(c):
    """Card int -> string, e.g. 50 -> 'Ah'."""
    return CARD_STR[c]


def card_int(s):
    """Card string -> int, e.g. 'Ah' -> 50. Raises KeyError on a malformed card."""
    return CARD_INT[s]


# ---------------------------------------------------------------- evaluator tables (from sng_engine.py)
STRAIGHT = [-1] * 8192
for _m in range(8192):
    _h = -1
    for _r in range(12, 3, -1):
        if (_m >> (_r - 4)) & 31 == 31:
            _h = _r
            break
    if _h < 0 and (_m & 0x100F) == 0x100F:
        _h = 3
    STRAIGHT[_m] = _h
POP = [bin(_m).count("1") for _m in range(8192)]


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


TOP5 = [_topk(_m, 5) for _m in range(8192)]
TOP3 = [_topk(_m, 3) for _m in range(8192)]
TOP2 = [_topk(_m, 2) for _m in range(8192)]
TOP1 = [_topk(_m, 1) for _m in range(8192)]


def evaluate(cards):
    """Score for 5..7 card ints; higher is better. category = score >> 20."""
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


def evaluate_str(cards):
    """Same as evaluate() for card strings, e.g. evaluate_str(['Ah', 'Kh', 'Qh', 'Jh', 'Th'])."""
    return evaluate([CARD_INT[s] for s in cards])


def category(cards):
    """Hand category 0..8 (see CATEGORY_NAMES) for 5..7 card strings or ints."""
    if cards and isinstance(cards[0], str):
        return evaluate_str(cards) >> 20
    return evaluate(cards) >> 20
