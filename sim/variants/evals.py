"""Hand evaluators of the rehearsal variants. Same signature and score format as sim.kit.cards.evaluate:
`evaluate(cards)` takes card INTS (kit numbering: rank = c >> 2, 0 = deuce ... 12 = ace; suit = c & 3) and returns
an int, higher is better, equal = tie, score >> 20 = category. Pure standard library.

* evaluate_any(cards): standard 52-card ranking for ANY number of cards >= 5 (the kit evaluator is exact for 5-7
  cards only; with 8+ cards it can pick the wrong quads and ignores a second flush suit). Used by `toss`, where the
  board can hold 6 to 12 cards.
* evaluate_short(cards): 36-card ranking (6..A). Flush beats full house; A-6-7-8-9 is a straight (the lowest one,
  ranked as a 9-high straight so it loses to 6-7-8-9-T); straight flush likewise. Three of a kind beats a straight
  only when built with trips_beat_straight=True (the Triton rule); the default keeps the straight above trips.
  Categories of the default short-deck ranking: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 full house,
  6 flush, 7 quads, 8 straight flush.
"""
try:
    from sim.kit.cards import POP, TOP1, TOP2, TOP3, TOP5, STRAIGHT as STRAIGHT52, CARD_INT, CARD_STR
except ImportError:   # pragma: no cover
    from cards import POP, TOP1, TOP2, TOP3, TOP5, STRAIGHT as STRAIGHT52, CARD_INT, CARD_STR

CATEGORY_NAMES_52 = ("high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads",
                     "straight flush")
CATEGORY_NAMES_SHORT = ("high card", "pair", "two pair", "trips", "straight", "full house", "flush", "quads",
                        "straight flush")
CATEGORY_NAMES_SHORT_TRIPS_OVER_STRAIGHT = ("high card", "pair", "two pair", "straight", "trips", "full house",
                                            "flush", "quads", "straight flush")

# short deck: ranks 6..A are 4..12; straights 6-T .. T-A plus the wheel A-6-7-8-9 (bits 12,4,5,6,7)
STRAIGHT_SHORT = [-1] * 8192
for _m in range(8192):
    _h = -1
    for _r in range(12, 7, -1):
        if (_m >> (_r - 4)) & 31 == 31:
            _h = _r
            break
    if _h < 0 and (_m & 0x10F0) == 0x10F0:
        _h = 7
    STRAIGHT_SHORT[_m] = _h


def _build(straight, cat_trips, cat_straight, cat_fh, cat_flush):
    def evaluate(cards):
        cnt = [0] * 13
        sm = [0, 0, 0, 0]
        for c in cards:
            r = c >> 2
            cnt[r] += 1
            sm[c & 3] |= 1 << r
        allm = sm[0] | sm[1] | sm[2] | sm[3]
        best_sf = -1
        best_fl = 0
        for m in sm:
            if POP[m] >= 5:
                h = straight[m]
                if h > best_sf:
                    best_sf = h
                t5 = TOP5[m]
                if t5 > best_fl:
                    best_fl = t5
        if best_sf >= 0:
            return (8 << 20) | best_sf
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
        if q >= 0:
            return (7 << 20) | (q << 16) | (TOP1[allm & ~(1 << q)] << 12)
        fh = -1
        if t1 >= 0 and (t2 >= 0 or p1 >= 0):
            fh = (cat_fh << 20) | (t1 << 16) | ((t2 if t2 > p1 else p1) << 12)
        fl = (cat_flush << 20) | best_fl if best_fl else -1
        if cat_fh > cat_flush:
            if fh >= 0:
                return fh
            if fl >= 0:
                return fl
        else:
            if fl >= 0:
                return fl
            if fh >= 0:
                return fh
        st = straight[allm]
        # trips vs straight order depends on the ranking (cat_trips / cat_straight)
        if cat_trips > cat_straight:
            if t1 >= 0:
                return (cat_trips << 20) | (t1 << 16) | (TOP2[allm & ~(1 << t1)] << 8)
            if st >= 0:
                return (cat_straight << 20) | st
        else:
            if st >= 0:
                return (cat_straight << 20) | st
            if t1 >= 0:
                return (cat_trips << 20) | (t1 << 16) | (TOP2[allm & ~(1 << t1)] << 8)
        if p2 >= 0:
            return (2 << 20) | (p1 << 16) | (p2 << 12) | (TOP1[allm & ~(1 << p1) & ~(1 << p2)] << 8)
        if p1 >= 0:
            return (1 << 20) | (p1 << 16) | (TOP3[allm & ~(1 << p1)] << 4)
        return TOP5[allm]
    return evaluate


# standard ranking, any number of cards: trips 3 < straight 4 < flush 5 < full house 6
evaluate_any = _build(STRAIGHT52, 3, 4, 6, 5)
evaluate_any.__doc__ = "Standard 52-card ranking for 5 or more cards (kit score format)."
# short deck: trips 3 < straight 4 < full house 5 < flush 6
evaluate_short = _build(STRAIGHT_SHORT, 3, 4, 5, 6)
evaluate_short.__doc__ = "36-card short-deck ranking for 5 or more cards: flush > full house, A-6-7-8-9 straight."
# Triton variant: straight 3 < trips 4
evaluate_short_trips = _build(STRAIGHT_SHORT, 4, 3, 5, 6)
evaluate_short_trips.__doc__ = "Short-deck ranking with three of a kind above a straight."


def evaluate_str(cards, evaluate=evaluate_any):
    """Evaluate card strings like ['Ah', 'Kh', ...] with any of the evaluators above."""
    return evaluate([CARD_INT[s] for s in cards])


SHORT_DECK = tuple(c for c in range(52) if (c >> 2) >= 4)   # 36 cards, 6..A
FULL_DECK = tuple(range(52))
