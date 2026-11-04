"""Bridge from a HandSpace + generated tables to the names bot/brain.py uses today. Pure standard library.

    from variant.bridge import brain_names
    names = brain_names(hs, tables_module)      # dict: LABELS, CLASS_ID, PUSH_PCT, CALL_PCT, EQ_GRID, EQ_ROWS,
                                                #       CLASS_OF (2-card pair table or None), class_of, eq_vs,
                                                #       JAM_FIT, CALL_FIT, TOTAL_COMBOS
For NLHE with bot/bot_tables.py these equal brain.py's module globals exactly (tests/test_bridge.py), so the
integration is a rename: brain.py keeps its logic and takes these values from the variant layer.
"""
import bisect

# Gate thresholds for swapping a generated tables module into the bot (verify 2026-11-03: a 90 s toss build had
# SANITY ok but per-class stderr 0.024 and a different top class than a 300 s build; gate on the error, not only ok)
MAX_STDERR_A = 0.015          # per-class standard error of equity vs random (stage a), from META["stderr"]["a"]


def gate_tables(T, max_stderr=MAX_STDERR_A, require_final=False):
    """(ok, reasons) for a generated tables module: SANITY ok, the equity stage done, META.stderr["a"] at most
    max_stderr (a missing stderr fails: the tables are provisional), and optionally a final (not checkpoint) write."""
    reasons = []
    san = getattr(T, "SANITY", None) or {}
    meta = getattr(T, "META", None) or {}
    if not san.get("ok"):
        reasons.append(f"SANITY not ok (failed {san.get('failed')})")
    done = meta.get("stage_done") or {}
    if not done.get("a"):
        reasons.append("stage a (equity vs random) not done: tables are a static-score placeholder")
    se = (meta.get("stderr") or {}).get("a")
    if se is None:
        reasons.append("no stderr for stage a")
    elif se > max_stderr:
        reasons.append(f"stderr a {se:.4f} > {max_stderr} (build longer: 300-900 s for 3-card or rollout games)")
    if require_final and not meta.get("final"):
        reasons.append("checkpoint, not the final write")
    src = str((getattr(T, "SOURCES", None) or {}).get("EQ_TABLE", ""))
    if src.startswith("linear model"):
        reasons.append("EQ_TABLE is the provisional linear model (up to 0.3 off at J = 3%); wait for stage b or m")
    for name in ("EQ_LABELS", "PUSH_ORDER", "CALL_ORDER", "EQ_GRID", "EQ_TABLE"):
        if not hasattr(T, name):
            reasons.append(f"missing {name}")
    return (not reasons), reasons


def brain_names(hs, T):
    """T: a tables module (or any object) with EQ_LABELS, PUSH_ORDER, CALL_ORDER, EQ_GRID, EQ_TABLE and optionally
    JAM_FIT, CALL_FIT. Rows are re-ordered to the HandSpace's class order if the labels come in another order."""
    labels = list(hs.labels)
    t_labels = T.EQ_LABELS.split()
    if sorted(t_labels) != sorted(labels):
        raise ValueError("tables do not belong to this HandSpace (label sets differ)")
    pos = {lab: i for i, lab in enumerate(t_labels)}
    rows = [list(T.EQ_TABLE[pos[lab]]) for lab in labels]
    push_pct = hs.percentiles(T.PUSH_ORDER.split())
    call_pct = hs.percentiles(T.CALL_ORDER.split())
    grid = list(T.EQ_GRID)
    g0, gn = grid[0], grid[-1]

    def eq_vs(cls, J):
        """Same interpolation as brain.eq_vs."""
        row = rows[cls]
        if J <= g0:
            return row[0]
        if J >= gn:
            return row[-1]
        i = bisect.bisect_left(grid, J)
        x0, x1 = grid[i - 1], grid[i]
        return row[i - 1] + (row[i] - row[i - 1]) * (J - x0) / (x1 - x0)

    if hs.order is None and getattr(T, "EQ_RANDOM", None) is not None:
        # class_of_safe maps unknown hands to the median class by strength; install it from the tables
        # (verify 2026-11-03: without this an unknown hand fell back to class 0, the best class)
        try:
            hs.set_strength([T.EQ_RANDOM[pos[lab]] for lab in labels])
        except Exception:  # noqa: BLE001
            pass
    class_of_list = None
    if hs._pair_table is not None:
        class_of_list = [c if c >= 0 else 0 for c in hs._pair_table]   # impossible pairs (a == b) -> 0, as brain.py
    return {
        "LABELS": labels,
        "CLASS_ID": {lab: i for i, lab in enumerate(labels)},
        "PUSH_PCT": push_pct,
        "CALL_PCT": call_pct,
        "EQ_GRID": grid,
        "EQ_ROWS": rows,
        "CLASS_OF": class_of_list,
        "CLASS_OF_STRIDE": hs._pair_M if class_of_list is not None else None,
        "class_of": hs.class_of_safe,
        "eq_vs": eq_vs,
        "JAM_FIT": tuple(getattr(T, "JAM_FIT", ())) or None,
        "CALL_FIT": tuple(getattr(T, "CALL_FIT", ())) or None,
        "TOTAL_COMBOS": hs.total_combos,
    }


# ===================================================================== variant-aware postflop class (brain seam)
# The brain's postflop logic reads one number per decision: the hand class on a 0..3 scale (bot/brain.py
# hand_class). The scale is STRUCTURAL and stays the same in every variant; what changes with the variant is which
# card sets are classed (the hand rule) and which 5-rank sets are straights (the ranking):
#   3 strong   a flush or a straight made with our hole cards (a straight better than the board's own), a set,
#              trips with a hole card, two pair with two hole ranks, a full house or quads that use a hole card
#   2 medium   top pair (a hole card pairs the board's top rank), an overpair, a pocket pair with board trips;
#              the deck's lowest straight when the straights are not hold'em's (short deck A-6-7-8-9, measured)
#   1 weak     a lower pair, a low flush (below Q) on a four-flush board, or with cards to come a flush draw (4 to a
#              flush with a hole card) or an open-ended straight draw (4 ranks that 2 or more of the variant's
#              straights contain: 6789 in a short deck completes with A or T)
#   0 air
# Category order (flush above full house, trips above straight) cannot move a hand between these classes: every
# category it flips is class 3 when a hole card makes it. It is probed and kept (PostflopRules) for the record.
# With the 52-card ranking and exactly 2 hole cards class_any is bot/brain.py hand_class card for card (tested on
# boards of 3-5 cards, bot/tests/test_postflop_variant.py) except "cards to come": hand_class reads "fewer than 5
# board cards", class_any reads the street when the brain passes it (a multiway toss flop already holds 5+ cards).
# Hand rules: must_use (k_hole, k_board) = Omaha style, exactly k_hole hole cards with k_board board cards; keep k of
# n with the choice still ahead = the best keepable subset (its discards join the board when they go there);
# otherwise any combination of all hole cards (the best subset).

def _bits(m):
    return [r for r in range(13) if m >> r & 1]


class PostflopRules:
    """Precomputed ranking facts for one variant: straight masks and table, open-ended windows, category flips."""

    def __init__(self, straights, flush_over_full_house=False, trips_over_straight=False, ranks_present=None):
        try:
            from .cards import straight_table_from
        except ImportError:                       # loaded without its package
            from cards import straight_table_from
        self.straights = tuple(sorted(straights))
        rp = ranks_present if ranks_present is not None else sorted({r for m in self.straights for r in _bits(m)})
        self.table = straight_table_from(self.straights, rp)
        cnt = {}
        for s in self.straights:
            for r in _bits(s):
                w = s & ~(1 << r)
                cnt[w] = cnt.get(w, 0) + 1
        # open-ended windows: 4 ranks that complete to a straight with 2 or more different ranks (2345 via A or 6;
        # 6789 in a short deck via A6789 and 6789T); one-ended (JQKA, A234) and gutshots are not draws (hand_class)
        self.windows = tuple(sorted(w for w, c in cnt.items() if c >= 2))
        self.flush_over_fh = bool(flush_over_full_house)
        self.trips_over_straight = bool(trips_over_straight)
        # weak straights: the deck's lowest straight when the straight set is not hold'em's (short deck A-6-7-8-9).
        # Class 2, not 3: as 3 it lost -0.117 [-0.179, -0.055] pts/table over 20,000 paired short-deck tables (the
        # bot raises the lowest straight of a deck where straights are common); as 2 it gained +0.046 [+0.010,
        # +0.082] (bot/tests/ab_postflop_ablation.py, POSTFLOP_VARIANT notes section 4). Hold'em: none (identity).
        try:
            from .cards import STRAIGHTS_52, straight_height
        except ImportError:
            from cards import STRAIGHTS_52, straight_height
        self.heights = {m: straight_height(m, rp) for m in self.straights}
        self.weak_heights = frozenset() if self.straights == STRAIGHTS_52 or not self.straights else \
            frozenset((min(self.heights.values()),))


def rules_from_ranking(rk):
    """PostflopRules for a probe_ranking dict (None: the 52-card ranking)."""
    try:
        from .cards import STRAIGHTS_52
    except ImportError:
        from cards import STRAIGHTS_52
    if not rk:
        return PostflopRules(STRAIGHTS_52)
    return PostflopRules(rk["straights"], rk.get("flush_over_full_house"), rk.get("trips_over_straight"))


def class_any(hole, board, R, more=None):
    """Class 0..3 of `hole` (any number of cards, any combination with the board) under PostflopRules R.
    more: True/False = cards still to come; None = fewer than 5 board cards (hand_class's rule). The tests run in
    hand_class's order, so 2 hole cards with the 52-card ranking give hand_class exactly."""
    bcnt = [0] * 13
    bsuit = [0, 0, 0, 0]
    hcnt = [0] * 13
    hsuit = [0, 0, 0, 0]
    bmask = 0
    hmask = 0
    top = 0
    for c in board:
        r = c >> 2
        bcnt[r] += 1
        bsuit[c & 3] += 1
        bmask |= 1 << r
        if r > top:
            top = r
    for c in hole:
        r = c >> 2
        hcnt[r] += 1
        hsuit[c & 3] += 1
        hmask |= 1 << r
    if more is None:
        more = len(board) < 5
    fs = -1                                        # flush using a hole card (the last qualifying hole card decides)
    for c in hole:
        s = c & 3
        if bsuit[s] + hsuit[s] >= 5:
            fs = s
    if fs >= 0:
        if bsuit[fs] >= 4:                         # four-flush board: only a high flush card is strong
            hi = max(c >> 2 for c in hole if c & 3 == fs)
            return 3 if hi >= 10 else 1
        return 3
    T = R.table
    amask = bmask | hmask
    sh = T[amask]
    flo = 0
    if sh >= 0 and sh > T[bmask]:
        if sh not in R.weak_heights:
            return 3                               # straight using a hole card (the variant's straights)
        flo = 2                                    # the deck's lowest straight in a non-hold'em ranking: 2, look on
    bmax = max(bcnt)
    cls = 0
    paired = 0                                     # distinct unpaired hole ranks the board pairs
    m = hmask
    while m:
        low = m & -m
        r = low.bit_length() - 1
        m ^= low
        h, b = hcnt[r], bcnt[r]
        if h >= 2:                                 # pocket pair
            if b >= 1:
                return 3                           # set or quads
            if bmax >= 3:
                c = 2                              # full house with the board's trips (weak-ish)
            elif r > top:
                c = 2                              # overpair
            else:
                c = 1                              # underpair / middle pocket pair
        elif b >= 2:
            return 3                               # trips (or quads) with a hole card
        elif b == 1:
            paired += 1
            c = 2 if r == top else 1               # top pair / lower pair
        else:
            c = 0
        if c > cls:
            cls = c
    if paired >= 2:
        return 3                                   # two pair with two hole ranks
    if cls == 0 and flo == 0 and more:
        for c in hole:
            s = c & 3
            if bsuit[s] + hsuit[s] == 4:
                return 1                           # flush draw with a hole card
        for w in R.windows:
            if amask & w == w and hmask & w and bmask & w != w:
                return 1                           # open-ended straight draw using a hole card
    return cls if cls > flo else flo


def class_must(sub, board, R, kh, kb, more=None):
    """Class 0..3 of exactly the kh cards in `sub` played with exactly kb board cards (kh + kb = 5): Omaha's rule.
    A flush needs all kh hole cards in one suit and kb board cards of it; a straight uses all kh hole ranks
    (distinct) and kb board ranks; pairs as class_any. Board-only hands never count (they cannot be played).
    Unlike hand_class, a low flush on a four-flush board does not stop the search (a straight flush or a set made
    with the same cards still counts as 3)."""
    bcnt = [0] * 13
    bsuit = [0, 0, 0, 0]
    bmask = 0
    top = 0
    for c in board:
        r = c >> 2
        bcnt[r] += 1
        bsuit[c & 3] += 1
        bmask |= 1 << r
        if r > top:
            top = r
    if more is None:
        more = len(board) < 5
    s0 = sub[0] & 3
    suited = all(c & 3 == s0 for c in sub)
    flo = 0                                        # a low flush on a four-flush board: at least 1, keep looking
    if suited and bsuit[s0] >= kb:
        if bsuit[s0] < 4 or max(c >> 2 for c in sub) >= 10:
            return 3
        flo = 1
    hm = 0
    hcnt = [0] * 13
    for c in sub:
        hcnt[c >> 2] += 1
        hm |= 1 << (c >> 2)
    distinct = bin(hm).count("1") == kh
    if distinct:
        for s in R.straights:
            rest = s & ~hm
            if s & hm == hm and rest & bmask == rest:
                if R.heights[s] not in R.weak_heights:
                    return 3
                if flo < 2:
                    flo = 2                        # the deck's lowest straight (non-hold'em ranking): 2
    bmax = max(bcnt)
    cls = 0
    paired = 0
    for r in _bits(hm):
        h, b = hcnt[r], bcnt[r]
        if h >= 2:
            if b >= 1:
                return 3                           # set (pair + 1 board card + 2 more board cards)
            if bmax >= 3 and kb >= 3:
                c = 2
            elif r > top:
                c = 2
            else:
                c = 1
        elif b >= 2 and kb >= 2:
            return 3                               # trips with one hole card (the other is the kicker)
        elif b >= 1:
            paired += 1
            c = 2 if r == top else 1
        else:
            c = 0
        if c > cls:
            cls = c
    if paired >= 2:
        return 3
    if cls == 0 and flo == 0 and more:
        if suited and bsuit[s0] == kb - 1:
            return 1                               # flush draw: one more board card of our suit
        if distinct:
            for w in R.windows:
                rest = w & ~hm
                if w & hm == hm and rest & bmask == rest and bin(rest).count("1") == kb - 1:
                    return 1                       # open-ended: all hole ranks plus kb - 1 board ranks
    return cls if cls > flo else flo


def postflop_classifier(spec, ranking=None):
    """None when the spec's postflop is plain hold'em (52-card ranking, exactly 2 hole cards, no keep choice, no
    must-use rule, 4 suits): the brain keeps its own hand_class. Otherwise cls(hole, board, street=None) -> 0..3.
    ranking: a probe_ranking dict; default: probed from spec.evaluator on spec.deck (spec_from_params installs the
    tables' ranking there). Never raises while building (None on any failure)."""
    try:
        try:
            from .spec import is_standard_ranking, probe_ranking
        except ImportError:
            from spec import is_standard_ranking, probe_ranking
        from itertools import combinations
        if spec.codec.slot != 4 or len(spec.suit_chars) > 4:
            return None                            # exotic suit counts: kit numbering only
        rk = ranking if ranking is not None else probe_ranking(spec.evaluator, spec.deck, spec.codec.slot)
        std_rank = is_standard_ranking(rk)
        mu = tuple(spec.must_use) if spec.must_use else None
        keep = spec.hole_kept if spec.hole_kept and spec.hole_kept < spec.hole_dealt else None
        if std_rank and mu is None and spec.hole_dealt == 2 and keep is None:
            return None
        R = rules_from_ranking(rk)
        n_streets = len(spec.board_schedule) or 3
        to_board = spec.discard_to == "board"
        if mu is not None and sum(mu) != 5:
            mu = None                              # unusual must-use shapes: any combination (documented)
    except Exception:  # noqa: BLE001
        return None

    def base(hole, board, more):
        if mu is None or len(hole) < mu[0]:
            return class_any(hole, board, R, more)
        kh, kb = mu
        best = 0
        for sub in combinations(hole, kh):
            c = class_must(sub, board, R, kh, kb, more)
            if c > best:
                best = c
                if c == 3:
                    break
        return best

    def classify(hole, board, street=None):
        more = None if street is None else street < n_streets
        if keep is not None and len(hole) > keep:   # the choice is still ahead: the best keepable subset
            n = len(hole)
            best = 0
            for sub in combinations(range(n), keep):
                kept = [hole[i] for i in sub]
                bd = list(board) + [hole[i] for i in range(n) if i not in sub] if to_board else board
                c = base(kept, bd, more)
                if c > best:
                    best = c
                    if c == 3:
                        break
            return best
        return base(hole, board, more)

    classify.rules = R
    classify.describe = (f"variant classifier, {'52-card' if std_rank else 'variant'} ranking ({len(R.straights)} "
                         f"straights, flush>FH {R.flush_over_fh}, trips>straight {R.trips_over_straight}), "
                         f"{'must use ' + str(mu) if mu else 'any combination'}"
                         f"{', keep ' + str(keep) + ' of ' + str(spec.hole_dealt) if keep else ''}")
    return classify
