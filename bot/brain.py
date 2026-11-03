"""Brain of the rehearsal bot (written for No Limit, Bay Area regional). Pure standard library.

Three versions, selected by PARAMS["VERSION"] (or the shell classes BotV0 / BotV1 / BotV2):
  v0  chip-EV jam/fold skeleton keyed on effective stack S and players behind k. First-in jams: S = min(my chips,
      biggest live opponent). Calls of all-ins: equity of our class against the jammer's estimated range (push-order
      top-J table) versus the real pot odds from contributions, so S is implicitly min(my chips, the jammer's chips)
      (P4). Deep opens by position, the tested facing-raise rules, value-heavy postflop by hand class, no bluffs.
  v1  v0 + exact ICM bubble factors (BF) from the announced payout, padded to 8 places. Call side always (risk
      premium on every all-in call, BF^-2.6 on re-jam ranges, BF gates on big postflop calls). Jam side: jam% x
      BF_hero^-2.5 by default; the tested jam% x BF_callers^2 x BF_hero^-1.5 only when every seat behind has been
      OBSERVED folding to jams more than chip-EV Nash (P2). Missing BFs come from an equal-stack BF table, never 1.0
      (P3).
  v2  v1 + per-seat typing from ACTIONS only (showdowns are censored), shrunk to pooled priors with minimum samples:
      folder (cleared at its first voluntary chip in the current hand, P7), maniac (its own shrunk jam rate as the
      jam range, E10), caller (light-jam cap, bigger value bets), loose, stealer (re-jam floor, E11), nit; gated SB
      widen (typed-folder BB or a BB whose shrunk jam-call rate is under the break-even, P1); a separate, tighter
      prior for re-jams over a raise (P5); BB defence against late small raises (P6); air c-bets only against seats
      with observed flop folds (E13 corrected, P10).

Every tunable number is in PARAMS below. Decisions are lookups and closed forms only (no Monte Carlo); the only
costly step is an exact BF (about 10-40 us), computed at most BF_PER_DECISION times per decision, cached per hand, and
skipped once the own clock passes GUARD_NS (the default table value is used instead).

The brain never sees the kit's raw state: adapter.py turns it into a View. It returns internal decisions:
("fold",), ("check",), ("call",), ("raise", raise_to_total), ("allin",); adapter.py makes them legal.
"""
import bisect
import random
import time

# ===================================== PARAMS: every tunable number =====================================
PARAMS = {
    "VERSION": 2,                 # 0, 1 or 2 (see the module docstring)
    # ---- shell and adapter (shell.py reads these; its own copy is used only if this module fails to import)
    "GC_GUARD_ACT": True,         # pause the collector for the duration of act() (try/finally, about 0.1 us), so no
                                  # collection can START inside our timed call; a due one runs after we return
    "GC_HYGIENE": True,           # at each hand_start event (observe, untimed in the kit) run a gen-1 or full
                                  # collection that is about to come due; no effect if observe() is never called
    "GC_FULL_MARGIN": 2,          # hygiene: full collection when gen-2's count is within this of its threshold
    "GC_GEN1_MARGIN": 1,          # hygiene: gen-1 collection when gen-1's count is within this of its threshold
    "GC_MODE": "freeze",          # at import, after the warm-up: "freeze" (gc.collect + gc.freeze, collector ON),
                                  # "disable" (freeze + gc.disable), "none"
    "WARMUP_DECISIONS": 300,      # synthetic decisions per version through a throwaway instance at import (0 = skip)
    "ALLIN_SNAP": 0.85,           # adapter: a raise-to at or above this share of max_raise_to goes all-in
    "FALLBACK_V0": True,          # P15: an exception in v1/v2 work returns the v0 decision (ICM and typing off),
                                  # not check/fold; only if that also fails does the shell check/fold
    # ---- payout and ICM (v1+)
    "ICM_ON": None,               # None = auto (off under winner-take-all, where every BF is 1); True/False forces
    "BF_MIN": 0.5, "BF_MAX": 6.0,  # BF clamp (tested values)
    "BF_NO_GAIN": 4.0,            # BF when winning the all-in gains no ICM equity (degenerate; then clamped)
    "BF_DEFAULT": 1.25,           # BF if the equal-stack table has no entry for the players left (8-max 50/30/20)
    "BF_PER_DECISION": 4,         # exact BFs computed in one decision at most; the rest: cache or table default
    "CALL_LAM": 2.6,              # re-jam / fit calls: call% /= BF ** CALL_LAM (fit on solved ICM spots)
    "JAM_HERO_EXP": -2.5,         # default jam side: jam% *= mean BF(me, caller) ** this (P2, review's best response)
    "JAM_OPP_EXP": 2.0,           # gated jam side (every possible caller observed jam-tight): * mean BF(caller, me)**this
    "JAM_HERO_EXP_GATED": -1.5,   # ... * mean BF(me, caller) ** this (the tested formula)
    # ---- latency
    "GUARD_NS": 250_000,          # own-clock budget per decision (ns); optional work (BFs) must fit inside it
    "BF_COST_SAFETY": 3.0,        # a BF starts only if elapsed + this x the BF cost measured at start-up < GUARD_NS
    # ---- chip-EV Nash fits (sng_icm fit_rules.json; refit these for antes or a card variant)
    "JAM_FIT": (3.29, -0.717, -0.917, 1.41, -0.41),   # jam% = a S^b k^c for k >= 2 callers; a1 S^b1 for k = 1
    "CALL_FIT": (2.64, 0.971, -0.582),                # call% = a J^b S^c (J = jammer's range, S = effective bb)
    "DEAD_BASE_BB": 1.5,          # first-in jams use S_eq = S x this / pot (bb): antes and limps widen like depth
    "DEAD_MIN_BB": 0.5,           # ... with the pot floored at this (an almost empty pot: about 3x tighter)
    # ---- preflop (v0+)
    "PF_DEPTH_BB": 20.0,          # jam/fold zone: effective stack in big blinds
    "OPEN_PCT": {7: .12, 6: .14, 5: .17, 4: .20, 3: .26, 2: .36, 1: .45},   # deep first-in opens by players behind
    "OPEN_BB": 2.5,               # deep open size (raise-to, in bb); +OPEN_LIMPER_BB per limper
    "OPEN_LIMPER_BB": 1.0,
    "LATE_SEATS": 3,              # a first-in raise with at most this many players behind is a late (steal) open
    "JAM_MULT": 1.0,              # x every first-in jam range
    "CALL_MODEL": "equity",       # "equity" (default) or "fit" (tested call_pct(J, S) with S vs the jammer)
    "CALL_EQ_ADJ": 0.0,           # added to the required equity of every all-in call (+ = tighter)
    "CALL_BEHIND_MARGIN": 0.02,   # required equity + this per opponent who can still act after us (overcall risk)
    "CALLER_RANGE_FRAC": 0.6,     # a player who already called the jam holds about this fraction of the jam range
    "CALL_MULT": 1.0,             # "fit" model only: x call range
    "RESTEAL_MULT": 1.2,          # committed vs a non-all-in raise: re-jam if call pct <= C x this (tested)
    "COMMIT_TO_CALL_FRAC": 0.4,   # facing a raise: committed if to_call >= this x my chips (or S <= PF_DEPTH_BB)
    "RAISE_TOP": 0.03,            # deep, facing a raise: 3-bet (jam if S <= FOURBET_JAM_MAX_BB) this top fraction
    "FOURBET_JAM_MAX_BB": 40.0,
    "THREEBET_MULT": 3.0,         # deep 3-bet size: raise to this x the current bet
    "FLAT_TOP": 0.10,             # deep, facing a raise: call this top fraction ...
    "FLAT_MAX_STACK": 0.10,       # ... if the call costs at most this share of my chips
    "BB_ISO_PCT": 0.06,           # BB with limpers and no raise: value-raise (jam when shallow) this top fraction
    "BB_ISO_BB": 4.0,             # deep iso size in bb ...
    "BB_ISO_LIMPER_BB": 1.0,      # ... plus this per limper
    "NEAR_ALLIN_FRAC": 0.25,      # a raise leaving the raiser <= this share of its chips ...
    "NEAR_ALLIN_BB": 1.0,         # ... or <= this many bb counts as a jam
    # ---- postflop (v0+)
    "VALUE_FRAC": 0.70,           # value bet (class >= 2), pot fraction
    "RAISE_MULT": 3.0,            # class 3 facing a bet: raise to this x the bet
    "BIG_CALL_FRAC": 0.4,         # a call is "big" when it costs >= this share of my chips behind
    "CALL2_MAX_POT": 1.0,         # class 2 calls a bet up to this x pot (not big, or BF below BF_CALL2_MAX)
    "CALL1_MAX_POT": 0.35,        # class 1 calls a bet up to this x pot (never big)
    "RAISED_CALL2_MAX_POT": 0.35,  # class 2 facing a RAISE of our bet: call only this cheap (raises are honest)
    "BF_CALL2_MAX": 1.3,          # v1+: big class-2 call only if BF < this
    "BF_RAISE3_MAX": 2.0,         # v1+: big class-3 spot: just call when BF >= this
    "COMMIT_FRAC": 0.45,          # a bet or raise that puts in >= this share of my chips goes all-in instead
    "CBET_AIR_UNKNOWN": 0.0,      # air c-bet frequency vs a seat with no fold evidence (P10: default off)
    # ---- v2 typing (shrinkage: estimate = (hits + k * mean) / (n + k))
    "PRIOR": {"vol": (0.30, 12), "jam": (0.05, 10), "calljam": (0.35, 6), "post_call": (0.45, 8),
              "flop_fold": (0.30, 6), "late_open": (0.25, 8)},
    "OE_PRIOR": 0.5,              # jam-call observed/expected ratio shrinkage, in expected-call units
    "MIN_OBS_TYPE": 6,            # preflop decisions before a seat gets any type other than folder ("unknown")
    "MIN_OBS_FOLDER": 8,          # preflop decisions with 0 voluntary chips before a seat is a folder
    "T_MANIAC_JAM": 0.25,         # shrunk first-in jam rate
    "T_CALLER_CALLJAM": 0.50, "MIN_JAMS_FACED": 3,   # shrunk jam-call rate over >= 3 jams faced
    "T_POST_CALL": 0.60, "MIN_POST_FACED": 5,         # or shrunk postflop call rate over >= 5 bets faced
    "T_LOOSE_VOL": 0.45,          # shrunk voluntary rate
    "T_NIT_VOL": 0.12, "MIN_OBS_NIT": 30,
    "T_STEAL_LATE": 0.40, "MIN_LATE_OPP": 8,          # stealer: shrunk late first-in open rate
    "OE_TIGHT": 0.6, "MIN_JAMS_FACED_TIGHT": 4,       # jam-tight: shrunk calls/expected-Nash-calls <= this
    "FOLDER_STEAL": True,         # min-raise any two when every seat behind is a folder (E09)
    "LIGHT_JAM_CAP": 0.35,        # max jam range with a caller, loose or maniac seat behind (E05)
    "SB_WIDEN": 1.6,              # x jam range when the only seat behind is a folder or folds to jams (P1)
    "SB_WIDEN_MARGIN": 0.05,      # widen only if the BB's shrunk jam-call rate < break-even(S) - this
    "SB_WIDEN_BE": (0.33, 0.018, 0.10),   # break-even jam-call rate of the widen: 0.33 at <= 10 bb, minus 0.018 per bb
                                          # above 10 (24% at 15 bb, review's sb_widen_ev), floored at 0.10
    "JAMFOLD_WIDEN": 1.6,         # x jam range when every seat behind is jam-tight (E15, payout-independent)
    "J_FLOOR": {"maniac": 0.50, "loose": 0.35, "stealer": 0.40},   # range floors of typed raisers (no caller floor)
    "REJAM_PRIOR": 0.06,          # P5: range of a re-jam over a raise (unless the seat is a typed maniac)
    "OPEN_PRIOR": None,           # v2 range prior of a first-in opener by players behind (None = OPEN_PCT)
    "OPEN_K_UNKNOWN": 3,          # v2: players-behind assumed for a raiser that was not the first-in opener
    "LATE_OPEN_FLOOR": 0.35,      # P6: range floor of a small late open (<= LATE_SEATS behind, raise <=
                                  #     BB_DEFEND_MAX_RAISE_BB)
    "BB_DEFEND_PCT": 0.30,        # P6: BB flats this top fraction (call order) vs a late small raise ...
    "BB_DEFEND_MAX_RAISE_BB": 3.0,  # ... of at most this many bb ...
    "BB_DEFEND_MAX_FRAC": 0.15,   # ... when the call costs at most this share of my chips
    "VALUE_FRAC_CALLER": 1.00,    # value bet vs a typed caller (E16)
    "CBET_FRAC": 0.33,            # air c-bet size vs a seat with fold evidence (E13)
    "CBET_FOLD_T": 0.40, "CBET_MIN_OBS": 3,           # shrunk fold-to-flop-bet rate over >= 3 flop bets faced
    "LIGHT_CALL_TYPES": ("maniac", "stealer"),        # call their postflop bets one class lighter (E11)
}

RANKS = "23456789TJQKA"
STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# ===================================== data: classes, percentiles, equity table =====================================
try:
    from . import bot_tables as _T
except ImportError:   # loaded by file path (kit runner) or flat: load the sibling file explicitly
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _name = "nolimit_bot_tables"
    _T = _sys.modules.get(_name)
    if _T is None:
        _spec = _ilu.spec_from_file_location(_name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                                                   "bot_tables.py"))
        _T = _ilu.module_from_spec(_spec)
        _sys.modules[_name] = _T
        _spec.loader.exec_module(_T)


def combos(lab):
    return 6 if len(lab) == 2 else (4 if lab[2] == "s" else 12)


def percentiles(order):
    """Midpoint percentile of each class in a best-first order: (combos of better classes + half its own) / 1326."""
    out, acc = {}, 0
    for lab in order:
        c = combos(lab)
        out[lab] = (acc + c / 2.0) / 1326.0
        acc += c
    return out


def label_of(r1, s1, r2, s2):
    if r1 < r2:
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if s1 == s2 else "o")


LABELS = _T.EQ_LABELS.split()                     # class index -> label (equity table row order)
CLASS_ID = {lab: i for i, lab in enumerate(LABELS)}
_pp = percentiles(_T.PUSH_ORDER.split())
_cp = percentiles(_T.CALL_ORDER.split())
PUSH_PCT = [_pp[lab] for lab in LABELS]           # 0 = best
CALL_PCT = [_cp[lab] for lab in LABELS]
EQ_GRID = list(_T.EQ_GRID)
EQ_ROWS = [list(r) for r in _T.EQ_TABLE]
CLASS_OF = [0] * 2704                             # card pair (c1 * 52 + c2, kit encoding) -> class index
for _a in range(52):
    for _b in range(52):
        if _a != _b:
            CLASS_OF[_a * 52 + _b] = CLASS_ID[label_of(_a >> 2, _a & 3, _b >> 2, _b & 3)]
_G0, _GN = EQ_GRID[0], EQ_GRID[-1]


def eq_vs(cls, J):
    """All-in equity of class `cls` against the top-J fraction of hands (push order), linear in J between grid
    points; J below the first grid point uses it (slightly optimistic vs ultra-tight ranges)."""
    row = EQ_ROWS[cls]
    if J <= _G0:
        return row[0]
    if J >= _GN:
        return row[-1]
    i = bisect.bisect_left(EQ_GRID, J)
    x0, x1 = EQ_GRID[i - 1], EQ_GRID[i]
    return row[i - 1] + (row[i] - row[i - 1]) * (J - x0) / (x1 - x0)


def jam_pct(S, k, f=PARAMS["JAM_FIT"]):
    """Chip-EV Nash first-in jam range (fraction of hands); S = effective bb, k = players who can call;
    f = PARAMS["JAM_FIT"] = (a, b, c, a1, b1)."""
    S = S if S > 1.0 else 1.0
    if k <= 1:
        v = f[3] * S ** f[4]
    else:
        v = f[0] * S ** f[1] * k ** f[2]
    return v if v < 1.0 else 1.0


def call_pct(J, S, f=PARAMS["CALL_FIT"]):
    """Chip-EV Nash big-blind call range against a jam range J at effective depth S; f = PARAMS["CALL_FIT"]."""
    v = f[0] * (J if J > 0.01 else 0.01) ** f[1] * (S if S > 1.0 else 1.0) ** f[2]
    return v if v < 1.0 else 1.0


def by_k(table, k):
    """Value of a {players behind: value} table with k clamped to its key range (k = 0 uses the smallest key)."""
    lo, hi = min(table), max(table)
    return table[lo if k < lo else hi if k > hi else k]


def equiv_depth(S, pot_bb, P):
    """Dead-money equivalent depth for first-in jams: a blinds-only pot (DEAD_BASE_BB) leaves S unchanged; antes
    and limps make the jam behave like a shallower one (P11: the solver widens jams x1.24 SB to x1.55 UTG at
    10 bb with 0.125 bb antes; this gives x1.23 and x1.44)."""
    floor = P["DEAD_MIN_BB"]
    return S * P["DEAD_BASE_BB"] / (pot_bb if pot_bb > floor else floor)


# ===================================== postflop hand class =====================================
def _straight_high(mask):
    """Highest straight rank (index of the top card) in a 13-bit rank mask, -1 if none; the wheel counts as 3."""
    for top in range(12, 3, -1):
        if (mask >> (top - 4)) & 31 == 31:
            return top
    if mask & 0x100F == 0x100F:
        return 3
    return -1


def hand_class(h1, h2, board):
    """0 air, 1 weak pair or draw, 2 top pair or overpair, 3 strong (two pair using both hole cards, set, trips with
    a hole card, a straight or flush using a hole card, full house or quads using a hole card). Kit card ints:
    rank = c >> 2 (0 = deuce .. 12 = ace), suit = c & 3. Board-only made hands do not count."""
    r1, r2 = h1 >> 2, h2 >> 2
    s1, s2 = h1 & 3, h2 & 3
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
    nb = len(board)
    # flush using a hole card
    fs = -1
    for s in (s1, s2):
        if bsuit[s] + (s1 == s) + (s2 == s) >= 5:
            fs = s
    if fs >= 0:
        if bsuit[fs] >= 4:   # four-flush board: only a high flush card is strong
            hi = max(r for r, s in ((r1, s1), (r2, s2)) if s == fs)
            return 3 if hi >= 10 else 1
        return 3
    # straight using a hole card
    amask = bmask | (1 << r1) | (1 << r2)
    sh = _straight_high(amask)
    if sh >= 0 and sh > _straight_high(bmask):
        return 3
    bmax = max(bcnt)
    if r1 == r2:                                   # pocket pair
        if bcnt[r1] >= 1:
            return 3                               # set or quads
        if bmax >= 3:
            return 2                               # full house made with the board's trips (weak-ish)
        if r1 > top:
            return 2                               # overpair (possibly with a board pair)
        cls = 1                                    # underpair / middle pocket pair
    else:
        c1, c2 = bcnt[r1], bcnt[r2]
        if c1 and c2:
            return 3                               # two pair with both hole cards (or a full house)
        if c1 >= 2 or c2 >= 2:
            return 3                               # trips (or quads) with a hole card
        if c1 or c2:
            pr = r1 if c1 else r2
            if pr == top:
                return 2                           # top pair (also on a paired board)
            cls = 1                                # second pair or lower
        else:
            cls = 0
    if cls == 0 and nb < 5:
        # draws: four to a flush with a hole card, or an open-ended straight draw using a hole card
        for s in (s1, s2):
            if bsuit[s] + (s1 == s) + (s2 == s) == 4:
                return 1
        hm = (1 << r1) | (1 << r2)
        for lo in range(0, 9):                     # windows 2345 .. TJQK are open at both ends (A2345 counts)
            w = 15 << lo
            if amask & w == w and hm & w and bmask & w != w:
                return 1
    return cls


# ===================================== ICM (exact Malmuth-Harville; padded payouts) =====================
def pad_payouts(pays, n=8):
    p = [float(x) for x in (pays or ())]
    if len(p) < n:
        p += [0.0] * (n - len(p))
    return p


def is_wta(pays):
    """Winner-take-all (every place after the first pays the same): equity is linear in chips, every BF is 1."""
    return len(pays) < 2 or all(abs(x - pays[1]) < 1e-12 for x in pays[1:])


def icm_hero(stacks, i, pays):
    """Closed form, EXACT when places 4..n are affine in place (WTA, top-3, linear, ...); pays padded."""
    live = [s for s in stacks if s > 0]
    k, si = len(live), stacks[i]
    if si <= 0 or k == 0:
        return 0.0
    p = list(pays[:k]) + [0.0] * max(0, k - len(pays))
    if k == 1:
        return p[0]
    step, base = p[k - 2] - p[k - 1], p[k - 1]
    acc = 0.0
    for sj in live:
        acc += si / (si + sj)
    lin = base + step * (acc - 0.5)
    c0 = p[0] - (base + step * (k - 1))
    c1 = p[1] - (base + step * (k - 2)) if k > 1 else 0.0
    c2 = p[2] - (base + step * (k - 3)) if k > 2 else 0.0
    if not (c0 or c1 or c2):
        return lin
    T = float(sum(live))
    e2 = e3 = 0.0
    n = len(stacks)
    for j in range(n):
        sj = stacks[j]
        if j == i or sj <= 0:
            continue
        Tj = T - sj
        pj = sj / T
        e2 += pj * si / Tj
        if c2:
            for m in range(n):
                sm = stacks[m]
                if m != i and m != j and sm > 0:
                    e3 += pj * (sm / Tj) * si / (Tj - sm)
    return lin + c0 * (si / T) + c1 * e2 + c2 * e3


def make_icm_exact(pays, max_players=10):
    """icm(stacks, i): exact Malmuth-Harville equity of seat i for ANY payout (exponential-race identity:
    E_i = sum over subsets B of the live others of coef[m][|B|] * s_i / (s_i + s(B)))."""
    C = [[1]]
    for n in range(1, max_players + 1):
        C.append([1] + [C[n - 1][t - 1] + C[n - 1][t] for t in range(1, n)] + [1])
    coef = []
    for m in range(max_players):
        p = list(pays[:m + 1]) + [0.0] * max(0, m + 1 - len(pays))
        row = []
        for b in range(m + 1):
            acc, sign = 0.0, 1.0
            for t in range(b + 1):
                acc += sign * C[b][t] * p[m - b + t]
                sign = -sign
            row.append(acc)
        coef.append(row)

    def icm(stacks, i):
        si = stacks[i]
        if si <= 0:
            return 0.0
        sums, sizes = [0.0], [0]
        for j, s in enumerate(stacks):
            if j != i and s > 0:
                sums += [x + s for x in sums]
                sizes += [b + 1 for b in sizes]
        row = coef[len(sizes).bit_length() - 1]
        ev = 0.0
        for x, b in zip(sums, sizes):
            ev += row[b] * si / (si + x)
        return ev
    return icm


def make_icm(pays):
    """Fastest exact method for this payout (pays padded to >= 8 places): closed form for affine tails."""
    pays = pad_payouts(pays, max(8, len(pays)))
    t = pays[3:8]
    if all(abs((t[q] - t[q + 1]) - (t[0] - t[1])) < 1e-9 for q in range(len(t) - 1)):
        return lambda stacks, i: icm_hero(stacks, i, pays)
    return make_icm_exact(pays, max(10, len(pays)))


def bubble_factor(icm, stacks, i, j, pays, lo=0.5, hi=6.0, no_gain=4.0):
    """ICM loss / ICM gain for seat i in an all-in of min(s_i, s_j) against seat j (start-of-hand stacks)."""
    a = min(stacks[i], stacks[j])
    if a <= 0:
        return 1.0
    k = 0
    for s in stacks:
        if s > 0:
            k += 1
    now = icm(stacks, i)
    win = list(stacks)
    win[i] += a
    win[j] -= a
    lose = list(stacks)
    lose[i] -= a
    lose[j] += a
    gain = icm(win, i) - now
    lost = now - (pays[k - 1] if lose[i] <= 0 else icm(lose, i))
    if gain <= 1e-12:
        v = no_gain
    else:
        v = lost / gain
    return lo if v < lo else hi if v > hi else v


_ICM_PACKS = {}   # payout key -> (icm function, equal-stack BF by players left, BF cost ns); pure function of the payout


def icm_pack(pays, n_seats, lo, hi, no_gain=4.0):
    """(icm, equal-stack BF table, measured BF cost in ns) for a payout, built once per process (construction time,
    untimed). The cost is the median thread-CPU time of 7 BFs on an 8-stack spot, so the guard adapts to the box."""
    key = (tuple(pays), n_seats, lo, hi, no_gain)
    pk = _ICM_PACKS.get(key)
    if pk is None:
        icm = make_icm(pays)
        table = {}
        for k in range(2, max(n_seats, 2) + 1):
            st = [1000] * k
            table[k] = bubble_factor(icm, st, 0, 1, pays, lo, hi, no_gain)
        table[1] = 1.0
        spot = [2600, 2100, 1900, 1700, 1500, 1300, 1100, 900][:max(2, n_seats)]
        ts = []
        tt = time.thread_time_ns
        for _ in range(7):
            c0 = tt()
            bubble_factor(icm, spot, 0, 1, pays, lo, hi, no_gain)
            ts.append(tt() - c0)
        ts.sort()
        pk = (icm, table, ts[len(ts) // 2])
        _ICM_PACKS[key] = pk
    return pk


# ===================================== per-seat statistics and the hand tracker =====================================
class SeatStat:
    __slots__ = ("pf_dec", "pf_vol", "pf_fold", "jam_opp", "jam_first", "jams", "jam_faced", "jam_called", "jam_exp",
                 "late_opp", "late_open", "post_faced", "post_call", "post_fold", "flop_faced", "flop_fold",
                 "vol_ever", "vol_hand")

    def __init__(self):
        self.pf_dec = self.pf_vol = self.pf_fold = 0
        self.jam_opp = self.jam_first = self.jams = 0
        self.jam_faced = self.jam_called = 0
        self.jam_exp = 0.0
        self.late_opp = self.late_open = 0
        self.post_faced = self.post_call = self.post_fold = 0
        self.flop_faced = self.flop_fold = 0
        self.vol_ever = 0
        self.vol_hand = False


def preflop_order(n, alive, sb_seat, bb_seat):
    """Preflop action order of the dealt-in seats: from the seat after the big blind clockwise to the big blind
    (heads-up: button/SB first, then BB)."""
    seq = []
    j = bb_seat
    for _ in range(n):
        j = (j + 1) % n
        if alive[j]:
            seq.append(j)
        if j == bb_seat:
            break
    return seq


class HandTracker:
    """The current hand's public action sequence, fed in order (from observe() events, or from the state's history
    at decision time when no events arrive). Updates the per-seat SeatStat counters as actions arrive, and keeps what
    the policy needs: raise count, last raiser, limpers, the outstanding jam and its estimated range."""
    __slots__ = ("P", "stats", "n", "hand_no", "street", "n_done", "bb", "ante", "start", "stake", "contrib", "bets",
                 "folded", "allin", "pre", "pos", "bb_seat", "cur", "pf_raises", "last_raiser", "raise_to",
                 "open_k", "open_seat", "limpers", "limper_seats", "callers_since_raise", "jam", "jam_responders",
                 "pf_aggr", "posted", "jam_ranges")

    def __init__(self, P, stats, n):
        self.P = P                        # the brain's PARAMS and per-seat stats (no back-reference to the brain,
        self.stats = stats                # so a finished table's objects are freed by reference counting alone)
        self.n = n
        self.hand_no = None
        self._reset(None, [0] * n, [False] * n, 0, 1, 20, 0)

    def _reset(self, hand_no, start, alive, sb_seat, bb_seat, bb, ante):
        n = self.n
        self.hand_no = hand_no
        self.street = 0
        self.n_done = 0
        self.bb = bb if bb > 0 else 1
        self.ante = ante
        self.start = list(start)
        self.stake = list(start)          # chips at stake preflop = start minus ante paid
        self.contrib = [0] * n
        self.bets = [0] * n
        self.folded = [not a for a in alive]
        self.allin = [False] * n
        self.bb_seat = bb_seat
        self.pre = preflop_order(n, alive, sb_seat, bb_seat) if any(alive) else []
        self.pos = {s: i for i, s in enumerate(self.pre)}
        self.cur = self.bb
        self.pf_raises = 0
        self.last_raiser = -1
        self.raise_to = 0
        self.open_k = 0
        self.open_seat = -1
        self.limpers = 0
        self.limper_seats = []
        self.callers_since_raise = 0
        self.jam = None                   # (seat, J_first_in_formula, is_rejam, jam_total)
        self.jam_responders = set()
        self.jam_ranges = {}              # every all-in raiser this hand: seat -> (J_first_in_formula, is_rejam)
        self.pf_aggr = -1
        self.posted = 0
        for st in self.stats:
            st.vol_hand = False

    def start_hand(self, hand_no, start, alive, sb_seat, bb_seat, bb, ante):
        self._reset(hand_no, start, alive, sb_seat, bb_seat, bb, ante)

    def new_street(self, idx):
        while self.street < idx:
            self.street += 1
            for j in range(self.n):
                self.bets[j] = 0
            self.cur = 0

    def _k_behind(self, s):
        """Dealt-in seats after s in the preflop order that have not folded."""
        p = self.pos.get(s)
        if p is None:
            return 0
        k = 0
        for j in self.pre[p + 1:]:
            if not self.folded[j]:
                k += 1
        return k

    def feed(self, s, kind, amt):
        self.n_done += 1
        if s < 0 or s >= self.n:
            return
        if kind == "ante":
            self.contrib[s] += amt
            self.stake[s] -= amt
            if self.contrib[s] >= self.start[s]:
                self.allin[s] = True
            return
        if kind == "post_sb" or kind == "post_bb":
            add = amt - self.bets[s]
            self.bets[s] = amt
            self.contrib[s] += add
            self.posted += add
            if self.contrib[s] >= self.start[s]:
                self.allin[s] = True
            return
        P = self.P
        st = self.stats[s]
        before = self.bets[s]
        cur = self.cur
        vol = kind == "call" or kind == "raise"
        if kind == "fold":
            self.folded[s] = True
        elif vol:
            add = amt - before
            if add > 0:
                self.bets[s] = amt
                self.contrib[s] += add
            if self.contrib[s] >= self.start[s]:
                self.allin[s] = True
        if self.street == 0:
            st.pf_dec += 1
            if vol:
                st.pf_vol += 1
                st.vol_ever += 1
                st.vol_hand = True
            elif kind == "fold":
                st.pf_fold += 1
            unopened = self.pf_raises == 0
            if unopened and kind != "check":
                st.jam_opp += 1
                if kind == "raise" and self.allin[s]:
                    st.jam_first += 1
                p = self.pos.get(s, 0)
                if self.limpers == 0 and s != self.bb_seat and len(self.pre) - 1 - p <= P["LATE_SEATS"]:
                    st.late_opp += 1
                    if kind == "raise":
                        st.late_open += 1
            jam = self.jam
            if jam is not None and jam[0] != s and kind != "check" and s not in self.jam_responders:
                self.jam_responders.add(s)
                st.jam_faced += 1
                if vol:
                    st.jam_called += 1
                S_resp = min(self.stake[s], jam[3]) / self.bb
                st.jam_exp += call_pct(P["REJAM_PRIOR"] if jam[2] else jam[1], S_resp, P["CALL_FIT"])
            if kind == "raise":
                first_in = self.pf_raises == 0
                if self.allin[s]:
                    # estimated range of this jam: the first-in formula at the jammer's spot (+ a re-jam flag)
                    k = self._k_behind(s) + (self.limpers if first_in else 0)
                    opp = [self.stake[j] for j in self.pre[self.pos.get(s, 0) + 1:] if not self.folded[j]]
                    S_r = min(self.stake[s], max(opp)) / self.bb if opp else self.stake[s] / self.bb
                    pot_before = sum(self.contrib) - (amt - before)
                    S_eq = equiv_depth(S_r, pot_before / self.bb, P) if first_in and pot_before > 0 else S_r
                    self.jam = (s, jam_pct(S_eq, k if k > 0 else 1, P["JAM_FIT"]), not first_in, self.stake[s])
                    self.jam_ranges[s] = (self.jam[1], self.jam[2])
                    self.jam_responders = set()
                if first_in:
                    self.open_k = self._k_behind(s)
                    self.open_seat = s
                self.pf_raises += 1
                self.last_raiser = s
                self.raise_to = amt
                self.callers_since_raise = 0
                self.pf_aggr = s
            elif kind == "call":
                if self.pf_raises == 0:
                    self.limpers += 1
                    self.limper_seats.append(s)
                else:
                    self.callers_since_raise += 1
        else:
            if cur > before:
                st.post_faced += 1
                if kind == "call" or kind == "raise":
                    st.post_call += 1
                elif kind == "fold":
                    st.post_fold += 1
                if self.street == 1:
                    st.flop_faced += 1
                    if kind == "fold":
                        st.flop_fold += 1
            if vol:
                st.vol_ever += 1
                st.vol_hand = True
        if kind == "raise" and amt > self.cur:
            self.cur = amt

    def sync_from_view(self, v):
        """Bring the tracker up to date with the view's history (no-op when observe() already delivered it)."""
        if self.hand_no != v.hand_no:
            self.start_hand(v.hand_no, v.start, v.alive, v.sb_seat, v.bb_seat, v.bb, v.ante)
        hist = v.history
        ss = v.street_starts
        i = self.n_done
        nh = len(hist)
        while i < nh:
            nxt = self.street + 1
            while nxt < len(ss) and i >= ss[nxt]:
                self.new_street(nxt)
                nxt += 1
            e = hist[i]
            self.feed(e[0], e[1], e[2])
            i += 1
        if v.street > self.street:
            self.new_street(v.street)


# ===================================== the brain =====================================
class Brain:
    """One per table (state resets each table). decide(view, t0) -> internal decision tuple."""

    def __init__(self, seat, config, version=None, params=None):
        P = dict(PARAMS)
        if params is None and isinstance(config.get("bot_params"), dict):
            params = config["bot_params"]          # rehearsal A/B hook (runner extra_config); absent on the platform
        if params:
            P.update(params)
        self.P = P
        if version is None:
            version = config.get("bot_version", P["VERSION"])
        self.version = int(version)
        self.seat = seat
        n = int(config.get("n_seats", 8) or 8)
        self.n = n
        self.pays = pad_payouts(config.get("payouts"), max(8, n))
        icm_on = P["ICM_ON"]
        if icm_on is None:
            icm_on = not is_wta(self.pays)
        self.icm_on = bool(icm_on) and self.version >= 1
        self.icm = None
        self.bf_table = {}
        self.bf_cost_ns = 0
        if self.icm_on:
            self.icm, self.bf_table, self.bf_cost_ns = icm_pack(self.pays, n, P["BF_MIN"], P["BF_MAX"],
                                                                P["BF_NO_GAIN"])
        # a BF may start only while it can still finish inside the budget (own clock, wall time like the rules)
        self.bf_start_limit = P["GUARD_NS"] - P["BF_COST_SAFETY"] * self.bf_cost_ns
        seed = config.get("seed", 0)
        self.rng = random.Random(seed if isinstance(seed, (int, str)) else 0)
        self.stats = [SeatStat() for _ in range(n)]
        self.tr = HandTracker(P, self.stats, n)
        self.bf_cache = {}
        self.bf_hand = None
        self.bf_budget = 0
        self.t0 = 0
        self.types = {}
        self.open_prior = P["OPEN_PRIOR"] or P["OPEN_PCT"]
        self.guard_trips = 0      # decisions where the guard stopped optional work (diagnostic)
        self.bf_computed = 0
        self.n_jam_gated = 0      # v1+ first-in jam decisions that used the gated (caller-side) ICM formula
        self.n_jam_hero = 0       # ... that used the default hero-only tightening

    # ------------------------------------------------------------------ events (untimed in the kit)
    def observe(self, ev):
        t = ev.get("type")
        tr = self.tr
        if t == "action":
            if ev.get("hand_no") == tr.hand_no:
                si = STREET_INDEX.get(ev.get("street"), tr.street)
                if si > tr.street:
                    tr.new_street(si)
                tr.feed(ev["seat"], ev["action"], ev.get("amount", 0))
        elif t == "hand_start":
            n = self.n
            stacks = list(ev.get("stacks", [0] * n))
            alive = list(ev.get("alive", [s > 0 for s in stacks]))
            tr.start_hand(ev.get("hand_no"), stacks, alive, ev.get("sb_seat", 0), ev.get("bb_seat", 1),
                          ev.get("bb", 20), ev.get("ante", 0))
            for p in ev.get("posts", ()):
                tr.feed(p[0], p[1], p[2])
        elif t == "street":
            si = STREET_INDEX.get(ev.get("street"), tr.street)
            if ev.get("hand_no") == tr.hand_no and si > tr.street:
                tr.new_street(si)

    # ------------------------------------------------------------------ bubble factors
    def bf(self, i, j):
        """BF of seat i vs seat j at this hand's start stacks: cache, else exact (budget and guard permitting),
        else the equal-stack table value for the players left (never 1.0 while ICM is on)."""
        if not self.icm_on:
            return 1.0
        key = i * 64 + j
        v = self.bf_cache.get(key)
        if v is not None:
            return v
        if self.bf_budget > 0 and time.perf_counter_ns() - self.t0 < self.bf_start_limit:
            self.bf_budget -= 1
            try:
                v = bubble_factor(self.icm, self.view_start, i, j, self.pays, self.P["BF_MIN"], self.P["BF_MAX"],
                                  self.P["BF_NO_GAIN"])
                self.bf_computed += 1
            except Exception:
                v = None
            if v is not None:
                self.bf_cache[key] = v
                return v
        elif self.bf_budget > 0:
            self.guard_trips += 1
            self.bf_budget = 0
        return self.bf_table.get(self.n_live, self.P["BF_DEFAULT"])

    def bf_mean(self, me, seats, hero_side):
        """Mean BF over seats (hero_side: BF(me, j); else BF(j, me)), biggest stacks first; uncomputed ones take the
        mean of the computed ones (or the table value)."""
        if not seats:
            return 1.0
        st = self.view_start
        order = sorted(seats, key=lambda j: -st[j])
        vals = []
        exact = []
        for j in order:
            key = (me * 64 + j) if hero_side else (j * 64 + me)
            v = self.bf_cache.get(key)
            if v is None and self.bf_budget > 0:
                v = self.bf(me, j) if hero_side else self.bf(j, me)
                if key not in self.bf_cache:
                    v = None
            if v is not None:
                exact.append(v)
            vals.append(v)
        fill = (sum(exact) / len(exact)) if exact else self.bf_table.get(self.n_live, self.P["BF_DEFAULT"])
        tot = 0.0
        for v in vals:
            tot += v if v is not None else fill
        return tot / len(vals)

    # ------------------------------------------------------------------ typing (v2; also the v1 jam-tight gate)
    def shrunk(self, hits, n, key):
        m, k = self.P["PRIOR"][key]
        return (hits + k * m) / (n + k)

    def jam_tight(self, j):
        """Observed folding to jams more than chip-EV Nash (v1 gate); v2 adds typed folders and nits."""
        st = self.stats[j]
        P = self.P
        if st.jam_faced >= P["MIN_JAMS_FACED_TIGHT"]:
            oe = (st.jam_called + P["OE_PRIOR"]) / (st.jam_exp + P["OE_PRIOR"])
            if oe <= P["OE_TIGHT"]:
                return True
        if self.version >= 2:
            t = self.seat_type(j)
            return t == "folder" or t == "nit"
        return False

    def is_folder(self, j):
        st = self.stats[j]
        return st.pf_dec >= self.P["MIN_OBS_FOLDER"] and st.pf_vol == 0 and st.vol_ever == 0 and not st.vol_hand

    def seat_type(self, j):
        t = self.types.get(j)
        if t is not None:
            return t
        P = self.P
        st = self.stats[j]
        if self.is_folder(j):
            t = "folder"
        elif st.pf_dec < P["MIN_OBS_TYPE"]:
            t = "unknown"
        elif self.shrunk(st.jam_first, st.jam_opp, "jam") >= P["T_MANIAC_JAM"]:
            t = "maniac"
        elif (st.jam_faced >= P["MIN_JAMS_FACED"]
              and self.shrunk(st.jam_called, st.jam_faced, "calljam") >= P["T_CALLER_CALLJAM"]) or \
                (st.post_faced >= P["MIN_POST_FACED"]
                 and self.shrunk(st.post_call, st.post_faced, "post_call") >= P["T_POST_CALL"]):
            t = "caller"
        elif st.late_opp >= P["MIN_LATE_OPP"] and \
                self.shrunk(st.late_open, st.late_opp, "late_open") >= P["T_STEAL_LATE"]:
            t = "stealer"
        elif self.shrunk(st.pf_vol, st.pf_dec, "vol") >= P["T_LOOSE_VOL"]:
            t = "loose"
        elif st.pf_dec >= P["MIN_OBS_NIT"] and self.shrunk(st.pf_vol, st.pf_dec, "vol") < P["T_NIT_VOL"]:
            t = "nit"
        else:
            t = "normal"
        self.types[j] = t
        return t

    def jam_range_of(self, j, J_formula, is_rejam):
        """Estimated range of seat j's jam. v0/v1: the first-in formula at its spot (also for re-jams, as tested);
        v2: re-jams get REJAM_PRIOR (P5); a typed maniac's range is its own shrunk jam rate (E10); loose floor."""
        if self.version < 2:
            return J_formula
        P = self.P
        t = self.seat_type(j)
        J = P["REJAM_PRIOR"] if is_rejam else J_formula
        if t == "maniac":
            st = self.stats[j]
            J = max(J, self.shrunk(st.jam_first, st.jam_opp, "jam"), P["J_FLOOR"]["maniac"])
        elif t == "loose" and not is_rejam:
            J = max(J, P["J_FLOOR"]["loose"])
        return J

    # ------------------------------------------------------------------ decide
    def decide(self, v, t0):
        self.t0 = t0
        self.bf_budget = self.P["BF_PER_DECISION"]
        tr = self.tr
        tr.sync_from_view(v)
        if self.bf_hand != v.hand_no:
            self.bf_hand = v.hand_no
            self.bf_cache = {}
        self.types = {}
        self.view_start = v.start
        nl = 0
        for s in v.start:
            if s > 0:
                nl += 1
        self.n_live = nl
        if v.street == 0:
            return self.preflop(v)
        return self.postflop(v)

    def decide_fallback(self, v, t0):
        """P15: the v0 decision (no ICM, no typing) for the same view. The shell calls this only when decide()
        raised, so an error in optional work costs the refinement, not the whole decision. The tracker resumes
        after the entry that failed (feed() counts an entry before processing it), so nothing is fed twice."""
        ver, icm = self.version, self.icm_on
        self.version, self.icm_on = 0, False
        try:
            return self.decide(v, t0)
        finally:
            self.version, self.icm_on = ver, icm

    # ------------------------------------------------------------------ preflop
    def preflop(self, v):
        P = self.P
        tr = self.tr
        me = v.me
        bb = v.bb
        ver = self.version
        n = v.n
        stacks, bets, in_hand = v.stacks, v.bets, v.in_hand
        tot = [stacks[j] + bets[j] for j in range(n)]
        cls = CLASS_OF[v.hole[0] * 52 + v.hole[1]]
        pp = PUSH_PCT[cls]
        cp = CALL_PCT[cls]
        pos = tr.pos.get(me)
        if tr.pf_raises == 0:
            # ---------------- unopened (limpers allowed)
            limpers = tr.limper_seats
            others = [j for j in range(n) if j != me and in_hand[j]]
            if not others:
                return ("check",) if v.to_call <= 0 else ("fold",)
            if v.to_call <= 0:
                # big blind option (limpers or a completed small blind): value-raise premiums only
                if limpers and pp <= P["BB_ISO_PCT"]:
                    S = min(tot[me], max(tot[j] for j in others)) / bb
                    if S <= P["PF_DEPTH_BB"]:
                        return ("allin",)
                    return ("raise", int((P["BB_ISO_BB"] + P["BB_ISO_LIMPER_BB"] * len(limpers)) * bb))
                return ("check",)
            behind = [j for j in tr.pre[pos + 1:] if in_hand[j]] if pos is not None else others
            callers = behind + [j for j in limpers if in_hand[j] and j not in behind]
            k = len(behind)
            if not callers:
                return ("check",) if v.to_call <= 0 else ("fold",)
            if ver >= 2 and P["FOLDER_STEAL"] and not limpers and behind and \
                    all(self.is_folder(j) for j in behind):
                return ("raise", v.min_to)                                # E09: free steal (min-raise any two)
            S = min(tot[me], max(tot[j] for j in callers)) / bb
            if S <= P["PF_DEPTH_BB"]:
                S_eq = equiv_depth(S, v.pot / bb, P)                      # antes and limps = more dead money
                J = jam_pct(S_eq, len(callers), P["JAM_FIT"]) * P["JAM_MULT"]
                if self.icm_on:
                    J = self.icm_jam(J, me, callers)
                if ver >= 2:
                    J = self.typed_jam(J, callers, S)
                if pp <= J:
                    return ("allin",)
                return ("fold",)
            op = by_k(P["OPEN_PCT"], k)
            if pp <= op:
                return ("raise", int((P["OPEN_BB"] + P["OPEN_LIMPER_BB"] * len(limpers)) * bb))
            return ("fold",)
        # ---------------- facing a raise
        r = tr.last_raiser
        if r < 0 or r == me:
            r = max((j for j in range(n) if j != me and in_hand[j]), key=lambda j: bets[j], default=-1)
            if r < 0:
                return ("check",) if v.to_call <= 0 else ("fold",)
        jam = tr.jam
        if jam is not None and in_hand[jam[0]] and v.all_in[jam[0]]:
            return self.call_jam(v, jam[0], jam[1], jam[2], cls, tot, near=False)
        near = stacks[r] <= max(P["NEAR_ALLIN_BB"] * bb, P["NEAR_ALLIN_FRAC"] * tot[r])
        S = min(tot[me], tot[r]) / bb
        if near:
            Jf = jam_pct(S, max(1, tr.open_k if tr.pf_raises == 1 else 1), P["JAM_FIT"])
            return self.call_jam(v, r, Jf, tr.pf_raises > 1, cls, tot, near=True)
        if v.to_call <= 0:
            return ("check",)
        committed = v.to_call >= P["COMMIT_TO_CALL_FRAC"] * tot[me] or S <= P["PF_DEPTH_BB"]
        late_small = False
        if ver >= 2:
            O, late_small = self.open_range(r, v)
        else:
            # tested brain: the raiser's range = the first-in jam formula at its own depth and position
            pr = tr.pos.get(r, 0)
            kr = max(1, len(tr.pre) - 1 - pr)
            O = jam_pct(tot[r] / bb, kr, P["JAM_FIT"])
        if committed:
            C = call_pct(O, S, P["CALL_FIT"])
            if self.icm_on:
                C = C / self.bf(me, r) ** P["CALL_LAM"]
            if cp <= P["RESTEAL_MULT"] * C:
                return ("allin",)
            if late_small and me == v.bb_seat and cp <= P["BB_DEFEND_PCT"] and \
                    v.to_call <= P["BB_DEFEND_MAX_FRAC"] * tot[me]:
                return ("call",)                                          # P6: BB defence vs a late small raise
            return ("fold",)
        if pp <= P["RAISE_TOP"]:
            if S <= P["FOURBET_JAM_MAX_BB"]:
                return ("allin",)
            return ("raise", int(P["THREEBET_MULT"] * max(bets)))
        if pp <= P["FLAT_TOP"] and v.to_call <= P["FLAT_MAX_STACK"] * tot[me]:
            return ("call",)
        if late_small and me == v.bb_seat and cp <= P["BB_DEFEND_PCT"] and \
                v.to_call <= P["BB_DEFEND_MAX_FRAC"] * tot[me]:
            return ("call",)
        return ("fold",)

    def icm_jam(self, J, me, callers):
        """v1 jam side: hero-only tightening by default; the tested caller-side widening only if every possible
        caller was observed folding to jams more than chip-EV Nash."""
        P = self.P
        bfh = self.bf_mean(me, callers, True)
        if all(self.jam_tight(j) for j in callers):
            bfo = self.bf_mean(me, callers, False)
            J = J * bfo ** P["JAM_OPP_EXP"] * bfh ** P["JAM_HERO_EXP_GATED"]
            self.n_jam_gated += 1
        else:
            J = J * bfh ** P["JAM_HERO_EXP"]
            self.n_jam_hero += 1
        return J if J < 1.0 else 1.0

    def typed_jam(self, J, callers, S):
        """v2: cap light jams into callers; widen only on evidence (folder BB or a BB folding to jams; jam-tight
        seats behind)."""
        P = self.P
        types = [self.seat_type(j) for j in callers]
        if any(t in ("caller", "loose", "maniac") for t in types):
            return min(J, P["LIGHT_JAM_CAP"])
        mult = 1.0
        if len(callers) == 1:
            j = callers[0]
            st = self.stats[j]
            b10, slope, bmin = P["SB_WIDEN_BE"]                            # break-even jam-call rate of the widen
            be = b10 - slope * (S - 10.0) if S > 10.0 else b10
            be = max(bmin, be) - P["SB_WIDEN_MARGIN"]
            if types[0] == "folder" or (st.jam_faced >= P["MIN_JAMS_FACED"]
                                        and self.shrunk(st.jam_called, st.jam_faced, "calljam") < be):
                mult = P["SB_WIDEN"]
        if all(self.jam_tight(j) for j in callers):
            mult = max(mult, P["JAMFOLD_WIDEN"])
        J *= mult
        return J if J < 1.0 else 1.0

    def open_range(self, r, v):
        """v2: estimated range of the raiser facing us (non-all-in): a first-in open gets the open prior by its
        position (floors for typed stealers / loose / maniacs; P6 floor for small late opens); a re-raise gets the
        tight re-jam prior unless the seat is a maniac."""
        P = self.P
        tr = self.tr
        t = self.seat_type(r)
        late_small = False
        if tr.pf_raises <= 1:
            k_r = tr.open_k if tr.open_seat == r else P["OPEN_K_UNKNOWN"]
            O = by_k(self.open_prior, k_r)
            if k_r <= P["LATE_SEATS"] and tr.raise_to <= P["BB_DEFEND_MAX_RAISE_BB"] * v.bb and \
                    tr.callers_since_raise == 0:
                late_small = True
                O = max(O, P["LATE_OPEN_FLOOR"])
            fl = P["J_FLOOR"].get(t)
            if fl is not None:
                O = max(O, fl)
        else:
            O = P["REJAM_PRIOR"]
            if t == "maniac":
                O = max(O, P["J_FLOOR"]["maniac"])
        return O, late_small

    def call_jam(self, v, jammer, J_formula, is_rejam, cls, tot, near):
        """Call (or re-jam over a near-all-in raise) by equity against the jammer's estimated range."""
        P = self.P
        tr = self.tr
        me = v.me
        n = v.n
        J = self.jam_range_of(jammer, J_formula, is_rejam)
        if P["CALL_MODEL"] == "fit":
            S = min(tot[me], tot[jammer]) / v.bb                          # P4: the jammer's stack
            C = call_pct(J, S, P["CALL_FIT"]) * P["CALL_MULT"]
            if self.icm_on:
                C = C / self.bf(me, jammer) ** P["CALL_LAM"]
            ok = CALL_PCT[cls] <= C
            return (("allin",) if near else ("call",)) if ok else ("fold",)
        eq = eq_vs(cls, J)
        Jc = J * P["CALLER_RANGE_FRAC"]
        n_left = 0
        responders = tr.jam_responders if tr.jam is not None and tr.jam[0] == jammer else ()
        for j in range(n):
            if j == me or j == jammer or not v.in_hand[j]:
                continue
            if j in responders:
                eq *= eq_vs(cls, Jc)                                      # already called the jam
            elif v.all_in[j]:
                jr = tr.jam_ranges.get(j)                                 # an earlier jam: its own estimated range;
                eq *= eq_vs(cls, self.jam_range_of(j, jr[0], jr[1]) if jr else 1.0)   # all-in from a blind: any two
            else:
                n_left += 1
        R = v.to_call
        contrib = v.contrib
        if near:
            R = min(v.stacks[me], v.to_call + v.stacks[jammer])
        if R <= 0:
            return ("check",)
        mine = contrib[me] + R
        W = 0
        for j in range(n):
            if j != me:
                c = contrib[j]
                if near and j == jammer:
                    c += v.stacks[jammer]
                W += c if c < mine else mine
        # winning returns our own chips already in (contrib) plus W; risk premium BF on the R chips we add
        gain = W + contrib[me]
        b = self.bf(me, jammer) if self.icm_on else 1.0
        need = b * R / (b * R + gain) + P["CALL_BEHIND_MARGIN"] * n_left + P["CALL_EQ_ADJ"]
        if eq >= need:
            return ("allin",) if near else ("call",)
        return ("fold",)

    # ------------------------------------------------------------------ postflop
    def postflop(self, v):
        P = self.P
        tr = self.tr
        me = v.me
        n = v.n
        ver = self.version
        cls = hand_class(v.hole[0], v.hole[1], v.board)
        pot = v.pot
        to_call = v.to_call
        stack = v.stacks[me]
        bets = v.bets
        cur = max(bets)
        opps = [j for j in range(n) if j != me and v.in_hand[j]]
        if not opps:
            return ("check",) if to_call <= 0 else ("call",)
        hu = len(opps) == 1
        if to_call <= 0:
            if cls >= 2:
                frac = P["VALUE_FRAC"]
                if ver >= 2 and any(self.seat_type(j) == "caller" for j in opps):
                    frac = P["VALUE_FRAC_CALLER"]
                return self.bet(v, cur + max(v.bb, int(frac * pot)))
            if ver >= 2 and hu:
                j = opps[0]
                if self.is_folder(j):
                    return ("raise", v.min_to)                            # E09: bet the minimum into a folder
                if v.street == 1 and tr.pf_aggr == me:
                    st = self.stats[j]
                    if st.flop_faced >= P["CBET_MIN_OBS"] and \
                            self.shrunk(st.flop_fold, st.flop_faced, "flop_fold") >= P["CBET_FOLD_T"]:
                        return self.bet(v, cur + max(v.bb, int(P["CBET_FRAC"] * pot)))
            if hu and v.street == 1 and tr.pf_aggr == me and P["CBET_AIR_UNKNOWN"] > 0 and \
                    self.rng.random() < P["CBET_AIR_UNKNOWN"]:
                return self.bet(v, cur + max(v.bb, int(P["CBET_FRAC"] * pot)))
            return ("check",)
        bettor = max(opps, key=lambda j: bets[j])
        big = to_call >= P["BIG_CALL_FRAC"] * stack
        b = self.bf(me, bettor) if (self.icm_on and big) else 1.0
        raised_me = bets[me] > 0
        light = ver >= 2 and self.seat_type(bettor) in P["LIGHT_CALL_TYPES"]
        if cls == 3:
            if big and b >= P["BF_RAISE3_MAX"]:
                return ("call",)
            return self.bet(v, int(P["RAISE_MULT"] * cur))
        ec = 2 if (light and cls >= 1) else cls
        if ec >= 2:
            if raised_me and not light:
                if to_call <= P["RAISED_CALL2_MAX_POT"] * pot and not big:
                    return ("call",)
                return ("fold",)
            if to_call <= P["CALL2_MAX_POT"] * pot and (not big or b < P["BF_CALL2_MAX"]):
                return ("call",)
            return ("fold",)
        if ec == 1 and not raised_me and to_call <= P["CALL1_MAX_POT"] * pot and not big:
            return ("call",)
        return ("fold",)

    def bet(self, v, to):
        """Bet or raise to `to` (street total); all-in if it would commit COMMIT_FRAC of our chips."""
        me = v.me
        chips = v.stacks[me] + v.bets[me]
        if to >= self.P["COMMIT_FRAC"] * chips:
            return ("allin",)
        return ("raise", to)
