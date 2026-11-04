"""Postflop engine: equity against an action-conditioned range, by lookup (POSTFLOP_ENGINE notes).

The idea behind the MIT 2024 GTO Wizard entry: equities are precomputed offline (bot/tools/make_postflop_tables.py,
Monte Carlo with the kit evaluator) and act() only looks them up. Pure standard library, no simulation here.

Lookup key: (street, board texture, hero class, opponent width).
  street    0 flop, 1 turn, 2 river
  texture   96 buckets: paired (2) x suits (4: rainbow / two-tone / three of a suit / four+) x connectedness (4: the
            most board ranks inside one 5-rank straight window, <=1 / 2 / 3 / 4+) x top card (3: 2-9 / T-J / Q-A)
  hero      75 classes = made (15) x draw (5)
            made: 0 no pair no overcard, 1 one overcard, 2 two overcards, 3 weak pair, 4 middle pair, 5 top pair kicker
            below T, 6 top pair kicker T+, 7 overpair, 8 two pair (both hole cards), 9 trips (one hole card + board
            pair), 10 set, 11 straight, 12 flush (best hole flush card below J), 13 flush J+, 14 full house or better
            (all made hands use a hole card; board-only hands count as the kicker classes below them)
            draw (flop and turn only, made below straight): 0 none, 1 gutshot, 2 open-ender (two or more straight
            ranks), 3 flush draw, 4 flush draw + straight draw
  width     0 tight (top 12% preflop), 1 medium (28%), 2 wide (55%) (push order; the brain maps typing labels)
Stored per key: the opponent's frequency in 4 strata of ITS hand on this board, with our cards removed (so the
table is blocker-aware), and our all-in equity against each stratum:
  s0 air (no pair, at most a gutshot), s1 weak (weak or middle pair, or a draw of 8+ outs), s2 medium (top pair,
  overpair), s3 strong (two pair or better)
Online, an action picks weights per stratum and equity = sum(w f e) / sum(w f). The weights are the field's
MEASURED behaviour by stratum (PARAMS["PE"]: who bets, who continues against a bet, who raises; bigger bets weigh
the weak strata less), so the engine plays one level of the game tree against that model:
  facing a bet   call when realized equity against the bettor's range (times the others' calling ranges) beats
                 the pot odds (bubble-factor weighted on big calls); raise heads-up when the raise EV beats the call
  checked to us  top pair or better always bets, heads-up at the size with the best EV (texture-dependent size set;
                 VALUE_FRAC_CALLER vs a typed caller, VALUE_FRAC multiway); weak / middle pairs check; strong draws
                 semi-bluff heads-up when the bet EV beats the check EV (which models the opponent betting after our
                 check when it acts after us); pure bluffs only heads-up, on dry boards, into a seat with observed
                 fold evidence, at a frequency that is higher with an A/K/Q blocker above the board
Fold equity uses the same continue probabilities, scaled per seat toward its shrunk observed postflop fold rate.
"""
import base64 as _b64

# ===================================== layout (must match bot/tools/make_postflop_tables.py) =========================
N_STREETS, N_TEX, N_MADE, N_DRAW, N_WIDTH, N_K = 3, 96, 15, 5, 3, 7
N_CLS = N_MADE * N_DRAW
WIDTH_NAMES = ("tight", "medium", "wide")
# typing label (brain.seat_type) -> width index (its range, and how often it continues against bets); anything else
# (normal, unknown, stealer, no typing) = medium. Betting style (how often it bets air and weak hands) is separate:
# a typed caller calls wide but does not bluff more, so only maniac / loose seats get the wide betting style.
WIDTH_OF_TYPE = {"folder": 0, "nit": 0, "maniac": 2, "loose": 2, "caller": 2}
BET_STYLE_OF_TYPE = {"folder": 0, "nit": 0, "maniac": 2, "loose": 2}

# ===================================== straight table (13-bit rank mask -> top rank of the best straight, -1) =====
STRAIGHT = [-1] * 8192


def _fill_straight():
    wins = [(3, 0x100F)] + [(top, 31 << (top - 4)) for top in range(4, 13)]   # ascending: higher tops overwrite
    full = 8191
    for top, w in wins:
        comp = full & ~w
        sub = comp
        while True:
            STRAIGHT[w | sub] = top
            if sub == 0:
                break
            sub = (sub - 1) & comp


_fill_straight()
POP5 = [bin(i).count("1") for i in range(32)]
_WHEEL = 0x100F


# ===================================== classifiers (shared with the table generator) ===============================
def texture(board):
    """Texture bucket 0..95 of a 3-5 card board (kit card ints)."""
    cnt = [0] * 13
    su = [0, 0, 0, 0]
    bm = 0
    top = 0
    paired = 0
    for c in board:
        r = c >> 2
        if cnt[r]:
            paired = 1
        cnt[r] += 1
        su[c & 3] += 1
        bm |= 1 << r
        if r > top:
            top = r
    m = max(su)
    sb = 0 if m <= 1 else (1 if m == 2 else (2 if m == 3 else 3))
    cn = 0
    for lo in range(9):
        x = POP5[(bm >> lo) & 31]
        if x > cn:
            cn = x
    x = bin(bm & _WHEEL).count("1")
    if x > cn:
        cn = x
    cb = 0 if cn <= 1 else (cn - 1 if cn <= 4 else 3)
    hb = 2 if top >= 10 else (1 if top >= 8 else 0)
    return ((paired * 4 + sb) * 4 + cb) * 3 + hb


def tex_parts(tex):
    """(paired, suits 0..3, connectedness 0..3, top 0..2) of a texture id."""
    hb = tex % 3
    t = tex // 3
    cb = t % 4
    t //= 4
    return t // 4, t % 4, cb, hb


def hand_feat(h1, h2, board):
    """(made 0..14, draw 0..4) of a two-card hand on a 3-5 card board."""
    r1, r2 = h1 >> 2, h2 >> 2
    s1, s2 = h1 & 3, h2 & 3
    cnt = [0] * 13
    su = [0, 0, 0, 0]
    bm = 0
    for c in board:
        r = c >> 2
        cnt[r] += 1
        su[c & 3] += 1
        bm |= 1 << r
    nb = len(board)
    top = bm.bit_length() - 1
    rest = bm & ~(1 << top)
    second = rest.bit_length() - 1          # -1 when every board card has the top rank
    board_pair = 0                           # a board rank seen twice or more (rank + 1, 0 = none)
    for r in range(13):
        if cnt[r] >= 2:
            board_pair = r + 1
    # ---- full house or better with a hole card
    if r1 == r2:
        c = cnt[r1]
        if c >= 2 or (c == 1 and board_pair):
            return 14, 0
    else:
        c1, c2 = cnt[r1], cnt[r2]
        if c1 >= 3 or c2 >= 3:
            return 14, 0
        if c1 and c2 and (c1 >= 2 or c2 >= 2):
            return 14, 0
        if (c1 >= 2 or c2 >= 2):
            tr = r1 if c1 >= 2 else r2
            for r in range(13):
                if r != tr and cnt[r] >= 2:
                    return 14, 0
    # ---- flush with a hole card
    fs = -1
    if su[s1] + 1 + (s2 == s1) >= 5:
        fs = s1
    elif su[s2] + 1 + (s1 == s2) >= 5:
        fs = s2
    am = bm | (1 << r1) | (1 << r2)
    if fs >= 0:
        hi = -1
        if s1 == fs:
            hi = r1
        if s2 == fs and r2 > hi:
            hi = r2
        # straight flush with a hole card
        fm = 0
        for c in board:
            if c & 3 == fs:
                fm |= 1 << (c >> 2)
        bfm = fm
        if s1 == fs:
            fm |= 1 << r1
        if s2 == fs:
            fm |= 1 << r2
        if STRAIGHT[fm] > STRAIGHT[bfm]:
            return 14, 0
        return (13 if hi >= 9 else 12), 0
    # ---- straight with a hole card
    if STRAIGHT[am] > STRAIGHT[bm]:
        return 11, 0
    # ---- sets, trips, two pair, pairs
    if r1 == r2:
        if cnt[r1] == 1:
            made = 10
        elif r1 > top:
            made = 7
        elif r1 > second:
            made = 4 if second >= 0 else 3   # between the board's top two ranks
        else:
            made = 3
    else:
        c1, c2 = cnt[r1], cnt[r2]
        if c1 >= 2 or c2 >= 2:
            made = 9
        elif c1 and c2:
            made = 8
        elif c1 or c2:
            pr, kick = (r1, r2) if c1 else (r2, r1)
            if pr == top:
                made = 6 if kick >= 8 else 5
            elif pr == second:
                made = 4
            else:
                made = 3
        else:
            made = (1 if r1 > top else 0) + (1 if r2 > top else 0)
    if nb >= 5:
        return made, 0
    # ---- draws
    fd = su[s1] + 1 + (s2 == s1) == 4 or su[s2] + 1 + (s1 == s2) == 4
    outs = 0
    for r in range(13):
        b = 1 << r
        if am & b:
            continue
        if STRAIGHT[am | b] > STRAIGHT[bm | b]:
            outs += 1
    if fd:
        draw = 4 if outs else 3
    elif outs >= 2:
        draw = 2
    elif outs == 1:
        draw = 1
    else:
        draw = 0
    return made, draw


def stratum(made, draw):
    """Opponent stratum of a (made, draw) feature: 0 air, 1 weak pair or strong draw, 2 top pair / overpair,
    3 two pair or better."""
    if made >= 8:
        return 3
    if made >= 5:
        return 2
    if made >= 3 or draw >= 2:
        return 1
    return 0


# ===================================== tables =====================================
DATA = None
TABLE_INFO = "postflop engine: no tables"
try:
    try:
        from . import postflop_tables as _PT
    except ImportError:
        import importlib.util as _ilu
        import os as _os
        import sys as _sys
        _nm = "nolimit_bot_postflop_tables"
        _PT = _sys.modules.get(_nm)
        if _PT is None:
            _sp = _ilu.spec_from_file_location(_nm, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                                                  "postflop_tables.py"))
            _PT = _ilu.module_from_spec(_sp)
            _sys.modules[_nm] = _PT
            _sp.loader.exec_module(_PT)
    if tuple(_PT.LAYOUT) == (N_STREETS, N_TEX, N_CLS, N_WIDTH, N_K):
        _d = _b64.b64decode(_PT.DATA_B64)
        if len(_d) == N_STREETS * N_TEX * N_CLS * N_WIDTH * N_K:
            DATA = _d
            TABLE_INFO = "postflop engine: %s" % getattr(_PT, "INFO", "tables")
except Exception:          # noqa: BLE001 - no tables: the engine is unavailable and the brain keeps its old path
    DATA = None


def offset(street, tex, cls, w):
    return (((street * N_TEX + tex) * N_CLS + cls) * N_WIDTH + w) * N_K


def cell(street, tex, cls, w):
    """(f0, f1, f2, f3, e0, e1, e2, e3) floats of one key (tests and tools)."""
    o = offset(street, tex, cls, w)
    d = DATA
    f0, f1, f2 = d[o] / 255.0, d[o + 1] / 255.0, d[o + 2] / 255.0
    f3 = 1.0 - f0 - f1 - f2
    if f3 < 0.0:
        f3 = 0.0
    return f0, f1, f2, f3, d[o + 3] / 255.0, d[o + 4] / 255.0, d[o + 5] / 255.0, d[o + 6] / 255.0


def eq_weighted(c, w0, w1, w2, w3):
    """Equity against the range weighted by stratum weights; c = cell(...)."""
    a0, a1, a2, a3 = w0 * c[0], w1 * c[1], w2 * c[2], w3 * c[3]
    den = a0 + a1 + a2 + a3
    if den <= 1e-9:
        return c[0] * c[4] + c[1] * c[5] + c[2] * c[6] + c[3] * c[7]
    return (a0 * c[4] + a1 * c[5] + a2 * c[6] + a3 * c[7]) / den


def call_weights(c, base, frac, ref, obs_hits, obs_n, k_obs, strong_floor):
    """Probabilities that each stratum calls a bet of `frac` x pot: base^(frac/ref) (strong always calls), then
    scaled so the implied fold rate moves toward the seat's observed fold rate (shrunk to the model's own rate with
    k_obs pseudo-observations). Returns (c0, c1, c2, c3, fold_prob)."""
    e = frac / ref
    c0 = base[0] ** e
    c1 = base[1] ** e
    c2 = base[2] ** e
    c3 = base[3]
    fm = c[0] * (1.0 - c0) + c[1] * (1.0 - c1) + c[2] * (1.0 - c2) + c[3] * (1.0 - c3)
    if obs_n > 0 and fm < 0.99:
        o = (obs_hits + k_obs * fm) / (obs_n + k_obs)
        r = (1.0 - o) / (1.0 - fm)
        c0 = min(1.0, c0 * r)
        c1 = min(1.0, c1 * r)
        c2 = min(1.0, c2 * r)
        c3 = min(1.0, max(strong_floor, c3 * r))
        fm = c[0] * (1.0 - c0) + c[1] * (1.0 - c1) + c[2] * (1.0 - c2) + c[3] * (1.0 - c3)
    return c0, c1, c2, c3, fm


# ===================================== the decision =====================================
# One level of the game tree, with the opponents' responses taken from measured stratum frequencies (PARAMS["PE"]):
#   W_BET[width][street][s]   P(an opponent holding stratum s bets when it can) - also the shape of a bettor's range
#   W_CALL[width][street][s]  P(it continues against a bet of REF_FRAC x pot); a bet of f x pot uses w ** (f / REF)
#   W_RAISE[street][s]        P(it raises our bet) - also the shape of a raiser's range
#   CONT_RAISE[s]             P(it continues when we raise its bet)
def decide(h1, h2, board, pot, to_call, stack, opps, bettor, raised_me, bf, caller_vs, cur, bb, last, cfg, rnd):
    """One postflop decision. opps: list of (seat, width, fold_hits, fold_n, matched, bluff_ok, cbet, allin, style) for every
    opponent still in the hand: fold counts 0/0 = no evidence (the model's own fold rate is used); matched = it has
    already put in the current bet; bluff_ok = observed fold evidence that allows a pure bluff into it; cbet = it
    was the preflop aggressor and this is the flop (its betting range is a c-bet range); allin = it cannot act (it
    holds its whole range and cannot fold); style = betting-style index of W_BET. bettor: seat of the biggest bet when to_call > 0; raised_me: that bet
    raised ours. bf: bubble factor for a big call (1.0 = chip EV). caller_vs: a typed caller is in the hand (value
    bets use VALUE_FRAC_CALLER). last: nobody acts after us on this street (a check closes it). The brain keeps E09
    (the minimum into a typed folder) before calling this.
    Returns ("check",) / ("fold",) / ("call",) / ("bet", to_total) / ("raise", to_total) / None (no tables or the
    component is switched off: the class rules decide)."""
    d = DATA
    if d is None:
        return None
    nb = len(board)
    if nb < 3 or nb > 5:
        return None
    st = nb - 3
    tex = texture(board)
    made, draw = hand_feat(h1, h2, board)
    cls = made * N_DRAW + draw
    grp = 0 if (made <= 2 and draw <= 1) else (1 if (made <= 4 and draw >= 2) else 2)
    real = cfg["REAL"][grp][st]
    if to_call <= 0:
        if not cfg["FIRST"]:
            return None
        return _first(h1, h2, board, st, tex, made, draw, cls, pot, stack, opps, caller_vs, cur, bb, last, real, cfg,
                      rnd)
    if not cfg["FACING"]:
        return None
    return _facing(st, tex, cls, pot, to_call, stack, opps, bettor, raised_me, bf, cur, real, cfg)


def _first(h1, h2, board, st, tex, made, draw, cls, pot, stack, opps, caller_vs, cur, bb, last, real, cfg, rnd):
    paired, sb, cb, hb = tex_parts(tex)
    dry = sb <= 1 and cb <= 1
    n_opp = len(opps)
    bluff = made <= 2 and draw <= 1
    semi = made <= 4 and draw >= 2
    if bluff:
        if n_opp != 1 or not dry or not opps[0][5] or opps[0][7]:
            return ("check",)
        top = 0
        for c in board:
            r = c >> 2
            if r > top:
                top = r
        hi = max(h1 >> 2, h2 >> 2)
        blocker = hi > top and hi >= 10           # an A/K/Q above the board: blocks top-kicker and overpair calls
        freq = cfg["BLUFF_FREQ"] * (1.0 if blocker else cfg["BLUFF_NOBLOCK"])
        if rnd.random() >= freq:
            return ("check",)
    elif semi and n_opp > cfg["SEMI_MAX_OPP"]:
        return ("check",)
    value = made >= 5                         # top pair or better: always bet (the class rules' value bet) ...
    if not value and not semi and not bluff:
        return ("check",)                     # weak and middle pairs without a strong draw check (class rules)
    hu = n_opp == 1 and not opps[0][7]
    if value and (caller_vs or not hu):       # ... sized VALUE_FRAC_CALLER vs a typed caller, VALUE_FRAC multiway
        frac = cfg["VALUE_FRAC_CALLER"] if caller_vs else cfg["VALUE_FRAC"]
        return ("bet", cur + max(bb, int(frac * pot)))
    sizes = cfg["SIZES_DRY"] if dry else cfg["SIZES_WET"]
    ref = cfg["REF_FRAC"]
    kobs = cfg["K_OBS"]
    sfl = cfg["STRONG_FLOOR"]
    cells = []
    g = 1.0
    for o in opps:
        c = cell(st, tex, cls, o[1])
        cells.append(c)
        g *= c[0] * c[4] + c[1] * c[5] + c[2] * c[6] + c[3] * c[7]
    if hu and not last:
        # we act first: after our check the opponent bets with W_BET; we then call (all its bets) or fold
        c = cells[0]
        wb = cfg["W_BET"][opps[0][8]][st]
        bo = cfg["OPP_BET_FRAC"] * pot
        chk = 0.0
        bet_val = 0.0
        for s in range(4):
            fs = c[s]
            es = c[4 + s]
            chk += fs * (1.0 - wb[s]) * real * es * pot
            bet_val += fs * wb[s] * (real * es * (pot + 2.0 * bo) - bo)
        ev_check = chk + (bet_val if bet_val > 0.0 else 0.0)
    else:
        ev_check = real * g * pot
    best, best_ev = 0.0, ev_check + cfg["BET_MARGIN"] * pot
    if value:                                 # heads-up value: the size with the best EV, never a check
        best_ev = -1e18
    wr = cfg["W_RAISE"][st]
    rm = cfg["RAISE_MULT"]
    for frac in sizes:
        B = frac * pot
        if B > stack:
            B = stack
        if hu:
            c = cells[0]
            o = opps[0]
            c0, c1, c2, c3, fp = call_weights(c, cfg["W_CALL"][o[1]][st], frac, ref, o[2], o[3], kobs, sfl)
            cs = (c0, c1, c2, c3)
            ev = fp * pot
            rv_call = 0.0
            rv_fold = 0.0
            R = rm * B
            if R > stack:
                R = stack
            for s in range(4):
                fs = c[s]
                es = c[4 + s]
                rho = wr[s] if wr[s] < cs[s] else cs[s]
                ev += fs * (cs[s] - rho) * (real * es * (pot + 2.0 * B) - B)
                rv_fold -= fs * rho * B
                rv_call += fs * rho * (real * es * (pot + 2.0 * R) - R)
            ev += rv_call if rv_call > rv_fold else rv_fold
        else:
            pf = 1.0
            eqc = 1.0
            for i, o in enumerate(opps):
                c = cells[i]
                if o[7]:                           # all-in: cannot fold, holds its whole range; no steal possible
                    eqc *= c[0] * c[4] + c[1] * c[5] + c[2] * c[6] + c[3] * c[7]
                    pf = 0.0
                    continue
                c0, c1, c2, c3, fp = call_weights(c, cfg["W_CALL"][o[1]][st], frac, ref, o[2], o[3], kobs, sfl)
                if pf > 0.0:
                    pf *= fp
                eqc *= eq_weighted(c, c0, c1, c2, c3)
            ev = pf * pot + (1.0 - pf) * (real * eqc * (pot + 2.0 * B) - B)
        if ev > best_ev:
            best, best_ev = frac, ev
    if best <= 0.0:
        return ("check",)
    return ("bet", cur + max(bb, int(best * pot)))


def _facing(st, tex, cls, pot, to_call, stack, opps, bettor, raised_me, bf, cur, real, cfg):
    tc = to_call if to_call < stack else stack
    pot_eff = pot - (to_call - tc)                 # chips we cannot match go back
    before = pot - to_call                         # pot before the bet we face (the kit's pot includes it)
    frac = to_call / before if before > 0 else 1.0
    ref = cfg["REF_FRAC"]
    eq = 1.0
    cb = None
    wb = None
    n_opp = len(opps)
    for o in opps:
        c = cell(st, tex, cls, o[1])
        if o[0] == bettor:
            if raised_me:
                w = cfg["W_RAISE"][st]
            else:
                w = cfg["W_BET"][o[8]][st]
                if o[6]:                           # the bettor was the preflop aggressor on the flop: c-bet range
                    cw = cfg["CBET_W"]
                    w = (w[0] if w[0] > cw[0] else cw[0], w[1] if w[1] > cw[1] else cw[1], w[2], w[3])
                if frac > ref:                     # bigger bets weigh the weak strata less
                    m = (ref / frac) ** cfg["SIZE_EXP"]
                    w = (w[0] * m, w[1] * m, w[2], w[3])
            cb, wb = c, w
            eq *= eq_weighted(c, w[0], w[1], w[2], w[3])
        elif o[7]:
            eq *= c[0] * c[4] + c[1] * c[5] + c[2] * c[6] + c[3] * c[7]
        else:
            c0, c1, c2, c3, fp = call_weights(c, cfg["W_CALL"][o[1]][st], frac, ref, o[2], o[3], cfg["K_OBS"],
                                              cfg["STRONG_FLOOR"])
            e = eq_weighted(c, c0, c1, c2, c3)
            eq *= e if o[4] else fp + (1.0 - fp) * e
    need = bf * tc / (bf * tc + pot_eff) + cfg["CALL_MARGIN"]
    ev_call = real * eq * (pot_eff + tc) - tc
    can_raise = tc < stack and bf < cfg["BF_RAISE_MAX"]
    if can_raise and raised_me:
        if eq >= cfg["RERAISE_EQ"]:
            return ("raise", int(cfg["RAISE_MULT"] * cur))
    elif can_raise and eq >= cfg["RAISE_MIN_EQ"][st]:
        R = cfg["RAISE_MULT"] * cur
        if n_opp == 1 and cb is not None:
            if R > stack + cur - to_call:          # our chips behind plus what we already put in this street
                R = stack + cur - to_call
            kr = cfg["CONT_RAISE"]
            num = 0.0
            den = 0.0
            for s in range(4):
                m = cb[s] * wb[s]
                den += m
                num += m * ((1.0 - kr[s]) * pot + kr[s] * (real * cb[4 + s] * (pot + 2.0 * R - cur) - R))
            if den > 1e-9:
                ev_raise = num / den
                if ev_raise > (ev_call if ev_call > 0.0 else 0.0) + cfg["BET_MARGIN"] * pot:
                    return ("raise", int(R))
        elif eq >= cfg["RAISE_EQ_MULTI"]:
            return ("raise", int(R))
    if real * eq >= need:
        return ("call",)
    return ("fold",)
