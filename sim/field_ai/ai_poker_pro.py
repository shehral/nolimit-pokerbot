"""ai_poker_pro: a sit-and-go style bot for the No Limit kit (standard library + sim.kit only).

Strategy, in order of priority inside act():
  1. Short stack (<= ~12 bb effective, ante-adjusted) and unopened pot: Nash-style push/fold. Push range is a
     percentile of the 169 starting hands that depends on effective stack and how many players are left to act,
     tightened by an ICM "bubble factor" (risk premium) computed with the kit's exact ICM.
  2. Facing an all-in (or a bet that commits most of our stack): estimate the villain's range from the action,
     estimate our equity against it, and call only if it beats the ICM-required equity
     req = (EV_fold - EV_lose) / (EV_win - EV_lose)  (dollar/points EV via Malmuth-Harville ICM).
  3. Deeper stacks: position-aware open-raise ranges, 3-bet (value + a few suited-wheel-ace bluffs) / call / fold,
     4-bet-shove or fold versus 3-bets, 3-bet-shove resteals at 12-25 bb.
  4. Postflop: simple made-hand / draw classification. Preflop aggressor c-bets the flop (heads-up: most hands,
     multiway: pair+), barrels the turn with good hands and draws, value-bets rivers. Facing bets: pot odds vs
     hand class. Low SPR with a good hand -> get it in.
No Monte Carlo inside act(); the decision cost is dominated by up to 3 ICM evaluations (~tens of us).
Every path is wrapped: on any internal error the bot returns check-or-fold (legal by construction).
"""
import random

try:
    from sim.kit.agent import Agent
    from sim.kit.cards import CARD_INT, evaluate
    from sim.kit.icm import make_icm
except ImportError:  # flat layout
    from agent import Agent
    from cards import CARD_INT, evaluate
    from icm import make_icm

RANKS = "23456789TJQKA"
_R = {r: i for i, r in enumerate(RANKS)}

# ------------------------------------------------------------------ preflop hand table (169 classes)
# Approximate all-in equity vs a random hand (fitted to well-known values: AA .852, 22 .503, AKo .654,
# A2o .555, 32o .32). Only the ORDER and rough level matter.
_PAIR_EQ = [0.503, 0.537, 0.570, 0.603, 0.633, 0.662, 0.691, 0.721, 0.751, 0.775, 0.799, 0.824, 0.852]
_HI_BASE = [0.0, 0.315, 0.315, 0.325, 0.33, 0.345, 0.355, 0.375, 0.405, 0.425, 0.455, 0.485, 0.555]


def _eqr(hi, lo, suited):
    if hi == lo:
        return _PAIR_EQ[hi]
    slope = 0.009 + (12 - hi) * 0.0025
    e = _HI_BASE[hi] + (lo) * slope
    if suited:
        e += 0.025 if hi >= 10 else 0.03
    gap = hi - lo
    if gap == 1:
        e += 0.008
    elif gap == 2:
        e += 0.004
    if hi == 12 and lo <= 3:      # wheel aces play a bit better
        e += 0.004
    return e


def _build_tables():
    classes = []
    for hi in range(13):
        for lo in range(hi + 1):
            if hi == lo:
                classes.append(((hi, lo, False), _eqr(hi, lo, False), 6))
            else:
                classes.append(((hi, lo, True), _eqr(hi, lo, True), 4))
                classes.append(((hi, lo, False), _eqr(hi, lo, False), 12))
    classes.sort(key=lambda x: -x[1])
    pct = {}
    cum = 0
    eqs = []   # (cumulative fraction, mean eqr of range so far)
    wsum = 0.0
    for key, e, w in classes:
        cum += w
        wsum += e * w
        pct[key] = cum / 1326.0          # fraction of combos at least as good (including this class)
        eqs.append((cum / 1326.0, wsum / cum))
    return pct, eqs


PCT, _RANGE_EQ = _build_tables()


def range_avg_eqr(frac):
    """Mean equity-vs-random of the top `frac` of hands."""
    for f, e in _RANGE_EQ:
        if f >= frac:
            return e
    return _RANGE_EQ[-1][1]


def hand_key(hole):
    a, b = _R[hole[0][0]], _R[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    return (hi, lo, hi != lo and hole[0][1] == hole[1][1])


def equity_vs_range(hole, frac):
    k = hand_key(hole)
    me = _eqr(*k)
    e = 0.5 + 1.45 * (me - range_avg_eqr(max(frac, 0.005)))
    if k[0] == k[1]:          # pairs are flips vs overcards: pull toward the middle less harshly
        e = max(e, 0.33)
    return 0.12 if e < 0.12 else 0.88 if e > 0.88 else e


def premium(hole):
    """2 = QQ+ / AK, 1 = JJ, TT, AQs, AQo, AJs, KQs, 0 = other."""
    hi, lo, suited = hand_key(hole)
    if hi == lo:
        return 2 if hi >= 10 else 1 if hi >= 8 else 0
    if hi == 12 and lo == 11:
        return 2
    if (hi == 12 and lo == 10) or (hi == 12 and lo == 9 and suited) or (hi == 11 and lo == 10 and suited):
        return 1
    return 0


# Nash-ish push fractions at 10 bb by number of players left to act behind (1 = SB vs BB).
_PUSH10 = [1.0, 0.58, 0.42, 0.31, 0.24, 0.19, 0.16, 0.14, 0.13, 0.12]
# Opening (raise-first-in) ranges for deeper stacks by players left behind.
_OPEN = [1.0, 0.42, 0.40, 0.29, 0.22, 0.18, 0.15, 0.13, 0.12, 0.11]


def push_fraction(eff_bb, n_behind):
    base = _PUSH10[min(n_behind, len(_PUSH10) - 1)]
    if eff_bb <= 0.5:
        return 1.0
    f = base * (10.0 / eff_bb) ** 0.85
    return 1.0 if f > 1.0 else f


def _straight_draw(ranks):
    """2 = open-ended (4 in a row, both ends live), 1 = gutshot (4 of a 5-window), 0 = none."""
    m = 0
    for r in ranks:
        m |= 1 << (r + 1)
        if r == 12:
            m |= 1
    best = 0
    for lo in range(0, 10):
        w = (m >> lo) & 0x1F
        if w == 0x1F:
            return 0   # already a straight
        if bin(w).count("1") == 4:
            best = max(best, 1)
    for lo in range(1, 10):
        if (m >> lo) & 0xF == 0xF and lo + 4 <= 13:
            best = 2
    return best


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        try:
            self.rng = random.Random(config.get("seed", seat))
        except Exception:
            self.rng = random.Random(seat)
        try:
            self.payouts = list(config.get("payouts") or [50, 30, 20, 0, 0, 0, 0, 0])
        except Exception:
            self.payouts = [50, 30, 20, 0, 0, 0, 0, 0]
        n = 10
        try:
            n = max(10, int(config.get("n_seats", 8)))
        except Exception:
            pass
        self.icm = make_icm(self.payouts, max_players=n)
        self.paid = sum(1 for p in self.payouts if p > 0)
        self.wta = self.paid <= 1
        # light opponent model: per seat, hands with a preflop action, and preflop all-in raises (jams)
        self.pf_hands = [0] * 16
        self.pf_jams = [0] * 16
        self._seen = set()

    def observe(self, event):
        try:
            if event.get("type") == "hand_start":
                self._seen = set()
            elif event.get("type") == "action" and event.get("street") == "preflop":
                s = event["seat"]
                if s != self.seat and 0 <= s < 16 and s not in self._seen:
                    self._seen.add(s)
                    self.pf_hands[s] += 1
                    if event.get("action") == "raise" and event.get("all_in"):
                        self.pf_jams[s] += 1
        except Exception:
            pass

    def _jam_rate(self, s):
        h = self.pf_hands[s] if 0 <= s < 16 else 0
        if h < 8:
            return None
        return (self.pf_jams[s] + 0.5) / (h + 4.0)

    # ---------------------------------------------------------------- entry
    def act(self, state):
        try:
            a = self._act(state)
            return self._legalize(a, state)
        except Exception:
            return {"action": "check"} if "check" in state.get("legal_actions", ()) else {"action": "fold"}

    @staticmethod
    def _legalize(a, state):
        legal = state["legal_actions"]
        kind = a.get("action")
        if kind == "raise":
            if "raise" not in legal:
                return {"action": "call"} if "call" in legal else {"action": "check"}
            lo, hi = state["min_raise_to"], state["max_raise_to"]
            t = int(a.get("amount", lo))
            t = lo if t < lo else hi if t > hi else t
            return {"action": "raise", "amount": t}
        if kind in legal:
            return {"action": kind}
        if kind == "fold" and "check" in legal:
            return {"action": "check"}
        if kind == "check" and "call" in legal:
            return {"action": "fold"}
        if kind == "call" and "check" in legal:
            return {"action": "check"}
        return {"action": "check"} if "check" in legal else {"action": "fold"}

    # ---------------------------------------------------------------- helpers
    def _shove(self, state):
        if "raise" in state["legal_actions"]:
            return {"action": "raise", "amount": state["max_raise_to"]}
        return {"action": "call"} if "call" in state["legal_actions"] else {"action": "check"}

    @staticmethod
    def _cf(state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}

    def _icm_ev(self, stacks, i):
        return self.icm(stacks, i)

    def _icm_required(self, state, v, call_amt):
        """ICM-required equity to call `call_amt` against seat v (heads-up approximation)."""
        me = self.seat
        stacks = list(state["stacks"])
        contrib = state["contributed"]
        pot = state["pot"]
        t_me = contrib[me] + call_amt
        t_v = contrib[v]
        P = pot + call_amt
        refund_v = t_v - t_me if t_v > t_me else 0
        refund_me = t_me - t_v if t_me > t_v else 0
        s_fold = stacks[:]
        s_fold[v] += pot
        s_win = stacks[:]
        s_win[me] = stacks[me] - call_amt + P - refund_v
        s_win[v] = stacks[v] + refund_v
        s_lose = stacks[:]
        s_lose[me] = stacks[me] - call_amt + refund_me
        s_lose[v] = stacks[v] + P - refund_me
        e_f = self.icm(s_fold, me)
        e_w = self.icm(s_win, me)
        e_l = self.icm(s_lose, me)
        den = e_w - e_l
        if den <= 1e-12:
            return 1.0
        r = (e_f - e_l) / den
        return 0.0 if r < 0 else 1.0 if r > 1 else r

    @staticmethod
    def _chip_required(state, call_amt):
        return call_amt / float(state["pot"] + call_amt) if call_amt > 0 else 0.0

    def _players_behind_preflop(self, state):
        """Seats still to act after us this preflop round (active, not all-in, with chips)."""
        n = state["n_seats"]
        bb = state["bb_seat"]
        me = self.seat
        if me == bb:
            return []
        out = []
        s = me
        for _ in range(n):
            s = (s + 1) % n
            if s == me:
                break
            if state["active"][s] and not state["all_in"][s] and state["stacks"][s] > 0:
                out.append(s)
            if s == bb:
                break
        return out

    def _bubble_factor(self, state, opp):
        """ICM risk premium vs chipEV for a hypothetical all-in against `opp` (>= 1 means tighten)."""
        if self.wta or opp is None:
            return 1.0
        me = self.seat
        tot_me = state["stacks"][me] + state["bets"][me]
        tot_o = state["stacks"][opp] + state["bets"][opp]
        risk = min(tot_me, tot_o)
        if risk <= 0:
            return 1.0
        base = [state["stacks"][i] + state["bets"][i] if state["alive"][i] else 0 for i in range(state["n_seats"])]
        dead = state["pot"] - sum(state["bets"])
        e0 = self.icm(base, me)
        w = base[:]
        w[me] += risk + max(dead, 0)
        w[opp] -= risk
        l = base[:]
        l[me] -= risk
        l[opp] += risk + max(dead, 0)
        ew, el = self.icm(w, me), self.icm(l, me)
        if ew - e0 <= 1e-12:
            return 2.0
        bf = (e0 - el) / (ew - e0)
        return 1.0 if bf < 1.0 else 3.0 if bf > 3.0 else bf

    # ---------------------------------------------------------------- main logic
    def _act(self, state):
        if state["street"] == "preflop":
            return self._preflop(state)
        return self._postflop(state)

    def _preflop(self, state):
        me = self.seat
        legal = state["legal_actions"]
        bb = max(1, state["bb"])
        sb = state["sb"]
        ante = state["ante"]
        stacks = state["stacks"]
        bets = state["bets"]
        to_call = state["to_call"]
        hole = state["hole"]
        pct = PCT[hand_key(hole)]
        prem = premium(hole)
        my_total = stacks[me] + bets[me]

        hist = state["history_this_hand"]
        raises = []
        limpers = 0
        for (s, a, amt) in hist:
            if a == "raise":
                raises.append((s, amt))
            elif a == "call" and not raises:
                limpers += 1
        n_raises = len(raises)
        behind = self._players_behind_preflop(state)
        n_behind = len(behind)
        others = [i for i in range(state["n_seats"]) if i != me and state["active"][i]]
        max_other = max([stacks[i] + bets[i] for i in others] or [0])
        eff = min(my_total, max_other)
        n_alive = state["n_alive"]
        orbit_cost = sb + bb + ante * n_alive
        eff_bb = eff * 1.5 / orbit_cost if orbit_cost > 0 else eff / bb   # ante-adjusted "bb"
        top_bet = max(bets)

        # ----- unopened (or limped) pot
        if n_raises == 0:
            if eff_bb <= 12.5 or (limpers == 0 and eff_bb <= 15 and n_behind <= 2):
                frac = push_fraction(eff_bb, n_behind) if n_behind > 0 else 0.0
                if limpers:
                    frac *= 0.7
                if n_behind > 0 and not self.wta and frac < 1.0:
                    big = max(behind, key=lambda i: stacks[i] + bets[i])
                    bf = self._bubble_factor(state, big)
                    frac *= (1.0 / bf) ** 1.4
                if n_behind == 0:   # BB, limped to us: shove strong hands, else check
                    if limpers and pct <= 0.12:
                        return self._shove(state)
                    return self._cf(state)
                if pct <= frac:
                    return self._shove(state)
                return self._cf(state)
            # deeper: raise-first-in
            frac = _OPEN[min(n_behind, len(_OPEN) - 1)] if n_behind > 0 else 0.0
            if eff_bb < 22:
                frac *= 0.85
            if limpers:
                frac *= 0.6
            if n_behind == 0:   # BB option after limps
                if pct <= 0.10 and "raise" in legal:
                    return {"action": "raise", "amount": top_bet + (3 + limpers) * bb}
                return self._cf(state)
            if pct <= frac:
                size = (2.5 if eff_bb >= 30 else 2.2) * bb + limpers * bb
                if state["sb_seat"] == me and n_behind == 1:
                    size = 3 * bb
                return {"action": "raise", "amount": int(size)}
            # SB completing in a limped pot cheaply with decent hands
            if limpers and "call" in legal and to_call <= bb // 2 + 1 and pct <= 0.45:
                return {"action": "call"}
            return self._cf(state)

        # ----- facing raise(s)
        last_raiser, last_amt = raises[-1]
        commit_frac = to_call / float(stacks[me]) if stacks[me] > 0 else 1.0
        villain_allin = state["all_in"][last_raiser] or stacks[last_raiser] == 0
        # villain range estimate
        if n_raises == 1:
            r_bb = (stacks[last_raiser] + bets[last_raiser]) * 1.5 / orbit_cost if orbit_cost else 10
            if villain_allin and last_amt >= 0.8 * (stacks[last_raiser] + bets[last_raiser]):
                # open shove: assume a Nash-ish push range for its stack/position, a bit wider (bots over-shove)
                vbehind = max(1, n_behind + 1)
                vr = min(1.0, push_fraction(r_bb, vbehind) * 1.15)
                jr = self._jam_rate(last_raiser)
                if jr is not None and jr > 0.12:
                    # a frequent jammer shoves far wider than its first-in share suggests
                    vr = max(vr, min(1.0, jr * 2.0))
            else:
                vr = 0.25
        elif n_raises == 2:
            vr = 0.07 if not villain_allin else 0.10
        else:
            vr = 0.035

        if villain_allin or commit_frac >= 0.4 or (my_total - bets[me] - to_call) < 3 * bb:
            # all-in style decision
            eq = equity_vs_range(hole, vr)
            if len([i for i in others if state["active"][i]]) > 1 and n_behind > 0:
                eq -= 0.05 * n_behind / 2.0          # others may still wake up
            call_amt = to_call
            req = self._icm_required(state, last_raiser, call_amt)
            if eq >= req + 0.01:
                # with the best hands prefer shoving over calling when raising is legal and not all-in already
                if "raise" in legal and prem >= 2:
                    return self._shove(state)
                return {"action": "call"} if "call" in legal else self._shove(state)
            return self._cf(state)

        # non-committing raise
        open_size_bb = last_amt / float(bb)
        if n_raises == 1:
            # 12-25 bb: resteal shove
            if eff_bb <= 25:
                vbase = 0.25
                shove_frac = 0.06 + (0.06 if eff_bb <= 18 else 0.0)
                if self._bubble_factor(state, last_raiser) > 1.3:
                    shove_frac *= 0.7
                if pct <= shove_frac or prem >= 2:
                    return self._shove(state)
                if pct <= 0.14 and to_call <= 2.5 * bb and n_behind <= 1 and eff_bb > 16:
                    return {"action": "call"}
                return self._cf(state)
            in_pos = n_behind == 0 or (state["button"] == me)
            is_bb = me == state["bb_seat"]
            late_open = open_size_bb <= 3.5
            if prem >= 2 or (prem == 1 and late_open and self.rng.random() < 0.5):
                size = last_amt * (3 if in_pos else 3.8) + limpers * bb
                return {"action": "raise", "amount": int(size)}
            k = hand_key(hole)
            if k[0] == 12 and k[2] and k[1] <= 3 and self.rng.random() < 0.4 and late_open:
                size = last_amt * (3 if in_pos else 3.8)
                return {"action": "raise", "amount": int(size)}
            call_frac = 0.12 if in_pos else 0.08
            if is_bb:
                call_frac = 0.30 if to_call <= 2 * bb else 0.18
            if to_call > 4 * bb:
                call_frac *= 0.6
            if pct <= call_frac and "call" in legal:
                return {"action": "call"}
            return self._cf(state)
        # facing a 3-bet or more (non-committing)
        if prem >= 2 and (n_raises == 2 or pct <= 0.012 or hand_key(hole)[0] == hand_key(hole)[1]):
            return self._shove(state)
        if n_raises == 2 and prem >= 1 and commit_frac <= 0.25:
            return {"action": "call"}
        return self._cf(state)

    # ---------------------------------------------------------------- postflop
    def _classify(self, hole, board):
        """Return (strength 0..5, draw 0..2)."""
        cards = [CARD_INT[c] for c in hole] + [CARD_INT[c] for c in board]
        cat = evaluate(cards) >> 20
        h1, h2 = _R[hole[0][0]], _R[hole[1][0]]
        br = [_R[c[0]] for c in board]
        bmax = max(br)
        bcount = {}
        for r in br:
            bcount[r] = bcount.get(r, 0) + 1
        board_pairs = sum(1 for v in bcount.values() if v >= 2)
        board_trips = any(v >= 3 for v in bcount.values())
        s = 0
        if cat >= 6:
            s = 5
            if board_trips and h1 not in bcount and h2 not in bcount and h1 != h2:
                s = 2
        elif cat == 5 or cat == 4:
            s = 4
            # flush/straight on a paired board, or needing just one hole card low: still decent
        elif cat == 3:
            if h1 == h2 and h1 in bcount:
                s = 5   # set
            elif board_trips:
                s = 1 if max(h1, h2) >= 11 else 0
            else:
                s = 4   # trips with one hole card
        elif cat == 2:
            if h1 != h2 and h1 in bcount and h2 in bcount:
                s = 4
            elif h1 == h2 and h1 > bmax:
                s = 3
            elif board_pairs >= 2 and h1 not in bcount and h2 not in bcount and h1 != h2:
                s = 1 if max(h1, h2) >= 11 else 0
            else:
                # one pair from hole + board pair
                hr = h1 if h1 in bcount else h2
                s = 3 if hr == bmax else 2
        elif cat == 1:
            if h1 == h2:
                if h1 > bmax:
                    s = 3
                elif len(set(br)) >= 2 and h1 > sorted(set(br))[-2]:
                    s = 2
                else:
                    s = 1
            elif h1 in bcount or h2 in bcount:
                hr, kick = (h1, h2) if h1 in bcount else (h2, h1)
                if hr == bmax:
                    s = 3 if kick >= 8 or hr == 12 else 2
                else:
                    srt = sorted(set(br), reverse=True)
                    s = 2 if len(srt) > 1 and hr == srt[1] else 1
            else:
                s = 0   # board pair only
        draw = 0
        if len(board) < 5:
            suits = {}
            for c in list(hole) + list(board):
                suits[c[1]] = suits.get(c[1], 0) + 1
            for su, cnt in suits.items():
                if cnt == 4 and (hole[0][1] == su or hole[1][1] == su):
                    draw = 2
            sd = _straight_draw([h1, h2] + br)
            if sd and sd > _straight_draw(br):
                draw = max(draw, sd)
        return s, draw

    def _postflop(self, state):
        me = self.seat
        legal = state["legal_actions"]
        pot = state["pot"]
        to_call = state["to_call"]
        stacks = state["stacks"]
        bets = state["bets"]
        street = state["street"]
        bb = max(1, state["bb"])
        board = state["board"]
        hist = state["history_this_hand"]
        ss = state["street_starts"]
        strength, draw = self._classify(state["hole"], board)
        n_opp = sum(1 for i in range(state["n_seats"]) if i != me and state["active"][i])
        top_bet = max(bets)
        my_stack = stacks[me]
        spr = my_stack / float(pot) if pot > 0 else 10.0

        # preflop aggressor?
        pf_end = ss[1] if len(ss) > 1 else len(hist)
        pf_aggr = None
        for (s, a, amt) in hist[:pf_end]:
            if a == "raise":
                pf_aggr = s
        # who has bet on previous street (was I the aggressor last street?)
        i_was_aggr = pf_aggr == me
        if street in ("turn", "river") and len(ss) >= 2:
            prev_start = ss[-2] if len(ss) >= 2 else 0
            cur_start = ss[-1]
            last_aggr = None
            for (s, a, amt) in hist[prev_start:cur_start]:
                if a == "raise":
                    last_aggr = s
            i_was_aggr = last_aggr == me

        def bet(frac):
            return {"action": "raise", "amount": int(top_bet + max(bb, frac * (pot + to_call)))}

        # low SPR with a good hand: get it in
        if spr <= 1.5 and (strength >= 3 or (strength >= 2 and n_opp == 1) or (draw >= 2 and street == "flop")):
            return self._shove(state)

        if to_call == 0:
            # betting opportunity
            if strength >= 4:
                return bet(0.66)
            if strength == 3:
                if n_opp <= 2 or i_was_aggr:
                    return bet(0.5 if street != "river" else 0.6)
                return self._cf(state)
            if street == "flop" and i_was_aggr:
                if n_opp == 1:
                    if strength >= 1 or draw >= 1 or self.rng.random() < 0.65:
                        return bet(0.33)
                elif n_opp == 2 and (strength >= 2 or draw >= 2):
                    return bet(0.4)
                return self._cf(state)
            if street == "turn" and i_was_aggr and n_opp == 1:
                if strength >= 2 or draw >= 2:
                    return bet(0.5)
                if draw >= 1 and self.rng.random() < 0.5:
                    return bet(0.5)
            if street == "river" and strength >= 2 and n_opp == 1 and i_was_aggr and self.rng.random() < 0.4:
                return bet(0.4)
            return self._cf(state)

        # facing a bet
        price = to_call / float(pot + to_call)
        big = to_call >= 0.5 * my_stack
        if strength >= 5:
            if "raise" in legal and (street != "river" or self.rng.random() < 0.8):
                return {"action": "raise", "amount": int(top_bet * 3 + (pot - sum(bets)) * 0.3)}
            return {"action": "call"}
        if strength == 4:
            if "raise" in legal and n_opp == 1 and self.rng.random() < 0.35 and not big:
                return {"action": "raise", "amount": int(top_bet * 2.8)}
            return {"action": "call"}
        if strength == 3:
            if price <= 0.40 or (n_opp == 1 and price <= 0.45 and not big):
                return {"action": "call"}
            if spr <= 3 and n_opp == 1:
                return {"action": "call"}
            return self._cf(state)
        if strength == 2:
            if price <= (0.30 if n_opp == 1 else 0.22) and not big:
                return {"action": "call"}
            return self._cf(state)
        # draws
        if draw >= 2 and street != "river":
            need = 0.36 if street == "flop" else 0.19
            if price <= need or (price <= need + 0.08 and spr > 3):
                return {"action": "call"}
            if street == "flop" and "raise" in legal and spr <= 3:
                return self._shove(state)
            return self._cf(state)
        if draw == 1 and street != "river" and price <= 0.18:
            return {"action": "call"}
        if strength == 1 and price <= 0.18 and n_opp == 1 and street != "river":
            return {"action": "call"}
        return self._cf(state)


AGENT = Bot
