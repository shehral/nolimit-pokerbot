"""ai_icm_plan: ICM-aware push/fold + opponent typing + value-heavy postflop, with a 1 ms guard.

Plan (as given to us):
  1. ICM-aware push/fold with bubble factors when short (effective stack <= PUSH_BB big blinds).
  2. Opponent typing from their actions (VPIP / PFR / postflop aggression -> tight / loose / aggressive),
     used to widen or narrow the ranges we assume they hold, call with, or shove with.
  3. Value-heavy postflop play: bet made hands for value, few bluffs, fold weak hands to bets.
  4. A guard for the 1 ms limit: every act() is wrapped in try/except with a cheap fallback, and the expensive
     ICM loop checks a perf_counter deadline and bails out to a chart decision.

Standard library + sim.kit only. No Monte Carlo inside act(): equities come from a closed-form heuristic.
"""
import gc
import random
import time

try:
    from sim.kit.agent import Agent
    from sim.kit.cards import CARD_INT, evaluate
    from sim.kit.icm import make_icm
except ImportError:  # flat layout fallback
    from agent import Agent
    from cards import CARD_INT, evaluate
    from icm import make_icm

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}
PUSH_BB = 12.0          # at or below this many BB effective we play push/fold
SHORT_CALL_BB = 16.0    # facing a raise at or below this, it's reshove-or-fold
GUARD_S = 0.00045       # bail out of the ICM loop after 450 us
_perf = time.perf_counter


# ------------------------------------------------------------------ preflop hand table (Chen formula -> percentile)
def _chen(hi, lo, suited):
    hv = (10.0, 8.0, 7.0, 6.0)[12 - hi] if hi >= 9 else (hi + 2) / 2.0
    if hi == lo:
        return max(5.0, 2 * hv)
    s = hv + (2.0 if suited else 0.0)
    gap = hi - lo - 1
    s -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and hi < 10:
        s += 1
    return s


def _build_table():
    hands = []
    for hi in range(13):
        for lo in range(hi + 1):
            if hi == lo:
                hands.append((hi, lo, False, 6))
            else:
                hands.append((hi, lo, True, 4))
                hands.append((hi, lo, False, 12))
    hands.sort(key=lambda h: -(_chen(h[0], h[1], h[2]) + h[0] / 100.0 + h[1] / 1000.0))
    table = {}
    cum = 0
    for hi, lo, su, combos in hands:
        frac = (cum + combos / 2.0) / 1326.0     # 0 = best hand, 1 = worst
        table[(hi, lo, su)] = frac
        cum += combos
    return table


_TABLE = _build_table()


def hand_info(hole):
    a, b = _RANK[hole[0][0]], _RANK[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    su = hi != lo and hole[0][1] == hole[1][1]
    return _TABLE[(hi, lo, su)], hi == lo, hi, lo


def equity_vs_range(frac, is_pair, hi, p):
    """Heuristic all-in equity of a hand (top-fraction `frac`) against a range of the top `p` of hands."""
    s = 1.0 - frac
    eq_rand = 0.32 + 0.53 * (s ** 1.5)
    if is_pair:
        pen = 0.04 if hi == 12 else 0.15
    else:
        pen = 0.18 + 0.1 * (s - 0.5)
    p = 0.02 if p < 0.02 else 1.0 if p > 1.0 else p
    eq = eq_rand - pen * (1.0 - p) / 0.9
    return 0.2 if eq < 0.2 else 0.87 if eq > 0.87 else eq


# ------------------------------------------------------------------ the bot
class Bot(Agent):
    DEBUG = False   # set True in local tests to surface swallowed errors

    def __init__(self, seat, config):
        super().__init__(seat, config)
        n = int(config.get("n_seats", 8))
        self.n = n
        self.rng = random.Random(config.get("seed", 0))
        self.icm = make_icm(config.get("payouts") or (50, 30, 20, 0, 0, 0, 0, 0), max_players=max(10, n))
        # opponent stats with priors
        self.hands = [0] * n
        self.vpip = [0] * n
        self.pfr = [0] * n
        self.shoves = [0] * n
        self.agg = [0] * n      # postflop bets/raises
        self.pas = [0] * n      # postflop calls
        self._vp = [False] * n
        self._pf = [False] * n
        self._bf_hand = -1
        self._bf = 1.0
        self._deadline = 0.0
        # warm-up (untimed): touch the ICM closure and the evaluator, then collect garbage so a collector pause
        # is less likely to land inside the first timed act() calls (we saw 1-2 ms CPU spikes in hands 1-2).
        try:
            st = list(config.get("start_stacks") or [2000] * n)
            self.icm(st, seat)
            self._classify(("Ah", "Kd"), ("2c", "7h", "Ts"))
            equity_vs_range(0.1, False, 12, 0.2)
            gc.collect()
        except Exception:
            pass

    # ---------------------------------------------------------------- opponent model
    def observe(self, event):
        t = event.get("type")
        if t == "action":
            s = event["seat"]
            a = event["action"]
            if event.get("street") == "preflop":
                if a in ("call", "raise") and not self._vp[s]:
                    self._vp[s] = True
                    self.vpip[s] += 1
                if a == "raise" and not self._pf[s]:
                    self._pf[s] = True
                    self.pfr[s] += 1
                    if event.get("all_in"):
                        self.shoves[s] += 1
            else:
                if a == "raise":
                    self.agg[s] += 1
                elif a == "call":
                    self.pas[s] += 1
        elif t == "hand_start":
            alive = event.get("alive") or ()
            for s in range(self.n):
                self._vp[s] = False
                self._pf[s] = False
                if s < len(alive) and alive[s]:
                    self.hands[s] += 1

    def vpip_est(self, s):
        return (self.vpip[s] + 0.28 * 6) / (self.hands[s] + 6)

    def pfr_est(self, s):
        return (self.pfr[s] + 0.14 * 6) / (self.hands[s] + 6)

    def af_est(self, s):
        return (self.agg[s] + 1.0) / (self.pas[s] + 1.0)

    def opp_type(self, s):
        v, pf, af = self.vpip_est(s), self.pfr_est(s), self.af_est(s)
        loose = v > 0.40
        tight = v < 0.20
        aggressive = af > 1.8 or (pf > 0.6 * v and pf > 0.2)
        return loose, tight, aggressive

    def call_width(self, s, risk_bb, is_bb):
        """Fraction of hands seat s calls an all-in of risk_bb big blinds with."""
        w = 2.2 / (risk_bb + 2.0)
        w *= min(2.0, max(0.5, self.vpip_est(s) / 0.28))
        if is_bb:
            w *= 1.3
        return 0.03 if w < 0.03 else 0.7 if w > 0.7 else w

    def raise_width(self, s, v_bb, n_raises, all_in):
        pf = self.pfr_est(s)
        if all_in and v_bb <= 20:
            pf = 0.5 * pf + 0.5 * min(0.6, 2.0 / (v_bb + 2.0))
        if self.shoves[s] >= 3 and self.hands[s] > 0:
            pf = max(pf, self.shoves[s] / self.hands[s])
        if n_raises >= 2:
            pf *= 0.45
        return 0.02 if pf < 0.02 else 0.9 if pf > 0.9 else pf

    # ---------------------------------------------------------------- ICM helpers
    def _ev(self, stacks):
        return self.icm(stacks, self.seat)

    def allin_required_eq(self, state, v):
        """Equity needed to call/shove all-in against seat v (two-way, exact ICM on the outcomes)."""
        me = self.seat
        stacks = list(state["stacks"])
        contrib = state["contributed"]
        pot = state["pot"]
        t_me = stacks[me] + contrib[me]
        t_v = stacks[v] + contrib[v]
        a = t_me if t_me < t_v else t_v
        dead = pot - contrib[me] - contrib[v]
        fold = list(stacks)
        fold[v] = stacks[v] + pot
        win = list(stacks)
        win[me] = t_me - a + 2 * a + dead
        win[v] = t_v - a
        lose = list(stacks)
        lose[me] = t_me - a
        lose[v] = t_v + a + dead
        e_f, e_w, e_l = self._ev(fold), self._ev(win), self._ev(lose)
        if e_w - e_l <= 1e-12:
            return 1.0
        return (e_f - e_l) / (e_w - e_l)

    def bubble_factor(self, state):
        """Once per hand: BF vs the biggest other stack for risking our whole stack (chips at hand start)."""
        h = state["hand_no"]
        if h == self._bf_hand:
            return self._bf
        self._bf_hand = h
        st = list(state["start_stacks"])
        me = self.seat
        others = [j for j in range(self.n) if j != me and st[j] > 0]
        bf = 1.0
        if others and st[me] > 0:
            v = max(others, key=lambda j: st[j])
            a = min(st[me], st[v])
            e0 = self._ev(st)
            w = list(st); w[me] += a; w[v] -= a
            l = list(st); l[me] -= a; l[v] += a
            gain = self._ev(w) - e0
            loss = e0 - self._ev(l)
            if gain > 1e-12:
                bf = loss / gain
        self._bf = 1.0 if bf < 1.0 else 2.5 if bf > 2.5 else bf
        return self._bf

    # ---------------------------------------------------------------- act with the guard
    def act(self, state):
        try:
            self._deadline = _perf() + GUARD_S
            act = self._act(state)
            if self._legal(act, state):
                return act
        except Exception:
            if self.DEBUG:
                raise
        else:
            if self.DEBUG:
                raise RuntimeError(f"illegal {act!r}")
        return self._fallback(state)

    @staticmethod
    def _legal(a, state):
        if not isinstance(a, dict):
            return False
        k = a.get("action")
        if k not in state["legal_actions"]:
            return False
        if k == "raise":
            amt = a.get("amount")
            return isinstance(amt, int) and state["min_raise_to"] <= amt <= state["max_raise_to"]
        return True

    @staticmethod
    def _fallback(state):
        legal = state["legal_actions"]
        if "check" in legal:
            return {"action": "check"}
        return {"action": "fold"} if "fold" in legal else {"action": legal[0]}

    # ---------------------------------------------------------------- action builders
    @staticmethod
    def _raise(state, target):
        if "raise" not in state["legal_actions"]:
            return None
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        t = int(target)
        return {"action": "raise", "amount": lo if t < lo else hi if t > hi else t}

    def _shove(self, state):
        legal = state["legal_actions"]
        if "raise" in legal:
            return {"action": "raise", "amount": state["max_raise_to"]}
        if "call" in legal:
            return {"action": "call"}
        return {"action": "check"}

    def _bet_or(self, state, target):
        """Raise to target, going all-in if the bet would leave a small stack behind."""
        me = self.seat
        total = state["stacks"][me] + state["bets"][me]
        if target >= 0.6 * total:
            return self._shove(state)
        r = self._raise(state, target)
        if r is not None:
            return r
        return {"action": "call"} if "call" in state["legal_actions"] else {"action": "check"}

    @staticmethod
    def _passive(state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}

    # ---------------------------------------------------------------- decision
    def _act(self, state):
        if state["street"] == "preflop":
            return self._preflop(state)
        return self._postflop(state)

    def _behind(self, state):
        """Seats still to act after us preflop (up to and including the big blind)."""
        me, bbs, n = self.seat, state["bb_seat"], self.n
        if me == bbs:
            return []
        out = []
        act, alli, stacks = state["active"], state["all_in"], state["stacks"]
        j = me
        for _ in range(n - 1):
            j = (j + 1) % n
            if act[j] and not alli[j] and stacks[j] > 0:
                out.append(j)
            if j == bbs:
                break
        return out

    def _preflop(self, state):
        me = self.seat
        legal = state["legal_actions"]
        bb = state["bb"] or 1
        stacks = state["stacks"]
        bets = state["bets"]
        my_total = stacks[me] + bets[me]
        frac, is_pair, hi, lo = hand_info(state["hole"])
        # effective stack: vs the biggest other live stack
        others = [j for j in range(self.n) if j != me and state["active"][j]]
        if not others:
            return self._passive(state)
        big_other = max(stacks[j] + bets[j] for j in others)
        eff_bb = min(my_total, big_other) / bb
        to_call = state["to_call"]
        top_bet = max(bets)
        hist = state["history_this_hand"]
        raises = [(s, amt) for (s, a, amt) in hist if a == "raise"]
        n_raises = len(raises)
        bf = self.bubble_factor(state) if eff_bb <= 30 else 1.0

        if n_raises == 0:
            # unopened (maybe limped) pot
            if eff_bb <= PUSH_BB:
                return self._push_decision(state, frac, is_pair, hi, bf)
            behind = self._behind(state)
            nb = len(behind)
            open_w = {0: 0.0, 1: 0.5, 2: 0.38, 3: 0.27, 4: 0.21, 5: 0.17, 6: 0.14}.get(nb, 0.12)
            # opponent typing: tight players behind -> steal wider; loose ones -> tighter, value-heavy
            if behind:
                lt = sum(self.vpip_est(j) for j in behind) / nb
                open_w *= min(1.4, max(0.7, 0.28 / lt))
            open_w /= bf ** 0.5
            limpers = sum(1 for (s, a, amt) in hist if a == "call")
            if "check" in legal:   # we are the BB with a free option
                if frac <= 0.08 + 0.02 * (limpers == 0) and "raise" in legal:
                    return self._bet_or(state, bb * (3 + limpers))
                return {"action": "check"}
            if frac <= open_w * (0.8 if limpers else 1.0):
                return self._bet_or(state, bb * (2.5 + limpers))
            if limpers and frac <= 0.25 and to_call <= bb and eff_bb > 25:
                return {"action": "call"}
            return self._passive(state)

        # facing a raise
        rs, ramt = raises[-1]
        v = rs
        v_total = stacks[v] + bets[v]
        v_allin = bool(state["all_in"][v])
        p_v = self.raise_width(v, min(v_total, my_total) / bb, n_raises, v_allin)
        committing = to_call >= 0.4 * stacks[me] or my_total / bb <= SHORT_CALL_BB or v_allin and to_call >= 0.25 * stacks[me]
        if committing:
            eq = equity_vs_range(frac, is_pair, hi, p_v)
            callers = sum(1 for j in others if j != v and bets[j] >= ramt and bets[j] > 0)
            eq *= 0.72 ** callers
            if _perf() < self._deadline:
                need = self.allin_required_eq(state, v)
            else:
                need = 0.5 * bf / (bf + 0.5)
            if eq >= need:
                return self._shove(state)
            return self._passive(state)
        # deep: 3-bet for value, call with a tight range, fold the rest
        val_w = max(0.025, 0.3 * p_v) / bf
        call_w = max(0.06, 0.7 * p_v) / bf
        if frac <= val_w and "raise" in legal:
            return self._bet_or(state, 3 * top_bet + (state["pot"] - sum(bets)))
        price_ok = to_call <= max(3.5 * bb, 0.08 * stacks[me])
        if frac <= call_w and price_ok and "call" in legal:
            return {"action": "call"}
        if is_pair and price_ok and eff_bb >= 30 and to_call <= 0.06 * stacks[me] and "call" in legal:
            return {"action": "call"}   # set-mine
        return self._passive(state)

    def _push_decision(self, state, frac, is_pair, hi, bf):
        """Open-shove decision: ICM EV of shove vs fold over the players left to act."""
        me = self.seat
        stacks = state["stacks"]
        bets = state["bets"]
        contrib = state["contributed"]
        bb = state["bb"] or 1
        pot = state["pot"]
        legal = state["legal_actions"]
        behind = self._behind(state)
        if not behind:
            return {"action": "check"} if "check" in legal else self._shove(state)
        t_me = stacks[me] + contrib[me]
        # baseline: fold (or check)
        base = list(stacks)
        if "check" not in legal:
            base[state["bb_seat"]] += pot   # approximate: blinds go to the BB
        e_fold = self._ev(base) if "check" not in legal else self._ev(stacks)
        p_none = 1.0
        ev = 0.0
        for j in behind:
            if _perf() > self._deadline:
                return self._push_chart(state, frac, bf, len(behind))
            t_j = stacks[j] + contrib[j]
            a = t_me if t_me < t_j else t_j
            pc = self.call_width(j, a / bb, j == state["bb_seat"])
            eq = equity_vs_range(frac, is_pair, hi, pc)
            dead = pot - contrib[me] - contrib[j]
            w = list(stacks); w[me] = t_me + a + dead; w[j] = t_j - a
            l = list(stacks); l[me] = t_me - a; l[j] = t_j + a + dead
            prob = p_none * pc
            ev += prob * (eq * self._ev(w) + (1.0 - eq) * self._ev(l))
            p_none *= (1.0 - pc)
        steal = list(stacks); steal[me] = stacks[me] + pot
        ev += p_none * self._ev(steal)
        if ev > e_fold:
            return self._shove(state)
        return self._passive(state)

    @staticmethod
    def _push_chart(state, frac, bf, nb):
        bb = state["bb"] or 1
        me = state["seat"]
        m = (state["stacks"][me] + state["bets"][me]) / bb
        w = min(0.9, 1.6 / (m + 1.0) * (1.0 + 0.25 * (6 - min(6, nb)))) / bf
        if frac <= w:
            return Bot._shove(None, state)
        return Bot._passive(state)

    # ---------------------------------------------------------------- postflop
    def _classify(self, hole, board):
        hc = [CARD_INT[c] for c in hole]
        bc = [CARD_INT[c] for c in board]
        cat = evaluate(hc + bc) >> 20
        br = [c >> 2 for c in bc]
        hr = [c >> 2 for c in hc]
        cnt = {}
        for r in br:
            cnt[r] = cnt.get(r, 0) + 1
        mx = max(cnt.values())
        board_cat = 0
        if mx >= 3:
            board_cat = 3 if mx == 3 else 7
            if mx == 3 and any(v >= 2 for r, v in cnt.items() if v < 3):
                board_cat = 6
        elif mx == 2:
            board_cat = 2 if sum(1 for v in cnt.values() if v == 2) >= 2 else 1
        if len(bc) == 5:
            board_cat = max(board_cat, evaluate(bc) >> 20)
        top = max(br)
        # draws
        suits = {}
        for c in hc + bc:
            suits[c & 3] = suits.get(c & 3, 0) + 1
        fd = any(v == 4 and any((c & 3) == s for c in hc) for s, v in suits.items()) and len(bc) < 5
        ranks = 0
        for r in hr + br:
            ranks |= 1 << r
            if r == 12:
                ranks |= 1 << 13   # ace low
        outs = 0
        if len(bc) < 5:
            for r in range(13):
                if ranks >> r & 1:
                    continue
                m = ranks | (1 << r) | ((1 << 13) if r == 12 else 0)
                for lo in range(0, 10):
                    win = ((m >> lo) & 31) == 31
                    if win:
                        outs += 1
                        break
        sd = outs  # 1 = gutshot, 2 = open-ended
        if cat > board_cat and cat >= 3:
            # trips: weaker if it's a board pair + one hole card with weak kicker; still strong
            return "monster", fd, sd
        if cat > board_cat and cat == 2:
            both = hr[0] != hr[1] and hr[0] in cnt and hr[1] in cnt
            if both or (hr[0] == hr[1] and hr[0] > top):
                return "strong", fd, sd
            return "medium", fd, sd
        if cat >= 1 and cat > board_cat - (1 if board_cat == 1 else 0):
            # one pair made with our cards (or a pocket pair over a paired board)
            if hr[0] == hr[1]:
                if hr[0] > top:
                    return "strong", fd, sd
                second = sorted(set(br))[-2] if len(set(br)) >= 2 else -1
                return ("medium" if hr[0] > second else "weak"), fd, sd
            for i in (0, 1):
                if hr[i] in cnt and cnt[hr[i]] == 1:
                    kick = hr[1 - i]
                    if hr[i] == top:
                        return ("strong" if kick >= 9 or hr[i] >= 11 and kick >= 8 else "medium"), fd, sd
                    second = sorted(set(br))[-2] if len(set(br)) >= 2 else -1
                    return ("medium" if hr[i] == second else "weak"), fd, sd
            return "weak", fd, sd
        return "air", fd, sd

    def _postflop(self, state):
        me = self.seat
        legal = state["legal_actions"]
        pot = state["pot"]
        to_call = state["to_call"]
        stacks = state["stacks"]
        bets = state["bets"]
        top_bet = max(bets)
        street = state["street"]
        cls, fd, sd = self._classify(state["hole"], state["board"])
        opps = [j for j in range(self.n) if j != me and state["active"][j]]
        k = len(opps)
        hist = state["history_this_hand"]
        ss = state["street_starts"]
        pre_end = ss[1] if len(ss) > 1 else len(hist)
        pfr_seat = -1
        for (s, a, amt) in hist[:pre_end]:
            if a == "raise":
                pfr_seat = s
        i_raised = pfr_seat == me
        stations = all(self.opp_type(j)[0] and not self.opp_type(j)[2] for j in opps) if opps else False
        bf = self.bubble_factor(state)

        draw_eq = 0.0
        if fd and sd >= 2:
            draw_eq = 0.52 if street == "flop" else 0.30
        elif fd:
            draw_eq = 0.36 if street == "flop" else 0.19
        elif sd >= 2:
            draw_eq = 0.32 if street == "flop" else 0.17
        elif sd == 1:
            draw_eq = 0.17 if street == "flop" else 0.09
        base_eq = {"monster": 0.88, "strong": 0.70, "medium": 0.50, "weak": 0.33, "air": 0.12}[cls]
        eq = max(base_eq, draw_eq) if cls in ("weak", "air") else base_eq + 0.3 * draw_eq * (1 - base_eq)
        if cls != "monster":
            eq *= 0.85 ** max(0, k - 1)

        if "check" in legal:
            # not facing a bet: value-heavy betting
            if cls in ("monster", "strong"):
                size = 0.8 if stations else 0.66
                if cls == "strong" and street == "river" and k > 1:
                    size = 0.5
                return self._bet_or(state, top_bet + int(size * pot))
            if cls == "medium":
                if k == 1 and (street == "flop" and i_raised or street == "river" and stations):
                    return self._bet_or(state, int(0.45 * pot))
                return {"action": "check"}
            if street == "flop" and k == 1 and i_raised and not stations:
                if draw_eq >= 0.32 or (cls == "air" and self.rng.random() < 0.45):
                    return self._bet_or(state, int(0.4 * pot) if draw_eq < 0.32 else int(0.55 * pot))
            return {"action": "check"}

        # facing a bet
        bettor = -1
        for j in opps:
            if bets[j] == top_bet:
                bettor = j
        if bettor >= 0:
            loose, tight, aggr = self.opp_type(bettor)
            if cls in ("medium", "weak"):
                if aggr:
                    eq += 0.07
                elif not loose and self.af_est(bettor) < 0.8:
                    eq -= 0.08
        need = to_call / float(pot + to_call)
        commit = to_call / float(stacks[me] + bets[me] + 1e-9)
        need *= 1.0 + (bf - 1.0) * min(1.0, commit * 1.5)
        allin_spot = to_call >= 0.5 * stacks[me] or (bettor >= 0 and state["all_in"][bettor])
        if allin_spot and bettor >= 0 and _perf() < self._deadline and k == 1:
            need = max(need, self.allin_required_eq(state, bettor))
        if cls == "monster":
            r = self._bet_or(state, top_bet + int(0.9 * (pot + to_call)))
            return r
        if cls == "strong" and to_call <= 0.35 * pot and street != "river" and "raise" in legal and k == 1:
            return self._bet_or(state, top_bet * 3)
        if eq >= need and "call" in legal:
            return {"action": "call"}
        return self._passive(state)
