"""ai_gto_fan: an equilibrium-style tournament bot for No Limit (8-max SNG, 2,000 chips, rising blinds).

Design (what fits a 1 ms decision budget in pure Python):
- Preflop: a hand ranking precomputed offline (heads-up equity vs a random hand, 15k samples per class, small
  playability bonus), turned into "top x% of hands" percentiles. Ranges are width-by-position:
  * effective stack <= 15 bb: Nash-like push/fold (push width shrinks with players left to act, ICM-tightened);
  * deeper: raise-first-in by position with one size, 3-bet value + a mixed bluff region, BB defends by price;
  * facing an all-in: call width from (ICM-adjusted) pot odds.
  Every threshold has a mixing band, so boundary hands are played with probabilities, not a hard line.
- Postflop: equity vs one random hand by a small Monte Carlo done in observe() when a street is dealt (untimed in
  the kit), with a tiny in-act() fallback; multiway and "they bet into me" discounts; one bet size per street
  (balanced value/bluff mix), calls by pot odds with a minimum-defence mix so we do not over-fold to aggression.
- ICM: a bubble factor from the kit's exact ICM at each hand start scales all-in calling/pushing.
Every act() is wrapped: any internal error falls back to a legal check/fold (or call when it is the only option).
"""
import random

try:
    from sim.kit.agent import Agent
    from sim.kit.cards import CARD_INT, evaluate
    from sim.kit.icm import icm_equities
except ImportError:  # flat layout
    from agent import Agent
    from cards import CARD_INT, evaluate
    from icm import icm_equities

_R = "23456789TJQKA"
_RI = {r: i for i, r in enumerate(_R)}

# Heads-up all-in equity vs a uniformly random hand (offline Monte Carlo, 15,000 samples per class).
_EQ = {'AA': 0.852, 'AKs': 0.675, 'AKo': 0.651, 'AQs': 0.658, 'AQo': 0.651, 'AJs': 0.654, 'AJo': 0.635, 'ATs': 0.644, 'ATo': 0.632, 'A9s': 0.628, 'A9o': 0.606, 'A8s': 0.624, 'A8o': 0.6, 'A7s': 0.617, 'A7o': 0.596, 'A6s': 0.596, 'A6o': 0.576, 'A5s': 0.592, 'A5o': 0.577, 'A4s': 0.586, 'A4o': 0.569, 'A3s': 0.579, 'A3o': 0.555, 'A2s': 0.574, 'A2o': 0.546, 'KK': 0.828, 'KQs': 0.635, 'KQo': 0.616, 'KJs': 0.628, 'KJo': 0.606, 'KTs': 0.62, 'KTo': 0.602, 'K9s': 0.605, 'K9o': 0.581, 'K8s': 0.582, 'K8o': 0.561, 'K7s': 0.573, 'K7o': 0.548, 'K6s': 0.567, 'K6o': 0.538, 'K5s': 0.553, 'K5o': 0.539, 'K4s': 0.541, 'K4o': 0.522, 'K3s': 0.536, 'K3o': 0.519, 'K2s': 0.543, 'K2o': 0.505, 'QQ': 0.799, 'QJs': 0.605, 'QJo': 0.577, 'QTs': 0.599, 'QTo': 0.574, 'Q9s': 0.576, 'Q9o': 0.55, 'Q8s': 0.562, 'Q8o': 0.535, 'Q7s': 0.54, 'Q7o': 0.52, 'Q6s': 0.54, 'Q6o': 0.508, 'Q5s': 0.524, 'Q5o': 0.508, 'Q4s': 0.517, 'Q4o': 0.495, 'Q3s': 0.504, 'Q3o': 0.488, 'Q2s': 0.501, 'Q2o': 0.477, 'JJ': 0.775, 'JTs': 0.572, 'JTo': 0.552, 'J9s': 0.559, 'J9o': 0.53, 'J8s': 0.544, 'J8o': 0.511, 'J7s': 0.528, 'J7o': 0.493, 'J6s': 0.503, 'J6o': 0.478, 'J5s': 0.503, 'J5o': 0.47, 'J4s': 0.493, 'J4o': 0.458, 'J3s': 0.482, 'J3o': 0.454, 'J2s': 0.483, 'J2o': 0.438, 'TT': 0.749, 'T9s': 0.547, 'T9o': 0.511, 'T8s': 0.529, 'T8o': 0.498, 'T7s': 0.5, 'T7o': 0.477, 'T6s': 0.494, 'T6o': 0.468, 'T5s': 0.469, 'T5o': 0.447, 'T4s': 0.47, 'T4o': 0.438, 'T3s': 0.455, 'T3o': 0.425, 'T2s': 0.45, 'T2o': 0.412, '99': 0.718, '98s': 0.508, '98o': 0.478, '97s': 0.493, '97o': 0.464, '96s': 0.476, '96o': 0.447, '95s': 0.452, '95o': 0.429, '94s': 0.441, '94o': 0.405, '93s': 0.437, '93o': 0.406, '92s': 0.424, '92o': 0.4, '88': 0.69, '87s': 0.482, '87o': 0.454, '86s': 0.459, '86o': 0.434, '85s': 0.448, '85o': 0.419, '84s': 0.43, '84o': 0.396, '83s': 0.412, '83o': 0.373, '82s': 0.402, '82o': 0.373, '77': 0.665, '76s': 0.452, '76o': 0.426, '75s': 0.433, '75o': 0.407, '74s': 0.415, '74o': 0.384, '73s': 0.41, '73o': 0.371, '72s': 0.378, '72o': 0.342, '66': 0.641, '65s': 0.427, '65o': 0.401, '64s': 0.418, '64o': 0.376, '63s': 0.395, '63o': 0.356, '62s': 0.38, '62o': 0.343, '55': 0.611, '54s': 0.41, '54o': 0.386, '53s': 0.393, '53o': 0.362, '52s': 0.38, '52o': 0.34, '44': 0.572, '43s': 0.392, '43o': 0.351, '42s': 0.365, '42o': 0.329, '33': 0.535, '32s': 0.363, '32o': 0.325, '22': 0.506}


def _build_pct():
    """Class -> fraction of all 1326 combos at least as strong (AA ~ 0.005, 72o ~ 1.0)."""
    score = {}
    for k, e in _EQ.items():
        s = e
        if len(k) == 3:
            gap = _RI[k[0]] - _RI[k[1]]
            if k[2] == "s":
                s += 0.012
                if gap <= 2 and _RI[k[0]] <= 10:
                    s += 0.012   # suited connectors/gappers play better than raw equity says
            elif gap == 1 and _RI[k[0]] <= 10:
                s += 0.004
        score[k] = s
    order = sorted(score, key=lambda k: -score[k])
    pct, cum = {}, 0
    for k in order:
        cum += 6 if len(k) == 2 else (4 if k[2] == "s" else 12)
        pct[k] = cum / 1326.0
    return pct


_PCT = _build_pct()
_OPEN_W = {0: 0.0, 1: 0.46, 2: 0.42, 3: 0.29, 4: 0.22, 5: 0.18, 6: 0.155, 7: 0.135, 8: 0.12}
_STREET_FRAC = {"flop": 0.5, "turn": 0.66, "river": 0.75}


def _key(hole):
    a, b = hole[0][0], hole[1][0]
    if _RI[a] < _RI[b]:
        a, b = b, a
    if a == b:
        return a + b
    return a + b + ("s" if hole[0][1] == hole[1][1] else "o")


def _equity_vs_random(hole_i, board_i, k, rng):
    dead = set(hole_i) | set(board_i)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board_i)
    w = 0.0
    sample = rng.sample
    for _ in range(k):
        s = sample(deck, 2 + need)
        full = board_i + s[2:]
        me = evaluate(hole_i + full)
        op = evaluate(s[:2] + full)
        w += 1.0 if me > op else (0.5 if me == op else 0.0)
    return w / k


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.rng = random.Random(config.get("seed", seat) if isinstance(config, dict) else seat)
        self.payouts = list(config.get("payouts") or (50.0, 30.0, 20.0, 0, 0, 0, 0, 0))
        self.hole = None
        self.hole_i = None
        self.hs = {}          # board tuple -> equity vs one random hand
        self.bf = 1.0         # ICM bubble factor for this hand
        self.folded = False
        # anti-exploit guard (not a full opponent model): how often each seat jams preflop
        self.hands_seen = 0
        self.jams = [0] * int(config.get("n_seats", 8) if isinstance(config, dict) else 8)

    # ------------------------------------------------------------------ untimed bookkeeping
    def observe(self, event):
        t = event.get("type")
        if t == "hand_start":
            self.hs = {}
            self.hole = None
            self.hole_i = None
            self.folded = False
            self.hands_seen += 1
            self.bf = self._bubble_factor(list(event.get("stacks", ())))
        elif t == "deal":
            self.hole = tuple(event["hole"])
            self.hole_i = [CARD_INT[c] for c in self.hole]
        elif t == "action":
            sq = event.get("seat")
            if sq == self.seat and event.get("action") == "fold":
                self.folded = True
            elif (event.get("street") == "preflop" and event.get("action") == "raise" and event.get("all_in")
                  and isinstance(sq, int) and 0 <= sq < len(self.jams)):
                self.jams[sq] += 1
        elif t == "street":
            if self.hole_i and not self.folded:
                board = tuple(event["board"])
                if board not in self.hs:
                    k = 70 if len(board) == 5 else 55
                    self.hs[board] = _equity_vs_random(self.hole_i, [CARD_INT[c] for c in board], k, self.rng)

    def _bubble_factor(self, stacks):
        try:
            me = self.seat
            if not stacks or stacks[me] <= 0:
                return 1.0
            live = [i for i in range(len(stacks)) if stacks[i] > 0]
            paid = sum(1 for p in self.payouts if p > 0)
            if len(live) <= 1 or paid >= len(live) and len(live) <= 2:
                return 1.0
            others = [i for i in live if i != me]
            big = max(others, key=lambda i: stacks[i])
            x = min(stacks[me], stacks[big])
            now = icm_equities(stacks, self.payouts)[me]
            w = list(stacks); w[me] += x; w[big] -= x
            l = list(stacks); l[me] -= x; l[big] += x
            win = icm_equities(w, self.payouts)[me]
            lose = icm_equities(l, self.payouts)[me]
            gain = win - now
            if gain <= 1e-9:
                return 1.0
            bf = (now - lose) / gain
            return 1.0 if bf < 1.0 else (2.5 if bf > 2.5 else bf)
        except Exception:
            return 1.0

    # ------------------------------------------------------------------ timed decision
    def act(self, state):
        try:
            a = self._act(state)
            if self._legal(a, state):
                return a
        except Exception:
            pass
        return self._safe(state)

    @staticmethod
    def _safe(state):
        legal = state.get("legal_actions", ())
        if "check" in legal:
            return {"action": "check"}
        if "fold" in legal:
            return {"action": "fold"}
        return {"action": "call"}

    @staticmethod
    def _legal(a, state):
        if not isinstance(a, dict) or a.get("action") not in state["legal_actions"]:
            return False
        if a["action"] == "raise":
            amt = a.get("amount")
            return isinstance(amt, int) and state["min_raise_to"] <= amt <= state["max_raise_to"]
        return True

    # helpers
    def _mix(self, width, pct, band=0.025):
        """Probability-smoothed 'pct is inside the top width': 1 well inside, 0 well outside, mixed near the edge."""
        if width <= 0:
            return False
        d = (width - pct) / band
        if d >= 1:
            return True
        if d <= -1:
            return False
        return self.rng.random() < 0.5 + 0.5 * d

    @staticmethod
    def _raise(state, target):
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        t = int(target)
        # if the raise commits most of the stack, just go all-in
        if t >= lo and t > 0.6 * hi:
            t = hi
        return {"action": "raise", "amount": lo if t < lo else (hi if t > hi else t)}

    @staticmethod
    def _shove(state):
        legal = state["legal_actions"]
        if "raise" in legal:
            return {"action": "raise", "amount": state["max_raise_to"]}
        if "call" in legal:
            return {"action": "call"}
        return {"action": "check"}

    def _passive(self, state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}

    def _call_width(self, odds):
        """Top-x width to call an all-in at these (ICM-adjusted) odds vs an unknown shover."""
        if odds <= 0.05:
            return 1.0
        w = 0.12 * (0.5 / odds) ** 2
        return 1.0 if w > 1.0 else w

    def _icm_odds(self, risk, pot_win):
        r = risk * self.bf
        return r / (r + pot_win) if r + pot_win > 0 else 0.0

    # ------------------------------------------------------------------ main logic
    def _act(self, state):
        if self.hole is None or tuple(state["hole"]) != self.hole:
            self.hole = tuple(state["hole"])
            self.hole_i = [CARD_INT[c] for c in self.hole]
        if state["street"] == "preflop":
            return self._preflop(state)
        return self._postflop(state)

    def _preflop(self, state):
        legal = state["legal_actions"]
        me = self.seat
        bb = state["bb"] or 1
        stacks, bets, active = state["stacks"], state["bets"], state["active"]
        n = state["n_seats"]
        my_total = stacks[me] + bets[me]
        opp_tot = [stacks[i] + bets[i] for i in range(n) if i != me and active[i]]
        eff = min(my_total, max(opp_tot)) if opp_tot else my_total
        eff_bb = eff / bb
        pct = _PCT[_key(self.hole)]
        to_call = state["to_call"]
        pot = state["pot"]
        top = max(bets)

        hist = state["history_this_hand"]
        ss = state["street_starts"]
        pre = hist[ss[0]:(ss[1] if len(ss) > 1 else len(hist))]
        raises, limpers, seen_raise = 0, 0, False
        for (s, act, amt) in pre:
            if act == "raise":
                raises += 1
                seen_raise = True
            elif act == "call" and not seen_raise:
                limpers += 1

        # players still to act behind us this round (up to and including the big blind)
        behind = 0
        bbs = state["bb_seat"]
        if me != bbs:
            j = me
            for _ in range(n):
                j = (j + 1) % n
                if active[j] and not state["all_in"][j] and j != me:
                    behind += 1
                if j == bbs:
                    break
        icm_tight = 1.0 + 0.6 * (self.bf - 1.0)

        # ---------------- facing an all-in-sized bet (or a call that commits us): call by odds
        if to_call > 0 and raises >= 1 and (to_call >= 0.4 * stacks[me] or eff_bb <= 15):
            if raises >= 1 and "raise" in legal and eff_bb <= 15 and to_call < 0.4 * stacks[me]:
                # a small open vs our short stack: re-shove top range
                jam_w = min(0.5, 1.8 / max(eff_bb, 1.0)) ** icm_tight
                if self._mix(jam_w, pct):
                    return self._shove(state)
            odds = self._icm_odds(to_call, pot)
            w = self._call_width(odds)
            # who shoved: a very short shover has a wide range
            shover_bb = max((stacks[i] + bets[i]) for i in range(n) if i != me and active[i]) / bb
            if shover_bb < 8:
                w = min(1.0, w * 1.4)
            # a seat that jams far more often than any equilibrium range gets called wider
            jammer = None
            for (sj, aj, _) in pre:
                if aj == "raise":
                    jammer = sj
            if jammer is not None and self.hands_seen >= 4 and jammer != me:
                rate = self.jams[jammer] / self.hands_seen
                if rate > 0.25:
                    w = max(w, (0.30 if rate < 0.5 else 0.45) * (0.5 / max(odds, 0.25)))
            if raises >= 2:
                w *= 0.6
            n_in = sum(1 for i in range(n) if i != me and active[i] and bets[i] > 0 and bets[i] >= bb)
            if n_in >= 2 and raises >= 1:
                w *= 0.7
            if self._mix(w, pct, 0.02):
                if pct < 0.03 and "raise" in legal:
                    return self._shove(state)
                return {"action": "call"} if "call" in legal else self._shove(state)
            return self._passive(state)

        # ---------------- unopened or limped pot
        if raises == 0:
            if eff_bb <= 15 and ("raise" in legal or (to_call > 0 and "call" in legal)):
                k = max(behind, 1)
                w1 = min(1.0, 6.0 / max(eff_bb, 1.0))
                w = w1 ** (k ** 0.85)
                w = w ** icm_tight
                if limpers:
                    w = w ** (1.0 + 0.4 * limpers)
                if self._mix(w, pct, 0.03) or (to_call >= 0.5 * stacks[me] and to_call > 0
                                                and pct < 0.6):
                    return self._shove(state)
                if "check" in legal:
                    return {"action": "check"}
                # SB completing vs BB when very cheap: mixed limp with playable hands
                if behind == 1 and to_call <= bb // 2 + 1 and pct < 0.75 and self.rng.random() < 0.3:
                    return {"action": "call"}
                return self._passive(state)
            w = _OPEN_W.get(behind, 0.12)
            if limpers:
                w *= 0.6
            if behind == 0:   # big blind in a limped pot
                w = 0.12
            if self._mix(w, pct) and "raise" in legal:
                size = (2.2 if eff_bb < 30 else 2.5) * bb + limpers * bb
                if behind == 0:
                    size = top + (pot * 1.0)
                return self._raise(state, size)
            if "check" in legal:
                return {"action": "check"}
            # SB complete / overlimp with decent playable hands when deep and cheap
            if to_call <= bb and eff_bb > 25:
                if (behind == 1 and pct < 0.65 and self.rng.random() < 0.45) or (limpers and pct < 0.25):
                    return {"action": "call"}
            return self._passive(state)

        # ---------------- facing a raise (not committing)
        price = to_call / (pot + to_call)
        in_bb = me == bbs
        if raises == 1:
            val_w = 0.045
            if behind == 0 or me == state["sb_seat"]:
                val_w = 0.055
            if self._mix(val_w, pct, 0.01) and "raise" in legal:
                mult = 3.5 if (in_bb or me == state["sb_seat"]) else 3.0
                return self._raise(state, top * mult + limpers * bb)
            # mixed bluff 3-bets: suited aces / suited broadways just outside the calling top
            k = _key(self.hole)
            if (0.10 < pct < 0.32 and len(k) == 3 and k[2] == "s" and k[0] in "AK" and "raise" in legal
                    and self.rng.random() < 0.3 and eff_bb > 25):
                mult = 3.5 if (in_bb or me == state["sb_seat"]) else 3.0
                return self._raise(state, top * mult)
            # flat-call width by price; BB defends widest (do not over-fold to min-raises)
            if in_bb:
                call_w = 0.42 if price <= 0.25 else (0.30 if price <= 0.33 else 0.18)
            elif behind == 0 or me == state["sb_seat"]:
                call_w = 0.12
            else:
                call_w = 0.10 if price <= 0.3 else 0.07
            if eff_bb < 25:
                call_w *= 0.7
            call_w = call_w ** icm_tight
            if self._mix(call_w, pct) and "call" in legal:
                return {"action": "call"}
            return self._passive(state)

        # facing a 3-bet or more
        if self._mix(0.025, pct, 0.008):
            if "raise" in legal:
                if eff_bb <= 60:
                    return self._shove(state)
                return self._raise(state, top * 2.3)
            return {"action": "call"} if "call" in legal else self._passive(state)
        k = _key(self.hole)
        if (len(k) == 3 and k[0] == "A" and k[2] == "s" and _RI[k[1]] <= 3 and eff_bb > 60
                and "raise" in legal and self.rng.random() < 0.12):
            return self._raise(state, top * 2.3)   # rare blocker 4-bet bluff
        call_w = 0.07 if price <= 0.33 else 0.045
        if self._mix(call_w, pct, 0.01) and "call" in legal and to_call < 0.3 * stacks[me]:
            return {"action": "call"}
        return self._passive(state)

    def _postflop(self, state):
        legal = state["legal_actions"]
        me = self.seat
        n = state["n_seats"]
        board = tuple(state["board"])
        eq1 = self.hs.get(board)
        if eq1 is None:
            eq1 = _equity_vs_random(self.hole_i, [CARD_INT[c] for c in board], 16, self.rng)
            self.hs[board] = eq1
        active = state["active"]
        n_opp = sum(1 for i in range(n) if active[i] and i != me)
        n_opp_live = sum(1 for i in range(n) if active[i] and i != me and not state["all_in"][i])
        eq = eq1 ** (max(n_opp, 1) ** 0.85)
        pot = state["pot"]
        to_call = state["to_call"]
        stacks, bets = state["stacks"], state["bets"]
        top = max(bets)
        street = state["street"]
        frac = _STREET_FRAC.get(street, 0.66)

        hist = state["history_this_hand"]
        ss = state["street_starts"]
        cur_start = ss[-1] if ss else 0
        raises_now = sum(1 for (s, a, _) in hist[cur_start:] if a == "raise" and s != me)
        # were we the last aggressor of the previous street?
        prev = hist[ss[-2]:ss[-1]] if len(ss) >= 2 else []
        last_aggr = None
        for (s, a, _) in prev:
            if a == "raise":
                last_aggr = s
        aggressor = last_aggr == me
        # earlier opponent aggression on prior streets tightens our read of their range
        opp_prior_raises = sum(1 for (s, a, _) in hist[ss[1] if len(ss) > 1 else len(hist):cur_start]
                               if a == "raise" and s != me)

        if "check" in legal:
            if "raise" not in legal or n_opp_live == 0:
                return {"action": "check"}
            bet_to = top + frac * pot
            r = self.rng.random()
            if eq > 0.72:
                if r < 0.85:
                    return self._raise(state, bet_to)
                return {"action": "check"}   # slowplay mix keeps checking range protected
            if eq > 0.5:
                p = 0.4 if (aggressor and n_opp == 1) else 0.15
                if street == "river":
                    p *= 0.5
                if r < p:
                    return self._raise(state, bet_to)
                return {"action": "check"}
            # weak region: bluffs, sized the same as value bets
            if n_opp == 1:
                p = {"flop": 0.45 if aggressor else 0.15, "turn": 0.25 if aggressor else 0.1,
                     "river": 0.18 if eq < 0.3 else 0.05}[street]
            else:
                p = 0.08 if (aggressor and street == "flop") else 0.02
            if r < p:
                return self._raise(state, bet_to)
            return {"action": "check"}

        # facing a bet
        tax = 0.04 + 0.06 * raises_now + 0.02 * opp_prior_raises
        eqa = eq - tax
        if eqa < 0:
            eqa = 0.0
        price = to_call / (pot + to_call)
        my_stack = stacks[me]
        big = to_call >= 0.35 * my_stack
        need = self._icm_odds(to_call, pot) if big else price
        r = self.rng.random()
        if eqa > 0.78 and "raise" in legal and n_opp_live >= 1:
            if r < 0.75:
                return self._raise(state, top + 0.85 * (pot + to_call))
            return {"action": "call"}
        if eqa >= need + 0.02:
            # occasional semi-bluff raise with mid equity on the flop heads-up
            if street == "flop" and n_opp == 1 and 0.45 < eqa < 0.6 and "raise" in legal and r < 0.1 \
                    and not big:
                return self._raise(state, top + 0.85 * (pot + to_call))
            return {"action": "call"}
        # draws / air raise as bluffs at low frequency (balance for value raises)
        if street == "flop" and n_opp == 1 and 0.3 < eq1 < 0.5 and "raise" in legal and r < 0.06 \
                and not big and raises_now <= 1:
            return self._raise(state, top + 0.85 * (pot + to_call))
        # minimum-defence mix vs small bets: do not auto-fold to aggression
        if not big and to_call <= 0.6 * pot and eqa > need - 0.1 and r < 0.35:
            return {"action": "call"}
        return {"action": "fold"} if "fold" in legal else {"action": "call"}
