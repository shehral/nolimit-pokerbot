"""Latency-first tight-aggressive bot. Precomputed preflop table (Chen-style score per 169 hands),
cheap postflop rules from the evaluator category plus draw flags. No simulation; everything wrapped
so any error degrades to a legal check/fold."""
from sim.kit.agent import Agent
from sim.kit.cards import CARD_INT, evaluate

_R = {r: i for i, r in enumerate("23456789TJQKA")}


def _chen(hi, lo, suited):
    pts = {12: 10, 11: 8, 10: 7, 9: 6}.get(hi, (hi + 2) / 2.0)
    if hi == lo:
        return max(5.0, pts * 2)
    if suited:
        pts += 2
    gap = hi - lo - 1
    pts -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:
        pts += 1
    return pts


# TABLE[(hi, lo, suited)] -> chen score, precomputed at import
TABLE = {}
for _h in range(13):
    for _l in range(_h + 1):
        TABLE[(_h, _l, False)] = _chen(_h, _l, False)
        if _h != _l:
            TABLE[(_h, _l, True)] = _chen(_h, _l, True)


def _score(hole):
    a, b = _R[hole[0][0]], _R[hole[1][0]]
    if a < b:
        a, b = b, a
    return TABLE[(a, b, hole[0][1] == hole[1][1])]


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.seat = seat

    def act(self, state):
        try:
            r = self._act(state)
            if r is not None:
                return r
        except Exception:
            pass
        return self._safe(state)

    @staticmethod
    def _safe(state):
        leg = state["legal_actions"]
        return {"action": "check"} if "check" in leg else {"action": "fold"}

    @staticmethod
    def _raise(state, target):
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        t = int(target)
        t = lo if t < lo else hi if t > hi else t
        return {"action": "raise", "amount": t}

    def _act(self, s):
        leg = s["legal_actions"]
        me = self.seat
        to_call = s["to_call"]
        bb = s["bb"]
        stack = s["stacks"][me]
        pot = s["pot"]
        can_raise = "raise" in leg
        if s["street"] == "preflop":
            sc = _score(s["hole"])
            n = s["n_seats"]
            # late position bonus: distance from button (0 = button)
            pos = (s["button"] - me) % n
            if pos <= 1:
                sc += 1.0
            elif pos >= n - 2 or pos >= 5:
                sc -= 0.5
            eff = (stack + s["bets"][me]) / bb
            top = max(s["bets"])
            facing_raise = top > bb
            if eff <= 12:
                if sc >= (8 if facing_raise else 6.5):
                    return self._raise(s, s["max_raise_to"]) if can_raise else {"action": "call"}
            if sc >= 10:  # premium: 99+, AQ+
                if can_raise:
                    return self._raise(s, max(top * 3, 3 * bb) if not facing_raise else top * 3)
                return {"action": "call"}
            if sc >= 8.5 and not facing_raise and can_raise:
                return self._raise(s, 3 * bb)
            if sc >= 8.5 and "call" in leg and to_call <= stack // 8:
                return {"action": "call"}
            if sc >= 7 and not facing_raise:
                if "check" in leg:
                    return {"action": "check"}
                if to_call <= bb and can_raise and pos <= 2:
                    return self._raise(s, 2.5 * bb)
                return {"action": "call"} if to_call <= bb else {"action": "fold"}
            if sc >= 6 and "call" in leg and to_call <= 2 * bb and pos <= 2:
                return {"action": "call"}
            return {"action": "check"} if "check" in leg else {"action": "fold"}
        hole, board = s["hole"], s["board"]
        cards = [CARD_INT[c] for c in hole] + [CARD_INT[c] for c in board]
        cat = evaluate(cards) >> 20
        # draw flags
        suits = [c[1] for c in hole + board]
        flush_draw = len(board) < 5 and max(suits.count(x) for x in "cdhs") == 4
        bet = int(pot * 0.66)
        top = max(s["bets"])
        if cat >= 4:  # trips+ : value bet / raise big
            if can_raise:
                return self._raise(s, top + max(bet, pot))
            return {"action": "call"} if "call" in leg else {"action": "check"}
        if cat >= 2:  # two pair or better-ish
            if can_raise:
                return self._raise(s, top + bet)
            return {"action": "call"} if "call" in leg else {"action": "check"}
        if cat == 1:
            # pair: top pair or better plays; weak pair just calls small
            bmax = max(_R[c[0]] for c in board)
            pr = [_R[c[0]] for c in hole]
            toppair = max(pr) >= bmax and (pr[0] == pr[1] or any(
                _R[c[0]] == bmax for c in hole) or pr[0] == pr[1])
            if "check" in leg:
                if toppair and can_raise:
                    return self._raise(s, top + int(pot * 0.5))
                return {"action": "check"}
            if toppair and to_call * 2 <= pot + to_call:
                return {"action": "call"}
            if to_call * 4 <= pot:
                return {"action": "call"}
            return {"action": "fold"}
        if flush_draw:
            if "check" in leg:
                return self._raise(s, top + int(pot * 0.5)) if can_raise and len(board) == 3 else {"action": "check"}
            if to_call * 3 <= pot and len(board) < 5:
                return {"action": "call"}
        return {"action": "check"} if "check" in leg else {"action": "fold"}
