"""ai_aggro: hyper-aggressive bot. Raises wide, 3-bets light, c-bets every flop, keeps barreling."""
import random
from sim.kit.agent import Agent
from sim.kit.cards import CARD_INT, evaluate

_R = {r: i for i, r in enumerate("23456789TJQKA")}


def _score(hole):
    a, b = _R[hole[0][0]], _R[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == lo:
        return 0.55 + 0.45 * hi / 12.0
    s = 0.45 * hi / 12.0 + 0.35 * lo / 12.0
    if hole[0][1] == hole[1][1]:
        s += 0.08
    if hi - lo == 1:
        s += 0.04
    return s


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.rng = random.Random(config.get("seed", 0))

    def act(self, state):
        try:
            a = self._act(state)
            if a["action"] in state["legal_actions"]:
                return a
        except Exception:
            pass
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}

    def _raise(self, st, target):
        lo, hi = st["min_raise_to"], st["max_raise_to"]
        t = int(target)
        return {"action": "raise", "amount": lo if t < lo else hi if t > hi else t}

    def _passive(self, st):
        return {"action": "check"} if "check" in st["legal_actions"] else {"action": "fold"}

    def _act(self, st):
        legal = st["legal_actions"]
        me = self.seat
        bb = st["bb"]
        pot = st["pot"]
        to_call = st["to_call"]
        top = max(st["bets"])
        stack = st["stacks"][me]
        can_raise = "raise" in legal
        rng = self.rng
        if st["street"] == "preflop":
            sc = _score(st["hole"])
            short = (stack + st["bets"][me]) <= 12 * bb
            if short and sc >= 0.5:
                return {"action": "raise", "amount": st["max_raise_to"]} if can_raise else (
                    {"action": "call"} if "call" in legal else {"action": "check"})
            nraise = sum(1 for h in st["history_this_hand"] if h[1] == "raise")
            if nraise == 0:   # open
                if can_raise and sc >= 0.35:
                    return self._raise(st, 3 * bb)
                return self._passive(st)
            if nraise == 1:   # 3-bet light
                if can_raise and (sc >= 0.55 or (sc >= 0.38 and rng.random() < 0.4)):
                    return self._raise(st, 3 * top)
                if "call" in legal and sc >= 0.45 and to_call <= 0.15 * stack + bb:
                    return {"action": "call"}
                return self._passive(st)
            # facing 4-bet+
            if sc >= 0.8:
                return {"action": "raise", "amount": st["max_raise_to"]} if can_raise else {"action": "call"}
            if "call" in legal and sc >= 0.62 and to_call <= 0.3 * stack:
                return {"action": "call"}
            return self._passive(st)
        cards = [CARD_INT[c] for c in st["hole"]] + [CARD_INT[c] for c in st["board"]]
        cat = evaluate(cards) >> 20
        if to_call == 0:   # barrel / c-bet
            if can_raise:
                if cat >= 1 or rng.random() < 0.8:
                    frac = 0.75 if cat >= 2 else 0.6
                    return self._raise(st, top + max(bb, int(pot * frac)))
            return {"action": "check"}
        # facing a bet
        if cat >= 2:
            if can_raise:
                return self._raise(st, top + int(2 * to_call + pot * 0.5))
            return {"action": "call"}
        price = to_call / (pot + to_call)
        if cat == 1 and price <= 0.4:
            return {"action": "call"}
        if cat == 0 and can_raise and price <= 0.35 and rng.random() < 0.15:
            return self._raise(st, top + 2 * to_call + int(pot * 0.5))   # bluff-raise
        if "call" in legal and price <= 0.2 and rng.random() < 0.3:
            return {"action": "call"}
        return {"action": "fold"}
