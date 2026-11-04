"""Heuristic NLHE tournament bot: Chen-style preflop, category+draw postflop, pot odds, push/fold when short."""
from sim.kit.agent import Agent
from sim.kit.cards import CARD_INT, evaluate

_R = {r: i for i, r in enumerate("23456789TJQKA")}


def _pre(hole):
    a, b = _R[hole[0][0]], _R[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == lo:
        return 0.55 + 0.45 * hi / 12.0
    s = 0.45 * hi / 12.0 + 0.35 * lo / 12.0
    if hole[0][1] == hole[1][1]:
        s += 0.08
    if hi - lo == 1:
        s += 0.04
    elif hi - lo == 2:
        s += 0.02
    return s


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.seat = seat

    def act(self, state):
        try:
            a = self._decide(state)
            return a
        except Exception:
            return self._safe(state)

    @staticmethod
    def _safe(st):
        la = st.get("legal_actions", [])
        if "check" in la:
            return {"action": "check"}
        return {"action": "fold"}

    @staticmethod
    def _raise(st, target):
        lo, hi = st["min_raise_to"], st["max_raise_to"]
        if hi <= 0 or "raise" not in st["legal_actions"]:
            return None
        t = int(target)
        t = lo if t < lo else hi if t > hi else t
        return {"action": "raise", "amount": t}

    def _decide(self, st):
        la = st["legal_actions"]
        me = self.seat
        to_call = st["to_call"]
        pot = st["pot"]
        bb = st["bb"]
        stack = st["stacks"][me]
        bets = st["bets"]
        top = max(bets)
        nact = sum(1 for x in st["active"] if x)
        free = "check" in la

        def passive():
            return {"action": "check"} if free else {"action": "fold"}

        def call_or(x):
            return {"action": "call"} if "call" in la else x

        if st["street"] == "preflop":
            s = _pre(st["hole"])
            eff = (stack + bets[me]) / bb
            facing = top > bb          # someone raised
            # short stack: push/fold
            if eff <= 12:
                thr = 0.62 if not facing else 0.74
                if eff <= 6:
                    thr -= 0.06
                if s >= thr:
                    r = self._raise(st, st["max_raise_to"])
                    if r:
                        return r
                    return call_or(passive())
                return passive()
            if s >= 0.86:
                r = self._raise(st, 3 * top if facing else 3 * bb) if to_call < stack // 3 else None
                if r:
                    return r
                return call_or(passive())
            if s >= 0.72:
                if not facing:
                    r = self._raise(st, 2.5 * bb)
                    if r:
                        return r
                if to_call <= max(3 * bb, stack // 10):
                    return call_or(passive())
                return passive()
            if s >= 0.58 and not facing:
                r = self._raise(st, 2.5 * bb) if nact <= 4 else None
                if r:
                    return r
                return call_or(passive())
            if s >= 0.55 and to_call <= 2 * bb:
                return call_or(passive())
            return passive()

        cards = [CARD_INT[c] for c in st["hole"]] + [CARD_INT[c] for c in st["board"]]
        cat = evaluate(cards) >> 20
        # draw detection
        draw = 0.0
        suits = {}
        for c in cards:
            suits[c & 3] = suits.get(c & 3, 0) + 1
        if st["street"] != "river":
            if max(suits.values()) == 4:
                draw = 0.19 * (2 if st["street"] == "flop" else 1)
            ranks = sorted({c >> 2 for c in cards})
            if 12 in ranks:
                ranks.append(-1)
            ranks.sort()
            for i in range(len(ranks) - 3):
                if ranks[i + 3] - ranks[i] == 3 and len(set(ranks[i:i + 4])) == 4:
                    draw = max(draw, 0.17 if st["street"] == "flop" else 0.09)
        # pair quality: does a hole card pair the board / overpair
        strength = 0.0
        if cat >= 4:
            strength = 1.0
        elif cat >= 2:
            strength = 0.9 if cat == 2 else 0.95
        elif cat == 1:
            bh = max((CARD_INT[c] >> 2) for c in st["board"])
            hh = max(_R[st["hole"][0][0]], _R[st["hole"][1][0]])
            strength = 0.6 if hh >= bh else 0.45
        else:
            strength = 0.1
        if nact >= 4:
            strength -= 0.1
        if to_call == 0:
            if strength >= 0.85 or (strength >= 0.55 and nact <= 3):
                r = self._raise(st, top + 0.65 * pot)
                if r:
                    return r
            if draw >= 0.17 and nact <= 3:
                r = self._raise(st, top + 0.5 * pot)
                if r:
                    return r
            return {"action": "check"}
        odds = to_call / (pot + to_call)
        equity = strength * 0.85 + draw
        if strength >= 0.9:
            r = self._raise(st, top + 0.8 * pot) if to_call < stack // 2 else None
            if r:
                return r
            return call_or(passive())
        if equity > odds + 0.05 or (strength >= 0.55 and odds < 0.35):
            return call_or(passive())
        return passive()

    def observe(self, event):
        pass
