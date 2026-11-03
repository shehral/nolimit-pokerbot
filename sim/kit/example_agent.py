"""Example agent: the kind of sample bot a starter kit ships. Simple and always legal; measured act() p50 about
1.2-1.7 us and p99 about 5-10 us on an M4.

Preflop it scores the two hole cards 0..1 (pairs, high cards, suited, connected): it raises to 3x the current bet
with the top hands, calls cheap prices with medium hands, shoves good hands when short (10 bb or less), and
otherwise checks or folds. Postflop it uses the kit evaluator's hand category: two pair or better bets/raises about
2/3 pot, one pair calls a bet up to half the pot, anything else checks or folds. No ICM, no opponent model, no
randomness. Every raise is clamped to [min_raise_to, max_raise_to] and only made when "raise" is legal.

Run it:  .venv/bin/python -m sim.kit.runner -e 'ex=sim/kit/example_agent.py:ExampleAgent*8' -n 1000
"""
try:
    from sim.kit.agent import Agent
    from sim.kit.cards import CARD_INT, evaluate
except ImportError:   # flat layout: this file sitting next to the kit modules
    from agent import Agent
    from cards import CARD_INT, evaluate

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}


def preflop_score(hole):
    """0..1 heuristic hand score: 22 = 0.55, AA = 1.0, AKs = 0.89, AKo = 0.81, T9s = 0.62, 72o = 0.19."""
    a, b = _RANK[hole[0][0]], _RANK[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == lo:
        return 0.55 + 0.45 * hi / 12.0
    s = 0.45 * hi / 12.0 + 0.35 * lo / 12.0
    if hole[0][1] == hole[1][1]:
        s += 0.08
    if hi - lo == 1:
        s += 0.04
    return s


class ExampleAgent(Agent):
    def act(self, state):
        legal = state["legal_actions"]
        to_call = state["to_call"]
        bb = state["bb"]
        stack = state["stacks"][self.seat]
        pot = state["pot"]
        top_bet = max(state["bets"])
        if state["street"] == "preflop":
            score = preflop_score(state["hole"])
            if stack + state["bets"][self.seat] <= 10 * bb and score >= 0.65:
                return self._all_in(legal, state)
            if score >= 0.80:
                if "raise" in legal and to_call <= 4 * bb:
                    return self._raise(state, 3 * max(top_bet, bb))
                if "call" in legal and (score >= 0.88 or to_call <= stack // 5):
                    return {"action": "call"}
            elif score >= 0.62 and "call" in legal and to_call <= 3 * bb:
                return {"action": "call"}
            return {"action": "check"} if "check" in legal else {"action": "fold"}
        cards = [CARD_INT[c] for c in state["hole"]] + [CARD_INT[c] for c in state["board"]]
        cat = evaluate(cards) >> 20
        if cat >= 2:
            if "raise" in legal:
                return self._raise(state, top_bet + (2 * (pot + to_call)) // 3)
            return {"action": "call"} if "call" in legal else {"action": "check"}
        if "check" in legal:
            return {"action": "check"}
        if cat == 1 and to_call * 2 <= pot:
            return {"action": "call"}
        return {"action": "fold"}

    @staticmethod
    def _raise(state, target):
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        t = lo if target < lo else hi if target > hi else int(target)
        return {"action": "raise", "amount": t}

    @staticmethod
    def _all_in(legal, state):
        if "raise" in legal:
            return {"action": "raise", "amount": state["max_raise_to"]}
        if "call" in legal:
            return {"action": "call"}
        return {"action": "check"}


Agent = ExampleAgent   # so the spec 'sim/kit/example_agent.py' (no attribute) also resolves to this class
