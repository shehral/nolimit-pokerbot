"""ai_student_fast: a simple, defensive, stateless No Limit bot (rehearsal field entrant).

Preflop: a hand-strength score (pairs, high cards, suited, connected). Strong hands raise, medium hands call
cheap prices, short stacks shove good hands. Postflop: the kit's 7-card evaluator sets the hand category; two
pair or better bets about 2/3 pot, one pair calls small bets and otherwise checks, nothing else is played.
Every decision is wrapped: any exception or illegal answer falls back to check-or-fold.
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim.kit.agent import Agent, check_or_fold, validate_action  # noqa: E402
from sim.kit.cards import CARD_INT, evaluate  # noqa: E402

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}


def _pre_score(hole):
    a, b = _RANK[hole[0][0]], _RANK[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == lo:
        return 0.55 + 0.45 * hi / 12.0
    s = 0.45 * hi / 12.0 + 0.35 * lo / 12.0
    if hole[0][1] == hole[1][1]:
        s += 0.08
    if hi - lo <= 2:
        s += 0.04
    return s


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.seat = seat

    def act(self, state):
        try:
            action = self._decide(state)
            kind, amt, err = validate_action(action, state["legal_actions"],
                                             state["min_raise_to"], state["max_raise_to"])
            if err is None:
                return {"action": kind, "amount": amt} if kind == "raise" else {"action": kind}
        except Exception:
            pass
        return check_or_fold(state)

    def observe(self, event):
        return None

    # ------------------------------------------------------------------ strategy
    def _decide(self, state):
        legal = state["legal_actions"]
        to_call = state["to_call"]
        bb = max(state["bb"], 1)
        stack = state["stacks"][self.seat]
        pot = max(state["pot"], 1)
        top_bet = max(state["bets"])
        can_raise = "raise" in legal and state["min_raise_to"] > 0

        def check_or_fold_():
            return {"action": "check"} if "check" in legal else {"action": "fold"}

        def raise_to(target):
            lo, hi = state["min_raise_to"], state["max_raise_to"]
            t = lo if target < lo else hi if target > hi else int(target)
            return {"action": "raise", "amount": t}

        def shove():
            if can_raise:
                return {"action": "raise", "amount": state["max_raise_to"]}
            if "call" in legal:
                return {"action": "call"}
            return check_or_fold_()

        if state["street"] == "preflop":
            score = _pre_score(state["hole"])
            effective = stack + state["bets"][self.seat]
            if effective <= 10 * bb and score >= 0.6:
                return shove()
            if score >= 0.80:
                if can_raise and to_call <= 4 * bb:
                    return raise_to(3 * max(top_bet, bb))
                if "call" in legal:
                    return {"action": "call"}
            if score >= 0.62 and "call" in legal and to_call <= 3 * bb:
                return {"action": "call"}
            return check_or_fold_()

        cards = [CARD_INT[c] for c in state["hole"]] + [CARD_INT[c] for c in state["board"]]
        cat = evaluate(cards) >> 20
        if cat >= 2:
            if can_raise:
                return raise_to(top_bet + (2 * (pot + to_call)) // 3)
            return {"action": "call"} if "call" in legal else check_or_fold_()
        if "check" in legal:
            return {"action": "check"}
        if cat == 1 and to_call * 2 <= pot and "call" in legal:
            return {"action": "call"}
        return {"action": "fold"} if "fold" in legal else check_or_fold_()


AGENT = Bot
