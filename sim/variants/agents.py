"""Simple legal example agents, one per variant (the kind of sample bot a starter kit ships).

VariantExample plays like sim.kit.example_agent.ExampleAgent (preflop score tiers, postflop by made-hand category)
and handles the extra request/fields of each variant:
  toss          request "discard": toss the card whose removal leaves the best two-card preflop score + board fit
  shortdeck     categories from the short-deck evaluator (flush above full house); preflop score as usual
  bounty        a hole/board card of your own bounty rank nudges the hand up one tier (more aggression)
  doubleboard   hand category = better of the two boards, pairs need to be made with BOTH hole cards' help
  antes/bbante  plain play (the ante only changes the pot odds)
Every answer is legal by construction (raises clamped, discard index in range). act() takes a few microseconds.
"""
from sim.kit.agent import Agent
from sim.kit.cards import CARD_INT
from sim.kit.example_agent import preflop_score

from .evals import evaluate_any, evaluate_short

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}


def best_pair_score(hole):
    """Best preflop_score over the 2-card subsets of 2 or 3 hole cards."""
    if len(hole) == 2:
        return preflop_score(hole)
    a, b, c = hole[0], hole[1], hole[2]
    return max(preflop_score((a, b)), preflop_score((a, c)), preflop_score((b, c)))


class VariantExample(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.variant = config.get("variant", "plain")
        self.ev = evaluate_short if self.variant == "shortdeck" else evaluate_any

    def act(self, state):
        if state.get("request") == "discard":
            return self._discard(state)
        legal = state["legal_actions"]
        to_call = state["to_call"]
        bb = state["bb"]
        stack = state["stacks"][self.seat]
        pot = state["pot"]
        top_bet = max(state["bets"])
        brank = state.get("bounty_rank")
        if state["street"] == "preflop":
            hole = state["hole"]
            score = best_pair_score(hole)
            if brank is not None and any(c[0] == brank for c in hole):
                score += 0.08
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
        hole_c = [CARD_INT[c] for c in state["hole"]]
        cat = 0
        for bd in (state.get("boards") or [state["board"]]):
            bc = [CARD_INT[c] for c in bd]
            s_with = self.ev(hole_c + bc) >> 20
            s_board = self.ev(bc + hole_c[:0]) >> 20 if len(bc) >= 5 else 0
            c = s_with if s_with > s_board else 0
            if c > cat:
                cat = c
        if brank is not None and cat == 0 and any(c[0] == brank for c in state["hole"] + state["board"]):
            cat = 1
        if cat >= 2:
            if "raise" in legal:
                return self._raise(state, top_bet + (2 * (pot + to_call)) // 3)
            return {"action": "call"} if "call" in legal else {"action": "check"}
        if "check" in legal:
            return {"action": "check"}
        if cat == 1 and to_call * 2 <= pot:
            return {"action": "call"}
        return {"action": "fold"}

    def _discard(self, state):
        hole = state["hole"]
        board_ranks = [_RANK[c[0]] for c in state["board"]]
        best_i, best_s = 0, -1.0
        for i in range(len(hole)):
            kept = [c for j, c in enumerate(hole) if j != i]
            s = preflop_score(kept)
            for c in kept:   # prefer keeping cards that pair the board
                if _RANK[c[0]] in board_ranks:
                    s += 0.25
            if s > best_s:
                best_i, best_s = i, s
        return {"action": "discard", "index": best_i}

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


class TossExample(VariantExample):
    pass


class ShortDeckExample(VariantExample):
    pass


class BountyExample(VariantExample):
    pass


class DoubleBoardExample(VariantExample):
    pass


class AnteExample(VariantExample):
    pass


EXAMPLES = {"plain": VariantExample, "toss": TossExample, "shortdeck": ShortDeckExample, "bounty": BountyExample,
            "doubleboard": DoubleBoardExample, "antes": AnteExample, "bbante": AnteExample}


class CrashDiscard(VariantExample):
    """Test fixture: answers every discard request illegally (exercises the strictness path)."""
    def _discard(self, state):
        return {"action": "discard", "index": 7}
