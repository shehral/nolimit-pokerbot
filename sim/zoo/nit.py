"""Nit (ZOO #7, archetype A8, 5-6% of seats in llm_field notes section 3).

Stands for: the over-cautious build ("survival first" prompt, or over-reading the ICM calculator): premiums only.
Preflop it raises to 3 BB with the top 7% of hands (AA-TT, AK, AQs), re-raises or calls raises only with the top 3%
(AA-QQ, AK), and shoves/calls shoves under 10 BB with the top 8%. Postflop it bets two pair or better 0.7 pot, calls
with top pair or better, and folds everything else. VPIP under 10%. Folds to raises, so steals work; its raises are real.
"""
try:
    from sim.zoo._common import (ZooAgent, all_in, call, check, fold, CHEN_PCT, label, hole_ints, board_ints, post_class,
                                 top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)
except ImportError:
    from _common import (ZooAgent, all_in, call, check, fold, CHEN_PCT, label, hole_ints, board_ints, post_class,
                         top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)


class Nit(ZooAgent):
    PARAMS = {"open_pct": 0.07, "continue_pct": 0.03, "short_bb": 10.0, "short_pct": 0.08}
    JITTER = {"open_pct": (0.05, 0.10), "continue_pct": (0.02, 0.045), "short_bb": (8.0, 12.0), "short_pct": (0.06, 0.11)}

    def act(self, state):
        p, legal, bb = self.p, state["legal_actions"], state["bb"]
        pct = CHEN_PCT[label(state["hole"])]
        if state["street"] == "preflop":
            if eff_stack_bb(state) <= p["short_bb"]:
                return all_in(state) if pct <= p["short_pct"] else fold_or_check(state)
            if top_bet(state) > bb:
                if pct <= p["continue_pct"]:
                    return call_or_check(state) if state["to_call"] >= 0.3 * state["stacks"][state["seat"]] else \
                        raise_amt(state, 3 * top_bet(state))
                return fold_or_check(state)
            if pct <= p["open_pct"]:
                return raise_amt(state, 3 * bb)
            return check() if state["to_call"] == 0 else fold()
        pc = post_class(hole_ints(state), board_ints(state))
        if state["to_call"] == 0:
            return raise_amt(state, top_bet(state) + int(0.7 * state["pot"])) if pc >= 3 else check()
        return call() if pc >= 2 else fold()
