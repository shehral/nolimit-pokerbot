"""Calling station (ZOO #6, archetype A6 "inverted pot odds", 5-6% of seats in llm_field notes section 3).

Stands for: the Husky Hold'em leaderboard winner's leak ("calls every shove that covers it", llm_field notes
section 1.1: `hand_strength >= 0.6 or pot_odds > 0.25`) and every bot with a "call if pot_odds > x" bug. Preflop it calls
any pair, ace, king, suited or connected hand up to a third of its stack (and any hand for a min-raise), almost never
raises (3% with a big hand). Postflop it calls down with any pair or draw at any price, folds only air facing a bet,
and bets only with two pair or better. High VPIP, weak showdown hands, light all-in calls: best answered by
maximum value and no bluffs.
"""
try:
    from sim.zoo._common import (ZooAgent, call, check, fold, label, ranks_of, is_suited, chen, hole_ints, board_ints,
                                 post_class, top_bet, raise_amt, call_or_check, fold_or_check)
except ImportError:
    from _common import (ZooAgent, call, check, fold, label, ranks_of, is_suited, chen, hole_ints, board_ints,
                         post_class, top_bet, raise_amt, call_or_check, fold_or_check)


class Station(ZooAgent):
    PARAMS = {"raise_prob": 0.03, "preflop_cap": 0.33, "min_chen": 4.0}
    JITTER = {"raise_prob": (0.0, 0.06), "preflop_cap": (0.2, 0.5), "min_chen": (3.0, 6.0)}

    def act(self, state):
        p, legal = self.p, state["legal_actions"]
        stack = state["stacks"][state["seat"]]
        if state["street"] == "preflop":
            hi, lo = ranks_of(state["hole"])
            playable = hi == lo or hi >= 11 or is_suited(state["hole"]) or hi - lo <= 2 or chen(state["hole"]) >= p["min_chen"] + 2
            if chen(state["hole"]) >= 12 and "raise" in legal and self.rng.random() < 0.5:
                return raise_amt(state, 3 * max(top_bet(state), state["bb"]))
            if state["to_call"] == 0:
                return check()
            if state["to_call"] <= state["bb"] or (playable and state["to_call"] <= p["preflop_cap"] * stack):
                return call()
            return fold()
        pc = post_class(hole_ints(state), board_ints(state))
        if pc >= 3 and "raise" in legal and self.rng.random() < 0.5:
            return raise_amt(state, top_bet(state) + int(0.6 * state["pot"]))
        if "raise" in legal and state["to_call"] == 0 and self.rng.random() < p["raise_prob"]:
            return raise_amt(state, top_bet(state) + int(0.5 * state["pot"]))
        if state["to_call"] == 0:
            return check()
        return call() if pc >= 1 else fold()
