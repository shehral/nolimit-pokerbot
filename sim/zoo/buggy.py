"""Buggy chart bot (ZOO #11, archetype A2 "illegal action" plus crash branches, 1-6% of seats).

Stands for: the Husky Hold'em failure mode (llm_field notes section 1.1: 24 of 50 testable drafts broke; the top
error is an illegal raise such as "RAISE amount 0", raise-to vs raise-by confusion; others are crashes in a rarely
reached branch). Plays exactly like chart_tag, but 10% of decisions raise an exception and 5% return an illegal raise
(amount 0, below the minimum, or above the all-in). The engine auto-folds (or checks) every one of those. Rates are
parameters; draws are made before the logic so they are exact in expectation.
"""
try:
    from sim.zoo.chart_tag import ChartTag
except ImportError:
    from chart_tag import ChartTag


class Buggy(ChartTag):
    PARAMS = {**ChartTag.PARAMS, "p_exception": 0.10, "p_illegal": 0.05}
    JITTER = {**ChartTag.JITTER, "p_exception": (0.05, 0.15), "p_illegal": (0.02, 0.08)}

    def act(self, state):
        r = self.rng.random()
        if r < self.p["p_exception"]:
            raise KeyError("player_hands")       # reads a state field that does not exist (Husky AttributeError class)
        if r < self.p["p_exception"] + self.p["p_illegal"]:
            bad = (0, state["min_raise_to"] - 1, state["max_raise_to"] + 1)[self.rng.randrange(3)]
            return {"action": "raise", "amount": bad}     # illegal even when raise is not legal at all
        return super().act(state)
