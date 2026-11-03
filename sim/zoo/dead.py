"""Dead bot (ZOO #10, archetype A0, 16% of T1 seats in the book, 3-8% after the realism review).

Stands for: a bot that crashed, timed out or never loaded, which the engine turns into "fold, or check when free"
on every decision (rules s3.7; llm_field notes section 3, A0). Implemented directly as always check-or-fold.
It posts its blinds, never bets and folds to any bet: free chips for whoever steals and bets into it.
"""
try:
    from sim.zoo._common import ZooAgent, check_or_fold
except ImportError:
    from _common import ZooAgent, check_or_fold


class Dead(ZooAgent):
    def act(self, state):
        return check_or_fold(state)
