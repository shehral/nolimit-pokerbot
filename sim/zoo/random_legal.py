"""Random legal bot (ZOO #9, archetype A3 "placeholder", 0.5-4% of seats in llm_field notes section 3).

Stands for: the template sample, an always-random bot, or a placeholder submitted just to beat the clock. Uniform over
the legal actions of each decision; a raise picks a uniform amount between the minimum and maximum raise-to (so about
half of its raises are huge). Always legal and fast, plays no strategy, goes all-in often and busts early.
"""
try:
    from sim.zoo._common import ZooAgent
except ImportError:
    from _common import ZooAgent


class RandomLegal(ZooAgent):
    PARAMS = {"raise_weight": 1.0, "small_raise_prob": 0.0}
    JITTER = {"raise_weight": (0.5, 1.5), "small_raise_prob": (0.0, 0.5)}

    def act(self, state):
        legal = state["legal_actions"]
        weights = [self.p["raise_weight"] if a == "raise" else 1.0 for a in legal]
        kind = self.rng.choices(legal, weights)[0]
        if kind != "raise":
            return {"action": kind}
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        if self.rng.random() < self.p["small_raise_prob"]:
            hi = min(hi, lo + 3 * state["bb"])
        return {"action": "raise", "amount": self.rng.randint(lo, hi)}
