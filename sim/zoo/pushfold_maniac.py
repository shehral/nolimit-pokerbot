"""Push/fold maniac (ZOO #5, archetype A7, 6-7.5% of seats in llm_field notes section 3).

Stands for: the team whose assistant said "push/fold is optimal in sit-and-gos" and applied it too early and too wide,
or the "exploit the folders" plan with a short-stack mode that switches on too soon. Under about 15 BB (effective
stack, param) it shoves the top X% of hands (default 35%, ranked by Chen score) first in, and calls shoves with the top
25%; above 15 BB it plays loose-aggressive: opens 55% of hands to 3 BB, re-raises the top 12%, c-bets 0.6 pot with
any pair or draw, barrels half of the rest. Signature: all-in on 20-30% of hands from the first levels, always the
same size. No ICM, no opponent model.
"""
try:
    from sim.zoo._common import (ZooAgent, all_in, call, check, fold, CHEN_PCT, label, hole_ints, board_ints, post_class,
                                 top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)
except ImportError:
    from _common import (ZooAgent, all_in, call, check, fold, CHEN_PCT, label, hole_ints, board_ints, post_class,
                         top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)


class PushFoldManiac(ZooAgent):
    PARAMS = {"short_bb": 15.0, "jam_pct": 0.35, "call_pct": 0.25, "open_pct": 0.55, "reraise_pct": 0.12}
    JITTER = {"short_bb": (12.0, 20.0), "jam_pct": (0.25, 0.5), "call_pct": (0.18, 0.32), "open_pct": (0.4, 0.7),
              "reraise_pct": (0.08, 0.16)}

    def act(self, state):
        p, legal, bb = self.p, state["legal_actions"], state["bb"]
        pct = CHEN_PCT[label(state["hole"])]
        if state["street"] == "preflop":
            eff = eff_stack_bb(state)
            facing = top_bet(state) > bb
            if eff <= p["short_bb"]:
                if pct <= (p["call_pct"] if facing else p["jam_pct"]):
                    return all_in(state)
                return fold_or_check(state)
            if not facing:
                if pct <= p["open_pct"] and "raise" in legal:
                    return raise_amt(state, 3 * bb)
                return call_or_check(state) if state["to_call"] == 0 else fold()
            if pct <= p["reraise_pct"] and "raise" in legal:
                return raise_amt(state, 3 * top_bet(state))
            return call_or_check(state) if pct <= 0.25 and state["to_call"] <= 0.1 * state["stacks"][state["seat"]] else fold_or_check(state)
        pc = post_class(hole_ints(state), board_ints(state))
        pot = state["pot"]
        if state["to_call"] == 0:
            if (pc >= 1 or self.rng.random() < 0.5) and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(0.6 * pot))
            return check()
        return call() if pc >= 1 else fold()
