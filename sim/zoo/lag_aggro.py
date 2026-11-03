"""Loose-aggressive "be aggressive against bots" bot (ZOO #8, archetype A10 steal-heavy / L2b, 8% of T6 seats).

Stands for: the team (or assistant) that reasons "AI bots are passive, aggression wins" (llm_field notes sections
3 and 4.1: the likely level-2 build, "the trap"). It opens 60% of hands from late position and 35% from early to 2.5 BB,
3-bets the top 12%, c-bets every flop 0.5 pot regardless of the hand, barrels the turn 70% and the river 40% when
called, bets any check-through spot, and shoves wide under 10 BB. It folds to a raise only with air on the flop or
later. It bleeds against the over-calling bloc (station) and wins against folders (nit, chart_tag air).
"""
try:
    from sim.zoo._common import (ZooAgent, all_in, call, check, fold, CHEN_PCT, label, hole_ints, board_ints, post_class,
                                 preflop_behind, top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)
except ImportError:
    from _common import (ZooAgent, all_in, call, check, fold, CHEN_PCT, label, hole_ints, board_ints, post_class,
                         preflop_behind, top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)


class LagAggro(ZooAgent):
    PARAMS = {"open_late": 0.60, "open_early": 0.35, "reraise_pct": 0.12, "cbet_frac": 0.5, "turn_barrel": 0.7,
              "river_barrel": 0.4, "short_bb": 10.0, "jam_pct": 0.40}
    JITTER = {"open_late": (0.5, 0.75), "open_early": (0.25, 0.45), "reraise_pct": (0.08, 0.18),
              "cbet_frac": (0.4, 0.66), "turn_barrel": (0.5, 0.85), "river_barrel": (0.25, 0.55),
              "jam_pct": (0.3, 0.55)}

    def act(self, state):
        if self.new_hand(state):
            self.hand_state["agg"] = False
        p, legal, bb = self.p, state["legal_actions"], state["bb"]
        pct = CHEN_PCT[label(state["hole"])]
        if state["street"] == "preflop":
            facing = top_bet(state) > bb
            if eff_stack_bb(state) <= p["short_bb"]:
                if pct <= (0.25 if facing else p["jam_pct"]):
                    self.hand_state["agg"] = True
                    return all_in(state)
                return fold_or_check(state)
            if not facing:
                k = len(preflop_behind(state))
                if pct <= (p["open_late"] if k <= 3 else p["open_early"]) and "raise" in legal:
                    self.hand_state["agg"] = True
                    return raise_amt(state, int(2.5 * bb))
                return check() if state["to_call"] == 0 else fold()
            if pct <= p["reraise_pct"] and "raise" in legal:
                self.hand_state["agg"] = True
                return raise_amt(state, 3 * top_bet(state))
            return call_or_check(state) if pct <= 0.3 and state["to_call"] <= 0.12 * state["stacks"][state["seat"]] \
                else fold_or_check(state)
        pc = post_class(hole_ints(state), board_ints(state))
        pot = state["pot"]
        if state["to_call"] == 0:
            freq = {"flop": 1.0, "turn": p["turn_barrel"], "river": p["river_barrel"]}[state["street"]]
            if (pc >= 2 or self.hand_state.get("agg") or self.rng.random() < 0.5) and self.rng.random() < freq \
                    and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(p["cbet_frac"] * pot))
            return check()
        if pc >= 3 and "raise" in legal:
            return raise_amt(state, top_bet(state) + int(0.75 * pot))
        return call() if pc >= 1 else fold()
