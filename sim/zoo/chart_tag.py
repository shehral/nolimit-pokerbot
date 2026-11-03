"""Textbook heuristic TAG (ZOO #4, archetype A5, the MODAL bot: 26-29% of seats in llm_field notes section 3).

Stands for: the bot an AI assistant writes first when told "build a poker bot, 1 ms limit" (llm_field notes
section 1.1, Husky Hold'em Bench census: 0 of 12 final bots used Monte Carlo; all map the hand to a strength scalar and
compare it with fixed thresholds). Preflop: Chen score tiers, position-aware opens of exactly 3 BB (param), 3-bets
premiums to 3x, calls with strong hands, push/fold by Chen score under 12 BB, no ICM. Postflop: made-hand gates
(0 air / 1 weak pair or draw / 2 top pair+ / 3 two pair+): bet 0.66 pot with class >= 2, c-bet 0.5 pot with class >= 1
when it was the preflop raiser, call bets when the pot-odds ratio is under 0.30 (class 1) or always (class >= 2), fold
air to any bet. No opponent model, no bluffs by default (param bluff), fixed sizes: fingerprint = opens of exactly
3 BB and postflop bets at one pot fraction. ICM-blind: it makes the same plays on the bubble as in chips.
"""
try:
    from sim.zoo._common import (ZooAgent, all_in, call, check, fold, chen, hole_ints, board_ints, post_class,
                                 preflop_behind, top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)
except ImportError:
    from _common import (ZooAgent, all_in, call, check, fold, chen, hole_ints, board_ints, post_class,
                         preflop_behind, top_bet, raise_amt, eff_stack_bb, fold_or_check, call_or_check)


class ChartTag(ZooAgent):
    PARAMS = {"open_bb": 3.0, "open_late": 5.5, "open_mid": 7.0, "open_early": 8.0, "premium": 10.0, "strong": 8.0,
              "short_bb": 12.0, "short_jam": 7.5, "short_call": 9.0, "bet_frac": 0.66, "cbet_frac": 0.5,
              "odds_gate": 0.30, "bluff": 0.0}
    JITTER = {"open_bb": (2.5, 3.0), "open_late": (4.5, 6.5), "open_mid": (6.0, 8.0), "open_early": (7.0, 9.0),
              "premium": (9.5, 11.0), "short_bb": (10.0, 14.0), "short_jam": (6.5, 8.5), "bet_frac": (0.55, 0.75),
              "cbet_frac": (0.4, 0.66), "odds_gate": (0.2, 0.3)}

    def act(self, state):
        if self.new_hand(state):
            self.hand_state["pf_raiser"] = False
        return self.preflop(state) if state["street"] == "preflop" else self.postflop(state)

    # ---------------------------------------------------------------- preflop
    def preflop(self, state):
        p, legal, bb = self.p, state["legal_actions"], state["bb"]
        score = chen(state["hole"])
        eff = eff_stack_bb(state)
        to_call = state["to_call"]
        facing_raise = top_bet(state) > bb
        if eff <= p["short_bb"]:                       # chip-EV push/fold, no ICM
            need = p["short_call"] if facing_raise else p["short_jam"]
            if score >= need:
                self.hand_state["pf_raiser"] = True
                return all_in(state)
            return fold_or_check(state)
        if not facing_raise:
            k = len(preflop_behind(state))
            need = p["open_late"] if k <= 2 else p["open_mid"] if k <= 4 else p["open_early"]
            if to_call == 0:                           # big blind with the option
                need = max(need, p["strong"])
            if score >= need and "raise" in legal:
                self.hand_state["pf_raiser"] = True
                return raise_amt(state, int(p["open_bb"] * bb))
            return call_or_check(state) if (to_call == 0 or (score >= p["strong"] and to_call <= bb)) else fold_or_check(state)
        if score >= p["premium"]:
            if "raise" in legal and to_call < 0.3 * state["stacks"][state["seat"]]:
                self.hand_state["pf_raiser"] = True
                return raise_amt(state, 3 * top_bet(state))
            return call_or_check(state)
        if score >= p["strong"] and to_call <= 0.12 * state["stacks"][state["seat"]]:
            return call_or_check(state)
        return fold_or_check(state)

    # ---------------------------------------------------------------- postflop
    def postflop(self, state):
        p, legal = self.p, state["legal_actions"]
        pc = post_class(hole_ints(state), board_ints(state))
        pot, to_call = state["pot"], state["to_call"]
        if to_call == 0:
            if pc >= 2 and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(p["bet_frac"] * pot))
            if pc >= 1 and self.hand_state.get("pf_raiser") and state["street"] == "flop" and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(p["cbet_frac"] * pot))
            if p["bluff"] and self.rng.random() < p["bluff"] and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(p["cbet_frac"] * pot))
            return check()
        odds = to_call / (pot + to_call)
        if pc == 3:
            if "raise" in legal and odds < 0.4 and self.rng.random() < 0.5:
                return raise_amt(state, top_bet(state) + int(p["bet_frac"] * pot))
            return call()
        if pc == 2:
            return call()
        if pc == 1 and odds < p["odds_gate"]:
            return call()
        return fold()
