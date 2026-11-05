"""The fragile zoo: six chart_tag bots, each carrying ONE bug of the kind hour-old LLM-written poker code ships with.

Each bot plays exactly like sim/zoo/chart_tag.py (the modal "textbook TAG" an assistant writes first) except in the
branch that holds its bug. The bugs are written the way such code is written (no deliberate random failure: every
error has a real trigger), so the error RATE depends on what the other seats do, which is what FRAGILITY notes
measures (does an unusual but legal size from our seat raise it?).

  key               bug (the idiom)                                         trigger in the mock engine
  frag_bbround      rounds every raise down to whole big blinds AFTER        min_raise_to (or a short all-in) that is
                    clamping it ("amount // bb * bb")                        not a multiple of the big blind
  frag_shortallin   computes the minimum raise itself from the last raise    an all-in raise for less than a full raise
                    increment (ignores min_raise_to; it min-raises two pair  (the engine keeps the old increment) or a
                    or better postflop), and looks up which bet level a      call all-in for less than the bet
                    call matched (levels.index(amount))
  frag_headsup      8-max blind order (SB = the seat after the button) also  two players left: as the button (who is
                    used heads-up, where the button posts the small blind   the SB) it "checks its option" facing the
                                                                             blind (illegal -> folded); as the BB it
                                                                             "completes" with call when it may check
  frag_keyerror     facing one open, reads an open-size table keyed by the   an open that is not a whole or half big
                    open in big blinds (OPEN_ADJ[cur / bb]): KeyError        blind multiple (2.2, 2.55, 2.75 bb ...)
  frag_zerodiv      postflop "stack-off" ratio 1 / (min(stacks) / pot):      a bet or raise from a player who is now
                    ZeroDivisionError when the bettor has no chips behind    all-in (stack 0)
  frag_index        "who acts after me" = live_behind[0] in the premium /    every other player still in the hand is
                    nuts branch (IndexError on an empty list); its history   all-in; the name map fires only on a
                    parser maps action names through a fixed tuple           platform with other names ("bet", "allin",
                    (ValueError on an unexpected name)                       "ante"): inert in this engine (TURBO: no
                                                                             antes)

The engine treats an exception or an illegal action as the rules do (s3.7): fold, or check when checking is legal.
"""
try:
    from sim.zoo.chart_tag import ChartTag
    from sim.zoo._common import board_ints, call, check, chen, eff_stack_bb, hole_ints, post_class, top_bet
except ImportError:          # flat import (file path)
    from chart_tag import ChartTag                       # noqa: F401
    from _common import board_ints, call, check, chen, eff_stack_bb, hole_ints, post_class, top_bet   # noqa: F401

STREETS = ("preflop", "flop", "turn", "river")


def _street_history(state):
    start = state["street_starts"][STREETS.index(state["street"])]
    return state["history_this_hand"][start:]


class FragBBRound(ChartTag):
    """Bug 1 (integer division): keeps every bet in whole big blinds, rounding down after the clamp."""

    def act(self, state):
        out = super().act(state)
        if out.get("action") == "raise":
            bb = state["bb"]
            out = {"action": "raise", "amount": out["amount"] // bb * bb}   # "bets in whole big blinds"
        return out


class FragShortAllIn(ChartTag):
    """Bug 2 (all-in for less): its own minimum-raise arithmetic and bet-level bookkeeping assume every raise is a
    full raise and every call matches a bet level."""

    def _levels(self, state):
        levels = [state["bb"] if state["street"] == "preflop" else 0]
        for _seat, kind, amt in _street_history(state):
            if kind == "raise":
                levels.append(amt)
            elif kind == "call" and amt > 0:
                levels.index(amt)          # "which level did they call" (ValueError: a call all-in for less)
        return levels

    def act(self, state):
        levels = self._levels(state)
        last_inc = levels[-1] - levels[-2] if len(levels) >= 2 else 0
        my_min = levels[-1] + max(state["bb"], last_inc)           # ignores the engine's min_raise_to
        if state["street"] != "preflop" and state["to_call"] > 0 and "raise" in state["legal_actions"] and \
                post_class(hole_ints(state), board_ints(state)) == 3:
            return {"action": "raise", "amount": min(my_min, state["max_raise_to"])}   # min-raises its monsters
        out = super().act(state)
        if out.get("action") == "raise" and out["amount"] < state["max_raise_to"]:
            amt = max(out["amount"] if out["amount"] > state["min_raise_to"] else 0, my_min)
            out = {"action": "raise", "amount": min(amt, state["max_raise_to"])}
        return out


class FragHeadsUp(ChartTag):
    """Bug 3 (heads-up blind order): the 8-max rule SB = first live seat after the button, used heads-up too."""

    def act(self, state):
        if state["street"] == "preflop" and state["n_alive"] == 2:
            n, b, alive = state["n_seats"], state["button"], state["alive"]
            sb_guess = next((b + d) % n for d in range(1, n + 1) if alive[(b + d) % n])
            bb_guess = next((sb_guess + d) % n for d in range(1, n + 1) if alive[(sb_guess + d) % n])
            me = state["seat"]
            if self.new_hand(state):
                self.hand_state["pf_raiser"] = False
            if me == bb_guess and top_bet(state) <= state["bb"]:         # "the big blind's option"
                if chen(state["hole"]) < self.p["strong"] or "raise" not in state["legal_actions"]:
                    return check()                                       # illegal when facing the blind
            if me == sb_guess and state["to_call"] == 0:                 # "complete from the small blind"
                return call()                                            # illegal when it may check
        return super().act(state)


class FragKeyError(ChartTag):
    """Bug 4 (KeyError): an open-size adjustment table keyed by the open in big blinds, looked up without rounding."""
    OPEN_ADJ = {2.0: 0.0, 2.5: 0.25, 3.0: 0.5, 3.5: 0.75, 4.0: 1.0, 4.5: 1.25}

    def act(self, state):
        if state["street"] == "preflop" and eff_stack_bb(state) > self.p["short_bb"]:
            raises = [a for a in _street_history(state) if a[1] == "raise"]
            if len(raises) == 1 and state["to_call"] > 0:
                size = raises[0][2] / state["bb"]
                if size < 5.0:                                           # bigger opens are treated as jams
                    adj = self.OPEN_ADJ[size]                            # KeyError on 2.55 bb, 2.75 bb, ...
                    if chen(state["hole"]) < self.p["strong"] + adj:
                        return {"action": "fold"}
        return super().act(state)


class FragZeroDiv(ChartTag):
    """Bug 5 (ZeroDivisionError): a postflop stack-to-pot ratio that divides by the smaller stack."""

    def act(self, state):
        if state["street"] != "preflop" and state["to_call"] > 0:
            me = state["seat"]
            bets, stacks = state["bets"], state["stacks"]
            bettor = max((j for j in range(state["n_seats"]) if j != me and state["active"][j]), key=lambda j: bets[j])
            spr = min(stacks[me], stacks[bettor]) / state["pot"]
            stackoff = 1.0 / spr                                         # ZeroDivisionError: the bettor is all-in
            if stackoff > 3.0 and "call" in state["legal_actions"]:
                return {"action": "call"}                                # "pot committed"
        return super().act(state)


class FragIndex(ChartTag):
    """Bug 6 (IndexError / unexpected action name): 'the player after me' read from a list that can be empty, and a
    history parser with a fixed action-name tuple."""
    ACTION_NAMES = ("post_sb", "post_bb", "fold", "check", "call", "raise")

    def act(self, state):
        for _seat, kind, _amt in state["history_this_hand"]:
            self.ACTION_NAMES.index(kind)                                # ValueError on "bet", "allin", "ante", ...
        strong = False
        if state["street"] == "preflop":
            strong = top_bet(state) > state["bb"] and chen(state["hole"]) >= self.p["premium"]
        elif state["to_call"] > 0:
            strong = post_class(hole_ints(state), board_ints(state)) == 3
        if strong:
            n, me = state["n_seats"], state["seat"]
            order = [(me + d) % n for d in range(1, n)]
            live_behind = [j for j in order if state["active"][j] and not state["all_in"][j]]
            nxt = live_behind[0]                                         # IndexError: everyone else is all-in
            self.hand_state["next_seat"] = nxt
        return super().act(state)


FRAGILE = {
    "frag_bbround": FragBBRound,
    "frag_shortallin": FragShortAllIn,
    "frag_headsup": FragHeadsUp,
    "frag_keyerror": FragKeyError,
    "frag_zerodiv": FragZeroDiv,
    "frag_index": FragIndex,
}
FRAG_MIX = {k: 100.0 / len(FRAGILE) for k in FRAGILE}
