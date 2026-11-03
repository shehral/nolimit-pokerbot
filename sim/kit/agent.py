"""The Agent interface of the mock starter kit (modeled on a typical "provided Agent" template).

    class MyBot(Agent):
        def __init__(self, seat, config):     # one NEW instance per table (bot state resets each table, s3.8)
            super().__init__(seat, config)
        def act(self, state):                  # TIMED: must return within config["time_limit_ms"] (1 ms)
            return {"action": "call"}
        def observe(self, event):              # optional, untimed: public events (and your own private deal)

act() returns {"action": "fold" | "check" | "call" | "raise", "amount": raise_to_total}. "amount" is used only for
"raise" and is the TOTAL bet on this street after the raise (raise-to, not raise-by). All-in = raise to
state["max_raise_to"] (or "call" when the stack does not cover the call). Legal means:
  - "action" is one of state["legal_actions"] (fold only when facing a bet; check only when not; call only when
    facing a bet; raise only when listed);
  - for "raise": an int (or an integral float) with min_raise_to <= amount <= max_raise_to.
Anything else (a non-dict, a missing key, an action not listed, a bad amount), an exception, or a decision slower
than the time limit is an ERROR: the engine counts it and plays fold, or check if checking is legal (rules s3.7).
There is no silent repair: the engine never rounds, clamps or converts your action.

The helpers below build legal actions from a state; validate_action() is the exact check the engine applies.
"""
import operator

ACTIONS = ("fold", "check", "call", "raise")


class Agent:
    """Base class. Subclass it, or write any class with the same three methods (duck typing works too)."""
    _kit_base = True   # marks this abstract base so the runner never mistakes it for a bot

    def __init__(self, seat, config):
        self.seat = seat
        self.config = config

    def act(self, state):
        raise NotImplementedError("override act(state) -> {'action': ..., 'amount': ...}")

    def observe(self, event):
        return None

    observe.kit_default = True   # the engine skips observe() calls for agents that keep this default


def fold():
    return {"action": "fold"}


def check():
    return {"action": "check"}


def call():
    return {"action": "call"}


def raise_to(amount):
    """Raise (or bet) to a TOTAL of `amount` chips on this street."""
    return {"action": "raise", "amount": int(amount)}


def check_or_fold(state):
    """The engine's own fallback: check if legal, else fold."""
    return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}


def check_or_call(state):
    return {"action": "check"} if "check" in state["legal_actions"] else {"action": "call"}


def all_in(state):
    """Put the whole stack in: raise to max_raise_to when raising is legal, else call (or check)."""
    legal = state["legal_actions"]
    if "raise" in legal:
        return {"action": "raise", "amount": state["max_raise_to"]}
    if "call" in legal:
        return {"action": "call"}
    return {"action": "check"}


def clamp_raise(state, target):
    """A LEGAL raise as close as possible to `target` (raise-to total), or None if raising is not legal.
    Use this in your bot; the engine itself never clamps."""
    if "raise" not in state["legal_actions"]:
        return None
    lo, hi = state["min_raise_to"], state["max_raise_to"]
    t = int(target)
    return {"action": "raise", "amount": lo if t < lo else hi if t > hi else t}


def validate_action(action, legal_actions, min_raise_to, max_raise_to):
    """The engine's legality check. Returns (kind, amount, None) when legal, else (None, 0, reason)."""
    if not isinstance(action, dict):
        return None, 0, f"act() returned {type(action).__name__}, not a dict"
    kind = action.get("action")
    if kind not in legal_actions:
        return None, 0, f"action {kind!r} not in legal_actions {list(legal_actions)}"
    if kind != "raise":
        return kind, 0, None
    amt = action.get("amount")
    if isinstance(amt, bool):
        return None, 0, f"raise amount {amt!r} is a bool"
    try:
        amt_i = operator.index(amt)
    except TypeError:
        if isinstance(amt, float) and amt.is_integer():
            amt_i = int(amt)
        else:
            return None, 0, f"raise amount {amt!r} is not an integer"
    if amt_i < min_raise_to or amt_i > max_raise_to:
        return None, 0, f"raise to {amt_i} outside [{min_raise_to}, {max_raise_to}]"
    return "raise", amt_i, None
