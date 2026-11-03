"""Shared test helpers: scripted tables with rigged decks."""
from sim.kit.engine import Table, make_deck
from sim.kit.fixtures import ScriptedAgent

ONE_LEVEL = ((10, 20, 0),)


def R(x):
    return {"action": "raise", "amount": x}


C = {"action": "call"}
X = {"action": "check"}
F = {"action": "fold"}


def ALLIN(state):
    """All-in: raise to max when raising is legal, else call (all-in for less or covering)."""
    if "raise" in state["legal_actions"]:
        return {"action": "raise", "amount": state["max_raise_to"]}
    return {"action": "call"}


def scripted_table(stacks, scripts, button, holes=None, board=None, schedule=ONE_LEVEL, max_hands=1, **kw):
    """Play `max_hands` hands with ScriptedAgents. scripts = {seat: [action or callable, ...]}.
    Returns (TableResult, logs) where logs[seat] is the list of states that seat saw."""
    n = len(stacks)
    logs = {s: [] for s in range(n)}
    deck = make_deck(n, holes or {}, board or [])

    def make(seat, cfg):
        return ScriptedAgent(seat, cfg, scripts.get(seat, []), logs[seat])

    t = Table([make] * n, "TOP3", schedule, seed="test", stacks=stacks, button=button, max_hands=max_hands,
              deck_fn=lambda h: deck, record_events=True, **kw)
    res = t.play()
    return res, logs


def events_of(res, kind):
    return [ev for only, ev in res.events if ev["type"] == kind]
