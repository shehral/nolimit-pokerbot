"""Test helpers: scripted variant tables with rigged decks (HU unless stacks says otherwise)."""
from sim.kit.fixtures import ScriptedAgent
from sim.variants.engine import VTable, make_variant_deck

ONE_LEVEL = ((10, 20, 0),)


def R(x):
    return {"action": "raise", "amount": x}


C = {"action": "call"}
X = {"action": "check"}
F = {"action": "fold"}


def D(i):
    return {"action": "discard", "index": i}


def ALLIN(state):
    if state.get("request") == "discard":
        return {"action": "discard", "index": 0}
    if "raise" in state["legal_actions"]:
        return {"action": "raise", "amount": state["max_raise_to"]}
    return {"action": "call"}


def vtable(variant, stacks, scripts, button, holes=None, board=None, schedule=ONE_LEVEL, max_hands=1, deck=True,
           **kw):
    """Returns (table, logs). logs[seat] = states seen; agents[seat].events = events. Call table.play()."""
    n = len(stacks)
    logs = {s: [] for s in range(n)}
    agents = {}

    def make(seat, cfg):
        a = ScriptedAgent(seat, cfg, scripts.get(seat, []), logs[seat])
        agents[seat] = a
        return a

    t = VTable(variant, [make] * n, "TOP3", schedule, seed="test", stacks=stacks, button=button,
               max_hands=max_hands, record_events=True, **kw)
    if deck:
        d = make_variant_deck(t.variant, n, holes or {}, board or [])
        t.deck_fn = lambda h: d
    t.agents = agents
    t.logs = logs
    return t


def events_of(res, kind):
    return [ev for only, ev in res.events if ev["type"] == kind]
