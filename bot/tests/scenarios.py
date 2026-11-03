"""Scenario harness for the bot tests: real engine states from the mock kit (rigged decks, scripted opponents).

    decisions = run_spot(spot, version)       # our seat's first kit action for every one of the 169 classes
    frac = action_share(decisions, "call")    # combo-weighted share of hands taking an action

A spot is a dict: n, stacks, button, our seat, scripts {seat: [actions]}, schedule, payouts (default TOP3).
Used by test_ranges.py and for manual exploration (python -m bot.tests.scenarios).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import Table, make_deck            # noqa: E402
from sim.kit.fixtures import ScriptedAgent              # noqa: E402
from sim.kit.runner import load_factory                 # noqa: E402

RANKS = "23456789TJQKA"


def class_cards(label):
    """Two concrete cards for a class label: 'AKs' -> ('Ah', 'Kh'), 'AKo' -> ('Ah', 'Kd'), 'TT' -> ('Th', 'Td')."""
    if len(label) == 2:
        return (label[0] + "h", label[1] + "d")
    if label[2] == "s":
        return (label[0] + "h", label[1] + "h")
    return (label[0] + "h", label[1] + "d")


def all_labels():
    out = []
    for i in range(12, -1, -1):
        for j in range(i, -1, -1):
            if i == j:
                out.append(RANKS[i] * 2)
            else:
                out.append(RANKS[i] + RANKS[j] + "s")
                out.append(RANKS[i] + RANKS[j] + "o")
    return out


def combos(label):
    return 6 if len(label) == 2 else (4 if label[2] == "s" else 12)


def ALLIN(state):
    if "raise" in state["legal_actions"]:
        return {"action": "raise", "amount": state["max_raise_to"]}
    return {"action": "call"}


def RAISE(x):
    return lambda state: {"action": "raise", "amount": max(state["min_raise_to"], min(int(x), state["max_raise_to"]))}


FOLD = {"action": "fold"}
CALL = {"action": "call"}
CHECK = {"action": "check"}


def run_spot(spot, version=2, labels=None, factory=None):
    """Our seat's first action (kit dict) for each class label, in a fresh table per label."""
    n = spot["n"]
    me = spot["me"]
    fac = factory or load_factory(f"bot/shell.py:BotV{version}")
    out = {}
    taken = set()
    filler = []
    for lab in all_labels():
        pass
    for lab in (labels or all_labels()):
        mine = class_cards(lab)
        holes = {me: mine}
        used = set(mine)
        # opponents get fixed low cards that avoid ours (their cards do not affect our first decision)
        pool = [r + s for r in "23456789TJQKA" for s in "cdhs" if r + s not in used]
        k = 0
        for s in range(n):
            if s != me:
                holes[s] = (pool[k], pool[k + 1])
                k += 2
        deck = make_deck(n, holes, [])
        seen = []

        def make(seat, cfg, _seen=seen):
            if seat == me:
                bot = fac(seat, cfg)
                orig = bot.act

                def act(state, _o=orig):
                    a = _o(state)
                    _seen.append((state, a))
                    return a
                bot.act = act
                return bot
            return ScriptedAgent(seat, cfg, list(spot["scripts"].get(seat, [])))
        t = Table([make] * n, spot.get("payouts", "TOP3"), spot.get("schedule", ((10, 20, 0),)), seed="spot",
                  stacks=spot["stacks"], button=spot["button"], max_hands=1, deck_fn=lambda h, d=deck: d,
                  timing="cpu")
        t.play()
        out[lab] = seen[0] if seen else (None, None)
    return out


def action_share(decisions, kind):
    tot = 0
    hit = 0
    for lab, (state, a) in decisions.items():
        c = combos(lab)
        tot += c
        if a is not None and a.get("action") == kind:
            if kind == "raise" or True:
                hit += c
    return hit / tot if tot else 0.0


def allin_share(decisions):
    tot = hit = 0
    for lab, (state, a) in decisions.items():
        c = combos(lab)
        tot += c
        if a is not None and a.get("action") == "raise" and a.get("amount") == state["max_raise_to"]:
            hit += c
    return hit / tot if tot else 0.0


def range_text(decisions, pred):
    return " ".join(lab for lab in all_labels() if lab in decisions and decisions[lab][1] is not None
                    and pred(decisions[lab][0], decisions[lab][1]))


# ------------------------------------------------------------------ common spots
def hu_sb_jam(bb_stack_bb=10, version=2):
    """Heads-up, 10/20 blinds: SB (button, seat 0) jams; we are the BB (seat 1)."""
    s = bb_stack_bb * 20
    return {"n": 2, "stacks": [s, s], "button": 0, "me": 1, "scripts": {0: [ALLIN]}}


def eight_first_in(me_k, stack_bb=10):
    """8 seats, button 7 (SB 0, BB 1, UTG 2): everyone before us folds; we are first in with me_k players behind."""
    n = 8
    pre = [2, 3, 4, 5, 6, 7, 0, 1]
    me = pre[len(pre) - 1 - me_k]
    scripts = {s: [FOLD] for s in pre[:pre.index(me)]}
    return {"n": n, "stacks": [stack_bb * 20] * n, "button": 7, "me": me, "scripts": scripts}


def eight_bb_vs_jam(jammer_k, stack_bb=10):
    """8 seats: the seat with jammer_k players behind jams first-in, everyone else folds; we are the BB."""
    n = 8
    pre = [2, 3, 4, 5, 6, 7, 0, 1]
    jammer = pre[len(pre) - 1 - jammer_k]
    scripts = {s: [FOLD] for s in pre if s not in (jammer, 1)}
    scripts[jammer] = [ALLIN]
    return {"n": n, "stacks": [stack_bb * 20] * n, "button": 7, "me": 1, "scripts": scripts}


if __name__ == "__main__":
    for ver in (0, 1, 2):
        d = run_spot(hu_sb_jam(10), ver)
        print(f"v{ver} HU BB vs SB jam 10bb: call {action_share(d, 'call'):.3f}")
