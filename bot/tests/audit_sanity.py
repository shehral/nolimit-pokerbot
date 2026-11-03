"""Audit 2026-11-03: poker-sanity spot checks of our bot on real engine states. Run: .venv/bin/python bot/tests/audit_sanity.py"""
import os, sys
os.environ["NOLIMIT_BOT_STRICT"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from sim.kit.engine import Table, make_deck
from sim.kit.fixtures import ScriptedAgent
from bot import shell as SH

def ALLIN(s):
    return {"action": "raise", "amount": s["max_raise_to"]} if "raise" in s["legal_actions"] else {"action": "call"}
C, X, F = {"action": "call"}, {"action": "check"}, {"action": "fold"}
def R(x): return {"action": "raise", "amount": x}

def first_action(ver, n, stacks, button, our, holes, board, scripts, street="preflop", sched=((10, 20, 0),), pay="TOP3"):
    """Our seat's first action on `street` (kit action dict) plus the state it saw."""
    seen = []
    class Spy:
        def __init__(self, seat, cfg):
            self.b = SH.make(seat, cfg, version=ver)
            self.observe = self.b.observe
        def act(self, s):
            out = self.b.act(s)
            seen.append((s, out))
            return out
    deck = make_deck(n, holes, board)
    facs = []
    for s in range(n):
        if s == our:
            facs.append(Spy)
        else:
            facs.append(lambda seat, cfg, sc=scripts.get(s, []): ScriptedAgent(seat, cfg, list(sc)))
    Table(facs, pay, sched, seed="sanity", stacks=stacks, button=button, max_hands=1, deck_fn=lambda h: deck,
          time_limit_ms=None).play()
    for s, out in seen:
        if s["street"] == street:
            return out, s
    return None, None

BOARD = ["2c", "7d", "9h", "Jc", "4s"]
OPP_PF = lambda s: {"action": "call"} if "call" in s["legal_actions"] else {"action": "check"}
POTBET = lambda s: {"action": "raise", "amount": max(s["min_raise_to"], min(s["max_raise_to"], s["pot"]))}
checks = []
def chk(name, ok, out):
    checks.append((name, ok, out))

for ver in (0, 1, 2):
    # 1. AA first in UTG, 8 handed 100 bb: must raise
    out, s = first_action(ver, 8, [2000]*8, 7, 2, {2: ("As", "Ad")}, BOARD, {})
    chk(f"v{ver} AA UTG first in raises", out["action"] == "raise", out)
    # 2. 72o UTG first in deep: fold
    out, s = first_action(ver, 8, [2000]*8, 7, 2, {2: ("7s", "2d")}, BOARD, {})
    chk(f"v{ver} 72o UTG first in folds", out["action"] == "fold", out)
    # 3. AA facing a 100 bb shove from UTG (we are the button): call
    out, s = first_action(ver, 8, [2000]*8, 7, 7, {7: ("As", "Ad")}, BOARD, {2: [ALLIN], 3: [F], 4: [F], 5: [F], 6: [F]})
    chk(f"v{ver} AA calls a 100 bb shove", out["action"] == "call", out)
    # 4. 72o facing a 100 bb shove: fold
    out, s = first_action(ver, 8, [2000]*8, 7, 7, {7: ("7s", "2d")}, BOARD, {2: [ALLIN], 3: [F], 4: [F], 5: [F], 6: [F]})
    chk(f"v{ver} 72o folds to a 100 bb shove", out["action"] == "fold", out)
    # 5. BB option after limps with 72o: check, never fold
    out, s = first_action(ver, 4, [2000]*4, 3, 1, {1: ("7s", "2d")}, BOARD, {2: [C], 3: [C], 0: [C]})
    chk(f"v{ver} BB option 72o checks", out["action"] == "check", out)
    # 6. button 6 bb, folded to us, A5s: jam (all in)
    out, s = first_action(ver, 8, [2000, 2000, 2000, 2000, 2000, 2000, 2000, 120], 7, 7, {7: ("Ah", "5h")}, BOARD,
                          {2: [F], 3: [F], 4: [F], 5: [F], 6: [F]})
    chk(f"v{ver} 6 bb button A5s jams", out["action"] == "raise" and out["amount"] == s["max_raise_to"], out)
    # 7. heads-up SB (button) 3 bb with any two: K2o jams
    out, s = first_action(ver, 2, [60, 3940], 0, 0, {0: ("Kh", "2d")}, BOARD, {})
    chk(f"v{ver} HU 3 bb SB K2o jams", out["action"] == "raise" and out["amount"] == s["max_raise_to"], out)
    # 8. flop: we hold a set of nines, checked to us heads-up: bet
    out, s = first_action(ver, 2, [2000, 2000], 0, 0, {0: ("9s", "9d"), 1: ("Kd", "Qc")}, BOARD, {1: [OPP_PF, X]},
                          street="flop")
    # (we are the button/SB heads-up: preflop we act first; take our flop action)
    chk(f"v{ver} flop set of 9s bets when checked to", out is not None and out["action"] == "raise", out)
    # 9. flop: air (KQ on 2-7-9) facing a pot-size bet: fold
    out, s = first_action(ver, 2, [2000, 2000], 0, 0, {0: ("Kd", "Qh"), 1: ("9s", "9d")}, BOARD, {1: [OPP_PF, POTBET]},
                          street="flop")
    chk(f"v{ver}   (saw a bet of {s and s['to_call']} into pot {s and s['pot']})", True, out)
    chk(f"v{ver} flop KQ air folds to a pot bet", out is not None and out["action"] == "fold", out)
for name, ok, out in checks:
    print(("OK  " if ok else "FAIL"), name, out)
print(sum(1 for c in checks if c[1]), "of", len(checks), "pass")
