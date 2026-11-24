"""Shared helpers for the field zoo: card parsing, hand strength, equity by Monte Carlo, position, sizing.

Pure standard library. Not an archetype itself. Every archetype subclasses ZooAgent, which gives it
  * self.rng        a random.Random seeded from config["seed"] (reproducible per table and seat)
  * self.p          the archetype's PARAMS, overridden by constructor keywords (used for jittered variants)
  * self.hands      hands seen at this table (counted from state["hand_no"])
"""
import json
import os
import random
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim.kit.agent import (Agent, all_in, call, check, check_or_call, check_or_fold, clamp_raise, fold,  # noqa: E402,F401
                           raise_to)
from sim.kit.cards import CARD_INT, evaluate  # noqa: E402

RANKS = "23456789TJQKA"
RANK_OF = {r: i for i, r in enumerate(RANKS)}


class ZooAgent(Agent):
    """Base class: seeded RNG and keyword-overridable parameters.  Subclasses set PARAMS and JITTER."""
    PARAMS = {}     # name -> default value
    JITTER = {}     # name -> (lo, hi) uniform range used by fields.jittered_variants (only listed names move)

    def __init__(self, seat, config, **params):
        super().__init__(seat, config)
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown parameters {sorted(unknown)}")
        self.p = {**self.PARAMS, **params}
        self.rng = random.Random(config.get("seed", seat))
        self.last_hand = -1
        self.hand_state = {}

    def new_hand(self, state):
        """True the first time act() is called in a hand; resets self.hand_state."""
        if state["hand_no"] != self.last_hand:
            self.last_hand = state["hand_no"]
            self.hand_state = {}
            return True
        return False


# ---------------------------------------------------------------- cards and preflop strength
def ranks_of(hole):
    a, b = RANK_OF[hole[0][0]], RANK_OF[hole[1][0]]
    return (a, b) if a >= b else (b, a)


def is_suited(hole):
    return hole[0][1] == hole[1][1]


def label(hole):
    """'AKs', 'QTo', '77' style class label."""
    hi, lo = ranks_of(hole)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if is_suited(hole) else "o")


def chen(hole):
    """Chen formula, 0..20 (AA = 20, 72o = -1). The textbook preflop score an AI assistant writes first."""
    hi, lo = ranks_of(hole)
    pts = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(hi, (hi + 2) / 2.0)
    if hi == lo:
        return float(max(5.0, 2 * pts))
    if is_suited(hole):
        pts += 2
    gap = hi - lo - 1
    pts -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and hi < 10:
        pts += 1
    return float(-(-pts // 1)) if pts % 1 else float(pts)   # round half up


def _build_pct():
    """Percentile (0 = best, 1 = worst) of each class label by Chen score, ties by high card, over 1,326 combos."""
    rows = []
    for hi in range(13):
        for lo in range(hi + 1):
            if hi == lo:
                rows.append((RANKS[hi] * 2, 6, chen((RANKS[hi] + "c", RANKS[lo] + "d")) + 0.01 * hi))
            else:
                rows.append((RANKS[hi] + RANKS[lo] + "s", 4, chen((RANKS[hi] + "c", RANKS[lo] + "c")) + 0.01 * hi + 0.001 * lo))
                rows.append((RANKS[hi] + RANKS[lo] + "o", 12, chen((RANKS[hi] + "c", RANKS[lo] + "d")) + 0.01 * hi + 0.001 * lo))
    rows.sort(key=lambda r: -r[2])
    out, acc = {}, 0
    for lab, combos, _ in rows:
        out[lab] = (acc + combos / 2.0) / 1326.0
        acc += combos
    return out


CHEN_PCT = _build_pct()


def combos(lab):
    return 6 if len(lab) == 2 else (4 if lab[2] == "s" else 12)


def _build_orders():
    path = os.path.join(os.path.dirname(__file__), "data", "orders.json")
    with open(path) as f:
        d = json.load(f)
    out = {}
    for key in ("push", "call"):
        acc, pct = 0, {}
        for lab in d[key]:
            pct[lab] = (acc + combos(lab) / 2.0) / 1326.0
            acc += combos(lab)
        out[key] = pct
    return out["push"], out["call"]


PUSH_PCT, CALL_PCT = _build_orders()   # Nash-style jam / call orderings from the research assets


def hole_ints(state):
    return [CARD_INT[c] for c in state["hole"]]


def board_ints(state):
    return [CARD_INT[c] for c in state["board"]]


# ---------------------------------------------------------------- equity by Monte Carlo
def equity_vs_random(hole, board, iters, rng):
    """Win share of `hole` (card ints) against ONE uniformly random hand: ties count half."""
    used = set(hole)
    used.update(board)
    rest = [c for c in range(52) if c not in used]
    need = 5 - len(board)
    sample = rng.sample
    win = 0.0
    for _ in range(iters):
        s = sample(rest, 2 + need)
        b = board + s[2:]
        me = evaluate(hole + b)
        op = evaluate(s[:2] + b)
        if me > op:
            win += 1.0
        elif me == op:
            win += 0.5
    return win / iters


# ---------------------------------------------------------------- postflop made-hand class
def post_class(hole, board):
    """0 air, 1 weak pair or draw, 2 top pair / overpair / trips-ish, 3 two pair or better using a hole card."""
    sc = evaluate(hole + board)
    cat = sc >> 20
    bc = evaluate(board) >> 20 if len(board) >= 3 else 0
    if cat >= 2 and cat > bc:
        return 3
    if cat == 1 and bc == 0:
        pair_rank = (sc >> 16) & 15
        return 2 if pair_rank >= max(c >> 2 for c in board) else 1
    if cat == 1 and bc == 1:
        return 0
    return 1 if has_draw(hole, board) else 0


def has_draw(hole, board):
    """Four to a flush with a hole card, or four ranks within a five-rank window with a hole card (straight draw)."""
    if len(board) >= 5:
        return False
    cards = hole + board
    for s in range(4):
        if sum(1 for c in cards if c & 3 == s) >= 4 and any(c & 3 == s for c in hole):
            return True
    mask = 0
    for c in cards:
        mask |= 1 << (c >> 2)
    mask |= (mask >> 12 & 1)   # the ace plays low
    hmask = 0
    for c in hole:
        hmask |= 1 << (c >> 2)
    for lo in range(0, 9):
        window = 31 << lo
        if bin(mask & window).count("1") >= 4 and (hmask & window):
            return True
    return False


# ---------------------------------------------------------------- position and betting helpers
def preflop_behind(state):
    """Seats still to act after us preflop that are in the hand and have chips (the 'players behind')."""
    n = state["n_seats"]
    me = state["seat"]
    order = [(state["bb_seat"] + 1 + i) % n for i in range(n)]
    pos = order.index(me)
    return [j for j in order[pos + 1:] if state["active"][j] and not state["all_in"][j] and state["stacks"][j] > 0]


def top_bet(state):
    return max(state["bets"])


def raises_this_street(state):
    start = state["street_starts"][["preflop", "flop", "turn", "river"].index(state["street"])]
    return sum(1 for (_s, a, _x) in state["history_this_hand"][start:] if a == "raise")


def my_total_stack(state):
    """Chips behind plus chips already bet this street."""
    s = state["seat"]
    return state["stacks"][s] + state["bets"][s]


def eff_stack_bb(state):
    """Effective stack in big blinds: mine against the biggest live opponent."""
    me = state["seat"]
    opp = [state["stacks"][j] + state["bets"][j] for j in range(state["n_seats"])
           if j != me and state["active"][j]]
    if not opp:
        return my_total_stack(state) / state["bb"]
    return min(my_total_stack(state), max(opp)) / state["bb"]


def raise_amt(state, target):
    """Legal raise-to as close to target as possible, else call/check/fold fallbacks (never illegal)."""
    r = clamp_raise(state, target)
    if r is not None:
        return r
    legal = state["legal_actions"]
    if "call" in legal:
        return call()
    return check() if "check" in legal else fold()


def check_else(state, other):
    return check() if "check" in state["legal_actions"] else other


def call_or_check(state):
    legal = state["legal_actions"]
    return call() if "call" in legal else (check() if "check" in legal else fold())


def fold_or_check(state):
    return check() if "check" in state["legal_actions"] else fold()
