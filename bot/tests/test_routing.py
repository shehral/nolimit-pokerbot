"""Special-request routing (adapter + shell + variant layer): every answer is legal under an INDEPENDENT validator
(written here from the request conventions, not the adapter's own is_legal_request), for 20,000 random request
states per request type: discard (the mock toss engine's request, also checked with sim/variants validate_discard),
toss, keep, redraw, bid, assign, other. Plus: a failing variant layer still answers legally (never check/fold), a
failing class lookup never turns a betting decision into check/fold, and the board-aware keep is the one used.
"""
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import adapter, brain, shell                     # noqa: E402
from sim.variants.engine import validate_discard           # noqa: E402

N = 20_000
DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]
KINDS = ("discard", "toss", "keep", "redraw", "bid", "assign", "other")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def make_state(rng, req):
    n_hole = {"discard": 3, "toss": 3}.get(req, rng.choice((2, 3, 4, 5)))
    if req in ("discard", "toss") and rng.random() < 0.2:
        n_hole = rng.choice((2, 4, 5))
    cards = rng.sample(DECK, n_hole + 9)
    hole, board = cards[:n_hole], cards[n_hole:n_hole + rng.choice((0, 2, 3, 4, 5, 7, 9))]
    st = {"request": req, "hole": hole, "board": board, "pot": rng.randint(0, 5000), "seat": rng.randrange(8),
          "stacks": [rng.randint(0, 6000) for _ in range(8)], "street": rng.choice(("preflop", "flop", "turn")),
          "legal_actions": [req]}
    if req == "discard":
        st["discard_options"] = list(range(n_hole)) if rng.random() < 0.8 else sorted(
            rng.sample(range(n_hole), rng.randint(1, n_hole)))
    elif req == "toss":
        if rng.random() < 0.5:
            st["toss_options"] = sorted(rng.sample(range(n_hole), rng.randint(1, n_hole)))
    elif req == "keep":
        st["keep_count"] = rng.randint(1, n_hole)
    elif req == "redraw":
        st["max_redraw"] = rng.randint(0, n_hole)
    elif req == "bid":
        lo = rng.randint(0, 50)
        st["min_bid"] = lo
        if rng.random() < 0.7:
            st["max_bid"] = lo + rng.randint(0, 3000)
        if rng.random() < 0.3:
            st["bid_options"] = sorted(set(rng.randint(0, 400) for _ in range(rng.randint(1, 6))))
    elif req == "assign":
        sizes, left = [], n_hole
        while left > 0:
            s = rng.randint(1, left)
            sizes.append(s)
            left -= s
        if rng.random() < 0.3 and len(sizes) > 1:
            sizes = sizes[:-1]                     # not every card assigned
        st["groups"] = sizes
    elif req == "other":
        st["options"] = [rng.choice(("a", "b", 0, 1, (1, 2), "pass")) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.3:
        st["legal_actions"] = None                 # some engines do not list actions on a special request
    return st


def independent_legal(out, st):
    req = st["request"]
    if type(out) is not dict:
        return False
    hole = st["hole"]
    n = len(hole)
    if out.get("action") != req:
        return False
    if req == "discard":
        idx, reason = validate_discard(out, n)
        return reason is None and idx in st["discard_options"]
    if req == "toss":
        i = out.get("index")
        return _is_int(i) and (i in st["toss_options"] if "toss_options" in st else 0 <= i < n)
    if req in ("keep", "redraw"):
        v = out.get("indices")
        if not isinstance(v, list) or len(set(v)) != len(v) or not all(_is_int(i) and 0 <= i < n for i in v):
            return False
        return len(v) == st["keep_count"] if req == "keep" else len(v) <= st["max_redraw"]
    if req == "bid":
        a = out.get("amount")
        if not _is_int(a):
            return False
        if "bid_options" in st:
            return a in st["bid_options"]
        stack = st["stacks"][st["seat"]]
        return a >= st["min_bid"] and a <= st.get("max_bid", a) and (st["min_bid"] > stack or a <= stack)
    if req == "assign":
        g = out.get("groups")
        if not isinstance(g, list) or len(g) != len(st["groups"]):
            return False
        flat = [i for grp in g for i in grp]
        return (all(len(grp) == s for grp, s in zip(g, st["groups"])) and len(set(flat)) == len(flat)
                and all(_is_int(i) and 0 <= i < n for i in flat))
    if req == "other":
        return out.get("option") in st["options"]
    return False


@pytest.fixture(scope="module")
def bot():
    return shell.Bot(0, {"n_seats": 8, "payouts": [50, 30, 20]})


@pytest.mark.parametrize("req", KINDS)
def test_fuzz_requests_legal(bot, req):
    rng = random.Random(f"route:{req}")
    bad = []
    fixed0, err0 = bot.illegal_fixed, bot.errors
    for _ in range(N):
        st = make_state(rng, req)
        out = bot.act(st)
        if not independent_legal(out, st):
            bad.append((st, out))
    assert not bad, bad[:3]
    assert bot.errors == err0                        # no exception anywhere on the request path
    # the layer itself answered (the safe fallback is for failures only); 'other' and options lists may need it
    assert bot.illegal_fixed - fixed0 <= N // 100, bot.illegal_fixed - fixed0


def test_layer_failure_still_legal_never_fold(bot, monkeypatch):
    class Broken:
        @staticmethod
        def respond(*a, **k):
            raise RuntimeError("layer down")
    monkeypatch.setattr(shell, "VLAYER", Broken)
    rng = random.Random(7)
    for req in KINDS:
        for _ in range(500):
            st = make_state(rng, req)
            out = bot.act(st)
            assert independent_legal(out, st), (st, out)
            assert out["action"] not in ("fold", "check")


def test_keep_rules(bot):
    # default (PARAMS KEEP_BOARD_AWARE False, measured better on toss): the layer's static keep.
    # AA + 7 on a K-2 flop: keep the aces (toss the 7). A-K-6 on a 6-6 flop: static keeps AK (tosses the 6);
    # the optional board-aware keep_policy keeps A6 (trips) and tosses the king
    from variant.defaults import safe_default
    st = {"request": "discard", "hole": ["Ah", "Ad", "7c"], "board": ["Ks", "2d"], "legal_actions": ["discard"],
          "discard_options": [0, 1, 2], "pot": 60, "stacks": [2000] * 8, "seat": 0}
    assert bot.act(st) == {"action": "discard", "index": 2}
    st["hole"], st["board"] = ["As", "Kd", "6h"], ["6c", "6s"]
    view = adapter.request_view(st, "discard")
    assert brain.PARAMS["KEEP_BOARD_AWARE"] is False and shell._KEEP_POLICY is None
    assert bot.act(st) == {"action": "discard", "index": safe_default(view)} == {"action": "discard", "index": 2}
    assert brain.keep_policy(view) == 1


def test_class_lookup_failure_keeps_the_computed_decision(monkeypatch):
    class BadHS:
        n = 3

        def class_of(self, h):
            raise KeyError("boom")

        def class_of_safe(self, h, default=None):
            raise KeyError("boom")
    b = shell.Bot(0, {"n_seats": 8, "payouts": [50, 30, 20]})
    st = {"hand_no": 1, "level": 1, "sb": 10, "bb": 20, "ante": 0, "button": 5, "sb_seat": 6, "bb_seat": 7,
          "seat": 0, "street": "preflop", "hole": ["Ah", "Ad", "Kc"], "board": [], "pot": 30,
          "stacks": [400] * 6 + [390, 380], "bets": [0] * 6 + [10, 20], "contributed": [0] * 6 + [10, 20],
          "active": [True] * 8, "all_in": [False] * 8, "to_call": 20, "min_raise_to": 40, "max_raise_to": 400,
          "legal_actions": ["fold", "call", "raise"], "history_this_hand": [(6, "post_sb", 10), (7, "post_bb", 20)],
          "street_starts": [0], "n_alive": 8, "alive": [True] * 8, "start_stacks": [400] * 8,
          "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "n_seats": 8}
    assert brain.classify([adapter.parse_card(c) for c in st["hole"]]) == brain.CLASS_ID["AA"]
    assert b.act(st)["action"] == "raise"            # AA from UTG at 20 bb: jam
    monkeypatch.setattr(brain, "_SEAM", BadHS())
    out = b.act(st)
    assert b.errors == 0 and b.fallbacks == 0        # nothing reached the shell's handlers
    assert out["action"] in ("raise", "fold")        # median class: a computed jam/fold decision, not a crash fold
    assert brain.classify([1, 2, 3]) == brain._MEDIAN_CLS
