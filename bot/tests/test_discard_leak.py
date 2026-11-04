"""Revealed-discard features (DISCARD_LEAK notes): LEAK_PENALTY (leak-aware keep) and USE_REVEALED (reader model on
the bettor's revealed discards). Both default OFF; both must be inert in plain hold'em."""
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import brain as B          # noqa: E402
from bot import shell as SH         # noqa: E402
from bot.adapter import parse_card as pc   # noqa: E402
from variant import defaults as VD  # noqa: E402


def test_defaults_off_and_shell_policy_unchanged():
    assert B.PARAMS["LEAK_PENALTY"] is False and B.PARAMS["USE_REVEALED"] is False
    assert B.KEEP_POLICY is None and SH._KEEP_POLICY is None


def _views(rng, n=400):
    out = []
    for i in range(n):
        cards = rng.sample(range(52), 3 + 2 + i % 3)
        fmt = ("index", "keep_indices", "discard_indices", "keep_cards", "discard_cards", "card")[i % 6]
        v = {"kind": "toss", "response": fmt, "k": 1, "hole": cards[:3], "board": cards[3:]}
        if i % 4 == 0 and fmt == "index":
            v["options"] = [0, 1, 2]
        out.append(v)
    return out


def test_zero_penalty_is_the_static_keep():
    for v in _views(random.Random(7)):
        assert VD.keep_with_penalty(v, lambda kept, out: 0.0) == VD.safe_default(v), v


def test_keep_dispatch_none_by_default_equals_static_answer():
    for v in _views(random.Random(8), 200):
        assert B.keep_dispatch(v) is None
        assert VD.respond(v, policy=B.keep_dispatch) == VD.respond(v)


def test_leak_table_and_keep_change():
    leak = [B.leak_of((r << 2,), B.PARAMS["REVEAL_EPS"]) for r in range(13)]
    assert all(x >= 0 for x in leak) and leak[0] < 0.05 and max(leak) > 0.2      # a deuce reveals little
    rng = random.Random(3)
    P = dict(B.PARAMS, LEAK_PENALTY=True, LEAK_LAMBDA=20.0)
    changed = 0
    for _ in range(3000):
        h = rng.sample(range(52), 3)
        v = {"kind": "toss", "response": "index", "k": 1, "hole": h, "board": []}
        a, s = B.keep_leak(v, P), VD.safe_default(v)
        assert a in (0, 1, 2)
        if a != s:
            changed += 1
            assert leak[h[a] >> 2] < leak[h[s] >> 2]       # the penalised keep gives up a less telling card
    assert 0.02 < changed / 3000 < 0.10                    # about 5% of keeps (a little equity for less leak)


def test_keep_leak_only_for_two_card_keeps():
    v = {"kind": "keep", "response": "keep_indices", "k": 3, "hole": [0, 5, 9, 13, 30], "board": []}
    assert B.keep_leak(v, dict(B.PARAMS, LEAK_PENALTY=True)) is None
    assert B.keep_leak({"kind": "bid", "hole": [1, 2, 3]}) is None


def test_registry_only_for_overrides():
    B._KEEP_OWNER.clear()
    base = B.Brain(0, {"n_seats": 8, "payouts": [50, 30, 20]}, version=2)
    on = B.Brain(1, {"n_seats": 8, "payouts": [50, 30, 20]}, version=2, params={"LEAK_PENALTY": True})
    assert not base.keep_override and on.keep_override
    hole = (pc("7c"), pc("7d"), pc("Kh"))
    B._KEEP_OWNER[frozenset(hole)] = on.P
    v = {"kind": "toss", "response": "index", "k": 1, "hole": list(hole), "board": [pc("2c"), pc("9d")]}
    assert B.keep_dispatch(v) == B.keep_leak(v, on.P)
    B._KEEP_OWNER.clear()
    assert B.keep_dispatch(v) is None


def test_reveal_tracker_events():
    b = B.Brain(2, {"n_seats": 8, "payouts": [50, 30, 20]}, version=2, params={"USE_REVEALED": True})
    b.observe({"type": "discard", "hand_no": 5, "seat": 3, "card": "Kh"})
    b.observe({"type": "discard", "hand_no": 5, "seat": 2, "card": "2c"})        # our own: ignored
    b.observe({"type": "discard", "hand_no": 5, "seat": 4, "card": "??"})        # unreadable: ignored
    b.observe({"type": "toss", "hand_no": 5, "seat": 6, "cards": ["10s", "4d"]})
    assert b.rev.cards == {3: [pc("Kh")], 4: [], 6: [pc("Ts"), pc("4d")]}
    b.observe({"type": "discard", "hand_no": 6, "seat": 1, "card": "As"})         # a new hand resets
    assert b.rev.cards == {1: [pc("As")]}


def test_reader_reads_a_high_toss_as_strength():
    board = [pc("7c"), pc("2d"), pc("Kh")]                 # the K was tossed by the bettor
    lik = B.reveal_lik([11], 0.1)
    assert B.reveal_delta([pc("As"), pc("7d")], board, [], lik, 1) > 0.10    # second pair: fold more
    assert abs(B.reveal_delta([pc("Ks"), pc("9d")], board, [], lik, 2)) < 0.10   # top pair: unchanged
    uniform = [1.0] * len(B._RANK_PAIRS)
    assert B.reveal_delta([pc("As"), pc("7d")], board, [], uniform, 1) == 0.0   # no information, no shift


def _nlhe_rows(params, tables=6):
    from sim.kit.engine import Table
    from sim.kit.example_agent import ExampleAgent
    from sim.kit.fixtures import AllInAgent, CheckCallAgent, RandomAgent
    rows = []

    def ours(seat, cfg):
        bot = SH.make(seat, cfg, version=2, params=dict(params, GUARD_NS=10 ** 15))
        orig = bot.act

        def act(state, _o=orig):
            a = _o(state)
            rows.append((state["hand_no"], state["seat"], state["street"], a.get("action"), a.get("amount")))
            return a
        bot.act = act
        return bot

    opps = [RandomAgent, AllInAgent, CheckCallAgent, ExampleAgent]
    for t in range(tables):
        facs = [ours if s % 2 == 0 else opps[(s + t) % 4] for s in range(8)]
        Table(facs, "TOP3", "TURBO", seed=f"leak-inert:{t}", timing="cpu", time_limit_ms=None).play()
    return rows


def test_features_inert_in_plain_holdem():
    off = _nlhe_rows({})
    on = _nlhe_rows({"LEAK_PENALTY": True, "USE_REVEALED": True})
    assert len(off) > 1000 and on == off


@pytest.mark.parametrize("params", [{"LEAK_PENALTY": True, "USE_REVEALED": True}, {}])
def test_toss_strict_no_errors(params, monkeypatch):
    import functools
    from sim.variants.runner import run_variant
    monkeypatch.setattr(SH, "STRICT", True)
    monkeypatch.setattr(SH, "_KEEP_POLICY", B.keep_dispatch)
    me = functools.partial(SH.make, version=2, params=params)
    rd = functools.partial(SH.make, version=2, params={"USE_REVEALED": True})
    r = run_variant("toss", [("me", me)] + [("rd", rd)] * 3 + [("ex", "sim.variants.agents:TossExample")] * 4, 20,
                    seed=3, workers=1, time_limit_ms=None)
    st = r.stats["me"]
    assert st.decisions > 200 and st.exceptions == 0 and st.illegal == 0
