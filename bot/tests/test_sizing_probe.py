"""SIZING_PROBE (FRAGILITY notes): amounts only, inside the legal bounds, default OFF; and the fragile zoo's bugs fire
on their documented triggers (sim/zoo_fragile/bots.py)."""
import os
import random
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import adapter as AD                     # noqa: E402
from bot import brain as B                        # noqa: E402
from bot import shell as SH                       # noqa: E402
from sim.kit.agent import validate_action         # noqa: E402
from sim.kit.engine import Table                  # noqa: E402
from sim.zoo_fragile import bots as FB            # noqa: E402

CFG = {"n_seats": 8, "payouts": [50, 30, 20], "seed": 3}


def test_default_off():
    assert B.PARAMS["SIZING_PROBE"] is False
    assert B.Brain(0, CFG).probe is False
    assert B.Brain(0, CFG, params={"SIZING_PROBE": True}).probe is True


def test_probe_size_stays_legal_and_below_the_snap():
    br = B.Brain(0, CFG, params={"SIZING_PROBE": True})
    snap = br.P["ALLIN_SNAP"]
    rng = random.Random(5)
    moved = 0
    for _ in range(20000):
        bb = rng.choice((20, 30, 50, 100, 150, 200, 300, 400, 25, 12))
        lo = rng.randint(bb, 30 * bb)
        hi = lo + rng.randint(0, 60 * bb)
        t = rng.randint(lo // 2, hi + bb)
        v = SimpleNamespace(min_to=lo, max_to=hi, bb=bb)
        out = br.probe_size(v, t)
        if t >= snap * hi or t >= hi:
            assert out == t                          # all-in territory: untouched
            continue
        if out == t:
            continue
        moved += 1
        assert lo <= out <= hi and out < snap * hi
        assert out >= t
        assert out - max(t, lo) <= max(4, int(br.P["PROBE_BAND"] * max(t, lo)))
        assert out % 2 == 1 and out % 5 != 0 and out % bb != 0
    assert moved > 5000


def test_probe_min_style_returns_the_exact_minimum():
    br = B.Brain(0, CFG, params={"SIZING_PROBE": True, "PROBE_STYLE": "min"})
    v = SimpleNamespace(min_to=80, max_to=2000, bb=20)
    assert br.probe_size(v, 82) == 80
    assert br.probe_size(v, 400) not in (400,) and br.probe_size(v, 400) % 2 == 1


def test_raising_not_legal_is_untouched():
    br = B.Brain(0, CFG, params={"SIZING_PROBE": True})
    assert br.probe_size(SimpleNamespace(min_to=0, max_to=0, bb=20), 120) == 120


class _Twin:
    """Our bot with the probe ON; at every decision it also asks the same brain with the probe OFF (same view) and
    checks the action type is identical and the probe's output legal."""

    def __init__(self, seat, config):
        self.bot = SH.make(seat, config, version=2, params={"SIZING_PROBE": True, "GUARD_NS": 10 ** 15})
        self.checked = 0
        self.moved = 0
        ob = getattr(self.bot, "observe", None)
        if ob is not None:
            self.observe = ob

    def act(self, state):
        br = self.bot._brain
        view = AD.normalize(state)
        br.probe = False
        d_off = br.decide(view, 0)
        br.probe = True
        d_on = br.decide(view, 0)
        assert d_on[0] == d_off[0], (d_off, d_on)
        a_off = AD.to_action(d_off, state, B.PARAMS["ALLIN_SNAP"])
        a_on = AD.to_action(d_on, state, B.PARAMS["ALLIN_SNAP"])
        assert a_on["action"] == a_off["action"]
        if a_on["action"] == "raise":
            assert (a_on["amount"] == state["max_raise_to"]) == (a_off["amount"] == state["max_raise_to"])
            self.moved += a_on["amount"] != a_off["amount"]
        assert validate_action(a_on, state["legal_actions"], state["min_raise_to"], state["max_raise_to"])[0]
        self.checked += 1
        out = self.bot.act(state)
        assert out == a_on or not AD.is_legal(a_on, state)
        return out


def test_probe_changes_amounts_only_in_real_play():
    from sim.zoo.chart_tag import ChartTag
    from sim.zoo.icm_pro import IcmPro
    made = []

    def twin(seat, config):
        t = _Twin(seat, config)
        made.append(t)
        return t
    for seed in range(6):
        res = Table([twin, ChartTag, IcmPro, ChartTag, IcmPro, FB.FragKeyError, FB.FragBBRound, twin],
                    seed=f"probe-test:{seed}", time_limit_ms=None).play()
        for s in (0, 7):
            assert res.stats[s].exceptions == 0 and res.stats[s].illegal == 0
    assert sum(t.checked for t in made) > 300
    assert sum(t.moved for t in made) > 20


# ------------------------------------------------------------------ fragile zoo: each bug fires on its trigger
def _state(**kw):
    n = 8
    s = {"hand_no": 1, "level": 1, "hands_to_next_level": 25, "hands_left": 199, "sb": 10, "bb": 20, "ante": 0,
         "button": 7, "sb_seat": 0, "bb_seat": 1, "seat": 4, "street": "preflop", "hole": ["As", "Ah"], "board": [],
         "pot": 30, "stacks": [1990, 1980, 2000, 2000, 2000, 2000, 2000, 2000], "bets": [10, 20, 0, 0, 0, 0, 0, 0],
         "contributed": [10, 20, 0, 0, 0, 0, 0, 0], "active": [True] * n, "all_in": [False] * n, "to_call": 20,
         "min_raise_to": 40, "max_raise_to": 2000, "legal_actions": ["fold", "call", "raise"],
         "history_this_hand": [(0, "post_sb", 10), (1, "post_bb", 20)], "street_starts": [0], "n_alive": n,
         "alive": [True] * n, "start_stacks": [2000] * n, "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "n_seats": n,
         "time_limit_ms": 1.0}
    s.update(kw)
    return s


def _agent(cls):
    return cls(4, {"seed": 1, "n_seats": 8})


def _legal(state, out):
    return validate_action(out, state["legal_actions"], state["min_raise_to"], state["max_raise_to"])[0] is not None


def test_frag_keyerror_on_an_odd_open_only():
    st = _state(seat=4, hole=["7c", "2d"], bets=[10, 20, 51, 0, 0, 0, 0, 0], stacks=[1990, 1980, 1949] + [2000] * 5,
                pot=81, to_call=51, min_raise_to=82,
                history_this_hand=[(0, "post_sb", 10), (1, "post_bb", 20), (2, "raise", 51)])
    with pytest.raises(KeyError):
        _agent(FB.FragKeyError).act(st)
    st2 = dict(st, bets=[10, 20, 50, 0, 0, 0, 0, 0], to_call=50, min_raise_to=80,
               history_this_hand=[(0, "post_sb", 10), (1, "post_bb", 20), (2, "raise", 50)])
    assert _legal(st2, _agent(FB.FragKeyError).act(st2))


def test_frag_bbround_illegal_on_a_short_odd_allin():
    st = _state(seat=4, hole=["As", "Ah"], stacks=[1990, 1980, 2000, 2000, 37, 2000, 2000, 2000],
                min_raise_to=37, max_raise_to=37)
    out = _agent(FB.FragBBRound).act(st)
    assert out == {"action": "raise", "amount": 20} and not _legal(st, out)


def test_frag_shortallin_on_a_call_for_less():
    st = _state(seat=4, bets=[10, 20, 100, 70, 0, 0, 0, 0], stacks=[1990, 1980, 1900, 0] + [2000] * 4,
                all_in=[False, False, False, True] + [False] * 4, pot=200, to_call=100, min_raise_to=180,
                history_this_hand=[(0, "post_sb", 10), (1, "post_bb", 20), (2, "raise", 100), (3, "call", 70)])
    with pytest.raises(ValueError):
        _agent(FB.FragShortAllIn).act(st)


def test_frag_shortallin_illegal_after_an_incomplete_allin_raise():
    # flop: seat 2 bets 100, seat 3 all-in raises to 150 (incomplete: +50 < 100); the engine's minimum stays 250
    st = _state(seat=4, street="flop", board=["Ks", "Kd", "7c"], hole=["Kh", "7d"], pot=650,
                bets=[0, 0, 100, 150, 0, 0, 0, 0], stacks=[1900, 1900, 1800, 0, 1900, 1900, 1900, 1900],
                all_in=[False, False, False, True] + [False] * 4, to_call=150, min_raise_to=250, max_raise_to=2050,
                history_this_hand=[(0, "post_sb", 10), (1, "post_bb", 20), (2, "raise", 100), (3, "raise", 150)],
                street_starts=[0, 2])
    out = _agent(FB.FragShortAllIn).act(st)                 # min-raises its full house: to 150 + 50 = 200
    assert out == {"action": "raise", "amount": 200} and not _legal(st, out)
    st2 = dict(st, bets=[0, 0, 100, 200, 0, 0, 0, 0], to_call=200, min_raise_to=300,
               history_this_hand=[(0, "post_sb", 10), (1, "post_bb", 20), (2, "raise", 100), (3, "raise", 200)])
    assert _legal(st2, _agent(FB.FragShortAllIn).act(st2))  # after a full raise its arithmetic is right


def test_frag_headsup_checks_facing_the_blind():
    st = _state(seat=7, button=7, sb_seat=7, bb_seat=1, hole=["9c", "4d"], n_alive=2,
                alive=[False, True, False, False, False, False, False, True], bets=[0, 20, 0, 0, 0, 0, 0, 10],
                stacks=[0, 3980, 0, 0, 0, 0, 0, 3990], to_call=10, min_raise_to=40, max_raise_to=4000,
                active=[False, True, False, False, False, False, False, True],
                history_this_hand=[(7, "post_sb", 10), (1, "post_bb", 20)])
    out = _agent(FB.FragHeadsUp).act(st)
    assert out == {"action": "check"} and not _legal(st, out)


def test_frag_zerodiv_facing_an_allin_bettor():
    st = _state(seat=4, street="flop", board=["Ks", "8d", "7c"], hole=["Kh", "Qd"], pot=900,
                bets=[0, 0, 400, 0, 0, 0, 0, 0], stacks=[1900, 1900, 0, 0, 1600, 1900, 1900, 1900],
                all_in=[False, False, True] + [False] * 5, to_call=400, min_raise_to=0, max_raise_to=0,
                legal_actions=["fold", "call"], street_starts=[0, 2])
    with pytest.raises(ZeroDivisionError):
        _agent(FB.FragZeroDiv).act(st)


def test_frag_index_premium_facing_a_lone_jam():
    st = _state(seat=4, hole=["As", "Ad"], bets=[10, 20, 600, 0, 0, 0, 0, 0], stacks=[1990, 1980, 0] + [2000] * 5,
                active=[True, True, True, False, True, False, False, False],
                all_in=[False, False, True] + [False] * 5, to_call=600, min_raise_to=0, max_raise_to=0,
                legal_actions=["fold", "call"],
                history_this_hand=[(0, "post_sb", 10), (1, "post_bb", 20), (2, "raise", 600)])
    st["active"][0] = st["active"][1] = False
    with pytest.raises(IndexError):
        _agent(FB.FragIndex).act(st)
