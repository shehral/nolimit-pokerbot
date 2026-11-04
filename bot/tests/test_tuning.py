"""Tuning rules added 2026-11-03 (TUNING_LOG notes): payout-chosen tuning (TUNE_BY_PAYOUT / TUNE_MULTI), the
class-2 bet-to-pot call limit (CALL2_MAX_BET_POT, pot_before_street) and the caller jam-cap gate
(CALLER_CAP_JAM_EVIDENCE). Each rule is checked with the switch on and off, so the tests hold whatever the shipped
defaults are."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.runner import load_factory                          # noqa: E402

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
B = SHELL.brain
P = B.PARAMS
TOP3 = [50, 30, 20, 0, 0, 0, 0, 0]
LINEAR = [100 * x / 36 for x in (8, 7, 6, 5, 4, 3, 2, 1)]
WTA = [100, 0, 0, 0, 0, 0, 0, 0]


def _brain(payouts=TOP3, params=None, config_params=None, version=2):
    cfg = {"n_seats": 8, "payouts": list(payouts) if payouts is not None else None, "seed": 1}
    if config_params is not None:
        cfg["bot_params"] = dict(config_params)
    return B.Brain(0, cfg, version=version, params=params)


# ------------------------------------------------------------------ TUNE_BY_PAYOUT
def test_tune_by_payout_applies_only_when_several_places_pay():
    for pays in (TOP3, LINEAR, [40, 30, 20, 10, 0, 0, 0, 0], [65, 35]):
        b = _brain(pays, params={"TUNE_BY_PAYOUT": True})
        assert b.P["PF_DEPTH_BB"] == 12.0 and b.P["JAM_MULT"] == 1.15, pays
        assert b.tuned == {"PF_DEPTH_BB": 12.0, "JAM_MULT": 1.15}
    for pays in (WTA, [100], [], None):
        b = _brain(pays, params={"TUNE_BY_PAYOUT": True})
        assert b.P["PF_DEPTH_BB"] == 20.0 and b.P["JAM_MULT"] == 1.0, pays
        assert b.tuned == {}


def test_tune_by_payout_off_keeps_the_built_values():
    b = _brain(TOP3, params={"TUNE_BY_PAYOUT": False})
    assert b.P["PF_DEPTH_BB"] == 20.0 and b.P["JAM_MULT"] == 1.0 and b.tuned == {}


def test_explicit_params_beat_the_payout_tuning():
    b = _brain(TOP3, params={"TUNE_BY_PAYOUT": True, "JAM_MULT": 1.3})
    assert b.P["JAM_MULT"] == 1.3 and b.P["PF_DEPTH_BB"] == 12.0
    b = _brain(TOP3, config_params={"TUNE_BY_PAYOUT": True, "PF_DEPTH_BB": 15.0})
    assert b.P["PF_DEPTH_BB"] == 15.0 and b.P["JAM_MULT"] == 1.15
    if P["TUNE_BY_PAYOUT"]:                               # the shipped default: an explicit key still wins
        b = _brain(TOP3, params={"PF_DEPTH_BB": 20.0})
        assert b.P["PF_DEPTH_BB"] == 20.0 and b.P["JAM_MULT"] == 1.15


def test_module_params_are_not_mutated_by_tuning():
    before = dict(P)
    _brain(TOP3, params={"TUNE_BY_PAYOUT": True})
    assert P == before


# ------------------------------------------------------------------ class-2 bet-to-pot limit
def _view(bet, base=100, stack=1000, pot_includes_bets=True):
    """Heads-up flop, we hold top pair top kicker (class 2) and face a first bet of `bet` into `base`."""
    v = SHELL.adapter.View()
    pc = SHELL.adapter.parse_card
    v.hand_no, v.street, v.me, v.n = 1, 1, 0, 2
    v.hole = (pc("Ah"), pc("Kd"))
    v.board = [pc("Ac"), pc("7s"), pc("2d")]
    v.boards = None
    v.bounty = None
    v.sb, v.bb, v.ante = 10, 20, 0
    v.bets = [0, bet]
    v.contrib = [base // 2, base // 2 + bet]
    v.pot = base + bet if pot_includes_bets else base
    v.stacks = [stack, stack - bet]
    v.start = [stack + base // 2, stack + base // 2]
    v.in_hand = [True, True]
    v.all_in = [False, False]
    v.alive = [True, True]
    v.to_call = bet
    v.min_to, v.max_to = 2 * bet, stack
    v.legal = ["fold", "call", "raise"]
    return v


def test_pot_before_street():
    assert B.pot_before_street(_view(150)) == 100                       # kit: pot includes the bet
    assert B.pot_before_street(_view(150, pot_includes_bets=False)) == 100   # a platform pot without it
    v = _view(150)
    v.pot, v.contrib = 150, [0, 150]
    assert B.pot_before_street(v) == v.bb                                # floored at 1 bb


def test_class2_calls_small_bets_and_folds_big_ones_with_the_limit():
    b = _brain(WTA, params={"CALL2_MAX_BET_POT": 1.0})
    assert B.hand_class(*_view(10).hole, _view(10).board) == 2
    for bet in (10, 50, 100):
        assert b.postflop(_view(bet)) == ("call",), bet
    for bet in (101, 150, 300):
        assert b.postflop(_view(bet)) == ("fold",), bet
    b = _brain(WTA, params={"CALL2_MAX_BET_POT": 1.0})
    assert b.postflop(_view(150, pot_includes_bets=False)) == ("fold",)


def test_class2_without_the_limit_calls_any_non_big_bet():
    b = _brain(WTA, params={"CALL2_MAX_BET_POT": None})
    for bet in (10, 100, 150, 300, 399):
        assert b.postflop(_view(bet)) == ("call",), bet


# ------------------------------------------------------------------ caller jam-cap gate
def _postflop_caller(b, j):
    st = b.stats[j]
    st.pf_dec, st.pf_vol, st.vol_ever = 20, 4, 4             # tight preflop (shrunk vol 0.23, not loose)
    st.post_faced, st.post_call = 10, 9                       # calls postflop bets 90%: typed caller
    b.types = {}
    assert b.seat_type(j) == "caller"


def test_caller_typed_from_postflop_only_caps_jams_only_without_the_gate():
    b = _brain(params={"CALLER_CAP_JAM_EVIDENCE": False})
    _postflop_caller(b, 1)
    assert b.typed_jam(0.6, [1], 10.0) == P["LIGHT_JAM_CAP"]
    b = _brain(params={"CALLER_CAP_JAM_EVIDENCE": True})
    _postflop_caller(b, 1)
    assert b.typed_jam(0.6, [1], 10.0) == 0.6


def test_caller_with_jam_call_evidence_still_caps_with_the_gate():
    b = _brain(params={"CALLER_CAP_JAM_EVIDENCE": True})
    st = b.stats[1]
    st.pf_dec, st.pf_vol, st.vol_ever, st.jam_faced, st.jam_called = 20, 4, 4, 6, 5
    b.types = {}
    assert b.seat_type(1) == "caller"
    assert b.typed_jam(0.6, [1], 10.0) == P["LIGHT_JAM_CAP"]


def test_loose_postflop_caller_still_caps_with_the_gate():
    b = _brain(params={"CALLER_CAP_JAM_EVIDENCE": True})
    _postflop_caller(b, 1)
    st = b.stats[1]
    st.pf_vol = st.vol_ever = 14                              # 70% voluntary: would be loose
    b.types = {}
    assert b.seat_type(1) == "caller"
    assert b.typed_jam(0.6, [1], 10.0) == P["LIGHT_JAM_CAP"]
