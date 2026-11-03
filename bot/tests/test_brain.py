"""Unit tests of the brain: hand classes (hand-checked), exact ICM vs the kit's brute force, BF values, tables,
the hand tracker (observe() and history feeds agree), and the review fixes per version (P1-P7)."""
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.icm import icm_brute                                # noqa: E402
from sim.kit.runner import load_factory                          # noqa: E402

from bot.tests.scenarios import (ALLIN, FOLD, CALL, RAISE, action_share, allin_share, eight_bb_vs_jam,  # noqa: E402
                                 eight_first_in, hu_sb_jam, run_spot)

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
B = SHELL.brain
P = B.PARAMS


def cards(*names):
    return [SHELL.adapter.parse_card(c) for c in names]


# ------------------------------------------------------------------ hand classes (10+ hand-checked cases)
@pytest.mark.parametrize("hole,board,want", [
    (("Ah", "Kd"), ("Ac", "7s", "2d"), 2),          # top pair, top kicker
    (("Qh", "Qd"), ("Jc", "7s", "2d"), 2),          # overpair
    (("9h", "9d"), ("Jc", "7s", "2d"), 1),          # underpair to the top card
    (("Kh", "7d"), ("Ac", "7s", "2d"), 1),          # second pair
    (("7h", "7d"), ("Ac", "7s", "2d"), 3),          # set
    (("Ah", "7d"), ("Ac", "7s", "2d"), 3),          # two pair with both hole cards
    (("3h", "3d"), ("Kc", "Ks", "5d"), 1),          # small pocket pair on a paired board: weak
    (("Ah", "Ad"), ("Kc", "Ks", "5d"), 2),          # overpair on a paired board
    (("Kh", "5d"), ("Kc", "Ks", "5d"), 3),          # full house with a hole card
    (("Ah", "Qh"), ("Kh", "7h", "2d"), 1),          # nut flush draw (no pair)
    (("9h", "8d"), ("7c", "6s", "2d"), 1),          # open-ended straight draw
    (("9h", "8d"), ("7c", "6s", "5d"), 3),          # straight with hole cards
    (("Ah", "Kd"), ("Qc", "7s", "2d"), 0),          # air (overcards)
    (("Ah", "Kd"), ("Qc", "Qs", "2d", "2h", "7c"), 0),   # board two pair, we play the board
    (("2h", "3h"), ("Ah", "Kh", "Qh", "Jd"), 3),    # flush with both hole cards (three-flush board)
    (("2h", "3d"), ("Ah", "Kh", "Qh", "9h"), 1),    # low flush card on a four-flush board: weak
    (("Ah", "3d"), ("Kh", "Th", "7h", "4h", "2c"), 3),   # nut flush card on a four-flush board
    (("Ah", "Kd"), ("Qc", "Js", "Td"), 3),          # broadway straight
    (("5h", "4d"), ("Ac", "2s", "3d"), 3),          # wheel
    (("9h", "8d"), ("7c", "6s", "2d", "Kh", "Qc"), 0),   # river: a missed draw is air
])
def test_hand_class(hole, board, want):
    h = cards(*hole)
    assert B.hand_class(h[0], h[1], cards(*board)) == want


# ------------------------------------------------------------------ ICM and bubble factors
PAYOUTS = {"TOP3": [50, 30, 20, 0, 0, 0, 0, 0], "LINEAR": [100 * x / 36 for x in (8, 7, 6, 5, 4, 3, 2, 1)],
           "TOP4": [40, 30, 20, 10, 0, 0, 0, 0], "WTA": [100, 0, 0, 0, 0, 0, 0, 0],
           "ODD": [40, 25, 15, 10, 6, 4, 0, 0]}


@pytest.mark.parametrize("name", list(PAYOUTS))
def test_icm_matches_kit_brute_force(name):
    pays = PAYOUTS[name]
    icm = B.make_icm(pays)
    rng = random.Random(7)
    for _ in range(25):
        n = rng.randint(2, 8)
        stacks = [rng.randint(0, 5000) if rng.random() < 0.85 else 0 for _ in range(n)]
        if sum(1 for s in stacks if s > 0) < 2:
            stacks[0], stacks[1] = 1500, 700
        ref = icm_brute(stacks, pays)
        for i in range(n):
            assert abs(icm(stacks, i) - ref[i]) < 1e-9, (name, stacks, i)


def test_equal_stack_bubble_factor_table():
    _, t3, _ = B.icm_pack(B.pad_payouts(PAYOUTS["TOP3"]), 8, 0.5, 6.0)
    _, lin, _ = B.icm_pack(B.pad_payouts([7, 6, 5, 4, 3, 2, 1, 0]), 8, 0.5, 6.0)
    assert abs(t3[8] - 1.25) < 0.01 and abs(t3[4] - 1.875) < 0.01 and abs(t3[2] - 1.0) < 1e-9
    assert abs(lin[8] - 2.33) < 0.01 and abs(lin[4] - 1.80) < 0.01
    assert B.is_wta(B.pad_payouts(PAYOUTS["WTA"])) and not B.is_wta(B.pad_payouts(PAYOUTS["TOP3"]))


def test_bubble_factor_pads_short_payout_lists():
    icm = B.make_icm(B.pad_payouts([50, 30, 20]))
    bf = B.bubble_factor(icm, [2000] * 8, 0, 1, B.pad_payouts([50, 30, 20]))
    assert abs(bf - 1.25) < 0.01


# ------------------------------------------------------------------ tables
def test_tables_and_percentiles():
    assert len(B.LABELS) == 169 and abs(B.PUSH_PCT[B.CLASS_ID["AA"]] - 3 / 1326) < 1e-12
    assert B.PUSH_PCT[B.CLASS_ID["32o"]] > 0.99
    tot = sum(6 if len(lab) == 2 else (4 if lab[2] == "s" else 12) for lab in B.LABELS)
    assert tot == 1326
    aa, kk = B.CLASS_ID["AA"], B.CLASS_ID["KK"]
    assert B.eq_vs(aa, 1.0) > 0.84 and B.eq_vs(kk, 0.03) < 0.72
    e1, e2 = B.eq_vs(kk, 0.2), B.eq_vs(kk, 0.25)
    assert min(e1, e2) <= B.eq_vs(kk, 0.22) <= max(e1, e2)


def test_jam_and_call_fits_reproduce_the_solver():
    assert abs(B.jam_pct(10, 7) - 0.106) < 0.005
    assert abs(B.jam_pct(10, 1) - 0.548) < 0.005
    assert abs(B.call_pct(0.548, 10) - 0.38) < 0.03


# ------------------------------------------------------------------ calling model (P4) and ICM call side (v1)
def test_equity_calls_reproduce_nash_calls():
    for S, nash in ((8, 0.45), (10, 0.373), (15, 0.284), (20, 0.218)):
        d = run_spot(hu_sb_jam(S), 0)
        assert abs(action_share(d, "call") - nash) < 0.03, S
    d = run_spot(eight_bb_vs_jam(7, 10), 0)
    assert abs(action_share(d, "call") - 0.067) < 0.02


def test_call_uses_the_jammers_stack_not_the_biggest(monkeypatch):
    # P4: 8 seats, UTG (100 chips = 5 bb) jams; we are the BB with 1000; the big stacks behind do not matter.
    # UTG's chip-EV range at 5 bb with 7 behind is about 17.5%; we risk 80 to gain 130 (38% needed): the review
    # puts the right call at "about 20%" (vs 7% when S is taken against the biggest stack)
    pre = [2, 3, 4, 5, 6, 7, 0, 1]
    stacks = [1000] * 8
    stacks[2] = 100
    scripts = {s: [FOLD] for s in pre if s not in (2, 1)}
    scripts[2] = [ALLIN]
    d = run_spot({"n": 8, "stacks": stacks, "button": 7, "me": 1, "scripts": scripts}, 0)
    share = action_share(d, "call")
    assert 0.15 < share < 0.28, share
    monkeypatch.setitem(B.PARAMS, "CALL_MODEL", "fit")          # the fit model also uses the jammer's stack
    share_fit = action_share(run_spot({"n": 8, "stacks": stacks, "button": 7, "me": 1, "scripts": scripts}, 0), "call")
    assert 0.15 < share_fit < 0.28, share_fit


def test_v1_tightens_calls_under_icm_and_not_under_wta():
    spot = eight_bb_vs_jam(1, 10)
    c0 = action_share(run_spot(spot, 0), "call")
    c1 = action_share(run_spot(spot, 1), "call")
    assert c1 < c0 - 0.05                      # 8 left, 50/30/20: BF 1.25 on the call
    spot_wta = dict(spot, payouts="WTA")
    assert abs(action_share(run_spot(spot_wta, 1), "call") - c0) < 1e-9      # ICM off under WTA


def test_v1_jam_side_is_hero_only_by_default():
    # P2: no caller evidence -> jam% x BF_hero^-2.5 (never the caller-side widening)
    j0 = allin_share(run_spot(eight_first_in(1, 10), 0))
    j1 = allin_share(run_spot(eight_first_in(1, 10), 1))
    assert j1 < j0 and abs(j1 / j0 - 1.25 ** -2.5) < 0.12


# ------------------------------------------------------------------ hand tracker: observe() and history agree
def test_tracker_observe_and_history_feeds_agree():
    from sim.kit.engine import Table
    from sim.kit.fixtures import RandomAgent
    fac = load_factory("bot/shell.py:BotV2")
    captured = {}

    class NoObserve:
        """Our bot without observe(): the tracker must be fed from the state's history only."""
        def __init__(self, seat, cfg):
            self.b = fac(seat, cfg)
            captured.setdefault("hist", []).append(self.b)

        def act(self, state):
            return self.b.act(state)

    def with_obs(seat, cfg):
        b = fac(seat, cfg)
        captured.setdefault("obs", []).append(b)
        return b
    for seed in range(3):
        captured.clear()
        Table([with_obs, NoObserve] + [RandomAgent] * 6, "TOP3", "TURBO", seed=f"trk{seed}", max_hands=40).play()
        so = captured["obs"][0]._brain.stats
        sh = captured["hist"][0]._brain.stats
        # the observe-fed tracker sees every action; the history-fed one only up to its own decisions
        assert sum(st.pf_dec for st in so) >= sum(st.pf_dec for st in sh) > 0


# ------------------------------------------------------------------ v2 fixes
def _brain(version=2, payouts=(50, 30, 20, 0, 0, 0, 0, 0), n=8):
    return B.Brain(0, {"n_seats": n, "payouts": list(payouts), "seed": 1}, version=version)


def test_p7_folder_label_clears_at_first_voluntary_chip():
    b = _brain()
    st = b.stats[3]
    st.pf_dec = 20
    assert b.is_folder(3)
    st.vol_hand = True                       # it just called our steal in this hand
    assert not b.is_folder(3)
    st.vol_hand = False
    st.vol_ever = 1
    assert not b.is_folder(3)


def test_p1_sb_widen_needs_evidence():
    b = _brain()
    b.types = {}
    # untyped BB: no widening
    assert b.typed_jam(0.3, [1], 10.0) == 0.3
    # BB folding to jams: 0 of 6 jams called -> shrunk 0.15 < break-even 0.33 - 0.05 -> widen x1.6
    st = b.stats[1]
    st.pf_dec, st.pf_vol, st.vol_ever, st.jam_faced, st.jam_called = 20, 4, 4, 6, 0
    b.types = {}
    assert abs(b.typed_jam(0.3, [1], 10.0) - 0.48) < 1e-9
    # BB calling jams at 4/6: no widening; light-jam cap only if typed caller/loose/maniac
    st.jam_called = 4
    b.types = {}
    assert b.typed_jam(0.3, [1], 10.0) == 0.3
    assert b.seat_type(1) == "caller"
    b.types = {}
    assert b.typed_jam(0.6, [1], 10.0) == P["LIGHT_JAM_CAP"]


def test_p5_rejam_prior_is_tighter_in_v2():
    for ver, want_tight in ((1, False), (2, True)):
        b = _brain(ver)
        b.types = {}
        J = b.jam_range_of(2, 0.40, True)
        assert (J == P["REJAM_PRIOR"]) == want_tight
    b = _brain(2)
    st = b.stats[2]
    st.pf_dec, st.pf_vol, st.vol_ever, st.jam_opp, st.jam_first = 40, 24, 24, 40, 24   # a 60% jammer: maniac
    b.types = {}
    assert b.seat_type(2) == "maniac" and b.jam_range_of(2, 0.10, True) >= 0.5


def test_p6_bb_defends_vs_late_min_raise_in_v2():
    # 8 seats at 15 bb; the button min-raises to 40, SB folds; we are the BB
    pre = [2, 3, 4, 5, 6, 7, 0, 1]
    scripts = {s: [FOLD] for s in pre if s not in (7, 1)}
    scripts[7] = [RAISE(40)]
    spot = {"n": 8, "stacks": [300] * 8, "button": 7, "me": 1, "scripts": scripts}
    cont1 = 1.0 - action_share(run_spot(spot, 1), "fold")
    cont2 = 1.0 - action_share(run_spot(spot, 2), "fold")
    assert cont2 > cont1 + 0.10 and cont2 >= 0.25, (cont1, cont2)


def test_v2_free_steal_vs_folders():
    b = _brain()
    for s in (1, 2):
        b.stats[s].pf_dec = 12
    assert b.is_folder(1) and b.is_folder(2)
    assert not b.is_folder(3)
