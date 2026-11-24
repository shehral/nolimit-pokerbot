import pytest
"""Pass-2 tuning switches (2026-11-04, TUNING_LOG notes "pass 2"; DOCTRINE notes 2.6, 2.8, 2.9):
JAM_OPP_EXP_DEFAULT, REJAM_LEARN (+ REJAM_LEARN_MAX_BB / REJAM_LEARN_LATE / PRIOR["rejam"]), REJAM_BY_LEFT,
BB_DEFEND_PCT_HU, CALLJAM_PRIOR and SB_WIDEN_MIN_JAMS, plus the rejam_any_two test opponent. Each rule is checked
with its switch set explicitly on and off, so the tests hold whatever the shipped defaults are."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.runner import load_factory                          # noqa: E402
from bot.tests.scenarios import RAISE, FOLD, action_share, run_spot   # noqa: E402
from bot.tests.rejam_opponent import RejamAnyTwo                  # noqa: E402

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
B = SHELL.brain
P = B.PARAMS
TOP3 = [50, 30, 20, 0, 0, 0, 0, 0]
OFF = {"JAM_OPP_EXP_DEFAULT": None, "REJAM_LEARN": False, "BB_DEFEND_PCT_HU": None,
       "CALLJAM_PRIOR": None, "SB_WIDEN_MIN_JAMS": None}


def _brain(params=None, n=8):
    p = dict(OFF)
    p.update(params or {})
    return B.Brain(0, {"n_seats": n, "payouts": list(TOP3)[:max(n, 3)], "seed": 1}, version=2, params=p)


def _hand(tr, hand_no, stack=300, bb=20):
    """8 seats, SB = 0, BB = 1; preflop order 2, 3, 4, 5, 6, 7, 0, 1."""
    tr.start_hand(hand_no, [stack] * 8, [True] * 8, 0, 1, bb, 0)
    tr.feed(0, "post_sb", bb // 2)
    tr.feed(1, "post_bb", bb)


# ------------------------------------------------------------------ JAM_OPP_EXP_DEFAULT
def test_default_caller_side_term_on_and_off():
    for exp, want in ((None, 0.2 * 2.0 ** -2.5), (1.0, 0.2 * 2.0 ** -2.5 * 1.5)):
        b = _brain({"JAM_OPP_EXP_DEFAULT": exp, "JAM_HERO_EXP": -2.5})
        b.bf_mean = lambda me, seats, hero_side: 2.0 if hero_side else 1.5
        got = b.icm_jam(0.2, 0, [3, 4])
        assert abs(got - want) < 1e-12, (exp, got, want)
        assert b.n_jam_hero == 1 and b.n_jam_gated == 0


# ------------------------------------------------------------------ REJAM_LEARN counters
def _late_open_rejam(tr, hand_no, stack=300, opener=7, rejammer=0):
    _hand(tr, hand_no, stack)
    for s in (2, 3, 4, 5, 6, 7):
        if s != opener:
            tr.feed(s, "fold", 0)
        else:
            tr.feed(s, "raise", 50)
    tr.feed(rejammer, "raise", stack)              # all-in re-raise
    tr.feed(1, "fold", 0)                          # faces the re-jam: never a re-jam chance


def test_rejam_counters_commit_at_next_hand_and_respect_the_filters():
    b = _brain({"REJAM_LEARN": True, "REJAM_LEARN_MAX_BB": 25.0, "REJAM_LEARN_LATE": True})
    tr = b.tr
    _late_open_rejam(tr, 1)
    assert b.stats[0].rj_opp == 0                  # pending: the re-jam we would face is not its own evidence
    _hand(tr, 2)
    assert (b.stats[0].rj_opp, b.stats[0].rj_hit) == (1, 1)
    assert b.stats[1].rj_opp == 0                  # acted after the jam
    for s in (2, 3, 4, 5, 6):
        assert b.stats[s].rj_opp == 0              # folded before the open
    _late_open_rejam(tr, 3, stack=1000)            # 50 bb deep: not a chance
    _hand(tr, 4)
    assert b.stats[0].rj_opp == 1
    _late_open_rejam(tr, 5, opener=2)              # an early open (7 behind): not a chance with REJAM_LEARN_LATE
    _hand(tr, 6)
    assert b.stats[0].rj_opp == 1
    off = _brain({"REJAM_LEARN": False})
    _late_open_rejam(off.tr, 1)
    _hand(off.tr, 2)
    assert off.stats[0].rj_opp == 0                # the switch off records nothing
    wide = _brain({"REJAM_LEARN": True, "REJAM_LEARN_MAX_BB": None, "REJAM_LEARN_LATE": False})
    _late_open_rejam(wide.tr, 1, stack=1000, opener=2)
    _hand(wide.tr, 2)
    assert (wide.stats[0].rj_opp, wide.stats[0].rj_hit) == (1, 1)


@pytest.mark.skip(reason="pass-2 re-jam variant superseded by the red-team implementation at the 2026-11-05 merge (REDTEAM notes)")
def test_rejam_range_learned_and_by_players_left():
    for rm in (False, True):
        b = _brain({"REJAM_LEARN": True, "RANGE_MODEL": rm})
        b.n_live = 8
        assert abs(b.jam_range_of(5, 0.10, True) - P["REJAM_PRIOR"]) < 1e-12      # no evidence: the prior
        st = b.stats[5]
        st.rj_opp, st.rj_hit = 4, 3
        assert abs(b.jam_range_of(5, 0.10, True) - (3 + 6 * 0.06) / 10) < 1e-12   # DOCTRINE 2.8: 34%
        assert b.jam_range_of(5, 0.10, False) == 0.10 or rm                       # first-in jams untouched
        off = _brain({"REJAM_LEARN": False, "RANGE_MODEL": rm})
        off.n_live = 8
        off.stats[5].rj_opp, off.stats[5].rj_hit = 4, 3
        assert abs(off.jam_range_of(5, 0.10, True) - P["REJAM_PRIOR"]) < 1e-12
    b = _brain({"REJAM_BY_LEFT": True, "RANGE_MODEL": False})
    for left, want in ((2, 0.20), (3, 0.12), (4, P["REJAM_PRIOR"]), (8, P["REJAM_PRIOR"])):
        b.n_live = left
        assert abs(b.jam_range_of(5, 0.10, True) - want) < 1e-12, left
    b = _brain({"REJAM_BY_LEFT": False, "RANGE_MODEL": False})
    b.n_live = 2
    assert abs(b.jam_range_of(5, 0.10, True) - P["REJAM_PRIOR"]) < 1e-12


# ------------------------------------------------------------------ BB_DEFEND_PCT_HU
def _hu_bb_vs_open():
    """Heads-up 15 bb, the button (SB, seat 0) raises to 2.5 bb; we are the BB."""
    return {"n": 2, "stacks": [300, 300], "button": 0, "me": 1, "scripts": {0: [RAISE(50)]}}


def _fac(params):
    p = dict(OFF)
    p.update(params)
    return lambda s, cfg: SHELL.make(s, cfg, version=2, params=p)


def test_heads_up_bb_defence_fraction():
    spot = _hu_bb_vs_open()
    base = 1.0 - action_share(run_spot(spot, 2, factory=_fac({})), "fold")
    hu = 1.0 - action_share(run_spot(spot, 2, factory=_fac({"BB_DEFEND_PCT_HU": 0.5})), "fold")
    assert hu > base + 0.10, (base, hu)
    # 8-handed the HU value does not apply
    pre = [2, 3, 4, 5, 6, 7, 0, 1]
    scripts = {s: [FOLD] for s in pre if s not in (7, 1)}
    scripts[7] = [RAISE(50)]
    spot8 = {"n": 8, "stacks": [300] * 8, "button": 7, "me": 1, "scripts": scripts}
    a = run_spot(spot8, 2, factory=_fac({}))
    c = run_spot(spot8, 2, factory=_fac({"BB_DEFEND_PCT_HU": 0.5}))
    assert action_share(a, "fold") == action_share(c, "fold")


# ------------------------------------------------------------------ CALLJAM_PRIOR, SB_WIDEN_MIN_JAMS
def test_calljam_prior_replaces_the_mean_only_and_does_not_mutate_params():
    before = dict(P["PRIOR"])
    b = _brain({"CALLJAM_PRIOR": 0.2})
    assert b.P["PRIOR"]["calljam"] == (0.2, P["PRIOR"]["calljam"][1])
    assert P["PRIOR"] == before and b.P["PRIOR"]["vol"] == before["vol"]
    assert _brain({"CALLJAM_PRIOR": None}).P["PRIOR"]["calljam"] == before["calljam"]


def test_sb_widen_fires_by_default_at_10bb_only_with_the_new_prior():
    on = {"CALLJAM_PRIOR": 0.2, "SB_WIDEN_MIN_JAMS": 0, "RANGE_MODEL": False}
    b = _brain(on)
    assert abs(b.typed_jam(0.3, [1], 10.0) - 0.3 * P["SB_WIDEN"]) < 1e-12        # unknown BB presumed to under-call
    assert b.typed_jam(0.3, [1], 15.0) == 0.3                                    # break-even 0.19 at 15 bb: no widen
    assert _brain({"RANGE_MODEL": False}).typed_jam(0.3, [1], 10.0) == 0.3       # built rule: needs 3 jams faced
    st = b.stats[1]
    st.jam_faced, st.jam_called = 1, 1                                           # one call in one jam: (1+1.2)/7
    b.types = {}
    assert b.typed_jam(0.3, [1], 10.0) == 0.3


# ------------------------------------------------------------------ the rejam_any_two opponent
def test_rejam_any_two_opponent_rejams_late_opens_at_13_to_20_bb_only():
    pre = [2, 3, 4, 5, 6, 7, 0, 1]

    def spot(opener, stack):
        scripts = {s: [FOLD] for s in pre if s not in (opener, 1)}
        scripts[opener] = [RAISE(50)]
        return {"n": 8, "stacks": [stack] * 8, "button": 7, "me": 1, "scripts": scripts}
    fac = lambda s, cfg: RejamAnyTwo(s, cfg)                                     # noqa: E731
    late = run_spot(spot(7, 300), labels=["72o", "83o"], factory=fac)            # button open, 15 bb
    assert all(a.get("action") == "raise" and a.get("amount") == st["stacks"][1] + st["bets"][1]
               for st, a in late.values()), late
    early = run_spot(spot(2, 300), labels=["72o"], factory=fac)                  # UTG open: icm_pro folds 72o
    deep = run_spot(spot(7, 600), labels=["72o"], factory=fac)                   # 30 bb: icm_pro folds 72o
    assert action_share(early, "fold") == 1.0 and action_share(deep, "fold") == 1.0


@pytest.mark.skip(reason="pass-2 re-jam variant superseded by the red-team implementation at the 2026-11-05 merge (REDTEAM notes)")
def test_rejam_learned_rate_threshold():
    st_args = (4, 3)                                                             # shrunk (3 + 0.36) / 10 = 0.336
    for mn, want in ((0.0, 0.336), (0.30, 0.336), (0.35, P["REJAM_PRIOR"])):
        b = _brain({"REJAM_LEARN": True, "REJAM_LEARN_MIN": mn, "RANGE_MODEL": False})
        b.n_live = 8
        b.stats[5].rj_opp, b.stats[5].rj_hit = st_args
        assert abs(b.jam_range_of(5, 0.10, True) - want) < 1e-12, mn
