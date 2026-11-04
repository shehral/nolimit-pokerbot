"""Range model (PARAMS["RANGE_MODEL"], RANGE_MODEL notes): per-seat Beta-binomial posteriors of range width from
preflop evidence of completed hands. Checked with the switch on and off, so the tests hold whatever the default is."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import Table                                  # noqa: E402
from sim.kit.example_agent import ExampleAgent                    # noqa: E402
from sim.kit.fixtures import AllInAgent, CheckCallAgent, RandomAgent   # noqa: E402
from sim.kit.runner import load_factory                           # noqa: E402

from bot.tests.scenarios import RAISE, FOLD, action_share, eight_bb_vs_jam, run_spot   # noqa: E402

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
B = SHELL.brain
P = B.PARAMS
ON = {"RANGE_MODEL": True}
OFF = {"RANGE_MODEL": False}
TOP3 = [50, 30, 20, 0, 0, 0, 0, 0]


def _brain(params=ON, version=2):
    return B.Brain(0, {"n_seats": 8, "payouts": list(TOP3), "seed": 1}, version=version, params=dict(params))


def _hand(tr, hand_no, stack=400, bb=20):
    """8 seats, SB = 0, BB = 1; preflop order 2, 3, 4, 5, 6, 7, 0, 1."""
    tr.start_hand(hand_no, [stack] * 8, [True] * 8, 0, 1, bb, 0)
    tr.feed(0, "post_sb", bb // 2)
    tr.feed(1, "post_bb", bb)


# ------------------------------------------------------------------ the posterior ratio
def test_ratio_is_one_without_evidence_and_clamped():
    b = _brain()
    assert b.rm_ratio(0, 0.0, 3.0) == 1.0
    assert abs(b.rm_ratio(2, 1.0, 3.0) - 5.0 / 4.0) < 1e-12           # (hits + a) / (E + a)
    assert b.rm_ratio(60, 3.0, 3.0) == P["RM_CAP"]                   # never wider than 2x the prior
    assert b.rm_ratio(0, 60.0, 3.0) == 1.0 / P["RM_CAP"]             # never narrower than half
    assert b.rm_ratio(0, 3.0, 3.0) == 0.5                            # 0 hits where 3 were expected


def test_one_jam_cannot_swing_the_range_much():
    b = _brain()
    st = b.stats[2]
    st.rm_jam_n, st.rm_jam_hit, st.rm_jam_exp = 3, 1, 0.24             # 1 jam in 3 spots at about 8% each
    r = b.rm_jam_ratio(2, False)
    assert 1.0 < r < 1.35, r


# ------------------------------------------------------------------ evidence collection
def test_evidence_kinds_and_commit_at_next_hand():
    b = _brain()
    tr = b.tr
    _hand(tr, 1)                                     # 20 bb deep: both jam and open zone
    tr.feed(2, "fold", 0)
    tr.feed(3, "raise", 50)                          # first-in open (not all-in)
    tr.feed(4, "raise", 150)                         # 3-bet
    tr.feed(5, "fold", 0)                            # facing two raises: no evidence
    st2, st3, st4, st5 = (b.stats[j] for j in (2, 3, 4, 5))
    assert st2.rm_jam_n == st2.rm_open_n == 0        # pending until the hand is over
    _hand(tr, 2)
    assert st2.rm_jam_n == 1 and st2.rm_jam_hit == 0 and st2.rm_jam_exp > 0
    assert st2.rm_open_n == 1 and st2.rm_open_hit == 0 and abs(st2.rm_open_exp - P["OPEN_PCT"][7]) < 1e-12
    assert st3.rm_open_n == 1 and st3.rm_open_hit == 1 and st3.rm_jam_n == 1 and st3.rm_jam_hit == 0
    assert st4.rm_3b_n == 1 and st4.rm_3b_hit == 1 and st4.rm_open_n == 0
    assert st5.rm_3b_n == 0 and st5.rm_open_n == 0 and st5.rm_jam_n == 0
    # hand 2: seat 2 jams first-in; seat 3 faces the jam (jam-call evidence, not 3-bet evidence) and calls
    tr.feed(2, "raise", 400)
    tr.feed(3, "call", 400)
    _hand(tr, 3)
    assert st2.rm_jam_n == 2 and st2.rm_jam_hit == 1
    assert st3.rm_3b_n == 0 and st3.rm_cj_n == 1 and st3.rm_cj_hit == 1 and st3.rm_cj_exp > 0


def test_depth_gates_of_the_evidence():
    b = _brain()
    tr = b.tr
    _hand(tr, 1, stack=1000)                         # 50 bb: open evidence only
    tr.feed(2, "raise", 50)
    _hand(tr, 2, stack=200)                          # 10 bb: jam evidence only
    tr.feed(2, "raise", 200)
    _hand(tr, 3)
    st = b.stats[2]
    assert st.rm_open_n == 1 and st.rm_open_hit == 1
    assert st.rm_jam_n == 1 and st.rm_jam_hit == 1


def test_postflop_calls_and_limps_are_never_range_evidence():
    b = _brain()
    tr = b.tr
    for h in range(1, 30):
        _hand(tr, h, stack=2000)
        tr.feed(2, "call", 20)                       # limp: a voluntary chip but not an open
        for j in (3, 4, 5, 6, 7, 0):
            tr.feed(j, "fold", 0)
        tr.feed(1, "check", 0)
        tr.new_street(1)
        tr.feed(1, "raise", 40)                      # BB bets the flop
        tr.feed(2, "call", 40)                       # seat 2 calls every postflop bet
    _hand(tr, 30)
    st = b.stats[2]
    assert st.post_call >= 25 and st.rm_open_hit == 0 and st.rm_3b_n == 0 and st.rm_cj_n == 0
    assert b.rm_call_ratio(2) == 1.0                 # a postflop "caller" is not a wide jam caller
    assert b.seat_type(2) == "caller"


def test_switch_off_records_nothing():
    b = _brain(OFF)
    tr = b.tr
    _hand(tr, 1)
    tr.feed(2, "raise", 400)
    tr.feed(3, "call", 400)
    _hand(tr, 2)
    for st in b.stats:
        assert all(getattr(st, f) == 0 for f in B.RM_FIELDS)
    assert not b.rm


# ------------------------------------------------------------------ ranges
def test_jam_and_rejam_ranges_scale_with_the_posterior_and_stay_capped():
    b = _brain()
    st = b.stats[2]
    b.types = {}
    assert b.jam_range_of(2, 0.10, False) == 0.10                     # no evidence: the prior
    st.rm_jam_n, st.rm_jam_hit, st.rm_jam_exp = 100, 14, 8.0           # jams 1.75x the formula
    b.types = {}
    assert abs(b.jam_range_of(2, 0.10, False) - 0.10 * 17.0 / 11.0) < 1e-12
    st.rm_jam_hit = 80
    assert abs(b.jam_range_of(2, 0.10, False) - 0.20) < 1e-12         # capped at 2x
    st.rm_jam_hit = 0
    assert abs(b.jam_range_of(2, 0.10, False) - 0.05) < 1e-12         # ... and at half
    st.rm_3b_n, st.rm_3b_hit = 100, 0                                 # never re-raises in 100 spots
    assert abs(b.jam_range_of(2, 0.40, True) - P["REJAM_PRIOR"] / 2.0) < 1e-12
    off = _brain(OFF)
    off.stats[2].rm_jam_n, off.stats[2].rm_jam_hit, off.stats[2].rm_jam_exp = 100, 80, 8.0
    assert off.jam_range_of(2, 0.10, False) == 0.10                   # switch off: evidence ignored
    v1 = _brain(version=1)
    v1.stats[2].rm_jam_n, v1.stats[2].rm_jam_hit, v1.stats[2].rm_jam_exp = 100, 80, 8.0
    assert v1.jam_range_of(2, 0.10, False) == 0.10                    # v2 only


def test_fallback_turns_the_model_off_and_restores_it():
    b = _brain()
    seen = []
    b.decide = lambda v, t0: seen.append((b.version, b.rm, b.icm_on))
    B.Brain.decide_fallback(b, None, 0)
    assert seen == [(0, False, False)] and b.rm and b.version == 2


# ------------------------------------------------------------------ decisions
def _with_evidence(params, seat, **fields):
    def fac(s, cfg):
        bot = SHELL.make(s, cfg, version=2, params=params)
        st = bot._brain.stats[seat]
        for k, x in fields.items():
            setattr(st, k, x)
        return bot
    return fac


def test_bb_calls_wider_vs_a_wide_jammer_and_tighter_vs_a_tight_one():
    spot = eight_bb_vs_jam(7, 10)                    # UTG (seat 2) jams 10 bb, we are the BB
    base_off = action_share(run_spot(spot, 2, factory=_with_evidence(OFF, 2)), "call")
    base_on = action_share(run_spot(spot, 2, factory=_with_evidence(ON, 2)), "call")
    assert base_on == base_off                       # no evidence: identical decisions
    wide = action_share(run_spot(spot, 2, factory=_with_evidence(ON, 2, rm_jam_n=100, rm_jam_hit=40,
                                                                 rm_jam_exp=8.0)), "call")
    tight = action_share(run_spot(spot, 2, factory=_with_evidence(ON, 2, rm_jam_n=100, rm_jam_hit=0,
                                                                  rm_jam_exp=8.0)), "call")
    off_wide = action_share(run_spot(spot, 2, factory=_with_evidence(OFF, 2, rm_jam_n=100, rm_jam_hit=40,
                                                                     rm_jam_exp=8.0)), "call")
    assert tight < base_on < wide, (tight, base_on, wide)
    assert off_wide == base_off


def test_bb_defence_narrows_vs_a_tight_opener():
    pre = [2, 3, 4, 5, 6, 7, 0, 1]
    scripts = {s: [FOLD] for s in pre if s not in (7, 1)}
    scripts[7] = [RAISE(40)]                         # button min-raises at 15 bb, we are the BB
    spot = {"n": 8, "stacks": [300] * 8, "button": 7, "me": 1, "scripts": scripts}
    base = 1.0 - action_share(run_spot(spot, 2, factory=_with_evidence(ON, 7)), "fold")
    off = 1.0 - action_share(run_spot(spot, 2, factory=_with_evidence(OFF, 7)), "fold")
    tight = 1.0 - action_share(run_spot(spot, 2, factory=_with_evidence(ON, 7, rm_open_n=80, rm_open_hit=2,
                                                                        rm_open_exp=20.0)), "fold")
    assert base == off and tight < base, (tight, base, off)


def test_plays_legally_in_strict_mode_and_collects_evidence(monkeypatch):
    monkeypatch.setattr(SHELL, "STRICT", True)
    made = []

    def ours(seat, cfg):
        bot = SHELL.make(seat, cfg, version=2, params=ON)
        made.append(bot)
        return bot
    opps = [RandomAgent, AllInAgent, CheckCallAgent, ExampleAgent]
    for t in range(12):
        facs = [ours if s % 2 == 0 else opps[(s + t) % 4] for s in range(8)]
        tab = Table(facs, ("TOP3", "LINEAR", "WTA", "TOP4")[t % 4], "TURBO", seed=f"rm:{t}", timing="cpu",
                    time_limit_ms=None)
        tab.play()
    assert made and all(b.errors == 0 and b.fallbacks == 0 for b in made)
    tot = [0] * len(B.RM_FIELDS)
    for bot in made:
        for st in bot._brain.stats:
            for i, f in enumerate(B.RM_FIELDS):
                tot[i] += getattr(st, f)
    assert tot[0] > 0 and tot[1] > 0 and tot[3] > 0 and tot[8] > 0, tot
