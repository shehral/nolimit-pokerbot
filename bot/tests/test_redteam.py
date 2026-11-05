"""Red-team switches (REDTEAM notes): the exploiter knobs (STAB_AIR, BARREL_AIR) and the hardening switches
(REJAM_LEARN, REJAM_PRIOR_BY_LEFT, BB_DEFEND_PCT_HU, REJAM_JAMZONE_BB, PROBE_LEARN, RESTEAL_WIDE_BB, PROBE_BET_FIRST).
Every switch is off by default except the hardening pair RESTEAL_WIDE_BB 25 + PROBE_BET_FIRST 1.0 (on 2026-11-05,
REDTEAM notes); with every switch off the bot plays exactly the pre-red-team v2; each rule is checked on and off."""
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
OFF = {"STAB_AIR": 0.0, "BARREL_AIR": 0.0, "REJAM_LEARN": False, "REJAM_PRIOR_BY_LEFT": None,
       "BB_DEFEND_PCT_HU": None, "REJAM_JAMZONE_BB": None, "PROBE_LEARN": False, "PROBE_AIR_CALL_POT": 0.0,
       "PROBE_BET_FIRST": 0.0, "RESTEAL_WIDE_BB": None}
SHIPPED_ON = {"RESTEAL_WIDE_BB": 25.0, "PROBE_BET_FIRST": 1.0}   # the applied pair


def _brain(params=None, version=2):
    return B.Brain(0, {"n_seats": 8, "payouts": TOP3, "seed": 1}, version=version, params=params)


def test_switch_defaults():
    for k, v in OFF.items():
        assert P[k] == SHIPPED_ON.get(k, v), k


def _play(params, tables=12, seed=0):
    """Our decisions in 4 seats (v2 with params) vs the kit's example agent and a calling fixture."""
    from sim.kit.engine import Table
    from sim.kit.example_agent import ExampleAgent
    from sim.kit.fixtures import CheckCallAgent
    rows = []

    def ours(seat, cfg):
        bot = SHELL.make(seat, cfg, version=2, params=dict(params, GUARD_NS=10 ** 15))
        orig = bot.act

        def act(state, _o=orig):
            a = _o(state)
            rows.append((state["hand_no"], state["seat"], state["street"], a.get("action"), a.get("amount")))
            return a
        bot.act = act
        return bot
    for t in range(tables):
        facs = [ours if s % 2 == 0 else (ExampleAgent if s % 4 == 1 else CheckCallAgent) for s in range(8)]
        Table(facs, seed=f"redteam:{seed}:{t}", payouts="TOP3", schedule="TURBO", time_limit_ms=None).play()
    return rows


def test_explicit_defaults_play_the_shipped_bot_and_off_differs():
    shipped = _play({})
    assert shipped == _play(dict(OFF, **SHIPPED_ON))
    assert shipped != _play(OFF)              # the pair changes play against these opponents


def test_exploiter_knobs_change_play():
    assert _play({}, tables=6) != _play({"STAB_AIR": 1.0, "BARREL_AIR": 1.0}, tables=6)


def test_hardening_on_plays_legally_and_differs_somewhere():
    on = {"REJAM_LEARN": True, "REJAM_PRIOR_BY_LEFT": {2: 0.20, 3: 0.12}, "BB_DEFEND_PCT_HU": 0.5,
          "REJAM_JAMZONE_BB": 20.0, "PROBE_LEARN": True, "PROBE_AIR_CALL_POT": 0.26}
    a = _play(on, tables=12)
    assert len(a) > 500
    assert a != _play(OFF, tables=12)


def test_rejam_learn_widens_a_frequent_rejammer():
    b = _brain({"REJAM_LEARN": True})
    b.n_live = 8
    st = b.stats[3]
    J0 = b.jam_range_of(3, 0.2, True)
    st.rm_3b_n, st.rm_3b_hit = 4, 3                    # 3 re-raises in 4 chances: (3 + 0.36) / 10 = 0.336
    J1 = b.jam_range_of(3, 0.2, True)
    assert abs(b.rejam_rate(3) - 0.336) < 1e-9
    assert J1 >= 0.336 - 1e-9 and J1 > J0
    off = _brain({"REJAM_LEARN": False})
    off.n_live = 8
    off.stats[3].rm_3b_n, off.stats[3].rm_3b_hit = 4, 3
    assert off.jam_range_of(3, 0.2, True) <= P["REJAM_PRIOR"] * P["RM_CAP"] + 1e-9   # capped without the switch


def test_rejam_learn_never_narrows_and_ignores_first_in_jams():
    b = _brain({"REJAM_LEARN": True})
    b.n_live = 8
    b.stats[2].rm_3b_n = 40                             # never re-raised in 40 chances
    off = _brain()
    off.n_live = 8
    off.stats[2].rm_3b_n = 40
    assert b.jam_range_of(2, 0.2, True) >= off.jam_range_of(2, 0.2, True) - 1e-12
    assert b.rejam_learned(2, 0.10) == 0.10
    assert b.jam_range_of(2, 0.25, False) == _brain().jam_range_of(2, 0.25, False)


def test_rejam_prior_by_left():
    b = _brain({"REJAM_PRIOR_BY_LEFT": {2: 0.20, 3: 0.12}})
    for nl, want in ((2, 0.20), (3, 0.12), (5, P["REJAM_PRIOR"])):
        b.n_live = nl
        assert b.rejam_prior() == want
    b2 = _brain()
    b2.n_live = 2
    assert b2.rejam_prior() == P["REJAM_PRIOR"]


def test_rejammer_needs_observations():
    b = _brain({"REJAM_JAMZONE_BB": 20.0})
    st = b.stats[4]
    st.rm_3b_n, st.rm_3b_hit = 2, 2
    assert not b.rejammer(4)                            # under REJAM_MIN_OBS chances
    st.rm_3b_n, st.rm_3b_hit = 3, 3
    assert b.rejammer(4)
    st.rm_3b_n, st.rm_3b_hit = 20, 1
    assert not b.rejammer(4)


def test_resteal_wide_needs_an_observed_wide_opener():
    b = _brain({"RESTEAL_WIDE_BB": 25.0})
    st = b.stats[5]
    assert b.rm_open_ratio(5) == 1.0                    # no evidence: the switch cannot fire
    st.rm_open_n, st.rm_open_hit, st.rm_open_exp = 20, 12, 4.0   # opened 12 of 20 where the prior expects 4
    assert b.rm_open_ratio(5) >= P["RESTEAL_WIDE_RATIO"]


def _flop_view(hole, board, bet, base=100, stack=1000):
    """Heads-up flop facing a first bet of `bet` into `base` from seat 1 (the kit's pot includes the bet)."""
    v = SHELL.adapter.View()
    pc = SHELL.adapter.parse_card
    v.hand_no, v.street, v.me, v.n = 1, 1, 0, 2
    v.hole = tuple(pc(c) for c in hole)
    v.board = [pc(c) for c in board]
    v.boards = None
    v.bounty = None
    v.sb, v.bb, v.ante = 10, 20, 0
    v.bets = [0, bet]
    v.contrib = [base // 2, base // 2 + bet]
    v.pot = base + bet
    v.stacks = [stack, stack - bet]
    v.start = [stack + base // 2, stack + base // 2]
    v.in_hand = [True, True]
    v.all_in = [False, False]
    v.alive = [True, True]
    v.to_call = bet
    v.min_to, v.max_to = 2 * bet, stack
    v.legal = ["fold", "call", "raise"]
    v.history = []
    v.street_starts = [0, 0]
    return v


def test_probe_learn_light_calls_a_frequent_bettor():
    weak = (("9h", "8d"), ("Ac", "Kd", "8s"))          # bottom pair: class 1
    air = (("5h", "4d"), ("Ac", "Kd", "9s"))           # nothing: class 0
    assert B.hand_class(*_flop_view(*weak, 80).hole, _flop_view(*weak, 80).board) == 1
    assert B.hand_class(*_flop_view(*air, 30).hole, _flop_view(*air, 30).board) == 0
    for on in (False, True):
        b = _brain({"PROBE_LEARN": on, "PROBE_AIR_CALL_POT": 0.26})
        st = b.stats[1]
        st.probe_opp, st.probe_bet = 10, 9                 # bets 9 of 10 unbet streets: shrunk 0.73 >= 0.65
        st.pf_dec = 2                                      # too few preflop decisions for any other type
        want = ("call",) if on else ("fold",)
        assert b.postflop(_flop_view(*weak, 80)) == want  # 80 into 100: over CALL1_MAX_POT, light call when on
        assert b.postflop(_flop_view(*air, 30)) == want   # 30 into 100 = 0.23 of the pot with the bet
        assert b.postflop(_flop_view(*air, 60)) == ("fold",)


def test_probe_bet_first_bets_air_into_a_prober_only_when_on():
    air = (("5h", "4d"), ("Ac", "Kd", "9s"))
    for on, want in ((0.0, "check"), (1.0, "raise")):
        b = _brain({"PROBE_BET_FIRST": on})
        b.stats[1].probe_opp, b.stats[1].probe_bet = 10, 9
        b.stats[1].pf_dec = 2
        v = _flop_view(*air, 0)
        v.legal = ["check", "raise"]
        assert b.postflop(v)[0] == want
    b = _brain({"PROBE_BET_FIRST": 1.0})
    b.stats[1].probe_opp, b.stats[1].probe_bet = 10, 2  # a passive seat: no lead
    b.stats[1].pf_dec = 2
    v = _flop_view(*air, 0)
    v.legal = ["check", "raise"]
    assert b.postflop(v) == ("check",)


def test_probe_counters_count_unbet_streets_only():
    b = _brain()
    tr = b.tr
    tr.start_hand(1, [2000] * 8, [True] * 8, 0, 1, 20, 0)
    tr.new_street(1)
    tr.feed(2, "check", 0)
    tr.feed(3, "raise", 40)                             # bet into an unbet street
    tr.feed(2, "call", 40)                              # facing a bet: not a probe chance
    st2, st3 = b.stats[2], b.stats[3]
    assert (st2.probe_opp, st2.probe_bet) == (1, 0)
    assert (st3.probe_opp, st3.probe_bet) == (1, 1)
