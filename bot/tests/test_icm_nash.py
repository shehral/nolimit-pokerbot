"""ICM_NASH (ICM_NASH notes): the equilibrium tables load and are well formed, the switch is OFF by default and inert
when off, the bot plays exactly the tables' sets on real engine states when on, and the tables are never used under
winner-take-all (which routes to the chip-EV formula)."""
import functools
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import brain as B                                          # noqa: E402
from bot import shell                                               # noqa: E402
from bot.tests.scenarios import ALLIN, CALL, FOLD, action_share, allin_share, run_spot   # noqa: E402

NT = B._NT
TABLES = os.path.join(ROOT, "bot", "icm_nash_tables.py")
ON = {"ICM_NASH": True, "GUARD_NS": 10 ** 15}


def fac(version, params):
    return functools.partial(shell.make, version=version, params=params)


def test_tables_load_and_are_small():
    assert NT is not None, "bot/icm_nash_tables.py missing or broken"
    assert os.path.getsize(TABLES) < 3_000_000
    assert set(NT.KINDS) >= {"TOP3", "LINEAR"}
    for n in (2, 3, 4):
        assert len(NT.NODES[n]) == {2: 2, 3: 6, 4: 13}[n]
        for kind in NT.KINDS:
            assert len(NT.ROWS[(n, kind)]) == 4 * len(NT.NODES[n]) * len(NT.GRID) ** n


def test_bucket_is_nearest_grid_value():
    g = NT.GRID
    assert [NT.bucket(x) for x in g] == list(range(len(g)))
    assert NT.bucket(0.4) == 0 and NT.bucket(3.9) == 0 and NT.bucket(4.0) == 1
    assert NT.bucket(11.4) == g.index(10) and NT.bucket(11.6) == g.index(13)
    assert NT.bucket(80.0) == len(g) - 1
    assert NT.lookup(5, "TOP3", (10,) * 5, ("push", 0)) is None
    assert NT.lookup(4, "TOP4", (10,) * 4, ("push", 3)) is None


@pytest.mark.parametrize("kind", ["TOP3", "LINEAR"])
def test_heads_up_is_chip_ev_nash(kind):
    """Two left: ICM is linear in chips for every payout, so the tables must reproduce chip-EV Nash (SB jam 58%,
    BB call 37.3% at 10 bb; bot design notes, range checks)."""
    push = B.mask_pct(NT.lookup(2, kind, (10, 10), ("push", 0)))
    call = B.mask_pct(NT.lookup(2, kind, (10, 10), ("call", 0, 1)))
    assert 0.55 <= push <= 0.61
    assert 0.35 <= call <= 0.40


def test_bubble_tightens_calls_vs_wta():
    """Four left, TOP3 (4th pays 0): bubble calls are far tighter than the chip-EV (WTA) calls."""
    st = (10, 10, 10, 10)
    for node in (("call", 0, 1), ("call", 3, 1), ("call", 2, 0)):
        top3 = B.mask_pct(NT.lookup(4, "TOP3", st, node))
        if "WTA" in NT.KINDS:
            assert top3 < 0.75 * B.mask_pct(NT.lookup(4, "WTA", st, node))
        assert top3 < 0.20


def test_switch_and_never_under_wta():
    assert B.PARAMS["ICM_NASH"] is True                                 # ON since 2026-11-05 (ICM_NASH notes s6)
    cfg = {"n_seats": 8, "payouts": [50, 30, 20]}
    assert B.Brain(0, cfg, 2, {"ICM_NASH": False}).nash_kind is None    # the switch
    assert B.Brain(0, cfg, 2, dict(ON)).nash_kind == "TOP3"
    assert B.Brain(0, {"n_seats": 8, "payouts": [100]}, 2, dict(ON)).nash_kind is None      # WTA: chip-EV
    assert B.Brain(0, {"n_seats": 8, "payouts": [40, 30, 20, 10]}, 2, dict(ON)).nash_kind is None   # not solved
    assert B.Brain(0, cfg, 0, dict(ON)).nash_kind is None               # v0 has no ICM
    assert B.Brain(0, cfg, 2, dict(ON, ICM_ON=False)).nash_kind is None


def _spot4(me, scripts, stacks_bb=(10, 10, 10, 10), payouts="TOP3"):
    # 4 seats, button 0: SB 1, BB 2, CO 3 (first to act), BTN 0. Stacks given by position (SB, BB, BTN, CO).
    sb, bb, btn, co = stacks_bb
    stacks = [btn * 20, sb * 20, bb * 20, co * 20]
    return {"n": 4, "stacks": stacks, "button": 0, "me": me, "scripts": scripts, "payouts": payouts}


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("stacks", [(10, 10, 10, 10), (20, 5, 5, 10)])
def test_first_in_jams_are_the_table_sets(version, stacks):
    co = run_spot(_spot4(3, {}, stacks), factory=fac(version, ON))
    assert abs(allin_share(co) - B.mask_pct(NT.lookup(4, "TOP3", stacks, ("push", 3)))) < 1e-9
    sb = run_spot(_spot4(1, {3: [FOLD], 0: [FOLD]}, stacks), factory=fac(version, ON))
    assert abs(allin_share(sb) - B.mask_pct(NT.lookup(4, "TOP3", stacks, ("push", 0)))) < 1e-9


@pytest.mark.parametrize("version", [1, 2])
def test_jam_calls_and_overcalls_are_the_table_sets(version):
    st = (10, 10, 10, 10)
    bb_vs_sb = run_spot(_spot4(2, {3: [FOLD], 0: [FOLD], 1: [ALLIN]}), factory=fac(version, ON))
    assert abs(action_share(bb_vs_sb, "call") - B.mask_pct(NT.lookup(4, "TOP3", st, ("call", 0, 1)))) < 1e-9
    bb_vs_co = run_spot(_spot4(2, {3: [ALLIN], 0: [FOLD], 1: [FOLD]}), factory=fac(version, ON))
    assert abs(action_share(bb_vs_co, "call") - B.mask_pct(NT.lookup(4, "TOP3", st, ("call", 3, 1)))) < 1e-9
    over = run_spot(_spot4(2, {3: [ALLIN], 0: [CALL], 1: [FOLD]}), factory=fac(version, ON))
    assert abs(action_share(over, "call") - B.mask_pct(NT.lookup(4, "TOP3", st, ("over", 3, 2, 1)))) < 1e-9


def test_off_switch_is_the_heuristic():
    """ICM_NASH False plays the BF heuristic (the identity test pins it over 20,000 decisions): on this bubble spot
    it differs from the tables, and ON (the default) equals the explicit ON."""
    off = run_spot(_spot4(3, {}), factory=fac(2, {"GUARD_NS": 10 ** 15, "ICM_NASH": False}))
    dflt = run_spot(_spot4(3, {}), factory=fac(2, {"GUARD_NS": 10 ** 15}))
    on = run_spot(_spot4(3, {}), factory=fac(2, dict(ON)))
    assert {k: v[1] for k, v in dflt.items()} == {k: v[1] for k, v in on.items()}
    assert allin_share(off) < 0.5 * allin_share(on)


def test_tables_unused_under_wta_in_play():
    """Real tables under WTA with the switch ON: no decision may come from the equilibrium tables."""
    from sim.kit.engine import Table
    from sim.kit.fixtures import AllInAgent, RandomAgent
    made = []

    def ours(seat, cfg):
        b = shell.make(seat, cfg, version=2, params=dict(ON))
        made.append(b)
        return b
    for t in range(6):
        Table([ours, AllInAgent, RandomAgent, ours, AllInAgent, RandomAgent, ours, RandomAgent], "WTA",
              seed=f"wta{t}", max_hands=200, time_limit_ms=None, stacks=[300] * 8).play()
    used = 0
    for b in made:
        br = b._brain
        assert br.nash_kind is None
        used += sum(br.n_nash)
    assert used == 0


def test_tables_used_in_play_under_top3():
    from sim.kit.engine import Table
    from sim.kit.fixtures import AllInAgent, RandomAgent
    made = []

    def ours(seat, cfg):
        b = shell.make(seat, cfg, version=2, params=dict(ON))
        made.append(b)
        return b
    for t in range(6):
        Table([ours, AllInAgent, RandomAgent, ours, AllInAgent, RandomAgent, ours, RandomAgent], "TOP3",
              seed=f"top3{t}", max_hands=200, time_limit_ms=None, stacks=[300] * 8).play()
    assert sum(sum(b._brain.n_nash[:3]) for b in made) > 0
    assert all(b.errors == 0 for b in made)


def test_parts_switch_limits_what_the_tables_decide():
    heur = run_spot(_spot4(3, {}), factory=fac(2, {"GUARD_NS": 10 ** 15, "ICM_NASH": False}))
    calls_only = run_spot(_spot4(3, {}), factory=fac(2, dict(ON, ICM_NASH_PARTS="call")))
    assert {k: v[1] for k, v in heur.items()} == {k: v[1] for k, v in calls_only.items()}
    st = (10, 10, 10, 10)
    bb_vs_sb = _spot4(2, {3: [FOLD], 0: [FOLD], 1: [ALLIN]})
    jam_only = run_spot(bb_vs_sb, factory=fac(1, dict(ON, ICM_NASH_PARTS="jam")))
    heur_bb = run_spot(bb_vs_sb, factory=fac(1, {"GUARD_NS": 10 ** 15, "ICM_NASH": False}))
    assert action_share(jam_only, "call") == action_share(heur_bb, "call")
    assert action_share(jam_only, "call") != B.mask_pct(NT.lookup(4, "TOP3", st, ("call", 0, 1)))
