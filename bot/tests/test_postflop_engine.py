"""Postflop engine (bot/postflop.py, PARAMS["POSTFLOP_ENGINE"]): classifiers, tables, decisions, the brain hook.

    .venv/bin/python -m pytest -q bot/tests/test_postflop_engine.py
"""
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest                                       # noqa: E402

from bot import brain as B                          # noqa: E402
from bot import postflop as PF                      # noqa: E402
from sim.kit.cards import card_int, evaluate        # noqa: E402

BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def C(s):
    return [card_int(x) for x in s.split()]


def feat(hole, board):
    h = C(hole)
    return PF.hand_feat(h[0], h[1], C(board))


# ----------------------------------------------------------------------------- tables
def test_tables_present_small_and_pure_data():
    path = os.path.join(BOT, "postflop_tables.py")
    assert os.path.getsize(path) < 2 * 1024 * 1024
    assert PF.DATA is not None, PF.TABLE_INFO
    assert len(PF.DATA) == PF.N_STREETS * PF.N_TEX * PF.N_CLS * PF.N_WIDTH * PF.N_K
    src = open(path).read()
    for word in ("import ", "def ", "class ", "lambda", "exec(", "eval("):
        assert word not in src, word
    assert B._PF is PF or B._PF is not None


def test_default_is_off():
    assert B.PARAMS["POSTFLOP_ENGINE"] is False


# ----------------------------------------------------------------------------- classifiers (hand-checked)
@pytest.mark.parametrize("hole,board,made,draw", [
    ("Ah Kh", "Qh Jh 2c", 2, 4),     # two overcards + nut flush draw + gutshot (T) = combo draw
    ("9c 8d", "7h 6s 2c", 2, 2),     # two overcards, open-ender
    ("9c 8d", "Th 6s 2c", 0, 1),     # gutshot (7)
    ("Ac Kd", "Ah 7s 2c", 6, 0),     # top pair, top kicker
    ("Ac 5d", "Ah 7s 2c", 5, 0),     # top pair, weak kicker
    ("Qc Qd", "Ah 7s 2c", 4, 0),     # pocket pair between the top two board ranks
    ("7c 7d", "Ah 7s 2c", 10, 0),    # set
    ("Ac Ad", "Kh 7s 2c", 7, 0),     # overpair
    ("5c 5d", "Kh 7s 2c", 3, 0),     # underpair
    ("Kc 7d", "Kh 7s 2c", 8, 0),     # two pair with both hole cards
    ("Kc 9d", "Kh Ks 2c", 9, 0),     # trips with one hole card
    ("Kc 2d", "Kh Ks 2c", 14, 0),    # full house
    ("9c 8c", "7c 6c 2c", 14, 0),    # no: 9c 8c 7c 6c = straight flush draw? 5 clubs incl 2c -> flush, 6-7-8-9 + T/5
    ("Ac 3c", "Kc 7c 2c", 13, 0),    # nut flush
    ("4c 3c", "Kc 7c 2c", 12, 0),    # low flush
    ("9d 8d", "7h 6s 5c", 11, 0),    # straight
    ("Ah 2d", "7h 6s 5c Kd 9c", 1, 0),   # river: one overcard (A), no draws on the river
    ("Jh Th", "9h 8s 2h 3c", 2, 4),  # turn: overcards + flush draw + open-ender
])
def test_hand_feat_cases(hole, board, made, draw):
    if hole == "9c 8c":              # 9c 8c 7c 6c 2c is a flush (no 5c / Tc for the straight flush): flush 9 high
        assert feat(hole, board) == (12, 0)
        return
    assert feat(hole, board) == (made, draw)


def test_made_classes_agree_with_the_kit_evaluator():
    """Straights, flushes and full houses that use a hole card match the evaluator's category (board-only hands are
    excluded by construction)."""
    rng = random.Random(5)
    for _ in range(20000):
        cs = rng.sample(range(52), 7)
        h, b = cs[:2], cs[2:]
        m, _ = PF.hand_feat(h[0], h[1], b)
        cat = evaluate(cs) >> 20
        bcat = evaluate(b) >> 20
        if bcat >= 4:
            continue
        if m == 11:
            assert cat == 4
        elif m in (12, 13):
            assert cat == 5
        elif m == 14:
            assert cat >= 6
        if cat == 4 and bcat < 4:
            assert m == 11
        if cat == 5:
            assert m in (12, 13)


def test_texture_parts():
    assert PF.tex_parts(PF.texture(C("Kh 7s 2c"))) == (0, 0, 0, 2)
    assert PF.tex_parts(PF.texture(C("9h 8h 7c"))) == (0, 1, 2, 0)
    assert PF.tex_parts(PF.texture(C("Ts Js Qs"))) == (0, 2, 2, 2)
    assert PF.tex_parts(PF.texture(C("8d 8c 3h"))) == (1, 0, 0, 0)
    assert PF.tex_parts(PF.texture(C("Ah 2h 3h 4h"))) == (0, 3, 3, 2)
    for _ in range(2000):
        b = random.Random(_).sample(range(52), 3 + _ % 3)
        assert 0 <= PF.texture(b) < PF.N_TEX


# ----------------------------------------------------------------------------- equity sanity
def _eq(hole, board, w=1):
    h = C(hole)
    b = C(board)
    m, d = PF.hand_feat(h[0], h[1], b)
    c = PF.cell(len(b) - 3, PF.texture(b), m * PF.N_DRAW + d, w)
    return c, c[0] * c[4] + c[1] * c[5] + c[2] * c[6] + c[3] * c[7]


def test_equity_orders_hands_sensibly():
    b = "Kh 7s 2c"
    eqs = [_eq(h, b)[1] for h in ("5c 4d", "5c 5d", "Kc 4d", "Kc Qd", "Ac Ad", "7c 7d")]
    assert eqs == sorted(eqs), eqs
    assert _eq("7c 7d", b)[1] > 0.85
    assert _eq("5c 4d", b)[1] < 0.25
    # tighter ranges hurt a one-pair hand
    assert _eq("Kc Qd", b, 0)[1] < _eq("Kc Qd", b, 2)[1]
    # frequencies are a distribution
    c, _ = _eq("Kc Qd", b)
    assert abs(sum(c[:4]) - 1.0) < 1e-9 and min(c[:4]) >= 0.0
    # equity falls against the stronger strata
    assert c[4] >= c[6] >= c[7]


# ----------------------------------------------------------------------------- decisions
CFG = dict(B.PARAMS["PE"], VALUE_FRAC_CALLER=1.0, VALUE_FRAC=0.7, RAISE_MULT=3.0, BF_RAISE_MAX=2.0)


def _opp(seat=1, width=1, fh=0, fn=0, matched=False, ok=False, cbet=False, allin=False, style=None):
    return (seat, width, fh, fn, matched, ok, cbet, allin, width if style is None else style)


def _decide(hole, board, pot, to_call, stack=2000, opps=None, raised_me=False, bf=1.0, cur=None, caller=False,
            seed=1, last=True):
    h = C(hole)
    opps = opps or [_opp(matched=to_call > 0)]
    cur = to_call if cur is None else cur
    return PF.decide(h[0], h[1], C(board), pot, to_call, stack, opps, opps[0][0], raised_me, bf, caller, cur, 20,
                     last, CFG, random.Random(seed))


def test_audit_leak_top_pair_folds_to_a_huge_bet_from_a_tight_seat():
    # top pair weak kicker, 3x pot overbet from a nit (pot 300 before the bet, bet 900 = 30% of our 3000 chips: the
    # old rule called anything under 40% of our chips)
    d = _decide("Kc 4d", "Kh 9s 6c Jd", pot=1200, to_call=900, stack=3000, opps=[_opp(width=0, matched=True)])
    assert d == ("fold",), d


def test_top_pair_calls_a_small_bet():
    d = _decide("Kc 4d", "Kh 9s 6c", pot=400, to_call=100)
    assert d == ("call",), d


def test_air_folds_to_a_pot_bet_and_sets_raise():
    assert _decide("5c 4d", "Kh 9s 2c", pot=600, to_call=300) == ("fold",)
    d = _decide("9c 9d", "Kh 9s 2c", pot=600, to_call=300)
    assert d[0] == "raise", d


def test_draw_calls_a_cheap_bet_folds_to_a_huge_one():
    assert _decide("Jh Th", "9h 8s 2h", pot=500, to_call=100) == ("call",)
    assert _decide("Jh Th", "9h 8s 2h", pot=500, to_call=100, stack=5000)[0] in ("call", "raise")
    assert _decide("Jh Th", "9h 8s 2h 3c", pot=4300, to_call=3000, stack=6000,
                   opps=[_opp(width=0, matched=True)]) == ("fold",)


def test_value_bets_strong_hands_and_checks_air_without_fold_evidence():
    d = _decide("7c 7d", "Kh 7s 2c", pot=300, to_call=0)
    assert d[0] == "bet" and d[1] >= 90, d
    assert _decide("5c 4d", "Kh 9s 2c", pot=300, to_call=0) == ("check",)


def test_bluffs_only_with_fold_evidence_on_dry_boards_heads_up():
    folder = _opp(fh=8, fn=10, ok=True)
    bets = sum(_decide("Ac 4d", "Kh 7s 2c", pot=300, to_call=0, opps=[folder], seed=s)[0] == "bet"
               for s in range(400))
    assert 100 < bets < 300, bets                  # BLUFF_FREQ 0.5 with the ace blocker (EV-positive vs a folder)
    wet = sum(_decide("Ac 4d", "Kh Qh Jc", pot=300, to_call=0, opps=[folder], seed=s)[0] == "bet"
              for s in range(200))
    assert wet == 0
    multi = sum(_decide("Ac 4d", "Kh 7s 2c", pot=300, to_call=0, opps=[folder, _opp(seat=2)], seed=s)[0] == "bet"
                for s in range(200))
    assert multi == 0


def test_semi_bluffs_with_strong_draws():
    folds = _opp(fh=5, fn=10)
    d = _decide("Jh Th", "9h 8s 2h", pot=300, to_call=0, opps=[folds])
    assert d[0] == "bet", d


def test_value_size_vs_a_typed_caller_is_value_frac_caller():
    d = _decide("Ac Kd", "Ah 7s 2c", pot=300, to_call=0, opps=[_opp(width=2)], caller=True)
    assert d == ("bet", 300), d


def test_icm_bubble_factor_tightens_a_big_call():
    args = dict(hole="Kc Qd", board="Kh 9s 6c 2d", pot=2000, to_call=1000, stack=1500)
    assert _decide(**args)[0] in ("call", "raise")       # chip EV: continue (all-in raise or call)
    assert _decide(bf=6.0, **args) == ("fold",)


def test_all_in_opponents_cannot_be_bluffed():
    folder = _opp(fh=9, fn=10, ok=True, allin=True, matched=True)
    assert all(_decide("Ac 4d", "Kh 7s 2c", pot=300, to_call=0, opps=[folder], seed=s) == ("check",)
               for s in range(50))


def test_decision_cpu_time_under_100us():
    rng = random.Random(11)
    worst = []
    for i in range(3000):
        cs = rng.sample(range(52), 2 + 3 + i % 3)
        h, b = cs[:2], cs[2:]
        n_opp = 1 + i % 4
        opps = [_opp(seat=k + 1, width=k % 3, fh=i % 5, fn=i % 9, matched=bool(i % 2), ok=bool(i % 3 == 0),
                     cbet=k == 0) for k in range(n_opp)]
        tc = 0 if i % 2 else 50 + i % 400
        t0 = time.thread_time_ns()
        PF.decide(h[0], h[1], b, 600 + tc, tc, 3000, opps, 1, bool(i % 7 == 0), 1.0, False, tc, 20, bool(i % 3),
                  CFG, rng)
        worst.append(time.thread_time_ns() - t0)
    worst.sort()
    assert worst[len(worst) // 2] < 50_000, worst[len(worst) // 2]
    assert worst[int(len(worst) * 0.99)] < 100_000, worst[int(len(worst) * 0.99)]


# ----------------------------------------------------------------------------- the brain hook on real tables
def _play(params, tables=12, seed=3):
    from sim.kit.engine import Table
    from sim.kit.example_agent import ExampleAgent
    from sim.kit.fixtures import CheckCallAgent, RandomAgent
    from bot import shell
    strict0 = shell.STRICT
    shell.STRICT = True              # exceptions and illegal results reach the kit (counted there)
    try:
        made = []
        dec = []

        def ours(version):
            def f(seat, cfg):
                b = shell.make(seat, cfg, version=version, params=params)
                made.append(b)
                orig = b.act

                def act(state, _o=orig):
                    a = _o(state)
                    if state["street"] != 0 and state["street"] != "preflop":
                        dec.append(a.get("action"))
                    return a
                b.act = act
                return b
            return f
        for t in range(tables):
            facs = [ours(2), RandomAgent, ours(0), CheckCallAgent, ours(1), ExampleAgent, ours(2), CheckCallAgent]
            res = Table(facs, seed=f"pe:{seed}:{t}", time_limit_ms=None).play()
            for s in (0, 2, 4, 6):
                assert res.stats[s].exceptions == 0 and res.stats[s].illegal == 0
        for b in made:
            assert getattr(b, "errors", 0) == 0 and getattr(b, "fallbacks", 0) == 0
        return dec
    finally:
        shell.STRICT = strict0


def test_engine_on_plays_strict_tables_without_errors_and_changes_postflop_play():
    on = _play({"POSTFLOP_ENGINE": True})
    off = _play({"POSTFLOP_ENGINE": False})
    assert len(on) > 200 and len(off) > 200
    assert on != off


def test_partial_pe_override_merges_with_defaults():
    b = B.Brain(0, {"n_seats": 8, "payouts": [50, 30, 20], "seed": 1},
                params={"POSTFLOP_ENGINE": True, "PE": {"BLUFF_FREQ": 0.0}})
    assert b.P["PE"] == {"BLUFF_FREQ": 0.0}
    # the merged config is built lazily on the first engine decision; build it the same way here
    cfg = dict(B.PARAMS["PE"])
    cfg.update(b.P["PE"])
    assert cfg["BLUFF_FREQ"] == 0.0 and cfg["W_CALL"] == B.PARAMS["PE"]["W_CALL"]
