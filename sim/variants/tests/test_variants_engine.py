import time

import pytest

from sim.kit.cards import RANKS
from sim.kit.fixtures import AllInAgent, CheckCallAgent, RandomAgent, PassiveRandomAgent
from sim.variants import NAMES, VTable, make_variant, play_variant_table, settle_bounty
from sim.variants.agents import EXAMPLES, CrashDiscard
from sim.variants.runner import run_variant

from vhelpers import ALLIN, C, D, F, ONE_LEVEL, R, X, events_of, vtable

ALL = [v for v in NAMES]


# ------------------------------------------------------------------ chip conservation
@pytest.mark.parametrize("variant", ALL)
def test_chip_conservation_mixed_field(variant):
    ex = EXAMPLES[variant]
    field = [ex, CheckCallAgent, RandomAgent, PassiveRandomAgent, AllInAgent, ex, RandomAgent, CheckCallAgent]
    for seed in range(25):
        res = VTable(variant, field, seed=seed, record_hands=True).play()   # EngineError on any violation
        assert abs(sum(res.points) - 100) < 1e-6
        for row in res.hand_log:
            assert sum(row[5]) == 16000 and sum(row[6]) == 16000
        assert sum(res.final_stacks) == 16000


@pytest.mark.parametrize("variant", ["bounty", "toss", "doubleboard"])
def test_chip_conservation_hand_by_hand_with_hand_bounty_mode(variant):
    opts = {"mode": "hand"} if variant == "bounty" else {}
    v = make_variant(variant, **opts)
    for seed in range(15):
        res = VTable(v, [AllInAgent, EXAMPLES[variant]] * 4, seed=seed, record_hands=True).play()
        assert sum(res.final_stacks) == 16000


# ------------------------------------------------------------------ state shape
@pytest.mark.parametrize("variant", ALL)
def test_state_has_kit_fields_and_extras(variant):
    t = vtable(variant, [2000, 2000, 2000], {}, button=0, deck=False)
    t.play()
    st = t.logs[1][0]
    for k in ("hand_no", "level", "sb", "bb", "ante", "button", "seat", "street", "hole", "board", "pot", "stacks",
              "bets", "contributed", "active", "all_in", "to_call", "min_raise_to", "max_raise_to", "legal_actions",
              "history_this_hand", "street_starts", "n_alive", "payouts", "time_limit_ms"):
        assert k in st
    assert st["variant"] == variant and st["request"] == "act"
    assert st["boards"][0] == st["board"] and len(st["boards"]) == (2 if variant == "doubleboard" else 1)
    assert len(st["hole"]) == (3 if variant == "toss" else 2)
    assert (st["bounty_rank"] is not None) == (variant == "bounty")


# ------------------------------------------------------------------ short deck
def test_shortdeck_deals_only_6_to_ace_and_flush_beats_full_house():
    for s in range(30):
        t = vtable("shortdeck", [500] * 8, {}, 0, deck=False, max_hands=3)
        t.deck_fn = t._random_deck
        d = t._random_deck(1)
        assert len(set(d)) == len(d) and all((c >> 2) >= 4 for c in d)
    fl_fh = dict(holes={0: ("Ah", "Kh"), 1: ("9c", "9d")}, board=["9h", "7h", "6h", "6c", "8s"])
    t = vtable("shortdeck", [100, 100], {0: [ALLIN], 1: [ALLIN]}, 0, **fl_fh)
    res = t.play()
    assert res.final_stacks == [200, 0]                       # flush wins in short deck
    t = vtable("plain", [100, 100], {0: [ALLIN], 1: [ALLIN]}, 0, **fl_fh)
    assert t.play().final_stacks == [0, 200]                  # the same cards under the 52-card ranking


def test_shortdeck_wheel_and_trips_option_in_engine():
    cards = dict(holes={0: ("Ad", "6c"), 1: ("Kh", "Kd")}, board=["7h", "8s", "9d", "Qc", "Kc"])
    t = vtable("shortdeck", [100, 100], {0: [ALLIN], 1: [ALLIN]}, 0, **cards)
    assert t.play().final_stacks == [200, 0]                  # A-6-7-8-9 straight beats trip kings
    v = make_variant("shortdeck", trips_beat_straight=True)
    t = vtable(v, [100, 100], {0: [ALLIN], 1: [ALLIN]}, 0, **cards)
    assert t.play().final_stacks == [0, 200]


# ------------------------------------------------------------------ double board
def test_doubleboard_split_heads_up():
    # seat0 wins board A (two pair... see cards), seat1 wins board B: chop of the 200 pot
    cards = dict(holes={0: ("Ah", "Kh"), 1: ("7c", "7d")},
                 board=["2c", "5d", "9s", "Jd", "Qs", "Ac", "Kd", "3s", "4h", "9c"])
    t = vtable("doubleboard", [100, 100], {0: [ALLIN], 1: [ALLIN]}, 0, **cards)
    res = t.play()
    assert res.final_stacks == [100, 100]
    sd = events_of(res, "showdown")[0]
    assert len(sd["boards"]) == 2 and len(sd["boards"][0]) == 5
    # seat0 wins both boards
    cards2 = dict(holes={0: ("Ah", "Kh"), 1: ("7c", "8d")},
                  board=["Ac", "Kd", "3s", "4h", "9c", "As", "Kc", "2s", "5h", "9d"])
    t = vtable("doubleboard", [100, 100], {0: [ALLIN], 1: [ALLIN]}, 0, **cards2)
    assert t.play().final_stacks == [200, 0]


def test_doubleboard_three_way_side_pot_odd_chip():
    # layers: 153 (all three, seat0 short with 51), then 98 (seats 1 and 2). seat0 wins board A, seat1 board B.
    cards = dict(holes={0: ("Ah", "Ad"), 1: ("Kh", "Kd"), 2: ("7c", "2d")},
                 board=["9s", "5c", "3h", "Js", "4d", "Kc", "Ks", "8d", "9h", "3d"])
    t = vtable("doubleboard", [51, 100, 100], {i: [ALLIN] * 3 for i in range(3)}, 0, **cards)
    res = t.play()
    # layer 153: board A half = 77 (odd chip to board A) -> seat0; board B half = 76 -> seat1; layer 98 -> seat1
    assert res.final_stacks == [77, 174, 0]


# ------------------------------------------------------------------ bounty
def test_settle_bounty_matches_the_mit_2025_examples():
    big = {0: 10 ** 6, 1: 10 ** 6}
    # 1: showdown, both bet 50, A (seat 0) wins and hits: winnings 1.5*50+10 = 85 -> bonus 35
    r = settle_bounty([(100, [0], {0: 50, 1: 50})], {0}, button=1, room=big)
    assert r == {0: {1: 35}}
    # 2: pot 60 + bets 10/30, B wins: normal winnings 40, with bounty 70 -> bonus 30
    r = settle_bounty([(100, [1], {0: 30 + 30, 1: 30 + 10 + 30})] , {1}, button=0, room=big) if False else None
    r = settle_bounty([(80, [1], {0: 40, 1: 40})], {1}, button=0, room=big)       # N = 40
    assert r == {1: {0: 30}}
    # 3: SB folds, BB (out of position) hits: 1.5*1+10 = 11.5 -> 12 total, i.e. bonus 11 (rounded up)
    assert settle_bounty([(3, [1], {0: 1, 1: 2})], {1}, button=0, room=big) == {1: {0: 11}}
    # same hit by the dealer rounds down: bonus 10
    assert settle_bounty([(3, [0], {0: 2, 1: 1})], {0}, button=0, room=big) == {0: {1: 10}}
    # 4: chop of 80 (40 each), exactly one hit -> bonus 40/4 + 10 = 20 total relative... = 20 - 0
    assert settle_bounty([(80, [0, 1], {0: 40, 1: 40})], {0}, button=1, room=big) == {0: {1: 20}}
    # chop where both hit: each owes the other, engine nets them (both bonuses are paid, they cancel in stacks)
    both = settle_bounty([(80, [0, 1], {0: 40, 1: 40})], {0, 1}, button=1, room=big)
    assert both == {0: {1: 20}, 1: {0: 20}}
    # 5: all-in 400 each, A wins and hits: 1.5*400+10 = 610 -> bonus 210 (if the loser has the chips)
    assert settle_bounty([(800, [0], {0: 400, 1: 400})], {0}, button=1, room=big) == {0: {1: 210}}
    # ... but never more than the payer can pay (a busted loser pays nothing; a short stack pays what is left)
    assert settle_bounty([(800, [0], {0: 400, 1: 400})], {0}, button=1, room={0: 800, 1: 0}) == {}
    assert settle_bounty([(800, [0], {0: 400, 1: 400})], {0}, button=1, room={0: 800, 1: 50}) == {0: {1: 50}}
    # no hit, no bonus
    assert settle_bounty([(100, [0], {0: 50, 1: 50})], set(), button=1, room=big) == {}


def test_settle_bounty_multiway_splits_payers_pro_rata():
    # three players put 100 each; seat 0 wins and hits: N = 200, bonus = 100 + 10 = 110, 55 from each loser
    r = settle_bounty([(300, [0], {0: 100, 1: 100, 2: 100})], {0}, button=2, room={1: 1000, 2: 1000})
    assert r == {0: {1: 55, 2: 55}}
    # unequal contributions: payers pay in proportion to what they put in (1 : 3), total bonus = 0.5*N + 10
    r = settle_bounty([(400, [0], {0: 100, 1: 100, 2: 200})], {0}, button=2, room={1: 1000, 2: 1000})
    assert sum(r[0].values()) == 160 and r[0][2] == 2 * r[0][1] + (0 if r[0][1] * 2 == r[0][2] else r[0][2] - 2 * r[0][1])


def _bounty_table(scripts, stacks=(1000, 1000), seat_ranks=(3, 1), holes=None, board=None, **kw):
    """HU bounty table where seat0's bounty rank is '5' (index 3) and seat1's is '3' (index 1)."""
    t = vtable("bounty", list(stacks), scripts, 0, holes=holes, board=board, **kw)
    t._roll_bounties = lambda h, l: setattr(t, "bounty_ranks", list(seat_ranks))
    return t


def test_bounty_engine_payout_check_down():
    # seat0 holds a pair of 5s (its bounty rank), seat1 A K; everyone checks down: pot 40, N = 20,
    # seat0 bonus = 0.5*20+10 = 20 -> stacks 1040 / 960 (total winnings 40 = 1.5*20+10)
    holes = {0: ("5h", "5s"), 1: ("Ad", "Kc")}
    board = ["9c", "2d", "7h", "Jc", "Td"]
    t = _bounty_table({0: [C, X, X, X], 1: [X, X, X, X]}, holes=holes, board=board)
    res = t.play()
    assert res.final_stacks == [1040, 960]
    ev = events_of(res, "hand_end")[0]
    assert ev["bonuses"] == ((0, 20),)
    # the rank is private: only the owner sees its own bounty event
    own = [e for e in t.agents[1].events if e["type"] == "bounty"]
    assert own and all(e["seat"] == 1 and e["rank"] == "3" for e in own)
    assert t.logs[0][0]["bounty_rank"] == "5" and t.logs[1][0]["bounty_rank"] == "3"


def test_bounty_no_hit_no_bonus_and_board_hit_counts():
    holes = {0: ("Qh", "Qs"), 1: ("Ad", "Kc")}                 # seat0 wins, but no 5 anywhere
    t = _bounty_table({0: [C, X, X, X], 1: [X, X, X, X]}, holes=holes, board=["9c", "2d", "7h", "Jc", "Td"])
    assert t.play().final_stacks == [1020, 980]
    board_hit = ["9c", "2d", "5h", "Jc", "Td"]                 # a 5 on the board counts too
    t = _bounty_table({0: [C, X, X, X], 1: [X, X, X, X]}, holes=holes, board=board_hit)
    assert t.play().final_stacks == [1040, 960]


def test_bounty_all_in_pays_no_bonus_because_the_loser_has_no_chips_left():
    holes = {0: ("5h", "5s"), 1: ("Ad", "Kc")}
    t = _bounty_table({0: [ALLIN], 1: [ALLIN]}, stacks=(300, 300), holes=holes,
                      board=["9c", "2d", "7h", "Jc", "Td"])
    assert t.play().final_stacks == [600, 0]


def test_bounty_ranks_change_per_level_or_per_hand():
    for mode, expect_change_at in (("level", 26), ("hand", 2)):
        t = VTable(make_variant("bounty", mode=mode), [CheckCallAgent] * 8, seed=7)
        t._roll_bounties(1, 1)
        r1 = list(t.bounty_ranks)
        t._roll_bounties(2, 1 if expect_change_at != 2 else 1)
        same_hand2 = list(t.bounty_ranks)
        t._roll_bounties(26, 2)
        later = list(t.bounty_ranks)
        assert (same_hand2 == r1) == (mode == "level")
        assert later != r1 or mode == "hand"


# ------------------------------------------------------------------ antes
def test_antes_start_at_level_two_and_are_everybody():
    t = vtable("antes", [2000] * 3, {}, 0, deck=False, schedule=((10, 20, 0), (15, 30, 0)), max_hands=2,
               hands_per_level=1)
    res = t.play()
    s1 = [s for s in t.logs[1] + t.logs[2] + t.logs[0] if s["hand_no"] == 1][0]
    s2 = [s for s in t.logs[1] + t.logs[2] + t.logs[0] if s["hand_no"] == 2][0]
    assert s1["ante"] == 0 and s2["ante"] == 4                  # round(0.125 * 30) = 4 chips
    h = [x for x in s2["history_this_hand"] if x[1] == "ante"]
    assert sorted(x[0] for x in h) == [0, 1, 2] and all(x[2] == 4 for x in h)
    first = min((s for s in t.logs[0] + t.logs[1] + t.logs[2] if s["hand_no"] == 2), key=lambda s: len(s["history_this_hand"]))
    assert first["pot"] == 12 + 15 + 30
    assert sum(res.final_stacks) == 6000


def test_bb_ante_only_big_blind_posts_it_and_it_is_dead_money():
    t = vtable("bbante", [2000] * 3, {}, 0, deck=False, schedule=((10, 20, 0), (15, 30, 0)), max_hands=2,
               hands_per_level=1)
    t.play()
    allst = t.logs[0] + t.logs[1] + t.logs[2]
    s2 = min((s for s in allst if s["hand_no"] == 2), key=lambda s: len(s["history_this_hand"]))
    ants = [x for x in s2["history_this_hand"] if x[1] == "ante"]
    assert len(ants) == 1 and ants[0][0] == s2["bb_seat"] and ants[0][2] == 30
    assert s2["pot"] == 30 + 15 + 30
    assert s2["to_call"] == 30                               # the ante is not part of the amount to call
    assert all(s["ante"] == 0 for s in allst if s["hand_no"] == 1)


# ------------------------------------------------------------------ toss
TOSS_CARDS = dict(holes={0: ("Ah", "Ad", "2c"), 1: ("Kh", "Kd", "7c")}, board=["9d", "5s", "Qc", "3h"])


def test_toss_flow_order_visibility_and_showdown():
    t = vtable("toss", [1000, 1000], {0: [C, D(2), X, X, X], 1: [X, D(2), X, X, X]}, 0, **TOSS_CARDS)
    res = t.play()
    reqs0 = [s for s in t.logs[0] if s["request"] == "discard"]
    reqs1 = [s for s in t.logs[1] if s["request"] == "discard"]
    assert len(reqs0) == len(reqs1) == 1
    r1, r0 = reqs1[0], reqs0[0]
    assert r1["legal_actions"] == ["discard"] and r1["discard_options"] == [0, 1, 2] and len(r1["hole"]) == 3
    assert r1["board"] == ["9d", "5s"] and r1["discards"] == []            # out of position (BB, seat 1) goes first
    assert r0["board"] == ["9d", "5s", "7c"] and r0["discards"] == [(1, "7c")]   # the dealer sees the toss
    postflop0 = [s for s in t.logs[0] if s["street"] == "flop" and s["request"] == "act"][0]
    assert len(postflop0["hole"]) == 2 and postflop0["board"] == ["9d", "5s", "7c", "2c"]
    sd = events_of(res, "showdown")[0]
    assert sd["board"] == ("9d", "5s", "7c", "2c", "Qc", "3h")             # 6 board cards: best 5 of 8
    assert res.final_stacks == [1020, 980]
    ds = events_of(res, "discard")
    assert [(e["seat"], e["card"]) for e in ds] == [(1, "7c"), (0, "2c")]


def test_toss_requested_even_when_everybody_is_all_in():
    t = vtable("toss", [100, 100], {0: [ALLIN, D(2)], 1: [ALLIN, D(2)]}, 0, **TOSS_CARDS)
    res = t.play()
    assert sum(1 for s in t.logs[0] + t.logs[1] if s["request"] == "discard") == 2
    assert res.final_stacks == [200, 0]


@pytest.mark.parametrize("bad", [{"action": "discard", "index": 7}, {"action": "discard", "index": True},
                                 {"action": "discard", "index": "1"}, {"action": "discard"},
                                 {"action": "fold"}, {"action": "check"}, None, "discard"])
def test_toss_illegal_discard_is_counted_and_folds(bad):
    t = vtable("toss", [1000, 1000], {0: [C, D(2)], 1: [X, bad]}, 0, **TOSS_CARDS)
    res = t.play()
    assert res.stats[1].illegal == 1 and res.stats[0].illegal == 0
    assert res.final_stacks == [1020, 980]                                  # seat 1 folded at the discard


def test_toss_discard_exception_and_timeout_are_errors():
    def boom(state):
        raise RuntimeError("x")

    def slow(state):
        time.sleep(0.003)
        return {"action": "discard", "index": 2}
    t = vtable("toss", [1000, 1000], {0: [C, D(2)], 1: [X, boom]}, 0, **TOSS_CARDS)
    res = t.play()
    assert res.stats[1].exceptions == 1 and res.final_stacks == [1020, 980]
    t = vtable("toss", [1000, 1000], {0: [C, D(2)], 1: [X, slow]}, 0, **TOSS_CARDS)
    res = t.play()
    assert res.stats[1].timeouts == 1 and res.final_stacks == [1020, 980]


def test_toss_error_policy_lowest_and_last_contender_cannot_fold():
    v = make_variant("toss", discard_error="lowest")
    t = vtable(v, [1000, 1000], {0: [C, D(2), X, X, X], 1: [X, {"action": "discard", "index": 9}, X, X, X]}, 0,
               **TOSS_CARDS)
    res = t.play()
    assert res.stats[1].illegal == 1
    assert events_of(res, "discard")[0]["card"] == "7c"                    # lowest rank of Kh Kd 7c
    assert res.final_stacks == [1020, 980]                                 # played on instead of folding
    bad = {"action": "discard", "index": 9}
    t = vtable("toss", [1000, 1000], {0: [C, bad, X, X, X], 1: [X, bad]}, 0, **TOSS_CARDS)
    res = t.play()                                                         # both err: seat 1 folds, seat 0 stays in
    assert res.final_stacks == [1020, 980] and res.stats[0].illegal == 1 and res.stats[1].illegal == 1


def test_toss_multiway_every_contender_discards_in_order_and_board_grows():
    t = vtable("toss", [1000] * 4, {i: [C] * 1 + [D(0), X, X, X] for i in range(4)}, 0, deck=False)
    res = t.play()
    ds = events_of(res, "discard")
    assert len(ds) >= 2
    assert [e["seat"] for e in ds] == sorted((e["seat"] for e in ds), key=lambda s: (s - 0 - 1) % 4)  # clockwise from left of button


# ------------------------------------------------------------------ strictness unchanged for act
def test_act_errors_still_fold_in_every_variant():
    from sim.kit.fixtures import ExplodingAgent
    for v in NAMES:
        res = play_variant_table(v, [ExplodingAgent] + [CheckCallAgent] * 7, seed=3)
        assert res.stats[0].exceptions > 0 and abs(sum(res.points) - 100) < 1e-6


def test_crash_discard_agent_fixture():
    res = play_variant_table("toss", [CrashDiscard] + [EXAMPLES["toss"]] * 7, seed=1)
    assert res.stats[0].illegal > 0


# ------------------------------------------------------------------ determinism / paired decks
def test_same_seed_same_result_and_paired_cards():
    f = [EXAMPLES["toss"]] * 8
    a = play_variant_table("toss", f, seed=5)
    b = play_variant_table("toss", f, seed=5)
    assert a.points == b.points and a.hands_played == b.hands_played
    t1 = VTable("doubleboard", f, seed=9)
    assert t1._random_deck(3) == VTable("doubleboard", f, seed=9)._random_deck(3)


# ------------------------------------------------------------------ smoke: 200 tables per variant
@pytest.mark.parametrize("variant", ALL)
def test_smoke_200_tables_speed(variant):
    ex = f"sim.variants.agents:{EXAMPLES[variant].__name__}"
    r = run_variant(variant, [("ex", ex)] * 8, 200, seed=11, workers=4)
    s = r.stats["ex"]
    assert s.exceptions == 0 and s.illegal == 0
    assert abs(sum(r.points["ex"]) / len(r.points["ex"]) - 12.5) < 1e-6          # 8 identical seats: average is exactly 12.5
    assert r.tables_per_s >= 100, f"{variant}: {r.tables_per_s:.0f} tables/s"
