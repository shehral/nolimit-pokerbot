"""Betting and pot rules: side pots, min-raise/reopen, heads-up blinds, BB option, refunds, short blinds, antes,
odd chips, and chip conservation every hand over many random tables."""
import pytest

from helpers import ALLIN, C, F, R, X, events_of, scripted_table
from sim.kit.engine import Table
from sim.kit.fixtures import AllInAgent, CheckCallAgent, PassiveRandomAgent, RandomAgent
from sim.kit.example_agent import ExampleAgent

NO_STRAIGHT_BOARD = ["2c", "7d", "9h", "Jc", "4s"]


# ---------------------------------------------------------------- side pots
def test_three_way_all_in_side_pot_and_refund():
    # button = seat 2: SB = 0 (100 chips), BB = 1 (300), button = 2 (500); preflop order 2, 0, 1
    res, logs = scripted_table(
        [100, 300, 500], {2: [ALLIN], 0: [C], 1: [C]}, button=2,
        holes={0: ("As", "Ah"), 1: ("Ks", "Kh"), 2: ("Qs", "Qh")}, board=NO_STRAIGHT_BOARD)
    # main pot 3 x 100 -> seat 0; side pot 2 x 200 -> seat 1; seat 2's uncalled 200 comes back
    assert res.final_stacks == [300, 400, 200]
    refunds = events_of(res, "refund")
    assert [(e["seat"], e["amount"]) for e in refunds] == [(2, 200)]
    assert logs[0][0]["to_call"] == 90 and logs[0][0]["legal_actions"] == ["fold", "call"]   # covers only 90


def test_side_pot_with_folded_dead_money():
    # 4 seats, button 3: SB 0, BB 1 (200 chips); preflop order 2, 3, 0, 1
    scripts = {2: [R(100), C], 3: [C, F], 0: [F], 1: [ALLIN]}
    holes = {0: ("2d", "3d"), 1: ("As", "Ah"), 2: ("Ks", "Kh"), 3: ("5c", "6c")}
    res, logs = scripted_table([1000, 200, 1000, 1000], scripts, button=3, holes=holes, board=NO_STRAIGHT_BOARD)
    # pot 510 = 10 (SB dead) + 200 + 200 + 100 (seat 3 dead); seat 1 wins it all
    assert res.final_stacks == [990, 510, 800, 900]
    assert sum(res.final_stacks) == 3200
    # seat 2 faced a FULL re-raise (100 -> 200, increment 100 >= 80) and so could raise again
    s2 = logs[2][1]
    assert s2["legal_actions"] == ["fold", "call", "raise"] and s2["min_raise_to"] == 300


def test_side_pot_second_best_wins_side():
    # the short stack wins the main pot, the bigger of the other two wins the side pot
    holes = {0: ("Qs", "Qh"), 1: ("As", "Ah"), 2: ("Ks", "Kh")}
    res, _ = scripted_table([500, 100, 500], {2: [ALLIN], 0: [C], 1: [C]}, button=2, holes=holes,
                            board=NO_STRAIGHT_BOARD)
    # main 3 x 100 -> seat 1; side 2 x 400 -> seat 2 (KK beats QQ)
    assert res.final_stacks == [0, 300, 800]
    assert res.places[0] == 3 and res.bust_hand[0] == 1


def test_split_pot_odd_chip_goes_left_of_button():
    # ante 1: pot = 3 antes + 20 + 20 = 43, split two ways: 22 to seat 0 (first left of the button), 21 to seat 1
    board = ["As", "Ks", "Qs", "Js", "Ts"]        # royal flush on board: every contender ties
    holes = {0: ("2c", "3d"), 1: ("4c", "5d"), 2: ("6c", "7d")}
    res, _ = scripted_table([2000, 2000, 2000], {2: [F], 0: [C, X, X, X], 1: [X, X, X, X]}, button=2,
                            holes=holes, board=board, schedule=((10, 20, 1),))
    assert res.final_stacks == [2001, 2000, 1999]


# ---------------------------------------------------------------- min-raise and reopening
def test_short_all_in_does_not_reopen():
    # button 2 opens to 100 (increment 80), SB calls, BB (150 chips) shoves 150: a short raise (+50 < 80)
    res, logs = scripted_table([2000, 150, 2000], {2: [R(100), C], 0: [C, C], 1: [ALLIN]}, button=2)
    bb = logs[1][0]
    assert bb["to_call"] == 80 and bb["min_raise_to"] == 150 and bb["max_raise_to"] == 150   # all-in for less
    assert "raise" in bb["legal_actions"]
    again = logs[2][1]
    assert again["legal_actions"] == ["fold", "call"] and again["to_call"] == 50
    assert logs[0][1]["legal_actions"] == ["fold", "call"]


def test_full_raise_reopens_and_sets_min_raise():
    # BB re-raises to 180 (+80 = a full raise): the opener may raise again, minimum 180 + 80 = 260
    res, logs = scripted_table([2000, 2000, 2000], {2: [R(100), C], 0: [C, C], 1: [R(180)]}, button=2)
    again = logs[2][1]
    assert again["legal_actions"] == ["fold", "call", "raise"] and again["min_raise_to"] == 260
    assert logs[0][0]["min_raise_to"] == 180                  # SB facing the open: 100 + 80


def test_cumulative_short_all_ins_reopen():
    # 5 seats, button 4: seat 2 opens 100 (increment 80); seat 3 shoves 150 (+50, short); seat 4 shoves 210
    # (+60, short); the SB calls. Seat 2 now faces +110 >= one full raise (80) since it acted: raising is open
    # again (TDA-style cumulative rule), minimum 210 + 80 = 290.
    scripts = {2: [R(100), C], 3: [ALLIN], 4: [ALLIN], 0: [C], 1: [F]}
    res, logs = scripted_table([2000, 2000, 2000, 150, 210], scripts, button=4)
    s4 = logs[4][0]
    assert s4["min_raise_to"] == 210 and s4["max_raise_to"] == 210 and "raise" in s4["legal_actions"]
    again = logs[2][1]
    assert again["legal_actions"] == ["fold", "call", "raise"] and again["min_raise_to"] == 290


def test_no_raise_when_everyone_else_is_all_in_or_folded():
    # same shoves, but the SB is the short stack and the BB folds: nobody could answer a raise from seat 2
    scripts = {2: [R(100), C], 3: [ALLIN], 0: [ALLIN], 1: [F]}
    res, logs = scripted_table([210, 2000, 2000, 150], scripts, button=3)
    sb = logs[0][0]
    assert sb["min_raise_to"] == 210 and sb["max_raise_to"] == 210
    again = logs[2][1]
    assert again["legal_actions"] == ["fold", "call"] and again["to_call"] == 110


def test_min_raise_tracks_largest_raise():
    res, logs = scripted_table([2000] * 4, {2: [R(100)], 3: [R(300)], 0: [F], 1: [F]}, button=3)
    assert logs[3][0]["min_raise_to"] == 180              # 100 + 80
    assert logs[2][1]["min_raise_to"] == 500              # 300 + the 200 increment


def test_illegal_raise_size_is_refused_not_repaired():
    # a raise to 30 is below the minimum raise-to of 40: refused (counted illegal), never bumped up to 40
    res, logs = scripted_table([2000, 2000, 2000], {2: [R(30)]}, button=2)
    assert res.stats[2].illegal == 1
    assert events_of(res, "action")[0]["action"] == "fold"     # facing the blind: auto-fold, never a min-raise


# ---------------------------------------------------------------- blinds and positions
def test_heads_up_button_posts_sb_acts_first_preflop_last_postflop():
    res, logs = scripted_table([2000, 2000], {0: [C, X], 1: [X, X]}, button=0)
    first = logs[0][0]
    assert first["street"] == "preflop" and first["sb_seat"] == 0 and first["bb_seat"] == 1
    assert first["to_call"] == 10 and first["bets"] == [10, 20]
    acts = [(e["seat"], e["street"]) for e in events_of(res, "action")]   # BB acts first postflop heads-up
    assert acts[:4] == [(0, "preflop"), (1, "preflop"), (1, "flop"), (0, "flop")]


def test_bb_option_check_or_raise_after_limps():
    res, logs = scripted_table([2000] * 3, {2: [C], 0: [C], 1: [X]}, button=2)
    bb = logs[1][0]
    assert bb["legal_actions"] == ["check", "raise"] and bb["to_call"] == 0 and bb["min_raise_to"] == 40


def test_postflop_open_bet_minimum_is_big_blind():
    res, logs = scripted_table([2000] * 3, {2: [C], 0: [C], 1: [X, X]}, button=2)
    flop = [s for s in logs[0] if s["street"] == "flop"][0]
    assert flop["legal_actions"] == ["check", "raise"] and flop["min_raise_to"] == 20 and flop["to_call"] == 0
    assert flop["board"] == logs[0][-1]["board"][:3] and len(flop["board"]) == 3


def test_no_raise_offered_when_nobody_can_respond():
    # heads-up: SB shoves, the BB may only fold or call
    res, logs = scripted_table([2000, 2000], {0: [ALLIN], 1: [C]}, button=0)
    assert logs[1][0]["legal_actions"] == ["fold", "call"] and logs[1][0]["max_raise_to"] == 0


def test_all_in_for_less_than_the_call():
    # button 0 opens to 400; the SB (seat 1) has 50 chips: to_call is capped at its 40 behind and it cannot raise
    holes = {0: ("Ks", "Kh"), 1: ("As", "Ah"), 2: ("2d", "3c")}
    res, logs = scripted_table([2000, 50, 2000], {0: [R(400)], 1: [C], 2: [F]}, button=0, holes=holes,
                               board=NO_STRAIGHT_BOARD)
    s = logs[1][0]
    assert s["to_call"] == 40 and s["legal_actions"] == ["fold", "call"] and s["max_raise_to"] == 0
    # the opener's 350 above the SB's 50 comes back; the SB wins 50 + 50 + the BB's dead 20
    assert [(e["seat"], e["amount"]) for e in events_of(res, "refund")] == [(0, 350)]
    assert res.final_stacks == [1950, 120, 1980]


def test_short_big_blind_uses_nominal_big_blind():
    # BB has 5 chips: posts 5 all-in; UTG must still call the full 20
    res, logs = scripted_table([2000, 5, 2000, 2000], {2: [C], 3: [F], 0: [F]}, button=3)
    utg = logs[2][0]
    assert utg["to_call"] == 20 and utg["bets"][1] == 5 and utg["all_in"][1]


def test_raise_everyone_folds_refund_event():
    res, logs = scripted_table([2000] * 3, {2: [R(60)], 0: [F], 1: [F]}, button=2)
    ref = events_of(res, "refund")
    assert [(e["seat"], e["amount"]) for e in ref] == [(2, 40)]
    assert res.final_stacks == [1990, 1980, 2030]


def test_antes_are_dead_money():
    res, logs = scripted_table([2000] * 3, {2: [C], 0: [C], 1: [X]}, button=2, schedule=((10, 20, 5),))
    s = logs[2][0]
    assert s["pot"] == 15 + 30 and s["to_call"] == 20 and s["contributed"] == [15, 25, 5]


def test_button_moves_and_blinds_level_up():
    res = Table([CheckCallAgent] * 3, "TOP3", "TURBO", seed=4, record_hands=True, max_hands=60).play()
    log = res.hand_log
    assert [h[2] for h in log[:26]] == [20] * 25 + [30]
    buttons = [h[4] for h in log[:6]]
    assert buttons == [(buttons[0] + k) % 3 for k in range(6)]


def test_tied_bust_same_starting_stack_split_places():
    holes = {0: ("As", "Ah"), 1: ("2s", "3h"), 2: ("2d", "3c")}
    res, _ = scripted_table([2000, 500, 500], {1: [ALLIN], 2: [ALLIN], 0: [C, C]}, button=0, holes=holes,
                            board=NO_STRAIGHT_BOARD)
    # seats 1 and 2 bust in the same hand with equal starting stacks: they share places 2 and 3 (30 + 20) / 2
    assert res.places == [1, 2.5, 2.5] and res.points == [50.0, 25.0, 25.0]


def test_bust_order_by_starting_stack():
    holes = {0: ("As", "Ah"), 1: ("2s", "3h"), 2: ("2d", "3c")}
    res, _ = scripted_table([2000, 600, 400], {1: [ALLIN], 2: [ALLIN], 0: [C, C]}, button=0, holes=holes,
                            board=NO_STRAIGHT_BOARD)
    assert res.places == [1, 2, 3] and res.points == [50.0, 30.0, 20.0]


# ---------------------------------------------------------------- conservation over many tables
@pytest.mark.parametrize("field", ["random", "mixed", "passive"])
def test_chip_conservation_every_hand(field):
    fields = {
        "random": [RandomAgent] * 8,
        "mixed": [RandomAgent, AllInAgent, CheckCallAgent, ExampleAgent, PassiveRandomAgent, RandomAgent,
                  ExampleAgent, CheckCallAgent],
        "passive": [PassiveRandomAgent] * 8,
    }
    hands = 0
    for t in range(60):
        res = Table(fields[field], "TOP3", ["TURBO", "FAST", "SLOW"][t % 3], seed=f"cons{t}",
                    record_hands=True).play()
        for (_h, _sb, _bb, _ante, _b, before, after) in res.hand_log:
            assert sum(before) == 16000 and sum(after) == 16000 and min(after) >= 0
            hands += 1
        assert abs(sum(res.points) - 100.0) < 1e-9
        assert abs(sum(res.places) - 36) < 1e-9
    assert hands > 500
