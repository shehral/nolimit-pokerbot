"""Legality tests for the bot's adapter + shell (SPEC s3; about 15 direct cases before v0).

Every engine-based case plays a scripted table on the mock kit with our bot in one seat and checks our action
with the kit's own validate_action (the exact rule the engine applies), plus the specific expectation of the case.
Run: .venv/bin/python -m pytest -q bot/tests
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.agent import validate_action                       # noqa: E402
from sim.kit.engine import Table, make_deck                     # noqa: E402
from sim.kit.fixtures import ScriptedAgent                      # noqa: E402
from sim.kit.runner import load_factory                         # noqa: E402

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
ADAPTER = SHELL.adapter
BOTS = {v: load_factory(f"bot/shell.py:BotV{v}") for v in (0, 1, 2)}
ONE_LEVEL = ((10, 20, 0),)
F = {"action": "fold"}
C = {"action": "call"}
X = {"action": "check"}


def R(x):
    return {"action": "raise", "amount": x}


def ALLIN(state):
    if "raise" in state["legal_actions"]:
        return {"action": "raise", "amount": state["max_raise_to"]}
    return {"action": "call"}


def legal(state, action):
    kind, amt, reason = validate_action(action, state["legal_actions"], state["min_raise_to"], state["max_raise_to"])
    return kind is not None


def play(stacks, scripts, button, me, holes=None, board=None, version=2, schedule=ONE_LEVEL, max_hands=1, **kw):
    """Scripted table with our bot in seat `me`; returns [(state, action)] for every decision our bot made."""
    n = len(stacks)
    deck = make_deck(n, holes or {}, board or [])
    seen = []
    fac = BOTS[version]

    def make(seat, cfg):
        if seat == me:
            bot = fac(seat, cfg)
            orig = bot.act

            def act(state, _o=orig):
                a = _o(state)
                seen.append((state, a))
                return a
            bot.act = act
            return bot
        return ScriptedAgent(seat, cfg, list(scripts.get(seat, [])))
    t = Table([make] * n, kw.pop("payouts", "TOP3"), schedule, seed="legal", stacks=stacks, button=button,
              max_hands=max_hands, deck_fn=lambda h: deck, timing="cpu", **kw)
    res = t.play()
    for st, a in seen:
        assert legal(st, a), (st["legal_actions"], st["min_raise_to"], st["max_raise_to"], a)
    return seen, res


# ------------------------------------------------------------------ 1. raise-to semantics (the kit's)
@pytest.mark.parametrize("version", [0, 1, 2])
def test_raise_amount_is_a_street_total(version):
    # deep stacks (100 bb); we hold AA on the button facing nothing: we open; the amount must be raise-TO
    stacks = [2000] * 4
    seen, _ = play(stacks, {0: [F], 1: [F]}, button=3, me=3, holes={3: ("Ah", "Ad")}, version=version)
    st, a = seen[0]
    assert a["action"] == "raise"
    assert isinstance(a["amount"], int)
    assert st["min_raise_to"] <= a["amount"] <= st["max_raise_to"]
    assert a["amount"] == 50          # OPEN_BB 2.5 x 20 = 50 total (not 50 on top of the 20)


# ------------------------------------------------------------------ 2. raise-by / add semantics in the adapter
def test_adapter_raise_by_and_add_conversion():
    state = {"seat": 1, "legal_actions": ["fold", "call", "raise"], "bets": [100, 20, 0], "min_raise_to": 180,
             "max_raise_to": 2000, "to_call": 80}
    old = ADAPTER.ADAPTER_PARAMS["RAISE_SEMANTICS"]
    try:
        ADAPTER.ADAPTER_PARAMS["RAISE_SEMANTICS"] = "to"
        assert ADAPTER.to_action(("raise", 300), state) == {"action": "raise", "amount": 300}
        # "by": bounds and amount are increases over the current bet (100)
        ADAPTER.ADAPTER_PARAMS["RAISE_SEMANTICS"] = "by"
        st_by = dict(state, min_raise_to=80, max_raise_to=1900)
        assert ADAPTER.to_action(("raise", 300), st_by) == {"action": "raise", "amount": 200}
        assert ADAPTER.to_action(("allin",), st_by) == {"action": "raise", "amount": 1900}
        # "add": amounts are chips added now (our bet is 20)
        ADAPTER.ADAPTER_PARAMS["RAISE_SEMANTICS"] = "add"
        st_add = dict(state, min_raise_to=160, max_raise_to=1980)
        assert ADAPTER.to_action(("raise", 300), st_add) == {"action": "raise", "amount": 280}
    finally:
        ADAPTER.ADAPTER_PARAMS["RAISE_SEMANTICS"] = old


# ------------------------------------------------------------------ 3. min-raise after a short all-in (no reopen)
@pytest.mark.parametrize("version", [0, 1, 2])
def test_short_all_in_does_not_reopen_us(version):
    # 3 seats, button 2: we (button, AA, 100 bb) open to 50 (increment 30); SB calls; BB (70 chips) shoves 70
    # (+20 < 30, a short all-in): we may only fold or call now; a raise/jam decision must come out as a call
    seen, _ = play([2000, 70, 2000], {0: [C, C], 1: [ALLIN]}, button=2, me=2, holes={2: ("As", "Ah")},
                   version=version)
    assert seen[0][1] == {"action": "raise", "amount": 50}
    st, a = seen[1]
    assert st["legal_actions"] == ["fold", "call"]
    assert a == {"action": "call"}


def test_cumulative_short_all_ins_reopen_and_min_raise_is_respected():
    # 5 seats, button 4: seat 2 (us, AA) opens 50 (increment 30); seat 3 shoves 70, seat 4 shoves 90 (two short
    # raises, +20 each, that add up to a full raise since we acted); SB calls, BB folds: raising is open again
    stacks = [2000, 2000, 2000, 70, 90]
    seen, _ = play(stacks, {3: [ALLIN], 4: [ALLIN], 0: [C], 1: [F]}, button=4, me=2, holes={2: ("As", "Ah")})
    assert seen[0][1] == {"action": "raise", "amount": 50}
    st, a = seen[1]
    assert "raise" in st["legal_actions"] and st["min_raise_to"] == 120     # 90 + the last full increment (30)
    assert a["action"] in ("raise", "call")
    if a["action"] == "raise":
        assert st["min_raise_to"] <= a["amount"] <= st["max_raise_to"]


# ------------------------------------------------------------------ 4. all-in for less than a call
@pytest.mark.parametrize("version", [0, 1, 2])
def test_all_in_for_less_than_a_call(version):
    # we (BB, 100 chips) hold AA facing a 600 jam: to_call is capped at our stack; a jam decision = call
    seen, _ = play([2000, 100, 2000], {2: [ALLIN], 0: [F]}, button=2, me=1, holes={1: ("As", "Ah")},
                   version=version)
    st, a = seen[0]
    assert st["to_call"] == 80 and "raise" not in st["legal_actions"]
    assert a == {"action": "call"}


# ------------------------------------------------------------------ 5. heads-up blinds
@pytest.mark.parametrize("version", [0, 1, 2])
def test_heads_up_button_posts_small_blind_and_acts_first(version):
    seen, _ = play([400, 400], {1: [F]}, button=0, me=0, holes={0: ("Ks", "Kh")}, version=version)
    st, a = seen[0]
    assert st["sb_seat"] == 0 and st["bb_seat"] == 1 and st["to_call"] == 10
    # 20 bb deep, KK raises. TOP3 pays places, so the 2026-11-03 payout tuning applies (PF_DEPTH_BB 12): a 2.5 bb
    # open; under winner-take-all PF_DEPTH_BB stays 20 and KK jams
    tuned = SHELL.brain.PARAMS["TUNE_BY_PAYOUT"]
    assert a["action"] == "raise" and a["amount"] == (2.5 * st["bb"] if tuned else st["max_raise_to"])
    seen, _ = play([400, 400], {1: [F]}, button=0, me=0, holes={0: ("Ks", "Kh")}, version=version, payouts="WTA")
    st, a = seen[0]
    assert st["sb_seat"] == 0 and st["to_call"] == 10
    assert a["action"] == "raise" and a["amount"] == st["max_raise_to"]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_heads_up_big_blind_checks_option_with_trash(version):
    seen, _ = play([2000, 2000], {0: [C]}, button=0, me=1, holes={1: ("7d", "2c")}, version=version)
    st, a = seen[0]
    assert st["legal_actions"] == ["check", "raise"]
    assert a == {"action": "check"}


# ------------------------------------------------------------------ 6. checking option in the BB, multiway
@pytest.mark.parametrize("version", [0, 1, 2])
def test_bb_option_after_limps_check_or_value_raise(version):
    stacks = [2000] * 4
    seen, _ = play(stacks, {2: [C], 3: [C], 0: [C]}, button=3, me=1, holes={1: ("8d", "3c")}, version=version)
    assert seen[0][1] == {"action": "check"}
    seen, _ = play(stacks, {2: [C], 3: [C], 0: [C]}, button=3, me=1, holes={1: ("Ad", "Ac")}, version=version)
    st, a = seen[0]
    assert a["action"] == "raise" and st["min_raise_to"] <= a["amount"] <= st["max_raise_to"]


# ------------------------------------------------------------------ 7. never fold when checking is free
def test_fold_decision_becomes_check_when_free():
    state = {"seat": 0, "legal_actions": ["check", "raise"], "bets": [0, 0], "min_raise_to": 20,
             "max_raise_to": 1000}
    assert ADAPTER.to_action(("fold",), state) == {"action": "check"}
    assert ADAPTER.to_action(("call",), state) == {"action": "check"}
    assert ADAPTER.to_action(("check",), dict(state, legal_actions=["fold", "call", "raise"])) == {"action": "fold"}


# ------------------------------------------------------------------ 8. raise when nobody can respond
@pytest.mark.parametrize("version", [0, 1, 2])
def test_no_raise_when_everyone_else_is_all_in(version):
    # seat 2 jams 300 (covered by us); the SB folds; we (BB, AA, 2000) can only call: the jam decision -> call
    seen, _ = play([2000, 2000, 300], {2: [ALLIN], 0: [F]}, button=2, me=1, holes={1: ("As", "Ah")},
                   version=version)
    st, a = seen[0]
    assert "raise" not in st["legal_actions"]
    assert a == {"action": "call"}


# ------------------------------------------------------------------ 9. bet into an unopened postflop pot
@pytest.mark.parametrize("version", [0, 1, 2])
def test_postflop_bet_respects_min_bet_and_is_a_total(version):
    # limped pot: the button limps, the SB completes, we (BB, 7d 2c) check the option; the flop 7h 2s Kd gives us
    # two pair with both hole cards (class 3); the SB checks: our bet must be a legal total >= the min bet (bb)
    holes = {1: ("7d", "2c"), 0: ("4d", "5c"), 2: ("3d", "8c")}
    seen, _ = play([2000, 2000, 2000], {2: [C, F], 0: [C, X, F]}, button=2, me=1, holes=holes,
                   board=["7h", "2s", "Kd", "9c", "Jd"], version=version)
    assert seen[0][1] == {"action": "check"}
    post = [(st, a) for st, a in seen if st["street"] != "preflop"]
    assert post, "we should reach the flop"
    st, a = post[0]
    assert st["street"] == "flop" and st["to_call"] == 0
    assert a["action"] == "raise" and a["amount"] >= st["min_raise_to"] >= 20
    assert a["amount"] == int(0.70 * 60)                       # VALUE_FRAC x pot (60 = three 20s)


# ------------------------------------------------------------------ 10. clamps: above max, below min, snap
def test_clamp_above_max_below_min_and_allin_snap():
    state = {"seat": 0, "legal_actions": ["fold", "call", "raise"], "bets": [0, 100], "min_raise_to": 200,
             "max_raise_to": 1000}
    assert ADAPTER.to_action(("raise", 5000), state) == {"action": "raise", "amount": 1000}
    assert ADAPTER.to_action(("raise", 120), state) == {"action": "raise", "amount": 200}
    assert ADAPTER.to_action(("raise", 900), state) == {"action": "raise", "amount": 1000}    # >= 85% of max
    assert ADAPTER.to_action(("raise", 400.7), state) == {"action": "raise", "amount": 400}
    assert ADAPTER.to_action(("allin",), state) == {"action": "raise", "amount": 1000}


def test_raise_decision_without_raise_option_calls_or_checks():
    st_call = {"seat": 0, "legal_actions": ["fold", "call"], "bets": [0, 100], "min_raise_to": 0, "max_raise_to": 0}
    st_check = {"seat": 0, "legal_actions": ["check"], "bets": [0, 0], "min_raise_to": 0, "max_raise_to": 0}
    for dec in (("raise", 500), ("allin",)):
        assert ADAPTER.to_action(dec, st_call) == {"action": "call"}
        assert ADAPTER.to_action(dec, st_check) == {"action": "check"}


# ------------------------------------------------------------------ 11. calling exactly our stack
@pytest.mark.parametrize("version", [0, 1, 2])
def test_call_that_exactly_covers_our_stack(version):
    # BB has 400 (posted 20, 380 behind); button jams 2000: to_call == 380 == our stack
    seen, _ = play([2000, 400, 2000], {2: [ALLIN], 0: [F]}, button=2, me=1, holes={1: ("Ks", "Kh")},
                   version=version)
    st, a = seen[0]
    assert st["to_call"] == st["stacks"][1]
    assert a == {"action": "call"}


# ------------------------------------------------------------------ 12. a short big blind (all-in from posting)
@pytest.mark.parametrize("version", [0, 1, 2])
def test_short_big_blind_all_in_from_posting(version):
    # button 3: SB 0, BB 1 with 15 chips (< 20) is all-in from the post; we are UTG (seat 2) with QQ:
    # the nominal bb (20) is still the amount to match
    seen, _ = play([2000, 15, 2000, 2000], {3: [F], 0: [F]}, button=3, me=2, holes={2: ("Qs", "Qh")},
                   version=version)
    assert seen[0][0]["all_in"][1]
    st, a = seen[0]
    assert st["to_call"] == 20
    assert a["action"] in ("raise", "call", "fold")


# ------------------------------------------------------------------ 13. shell: exceptions and bad states
def test_shell_returns_check_or_fold_on_brain_exception(monkeypatch):
    bot = BOTS[2](0, {"n_seats": 2, "payouts": [50, 30], "seed": 1})

    def boom(view, t0):
        raise RuntimeError("injected")
    monkeypatch.setattr(bot._brain, "decide", boom)
    monkeypatch.setattr(SHELL, "STRICT", False)
    st = {"seat": 0, "legal_actions": ["check", "raise"]}
    assert bot.act(dict(st, stacks=[100, 100], hole=["Ah", "Kd"], bb=20, min_raise_to=40,
                        max_raise_to=100)) == {"action": "check"}
    assert bot.act({"seat": 0, "legal_actions": ["fold", "call"]}) == {"action": "fold"}   # malformed state
    assert bot.act(None) == {"action": "fold"}                                               # garbage state
    assert bot.errors == 3


def test_shell_replaces_an_illegal_result(monkeypatch):
    bot = BOTS[0](0, {"n_seats": 2, "payouts": [50, 30], "seed": 1})
    monkeypatch.setattr(SHELL, "STRICT", False)
    monkeypatch.setattr(SHELL.adapter, "to_action", lambda dec, state, snap=0.85: {"action": "raise", "amount": -5})
    seen, _ = (None, None)
    st = {"seat": 0, "hand_no": 1, "street": "preflop", "stacks": [390, 380], "bets": [10, 20],
          "contributed": [10, 20], "start_stacks": [400, 400], "active": [True, True], "all_in": [False, False],
          "alive": [True, True], "hole": ["Ah", "Kd"], "board": [], "sb": 10, "bb": 20, "ante": 0, "pot": 30,
          "button": 0, "sb_seat": 0, "bb_seat": 1, "to_call": 10, "min_raise_to": 40, "max_raise_to": 400,
          "legal_actions": ["fold", "call", "raise"], "history_this_hand": [(0, "post_sb", 10), (1, "post_bb", 20)],
          "street_starts": [0], "payouts": [50, 30], "n_alive": 2}
    assert bot.act(st) == {"action": "fold"}
    assert bot.illegal_fixed == 1


def test_shell_survives_a_failed_brain_import(monkeypatch):
    monkeypatch.setattr(SHELL, "brain", None)
    monkeypatch.setattr(SHELL, "STRICT", False)
    bot = SHELL.Bot(0, {"n_seats": 2})
    assert bot._brain is None
    assert bot.act({"legal_actions": ["check", "raise"]}) == {"action": "check"}
    assert bot.act({"legal_actions": ["fold", "call", "raise"]}) == {"action": "fold"}


# ------------------------------------------------------------------ 14. card formats
def test_card_parsing_formats():
    p = ADAPTER.parse_card
    assert p("Ah") == p("AH") == p("ah") == 12 * 4 + 2
    assert p("10s") == p("Ts") == 8 * 4 + 3
    assert p("2c") == 0 and p(51) == 51
    with pytest.raises(Exception):
        p("Xx")


# ------------------------------------------------------------------ 15. fuzz: random tables, every action legal
@pytest.mark.parametrize("version", [0, 1, 2])
def test_fuzz_every_action_legal_against_random_opponents(version):
    from sim.kit.fixtures import RandomAgent, AllInAgent, CheckCallAgent
    fac = BOTS[version]
    errors = []

    def make_ours(seat, cfg):
        bot = fac(seat, cfg)
        orig = bot.act

        def act(state, _o=orig):
            a = _o(state)
            if not legal(state, a):
                errors.append((state, a))
            return a
        bot.act = act
        return bot
    opp = [RandomAgent, AllInAgent, CheckCallAgent]
    for t in range(60):
        facs = [make_ours if s % 2 == 0 else opp[(s + t) % 3] for s in range(8)]
        stacks = None
        if t % 3 == 0:
            stacks = [((t * 37 + s * 101) % 1900) + 30 for s in range(8)]
        res = Table(facs, ("TOP3", "LINEAR", "WTA")[t % 3], ("TURBO", "FAST", "SLOW")[t % 3], seed=f"fuzz{t}",
                    stacks=stacks, timing="cpu").play()
        for s in range(0, 8, 2):
            assert res.stats[s].exceptions == 0 and res.stats[s].illegal == 0
    assert not errors, errors[:3]
