"""Tests for the second-pass fixes: P15 fallback, no reference cycles, every PARAMS key used, the BF guard."""
import gc
import os
import random
import re
import sys
import weakref

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.runner import load_factory                          # noqa: E402

from bot.tests.scenarios import eight_bb_vs_jam, eight_first_in, run_spot   # noqa: E402

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
B = SHELL.brain
BOT_DIR = os.path.join(ROOT, "bot")
LABELS = ["AA", "TT", "55", "AKo", "A5o", "K9s", "QJo", "J8s", "T9s", "72o", "93o", "K2o"]


def test_p15_exception_in_optional_work_returns_the_v0_decision(monkeypatch):
    """An exception inside v2-only or ICM-only code (here: typed_jam and icm_jam raise) must give the v0 decision
    for the same spot, not check/fold; the kit sees a legal action and our counters record the recovery."""
    monkeypatch.setattr(SHELL, "STRICT", False)
    fac2 = load_factory("bot/shell.py:BotV2")
    made = []

    def broken(seat, cfg):
        bot = fac2(seat, cfg)

        def boom(*a, **k):
            raise RuntimeError("injected failure in optional work")
        bot._brain.typed_jam = boom          # instance attributes shadow the methods
        bot._brain.icm_jam = boom
        made.append(bot)
        return bot
    for spot in (eight_first_in(1, 10), eight_first_in(4, 14)):
        d0 = run_spot(spot, 0, labels=LABELS)
        made.clear()
        db = run_spot(spot, 2, labels=LABELS, factory=broken)
        for lab in LABELS:
            assert db[lab][1] == d0[lab][1], (lab, db[lab][1], d0[lab][1])
        assert any(d0[lab][1]["action"] == "raise" for lab in LABELS)      # the spot has jams, not only folds
        assert made and all(b.fallbacks == 1 and b.errors == 1 for b in made)


def test_p15_fallback_failure_still_check_folds(monkeypatch):
    monkeypatch.setattr(SHELL, "STRICT", False)
    bot = load_factory("bot/shell.py:BotV2")(0, {"n_seats": 8, "payouts": [50, 30, 20], "seed": 1})

    def boom(*a, **k):
        raise RuntimeError("injected")
    bot._brain.decide = boom
    bot._brain.decide_fallback = boom
    st = SHELL._synthetic_states(random.Random(3), 1)[0]
    out = bot.act(st)
    assert out == ({"action": "check"} if "check" in st["legal_actions"] else {"action": "fold"})
    assert bot.errors == 1 and bot.fallbacks == 0


def test_brain_is_freed_by_reference_counting_alone():
    """No reference cycles in our per-table objects: with the collector off, dropping the bot frees its brain
    and tracker (a cycle would leak one brain per table in a long-lived process that rarely runs full collections)."""
    fac = load_factory("bot/shell.py:Bot")                # the base class honours config["bot_version"]
    was = gc.isenabled()
    gc.disable()
    try:
        for ver in (0, 1, 2):
            bot = fac(0, {"n_seats": 8, "payouts": [50, 30, 20], "seed": 2, "bot_version": ver})
            assert bot._brain.version == ver
            wb = weakref.ref(bot._brain)                       # the tracker (slots, no weakref) lives only on it
            for st in SHELL._synthetic_states(random.Random(ver), 40):
                bot._brain.seat = st["seat"]
                bot.act(st)
            del bot
            assert wb() is None, ver
    finally:
        if was:
            gc.enable()


def _code_without(src, start_marker, end_marker):
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[:i] + src[j + len(end_marker):]


def test_every_params_key_is_used():
    """Each PARAMS key is read by code (not only declared), so no tunable is silently inert."""
    brain_src = open(os.path.join(BOT_DIR, "brain.py")).read()
    brain_code = _code_without(brain_src, "PARAMS = {", "\n}\n")
    shell_src = open(os.path.join(BOT_DIR, "shell.py")).read()
    shell_code = _code_without(shell_src, "_SHELL_DEFAULTS = {", "}\n")
    unused = []
    for k in B.PARAMS:
        pat = re.compile(r'\[\s*"%s"\s*\]|\.get\(\s*"%s"' % (k, k))
        in_brain = bool(pat.search(brain_code))
        in_shell = k in SHELL._SHELL_DEFAULTS and bool(pat.search(shell_code))
        if not (in_brain or in_shell):
            unused.append(k)
    assert not unused, unused


def test_shell_reads_its_settings_from_brain_params():
    for k in SHELL._SHELL_DEFAULTS:
        assert k in B.PARAMS, k
        assert SHELL.SHELL_PARAMS[k] == B.PARAMS[k], k


def test_bf_guard_uses_the_measured_cost_and_falls_back_to_the_table():
    b = B.Brain(0, {"n_seats": 8, "payouts": [40, 30, 20, 10], "seed": 1}, version=1)   # exact (non-affine) ICM
    assert b.icm_on and b.bf_cost_ns > 0
    assert b.bf_start_limit == B.PARAMS["GUARD_NS"] - B.PARAMS["BF_COST_SAFETY"] * b.bf_cost_ns
    # a zero budget: no exact BF can start, the equal-stack table supplies the value (never 1.0 while ICM is on)
    spot = eight_bb_vs_jam(1, 10)
    base = run_spot(spot, 1, labels=LABELS)
    made = []

    def tight(seat, cfg):
        bot = SHELL.make(seat, cfg, version=1, params={"GUARD_NS": 0})
        made.append(bot)
        return bot
    d = run_spot(spot, 1, labels=LABELS, factory=tight)
    for lab in LABELS:                          # equal stacks: the table BF equals the exact BF, same actions
        assert d[lab][1] == base[lab][1], lab
    assert all(m._brain.bf_computed == 0 and m._brain.guard_trips >= 1 for m in made)
