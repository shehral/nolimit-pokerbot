"""Rules s3.7: an exception, an illegal action, or a decision slower than the limit is an error; the engine folds,
or checks when checking is legal. Nothing is repaired."""
import pytest

from sim.kit.agent import Agent, validate_action
from sim.kit.engine import Table
from sim.kit.fixtures import (BadInitAgent, BusyAgent, CheckCallAgent, ExplodingAgent, FoldAlwaysAgent,
                              IllegalRaiseAgent, RecorderAgent, SleepyAgent)


def table_with(bad_cls, seat=0, n=8, max_hands=12, **kw):
    facs = [CheckCallAgent] * n
    facs[seat] = bad_cls
    return Table(facs, "TOP3", "TURBO", seed=f"strict-{bad_cls.__name__}", max_hands=max_hands, record_events=True,
                 **kw).play()


def seat_actions(res, seat):
    return [ev for only, ev in res.events if ev["type"] == "action" and ev["seat"] == seat]


def assert_auto_fold_or_check(res, seat):
    acts = seat_actions(res, seat)
    assert acts, "the bot never got a decision"
    for ev in acts:
        assert ev["action"] == ("check" if ev["to_call"] == 0 else "fold"), ev
    return acts


def test_sleeping_bot_is_auto_folded():
    res = table_with(SleepyAgent)
    st = res.stats[0]
    assert st.decisions > 0 and st.timeouts == st.decisions
    assert st.exceptions == 0 and st.illegal == 0 and st.n_call == 0 and st.n_raise == 0
    assert st.lat_max_ns >= 2_000_000
    assert_auto_fold_or_check(res, 0)
    assert "act() took" in st.examples["timeout"][0]


def test_exception_bot_is_auto_folded():
    res = table_with(ExplodingAgent)
    st = res.stats[0]
    assert st.decisions > 0 and st.exceptions == st.decisions and st.timeouts == 0 and st.illegal == 0
    assert_auto_fold_or_check(res, 0)
    assert st.examples["exception"][0].startswith("RuntimeError: boom")


def test_illegal_raise_is_auto_folded():
    res = table_with(IllegalRaiseAgent)
    st = res.stats[0]
    assert st.decisions > 0 and st.illegal == st.decisions and st.exceptions == 0 and st.timeouts == 0
    assert st.n_raise == 0
    assert_auto_fold_or_check(res, 0)


def test_checking_is_used_instead_of_folding_when_free():
    # an erroring big blind after everyone limps must CHECK (not fold), and so on every free street
    res = table_with(ExplodingAgent, seat=0, n=3, max_hands=30)
    acts = assert_auto_fold_or_check(res, 0)
    checks = [ev for ev in acts if ev["action"] == "check"]
    assert checks, "expected at least one free spot (BB option or postflop) for the erroring seat"
    assert res.stats[0].n_check == len(checks)


def test_fold_when_check_is_free_is_illegal_and_becomes_check():
    res = table_with(FoldAlwaysAgent, seat=0, n=3, max_hands=30)
    st = res.stats[0]
    acts = assert_auto_fold_or_check(res, 0)
    free = [ev for ev in acts if ev["to_call"] == 0]
    assert free and st.illegal == len(free)                 # each free 'fold' was refused and checked
    assert st.decisions == len(acts)


def test_bot_that_fails_to_construct_errors_every_decision():
    res = table_with(BadInitAgent)
    st = res.stats[0]
    assert st.init_exceptions == 1 and st.decisions > 0 and st.exceptions == st.decisions
    assert_auto_fold_or_check(res, 0)
    assert "cannot construct" in st.examples["init"][0]


def test_time_scale_simulates_a_slower_sandbox():
    # 100 us of real work x 20 = 2 ms > 1 ms: every decision is a timeout
    res = table_with(BusyAgent, time_scale=20.0, extra_config={"busy_us": 100})
    st = res.stats[0]
    assert st.decisions > 0 and st.timeouts == st.decisions


def test_no_time_limit_when_disabled():
    res = table_with(SleepyAgent, time_limit_ms=None)
    st = res.stats[0]
    assert st.timeouts == 0 and st.n_call + st.n_check == st.decisions


def test_errors_never_leak_between_seats():
    res = table_with(ExplodingAgent)
    for s in range(1, 8):
        st = res.stats[s]
        assert st.exceptions == st.timeouts == st.illegal == 0


class ObserveRaises(CheckCallAgent):
    def observe(self, event):
        raise KeyError("bad observe")


def test_observe_exceptions_are_counted_but_do_not_fold():
    res = table_with(ObserveRaises)
    st = res.stats[0]
    assert st.observe_exceptions == st.observe_calls > 0
    assert st.exceptions == 0 and st.n_fold == 0     # a check/call bot whose act() is fine never folds


@pytest.mark.parametrize("ret,why", [
    (None, "not a dict"),
    ("call", "not a dict"),
    ({"act": "call"}, "not in legal_actions"),
    ({"action": "CALL"}, "not in legal_actions"),
    ({"action": "bet", "amount": 100}, "not in legal_actions"),
    ({"action": "raise"}, "not an integer"),
    ({"action": "raise", "amount": "100"}, "not an integer"),
    ({"action": "raise", "amount": 100.5}, "not an integer"),
    ({"action": "raise", "amount": True}, "bool"),
    ({"action": "raise", "amount": 39}, "outside"),
    ({"action": "raise", "amount": 2001}, "outside"),
    ({"action": "check"}, "not in legal_actions"),
])
def test_validate_action_rejects(ret, why):
    kind, amt, reason = validate_action(ret, ("fold", "call", "raise"), 40, 2000)
    assert kind is None and why in reason


def test_validate_action_accepts_legal_forms():
    assert validate_action({"action": "raise", "amount": 40}, ("fold", "call", "raise"), 40, 2000)[:2] == ("raise", 40)
    assert validate_action({"action": "raise", "amount": 2000.0}, ("fold", "call", "raise"), 40, 2000)[:2] == \
        ("raise", 2000)
    assert validate_action({"action": "call", "amount": 999}, ("fold", "call"), 0, 0)[:2] == ("call", 0)
    assert validate_action({"action": "fold"}, ("fold", "call"), 0, 0)[:2] == ("fold", 0)


class Weird(Agent):
    """Returns objects that are hostile to validation (comparison raises)."""

    class Bad:
        def __eq__(self, other):
            raise ValueError("no comparisons")
        __hash__ = object.__hash__

    def act(self, state):
        return {"action": self.Bad()}


def test_hostile_return_value_cannot_crash_the_engine():
    res = table_with(Weird)
    st = res.stats[0]
    assert st.illegal == st.decisions > 0
    assert_auto_fold_or_check(res, 0)


def test_cpu_timing_splits_wall_noise_from_bot_work():
    # sleeping uses no CPU: over the limit on the wall clock only (counted as 'preempted'-style noise)
    res = table_with(SleepyAgent)
    st = res.stats[0]
    assert st.wall_over == st.decisions and st.cpu_over == 0 and st.preempted == st.decisions
    # with timing='cpu' the same bot is not penalized
    res = table_with(SleepyAgent, timing="cpu")
    st = res.stats[0]
    assert st.timeouts == 0 and st.wall_over == st.decisions and st.n_call + st.n_check == st.decisions
    # busy-waiting burns CPU: over on both clocks, a timeout in both modes
    res = table_with(BusyAgent, timing="cpu", time_scale=20.0, extra_config={"busy_us": 100})
    st = res.stats[0]
    assert st.timeouts == st.decisions == st.cpu_over and st.preempted == 0


def test_bad_timing_mode_rejected():
    with pytest.raises(ValueError):
        Table([CheckCallAgent] * 2, timing="gpu")
