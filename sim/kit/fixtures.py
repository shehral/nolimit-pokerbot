"""Test fixtures for the kit: cheap agents for speed runs and deliberately faulty agents for the strictness tests.
These are NOT the field zoo (sim/zoo/); they exist to exercise the engine.
"""
import random
import time

try:
    from .agent import Agent
except ImportError:
    from agent import Agent


class CheckFoldAgent(Agent):
    """Check when free, else fold (what a crashed bot becomes under the auto-fold rule)."""

    def act(self, state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}


class CheckCallAgent(Agent):
    """Calling station: check when free, else call. Never raises."""

    def act(self, state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "call"}


class AllInAgent(Agent):
    """Always puts the whole stack in."""

    def act(self, state):
        if "raise" in state["legal_actions"]:
            return {"action": "raise", "amount": state["max_raise_to"]}
        return {"action": "call"} if "call" in state["legal_actions"] else {"action": "check"}


class RandomAgent(Agent):
    """Uniform over legal actions; raise sizes uniform between min and max raise-to (seeded per table)."""

    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.rng = random.Random(config["seed"])

    def act(self, state):
        legal = state["legal_actions"]
        a = legal[int(self.rng.random() * len(legal))]
        if a == "raise":
            lo, hi = state["min_raise_to"], state["max_raise_to"]
            return {"action": "raise", "amount": lo + int(self.rng.random() * (hi - lo + 1))}
        return {"action": a}


class PassiveRandomAgent(Agent):
    """Mostly checks/calls, sometimes folds, raises 10% (small raises): long tables, many showdowns."""

    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.rng = random.Random(config["seed"])

    def act(self, state):
        legal = state["legal_actions"]
        u = self.rng.random()
        if "raise" in legal and u < 0.10:
            lo, hi = state["min_raise_to"], state["max_raise_to"]
            return {"action": "raise", "amount": min(hi, lo + int(self.rng.random() * lo))}
        if "check" in legal:
            return {"action": "check"}
        if u < 0.35:
            return {"action": "fold"}
        return {"action": "call"}


class SleepyAgent(Agent):
    """Wants to call/check but sleeps 2 ms first: every decision is a timeout."""
    SLEEP_S = 0.002

    def act(self, state):
        time.sleep(self.SLEEP_S)
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "call"}


class BusyAgent(Agent):
    """Busy-waits config['busy_us'] microseconds (default 600) then check/calls: a bot that is close to the limit."""

    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.busy_ns = int(config.get("busy_us", 600) * 1000)

    def act(self, state):
        end = time.perf_counter_ns() + self.busy_ns
        while time.perf_counter_ns() < end:
            pass
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "call"}


class ExplodingAgent(Agent):
    """Raises an exception on every decision."""

    def act(self, state):
        raise RuntimeError("boom")


class IllegalRaiseAgent(Agent):
    """Raises one chip below the minimum when raising is legal, else raises anyway (not legal), else calls."""

    def act(self, state):
        if "raise" in state["legal_actions"] and state["min_raise_to"] < state["max_raise_to"]:
            return {"action": "raise", "amount": state["min_raise_to"] - 1}
        return {"action": "raise", "amount": state["max_raise_to"] + 1}


class FoldAlwaysAgent(Agent):
    """Returns 'fold' even when checking is free: illegal there (fold is offered only facing a bet)."""

    def act(self, state):
        return {"action": "fold"}


class BadInitAgent(Agent):
    def __init__(self, seat, config):
        raise ValueError("cannot construct")


class ScriptedAgent(Agent):
    """Plays a fixed script and records every state it sees (tests). script items: dicts returned as-is, or
    callables state -> action. When the script runs out it check/folds."""

    def __init__(self, seat, config, script=(), log=None):
        super().__init__(seat, config)
        self.script = list(script)
        self.states = log if log is not None else []
        self.events = []

    def act(self, state):
        self.states.append(state)
        if not self.script:
            return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
        item = self.script.pop(0)
        return item(state) if callable(item) else item

    def observe(self, event):
        self.events.append(event)


class RecorderAgent(CheckCallAgent):
    """Check/call agent that records all events it observes (tests of the public event stream)."""

    def __init__(self, seat, config, sink=None):
        super().__init__(seat, config)
        self.events = sink if sink is not None else []

    def observe(self, event):
        self.events.append(event)


class WorkAgent(Agent):
    """CPU-bound: config['work_evals'] 7-card evaluations per decision (default 290, about 0.6 ms on an M4
    performance core), then check/call. Models a Monte Carlo bot trimmed to sit just under the limit."""

    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.k = int(config.get("work_evals", 290))
        self.cards = [(i * 7) % 52 for i in range(7)]

    def act(self, state):
        try:
            from .cards import evaluate
        except ImportError:
            from cards import evaluate
        c = self.cards
        acc = 0
        for _ in range(self.k):
            acc ^= evaluate(c)
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "call"}
