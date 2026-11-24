"""AI-written field: opponent bots written by independent AI agents from public materials (sim/field_ai/ai_*.py,
class Bot). Each agent was asked to write the kind of bot a different kind of team might submit.

Seat occupants for the bench (sim/bench/common.py recognizes any name that starts with "ai_"):
  "ai_quant", "ai_gto_fan", ...    one AI-written bot behind a seeded timeout model (AiSeat)

Fields (seat shares, sum 100; each of the 7 opponent seats is drawn i.i.d. per table, as for the zoo mixes):
  AI_ONLY   uniform over the AI bots
  AI_MIX    50% uniform over the AI bots + 50% T6_like shares (each zoo share halved)

Loading. Each bot is loaded the way a real platform would load a team's file: by path, through the kit runner's
load_factory("sim/field_ai/<slug>.py:Bot"). A bot whose file fails to import, or whose constructor raises, is a dead
seat: the engine catches the constructor error and plays check/fold for that seat for the whole table. Exceptions
and illegal actions inside act() go to the engine unmasked (it folds or checks and counts them).

Clock. Score runs switch the kit's clock off and give each AI bot a seeded per-decision timeout draw at its MEASURED
rate (TIMEOUT_P_1_5 by default: thread-CPU enforcement at time_scale 1.5; TIMEOUT_P_1_0 for a fast sandbox). The
rates were measured once (one seat, 400 tables against T6_like, enforced CPU clock) and are kept as constants so that
a cell's clock model stays fixed. Two bots still replay inexactly: their Monte Carlo stops on a clock.
"""
import functools
import glob
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SLUGS = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HERE, "ai_*.py")))

# Measured timeout shares (timeouts / decisions of the bot's seat), CPU clock enforced, 400 tables each.
# 0.0 = no timeout seen in that run. Measured 2026-11-03, seed 7, load 13.7->9.4. Every timeout seen
# was a rare CPU spike (collector / allocation pauses: counts are not monotone in the time scale), none structural.
TIMEOUT_P_1_0 = {"ai_aggro": 0.000e+00, "ai_default_ml": 6.907e-05, "ai_frontier_careful": 3.664e-05, "ai_gto_fan": 4.168e-05, "ai_icm_plan": 0.000e+00, "ai_latency_first": 0.000e+00, "ai_poker_pro": 0.000e+00, "ai_quant": 1.875e-05, "ai_student_fast": 0.000e+00}
TIMEOUT_P_1_5 = {"ai_aggro": 0.000e+00, "ai_default_ml": 8.494e-05, "ai_frontier_careful": 1.775e-05, "ai_gto_fan": 1.994e-05, "ai_icm_plan": 0.000e+00, "ai_latency_first": 1.797e-05, "ai_poker_pro": 3.691e-05, "ai_quant": 3.627e-05, "ai_student_fast": 1.284e-04}
TIMEOUT_P_2_0 = {"ai_aggro": 7.069e-05, "ai_default_ml": 6.471e-05, "ai_frontier_careful": 1.793e-05, "ai_gto_fan": 8.093e-05, "ai_icm_plan": 0.000e+00, "ai_latency_first": 0.000e+00, "ai_poker_pro": 3.621e-05, "ai_quant": 5.560e-05, "ai_student_fast": 2.011e-05}


def is_ai(name):
    return name.partition(":")[0] in SLUGS


def bot_factory(slug):
    """The bot's class loaded by path through the kit runner (cached per process)."""
    from sim.kit.runner import load_factory
    return load_factory(os.path.join(HERE, slug + ".py") + ":Bot")


class AiSeat:
    """One AI-written bot behind a seeded timeout model: each decision times out with probability p (a random stream
    separate from the bot's own); a timed-out decision returns the rules' auto action without running the bot.
    Construction errors propagate (the engine makes the seat dead), act() errors propagate (the engine counts them)."""
    is_ai_seat = True

    def __init__(self, seat, config, slug, p=0.0):
        self.inner = bot_factory(slug)(seat, config)
        self.slug = slug
        self.p = float(p)
        self.rng = random.Random(f"aiseat:{config.get('seed', seat)}")
        self.n_decisions = 0
        self.n_timeouts = 0
        obs = getattr(self.inner, "observe", None)
        if callable(obs) and not getattr(obs, "kit_default", False):
            self.observe = obs            # instance attribute: the engine registers it only when the bot has one

    def act(self, state):
        self.n_decisions += 1
        if self.p > 0.0 and self.rng.random() < self.p:
            self.n_timeouts += 1
            return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
        return self.inner.act(state)


def factory(slug, p=0.0):
    return functools.partial(AiSeat, slug=slug, p=p)


def _t6():
    from sim.zoo import fields as ZF
    return ZF.T6_like


AI_ONLY = {s: 100.0 / len(SLUGS) for s in SLUGS}
AI_MIX = {s: 50.0 / len(SLUGS) for s in SLUGS}
for _k, _v in _t6().items():
    AI_MIX[_k] = AI_MIX.get(_k, 0.0) + _v / 2.0
MIXES = {"AI_ONLY": AI_ONLY, "AI_MIX": AI_MIX}
for _name, _m in MIXES.items():
    assert abs(sum(_m.values()) - 100.0) < 1e-9, _name


def sandbox_p(scale="1.5"):
    """The bench's zoo sandbox rates plus the AI bots' measured rates at the given time scale."""
    from sim.bench.common import SANDBOX_P
    ai = {"1.0": TIMEOUT_P_1_0, "1.5": TIMEOUT_P_1_5, "2.0": TIMEOUT_P_2_0}[str(scale)]
    out = dict(SANDBOX_P)
    out.update(ai)
    return out
