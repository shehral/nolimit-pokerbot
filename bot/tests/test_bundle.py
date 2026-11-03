"""The single-file bundle (bot/dist/nolimit_bot.py, built by bot/tools/bundle.py) is current and behaves exactly like
the sources: same actions on the same tables for v0, v1 and v2. Rebuild after any source change:
    .venv/bin/python bot/tools/bundle.py
"""
import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import Table                                  # noqa: E402
from sim.kit.fixtures import AllInAgent, CheckCallAgent, RandomAgent   # noqa: E402
from sim.kit.runner import load_factory                           # noqa: E402

BOT = os.path.join(ROOT, "bot")
DIST = os.path.join(BOT, "dist", "nolimit_bot.py")


def _digest():
    src = ""
    for name in ("bot_tables", "brain", "adapter", "shell"):
        with open(os.path.join(BOT, name + ".py")) as f:
            src += f.read()
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def test_bundle_is_current():
    assert os.path.exists(DIST), "run bot/tools/bundle.py"
    with open(DIST) as f:
        head = f.read(2000)
    assert _digest() in head, "bundle is stale: run .venv/bin/python bot/tools/bundle.py"


def _play(fac, tables=24):
    log = []

    def make(seat, cfg):
        bot = fac(seat, cfg)
        orig = bot.act

        def act(state, _o=orig):
            a = _o(state)
            log.append((state["hand_no"], state["seat"], state["street"], a.get("action"), a.get("amount")))
            return a
        bot.act = act
        return bot
    opps = [RandomAgent, AllInAgent, CheckCallAgent]
    for t in range(tables):
        facs = [make if s % 2 == 0 else opps[(s + t) % 3] for s in range(8)]
        Table(facs, ("TOP3", "LINEAR", "WTA")[t % 3], "TURBO", seed=f"bundle:{t}", timing="cpu",
              time_limit_ms=None).play()
    return log


@pytest.mark.parametrize("version", [0, 1, 2])
def test_bundle_matches_sources(version):
    src = load_factory(f"bot/shell.py:BotV{version}")          # sources first (see the module-name note in bundle.py)
    bun = load_factory(f"bot/dist/nolimit_bot.py:BotV{version}")
    src_brain = sys.modules[src.__module__].brain
    bun_brain = sys.modules[bun.__module__].brain
    assert src_brain is not bun_brain
    a, b = _play(src), _play(bun)
    assert len(a) > 1000
    assert a == b
