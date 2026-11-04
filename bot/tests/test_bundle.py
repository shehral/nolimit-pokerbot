"""The single-file bundle (bot/dist/nolimit_bot.py, built by bot/tools/bundle.py) is current and behaves exactly like
the sources: same actions on the same tables for v0, v1 and v2 on the plain kit and on the toss variant; and a copy
of the bundle alone in an empty directory (no bot/, no variant/, no tables) imports, plays the built-in NLHE tables
and answers toss requests legally. Rebuild after any source change:
    .venv/bin/python bot/tools/bundle.py
"""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import Table                                  # noqa: E402
from sim.kit.fixtures import AllInAgent, CheckCallAgent, RandomAgent   # noqa: E402
from sim.kit.runner import load_factory                           # noqa: E402
from bot.tools.bundle import sources_digest                       # noqa: E402

BOT = os.path.join(ROOT, "bot")
DIST = os.path.join(BOT, "dist", "nolimit_bot.py")
DET = {"GUARD_NS": 10 ** 15}      # no clock-dependent BF skipping, so two runs are comparable under any load


def test_bundle_is_current():
    assert os.path.exists(DIST), "run bot/tools/bundle.py"
    with open(DIST) as f:
        head = f.read(2000)
    assert sources_digest() in head, "bundle is stale: run .venv/bin/python bot/tools/bundle.py"


def _wrap(make, version, log):
    def fac(seat, cfg):
        bot = make(seat, cfg, version=version, params=DET)
        orig = bot.act

        def act(state, _o=orig):
            a = _o(state)
            log.append((state["hand_no"], state["seat"], state["street"], state.get("request"), tuple(a.items())))
            return a
        bot.act = act
        return bot
    return fac


def _play(make, version, tables=24):
    log = []
    fac = _wrap(make, version, log)
    opps = [RandomAgent, AllInAgent, CheckCallAgent]
    for t in range(tables):
        facs = [fac if s % 2 == 0 else opps[(s + t) % 3] for s in range(8)]
        Table(facs, ("TOP3", "LINEAR", "WTA")[t % 3], "TURBO", seed=f"bundle:{t}", timing="cpu",
              time_limit_ms=None).play()
    return log


def _play_toss(make, version, tables=12):
    from sim.variants.agents import TossExample
    from sim.variants.engine import VTable
    from sim.variants.spec import make_variant
    log = []
    fac = _wrap(make, version, log)
    v = make_variant("toss")
    for t in range(tables):
        facs = [fac if s % 2 == 0 else TossExample for s in range(8)]
        VTable(v, facs, seed=f"btoss:{t}", timing="cpu", time_limit_ms=None).play()
    return log


def _makes():
    src = load_factory("bot/shell.py:Bot")          # sources first (see the module-name note in bundle.py)
    bun = load_factory("bot/dist/nolimit_bot.py:Bot")
    sm, bm = sys.modules[src.__module__], sys.modules[bun.__module__]
    assert sm.brain is not bm.brain
    return sm.make, bm.make


@pytest.mark.parametrize("version", [0, 1, 2])
def test_bundle_matches_sources(version):
    s, b = _makes()
    a, c = _play(s, version), _play(b, version)
    assert len(a) > 1000
    assert a == c


def test_bundle_matches_sources_on_toss():
    s, b = _makes()
    a, c = _play_toss(s, 2), _play_toss(b, 2)
    assert sum(1 for r in a if r[3] == "discard") > 50
    assert a == c


def test_bundle_alone_in_an_empty_directory(tmp_path):
    shutil.copy(DIST, tmp_path / "nolimit_bot.py")
    code = (
        "import importlib.util as u, sys\n"
        "s = u.spec_from_file_location('b', 'nolimit_bot.py'); m = u.module_from_spec(s); s.loader.exec_module(m)\n"
        "assert m.VLAYER is not None and m.VLAYER.__name__ == 'nolimit_variant', m.VLAYER\n"
        "assert m.brain._SEAM is None and 'built-in' in m.brain.TABLES_INFO, m.brain.TABLES_INFO\n"
        "assert 'warmup_ms' in m.WARMUP_INFO, m.WARMUP_INFO\n"
        "b = m.Bot(0, {'n_seats': 8, 'payouts': [50, 30, 20]})\n"
        "st = {'request': 'discard', 'hole': ['Ah', 'Ad', '7c'], 'board': ['Ks', '2d'], 'legal_actions': ['discard'],\n"
        "      'discard_options': [0, 1, 2], 'pot': 60, 'stacks': [2000] * 8, 'seat': 0}\n"
        "out = b.act(st)\n"
        "assert out == {'action': 'discard', 'index': 2}, out\n"
        "print('ok', m.brain.TABLES_INFO)\n"
    )
    env = dict(os.environ)
    env.pop("NOLIMIT_TABLES", None)
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "ok" in r.stdout
