"""Variant-aware postflop class in the brain (POSTFLOP_VARIANT notes).

* identity: with the 52-card ranking and 2 hole cards the variant classifier (variant/bridge.class_any) is
  brain.hand_class card for card, on boards of 3-5 cards (hold'em) and 6-12 cards (toss boards)
* routing: brain._POST (the classifier postflop() uses) is set only when swapped tables carry a variant spec whose
  postflop differs from hold'em (short deck, toss); None with the built-in tables, with plain / bounty / antes /
  double-board tables, and with NOLIMIT_POSTFLOP_VARIANT=0. In-process: postflop() calls _POST exactly when set.
* latency: classifier CPU time (thread_time) per call, and act() CPU p99.9 on short-deck strict tables < 250 us
The swap is decided at import, so the routing cases run in fresh processes.
"""
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import brain                                              # noqa: E402
from variant.bridge import class_any, postflop_classifier, rules_from_ranking   # noqa: E402
from variant.spec import short_deck_spec, toss_spec                # noqa: E402

DRILL = os.path.join(ROOT, "bot", "out", "drill")
TABLES = {v: os.path.join(DRILL, v, "variant_tables.py") for v in
          ("shortdeck", "toss", "plain", "bounty", "antes", "doubleboard")}


def test_holdem_identity_with_hand_class():
    R = rules_from_ranking(None)
    rng = random.Random(1103)
    for nb in (3, 4, 5, 6, 8, 10, 12):
        for _ in range(12000 if nb <= 5 else 3000):
            cs = rng.sample(range(52), 2 + nb)
            assert class_any(cs[:2], cs[2:], R) == brain.hand_class(cs[0], cs[1], cs[2:]), cs


def _probe(tables, extra_env=None):
    env = dict(os.environ, NOLIMIT_TABLES=tables, NOLIMIT_TABLES_QUIET="1")
    env.update(extra_env or {})
    code = ("import json; from bot import brain; print(json.dumps({'seam': brain._SEAM is not None, "
            "'post': brain._POST is not None, 'info': brain.POST_INFO}))")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_routing_only_with_a_variant_spec():
    r = _probe("builtin")
    assert not r["seam"] and not r["post"] and "hand_class" in r["info"]
    for v in ("plain", "bounty", "antes", "doubleboard"):
        r = _probe(TABLES[v])
        assert r["seam"] and not r["post"], (v, r)                 # hold'em postflop: hand_class, unchanged
    r = _probe(TABLES["shortdeck"])
    assert r["seam"] and r["post"] and "variant ranking (6 straights, flush>FH True" in r["info"], r
    r = _probe(TABLES["toss"])
    assert r["seam"] and r["post"] and "keep 2 of 3" in r["info"], r
    r = _probe(TABLES["shortdeck"], {"NOLIMIT_POSTFLOP_VARIANT": "0"})
    assert r["seam"] and not r["post"] and "off" in r["info"], r


def test_postflop_calls_the_classifier_only_when_set(monkeypatch):
    from sim.kit.engine import Table
    from sim.kit.fixtures import CheckCallAgent, RandomAgent
    from sim.kit.runner import load_factory
    fac = load_factory("bot.shell:Bot")
    calls = []

    def fake(hole, board, street=None):
        calls.append(street)
        return 1

    def play():
        for t in range(3):
            facs = [fac if s % 2 == 0 else (RandomAgent, CheckCallAgent)[(s + t) % 2] for s in range(8)]
            Table(facs, "TOP3", "TURBO", seed=f"postvar:{t}", timing="cpu", time_limit_ms=None).play()

    monkeypatch.setattr(brain, "_POST", None)
    play()
    assert calls == []
    monkeypatch.setattr(brain, "_POST", fake)
    play()
    assert len(calls) > 20 and all(s in (1, 2, 3) for s in calls)
    # a classifier that raises falls back to post_class (never breaks a decision)
    monkeypatch.setattr(brain, "_POST", lambda *a: 1 // 0)
    assert brain.post_class_variant([48, 44], [0, 5, 9], None, 1) == brain.post_class([48, 44], [0, 5, 9])


def _cpu_quantiles(f, spots):
    tt = time.thread_time_ns
    xs = []
    for h, b, st in spots:
        t0 = tt()
        f(h, b, st)
        xs.append(tt() - t0)
    xs.sort()
    n = len(xs)
    return {q: xs[min(n - 1, int(q * n))] / 1000.0 for q in (0.5, 0.99, 0.999)} | {"max": xs[-1] / 1000.0}


def test_classifier_cpu_latency():
    rng = random.Random(9)
    short = tuple(c for c in range(52) if c >> 2 >= 4)
    f = postflop_classifier(short_deck_spec())
    spots = []
    for _ in range(20000):
        nb = rng.choice((3, 4, 5))
        cs = rng.sample(short, 2 + nb)
        spots.append((cs[:2], cs[2:], nb - 2))
    q = _cpu_quantiles(f, spots)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')} load {os.getloadavg()[0]:.1f}] short-deck classifier CPU us {q}")
    assert q[0.5] < 25 and q[0.999] < 120, q
    g = postflop_classifier(toss_spec())
    spots = []
    for _ in range(5000):
        nb = rng.choice((5, 7, 9))
        cs = rng.sample(range(52), 3 + nb)
        spots.append((cs[:3], cs[3:], 1))                          # worst case: the choice still ahead (3 subsets)
    q = _cpu_quantiles(g, spots)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} load {os.getloadavg()[0]:.1f}] toss 3-card classifier CPU us {q}")
    assert q[0.5] < 60 and q[0.999] < 250, q


def test_act_cpu_latency_short_deck_strict(tmp_path):
    out = tmp_path / "bench.json"
    env = dict(os.environ, NOLIMIT_TABLES=TABLES["shortdeck"], NOLIMIT_BOT_STRICT="1", NOLIMIT_TABLES_QUIET="1",
               NOLIMIT_POSTFLOP_VARIANT="1")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "bot", "tools", "drill.py"), "--bench", "shortdeck",
                        "--tables", "60", "--seed", "3", "--workers", "1", "--bench-out", str(out)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(out.read_text())
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')} load {os.getloadavg()[0]:.1f}] act() CPU us {r['cpu_us']} "
          f"over {r['decisions']} decisions")
    assert r["exceptions"] == 0 and r["illegal"] == 0 and r["timeouts"] == 0
    assert r["cpu_us"]["p99.9"] < 250, r["cpu_us"]
    assert "variant_tables" in r["tables_in_play"]
