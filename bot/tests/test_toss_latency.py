"""Latency on the toss variant (pytest wrapper of bot/tests/latency_toss.py): betting decisions with 3 dealt cards
preflop, 2 kept after the toss, and the discard requests themselves, all through the timed act() path.

Default: 20,000 decisions (quick). The full run (200,000 decisions; results in VARIANT_INTEGRATION notes): set
NOLIMIT_LATENCY_FULL=1, or .venv/bin/python bot/tests/latency_toss.py --json bot/out/latency_toss.json
Gate: thread CPU p99.9 <= 250 us and no decision over 1 ms of CPU (wall clock on this shared laptop includes other
processes' preemption; the wall numbers are printed), zero exceptions and zero illegal answers.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot.tests.latency_toss import measure      # noqa: E402

N = 200_000 if os.environ.get("NOLIMIT_LATENCY_FULL") == "1" else 20_000


def test_toss_latency_and_zero_errors():
    r = measure(N)
    assert r["decisions"] >= N and r["discard_requests"] > N // 20
    assert r["shell_exceptions"] == 0 and r["shell_illegal_fixed"] == 0
    assert r["kit_exceptions"] == 0 and r["kit_illegal"] == 0
    assert r["cpu_us"]["p99.9"] <= 250.0, r["cpu_us"]
    assert r["over_1ms"]["cpu"] == 0, r["over_1ms"]
    print(f"\ntoss: {r['decisions']} decisions; wall {r['wall_us']}; cpu {r['cpu_us']}; discard {r['discard_wall_us']}")
