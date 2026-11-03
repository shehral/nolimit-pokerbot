"""Latency test (pytest wrapper of bot/tests/latency.py).

Default: 20,000 decisions per version (a quick check). Full SPEC run (>= 200,000 decisions): set
NOLIMIT_LATENCY_FULL=1, or run the script directly: .venv/bin/python bot/tests/latency.py --json bot/out/latency.json
Assertions use thread CPU time for the 250 us gate (wall clock on a loaded laptop includes other processes;
the wall numbers are printed and saved by the script) and require zero exceptions and zero illegal results.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot.tests.latency import measure      # noqa: E402

N = 200_000 if os.environ.get("NOLIMIT_LATENCY_FULL") == "1" else 20_000


@pytest.mark.parametrize("version", [0, 1, 2])
def test_latency_and_zero_errors(version):
    r = measure(version, N)
    assert r["decisions"] >= N
    assert r["shell_exceptions"] == 0 and r["shell_illegal_fixed"] == 0
    assert r["kit_exceptions"] == 0 and r["kit_illegal"] == 0
    assert r["cpu_us"]["p99.9"] <= 250.0, r["cpu_us"]
    assert r["over_1ms"]["cpu"] == 0, r["over_1ms"]
    print(f"\nv{version}: {r['decisions']} decisions; wall p50 {r['wall_us']['p50']} p99 {r['wall_us']['p99']} "
          f"p99.9 {r['wall_us']['p99.9']} max {r['wall_us']['max']:.1f} us")
