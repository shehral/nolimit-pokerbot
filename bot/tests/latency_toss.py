"""Latency of our act() on the TOSS variant (sim/variants), betting decisions and discard requests together, plus the
discard requests on their own. Single process, real variant tables: our bot in 4 seats, 4 TossExample agents.
Wall clock and thread CPU time per decision from the engine's 0.1-us histograms (what the rules limit is wall);
the discard-only numbers come from a thin timer around act(). Tables in play: the built-in NLHE tables unless
NOLIMIT_TABLES points at rebuilt toss tables (then the 1,755-class lookups are measured).

    .venv/bin/python bot/tests/latency_toss.py [--decisions 200000] [--json bot/out/latency_toss.json]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import SeatStats              # noqa: E402
from sim.kit.runner import load_factory           # noqa: E402
from sim.variants.agents import TossExample       # noqa: E402
from sim.variants.engine import VTable            # noqa: E402
from sim.variants.spec import make_variant        # noqa: E402


def _q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def measure(decisions, version=2, seed=0):
    fac = load_factory(f"bot/shell.py:BotV{version}")
    shell = sys.modules[fac.__module__]
    v = make_variant("toss")
    ours = SeatStats()
    bots, disc = [], []
    pc = time.perf_counter_ns

    def make(seat, cfg):
        b = fac(seat, cfg)
        orig = b.act

        def act(state, _o=orig):
            if state.get("request") == "discard":
                t0 = pc()
                out = _o(state)
                disc.append(pc() - t0)
                return out
            return _o(state)
        b.act = act
        bots.append(b)
        return b
    t_start = time.perf_counter()
    load0 = os.getloadavg()
    tables = errors = fixed = reqs = 0
    while ours.decisions < decisions:
        facs = [make, TossExample] * 4
        rot = tables % 8
        facs = facs[rot:] + facs[:rot]
        bots.clear()
        res = VTable(v, facs, seed=f"latt:{seed}:{tables}", timing="wall", time_limit_ms=None).play()
        for s in range(8):
            if facs[s] is make:
                ours.merge(res.stats[s])
        for b in bots:
            errors += b.errors
            fixed += b.illegal_fixed
            reqs += b.requests
        tables += 1
    s = ours.summary()
    disc.sort()
    return {
        "variant": "toss", "version": version, "decisions": s["decisions"], "discard_requests": reqs,
        "tables": tables, "wall_us": s["latency_us"], "cpu_us": s["cpu_us"],
        "over_1ms": {"wall": sum(c for b, c in ours.lat_hist.items() if b > 10000),
                     "cpu": sum(c for b, c in ours.cpu_hist.items() if b > 10000)},
        "over_250us": {"wall": sum(c for b, c in ours.lat_hist.items() if b >= 2500),
                       "cpu": sum(c for b, c in ours.cpu_hist.items() if b >= 2500)},
        "discard_wall_us": {k: round(_q(disc, p) / 1000, 2) for k, p in
                            (("p50", 0.5), ("p99", 0.99), ("p99.9", 0.999))} | {"max": round(disc[-1] / 1000, 2)},
        "shell_exceptions": errors, "shell_illegal_fixed": fixed, "kit_exceptions": s["exceptions"],
        "kit_illegal": s["illegal"], "elapsed_s": round(time.perf_counter() - t_start, 1),
        "loadavg_start": list(load0), "loadavg_end": list(os.getloadavg()), "python": sys.version.split()[0],
        "tables_in_play": shell.brain.TABLES_INFO,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=200_000)
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    r = measure(a.decisions, a.version)
    w, c, d = r["wall_us"], r["cpu_us"], r["discard_wall_us"]
    print(f"toss v{a.version}: {r['decisions']:,} decisions ({r['discard_requests']:,} discards) in {r['tables']} "
          f"tables, {r['elapsed_s']} s | wall p50 {w['p50']:.1f} p99 {w['p99']:.1f} p99.9 {w['p99.9']:.1f} "
          f"max {w['max']:.1f} us | cpu p50 {c['p50']:.1f} p99 {c['p99']:.1f} p99.9 {c['p99.9']:.1f} max "
          f"{c['max']:.1f} | discard wall {d} | >1ms {r['over_1ms']} | errors shell {r['shell_exceptions']} "
          f"fixed {r['shell_illegal_fixed']} kit {r['kit_exceptions']}/{r['kit_illegal']} | load "
          f"{r['loadavg_start'][0]:.0f}->{r['loadavg_end'][0]:.0f} | {r['tables_in_play']}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
