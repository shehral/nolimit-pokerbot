"""Paired A/B of bot variants on identical tables (same seating, buttons, decks and agent seeds; sim/kit run_paired).

A variant is "v<N>" optionally followed by PARAMS overrides: "v1", "v2:CALL_LAM=2.0,PF_DEPTH_BB=12".
Values are parsed as Python literals (numbers, True/False/None, tuples, dicts). The first variant is the baseline.
Opponents (7 seats):
  an archetype key   7 copies of sim/zoo/<key>.py            e.g. --field icm_pro
  a mix name         each seat drawn per table from the mix   e.g. --field T6_like  (T1_like, T6_like, STRONG)
  self               7 copies of the baseline variant
  example            7 copies of the kit's example agent
Scores are relative (our points / 12.5) with 95% CIs; the paired difference is in points and in relative units.
Score runs use CPU-time enforcement (laptop preemption excluded; see sim/kit/engine.py).

    .venv/bin/python bot/tools/ab.py --field icm_pro --tables 2000 v0 v1 v2
    .venv/bin/python bot/tools/ab.py --field T6_like --payouts LINEAR v1 "v1:CALL_LAM=2.0"
"""
import argparse
import ast
import functools
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.runner import run_paired              # noqa: E402
from bot import shell as SH                         # noqa: E402  (package import: partials pickle by name)

SHARES_SUM = 100.0


def parse_variant(text):
    """'v2:A=1,B=(1,2)' -> (label, version, params)."""
    head, _, tail = text.partition(":")
    if not head.startswith("v"):
        raise SystemExit(f"variant must start with v<N>: {text!r}")
    version = int(head[1:])
    params = {}
    if tail:
        for item in _split_top(tail):
            k, _, val = item.partition("=")
            k = k.strip()
            if k not in SH.brain.PARAMS:
                raise SystemExit(f"unknown PARAMS key {k!r} in {text!r}")
            params[k] = ast.literal_eval(val.strip())
    return text, version, params


def _split_top(s):
    """Split on commas that are not inside brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def mixed_opponent(seat, config, mix_name):
    """One seat drawn from a zoo mix, deterministically from the agent seed (identical across paired runs)."""
    from sim.zoo import fields as F
    mix = F.MIXES[mix_name]
    keys = sorted(mix)
    rng = random.Random(config["seed"] * 7 + 12345)
    arch = rng.choices(keys, [mix[k] for k in keys], k=1)[0]
    return F.factory(arch)(seat, config)


def opponents(field, base_variant):
    if field == "self":
        _, ver, params = base_variant
        return [("self", functools.partial(SH.make, version=ver, params=params))] * 7
    if field == "example":
        return [("example", "sim/kit/example_agent.py:ExampleAgent")] * 7
    from sim.zoo import fields as F
    if field in F.MIXES:
        return [(field, functools.partial(mixed_opponent, mix_name=field))] * 7
    if field in F.ARCHETYPES:
        return [(field, F.factory(field))] * 7
    raise SystemExit(f"unknown field {field!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--field", default="T6_like")
    ap.add_argument("--tables", type=int, default=2000)
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true", help="print example error messages")
    a = ap.parse_args(argv)
    vs = [parse_variant(t) for t in a.variants]
    opp = opponents(a.field, vs[0])
    configs = {}
    for label, ver, params in vs:
        configs[label] = [("ours", functools.partial(SH.make, version=ver, params=params))] + opp
    t0 = time.time()
    pr = run_paired(configs, a.tables, seed=a.seed, focus=0, payouts=a.payouts, schedule=a.schedule,
                    workers=a.workers, timing="cpu")
    secs = time.time() - t0
    base = vs[0][0]
    print(f"field {a.field}, {a.tables} paired tables, {a.payouts} {a.schedule}, seed {a.seed}, {secs:.0f} s")
    out = {"field": a.field, "tables": a.tables, "payouts": a.payouts, "schedule": a.schedule, "seed": a.seed,
           "secs": secs, "variants": {}}
    for label, _, _ in vs:
        e = pr.results[label].entrants[0]
        r = e.rel()
        s = e.stats.summary()
        row = {"rel": r, "errors": s["errors"], "timeouts": s["timeouts"], "exceptions": s["exceptions"],
               "illegal": s["illegal"]}
        line = (f"  {label:<40} rel {r['rel']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  "
                f"exc/illegal/timeouts {s['exceptions']}/{s['illegal']}/{s['timeouts']}")
        row["examples"] = s.get("examples")
        if label != base:
            d = pr.comparisons[label]
            row["diff_vs_base"] = d
            line += (f"  | vs {base}: {d['mean_diff']:+.2f} pts [{d['lo']:+.2f}, {d['hi']:+.2f}] "
                     f"({d['mean_diff'] / 12.5:+.3f} rel), se {d['se']:.2f} (unpaired {d['unpaired_se']:.2f})")
        out["variants"][label] = row
        print(line, flush=True)
        if s["errors"] and a.verbose:
            for kind, ex in (s.get("examples") or {}).items():
                for x in ex[:3]:
                    print(f"      {kind}: {x}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
