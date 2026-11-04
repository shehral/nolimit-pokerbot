"""Ablation of the short-deck postflop change (POSTFLOP_VARIANT notes section 4): what reading of the A-6-7-8-9
straight plays best against the example agents. Every class change in short deck comes from an A-6-7-8-9 straight
(the only straight hold'em's reading misses; the draws are the same), so the arms are:

    holdem   brain._POST = None (hand_class)
    shipped  the shipped classifier: A6789 made with a hole card = class 2 (PostflopRules.weak_heights)
    wheel3   A6789 made with a hole card = class 3 like every other straight (the first version)
    two_hole wheel3, but only when both hole cards are in it; with one hole card, the hold'em class

In-process, brain._POST patched before the fork pool starts (children inherit it). Same tables, seeds, seats.
    .venv/bin/python bot/tests/ab_postflop_ablation.py --tables 10000 --seeds 2026 1106 --workers 4
    .venv/bin/python bot/tests/ab_postflop_ablation.py --tables 10000 --seeds 7 99 4242 --arms holdem shipped wheel3
Writes bot/out/postflop_variant/shortdeck_ablation.json; log lines carry the real clock.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
ROOT = os.path.dirname(BOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TP = os.path.join(BOT, "out", "drill", "shortdeck", "variant_tables.py")
os.environ.setdefault("NOLIMIT_TABLES", TP)
os.environ.setdefault("NOLIMIT_TABLES_QUIET", "1")
os.environ.setdefault("NOLIMIT_BOT_STRICT", "1")

from bot import brain                               # noqa: E402
from sim.variants.runner import run_variant         # noqa: E402

WHEEL = (1 << 12) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
FULL = brain._POST


def _n_wheel(hole):
    return sum(1 for c in hole if (1 << (c >> 2)) & WHEEL)


R = FULL.rules if FULL is not None else None
WEAK = R.weak_heights if R is not None else frozenset()


def two_hole(hole, board, street=None):
    c = FULL(hole, board, street)
    h = brain.hand_class(hole[0], hole[1], board) if len(hole) == 2 else c
    if c != h and _n_wheel(hole) < 2:
        return h
    return c


def arm_setup(name):
    """Install an arm: (brain._POST, R.weak_heights)."""
    if name == "holdem":
        brain._POST, R.weak_heights = None, WEAK
    elif name == "shipped":
        brain._POST, R.weak_heights = FULL, WEAK
    elif name == "wheel3":
        brain._POST, R.weak_heights = FULL, frozenset()
    elif name == "two_hole":
        brain._POST, R.weak_heights = two_hole, frozenset()
    else:
        raise ValueError(name)


ARMS = ("holdem", "shipped", "wheel3", "two_hole")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} load {os.getloadavg()[0]:.1f}] {msg}", flush=True)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    h = 1.96 * sd / math.sqrt(n)
    return {"mean": round(m, 3), "lo": round(m - h, 3), "hi": round(m + h, 3), "n": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1106])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="shortdeck_ablation.json")
    a = ap.parse_args(argv)
    assert FULL is not None, brain.POST_INFO
    ents = [("me", "bot.shell:Bot")] + [("ex", "sim.variants.agents:ShortDeckExample")] * 7
    arms = list(a.arms)
    pts = {k: [] for k in arms}
    cpu = {}
    for seed in a.seeds:
        for name in arms:
            arm_setup(name)
            r = run_variant("shortdeck", ents, a.tables, seed=seed, workers=a.workers, time_limit_ms=1.0,
                            timing="cpu")
            s = r.stats["me"].summary()
            pts[name] += r.points["me"]
            cpu[f"{name}:{seed}"] = s["cpu_us"]
            log(f"seed {seed} {name:9s} mean {sum(r.points['me']) / len(r.points['me']):.3f}, errors "
                f"{s['exceptions']}/{s['illegal']}/{s['timeouts']}, CPU us {s['cpu_us']}")
    arm_setup("shipped")
    res = {"arms": arms, "tables_per_seed": a.tables, "seeds": a.seeds, "cpu_us": cpu, "loadavg_end": list(os.getloadavg()),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "vs_holdem": {}}
    for name in arms:
        if name != "holdem":
            d = paired(pts[name], pts["holdem"])
            res["vs_holdem"][name] = d
            log(f"{name:9s} - holdem: {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] pts/table over {d['n']}")
    out = os.path.join(BOT, "out", "postflop_variant", a.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
