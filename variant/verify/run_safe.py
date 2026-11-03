"""300 strict tables per variant for the safe-default bot (variant/verify/safe_bot.py).

Two fields per variant: "self" = 8 safe-default bots (most decisions), "mix" = 1 safe-default bot + 7 of the
variant's own example agents (sim/variants/agents.py) for a points context. Strict kit rules: 1 ms wall clock,
errors fold (check when free); discard errors fold. Also a direct latency micro-benchmark of the discard path.

    .venv/bin/python -m variant.verify.run_safe [--tables 300] [--workers 1]
"""
import argparse
import json
import os
import random
import time

from sim.kit.cards import CARD_STR
from sim.variants.runner import run_variant

HERE = os.path.dirname(os.path.abspath(__file__))
SAFE = "variant.verify.safe_bot:SafeDefaultBot"
EX = {"plain": "VariantExample", "toss": "TossExample", "shortdeck": "ShortDeckExample", "bounty": "BountyExample",
      "doubleboard": "DoubleBoardExample", "antes": "AnteExample", "bbante": "AnteExample"}
KEYS = ("decisions", "timeouts", "exceptions", "illegal", "errors", "latency_us", "cpu_us", "over_limit", "examples",
        "hands")


def discard_bench(n=20000, seed=5):
    from variant.verify.safe_bot import SafeDefaultBot
    rng = random.Random(seed)
    bot = SafeDefaultBot(0, {"seat": 0})
    lat = []
    bad = 0
    pc = time.perf_counter_ns
    for _ in range(n):
        cs = rng.sample(range(52), 3 + 2 + rng.randrange(0, 4))
        st = {"request": "discard", "hole": [CARD_STR[c] for c in cs[:3]], "board": [CARD_STR[c] for c in cs[3:]],
              "discard_options": [0, 1, 2], "legal_actions": ["discard"], "pot": 60, "stacks": [2000] * 8, "seat": 0}
        t0 = pc()
        r = bot.act(st)
        lat.append(pc() - t0)
        if not (type(r) is dict and r.get("action") == "discard" and r.get("index") in (0, 1, 2)):
            bad += 1
    lat.sort()
    q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))] / 1000, 2)
    return {"n": n, "illegal": bad, "p50_us": q(0.5), "p99_us": q(0.99), "p999_us": q(0.999),
            "max_us": round(lat[-1] / 1000, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=["plain", "toss", "shortdeck", "bounty", "doubleboard", "antes",
                                                       "bbante"])
    a = ap.parse_args(argv)
    out = {"discard_bench": discard_bench(), "runs": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "loadavg_start": os.getloadavg()}
    for v in a.variants:
        for field in ("self", "mix"):
            ents = [("safe", SAFE)] * 8 if field == "self" else \
                [("safe", SAFE)] + [("ex", f"sim.variants.agents:{EX[v]}")] * 7
            t0 = time.time()
            r = run_variant(v, ents, a.tables, seed=17, workers=a.workers, time_limit_ms=1.0, timing="wall")
            s = r.stats["safe"].summary()
            pts = r.points["safe"]
            row = {k: s.get(k) for k in KEYS}
            row.update(tables=a.tables, seats=len(pts), mean_points=round(sum(pts) / len(pts), 3),
                       rel=round(sum(pts) / len(pts) / 12.5, 3), tables_per_s=round(r.tables_per_s, 1),
                       wall_s=round(time.time() - t0, 1), loadavg=os.getloadavg())
            if field == "mix":
                e = r.stats["ex"].summary()
                row["ex_errors"] = {k: e.get(k) for k in ("timeouts", "exceptions", "illegal")}
            out["runs"][f"{v}/{field}"] = row
            print(f"{v:12s} {field:4s} dec {row['decisions']:7d} to {row['timeouts']} exc {row['exceptions']} "
                  f"ill {row['illegal']} cpu_over {s['over_limit']['cpu']} p50 {s['latency_us']['p50']} "
                  f"p99 {s['latency_us']['p99']} rel {row['rel']}", flush=True)
    out["loadavg_end"] = os.getloadavg()
    with open(os.path.join(HERE, "out", "safe_runs.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out["discard_bench"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
