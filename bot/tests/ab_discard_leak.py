"""Paired A/B of the revealed-discard features (DISCARD_LEAK notes) on the sim/variants toss game.

    .venv/bin/python bot/tests/ab_discard_leak.py --field EX --arms off,leak,reveal --tables 5000 --seed 501
    .venv/bin/python bot/tests/ab_discard_leak.py --diag 300            # how often each feature changes a decision

Our bot (v2, shipped defaults) in 1 seat; the arms differ only in its bot_params:
  off      shipped defaults (LEAK_PENALTY False, USE_REVEALED False)
  leak     LEAK_PENALTY True (LEAK_LAMBDA from --lam)
  reveal   USE_REVEALED True
  both     both on
Fields (7 opponents): EX = 7 x TossExample (the kit-style sample bot), READ = 7 x our v2 with USE_REVEALED True (a
field that reads our tosses with the same reader model; the test for the leak penalty), MIX = 3 x our v2 (shipped)
+ 4 x TossExample. Same seed = same seating, decks and agent seeds in every arm (paired). STRICT mode (exceptions
and illegal answers reach the engine's counters), the 1 ms limit enforced on thread CPU time. Tables: the rebuilt
toss tables bot/out/drill/toss/variant_tables.py (--tables-path builtin = the NLHE tables). Every log line carries
the real clock and the load average. Output: JSON with per-arm points, errors and latency, and paired differences
(arm - off) in relative units (points / 12.5) with 95% intervals.
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
if __name__ == "__main__":                       # env before the bot is imported (fork workers inherit it)
    _tp = None
    for _i, _a in enumerate(sys.argv):
        if _a == "--tables-path" and _i + 1 < len(sys.argv):
            _tp = sys.argv[_i + 1]
    os.environ["NOLIMIT_TABLES"] = _tp or os.path.join(BOT, "out", "drill", "toss", "variant_tables.py")
    os.environ.setdefault("NOLIMIT_BOT_STRICT", "1")
    os.environ.setdefault("NOLIMIT_TABLES_QUIET", "1")

import functools                                  # noqa: E402

from bot import brain as BR                       # noqa: E402
from bot import shell as SH                       # noqa: E402

SH._KEEP_POLICY = BR.keep_dispatch                # per-instance keep settings (keep_dispatch: None = static keep)
ARMS = {"off": {}, "leak": {"LEAK_PENALTY": True}, "reveal": {"USE_REVEALED": True},
        "both": {"LEAK_PENALTY": True, "USE_REVEALED": True}}
EX = "sim.variants.agents:TossExample"


def log(msg, fh=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')} load {' '.join(f'{x:.2f}' for x in os.getloadavg())}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def ours(params):
    return functools.partial(SH.make, version=2, params=dict(params) if params else None)


def entrants(field, me):
    if field == "EX":
        opp = [("ex", EX)] * 7
    elif field == "READ":
        opp = [("rd", ours({"USE_REVEALED": True}))] * 7
    elif field == "MIX":
        opp = [("v2", ours({}))] * 3 + [("ex", EX)] * 4
    else:
        raise SystemExit(f"unknown field {field}")
    return [("me", ours(me))] + opp


def paired(a, b, avg=12.5):
    d = [(x - y) / avg for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return {"mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4), "n": n,
            "identical_tables": sum(1 for x in d if x == 0)}


def pct(hist, q):
    tot = sum(hist.values())
    if not tot:
        return 0.0
    k = q * tot
    acc = 0
    for b in sorted(hist):
        acc += hist[b]
        if acc >= k:
            return b / 10.0
    return max(hist) / 10.0


def run_arm(field, arm, params, n, seed, workers):
    from sim.variants.runner import run_variant
    t0 = time.time()
    r = run_variant("toss", entrants(field, params), n, seed=seed, workers=workers, time_limit_ms=1.0, timing="cpu")
    st = r.stats["me"]
    pts = r.points["me"]
    return {"arm": arm, "params": params, "points": pts, "mean_points": sum(pts) / len(pts),
            "rel": sum(pts) / len(pts) / 12.5, "decisions": st.decisions, "exceptions": st.exceptions,
            "illegal": st.illegal, "timeouts": st.timeouts, "cpu_over_1ms": st.cpu_over,
            "cpu_us": {"p50": pct(st.cpu_hist, 0.5), "p99.9": pct(st.cpu_hist, 0.999), "max": st.cpu_max_ns / 1000},
            "secs": round(time.time() - t0, 1)}


def diag(n, seed, lam):
    """In-process: share of our toss requests whose keep changes with LEAK_PENALTY, and of our postflop calls facing
    a revealed bettor that USE_REVEALED moves (down / up), vs 7 TossExample on the toss tables."""
    from sim.variants.runner import run_variant
    from variant import defaults as VD
    cnt = {"requests": 0, "keep_changed": 0, "reads": 0, "down": 0, "up": 0, "delta_hist": {}}
    orig_rc = BR.Brain.reveal_class
    orig_kd = BR.keep_dispatch

    def kd(view):
        r = orig_kd(view)
        cnt["requests"] += 1
        lk = BR.keep_leak(view, dict(BR.PARAMS, LEAK_LAMBDA=lam))
        if lk is not None and lk != VD.safe_default(view):
            cnt["keep_changed"] += 1
        return r

    def rc(self, v, j, cls):
        out = orig_rc(self, v, j, cls)
        cnt["reads"] += 1
        d = self.rev.cache.get((j, len(v.board), cls), 0.0)
        b = f"{round(d * 20) / 20:+.2f}"
        cnt["delta_hist"][b] = cnt["delta_hist"].get(b, 0) + 1
        cnt["down" if out < cls else "up" if out > cls else "same"] = \
            cnt.get("down" if out < cls else "up" if out > cls else "same", 0) + 1
        return out

    BR.Brain.reveal_class = rc
    SH._KEEP_POLICY = kd
    try:
        run_variant("toss", entrants("EX", {"USE_REVEALED": True}), n, seed=seed, workers=1, time_limit_ms=None)
    finally:
        BR.Brain.reveal_class = orig_rc
        SH._KEEP_POLICY = BR.keep_dispatch
    cnt["delta_hist"] = dict(sorted(cnt["delta_hist"].items()))
    return cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="EX")
    ap.add_argument("--arms", default="off,leak,reveal")
    ap.add_argument("--tables", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=501)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lam", type=float, default=None, help="LEAK_LAMBDA for the leak arms (default PARAMS)")
    ap.add_argument("--tables-path", default=None)
    ap.add_argument("--diag", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    fh = open(a.log, "a") if a.log else None
    if a.diag:
        lam = a.lam if a.lam is not None else BR.PARAMS["LEAK_LAMBDA"]
        log(f"diag: {a.diag} tables vs 7 TossExample, seed {a.seed}, lam {lam}, {BR.TABLES_INFO}", fh)
        c = diag(a.diag, a.seed, lam)
        log(f"diag: {json.dumps(c)}", fh)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(c, f, indent=1)
        return 0
    arms = a.arms.split(",")
    log(f"A/B toss field {a.field}: arms {arms}, {a.tables} paired tables, seed {a.seed}, workers {a.workers}, "
        f"lam {a.lam}, {BR.TABLES_INFO}, STRICT {SH.STRICT}", fh)
    res = {"field": a.field, "tables": a.tables, "seed": a.seed, "workers": a.workers, "tables_info": BR.TABLES_INFO,
           "strict": SH.STRICT, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "loadavg_start": os.getloadavg(),
           "arms": {}, "diff_vs_off": {}}
    rows = {}
    for arm in arms:
        p = dict(ARMS[arm.split("@")[0]])
        if a.lam is not None and p.get("LEAK_PENALTY"):
            p["LEAK_LAMBDA"] = a.lam
        if "@" in arm:                                 # e.g. "leak@40": LEAK_LAMBDA 40 for this arm
            p["LEAK_LAMBDA"] = float(arm.split("@")[1])
        r = run_arm(a.field, arm, p, a.tables, a.seed, a.workers)
        rows[arm] = r
        log(f"  {a.field} {arm} {p}: rel {r['rel']:.4f}, {r['decisions']:,} decisions, exc/illegal/timeouts "
            f"{r['exceptions']}/{r['illegal']}/{r['timeouts']}, CPU us p50/p99.9/max {r['cpu_us']['p50']}/"
            f"{r['cpu_us']['p99.9']}/{r['cpu_us']['max']:.1f}, {r['secs']} s", fh)
        res["arms"][arm] = {k: v for k, v in r.items() if k != "points"}
    if "off" in rows:
        for arm in arms:
            if arm != "off":
                d = paired(rows[arm]["points"], rows["off"]["points"])
                res["diff_vs_off"][arm] = d
                log(f"  {a.field} {arm} - off: {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] rel/table, "
                    f"{d['identical_tables']} of {d['n']} tables identical", fh)
    res["loadavg_end"] = os.getloadavg()
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
