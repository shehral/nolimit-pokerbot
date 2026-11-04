"""Surprise-variant drill: the event day (No Limit) rehearsed end to end on a sim/variants game.

    .venv/bin/python bot/tools/drill.py toss            # or shortdeck, bounty, doubleboard, antes, plain, random
    .venv/bin/python bot/tools/drill.py all --json bot/out/drill_all.json

Steps (elapsed minutes printed after each):
  1 triage      the five triage questions (the variant checklist) with this variant's answers
  2 spec        the VariantSpec the event-day way (variant/verify/specs.py: spec_from_kit + evaluator probe), its
                HandSpace, and the table spec (a pure double board builds on the NLHE spec)
  3 build       variant.build_tables in the BACKGROUND (a subprocess) with the right budget: 300 s for rollout games
                (toss: 300-900 s, verify 2026-11-03), 90 s otherwise; output bot/out/drill/<variant>/variant_tables.py
  4 safe bot    immediately, while the build runs: the integrated bot on the BUILT-IN NLHE tables (NOLIMIT_TABLES=
                builtin), 300 strict tables, 1 seat vs 7 of the variant's example agents; STRICT mode (exceptions
                and illegal answers reach the engine's counters, nothing is masked)
  5 swap        wait for the build, gate it (tables_loader: SANITY ok, stage a done, stderr a <= 0.015, labels match)
  6 rerun       the same 300 tables (same seed: same seats and decks, so the comparison is paired) with
                NOLIMIT_TABLES=<the build>; if the gate rejected the tables this run plays the built-in tables again
Timeouts are enforced on thread CPU time (1 ms): this laptop is shared (load 20-100 while these drills ran), so the
wall clock includes other processes' preemption; wall overruns are reported next to it (wall_over, preempted).
Relative score = our mean points / 12.5 (1.00 = an average seat), with a 95% interval; the paired difference
(rerun - safe) is in points per table.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
ROOT = os.path.dirname(BOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PY = sys.executable
VARIANTS = ("toss", "shortdeck", "bounty", "doubleboard", "antes", "plain")
EXAMPLE = {"plain": "VariantExample", "toss": "TossExample", "shortdeck": "ShortDeckExample",
           "bounty": "BountyExample", "doubleboard": "DoubleBoardExample", "antes": "AnteExample"}
BUDGET = {"toss": 300.0}          # rollout games; everything else 90 s (a pure double board builds on the NLHE spec)
QUESTIONS = (
    "1. Hole-card count changed? Keep/discard/toss/assign choice? When? Discards public?",
    "2. Deck or ranking changed (removed ranks, short deck, new ranking, wild cards)?",
    "3. Any new request or action beyond fold/check/call/raise/all-in (bid, discard, redraw, assign, other)?",
    "4. Any card or value revealed that was not before (discards, a hole card, bids)?",
    "5. Payouts, antes, straddles or the blind schedule changed?",
)
ANSWERS = {
    "toss": ("3 dealt; every contender tosses 1 face up to the board before flop betting (2-card flop)",
             "no", "yes: request 'discard', answer {'action': 'discard', 'index': k}, discard_options",
             "yes: every tossed card (public 'discard' events, state['discards'])", "no"),
    "shortdeck": ("no", "yes: 36 cards (6..A), flush > full house, A-6-7-8-9 straight", "no", "no", "no"),
    "bounty": ("no", "no", "no", "a hidden per-level bounty rank (ours: state['bounty_rank']); bounty_bonus events",
               "payoff: a pot won with the rank in hole or board pays 1.5 x normal + 10"),
    "doubleboard": ("no", "no (two boards; each pot split in halves per board)", "no", "state['boards']", "no"),
    "antes": ("no", "no", "no", "no", "antes 0.125 bb from level 2 (dead money)"),
    "plain": ("no", "no", "no", "no", "no"),
}


def _mins(t0):
    return round((time.time() - t0) / 60.0, 2)


def _say(t0, msg):
    print(f"[drill +{_mins(t0):5.2f} min] {msg}", flush=True)


# ------------------------------------------------------------------ bench (runs in a fresh subprocess)
def bench(variant, tables, seed, workers, out_json):
    """300 strict tables: our bot (bot.shell:Bot) in 1 seat, 7 example agents; writes our seat's numbers."""
    os.environ.setdefault("NOLIMIT_BOT_STRICT", "1")
    from sim.variants.runner import run_variant
    from bot import brain
    ents = [("me", "bot.shell:Bot")] + [("ex", f"sim.variants.agents:{EXAMPLE[variant]}")] * 7
    t0 = time.time()
    r = run_variant(variant, ents, tables, seed=seed, workers=workers, time_limit_ms=1.0, timing="cpu")
    s = r.stats["me"].summary()
    pts = r.points["me"]
    ex = r.stats["ex"].summary()
    row = {"variant": variant, "tables": tables, "seed": seed, "points": pts,
           "mean_points": sum(pts) / len(pts), "decisions": s["decisions"], "timeouts": s["timeouts"],
           "exceptions": s["exceptions"], "illegal": s["illegal"], "over_limit": s.get("over_limit"),
           "latency_us": s["latency_us"], "cpu_us": s.get("cpu_us"), "examples": s.get("examples"),
           "ex_errors": {k: ex.get(k) for k in ("timeouts", "exceptions", "illegal")},
           "tables_in_play": brain.TABLES_INFO, "strict": os.environ.get("NOLIMIT_BOT_STRICT") == "1",
           "wall_s": round(time.time() - t0, 1), "loadavg": list(os.getloadavg())}
    with open(out_json, "w") as f:
        json.dump(row, f)
    return row


def _run_bench(variant, tables, seed, workers, out_json, tables_env):
    env = dict(os.environ, NOLIMIT_TABLES=tables_env, NOLIMIT_BOT_STRICT="1", NOLIMIT_TABLES_QUIET="1")
    cmd = [PY, os.path.abspath(__file__), "--bench", variant, "--tables", str(tables), "--seed", str(seed),
           "--workers", str(workers), "--bench-out", out_json]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"bench failed: {p.stderr[-2000:]}")
    with open(out_json) as f:
        return json.load(f)


def _rel(pts):
    n = len(pts)
    m = sum(pts) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in pts) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return round(m / 12.5, 3), round((m - h) / 12.5, 3), round((m + h) / 12.5, 3)


def _paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return round(m, 3), round(m - h, 3), round(m + h, 3)


def _brief(row):
    ol = row.get("over_limit") or {}
    return (f"{row['decisions']:,} decisions, exceptions {row['exceptions']}, illegal {row['illegal']}, timeouts "
            f"{row['timeouts']} (cpu over {ol.get('cpu')}, wall over {ol.get('wall')}, preempted "
            f"{ol.get('preempted')}), wall p50/p99/p99.9/max {row['latency_us']['p50']:.1f}/"
            f"{row['latency_us']['p99']:.1f}/{row['latency_us']['p99.9']:.1f}/{row['latency_us']['max']:.0f} us, "
            f"rel {_rel(row['points'])}")


# ------------------------------------------------------------------ the drill
def drill(variant, tables=300, seed=1106, bench_workers=4, build_workers=8, budget=None, outdir=None):
    t0 = time.time()
    steps = {}
    out = {"variant": variant, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "loadavg_start": list(os.getloadavg())}
    outdir = outdir or os.path.join(BOT, "out", "drill", variant)
    os.makedirs(outdir, exist_ok=True)
    # 1 triage
    print(f"\n=== DRILL: {variant} ===")
    for q, a in zip(QUESTIONS, ANSWERS[variant]):
        print(f"  {q}\n     -> {a}")
    steps["triage"] = _mins(t0)
    _say(t0, "triage done")
    # 2 spec
    from variant.verify import specs
    from variant.handspace import HandSpace
    spec, rep = specs.get(variant)
    tspec = spec.table_spec()
    hs = HandSpace(tspec)
    print(f"  spec: {spec.summary()}")
    print(f"  probe: ok {rep['ok']}, exact {rep.get('exact')}, warnings {rep.get('warnings')}, errors {rep['errors']}")
    print(f"  table spec: {tspec.notes or 'same as the game'}; {hs.describe()}")
    out["spec"] = {"summary": spec.summary(), "probe_ok": rep["ok"], "probe_warnings": rep.get("warnings"),
                   "table_spec": tspec.notes or "", "classes": hs.n_classes}
    steps["spec"] = _mins(t0)
    _say(t0, "spec + probe + HandSpace done")
    # 3 build in the background
    b = budget or BUDGET.get(variant, 90.0)
    tpath = os.path.join(outdir, "variant_tables.py")
    for p in (tpath, tpath + ".tmp"):
        if os.path.exists(p):
            os.remove(p)
    blog = open(os.path.join(outdir, "build.log"), "w")
    build = subprocess.Popen([PY, "-m", "variant.build_tables", "--spec", f"variant.verify.specs:{variant}_spec",
                              "--workers", str(build_workers), "--budget", str(b), "--out", tpath],
                             cwd=ROOT, stdout=blog, stderr=subprocess.STDOUT)
    out["build"] = {"budget_s": b, "workers": build_workers, "path": tpath}
    steps["build_started"] = _mins(t0)
    _say(t0, f"build started in the background: budget {b:.0f} s, {build_workers} workers -> {tpath}")
    # 4 safe-default bot, immediately
    safe = _run_bench(variant, tables, seed, bench_workers, os.path.join(outdir, "safe.json"), "builtin")
    steps["safe_bot_verified"] = _mins(t0)
    out["safe"] = {k: v for k, v in safe.items() if k != "points"}
    out["safe"]["rel"] = _rel(safe["points"])
    _say(t0, f"SAFE BOT (built-in tables, submit-ready): {_brief(safe)}")
    # 5 wait for the build, gate it
    rc = build.wait()
    blog.close()
    steps["build_done"] = _mins(t0)
    from bot import tables_loader as TL
    act = TL.load(tpath, quiet=True)
    out["build"].update(returncode=rc, accepted=act.kind == "variant", reasons=act.reasons, line=TL.INFO["line"])
    try:
        T = TL.import_tables(tpath)
        out["build"].update(stderr=T.META.get("stderr"), stage_s=T.META.get("stage_s"),
                            stage_done=T.META.get("stage_done"), sanity_ok=T.SANITY.get("ok"),
                            sources=T.SOURCES, elapsed_s=T.META.get("elapsed_s"))
    except Exception as e:  # noqa: BLE001
        out["build"]["import_error"] = f"{type(e).__name__}: {e}"
    _say(t0, f"build finished (rc {rc}); gate: {TL.INFO['line']}")
    # 6 rerun with the swapped tables (paired: same seed)
    env_tables = tpath if act.kind == "variant" else "builtin"
    swap = _run_bench(variant, tables, seed, max(bench_workers, 8), os.path.join(outdir, "swapped.json"),
                      env_tables)
    steps["swapped_verified"] = _mins(t0)
    out["swapped"] = {k: v for k, v in swap.items() if k != "points"}
    out["swapped"]["rel"] = _rel(swap["points"])
    out["paired_diff_pts"] = _paired(swap["points"], safe["points"])
    _say(t0, f"SWAPPED ({'rebuilt tables' if act.kind == 'variant' else 'gate rejected: built-in again'}): "
             f"{_brief(swap)}")
    _say(t0, f"paired difference swapped - safe: {out['paired_diff_pts']} points per table")
    out["steps_min"] = steps
    out["loadavg_end"] = list(os.getloadavg())
    with open(os.path.join(outdir, "drill.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"  steps (elapsed minutes): {steps}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variant", nargs="?", default="random", help="toss shortdeck bounty doubleboard antes plain "
                                                                 "random all")
    ap.add_argument("--tables", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1106)
    ap.add_argument("--workers", type=int, default=4, help="bench workers while the build runs")
    ap.add_argument("--build-workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=None, help="build seconds (default: toss 300, others 90)")
    ap.add_argument("--json", default="")
    ap.add_argument("--bench", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--bench-out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.bench:
        bench(a.bench, a.tables, a.seed, a.workers, a.bench_out)
        return 0
    names = list(VARIANTS) if a.variant == "all" else [
        random.choice(VARIANTS) if a.variant == "random" else a.variant]
    res = {}
    for v in names:
        res[v] = drill(v, a.tables, a.seed, a.workers, a.build_workers, a.budget)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    print("\n=== SUMMARY ===")
    for v, r in res.items():
        s, w = r["safe"], r["swapped"]
        print(f"{v:12s} safe at {r['steps_min']['safe_bot_verified']:5.2f} min rel {s['rel']} err "
              f"{s['exceptions']}/{s['illegal']}/{s['timeouts']} | swapped at {r['steps_min']['swapped_verified']:5.2f}"
              f" min ({'rebuilt' if r['build']['accepted'] else 'REJECTED'}) rel {w['rel']} err "
              f"{w['exceptions']}/{w['illegal']}/{w['timeouts']} | paired diff {r['paired_diff_pts']} pts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
