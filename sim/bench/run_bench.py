"""Run the benchmarks B0-B5 and write JSON to sim/bench/out/.

    .venv/bin/python sim/bench/run_bench.py b0            # sanity: 8 copies, determinism, pairing, test suites
    .venv/bin/python sim/bench/run_bench.py b2top3        # headline: fields x occupants, TOP3/TURBO
    .venv/bin/python sim/bench/run_bench.py b1            # 7 copies of each archetype, v0/v1/v2
    .venv/bin/python sim/bench/run_bench.py b2pay         # LINEAR and WTA (TURBO), v0/v1/v2
    .venv/bin/python sim/bench/run_bench.py b2sched       # FAST and SLOW (TOP3), v0/v1/v2
    .venv/bin/python sim/bench/run_bench.py b3            # simulated event: 6 drifting tournaments, weights 1..6
    .venv/bin/python sim/bench/run_bench.py b4            # reliability: >= 1M decisions, wall clock enforced, STRICT
    .venv/bin/python sim/bench/run_bench.py b5knobs       # extra paired A/Bs of PARAMS knobs (v2 vs v2:KEY=VAL)
    .venv/bin/python sim/bench/run_bench.py ai            # the AI-written fields AI_ONLY and AI_MIX, same occupants as b2top3
Each step skips cells whose JSON already exists (resume after a cutoff); --force reruns them.
Score runs: kit clock enforcement off, seeded timeout model for near-limit opponents (see common.py).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.bench import common as C        # noqa: E402
from sim.zoo import fields as F           # noqa: E402

VERSIONS = ["v0", "v1", "v2"]
SUBSTITUTES = ["chart_tag", "icm_pro", "lag_aggro"]      # "what if we had submitted X" in our seat
FIELDS = ["T1_like", "T6_like", "STRONG"]


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def exists(name):
    return os.path.exists(os.path.join(C.OUT, name))


def cell(name, force, pool, **kw):
    if exists(name) and not force:
        print(f"skip {name} (exists)", flush=True)
        return C.load(name)
    t0 = time.time()
    out = C.run_cell(pool=pool, **kw)
    out["finished"] = stamp()
    C.save(out, name)
    print(f"  saved {name} in {time.time() - t0:.0f} s", flush=True)
    return out


# ----------------------------------------------------------------------------- B0
def b0(args, pool):
    res = {}
    # 8 copies of the same bot: the focus seat must score 1.00 within its CI.
    for who, n in (("v2", 10000), ("random_legal", 10000), ("v0", 4000)):
        res[f"self_{who}"] = cell(f"b0_self_{who}.json", args.force, pool, field=who, occupants=[who], n_tables=n,
                                  seed=11, label=f"B0 8 x {who}")
    # Determinism: the same cell twice. v0 has no clock-dependent logic; v1/v2 start an exact bubble factor only
    # while their own clock allows (GUARD_NS), so under heavy machine load a few decisions fall back to the table BF.
    # The huge-guard variant shows that this guard is the only source of run-to-run differences.
    rep = {}
    for occ in ("v0", "v2", "v2:GUARD_NS=1000000000000"):
        a = C.run_cell("T6_like", [occ], 600, seed=12, pool=pool, label="B0 repeat A", verbose=False)
        b = C.run_cell("T6_like", [occ], 600, seed=12, pool=pool, label="B0 repeat B", verbose=False)
        pa, pb = a["occ"][occ]["points"], b["occ"][occ]["points"]
        rep[occ] = {"tables": len(pa), "differing_tables": sum(1 for x, y in zip(pa, pb) if x != y),
                    "guard_trips": [a["occ"][occ]["shell"]["guard_trips"], b["occ"][occ]["shell"]["guard_trips"]],
                    "decisions": a["occ"][occ]["stats"]["decisions"],
                    "loadavg": [a["loadavg_start"][0], b["loadavg_end"][0]]}
        print(f"  repeat {occ}: {rep[occ]}", flush=True)
    res["repeat"] = rep
    w1 = C.run_cell("T6_like", ["v0"], 100, seed=12, workers=1, label="B0 repeat 1 worker", verbose=False)
    a = C.run_cell("T6_like", ["v0"], 100, seed=12, pool=pool, label="B0 repeat pool", verbose=False)
    res["one_worker_identical_v0"] = a["occ"]["v0"]["points"] == w1["occ"]["v0"]["points"]
    # Pairing: under WTA every bubble factor is 1 and v1 turns ICM off, so v1 must equal v0 on every table.
    res["wta_v1_v0"] = cell("b0_wta_pairing.json", args.force, pool, field="T6_like", occupants=["v0", "v1"],
                            n_tables=2000, seed=13, payouts="WTA", label="B0 WTA v1 == v0")["pairs"]["v1 - v0"]
    # Test suites (kit, zoo, bot), run as subprocesses.
    suites = {}
    for path in ("sim/kit/tests", "sim/zoo/tests", "bot/tests"):
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", path], cwd=ROOT,
                           capture_output=True, text=True)
        tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-3:]
        suites[path] = {"returncode": p.returncode, "tail": tail, "secs": round(time.time() - t0, 1)}
        print(f"  pytest {path}: rc {p.returncode} | {tail[-1] if tail else ''}", flush=True)
    res["tests"] = suites
    res["finished"] = stamp()
    C.save(res, "b0.json")
    return res


# ----------------------------------------------------------------------------- B2 headline
def b2top3(args, pool):
    n = args.tables or 12000
    for f in FIELDS:
        cell(f"b2_{f}_TOP3_TURBO.json", args.force, pool, field=f, occupants=VERSIONS + SUBSTITUTES, n_tables=n,
             seed=args.seed or 21, context="v2", label=f"B2 {f} TOP3 TURBO")


# ----------------------------------------------------------------------------- AI-written fields
def ai(args, pool):
    """b2top3's occupants against the AI-written fields. AI bots time out at their measured rates (time scale 1.5)."""
    from sim.field_ai import fields as AF
    n = args.tables or 5000
    for f in (args.fields.split(",") if args.fields else ["AI_ONLY", "AI_MIX"]):
        cell(f"ai_{f}_TOP3_TURBO.json", args.force, pool, field=f, occupants=VERSIONS + SUBSTITUTES, n_tables=n,
             seed=args.seed or 24, context="v2", label=f"AI {f} TOP3 TURBO", sandbox_p=AF.sandbox_p("1.5"))


# ----------------------------------------------------------------------------- B1 grid
def b1(args, pool):
    order = ["icm_pro", "chart_tag", "nit", "pushfold_maniac", "lag_aggro", "station", "buggy", "random_legal", "dead"]
    for k in order:
        n = args.tables or 10000
        cell(f"b1_{k}_TOP3_TURBO.json", args.force, pool, field=k, occupants=VERSIONS, n_tables=n, seed=31,
             label=f"B1 7 x {k} TOP3 TURBO")


# ----------------------------------------------------------------------------- B2 payouts and schedules
def b2pay(args, pool):
    for pay, n in (("LINEAR", 5000), ("WTA", 20000)):
        for f in FIELDS:
            cell(f"b2_{f}_{pay}_TURBO.json", args.force, pool, field=f, occupants=VERSIONS, n_tables=args.tables or n,
                 seed=22, payouts=pay, label=f"B2 {f} {pay} TURBO")


def b2sched(args, pool):
    for sch in ("FAST", "SLOW"):
        for f in FIELDS:
            cell(f"b2_{f}_TOP3_{sch}.json", args.force, pool, field=f, occupants=VERSIONS, n_tables=args.tables or 10000,
                 seed=23, schedule=sch, label=f"B2 {f} TOP3 {sch}")


# ----------------------------------------------------------------------------- B3 simulated event
LADDER = ["dead", "random_legal", "buggy", "station", "nit", "lag_aggro", "pushfold_maniac", "chart_tag", "icm_pro"]


def drift_mix(t):
    """Seat shares of tournament t (1..6): linear drift from T1_like (t=1) to T6_like (t=6)."""
    a = (t - 1) / 5.0
    keys = set(F.T1_like) | set(F.T6_like)
    return {k: (1 - a) * F.T1_like.get(k, 0) + a * F.T6_like.get(k, 0) for k in keys}


def team_archetype(u, mix):
    """Quantile coupling: the archetype at quantile u of the mix ordered by the sophistication LADDER. As the mix
    drifts toward T6, a team's quantile maps to the same or a more sophisticated archetype (it 'fixes its bot')."""
    tot = float(sum(mix.values()))
    cum = 0.0
    for k in LADDER:
        cum += mix.get(k, 0.0) / tot
        if u < cum:
            return k
    return LADDER[-1]


def b3_field(rep, n_rivals):
    import random
    rng = random.Random(f"b3-field:{rep}")
    if rep == 0:
        us = [(i + 0.5) / n_rivals for i in range(n_rivals)]          # stratified: shares match the mix
    else:
        us = sorted(rng.random() for _ in range(n_rivals))
    teams = []
    for i, u in enumerate(us):
        traj = [team_archetype(u, drift_mix(t)) for t in range(1, 7)]
        params = {}
        for arch in sorted(set(traj)):
            params[arch] = F.jitter_params(arch, random.Random(f"b3-team:{rep}:{i}:{arch}"))
        teams.append({"team": f"team{i + 1:02d}", "u": u, "trajectory": traj, "params": params})
    return teams


def b3_entrants(teams, t, ours="v2"):
    import functools
    ents = [("ours", C.factory_for(ours))]
    for tm in teams:
        arch = tm["trajectory"][t - 1]
        p = tm["params"][arch]
        sp = C.SANDBOX_P.get(arch, 0.0)
        fac = functools.partial(C.Sandboxed, key=arch, p=sp, params=p) if sp > 0 else F.factory(arch, **p)
        ents.append((tm["team"], fac))
    return ents


def b3(args, pool):
    from sim.kit.runner import run
    n_rivals = 19
    reps = [(0, args.iters or 10000)] + [(r, args.iters_extra or 4000) for r in range(1, 1 + args.reps_extra)]
    for rep, iters in reps:
        name = f"b3_rep{rep}.json"
        if exists(name) and not args.force:
            print(f"skip {name} (exists)", flush=True)
            continue
        teams = b3_field(rep, n_rivals)
        k = n_rivals + 1
        per_iter = -(-k // 8)
        n_tables = iters * per_iter
        tours = []
        t_rep = time.time()
        for t in range(1, 7):
            ents = b3_entrants(teams, t, args.ours)
            t0 = time.time()
            res = run(ents, n_tables, seed=3000 + 10 * rep + t, payouts="TOP3", schedule="TURBO", workers=args.workers,
                      seating="event", time_limit_ms=None)
            scores = {}
            for e in res.entrants:
                r = C.rel_ci(e.points)
                scores[e.name] = {"rel": r["rel"], "half": r["half"], "n": r["n"], "points_mean": r["points"],
                                  "sd": r["sd_points"],
                                  "archetype": ("ours:" + args.ours) if e.name == "ours" else
                                  next(tm["trajectory"][t - 1] for tm in teams if tm["team"] == e.name)}
            ours_st = C.stats_digest(res.entrants[0].stats)
            tours.append({"t": t, "mix": drift_mix(t), "n_tables": n_tables, "iterations": iters,
                          "elapsed_s": time.time() - t0, "tables_per_s": res.meta["tables_per_s"],
                          "loadavg": [res.meta["loadavg_start"], res.meta["loadavg_end"]], "scores": scores,
                          "ours_stats": ours_st, "ours_points": [round(p, 4) for p in res.entrants[0].points]})
            our = scores["ours"]
            best = max((v["rel"], n) for n, v in scores.items() if n != "ours")
            print(f"  B3 rep {rep} T{t}: {n_tables} tables in {time.time() - t0:.0f} s; ours rel {our['rel']:.3f} "
                  f"+-{our['half']:.3f}; best rival {best[1]} {best[0]:.3f}", flush=True)
        names = [n for n in tours[0]["scores"]]
        final = {}
        for nme in names:
            w = sum(t["t"] * t["scores"][nme]["rel"] for t in tours) / 21.0
            se = (sum((t["t"] / 21.0) ** 2 * (t["scores"][nme]["half"] / 1.959964) ** 2 for t in tours)) ** 0.5
            final[nme] = {"weighted_rel": w, "half": 1.959964 * se,
                          "per_t": [t["scores"][nme]["rel"] for t in tours],
                          "trajectory": [t["scores"][nme]["archetype"] for t in tours]}
        ranking = sorted(final, key=lambda x: -final[x]["weighted_rel"])
        out = {"rep": rep, "iterations": iters, "n_entrants": k, "tables_per_tournament": n_tables, "ours": args.ours,
               "teams": teams, "tournaments": tours, "final": final, "ranking": ranking,
               "our_rank": ranking.index("ours") + 1, "elapsed_s": time.time() - t_rep, "finished": stamp()}
        C.save(out, name)
        print(f"  B3 rep {rep}: our rank {out['our_rank']} of {k}; weighted {final['ours']['weighted_rel']:.3f}; "
              f"top 3: {[(n, round(final[n]['weighted_rel'], 3)) for n in ranking[:3]]}", flush=True)


# ----------------------------------------------------------------------------- B4 reliability
def b4(args, pool_unused):
    env = {"NOLIMIT_BOT_STRICT": "1"}
    pool = C.make_pool(args.workers or 8, env=env)
    grid = [("T6_like", "TOP3", "TURBO"), ("T1_like", "TOP3", "TURBO"), ("STRONG", "TOP3", "TURBO"),
            ("T6_like", "LINEAR", "FAST"), ("T6_like", "WTA", "SLOW"), ("T6_like", "TOP4", "TURBO"),
            ("STRONG", "TOP4", "SLOW"), ("T1_like", "LINEAR", "FAST")]
    n = args.tables or 1100
    try:
        for f, pay, sch in grid:
            cell(f"b4_{f}_{pay}_{sch}.json", args.force, pool, field=f, occupants=["v2"], n_tables=n, seed=41,
                 payouts=pay, schedule=sch, time_limit_ms=1.0, timing="wall", label=f"B4 STRICT wall {f} {pay} {sch}",
                 keep_hist=True)
    finally:
        pool.close()
        pool.join()


def b4q(args, pool_unused):
    """B4 in ONE process (no worker pool): the cleanest wall clock this shared laptop can give. STRICT is set here,
    so run this step in its own invocation (bot.shell reads NOLIMIT_BOT_STRICT at import)."""
    os.environ["NOLIMIT_BOT_STRICT"] = "1"
    if "bot.shell" in sys.modules and not sys.modules["bot.shell"].STRICT:
        raise SystemExit("b4q: bot.shell was imported before STRICT was set; run 'run_bench.py b4q' on its own")
    grid = [("T6_like", "TOP3", "TURBO", 3000), ("STRONG", "TOP3", "TURBO", 2500), ("T1_like", "TOP3", "TURBO", 2200)]
    for f, pay, sch, n in grid:
        cell(f"b4q_{f}_{pay}_{sch}.json", args.force, None, field=f, occupants=["v2"], n_tables=args.tables or n,
             seed=42, payouts=pay, schedule=sch, time_limit_ms=1.0, timing="wall", workers=1, keep_hist=True,
             label=f"B4q one process STRICT wall {f} {pay} {sch}")
    from bot import shell as SH
    print(f"  STRICT in this process: {SH.STRICT}", flush=True)


# ----------------------------------------------------------------------------- B5 knob A/Bs
KNOBS = ["v2:PF_DEPTH_BB=12", "v2:CALL_LAM=2.0", "v2:VALUE_FRAC=0.6", "v2:REJAM_PRIOR=0.12", "v2:BB_DEFEND_PCT=0.15",
         "v2:CBET_AIR_UNKNOWN=0.5"]


def b5knobs(args, pool):
    from bot import brain
    knobs = [k for k in (args.knobs.split(";") if args.knobs else KNOBS)
             if k.split(":", 1)[1].split("=")[0] in brain.PARAMS]
    tag = f"_{args.tag}" if args.tag else ""
    for f in (args.fields.split(",") if args.fields else ["T6_like"]):
        cell(f"b5_knobs{tag}_{f}_TOP3_TURBO.json", args.force, pool, field=f, occupants=["v2"] + knobs,
             n_tables=args.tables or 10000, seed=args.seed or 51, context="v2", label=f"B5 knobs{tag} {f} TOP3 TURBO")


COMBO = "v2:PF_DEPTH_BB=12,JAM_MULT=1.15"


def b5cross(args, pool):
    """The confirmed TOP3/TURBO knob change, checked under the other payouts and schedules (paired against v2)."""
    combo = args.knobs or COMBO
    grid = [("T6_like", "LINEAR", "TURBO", 5000), ("T6_like", "WTA", "TURBO", 20000), ("T6_like", "TOP3", "FAST", 10000),
            ("T6_like", "TOP3", "SLOW", 10000), ("STRONG", "LINEAR", "TURBO", 5000), ("STRONG", "WTA", "TURBO", 20000)]
    for f, pay, sch, n in grid:
        cell(f"b5_cross_{f}_{pay}_{sch}.json", args.force, pool, field=f, occupants=["v2", combo],
             n_tables=args.tables or n, seed=81, payouts=pay, schedule=sch, context="v2",
             label=f"B5 cross {f} {pay} {sch}")


def _parse_levels(text):
    """'TURBO' or '10/20,15/30,25/50' or '10/20/0,20/40/5' -> a schedule the kit accepts."""
    if "/" not in text:
        return text
    return [tuple(int(x) for x in lv.split("/")) for lv in text.split(",")]


def _parse_payouts(text):
    """'TOP3' or '50,30,20' -> a payout the kit accepts (padded to 8 places by the kit)."""
    if "," not in text and not text.replace(".", "").isdigit():
        return text
    return [float(x) for x in text.split(",")]


def quick(args, pool):
    """Quick check for a given payout and blind schedule: v0/v1/v2 (and icm_pro in our seat) on the same tables in
    T6_like and STRONG. Example: run_bench.py quick --payouts 40,30,20,10 --schedule 10/20,20/40,30/60 --tables 2000"""
    pay = _parse_payouts(args.payouts or "TOP3")
    sch = _parse_levels(args.schedule or "TURBO")
    for f in (args.fields.split(",") if args.fields else ["T6_like", "STRONG"]):
        extra = [k for k in args.knobs.split(";") if k] if args.knobs else []
        out = C.run_cell(f, ["v0", "v1", "v2", "icm_pro"] + extra, args.tables or 2000, seed=61, payouts=pay, schedule=sch,
                         pool=pool, context="v2", label=f"QUICK {f} {args.payouts or 'TOP3'} {args.schedule or 'TURBO'}")
        out["finished"] = stamp()
        C.save(out, f"quick_{f}.json")


STEPS = {"b0": b0, "b2top3": b2top3, "ai": ai, "b1": b1, "b2pay": b2pay, "b2sched": b2sched, "b3": b3, "b4": b4,
         "b4q": b4q, "b5knobs": b5knobs, "b5cross": b5cross, "quick": quick}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=sorted(STEPS) + ["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tables", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--iters-extra", type=int, default=None)
    ap.add_argument("--reps-extra", type=int, default=2)
    ap.add_argument("--ours", default="v2")
    ap.add_argument("--knobs", default="", help="';'-separated occupants such as 'v2:CALL_LAM=2.0;v2:JAM_MULT=1.15'")
    ap.add_argument("--tag", default="", help="suffix for the b5knobs output file")
    ap.add_argument("--seed", type=int, default=None, help="b2top3 / ai / b5knobs: bench seed (defaults 21 / 24 / 51)")
    ap.add_argument("--fields", default="", help="comma-separated fields for ai / b5knobs / quick")
    ap.add_argument("--payouts", default="", help="quick: a name (TOP3, LINEAR, WTA, TOP4) or points by place, '40,30,20,10'")
    ap.add_argument("--schedule", default="", help="quick: a name (TURBO, FAST, SLOW) or levels 'sb/bb[/ante],...'")
    a = ap.parse_args(argv)
    steps = ["b0", "b2top3", "b1", "b3", "b2pay", "b2sched", "b4", "b5knobs"] if "all" in a.steps else a.steps
    os.makedirs(C.OUT, exist_ok=True)
    pool = None
    try:
        for s in steps:
            t0 = time.time()
            print(f"=== {s} start {stamp()}", flush=True)
            if s not in ("b4", "b4q", "b3") and pool is None:
                pool = C.make_pool(a.workers)
            STEPS[s](a, pool)
            print(f"=== {s} done in {time.time() - t0:.0f} s ({stamp()})", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
