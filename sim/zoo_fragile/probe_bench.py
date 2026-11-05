"""Fragility-probe bench: paired cells that also count the errors OUR action sizes induce in the opponents' seats.

A cell = one field, our-seat occupants (e.g. "v2" and "v2:SIZING_PROBE=True"), n tables, a seed. Same machinery and
pairing as sim/bench/common.run_cell (same table plan, kit table seed, AI/zoo sandbox timeout model sandbox_p("1.5"),
clock off in score runs), plus, per occupant and per table: the number of exceptions and illegal actions the engine
counted in the 7 opponent seats, the table's hands and the opponents' decisions. The paired difference of the
per-table error counts (occupant minus baseline) says whether our sizes cause the errors.

Fields: any sim/bench field (zoo mixes, AI_ONLY, AI_MIX, an archetype or ai_ slug for 7 copies), a fragile key
(frag_bbround, ...; 7 copies) or FRAG_MIX (each seat drawn uniformly from the six fragile bots).

    .venv/bin/python sim/zoo_fragile/probe_bench.py --field FRAG_MIX --tables 1000 --seed 501 \
        --occupants "v2;v2:SIZING_PROBE=True" --json sim/zoo_fragile/out/x.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.bench import common as C                    # noqa: E402
from sim.kit.engine import Table                     # noqa: E402
from sim.kit.payouts import average_seat, get_payouts  # noqa: E402
from sim.zoo import fields as ZF                     # noqa: E402
from sim.zoo_fragile.bots import FRAG_MIX, FRAGILE   # noqa: E402

N = 8


def plan(field, seed, tid):
    if field in FRAGILE or field == "FRAG_MIX":
        rng = random.Random(f"bench-plan:{seed}:{field}:{tid}")
        our = rng.randrange(N)
        opp = ZF.sample_opponents(FRAG_MIX, rng, N - 1) if field == "FRAG_MIX" else [field] * (N - 1)
        return our, opp
    return C.plan_table(field, seed, tid)


_SP = None

# Masked errors. Every AI-written bot wraps act() in try/except and returns a safe action on any exception, so the
# engine never sees those errors. sys.monitoring (Python 3.12+) reports every exception CAUGHT inside a function
# defined in sim/field_ai/ (EXCEPTION_HANDLED, attributed to the handling function); the per-table count is the
# opponents' masked errors. Exceptions the bots use for control flow show up too, by (file, function, type).
_MASKED = {}            # (slug, handler function, exception type) -> count
_MASKED_N = [0]
_MON = [False]          # monitoring active in this worker


def _on_handled(code, offset, exc):
    fn = code.co_filename
    if "field_ai" in fn:
        key = (os.path.basename(fn)[:-3], code.co_name, type(exc).__name__)
        _MASKED[key] = _MASKED.get(key, 0) + 1
        _MASKED_N[0] += 1


def _monitor():
    mon = getattr(sys, "monitoring", None)
    if mon is None:
        return False
    tid = 4                                      # a free tool id (0-5); 4 is unassigned by CPython
    try:
        mon.use_tool_id(tid, "probe_bench")
    except ValueError:
        return False
    mon.register_callback(tid, mon.events.EXCEPTION_HANDLED, _on_handled)
    mon.set_events(tid, mon.events.EXCEPTION_HANDLED)
    return True


def fac(name):
    if name in FRAGILE:
        return FRAGILE[name]
    return C._fac(name, _SP)


def _init(root, sp):
    global _SP
    if root not in sys.path:
        sys.path.insert(0, root)
    _SP = sp
    _MON[0] = _monitor()


def run_chunk(task):
    cell, tids = task
    occs = cell["occupants"]
    kw = dict(payouts=cell["payouts"], schedule=cell["schedule"], time_limit_ms=None)
    out = {o: {"rows": [], "opp": {}, "ours": [0, 0, 0], "ex": {}, "masked": {}} for o in occs}
    t0 = time.perf_counter()
    for tid in tids:
        our, opp = plan(cell["field"], cell["seed"], tid)
        for o in occs:
            facs, names, grabs = [], [], []
            k = 0
            for s in range(N):
                nm = o if s == our else opp[k]
                if s != our:
                    k += 1
                g = C._Grab(fac(nm))
                facs.append(g)
                names.append(nm)
                grabs.append(g)
            m0 = _MASKED_N[0]
            snap = dict(_MASKED)
            res = Table(facs, seed=C.table_seed(cell["seed"], tid), **kw).play()
            masked = _MASKED_N[0] - m0
            if masked:
                for key, c in _MASKED.items():
                    d = c - snap.get(key, 0)
                    if d:
                        k2 = "|".join(key)
                        out[o]["masked"][k2] = out[o]["masked"].get(k2, 0) + d
            rec = out[o]
            exc = ill = dec = 0
            for s in range(N):
                if s == our:
                    continue
                st = res.stats[s]
                a = C.archetype_of(names[s]) if names[s] not in FRAGILE else names[s]
                g = rec["opp"].setdefault(a, [0, 0, 0, 0])      # exceptions, illegal, decisions, seat-tables
                g[0] += st.exceptions
                g[1] += st.illegal
                g[2] += st.decisions
                g[3] += 1
                exc += st.exceptions
                ill += st.illegal
                dec += st.decisions
                for cat in ("exception", "illegal"):
                    for msg in st.examples.get(cat, ())[:2]:
                        lst = rec["ex"].setdefault(f"{a}|{cat}", [])
                        if len(lst) < 4:
                            lst.append(msg)
            sto = res.stats[our]
            rec["ours"][0] += sto.exceptions + sto.illegal
            for a in grabs[our].made:
                rec["ours"][1] += getattr(a, "errors", 0)
                rec["ours"][2] += getattr(a, "illegal_fixed", 0)
            rec["rows"].append((tid, res.points[our], res.hands_played, exc, ill, dec, masked))
    return {"occ": out, "secs": time.perf_counter() - t0, "monitor": _MON[0]}


def _mci(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = 1.959964 * sd / math.sqrt(n) if n > 1 else float("nan")
    return m, m - h, m + h


def run_cell(field, occupants, n_tables, seed, workers=4, payouts="TOP3", schedule="TURBO", pool=None):
    cell = {"field": field, "occupants": occupants, "payouts": payouts, "schedule": schedule, "seed": seed}
    chunk = max(1, min(40, n_tables // (workers * 6) or 1))
    tasks = [(cell, list(range(lo, min(n_tables, lo + chunk)))) for lo in range(0, n_tables, chunk)]
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    results = list(pool.imap_unordered(run_chunk, tasks))
    elapsed = time.perf_counter() - t0
    avg = average_seat(get_payouts(payouts, N), N)
    occ = {}
    for o in occupants:
        rows, opp, ours, ex, masked = [], {}, [0, 0, 0], {}, {}
        for r in results:
            x = r["occ"][o]
            rows.extend(x["rows"])
            for k, v in x.get("masked", {}).items():
                masked[k] = masked.get(k, 0) + v
            for a, g in x["opp"].items():
                acc = opp.setdefault(a, [0, 0, 0, 0])
                for i in range(4):
                    acc[i] += g[i]
            for i in range(3):
                ours[i] += x["ours"][i]
            for k, v in x["ex"].items():
                ex.setdefault(k, []).extend(v[: max(0, 4 - len(ex.get(k, [])))])
        rows.sort()
        hands = sum(r[2] for r in rows)
        errs = sum(r[3] + r[4] for r in rows)
        occ[o] = {"rows": rows, "rel": C.rel_ci([r[1] for r in rows], avg), "hands": hands,
                  "opp_exceptions": sum(r[3] for r in rows), "opp_illegal": sum(r[4] for r in rows),
                  "opp_decisions": sum(r[5] for r in rows), "opp_errors_per_1k_hands": 1000.0 * errs / hands,
                  "ours_errors_engine": ours[0], "ours_errors_masked": ours[1], "ours_illegal_fixed": ours[2],
                  "by_opponent": {a: {"exceptions": g[0], "illegal": g[1], "decisions": g[2], "seat_tables": g[3],
                                      "errors_per_1k_decisions": 1000.0 * (g[0] + g[1]) / g[2] if g[2] else 0.0}
                                  for a, g in sorted(opp.items())},
                  "opp_masked": sum(r[6] for r in rows),
                  "opp_masked_per_1k_hands": 1000.0 * sum(r[6] for r in rows) / hands,
                  "masked_by_handler": dict(sorted(masked.items(), key=lambda kv: -kv[1])),
                  "examples": ex}
    base = occupants[0]
    comps = {}
    b = occ[base]["rows"]
    for o in occupants[1:]:
        x = occ[o]["rows"]
        assert [r[0] for r in x] == [r[0] for r in b]
        p = C.paired_ci([r[1] for r in x], [r[1] for r in b], avg)
        de = [(r[3] + r[4]) - (q[3] + q[4]) for r, q in zip(x, b)]
        dr = [1000.0 * (r[3] + r[4]) / r[2] - 1000.0 * (q[3] + q[4]) / q[2] for r, q in zip(x, b)]
        dm = [1000.0 * r[6] / r[2] - 1000.0 * q[6] / q[2] for r, q in zip(x, b)]
        m, lo, hi = _mci(de)
        m2, lo2, hi2 = _mci(dr)
        m3, lo3, hi3 = _mci(dm)
        comps[o] = {"score": p, "opp_errors_per_table_diff": [m, lo, hi], "opp_errors_per_1k_hands_diff": [m2, lo2, hi2],
                    "opp_masked_per_1k_hands_diff": [m3, lo3, hi3]}
    out = {"field": field, "seed": seed, "n_tables": n_tables, "payouts": payouts, "schedule": schedule,
           "occupants": occupants, "elapsed_s": elapsed, "loadavg_start": list(load0),
           "masked_monitor_active": all(r.get("monitor") for r in results),
           "loadavg_end": list(os.getloadavg()), "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "occ": {o: {k: v for k, v in d.items() if k != "rows"} for o, d in occ.items()},
           "points": {o: [round(r[1], 4) for r in d["rows"]] for o, d in occ.items()},
           "comparisons": comps}
    return out


def brief(c):
    lines = [f"[{c['field']} seed {c['seed']}] {c['n_tables']} tables, {c['elapsed_s']:.0f} s, load "
             f"{c['loadavg_start'][0]:.1f}->{c['loadavg_end'][0]:.1f}, {c['finished']}"]
    for o, d in c["occ"].items():
        r = d["rel"]
        lines.append(f"   {o:<28} rel {r['rel']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  opp exc {d['opp_exceptions']} "
                     f"ill {d['opp_illegal']} = {d['opp_errors_per_1k_hands']:.2f} per 1k hands; masked "
                     f"{d['opp_masked']} = {d['opp_masked_per_1k_hands']:.2f}; ours "
                     f"{d['ours_errors_engine']}/{d['ours_errors_masked']}/{d['ours_illegal_fixed']}")
    for o, p in c["comparisons"].items():
        s = p["score"]
        e = p["opp_errors_per_1k_hands_diff"]
        lines.append(f"   {o} - {c['occupants'][0]}: score {s['diff_rel']:+.3f} [{s['lo_rel']:+.3f}, {s['hi_rel']:+.3f}]"
                     f"  opp errors/1k hands {e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}]"
                     f"  masked/1k hands {p['opp_masked_per_1k_hands_diff'][0]:+.2f} "
                     f"[{p['opp_masked_per_1k_hands_diff'][1]:+.2f}, {p['opp_masked_per_1k_hands_diff'][2]:+.2f}]")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fields", required=True, help="comma-separated")
    ap.add_argument("--occupants", default="v2;v2:SIZING_PROBE=True", help="';'-separated, the first is the baseline")
    ap.add_argument("--tables", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=501)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    from sim.field_ai import fields as AF
    sp = AF.sandbox_p("1.5")
    occs = [o.strip() for o in a.occupants.split(";") if o.strip()]
    os.makedirs(a.outdir, exist_ok=True)
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(ROOT, sp))
    try:
        for field in [f.strip() for f in a.fields.split(",") if f.strip()]:
            c = run_cell(field, occs, a.tables, a.seed, a.workers, a.payouts, a.schedule, pool)
            print(brief(c), flush=True)
            path = os.path.join(a.outdir, f"{field}_s{a.seed}_n{a.tables}{a.tag}.json")
            with open(path + ".tmp", "w") as f:
                json.dump(c, f, default=str)
            os.replace(path + ".tmp", path)
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
