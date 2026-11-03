"""build_tables: regenerate the bot's tables for any VariantSpec (variants_deep notes s6.2, R6 and R7).

    from variant.spec import nlhe_spec
    from variant.build_tables import build_tables
    res = build_tables(nlhe_spec(), workers=8, time_budget=60, out_path="variant/out/tables_nlhe.py")

or from the shell (example specs, or any "module:function" returning a VariantSpec):

    .venv/bin/python -m variant.build_tables --spec nlhe --workers 8 --budget 60 --out variant/out/tables_nlhe.py

Stages (all multi-core, standard library multiprocessing; the budget is shared and leftovers roll forward):
  a  equity of every class vs a random hand (range engine, one all-ones range)
  m  class-vs-class equity matrix (matrix engine; only when n_classes <= matrix_max and the game is separable)
  c  push/fold re-solve on the matrix (solver.py, a port of sng_icm pushfold.py): heads-up Nash by stack -> PUSH_ORDER,
     CALL_ORDER and the SB jam fit; 8-handed equal-stack Nash -> the k >= 2 jam fit and CALL_FIT
  b  equity of every class vs the top-J range (J on the bot's EQ_GRID; ranges in PUSH order as the bot assumes, else
     the given range_order, else equity-vs-random order); the bot reads it through eq_vs(class, J)
After every stage, and every `checkpoint_every` seconds inside the MC stages, the tables module is rewritten
atomically with whatever exists, so a late build always has something to ship. Missing parts use labelled fallbacks:
EQ_TABLE from a linear model on equity vs random (NLHE-fitted slopes, up to 0.3 off at J = 3%), orders = equity-vs-
random order, JAM_FIT/CALL_FIT = the bot's NLHE fits.

Output: a Python module of plain literals with the bot_tables.py names (PUSH_ORDER, CALL_ORDER, EQ_GRID, EQ_LABELS,
EQ_TABLE) plus EQ_RANDOM, COMBOS, TOTAL_COMBOS, JAM_FIT, CALL_FIT, SANITY, META. Labels are space-free.
"""
import dataclasses
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time

from . import mc
from . import solver as SV
from .handspace import HandSpace

EQ_GRID = (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)     # bot_tables.EQ_GRID
NLHE_JAM_FIT = (3.29, -0.717, -0.917, 1.41, -0.41)      # bot PARAMS (sng_icm fit_rules.json, exp of the log fits)
NLHE_CALL_FIT = (2.64, 0.971, -0.582)
# eq(class, J) ~ c0(J) + c1(J) * (eq_random(class) - mean eq_random of the top-J range), fitted on the bot's NLHE table
MODEL_PRIOR = ((0.03, 0.4425, 0.77), (0.05, 0.4435, 0.7257), (0.08, 0.4357, 0.6327), (0.1, 0.4463, 0.6628),
               (0.15, 0.4659, 0.7352), (0.2, 0.4783, 0.7932), (0.25, 0.4862, 0.8367), (0.3, 0.4923, 0.8782),
               (0.4, 0.4988, 0.936), (0.5, 0.5023, 0.9815), (0.6, 0.5031, 1.0048), (0.7, 0.5037, 1.0389),
               (0.85, 0.502, 1.0517), (1.0, 0.5, 1.0))
NLHE_ANCHORS = {"AA": (0.845, 0.860), "KK": (0.816, 0.832), "AKs": (0.66, 0.68), "72o": (0.335, 0.355),
                "22": (0.495, 0.512), "32o": (0.315, 0.335)}
DEFAULT_SHARES = {"a": 0.10, "m": 0.20, "c": 0.20, "b": 0.50}
GIVEN_ORDER_SHARES = {"a": 0.05, "m": 0.15, "c": 0.15, "b": 0.65}   # stage b does not wait for the solver's order


def _log_default(msg):
    print(f"[build_tables {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _State:
    """Accumulators and results of a build (what the checkpoint writer reads)."""

    def __init__(self, hs, grid):
        C = hs.n_classes
        self.hs = hs
        self.grid = tuple(grid)
        self.a_num = [0.0] * C
        self.a_den = [0.0] * C
        self.a_boards = 0
        self.m_num2 = None
        self.m_boards = 0
        self.E = None
        self.Wavg = None
        self.b_num = [[0.0] * len(grid) for _ in range(C)]
        self.b_den = [[0.0] * len(grid) for _ in range(C)]
        self.b_boards = 0
        self.b_order = None
        self.b_order_src = None
        self.push = None
        self.call = None
        self.jam_fit = None
        self.call_fit = None
        self.fit_info = None
        self.stage_s = {}
        self.stage_done = {}
        self.stage_note = {}
        self.stderr = {}

    def eq_random(self):
        """Pooled equity vs random from stage a and the J = 1 column of stage b (same estimand)."""
        C = self.hs.n_classes
        num = list(self.a_num)
        den = list(self.a_den)
        if self.b_boards and self.grid[-1] >= 1.0:
            for i in range(C):
                num[i] += self.b_num[i][-1]
                den[i] += self.b_den[i][-1]
        if all(d > 0 for d in den):
            return [n / (2.0 * d) for n, d in zip(num, den)], "mc"
        if self.E is not None:
            return [sum(w * e for w, e in zip(rw, re)) / sum(rw) for rw, re in zip(self.Wavg, self.E)], "matrix"
        return None, None


def _static_strength(hs):
    """Fallback strength before any MC: static score of each class's kept cards, scaled to 0.3 .. 0.85."""
    from .spec import static_score
    sc = [static_score(hs.kept_of(i), hs.slot) for i in range(hs.n_classes)]
    lo, hi = min(sc), max(sc)
    return [0.3 + 0.55 * (s - lo) / ((hi - lo) or 1.0) for s in sc]


def _model_table(hs, ev, order, grid):
    """Provisional EQ_TABLE from equity vs random alone (MODEL_PRIOR slopes, interpolated in J)."""
    pj = [p[0] for p in MODEL_PRIOR]

    def coef(J):
        if J <= pj[0]:
            return MODEL_PRIOR[0][1:]
        for k in range(1, len(pj)):
            if J <= pj[k]:
                t = (J - pj[k - 1]) / (pj[k] - pj[k - 1])
                a, b = MODEL_PRIOR[k - 1][1:], MODEL_PRIOR[k][1:]
                return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        return MODEL_PRIOR[-1][1:]
    C = hs.n_classes
    cols = []
    for J in grid:
        r = hs.top_range(order, J)
        w = [a * b for a, b in zip(r, hs.combos)]
        m = sum(x * e for x, e in zip(w, ev)) / sum(w)
        c0, c1 = coef(J)
        cols.append([min(0.99, max(0.01, c0 + c1 * (e - m))) for e in ev])
    return [[cols[g][i] for g in range(len(grid))] for i in range(C)]


def _matrix_table(st, order, grid):
    """EQ_TABLE from the matrix exactly as sng_icm fit_eq_vs_range.py: sum_j W_ij E_ij r_j / sum_j W_ij r_j."""
    hs = st.hs
    C = hs.n_classes
    out = [[0.0] * len(grid) for _ in range(C)]
    for g, J in enumerate(grid):
        r = hs.top_range(order, J)
        for i in range(C):
            rw, re = st.Wavg[i], st.E[i]
            den = sum(w * x for w, x in zip(rw, r))
            out[i][g] = sum(w * e * x for w, e, x in zip(rw, re, r)) / den if den > 0 else 0.5
    return out


def assemble(st, spec):
    """Current best tables (dict of plain values) from a build state, with provenance per part."""
    hs = st.hs
    C = hs.n_classes
    ev, ev_src = st.eq_random()
    if ev is None:
        ev, ev_src = _static_strength(hs), "static_score"
    evr_order = sorted(range(C), key=lambda i: (-ev[i], i))
    push = st.push or evr_order
    call = st.call or evr_order
    order_src = "solver" if st.push else "eq_random"
    b_order = st.b_order or push
    if st.E is not None:
        table = _matrix_table(st, b_order, st.grid)
        t_src = f"matrix ({st.m_boards} boards)"
    else:
        table = _model_table(hs, ev, b_order, st.grid)
        t_src = f"linear model on equity vs random ({ev_src}); provisional, up to 0.3 off at J = 3%"
    # stage b rows replace the matrix rows once b has at least as many boards (a young b is noisier per cell)
    if st.b_boards and (st.E is None or st.b_boards >= st.m_boards):
        filled = 0
        for i in range(C):
            rn, rd, row = st.b_num[i], st.b_den[i], table[i]
            for g in range(len(st.grid)):
                if rd[g] > 0:
                    row[g] = rn[g] / (2.0 * rd[g])
                    filled += 1
        base = t_src
        t_src = f"mc_range ({st.b_boards} boards, ranges in {st.b_order_src} order)"
        if filled < C * len(st.grid):
            t_src += f"; {C * len(st.grid) - filled} empty cells from {base}"
    lab = hs.labels
    return {
        "SPEC_NAME": spec.name,
        "SPEC_SUMMARY": spec.summary(),
        "PUSH_ORDER": " ".join(lab[i] for i in push),
        "CALL_ORDER": " ".join(lab[i] for i in call),
        "EQ_GRID": tuple(float(x) for x in st.grid),
        "EQ_LABELS": " ".join(lab),
        "EQ_TABLE": tuple(tuple(round(v, 4) for v in row) for row in table),
        "EQ_RANDOM": tuple(round(v, 4) for v in ev),
        "COMBOS": tuple(hs.combos),
        "TOTAL_COMBOS": hs.total_combos,
        "JAM_FIT": tuple(st.jam_fit or NLHE_JAM_FIT),
        "CALL_FIT": tuple(st.call_fit or NLHE_CALL_FIT),
        "SOURCES": {"EQ_RANDOM": ev_src, "EQ_TABLE": t_src, "ORDERS": order_src,
                    "FITS": ("solver refit: " + ", ".join((st.fit_info or {}).get("refit_parts", [])))
                    if st.jam_fit else "NLHE prior (bot PARAMS)"},
    }


def write_module(path, tables, sanity, meta):
    """Atomically write the tables as a Python module of literals (bot_tables.py-compatible names)."""
    if not path:
        return None
    lines = ['"""Tables GENERATED by variant/build_tables.py; do not edit by hand.',
             "", f"Spec: {tables['SPEC_SUMMARY']}", f"Sources: {tables['SOURCES']}",
             "Row order of EQ_TABLE, EQ_RANDOM and COMBOS = EQ_LABELS. EQ_TABLE[i][g] = equity of class i vs the",
             "top EQ_GRID[g] fraction of hands in PUSH order (partial class at the boundary).", '"""', ""]
    for k in ("SPEC_NAME", "PUSH_ORDER", "CALL_ORDER", "EQ_GRID", "EQ_LABELS"):
        lines.append(f"{k} = {tables[k]!r}")
    lines.append("EQ_TABLE = (")
    lab = tables["EQ_LABELS"].split()
    for l, row in zip(lab, tables["EQ_TABLE"]):
        lines.append(f"    {row!r},  # {l}")
    lines.append(")")
    for k in ("EQ_RANDOM", "COMBOS", "TOTAL_COMBOS", "JAM_FIT", "CALL_FIT", "SOURCES"):
        lines.append(f"{k} = {tables[k]!r}")
    body = "\n".join(lines) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    tail = [f"SANITY = {json.loads(json.dumps(sanity))!r}",
            f"META = {json.loads(json.dumps(dict(meta, checksum=digest)))!r}", ""]
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(body + "\n".join(tail))
    os.replace(tmp, path)
    return digest


def load_tables(path):
    """Import a generated tables module by file path."""
    name = "variant_tables_" + hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================================================================== sanity
def sanity_report(st, tables, spec):
    """Checks a table must pass before it may replace the bot's: returns a dict with ok and per-check values."""
    hs = st.hs
    rep = {"checks": {}, "warnings": []}
    ck = rep["checks"]
    C = hs.n_classes
    ev = list(tables["EQ_RANDOM"])
    ck["combos_sum_ok"] = abs(sum(hs.combos) - hs.total_combos) <= max(1, 0.001 * hs.total_combos)
    mean = sum(e * c for e, c in zip(ev, hs.combos)) / float(sum(hs.combos))
    ck["mean_eq_random"] = round(mean, 4)
    ck["mean_eq_random_ok"] = abs(mean - 0.5) <= 0.005 or tables["SOURCES"]["EQ_RANDOM"] == "static_score"
    ck["best_minus_worst"] = round(max(ev) - min(ev), 4)
    ck["spread_ok"] = max(ev) - min(ev) >= 0.15
    lab = tables["EQ_LABELS"].split()
    for name in ("PUSH_ORDER", "CALL_ORDER"):
        ck[name + "_is_permutation"] = sorted(tables[name].split()) == sorted(lab)
    tbl = tables["EQ_TABLE"]
    tol = 0.02
    viol = [lab[i] for i in range(C) if any(tbl[i][g + 1] < tbl[i][g] - tol for g in range(len(st.grid) - 1))]
    ck["monotone_J_violations"] = len(viol)
    ck["monotone_J_examples"] = viol[:8]
    top = sorted(range(C), key=lambda i: -ev[i])[:max(1, C // 20)]
    ck["top5pct_rows_rising"] = all(tbl[i][-1] >= tbl[i][0] - tol for i in top)
    ck["eq_in_unit_interval"] = all(0.0 <= v <= 1.0 for row in tbl for v in row)
    if st.grid[-1] >= 1.0 and tables["SOURCES"]["EQ_TABLE"].startswith("mc_range"):
        ck["table_J1_vs_eq_random_maxdiff"] = round(max(abs(row[-1] - e) for row, e in zip(tbl, ev)), 4)
    if st.E is not None:
        ck["matrix_symmetry_maxdev"] = round(max(abs(st.E[i][j] + st.E[j][i] - 1.0) for i in range(C)
                                                 for j in range(i, C) if st.Wavg[i][j] > 0), 4)
    jf, cf = tables["JAM_FIT"], tables["CALL_FIT"]
    ck["jam_fit_shrinks_with_depth"] = jf[1] < 0 and jf[4] < 0
    ck["call_fit_grows_with_jam_range"] = cf[1] > 0
    if spec.name == "nlhe" or (hs.n_classes == 169 and "AA" in hs.label_index):
        out = {}
        for k, (lo, hi) in NLHE_ANCHORS.items():
            v = ev[hs.label_index[k]]
            out[k] = (round(v, 4), lo <= v <= hi)
        ck["nlhe_anchors"] = out
        ck["nlhe_anchors_ok"] = all(v[1] for v in out.values())
    critical = [k for k in ("combos_sum_ok", "mean_eq_random_ok", "spread_ok", "PUSH_ORDER_is_permutation",
                            "CALL_ORDER_is_permutation", "eq_in_unit_interval", "nlhe_anchors_ok") if k in ck]
    rep["ok"] = all(ck[k] for k in critical)
    rep["failed"] = [k for k in critical if not ck[k]]
    if ck["monotone_J_violations"] > C // 4:
        rep["warnings"].append("many rows fall with J by more than 0.02 (normal for dominated hands; check order)")
    return rep


# ===================================================================== the build
def _pool(workers, spec, hs, start_method):
    method = start_method or ("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    ctx = mp.get_context(method)
    mc.init_worker(spec, hs)                      # parent: inherited by forked workers
    if method == "fork":
        return ctx.Pool(workers), method
    return ctx.Pool(workers, initializer=mc.init_worker, initargs=(spec, hs)), method


def _run_mc(pool, workers, fn, make_args, deadline, chunk_s, on_result, stop=None, min_left=0.05):
    """Keep `workers` tasks in flight until the deadline; each task stops at its own deadline (at least one
    board). on_result(result) merges; stop() -> True ends the stage early."""
    inflight = []
    k = 0
    while True:
        now = time.time()
        stopping = (stop is not None and stop()) or now >= deadline - min_left
        while not stopping and len(inflight) < workers:
            td = min(deadline, now + chunk_s)
            inflight.append(pool.apply_async(fn, (make_args(k, td),)))
            k += 1
        if not inflight:
            break
        progressed = False
        for ar in list(inflight):
            if ar.ready():
                inflight.remove(ar)
                on_result(ar.get())
                progressed = True
        if not progressed:
            time.sleep(0.005)
    return k


def build_tables(spec, evaluator=None, workers=8, time_budget=60.0, out_path=None, hs=None, grid=EQ_GRID,
                 range_order=None, stages=("a", "m", "c", "b"), shares=None, checkpoint_every=5.0, seed=1,
                 hu_iters=1500, multi_iters=200, matrix_max=400, target_a=0.0015, matrix_boards=3000,
                 chunk_s=2.0, ante=None, start_method=None, log=_log_default):
    """Build the tables for `spec` within `time_budget` seconds of wall clock (pool start-up included). Returns a dict
    with tables, sanity, per-stage seconds and counts, and the output path. See the module docstring."""
    t_start = time.time()
    hard_end = t_start + float(time_budget)
    margin = max(0.3, min(1.5, 0.02 * float(time_budget)))      # pool shutdown + final write
    if evaluator is not None:
        spec = dataclasses.replace(spec, evaluator=evaluator)
    if hs is None:
        hs = HandSpace(spec)
    st = _State(hs, grid)
    C = hs.n_classes
    base = DEFAULT_SHARES if range_order is None else GIVEN_ORDER_SHARES
    shares = dict(base, **(shares or {}))
    ante_bb = spec.ante_bb if ante is None else ante
    use_matrix = spec.separable and hs.universe is not None and C <= matrix_max
    plan = [s for s in ("a", "m", "c", "b") if s in stages]
    if not use_matrix:
        for s in ("m", "c"):
            if s in plan:
                plan.remove(s)
                st.stage_note[s] = (f"skipped: {C} classes > matrix_max {matrix_max}" if C > matrix_max else
                                    "skipped: not separable or universe not enumerated")
    if range_order is not None:
        st.b_order = [hs.label_index[o] if isinstance(o, str) else o for o in range_order]
        st.b_order_src = "given"
    engine = "range" if (spec.separable and hs.universe is not None) else "rollout"
    meta = {"spec": spec.name, "classes": C, "engine": engine, "workers": workers, "budget_s": time_budget,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "loadavg_start": _loadavg()}
    last_ck = [0.0]

    def checkpoint(force=False):
        now = time.time()
        if not out_path or (not force and now - last_ck[0] < checkpoint_every):
            return
        last_ck[0] = now
        tables = assemble(st, spec)
        san = sanity_report(st, tables, spec)
        m = dict(meta, stage_s=st.stage_s, stage_done=st.stage_done, stage_note=st.stage_note,
                 boards={"a": st.a_boards, "m": st.m_boards, "b": st.b_boards}, stderr=st.stderr,
                 elapsed_s=round(now - t_start, 2), final=False)
        write_module(out_path, tables, san, m)

    log(f"{hs.describe()}; plan {plan}; engine {engine}; budget {time_budget} s, {workers} workers")
    pool, method = _pool(workers, spec, hs, start_method)
    meta["start_method"] = method
    task_fn = mc.range_task if engine == "range" else mc.rollout_task
    try:
        checkpoint(force=True)
        for si, s in enumerate(plan):
            now = time.time()
            left = hard_end - now
            if left <= 0.2:
                st.stage_note[s] = "skipped: budget exhausted"
                continue
            rest = sum(shares[x] for x in plan[si:])
            deadline = now + left * shares[s] / rest if si < len(plan) - 1 else hard_end - margin
            t0 = time.time()
            if s == "a":
                wv = [(1.0,)] * C
                batches = []

                def on_a(res):
                    num, den, nb = res
                    for i in range(C):
                        st.a_num[i] += num[i][0]
                        st.a_den[i] += den[i][0]
                    st.a_boards += nb
                    batches.append([num[i][0] / (2 * den[i][0]) if den[i][0] > 0 else None for i in range(C)])
                    checkpoint()

                seen = [0, False]

                def stop_a():
                    if seen[1] or len(batches) < 2 * workers or len(batches) == seen[0]:
                        return seen[1]
                    seen[0] = len(batches)
                    se = _batch_stderr(batches)
                    st.stderr["a"] = se
                    seen[1] = se is not None and se <= target_a
                    return seen[1]
                _run_mc(pool, workers, task_fn, lambda k, td: (seed * 1_000_003 + k, 10 ** 9, td, wv),
                        deadline, chunk_s if engine == "range" else 4 * chunk_s, on_a, stop_a)
                st.stderr["a"] = _batch_stderr(batches)
                ev, _ = st.eq_random()
                if ev is not None:
                    hs.set_strength(ev)
                st.stage_done["a"] = ev is not None
            elif s == "m":
                st.m_num2 = [[0] * C for _ in range(C)]

                def on_m(res):
                    num2, nb = res
                    for i in range(C):
                        r = st.m_num2[i]
                        st.m_num2[i] = [x + y for x, y in zip(r, num2[i])]
                    st.m_boards += nb
                _run_mc(pool, workers, mc.matrix_task, lambda k, td: (seed * 2_000_003 + k, 10 ** 9, td),
                        deadline, chunk_s, on_m, lambda: st.m_boards >= matrix_boards)
                if st.m_boards:
                    st.E, st.Wavg = _matrix_from_counts(hs, st.m_num2, st.m_boards, spec)
                st.stage_done["m"] = st.m_boards > 0
            elif s == "c":
                if st.E is None:
                    st.stage_note["c"] = "skipped: no matrix"
                    continue
                ev, _ = st.eq_random()
                ok = _solve_stage(pool, workers, st, hs, ev, deadline, hu_iters, multi_iters, ante_bb, log)
                st.stage_done["c"] = ok
            elif s == "b":
                if st.b_order is None:
                    if st.push:
                        st.b_order, st.b_order_src = list(st.push), "push"
                    else:
                        ev, _ = st.eq_random()
                        st.b_order = sorted(range(C), key=lambda i: (-ev[i], i)) if ev else list(range(C))
                        st.b_order_src = "eq_random"
                wv = _range_weights(hs, st.b_order, st.grid)
                batches_b = []

                def on_b(res):
                    num, den, nb = res
                    K = len(st.grid)
                    for i in range(C):
                        rn, rd, n2, d2 = st.b_num[i], st.b_den[i], num[i], den[i]
                        for g in range(K):
                            rn[g] += n2[g]
                            rd[g] += d2[g]
                    st.b_boards += nb
                    batches_b.append([num[i][0] / (2 * den[i][0]) if den[i][0] > 0 else None for i in range(C)])
                    checkpoint()
                _run_mc(pool, workers, task_fn, lambda k, td: (seed * 3_000_017 + k, 10 ** 9, td, wv),
                        deadline, chunk_s if engine == "range" else 4 * chunk_s, on_b)
                st.stderr["b_J3pct_batch"] = _batch_stderr(batches_b)
                st.stage_done["b"] = st.b_boards > 0
            st.stage_s[s] = round(time.time() - t0, 2)
            log(f"stage {s}: {st.stage_s[s]} s, done={st.stage_done.get(s)}, "
                f"boards a/m/b = {st.a_boards}/{st.m_boards}/{st.b_boards}, stderr {st.stderr}")
            checkpoint(force=True)
    finally:
        pool.terminate()
        pool.join()
    tables = assemble(st, spec)
    san = sanity_report(st, tables, spec)
    elapsed = time.time() - t_start
    meta.update(stage_s=st.stage_s, stage_done=st.stage_done, stage_note=st.stage_note,
                boards={"a": st.a_boards, "m": st.m_boards, "b": st.b_boards}, stderr=st.stderr,
                fit_info=st.fit_info, elapsed_s=round(elapsed, 2), final=True, loadavg_end=_loadavg())
    digest = write_module(out_path, tables, san, meta)
    log(f"done in {elapsed:.1f} s; sanity ok={san['ok']} failed={san['failed']}; {out_path or '(not written)'}")
    return {"tables": tables, "sanity": san, "meta": meta, "path": out_path, "checksum": digest, "state": st}


def _loadavg():
    try:
        return [round(x, 1) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def _batch_stderr(batches):
    """Max over classes of the batch-means standard error of equity vs random (None if under 2 batches)."""
    if len(batches) < 2:
        return None
    C = len(batches[0])
    worst = 0.0
    for i in range(C):
        xs = [b[i] for b in batches if b[i] is not None]
        if len(xs) < 2:
            return None
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        worst = max(worst, math.sqrt(var / len(xs)))
    return round(worst, 5)


def _range_weights(hs, order, grid):
    cols = [hs.top_range(order, J) for J in grid]
    return [tuple(cols[g][i] for g in range(len(grid))) for i in range(hs.n_classes)]


def _matrix_from_counts(hs, num2, boards, spec):
    """E[i][j] = num2 / (2 x boards x p x pairs[i][j]); W[i][j] = pairs[i][j] / combos[i] (average compatible
    class-j hands per class-i hand). p = P(a random board misses 2n given cards) = C(N-2n, nb) / C(N, nb)."""
    C = hs.n_classes
    n = hs.n
    N = hs.N
    nb = sum(spec.board_schedule)
    p = math.comb(N - 2 * n, nb) / math.comb(N, nb)
    pairs = _compat_pairs(hs)
    E = [[0.5] * C for _ in range(C)]
    W = [[0.0] * C for _ in range(C)]
    for i in range(C):
        ci = hs.combos[i]
        for j in range(C):
            pr = pairs[i][j]
            if pr > 0:
                E[i][j] = num2[i][j] / (2.0 * boards * p * pr)
                W[i][j] = pr / ci
    return E, W


def _compat_pairs(hs):
    """pairs[i][j] = number of (a in i, b in j) with no shared card (exact, board-free)."""
    C = hs.n_classes
    U, UC = hs.universe, hs.univ_class
    from itertools import combinations as _cmb
    n = hs.n
    # villains containing a given subset T of cards, per class (inclusion-exclusion like the engines)
    cnt = {}
    for h, c in zip(U, UC):
        for size in range(1, n + 1):
            for t in _cmb(h, size):
                d = cnt.get(t)
                if d is None:
                    d = cnt[t] = {}
                d[c] = d.get(c, 0) + 1
    pairs = [[0] * C for _ in range(C)]
    for h, i in zip(U, UC):
        row = pairs[i]
        acc = {}
        for j in range(C):
            acc[j] = hs.combos[j]
        for size in range(1, n + 1):
            sign = -1 if size % 2 else 1
            for t in _cmb(h, size):
                for j, v in cnt[t].items():
                    acc[j] += sign * v
        for j, v in acc.items():
            row[j] += v
    return pairs


def _solve_stage(pool, workers, st, hs, ev, deadline, hu_iters, multi_iters, ante_bb, log):
    """Heads-up Nash at 12 stacks and 8-handed Nash at 6 stacks in parallel; adapt iterations to the time left."""
    C = hs.n_classes
    Q = math.comb(hs.N - hs.n, hs.n)
    left = deadline - time.time()
    # measured on this Mac: about 1.8 s for 1,500 heads-up iterations and 22 ms per 8-handed iteration at 169 classes
    scale = (C / 169.0) ** 2
    hu_cost = 1.8 / 1500 * scale
    mu_cost = 0.022 * scale
    free = max(1, workers - len(SV.MULTI_STACKS))
    hu_waves = math.ceil(len(SV.HU_STACKS) / free)
    hi = max(150, min(hu_iters, int(max(0.5, left * 0.85 / hu_waves) / hu_cost)))
    mi = max(60, min(multi_iters, int(max(0.5, left * 0.85) / mu_cost)))
    E, W, cmb, tot = st.E, st.Wavg, hs.combos, hs.total_combos
    # every task stops at the deadline (solve checks the clock each iteration), so no task outlives the stage
    # the 8-handed solves are the long ones: queue them first so they start at once; heads-up fills the rest
    jobs = [("m8", pool.apply_async(SV.multi_task, ((E, W, cmb, tot, Q, 8, S, mi, ante_bb, deadline),)))
            for S in SV.MULTI_STACKS]
    jobs += [("hu", pool.apply_async(SV.hu_task, ((E, W, cmb, tot, Q, S, hi, ante_bb, deadline),)))
             for S in SV.HU_STACKS]
    hu, m8 = [], []
    while time.time() < deadline + 1.0 and not all(ar.ready() for _, ar in jobs):
        time.sleep(0.01)
    for kind, ar in jobs:
        if ar.ready():
            (hu if kind == "hu" else m8).append(ar.get())
    st.fit_info = {"hu_iters": hi, "multi_iters": mi, "hu_done": len(hu), "multi_done": len(m8),
                   "hu_iters_done_min": min((d["iters"] or 0 for d in hu), default=0),
                   "multi_iters_done_min": min((d["iters"] or 0 for d in m8), default=0)}
    # parts are refit only when their solves converged enough; otherwise that part keeps the NLHE prior
    hu = [d for d in hu if (d["iters"] or 0) >= 300]
    m8 = [d for d in m8 if (d["iters"] or 0) >= 60]
    parts = []
    if len(hu) >= len(SV.HU_STACKS) - 2:
        po, co, _, _ = SV.orders(hu, ev)
        st.push, st.call = po, co
        parts.append("orders")
    hu_sb = [d for d in hu if d["S"] >= 4]
    if len(hu_sb) >= 6 and len(m8) >= len(SV.MULTI_STACKS) - 1:
        jf, cf, info = SV.fit(hu, m8)
        st.jam_fit, st.call_fit = jf, cf
        parts.append("jam+call fits")
        st.fit_info.update({k: v for k, v in info.items() if k != "rows"})
    elif len(m8) >= len(SV.MULTI_STACKS) - 1:
        jf, cf, info = SV.fit([{"S": 4, "sb_jam_pct": 1.0}, {"S": 5, "sb_jam_pct": 1.0}], m8)
        st.jam_fit = jf[:3] + NLHE_JAM_FIT[3:]
        st.call_fit = cf
        parts.append("k>=2 jam + call fits (SB part = NLHE prior)")
    st.fit_info["refit_parts"] = parts
    st.fit_info["hu"] = {d["S"]: (round(d["sb_jam_pct"], 4), round(d["bb_call_pct"], 4)) for d in hu}
    st.fit_info["m8_jam"] = {d["S"]: [round(x, 4) for x in d["jam"]] for d in m8}
    log(f"solver: {len(hu)}/12 heads-up, {len(m8)}/6 8-handed converged (iters {hi}/{mi}); refit {parts}; "
        f"fits {st.jam_fit} {st.call_fit}")
    return st.push is not None and st.jam_fit is not None


# ===================================================================== CLI
def _resolve_spec(name):
    from . import spec as S
    if name in S.EXAMPLES:
        return S.EXAMPLES[name]()
    mod, _, fn = name.partition(":")
    if mod.endswith(".py"):
        sp = importlib.util.spec_from_file_location("user_spec", mod)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn or "SPEC")() if callable(getattr(m, fn or "SPEC")) else getattr(m, fn or "SPEC")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", default="nlhe", help="example name or module:function returning a VariantSpec")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", default="amcb")
    ap.add_argument("--order", default=None, help="file with a space-separated best-first label order for stage b")
    args = ap.parse_args(argv)
    spec = _resolve_spec(args.spec)
    order = open(args.order).read().split() if args.order else None
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"tables_{spec.name}.py")
    res = build_tables(spec, workers=args.workers, time_budget=args.budget, out_path=out,
                       stages=tuple(args.stages), range_order=order)
    print(json.dumps({"path": res["path"], "sanity_ok": res["sanity"]["ok"], "failed": res["sanity"]["failed"],
                      "stage_s": res["meta"]["stage_s"], "boards": res["meta"]["boards"]}, indent=1))
    return 0 if res["sanity"]["ok"] else 1


if __name__ == "__main__":
    import importlib
    sys.exit(main())
