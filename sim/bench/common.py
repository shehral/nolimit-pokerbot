"""Bench driver: paired benchmark cells over the mock kit's engine.

A CELL is one benchmark condition:
    field      a zoo mix ("T1_like", "T6_like", "STRONG": each of the 7 opponent seats is drawn i.i.d. per table
               from the mix's seat shares), a zoo archetype key (7 copies), or any occupant name (7 copies, e.g.
               "v2" for the B0 sanity check)
    payouts    "TOP3" | "LINEAR" | "WTA" | "TOP4" (sim/kit/payouts.py), schedule "TURBO" | "FAST" | "SLOW"
    occupants  who sits in OUR seat, e.g. ["v0", "v1", "v2", "chart_tag"]
    n_tables, seed

PAIRING. Table t of a cell has one plan (our seat, the 7 opponents) and one kit table seed (button, deck stream,
agent seeds). Every occupant plays exactly that table, in the same worker task, so per-table differences between
occupants are paired (same cards dealt to the same seats, same opponents, same opponent random streams).

CLOCK. The kit's 1 ms enforcement is OFF in score runs (time_limit_ms=None). Opponents that sit near the limit
instead time out at their measured rates through a seeded draw per decision (SANDBOX_P below for zoo bots, the AI
field's own table for AI bots), and a timed-out decision gets the rules' auto action (check if legal, else fold),
exactly what the engine does after a real timeout. So score runs reproduce exactly on any machine load and paired
runs are exact. Our own act() is still timed on every decision (wall and CPU histograms) and B4 measures it with
enforcement on.

Our bot runs in production mode (the shell masks errors with its v0 fallback or check/fold); the shell's own error
counters are read back after every table, so masked errors are still counted. B4 runs STRICT (nothing masked).
"""
import functools
import math
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import SeatStats, Table          # noqa: E402
from sim.kit.payouts import average_seat, get_payouts, payouts_name  # noqa: E402
from sim.kit.schedules import schedule_name          # noqa: E402
from sim.kit.stats import Z95                        # noqa: E402
from sim.zoo import fields as F                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
N_SEATS = 8

# Measured timeout shares of near-limit zoo bots, by archetype key. None of the published archetypes is near the
# limit, so the table is empty; a slow archetype added later gets its measured share here.
SANDBOX_P = {}


# ----------------------------------------------------------------------------- seat occupants
class Sandboxed:
    """A zoo bot behind a seeded timeout model. Each decision times out with probability p (a random stream separate
    from the bot's own); a timed-out decision returns the rules' auto action without running the bot."""

    def __init__(self, seat, config, key, p, params=None):
        self.inner = F.factory(key, **(params or {}))(seat, config)
        self.p = float(p)
        self.rng = random.Random(f"sandbox:{config.get('seed', seat)}")
        self.n_decisions = 0
        self.n_timeouts = 0
        obs = getattr(self.inner, "observe", None)
        if callable(obs) and not getattr(obs, "kit_default", False):
            self.observe = obs            # instance attribute: the engine registers it only when the bot has one

    def act(self, state):
        self.n_decisions += 1
        if self.p > 0.0 and self.rng.random() < self.p:
            self.n_timeouts += 1
            return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
        return self.inner.act(state)


def parse_occupant(name):
    """'v2' -> ('ours', 2, {}); 'v2:CALL_LAM=2.0,PF_DEPTH_BB=12' -> ('ours', 2, {...}); 'chart_tag' -> ('zoo', key, {});
    'chart_tag:open_bb=2.5' -> ('zoo', key, {...}); 'example' -> ('example', None, {})."""
    import ast
    head, _, tail = name.partition(":")
    params = {}
    if tail:
        for item in tail.split(","):
            k, _, v = item.partition("=")
            params[k.strip()] = ast.literal_eval(v.strip())
    if head[:1] == "v" and head[1:].isdigit():
        return "ours", int(head[1:]), params
    if head == "example":
        return "example", None, params
    if head.startswith("ai_"):            # AI-written field (sim/field_ai); imported only when an ai_ name is used
        from sim.field_ai import fields as AF
        if head in AF.SLUGS:
            return "ai", head, params
    if head in F.ARCHETYPES:
        return "zoo", head, params
    raise ValueError(f"unknown occupant {name!r}")


def factory_for(name, sandbox=True, sandbox_p=None):
    """A picklable factory (seat, config) -> agent for an occupant/opponent name."""
    kind, what, params = parse_occupant(name)
    if kind == "ours":
        from bot import shell as SH       # package import, so partials pickle by name
        return functools.partial(SH.make, version=what, params=params or None)
    if kind == "example":
        from sim.kit.example_agent import ExampleAgent
        return ExampleAgent
    p = (sandbox_p or SANDBOX_P).get(what, 0.0) if sandbox else 0.0
    if kind == "ai":
        from sim.field_ai import fields as AF
        return AF.factory(what, p=p)
    if p > 0.0:
        return functools.partial(Sandboxed, key=what, p=p, params=params or None)
    return F.factory(what, **params)


def is_ours(name):
    return parse_occupant(name)[0] == "ours"


def archetype_of(name):
    kind, what, _ = parse_occupant(name)
    return what if kind in ("zoo", "ai") else name.partition(":")[0]


class _Grab:
    """Factory wrapper that keeps the agents it made (to read shell / sandbox counters after the table)."""
    __slots__ = ("fac", "made")

    def __init__(self, fac):
        self.fac = fac
        self.made = []

    def __call__(self, seat, config):
        a = self.fac(seat, config)
        self.made.append(a)
        return a


# ----------------------------------------------------------------------------- plan
def plan_table(field, seed, tid):
    """(our seat, [7 opponent names]) for table tid of a cell; depends only on (field, seed, tid)."""
    rng = random.Random(f"bench-plan:{seed}:{field}:{tid}")
    our = rng.randrange(N_SEATS)
    if field in F.MIXES:
        opp = F.sample_opponents(F.MIXES[field], rng, N_SEATS - 1)
    elif field.startswith("AI_"):          # AI-written fields (sim/field_ai/fields.py)
        from sim.field_ai import fields as AF
        opp = F.sample_opponents(AF.MIXES[field], rng, N_SEATS - 1)
    else:
        opp = [field] * (N_SEATS - 1)
    return our, opp


def table_seed(seed, tid):
    return f"bench:{seed}:{tid}"


# ----------------------------------------------------------------------------- worker
_FAC = {}


def _fac(name, sandbox_p=None):
    key = (name, None if sandbox_p is None else tuple(sorted(sandbox_p.items())))
    f = _FAC.get(key)
    if f is None:
        f = _FAC[key] = factory_for(name, sandbox_p=sandbox_p)
    return f


def _init_worker(root, env):
    if root not in sys.path:
        sys.path.insert(0, root)
    for k, v in (env or {}).items():
        os.environ[k] = v


def _new_agg():
    return [0, 0.0, 0.0, 0.0, 0.0, 0.0]     # tables, sum S, sum N, sum S^2, sum S*N, sum N^2


def run_chunk(task):
    """One chunk of tables of one cell; every occupant plays every table. Returns plain data."""
    cell, tids = task
    occs = cell["occupants"]
    context = cell.get("context") or occs[-1]
    kw = dict(payouts=cell["payouts"], schedule=cell["schedule"], start_stack=cell.get("start_stack", 2000),
              max_hands=cell.get("max_hands", 200), time_limit_ms=cell.get("time_limit_ms"),
              timing=cell.get("timing", "wall"), time_scale=cell.get("time_scale", 1.0))
    out = {o: {"rows": [], "stats": SeatStats(), "shell": [0, 0, 0, 0, 0, 0], "opp": {}} for o in occs}
    sandbox = {}                     # archetype -> [decisions, timeouts] (opponent seats, context occupant only)
    sp = cell.get("sandbox_p")       # None = SANDBOX_P (every pre-existing cell)
    t0 = time.perf_counter()
    for tid in tids:
        our, opp = plan_table(cell["field"], cell["seed"], tid)
        for o in occs:
            facs = []
            grabs = []
            k = 0
            for s in range(N_SEATS):
                if s == our:
                    g = _Grab(_fac(o, sp))
                    facs.append(g)
                    grabs.append((s, o, g))
                else:
                    g = _Grab(_fac(opp[k], sp))
                    facs.append(g)
                    grabs.append((s, opp[k], g))
                    k += 1
            res = Table(facs, seed=table_seed(cell["seed"], tid), **kw).play()
            rec = out[o]
            rec["rows"].append((tid, res.points[our], res.places[our], res.hands_played, res.survivors))
            rec["stats"].merge(res.stats[our])
            for s, name, g in grabs:
                for a in g.made:
                    if s == our and hasattr(a, "fallbacks"):
                        sh = rec["shell"]
                        sh[0] += 1
                        sh[1] += getattr(a, "errors", 0)
                        sh[2] += getattr(a, "fallbacks", 0)
                        sh[3] += getattr(a, "illegal_fixed", 0)
                        br = getattr(a, "_brain", None)
                        if br is not None:
                            sh[4] += getattr(br, "guard_trips", 0)
                            sh[5] += getattr(br, "bf_computed", 0)
                    elif s != our and o == context and (isinstance(a, Sandboxed) or getattr(a, "is_ai_seat", False)):
                        c = sandbox.setdefault(archetype_of(name), [0, 0])
                        c[0] += a.n_decisions
                        c[1] += a.n_timeouts
            if o == context:
                per = {}
                for s in range(N_SEATS):
                    if s == our:
                        continue
                    name = grabs[s][1]
                    a = archetype_of(name)
                    x = per.setdefault(a, [0.0, 0])
                    x[0] += res.points[s]
                    x[1] += 1
                for a, (S, N) in per.items():
                    g = rec["opp"].setdefault(a, _new_agg())
                    g[0] += 1
                    g[1] += S
                    g[2] += N
                    g[3] += S * S
                    g[4] += S * N
                    g[5] += N * N
    return {"occ": out, "sandbox": sandbox, "secs": time.perf_counter() - t0, "pid": os.getpid()}


# ----------------------------------------------------------------------------- statistics
def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "half": float("nan")}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = Z95 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h, "half": h}


def rel_ci(points, avg=12.5):
    c = mean_ci(points)
    return {"rel": c["mean"] / avg, "lo": c["lo"] / avg, "hi": c["hi"] / avg, "half": c["half"] / avg, "n": c["n"],
            "points": c["mean"], "sd_points": c["sd"]}


def paired_ci(xs, ys, avg=12.5):
    """Paired difference xs - ys (per table), in points and relative units, with the pairing gain."""
    d = [a - b for a, b in zip(xs, ys)]
    c = mean_ci(d)
    cx, cy = mean_ci(xs), mean_ci(ys)
    n = len(d)
    if n > 1 and cx["sd"] > 0 and cy["sd"] > 0:
        cov = sum((a - cx["mean"]) * (b - cy["mean"]) for a, b in zip(xs, ys)) / (n - 1)
        corr = cov / (cx["sd"] * cy["sd"])
    else:
        corr = 0.0
    unpaired_half = Z95 * math.sqrt(cx["sd"] ** 2 / n + cy["sd"] ** 2 / n) if n > 1 else float("nan")
    return {"n": n, "diff_points": c["mean"], "lo_points": c["lo"], "hi_points": c["hi"], "half_points": c["half"],
            "diff_rel": c["mean"] / avg, "lo_rel": c["lo"] / avg, "hi_rel": c["hi"] / avg, "half_rel": c["half"] / avg,
            "corr": corr, "unpaired_half_rel": unpaired_half / avg,
            "identical_tables": sum(1 for x in d if x == 0.0), "significant": (c["lo"] > 0) or (c["hi"] < 0)}


def ratio_ci(agg, avg=12.5):
    """Cluster-robust (by table) mean points per seat of an archetype, from _new_agg sums."""
    n, S, N, S2, SN, N2 = agg
    if N <= 0:
        return None
    m = S / N
    var = max(0.0, S2 - 2 * m * SN + m * m * N2)
    se = math.sqrt(var * n / max(1, n - 1)) / N
    return {"seats": N, "tables": n, "points": m, "rel": m / avg, "lo": (m - Z95 * se) / avg,
            "hi": (m + Z95 * se) / avg, "half": Z95 * se / avg}


def over_count(hist, limit_us=1000.0):
    """Decisions whose time exceeded limit_us, from a kit 0.1-us histogram (bin b covers [b, b+1) x 100 ns)."""
    b0 = int(limit_us * 10)
    return sum(c for b, c in hist.items() if b >= b0)


def stats_digest(st):
    s = st.summary()
    return {"tables": s["tables"], "decisions": s["decisions"], "timeouts_enforced": s["timeouts"],
            "exceptions": s["exceptions"], "illegal": s["illegal"],
            "wall_us": s["latency_us"], "cpu_us": s["cpu_us"],
            "over_1ms_wall": over_count(st.lat_hist), "over_1ms_cpu": over_count(st.cpu_hist),
            "over_250us_cpu": over_count(st.cpu_hist, 250.0), "over_limit_counters": s["over_limit"],
            "observe": s["observe"], "init": s["init"], "actions": s["actions"], "vpip": s["vpip"], "pfr": s["pfr"],
            "examples": s["examples"]}


# ----------------------------------------------------------------------------- run a cell
def default_workers():
    return max(1, (os.cpu_count() or 4) - 2)


def run_cell(field, occupants, n_tables, seed=1, payouts="TOP3", schedule="TURBO", workers=None, context=None,
             time_limit_ms=None, timing="wall", env=None, chunk=None, label=None, verbose=True, pool=None,
             keep_hist=False, sandbox_p=None):
    """Play a cell and return a JSON-ready dict (see module docstring)."""
    occupants = list(occupants)
    cell = {"field": field, "occupants": occupants, "payouts": payouts, "schedule": schedule, "seed": seed,
            "context": context or occupants[-1], "time_limit_ms": time_limit_ms, "timing": timing}
    if sandbox_p is not None:
        cell["sandbox_p"] = dict(sandbox_p)
    workers = default_workers() if workers is None else max(1, int(workers))
    if pool is not None:
        workers = getattr(pool, "_processes", workers) or workers     # record the pool's real size
    if chunk is None:
        chunk = max(1, min(40, n_tables // (workers * 6) or 1))
    tasks = [(cell, list(range(lo, min(n_tables, lo + chunk)))) for lo in range(0, n_tables, chunk)]
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    results = []
    own_pool = None
    if workers == 1 and pool is None:
        _init_worker(ROOT, env)
        results = [run_chunk(t) for t in tasks]
    else:
        if pool is None:
            ctx = mp.get_context("spawn")
            own_pool = pool = ctx.Pool(workers, initializer=_init_worker, initargs=(ROOT, env))
        try:
            for r in pool.imap_unordered(run_chunk, tasks):
                results.append(r)
        finally:
            if own_pool is not None:
                own_pool.close()
                own_pool.join()
    elapsed = time.perf_counter() - t0
    avg = average_seat(get_payouts(payouts, N_SEATS), N_SEATS)
    occ_out = {}
    sandbox = {}
    hist = {}
    busy = sum(r["secs"] for r in results)
    for o in occupants:
        rows = []
        st = SeatStats()
        shell = [0, 0, 0, 0, 0, 0]
        opp = {}
        for r in results:
            x = r["occ"][o]
            rows.extend(x["rows"])
            st.merge(x["stats"])
            for i in range(6):
                shell[i] += x["shell"][i]
            for a, g in x["opp"].items():
                acc = opp.setdefault(a, _new_agg())
                for i in range(6):
                    acc[i] += g[i]
        rows.sort()
        if keep_hist and is_ours(o):
            hist[o] = {"wall": {str(b): c for b, c in sorted(st.lat_hist.items())},
                       "cpu": {str(b): c for b, c in sorted(st.cpu_hist.items())}}
        pts = [r[1] for r in rows]
        places = {}
        for r in rows:
            places[str(r[2])] = places.get(str(r[2]), 0) + 1
        occ_out[o] = {"rel": rel_ci(pts, avg), "points": [round(p, 4) for p in pts],
                      "places": dict(sorted(places.items())), "stats": stats_digest(st),
                      "shell": {"instances": shell[0], "errors_masked": shell[1], "fallbacks_v0": shell[2],
                                "illegal_fixed": shell[3], "guard_trips": shell[4], "bf_exact": shell[5]}
                      if is_ours(o) else None,
                      "mean_hands": sum(r[3] for r in rows) / len(rows) if rows else 0.0,
                      "capped_share": sum(1 for r in rows if r[4] > 1) / len(rows) if rows else 0.0,
                      "opponents": {a: ratio_ci(g, avg) for a, g in sorted(opp.items())} if opp else None}
    for r in results:
        for a, (d, t) in r["sandbox"].items():
            c = sandbox.setdefault(a, [0, 0])
            c[0] += d
            c[1] += t
    pairs = {}
    names = occupants
    for i, a in enumerate(names):
        for b in names[:i]:
            pairs[f"{a} - {b}"] = paired_ci(occ_out[a]["points"], occ_out[b]["points"], avg)
    out = {"label": label or f"{field}/{payouts_name(payouts)}/{schedule_name(schedule)}", "field": field,
           "payouts": payouts_name(payouts), "schedule": schedule_name(schedule), "n_tables": n_tables, "seed": seed,
           "occupants": occupants, "context": cell["context"], "avg_seat": avg,
           "clock": {"time_limit_ms": time_limit_ms, "timing": timing,
                     "sandbox_p": SANDBOX_P if sandbox_p is None else dict(sandbox_p)},
           "workers": workers, "elapsed_s": elapsed, "table_plays_per_s": n_tables * len(occupants) / elapsed,
           "worker_busy_s": busy, "loadavg_start": list(load0), "loadavg_end": list(os.getloadavg()),
           "sandbox_timeouts": {a: {"decisions": d, "timeouts": t, "share": t / d if d else 0.0}
                                for a, (d, t) in sorted(sandbox.items())},
           "occ": occ_out, "pairs": pairs}
    if hist:
        out["hist_by_occupant"] = hist
        out["hist"] = hist[next(iter(hist))]      # 0.1-us bins of the first of our occupants (for charts)
    if verbose:
        print(brief(out), flush=True)
    return out


def brief(cell):
    lines = [f"[{cell['label']}] {cell['n_tables']} tables x {len(cell['occupants'])} occupants, "
             f"{cell['elapsed_s']:.0f} s ({cell['table_plays_per_s']:.0f} table-plays/s), "
             f"load {cell['loadavg_start'][0]:.1f}->{cell['loadavg_end'][0]:.1f}"]
    for o in cell["occupants"]:
        r = cell["occ"][o]["rel"]
        st = cell["occ"][o]["stats"]
        sh = cell["occ"][o]["shell"]
        extra = f" masked {sh['errors_masked']} guard {sh['guard_trips']}/{sh['bf_exact']}" if sh else ""
        lines.append(f"   {o:<24} rel {r['rel']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] +-{r['half']:.3f} "
                     f"(sd {r['sd_points']:.1f} pts)  dec {st['decisions']} t/o {st['timeouts_enforced']} exc {st['exceptions']} "
                     f"ill {st['illegal']}"
                     f" cpu>1ms {st['over_1ms_cpu']}{extra}")
    for k, p in cell["pairs"].items():
        lines.append(f"   {k:<34} {p['diff_rel']:+.3f} [{p['lo_rel']:+.3f}, {p['hi_rel']:+.3f}] rel "
                     f"(corr {p['corr']:.2f}, unpaired +-{p['unpaired_half_rel']:.3f}, same {p['identical_tables']})")
    return "\n".join(lines)


def save(obj, name):
    import json
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=None, separators=(",", ":"), default=str)
    os.replace(tmp, path)
    return path


def load(name):
    import json
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def make_pool(workers=None, env=None):
    workers = default_workers() if workers is None else workers
    return mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(ROOT, env))
