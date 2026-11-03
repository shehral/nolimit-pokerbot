"""Runner of the mock starter kit: N tables for a list of entrants, random seating, multiprocessing, seeded and
reproducible, and a PAIRED mode that replays the same tables (seating, buttons, decks, agent seeds) across compared
configurations.

    from sim.kit.runner import run, run_paired
    res = run([("ex", "sim/kit/example_agent.py:ExampleAgent")] * 8, n_tables=2000, seed=1)
    print(res.summary())

Entrants: a list of entries, each one of
  - ("name", spec) or Entrant("name", spec);
  - a spec alone (the name is derived from it).
A spec is a class or factory callable f(seat, config) -> agent (must be picklable for workers > 1: a module-level
class or functools.partial of one, not a lambda), or a string "package.module:Attr" or "path/to/file.py:Attr"
(path relative to the project root or absolute). Duplicate entries are separate entrants with the same name;
RunResult.by_name() pools them.

Seating (one new set of agent instances per table, seats 0..n-1, agents see seat numbers only):
  - "event" (default; the event's format, rules s4): each iteration shuffles all k entrants into ceil(k/n)
    tables; empty seats get random copies of entrants marked as fillers (they play, earn nothing, and are not
    recorded). With k == n every table seats everyone in a random order.
  - "sample": each table draws n distinct entrants uniformly (needs k >= n).
Determinism: table t of a run with seed s uses only (s, t): its seating comes from the seating stream of (s, k), and
its button, deck stream, per-agent seeds and global-random seed come from "kit-table:s:t". So a run is identical for
any worker count, and two configurations with the same seed and the same number of entrants see the same tables
(PAIRED mode). Exceptions to exact reproducibility: decisions near the time limit (wall clock), and bots whose
module-level state carries over between tables in a worker process.
"""
import hashlib
import importlib
import importlib.util
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

try:
    from .engine import SeatStats, Table
    from .payouts import average_seat, get_payouts, payouts_name
    from .schedules import HANDS_PER_LEVEL, schedule_name
    from .stats import ci95, paired, relative
except ImportError:   # flat import (sim/kit on sys.path)
    from engine import SeatStats, Table
    from payouts import average_seat, get_payouts, payouts_name
    from schedules import HANDS_PER_LEVEL, schedule_name
    from stats import ci95, paired, relative

KIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = KIT_DIR.parents[1]


# ---------------------------------------------------------------- entrants
class Entrant:
    __slots__ = ("name", "spec")

    def __init__(self, name, spec):
        self.name = str(name)
        self.spec = spec

    def __repr__(self):
        return f"Entrant({self.name!r}, {self.spec!r})"

    def __reduce__(self):
        return (Entrant, (self.name, self.spec))


def _spec_name(spec):
    if isinstance(spec, str):
        target = spec.rsplit(":", 1)
        if len(target) == 2 and target[1]:
            return target[1]
        return Path(target[0]).stem
    return getattr(spec, "__name__", None) or getattr(getattr(spec, "func", None), "__name__", None) or repr(spec)


def as_entrant(x):
    if isinstance(x, Entrant):
        return x
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
        return Entrant(x[0], x[1])
    return Entrant(_spec_name(x), x)


_LOADED = {}


def _is_kit_base(obj):
    return isinstance(obj, type) and "_kit_base" in vars(obj)


def _default_attr(mod, spec):
    """Attribute to use when a spec names only a module or file: make_agent, AGENT, an `Agent` that is not the
    kit's abstract base, else the single class defined in that module that has an act() method."""
    for cand in ("make_agent", "AGENT"):
        if callable(getattr(mod, cand, None)):
            return cand
    a = getattr(mod, "Agent", None)
    if callable(a) and not _is_kit_base(a):
        return "Agent"
    own = [name for name, obj in vars(mod).items()
           if isinstance(obj, type) and obj.__module__ == mod.__name__ and callable(getattr(obj, "act", None))
           and not _is_kit_base(obj)]
    if len(own) == 1:
        return own[0]
    raise AttributeError(f"{spec!r}: name the agent class explicitly ('file.py:ClassName'); found "
                         f"{own or 'no agent classes'}")


def load_factory(spec):
    """Resolve a spec to a factory callable f(seat, config) -> agent."""
    if not isinstance(spec, str):
        if not callable(spec):
            raise TypeError(f"entrant spec {spec!r} is neither callable nor a string")
        return spec
    if spec in _LOADED:
        return _LOADED[spec]
    target, _, attr = spec.rpartition(":") if ":" in spec else (spec, "", "")
    is_path = target.endswith(".py") or os.sep in target or "/" in target
    if is_path:
        path = Path(target)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path) if (PROJECT_ROOT / path).exists() else path.resolve()
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"bot file not found: {path}")
        modname = "kitbot_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
        mod = sys.modules.get(modname)
        if mod is None:
            d = str(path.parent)
            if d not in sys.path:
                sys.path.append(d)     # sibling imports work; appended so stdlib and the kit win name clashes
            ms = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(ms)
            sys.modules[modname] = mod
            try:
                ms.loader.exec_module(mod)
            except BaseException:
                del sys.modules[modname]
                raise
    else:
        mod = importlib.import_module(target)
    if not attr:
        attr = _default_attr(mod, spec)
    fac = getattr(mod, attr)
    if not callable(fac):
        raise TypeError(f"{spec!r} is not callable")
    _LOADED[spec] = fac
    return fac


# ---------------------------------------------------------------- seating
def seating_plan(k, n_tables, n_seats=8, seed=0, mode="event"):
    """List of n_tables tables; each is a tuple of (entrant_index, is_filler) per seat 0..n_seats-1."""
    if k < 1:
        raise ValueError("need at least one entrant")
    rng = random.Random(f"kit-seating:{seed}:{k}:{n_seats}:{mode}")
    tables = []
    if mode == "sample":
        if k < n_seats:
            raise ValueError(f"seating='sample' needs at least {n_seats} entrants, got {k}")
        while len(tables) < n_tables:
            tables.append(tuple((e, False) for e in rng.sample(range(k), n_seats)))
    elif mode == "event":
        per_iter = -(-k // n_seats)
        n_fill = per_iter * n_seats - k
        while len(tables) < n_tables:
            items = [(e, False) for e in range(k)] + [(rng.randrange(k), True) for _ in range(n_fill)]
            rng.shuffle(items)
            for t in range(per_iter):
                if len(tables) < n_tables:
                    tables.append(tuple(items[t * n_seats:(t + 1) * n_seats]))
    else:
        raise ValueError(f"unknown seating mode {mode!r} (use 'event' or 'sample')")
    return tables


# ---------------------------------------------------------------- workers
_W = {}


def _apply_gc_mode(mode):
    """'default': leave the collector alone (what an unknown sandbox most likely does); 'freeze': gc.freeze() after
    imports so long-lived kit/worker objects are never rescanned; 'disable': gc.disable() (no collector pauses at
    all; cyclic garbage then leaks, fine for a benchmark process)."""
    import gc
    if mode == "default":
        return
    gc.collect()
    if mode == "freeze":
        gc.freeze()
    elif mode == "disable":
        gc.disable()
    else:
        raise ValueError(f"gc_mode must be 'default', 'freeze' or 'disable', not {mode!r}")


def _worker_init(specs, table_kwargs, paths, gc_mode="default"):
    # A failing pool initializer makes multiprocessing respawn workers forever; record the error instead and raise
    # it from the first task, so it reaches the caller.
    _W.pop("init_error", None)
    try:
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)
        _W["factories"] = [load_factory(s) for s in specs]
        _W["kw"] = table_kwargs
        _apply_gc_mode(gc_mode)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        import traceback
        _W["init_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"


def _run_chunk(task):
    if "init_error" in _W:
        raise RuntimeError("kit worker initialization failed: " + _W["init_error"])
    chunk_id, seed, items = task
    factories = _W["factories"]
    kw = _W["kw"]
    agg = [SeatStats() for _ in factories]
    rows = []
    t0 = time.perf_counter()
    for table_id, seating in items:
        res = Table([factories[e] for e, _ in seating], seed=f"{seed}:{table_id}", **kw).play()
        for s, (e, filler) in enumerate(seating):
            if not filler:
                agg[e].merge(res.stats[s])
        rows.append((table_id, tuple(e for e, _ in seating), tuple(f for _, f in seating), tuple(res.points),
                     tuple(res.places), tuple(res.bust_hand), res.hands_played, res.survivors))
    return chunk_id, rows, agg, time.perf_counter() - t0, os.getpid()


# ---------------------------------------------------------------- results
class EntrantResult:
    """Per-entrant output: points per table (table order), places, and the merged SeatStats counters."""

    def __init__(self, index, name, points, places, table_ids, stats, avg_seat):
        self.index = index
        self.name = name
        self.points = points
        self.places = places
        self.table_ids = table_ids
        self.stats = stats
        self.avg_seat = avg_seat

    @property
    def mean(self):
        return sum(self.points) / len(self.points) if self.points else 0.0

    def ci(self):
        return ci95(self.points)

    def rel(self):
        return relative(self.points, self.avg_seat)

    def place_counts(self):
        out = {}
        for p in self.places:
            out[p] = out.get(p, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self, lists=True):
        d = {"index": self.index, "name": self.name, "points_ci95": self.ci(), "relative": self.rel(),
             "place_counts": {str(k): v for k, v in self.place_counts().items()}}
        d.update(self.stats.summary())
        if lists:
            d["points"] = self.points
            d["places"] = self.places
            d["table_ids"] = self.table_ids
        return d


class RunResult:
    def __init__(self, entrants, tables, meta):
        self.entrants = entrants      # list of EntrantResult, in entrant order
        self.tables = tables          # rows: (table_id, entrant_by_seat, filler_by_seat, points, places, bust_hand,
        self.meta = meta              #        hands_played, survivors)

    def by_name(self):
        """Pool entrants that share a name (copies). Pooled intervals treat copies as independent (approximate)."""
        out = {}
        for e in self.entrants:
            if e.name not in out:
                out[e.name] = EntrantResult(e.index, e.name, [], [], [], SeatStats(), e.avg_seat)
            p = out[e.name]
            p.points = p.points + e.points
            p.places = p.places + e.places
            p.table_ids = p.table_ids + e.table_ids
            p.stats.merge(e.stats)
        return out

    def summary(self, pooled=True):
        m = self.meta
        groups = list(self.by_name().values()) if pooled else self.entrants
        lines = [f"{m['n_tables']} tables, seed {m['seed']}, payouts {m['payouts']}, schedule {m['schedule']}, "
                 f"seating {m['seating']}, limit {m['time_limit_ms']} ms x{m['time_scale']:g} ({m['timing']}), "
                 f"workers {m['workers']}, {m['elapsed_s']:.1f} s ({m['tables_per_s']:.0f} tables/s), "
                 f"load {m['loadavg_start'][0]:.1f}->{m['loadavg_end'][0]:.1f}",
                 "t/o = enforced timeouts; cpu = over the limit on CPU time (bot slow); osn = over on wall clock only "
                 "(OS/other-process noise)",
                 f"{'entrant':<22}{'tables':>7}{'points':>8}{'95% CI':>16}{'rel':>6}{'decisions':>11}"
                 f"{'t/o':>6}{'(cpu':>6}{'osn)':>6}{'exc':>6}{'ill':>6}{'p50us':>8}{'p99us':>8}{'p99.9':>8}"
                 f"{'max_us':>9}"]
        for e in groups:
            c = e.ci()
            r = e.rel()
            s = e.stats
            lines.append(f"{e.name[:21]:<22}{len(e.points):>7}{c['mean']:>8.2f}"
                         f"{'[' + format(c['lo'], '.2f') + ', ' + format(c['hi'], '.2f') + ']':>16}"
                         f"{r['rel']:>6.2f}{s.decisions:>11}{s.timeouts:>6}{s.cpu_over:>6}{s.preempted:>6}"
                         f"{s.exceptions:>6}{s.illegal:>6}"
                         f"{s.latency_us(0.5):>8.1f}{s.latency_us(0.99):>8.1f}{s.latency_us(0.999):>8.1f}"
                         f"{s.lat_max_ns / 1000:>9.1f}")
        return "\n".join(lines)

    def to_dict(self, lists=True, tables=False):
        d = {"meta": self.meta, "entrants": [e.to_dict(lists) for e in self.entrants],
             "pooled": {k: v.to_dict(False) for k, v in self.by_name().items()}}
        if tables:
            d["tables"] = [list(r) for r in self.tables]
        return d

    def save_json(self, path, lists=True, tables=False):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(lists, tables), f, indent=1)
        return path


class _SpawnSafeMain:
    """spawn workers re-import the parent's __main__ from its __file__. When that is not a real file (code piped on
    stdin, some notebooks), every worker would crash at start-up and the pool would respawn them forever. Hide such
    a __file__ while the pool starts its workers, then restore it."""

    def __enter__(self):
        self.main = sys.modules.get("__main__")
        self.saved = None
        f = getattr(self.main, "__file__", None)
        spec_name = getattr(getattr(self.main, "__spec__", None), "name", None)
        if f is not None and spec_name is None and not os.path.isfile(f):
            self.saved = f
            del self.main.__file__
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.main.__file__ = self.saved
        return False


def default_workers():
    return max(1, (os.cpu_count() or 4) - 2)


def run(entrants, n_tables, seed=0, payouts="TOP3", schedule="TURBO", *, workers=None, seating="event",
        time_limit_ms=1.0, time_scale=1.0, timing="wall", start_stack=2000, max_hands=200,
        hands_per_level=HANDS_PER_LEVEL, n_seats=8, chunk_size=None, progress=False, extra_config=None,
        gc_mode="default"):
    """Play n_tables tables and return a RunResult. workers=None -> os.cpu_count() - 2; workers=1 runs in-process.
    timing="wall" enforces the limit on wall clock (the rules); "cpu" on thread CPU time (see engine.SeatStats).
    gc_mode sets each worker's garbage collector ('default', 'freeze', 'disable'; see _apply_gc_mode); with
    workers=1 it applies to this process for the duration of the run and is then restored."""
    if gc_mode not in ("default", "freeze", "disable"):
        raise ValueError(f"gc_mode must be 'default', 'freeze' or 'disable', not {gc_mode!r}")
    if timing not in ("wall", "cpu"):
        raise ValueError(f"timing must be 'wall' or 'cpu', not {timing!r}")
    ents = [as_entrant(x) for x in entrants]
    k = len(ents)
    plan = seating_plan(k, n_tables, n_seats, seed, seating)
    table_kwargs = {"payouts": get_payouts(payouts, n_seats), "schedule": schedule, "start_stack": start_stack,
                    "max_hands": max_hands, "hands_per_level": hands_per_level, "time_limit_ms": time_limit_ms,
                    "time_scale": time_scale, "timing": timing, "extra_config": extra_config}
    workers = default_workers() if workers is None else max(1, int(workers))
    workers = min(workers, max(1, n_tables))
    if chunk_size is None:
        chunk_size = max(1, min(100, -(-n_tables // (workers * 8))))
    tasks = []
    for cid, lo in enumerate(range(0, n_tables, chunk_size)):
        hi = min(n_tables, lo + chunk_size)
        tasks.append((cid, seed, [(t, plan[t]) for t in range(lo, hi)]))
    specs = [e.spec for e in ents]
    paths = [str(PROJECT_ROOT)]
    load_start = os.getloadavg()
    t_start = time.perf_counter()
    results = []
    done = 0
    next_report = max(1, n_tables // 10)
    if workers == 1:
        import gc
        was_enabled = gc.isenabled()
        _worker_init(specs, table_kwargs, paths, gc_mode)
        if "init_error" in _W:
            raise RuntimeError("kit initialization failed: " + _W.pop("init_error"))
        try:
            for task in tasks:
                results.append(_run_chunk(task))
                done += len(task[2])
                if progress and done >= next_report:
                    print(f"  {done}/{n_tables} tables, {time.perf_counter() - t_start:.1f} s", file=sys.stderr)
                    next_report += max(1, n_tables // 10)
        finally:
            if gc_mode == "freeze":
                gc.unfreeze()
            if was_enabled:
                gc.enable()
    else:
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)
        for e in ents:
            load_factory(e.spec)          # surface a bad path / attribute here, not as a hung pool
            try:
                pickle.dumps(e.spec)
            except Exception as ex:
                raise TypeError(f"entrant {e.name!r}: spec {e.spec!r} is not picklable ({ex}); use a module-level "
                                f"class, functools.partial, a 'module:Attr' or 'file.py:Attr' string, or workers=1")
        ctx = mp.get_context("spawn")
        with _SpawnSafeMain():
            pool = ctx.Pool(workers, initializer=_worker_init, initargs=(specs, table_kwargs, paths, gc_mode))
        with pool:
            for r in pool.imap_unordered(_run_chunk, tasks):
                results.append(r)
                done += len(r[1])
                if progress and done >= next_report:
                    print(f"  {done}/{n_tables} tables, {time.perf_counter() - t_start:.1f} s", file=sys.stderr)
                    next_report += max(1, n_tables // 10)
    elapsed = time.perf_counter() - t_start
    rows = []
    stats = [SeatStats() for _ in ents]
    busy = 0.0
    pids = set()
    for _cid, rws, agg, dt, pid in results:
        rows.extend(rws)
        for e in range(k):
            stats[e].merge(agg[e])
        busy += dt
        pids.add(pid)
    rows.sort(key=lambda r: r[0])
    pays = table_kwargs["payouts"]
    avg = average_seat(pays, n_seats)
    per_points = [[] for _ in ents]
    per_places = [[] for _ in ents]
    per_ids = [[] for _ in ents]
    hands_total = 0
    capped = 0
    for (tid, seat_ents, fillers, pts, places, _bust, hands, surv) in rows:
        hands_total += hands
        if surv > 1:
            capped += 1
        for s, e in enumerate(seat_ents):
            if not fillers[s]:
                per_points[e].append(pts[s])
                per_places[e].append(places[s])
                per_ids[e].append(tid)
    meta = {"n_tables": n_tables, "seed": seed, "payouts": payouts_name(payouts), "payouts_vector": list(pays),
            "avg_seat_points": avg, "schedule": schedule_name(schedule), "seating": seating, "n_seats": n_seats,
            "start_stack": start_stack, "max_hands": max_hands, "hands_per_level": hands_per_level,
            "time_limit_ms": time_limit_ms, "time_scale": time_scale, "timing": timing, "gc_mode": gc_mode,
            "workers": workers,
            "processes": len(pids), "loadavg_start": list(load_start), "loadavg_end": list(os.getloadavg()),
            "chunk_size": chunk_size, "elapsed_s": elapsed, "tables_per_s": n_tables / elapsed if elapsed else 0.0,
            "worker_busy_s": busy, "mean_hands_per_table": hands_total / n_tables if n_tables else 0.0,
            "capped_share": capped / n_tables if n_tables else 0.0, "entrant_names": [e.name for e in ents],
            "python": sys.version.split()[0]}
    out = [EntrantResult(i, ents[i].name, per_points[i], per_places[i], per_ids[i], stats[i], avg) for i in range(k)]
    return RunResult(out, rows, meta)


class PairedResult:
    def __init__(self, results, comparisons, focus, baseline):
        self.results = results            # name -> RunResult
        self.comparisons = comparisons    # name -> paired stats of focus points (config - baseline), per table
        self.focus = focus
        self.baseline = baseline

    def summary(self):
        lines = [f"PAIRED: focus entrant #{self.focus}, baseline config {self.baseline!r}"]
        base = self.results[self.baseline].entrants[self.focus]
        b = base.ci()
        lines.append(f"  {self.baseline:<20} {base.name:<18} points {b['mean']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}] "
                     f"rel {base.rel()['rel']:.3f}")
        for name, c in self.comparisons.items():
            e = self.results[name].entrants[self.focus]
            ec = e.ci()
            lines.append(f"  {name:<20} {e.name:<18} points {ec['mean']:.2f} [{ec['lo']:.2f}, {ec['hi']:.2f}] "
                         f"rel {e.rel()['rel']:.3f} | diff {c['mean_diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
                         f"(se {c['se']:.3f} vs unpaired {c['unpaired_se']:.3f}, corr {c['corr']:.2f})")
        return "\n".join(lines)

    def to_dict(self, lists=False):
        return {"focus": self.focus, "baseline": self.baseline, "comparisons": self.comparisons,
                "results": {k: v.to_dict(lists) for k, v in self.results.items()}}


def run_paired(configs, n_tables, seed=0, focus=0, baseline=None, **kw):
    """Run every configuration on the SAME tables (same seed: seating, buttons, decks, agent seeds) and compare the
    focus entrant's per-table points against the baseline configuration (default: the first).

    configs: dict name -> entrant list; every list must have the same length (so the seating plans coincide), and
    entrant `focus` is the one being compared (e.g. index 0 = our bot, v0 in one config and v1 in another)."""
    names = list(configs)
    if not names:
        raise ValueError("no configurations")
    ks = {len(configs[n]) for n in names}
    if len(ks) != 1:
        raise ValueError(f"PAIRED needs equally long entrant lists, got lengths {sorted(ks)}")
    baseline = names[0] if baseline is None else baseline
    results = {n: run(configs[n], n_tables, seed=seed, **kw) for n in names}
    base = results[baseline].entrants[focus]
    comps = {}
    for n in names:
        if n == baseline:
            continue
        e = results[n].entrants[focus]
        if e.table_ids != base.table_ids:
            raise RuntimeError("paired runs did not seat the focus entrant at the same tables")
        comps[n] = paired(e.points, base.points)
    return PairedResult(results, comps, focus, baseline)


# ---------------------------------------------------------------- CLI
def _parse_entrant(arg):
    name, _, spec = arg.partition("=") if "=" in arg.split(":", 1)[0] else ("", "", arg)
    copies = 1
    if "*" in spec:
        spec, _, c = spec.rpartition("*")
        copies = int(c)
    e = Entrant(name or _spec_name(spec), spec)
    return [e] * copies


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run N mock-kit tables for a list of entrants.")
    ap.add_argument("--entrant", "-e", action="append", required=True,
                    help="NAME=SPEC or SPEC, optional *COPIES suffix (quote it), e.g. 'ex=sim/kit/example_agent.py:"
                         "ExampleAgent*8'")
    ap.add_argument("--tables", "-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--payouts", default="TOP3")
    ap.add_argument("--schedule", default="TURBO")
    ap.add_argument("--seating", default="event", choices=["event", "sample"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--time-limit-ms", type=float, default=1.0)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--timing", default="wall", choices=["wall", "cpu"])
    ap.add_argument("--gc-mode", default="default", choices=["default", "freeze", "disable"])
    ap.add_argument("--json", default=None, help="write the full result as JSON here")
    ap.add_argument("--per-entrant", action="store_true", help="print copies separately instead of pooled")
    a = ap.parse_args(argv)
    ents = []
    for arg in a.entrant:
        ents.extend(_parse_entrant(arg))
    res = run(ents, a.tables, seed=a.seed, payouts=a.payouts, schedule=a.schedule, workers=a.workers,
              seating=a.seating, time_limit_ms=a.time_limit_ms, time_scale=a.time_scale, timing=a.timing,
              gc_mode=a.gc_mode, progress=True)
    print(res.summary(pooled=not a.per_entrant))
    if a.json:
        res.save_json(a.json)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sim.kit import runner as _runner   # run the package copy so worker functions pickle by package name
    sys.exit(_runner.main())
