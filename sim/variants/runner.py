"""Run many tables of a variant, optionally in a process pool (reuses the kit's entrant specs and seating plan).

    from sim.variants.runner import run_variant
    res = run_variant("toss", [("ex", "sim.variants.agents:TossExample")] * 8, n_tables=200, workers=4)
    res.tables_per_s, res.points["ex"], res.stats["ex"].summary()

CLI:  .venv/bin/python -m sim.variants.runner toss -n 200 --workers 4
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

from sim.kit.engine import SeatStats
from sim.kit.runner import as_entrant, load_factory, seating_plan

from .engine import VTable
from .spec import NAMES, make_variant

_W = {}


def _init(variant_name, variant_opts, specs, kw):
    _W["variant"] = make_variant(variant_name, **variant_opts)
    _W["facs"] = [load_factory(s) for s in specs]
    _W["kw"] = kw


def _chunk(task):
    seed, items = task
    facs, kw, v = _W["facs"], _W["kw"], _W["variant"]
    rows = []
    agg = [SeatStats() for _ in facs]
    for tid, seating in items:
        res = VTable(v, [facs[e] for e, _ in seating], seed=f"{seed}:{tid}", **kw).play()
        for s, (e, filler) in enumerate(seating):
            if not filler:
                agg[e].merge(res.stats[s])
        rows.append((tid, tuple(e for e, _ in seating), tuple(f for _, f in seating), tuple(res.points),
                     res.hands_played))
    return rows, agg


class VariantRun:
    def __init__(self, names, points, stats, tables, elapsed, hands):
        self.names, self.points, self.stats = names, points, stats
        self.tables, self.elapsed, self.hands = tables, elapsed, hands

    @property
    def tables_per_s(self):
        return self.tables / self.elapsed if self.elapsed else float("inf")

    def mean_points(self, name):
        p = self.points[name]
        return sum(p) / len(p) if p else float("nan")


def run_variant(variant, entrants, n_tables, seed=0, payouts="TOP3", schedule="TURBO", *, workers=1,
                seating="event", n_seats=8, variant_options=None, time_limit_ms=1.0, timing="wall", **table_kw):
    """variant: name or Variant (workers > 1 needs a name). entrants as in sim.kit.runner. Returns VariantRun."""
    vname = variant if isinstance(variant, str) else variant.name
    vopt = dict(variant_options or ({} if isinstance(variant, str) else variant.options))
    ents = [as_entrant(x) for x in entrants]
    plan = seating_plan(len(ents), n_tables, n_seats, seed, seating)
    kw = dict(payouts=payouts, schedule=schedule, time_limit_ms=time_limit_ms, timing=timing, **table_kw)
    specs = [e.spec for e in ents]
    workers = max(1, min(int(workers), n_tables))
    csz = max(1, min(50, -(-n_tables // (workers * 4))))
    tasks = [(seed, [(t, plan[t]) for t in range(lo, min(n_tables, lo + csz))]) for lo in range(0, n_tables, csz)]
    t0 = time.perf_counter()
    if workers == 1:
        _init(vname, vopt, specs, kw)
        out = [_chunk(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(workers, _init, (vname, vopt, specs, kw)) as pool:
            out = pool.map(_chunk, tasks)
    elapsed = time.perf_counter() - t0
    names = [e.name for e in ents]
    points = {nm: [] for nm in names}
    stats = {nm: SeatStats() for nm in names}
    hands = 0
    for rows, agg in out:
        for tid, ent_idx, fillers, pts, hp in rows:
            hands += hp
            for e, f, p in zip(ent_idx, fillers, pts):
                if not f:
                    points[names[e]].append(p)
        for e, a in enumerate(agg):
            stats[names[e]].merge(a)
    return VariantRun(names, points, stats, n_tables, elapsed, hands)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=NAMES)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    ex = f"sim.variants.agents:{ {'toss':'TossExample','shortdeck':'ShortDeckExample','bounty':'BountyExample','doubleboard':'DoubleBoardExample'}.get(a.variant,'AnteExample') if a.variant!='plain' else 'VariantExample'}"
    r = run_variant(a.variant, [("ex", ex)] * 8, a.n, seed=a.seed, workers=a.workers)
    s = r.stats["ex"]
    print(f"{a.variant}: {r.tables_per_s:.0f} tables/s, {r.hands / a.n:.1f} hands/table, errors "
          f"{s.timeouts + s.exceptions + s.illegal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
