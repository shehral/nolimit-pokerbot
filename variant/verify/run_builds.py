"""Build tables for every rehearsal variant (8 workers, 90 s) and record classes, stage seconds, sanity and
checkpoint behaviour. Output: variant/verify/out/tables_<v>.py, out/build_<v>.json, out/builds.json.

    .venv/bin/python -m variant.verify.run_builds [names...] [--budget 90] [--workers 8]
"""
import argparse
import json
import os
import re
import threading
import time

from variant.build_tables import build_tables, load_tables
from variant.handspace import HandSpace
from variant.verify import specs as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
_META = re.compile(r"^META = (.*)$", re.M)


class Watcher(threading.Thread):
    """Polls the output module; on every new write records time since start, META.final, stage_done, and whether the
    file imports and its SANITY.ok (an import every write is how the bot would pick it up mid-build)."""

    def __init__(self, path, t0, period=0.2):
        super().__init__(daemon=True)
        self.path, self.t0, self.period = path, t0, period
        self.events, self.stop_flag = [], False

    def run(self):
        last = None
        while not self.stop_flag:
            try:
                st = os.stat(self.path)
                key = (st.st_mtime_ns, st.st_size)
                if key != last:
                    last = key
                    ev = {"t": round(time.time() - self.t0, 2), "bytes": st.st_size}
                    try:
                        m = load_tables(self.path)
                        ev.update(imports=True, final=m.META.get("final"), stage_done=m.META.get("stage_done"),
                                  sanity_ok=m.SANITY.get("ok"), failed=m.SANITY.get("failed"),
                                  n_labels=len(m.EQ_LABELS.split()))
                    except Exception as e:  # noqa: BLE001
                        ev.update(imports=False, error=f"{type(e).__name__}: {e}"[:200])
                    self.events.append(ev)
            except FileNotFoundError:
                pass
            time.sleep(self.period)


def one(name, budget, workers):
    spec, probe = S.get(name)
    path = os.path.join(OUT, f"tables_{name}.py")
    if os.path.exists(path):
        os.remove(path)
    t_hs = time.time()
    hs = HandSpace(spec)
    hs_s = time.time() - t_hs
    logs = []
    t0 = time.time()
    w = Watcher(path, t0)
    w.start()
    res = build_tables(spec, workers=workers, time_budget=budget, out_path=path, hs=hs,
                       log=lambda m: logs.append(f"{time.strftime('%H:%M:%S')} {m}"))
    wall = time.time() - t0
    time.sleep(0.5)
    w.stop_flag = True
    w.join()
    meta, san, tab = res["meta"], res["sanity"], res["tables"]
    ev = w.events
    gaps = [b["t"] - a["t"] for a, b in zip(ev, ev[1:])]
    out = {
        "variant": name, "spec_summary": spec.summary(), "probe": probe, "handspace_s": round(hs_s, 2),
        "classes": hs.n_classes, "total_combos": hs.total_combos, "describe": hs.describe(),
        "engine": meta.get("engine"), "budget_s": budget, "workers": workers, "wall_s": round(wall, 2),
        "stage_s": meta.get("stage_s"), "stage_done": meta.get("stage_done"), "stage_note": meta.get("stage_note"),
        "boards": meta.get("boards"), "stderr": meta.get("stderr"), "fit_info": meta.get("fit_info"),
        "sources": tab.get("SOURCES"), "jam_fit": tab.get("JAM_FIT"), "call_fit": tab.get("CALL_FIT"),
        "loadavg": [meta.get("loadavg_start"), meta.get("loadavg_end")],
        "sanity_ok": san["ok"], "sanity_failed": san["failed"], "sanity_checks": san["checks"],
        "sanity_warnings": san["warnings"],
        "checkpoints": {"writes": len(ev), "first_write_s": ev[0]["t"] if ev else None,
                        "max_gap_s": round(max(gaps), 2) if gaps else None,
                        "all_imported": all(e.get("imports") for e in ev),
                        "nonfinal_sanity_ok": sum(1 for e in ev if e.get("imports") and not e.get("final")
                                                  and e.get("sanity_ok")),
                        "nonfinal": sum(1 for e in ev if not e.get("final")), "events": ev},
        "log": logs, "path": path,
    }
    with open(os.path.join(OUT, f"build_{name}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"{name}: classes {hs.n_classes} engine {out['engine']} wall {wall:.1f} s stages {out['stage_s']} "
          f"sanity {san['ok']} {san['failed']} ckpt writes {len(ev)} first {out['checkpoints']['first_write_s']} "
          f"maxgap {out['checkpoints']['max_gap_s']} load {out['loadavg']}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(S.SPECS))
    ap.add_argument("--budget", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    for n in a.names:
        one(n, a.budget, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
