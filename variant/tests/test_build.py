"""build_tables: NLHE in 60 s on this Mac reaches the bot's EQ table within 1.5 points on every class (every cell);
the time budget is respected and partial tables are checkpointed and loadable.

The 60 s test is slow by design (about 62 s); run the rest with  -k "not sixty".
"""
import os
import threading
import time

import pytest

from bot import bot_tables as T
from bot import brain
from variant.build_tables import build_tables, load_tables
from variant.spec import nlhe_spec, short_deck_spec

WORKERS = 8      # outputs go to pytest's tmp_path (audit 2, 2026-11-04): the committed copies under variant/out/test/
                 # are reference fixtures and used to be rewritten, and shown as modified, by every test run


def test_sixty_seconds_nlhe_matches_bot_tables(tmp_path):
    path = str(tmp_path / "tables_nlhe_60s.py")
    t0 = time.time()
    res = build_tables(nlhe_spec(), workers=WORKERS, time_budget=60, out_path=path,
                       range_order=T.PUSH_ORDER.split(), log=lambda m: None)
    wall = time.time() - t0
    assert wall <= 60.0 + 1.0, wall
    m = load_tables(path)
    assert m.EQ_LABELS == T.EQ_LABELS and m.EQ_GRID == T.EQ_GRID
    worst = 0.0
    for lab, ra, rb in zip(T.EQ_LABELS.split(), m.EQ_TABLE, T.EQ_TABLE):
        d = max(abs(a - b) for a, b in zip(ra, rb))
        worst = max(worst, d)
        assert d <= 0.015, (lab, d)
    assert m.SANITY["ok"] and m.SANITY["checks"]["nlhe_anchors_ok"]
    assert m.META["final"] and m.META["stage_done"].get("b")
    # refitted Nash fits close to the bot's (sng_icm) fits
    jf, cf = m.JAM_FIT, m.CALL_FIT
    bj, bc = brain.PARAMS["JAM_FIT"], brain.PARAMS["CALL_FIT"]
    fi = m.META.get("fit_info") or {}
    print("fits", jf, cf, "solver", {k: fi.get(k) for k in ("hu_iters_done_min", "multi_iters_done_min")},
          "matrix boards", m.META["boards"]["m"])
    if m.META["stage_done"].get("c") and fi.get("multi_iters_done_min", 0) >= 100 and m.META["boards"]["m"] >= 2500:
        assert abs(jf[0] / bj[0] - 1) < 0.15 and abs(jf[1] - bj[1]) < 0.06 and abs(jf[2] - bj[2]) < 0.06
        assert abs(jf[3] / bj[3] - 1) < 0.15 and abs(jf[4] - bj[4]) < 0.06
        assert abs(cf[0] / bc[0] - 1) < 0.15 and abs(cf[1] - bc[1]) < 0.06 and abs(cf[2] - bc[2]) < 0.06
    print(f"max cell diff {worst:.4f}, wall {wall:.1f} s, stages {m.META['stage_s']}, boards {m.META['boards']}")


def test_budget_respected_and_checkpoints(tmp_path):
    path = str(tmp_path / "tables_short_ckpt.py")
    if os.path.exists(path):
        os.remove(path)
    seen = []
    stop = threading.Event()

    def watch():
        last = None
        while not stop.is_set():
            try:
                mt = os.path.getmtime(path)
                if mt != last:
                    last = mt
                    m = load_tables(path)
                    seen.append((m.META.get("final"), dict(m.META.get("stage_done", {})), len(m.EQ_TABLE),
                                 m.SOURCES["EQ_TABLE"][:9]))
            except Exception:  # noqa: BLE001 (file mid-replace or not there yet)
                pass
            time.sleep(0.05)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t0 = time.time()
    res = build_tables(short_deck_spec(), workers=4, time_budget=12, out_path=path, checkpoint_every=1.0,
                       log=lambda m: None)
    wall = time.time() - t0
    time.sleep(0.2)
    stop.set()
    th.join()
    assert wall <= 12 + 1.0, wall
    partial = [s for s in seen if not s[0]]
    assert len(partial) >= 3, seen                                   # several checkpoints before the end
    assert all(s[2] == 81 for s in seen)                             # every checkpoint is a complete table
    assert any(s[3] == "linear mo" for s in partial)                 # provisional table shipped early
    assert any(s[3] == "mc_range " for s in seen)                    # then MC rows
    assert res["sanity"]["ok"]


def test_tiny_budget_still_ships(tmp_path):
    path = str(tmp_path / "tables_nlhe_tiny.py")
    t0 = time.time()
    res = build_tables(nlhe_spec(), workers=2, time_budget=2.5, out_path=path, log=lambda m: None)
    assert time.time() - t0 <= 2.5 + 1.0
    m = load_tables(path)
    assert len(m.EQ_TABLE) == 169 and len(m.EQ_TABLE[0]) == len(m.EQ_GRID)
    assert sorted(m.PUSH_ORDER.split()) == sorted(m.EQ_LABELS.split())
    assert m.META["final"]
