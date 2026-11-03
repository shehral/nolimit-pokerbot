"""Bridge from a HandSpace + generated tables to the names bot/brain.py uses today. Pure standard library.

    from variant.bridge import brain_names
    names = brain_names(hs, tables_module)      # dict: LABELS, CLASS_ID, PUSH_PCT, CALL_PCT, EQ_GRID, EQ_ROWS,
                                                #       CLASS_OF (2-card pair table or None), class_of, eq_vs,
                                                #       JAM_FIT, CALL_FIT, TOTAL_COMBOS
For NLHE with bot/bot_tables.py these equal brain.py's module globals exactly (tests/test_bridge.py), so the
integration is a rename: brain.py keeps its logic and takes these values from the variant layer.
"""
import bisect

# Gate thresholds for swapping a generated tables module into the bot (verify 2026-11-03: a 90 s toss build had
# SANITY ok but per-class stderr 0.024 and a different top class than a 300 s build; gate on the error, not only ok)
MAX_STDERR_A = 0.015          # per-class standard error of equity vs random (stage a), from META["stderr"]["a"]


def gate_tables(T, max_stderr=MAX_STDERR_A, require_final=False):
    """(ok, reasons) for a generated tables module: SANITY ok, the equity stage done, META.stderr["a"] at most
    max_stderr (a missing stderr fails: the tables are provisional), and optionally a final (not checkpoint) write."""
    reasons = []
    san = getattr(T, "SANITY", None) or {}
    meta = getattr(T, "META", None) or {}
    if not san.get("ok"):
        reasons.append(f"SANITY not ok (failed {san.get('failed')})")
    done = meta.get("stage_done") or {}
    if not done.get("a"):
        reasons.append("stage a (equity vs random) not done: tables are a static-score placeholder")
    se = (meta.get("stderr") or {}).get("a")
    if se is None:
        reasons.append("no stderr for stage a")
    elif se > max_stderr:
        reasons.append(f"stderr a {se:.4f} > {max_stderr} (build longer: 300-900 s for 3-card or rollout games)")
    if require_final and not meta.get("final"):
        reasons.append("checkpoint, not the final write")
    for name in ("EQ_LABELS", "PUSH_ORDER", "CALL_ORDER", "EQ_GRID", "EQ_TABLE"):
        if not hasattr(T, name):
            reasons.append(f"missing {name}")
    return (not reasons), reasons


def brain_names(hs, T):
    """T: a tables module (or any object) with EQ_LABELS, PUSH_ORDER, CALL_ORDER, EQ_GRID, EQ_TABLE and optionally
    JAM_FIT, CALL_FIT. Rows are re-ordered to the HandSpace's class order if the labels come in another order."""
    labels = list(hs.labels)
    t_labels = T.EQ_LABELS.split()
    if sorted(t_labels) != sorted(labels):
        raise ValueError("tables do not belong to this HandSpace (label sets differ)")
    pos = {lab: i for i, lab in enumerate(t_labels)}
    rows = [list(T.EQ_TABLE[pos[lab]]) for lab in labels]
    push_pct = hs.percentiles(T.PUSH_ORDER.split())
    call_pct = hs.percentiles(T.CALL_ORDER.split())
    grid = list(T.EQ_GRID)
    g0, gn = grid[0], grid[-1]

    def eq_vs(cls, J):
        """Same interpolation as brain.eq_vs."""
        row = rows[cls]
        if J <= g0:
            return row[0]
        if J >= gn:
            return row[-1]
        i = bisect.bisect_left(grid, J)
        x0, x1 = grid[i - 1], grid[i]
        return row[i - 1] + (row[i] - row[i - 1]) * (J - x0) / (x1 - x0)

    if hs.order is None and getattr(T, "EQ_RANDOM", None) is not None:
        # class_of_safe maps unknown hands to the median class by strength; install it from the tables
        # (verify 2026-11-03: without this an unknown hand fell back to class 0, the best class)
        try:
            hs.set_strength([T.EQ_RANDOM[pos[lab]] for lab in labels])
        except Exception:  # noqa: BLE001
            pass
    class_of_list = None
    if hs._pair_table is not None:
        class_of_list = [c if c >= 0 else 0 for c in hs._pair_table]   # impossible pairs (a == b) -> 0, as brain.py
    return {
        "LABELS": labels,
        "CLASS_ID": {lab: i for i, lab in enumerate(labels)},
        "PUSH_PCT": push_pct,
        "CALL_PCT": call_pct,
        "EQ_GRID": grid,
        "EQ_ROWS": rows,
        "CLASS_OF": class_of_list,
        "CLASS_OF_STRIDE": hs._pair_M if class_of_list is not None else None,
        "class_of": hs.class_of_safe,
        "eq_vs": eq_vs,
        "JAM_FIT": tuple(getattr(T, "JAM_FIT", ())) or None,
        "CALL_FIT": tuple(getattr(T, "CALL_FIT", ())) or None,
        "TOTAL_COMBOS": hs.total_combos,
    }
