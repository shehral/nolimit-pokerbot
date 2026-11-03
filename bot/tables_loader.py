"""Table swapping: which tables the brain plays with, and where the variant layer comes from. Standard library only.

load() picks the tables ONCE at import of brain.py:
  1. the environment variable NOLIMIT_TABLES: a path to a generated tables module (variant/build_tables.py output);
     "builtin" (or "", "nlhe", "none", "0") forces the built-in NLHE tables
  2. else bot/variant_tables.py if it exists (the event-day build target: --out bot/variant_tables.py)
  3. else the built-in NLHE tables (bot/bot_tables.py), the pre-integration bot exactly
A candidate is used only if it imports, passes the gate (variant/bridge.gate_tables: SANITY ok, equity stage done,
META stderr of stage a <= NOLIMIT_TABLES_MAX_STDERR, default 0.015) and its labels match the class space rebuilt from
its SPEC_PARAMS (or, for older modules without SPEC_PARAMS, inferred from the class count: 169 NLHE, 81 short deck,
1,755 three-dealt-keep-two). Anything else falls back to the built-in tables with the reason. One line on stderr says
which tables are active (NOLIMIT_TABLES_QUIET=1 silences it); INFO holds the same facts.

variant_layer() returns the variant package (defaults.respond, HandSpace, bridge) under the private name
nolimit_variant: the bundle's embedded copy if present, else ../variant (or ./variant) loaded by file path, so a
platform module that happens to be called "variant" can never shadow ours. None if neither exists (the adapter's
own safe_request then answers every special request).
"""
import importlib.util
import os
import sys

ENV_TABLES = "NOLIMIT_TABLES"
ENV_MAX_STDERR = "NOLIMIT_TABLES_MAX_STDERR"
ENV_QUIET = "NOLIMIT_TABLES_QUIET"
BUILTIN_WORDS = ("", "builtin", "nlhe", "none", "0", "off")
DEFAULT_FILE = "variant_tables.py"
MAX_STDERR = 0.015
_HERE = os.path.dirname(os.path.abspath(__file__))
_VNAME = "nolimit_variant"

INFO = {"active": "builtin", "path": None, "reasons": [], "line": "", "classes": 169, "spec": "nlhe"}


def variant_layer():
    """The variant package or None (never raises)."""
    m = sys.modules.get(_VNAME)
    if m is not None:
        return m
    for d in (os.path.join(os.path.dirname(_HERE), "variant"), os.path.join(_HERE, "variant")):
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            continue
        try:
            spec = importlib.util.spec_from_file_location(_VNAME, init, submodule_search_locations=[d])
            m = importlib.util.module_from_spec(spec)
            sys.modules[_VNAME] = m
            spec.loader.exec_module(m)
            return m
        except BaseException:          # noqa: BLE001 - a broken layer must never break the bot
            sys.modules.pop(_VNAME, None)
            for k in [k for k in sys.modules if k.startswith(_VNAME + ".")]:
                sys.modules.pop(k, None)
    return None


def _submodule(name):
    vl = variant_layer()
    if vl is None:
        return None
    m = sys.modules.get(f"{_VNAME}.{name}")
    if m is None:
        import importlib
        m = importlib.import_module(f"{_VNAME}.{name}")
    return m


def resolve_path():
    """(path or None, how it was chosen)."""
    env = os.environ.get(ENV_TABLES)
    if env is not None:
        if env.strip().lower() in BUILTIN_WORDS:
            return None, f"{ENV_TABLES}={env!r} forces the built-in tables"
        return env, f"{ENV_TABLES}"
    p = os.path.join(_HERE, DEFAULT_FILE)
    if os.path.exists(p):
        return p, DEFAULT_FILE
    return None, "no variant tables"


def import_tables(path):
    import hashlib
    name = "nolimit_vtables_" + hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spec_for(T, S):
    """VariantSpec for a tables module: SPEC_PARAMS, else inferred from the class count."""
    p = getattr(T, "SPEC_PARAMS", None)
    if p:
        return S.spec_from_params(p), "SPEC_PARAMS"
    n = len(T.EQ_LABELS.split())
    total = getattr(T, "TOTAL_COMBOS", None)
    guess = {(169, 1326): S.nlhe_spec, (81, 630): S.short_deck_spec, (1755, 22100): S.three_keep_two_spec}
    f = guess.get((n, total))
    if f is None:
        raise ValueError(f"no SPEC_PARAMS and no known class space for {n} classes / {total} combos")
    return f(), f"inferred from {n} classes"


class Active:
    """The tables in play. kind "builtin" (names None) or "variant" (names = bridge.brain_names(hs, T))."""

    def __init__(self, kind, path=None, T=None, hs=None, spec=None, names=None, use_fits=False, reasons=()):
        self.kind, self.path, self.T, self.hs, self.spec = kind, path, T, hs, spec
        self.names, self.use_fits, self.reasons = names, use_fits, list(reasons)


def load(path=None, quiet=None):
    """Active tables (never raises). path=None: resolve_path()."""
    how = "argument"
    if path is None:
        path, how = resolve_path()
    reasons = []
    act = None
    if path is not None:
        try:
            act = _load_variant(path, reasons)
        except BaseException as e:      # noqa: BLE001
            reasons.append(f"{type(e).__name__}: {e}")
            act = None
    else:
        reasons.append(how)
    if act is None:
        act = Active("builtin", reasons=reasons)
        INFO.update(active="builtin", path=path, reasons=reasons, classes=169, spec="nlhe")
        line = "tables: built-in NLHE (bot_tables.py, 169 classes)"
        if path is not None:
            line += f"; REJECTED {path}: {'; '.join(reasons)}"
        elif how != "no variant tables":
            line += f" ({how})"
    else:
        T = act.T
        se = ((getattr(T, "META", {}) or {}).get("stderr") or {}).get("a")
        INFO.update(active="variant", path=path, reasons=act.reasons, classes=act.hs.n_classes, spec=act.spec.name)
        line = (f"tables: {path} via {how} (spec {getattr(T, 'SPEC_NAME', '?')}, {act.hs.n_classes} classes, "
                f"stderr a {se}, fits {'refit' if act.use_fits else 'NLHE'}, final {T.META.get('final')})")
    INFO["line"] = line
    if quiet is None:
        quiet = os.environ.get(ENV_QUIET) == "1"
    if not quiet:
        try:
            sys.stderr.write(f"[nolimit_bot] {line}\n")
        except BaseException:           # noqa: BLE001
            pass
    return act


def _load_variant(path, reasons):
    if not os.path.exists(path):
        reasons.append("file not found")
        return None
    S = _submodule("spec")
    H = _submodule("handspace")
    B = _submodule("bridge")
    if S is None or H is None or B is None:
        reasons.append("variant layer not available (needed to rebuild the class space)")
        return None
    T = import_tables(path)
    try:
        max_se = float(os.environ.get(ENV_MAX_STDERR, MAX_STDERR))
    except ValueError:
        max_se = MAX_STDERR
    ok, why = B.gate_tables(T, max_stderr=max_se)
    if not ok:
        reasons.extend(why)
        return None
    spec, src = _spec_for(T, S)
    hs = H.HandSpace(spec)
    names = B.brain_names(hs, T)          # raises when the labels do not belong to this class space
    C = hs.n_classes
    if len(names["EQ_ROWS"]) != C or len(names["PUSH_PCT"]) != C or len(names["CALL_PCT"]) != C:
        reasons.append("table sizes do not match the class space")
        return None
    if any(len(r) != len(names["EQ_GRID"]) for r in names["EQ_ROWS"]):
        reasons.append("EQ_TABLE rows do not match EQ_GRID")
        return None
    fits_src = str((getattr(T, "SOURCES", {}) or {}).get("FITS", ""))
    # refit push/fold fits replace the NLHE ones only when they were solved WITHOUT antes: the brain already
    # prices antes as dead money (P11, equiv_depth); ante-solved fits would count the antes twice
    use_fits = (fits_src.startswith("solver refit") and "jam+call" in fits_src and not spec.ante_bb
                and names["JAM_FIT"] is not None and len(names["JAM_FIT"]) == 5
                and names["CALL_FIT"] is not None and len(names["CALL_FIT"]) == 3)
    return Active("variant", path, T, hs, spec, names, use_fits, reasons=[src])
