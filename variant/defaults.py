"""Safe, legal defaults for every special request type (variants_deep notes s6.1, R3). Pure standard library.

The adapter turns the kit's request into a REQUEST VIEW (a plain dict) and calls respond(view). Keys:
  kind      "discard" | "keep" | "toss" | "assign" | "bid" | "redraw" | "other"
  response  answer format: "keep_indices" | "discard_indices" | "keep_cards" | "discard_cards" | "index" | "card" |
            "int" | "index_or_none" | "groups" | "option"
  hole, board   card ints (kit numbering); k: keep -> cards kept, discard/toss -> cards given up, redraw -> max cards
  pot, stack, min, max   chips (bids); groups: list of group sizes (assign); options: the kit's explicit legal list
  default   the kit template's own answer, if any; optional: the request may be declined (redraw)

Defaults (cheap, no rollouts, measured well under 100 us):
  keep / discard / toss   keep the best subset by static score (pairs, suitedness, connectivity, high cards)
  redraw                  decline when optional; else redraw the weakest card
  bid                     min(floor(0.1 x pot), stack - 1), clamped to [min, max], never above the stack
  assign                  identity assignment (consecutive cards to groups in order)
  other                   the template default, else the first legal option, else None
respond(view, policy) runs the optional policy first and keeps its answer only if it is legal; any exception or
illegal answer falls back to the default, and respond itself never raises. guard() does the same for optional work
around a computed betting action: an exception returns the caller's own action, never check/fold.
"""
import time
from itertools import combinations

from .spec import static_score


# ===================================================================== legality
def _norm(x):
    if isinstance(x, (list, tuple)):
        return tuple(_norm(v) for v in x)
    return x


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def n_keep(view):
    """Number of cards kept implied by the view."""
    n = len(view.get("hole") or ())
    k = view.get("k")
    kind = view.get("kind")
    if kind == "keep":
        return k if _is_int(k) else n
    if kind in ("discard", "toss", "redraw"):
        return n - (k if _is_int(k) else 1)
    return n


def legal(view, resp, spec_rule=None):
    """True when `resp` is a legal answer to the request view (plus the kit's own rule if given)."""
    try:
        opts = view.get("options")
        if opts is not None:
            if _norm(resp) not in set(_norm(o) for o in opts):
                return False
            return spec_rule(view, resp) if spec_rule else True
        fmt = view.get("response")
        hole = list(view.get("hole") or ())
        n = len(hole)
        k = view.get("k")
        decline_ok = view.get("kind") == "redraw" and bool(view.get("optional", True))
        if fmt in ("keep_indices", "discard_indices"):
            want = n_keep(view) if fmt == "keep_indices" else n - n_keep(view)
            ok = (isinstance(resp, (list, tuple)) and all(_is_int(i) and 0 <= i < n for i in resp)
                  and len(set(resp)) == len(resp))
            ok = ok and (len(resp) == want or (decline_ok and len(resp) == (n if fmt == "keep_indices" else 0)))
        elif fmt in ("keep_cards", "discard_cards"):
            want = n_keep(view) if fmt == "keep_cards" else n - n_keep(view)
            ok = (isinstance(resp, (list, tuple)) and all(c in hole for c in resp) and len(set(resp)) == len(resp))
            ok = ok and (len(resp) == want or (decline_ok and len(resp) == (n if fmt == "keep_cards" else 0)))
        elif fmt == "index":
            ok = _is_int(resp) and 0 <= resp < n
        elif fmt == "card":
            ok = resp in hole
        elif fmt == "int":
            lo, hi = view.get("min", 0), view.get("max")
            stack = view.get("stack")
            ok = _is_int(resp) and resp >= lo and (hi is None or resp <= hi)
            if ok and stack is not None and lo <= stack:
                ok = resp <= stack
        elif fmt == "index_or_none":
            if resp is None:
                ok = bool(view.get("optional", True))
            else:
                ok = _is_int(resp) and 0 <= resp < n
        elif fmt == "groups":
            sizes = list(view.get("groups") or ())
            ok = (isinstance(resp, (list, tuple)) and len(resp) == len(sizes)
                  and all(isinstance(g, (list, tuple)) and len(g) == s for g, s in zip(resp, sizes)))
            if ok:
                flat = [i for g in resp for i in g]
                ok = all(_is_int(i) and 0 <= i < n for i in flat) and len(set(flat)) == len(flat)
        elif fmt == "option":
            ok = False      # an "option" request without an options list has no checkable legal set
        else:
            ok = False
        if ok and spec_rule:
            ok = bool(spec_rule(view, resp))
        return ok
    except Exception:  # noqa: BLE001
        return False


# ===================================================================== defaults
def best_keep_indices(hole, k, score=None, slot=4):
    """Indices of the best k cards by static score (ties: first in order). Never rolls out."""
    n = len(hole)
    if k >= n:
        return tuple(range(n))
    if k <= 0:
        return ()
    sc = score or (lambda cs: static_score(cs, slot))
    best, bk = None, None
    for idx in combinations(range(n), k):
        s = sc([hole[i] for i in idx])
        if bk is None or s > bk:
            best, bk = idx, s
    return best


def keep_with_penalty(view, penalty, score=None, slot=4):
    """Keep/discard/toss answer that maximises static score(kept) - penalty(kept, given_up) over every legal kept
    subset (DISCARD_LEAK notes: the information-leak penalty on revealed discards). penalty(kept_cards,
    given_up_cards) -> float in score units; a penalty of 0 gives exactly best_keep_indices (same order, same
    ties). Returns an answer in the view's format, or None for other request kinds. Never rolls out."""
    if view.get("kind") not in ("keep", "discard", "toss"):
        return None
    hole = list(view.get("hole") or ())
    n = len(hole)
    k = n_keep(view)
    if k >= n or k <= 0:
        return None
    sc = score or (lambda cs: static_score(cs, slot))
    best, bk = None, None
    for idx in combinations(range(n), k):
        kept = [hole[i] for i in idx]
        out = [hole[i] for i in range(n) if i not in idx]
        s = sc(kept) - penalty(kept, out)
        if bk is None or s > bk:
            best, bk = idx, s
    opts = view.get("options")
    if opts:
        return _best_option(view, opts, best)
    return _from_keep(view, best)


def _from_keep(view, keep_idx):
    fmt = view.get("response")
    hole = list(view.get("hole") or ())
    n = len(hole)
    disc = tuple(i for i in range(n) if i not in keep_idx)
    if fmt == "keep_indices":
        return list(keep_idx)
    if fmt == "discard_indices":
        return list(disc)
    if fmt == "keep_cards":
        return [hole[i] for i in keep_idx]
    if fmt == "discard_cards":
        return [hole[i] for i in disc]
    if fmt in ("index", "index_or_none"):
        return disc[0] if disc else 0
    if fmt == "card":
        return hole[disc[0]] if disc else (hole[0] if hole else None)
    return list(keep_idx)


def default_bid(view):
    pot = view.get("pot") or 0
    stack = view.get("stack")
    lo = view.get("min", 0)
    hi = view.get("max")
    v = int(0.1 * pot)
    if stack is not None:
        v = min(v, stack - 1)
    v = max(v, 0)
    if hi is not None:
        v = min(v, hi)
    if stack is not None and lo <= stack:
        v = min(v, stack)
    return max(v, lo)


def safe_default(view, spec=None, score=None):
    """The legal default answer for a request view (see the module docstring)."""
    kind = view.get("kind")
    opts = view.get("options")
    hole = list(view.get("hole") or ())
    slot = spec.codec.slot if spec is not None else 4
    sc = score or (spec.keep_score if spec is not None and spec.keep_score else None)
    if kind in ("keep", "discard", "toss"):
        kk = best_keep_indices(hole, n_keep(view), sc, slot)
        if opts:
            return _best_option(view, opts, kk)
        return _from_keep(view, kk)
    if kind == "redraw":
        if view.get("optional", True):
            if opts:
                for o in opts:
                    if o is None or _norm(o) in ((), "decline", "none", "no", False):
                        return o
                return opts[0]
            fmt = view.get("response")
            if fmt == "index_or_none":
                return None
            if fmt in ("discard_indices", "discard_cards"):
                return []
            if fmt == "keep_indices":
                return list(range(len(hole)))
            if fmt == "keep_cards":
                return list(hole)
        kk = best_keep_indices(hole, n_keep(view), sc, slot)
        if opts:
            return _best_option(view, opts, kk)
        return _from_keep(view, kk)
    if kind == "bid":
        if opts:
            ints = [o for o in opts if _is_int(o)]
            stack = view.get("stack")
            under = [o for o in ints if stack is None or o <= stack]
            ints = under or ints
            tgt = default_bid(dict(view, min=min(ints) if ints else 0))
            return min(ints, key=lambda o: (abs(o - tgt), o)) if ints else opts[0]
        return default_bid(view)
    if kind == "assign":
        sizes = list(view.get("groups") or ())
        if opts:
            return opts[0]
        out, i = [], 0
        for s in sizes:
            out.append(list(range(i, i + s)))
            i += s
        return out
    # other / unknown
    if "default" in view and view["default"] is not None:
        return view["default"]
    if opts:
        return opts[0]
    return None


def _best_option(view, opts, keep_idx):
    """Among explicit options pick the one closest to our keep (same kept set if present)."""
    hole = list(view.get("hole") or ())
    want = set(hole[i] for i in keep_idx)
    for o in opts:
        try:
            r = _from_keep(view, keep_idx)
            if _norm(o) == _norm(r):
                return o
        except Exception:  # noqa: BLE001
            pass
    for o in opts:
        if isinstance(o, (list, tuple)) and set(o) == want:
            return o
    return opts[0]


def respond(view, policy=None, spec=None, spec_rule=None, budget_us=None):
    """Answer a special request. Tries `policy(view)` (optional work: may raise, may be slow); keeps it only if it is
    legal and within budget_us; otherwise the safe default. Never raises."""
    t0 = time.perf_counter_ns()
    if policy is not None:
        try:
            r = policy(view)
            if (budget_us is None or (time.perf_counter_ns() - t0) <= budget_us * 1000) and legal(view, r, spec_rule):
                return r
        except Exception:  # noqa: BLE001
            pass
    try:
        r = safe_default(view, spec)
        if legal(view, r, spec_rule) or view.get("response") == "option":
            return r
    except Exception:  # noqa: BLE001
        r = None
    try:
        opts = view.get("options")
        if opts:
            return opts[0]
        if view.get("default") is not None:
            return view["default"]
    except Exception:  # noqa: BLE001
        pass
    return r


def guard(fn, *args, default=None, **kw):
    """Run optional variant work; on any exception return `default` (the caller's own computed action)."""
    try:
        return fn(*args, **kw)
    except Exception:  # noqa: BLE001
        return default


# ===================================================================== registry for spec requests
def view_for(spec_request, **fields):
    """Build a request view from a spec RequestType plus the state fields the adapter extracted."""
    v = {"kind": spec_request.kind, "response": spec_request.response, "optional": spec_request.optional}
    if spec_request.k is not None:
        v["k"] = spec_request.k
    v.update(fields)
    return v
