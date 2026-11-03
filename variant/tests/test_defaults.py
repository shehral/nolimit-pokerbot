"""Safe defaults are legal for a fuzz of 10,000 random request states, fast, and never raise."""
import random
import time

from variant.defaults import guard, legal, respond, safe_default
from variant.spec import REQUEST_KINDS

FORMATS = {
    "keep": ("keep_indices", "discard_indices", "keep_cards", "discard_cards"),
    "discard": ("keep_indices", "discard_indices", "keep_cards", "discard_cards", "index", "card"),
    "toss": ("index", "card", "discard_indices", "discard_cards"),
    "redraw": ("index_or_none", "discard_indices", "discard_cards", "keep_indices"),
    "bid": ("int",),
    "assign": ("groups",),
    "other": ("option",),
}


def random_view(rng):
    kind = rng.choice(REQUEST_KINDS)
    fmt = rng.choice(FORMATS[kind])
    n = rng.randint(1, 7)
    cards = rng.sample(range(52), n + 5)
    v = {"kind": kind, "response": fmt, "hole": cards[:n], "board": cards[n:n + rng.randint(0, 5)],
         "pot": rng.choice((0, 1, 3, 30, 150, 1000, 12345)), "stack": rng.choice((0, 1, 2, 9, 100, 1500, 16000))}
    if fmt in ("index", "card"):
        v["k"] = 1
    elif kind == "keep":
        v["k"] = rng.randint(1, n)
    elif kind in ("discard", "toss", "redraw"):
        v["k"] = rng.randint(1 if kind != "redraw" else 0, n)
    if kind == "redraw":
        v["optional"] = rng.random() < 0.7
        if fmt in ("index", "index_or_none"):
            v["k"] = 1
    if kind == "bid":
        lo = rng.choice((0, 0, 1, 5, 50))
        v["min"] = lo
        v["max"] = lo + rng.choice((0, 1, 10, 500, 100000))
    if kind == "assign":
        sizes, left = [], n
        while left > 0:
            s = rng.randint(1, left)
            sizes.append(s)
            left -= s
        v["groups"] = sizes
    if kind == "other" or rng.random() < 0.1:
        opts = [rng.randint(0, 9) for _ in range(rng.randint(1, 4))]
        if kind in ("keep", "discard", "toss", "redraw") and fmt not in ("index", "card", "index_or_none"):
            opts = None
        if opts:
            v["options"] = opts
        if kind == "other" and rng.random() < 0.5:
            v["default"] = opts[0] if opts else None
    return v


def test_fuzz_10000_legal_and_fast():
    rng = random.Random(2026)
    worst = 0
    times = []
    for _ in range(10000):
        v = random_view(rng)
        t0 = time.perf_counter_ns()
        r = respond(v)
        times.append(time.perf_counter_ns() - t0)
        if v.get("options") is not None or v["response"] != "option":
            assert legal(v, r), (v, r)
        assert legal(v, safe_default(v)) or v["response"] == "option"
        if v["kind"] == "bid":
            if v.get("options") is None:
                assert v["min"] <= r <= max(v["stack"], v["min"])
            elif any(o <= v["stack"] for o in v["options"]):
                assert r <= v["stack"]
    times.sort()
    p99 = times[int(0.99 * len(times))] / 1000.0
    assert p99 < 100.0, p99


def test_specific_defaults():
    v = {"kind": "bid", "response": "int", "pot": 1000, "stack": 50, "min": 0, "max": 400}
    assert safe_default(v) == 49
    v = {"kind": "bid", "response": "int", "pot": 1000, "stack": 5000, "min": 0, "max": 400}
    assert safe_default(v) == 100
    v = {"kind": "redraw", "response": "index_or_none", "hole": [1, 2], "optional": True}
    assert safe_default(v) is None
    v = {"kind": "keep", "response": "keep_indices", "hole": [50, 51, 0], "k": 2}   # Ah As 2c
    assert sorted(safe_default(v)) == [0, 1]
    v = {"kind": "assign", "response": "groups", "hole": list(range(6)), "groups": [2, 2, 2]}
    assert safe_default(v) == [[0, 1], [2, 3], [4, 5]]
    v = {"kind": "other", "response": "option", "options": ["x", "y"], "default": "y"}
    assert safe_default(v) == "y"


def test_policy_failures_fall_back():
    v = {"kind": "discard", "response": "discard_indices", "hole": [50, 51, 0], "k": 1}

    def boom(view):
        raise RuntimeError("variant bug")
    assert respond(v, boom) == [2]
    assert respond(v, lambda view: [7]) == [2]                    # illegal answer -> default
    assert respond(v, lambda view: [0]) == [0]                    # legal answer kept
    assert respond(v, lambda view: time.sleep(0.002) or [0], budget_us=500) == [2]   # too slow -> default


def test_guard_keeps_callers_action():
    action = ("raise", 300)
    assert guard(lambda: 1 / 0, default=action) == action
    assert guard(lambda: ("call",), default=action) == ("call",)
