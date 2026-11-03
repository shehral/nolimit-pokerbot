"""Runner: places/points sums, determinism (any worker count), PAIRED alignment, event seating with fillers,
string specs, pooling."""
import functools

import pytest

from sim.kit.engine import Table
from sim.kit.fixtures import (AllInAgent, CheckCallAgent, CheckFoldAgent, PassiveRandomAgent, RandomAgent,
                              ScriptedAgent)
from sim.kit.example_agent import ExampleAgent
from sim.kit.runner import Entrant, as_entrant, run, run_paired, seating_plan

MIXED = [("random", RandomAgent), ("allin", AllInAgent), ("station", CheckCallAgent), ("ex", ExampleAgent),
         ("passive", PassiveRandomAgent), ("random", RandomAgent), ("ex", ExampleAgent), ("checkfold", CheckFoldAgent)]


@pytest.mark.parametrize("payouts,total", [("TOP3", 100.0), ("LINEAR", 100.0), ("WTA", 100.0), ("TOP4", 100.0)])
def test_places_and_points_sum(payouts, total):
    res = run(MIXED, 120, seed=3, payouts=payouts, workers=1)
    for (_tid, seats, fillers, pts, places, bust, hands, surv) in res.tables:
        assert abs(sum(pts) - total) < 1e-9
        assert abs(sum(places) - 36) < 1e-9          # 1 + ... + 8, ties averaged
        assert min(places) >= 1 and max(places) <= 8
        assert sorted(seats) == list(range(8)) and not any(fillers)
        assert 1 <= hands <= 200 and 1 <= surv <= 8
        assert sum(1 for b in bust if b is None) == surv      # survivors have no bust hand
    avg = sum(e.mean for e in res.entrants) / 8
    assert abs(avg - 12.5) < 1e-9                     # zero-sum: the mean seat earns exactly 12.5


def test_reproducible_and_worker_count_independent():
    a = run(MIXED, 64, seed=9, workers=1)
    b = run(MIXED, 64, seed=9, workers=1)
    c = run(MIXED, 64, seed=9, workers=3, chunk_size=5)
    for x, y, z in zip(a.entrants, b.entrants, c.entrants):
        assert x.points == y.points == z.points and x.places == y.places == z.places
        assert x.stats.decisions == y.stats.decisions == z.stats.decisions
    d = run(MIXED, 64, seed=10, workers=1)
    assert [e.points for e in d.entrants] != [e.points for e in a.entrants]


def test_string_specs_and_pooling():
    ents = [("ex", "sim/kit/example_agent.py:ExampleAgent")] * 4 + [("rnd", "sim.kit.fixtures:RandomAgent")] * 4
    res = run(ents, 40, seed=1, workers=2)
    pooled = res.by_name()
    assert set(pooled) == {"ex", "rnd"}
    assert len(pooled["ex"].points) == 160 and len(pooled["rnd"].points) == 160
    assert abs(sum(pooled["ex"].points) + sum(pooled["rnd"].points) - 40 * 100.0) < 1e-6
    assert as_entrant("sim/kit/example_agent.py").name == "example_agent"


def test_event_seating_with_fillers():
    k = 10
    plan = seating_plan(k, 2 * 50, 8, seed=4, mode="event")   # 2 tables per iteration, 6 filler seats
    for it in range(50):
        t1, t2 = plan[2 * it], plan[2 * it + 1]
        real = [e for e, f in t1 + t2 if not f]
        assert sorted(real) == list(range(k))                   # everyone plays exactly once per iteration
        assert sum(1 for _, f in t1 + t2 if f) == 6
    ents = [(f"b{i}", CheckCallAgent if i % 2 else RandomAgent) for i in range(k)]
    res = run(ents, 100, seed=4, workers=1)
    for e in res.entrants:
        assert len(e.points) == 50                              # fillers are not recorded
    real_total = sum(sum(e.points) for e in res.entrants)
    assert real_total < 100 * 100.0                             # fillers took some points (they earn nothing)


def test_sample_seating():
    plan = seating_plan(12, 200, 8, seed=2, mode="sample")
    for t in plan:
        assert len({e for e, _ in t}) == 8 and not any(f for _, f in t)
    with pytest.raises(ValueError):
        seating_plan(5, 10, 8, mode="sample")


def test_paired_mode_same_tables_and_decks():
    base = [("focus", CheckCallAgent)] + [("field", PassiveRandomAgent)] * 7
    alt = [("focus", ExampleAgent)] + [("field", PassiveRandomAgent)] * 7
    pr = run_paired({"station": base, "example": alt}, 60, seed=12, workers=1)
    a, b = pr.results["station"], pr.results["example"]
    assert [r[1] for r in a.tables] == [r[1] for r in b.tables]           # identical seating per table
    assert a.entrants[0].table_ids == b.entrants[0].table_ids
    c = pr.comparisons["example"]
    assert c["n"] == 60 and abs(c["mean_diff"] - (b.entrants[0].mean - a.entrants[0].mean)) < 1e-9
    same = run_paired({"x": base, "y": base}, 30, seed=12, workers=1)
    assert same.comparisons["y"]["mean_diff"] == 0.0 and same.comparisons["y"]["sd_diff"] == 0.0


def _deals(res):
    return {(ev["hand_no"], only): ev["hole"] for only, ev in res.events if ev["type"] == "deal"}


def _boards(res):
    return {ev["hand_no"]: ev["board"] for only, ev in res.events if ev["type"] == "showdown"}


def test_paired_decks_line_up_even_after_play_diverges():
    """Same table seed: every seat gets the same cards in hand h no matter what happened before (position-based
    dealing from a per-hand stream), the board is the same, and the bots' own seeds match too."""
    fa = [CheckCallAgent] * 8
    fb = [CheckFoldAgent] + [CheckCallAgent] * 7
    ta = Table(fa, seed="pair", record_events=True, max_hands=60)
    tb = Table(fb, seed="pair", record_events=True, max_hands=60)
    ra, rb = ta.play(), tb.play()
    assert [s.n_call for s in ra.stats] != [s.n_call for s in rb.stats]      # play really diverged
    da, db = _deals(ra), _deals(rb)
    common = set(da) & set(db)
    assert len(common) > 300
    assert all(da[k] == db[k] for k in common)
    ba, bb = _boards(ra), _boards(rb)
    both = set(ba) & set(bb)
    assert len(both) > 30 and all(ba[h] == bb[h] for h in both)
    assert ta.agent_seeds == tb.agent_seeds


def test_agents_get_fresh_instances_and_own_seeds():
    created = []

    class Tracker(CheckCallAgent):
        def __init__(self, seat, config):
            super().__init__(seat, config)
            created.append((id(self), seat, config["seed"], config["n_seats"], config["payouts"][0]))

    run([Tracker] * 8, 5, seed=0, workers=1)
    assert len(created) == 40 and len({c[0] for c in created}) >= 8
    assert len({c[2] for c in created}) == 40                   # a distinct seed per seat per table
    assert all(c[3] == 8 and c[4] == 50.0 for c in created)


def test_unpicklable_spec_rejected_with_clear_message():
    with pytest.raises(TypeError, match="not picklable"):
        run([("lam", lambda seat, cfg: CheckCallAgent(seat, cfg))] * 8, 4, workers=2)
    ok = run([("part", functools.partial(ScriptedAgent, script=()))] * 8, 4, workers=2)
    assert len(ok.entrants[0].points) == 4


def test_summary_and_json(tmp_path):
    res = run(MIXED, 16, seed=5, workers=1)
    text = res.summary()
    assert "entrant" in text and "station" in text
    p = res.save_json(tmp_path / "r.json")
    import json
    d = json.load(open(p))
    assert d["meta"]["n_tables"] == 16 and len(d["entrants"]) == 8
    assert {"p50", "p99", "p99.9", "max"} <= set(d["entrants"][0]["latency_us"])


def test_bare_path_spec_finds_the_bot_class_not_the_kit_base():
    from sim.kit.runner import load_factory
    fac = load_factory("sim/kit/tests/data/zoo_style_bot.py")
    assert fac.__name__ == "NitStyleAgent"
    res = run([("z", "sim/kit/tests/data/zoo_style_bot.py")] * 8, 4, seed=0, workers=2)
    assert res.entrants[0].stats.exceptions == 0 and res.entrants[0].stats.decisions > 0
    assert load_factory("sim/kit/example_agent.py").__name__ == "ExampleAgent"


def test_gc_mode_applied_and_restored_in_process():
    import gc
    assert gc.isenabled()
    seen = []

    class Probe(CheckCallAgent):
        def __init__(self, seat, config):
            super().__init__(seat, config)
            seen.append(gc.isenabled())

    run([Probe] * 8, 2, seed=0, workers=1, gc_mode="disable")
    assert seen and not any(seen) and gc.isenabled()          # disabled during the run, restored after
    run([Probe] * 8, 2, seed=0, workers=1, gc_mode="freeze")
    assert gc.isenabled() and gc.get_freeze_count() == 0
    with pytest.raises(ValueError):
        run([Probe] * 8, 2, seed=0, workers=1, gc_mode="sometimes")
    assert gc.isenabled()


def test_bad_spec_or_option_fails_fast_instead_of_hanging():
    with pytest.raises(FileNotFoundError):
        run([("typo", "sim/zoo/does_not_exist.py:Nope")] * 8, 4, workers=2)
    with pytest.raises(AttributeError):
        run([("noattr", "sim.kit.fixtures:NoSuchAgent")] * 8, 4, workers=2)
    with pytest.raises(ValueError):
        run([CheckCallAgent] * 8, 4, workers=2, gc_mode="sometimes")
    with pytest.raises(ValueError):
        run([CheckCallAgent] * 8, 4, workers=2, timing="gpu")


def test_worker_init_failure_is_raised_from_tasks():
    from sim.kit import runner as R
    saved = dict(R._W)
    try:
        R._worker_init(["sim.kit.fixtures:NoSuchAgent"], {}, [])
        assert "init_error" in R._W
        with pytest.raises(RuntimeError, match="initialization failed"):
            R._run_chunk((0, 0, []))
    finally:
        R._W.clear()
        R._W.update(saved)
