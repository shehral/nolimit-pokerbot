"""The variant-verify findings fixed for the bot integration (sim/build_variant_verify.json, 2026-11-03):
class_of_dealt / class_of_safe classify dealt hands of any size properly; the evaluator probe checks every showdown
size up to the 8-seat maximum; a pure double board builds on the NLHE spec; VariantSpec has a bounty rule and
payoff hook; tables carry SPEC_PARAMS; gate_tables gates on stderr; checkpoints are written every few seconds
INSIDE stages (m and c used to write only at their end)."""
import dataclasses
import os
import time

from variant import build_tables as BT
from variant.bridge import gate_tables
from variant.cards import STD_CODEC as C
from variant.handspace import HandSpace
from variant.spec import VariantSpec, nlhe_spec, probe_evaluator, spec_from_params, toss_spec


def _ints(s):
    return C.ints(s.split())


def test_class_of_dealt_any_size():
    hs = HandSpace(toss_spec())                       # 3-dealt classes, 1,755
    for two, lab in (("Ah Ad", "AA|"), ("Kh Qh", "KQs|")):
        ci = hs.class_of_safe(_ints(two))
        assert hs.labels[ci].startswith(lab), hs.labels[ci]
    worst = hs.class_of_safe(_ints("7h 2s"))          # never kept by the static rule: a weak class, not class 0
    assert worst != 0 and not hs.labels[worst].startswith(("AA", "KK", "22"))
    nl = HandSpace(nlhe_spec())
    assert nl.labels[nl.class_of_safe(_ints("Ah Ad Kc"))] == "AA"
    assert nl.labels[nl.class_of_safe(_ints("2c 2d Ah Ks 7d"))] == "AKo"


def test_probe_checks_multiway_showdown_sizes():
    from sim.kit.cards import evaluate as kit_evaluate
    rep = probe_evaluator(dataclasses.replace(toss_spec(), evaluator=kit_evaluate))
    assert rep["showdown_cards"] == {"heads_up": 8, "max_multiway": 14}
    assert rep["exact"][8] and not all(rep["exact"].values())          # the kit evaluator slips at 13-14 cards
    assert rep["ok"] and rep["warnings"]              # heads-up size exact (the build is fine), multiway flagged


def test_double_board_builds_on_the_nlhe_spec():
    db = VariantSpec(name="db", n_boards=2)
    assert not db.separable and db.pure_multi_board
    t = db.table_spec()
    assert t.n_boards == 1 and t.separable and "1 board" in t.notes
    assert nlhe_spec().table_spec() is not None and nlhe_spec().table_spec().n_boards == 1


def test_bounty_field_and_payoff_hook():
    b = VariantSpec(name="bounty", bounty={"mult": 0.5, "flat": 10})
    assert b.bounty_payoff(100) == 160.0 and b.bounty_payoff(100, hit=False) == 100
    assert nlhe_spec().bounty_payoff(100) == 100
    assert "bounty" in b.summary()
    assert spec_from_params(b.spec_params()).bounty == {"mult": 0.5, "flat": 10}


def test_spec_params_round_trip():
    for s in (nlhe_spec(), toss_spec(), VariantSpec(name="sd", removed_ranks="2345")):
        r = spec_from_params(s.spec_params())
        assert HandSpace(r).labels == HandSpace(s).labels


def test_gate_on_stderr():
    class T:
        SANITY = {"ok": True}
        META = {"stage_done": {"a": True}, "stderr": {"a": 0.024}, "final": True}
        SOURCES = {"EQ_TABLE": "mc_range (500 boards)"}
        EQ_LABELS = PUSH_ORDER = CALL_ORDER = ""
        EQ_GRID = EQ_TABLE = ()
    ok, why = gate_tables(T)
    assert not ok and "stderr" in why[0]
    T.META = dict(T.META, stderr={"a": 0.01})
    assert gate_tables(T)[0]
    T.SOURCES = {"EQ_TABLE": "linear model on equity vs random"}
    assert not gate_tables(T)[0]


def test_checkpoints_every_few_seconds_inside_stages(tmp_path):
    path = str(tmp_path / "tables_nlhe_ckpt5.py")   # audit 2: not the tracked variant/out/tables_nlhe_ckpt5.py
    writes = []
    orig = BT.write_module

    def spy(p, tables, sanity, meta):
        writes.append(time.time())
        return orig(p, tables, sanity, meta)
    BT.write_module = spy
    try:
        t0 = time.time()
        res = BT.build_tables(nlhe_spec(), workers=4, time_budget=24, out_path=path, checkpoint_every=2.0,
                              log=lambda m: None)
    finally:
        BT.write_module = orig
    ts = [t0] + writes
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert max(gaps) <= 2.0 + 1.5, gaps               # every stage writes on time (m and c included)
    m = BT.load_tables(path)
    assert m.SPEC_PARAMS["hole_dealt"] == 2 and m.META["checkpoint_every_s"] == 2.0
    assert res["sanity"]["ok"]
