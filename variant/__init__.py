"""Variant-agnostic layer for the No Limit rehearsal bot (variants_deep notes s6). Pure standard library.

spec.py          VariantSpec, RequestType, spec_from_kit (+ evaluator probe), worked examples (nlhe_spec ...)
handspace.py     HandSpace: classes, combos, class_of, labels, percentiles (replaces the 169-class constants)
build_tables.py  multi-core table rebuild (equity vs random, vs top-J, push/fold re-solve), checkpoints, sanity
defaults.py      safe legal defaults for discard/keep/toss/assign/bid/redraw/other requests; respond(); guard()
bridge.py        HandSpace + tables -> the names bot/brain.py reads today
cards.py, mc.py, solver.py   codec and evaluators; Monte Carlo engines; stdlib port of the sng_icm solver
"""
from .spec import (VariantSpec, RequestType, spec_from_kit, probe_evaluator, static_score, nlhe_spec,
                   short_deck_spec, three_keep_two_spec, omaha_spec, toss_spec, EXAMPLES)
from .handspace import HandSpace
from .defaults import respond, safe_default, legal, guard

__all__ = ["VariantSpec", "RequestType", "spec_from_kit", "probe_evaluator", "static_score", "nlhe_spec",
           "short_deck_spec", "three_keep_two_spec", "omaha_spec", "toss_spec", "EXAMPLES", "HandSpace",
           "respond", "safe_default", "legal", "guard"]
