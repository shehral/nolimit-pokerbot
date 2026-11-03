"""Rehearsal variants of the mock kit: toss, shortdeck, bounty, doubleboard, antes, bbante (rules in sim/variants/rules/)."""
from .engine import VTable, make_variant_deck, play_variant_table, settle_bounty, validate_discard
from .evals import evaluate_any, evaluate_short, evaluate_short_trips
from .spec import NAMES, Variant, make_variant

__all__ = ["VTable", "make_variant_deck", "play_variant_table", "settle_bounty", "validate_discard", "evaluate_any",
           "evaluate_short", "evaluate_short_trips", "NAMES", "Variant", "make_variant"]
