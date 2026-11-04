"""VariantSpecs for the five rehearsal variants of sim/variants plus plain NLHE, built the event-day way:
spec_from_kit(<the variant's own evaluator>, ...) with the values read off sim/variants/rules/*.md and engine.py.
Module-level functions so build workers can use them under fork or spawn.

What the spec CANNOT express: the multiway toss board (the spec's board is heads-up: 2 + 1 + 1 plus one toss per
player in the rollout) and level-dependent antes. Bounty payoffs are VariantSpec.bounty (since 2026-11-03).
"""
from sim.kit.cards import evaluate as kit_evaluate
from sim.variants.evals import evaluate_any, evaluate_short

from variant.spec import RequestType, spec_from_kit

ANTE_BB = 0.125      # sim/variants antes: round(0.125 x bb) from level 2


def plain():
    return spec_from_kit(kit_evaluate, name="plain")


def toss():
    return spec_from_kit(evaluate_any, name="toss", hole_dealt=3, hole_kept=2, keep_street=1, discard_to="board",
                         discards_public=True, board=(2, 1, 1),
                         requests=(RequestType("toss", "discard", response="index", k=1),),
                         revealed=("tossed card",))


def shortdeck():
    return spec_from_kit(evaluate_short, name="shortdeck", removed_ranks="2345")


def bounty():
    # card game = plain hold'em; the payoff rule is VariantSpec.bounty (bounty_payoff is the hook the bot uses)
    return spec_from_kit(kit_evaluate, name="bounty", revealed=("bounty_bonus chips",),
                         bounty={"mult": 0.5, "flat": 10, "mode": "level"})


def doubleboard():
    # a pure double board: build_tables builds it on the NLHE spec (table_spec(), identical equities), so the
    # separable engines, the matrix and the push/fold re-solve all run (verify 2026-11-03 used the rollout engine)
    return spec_from_kit(kit_evaluate, name="doubleboard", n_boards=2)


def antes():
    return spec_from_kit(kit_evaluate, name="antes", ante_bb=ANTE_BB,
                         notes="ante 0.125 bb from level 2 only; the spec has one ante for all levels")


SPECS = {"plain": plain, "toss": toss, "shortdeck": shortdeck, "bounty": bounty, "doubleboard": doubleboard,
         "antes": antes}


def get(name):
    """(spec, probe report) for a variant name."""
    return SPECS[name]()


# "module:function" entry points for `python -m variant.build_tables --spec variant/verify/specs.py:<name>_spec`
def plain_spec():
    return plain()[0]


def toss_spec():
    return toss()[0]


def shortdeck_spec():
    return shortdeck()[0]


def bounty_spec():
    return bounty()[0]


def doubleboard_spec():
    return doubleboard()[0]


def antes_spec():
    return antes()[0]
