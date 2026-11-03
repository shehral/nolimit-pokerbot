"""bridge.brain_names on NLHE + bot_tables reproduces brain.py's globals exactly (the golden test of R5/6.4.1)."""
import random

from bot import bot_tables as T
from bot import brain
from variant.bridge import brain_names
from variant.handspace import HandSpace
from variant.spec import nlhe_spec


def test_brain_names_equal_brain_globals():
    hs = HandSpace(nlhe_spec())
    nm = brain_names(hs, T)
    assert nm["LABELS"] == brain.LABELS
    assert nm["CLASS_ID"] == brain.CLASS_ID
    assert nm["PUSH_PCT"] == brain.PUSH_PCT and nm["CALL_PCT"] == brain.CALL_PCT
    assert nm["EQ_GRID"] == brain.EQ_GRID and nm["EQ_ROWS"] == brain.EQ_ROWS
    assert nm["CLASS_OF_STRIDE"] == 52 and nm["CLASS_OF"][:2704] == brain.CLASS_OF
    assert nm["TOTAL_COMBOS"] == 1326
    rng = random.Random(1)
    for _ in range(5000):
        c, J = rng.randrange(169), rng.random() * 1.1
        assert nm["eq_vs"](c, J) == brain.eq_vs(c, J)
    for a in range(52):
        for b in range(52):
            if a != b:
                assert nm["class_of"]((a, b)) == brain.CLASS_OF[a * 52 + b]
