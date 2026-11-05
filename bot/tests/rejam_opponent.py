"""Re-jam-any-two opponent for the pass-2 tuning screen (TUNING_LOG notes, 2026-11-04; DOCTRINE notes 2.8 and 3).

`RejamAnyTwo` = the zoo's icm_pro, except: facing a single non-all-in preflop open (a raise of at most `max_open_bb`)
from a late seat (at most `late_seats` players behind the opener), with an effective stack between `lo_bb` and `hi_bb`
big blinds, it goes all-in with any two cards. This is the level-2 exploit the strategy review names ("re-jam any two over
our 13-20 bb opens"), which no zoo or AI-field bot plays (their re-jammers are tight). Everything else, first-in jams
and jam calls included, is icm_pro, so the cell isolates the re-jam leak.

Not a test file (pytest collects test_*.py only). The tuning driver registers it as the archetype "rejam_any_two" in
sim.zoo.fields.ARCHETYPES at import time (also in the spawned workers), so sim/zoo is not modified.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.zoo._common import all_in, eff_stack_bb, raises_this_street   # noqa: E402
from sim.zoo.icm_pro import IcmPro                                     # noqa: E402


class RejamAnyTwo(IcmPro):
    PARAMS = dict(IcmPro.PARAMS, lo_bb=13.0, hi_bb=20.0, late_seats=3, max_open_bb=3.0)
    JITTER = dict(IcmPro.JITTER)

    def preflop(self, state):
        p = self.p
        bb = state["bb"]
        n = state["n_seats"]
        me = state["seat"]
        top = max(state["bets"])
        if top > bb and top <= p["max_open_bb"] * bb and raises_this_street(state) == 1:
            raiser = max((j for j in range(n) if j != me and state["active"][j]), key=lambda j: state["bets"][j])
            if not state["all_in"][raiser]:
                order = [(state["bb_seat"] + 1 + i) % n for i in range(n)]
                # players behind the opener when it opened = seats dealt in after it (folded since or not)
                behind = sum(1 for j in order[order.index(raiser) + 1:]
                             if state["stacks"][j] + state["contributed"][j] > 0)
                S = eff_stack_bb(state)
                if behind <= p["late_seats"] and p["lo_bb"] <= S <= p["hi_bb"]:
                    return all_in(state)
        return super().preflop(state)


def register():
    """Make 'rejam_any_two' a zoo archetype name for sim/bench/common (idempotent)."""
    from sim.zoo import fields as F
    sys.modules.setdefault("sim.zoo.rejam_any_two", sys.modules[__name__])
    F.ARCHETYPES.setdefault("rejam_any_two", ("rejam_any_two", "RejamAnyTwo"))
