"""The hand tracker gives the same per-hand picture whether it is fed by observe() events (the normal path) or
rebuilt from the decision state's history_this_hand alone (the fallback when a platform sends no events).

At every decision of our bot (v2, several seats, random/all-in/calling opponents, three payouts and schedules), a
fresh brain is synced from the view's history and its per-hand fields are compared with the live tracker's."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.kit.engine import Table                                  # noqa: E402
from sim.kit.fixtures import AllInAgent, CheckCallAgent, PassiveRandomAgent, RandomAgent   # noqa: E402
from sim.kit.runner import load_factory                           # noqa: E402

SHELL = sys.modules[load_factory("bot/shell.py:Bot").__module__]
B = SHELL.brain
A = SHELL.adapter
FIELDS = ("street", "contrib", "bets", "folded", "allin", "pf_raises", "last_raiser", "raise_to", "open_k",
          "open_seat", "limpers", "limper_seats", "callers_since_raise", "pf_aggr", "cur", "jam", "jam_ranges",
          "stake", "pre")


def test_observe_and_history_feeds_give_the_same_hand_state():
    fac = load_factory("bot/shell.py:BotV2")
    checked = [0]
    mismatches = []

    def make(seat, cfg):
        bot = fac(seat, cfg)
        orig = bot.act

        def act(state, _o=orig, _bot=bot, _cfg=cfg):
            out = _o(state)
            view = A.normalize(state)
            fresh = B.Brain(state["seat"], _cfg, version=2)
            fresh.tr.sync_from_view(view)
            live = _bot._brain.tr
            for f in FIELDS:
                a, b = getattr(live, f), getattr(fresh.tr, f)
                if a != b:
                    mismatches.append((state["hand_no"], state["street"], f, a, b))
            checked[0] += 1
            return out
        bot.act = act
        return bot
    opps = [RandomAgent, AllInAgent, CheckCallAgent, PassiveRandomAgent]
    for t in range(40):
        facs = [make if s % 3 == 0 else opps[(s + t) % 4] for s in range(8)]
        Table(facs, ("TOP3", "LINEAR", "WTA")[t % 3], ("TURBO", "FAST", "SLOW")[t % 3], seed=f"trk2:{t}",
              timing="cpu", time_limit_ms=None, max_hands=60).play()
    assert checked[0] > 2000, checked[0]
    assert not mismatches, mismatches[:5]


def test_no_collection_starts_inside_act():
    """GC_GUARD_ACT: with the collector paused inside act(), no garbage collection starts during our decisions."""
    import gc
    import time
    from sim.zoo import fields as F
    fac = load_factory("bot/shell.py:BotV1")
    inside = [False]
    hits = []
    total = [0]

    def cb(phase, info):
        if phase == "start":
            total[0] += 1
            if inside[0]:
                hits.append(info["generation"])

    def make(seat, cfg):
        bot = fac(seat, cfg)
        orig = bot.act

        def act(state, _o=orig):
            inside[0] = True
            try:
                return _o(state)
            finally:
                inside[0] = False
        bot.act = act
        return bot
    gc.callbacks.append(cb)
    try:
        opp = F.factory("icm_pro")
        for t in range(120):
            facs = [make] + [opp] * 7
            Table(facs[t % 8:] + facs[:t % 8], "TOP3", "TURBO", seed=f"gcg:{t}", timing="cpu",
                  time_limit_ms=None).play()
    finally:
        gc.callbacks.remove(cb)
    assert total[0] > 10, total[0]          # collections did happen during the run ...
    assert not hits, hits                   # ... but none started inside our act()
