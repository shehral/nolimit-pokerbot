"""NLHE byte-identity: with the plain NLHE spec (no variant tables, plain kit) the integrated bot makes exactly the
decisions the pre-integration bot made (commit df8e93b), decision by decision, over at least 20,000 decisions.

How: deterministic plain-kit tables (seeded decks and opponents, no time limit, BF guard disabled so no decision
depends on the clock), our bot in four seats (v0, v1, v2, v2) against seeded fixtures and the kit's example agent.
Every one of our decisions is serialized as (hand, seat, street, hole, action, amount) and hashed per block of 1,000.
The golden hashes in data/nlhe_golden.json were recorded with the pre-integration sources:

    NOLIMIT_TABLES= .venv/bin/python bot/tests/test_nlhe_identity.py --record    # only on the baseline commit
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nlhe_golden.json")
N_DECISIONS = 20_000
BLOCK = 1000


def record(n=N_DECISIONS):
    os.environ.pop("NOLIMIT_TABLES", None)
    from sim.kit.engine import Table
    from sim.kit.example_agent import ExampleAgent
    from sim.kit.fixtures import AllInAgent, CheckCallAgent, RandomAgent
    from bot import shell

    rows = []
    params = {"GUARD_NS": 10 ** 15,          # no clock dependence: every BF is computed exactly
              # the 2026-11-03 tuning switches (TUNING_LOG notes) pinned to the pre-tuning rules, so this test keeps
              # checking the decision code against the golden recorded at df8e93b (not regenerated); the tuned
              # defaults are covered by test_tuning.py
              "TUNE_BY_PAYOUT": False, "CALLER_CAP_JAM_EVIDENCE": False,
              # the range model (RANGE_MODEL notes) pinned off for the same reason, whatever its shipped default
              "RANGE_MODEL": False,
              # the 2026-11-04 pass-2 tuning switches (TUNING_LOG notes "pass 2") pinned to the built rules too
              "JAM_HERO_EXP": -2.5, "JAM_OPP_EXP_DEFAULT": None, "MIN_JAMS_FACED_TIGHT": 4,
              "REJAM_LEARN": False, "REJAM_BY_LEFT": False, "BB_DEFEND_PCT_HU": None,
              "CALLJAM_PRIOR": None, "SB_WIDEN_MIN_JAMS": None, "T_CALLER_CALLJAM": 0.50}

    def ours(version):
        def make(seat, cfg):
            bot = shell.make(seat, cfg, version=version, params=params)
            orig = bot.act

            def act(state, _o=orig):
                a = _o(state)
                rows.append((state["hand_no"], state["seat"], state["street"], tuple(state["hole"]),
                             tuple(state["board"]), a.get("action"), a.get("amount")))
                return a
            bot.act = act
            return bot
        return make

    opps = [RandomAgent, AllInAgent, CheckCallAgent, ExampleAgent]
    t = 0
    while len(rows) < n:
        facs = []
        for s in range(8):
            if s % 2 == 0:
                facs.append(ours((0, 1, 2, 2)[(s // 2 + t) % 4]))
            else:
                facs.append(opps[(s + t) % 4])
        Table(facs, ("TOP3", "LINEAR", "WTA", "TOP4")[t % 4], ("TURBO", "FAST", "SLOW")[t % 3],
              seed=f"identity:{t}", timing="cpu", time_limit_ms=None).play()
        t += 1
    rows = rows[:n]
    blocks = []
    for i in range(0, n, BLOCK):
        blocks.append(hashlib.sha256(repr(rows[i:i + BLOCK]).encode()).hexdigest()[:16])
    return {"n": n, "tables": t, "blocks": blocks,
            "total": hashlib.sha256(repr(rows).encode()).hexdigest()}, rows


def test_nlhe_decisions_identical_to_baseline():
    with open(GOLDEN) as f:
        gold = json.load(f)
    got, rows = record(gold["n"])
    assert got["n"] == gold["n"] >= N_DECISIONS
    for i, (a, b) in enumerate(zip(got["blocks"], gold["blocks"])):
        assert a == b, f"first differing block {i} (decisions {i * BLOCK}..{(i + 1) * BLOCK - 1}): {rows[i * BLOCK]}"
    assert got["total"] == gold["total"]


if __name__ == "__main__":
    if "--record" in sys.argv:
        g, _ = record()
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(g, f, indent=1)
        print(f"recorded {g['n']} decisions over {g['tables']} tables -> {GOLDEN}")
