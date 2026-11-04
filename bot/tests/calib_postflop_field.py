"""Measure the field's postflop behaviour by stratum: the source of PARAMS["PE"]'s W_BET / W_CALL / W_RAISE /
CONT_RAISE (POSTFLOP_ENGINE notes section 3). Ground truth from the opponents' own hole cards (a simulator-only view).

    .venv/bin/python bot/tests/calib_postflop_field.py measure 300 AI_MIX,STRONG 9002 bot/out/postflop_engine/calib_field_9002.json
    .venv/bin/python bot/tests/calib_postflop_field.py report bot/out/postflop_engine/calib_field_9002.json [ALL|arch1,arch2]

For every opponent postflop decision: street, stratum of its hand (postflop.stratum), situation (first = nothing to
call, face = facing a bet, face_raise = facing a raise of its own bet), bet-size bucket (s <= 0.4 pot < m <= 0.8 < l)
and the action. Our seat: v2 (engine off), so the spots come from the shipped bot's play (bench plan and table seeds
of sim/bench/common, AI-bot timeout model at time_scale 1.5). The report prints, per street: P(bet | stratum) when it
can bet, P(continue) and P(raise) facing a bet by size bucket, P(continue) facing a raise.
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STREET = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _parse(c):
    if isinstance(c, int):
        return c
    return "23456789TJQKA".index(c[0].upper()) * 4 + "cdhs".index(c[1].lower())


class Spy:
    """Wraps an opponent; records its postflop decisions with its own hand's stratum."""

    def __init__(self, inner, arch, rec):
        self.inner = inner
        self.arch = arch
        self.rec = rec
        obs = getattr(inner, "observe", None)
        if callable(obs) and not getattr(obs, "kit_default", False):
            self.observe = obs

    def act(self, state):
        from bot import postflop as PF
        a = self.inner.act(state)
        try:
            st = STREET.get(state["street"], state["street"])
            if st and len(state["hole"]) == 2:
                h = [_parse(c) for c in state["hole"]]
                b = [_parse(c) for c in state["board"]]
                s = PF.stratum(*PF.hand_feat(h[0], h[1], b))
                tc = state.get("to_call", 0)
                bets = state.get("bets") or []
                kind = a.get("action") if isinstance(a, dict) else str(a)
                if tc <= 0:
                    sit, sz = "first", "-"
                else:
                    sit = "face_raise" if bets and bets[state["seat"]] > 0 else "face"
                    before = state["pot"] - tc
                    f = tc / before if before > 0 else 9
                    sz = "s" if f <= 0.4 else ("m" if f <= 0.8 else "l")
                self.rec[(self.arch, st, sit, sz, s, kind)] += 1
        except Exception as e:                      # noqa: BLE001 - a measurement tool: count, never break play
            self.rec[("ERR", str(e)[:40], "", "", 0, "")] += 1
        return a


def measure(n, fields, seed, out):
    from sim.bench import common as C
    from sim.field_ai import fields as AF
    from sim.kit.engine import Table
    rec = collections.Counter()
    sp = AF.sandbox_p("1.5")
    for field in fields:
        for tid in range(n):
            our, opp = C.plan_table(field, seed, tid)
            facs, k = [], 0
            for s in range(8):
                if s == our:
                    facs.append(C.factory_for("v2"))
                else:
                    fac, arch = C.factory_for(opp[k], sandbox_p=sp), C.archetype_of(opp[k])
                    facs.append(lambda seat, cfg, _f=fac, _a=arch: Spy(_f(seat, cfg), _a, rec))
                    k += 1
            Table(facs, seed=C.table_seed(seed, tid), time_limit_ms=None).play()
    with open(out, "w") as f:
        json.dump({"|".join(map(str, k)): v for k, v in rec.items()}, f)
    print("records", sum(rec.values()), "->", out)


def report(path, group="ALL"):
    d = json.load(open(path))
    T = collections.Counter()
    for k, v in d.items():
        p = k.split("|")
        if p[0] == "ERR":
            print("ERR", k, v)
            continue
        arch, st, sit, sz, s, kind = p
        if group != "ALL" and arch not in group.split(","):
            continue
        T[(int(st), sit, sz, int(s), kind)] += v

    def rate(st, sit, sz, s, kinds):
        tot = hit = 0
        for (a, b, c, e, k), v in T.items():
            if a == st and b == sit and (sz is None or c == sz) and e == s:
                tot += v
                if k in kinds:
                    hit += v
        return (hit / tot if tot else float("nan")), tot

    for st in (1, 2, 3):
        print(f"street {st}")
        print("  first-to-act/checked-to:", "  ".join(
            "s%d bet %.2f (n %d)" % ((s,) + rate(st, "first", None, s, ("raise", "allin", "bet"))) for s in range(4)))
        for sz in ("s", "m", "l"):
            row = []
            for s in range(4):
                r, n = rate(st, "face", sz, s, ("call", "raise", "allin"))
                rr, _ = rate(st, "face", sz, s, ("raise", "allin"))
                row.append(f"s{s} cont {r:.2f} raise {rr:.2f} (n {n})")
            print(f"  face {sz}:", "  ".join(row))
        print("  face raise:", "  ".join(
            "s%d cont %.2f (n %d)" % ((s,) + rate(st, "face_raise", None, s, ("call", "raise", "allin")))
            for s in range(4)))


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(int(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4]), sys.argv[5])
    else:
        report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ALL")
