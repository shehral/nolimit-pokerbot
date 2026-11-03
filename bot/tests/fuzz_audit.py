"""Audit fuzz of our bot (2026-11-03, auditor-written). Run from the project root:

    .venv/bin/python bot/tests/fuzz_audit.py --tables 6000 --workers 8 --out bot/out/fuzz_audit.json

Part A (live, engine-driven): our bot (v0/v1/v2, 1-3 copies) at kit tables with randomized ADVERSARIAL configs:
  2-10 seats; stacks from 1 chip to 16,000 (many below one big blind); odd blind levels (1/1, 1/2, 5/5, 7/13,
  11/23, 33/67 + ante 3, 250/500 + ante 50, steep schedules, a level per hand); antes; payouts of lengths 1-10
  (TOP3, LINEAR, WTA, TOP4, [100], [65,35], 7 places, non-100 sums, all-zero); 1-200 hands; opponents that shove
  everything, min-raise everything, leave one chip behind, random sizes, check-call, plus zoo bots.
  STRICT mode (the shell masks nothing), 1 ms enforced on thread CPU time (wall clock recorded too).
Part B (cold replay): a sample of the live states replayed on a FRESH instance (no observe events, the tracker
  must rebuild the hand from history) and on a STALE instance (it last saw another table's hand).
Part C (mutation, production mode): replayed states with optional fields deleted or renamed, history entries
  with unknown action names, extra keys; the shell must still return a legal action (masked errors counted).
Every output is checked with the kit's own validate_action.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PAYOUTS = ["TOP3", "LINEAR", "WTA", "TOP4", (100,), (65, 35), (50, 30, 20), (40, 25, 15, 10, 5, 3, 2),
           (7, 2, 1), (1000, 600, 400), (34, 33, 33), (0, 0, 0, 0, 0, 0, 0, 0),
           (30, 20, 15, 10, 8, 7, 5, 3, 1, 1)]
SCHEDULES = ["TURBO", "FAST", "SLOW", ((1, 1, 0),), ((1, 2, 0),), ((5, 5, 0),), ((7, 13, 0), (11, 23, 0)),
             ((33, 67, 3), (50, 100, 5)), ((250, 500, 50), (500, 1000, 100)),
             ((10, 20, 0), (40, 80, 0), (160, 320, 0), (640, 1280, 0), (2560, 5120, 0), (5000, 10000, 1000))]
STACKS = [1, 2, 3, 5, 9, 15, 21, 40, 99, 150, 400, 777, 2000, 2000, 2000, 5000, 16000]


def _legal_actions_of(state):
    return state["legal_actions"]


class AllIn:
    def __init__(self, seat, config):
        pass

    def act(self, s):
        if "raise" in s["legal_actions"]:
            return {"action": "raise", "amount": s["max_raise_to"]}
        return {"action": "call"} if "call" in s["legal_actions"] else {"action": "check"}


class MinRaiser:
    def __init__(self, seat, config):
        self.r = random.Random(config["seed"])

    def act(self, s):
        if "raise" in s["legal_actions"] and self.r.random() < 0.7:
            return {"action": "raise", "amount": s["min_raise_to"]}
        return {"action": "call"} if "call" in s["legal_actions"] else {"action": "check"}


class OneChipLeaver:
    """Raises to one chip below all-in when that is legal (tiny stacks next), else min-raises or calls."""

    def __init__(self, seat, config):
        self.r = random.Random(config["seed"])

    def act(self, s):
        L = s["legal_actions"]
        if "raise" in L and self.r.random() < 0.5:
            lo, hi = s["min_raise_to"], s["max_raise_to"]
            return {"action": "raise", "amount": hi - 1 if hi - 1 >= lo else hi}
        if "call" in L:
            return {"action": "call"} if self.r.random() < 0.8 else {"action": "fold"}
        return {"action": "check"}


class RandomSizes:
    def __init__(self, seat, config):
        self.r = random.Random(config["seed"])

    def act(self, s):
        L = s["legal_actions"]
        a = L[int(self.r.random() * len(L))]
        if a == "raise":
            lo, hi = s["min_raise_to"], s["max_raise_to"]
            return {"action": "raise", "amount": lo + int(self.r.random() * (hi - lo + 1))}
        return {"action": a}


class CheckCall:
    def __init__(self, seat, config):
        pass

    def act(self, s):
        return {"action": "check"} if "check" in s["legal_actions"] else {"action": "call"}


def _zoo(key):
    from sim.zoo import fields as F
    return F.factory(key)


class Recorder:
    """Wraps our bot. Times ONLY the inner act() (wall + thread CPU, own 0.1 us histograms; the kit's limit is off so
    this wrapper's bookkeeping can never auto-fold us), tallies coverage categories, keeps a sample of
    (state, config) for the cold replay, and checks the bot never mutates the state it is given."""

    def __init__(self, inner, config, version, sink, cov, tim, p, rng):
        self.inner = inner
        self.config = dict(config, bot_version=version)
        self.version = version
        self.sink = sink
        self.cov = cov
        self.tim = tim
        self.p = p
        self.rng = rng
        obs = getattr(inner, "observe", None)
        if callable(obs):
            self.observe = obs

    def act(self, s):
        keep = self.rng.random() < self.p
        before = copy.deepcopy(s) if keep else None
        tc, pc = time.thread_time_ns, time.perf_counter_ns
        c0 = tc()
        t0 = pc()
        try:
            return self.inner.act(s)
        finally:
            el, cel = pc() - t0, tc() - c0
            T = self.tim[self.version]
            h = T["wall"]
            b = el // 100
            h[b] = h.get(b, 0) + 1
            h = T["cpu"]
            b = cel // 100
            h[b] = h.get(b, 0) + 1
            if el > T["wall_max"]:
                T["wall_max"] = el
            if cel > T["cpu_max"]:
                T["cpu_max"] = cel
            self._tally(s)
            if keep:
                if before != s:
                    self.cov["state_mutated_by_bot"] += 1
                self.sink.append((before, self.config))

    def _tally(self, s):
        c = self.cov
        L = s["legal_actions"]
        me = s["seat"]
        stk = s["stacks"][me]
        c["decisions"] += 1
        if s["n_alive"] == 2:
            c["heads_up"] += 1
        if s["street"] == "preflop" and s["to_call"] == 0 and me == s["bb_seat"]:
            c["bb_option"] += 1
        if s["to_call"] > 0 and s["to_call"] >= stk:
            c["call_is_allin_or_less"] += 1
        if s["to_call"] > 0 and s["bets"][me] + stk < max(s["bets"]):
            c["all_in_for_less_than_call"] += 1
        if "raise" in L and s["min_raise_to"] == s["max_raise_to"]:
            c["raise_only_allin"] += 1
        if stk + s["bets"][me] < s["bb"]:
            c["stack_below_bb"] += 1
        if any(s["all_in"][j] for j in range(len(s["all_in"])) if j != me):
            c["facing_some_allin"] += 1
        if s["ante"]:
            c["antes"] += 1
        if s["street"] != "preflop":
            c["postflop"] += 1


def _make_ours(version):
    from bot import shell as SH

    def fac(seat, config):
        return SH.make(seat, config, version=version)
    return fac


def run_live(task):
    """One block of fuzz tables (part A). Returns plain data (counters, histograms, sampled states)."""
    os.environ["NOLIMIT_BOT_STRICT"] = "1"
    from sim.kit.engine import SeatStats, Table, EngineError
    from sim.kit.payouts import get_payouts
    block, n_tables, sample_p, seed = task
    rng = random.Random(f"fuzz:{seed}:{block}")
    st_all = {v: SeatStats() for v in (0, 1, 2)}
    cov = {k: 0 for k in ("decisions", "heads_up", "bb_option", "call_is_allin_or_less", "all_in_for_less_than_call",
                          "raise_only_allin", "stack_below_bb", "facing_some_allin", "antes", "postflop",
                          "state_mutated_by_bot")}
    tim = {v: {"wall": {}, "cpu": {}, "wall_max": 0, "cpu_max": 0} for v in (0, 1, 2)}
    by_n = {}
    by_pay = {}
    sink = []
    engine_errors = []
    cfg_errors = 0
    opp_pool = [AllIn, MinRaiser, OneChipLeaver, RandomSizes, CheckCall, "chart_tag", "icm_pro", "pushfold_maniac",
                "lag_aggro", "station", "nit"]
    for t in range(n_tables):
        n = rng.choice([2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 9, 10])
        pay = rng.choice(PAYOUTS)
        if not isinstance(pay, str) and len(pay) > n and any(pay[n:]) and n >= 8:
            pay = "TOP3"
        sched = rng.choice(SCHEDULES)
        if rng.random() < 0.4:
            stacks = [2000] * n
        else:
            stacks = [rng.choice(STACKS) for _ in range(n)]
        k_ours = min(n, rng.choice([1, 1, 2, 3]))
        seats = rng.sample(range(n), k_ours)
        facs = []
        for s in range(n):
            if s in seats:
                ver = rng.choice([0, 1, 2])
                facs.append((ver, _make_ours(ver)))
            else:
                o = rng.choice(opp_pool)
                facs.append((None, _zoo(o) if isinstance(o, str) else o))
        rec_facs = []
        for s, (ver, f) in enumerate(facs):
            if ver is None:
                rec_facs.append(f)
            else:
                def wrap(seat, config, f=f, ver=ver):
                    return Recorder(f(seat, config), config, ver, sink, cov, tim, sample_p, rng)
                rec_facs.append(wrap)
        try:
            tb = Table(rec_facs, pay, sched, seed=f"fz:{seed}:{block}:{t}", stacks=stacks,
                       max_hands=rng.choice([200, 200, rng.randint(1, 200)]),
                       hands_per_level=rng.choice([1, 3, 25, 25]), time_limit_ms=None)
        except ValueError:
            cfg_errors += 1
            continue
        try:
            res = tb.play()
        except EngineError as e:
            engine_errors.append(f"{type(e).__name__}: {e}")
            continue
        for s, (ver, _f) in enumerate(facs):
            if ver is not None:
                st_all[ver].merge(res.stats[s])
                by_n[n] = by_n.get(n, 0) + res.stats[s].decisions
                pk = pay if isinstance(pay, str) else f"len{len(pay)}:{','.join(str(x) for x in pay[:3])}"
                by_pay[pk] = by_pay.get(pk, 0) + res.stats[s].decisions
    out_st = {}
    for v, st in st_all.items():
        T = tim[v]
        out_st[v] = {"decisions": st.decisions, "exceptions": st.exceptions, "illegal": st.illegal,
                     "timeouts_cpu": sum(c for b, c in T["cpu"].items() if b >= 10000),
                     "wall_over": sum(c for b, c in T["wall"].items() if b >= 10000),
                     "cpu_over": sum(c for b, c in T["cpu"].items() if b >= 10000),
                     "preempted": 0, "lat_hist": T["wall"], "cpu_hist": T["cpu"],
                     "lat_max_ns": T["wall_max"], "cpu_max_ns": T["cpu_max"], "examples": st.examples,
                     "observe_exceptions": st.observe_exceptions, "init_exceptions": st.init_exceptions}
    return {"stats": out_st, "cov": cov, "by_n": by_n, "by_pay": by_pay, "sample": sink,
            "engine_errors": engine_errors, "cfg_errors": cfg_errors}


def _validate(out, state):
    from sim.kit.agent import validate_action
    st = state
    kind, amt, reason = validate_action(out, st["legal_actions"], st["min_raise_to"], st["max_raise_to"])
    return kind is not None, reason


_OPTIONAL = ["sb_seat", "bb_seat", "contributed", "start_stacks", "history_this_hand", "street_starts", "active",
             "all_in", "alive", "payouts", "n_alive", "hands_left", "level", "hands_to_next_level", "ante", "sb",
             "pot", "button", "time_limit_ms", "n_seats"]


def mutate(state, rng):
    s = copy.deepcopy(state)
    kind = rng.randrange(6)
    if kind == 0:                                     # delete 1-4 optional fields
        for k in rng.sample(_OPTIONAL, rng.randint(1, 4)):
            s.pop(k, None)
    elif kind == 1:                                   # unknown action names in the history
        h = list(s.get("history_this_hand") or [])
        h = [(a, {"raise": "bet", "call": "CALL", "post_bb": "big_blind"}.get(b, b), c) for a, b, c in h]
        s["history_this_hand"] = h
    elif kind == 2:                                   # extra keys
        s["variant"] = "mystery"
        s["opponents"] = [{"seat": j} for j in range(len(s["stacks"]))]
    elif kind == 3:                                   # cards as upper-case / 10 notation
        s["hole"] = [c.replace("T", "10").upper() if rng.random() < 0.5 else c for c in s["hole"]]
    elif kind == 4:                                   # history as lists, not tuples
        s["history_this_hand"] = [list(e) for e in (s.get("history_this_hand") or [])]
    else:                                             # street as an int
        s["street"] = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[s["street"]]
    return s, kind


def run_replay(task):
    """Parts B and C on a chunk of sampled states: fresh instance, stale instance, mutated (production mode)."""
    mode, chunk = task
    if mode == "strict":
        os.environ["NOLIMIT_BOT_STRICT"] = "1"
    else:
        os.environ.pop("NOLIMIT_BOT_STRICT", None)
    from bot import shell as SH
    assert SH.STRICT == (mode == "strict")
    pc, tc = time.perf_counter_ns, time.thread_time_ns
    rng = random.Random(f"replay:{mode}:{len(chunk)}")
    res = {"n": 0, "exceptions": 0, "illegal": 0, "wall": [], "cpu": [], "examples": [], "masked": 0,
           "mut_kinds": {}, "mut_masked": {}, "fallbacks": 0}
    stales = {}
    for state, config in chunk:
        ver = config.get("bot_version")
        cases = []
        if mode == "strict":
            cases.append(("fresh", state, SH.make(state["seat"], config, version=ver)))
            key = (config.get("n_seats"), tuple(config.get("payouts") or ()), ver, state["seat"])
            stale = stales.get(key)
            if stale is None:
                stale = stales[key] = SH.make(state["seat"], config, version=ver)
            if stale._brain is None or stale._brain.tr.hand_no != state["hand_no"]:
                cases.append(("stale", state, stale))
        else:
            ms, kind = mutate(state, rng)
            cases.append((f"mut{kind}", ms, SH.make(state["seat"], config, version=ver)))
        for tag, st, bot in cases:
            res["n"] += 1
            e0 = bot.errors
            f0 = bot.fallbacks
            c0, t0 = tc(), pc()
            try:
                out = bot.act(st)
                exc = None
            except Exception as e:      # noqa: BLE001
                out, exc = None, e
            el, cel = pc() - t0, tc() - c0
            res["wall"].append(el)
            res["cpu"].append(cel)
            if exc is not None:
                res["exceptions"] += 1
                if len(res["examples"]) < 8:
                    res["examples"].append(f"{tag}: {type(exc).__name__}: {exc}")
                continue
            ok, why = _validate(out, st)
            if not ok:
                res["illegal"] += 1
                if len(res["examples"]) < 8:
                    res["examples"].append(f"{tag}: illegal {out!r}: {why}")
            if bot.errors > e0:
                res["masked"] += 1
                if tag.startswith("mut"):
                    res["mut_masked"][tag] = res["mut_masked"].get(tag, 0) + 1
            if bot.fallbacks > f0:
                res["fallbacks"] += 1
            if tag.startswith("mut"):
                res["mut_kinds"][tag] = res["mut_kinds"].get(tag, 0) + 1
    return res


def _q(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _hist_q(hist, q):
    tot = sum(hist.values())
    if not tot:
        return 0.0
    cum = 0
    for b in sorted(hist):
        cum += hist[b]
        if cum >= q * tot:
            return (b + 1) / 10.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", default=os.path.join(ROOT, "bot", "out", "fuzz_audit.json"))
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    blocks = a.workers * 6
    per = max(1, a.tables // blocks)
    t0 = time.time()
    load0 = os.getloadavg()
    with ctx.Pool(a.workers) as pool:
        live = pool.map(run_live, [(b, per, a.sample, a.seed) for b in range(blocks)])
    t_live = time.time() - t0
    agg = {v: {"decisions": 0, "exceptions": 0, "illegal": 0, "timeouts_cpu": 0, "wall_over": 0, "cpu_over": 0,
               "preempted": 0, "lat_hist": {}, "cpu_hist": {}, "lat_max_ns": 0, "cpu_max_ns": 0, "examples": {},
               "observe_exceptions": 0, "init_exceptions": 0} for v in (0, 1, 2)}
    cov, by_n, by_pay, sample, eng_err, cfg_err = {}, {}, {}, [], [], 0
    for r in live:
        for v, s in r["stats"].items():
            A = agg[v]
            for k in ("decisions", "exceptions", "illegal", "timeouts_cpu", "wall_over", "cpu_over", "preempted",
                      "observe_exceptions", "init_exceptions"):
                A[k] += s[k]
            for hk in ("lat_hist", "cpu_hist"):
                for b, c in s[hk].items():
                    A[hk][b] = A[hk].get(b, 0) + c
            A["lat_max_ns"] = max(A["lat_max_ns"], s["lat_max_ns"])
            A["cpu_max_ns"] = max(A["cpu_max_ns"], s["cpu_max_ns"])
            for cat, lst in s["examples"].items():
                A["examples"].setdefault(cat, []).extend(lst[:3])
        for k, c in r["cov"].items():
            cov[k] = cov.get(k, 0) + c
        for k, c in r["by_n"].items():
            by_n[k] = by_n.get(k, 0) + c
        for k, c in r["by_pay"].items():
            by_pay[k] = by_pay.get(k, 0) + c
        sample.extend(r["sample"])
        eng_err.extend(r["engine_errors"])
        cfg_err += r["cfg_errors"]
    live_out = {}
    for v, A in agg.items():
        live_out[f"v{v}"] = {k: A[k] for k in ("decisions", "exceptions", "illegal", "timeouts_cpu", "wall_over",
                                              "cpu_over", "preempted", "observe_exceptions", "init_exceptions")}
        live_out[f"v{v}"]["wall_us"] = {"p50": _hist_q(A["lat_hist"], .5), "p99": _hist_q(A["lat_hist"], .99),
                                        "p99.9": _hist_q(A["lat_hist"], .999), "max": A["lat_max_ns"] / 1000}
        live_out[f"v{v}"]["cpu_us"] = {"p50": _hist_q(A["cpu_hist"], .5), "p99": _hist_q(A["cpu_hist"], .99),
                                       "p99.9": _hist_q(A["cpu_hist"], .999), "max": A["cpu_max_ns"] / 1000}
        live_out[f"v{v}"]["cpu_over_250us"] = sum(c for b, c in A["cpu_hist"].items() if b >= 2500)
        live_out[f"v{v}"]["examples"] = {k: v2[:5] for k, v2 in A["examples"].items()}
    # parts B and C
    rng = random.Random(a.seed)
    rng.shuffle(sample)
    chunks = [sample[i::a.workers * 2] for i in range(a.workers * 2)]
    t1 = time.time()
    with ctx.Pool(a.workers) as pool:            # separate pools: STRICT is read when bot.shell is imported
        strict = pool.map(run_replay, [("strict", c) for c in chunks])
    with ctx.Pool(a.workers) as pool:
        prod = pool.map(run_replay, [("prod", c) for c in chunks])
    t_rep = time.time() - t1

    def merge(rs):
        m = {"n": 0, "exceptions": 0, "illegal": 0, "masked": 0, "fallbacks": 0, "examples": [], "mut_kinds": {},
             "mut_masked": {}}
        wall, cpu = [], []
        for r in rs:
            for k in ("n", "exceptions", "illegal", "masked", "fallbacks"):
                m[k] += r[k]
            m["examples"].extend(r["examples"][:3])
            for k, c in r["mut_kinds"].items():
                m["mut_kinds"][k] = m["mut_kinds"].get(k, 0) + c
            for k, c in r["mut_masked"].items():
                m["mut_masked"][k] = m["mut_masked"].get(k, 0) + c
            wall.extend(r["wall"])
            cpu.extend(r["cpu"])
        m["wall_us"] = {q: _q(wall, x) / 1000 for q, x in (("p50", .5), ("p99", .99), ("p99.9", .999))}
        m["wall_us"]["max"] = (max(wall) / 1000) if wall else 0
        m["cpu_us"] = {q: _q(cpu, x) / 1000 for q, x in (("p50", .5), ("p99", .99), ("p99.9", .999))}
        m["cpu_us"]["max"] = (max(cpu) / 1000) if cpu else 0
        m["cpu_over_1ms"] = sum(1 for x in cpu if x > 1_000_000)
        m["wall_over_1ms"] = sum(1 for x in wall if x > 1_000_000)
        m["examples"] = m["examples"][:10]
        return m

    out = {"meta": {"tables": a.tables, "blocks": blocks, "tables_run": blocks * per, "workers": a.workers,
                    "seed": a.seed, "live_s": t_live, "replay_s": t_rep, "loadavg_start": list(load0),
                    "loadavg_end": list(os.getloadavg()), "timing": "1 ms enforced on thread CPU time; wall recorded",
                    "python": sys.version.split()[0]},
           "live": live_out, "coverage": cov, "decisions_by_n_seats": dict(sorted(by_n.items())),
           "decisions_by_payout": by_pay, "engine_errors": eng_err[:10], "n_engine_errors": len(eng_err),
           "config_rejected": cfg_err, "replay_strict": merge(strict), "replay_mutated_prod": merge(prod)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    tot = sum(live_out[k]["decisions"] for k in live_out)
    print(f"live decisions {tot} in {t_live:.0f} s; replays strict {out['replay_strict']['n']} "
          f"prod {out['replay_mutated_prod']['n']} in {t_rep:.0f} s; engine errors {len(eng_err)}")
    for k, v in live_out.items():
        print(k, {x: v[x] for x in ("decisions", "exceptions", "illegal", "timeouts_cpu", "cpu_over", "wall_over",
                                     "preempted", "cpu_over_250us")}, "cpu", v["cpu_us"], "wall", v["wall_us"])
        if v["examples"]:
            print("   examples", v["examples"])
    print("coverage", cov)
    for k in ("replay_strict", "replay_mutated_prod"):
        r = out[k]
        print(k, {x: r[x] for x in ("n", "exceptions", "illegal", "masked", "fallbacks", "cpu_over_1ms",
                                     "wall_over_1ms")}, "cpu", r["cpu_us"], "wall", r["wall_us"])
        print("   examples", r["examples"][:6], r.get("mut_masked"))


if __name__ == "__main__":
    main()
