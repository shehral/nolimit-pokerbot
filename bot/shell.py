"""Shell of the rehearsal bot: the kit-facing Agent class. It never raises and never returns an illegal action.

    act(state):  t0 = own clock -> collector paused (GC_GUARD_ACT) -> adapter.normalize -> brain.decide (optional
                 work stops at the guard) -> on an exception, brain.decide_fallback (the v0 decision, P15) ->
                 adapter.to_action (clamped) -> adapter.is_legal check -> collector resumed -> return.
                 Anything that still fails returns check-if-legal-else-fold (the rules' own fallback).
    observe(e):  feeds the brain's hand tracker (untimed in the mock kit) and, at hand_start, runs a collection
                 that is about to come due (GC_HYGIENE); exceptions are swallowed.
    import time: everything is wrapped; a failed import leaves a bot that check/folds instead of no bot. A warm-up
                 runs synthetic decisions through a THROWAWAY instance, then gc.collect() and gc.freeze() (GC stays
                 on: the platform's process model is unknown, so gc.disable() is not the default).
    Settings:    brain.PARAMS (GC_*, WARMUP_DECISIONS, ALLIN_SNAP, FALLBACK_V0); _SHELL_DEFAULTS below is only the
                 copy used when brain.py itself fails to import.

Kit specs:  "bot/shell.py:Bot" (PARAMS["VERSION"], default 2), "bot/shell.py:BotV0", ":BotV1", ":BotV2".
Version can also come from config["bot_version"] or the environment variable NOLIMIT_BOT_VERSION.
"""
import gc
import os
import sys
import time

# Fallback copy of the shell settings, used ONLY if brain.py fails to import (the live values are brain.PARAMS;
# see the comments there). The GC settings were chosen from latency measurements on 2026-11-03.
_SHELL_DEFAULTS = {"GC_GUARD_ACT": True, "GC_HYGIENE": True, "GC_FULL_MARGIN": 2, "GC_GEN1_MARGIN": 1,
                   "GC_MODE": "freeze", "WARMUP_DECISIONS": 300, "ALLIN_SNAP": 0.85, "FALLBACK_V0": True}

_IMPORT_ERRORS = []
# Test-only: NOLIMIT_BOT_STRICT=1 lets exceptions and illegal results through to the engine, so the kit's own
# counters show every error the shell would otherwise hide. Never set it for a submission.
STRICT = os.environ.get("NOLIMIT_BOT_STRICT") == "1"


def _load_sibling(name):
    """Import a sibling module of this file: as a package member if possible, else by file path under a unique
    name (so other modules with the same name on sys.path can never shadow it)."""
    pkg = __package__
    if pkg:
        try:
            import importlib
            return importlib.import_module(f"{pkg}.{name}")
        except Exception:
            pass
    import importlib.util
    modname = "nolimit_bot_" + name
    mod = sys.modules.get(modname)
    if mod is not None:
        return mod
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[modname]
        raise
    return mod


try:
    adapter = _load_sibling("adapter")
except BaseException as _e:      # noqa: BLE001 - an import failure must never kill the bot
    adapter = None
    _IMPORT_ERRORS.append(f"adapter: {type(_e).__name__}: {_e}")
try:
    brain = _load_sibling("brain")
except BaseException as _e:      # noqa: BLE001
    brain = None
    _IMPORT_ERRORS.append(f"brain: {type(_e).__name__}: {_e}")

SHELL_PARAMS = dict(_SHELL_DEFAULTS)
try:
    if brain is not None:
        for _k in _SHELL_DEFAULTS:
            if _k in brain.PARAMS:
                SHELL_PARAMS[_k] = brain.PARAMS[_k]
except BaseException:            # noqa: BLE001
    pass

_perf = time.perf_counter_ns
_GC_HYGIENE = bool(SHELL_PARAMS.get("GC_HYGIENE"))
_GC_GUARD = bool(SHELL_PARAMS.get("GC_GUARD_ACT"))
try:
    _GC_T = gc.get_threshold()
except BaseException:            # noqa: BLE001
    _GC_T = (2000, 10, 10)
GC_STATS = {"full": 0, "gen1": 0}


def _gc_due():
    """Collections that would come due during the next hand run now, between hands (CPython's generational GC:
    a full collection fires once gen-2's count passes its threshold; it costs about 1.5 ms of CPU in a kit worker,
    over the 1 ms limit if it lands inside act())."""
    if not gc.isenabled():
        return
    c0, c1, c2 = gc.get_count()
    if c2 >= _GC_T[2] - SHELL_PARAMS["GC_FULL_MARGIN"]:
        gc.collect()
        GC_STATS["full"] += 1
    elif c1 >= _GC_T[1] - SHELL_PARAMS["GC_GEN1_MARGIN"]:
        gc.collect(1)
        GC_STATS["gen1"] += 1


def _safe(state):
    try:
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
    except BaseException:        # noqa: BLE001
        return {"action": "fold"}


def _env_version():
    v = os.environ.get("NOLIMIT_BOT_VERSION")
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


class Bot:
    """Kit Agent. One instance per table (all state lives on the instance and resets with it)."""
    VERSION = None    # None = config["bot_version"], else NOLIMIT_BOT_VERSION, else brain.PARAMS["VERSION"]

    def __init__(self, seat, config):
        self.seat = seat
        self.config = config
        self.errors = 0           # exceptions caught in act (diagnostic; the kit counts its own)
        self.fallbacks = 0        # ... of which recovered by the v0 decision (P15)
        self.illegal_fixed = 0    # results that failed our own legality check and were replaced by the fallback
        self._snap = SHELL_PARAMS["ALLIN_SNAP"]
        self._fallback = bool(SHELL_PARAMS["FALLBACK_V0"])
        self._brain = None
        try:
            cfg = config if isinstance(config, dict) else {}
            ver = self.VERSION
            if ver is None:
                ver = cfg.get("bot_version")
            if ver is None:
                ver = _env_version()
            if brain is not None and adapter is not None:
                self._brain = brain.Brain(seat, cfg, version=ver, params=cfg.get("bot_params"))
                self._snap = self._brain.P["ALLIN_SNAP"]          # per instance, so A/B overrides apply
                self._fallback = bool(self._brain.P["FALLBACK_V0"])
        except BaseException:     # noqa: BLE001
            self._brain = None
            if STRICT:
                raise

    def act(self, state):
        t0 = _perf()
        g = _GC_GUARD and gc.isenabled()
        if g:
            gc.disable()
        counted = False
        try:
            b = self._brain
            if b is None:
                if STRICT:
                    raise RuntimeError("bot failed to import or construct: " + "; ".join(_IMPORT_ERRORS))
                return _safe(state)
            view = adapter.normalize(state)
            try:
                dec = b.decide(view, t0)
            except Exception:     # noqa: BLE001 - P15: keep the mandatory (v0) decision when optional work fails
                if STRICT or not self._fallback:
                    raise
                self.errors += 1
                counted = True
                dec = b.decide_fallback(view, t0)       # if this raises too, the handler below check/folds
                self.fallbacks += 1
            out = adapter.to_action(dec, state, self._snap)
            if adapter.is_legal(out, state) or STRICT:
                return out
            self.illegal_fixed += 1
            return _safe(state)
        except BaseException:     # noqa: BLE001 - rules s3.7: an exception would auto-fold us anyway
            if not counted:
                self.errors += 1
            if STRICT:
                raise
            return _safe(state)
        finally:
            if g:
                gc.enable()

    def observe(self, event):
        try:
            if _GC_HYGIENE and event.get("type") == "hand_start":
                _gc_due()
            b = self._brain
            if b is not None:
                b.observe(event)
        except BaseException:     # noqa: BLE001
            if STRICT:
                raise


class BotV0(Bot):
    VERSION = 0


class BotV1(Bot):
    VERSION = 1


class BotV2(Bot):
    VERSION = 2


Agent = Bot          # the runner's default attribute for "bot/shell.py"


def make(seat, config, version=None, params=None):
    """Factory with overrides, for A/B runs: functools.partial(bot.shell.make, version=1, params={"CALL_LAM": 2.0}).
    Picklable for worker pools when this module is imported as the package `bot.shell`."""
    cfg = dict(config) if isinstance(config, dict) else {}
    if version is not None:
        cfg["bot_version"] = version
    if params:
        cfg["bot_params"] = dict(params)
    return Bot(seat, cfg)


# ----------------------------------------------------------------------------- import-time warm-up
def _synthetic_states(rng, count):
    """Kit-shaped states covering preflop (unopened, facing an open, facing a jam), postflop (free and facing a
    bet) on random stacks. Only for warming code paths; consistency is approximate and errors are harmless."""
    deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
    out = []
    for t in range(count):
        n = 8
        cards = rng.sample(deck, 7)
        stacks = [rng.randint(200, 6000) for _ in range(n)]
        bb = rng.choice((20, 50, 100, 200, 400))
        sb = bb // 2
        btn = rng.randrange(n)
        sbs, bbs = (btn + 1) % n, (btn + 2) % n
        me = rng.randrange(n)
        kind = t % 5
        street = "preflop" if kind < 3 else rng.choice(("flop", "turn", "river"))
        board = [] if street == "preflop" else cards[2:2 + {"flop": 3, "turn": 4, "river": 5}[street]]
        bets = [0] * n
        hist = [(sbs, "post_sb", sb), (bbs, "post_bb", bb)]
        if street == "preflop":
            bets[sbs], bets[bbs] = sb, bb
        if kind == 1:
            r = (bbs + 1) % n
            if r != me:
                bets[r] = int(2.5 * bb)
                hist.append((r, "raise", bets[r]))
        elif kind == 2:
            r = (bbs + 1) % n
            if r != me:
                bets[r] = stacks[r]
                hist.append((r, "raise", bets[r]))
        elif kind == 4:
            r = (me + 1) % n
            bets[r] = bb * 3
        cur = max(bets) if street != "preflop" else max(max(bets), bb)
        to_call = max(0, min(cur - bets[me], stacks[me]))
        legal = (["fold", "call", "raise"] if to_call > 0 else ["check", "raise"])
        allin_flags = [bets[j] >= stacks[j] for j in range(n)]
        live_stacks = [max(0, stacks[j] - bets[j]) for j in range(n)]
        st = {
            "hand_no": t + 1, "level": 1, "hands_to_next_level": 10, "hands_left": 100, "sb": sb, "bb": bb,
            "ante": 0, "button": btn, "sb_seat": sbs, "bb_seat": bbs, "seat": me, "street": street,
            "hole": cards[:2], "board": board, "pot": sum(bets) + 3 * bb, "stacks": live_stacks, "bets": bets,
            "contributed": [bets[j] for j in range(n)], "active": [True] * n, "all_in": allin_flags,
            "to_call": to_call, "min_raise_to": min(cur + bb, bets[me] + live_stacks[me]),
            "max_raise_to": bets[me] + live_stacks[me], "legal_actions": legal, "history_this_hand": hist,
            "street_starts": [0] if street == "preflop" else [0, len(hist)], "n_alive": n,
            "alive": [True] * n, "start_stacks": stacks, "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "n_seats": n,
            "time_limit_ms": 1.0,
        }
        out.append(st)
    return out


def _warm_up():
    n = SHELL_PARAMS["WARMUP_DECISIONS"]
    if n <= 0 or brain is None or adapter is None:
        return
    import random as _random
    rng = _random.Random(20261106)
    states = _synthetic_states(rng, n)
    cfg = {"seat": 0, "n_seats": 8, "start_stack": 2000, "payouts": [50, 30, 20, 0, 0, 0, 0, 0],
           "blind_schedule": [[10, 20, 0]], "hands_per_level": 25, "max_hands": 200, "time_limit_ms": 1.0, "seed": 1}
    for ver in (0, 1, 2):
        throwaway = Bot(0, dict(cfg, bot_version=ver))
        for st in states:
            throwaway.seat = st["seat"]
            if throwaway._brain is not None:
                throwaway._brain.seat = st["seat"]
            throwaway.act(st)


WARMUP_INFO = {}
try:
    _w0 = _perf()
    _warm_up()
    WARMUP_INFO["warmup_ms"] = (_perf() - _w0) / 1e6
except BaseException as _e:      # noqa: BLE001
    WARMUP_INFO["warmup_error"] = f"{type(_e).__name__}: {_e}"
try:
    _mode = SHELL_PARAMS["GC_MODE"]
    if _mode == "freeze":
        gc.collect()
        gc.freeze()
    elif _mode == "disable":
        gc.collect()
        gc.freeze()
        gc.disable()
except BaseException:            # noqa: BLE001
    pass
