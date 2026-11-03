"""8-max no-limit hold'em sit-and-go engine of the mock starter kit (pure standard library).

Format (rules s4 + SPEC section 1): every seat starts with 2,000 chips; blinds from a schedule, up one level every
25 hands; the table ends when one player holds all the chips or after 200 hands. Eliminated seats get their
finishing place's points (players busted in the same hand are ordered by their stack at the start of that hand,
larger = better place; exactly equal stacks split the points of the places they share). At the cap the remaining
places' points go to the survivors by exact ICM (icm.py).

Strictness (rules s3.7): act() is timed with time.perf_counter_ns around the call. An exception, an illegal action
(agent.validate_action), or a decision slower than the time limit (default 1.0 ms wall clock) is an ERROR: it is
counted and the engine plays FOLD, or CHECK when checking is legal. The engine never repairs an action.
Each decision's thread CPU time is recorded too; timing="cpu" enforces the limit on CPU time instead, for benchmark
runs where a busy laptop's preemption (wall over, CPU under) would otherwise count as bot errors.

Betting rules:
- Raise amounts are raise-to totals for the street. Minimum raise = current bet + the last FULL raise increment
  (the big blind when nothing was raised; also the minimum opening bet postflop). An all-in below a full raise is
  allowed (min_raise_to = max_raise_to = the all-in) but does not reopen the betting: a player who already acted
  may raise again only if the bet has risen by at least one full raise increment since they last acted
  (cumulative short all-ins can add up to a full raise, as in TDA rules).
- Preflop the amount to match is the nominal big blind even when the big blind is all-in for less.
- Fold is legal only when facing a bet; check only when not; raise is not offered when nobody else could respond.
- When the bet to call exceeds the stack, "call" puts in the whole stack (all-in for less); to_call is capped at
  the stack.
- Heads-up: the button posts the small blind and acts first preflop, last postflop. The button moves to the next
  seat that still has chips each hand (moving button; no dead-button rule).
- Uncalled bets are returned at the end of each betting round (refund event, a refund line in the hand log).
  Side pots by contribution layers; split pots share equally and odd chips go one each to the winners closest to
  the left of the button.
- Optional antes (third element of a schedule level) are dead money: they count in the pot, not toward calls.
  Antes that no live player can match (a folded player's ante above every contender's total) go back to their
  owner like an uncalled bet (refund event at the end of the hand).

Dealing is position-based so PAIRED comparisons line up: in hand h, seat i gets deck[2i], deck[2i+1] and the board is
deck[2n .. 2n+4], where the deck for hand h depends only on the table seed and h (never on bot actions). Bots never
see the deck RNG: each agent gets its own config["seed"], and the global random module is reseeded per table.
"""
import random
import time

try:
    from .agent import validate_action
    from .cards import CARD_STR, evaluate
    from .icm import icm_equities
    from .payouts import get_payouts
    from .schedules import HANDS_PER_LEVEL, get_schedule
except ImportError:   # flat import (sim/kit on sys.path)
    from agent import validate_action
    from cards import CARD_STR, evaluate
    from icm import icm_equities
    from payouts import get_payouts
    from schedules import HANDS_PER_LEVEL, get_schedule

STREETS = ("preflop", "flop", "turn", "river")
_FCR = ("fold", "call", "raise")
_FC = ("fold", "call")
_XR = ("check", "raise")
_X = ("check",)
_MAX_EXAMPLES = 5
_REPR_LIMIT = 160


class EngineError(AssertionError):
    """An engine invariant failed (chip conservation, pot layers, points sum). A kit bug, never a bot's fault."""


def _trunc(s, limit=_REPR_LIMIT):
    s = str(s)
    return s if len(s) <= limit else s[:limit] + "..."


def _short(x, limit=_REPR_LIMIT):
    try:
        s = repr(x)
    except Exception as e:   # a bot can return objects whose repr raises
        s = f"<unreprable {type(x).__name__}: {type(e).__name__}>"
    return s if len(s) <= limit else s[:limit] + "..."


class SeatStats:
    """Counters for one entrant (one seat of one table, or merged over many tables by the runner).

    Timing: every act() is measured twice, wall clock (time.perf_counter_ns, what the rules limit) and the
    thread's CPU time (time.thread_time_ns). wall_over / cpu_over count decisions over the limit on each clock
    whatever the enforcement mode; `preempted` = over on the wall clock but not on CPU time, i.e. the delay came
    from the OS or other processes (laptop noise), not from the bot's own work."""
    __slots__ = ("tables", "decisions", "timeouts", "exceptions", "illegal", "lat_hist", "lat_sum_ns", "lat_max_ns",
                 "cpu_hist", "cpu_max_ns", "wall_over", "cpu_over", "preempted",
                 "observe_calls", "observe_exceptions", "observe_ns", "observe_max_ns", "inits", "init_exceptions",
                 "init_ns_sum", "init_ns_max", "n_fold", "n_check", "n_call", "n_raise", "n_allin", "hands", "vpip",
                 "pfr", "examples")

    def __init__(self):
        self.tables = 0
        self.decisions = self.timeouts = self.exceptions = self.illegal = 0
        self.lat_hist = {}            # wall clock: 0.1-microsecond bin -> count (bin = elapsed_ns // 100)
        self.lat_sum_ns = self.lat_max_ns = 0
        self.cpu_hist = {}            # thread CPU time, same bins
        self.cpu_max_ns = 0
        self.wall_over = self.cpu_over = self.preempted = 0
        self.observe_calls = self.observe_exceptions = self.observe_ns = self.observe_max_ns = 0
        self.inits = self.init_exceptions = self.init_ns_sum = self.init_ns_max = 0
        self.n_fold = self.n_check = self.n_call = self.n_raise = self.n_allin = 0
        self.hands = self.vpip = self.pfr = 0
        self.examples = {}            # "exception" / "timeout" / "illegal" / "observe" / "init" -> [str, ...]

    def add_example(self, cat, msg):
        lst = self.examples.setdefault(cat, [])
        if len(lst) < _MAX_EXAMPLES:
            lst.append(msg)

    def merge(self, o):
        self.tables += o.tables
        self.decisions += o.decisions
        self.timeouts += o.timeouts
        self.exceptions += o.exceptions
        self.illegal += o.illegal
        h = self.lat_hist
        for b, c in o.lat_hist.items():
            h[b] = h.get(b, 0) + c
        h = self.cpu_hist
        for b, c in o.cpu_hist.items():
            h[b] = h.get(b, 0) + c
        self.lat_sum_ns += o.lat_sum_ns
        if o.lat_max_ns > self.lat_max_ns:
            self.lat_max_ns = o.lat_max_ns
        if o.cpu_max_ns > self.cpu_max_ns:
            self.cpu_max_ns = o.cpu_max_ns
        self.wall_over += o.wall_over
        self.cpu_over += o.cpu_over
        self.preempted += o.preempted
        self.observe_calls += o.observe_calls
        self.observe_exceptions += o.observe_exceptions
        self.observe_ns += o.observe_ns
        if o.observe_max_ns > self.observe_max_ns:
            self.observe_max_ns = o.observe_max_ns
        self.inits += o.inits
        self.init_exceptions += o.init_exceptions
        self.init_ns_sum += o.init_ns_sum
        if o.init_ns_max > self.init_ns_max:
            self.init_ns_max = o.init_ns_max
        self.n_fold += o.n_fold
        self.n_check += o.n_check
        self.n_call += o.n_call
        self.n_raise += o.n_raise
        self.n_allin += o.n_allin
        self.hands += o.hands
        self.vpip += o.vpip
        self.pfr += o.pfr
        for cat, lst in o.examples.items():
            for m in lst:
                self.add_example(cat, m)
        return self

    @staticmethod
    def _quantile(hist, q, fallback_ns):
        total = sum(hist.values())
        if total == 0:
            return 0.0
        rank = q * total
        cum = 0
        for b in sorted(hist):
            cum += hist[b]
            if cum >= rank:
                return (b + 1) / 10.0
        return fallback_ns / 1000.0

    COARSE_EDGES_US = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

    @classmethod
    def coarse(cls, hist):
        """Counts per coarse bucket: '<1us', '1-2us', ..., '>=5000us' (for charts)."""
        edges = cls.COARSE_EDGES_US
        out = {}
        labels = [f"<{edges[0]}us"] + [f"{a}-{b}us" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}us"]
        for lab in labels:
            out[lab] = 0
        for b, c in hist.items():
            us = b / 10.0
            k = 0
            while k < len(edges) and us >= edges[k]:
                k += 1
            out[labels[k]] += c
        return out

    def latency_us(self, q):
        """q-quantile of act() wall-clock latency in microseconds (upper edge of its 0.1 us bin)."""
        return self._quantile(self.lat_hist, q, self.lat_max_ns)

    def cpu_us(self, q):
        """q-quantile of act() thread CPU time in microseconds."""
        return self._quantile(self.cpu_hist, q, self.cpu_max_ns)

    def summary(self):
        d = self.decisions
        timed = sum(self.lat_hist.values())
        return {
            "tables": self.tables,
            "decisions": d,
            "timeouts": self.timeouts,
            "exceptions": self.exceptions,
            "illegal": self.illegal,
            "errors": self.timeouts + self.exceptions + self.illegal,
            "error_rate": (self.timeouts + self.exceptions + self.illegal) / d if d else 0.0,
            "latency_us": {
                "p50": self.latency_us(0.50),
                "p99": self.latency_us(0.99),
                "p99.9": self.latency_us(0.999),
                "max": self.lat_max_ns / 1000.0,
                "mean": (self.lat_sum_ns / timed / 1000.0) if timed else 0.0,
            },
            "cpu_us": {"p50": self.cpu_us(0.50), "p99": self.cpu_us(0.99), "p99.9": self.cpu_us(0.999),
                       "max": self.cpu_max_ns / 1000.0},
            "over_limit": {"wall": self.wall_over, "cpu": self.cpu_over, "preempted": self.preempted},
            "latency_hist_us": self.coarse(self.lat_hist),
            "cpu_hist_us": self.coarse(self.cpu_hist),
            "observe": {"calls": self.observe_calls, "exceptions": self.observe_exceptions,
                        "total_ms": self.observe_ns / 1e6, "max_us": self.observe_max_ns / 1000.0},
            "init": {"count": self.inits, "exceptions": self.init_exceptions,
                     "mean_ms": (self.init_ns_sum / self.inits / 1e6) if self.inits else 0.0,
                     "max_ms": self.init_ns_max / 1e6},
            "actions": {"fold": self.n_fold, "check": self.n_check, "call": self.n_call, "raise": self.n_raise,
                        "all_in": self.n_allin},
            "hands": self.hands,
            "vpip": self.vpip / self.hands if self.hands else 0.0,
            "pfr": self.pfr / self.hands if self.hands else 0.0,
            "examples": {k: list(v) for k, v in self.examples.items()},
        }


class TableResult:
    """Outcome of one table. Lists are indexed by seat."""
    __slots__ = ("points", "places", "bust_hand", "hands_played", "survivors", "capped", "final_stacks", "stats",
                 "hand_log", "events")

    def __init__(self, points, places, bust_hand, hands_played, survivors, capped, final_stacks, stats, hand_log,
                 events):
        self.points = points
        self.places = places
        self.bust_hand = bust_hand
        self.hands_played = hands_played
        self.survivors = survivors
        self.capped = capped
        self.final_stacks = final_stacks
        self.stats = stats
        self.hand_log = hand_log
        self.events = events


def make_deck(n_seats, holes=None, board=None, fill_seed=0):
    """A fixed deck layout for tests: holes = {seat: ("Ah", "Kd")}, board = ["2c", ...] (up to 5 cards).
    Unspecified positions get the remaining cards in a seeded random order. Returns a list of card ints."""
    from_str = {s: i for i, s in enumerate(CARD_STR)}
    need = 2 * n_seats + 5
    deck = [None] * need
    used = set()
    for seat, (a, b) in (holes or {}).items():
        deck[2 * seat], deck[2 * seat + 1] = from_str[a], from_str[b]
    for k, c in enumerate(board or []):
        deck[2 * n_seats + k] = from_str[c]
    for c in deck:
        if c is not None:
            if c in used:
                raise ValueError(f"duplicate card {CARD_STR[c]} in rigged deck")
            used.add(c)
    rest = [c for c in range(52) if c not in used]
    random.Random(fill_seed).shuffle(rest)
    it = iter(rest)
    return [c if c is not None else next(it) for c in deck]


class Table:
    """One sit-and-go table. factories[i](seat, config) -> agent for seat i (a NEW agent per table)."""

    def __init__(self, factories, payouts="TOP3", schedule="TURBO", seed=0, *, start_stack=2000, stacks=None,
                 max_hands=200, hands_per_level=HANDS_PER_LEVEL, time_limit_ms=1.0, time_scale=1.0, timing="wall",
                 button=None, deck_fn=None, seed_global_random=True, record_hands=False, record_events=False,
                 extra_config=None):
        n = len(factories)
        if not 2 <= n <= 10:
            raise ValueError(f"need 2..10 seats, got {n}")
        self.n = n
        self.factories = list(factories)
        self.pays = get_payouts(payouts, n, truncate=n < 8)   # tables under 8 seats drop places beyond n
        self.pays_list = list(self.pays)
        self.schedule = get_schedule(schedule)
        self.max_hands = int(max_hands)
        self.hands_per_level = int(hands_per_level)
        self.start_stack = int(start_stack)
        self.stacks = [int(s) for s in stacks] if stacks is not None else [self.start_stack] * n
        if len(self.stacks) != n or any(s < 0 for s in self.stacks) or sum(1 for s in self.stacks if s > 0) < 2:
            raise ValueError(f"bad starting stacks {self.stacks!r}")
        self.total_chips = sum(self.stacks)
        self.time_limit_ms = time_limit_ms
        self.limit_ns = float("inf") if time_limit_ms is None else float(time_limit_ms) * 1e6
        self.time_scale = float(time_scale)
        if timing not in ("wall", "cpu"):
            raise ValueError(f"timing must be 'wall' (rules: wall clock) or 'cpu' (thread CPU time), not {timing!r}")
        self.timing = timing
        trng = random.Random(f"kit-table:{seed}")
        first_button = trng.randrange(n)
        self.agent_seeds = [trng.getrandbits(63) for _ in range(n)]
        deck_seed = trng.getrandbits(64)
        self.global_seed = trng.getrandbits(64) if seed_global_random else None
        self.first_button = first_button if button is None else int(button) % n
        self.deck_rng = random.Random(deck_seed)
        self.deck_fn = deck_fn if deck_fn is not None else self._random_deck
        self.record_hands = record_hands
        self.record_events = record_events
        self.extra_config = dict(extra_config or {})
        self.button = None

    # ------------------------------------------------------------------ helpers
    def _random_deck(self, hand_no):
        """Partial Fisher-Yates: exactly 2n+5 draws per hand, so hand h's cards do not depend on earlier play."""
        need = 2 * self.n + 5
        d = list(range(52))
        rnd = self.deck_rng.random
        for k in range(need):
            j = k + int(rnd() * (52 - k))
            d[k], d[j] = d[j], d[k]
        return d[:need]

    def _config(self, seat):
        cfg = {
            "seat": seat,
            "n_seats": self.n,
            "start_stack": self.start_stack,
            "start_stacks": list(self.stacks),
            "payouts": list(self.pays),
            "blind_schedule": [list(lv) for lv in self.schedule],
            "hands_per_level": self.hands_per_level,
            "max_hands": self.max_hands,
            "time_limit_ms": self.time_limit_ms,
            "seed": self.agent_seeds[seat],
        }
        for k, v in self.extra_config.items():
            cfg[k] = v
        return cfg

    def _emit(self, ev, only=None):
        """Deliver an event (a fresh shallow copy per agent; values are immutable) to observing agents."""
        if self.events is not None:
            self.events.append((only, ev))
        pc = time.perf_counter_ns
        for seat, fn, st in self.observers:
            if only is not None and seat != only:
                continue
            st.observe_calls += 1
            t0 = pc()
            try:
                fn(dict(ev))
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                st.observe_exceptions += 1
                st.add_example("observe", f"{type(e).__name__}: {_trunc(e)} (event {ev.get('type')})")
            dt = pc() - t0
            st.observe_ns += dt
            if dt > st.observe_max_ns:
                st.observe_max_ns = dt

    # ------------------------------------------------------------------ table
    def play(self):
        n = self.n
        pc = time.perf_counter_ns
        self.stats = stats = [SeatStats() for _ in range(n)]
        self.events = [] if self.record_events else None
        if self.global_seed is not None:
            random.seed(self.global_seed)
        acts = [None] * n
        observers = []
        for seat in range(n):
            st = stats[seat]
            st.tables = 1
            st.inits = 1
            cfg = self._config(seat)
            t0 = pc()
            try:
                agent = self.factories[seat](seat, cfg)
                act = agent.act
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                agent = act = None
                st.init_exceptions = 1
                st.add_example("init", f"{type(e).__name__}: {_trunc(e)}")
            dt = pc() - t0
            st.init_ns_sum = st.init_ns_max = dt
            acts[seat] = act
            if agent is not None:
                obs = getattr(agent, "observe", None)
                if callable(obs) and not getattr(obs, "kit_default", False):
                    observers.append((seat, obs, st))
        self.acts = acts
        self.observers = observers
        self.points = [0.0] * n
        self.places = [None] * n
        self.bust_hand = [None] * n
        self.hand_log = [] if self.record_hands else None
        stacks = self.stacks
        hand_no = 0
        while True:
            alive = 0
            for s in stacks:
                if s > 0:
                    alive += 1
            if alive <= 1 or hand_no >= self.max_hands:
                break
            hand_no += 1
            self._play_hand(hand_no)
        return self._finish(hand_no)

    def _finish(self, hand_no):
        n = self.n
        stacks = self.stacks
        pays = self.pays
        points, places = self.points, self.places
        survivors = [i for i in range(n) if stacks[i] > 0]
        k = len(survivors)
        if k == 1:
            points[survivors[0]] = pays[0]
            places[survivors[0]] = 1
        else:
            eq = icm_equities([stacks[i] for i in survivors], pays[:k])
            for j, i in enumerate(survivors):
                points[i] = eq[j]
                bigger = sum(1 for x in survivors if stacks[x] > stacks[i])
                equal = sum(1 for x in survivors if stacks[x] == stacks[i])
                places[i] = bigger + (equal + 1) / 2.0
                if places[i] == int(places[i]):
                    places[i] = int(places[i])
        total = sum(points)
        if abs(total - sum(pays)) > 1e-6 * max(1.0, sum(pays)):
            raise EngineError(f"points sum {total} != payouts sum {sum(pays)}")
        if self.observers or self.events is not None:
            self._emit({"type": "table_end", "hand_no": hand_no, "points": tuple(points), "places": tuple(places),
                        "stacks": tuple(stacks)})
        return TableResult(list(points), list(places), list(self.bust_hand), hand_no, k, k > 1, list(stacks),
                           self.stats, self.hand_log, self.events)

    # ------------------------------------------------------------------ hand
    def _play_hand(self, hand_no):
        n = self.n
        stacks = self.stacks
        stats = self.stats
        acts = self.acts
        pays = self.pays
        pays_list = self.pays_list
        emitting = bool(self.observers) or self.events is not None
        emit = self._emit
        pc = time.perf_counter_ns
        tc = time.thread_time_ns
        limit_ns = self.limit_ns
        scale = self.time_scale
        cpu_mode = self.timing == "cpu"
        sched = self.schedule
        hpl = self.hands_per_level
        lv = (hand_no - 1) // hpl
        if lv >= len(sched):
            lv = len(sched) - 1
        sb, bb, ante = sched[lv]
        if lv == len(sched) - 1:
            hands_to_next = self.max_hands - hand_no + 1
        else:
            hands_to_next = (lv + 1) * hpl + 1 - hand_no
        level_no = lv + 1
        hands_left = self.max_hands - hand_no
        tlim = self.time_limit_ms

        # button and positions
        if self.button is None:
            b = self.first_button
        else:
            b = (self.button + 1) % n
        while stacks[b] <= 0:
            b = (b + 1) % n
        self.button = b
        order = []
        for d in range(1, n + 1):
            k = (b + d) % n
            if stacks[k] > 0:
                order.append(k)
        n_alive = len(order)
        hu = n_alive == 2
        if hu:
            sb_seat, bb_seat = b, order[0]
            pre_seq = [b, order[0]]
        else:
            sb_seat, bb_seat = order[0], order[1]
            pre_seq = order[2:] + order[:2]
        start_stacks = stacks[:]
        alive_list = [s > 0 for s in start_stacks]

        deck = self.deck_fn(hand_no)
        hole_s = [None] * n
        hole = [None] * n
        for i in order:
            c1, c2 = deck[2 * i], deck[2 * i + 1]
            hole[i] = (c1, c2)
            hole_s[i] = (CARD_STR[c1], CARD_STR[c2])
        bd = deck[2 * n: 2 * n + 5]

        bets = [0] * n
        contrib = [0] * n
        folded = [True] * n
        allin = [False] * n
        for i in order:
            folded[i] = False
        active = [not f for f in folded]
        history = []
        street_starts = [0]
        pot = 0
        if ante > 0:
            for i in order:
                a = ante if ante < stacks[i] else stacks[i]
                stacks[i] -= a
                contrib[i] += a
                pot += a
                if stacks[i] == 0:
                    allin[i] = True
                history.append((i, "ante", a))
        for seat, amt, tag in ((sb_seat, sb, "post_sb"), (bb_seat, bb, "post_bb")):
            a = amt if amt < stacks[seat] else stacks[seat]
            stacks[seat] -= a
            bets[seat] += a
            contrib[seat] += a
            pot += a
            if stacks[seat] == 0:
                allin[seat] = True
            history.append((seat, tag, bets[seat]))
        if emitting:
            emit({"type": "hand_start", "hand_no": hand_no, "level": level_no, "sb": sb, "bb": bb, "ante": ante,
                  "button": b, "sb_seat": sb_seat, "bb_seat": bb_seat, "stacks": tuple(start_stacks),
                  "alive": tuple(alive_list), "posts": tuple(history)})
            for i in order:
                emit({"type": "deal", "hand_no": hand_no, "seat": i, "hole": hole_s[i]}, only=i)
        vpip = [False] * n
        pfr = [False] * n
        n_in = n_alive
        board_s = []
        cur = bb
        min_inc = bb
        for street in range(4):
            if street:
                for i in order:
                    bets[i] = 0
                cur = 0
                min_inc = bb
                board_s = [CARD_STR[c] for c in bd[:street + 2]]
                street_starts.append(len(history))
                if emitting:
                    emit({"type": "street", "hand_no": hand_no, "street": STREETS[street], "board": tuple(board_s),
                          "pot": pot})
                can = 0
                for i in order:
                    if not folded[i] and not allin[i]:
                        can += 1
                if can < 2:
                    continue
                seq = order
            else:
                seq = pre_seq
            street_name = STREETS[street]
            can_act = 0                      # players in the hand who still have chips behind
            for i in seq:
                if not folded[i] and not allin[i]:
                    can_act += 1
            acted = [False] * n
            acted_level = [0] * n
            nseq = len(seq)
            p = 0
            idle = 0
            while idle < nseq:
                i = seq[p]
                p += 1
                if p == nseq:
                    p = 0
                if folded[i] or allin[i] or (acted[i] and bets[i] >= cur):
                    idle += 1
                    continue
                idle = 0
                stk = stacks[i]
                need = cur - bets[i]
                to_call = need if need < stk else stk
                others = can_act > 1             # i itself is pending, so it is one of the can_act players
                if to_call <= 0 and not others:
                    acted[i] = True       # nothing to decide: nobody left who could respond to a bet
                    acted_level[i] = cur
                    continue
                max_to = bets[i] + stk
                can_raise = others and max_to > cur and (not acted[i] or cur - acted_level[i] >= min_inc)
                if to_call > 0:
                    legal = _FCR if can_raise else _FC
                else:
                    legal = _XR if can_raise else _X
                if can_raise:
                    min_to = cur + min_inc
                    if min_to > max_to:
                        min_to = max_to
                    mx = max_to
                else:
                    min_to = mx = 0
                st = stats[i]
                st.decisions += 1
                kind = None
                actf = acts[i]
                if actf is None:
                    st.exceptions += 1
                    st.add_example("exception", "agent failed to construct; every decision errors")
                else:
                    state = {
                        "hand_no": hand_no, "level": level_no, "hands_to_next_level": hands_to_next,
                        "hands_left": hands_left, "sb": sb, "bb": bb, "ante": ante, "button": b,
                        "sb_seat": sb_seat, "bb_seat": bb_seat, "seat": i, "street": street_name,
                        "hole": list(hole_s[i]), "board": board_s[:], "pot": pot, "stacks": stacks[:],
                        "bets": bets[:], "contributed": contrib[:], "active": active[:],
                        "all_in": allin[:], "to_call": to_call, "min_raise_to": min_to, "max_raise_to": mx,
                        "legal_actions": list(legal), "history_this_hand": history[:],
                        "street_starts": street_starts[:], "n_alive": n_alive, "alive": alive_list[:],
                        "start_stacks": start_stacks[:], "payouts": pays_list[:], "n_seats": n,
                        "time_limit_ms": tlim,
                    }
                    exc = None
                    out = None
                    c0 = tc()
                    t0 = pc()
                    try:
                        out = actf(state)
                    except KeyboardInterrupt:
                        raise
                    except BaseException as e:
                        exc = e
                    el = pc() - t0
                    cel = tc() - c0
                    hb = el // 100
                    lh = st.lat_hist
                    lh[hb] = lh.get(hb, 0) + 1
                    st.lat_sum_ns += el
                    if el > st.lat_max_ns:
                        st.lat_max_ns = el
                    hb = cel // 100
                    lh = st.cpu_hist
                    lh[hb] = lh.get(hb, 0) + 1
                    if cel > st.cpu_max_ns:
                        st.cpu_max_ns = cel
                    w_over = el * scale > limit_ns
                    c_over = cel * scale > limit_ns
                    if w_over:
                        st.wall_over += 1
                        if not c_over:
                            st.preempted += 1
                    if c_over:
                        st.cpu_over += 1
                    if exc is not None:
                        st.exceptions += 1
                        if len(st.examples.get("exception", ())) < _MAX_EXAMPLES:
                            st.add_example("exception", f"{type(exc).__name__}: {_trunc(exc)} "
                                                        f"(hand {hand_no} {street_name})")
                    elif (c_over if cpu_mode else w_over):
                        st.timeouts += 1
                        if len(st.examples.get("timeout", ())) < _MAX_EXAMPLES:
                            st.add_example("timeout", f"act() took {el / 1e6:.3f} ms wall, {cel / 1e6:.3f} ms cpu "
                                                      f"x scale {scale:g} > {tlim} ms (hand {hand_no} "
                                                      f"{street_name})")
                    else:
                        try:
                            k = out.get("action") if type(out) is dict else None
                            if k is not None and k != "raise" and k in legal:   # fast path, same rule
                                kind, amt = k, 0
                            else:
                                kind, amt, reason = validate_action(out, legal, min_to, mx)
                        except Exception as e:   # a hostile return value must not crash the engine
                            kind, amt, reason = None, 0, f"validation raised {type(e).__name__}"
                        if kind is None:
                            st.illegal += 1
                            if len(st.examples.get("illegal", ())) < _MAX_EXAMPLES:
                                st.add_example("illegal", f"{reason}; returned {_short(out)} "
                                                          f"(hand {hand_no} {street_name})")
                if kind is None:              # rules s3.7: fold, or check when checking is legal
                    kind = "check" if to_call <= 0 else "fold"
                    amt = 0
                # apply
                all_in_now = False
                if kind == "fold":
                    folded[i] = True
                    active[i] = False
                    n_in -= 1
                    can_act -= 1
                    rec = 0
                    st.n_fold += 1
                elif kind == "check":
                    rec = 0
                    st.n_check += 1
                elif kind == "call":
                    stacks[i] = stk - to_call
                    bets[i] += to_call
                    contrib[i] += to_call
                    pot += to_call
                    if stacks[i] == 0:
                        allin[i] = True
                        all_in_now = True
                    rec = bets[i]
                    st.n_call += 1
                    if street == 0:
                        vpip[i] = True
                else:
                    add = amt - bets[i]
                    stacks[i] = stk - add
                    bets[i] = amt
                    contrib[i] += add
                    pot += add
                    if stacks[i] == 0:
                        allin[i] = True
                        all_in_now = True
                    inc = amt - cur
                    if inc >= min_inc:
                        min_inc = inc
                    cur = amt
                    rec = amt
                    st.n_raise += 1
                    if street == 0:
                        vpip[i] = True
                        pfr[i] = True
                if all_in_now:
                    st.n_allin += 1
                    can_act -= 1
                acted[i] = True
                acted_level[i] = cur
                history.append((i, kind, rec))
                if emitting:
                    emit({"type": "action", "hand_no": hand_no, "street": street_name, "seat": i, "action": kind,
                          "amount": rec, "to_call": to_call, "pot": pot, "stack": stacks[i], "all_in": all_in_now})
                if n_in == 1:
                    break
            # uncalled bet back to its owner (all others folded or are all-in for less)
            top = order[0]
            for j in order:
                if bets[j] > bets[top]:
                    top = j
            m1 = bets[top]
            if m1 > 0:
                m2 = 0
                for j in order:
                    if j != top and bets[j] > m2:
                        m2 = bets[j]
                if m1 > m2:
                    r = m1 - m2
                    stacks[top] += r
                    bets[top] -= r
                    contrib[top] -= r
                    pot -= r
                    allin[top] = False
                    if emitting:
                        emit({"type": "refund", "hand_no": hand_no, "street": street_name, "seat": top,
                              "amount": r})
            if n_in == 1:
                break

        # Money no live player can match (a folded player's antes above every contender's total) goes back to its
        # owner, like an uncalled bet. Without this a showdown layer had no eligible player and the engine raised
        # EngineError (audit fix 2026-11-03; reachable only with antes: blind excess is already refunded above).
        live_max = 0
        for i in order:
            if not folded[i] and contrib[i] > live_max:
                live_max = contrib[i]
        for j in order:
            if contrib[j] > live_max:
                r = contrib[j] - live_max
                stacks[j] += r
                contrib[j] -= r
                pot -= r
                if emitting:
                    emit({"type": "refund", "hand_no": hand_no, "street": STREETS[street], "seat": j, "amount": r})

        # award
        won = [0] * n
        shown = ()
        if n_in == 1:
            for i in order:
                if not folded[i]:
                    won[i] = pot
                    break
        else:
            board = bd
            contenders = [i for i in order if not folded[i]]      # clockwise from the left of the button
            score = [0] * n
            for i in contenders:
                score[i] = evaluate(hole[i] + tuple(board))
            levels = sorted(set(c for c in contrib if c > 0))
            prev = 0
            for lvl in levels:
                slice_ = 0
                for j in range(n):
                    cj = contrib[j]
                    if cj > prev:
                        slice_ += (cj if cj < lvl else lvl) - prev
                elig = [i for i in contenders if contrib[i] >= lvl]
                if not elig:
                    raise EngineError(f"pot layer {prev}-{lvl} has no eligible player (hand {hand_no})")
                best = max(score[i] for i in elig)
                winners = [i for i in elig if score[i] == best]
                share, rem = divmod(slice_, len(winners))
                for t, i in enumerate(winners):
                    won[i] += share + (1 if t < rem else 0)
                prev = lvl
            if emitting:
                shown = tuple((i, hole_s[i]) for i in contenders)
                emit({"type": "showdown", "hand_no": hand_no, "board": tuple(CARD_STR[c] for c in board),
                      "shown": shown})
        for i in range(n):
            if won[i]:
                stacks[i] += won[i]
        if sum(won) != pot:
            raise EngineError(f"awarded {sum(won)} != pot {pot} (hand {hand_no})")
        tot = 0
        for s in stacks:
            if s < 0:
                raise EngineError(f"negative stack {stacks!r} (hand {hand_no})")
            tot += s
        if tot != self.total_chips:
            raise EngineError(f"chips {tot} != {self.total_chips} after hand {hand_no}")
        for i in order:
            st = stats[i]
            st.hands += 1
            if vpip[i]:
                st.vpip += 1
            if pfr[i]:
                st.pfr += 1
        if emitting:
            emit({"type": "hand_end", "hand_no": hand_no, "pot": pot,
                  "won": tuple((i, won[i]) for i in range(n) if won[i]), "stacks": tuple(stacks),
                  "board": tuple(board_s if n_in == 1 else (CARD_STR[c] for c in bd))})
        if self.hand_log is not None:
            self.hand_log.append((hand_no, sb, bb, ante, b, tuple(start_stacks), tuple(stacks)))

        # eliminations: smaller starting stack = worse place; equal starting stacks share their places' points
        busted = [i for i in order if stacks[i] == 0]
        if busted:
            busted.sort(key=lambda i: start_stacks[i])
            place = n_alive
            idx = 0
            while idx < len(busted):
                grp = [busted[idx]]
                idx += 1
                while idx < len(busted) and start_stacks[busted[idx]] == start_stacks[grp[0]]:
                    grp.append(busted[idx])
                    idx += 1
                g = len(grp)
                pts = sum(pays[q - 1] for q in range(place - g + 1, place + 1)) / g
                avg_place = (place + (place - g + 1)) / 2.0
                if avg_place == int(avg_place):
                    avg_place = int(avg_place)
                for i in grp:
                    self.points[i] = pts
                    self.places[i] = avg_place
                    self.bust_hand[i] = hand_no
                    if emitting:
                        emit({"type": "elimination", "hand_no": hand_no, "seat": i, "place": avg_place,
                              "points": pts})
                place -= g


def play_table(factories, payouts="TOP3", schedule="TURBO", seed=0, **kw):
    """Convenience: Table(factories, payouts, schedule, seed, **kw).play() -> TableResult."""
    return Table(factories, payouts, schedule, seed, **kw).play()
