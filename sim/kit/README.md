# sim/kit: mock starter kit

A stand-in for a tournament starter kit, written from the published format only (the official rules list an agent
template, a local simulator, an ICM calculator, a payout structure and a blind schedule). No organizer code is used.
The real interface was to be revealed on the event day, so bots sit behind a thin adapter and only the adapter would
change. Pure standard library (tested here on Python 3.13).

```bash
.venv/bin/python -m pytest -q sim/kit/tests                                   # about 10 s
.venv/bin/python -m sim.kit.runner -e 'ex=sim/kit/example_agent.py:ExampleAgent*8' -n 1000
.venv/bin/python -m sim.kit.bench_speed                                        # throughput, writes out/speed.json
```

## 1. The Agent API

```python
from sim.kit.agent import Agent

class MyBot(Agent):
    def __init__(self, seat, config):    # a NEW instance per table (bot state resets each table, s3.8)
        super().__init__(seat, config)
    def act(self, state):                # TIMED: wall clock around this call must stay under 1 ms
        return {"action": "raise", "amount": state["min_raise_to"]}
    def observe(self, event):            # optional, NOT timed: public events (and your own private deal)
        pass
```

Subclassing is optional (any class with `act`, and optionally `observe`, works). A factory is anything called as
`factory(seat, config) -> agent`; a class is a factory.

**config** (a fresh dict per agent): `seat, n_seats, start_stack, start_stacks, payouts` (8 floats, 1st place
first), `blind_schedule` (list of `[sb, bb, ante]`), `hands_per_level` (25), `max_hands` (200), `time_limit_ms`
(1.0), `seed` (an int for your own `random.Random`; it differs per seat and table, and is identical across PAIRED
runs). The engine reseeds the global `random` module per table too, but your own seeded generator is the
reproducible choice.

**state** (a fresh dict per decision; lists are copies, indexed by seat 0..n-1):

| field | meaning |
|---|---|
| `hand_no`, `level`, `hands_to_next_level`, `hands_left` | 1-based hand number; 1-based blind level; hands until the blinds change, counting this one (at the last level: hands to the cap); hands after this one before the 200-hand cap |
| `sb`, `bb`, `ante` | blinds and ante of this hand |
| `button`, `sb_seat`, `bb_seat`, `seat` | positions; `seat` is you |
| `street` | `"preflop"`, `"flop"`, `"turn"`, `"river"` |
| `hole`, `board` | your two cards and the board, strings like `"Ah"`, `"Td"` (ranks `23456789TJQKA`, suits `cdhs`) |
| `pot` | every chip put in this hand so far, including this street's bets and antes |
| `stacks`, `bets`, `contributed` | chips behind now; chips bet this street; chips put in this hand |
| `active`, `all_in`, `alive` | in this hand and not folded; all-in; still in the tournament (had chips at hand start) |
| `start_stacks` | stacks at the start of this hand (before antes and blinds) |
| `to_call` | chips a call puts in, capped at your stack (a short call is an all-in for less) |
| `min_raise_to`, `max_raise_to` | raise-to bounds; both 0 when raising is not legal; `max_raise_to` = all-in |
| `legal_actions` | subset of `["fold", "check", "call", "raise"]` |
| `history_this_hand` | list of `(seat, action, amount)`: actions `ante`, `post_sb`, `post_bb`, `fold`, `check`, `call`, `raise`; amount = that seat's street total after the action (0 for fold/check) |
| `street_starts` | index into `history_this_hand` where each street began (`[0, i_flop, i_turn, i_river]`) |
| `n_alive`, `n_seats`, `payouts`, `time_limit_ms` | players still in; seats; points by place; the limit |

**Actions.** Return `{"action": "fold" | "check" | "call" | "raise", "amount": total}`. `amount` matters only for
`raise` and is the TOTAL street bet after raising (raise-to semantics; raise-by is never used). All-in = raise to `max_raise_to`, or
`call` when your stack does not cover the call. Legal means: the action is listed in `legal_actions`, and a raise
amount is an int (or an integral float) with `min_raise_to <= amount <= max_raise_to`. Note what is listed:
`fold` only when facing a bet, `check` only when not, `raise` only when somebody could still respond and you have
chips beyond the call. `agent.validate_action()` is the exact check; helpers `fold() check() call() raise_to(x)
all_in(state) check_or_fold(state) clamp_raise(state, target)` build legal actions.

**Errors (rules s3.7), no silent repair.** An exception from `act`, an illegal return (non-dict, unknown action,
`"CALL"`, `"bet"`, a raise outside the bounds, a string amount, `fold` when checking is free, ...), or a decision
over the time limit is counted, and the engine plays **fold, or check when checking is legal**. It never rounds,
clamps or converts what you returned. A bot whose constructor raises errors on every decision.

**Events** (`observe`, untimed, each agent gets its own copy; values are immutable): `hand_start` (stacks, button,
blind seats, `posts`), `deal` (your own hole cards only), `action` (seat, action, amount, to_call, pot, stack,
all_in), `street` (board, pot), `refund` (uncalled bet returned), `showdown` (board, `shown` = every contender's
cards), `hand_end` (`won` per seat, stacks), `elimination` (seat, place, points), `table_end` (points, places,
stacks). Nobody sees another seat's cards before showdown. Agents that keep the base-class `observe` are skipped
(no call cost). Exceptions inside `observe` are counted (`observe_exceptions`) and otherwise ignored.

## 2. Game rules implemented (engine.py)

- 8 seats, 2,000 chips each, blinds from a schedule, up one level every 25 hands; the table ends at one survivor or
  after hand 200. Eliminated seats get their place's points; players busted in the same hand are ordered by their
  stack at the start of that hand (bigger = better place) and exactly equal stacks split the points of the places
  they share (place reported as e.g. 6.5). At the cap the survivors split the remaining places' points by exact ICM;
  their reported place is the chip rank. Points per table always sum to the payout total (100).
- Raises are raise-to. Minimum raise = current bet + the last full raise increment (the big blind if none; also the
  minimum opening bet postflop). An all-in below a full raise is allowed but does not reopen the betting: a player
  who already acted may raise again only if the bet has risen by at least one full increment since they acted
  (short all-ins can add up to a full raise, TDA-style).
- Preflop the amount to match is the nominal big blind even when the BB is all-in for less.
- Heads-up: the button posts the small blind, acts first preflop and last postflop. Moving button (next seat with
  chips), no dead-button rule.
- Uncalled bets come back at the end of each betting round; side pots by contribution layers; split pots share
  equally, odd chips one each to the winners nearest the left of the button. Optional antes (dead money). A folded player's
  antes that no live player can match (every contender all-in for less) go back to it like an uncalled bet
  (audit fix 2026-11-03; before, such a hand raised EngineError).
- Dealing is by position: in hand h seat i gets `deck[2i], deck[2i+1]`, the board is `deck[2n..2n+4]`, and the deck
  of hand h depends only on the table seed and h. So PAIRED runs give each seat the same cards hand by hand even after
  play diverges. Bots never see the deck generator.
- Chip conservation, pot-award totals and points sums are checked every hand; a violation raises `EngineError`
  (a kit bug, never a bot's fault). The evaluator is tested in `tests/test_cards.py` against a brute-force 5-card evaluator.

**Payouts** (`payouts.py`, each sums to 100, so the average seat earns 12.5): `TOP3` 50/30/20, `LINEAR` 8..1 scaled,
`WTA` 100, `TOP4` 40/30/20/10, or any list. **Schedules** (`schedules.py`): `TURBO` 10/20, 15/30, 25/50, 50/100,
75/150, 100/200, 150/300, 200/400; `FAST` doubling 10/20 -> 1280/2560; `SLOW` about 1.25x per level (BB 20 -> 100),
or any list of `(sb, bb[, ante])`. **ICM** (`icm.py`): `icm_equities(stacks, payouts)` exact Malmuth-Harville for all
seats (bitmask DP), `make_icm(payouts)` -> `icm(stacks, i)` `icm_brute`
(reference); tested against each other and brute force.

## 3. The runner (runner.py)

```python
from sim.kit.runner import run, run_paired
field = [("ours", "bot/shell.py:Bot")] + [("nit", "sim/zoo/nit.py:NitAgent")] * 7   # illustrative paths
res = run(field, n_tables=5000, seed=1, payouts="TOP3", schedule="TURBO")   # workers default: cores - 2
print(res.summary())                      # pooled by name; res.summary(pooled=False) per entrant
ours = res.entrants[0]
ours.points, ours.places                  # per table, in table order
ours.ci(), ours.rel()                     # mean points with 95% CI; relative score (points / 12.5) with CI
ours.stats.summary()                      # decisions, timeouts, exceptions, illegal, latency p50/p99/p99.9/max, ...
res.save_json("sim/bench/out/x.json")
```

- Entrants: `("name", spec)`, where spec is a class/factory (picklable for workers > 1: module-level class or
  `functools.partial`; a lambda does not pickle) or a string `"package.module:Attr"` / `"path/file.py:Attr"` (relative to the
  project root). A bare `"path/file.py"` picks `make_agent`, `AGENT`, a non-base `Agent`, or the module's single
  agent class. Copies are separate entrants with the same name; `res.by_name()` pools them.
- Seating: `seating="event"` (default, the event's format): each iteration shuffles all k entrants into
  ceil(k/8) tables, empty seats get random copies marked as fillers (they play, earn nothing, are not recorded).
  `seating="sample"`: each table draws 8 distinct entrants. Agents see seat numbers only.
- Reproducible: table t of seed s depends only on (s, t), so any worker count gives identical results
  (tested), except decisions near the time limit (wall clock) and bots whose module-level state carries over between
  tables in a worker process.
- PAIRED: `run_paired({"v0": field_v0, "v1": field_v1}, n_tables, seed=1, focus=0)` replays the same tables
  (seating, buttons, decks, agent seeds) for every configuration and returns per-table paired differences of entrant
  `focus` versus the first configuration: mean diff, 95% CI, correlation, and the unpaired standard error for
  comparison. Lists must have equal length.
- **Per-entrant stats** (`SeatStats.summary()`): tables, decisions, timeouts, exceptions, illegal, error_rate,
  `latency_us` (wall p50/p99/p99.9/max/mean), `cpu_us` (thread CPU time percentiles), `over_limit` (wall, cpu,
  preempted), coarse histograms for charts, observe calls/exceptions/time, init time and failures, action counts,
  VPIP/PFR, and up to 5 example messages per error type. Exact per-decision latencies are kept as a 0.1 us
  histogram (`stats.lat_hist`).
- **gc_mode** (`"default"`, `"freeze"`, `"disable"`): the garbage-collector setting of each worker process (with
  `workers=1`, of this process for the run, then restored). See section 4 for why it matters.
- Fails fast: bad bot paths/attributes, unpicklable specs and invalid options raise before the pool starts;
  a worker that still fails to initialize raises from its first task (a failing pool initializer would otherwise
  make multiprocessing respawn workers forever). Driving the runner from stdin or a notebook also works.
- CLI: `python -m sim.kit.runner -e 'NAME=SPEC*COPIES' ... -n TABLES --seed S --payouts TOP3 --schedule TURBO
  --seating event --workers W --time-limit-ms 1.0 --time-scale 1.0 --timing wall --gc-mode default --json out.json`.

## 4. Timing: read this before trusting a timeout count

Every `act()` is timed with `time.perf_counter_ns` (wall clock, what the rules limit) and with `time.thread_time_ns`
(the thread's CPU time). Counters: `wall_over`, `cpu_over` (the bot's own work was over the limit), and `preempted`
(over on the wall clock only: the OS or other processes delayed it). `timing="wall"` (default, faithful to s3.7)
enforces on wall clock; `timing="cpu"` enforces on CPU time. `time_scale=k` multiplies measured times before the
comparison (emulates a sandbox k times slower than the test machine). `time_limit_ms=None` disables enforcement (debug).

Measured on the development machine (Apple M4 Pro, 8 performance + 4 efficiency cores) while other jobs were running:
- Cheap bots (1-20 us per act) in 10-worker runs: 4-145 wall-clock timeouts per run (up to 43 per million
  decisions), act wall max up to 18 ms, and **every one** was `preempted` (CPU time never above 463 us). Single
  worker: 0-1.
- A CPU-bound bot at about 0.6 ms per act (fixture `WorkAgent`, models a trimmed Monte Carlo bot), at load average
  26-30: over 1 ms on the wall clock 0.85% of decisions in-process, 1.78% with 8 workers, 9.32% with 10 workers,
  versus 0.01-0.10% on CPU time; its CPU p50 rose from 604 us (1 worker) to 811 us (10 workers: efficiency cores and
  contention). At load 6.5 in-process: 0.15% wall, 0 CPU.

- Garbage collection: in long-running workers (about 500 tables each), the example bot's worst act() CPU time was
  0.84-0.85 ms with the default collector and with `gc_mode="freeze"`, but 53 us with `gc_mode="disable"`
  (2.66M decisions each, 8 workers). One earlier run had 4 decisions at 2.1-2.9 ms of CPU, all in hands 1-2 of a
  table. So rare collector pauses land inside act() calls, independent of the bot; freezing at start-up does not
  remove them.

So: for score benchmarks use `timing="cpu"` and state any slow-sandbox assumption explicitly with `time_scale`; for
reliability (B4) measure wall clock on a quiet machine with `workers <= 8`, and report `cpu_over` and `preempted` next
to the wall-clock timeouts; add `gc_mode="disable"` when you want the bot's own latency without kit GC pauses.
Never read a laptop timeout count as a bot bug before checking `cpu_over`.

## 5. Speed (measured, `bench_speed.py`, cheap bots, includes pool start-up)

| field (8 seats) | hands/table | decisions/table | 1 worker | 10 workers |
|---|---|---|---|---|
| example x8 | 122 | 669 | 179-192 tables/s | 1,165-1,280 tables/s |
| check/call station x8 (worst case: 200 hands, 8-way showdowns) | 200 | 4,837 | 43-48 | 278-308 |
| passive-random x8 | 182 | 2,250 | 78-88 | 531-677 |
| random x8 (all-ins end tables fast) | 15.5 | 83 | 1,116-1,248 | 8,331-9,621 |
| mixed (example, station, passive, random, all-in, check/fold) | 151 | 808 | 165-174 | 1,056-1,384 |

Two runs, load average 3.7 -> 8.9 and 4.3 -> 7.2 (other jobs were busy). The build
target (>= 300 tables/s with cheap bots) is met by every field except the pathological all-calling station field,
which sits at it (278-308). Engine cost is about 4-5 us per decision (state dict 1.3 us, the four clock reads
0.44 us); real field bots will dominate run time once they are slower than that.

## 6. Known gaps versus the real event (fidelity notes)

- The variant may not be NLHE; the real interface, timing method (IPC included?), Python version and packages are
  unknown. The adapter is where these land.
- Agents share a process with the engine and the other seats of their table: a `gc.disable()`/`gc.freeze()` in one
  bot affects every bot in that worker, and module-level state persists across tables in a worker. There is no
  sandbox: a hostile bot could read engine memory. Rehearsal only.
- `observe` is untimed (a design choice), so a bot could move heavy work there; the real platform may time it. Its total
  and max time are recorded per entrant (`observe.total_ms`, `observe.max_us`) so this is visible.
- `fold` when checking is free counts as illegal here (auto-check, same outcome); a real engine might accept it.
- Showdown reveals every contender's cards (no mucking). Moving button, no dead-button rule.

## 7. Files

- `agent.py` interface, helpers, `validate_action` | `engine.py` table engine, `SeatStats`, `make_deck` (rigged decks)
- `cards.py` card strings, evaluator | `icm.py` exact ICM | `payouts.py` | `schedules.py` | `stats.py` CI helpers
- `runner.py` `run`, `run_paired`, `seating_plan`, CLI | `bench_speed.py` throughput measurement
- `example_agent.py` the sample bot (simple, legal; measured act() p50 about 1.2-1.7 us, p99 about 5-10 us)
- `fixtures.py` engine test bots: check/fold, check/call, all-in, random, passive-random, sleepy (2 ms), busy
  (busy-wait), work (CPU-bound, about 0.6 ms), exploding, illegal-raise, fold-always, bad-init, scripted, recorder.
  These exercise the engine; the field archetypes live in `sim/zoo/`.
- `tests/` evaluator vs brute force; ICM vs brute force; 24 rules cases (side pots, refunds, min-raise and
  reopening, heads-up, BB option, short blinds, antes, odd chips, ties) plus chip conservation every hand over 180
  random tables; strictness (sleep 2 ms, exception, illegal raise, check-not-fold when free, bad init, hostile
  returns, time scale, CPU vs wall); runner (sums, determinism across worker counts, PAIRED alignment, event seating
  with fillers, specs, JSON, gc_mode, fail-fast on bad specs/options); observe (privacy, sequence, copies).
