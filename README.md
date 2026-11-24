# No Limit pokerbot

A sit-and-go poker bot written for Sequoia's "No Limit: A Bot Building Tournament", plus the simulator and opponent
zoo used to test it. It answers every decision with lookups and closed-form math, so it stays far inside the
event's 1 ms limit.

**Status.** The bot was built for a solo-team application to the tournament. The Bay Area regional was full, so I did
not attend, and the bot never played a live table. Every result below comes from simulation against the opponents in
this repo.

## How the bot works

The bot is three layers, each a version you can select (`v0`, `v1`, `v2`), so the benchmark can price each one.

- **v0, push/fold core.** At an effective stack of 20 big blinds or less (12 when the payout pays two or more
  places), the bot plays jam or fold from chip-EV Nash ranges, adjusted for antes, limpers and the number of players
  who can call. Calling an all-in compares our hand class's equity against the jammer's estimated range with the
  real pot odds. Deeper stacks open 2.5 big blinds by position. Postflop play is value-heavy by hand class, with no
  unconditional bluffs.
- **v1, ICM.** Exact Malmuth-Harville ICM from the announced payout, and bubble factors that tighten calls and jams
  when busting costs more than doubling gains. With two to four players left at 20 big blinds or less, under top-3 and linear
  payouts, the bot uses precomputed ICM push/fold equilibrium tables (`bot/icm_nash_tables.py`). The solver that made
  them, `bot/tools/solve_icm_pushfold.py`, is included for reference; it reads heads-up equity matrices that are not
  in this repo, so it cannot regenerate the tables from this copy.
  ICM switches off under winner-take-all, where every bubble factor is 1.
- **v2, opponent range model.** Per-seat counters from public actions only (VPIP, raises, jams, calls of jams, flop
  folds), shrunk toward pooled priors, type each opponent (folder, caller, maniac, stealer, nit and so on). A
  Beta-binomial range model widens or narrows each seat's assumed range from its preflop evidence, capped at half to
  twice the prior. The types decide steals, light jams, re-jam ranges and value sizing.

**The 1 ms budget.** Nothing runs Monte Carlo at decision time. The shell pauses the garbage collector inside `act()`,
times itself, and starts an optional exact bubble factor only if it can finish inside a 200 us guard. If anything
raises, the shell answers with the v0 decision for the same spot, and if that fails too it checks or folds, which is
the rules' own fallback. The single-file bundle (`bot/dist/nolimit_bot.py`) imports with the standard library only.

**Variant handling.** The rules said the event's poker variant would be revealed at the start of the build phase. `variant/` describes a
game as a `VariantSpec` (deck, hole cards, board, special requests), rebuilds the push/fold orders and equity tables
for it (`variant/build_tables.py`), and the bot swaps those tables in at import after sanity checks
(`bot/tables_loader.py`). Special requests such as discards answer with legal defaults, and postflop hand classes
follow the variant's ranking (short deck and a toss game are tested). `sim/variants/` has five rehearsal variants
with their rules in `sim/variants/rules/`.

`bot/adapter.py` is the only file that knows the kit's interface. The mock kit stood in for the real starter kit,
which the organizers planned to hand out at the event, so the adapter is where a real platform would plug in.

## How it was tested

- `sim/kit/` is a standard-library stand-in for a starter kit: table engine, Agent interface, exact ICM, payouts,
  blind schedules and a paired runner that replays identical tables for every bot in a seat. See `sim/kit/README.md`.
- `sim/zoo/` holds nine hand-built opponent archetypes (`chart_tag`, `icm_pro`, `pushfold_maniac`, `lag_aggro`,
  `station`, `nit`, `buggy`, `random_legal`, `dead`) and three field mixes.
- `sim/field_ai/` holds nine opponent bots written by independent AI agents from public materials, each asked to
  write the bot a different kind of team might submit.
- The test suites cover legality (raise semantics, short all-ins, heads-up blinds and more), ICM against brute force,
  the tracker, the bundle, latency, the variant layer and the kit engine itself.

## Results (simulated)

These scores are simulated, against the opponent zoo and the AI-written bots in this repo. A relative score of
1.00 is an average seat and 4.00 is the ceiling (winning every table). TOP3 payout (50/30/20), 6,000 paired tables
per field, seed 1122, from `results/sim/`:

| field | v2 (the shipped bot) | v0 (push/fold core only) | `icm_pro` in the same seat |
|---|---|---|---|
| T1_like (30% broken seats) | 2.111 [2.069, 2.154] | 1.857 [1.814, 1.899] | 1.762 [1.720, 1.805] |
| T6_like (6% broken seats) | 1.719 [1.678, 1.761] | 1.501 [1.461, 1.542] | 1.431 [1.391, 1.471] |
| STRONG (half `icm_pro`) | 1.458 [1.418, 1.499] | 1.219 [1.179, 1.258] | 1.236 [1.197, 1.275] |
| AI_ONLY (the nine AI-written bots) | 1.599 [1.559, 1.639] | 1.460 [1.421, 1.499] | 1.370 [1.331, 1.410] |
| AI_MIX (half AI, half T6_like) | 1.673 [1.632, 1.713] | 1.458 [1.419, 1.498] | 1.431 [1.391, 1.471] |

Over those 30,000 tables v0, v1 and v2 made 12,998,674 decisions with 0 exceptions, 0 illegal actions and 0
decisions over 1 ms of CPU. In a 200,000-decision latency run, v2's slowest decision took 226.9 us of CPU
(`results/latency_mixed_v0_v1_v2.json`). The full tables, the paired differences and how to read them are in
[docs/BENCHMARK.md](docs/BENCHMARK.md).

## Run it

You need Python 3.13 (other versions may work; 3.13 is what this repo was tested on) and [uv](https://docs.astral.sh/uv/)
or pip.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt     # pytest; the bot itself has no dependencies

.venv/bin/python -m pytest -q sim/kit/tests sim/zoo/tests sim/variants/tests bot/tests   # about a minute
.venv/bin/python -m pytest -q variant/tests                                            # about two minutes

# a sample tournament: the bot in one seat, seven copies of the kit's example agent, 500 tables.
# It enforces the 1 ms limit on the wall clock, so a busy machine can show a few wall-clock timeouts for the bot;
# the benchmark runs turn the kit's clock off and record CPU time instead.
.venv/bin/python -m sim.kit.runner -e 'ours=bot/shell.py:BotV2' -e 'ex=sim/kit/example_agent.py:ExampleAgent*7' -n 500

# a paired A/B of the three layers against the T6_like field
.venv/bin/python bot/tools/ab.py --field T6_like --tables 1000 v0 v1 v2

# rebuild the single-file bundle after changing the bot
.venv/bin/python bot/tools/bundle.py
```

## Layout

| path | what it holds |
|---|---|
| `bot/` | adapter, shell and brain; generated lookup tables; `dist/nolimit_bot.py` bundle; tools; tests |
| `variant/` | VariantSpec, hand spaces, the table factory and its solver, request defaults, tests |
| `sim/kit/` | the mock starter kit |
| `sim/zoo/`, `sim/field_ai/` | opponent archetypes and AI-written opponent bots |
| `sim/variants/` | rehearsal variant engines and rules |
| `sim/bench/` | the paired benchmark driver and `summarize.py` |
| `sim/zoo_fragile/` | `chart_tag` copies with one planted bug each, for probing |
| `results/`, `docs/` | the published benchmark JSON and its write-up |

Code comments sometimes cite design notes by name (for example "ICM_NASH notes" or labels such as E02 and P4). Those
notes are not published.

## Limitations

- Every score is against modelled fields. The real field, payout and variant were never seen, so the numbers rank
  the bot's own versions and say little about a live event.
- The bot was tested against 12 opponent archetypes and 10 AI-written bots. This repo publishes 9 of each: the three
  slow Monte Carlo archetypes modelled on the organizer's public exhibit bots, and an AI-written port of those exhibit
  bots, are left out. The fields here are softer than the ones the bot was tuned against.
- The AI-written bots all came from one family of AI models, each on a first try.
- Typing thresholds and priors were tuned only against these fields.
- Postflop play outside the tested variants falls back to hold'em hand classes.
- Latency was measured on an Apple M4 Pro; a slower sandbox, or a platform that times inter-process overhead, would
  shrink the margin.

## License

Apache-2.0, see [LICENSE](LICENSE). Copyright 2026 Ali Shehral.
