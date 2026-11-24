# Benchmark: the bot against simulated fields

Every number on this page comes from simulation against opponents written for this repo. The bot never played a live
table, so none of it says how the bot would have done at the event. The JSON behind each table is in `results/`, and
`python sim/bench/summarize.py` prints the first three tables from it.

Run on 2026-11-22 from this repo: Apple M4 Pro, CPython 3.13.13, 8 worker processes, load average 7 to 12 from other
jobs on the same machine.

## How to read a score

Each table seats 8 bots. The headline payout (TOP3) gives 50 points for 1st, 30 for 2nd, 20 for 3rd and nothing for
4th to 8th, so the average seat earns 100 / 8 = 12.5 points per table. A **relative score** is points per table
divided by 12.5: 1.00 is an average seat, 1.30 is 30% above average, and the TOP3 ceiling is 4.00 (winning every
table). Brackets are 95% confidence intervals.

**Paired differences.** Every bot that takes our seat in a cell plays the same tables: the same seat, the same
opponents and the same cards dealt to every seat. A difference such as `v2 - v0` is computed table by table. If its
interval excludes 0, the two bots differ at the 95% level for that one comparison.

**Fields.** Our seat plus 7 opponents drawn per table from a mix:

- `T1_like`: a guess at the first tournament of an event, with 30% broken seats (crashed, buggy or random bots).
- `T6_like`: a guess at the last tournament, with 6% broken seats and 28% competent ICM bots.
- `STRONG`: a worst case where half the seats are competent ICM jam/fold bots (`icm_pro`).
- `AI_ONLY`: the nine bots in `sim/field_ai/`, each written by an independent AI agent from public materials.
- `AI_MIX`: half `AI_ONLY`, half `T6_like`.

The shares are in `sim/zoo/fields.py` and `sim/field_ai/fields.py`. They are priors. The private version of the zoo
had three more archetypes, slow Monte Carlo bots modelled on the organizer's public exhibit bots; they are left out
of this repo, and their seats went to `dead` and `station`. The private AI-written field also had a tenth bot, a port
of those exhibit bots, which is left out for the same reason. Those two are easy opponents, so the fields here are
softer than the ones the bot was tuned against, and scores here run higher than the private runs did.

## Our bot in each field

TOP3 payout, TURBO blinds (10/20 rising every 25 hands), 6,000 tables per field, seed 1122. v0, v1 and v2 are the
three layers of the bot (see the README); v2 is the shipped default.

| field | tables | seed | v0 | v1 | v2 | v2 - v0 (paired) | file |
|---|---|---|---|---|---|---|---|
| T1_like | 6,000 | 1122 | 1.857 [1.814, 1.899] | 1.914 [1.873, 1.956] | 2.111 [2.069, 2.154] | +0.255 [+0.214, +0.296] | `results/sim/b2_T1_like_TOP3_TURBO.json` |
| T6_like | 6,000 | 1122 | 1.501 [1.461, 1.542] | 1.568 [1.529, 1.608] | 1.719 [1.678, 1.761] | +0.218 [+0.177, +0.260] | `results/sim/b2_T6_like_TOP3_TURBO.json` |
| STRONG | 6,000 | 1122 | 1.219 [1.179, 1.258] | 1.308 [1.269, 1.347] | 1.458 [1.418, 1.499] | +0.239 [+0.196, +0.282] | `results/sim/b2_STRONG_TOP3_TURBO.json` |
| AI_ONLY | 6,000 | 1122 | 1.460 [1.421, 1.499] | 1.488 [1.450, 1.527] | 1.599 [1.559, 1.639] | +0.139 [+0.096, +0.183] | `results/sim/ai_AI_ONLY_TOP3_TURBO.json` |
| AI_MIX | 6,000 | 1122 | 1.458 [1.419, 1.498] | 1.496 [1.457, 1.535] | 1.673 [1.632, 1.713] | +0.214 [+0.172, +0.256] | `results/sim/ai_AI_MIX_TOP3_TURBO.json` |

v2 is above average in all five fields and above v0 in all five paired comparisons. Its lowest score is in `STRONG`
(1.458). Its smallest gain over v0 is against the AI-written bots (+0.139).

## Other bots in our seat, on the same tables

Three opponent archetypes also took our seat, as stand-ins for bots other teams might submit: `icm_pro` (Nash
jam/fold plus exact ICM), `chart_tag` (a textbook tight-aggressive bot) and `lag_aggro` (loose and aggressive).

| field | icm_pro in our seat | chart_tag in our seat | lag_aggro in our seat | v2 - icm_pro (paired) |
|---|---|---|---|---|
| T1_like | 1.762 [1.720, 1.805] | 1.293 [1.257, 1.329] | 1.100 [1.058, 1.141] | +0.349 [+0.302, +0.396] |
| T6_like | 1.431 [1.391, 1.471] | 1.089 [1.055, 1.122] | 0.837 [0.800, 0.874] | +0.289 [+0.243, +0.334] |
| STRONG | 1.236 [1.197, 1.275] | 0.850 [0.819, 0.881] | 0.704 [0.670, 0.738] | +0.222 [+0.177, +0.267] |
| AI_ONLY | 1.370 [1.331, 1.410] | 0.951 [0.920, 0.983] | 0.736 [0.700, 0.771] | +0.229 [+0.182, +0.276] |
| AI_MIX | 1.431 [1.391, 1.471] | 1.031 [0.999, 1.064] | 0.779 [0.743, 0.815] | +0.242 [+0.195, +0.288] |

`icm_pro` is the strongest of the three in every field, and v2 beats it by +0.222 to +0.349 on the same tables.
`lag_aggro` scores below average (under 1.00) in four of the five fields.

## Errors and the 1 ms budget

Score runs use the bot in production mode: its shell catches errors and falls back to a simpler decision, and it
counts every catch. The kit's clock is off in score runs, so no decision is cut off for time, and each decision's CPU
time is still recorded.

| field | our decisions (v0 + v1 + v2) | exceptions | illegal | CPU over 1 ms | masked by the shell |
|---|---|---|---|---|---|
| T1_like | 2,781,969 | 0 | 0 | 0 | 0 |
| T6_like | 2,566,891 | 0 | 0 | 0 | 0 |
| STRONG | 2,433,443 | 0 | 0 | 0 | 0 |
| AI_ONLY | 2,607,560 | 0 | 0 | 0 | 0 |
| AI_MIX | 2,608,811 | 0 | 0 | 0 | 0 |

A separate latency run (`bot/tests/latency.py --decisions 200000`, one process, our bot in 4 of 8 seats against
`chart_tag`, `icm_pro`, `pushfold_maniac` and `station`, load average 7 to 10) measured each version's `act()` on the
wall clock and on thread CPU time. File: `results/latency_mixed_v0_v1_v2.json`.

| version | decisions | wall p50 / p99.9 / max (us) | CPU p50 / p99.9 / max (us) | over 1 ms |
|---|---|---|---|---|
| v0 | 200,267 | 7.1 / 33.3 / 127.7 | 7.4 / 28.3 / 98.4 | 0 |
| v1 | 200,365 | 7.5 / 46.7 / 210.8 | 7.7 / 43.6 / 128.1 | 0 |
| v2 | 200,247 | 8.8 / 53.4 / 200.2 | 9.0 / 48.9 / 226.9 | 0 |

v2's CPU maximum (226.9 us) sits above its wall-clock maximum (200.2 us). The two maxima can come from different
decisions, and the kit's CPU window opens just before and closes just after its wall-clock window
(`sim/kit/engine.py`), so the two columns are reported as recorded. The slowest v2 decision used 226.9 us of CPU, under a quarter of the 1 ms limit on this machine. A sandbox several
times slower than an M4 Pro core would shrink that margin, and the real platform's speed and timing method were
never published.

## Reproduce

```bash
python sim/bench/run_bench.py b2top3 ai --tables 6000 --workers 8 --seed 1122   # writes sim/bench/out/*.json
python sim/bench/summarize.py sim/bench/out
python bot/tests/latency.py --decisions 200000 --json sim/bench/out/latency.json
```

A re-run with the same seed does not reproduce every cell to the last digit. In the three `b2top3` cells, v0 and the
zoo bots in our seat replayed exactly. v1 and v2 start an optional exact bubble factor only when it fits inside the bot's own 200 us wall-clock
guard (`bot/brain.py`), so machine load changes a few hundred of their decisions per cell: a re-run of `b2top3` from
a fresh clone moved v2 in `T1_like` from 2.111 to 2.112 and v2 in `STRONG` from 1.458 to 1.459, far inside the
intervals. Two of the AI-written bots also stop work on a clock (`ai_quant` its Monte Carlo, `ai_icm_plan` its ICM
loop), so cells that include them can differ slightly too. Latency numbers depend on the machine and its load.
