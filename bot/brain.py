"""Brain of the rehearsal bot (written for No Limit, Bay Area regional). Pure standard library.

Three versions, selected by PARAMS["VERSION"] (or the shell classes BotV0 / BotV1 / BotV2):
  v0  chip-EV jam/fold skeleton keyed on effective stack S and players behind k. First-in jams: S = min(my chips,
      biggest live opponent). Calls of all-ins: equity of our class against the jammer's estimated range (push-order
      top-J table) versus the real pot odds from contributions, so S is implicitly min(my chips, the jammer's chips)
      (P4). Deep opens by position, the tested facing-raise rules, value-heavy postflop by hand class, no bluffs.
  v1  v0 + exact ICM bubble factors (BF) from the announced payout, padded to 8 places. Call side always (risk
      premium on every all-in call, BF^-2.6 on re-jam ranges, BF gates on big postflop calls). Jam side: jam% x
      BF_hero^-2.5 by default; the tested jam% x BF_callers^2 x BF_hero^-1.5 only when every seat behind has been
      OBSERVED folding to jams more than chip-EV Nash (P2). Missing BFs come from an equal-stack BF table, never 1.0
      (P3).
  v2  v1 + per-seat typing from ACTIONS only (showdowns are censored), shrunk to pooled priors with minimum samples:
      folder (cleared at its first voluntary chip in the current hand, P7), maniac (its own shrunk jam rate as the
      jam range, E10), caller (light-jam cap, bigger value bets), loose, stealer (re-jam floor, E11), nit; gated SB
      widen (typed-folder BB or a BB whose shrunk jam-call rate is under the break-even, P1); a separate, tighter
      prior for re-jams over a raise (P5); BB defence against late small raises (P6); air c-bets only against seats
      with observed flop folds (E13 corrected, P10). RANGE_MODEL (RANGE_MODEL notes): per-seat Beta-binomial
      posteriors of jam / open / 3-bet / jam-call range width from preflop actions of completed hands (never postflop
      calls), within [prior / 2, prior x 2], price our calls, re-steals, flats, 3-bets and BB defence.

Every tunable number is in PARAMS below. Decisions are lookups and closed forms only (no Monte Carlo); the only
costly step is an exact BF (about 10-40 us), computed at most BF_PER_DECISION times per decision, cached per hand, and
skipped once the own clock passes GUARD_NS (the default table value is used instead).

The brain never sees the kit's raw state: adapter.py turns it into a View. It returns internal decisions:
("fold",), ("check",), ("call",), ("raise", raise_to_total), ("allin",); adapter.py makes them legal.
"""
import bisect
import random
import time
from itertools import combinations as _combinations

# ===================================== PARAMS: every tunable number =====================================
PARAMS = {
    "VERSION": 2,                 # 0, 1 or 2 (see the module docstring)
    # ---- shell and adapter (shell.py reads these; its own copy is used only if this module fails to import)
    "GC_GUARD_ACT": True,         # pause the collector for the duration of act() (try/finally, about 0.1 us), so no
                                  # collection can START inside our timed call; a due one runs after we return
    "GC_HYGIENE": True,           # at each hand_start event (observe, untimed in the kit) run a gen-1 or full
                                  # collection that is about to come due; no effect if observe() is never called
    "GC_FULL_MARGIN": 2,          # hygiene: full collection when gen-2's count is within this of its threshold
    "GC_GEN1_MARGIN": 1,          # hygiene: gen-1 collection when gen-1's count is within this of its threshold
    "GC_MODE": "freeze",          # at import, after the warm-up: "freeze" (gc.collect + gc.freeze, collector ON),
                                  # "disable" (freeze + gc.disable), "none"
    "WARMUP_DECISIONS": 300,      # synthetic decisions per version through a throwaway instance at import (0 = skip)
    "ALLIN_SNAP": 0.85,           # adapter: a raise-to at or above this share of max_raise_to goes all-in
    "FALLBACK_V0": True,          # P15: an exception in v1/v2 work returns the v0 decision (ICM and typing off),
                                  # not check/fold; only if that also fails does the shell check/fold
    "BOUNTY_PARTS": ("jam", "call"),   # bounty hit (adapter v.bounty): where the payoff hook re-prices: "jam" first-in
                                  # jam depth, "call" calls of all-ins, "post" postflop call thresholds. A/B 2026-11-03
                                  # (1,500 paired tables vs 7 BountyExample, pts vs no hook): jam +0.08 [-0.05, +0.22],
                                  # call -0.03 [-0.09, +0.04], post -0.83 [-1.29, -0.36] -> post OFF
    "KEEP_BOARD_AWARE": False,    # keep/discard/toss requests: True = keep_policy (postflop class on the known board
                                  # + NLHE order); False = the variant layer's static keep. Toss A/B 2026-11-03, 1,000
                                  # paired tables vs 7 TossExample: board-aware -2.6 pts [-3.7, -1.5], so OFF
    "LEAK_PENALTY": False,        # revealed discards (DISCARD_LEAK notes): keep = static score - LEAK_LAMBDA x the
                                  # strength a reader infers from the cards we give up (reader model below); keeps 2
    "LEAK_LAMBDA": 20.0,          # penalty in static-score units per unit of revealed strength shift
    "USE_REVEALED": False,        # postflop calls: the bettor's revealed discards narrow its kept range (reader model);
                                  # our class is moved one step when the read shifts its ahead-share by REVEAL_T
    "REVEAL_EPS": 0.1,            # reader model: likelihood of a discard that is not the model's worst card (measured
                                  # 2026-11-04: the NLHE-order toss-worst model names our static toss 93.5% and the
                                  # kit-style TossExample's 84.3% of the time on random 3-card hands)
    "REVEAL_T": 0.10,             # |shift in the bettor's ahead-share vs us| that moves our call class one step
    # ---- payout and ICM (v1+)
    "ICM_ON": None,               # None = auto (off under winner-take-all, where every BF is 1); True/False forces
    "BF_MIN": 0.5, "BF_MAX": 6.0,  # BF clamp (tested values)
    "BF_NO_GAIN": 4.0,            # BF when winning the all-in gains no ICM equity (degenerate; then clamped)
    "BF_DEFAULT": 1.25,           # BF if the equal-stack table has no entry for the players left (8-max 50/30/20)
    "BF_PER_DECISION": 4,         # exact BFs computed in one decision at most; the rest: cache or table default
    "CALL_LAM": 2.6,              # re-jam / fit calls: call% /= BF ** CALL_LAM (fit on solved ICM spots)
    "JAM_HERO_EXP": -2.5,         # default jam side: jam% *= mean BF(me, caller) ** this (P2, review's best response)
    "JAM_OPP_EXP": 2.0,           # gated jam side (every possible caller observed jam-tight): * mean BF(caller, me)**this
    "JAM_HERO_EXP_GATED": -1.5,   # ... * mean BF(me, caller) ** this (the tested formula)
    # ---- latency
    "GUARD_NS": 200_000,          # own-clock budget per decision (ns); optional work (BFs) must fit inside it
    #                               (250,000 until the 2026-11-03 hardening: the BF start limit GUARD - 3 x cost left
    #                               worst decisions at 230-380 us of CPU; 200,000 bounds them under 200 us, A/B 0)
    "BF_COST_SAFETY": 3.0,        # a BF starts only if elapsed + this x the BF cost measured at start-up < GUARD_NS
    # ---- chip-EV Nash fits (sng_icm fit_rules.json; refit these for antes or a card variant)
    "JAM_FIT": (3.29, -0.717, -0.917, 1.41, -0.41),   # jam% = a S^b k^c for k >= 2 callers; a1 S^b1 for k = 1
    "CALL_FIT": (2.64, 0.971, -0.582),                # call% = a J^b S^c (J = jammer's range, S = effective bb)
    "DEAD_BASE_BB": 1.5,          # first-in jams use S_eq = S x this / pot (bb): antes and limps widen like depth
    "DEAD_MIN_BB": 0.5,           # ... with the pot floored at this (an almost empty pot: about 3x tighter)
    # ---- preflop (v0+)
    "PF_DEPTH_BB": 20.0,          # jam/fold zone: effective stack in big blinds
    "OPEN_PCT": {7: .12, 6: .14, 5: .17, 4: .20, 3: .26, 2: .36, 1: .45},   # deep first-in opens by players behind
    "OPEN_BB": 2.5,               # deep open size (raise-to, in bb); +OPEN_LIMPER_BB per limper
    "OPEN_LIMPER_BB": 1.0,
    "LATE_SEATS": 3,              # a first-in raise with at most this many players behind is a late (steal) open
    "JAM_MULT": 1.0,              # x every first-in jam range
    "TUNE_BY_PAYOUT": True,       # at construction: when the payout pays two or more places (not winner-take-all),
                                  # apply TUNE_MULTI on top of PARAMS; explicit params / config["bot_params"] keys win.
                                  # ON 2026-11-03 (TUNING_LOG notes, seed 202, 10,000 paired tables, TOP3): AI_ONLY
                                  # +0.040, STRONG +0.058, T6_like +0.036, T1_like +0.055, AI_MIX +0.023 [-0.003, +0.048]
    "TUNE_MULTI": {"PF_DEPTH_BB": 12.0, "JAM_MULT": 1.15},   # B5 + AI-field confirmed under TOP3 / LINEAR; loses under
                                  # winner-take-all, where it is not applied (REPORT notes B5); JAM_MULT 1.3 screened
                                  # worse than 1.15 (AI_MIX -0.014)
    "CALL_MODEL": "equity",       # "equity" (default) or "fit" (tested call_pct(J, S) with S vs the jammer)
    "CALL_EQ_ADJ": 0.0,           # added to the required equity of every all-in call (+ = tighter)
    "CALL_BEHIND_MARGIN": 0.02,   # required equity + this per opponent who can still act after us (overcall risk)
    "CALLER_RANGE_FRAC": 0.6,     # a player who already called the jam holds about this fraction of the jam range
    "CALL_MULT": 1.0,             # "fit" model only: x call range
    "RESTEAL_MULT": 1.2,          # committed vs a non-all-in raise: re-jam if call pct <= C x this (tested)
    "COMMIT_TO_CALL_FRAC": 0.4,   # facing a raise: committed if to_call >= this x my chips (or S <= PF_DEPTH_BB)
    "RAISE_TOP": 0.03,            # deep, facing a raise: 3-bet (jam if S <= FOURBET_JAM_MAX_BB) this top fraction
    "FOURBET_JAM_MAX_BB": 40.0,
    "THREEBET_MULT": 3.0,         # deep 3-bet size: raise to this x the current bet
    "FLAT_TOP": 0.10,             # deep, facing a raise: call this top fraction ...
    "FLAT_MAX_STACK": 0.10,       # ... if the call costs at most this share of my chips
    "BB_ISO_PCT": 0.06,           # BB with limpers and no raise: value-raise (jam when shallow) this top fraction
    "BB_ISO_BB": 4.0,             # deep iso size in bb ...
    "BB_ISO_LIMPER_BB": 1.0,      # ... plus this per limper
    "NEAR_ALLIN_FRAC": 0.25,      # a raise leaving the raiser <= this share of its chips ...
    "NEAR_ALLIN_BB": 1.0,         # ... or <= this many bb counts as a jam
    # ---- postflop (v0+)
    "VALUE_FRAC": 0.70,           # value bet (class >= 2), pot fraction
    "RAISE_MULT": 3.0,            # class 3 facing a bet: raise to this x the bet
    "BIG_CALL_FRAC": 0.4,         # a call is "big" when it costs >= this share of my chips behind
    "CALL2_MAX_POT": 1.0,         # class 2 calls a bet up to this x pot (not big, or BF below BF_CALL2_MAX); the
                                  # pot here already includes the bet, so 1.0 never binds
    "CALL2_MAX_BET_POT": None,    # None = off; else class 2 facing a first bet also needs to_call <= this x the pot
                                  # BEFORE this street's bets (bet-to-pot ratio: 1.0 = a pot-size bet). Left off
                                  # 2026-11-03: 0.75 / 1.0 / 1.5 all within +-0.01 of v2 in every field (TUNING_LOG)
    "CALL1_MAX_POT": 0.35,        # class 1 calls a bet up to this x pot (never big)
    "RAISED_CALL2_MAX_POT": 0.35,  # class 2 facing a RAISE of our bet: call only this cheap (raises are honest)
    "BF_CALL2_MAX": 1.3,          # v1+: big class-2 call only if BF < this
    "BF_RAISE3_MAX": 2.0,         # v1+: big class-3 spot: just call when BF >= this
    "COMMIT_FRAC": 0.45,          # a bet or raise that puts in >= this share of my chips goes all-in instead
    "CBET_AIR_UNKNOWN": 0.0,      # air c-bet frequency vs a seat with no fold evidence (P10: default off)
    "POSTFLOP_ENGINE": False,     # True: hold'em postflop decisions by bot/postflop.py (equity of our hand class vs an
                                  # action-conditioned range, precomputed by tools/make_postflop_tables.py; lookups
                                  # only); False = the hand-class rules below (byte-identical). POSTFLOP_ENGINE notes
    "PE": {                       # postflop engine knobs (a partial dict in params overrides these key by key)
        # Opponent model: stratum s = 0 air, 1 weak pair or 8+ out draw, 2 top pair / overpair, 3 two pair+; width =
        # tight / medium / wide from the typing label. The probabilities below were MEASURED on the AI_MIX + STRONG
        # fields (seed 9002, 300 tables each, the opponents' own hole cards; POSTFLOP_ENGINE notes section 3).
        "FIRST": True, "FACING": True,    # engine decides when checked to us / when facing a bet (False = class rules)
        "REAL": ((0.70, 0.80, 1.0), (0.85, 0.90, 1.0), (0.95, 0.98, 1.0)),
        #   equity realization [hand group][street]: air / overcards / gutshots, 8+ out draws, made pairs or better
        "REF_FRAC": 0.6,                  # W_CALL is for a bet of this x pot; f x pot uses w ** (f / REF_FRAC)
        "W_CALL": (((0.21, 0.80, 1.0, 1.0), (0.49, 0.85, 1.0, 1.0), (0.18, 0.91, 0.97, 1.0)),      # tight
                   ((0.20, 0.71, 1.0, 0.99), (0.38, 0.75, 0.98, 1.0), (0.24, 0.81, 0.96, 0.99)),   # medium
                   ((0.16, 0.94, 1.0, 1.0), (0.42, 0.93, 1.0, 1.0), (0.13, 0.97, 1.0, 0.99))),    # wide
        #   [width][street][s]: P(continue vs a bet); scaled per seat toward its shrunk observed fold rate (v2)
        "W_BET": (((0.07, 0.45, 0.97, 1.0), (0.02, 0.15, 0.95, 1.0), (0.02, 0.28, 0.95, 1.0)),
                  ((0.17, 0.41, 0.90, 0.95), (0.04, 0.19, 0.84, 0.92), (0.05, 0.29, 0.81, 0.90)),
                  ((0.62, 0.81, 0.91, 0.91), (0.55, 0.65, 0.75, 0.76), (0.44, 0.58, 0.61, 0.69))),
        #   [style][street][s]: P(bet when it can); also the shape of a bettor's range. Styles: tight (folder, nit),
        #   medium (every other seat: the measured non-aggressive bots), wide (maniac, loose: the measured
        #   station / lag / pushfold maniac / ai_aggro group)
        "CBET_W": (0.40, 0.60),           # floors of the air / weak bet weights when the bettor c-bets the flop
        "W_RAISE": ((0.01, 0.06, 0.13, 0.70), (0.01, 0.07, 0.15, 0.55), (0.02, 0.12, 0.22, 0.55)),
        #   [street][s]: P(raise our bet); also the shape of a raiser's range
        "CONT_RAISE": (0.50, 0.95, 0.98, 1.0),   # P(continue when we raise its bet)
        "OPP_BET_FRAC": 0.6,              # its bet size (x pot) after our check, in the check EV when we act first
        "SIZE_EXP": 0.5,                  # facing f x pot > REF_FRAC: air / weak weights x (REF_FRAC / f) ** this
        "K_OBS": 6.0,                     # pseudo-observations of the model fold rate when scaling to observed folds
        "STRONG_FLOOR": 0.95,             # two pair+ continues at least this often whatever the observed fold rate
        "SIZES_DRY": (0.33, 0.5, 0.75),   # heads-up bet sizes (x pot) by EV, rainbow / two-tone and disconnected ...
        "SIZES_WET": (0.5, 0.75, 1.0),    # ... other boards. Top pair+ always bets (VALUE_FRAC_CALLER vs a typed
        #                                   caller, VALUE_FRAC multiway); weak / middle pairs without a draw check
        "BET_MARGIN": 0.02,               # a bet (raise) must beat checking (calling) by this x pot
        "SEMI_MAX_OPP": 1,                # semi-bluffs (8+ out draws, at most a middle pair) with at most this many opps
        "BLUFF_FREQ": 0.5,                # pure bluffs: heads-up, dry board, a seat with fold evidence, this often ...
        "BLUFF_NOBLOCK": 0.4,             # ... x this without an A/K/Q above the board (blocker)
        "BLUFF_MIN_OBS": 3, "BLUFF_FOLD_T": 0.45,   # fold evidence: shrunk postflop fold rate >= T over >= MIN bets
        "RAISE_MIN_EQ": (0.60, 0.60, 0.65),   # heads-up: a value raise is considered from this equity vs the range
        "RAISE_EQ_MULTI": 0.75,           # multiway: raise from this equity (x the others' calling ranges)
        "RERAISE_EQ": 0.85,               # re-raise a raise of our bet from this equity vs the raiser's range
        "CALL_MARGIN": 0.0,               # required equity of a call = pot odds (BF-weighted when big) + this
    },
    # ---- v2 typing (shrinkage: estimate = (hits + k * mean) / (n + k))
    "PRIOR": {"vol": (0.30, 12), "jam": (0.05, 10), "calljam": (0.35, 6), "post_call": (0.45, 8),
              "flop_fold": (0.30, 6), "late_open": (0.25, 8)},
    "OE_PRIOR": 0.5,              # jam-call observed/expected ratio shrinkage, in expected-call units
    "MIN_OBS_TYPE": 6,            # preflop decisions before a seat gets any type other than folder ("unknown")
    "MIN_OBS_FOLDER": 8,          # preflop decisions with 0 voluntary chips before a seat is a folder
    "T_MANIAC_JAM": 0.25,         # shrunk first-in jam rate
    "T_CALLER_CALLJAM": 0.50, "MIN_JAMS_FACED": 3,   # shrunk jam-call rate over >= 3 jams faced
    "T_POST_CALL": 0.60, "MIN_POST_FACED": 5,         # or shrunk postflop call rate over >= 5 bets faced
    "T_LOOSE_VOL": 0.45,          # shrunk voluntary rate
    "T_NIT_VOL": 0.12, "MIN_OBS_NIT": 30,
    "T_STEAL_LATE": 0.40, "MIN_LATE_OPP": 8,          # stealer: shrunk late first-in open rate
    "OE_TIGHT": 0.6, "MIN_JAMS_FACED_TIGHT": 4,       # jam-tight: shrunk calls/expected-Nash-calls <= this
    "FOLDER_STEAL": True,         # min-raise any two when every seat behind is a folder (E09)
    "LIGHT_JAM_CAP": 0.35,        # max jam range with a caller, loose or maniac seat behind (E05)
    "CALLER_CAP_JAM_EVIDENCE": True,    # True: a "caller" seat caps first-in jams only if it was typed from jam calls
                                  # (shrunk jam-call rate >= T_CALLER_CALLJAM over >= MIN_JAMS_FACED) or would also
                                  # be "loose"; a caller typed from postflop calls alone no longer caps. ON 2026-11-03
                                  # (TUNING_LOG, seed 202): STRONG +0.009, T6_like +0.013, T1_like +0.009, AI_ONLY
                                  # +0.027, AI_MIX +0.011 [-0.007, +0.030]; False = the built v2 rule
    "SB_WIDEN": 1.6,              # x jam range when the only seat behind is a folder or folds to jams (P1)
    "SB_WIDEN_MARGIN": 0.05,      # widen only if the BB's shrunk jam-call rate < break-even(S) - this
    "SB_WIDEN_BE": (0.33, 0.018, 0.10),   # break-even jam-call rate of the widen: 0.33 at <= 10 bb, minus 0.018 per bb
                                          # above 10 (24% at 15 bb, review's sb_widen_ev), floored at 0.10
    "JAMFOLD_WIDEN": 1.6,         # x jam range when every seat behind is jam-tight (E15, payout-independent)
    "J_FLOOR": {"maniac": 0.50, "loose": 0.35, "stealer": 0.40},   # range floors of typed raisers (no caller floor)
    "REJAM_PRIOR": 0.06,          # P5: range of a re-jam over a raise (unless the seat is a typed maniac)
    "OPEN_PRIOR": None,           # v2 range prior of a first-in opener by players behind (None = OPEN_PCT)
    "OPEN_K_UNKNOWN": 3,          # v2: players-behind assumed for a raiser that was not the first-in opener
    "LATE_OPEN_FLOOR": 0.35,      # P6: range floor of a small late open (<= LATE_SEATS behind, raise <=
                                  #     BB_DEFEND_MAX_RAISE_BB)
    "BB_DEFEND_PCT": 0.30,        # P6: BB flats this top fraction (call order) vs a late small raise ...
    "BB_DEFEND_MAX_RAISE_BB": 3.0,  # ... of at most this many bb ...
    "BB_DEFEND_MAX_FRAC": 0.15,   # ... when the call costs at most this share of my chips
    "VALUE_FRAC_CALLER": 1.00,    # value bet vs a typed caller (E16)
    "CBET_FRAC": 0.33,            # air c-bet size vs a seat with fold evidence (E13)
    "CBET_FOLD_T": 0.40, "CBET_MIN_OBS": 3,           # shrunk fold-to-flop-bet rate over >= 3 flop bets faced
    "LIGHT_CALL_TYPES": ("maniac", "stealer"),        # call their postflop bets one class lighter (E11)
    # ---- v2 range model (RANGE_MODEL notes): per-seat Beta-binomial posteriors of range WIDTH from preflop action
    #      evidence only (never postflop calls), from COMPLETED hands (the current hand's actions are committed at
    #      the next hand start, so the jam we face is not its own evidence). Each seat's hits over its
    #      opportunities are compared with the prior's expected hits E at those same spots; with a Beta prior whose
    #      mean is the prior rate and whose alpha (pseudo-hits) is a, the posterior/prior ratio is
    #      r = (hits + a) / (E + a) (prior strength in opportunities = a / prior rate); r scales the prior range at
    #      the current spot, clamped to [prior / RM_CAP, prior x RM_CAP].
    "RANGE_MODEL": True,          # ON 2026-11-04 (RANGE_MODEL notes, seed 4102, 5,000 paired tables, TOP3 TURBO):
                                  # AI_MIX +0.055 [+0.018, +0.093], STRONG +0.036 [+0.001, +0.072], T6_like +0.054
                                  # [+0.020, +0.088]; False = v2 as before (the NLHE golden identity test pins False)
    "RM_A_JAM": 3.0,              # Beta alpha (pseudo-hits): first-in jams in the jam zone -> jam ranges
    "RM_A_OPEN": 4.0,             # ... first-in raises (no limpers, deep enough) -> open ranges
    "RM_A_3BET": 2.0,             # ... re-raises facing one non-all-in raise -> re-jam / 3-bet ranges
    "RM_A_CALL": 2.0,             # ... jam calls vs expected Nash calls -> range of a seat that called a jam
    "RM_CAP": 2.0,                # posterior range within [prior / this, prior x this]
    "RM_JAM_MAX_BB": 25.0,        # a first-in decision is jam evidence only at effective depth <= this (bb)
    "RM_OPEN_MIN_BB": 15.0,       # ... open evidence only at effective depth >= this (bb; shallower = jam/fold zone)
    "RM_RULE_EXP": 1.0,           # our flat / 3-bet / BB-defence fractions x (posterior / prior raiser range) ** this
    "RM_BB_DEFEND_MAX": 0.50,     # ... with the BB defence fraction capped at this
    # ---- ICM push/fold equilibrium tables (ICM_NASH notes; bot/icm_nash_tables.py from
    #      tools/solve_icm_pushfold.py). With 2-4 players left, no antes, ICM on (never under winner-take-all) and a
    #      payout the tables were solved for (TOP3, LINEAR), first-in jams and calls of a first-in jam (and over-calls
    #      after one caller) come from the equilibrium sets at the nearest stack bucket instead of the fitted formula x
    #      bubble-factor heuristic. v2 typing still applies on top of the jam set (see ICM_NASH notes section 4).
    "ICM_NASH": True,             # ON 2026-11-05 (ICM_NASH notes section 6, seed 5102, 20,000 paired tables TOP3 /
                                  # 10,000 LINEAR): AI_MIX +0.023 [+0.009, +0.037], STRONG +0.007 [-0.003, +0.016],
                                  # AI_ONLY +0.016 [-0.001, +0.034], icm_pro +0.015, ai_icm_plan +0.025, ai_poker_pro
                                  # +0.052 (TOP3); every LINEAR lower bound >= -0.003. False = the BF heuristic
    "ICM_NASH_PARTS": "jam+call",  # which decisions use the tables: "jam" (first-in jams), "call" (jam calls and
                                  # over-calls), or both
    "ICM_NASH_MAX_BB": 20.0,      # only when our effective stack (bb) is at most this
    "ICM_NASH_CAP_BB": 30.0,      # ... and no live stack exceeds this (bigger stacks would sit in the 20 bb bucket)
    "ICM_NASH_CALL_DEV": 1.25,    # v2: when the jammer's estimated range (typing / range model, anchored on its
                                  # equilibrium jam range) is more than this factor wider or narrower than that range,
                                  # the call is priced by call_jam (equity vs the estimate) instead of the call set
}

RANKS = "23456789TJQKA"
STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# ===================================== data: classes, percentiles, equity table =====================================
try:
    from . import bot_tables as _T
except ImportError:   # loaded by file path (kit runner) or flat: load the sibling file explicitly
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _name = "nolimit_bot_tables"
    _T = _sys.modules.get(_name)
    if _T is None:
        _spec = _ilu.spec_from_file_location(_name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                                                   "bot_tables.py"))
        _T = _ilu.module_from_spec(_spec)
        _sys.modules[_name] = _T
        _spec.loader.exec_module(_T)


def combos(lab):
    return 6 if len(lab) == 2 else (4 if lab[2] == "s" else 12)


def percentiles(order):
    """Midpoint percentile of each class in a best-first order: (combos of better classes + half its own) / 1326."""
    out, acc = {}, 0
    for lab in order:
        c = combos(lab)
        out[lab] = (acc + c / 2.0) / 1326.0
        acc += c
    return out


def label_of(r1, s1, r2, s2):
    if r1 < r2:
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if s1 == s2 else "o")


LABELS = _T.EQ_LABELS.split()                     # class index -> label (equity table row order)
CLASS_ID = {lab: i for i, lab in enumerate(LABELS)}
_pp = percentiles(_T.PUSH_ORDER.split())
_cp = percentiles(_T.CALL_ORDER.split())
PUSH_PCT = [_pp[lab] for lab in LABELS]           # 0 = best
CALL_PCT = [_cp[lab] for lab in LABELS]
EQ_GRID = list(_T.EQ_GRID)
EQ_ROWS = [list(r) for r in _T.EQ_TABLE]
CLASS_OF = [0] * 2704                             # card pair (c1 * 52 + c2, kit encoding) -> class index
for _a in range(52):
    for _b in range(52):
        if _a != _b:
            CLASS_OF[_a * 52 + _b] = CLASS_ID[label_of(_a >> 2, _a & 3, _b >> 2, _b & 3)]
_G0, _GN = EQ_GRID[0], EQ_GRID[-1]


def eq_vs(cls, J):
    """All-in equity of class `cls` against the top-J fraction of hands (push order), linear in J between grid
    points; J below the first grid point uses it (slightly optimistic vs ultra-tight ranges)."""
    row = EQ_ROWS[cls]
    if J <= _G0:
        return row[0]
    if J >= _GN:
        return row[-1]
    i = bisect.bisect_left(EQ_GRID, J)
    x0, x1 = EQ_GRID[i - 1], EQ_GRID[i]
    return row[i - 1] + (row[i] - row[i - 1]) * (J - x0) / (x1 - x0)


def jam_pct(S, k, f=PARAMS["JAM_FIT"]):
    """Chip-EV Nash first-in jam range (fraction of hands); S = effective bb, k = players who can call;
    f = PARAMS["JAM_FIT"] = (a, b, c, a1, b1)."""
    S = S if S > 1.0 else 1.0
    if k <= 1:
        v = f[3] * S ** f[4]
    else:
        v = f[0] * S ** f[1] * k ** f[2]
    return v if v < 1.0 else 1.0


def call_pct(J, S, f=PARAMS["CALL_FIT"]):
    """Chip-EV Nash big-blind call range against a jam range J at effective depth S; f = PARAMS["CALL_FIT"]."""
    v = f[0] * (J if J > 0.01 else 0.01) ** f[1] * (S if S > 1.0 else 1.0) ** f[2]
    return v if v < 1.0 else 1.0


def by_k(table, k):
    """Value of a {players behind: value} table with k clamped to its key range (k = 0 uses the smallest key)."""
    lo, hi = min(table), max(table)
    return table[lo if k < lo else hi if k > hi else k]


def equiv_depth(S, pot_bb, P):
    """Dead-money equivalent depth for first-in jams: a blinds-only pot (DEAD_BASE_BB) leaves S unchanged; antes
    and limps make the jam behave like a shallower one (P11: the solver widens jams x1.24 SB to x1.55 UTG at
    10 bb with 0.125 bb antes; this gives x1.23 and x1.44)."""
    floor = P["DEAD_MIN_BB"]
    return S * P["DEAD_BASE_BB"] / (pot_bb if pot_bb > floor else floor)


# ===================================== postflop hand class =====================================
def _straight_high(mask):
    """Highest straight rank (index of the top card) in a 13-bit rank mask, -1 if none; the wheel counts as 3."""
    for top in range(12, 3, -1):
        if (mask >> (top - 4)) & 31 == 31:
            return top
    if mask & 0x100F == 0x100F:
        return 3
    return -1


def hand_class(h1, h2, board):
    """0 air, 1 weak pair or draw, 2 top pair or overpair, 3 strong (two pair using both hole cards, set, trips with
    a hole card, a straight or flush using a hole card, full house or quads using a hole card). Kit card ints:
    rank = c >> 2 (0 = deuce .. 12 = ace), suit = c & 3. Board-only made hands do not count."""
    r1, r2 = h1 >> 2, h2 >> 2
    s1, s2 = h1 & 3, h2 & 3
    bcnt = [0] * 13
    bsuit = [0, 0, 0, 0]
    bmask = 0
    top = 0
    for c in board:
        r = c >> 2
        bcnt[r] += 1
        bsuit[c & 3] += 1
        bmask |= 1 << r
        if r > top:
            top = r
    nb = len(board)
    # flush using a hole card
    fs = -1
    for s in (s1, s2):
        if bsuit[s] + (s1 == s) + (s2 == s) >= 5:
            fs = s
    if fs >= 0:
        if bsuit[fs] >= 4:   # four-flush board: only a high flush card is strong
            hi = max(r for r, s in ((r1, s1), (r2, s2)) if s == fs)
            return 3 if hi >= 10 else 1
        return 3
    # straight using a hole card
    amask = bmask | (1 << r1) | (1 << r2)
    sh = _straight_high(amask)
    if sh >= 0 and sh > _straight_high(bmask):
        return 3
    bmax = max(bcnt)
    if r1 == r2:                                   # pocket pair
        if bcnt[r1] >= 1:
            return 3                               # set or quads
        if bmax >= 3:
            return 2                               # full house made with the board's trips (weak-ish)
        if r1 > top:
            return 2                               # overpair (possibly with a board pair)
        cls = 1                                    # underpair / middle pocket pair
    else:
        c1, c2 = bcnt[r1], bcnt[r2]
        if c1 and c2:
            return 3                               # two pair with both hole cards (or a full house)
        if c1 >= 2 or c2 >= 2:
            return 3                               # trips (or quads) with a hole card
        if c1 or c2:
            pr = r1 if c1 else r2
            if pr == top:
                return 2                           # top pair (also on a paired board)
            cls = 1                                # second pair or lower
        else:
            cls = 0
    if cls == 0 and nb < 5:
        # draws: four to a flush with a hole card, or an open-ended straight draw using a hole card
        for s in (s1, s2):
            if bsuit[s] + (s1 == s) + (s2 == s) == 4:
                return 1
        hm = (1 << r1) | (1 << r2)
        for lo in range(0, 9):                     # windows 2345 .. TJQK are open at both ends (A2345 counts)
            w = 15 << lo
            if amask & w == w and hm & w and bmask & w != w:
                return 1
    return cls


# Postflop engine (bot/postflop.py, PARAMS["POSTFLOP_ENGINE"]): equity-vs-range lookups. Loaded at import (its
# tables decode in about a millisecond); None when it or its tables are missing, and the hand-class path decides.
def _load_postflop():
    try:
        from . import postflop as pf
    except ImportError:
        import importlib.util as _ilu4
        import os as _os4
        import sys as _sys4
        name = "nolimit_bot_postflop"
        pf = _sys4.modules.get(name)
        if pf is None:
            sp = _ilu4.spec_from_file_location(name, _os4.path.join(_os4.path.dirname(_os4.path.abspath(__file__)),
                                                                    "postflop.py"))
            pf = _ilu4.module_from_spec(sp)
            _sys4.modules[name] = pf
            try:
                sp.loader.exec_module(pf)
            except BaseException:
                del _sys4.modules[name]
                raise
    return pf if pf.DATA is not None else None


try:
    _PF = _load_postflop()
except BaseException:         # noqa: BLE001 - no engine: the hand-class path
    _PF = None


# ===================================== variant seam: swapped tables, any hole-card count =====================
# The built-in NLHE tables above stay in force unless tables_loader accepts a rebuilt tables module (env
# NOLIMIT_TABLES or bot/variant_tables.py; see tables_loader.py). Then LABELS ... EQ_ROWS come from
# variant/bridge.brain_names (same meaning, the variant's class space) and _SEAM holds its HandSpace. With _SEAM None
# and two hole cards every lookup is the pre-integration one (tests/test_nlhe_identity.py: 20,000 decisions equal).
_NL_CLASS_OF, _NL_PUSH_PCT = CLASS_OF, PUSH_PCT        # NLHE 2-card strength, kept for the keep policy
_SEAM = None
TABLES_INFO = "tables: built-in NLHE (bot_tables.py, 169 classes)"


def _load_tables_loader():
    try:
        from . import tables_loader as tl
        return tl
    except ImportError:
        import importlib.util as _ilu2
        import os as _os2
        import sys as _sys2
        name = "nolimit_bot_tables_loader"
        tl = _sys2.modules.get(name)
        if tl is None:
            sp = _ilu2.spec_from_file_location(name, _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)),
                                                                    "tables_loader.py"))
            tl = _ilu2.module_from_spec(sp)
            _sys2.modules[name] = tl
            try:
                sp.loader.exec_module(tl)
            except BaseException:
                del _sys2.modules[name]
                raise
        return tl


try:
    TABLES_LOADER = _load_tables_loader()
    _ACTIVE = TABLES_LOADER.load()
except BaseException:         # noqa: BLE001 - no loader, no swap: the built-in tables
    TABLES_LOADER = None
    _ACTIVE = None
if _ACTIVE is not None and _ACTIVE.kind == "variant":
    try:
        _nm = _ACTIVE.names
        _labels, _cid = list(_nm["LABELS"]), dict(_nm["CLASS_ID"])
        _push, _call = list(_nm["PUSH_PCT"]), list(_nm["CALL_PCT"])
        _grid, _rows = list(_nm["EQ_GRID"]), [list(r) for r in _nm["EQ_ROWS"]]
        LABELS, CLASS_ID, PUSH_PCT, CALL_PCT, EQ_GRID, EQ_ROWS = _labels, _cid, _push, _call, _grid, _rows
        _G0, _GN = EQ_GRID[0], EQ_GRID[-1]
        if _ACTIVE.use_fits:
            PARAMS["JAM_FIT"] = tuple(float(x) for x in _nm["JAM_FIT"])
            PARAMS["CALL_FIT"] = tuple(float(x) for x in _nm["CALL_FIT"])
        _SEAM = _ACTIVE.hs
        TABLES_INFO = TABLES_LOADER.INFO.get("line", "tables: variant")
    except BaseException:     # noqa: BLE001 - a half-applied swap is worse than none: restore the NLHE tables
        LABELS = _T.EQ_LABELS.split()
        CLASS_ID = {lab: i for i, lab in enumerate(LABELS)}
        PUSH_PCT, CALL_PCT = _NL_PUSH_PCT, [_cp[lab] for lab in LABELS]
        EQ_GRID, EQ_ROWS = list(_T.EQ_GRID), [list(r) for r in _T.EQ_TABLE]
        _G0, _GN = EQ_GRID[0], EQ_GRID[-1]
        _SEAM = None
        TABLES_INFO = "tables: built-in NLHE (swap failed)"
# Variant-aware postflop class (POSTFLOP_VARIANT notes). _POST is None with the built-in tables and with swapped
# tables of a plain hold'em spec (52-card ranking, 2 hole cards, no keep or must-use rule): postflop() then runs the
# pre-integration hand_class lines. Otherwise variant/bridge.postflop_classifier(spec): the variant's straights (short
# deck A-6-7-8-9), its hand rule (must use k of the hole cards, the best keepable subset) and its street count, on
# hand_class's 0..3 scale. NOLIMIT_POSTFLOP_VARIANT=0 switches it off (A/B, event-day rollback).
_POST = None
POST_INFO = "postflop: hand_class (52-card hold'em)"
if _SEAM is not None:
    try:
        import os as _os3
        if _os3.environ.get("NOLIMIT_POSTFLOP_VARIANT", "1").strip() not in ("0", "off", "no", "false"):
            _POST = TABLES_LOADER._submodule("bridge").postflop_classifier(_ACTIVE.spec)
            if _POST is not None:
                POST_INFO = "postflop: " + _POST.describe
        else:
            POST_INFO = "postflop: hand_class (NOLIMIT_POSTFLOP_VARIANT off)"
    except BaseException:     # noqa: BLE001 - no classifier: hand_class, as before
        _POST = None
_MEDIAN_CLS =min(range(len(PUSH_PCT)), key=lambda i: (abs(PUSH_PCT[i] - 0.5), i))


def classify(hole):
    """Preflop class (row of the active tables) of ANY dealt hand; never raises.
    Built-in NLHE tables: 2 cards = CLASS_OF; 3+ cards (a toss or keep game before its choice) = the pair with the
    best push percentile. Swapped tables: the HandSpace class of a hand of its dealt size (3-card toss classes for the
    3 dealt cards); more cards = the best subset by push percentile; fewer = HandSpace.class_of_safe (the class of
    the dealt hand that keeps these cards). Anything that fails: the median class, never the best one."""
    try:
        n = len(hole)
        hs = _SEAM
        if hs is None:
            if n == 2:
                return CLASS_OF[hole[0] * 52 + hole[1]]
            if n > 2:
                best, bp = -1, 2.0
                for i in range(n - 1):
                    hi = hole[i] * 52
                    for j in range(i + 1, n):
                        c = CLASS_OF[hi + hole[j]]
                        if PUSH_PCT[c] < bp:
                            best, bp = c, PUSH_PCT[c]
                if best >= 0:
                    return best
        else:
            if n == hs.n:
                ci = hs.class_of(hole)
                if ci is not None and ci >= 0:
                    return ci
            elif n > hs.n:
                best, bp = -1, 2.0
                for sub in _combinations(hole, hs.n):
                    c = hs.class_of(sub)
                    if c is not None and c >= 0 and PUSH_PCT[c] < bp:
                        best, bp = c, PUSH_PCT[c]
                if best >= 0:
                    return best
            return hs.class_of_safe(tuple(hole), default=_MEDIAN_CLS)
    except Exception:         # noqa: BLE001
        pass
    return _MEDIAN_CLS


def post_class(hole, board, boards=None):
    """Postflop class 0..3 of any hole-card count and any number of boards; never raises. 2 cards, one board =
    hand_class. More cards: the best pair (the 2 kept cards are what a toss game plays after the toss). Several
    boards: the floor of the mean class (a pot split per board: strong on one board and air on the other is a weak
    pair overall)."""
    try:
        bl = boards if boards else (board,)
        n = len(hole)
        tot = 0
        for b in bl:
            if n == 2:
                c = hand_class(hole[0], hole[1], b)
            else:
                c = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        x = hand_class(hole[i], hole[j], b)
                        if x > c:
                            c = x
            tot += c
        return tot // len(bl)
    except Exception:         # noqa: BLE001
        pass
    try:
        return hand_class(hole[0], hole[1], board)
    except Exception:         # noqa: BLE001
        return 0


def post_class_variant(hole, board, boards=None, street=None):
    """Postflop class 0..3 with the active variant's classifier _POST (any hole count, the variant's ranking and hand
    rule); several boards: the floor of the mean class, as post_class. Falls back to post_class on any failure."""
    try:
        f = _POST
        if boards:
            tot = 0
            for b in boards:
                tot += f(hole, b, street)
            return tot // len(boards)
        return f(hole, board, street)
    except Exception:         # noqa: BLE001
        return post_class(hole, board, boards)


def keep_policy(view):
    """Board-aware keep for keep/discard/toss requests that keep exactly 2 cards (variant layer view, see
    variant/defaults.py): score = 10 x postflop class of the kept pair on the board the answer leaves (a tossed card
    joins the board) + 4 x (1 - NLHE push percentile of the pair). Without a board this is the NLHE preflop order.
    Returns an answer in the view's format, or None (the layer's static default then answers). About 10-25 us."""
    kind = view.get("kind")
    if kind not in ("discard", "keep", "toss"):
        return None
    hole = view.get("hole") or ()
    n = len(hole)
    k = view.get("k")
    k = k if isinstance(k, int) and not isinstance(k, bool) else (2 if kind == "keep" else 1)
    keep_n = k if kind == "keep" else n - k
    if keep_n != 2 or n < 3:
        return None
    board = list(view.get("board") or ())
    best, bs = None, None
    for a in range(n - 1):
        for b in range(a + 1, n):
            ka, kb = hole[a], hole[b]
            out = [hole[i] for i in range(n) if i != a and i != b]
            bd = board + out if kind == "toss" else board
            s = 4.0 * (1.0 - _NL_PUSH_PCT[_NL_CLASS_OF[ka * 52 + kb]])
            if bd:
                s += 10.0 * hand_class(ka, kb, bd)
            if bs is None or s > bs:
                best, bs = (a, b), s
    keep_idx = best
    disc = [i for i in range(n) if i not in keep_idx]
    fmt = view.get("response")
    if fmt in ("index", "index_or_none"):
        return disc[0]
    if fmt == "keep_indices":
        return list(keep_idx)
    if fmt == "discard_indices":
        return disc
    if fmt == "keep_cards":
        return [hole[i] for i in keep_idx]
    if fmt == "discard_cards":
        return [hole[i] for i in disc]
    if fmt == "card":
        return hole[disc[0]]
    return None


# ===================================== revealed discards: reader model, leak-aware keep (DISCARD_LEAK notes) ======
# Reader model (rank level, suits ignored): a player dealt more cards than it keeps keeps the 2 cards that are
# strongest in the NLHE push order and gives up the rest; a discard that is not the model's choice has likelihood
# REVEAL_EPS relative to one that is. The posterior over a player's kept rank pair given its revealed discards D is
#   w(x, y) = combos(x, y | unseen cards) x (1 if S2[x][y] >= S2[x][d] and S2[y][d] for every d in D else EPS)
# S2 = 1 - NLHE push percentile of the offsuit (or pair) label: 1 = AA, 0 = the worst hand.
_RX = range(13)
S2 = [[1.0 - _NL_PUSH_PCT[_NL_CLASS_OF[(x << 2) * 52 + ((y << 2) | 1)]] for y in _RX] for x in _RX]   # NLHE tables
_RANK_PAIRS = tuple((x, y) for x in _RX for y in range(x, 13))         # 91 kept rank pairs, x <= y
_CARD_RANK = {r: i for i, r in enumerate(RANKS)}
_CARD_RANK.update({r.lower(): i for i, r in enumerate(RANKS)})
_CARD_RANK["10"] = 8
_CARD_SUIT = {"c": 0, "d": 1, "h": 2, "s": 3, "C": 0, "D": 1, "H": 2, "S": 3, "♣": 0, "♦": 1, "♥": 2, "♠": 3}


def card_int(c):
    """Event card (e.g. 'Ah', '10h', an int 0..51) -> int in the adapter's numbering; None if unreadable."""
    try:
        if isinstance(c, int) and not isinstance(c, bool):
            return c if 0 <= c < 52 else None
        s = str(c).strip()
        return (_CARD_RANK[s[:-1]] << 2) | _CARD_SUIT[s[-1]]
    except Exception:         # noqa: BLE001
        return None


def reveal_lik(dr, eps):
    """Reader-model likelihood of each kept rank pair (index order of _RANK_PAIRS) given discarded ranks dr."""
    out = []
    for x, y in _RANK_PAIRS:
        s = S2[x][y]
        ok = True
        for d in dr:
            if S2[x][d] > s or S2[y][d] > s:
                ok = False
                break
        out.append(1.0 if ok else eps)
    return out


def reveal_shift_prior(dr, eps):
    """Strength a reader infers from discarded ranks dr on a full deck: E[S2 kept | dr] - E[S2 kept] (no discards
    seen). The leak of giving up those cards; about 0.03 for a deuce up to 0.26 for a nine at EPS 0.1."""
    cnt = [4] * 13
    for d in dr:
        cnt[d] -= 1
    lik = reveal_lik(dr, eps)
    w0 = e0 = w1 = e1 = 0.0
    for i, (x, y) in enumerate(_RANK_PAIRS):
        n = cnt[x] * (cnt[x] - 1) // 2 if x == y else cnt[x] * cnt[y]
        if n <= 0:
            continue
        s = S2[x][y]
        w0 += n
        e0 += n * s
        wl = n * lik[i]
        w1 += wl
        e1 += wl * s
    return (e1 / w1 if w1 > 0 else e0 / w0) - e0 / w0


_LEAK_CACHE = {}


def leak_of(cards, eps):
    """Leak (reveal_shift_prior, floored at 0) of giving up these cards; cached by (ranks, eps)."""
    key = (tuple(sorted(c >> 2 for c in cards)), eps)
    v = _LEAK_CACHE.get(key)
    if v is None:
        v = reveal_shift_prior(key[0], eps)
        v = v if v > 0.0 else 0.0
        if len(_LEAK_CACHE) < 4096:
            _LEAK_CACHE[key] = v
    return v


for _r in _RX:                    # single-card leaks at the default EPS, so no request computes one inside act()
    leak_of((_r << 2,), PARAMS["REVEAL_EPS"])


def reveal_board(hole, board, dead):
    """The opponents' kept rank pairs on this board: [(pair index, combos left, rank-level class)] for pairs with
    combos left. Rank-level class on hand_class's scale without flushes or draws: set / trips / two pair / a straight
    using a hole card = 3, overpair or top pair = 2, a lower pair = 1, else 0. dead = revealed cards that are not on
    the board (muck variants). About 25 us; cached per board by the caller."""
    cnt = [4] * 13
    for c in hole:
        cnt[c >> 2] -= 1
    bcnt = [0] * 13
    bmask = 0
    top = 0
    for c in board:
        r = c >> 2
        cnt[r] -= 1
        bcnt[r] += 1
        bmask |= 1 << r
        if r > top:
            top = r
    for c in dead:
        cnt[c >> 2] -= 1
    bmax = max(bcnt) if board else 0
    bsh = _straight_high(bmask)
    miss = []                                      # rank masks a kept pair must cover to make a better straight
    for t in range(12, 3, -1):
        if t > bsh:
            m = (31 << (t - 4)) & ~bmask
            if bin(m).count("1") <= 2:
                miss.append(m)
    if 3 > bsh:
        m = 0x100F & ~bmask
        if bin(m).count("1") <= 2:
            miss.append(m)
    rows = []
    i = -1
    for x, y in _RANK_PAIRS:
        i += 1
        cx = cnt[x]
        if x == y:
            n = cx * (cx - 1) // 2
            if n <= 0:
                continue
            c = 3 if bcnt[x] else (2 if (bmax >= 3 or x > top) else 1)
        else:
            n = cx * cnt[y]
            if n <= 0:
                continue
            bx, by = bcnt[x], bcnt[y]
            if (bx and by) or bx >= 2 or by >= 2:
                c = 3
            elif by:
                c = 2 if y == top else 1
            elif bx:
                c = 2 if x == top else 1
            else:
                c = 0
        if c < 3 and miss:
            hm = (1 << x) | (1 << y)
            for m in miss:
                if m & hm == m:
                    c = 3
                    break
        rows.append((i, n, c))
    return rows


def reveal_delta_rows(rows, lik, cls):
    """Shift of a bettor's ahead-share against our class `cls` from its revealed discards: A(read) - A(no read),
    with A = P(its class > cls) + 0.5 P(== cls) over reveal_board rows; lik = reveal_lik of its discards."""
    w0 = [0.0, 0.0, 0.0, 0.0]
    w1 = [0.0, 0.0, 0.0, 0.0]
    for i, n, c in rows:
        w0[c] += n
        w1[c] += n * lik[i]
    t0 = w0[0] + w0[1] + w0[2] + w0[3]
    t1 = w1[0] + w1[1] + w1[2] + w1[3]
    if t0 <= 0 or t1 <= 0:
        return 0.0
    a0 = a1 = 0.0
    for c in range(cls, 4):
        f = 0.5 if c == cls else 1.0
        a0 += f * w0[c]
        a1 += f * w1[c]
    return a1 / t1 - a0 / t0


def reveal_delta(hole, board, dead, lik, cls):
    """reveal_delta_rows on reveal_board(hole, board, dead) (one call; the brain caches the board rows)."""
    return reveal_delta_rows(reveal_board(hole, board, dead), lik, cls)


class RevealTracker:
    """Cards each seat revealed this hand (public discard / toss events), fed by Brain.observe. Inert when no such
    event arrives (plain hold'em, or a platform that does not call observe)."""
    __slots__ = ("hand_no", "cards", "lik", "cache", "rows")

    def __init__(self):
        self.hand_no = None
        self.cards = {}           # seat -> [card ints]
        self.lik = {}             # seat -> reveal_lik of its discards (computed on first use)
        self.cache = {}           # (seat, board length, our class) -> reveal_delta
        self.rows = {}            # board length -> reveal_board rows

    def add(self, hand_no, seat, cards):
        if hand_no != self.hand_no:
            self.hand_no = hand_no
            self.cards = {}
            self.lik = {}
            self.cache = {}
        self.rows = {}
        lst = self.cards.setdefault(seat, [])
        for c in cards:
            ci = card_int(c)
            if ci is not None and ci not in lst:
                lst.append(ci)
        self.lik.pop(seat, None)
        self.cache = {}


def keep_leak(view, P=None):
    """Leak-aware keep (LEAK_PENALTY): the variant layer's static keep minus LEAK_LAMBDA x leak_of(cards given up),
    for requests that keep exactly 2 cards; None otherwise (the layer's static default then answers)."""
    P = P or PARAMS
    kind = view.get("kind")
    if kind not in ("discard", "keep", "toss"):
        return None
    hole = view.get("hole") or ()
    n = len(hole)
    k = view.get("k")
    k = k if isinstance(k, int) and not isinstance(k, bool) else (2 if kind == "keep" else 1)
    keep_n = k if kind == "keep" else n - k
    if keep_n != 2 or n < 3 or _VDEFAULTS is None:
        return None
    lam = float(P["LEAK_LAMBDA"])
    eps = P["REVEAL_EPS"]
    return _VDEFAULTS.keep_with_penalty(view, lambda kept, out: lam * leak_of(out, eps))


try:
    _VDEFAULTS = TABLES_LOADER._submodule("defaults") if TABLES_LOADER is not None else None
except BaseException:         # noqa: BLE001 - no layer: keep_leak answers None (the static default)
    _VDEFAULTS = None

# Per-instance keep settings for rehearsal A/Bs only: the shell passes one module-level policy, so a Brain whose
# params override LEAK_PENALTY / KEEP_BOARD_AWARE registers its params under its dealt hole cards at its decisions,
# and keep_dispatch looks them up by the request's hole cards. Production (no override) never registers.
_KEEP_OWNER = {}


def keep_dispatch(view):
    """The keep policy the shell passes to the variant layer: leak-aware, board-aware or None (static)."""
    try:
        P = _KEEP_OWNER.get(frozenset(view.get("hole") or ()), PARAMS)
    except TypeError:
        P = PARAMS
    if P["LEAK_PENALTY"]:
        return keep_leak(view, P)
    if P["KEEP_BOARD_AWARE"]:
        return keep_policy(view)
    return None


KEEP_POLICY = keep_dispatch if (PARAMS["KEEP_BOARD_AWARE"] or PARAMS["LEAK_PENALTY"]) else None   # shell -> respond


def bounty_bonus(v, exclude=-1):
    """Extra chips a bounty hit wins (v.bounty = (mult, flat)): mult x the chips the others put in + flat, paid by
    those others and capped by the chips they still hold. A player who is all-in (or `exclude`, a jammer we call)
    has nothing left after losing, so pays nothing (the rehearsal engine's rule; a stacks-reset rule like MIT 2025
    would pay; PLATFORM: check). Measured 2026-11-03: pricing the bonus into calls of all-ins as if it were paid lost
    0.8 pts per table vs no hook."""
    b = v.bounty
    me = v.me
    paid = 0
    room = 0
    for j in range(v.n):
        if j == me or j == exclude:
            continue
        c = v.contrib[j]
        s = v.stacks[j]
        if c > 0 and s > 0 and not v.all_in[j]:
            paid += c
            room += s
    if paid <= 0:
        return 0.0
    x = b[0] * paid + b[1]
    return x if x < room else room


def bounty_pot(v, pot):
    """Pot as the payoff sees it: a bounty hit adds bounty_bonus (payers with chips left only). Unchanged otherwise."""
    if v.bounty is None:
        return pot
    return pot + bounty_bonus(v)


def pot_before_street(v):
    """The pot before this street's bets (the base of a bet-to-pot ratio). The kit's pot includes the street's bets
    (pot == sum(contributed)); a platform whose pot leaves them out already reports the base. Floored at 1 bb."""
    pot, sb = v.pot, sum(v.bets)
    base = pot - sb if pot >= sum(v.contrib) else pot
    return base if base > v.bb else v.bb


# ===================================== ICM (exact Malmuth-Harville; padded payouts) =====================
def pad_payouts(pays, n=8):
    p = [float(x) for x in (pays or ())]
    if len(p) < n:
        p += [0.0] * (n - len(p))
    return p


def is_wta(pays):
    """Winner-take-all (every place after the first pays the same): equity is linear in chips, every BF is 1."""
    return len(pays) < 2 or all(abs(x - pays[1]) < 1e-12 for x in pays[1:])


def icm_hero(stacks, i, pays):
    """Closed form, EXACT when places 4..n are affine in place (WTA, top-3, linear, ...); pays padded.
    Hardening 2026-11-03: the double loop runs over a pre-filtered list of the live others (same terms, same order,
    same arithmetic: bit-identical results, checked on 359k random spots; about 17% faster)."""
    live = [s for s in stacks if s > 0]
    k, si = len(live), stacks[i]
    if si <= 0 or k == 0:
        return 0.0
    p = list(pays[:k]) + [0.0] * max(0, k - len(pays))
    if k == 1:
        return p[0]
    step, base = p[k - 2] - p[k - 1], p[k - 1]
    acc = 0.0
    for sj in live:
        acc += si / (si + sj)
    lin = base + step * (acc - 0.5)
    c0 = p[0] - (base + step * (k - 1))
    c1 = p[1] - (base + step * (k - 2)) if k > 1 else 0.0
    c2 = p[2] - (base + step * (k - 3)) if k > 2 else 0.0
    if not (c0 or c1 or c2):
        return lin
    T = float(sum(live))
    e2 = e3 = 0.0
    vals = [sj for j, sj in enumerate(stacks) if j != i and sj > 0]
    for a, sj in enumerate(vals):
        Tj = T - sj
        pj = sj / T
        e2 += pj * si / Tj
        if c2:
            for sm in vals[:a]:
                e3 += pj * (sm / Tj) * si / (Tj - sm)
            for sm in vals[a + 1:]:
                e3 += pj * (sm / Tj) * si / (Tj - sm)
    return lin + c0 * (si / T) + c1 * e2 + c2 * e3


def make_icm_exact(pays, max_players=10):
    """icm(stacks, i): exact Malmuth-Harville equity of seat i for ANY payout (exponential-race identity:
    E_i = sum over subsets B of the live others of coef[m][|B|] * s_i / (s_i + s(B)))."""
    C = [[1]]
    for n in range(1, max_players + 1):
        C.append([1] + [C[n - 1][t - 1] + C[n - 1][t] for t in range(1, n)] + [1])
    coef = []
    for m in range(max_players):
        p = list(pays[:m + 1]) + [0.0] * max(0, m + 1 - len(pays))
        row = []
        for b in range(m + 1):
            acc, sign = 0.0, 1.0
            for t in range(b + 1):
                acc += sign * C[b][t] * p[m - b + t]
                sign = -sign
            row.append(acc)
        coef.append(row)
    # weights[m][q] = coef[m][size of subset q] in the enumeration order below (hardening 2026-11-03: built once
    # instead of per call; same terms, same order, bit-identical results on 377k random spots, about 17% faster)
    weights = []
    sizes = [0]
    for m in range(max_players):
        weights.append([coef[m][b] for b in sizes])
        sizes = sizes + [b + 1 for b in sizes]

    def icm(stacks, i):
        si = stacks[i]
        if si <= 0:
            return 0.0
        sums = [0.0]
        for j, s in enumerate(stacks):
            if j != i and s > 0:
                sums += [x + s for x in sums]
        ev = 0.0
        for w, x in zip(weights[len(sums).bit_length() - 1], sums):
            ev += w * si / (si + x)
        return ev
    return icm


def make_icm(pays):
    """Fastest exact method for this payout (pays padded to >= 8 places): closed form for affine tails."""
    pays = pad_payouts(pays, max(8, len(pays)))
    t = pays[3:8]
    if all(abs((t[q] - t[q + 1]) - (t[0] - t[1])) < 1e-9 for q in range(len(t) - 1)):
        return lambda stacks, i: icm_hero(stacks, i, pays)
    return make_icm_exact(pays, max(10, len(pays)))


def bubble_factor(icm, stacks, i, j, pays, lo=0.5, hi=6.0, no_gain=4.0, now=None):
    """ICM loss / ICM gain for seat i in an all-in of min(s_i, s_j) against seat j (start-of-hand stacks).
    now: icm(stacks, i) if the caller already has it (Brain.bf caches it per hand and seat: the same float)."""
    a = min(stacks[i], stacks[j])
    if a <= 0:
        return 1.0
    k = 0
    for s in stacks:
        if s > 0:
            k += 1
    if now is None:
        now = icm(stacks, i)
    win = list(stacks)
    win[i] += a
    win[j] -= a
    lose = list(stacks)
    lose[i] -= a
    lose[j] += a
    gain = icm(win, i) - now
    lost = now - (pays[k - 1] if lose[i] <= 0 else icm(lose, i))
    if gain <= 1e-12:
        v = no_gain
    else:
        v = lost / gain
    return lo if v < lo else hi if v > hi else v


_ICM_PACKS = {}   # payout key -> (icm function, equal-stack BF by players left, BF cost ns); pure function of the payout


def icm_pack(pays, n_seats, lo, hi, no_gain=4.0):
    """(icm, equal-stack BF table, measured BF cost in ns) for a payout, built once per process (construction time,
    untimed). The cost is the median thread-CPU time of 7 BFs on an 8-stack spot, so the guard adapts to the box."""
    key = (tuple(pays), n_seats, lo, hi, no_gain)
    pk = _ICM_PACKS.get(key)
    if pk is None:
        icm = make_icm(pays)
        table = {}
        for k in range(2, max(n_seats, 2) + 1):
            st = [1000] * k
            table[k] = bubble_factor(icm, st, 0, 1, pays, lo, hi, no_gain)
        table[1] = 1.0
        # one stack per seat (audit fix 2026-11-03: the list stopped at 8, so on 9-10-seat tables the measured cost
        # was an 8-player BF, about 4x too low, and decisions reached 316 us against the 250 us budget)
        spot = [2600, 2100, 1900, 1700, 1500, 1300, 1100, 900, 800, 700][:max(2, min(n_seats, 10))]
        ts = []
        tt = time.thread_time_ns
        for _ in range(7):
            c0 = tt()
            bubble_factor(icm, spot, 0, 1, pays, lo, hi, no_gain)
            ts.append(tt() - c0)
        ts.sort()
        pk = (icm, table, ts[len(ts) // 2])
        _ICM_PACKS[key] = pk
    return pk


# ===================================== ICM push/fold equilibrium tables (ICM_NASH) =====================================
def _load_nash():
    """The ICM push/fold equilibrium tables (bot/icm_nash_tables.py), or None if missing or broken: the switch then
    has no effect. Never raises."""
    try:
        try:
            from . import icm_nash_tables as nt
        except ImportError:
            import importlib.util as ilu
            import os
            import sys
            name = "nolimit_icm_nash_tables"
            nt = sys.modules.get(name)
            if nt is None:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icm_nash_tables.py")
                if not os.path.exists(path):
                    return None
                spec = ilu.spec_from_file_location(name, path)
                nt = ilu.module_from_spec(spec)
                spec.loader.exec_module(nt)
                sys.modules[name] = nt
        nt.lookup(2, nt.KINDS[0], (10.0, 10.0), ("push", 0))       # smoke check: raises if the data is broken
        return nt
    except Exception:
        return None


_NT = _load_nash()
_NL_COMB = [combos(lab) for lab in _T.EQ_LABELS.split()]   # the tables' class order (NLHE, bot_tables.EQ_LABELS)
_MASK_PCT = {}        # 169-bit set -> combo share (a pure function of the table data)


def mask_pct(mask):
    """Combo share of a 169-bit class set."""
    v = _MASK_PCT.get(mask)
    if v is None:
        tot = 0
        c = 0
        m = mask
        while m:
            if m & 1:
                tot += _NL_COMB[c]
            m >>= 1
            c += 1
        v = tot / 1326.0
        if len(_MASK_PCT) < 100_000:
            _MASK_PCT[mask] = v
    return v


def nash_kind(pays):
    """Payout kind of the equilibrium tables whose first 4 places equal this payout's, else None. The tables are
    used only with at most 4 players left, whose ICM depends on places 1-4 alone (so a 4-seat test table with a
    truncated payout list, or any payout sharing the top 4, gets the same tables)."""
    if _NT is None:
        return None
    p = pad_payouts(pays, 8)[:4]
    for k in _NT.KINDS:
        q = pad_payouts(_NT.PAYOUTS[k], 8)[:4]
        if all(abs(a - b) < 1e-6 for a, b in zip(p, q)):
            return k
    return None


# ===================================== per-seat statistics and the hand tracker =====================================
RM_FIELDS = ("rm_jam_n", "rm_jam_hit", "rm_jam_exp", "rm_open_n", "rm_open_hit", "rm_open_exp",
             "rm_3b_n", "rm_3b_hit", "rm_cj_n", "rm_cj_hit", "rm_cj_exp")


class SeatStat:
    __slots__ = ("pf_dec", "pf_vol", "pf_fold", "jam_opp", "jam_first", "jams", "jam_faced", "jam_called", "jam_exp",
                 "late_opp", "late_open", "post_faced", "post_call", "post_fold", "flop_faced", "flop_fold",
                 "vol_ever", "vol_hand",
                 "rm_jam_n", "rm_jam_hit", "rm_jam_exp", "rm_open_n", "rm_open_hit", "rm_open_exp",
                 "rm_3b_n", "rm_3b_hit", "rm_cj_n", "rm_cj_hit", "rm_cj_exp")

    def __init__(self):
        self.pf_dec = self.pf_vol = self.pf_fold = 0
        self.jam_opp = self.jam_first = self.jams = 0
        self.jam_faced = self.jam_called = 0
        self.jam_exp = 0.0
        self.late_opp = self.late_open = 0
        self.post_faced = self.post_call = self.post_fold = 0
        self.flop_faced = self.flop_fold = 0
        self.vol_ever = 0
        self.vol_hand = False
        # range-model evidence (RANGE_MODEL only; preflop actions of completed hands): opportunities, hits,
        # prior-expected hits; the order matches the HandTracker.rm_pend lists (RM_FIELDS)
        self.rm_jam_n = self.rm_jam_hit = 0
        self.rm_jam_exp = 0.0
        self.rm_open_n = self.rm_open_hit = 0
        self.rm_open_exp = 0.0
        self.rm_3b_n = self.rm_3b_hit = 0
        self.rm_cj_n = self.rm_cj_hit = 0
        self.rm_cj_exp = 0.0



def preflop_order(n, alive, sb_seat, bb_seat):
    """Preflop action order of the dealt-in seats: from the seat after the big blind clockwise to the big blind
    (heads-up: button/SB first, then BB)."""
    seq = []
    j = bb_seat
    for _ in range(n):
        j = (j + 1) % n
        if alive[j]:
            seq.append(j)
        if j == bb_seat:
            break
    return seq


class HandTracker:
    """The current hand's public action sequence, fed in order (from observe() events, or from the state's history
    at decision time when no events arrive). Updates the per-seat SeatStat counters as actions arrive, and keeps what
    the policy needs: raise count, last raiser, limpers, the outstanding jam and its estimated range."""
    __slots__ = ("P", "stats", "n", "hand_no", "street", "n_done", "bb", "ante", "start", "stake", "contrib", "bets",
                 "folded", "allin", "pre", "pos", "bb_seat", "cur", "pf_raises", "last_raiser", "raise_to",
                 "open_k", "open_seat", "limpers", "limper_seats", "callers_since_raise", "jam", "jam_responders",
                 "pf_aggr", "posted", "jam_ranges", "rm_pend")

    def __init__(self, P, stats, n):
        self.P = P                        # the brain's PARAMS and per-seat stats (no back-reference to the brain,
        self.stats = stats                # so a finished table's objects are freed by reference counting alone)
        self.n = n
        self.hand_no = None
        self.rm_pend = None               # RANGE_MODEL: this hand's evidence per seat, committed at the next hand
        self._reset(None, [0] * n, [False] * n, 0, 1, 20, 0)

    def _reset(self, hand_no, start, alive, sb_seat, bb_seat, bb, ante):
        n = self.n
        if self.rm_pend:
            self.rm_commit()
        self.hand_no = hand_no
        self.street = 0
        self.n_done = 0
        self.bb = bb if bb > 0 else 1
        self.ante = ante
        self.start = list(start)
        self.stake = list(start)          # chips at stake preflop = start minus ante paid
        self.contrib = [0] * n
        self.bets = [0] * n
        self.folded = [not a for a in alive]
        self.allin = [False] * n
        self.bb_seat = bb_seat
        self.pre = preflop_order(n, alive, sb_seat, bb_seat) if any(alive) else []
        self.pos = {s: i for i, s in enumerate(self.pre)}
        self.cur = self.bb
        self.pf_raises = 0
        self.last_raiser = -1
        self.raise_to = 0
        self.open_k = 0
        self.open_seat = -1
        self.limpers = 0
        self.limper_seats = []
        self.callers_since_raise = 0
        self.jam = None                   # (seat, J_first_in_formula, is_rejam, jam_total)
        self.jam_responders = set()
        self.jam_ranges = {}              # every all-in raiser this hand: seat -> (J_first_in_formula, is_rejam)
        self.pf_aggr = -1
        self.posted = 0
        for st in self.stats:
            st.vol_hand = False

    def start_hand(self, hand_no, start, alive, sb_seat, bb_seat, bb, ante):
        self._reset(hand_no, start, alive, sb_seat, bb_seat, bb, ante)

    def new_street(self, idx):
        while self.street < idx:
            self.street += 1
            for j in range(self.n):
                self.bets[j] = 0
            self.cur = 0

    def _k_behind(self, s):
        """Dealt-in seats after s in the preflop order that have not folded."""
        p = self.pos.get(s)
        if p is None:
            return 0
        k = 0
        for j in self.pre[p + 1:]:
            if not self.folded[j]:
                k += 1
        return k

    def feed(self, s, kind, amt):
        self.n_done += 1
        if s < 0 or s >= self.n:
            return
        if kind == "ante":
            self.contrib[s] += amt
            self.stake[s] -= amt
            if self.contrib[s] >= self.start[s]:
                self.allin[s] = True
            return
        if kind == "post_sb" or kind == "post_bb":
            add = amt - self.bets[s]
            self.bets[s] = amt
            self.contrib[s] += add
            self.posted += add
            if self.contrib[s] >= self.start[s]:
                self.allin[s] = True
            return
        P = self.P
        st = self.stats[s]
        before = self.bets[s]
        cur = self.cur
        c0 = self.contrib[s]
        vol = kind == "call" or kind == "raise"
        if kind == "fold":
            self.folded[s] = True
        elif vol:
            add = amt - before
            if add > 0:
                self.bets[s] = amt
                self.contrib[s] += add
            if self.contrib[s] >= self.start[s]:
                self.allin[s] = True
        if self.street == 0:
            st.pf_dec += 1
            if vol:
                st.pf_vol += 1
                st.vol_ever += 1
                st.vol_hand = True
            elif kind == "fold":
                st.pf_fold += 1
            unopened = self.pf_raises == 0
            if P["RANGE_MODEL"] and kind != "check":
                self._rm_evidence(s, kind, unopened, c0)
            if unopened and kind != "check":
                st.jam_opp += 1
                if kind == "raise" and self.allin[s]:
                    st.jam_first += 1
                p = self.pos.get(s, 0)
                if self.limpers == 0 and s != self.bb_seat and len(self.pre) - 1 - p <= P["LATE_SEATS"]:
                    st.late_opp += 1
                    if kind == "raise":
                        st.late_open += 1
            jam = self.jam
            if jam is not None and jam[0] != s and kind != "check" and s not in self.jam_responders:
                self.jam_responders.add(s)
                st.jam_faced += 1
                if vol:
                    st.jam_called += 1
                S_resp = min(self.stake[s], jam[3]) / self.bb
                x = call_pct(P["REJAM_PRIOR"] if jam[2] else jam[1], S_resp, P["CALL_FIT"])
                st.jam_exp += x
                if P["RANGE_MODEL"]:
                    d = self._rm_d(s)
                    d[8] += 1
                    d[10] += x
                    if vol:
                        d[9] += 1
            if kind == "raise":
                first_in = self.pf_raises == 0
                if self.allin[s]:
                    # estimated range of this jam: the first-in formula at the jammer's spot (+ a re-jam flag)
                    k = self._k_behind(s) + (self.limpers if first_in else 0)
                    opp = [self.stake[j] for j in self.pre[self.pos.get(s, 0) + 1:] if not self.folded[j]]
                    S_r = min(self.stake[s], max(opp)) / self.bb if opp else self.stake[s] / self.bb
                    pot_before = sum(self.contrib) - (amt - before)
                    S_eq = equiv_depth(S_r, pot_before / self.bb, P) if first_in and pot_before > 0 else S_r
                    self.jam = (s, jam_pct(S_eq, k if k > 0 else 1, P["JAM_FIT"]), not first_in, self.stake[s])
                    self.jam_ranges[s] = (self.jam[1], self.jam[2])
                    self.jam_responders = set()
                if first_in:
                    self.open_k = self._k_behind(s)
                    self.open_seat = s
                self.pf_raises += 1
                self.last_raiser = s
                self.raise_to = amt
                self.callers_since_raise = 0
                self.pf_aggr = s
            elif kind == "call":
                if self.pf_raises == 0:
                    self.limpers += 1
                    self.limper_seats.append(s)
                else:
                    self.callers_since_raise += 1
        else:
            if cur > before:
                st.post_faced += 1
                if kind == "call" or kind == "raise":
                    st.post_call += 1
                elif kind == "fold":
                    st.post_fold += 1
                if self.street == 1:
                    st.flop_faced += 1
                    if kind == "fold":
                        st.flop_fold += 1
            if vol:
                st.vol_ever += 1
                st.vol_hand = True
        if kind == "raise" and amt > self.cur:
            self.cur = amt

    def _rm_d(self, s):
        if self.rm_pend is None:
            self.rm_pend = {}
        d = self.rm_pend.get(s)
        if d is None:
            d = self.rm_pend[s] = [0, 0, 0.0, 0, 0, 0.0, 0, 0, 0, 0, 0.0]
        return d

    def rm_commit(self):
        """Add the finished hand's range-model evidence to the per-seat counters."""
        stats = self.stats
        for s, d in self.rm_pend.items():
            st = stats[s]
            st.rm_jam_n += d[0]
            st.rm_jam_hit += d[1]
            st.rm_jam_exp += d[2]
            st.rm_open_n += d[3]
            st.rm_open_hit += d[4]
            st.rm_open_exp += d[5]
            st.rm_3b_n += d[6]
            st.rm_3b_hit += d[7]
            st.rm_cj_n += d[8]
            st.rm_cj_hit += d[9]
            st.rm_cj_exp += d[10]
        self.rm_pend = {}

    def _rm_evidence(self, s, kind, unopened, c0):
        """Range-model evidence from one preflop decision (fold / call / raise) of seat s, recorded BEFORE the
        tracker's raise state changes, into this hand's pending list (committed at the next hand start). Jam
        evidence: first-in at effective depth <= RM_JAM_MAX_BB, expected hits = the first-in jam formula at the
        spot (the estimate the tracker gives a jam there). Open evidence: first-in with no limpers, not the BB,
        depth >= RM_OPEN_MIN_BB, expected = the open prior by players behind, hit = any raise. 3-bet evidence:
        facing exactly one raise that was not all-in, with a raise still possible, hit = a re-raise. Jam-call
        evidence is recorded where the tracker counts a jam faced. Postflop actions never reach here."""
        P = self.P
        bb = self.bb
        if unopened:
            p = self.pos.get(s)
            if p is None:
                return
            opp = [self.stake[j] for j in self.pre[p + 1:] if not self.folded[j]]
            kb = len(opp)
            k = kb + self.limpers
            if k <= 0:
                return
            S_r = min(self.stake[s], max(opp)) / bb if opp else self.stake[s] / bb
            hit = kind == "raise"
            jam_zone = S_r <= P["RM_JAM_MAX_BB"]
            open_zone = self.limpers == 0 and kb > 0 and s != self.bb_seat and S_r >= P["RM_OPEN_MIN_BB"]
            if not (jam_zone or open_zone):
                return
            d = self._rm_d(s)
            if jam_zone:
                pot_before = sum(self.contrib) - (self.contrib[s] - c0)
                S_eq = equiv_depth(S_r, pot_before / bb, P) if pot_before > 0 else S_r
                d[0] += 1
                d[2] += jam_pct(S_eq, k, P["JAM_FIT"])
                if hit and self.allin[s]:
                    d[1] += 1
            if open_zone:
                d[3] += 1
                d[5] += by_k(P["OPEN_PRIOR"] or P["OPEN_PCT"], kb)
                if hit:
                    d[4] += 1
        elif self.pf_raises == 1 and self.jam is None and s != self.last_raiser and self.stake[s] > self.raise_to:
            d = self._rm_d(s)
            d[6] += 1
            if kind == "raise":
                d[7] += 1

    def sync_from_view(self, v):
        """Bring the tracker up to date with the view's history (no-op when observe() already delivered it)."""
        if self.hand_no != v.hand_no:
            self.start_hand(v.hand_no, v.start, v.alive, v.sb_seat, v.bb_seat, v.bb, v.ante)
        hist = v.history
        ss = v.street_starts
        i = self.n_done
        nh = len(hist)
        while i < nh:
            nxt = self.street + 1
            while nxt < len(ss) and i >= ss[nxt]:
                self.new_street(nxt)
                nxt += 1
            e = hist[i]
            self.feed(e[0], e[1], e[2])
            i += 1
        if v.street > self.street:
            self.new_street(v.street)


# ===================================== the brain =====================================
class Brain:
    """One per table (state resets each table). decide(view, t0) -> internal decision tuple."""

    def __init__(self, seat, config, version=None, params=None):
        P = dict(PARAMS)
        if params is None and isinstance(config.get("bot_params"), dict):
            params = config["bot_params"]          # rehearsal A/B hook (runner extra_config); absent on the platform
        if params:
            P.update(params)
        if version is None:
            version = config.get("bot_version", P["VERSION"])
        self.version = int(version)
        self.seat = seat
        n = int(config.get("n_seats", 8) or 8)
        self.n = n
        self.pays = pad_payouts(config.get("payouts"), max(8, n))
        self.tuned = {}           # PARAMS keys set by TUNE_BY_PAYOUT for this payout (diagnostic)
        if P["TUNE_BY_PAYOUT"] and not is_wta(self.pays):
            for k, val in (P["TUNE_MULTI"] or {}).items():
                if not (params and k in params):    # an explicit override still wins
                    P[k] = val
                    self.tuned[k] = val
        self.P = P
        icm_on = P["ICM_ON"]
        if icm_on is None:
            icm_on = not is_wta(self.pays)
        self.icm_on = bool(icm_on) and self.version >= 1
        self.icm = None
        self.bf_table = {}
        self.bf_cost_ns = 0
        if self.icm_on:
            self.icm, self.bf_table, self.bf_cost_ns = icm_pack(self.pays, n, P["BF_MIN"], P["BF_MAX"],
                                                                P["BF_NO_GAIN"])
        # ICM_NASH (ICM_NASH notes): equilibrium tables for this payout, only with ICM on (so never under
        # winner-take-all, which stays on the chip-EV formula), the built-in NLHE classes and a solved payout kind
        self.nash_kind = None
        if P["ICM_NASH"] and self.icm_on and _SEAM is None and not is_wta(self.pays):
            k = nash_kind(self.pays)
            if k is not None and k != "WTA":
                self.nash_kind = k
        parts = str(P["ICM_NASH_PARTS"])
        self.nash_jam = self.nash_kind is not None and "jam" in parts
        self.nash_call_on = self.nash_kind is not None and "call" in parts
        self.n_nash = [0, 0, 0, 0]  # decisions from the tables: first-in jam sets, call sets, over-call sets; and
        #                             calls priced by call_jam because the jammer's estimated range deviated (v2)
        self.nash_memo = (None, None)   # (hand_no, spot) cache of nash_spot
        # a BF may start only while it can still finish inside the budget (own clock, wall time like the rules)
        self.bf_start_limit = P["GUARD_NS"] - P["BF_COST_SAFETY"] * self.bf_cost_ns
        seed = config.get("seed", 0)
        self.rng = random.Random(seed if isinstance(seed, (int, str)) else 0)
        self.stats = [SeatStat() for _ in range(n)]
        self.tr = HandTracker(P, self.stats, n)
        self.bf_cache = {}
        self.icm_now = {}         # seat -> icm(start stacks, seat) for this hand (bf() reuses it across opponents)
        self.bf_hand = None
        self.bf_budget = 0
        self.t0 = 0
        self.types = {}
        self.open_prior = P["OPEN_PRIOR"] or P["OPEN_PCT"]
        self.rm = bool(P["RANGE_MODEL"]) and self.version >= 2    # range model (v2 only; the v0 fallback skips it)
        self.guard_trips = 0      # decisions where the guard stopped optional work (diagnostic)
        self.bf_computed = 0
        self.n_jam_gated = 0      # v1+ first-in jam decisions that used the gated (caller-side) ICM formula
        self.n_jam_hero = 0       # ... that used the default hero-only tightening
        self.rev = RevealTracker()  # revealed discards / tosses this hand (observe "discard" events)
        self.n_reveal_adj = [0, 0]  # USE_REVEALED: postflop calls moved down / up a class (diagnostic)
        self.keep_override = P["LEAK_PENALTY"] != PARAMS["LEAK_PENALTY"] or \
            P["KEEP_BOARD_AWARE"] != PARAMS["KEEP_BOARD_AWARE"]          # A/B only: register for keep_dispatch

    # ------------------------------------------------------------------ events (untimed in the kit)
    def observe(self, ev):
        t = ev.get("type")
        tr = self.tr
        if t == "action":
            if ev.get("hand_no") == tr.hand_no:
                si = STREET_INDEX.get(ev.get("street"), tr.street)
                if si > tr.street:
                    tr.new_street(si)
                tr.feed(ev["seat"], ev["action"], ev.get("amount", 0))
        elif t == "hand_start":
            n = self.n
            stacks = list(ev.get("stacks", [0] * n))
            alive = list(ev.get("alive", [s > 0 for s in stacks]))
            tr.start_hand(ev.get("hand_no"), stacks, alive, ev.get("sb_seat", 0), ev.get("bb_seat", 1),
                          ev.get("bb", 20), ev.get("ante", 0))
            for p in ev.get("posts", ()):
                tr.feed(p[0], p[1], p[2])
        elif t == "street":
            si = STREET_INDEX.get(ev.get("street"), tr.street)
            if ev.get("hand_no") == tr.hand_no and si > tr.street:
                tr.new_street(si)
        elif t == "discard" or t == "toss":
            s = ev.get("seat")
            if isinstance(s, int) and s != self.seat:
                cs = ev.get("cards")
                if cs is None:
                    cs = (ev.get("card"),)
                self.rev.add(ev.get("hand_no"), s, cs)

    # ------------------------------------------------------------------ bubble factors
    def bf(self, i, j):
        """BF of seat i vs seat j at this hand's start stacks: cache, else exact (budget and guard permitting),
        else the equal-stack table value for the players left (never 1.0 while ICM is on)."""
        if not self.icm_on:
            return 1.0
        key = i * 64 + j
        v = self.bf_cache.get(key)
        if v is not None:
            return v
        if self.bf_budget > 0 and time.perf_counter_ns() - self.t0 < self.bf_start_limit:
            self.bf_budget -= 1
            try:
                st = self.view_start
                now = self.icm_now.get(i)          # seat i's ICM at this hand's start stacks: one call per hand
                if now is None and st[i] > 0 and min(st[i], st[j]) > 0:
                    now = self.icm(st, i)
                    self.icm_now[i] = now
                v = bubble_factor(self.icm, st, i, j, self.pays, self.P["BF_MIN"], self.P["BF_MAX"],
                                  self.P["BF_NO_GAIN"], now)
                self.bf_computed += 1
            except Exception:
                v = None
            if v is not None:
                self.bf_cache[key] = v
                return v
        elif self.bf_budget > 0:
            self.guard_trips += 1
            self.bf_budget = 0
        return self.bf_table.get(self.n_live, self.P["BF_DEFAULT"])

    def bf_mean(self, me, seats, hero_side):
        """Mean BF over seats (hero_side: BF(me, j); else BF(j, me)), biggest stacks first; uncomputed ones take the
        mean of the computed ones (or the table value)."""
        if not seats:
            return 1.0
        st = self.view_start
        order = sorted(seats, key=lambda j: -st[j])
        vals = []
        exact = []
        for j in order:
            key = (me * 64 + j) if hero_side else (j * 64 + me)
            v = self.bf_cache.get(key)
            if v is None and self.bf_budget > 0:
                v = self.bf(me, j) if hero_side else self.bf(j, me)
                if key not in self.bf_cache:
                    v = None
            if v is not None:
                exact.append(v)
            vals.append(v)
        fill = (sum(exact) / len(exact)) if exact else self.bf_table.get(self.n_live, self.P["BF_DEFAULT"])
        tot = 0.0
        for v in vals:
            tot += v if v is not None else fill
        return tot / len(vals)

    # ------------------------------------------------------------------ typing (v2; also the v1 jam-tight gate)
    def shrunk(self, hits, n, key):
        m, k = self.P["PRIOR"][key]
        return (hits + k * m) / (n + k)

    def jam_tight(self, j):
        """Observed folding to jams more than chip-EV Nash (v1 gate); v2 adds typed folders and nits."""
        st = self.stats[j]
        P = self.P
        if st.jam_faced >= P["MIN_JAMS_FACED_TIGHT"]:
            oe = (st.jam_called + P["OE_PRIOR"]) / (st.jam_exp + P["OE_PRIOR"])
            if oe <= P["OE_TIGHT"]:
                return True
        if self.version >= 2:
            t = self.seat_type(j)
            return t == "folder" or t == "nit"
        return False

    def is_folder(self, j):
        st = self.stats[j]
        return st.pf_dec >= self.P["MIN_OBS_FOLDER"] and st.pf_vol == 0 and st.vol_ever == 0 and not st.vol_hand

    def seat_type(self, j):
        t = self.types.get(j)
        if t is not None:
            return t
        P = self.P
        st = self.stats[j]
        if self.is_folder(j):
            t = "folder"
        elif st.pf_dec < P["MIN_OBS_TYPE"]:
            t = "unknown"
        elif self.shrunk(st.jam_first, st.jam_opp, "jam") >= P["T_MANIAC_JAM"]:
            t = "maniac"
        elif (st.jam_faced >= P["MIN_JAMS_FACED"]
              and self.shrunk(st.jam_called, st.jam_faced, "calljam") >= P["T_CALLER_CALLJAM"]) or \
                (st.post_faced >= P["MIN_POST_FACED"]
                 and self.shrunk(st.post_call, st.post_faced, "post_call") >= P["T_POST_CALL"]):
            t = "caller"
        elif st.late_opp >= P["MIN_LATE_OPP"] and \
                self.shrunk(st.late_open, st.late_opp, "late_open") >= P["T_STEAL_LATE"]:
            t = "stealer"
        elif self.shrunk(st.pf_vol, st.pf_dec, "vol") >= P["T_LOOSE_VOL"]:
            t = "loose"
        elif st.pf_dec >= P["MIN_OBS_NIT"] and self.shrunk(st.pf_vol, st.pf_dec, "vol") < P["T_NIT_VOL"]:
            t = "nit"
        else:
            t = "normal"
        self.types[j] = t
        return t

    def rm_ratio(self, hits, expected, a):
        """Range model: posterior / prior width ratio (hits + a) / (expected + a) for a Beta prior with the prior rate
        as mean and alpha a, clamped to [1/RM_CAP, RM_CAP]; 1.0 without evidence."""
        if expected <= 0.0:
            return 1.0
        r = (hits + a) / (expected + a)
        cap = self.P["RM_CAP"]
        if r > cap:
            return cap
        lo = 1.0 / cap
        return r if r > lo else lo

    def rm_jam_ratio(self, j, is_rejam):
        """Width ratio of seat j's jam (first-in jam evidence) or re-raise (3-bet evidence) range."""
        st = self.stats[j]
        P = self.P
        if is_rejam:
            return self.rm_ratio(st.rm_3b_hit, st.rm_3b_n * P["REJAM_PRIOR"], P["RM_A_3BET"])
        return self.rm_ratio(st.rm_jam_hit, st.rm_jam_exp, P["RM_A_JAM"])

    def rm_open_ratio(self, j):
        st = self.stats[j]
        return self.rm_ratio(st.rm_open_hit, st.rm_open_exp, self.P["RM_A_OPEN"])

    def rm_call_ratio(self, j):
        """Width ratio of a seat that already called a jam, from its jam calls vs expected Nash calls in completed
        hands (jam-call evidence only; postflop calls never count)."""
        st = self.stats[j]
        return self.rm_ratio(st.rm_cj_hit, st.rm_cj_exp, self.P["RM_A_CALL"])

    def jam_range_of(self, j, J_formula, is_rejam):
        """Estimated range of seat j's jam. v0/v1: the first-in formula at its spot (also for re-jams, as tested);
        v2: re-jams get REJAM_PRIOR (P5); a typed maniac's range is its own shrunk jam rate (E10); loose floor.
        RANGE_MODEL: the prior (formula or REJAM_PRIOR) is scaled by the seat's posterior ratio first."""
        if self.version < 2:
            return J_formula
        P = self.P
        t = self.seat_type(j)
        J = P["REJAM_PRIOR"] if is_rejam else J_formula
        if self.rm:
            J *= self.rm_jam_ratio(j, is_rejam)
            if J > 1.0:
                J = 1.0
        if t == "maniac":
            st = self.stats[j]
            J = max(J, self.shrunk(st.jam_first, st.jam_opp, "jam"), P["J_FLOOR"]["maniac"])
        elif t == "loose" and not is_rejam:
            J = max(J, P["J_FLOOR"]["loose"])
        return J

    # ------------------------------------------------------------------ decide
    def decide(self, v, t0):
        self.t0 = t0
        self.bf_budget = self.P["BF_PER_DECISION"]
        tr = self.tr
        tr.sync_from_view(v)
        if self.bf_hand != v.hand_no:
            self.bf_hand = v.hand_no
            self.bf_cache = {}
            self.icm_now = {}
        self.types = {}
        self.view_start = v.start
        nl = 0
        for s in v.start:
            if s > 0:
                nl += 1
        self.n_live = nl
        if self.keep_override and len(v.hole) > 2:
            if len(_KEEP_OWNER) > 256:
                _KEEP_OWNER.clear()
            _KEEP_OWNER[frozenset(v.hole)] = self.P
        if v.street == 0:
            return self.preflop(v)
        return self.postflop(v)

    def decide_fallback(self, v, t0):
        """P15: the v0 decision (no ICM, no typing) for the same view. The shell calls this only when decide()
        raised, so an error in optional work costs the refinement, not the whole decision. The tracker resumes
        after the entry that failed (feed() counts an entry before processing it), so nothing is fed twice."""
        saved = (self.version, self.icm_on, self.rm, self.nash_kind, self.nash_jam, self.nash_call_on)
        self.version, self.icm_on, self.rm, self.nash_kind, self.nash_jam, self.nash_call_on = \
            0, False, False, None, False, False
        try:
            return self.decide(v, t0)
        finally:
            self.version, self.icm_on, self.rm, self.nash_kind, self.nash_jam, self.nash_call_on = saved

    # ------------------------------------------------------------------ preflop
    def preflop(self, v):
        P = self.P
        tr = self.tr
        me = v.me
        bb = v.bb
        ver = self.version
        n = v.n
        stacks, bets, in_hand = v.stacks, v.bets, v.in_hand
        tot = [stacks[j] + bets[j] for j in range(n)]
        h = v.hole
        if _SEAM is None and len(h) == 2:
            cls = CLASS_OF[h[0] * 52 + h[1]]          # the pre-integration lookup (NLHE byte identity)
        else:
            cls = classify(h)                          # any dealt count, swapped tables; never raises
        pp = PUSH_PCT[cls]
        cp = CALL_PCT[cls]
        pos = tr.pos.get(me)
        if tr.pf_raises == 0:
            # ---------------- unopened (limpers allowed)
            limpers = tr.limper_seats
            others = [j for j in range(n) if j != me and in_hand[j]]
            if not others:
                return ("check",) if v.to_call <= 0 else ("fold",)
            if v.to_call <= 0:
                # big blind option (limpers or a completed small blind): value-raise premiums only
                if limpers and pp <= P["BB_ISO_PCT"]:
                    S = min(tot[me], max(tot[j] for j in others)) / bb
                    if S <= P["PF_DEPTH_BB"]:
                        return ("allin",)
                    return ("raise", int((P["BB_ISO_BB"] + P["BB_ISO_LIMPER_BB"] * len(limpers)) * bb))
                return ("check",)
            behind = [j for j in tr.pre[pos + 1:] if in_hand[j]] if pos is not None else others
            callers = behind + [j for j in limpers if in_hand[j] and j not in behind]
            k = len(behind)
            if not callers:
                return ("check",) if v.to_call <= 0 else ("fold",)
            if ver >= 2 and P["FOLDER_STEAL"] and not limpers and behind and \
                    all(self.is_folder(j) for j in behind):
                return ("raise", v.min_to)                                # E09: free steal (min-raise any two)
            S = min(tot[me], max(tot[j] for j in callers)) / bb
            nash = None
            if self.nash_jam and not limpers and S <= P["ICM_NASH_MAX_BB"]:
                nash = self.nash_first_in(v, cls, pp, callers, S)     # True jam / False no jam / None no table
                if nash:
                    return ("allin",)
            if S <= P["PF_DEPTH_BB"]:
                if nash is not None:
                    return ("fold",)
                pot_w = v.pot if (v.bounty is None or "jam" not in P["BOUNTY_PARTS"]) else bounty_pot(v, v.pot)
                S_eq = equiv_depth(S, pot_w / bb, P)                      # antes and limps = more dead money
                J = jam_pct(S_eq, len(callers), P["JAM_FIT"]) * P["JAM_MULT"]
                if self.icm_on:
                    J = self.icm_jam(J, me, callers)
                if ver >= 2:
                    J = self.typed_jam(J, callers, S)
                if pp <= J:
                    return ("allin",)
                return ("fold",)
            op = by_k(P["OPEN_PCT"], k)
            if pp <= op:
                return ("raise", int((P["OPEN_BB"] + P["OPEN_LIMPER_BB"] * len(limpers)) * bb))
            return ("fold",)
        # ---------------- facing a raise
        r = tr.last_raiser
        if r < 0 or r == me:
            r = max((j for j in range(n) if j != me and in_hand[j]), key=lambda j: bets[j], default=-1)
            if r < 0:
                return ("check",) if v.to_call <= 0 else ("fold",)
        jam = tr.jam
        if jam is not None and in_hand[jam[0]] and v.all_in[jam[0]]:
            if self.nash_call_on:
                d = self.nash_call(v, jam, cls, tot)
                if d is not None:
                    return d
            return self.call_jam(v, jam[0], jam[1], jam[2], cls, tot, near=False)
        near = stacks[r] <= max(P["NEAR_ALLIN_BB"] * bb, P["NEAR_ALLIN_FRAC"] * tot[r])
        S = min(tot[me], tot[r]) / bb
        if near:
            Jf = jam_pct(S, max(1, tr.open_k if tr.pf_raises == 1 else 1), P["JAM_FIT"])
            return self.call_jam(v, r, Jf, tr.pf_raises > 1, cls, tot, near=True)
        if v.to_call <= 0:
            return ("check",)
        committed = v.to_call >= P["COMMIT_TO_CALL_FRAC"] * tot[me] or S <= P["PF_DEPTH_BB"]
        late_small = False
        raise_top, flat_top, bb_def = P["RAISE_TOP"], P["FLAT_TOP"], P["BB_DEFEND_PCT"]
        if ver >= 2 and self.rm:
            O, late_small, sc = self.open_range_rm(r, v)
            if sc != 1.0:
                sc = sc ** P["RM_RULE_EXP"]
                raise_top *= sc
                flat_top *= sc
                bb_def = min(bb_def * sc, P["RM_BB_DEFEND_MAX"])
        elif ver >= 2:
            O, late_small = self.open_range(r, v)
        else:
            # tested brain: the raiser's range = the first-in jam formula at its own depth and position
            pr = tr.pos.get(r, 0)
            kr = max(1, len(tr.pre) - 1 - pr)
            O = jam_pct(tot[r] / bb, kr, P["JAM_FIT"])
        if committed:
            C = call_pct(O, S, P["CALL_FIT"])
            if self.icm_on:
                C = C / self.bf(me, r) ** P["CALL_LAM"]
            if cp <= P["RESTEAL_MULT"] * C:
                return ("allin",)
            if late_small and me == v.bb_seat and cp <= bb_def and \
                    v.to_call <= P["BB_DEFEND_MAX_FRAC"] * tot[me]:
                return ("call",)                                          # P6: BB defence vs a late small raise
            return ("fold",)
        if pp <= raise_top:
            if S <= P["FOURBET_JAM_MAX_BB"]:
                return ("allin",)
            return ("raise", int(P["THREEBET_MULT"] * max(bets)))
        if pp <= flat_top and v.to_call <= P["FLAT_MAX_STACK"] * tot[me]:
            return ("call",)
        if late_small and me == v.bb_seat and cp <= bb_def and \
                v.to_call <= P["BB_DEFEND_MAX_FRAC"] * tot[me]:
            return ("call",)
        return ("fold",)

    def icm_jam(self, J, me, callers):
        """v1 jam side: hero-only tightening by default; the tested caller-side widening only if every possible
        caller was observed folding to jams more than chip-EV Nash."""
        P = self.P
        bfh = self.bf_mean(me, callers, True)
        if all(self.jam_tight(j) for j in callers):
            bfo = self.bf_mean(me, callers, False)
            J = J * bfo ** P["JAM_OPP_EXP"] * bfh ** P["JAM_HERO_EXP_GATED"]
            self.n_jam_gated += 1
        else:
            J = J * bfh ** P["JAM_HERO_EXP"]
            self.n_jam_hero += 1
        return J if J < 1.0 else 1.0

    # ------------------------------------------------------------------ ICM_NASH (ICM_NASH notes)
    def nash_spot(self, v):
        """(n, stacks in bb by position SB, BB, BTN, CO, seat -> position) when the push/fold tables apply to this
        hand (2-4 dealt in, two hole cards, no antes, blinds where the table expects them, no stack above
        ICM_NASH_CAP_BB), else None. Cached per hand (start stacks do not change within a hand)."""
        memo = self.nash_memo
        if memo[0] == v.hand_no:
            return memo[1]
        spot = None
        pre = self.tr.pre
        n = len(pre)
        if 2 <= n <= 4 and len(v.hole) == 2 and not v.ante and not self.tr.ante and \
                pre[-2] == v.sb_seat and pre[-1] == v.bb_seat:
            seats = (pre[-2], pre[-1]) + tuple(reversed(pre[:-2]))     # SB, BB, BTN, CO
            bb = v.bb
            st = tuple(v.start[s] / bb for s in seats)
            if max(st) <= self.P["ICM_NASH_CAP_BB"] and min(st) > 0:
                spot = (n, st, {s: i for i, s in enumerate(seats)})
        self.nash_memo = (v.hand_no, spot)
        return spot

    def nash_first_in(self, v, cls, pp, callers, S):
        """First-in jam from the equilibrium set at the nearest bucket: True / False, or None when no table applies.
        v2: the typing layer's adjustments still apply on top, as a range share: typed_jam(share of the set) wider
        than the set adds the push-order hands up to the new share; narrower (the light-jam cap vs callers) keeps
        only the set's hands inside the cap."""
        spot = self.nash_spot(v)
        if spot is None:
            return None
        n, st, pos = spot
        mask = _NT.lookup(n, self.nash_kind, st, ("push", pos[v.me]))
        if mask is None:
            return None
        inset = bool((mask >> cls) & 1)
        if self.version >= 2:
            Jt = mask_pct(mask)
            J2 = self.typed_jam(Jt, callers, S)
            if J2 > Jt:
                inset = inset or pp <= J2
            elif J2 < Jt:
                inset = inset and pp <= J2
        self.n_nash[0] += 1
        return inset

    def nash_call(self, v, jam, cls, tot):
        """Facing a first-in jam (no limpers, no other raise): call from the equilibrium call set, or over-call set
        after exactly one caller; None when no table applies (call_jam then decides). v2: a jammer whose estimated
        range deviates from its equilibrium jam range by more than ICM_NASH_CALL_DEV is priced by call_jam, anchored
        on that equilibrium range."""
        P = self.P
        tr = self.tr
        jammer = jam[0]
        if jam[2] or tr.pf_raises != 1 or tr.limpers or tr.open_seat != jammer:
            return None
        me = v.me
        if min(tot[me], tot[jammer]) / v.bb > P["ICM_NASH_MAX_BB"]:
            return None
        spot = self.nash_spot(v)
        if spot is None:
            return None
        n, st, pos = spot
        callers = []
        for j in range(v.n):
            if j == me or j == jammer or not v.in_hand[j]:
                continue
            if j in tr.jam_responders:
                callers.append(j)
            elif v.all_in[j]:
                return None                      # a forced all-in (short blind) makes it another game
        kind = self.nash_kind
        if not callers:
            node = ("call", pos[jammer], pos[me])
        elif len(callers) == 1:
            node = ("over", pos[jammer], pos[callers[0]], pos[me])
        else:
            return None
        mask = _NT.lookup(n, kind, st, node)
        if mask is None:
            return None
        if self.version >= 2 and not callers:
            pm = _NT.lookup(n, kind, st, ("push", pos[jammer]))
            Jn = mask_pct(pm) if pm else 0.0
            if Jn > 0.0:
                Je = self.jam_range_of(jammer, Jn, False)
                dev = P["ICM_NASH_CALL_DEV"]
                if Je > Jn * dev or Je * dev < Jn:
                    self.n_nash[3] += 1
                    return self.call_jam(v, jammer, Jn, False, cls, tot, near=False)
        self.n_nash[1 if not callers else 2] += 1
        return ("call",) if (mask >> cls) & 1 else ("fold",)

    def typed_jam(self, J, callers, S):
        """v2: cap light jams into callers; widen only on evidence (folder BB or a BB folding to jams; jam-tight
        seats behind)."""
        P = self.P
        types = [self.seat_type(j) for j in callers]
        if P["CALLER_CAP_JAM_EVIDENCE"]:
            capped = any(t in ("loose", "maniac") or (t == "caller" and self.caps_as_caller(j))
                         for j, t in zip(callers, types))
        else:
            capped = any(t in ("caller", "loose", "maniac") for t in types)
        if capped:
            return min(J, P["LIGHT_JAM_CAP"])
        mult = 1.0
        if len(callers) == 1:
            j = callers[0]
            st = self.stats[j]
            b10, slope, bmin = P["SB_WIDEN_BE"]                            # break-even jam-call rate of the widen
            be = b10 - slope * (S - 10.0) if S > 10.0 else b10
            be = max(bmin, be) - P["SB_WIDEN_MARGIN"]
            if types[0] == "folder" or (st.jam_faced >= P["MIN_JAMS_FACED"]
                                        and self.shrunk(st.jam_called, st.jam_faced, "calljam") < be):
                mult = P["SB_WIDEN"]
        if all(self.jam_tight(j) for j in callers):
            mult = max(mult, P["JAMFOLD_WIDEN"])
        J *= mult
        return J if J < 1.0 else 1.0

    def caps_as_caller(self, j):
        """CALLER_CAP_JAM_EVIDENCE: does a typed caller cap our jams? Only on jam-call evidence, or if its voluntary
        rate alone would have typed it loose (the caller branch comes first in seat_type)."""
        P = self.P
        st = self.stats[j]
        if st.jam_faced >= P["MIN_JAMS_FACED"] and \
                self.shrunk(st.jam_called, st.jam_faced, "calljam") >= P["T_CALLER_CALLJAM"]:
            return True
        return self.shrunk(st.pf_vol, st.pf_dec, "vol") >= P["T_LOOSE_VOL"]

    def open_range(self, r, v):
        """v2: estimated range of the raiser facing us (non-all-in): a first-in open gets the open prior by its
        position (floors for typed stealers / loose / maniacs; P6 floor for small late opens); a re-raise gets the
        tight re-jam prior unless the seat is a maniac."""
        P = self.P
        tr = self.tr
        t = self.seat_type(r)
        late_small = False
        if tr.pf_raises <= 1:
            k_r = tr.open_k if tr.open_seat == r else P["OPEN_K_UNKNOWN"]
            O = by_k(self.open_prior, k_r)
            if k_r <= P["LATE_SEATS"] and tr.raise_to <= P["BB_DEFEND_MAX_RAISE_BB"] * v.bb and \
                    tr.callers_since_raise == 0:
                late_small = True
                O = max(O, P["LATE_OPEN_FLOOR"])
            fl = P["J_FLOOR"].get(t)
            if fl is not None:
                O = max(O, fl)
        else:
            O = P["REJAM_PRIOR"]
            if t == "maniac":
                O = max(O, P["J_FLOOR"]["maniac"])
        return O, late_small

    def open_range_rm(self, r, v):
        """RANGE_MODEL version of open_range: the evidence-free prior (open prior by position, P6 late-small floor;
        re-raises: REJAM_PRIOR) x the seat's posterior ratio (open or 3-bet evidence), then the typing floors.
        Returns (O, late_small, scale) with scale = O / open_range's O, used for our flat / 3-bet / BB rules."""
        P = self.P
        tr = self.tr
        O0, late_small = self.open_range(r, v)
        t = self.seat_type(r)
        if tr.pf_raises <= 1:
            k_r = tr.open_k if tr.open_seat == r else P["OPEN_K_UNKNOWN"]
            O = by_k(self.open_prior, k_r)
            if late_small:
                O = max(O, P["LATE_OPEN_FLOOR"])
            O *= self.rm_open_ratio(r)
            fl = P["J_FLOOR"].get(t)
            if fl is not None:
                O = max(O, fl)
        else:
            O = P["REJAM_PRIOR"] * self.rm_jam_ratio(r, True)
            if t == "maniac":
                O = max(O, P["J_FLOOR"]["maniac"])
        if O > 1.0:
            O = 1.0
        return O, late_small, (O / O0 if O0 > 0.0 else 1.0)

    def call_jam(self, v, jammer, J_formula, is_rejam, cls, tot, near):
        """Call (or re-jam over a near-all-in raise) by equity against the jammer's estimated range."""
        P = self.P
        tr = self.tr
        me = v.me
        n = v.n
        J = self.jam_range_of(jammer, J_formula, is_rejam)
        if P["CALL_MODEL"] == "fit":
            S = min(tot[me], tot[jammer]) / v.bb                          # P4: the jammer's stack
            C = call_pct(J, S, P["CALL_FIT"]) * P["CALL_MULT"]
            if self.icm_on:
                C = C / self.bf(me, jammer) ** P["CALL_LAM"]
            ok = CALL_PCT[cls] <= C
            return (("allin",) if near else ("call",)) if ok else ("fold",)
        eq = eq_vs(cls, J)
        Jc = J * P["CALLER_RANGE_FRAC"]
        n_left = 0
        responders = tr.jam_responders if tr.jam is not None and tr.jam[0] == jammer else ()
        for j in range(n):
            if j == me or j == jammer or not v.in_hand[j]:
                continue
            if j in responders:
                if self.rm:                                               # its jam-call width, shrunk
                    eq *= eq_vs(cls, min(1.0, Jc * self.rm_call_ratio(j)))
                else:
                    eq *= eq_vs(cls, Jc)                                  # already called the jam
            elif v.all_in[j]:
                jr = tr.jam_ranges.get(j)                                 # an earlier jam: its own estimated range;
                eq *= eq_vs(cls, self.jam_range_of(j, jr[0], jr[1]) if jr else 1.0)   # all-in from a blind: any two
            else:
                n_left += 1
        R = v.to_call
        contrib = v.contrib
        if near:
            R = min(v.stacks[me], v.to_call + v.stacks[jammer])
        if R <= 0:
            return ("check",)
        mine = contrib[me] + R
        W = 0
        for j in range(n):
            if j != me:
                c = contrib[j]
                if near and j == jammer:
                    c += v.stacks[jammer]
                W += c if c < mine else mine
        # winning returns our own chips already in (contrib) plus W; risk premium BF on the R chips we add
        gain = W + contrib[me]
        if v.bounty is not None and "call" in P["BOUNTY_PARTS"]:
            gain += bounty_bonus(v, jammer)                               # bounty hit: only payers with chips left
        b = self.bf(me, jammer) if self.icm_on else 1.0
        need = b * R / (b * R + gain) + P["CALL_BEHIND_MARGIN"] * n_left + P["CALL_EQ_ADJ"]
        if eq >= need:
            return ("allin",) if near else ("call",)
        return ("fold",)

    # ------------------------------------------------------------------ postflop
    def postflop(self, v):
        P = self.P
        tr = self.tr
        me = v.me
        n = v.n
        ver = self.version
        h = v.hole
        if P["POSTFLOP_ENGINE"] and _PF is not None and _POST is None and len(h) == 2 and v.boards is None:
            d = self.postflop_engine(v)
            if d is not None:
                return d
        if _POST is not None:
            cls = post_class_variant(h, v.board, v.boards, v.street)    # the active variant's ranking and rule
        elif len(h) == 2 and v.boards is None:
            cls = hand_class(h[0], h[1], v.board)      # the pre-integration path (NLHE byte identity)
        else:
            cls = post_class(h, v.board, v.boards)     # 3+ cards or several boards; never raises
        pot = v.pot
        to_call = v.to_call
        stack = v.stacks[me]
        bets = v.bets
        cur = max(bets)
        opps = [j for j in range(n) if j != me and v.in_hand[j]]
        if not opps:
            return ("check",) if to_call <= 0 else ("call",)
        hu = len(opps) == 1
        if to_call <= 0:
            if cls >= 2:
                frac = P["VALUE_FRAC"]
                if ver >= 2 and any(self.seat_type(j) == "caller" for j in opps):
                    frac = P["VALUE_FRAC_CALLER"]
                return self.bet(v, cur + max(v.bb, int(frac * pot)))
            if ver >= 2 and hu:
                j = opps[0]
                if self.is_folder(j):
                    return ("raise", v.min_to)                            # E09: bet the minimum into a folder
                if v.street == 1 and tr.pf_aggr == me:
                    st = self.stats[j]
                    if st.flop_faced >= P["CBET_MIN_OBS"] and \
                            self.shrunk(st.flop_fold, st.flop_faced, "flop_fold") >= P["CBET_FOLD_T"]:
                        return self.bet(v, cur + max(v.bb, int(P["CBET_FRAC"] * pot)))
            if hu and v.street == 1 and tr.pf_aggr == me and P["CBET_AIR_UNKNOWN"] > 0 and \
                    self.rng.random() < P["CBET_AIR_UNKNOWN"]:
                return self.bet(v, cur + max(v.bb, int(P["CBET_FRAC"] * pot)))
            return ("check",)
        bettor = max(opps, key=lambda j: bets[j])
        big = to_call >= P["BIG_CALL_FRAC"] * stack
        b = self.bf(me, bettor) if (self.icm_on and big) else 1.0
        raised_me = bets[me] > 0
        light = ver >= 2 and self.seat_type(bettor) in P["LIGHT_CALL_TYPES"]
        if P["USE_REVEALED"] and 0 < cls < 3 and self.rev.hand_no == v.hand_no and bettor in self.rev.cards:
            cls = self.reveal_class(v, bettor, cls)
        if cls == 3:
            if big and b >= P["BF_RAISE3_MAX"]:
                return ("call",)
            return self.bet(v, int(P["RAISE_MULT"] * cur))
        ec = 2 if (light and cls >= 1) else cls
        if v.bounty is not None and "post" in P["BOUNTY_PARTS"]:
            pot = bounty_pot(v, pot)                                     # call thresholds see the bounty payoff
        if ec >= 2:
            if raised_me and not light:
                if to_call <= P["RAISED_CALL2_MAX_POT"] * pot and not big:
                    return ("call",)
                return ("fold",)
            if to_call <= P["CALL2_MAX_POT"] * pot and (not big or b < P["BF_CALL2_MAX"]):
                mb = P["CALL2_MAX_BET_POT"]
                if mb is None or to_call <= mb * pot_before_street(v):
                    return ("call",)
            return ("fold",)
        if ec == 1 and not raised_me and to_call <= P["CALL1_MAX_POT"] * pot and not big:
            return ("call",)
        return ("fold",)

    def postflop_engine(self, v):
        """POSTFLOP_ENGINE (POSTFLOP_ENGINE notes): bet / check / call / raise / fold from equity against an
        action-conditioned range (bot/postflop.py lookups). Widths come from the typing labels (v2; medium in v0/v1
        and for unknown seats), fold rates from each seat's observed postflop folds (v2). E09 (the minimum into a
        typed folder) and VALUE_FRAC_CALLER are kept. Returns None when there is nothing to decide here."""
        P = self.P
        cfg = getattr(self, "_pe_cfg", None)
        if cfg is None:
            cfg = dict(PARAMS["PE"])
            cfg.update(P["PE"] or {})
            cfg["VALUE_FRAC_CALLER"] = P["VALUE_FRAC_CALLER"]
            cfg["VALUE_FRAC"] = P["VALUE_FRAC"]
            cfg["RAISE_MULT"] = P["RAISE_MULT"]
            cfg["BF_RAISE_MAX"] = P["BF_RAISE3_MAX"]
            self._pe_cfg = cfg
        me = v.me
        bets = v.bets
        to_call = v.to_call
        in_hand = v.in_hand
        all_in = v.all_in
        seats = [j for j in range(v.n) if j != me and in_hand[j]]
        if not seats:
            return None
        ver = self.version
        tr = self.tr
        cur = max(bets)
        flop = v.street == 1
        typed = ver >= 2
        width_of = _PF.WIDTH_OF_TYPE
        style_of = _PF.BET_STYLE_OF_TYPE
        mo, ft = cfg["BLUFF_MIN_OBS"], cfg["BLUFF_FOLD_T"]
        opps = []
        caller_vs = False
        active = 0
        for j in seats:
            ai = bool(all_in[j])
            if typed:
                t = self.seat_type(j)
                w = width_of.get(t, 1)
                sty = style_of.get(t, 1)
                if t == "caller":
                    caller_vs = True
            else:
                w = sty = 1
            if typed and not ai:
                st = self.stats[j]
                fh, fn = st.post_fold, st.post_faced
                ok = fn >= mo and self.shrunk(fh, fn, "flop_fold") >= ft
            else:
                fh = fn = 0
                ok = False
            if not ai:
                active += 1
            opps.append((j, w, fh, fn, ai or (cur > 0 and bets[j] >= cur), ok, flop and tr.pf_aggr == j, ai, sty))
        if to_call <= 0:
            if active == 0:
                return ("check",)
            if typed and len(seats) == 1 and self.is_folder(seats[0]):
                return ("raise", v.min_to)                                # E09: bet the minimum into a folder
            bettor = -1
            b = 1.0
            pot = v.pot
        else:
            bettor = max(seats, key=lambda j: bets[j])
            stack = v.stacks[me]
            big = to_call >= P["BIG_CALL_FRAC"] * stack
            b = self.bf(me, bettor) if (self.icm_on and big) else 1.0
            pot = v.pot
            if v.bounty is not None and "post" in P["BOUNTY_PARTS"]:
                pot = bounty_pot(v, pot)
        last = True                                # nobody still to act after us on this street?
        if to_call <= 0:
            ss = v.street_starts
            acted = set()
            if v.street < len(ss):
                for e in v.history[ss[v.street]:]:
                    acted.add(e[0])
            for o in opps:
                if not o[7] and o[0] not in acted:
                    last = False
                    break
        h = v.hole
        d = _PF.decide(h[0], h[1], v.board, pot, to_call, v.stacks[me], opps, bettor, bets[me] > 0, b, caller_vs,
                       cur, v.bb, last, cfg, self.rng)
        if d is None:
            return None
        k = d[0]
        if k == "bet" or k == "raise":
            return self.bet(v, d[1])
        return d

    def reveal_class(self, v, j, cls):
        """USE_REVEALED: our call class against bettor j moved one step by the reader model's read of j's revealed
        discards (reveal_delta): down when the read makes j REVEAL_T more likely to be ahead than an unread hand,
        a weak pair up to top-pair calls when it makes j REVEAL_T less likely. Cached per (seat, board, class)."""
        rv = self.rev
        key = (j, len(v.board), cls)
        d = rv.cache.get(key)
        if d is None:
            lik = rv.lik.get(j)
            if lik is None:
                lik = rv.lik[j] = reveal_lik([c >> 2 for c in rv.cards[j]], self.P["REVEAL_EPS"])
            nb = len(v.board)
            rows = rv.rows.get(nb)
            if rows is None:
                bset = set(v.board)
                dead = [c for cs in rv.cards.values() for c in cs if c not in bset]
                rows = rv.rows[nb] = reveal_board(v.hole, v.board, dead)
            d = rv.cache[key] = reveal_delta_rows(rows, lik, cls)
        t = self.P["REVEAL_T"]
        if d >= t:
            self.n_reveal_adj[0] += 1
            return cls - 1
        if d <= -t and cls == 1:
            self.n_reveal_adj[1] += 1
            return 2
        return cls

    def bet(self, v, to):
        """Bet or raise to `to` (street total); all-in if it would commit COMMIT_FRAC of our chips."""
        me = v.me
        chips = v.stacks[me] + v.bets[me]
        if to >= self.P["COMMIT_FRAC"] * chips:
            return ("allin",)
        return ("raise", to)
