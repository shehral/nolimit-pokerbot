"""Mock starter kit for the No Limit rehearsal (see the top-level README): engine, Agent interface, blind schedules, payouts,
exact ICM, runner with seeded seating and PAIRED mode. Import as `sim.kit` from the project root."""
from .agent import (ACTIONS, Agent, all_in, call, check, check_or_call, check_or_fold, clamp_raise, fold, raise_to,
                    validate_action)
from .cards import CARD_INT, CARD_STR, category, evaluate, evaluate_str
from .engine import EngineError, SeatStats, Table, TableResult, make_deck, play_table
from .icm import icm_brute, icm_equities, make_icm
from .payouts import LINEAR, PAYOUTS, TOP3, TOP4, WTA, average_seat, get_payouts
from .runner import Entrant, RunResult, run, run_paired, seating_plan
from .schedules import FAST, SCHEDULES, SLOW, TURBO, blinds_for_hand, get_schedule

__all__ = [
    "ACTIONS", "Agent", "all_in", "call", "check", "check_or_call", "check_or_fold", "clamp_raise", "fold",
    "raise_to", "validate_action", "CARD_INT", "CARD_STR", "category", "evaluate", "evaluate_str", "EngineError",
    "SeatStats", "Table", "TableResult", "make_deck", "play_table", "icm_brute", "icm_equities", "make_icm",
    "LINEAR", "PAYOUTS", "TOP3", "TOP4", "WTA", "average_seat", "get_payouts", "Entrant", "RunResult", "run",
    "run_paired", "seating_plan", "FAST", "SCHEDULES", "SLOW", "TURBO", "blinds_for_hand", "get_schedule",
]
