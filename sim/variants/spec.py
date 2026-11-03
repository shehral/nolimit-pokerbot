"""Variant definitions: what changes versus the kit's plain 8-max no-limit hold'em.

make_variant(name, **options) -> Variant. Names: "plain" (kit parity, for speed comparison), "toss", "shortdeck",
"bounty", "doubleboard", "antes" (ante of 0.125 bb from everyone from level 2), "bbante" (a big-blind ante: the BB
seat posts one extra big blind of dead money from level 2). A Variant is a small picklable object; the engine
(engine.VTable) reads its fields, agents see its name and options in config["variant"] / config["variant_options"].
"""
from .evals import (CATEGORY_NAMES_52, CATEGORY_NAMES_SHORT, CATEGORY_NAMES_SHORT_TRIPS_OVER_STRAIGHT, FULL_DECK,
                    SHORT_DECK, evaluate_any, evaluate_short, evaluate_short_trips)

try:
    from sim.kit.cards import evaluate as kit_evaluate
except ImportError:   # pragma: no cover
    from cards import evaluate as kit_evaluate

NAMES = ("plain", "toss", "shortdeck", "bounty", "doubleboard", "antes", "bbante")


class Variant:
    """Fields read by the engine.

    name, options          label and the options it was built with
    deck                   tuple of card ints in play (36 for shortdeck, 52 otherwise)
    hole_dealt             hole cards dealt per seat (3 for toss)
    toss                   True: after the flop every contender discards one hole card face up to the board
    n_boards               1, or 2 for doubleboard
    street_new             cards dealt on flop, turn, river (per board)
    evaluate               score function, kit signature: evaluate(list of card ints) -> int
    bounty                 None, or dict(mode="level"|"hand", mult=0.5, flat=10)
    ante_mode              "all" (everyone antes) or "bb" (only the big blind posts the ante)
    ante_frac, ante_start  ante = round(frac * bb) chips (min 1), from level ante_start (1-based); 0 = no ante
    discard_error          what an error/timeout on a discard request does: "fold" (strict) or "lowest"
    """

    def __init__(self, name, options=None, *, deck=FULL_DECK, hole_dealt=2, toss=False, n_boards=1,
                 street_new=(3, 1, 1), evaluate=kit_evaluate, categories=CATEGORY_NAMES_52, bounty=None,
                 ante_mode="all", ante_frac=0.0, ante_start=2, discard_error="fold"):
        self.name = name
        self.options = dict(options or {})
        self.deck = tuple(deck)
        self.hole_dealt = hole_dealt
        self.toss = toss
        self.n_boards = n_boards
        self.street_new = tuple(street_new)
        self.board_len = sum(self.street_new)
        self.evaluate = evaluate
        self.categories = categories
        self.bounty = bounty
        self.ante_mode = ante_mode
        self.ante_frac = ante_frac
        self.ante_start = ante_start
        self.discard_error = discard_error
        self.need_cards_per_seat = hole_dealt
        if len(self.deck) < 8 * hole_dealt + n_boards * self.board_len:
            raise ValueError("deck too small for 8 seats")

    def deck_cards_needed(self, n_seats):
        return self.hole_dealt * n_seats + self.n_boards * self.board_len

    def apply_schedule(self, schedule):
        """Return the blind schedule with this variant's antes filled in (levels are (sb, bb, ante))."""
        if not self.ante_frac:
            return tuple(schedule)
        out = []
        for lv, (sb, bb, ante) in enumerate(schedule):
            if lv + 1 >= self.ante_start:
                ante = max(1, int(round(self.ante_frac * bb)))
            out.append((sb, bb, ante))
        return tuple(out)

    def public_options(self):
        return dict(self.options)

    def __repr__(self):
        return f"Variant({self.name!r}, {self.options!r})"


def make_variant(name, **opt):
    """Build a Variant. Options (all optional):
    toss:        discard_error="fold"|"lowest"
    shortdeck:   trips_beat_straight=False
    bounty:      mode="level" (new hidden rank per seat every blind level, 25 hands) or "hand"; mult=0.5; flat=10
    antes:       frac=0.125, start_level=2
    bbante:      frac=1.0, start_level=2
    """
    n = name.lower()
    if n == "plain":
        return Variant("plain", opt)
    if n == "toss":
        return Variant("toss", opt, hole_dealt=3, toss=True, street_new=(2, 1, 1), evaluate=evaluate_any,
                       discard_error=opt.get("discard_error", "fold"))
    if n == "shortdeck":
        tbs = bool(opt.get("trips_beat_straight", False))
        return Variant("shortdeck", opt, deck=SHORT_DECK,
                       evaluate=evaluate_short_trips if tbs else evaluate_short,
                       categories=CATEGORY_NAMES_SHORT_TRIPS_OVER_STRAIGHT if tbs else CATEGORY_NAMES_SHORT)
    if n == "bounty":
        mode = opt.get("mode", "level")
        if mode not in ("level", "hand"):
            raise ValueError("bounty mode must be 'level' or 'hand'")
        return Variant("bounty", opt, bounty={"mode": mode, "mult": float(opt.get("mult", 0.5)),
                                              "flat": int(opt.get("flat", 10))})
    if n == "doubleboard":
        return Variant("doubleboard", opt, n_boards=2)
    if n == "antes":
        return Variant("antes", opt, ante_mode="all", ante_frac=float(opt.get("frac", 0.125)),
                       ante_start=int(opt.get("start_level", 2)))
    if n == "bbante":
        return Variant("bbante", opt, ante_mode="bb", ante_frac=float(opt.get("frac", 1.0)),
                       ante_start=int(opt.get("start_level", 2)))
    raise ValueError(f"unknown variant {name!r}; known: {NAMES}")
