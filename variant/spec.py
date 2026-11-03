"""VariantSpec: the one object that describes the game (variants_deep notes s6.1, R1). Pure standard library.

Nothing else in the variant layer hard-codes 52 cards, 2 hole cards, 169 classes or 5 board cards: HandSpace,
build_tables and the safe defaults read them from a VariantSpec.

Build one in minutes from the kit's engine source with spec_from_kit(evaluator, ...) (it also probes the evaluator:
card counts it accepts, speed, category order on hand-checked cases). Worked examples: nlhe_spec() reproduces the
bot's current 169-class setup; short_deck_spec(), three_keep_two_spec(), omaha_spec(), toss_spec() are rehearsal
shapes.
"""
import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional, Tuple

from .cards import (STD_RANKS, STD_SUITS, Codec, evaluate52, evaluate_short, evaluate_upto9)

REQUEST_KINDS = ("discard", "keep", "toss", "assign", "bid", "redraw", "other")
# response formats a request can expect (the adapter converts the kit's request into one of these)
RESPONSE_FORMATS = ("keep_indices", "discard_indices", "keep_cards", "discard_cards", "index", "card",
                    "int", "index_or_none", "groups", "option")


@dataclass
class RequestType:
    """A special (non-betting) request of the variant.

    kind      one of REQUEST_KINDS
    name      the kit's own name (e.g. "DiscardAction", "ActionType.DISCARD"), for the adapter
    response  RESPONSE_FORMATS entry: the shape of a legal answer
    k         keep: cards kept; discard/toss: cards given up; redraw: max cards redrawn
    optional  the request may be declined (redraw); declining is then the safe default
    legal     optional kit legality rule callable(view, response) -> bool, checked on top of ours
    default   optional template default callable(view) -> response (the kit template's own answer)
    """
    kind: str
    name: str = ""
    response: str = "keep_indices"
    k: Optional[int] = None
    optional: bool = False
    legal: Optional[Callable] = None
    default: Optional[Callable] = None
    notes: str = ""

    def __post_init__(self):
        if self.kind not in REQUEST_KINDS:
            raise ValueError(f"unknown request kind {self.kind!r}; use one of {REQUEST_KINDS}")
        if self.response not in RESPONSE_FORMATS:
            raise ValueError(f"unknown response format {self.response!r}; use one of {RESPONSE_FORMATS}")


def static_score(cards, slot=4):
    """Cheap static strength of a set of hole cards (any size): high cards + pairs (by rank) + suitedness +
    connectivity. Used to pick the kept subset and in the safe defaults; never rolls out. About 2-4 us for 2 cards."""
    ranks = sorted((c // slot for c in cards), reverse=True)
    score = 0.0
    for r in ranks:
        score += r + 1
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    for r, m in cnt.items():
        if m >= 2:
            score += (m * (m - 1) // 2) * (14 + r)
    sc = {}
    for c in cards:
        s = c % slot
        sc[s] = sc.get(s, 0) + 1
    for m in sc.values():
        if m >= 2:
            score += 3 * (m - 1)
    distinct = sorted(cnt, reverse=True)
    for a, b in zip(distinct, distinct[1:]):
        g = a - b
        if g <= 3:
            score += 4 - g
    return score


@dataclass
class VariantSpec:
    """Everything the variant layer needs to know about the game.

    deck          rank_chars / suit_chars (encoding alphabet, low to high) minus removed ranks, suits, cards
    hole_dealt    cards dealt to each player; hole_kept: cards kept when the player must choose (None = no choice)
    keep_street   when the choice is made: 0 = right after the deal (preflop), 1 = after the first board street ...
    discard_to    "muck" (dead) or "board" (tossed face up to the board, MIT 2026 Toss)
    discards_public  are discards revealed to everybody
    board_schedule   cards dealt on each street after preflop, e.g. (3, 1, 1); () = no board
    n_boards      1, or 2 for a double board (pot split per board)
    evaluator     callable(cards: list[int]) -> comparable, higher better (the kit's evaluator)
    must_use      None ("any combination") or (k_hole, k_board) for an Omaha-style rule
    hi_lo, lo_evaluator  split pot: lo_evaluator(cards) -> comparable (higher = better low) or None (no qualifier)
    requests      tuple of RequestType (special requests the engine sends)
    revealed      free-text list of public events beyond hold'em (e.g. "discards", "bids", "one hole card")
    ante_bb, straddle, payouts, blind_schedule: forced bets and payout (the ICM layer reads payouts)
    keep_score    optional callable(cards) -> float to pick kept subsets (default static_score)
    share_fn      optional callable(spec, dealt_hands, board) -> pot shares, for showdowns the default cannot express
    """
    name: str = "custom"
    rank_chars: str = STD_RANKS
    suit_chars: str = STD_SUITS
    removed_ranks: str = ""
    removed_suits: str = ""
    removed_cards: Tuple[str, ...] = ()
    hole_dealt: int = 2
    hole_kept: Optional[int] = None
    keep_street: int = 0
    discard_to: str = "muck"
    discards_public: bool = False
    board_schedule: Tuple[int, ...] = (3, 1, 1)
    n_boards: int = 1
    evaluator: Callable = evaluate52
    must_use: Optional[Tuple[int, int]] = None
    hi_lo: bool = False
    lo_evaluator: Optional[Callable] = None
    requests: Tuple[RequestType, ...] = ()
    revealed: Tuple[str, ...] = ()
    ante_bb: float = 0.0
    straddle: bool = False
    payouts: Optional[Tuple[float, ...]] = None
    blind_schedule: Optional[Tuple] = None
    keep_score: Optional[Callable] = None
    share_fn: Optional[Callable] = None
    notes: str = ""
    _deck: tuple = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ deck and codec
    @property
    def codec(self):
        return Codec(self.rank_chars, self.suit_chars)

    @property
    def deck(self):
        if self._deck is None:
            cd = self.codec
            rm = set(cd.to_int(c) for c in self.removed_cards)
            out = []
            for r, rc in enumerate(self.rank_chars):
                if rc in self.removed_ranks:
                    continue
                for s, sc in enumerate(self.suit_chars):
                    if sc in self.removed_suits:
                        continue
                    c = cd.make(r, s)
                    if c not in rm:
                        out.append(c)
            self._deck = tuple(out)
        return self._deck

    @property
    def n_kept(self):
        return self.hole_kept if self.hole_kept else self.hole_dealt

    @property
    def board_cards(self):
        return sum(self.board_schedule) * self.n_boards

    @property
    def separable(self):
        """True when a player's showdown value depends only on its own dealt cards and the board (then the fast
        rank-every-hand engines apply); False for interacting showdowns (toss to board, hi/lo, double board,
        custom share_fn), which use the generic rollout engine."""
        return (self.share_fn is None and not self.hi_lo and self.n_boards == 1
                and not (self.hole_kept and self.discard_to == "board"))

    # ------------------------------------------------------------------ hands
    def hand_value(self, hole, board):
        """Showdown value of the cards a player plays (after any keep choice) on a full board."""
        ev = self.evaluator
        if self.must_use is None:
            return ev(list(hole) + list(board))
        kh, kb = self.must_use
        best = None
        for h in combinations(hole, kh):
            for b in combinations(board, kb):
                v = ev(list(h) + list(b))
                if best is None or v > best:
                    best = v
        return best

    def keep(self, dealt, board_known=()):
        """The kept subset (tuple) of a dealt hand. Before the board: best by keep_score / static_score. With at
        least 3 known board cards (choice after the flop): best by the evaluator on kept + board, static tie-break.
        Deterministic: ties go to the first subset in the dealt order."""
        k = self.hole_kept
        if not k or k >= len(dealt):
            return tuple(dealt)
        score = self.keep_score or static_score
        best, best_key = None, None
        use_ev = len(board_known) >= 3 and self.must_use is None and k + len(board_known) >= 5
        for sub in combinations(dealt, k):
            if use_ev:
                try:
                    key = (self.evaluator(list(sub) + list(board_known)), score(sub))
                except Exception:
                    key = (0, score(sub))
            else:
                key = (score(sub),)
            if best_key is None or key > best_key:
                best, best_key = sub, key
        return tuple(best)

    def known_board_at_keep(self, board):
        """Board cards visible when the keep choice is made."""
        return tuple(board[:sum(self.board_schedule[:self.keep_street])]) if self.keep_street else ()

    def value(self, dealt, board):
        """Separable showdown value of a DEALT hand on a full board (keep applied)."""
        if self.hole_kept:
            dealt = self.keep(dealt, self.known_board_at_keep(board))
        return self.hand_value(dealt, board)

    def shares(self, dealt_hands, board):
        """Pot share of each player at showdown (all players reach showdown; ties split). Handles keep, toss to
        board, double board and hi/lo; share_fn overrides everything."""
        if self.share_fn is not None:
            return self.share_fn(self, dealt_hands, board)
        n = len(dealt_hands)
        nb = sum(self.board_schedule)
        boards = [tuple(board[i * nb:(i + 1) * nb]) for i in range(self.n_boards)] if self.n_boards > 1 else [tuple(board)]
        kept = []
        extra = []
        for d in dealt_hands:
            if self.hole_kept:
                kk = self.keep(d, self.known_board_at_keep(boards[0]))
                kept.append(kk)
                if self.discard_to == "board":
                    extra.extend(c for c in d if c not in kk)
            else:
                kept.append(tuple(d))
        out = [0.0] * n
        w = 1.0 / len(boards)
        for bd in boards:
            full = bd + tuple(extra)
            hv = [self.hand_value(kept[i], full) for i in range(n)]
            if self.hi_lo and self.lo_evaluator is not None:
                lv = [self.lo_evaluator(list(kept[i]) + list(full)) for i in range(n)]
                if any(v is not None for v in lv):
                    _split(out, hv, w * 0.5)
                    _split(out, lv, w * 0.5)
                    continue
            _split(out, hv, w)
        return out

    # ------------------------------------------------------------------ description
    def summary(self):
        d = self.deck
        req = ", ".join(f"{r.kind}({r.response}{', k=' + str(r.k) if r.k else ''})" for r in self.requests) or "none"
        return (f"{self.name}: deck {len(d)} cards, dealt {self.hole_dealt}"
                f"{', keep ' + str(self.hole_kept) + ' at street ' + str(self.keep_street) + ' to ' + self.discard_to if self.hole_kept else ''}"
                f", board {self.board_schedule} x{self.n_boards}, must_use {self.must_use}, hi_lo {self.hi_lo}, "
                f"requests {req}, ante {self.ante_bb} bb, separable {self.separable}")


def _split(out, vals, w):
    best = None
    for v in vals:
        if v is not None and (best is None or v > best):
            best = v
    if best is None:
        return
    win = [i for i, v in enumerate(vals) if v == best]
    s = w / len(win)
    for i in win:
        out[i] += s


# ===================================================================== kit helper
def probe_evaluator(spec, n_calls=2000, seed=7):
    """Check a kit evaluator BEFORE building any table (variants_deep s5.1 D: "test the evaluator on 10 hand-checked
    cases"). Returns a dict: card counts accepted, microseconds per call, category order seen on hand-checked hands
    (standard encoding only), determinism and card-order invariance. Never raises."""
    rep = {"ok": True, "errors": []}
    rng = random.Random(seed)
    deck = list(spec.deck)
    n_play = spec.n_kept + sum(spec.board_schedule) + (spec.hole_dealt - spec.n_kept if spec.discard_to == "board" else 0)
    accepted = {}
    for n in range(5, min(len(deck), max(n_play, 7)) + 1):
        try:
            spec.evaluator(rng.sample(deck, n))
            accepted[n] = True
        except Exception as e:  # noqa: BLE001
            accepted[n] = f"{type(e).__name__}: {e}"
    rep["accepts"] = accepted
    if accepted.get(n_play) is not True:
        rep["ok"] = False
        rep["errors"].append(f"evaluator fails on {n_play} cards (hole kept + board)")
    hands = [rng.sample(deck, n_play if accepted.get(n_play) is True else 5) for _ in range(n_calls)]
    t0 = time.perf_counter()
    try:
        vals = [spec.evaluator(h) for h in hands]
        rep["us_per_call"] = round((time.perf_counter() - t0) / n_calls * 1e6, 2)
        rep["deterministic"] = all(spec.evaluator(h) == v for h, v in zip(hands[:200], vals[:200]))
        rep["order_invariant"] = all(spec.evaluator(list(reversed(h))) == v for h, v in zip(hands[:200], vals[:200]))
        if not (rep["deterministic"] and rep["order_invariant"]):
            rep["ok"] = False
            rep["errors"].append("evaluator not deterministic or depends on card order")
    except Exception as e:  # noqa: BLE001
        rep["ok"] = False
        rep["errors"].append(f"evaluator raised {type(e).__name__}: {e}")
    # hand-checked categories (needs the standard encoding and the cards to exist in the deck)
    cd = spec.codec
    cases = {
        "straight_flush": "9h 8h 7h 6h Th", "quads": "Ks Kh Kd Kc 7s", "full_house": "Qs Qh Qd 9c 9s",
        "flush": "As Js 8s 7s 6s", "straight": "Ts 9h 8d 7c Js", "trips": "8s 8h 8d Ac 6s",
        "two_pair": "As Ah 9d 9c 7s", "pair": "Ks Kh 9d 8c 6s", "high_card": "As Qh 9d 8c 6s",
        "wheel": "As 2h 3d 4c 5s", "short_wheel": "As 6h 7d 8c 9s",
    }
    dset = set(deck)
    seen = {}
    for name, s in cases.items():
        try:
            cs = [cd.to_int(x) for x in s.split()]
        except Exception:  # noqa: BLE001
            continue
        if all(c in dset for c in cs):
            try:
                seen[name] = spec.evaluator(cs)
            except Exception as e:  # noqa: BLE001
                rep["errors"].append(f"{name}: {type(e).__name__}")
    rep["category_order"] = [k for k, _ in sorted(seen.items(), key=lambda kv: kv[1])]
    if "flush" in seen and "full_house" in seen:
        rep["flush_beats_full_house"] = seen["flush"] > seen["full_house"]
    if "straight" in seen and "trips" in seen:
        rep["straight_beats_trips"] = seen["straight"] > seen["trips"]
    if "wheel" in seen and "high_card" in seen:
        rep["wheel_is_straight"] = seen["wheel"] > seen.get("trips", seen["high_card"])
    if "short_wheel" in seen and "high_card" in seen:
        rep["a6789_is_straight"] = seen["short_wheel"] > seen.get("trips", seen["high_card"])
    return rep


def spec_from_kit(evaluator, name="kit", hole_dealt=2, hole_kept=None, keep_street=0, discard_to="muck",
                  discards_public=False, board=(3, 1, 1), n_boards=1, removed_ranks="", removed_suits="",
                  removed_cards=(), rank_chars=STD_RANKS, suit_chars=STD_SUITS, must_use=None, hi_lo=False,
                  lo_evaluator=None, requests=(), revealed=(), ante_bb=0.0, payouts=None, probe=True, **extra):
    """Skeleton for event day: fill the arguments from the kit's engine source (the evaluator import, the deal and
    board code, the new action namedtuple/enum) and get a VariantSpec plus an evaluator probe report.

    Example (MIT-2026-like toss):  spec, rep = spec_from_kit(kit.evaluate, hole_dealt=3, hole_kept=2, keep_street=1,
        discard_to="board", discards_public=True, board=(2, 1, 1),
        requests=(RequestType("toss", "DiscardAction", response="index", k=1),))
    """
    spec = VariantSpec(name=name, evaluator=evaluator, hole_dealt=hole_dealt, hole_kept=hole_kept,
                       keep_street=keep_street, discard_to=discard_to, discards_public=discards_public,
                       board_schedule=tuple(board), n_boards=n_boards, removed_ranks=removed_ranks,
                       removed_suits=removed_suits, removed_cards=tuple(removed_cards), rank_chars=rank_chars,
                       suit_chars=suit_chars, must_use=must_use, hi_lo=hi_lo, lo_evaluator=lo_evaluator,
                       requests=tuple(requests), revealed=tuple(revealed), ante_bb=ante_bb,
                       payouts=tuple(payouts) if payouts else None, **extra)
    rep = probe_evaluator(spec) if probe else None
    return spec, rep


# ===================================================================== worked examples
def nlhe_spec():
    """Plain no-limit hold'em: the bot's current setup (169 classes, 1,326 combos, board 3-1-1)."""
    return VariantSpec(name="nlhe", evaluator=evaluate52)


def short_deck_spec():
    """36-card short deck (6..A), flush beats full house, A-6-7-8-9 is the lowest straight: 81 classes, 630 combos."""
    return VariantSpec(name="short_deck", removed_ranks="2345", evaluator=evaluate_short)


def three_keep_two_spec(keep_street=0, discards_public=False):
    """3 hole cards dealt, keep 2 (discard 1 to the muck) right after the deal; showdown best 5 of 7: 1,755 classes."""
    return VariantSpec(name="three_keep_two", hole_dealt=3, hole_kept=2, keep_street=keep_street,
                       discards_public=discards_public, evaluator=evaluate52,
                       requests=(RequestType("discard", "DiscardAction", response="discard_indices", k=1),),
                       revealed=("discards",) if discards_public else ())


def omaha_spec():
    """4 hole cards, exactly 2 hole + 3 board (Omaha rule): 16,432 classes (rollout engine; tables are slow)."""
    return VariantSpec(name="omaha", hole_dealt=4, must_use=(2, 3), evaluator=evaluate52)


def toss_spec():
    """MIT 2026 Toss or Hold'em: 3 hole cards, 2-card flop, each player tosses one card face up to the board, then
    turn and river; best 5 of 8. Interacting showdown (rollout engine)."""
    return VariantSpec(name="toss", hole_dealt=3, hole_kept=2, keep_street=1, discard_to="board",
                       discards_public=True, board_schedule=(2, 1, 1), evaluator=evaluate_upto9,
                       requests=(RequestType("toss", "DiscardAction", response="index", k=1),),
                       revealed=("tossed card",))


EXAMPLES = {"nlhe": nlhe_spec, "short_deck": short_deck_spec, "three_keep_two": three_keep_two_spec,
            "omaha": omaha_spec, "toss": toss_spec}
