"""Adapter: the ONLY file that knows the starter kit's interface (mock kit in sim/kit; a real platform's kit would replace it).

normalize(state) -> View    kit state dict -> the brain's normalized view (card ints, positions, totals)
to_action(decision, state)  internal decision -> a legal kit action (clamped to the legal bounds)
is_legal(action, state)     the kit's legality rule, re-implemented here (the bot must not import the kit)
safe_action(state)          check if legal, else fold (the rules' own fallback)

Internal decisions (from brain.py): ("fold",) ("check",) ("call",) ("raise", raise_to_total) ("allin",).
Kit actions: {"action": "fold" | "check" | "call" | "raise", "amount": raise_to_total}; all-in = raise to
max_raise_to, or call when the stack does not cover the call.

PLATFORM: when the real interface differs, change only ADAPTER_PARAMS and the two functions marked PLATFORM below
(field names, card format, raise semantics, action names). The brain and the shell do not change.
ADAPTER_PARAMS holds interface facts only; tunables (such as the all-in snap) live in brain.PARAMS.
"""

ADAPTER_PARAMS = {
    "RAISE_SEMANTICS": "to",     # "to" (kit: amount = street total), "by" (amount = increase over the current bet),
                                 # "add" (amount = chips added now); bounds are read in the same semantics
    "ACTION_NAMES": {"fold": "fold", "check": "check", "call": "call", "raise": "raise"},
    "STREETS": {"preflop": 0, "flop": 1, "turn": 2, "river": 3},
}

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_RANK = {r: i for i, r in enumerate(RANKS)}
_RANK.update({r.lower(): i for i, r in enumerate(RANKS)})
_RANK["10"] = 8
_SUIT = {s: i for i, s in enumerate(SUITS)}
_SUIT.update({s.upper(): i for i, s in enumerate(SUITS)})
_SUIT.update({"♣": 0, "♦": 1, "♥": 2, "♠": 3})
_CARD_CACHE = {}


def parse_card(c):
    """Card -> int 0..51 (rank = c >> 2 with 0 = deuce, suit = c & 3 with c, d, h, s). Accepts 'Ah', 'AH', 'ah',
    '10h', 'Th', suit symbols, and ints already in this encoding. Raises on anything else."""
    v = _CARD_CACHE.get(c)
    if v is not None:
        return v
    if isinstance(c, int) and not isinstance(c, bool):
        if 0 <= c < 52:
            return c
        raise ValueError(f"card int out of range: {c!r}")
    s = str(c).strip()
    r, su = s[:-1], s[-1]
    v = (_RANK[r] << 2) | _SUIT[su]
    _CARD_CACHE[c] = v
    return v


class View:
    """Normalized decision view. Lists are indexed by seat 0..n-1."""
    __slots__ = ("hand_no", "street", "me", "n", "hole", "board", "sb", "bb", "ante", "pot", "stacks", "bets",
                 "contrib", "start", "in_hand", "all_in", "alive", "to_call", "min_to", "max_to", "legal",
                 "button", "sb_seat", "bb_seat", "history", "street_starts", "payouts", "n_alive", "hands_left",
                 "level")


def _sem_to_total(x, state, sem):
    if sem == "to":
        return x
    bets = state.get("bets") or []
    seat = state["seat"]
    if sem == "by":
        return x + (max(bets) if bets else 0)
    return x + (bets[seat] if bets else 0)          # "add"


def _total_to_sem(t, state, sem):
    if sem == "to":
        return t
    bets = state.get("bets") or []
    seat = state["seat"]
    if sem == "by":
        return t - (max(bets) if bets else 0)
    return t - (bets[seat] if bets else 0)


def normalize(state):
    """PLATFORM: map the real state's fields here. Missing optional fields are derived where possible."""
    v = View()
    sem = ADAPTER_PARAMS["RAISE_SEMANTICS"]
    me = state["seat"]
    stacks = list(state["stacks"])
    n = len(stacks)
    bets = list(state.get("bets") or [0] * n)
    contrib = state.get("contributed")
    contrib = list(contrib) if contrib is not None else list(bets)
    start = state.get("start_stacks")
    start = list(start) if start is not None else [stacks[j] + contrib[j] for j in range(n)]
    active = state.get("active")
    alive = state.get("alive")
    alive = list(alive) if alive is not None else [start[j] > 0 for j in range(n)]
    v.hand_no = state.get("hand_no", 0)
    st = state.get("street", 0)
    v.street = ADAPTER_PARAMS["STREETS"][st] if isinstance(st, str) else int(st)
    v.me = me
    v.n = n
    h = state["hole"]
    v.hole = (parse_card(h[0]), parse_card(h[1]))
    v.board = [parse_card(c) for c in (state.get("board") or ())]
    v.sb = state.get("sb", 0)
    v.bb = state["bb"]
    v.ante = state.get("ante", 0) or 0
    v.pot = state.get("pot", sum(contrib))
    v.stacks = stacks
    v.bets = bets
    v.contrib = contrib
    v.start = start
    v.in_hand = list(active) if active is not None else list(alive)
    ai = state.get("all_in")
    v.all_in = list(ai) if ai is not None else [v.in_hand[j] and stacks[j] == 0 for j in range(n)]
    v.alive = alive
    v.to_call = state.get("to_call", max(bets) - bets[me])
    legal = state["legal_actions"]
    v.legal = legal
    if "raise" in legal:
        v.min_to = _sem_to_total(state["min_raise_to"], state, sem)
        v.max_to = _sem_to_total(state["max_raise_to"], state, sem)
    else:
        v.min_to = v.max_to = 0
    btn = state.get("button", 0)
    v.button = btn
    sbs, bbs = state.get("sb_seat"), state.get("bb_seat")
    if sbs is None or bbs is None:
        order = [(btn + d) % n for d in range(1, n + 1) if alive[(btn + d) % n]]
        if len(order) == 2:
            sbs, bbs = btn, order[0]
        elif len(order) > 2:
            sbs, bbs = order[0], order[1]
        else:
            sbs = bbs = btn
    v.sb_seat, v.bb_seat = sbs, bbs
    v.history = state.get("history_this_hand") or []
    v.street_starts = state.get("street_starts") or [0]
    v.payouts = state.get("payouts")
    v.n_alive = state.get("n_alive", sum(1 for a in alive if a))
    v.hands_left = state.get("hands_left", 0)
    v.level = state.get("level", 1)
    return v


def safe_action(state):
    try:
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
    except Exception:
        return {"action": "fold"}


def to_action(dec, state, snap=0.85):
    """PLATFORM: internal decision -> the kit's action dict, legal by construction (clamped, all-in converted).
    snap = brain.PARAMS["ALLIN_SNAP"]: a raise-to at or above this share of max_raise_to goes all-in."""
    names = ADAPTER_PARAMS["ACTION_NAMES"]
    legal = state["legal_actions"]
    kind = dec[0]
    can_check = "check" in legal
    can_call = "call" in legal
    can_raise = "raise" in legal
    if kind == "raise" or kind == "allin":
        if can_raise:
            sem = ADAPTER_PARAMS["RAISE_SEMANTICS"]
            lo = _sem_to_total(state["min_raise_to"], state, sem)
            hi = _sem_to_total(state["max_raise_to"], state, sem)
            if kind == "allin":
                t = hi
            else:
                t = int(dec[1])
                if t >= snap * hi:
                    t = hi
                if t < lo:
                    t = lo
                elif t > hi:
                    t = hi
            return {"action": names["raise"], "amount": int(_total_to_sem(t, state, sem))}
        if can_call:
            return {"action": names["call"]}
        return {"action": names["check"]} if can_check else {"action": names["fold"]}
    if kind == "call":
        if can_call:
            return {"action": names["call"]}
        return {"action": names["check"]} if can_check else {"action": names["fold"]}
    if can_check:
        return {"action": names["check"]}
    return {"action": names["fold"]}


def is_legal(action, state):
    """The mock kit's validate_action rule (sim/kit/agent.py), re-implemented: the action is listed in
    legal_actions; a raise amount is an int (or an integral float, not a bool) within [min_raise_to, max_raise_to]."""
    if type(action) is not dict:
        return False
    kind = action.get("action")
    legal = state["legal_actions"]
    if kind not in legal:
        return False
    if kind != "raise":
        return True
    amt = action.get("amount")
    if isinstance(amt, bool):
        return False
    if isinstance(amt, float):
        if not amt.is_integer():
            return False
        amt = int(amt)
    if not isinstance(amt, int):
        return False
    return state["min_raise_to"] <= amt <= state["max_raise_to"]
