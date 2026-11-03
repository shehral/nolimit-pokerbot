"""Adapter: the ONLY file that knows the starter kit's interface (mock kit in sim/kit; a real platform's kit would replace it).

normalize(state) -> View    kit state dict -> the brain's normalized view (card ints, positions, totals)
to_action(decision, state)  internal decision -> a legal kit action (clamped to the legal bounds)
is_legal(action, state)     the kit's legality rule, re-implemented here (the bot must not import the kit)
safe_action(state)          check if legal, else fold (the rules' own fallback)

Internal decisions (from brain.py): ("fold",) ("check",) ("call",) ("raise", raise_to_total) ("allin",).
Kit actions: {"action": "fold" | "check" | "call" | "raise", "amount": raise_to_total}; all-in = raise to
max_raise_to, or call when the stack does not cover the call.

Special (non-betting) requests (variant games: discard, keep, toss, bid, assign, redraw, other):
request_of(state)               None for a betting decision, else the kit's request name
request_view(state, req)        kit state -> the variant layer's request view (variant/defaults.py docstring)
answer_request(state, req, ...) view -> variant.defaults.respond (optional brain policy first) -> kit answer
is_legal_request(out, state)    the kit's legality rule for request answers, re-implemented (format by format)
safe_request(state)             a legal answer by construction that needs no variant code (the request fallback)
The kit's request shapes are ADAPTER_PARAMS["REQUESTS"]; the mock (sim/variants) sends request "discard" with
discard_options and wants {"action": "discard", "index": k}. The other entries are rehearsal guesses: PLATFORM, set
each one's state fields and answer key from the real engine.

PLATFORM: when the real interface differs, change only ADAPTER_PARAMS and the two functions marked PLATFORM below
(field names, card format, raise semantics, action names). The brain and the shell do not change.
ADAPTER_PARAMS holds interface facts only; tunables (such as the all-in snap) live in brain.PARAMS.
"""

ADAPTER_PARAMS = {
    "RAISE_SEMANTICS": "to",     # "to" (kit: amount = street total), "by" (amount = increase over the current bet),
                                 # "add" (amount = chips added now); bounds are read in the same semantics
    "ACTION_NAMES": {"fold": "fold", "check": "check", "call": "call", "raise": "raise"},
    "STREETS": {"preflop": 0, "flop": 1, "turn": 2, "river": 3},
    # ---- variant games (PLATFORM: map from the real engine). state[REQUEST_FIELD] in BET_REQUESTS = a betting decision
    "REQUEST_FIELD": "request",
    "BET_REQUESTS": (None, "act", "bet", "betting", "action"),
    # kit request name -> kind (variant layer), response format (layer), k (cards given up for discard/toss, kept for
    # keep, max for redraw), action (answer's action name; None = the request name), field (answer key holding the
    # value), options (state field listing the legal values, if the kit sends one), k_field (state field with k),
    # min_field/max_field (bids), groups_field (assign), optional (may be declined)
    "REQUESTS": {
        "discard": {"kind": "toss", "response": "index", "k": 1, "action": "discard", "field": "index",
                    "options": "discard_options"},
        "toss": {"kind": "toss", "response": "index", "k": 1, "action": "toss", "field": "index",
                 "options": "toss_options"},
        "keep": {"kind": "keep", "response": "keep_indices", "k": 2, "k_field": "keep_count", "action": "keep",
                 "field": "indices", "options": "keep_options"},
        "redraw": {"kind": "redraw", "response": "discard_indices", "k": 1, "k_field": "max_redraw",
                   "optional": True, "action": "redraw", "field": "indices", "options": "redraw_options"},
        "bid": {"kind": "bid", "response": "int", "action": "bid", "field": "amount", "options": "bid_options",
                "min_field": "min_bid", "max_field": "max_bid"},
        "assign": {"kind": "assign", "response": "groups", "action": "assign", "field": "groups",
                   "options": "assign_options", "groups_field": "groups"},
        "other": {"kind": "other", "response": "option", "action": None, "field": "option", "options": "options"},
    },
    # bounty games: state field with our own bounty rank, and the payoff (MIT 2025: 1.5 x normal + 10)
    "BOUNTY_FIELD": "bounty_rank",
    "BOUNTY": {"mult": 0.5, "flat": 10},
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
                 "level", "boards", "bounty")
    # hole: ALL dealt cards (2 in hold'em, 3 before a toss, 2 after it, 4-5 in other games); boards: None with one
    # board, else every board (board = boards[0]); bounty: None, or (mult, flat) when our bounty rank is in our hole
    # cards or on the board (a pot we win then pays normal x (1 + mult) + flat)


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
    if len(h) == 2:
        v.hole = (parse_card(h[0]), parse_card(h[1]))
    else:
        v.hole = tuple(parse_card(c) for c in h)
    v.board = [parse_card(c) for c in (state.get("board") or ())]
    bds = state.get("boards")
    v.boards = [[parse_card(c) for c in b] for b in bds] if bds is not None and len(bds) > 1 else None
    br = state.get(ADAPTER_PARAMS["BOUNTY_FIELD"])
    v.bounty = None
    if br is not None:
        rk = br if isinstance(br, int) and not isinstance(br, bool) else _RANK.get(str(br).strip())
        if rk is not None and (any(c >> 2 == rk for c in v.hole) or any(c >> 2 == rk for c in v.board)):
            b = ADAPTER_PARAMS["BOUNTY"]
            v.bounty = (float(b["mult"]), float(b["flat"]))
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


# ============================================================================ special requests (variant games)
_KINDS = ("discard", "keep", "toss", "assign", "bid", "redraw", "other")
_DEFAULT_FORMAT = {"discard": "index", "toss": "index", "keep": "keep_indices", "redraw": "discard_indices",
                   "bid": "int", "assign": "groups", "other": "option"}
_DEFAULT_FIELD = {"index": "index", "keep_indices": "indices", "discard_indices": "indices", "keep_cards": "cards",
                  "discard_cards": "cards", "card": "card", "int": "amount", "index_or_none": "index",
                  "groups": "groups", "option": "option"}


def request_of(state):
    """None for a betting decision (the plain kit has no request field), else the kit's request name."""
    r = state.get(ADAPTER_PARAMS["REQUEST_FIELD"])
    if r in ADAPTER_PARAMS["BET_REQUESTS"]:
        return None
    return r


def request_cfg(req):
    """The ADAPTER_PARAMS["REQUESTS"] entry of a request (an unknown name gets a guessed entry by its kind)."""
    cfg = ADAPTER_PARAMS["REQUESTS"].get(req)
    if cfg is not None:
        return cfg
    kind = req if req in _KINDS else "other"
    fmt = _DEFAULT_FORMAT[kind]
    return {"kind": kind, "response": fmt, "k": 1 if kind in ("discard", "toss", "redraw") else None,
            "action": None, "field": _DEFAULT_FIELD[fmt], "options": f"{req}_options",
            "optional": kind == "redraw"}


def _options(state, cfg):
    f = cfg.get("options")
    o = state.get(f) if f else None
    if o is None and f != "options":
        o = state.get("options")
    return list(o) if o is not None else None


def _req_k(state, cfg):
    kf = cfg.get("k_field")
    if kf and state.get(kf) is not None:
        return state[kf]
    return cfg.get("k")


def _stack(state):
    st = state.get("stacks")
    try:
        return st[state["seat"]] if st is not None else None
    except Exception:  # noqa: BLE001
        return None


def request_view(state, req, cfg=None):
    """Kit state -> the variant layer's request view (card ints in this file's numbering = the layer's)."""
    cfg = cfg or request_cfg(req)
    v = {"kind": cfg["kind"], "response": cfg["response"], "optional": bool(cfg.get("optional", False)),
         "hole": [parse_card(c) for c in (state.get("hole") or ())],
         "board": [parse_card(c) for c in (state.get("board") or ())],
         "pot": state.get("pot", 0) or 0, "stack": _stack(state), "options": _options(state, cfg),
         "street": state.get("street"), "request": req}
    k = _req_k(state, cfg)
    if k is not None:
        v["k"] = k
    if cfg.get("min_field"):
        v["min"] = state.get(cfg["min_field"], 0) or 0
    if cfg.get("max_field") and state.get(cfg["max_field"]) is not None:
        v["max"] = state[cfg["max_field"]]
    if cfg.get("groups_field"):
        v["groups"] = list(state.get(cfg["groups_field"]) or ())
    return v


def _answer(state, req, cfg, value):
    name = cfg.get("action") or req
    legal = state.get("legal_actions")
    if legal and name not in legal and len(legal) == 1:
        name = legal[0]
    return {"action": name, cfg.get("field") or _DEFAULT_FIELD.get(cfg["response"], "value"): value}


def answer_request(state, req, layer=None, policy=None, budget_us=200):
    """Answer a special request: the variant layer's respond(view, policy) (policy = an optional cheap rule, kept only
    when legal and inside budget_us; respond never raises), wrapped in the kit's answer dict. Without the layer: the
    policy, else safe_request. The shell checks the result with is_legal_request and falls back to safe_request."""
    cfg = request_cfg(req)
    view = request_view(state, req, cfg)
    if layer is not None:
        r = layer.respond(view, policy=policy, budget_us=budget_us)
    elif policy is not None:
        r = policy(view)
    else:
        return safe_request(state, req)
    if cfg["response"] in ("keep_cards", "discard_cards", "card"):     # layer card ints -> the kit's own cards
        back = dict(zip(view["hole"], state.get("hole") or ()))
        r = back.get(r, r) if cfg["response"] == "card" else [back.get(c, c) for c in (r or ())]
    return _answer(state, req, cfg, r)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _n_kept(n, kind, k):
    if kind == "keep":
        return k if _is_int(k) else n
    if kind in ("discard", "toss", "redraw"):
        return n - (k if _is_int(k) else 1)
    return n


def _norm(x):
    if isinstance(x, (list, tuple)):
        return tuple(_norm(g) for g in x)
    return x


def is_legal_request(out, state, req=None):
    """The kit's legality rule for a request answer, re-implemented per response format (the mock's discard rule is
    sim/variants/engine.validate_discard: a dict, action "discard", an integer index in range). With an options list
    in the state, the answer must be one of the options."""
    try:
        if type(out) is not dict:
            return False
        req = req if req is not None else request_of(state)
        cfg = request_cfg(req)
        name = cfg.get("action") or req
        legal = state.get("legal_actions")
        if legal and name not in legal and len(legal) == 1:
            name = legal[0]
        if out.get("action") != name:
            return False
        field = cfg.get("field") or _DEFAULT_FIELD.get(cfg["response"], "value")
        if field not in out:
            return False
        val = out[field]
        opts = _options(state, cfg)
        fmt = cfg["response"]
        hole = list(state.get("hole") or ())
        n = len(hole)
        if opts is not None:
            if fmt in ("index", "int") and not _is_int(val):
                return False
            return _norm(val) in set(_norm(o) for o in opts)
        k = _req_k(state, cfg)
        optional = bool(cfg.get("optional", False))
        if fmt in ("index", "index_or_none"):
            if val is None:
                return fmt == "index_or_none" and optional
            return _is_int(val) and 0 <= val < n
        if fmt in ("keep_indices", "discard_indices", "keep_cards", "discard_cards"):
            if not isinstance(val, (list, tuple)) or len(set(val)) != len(val):
                return False
            if fmt.endswith("indices"):
                if not all(_is_int(i) and 0 <= i < n for i in val):
                    return False
            elif not all(c in hole for c in val):
                return False
            keep = _n_kept(n, cfg["kind"], k)
            if cfg["kind"] == "redraw":           # redraw up to k cards (none = decline)
                return len(val) <= n - keep if fmt.startswith("discard") else len(val) >= keep
            want = keep if fmt.startswith("keep") else n - keep
            decline = n if fmt.startswith("keep") else 0
            return len(val) == want or (optional and len(val) == decline)
        if fmt == "card":
            return val in hole
        if fmt == "int":
            if not _is_int(val):
                return False
            lo = state.get(cfg.get("min_field") or "", 0) or 0
            hi = state.get(cfg.get("max_field") or "")
            st = _stack(state)
            return val >= lo and (hi is None or val <= hi) and (st is None or lo > st or val <= st)
        if fmt == "groups":
            sizes = list(state.get(cfg.get("groups_field") or "groups") or ())
            if not isinstance(val, (list, tuple)) or len(val) != len(sizes):
                return False
            flat = []
            for g, s in zip(val, sizes):
                if not isinstance(g, (list, tuple)) or len(g) != s:
                    return False
                flat.extend(g)
            return all(_is_int(i) and 0 <= i < n for i in flat) and len(set(flat)) == len(flat)
        return False                      # "option" without an options list: nothing is checkably legal
    except Exception:  # noqa: BLE001
        return False


def safe_request(state, req=None):
    """A legal request answer by construction, with no variant code: toss/discard the lowest card (the engine's own
    "lowest" default), keep the highest cards, decline an optional redraw, bid the minimum, identity assignment,
    else the first listed option."""
    req = req if req is not None else request_of(state)
    cfg = request_cfg(req)
    fmt = cfg["response"]
    hole = list(state.get("hole") or ())
    n = len(hole)
    opts = _options(state, cfg)
    ranks = []
    for i, c in enumerate(hole):
        try:
            ranks.append((parse_card(c) >> 2, i))
        except Exception:  # noqa: BLE001
            ranks.append((-1, i))
    ranks.sort()
    keep = _n_kept(n, cfg["kind"], _req_k(state, cfg))
    if fmt == "int":
        if opts:
            st = _stack(state)
            under = [o for o in opts if _is_int(o) and (st is None or o <= st)]
            val = min(under or opts)
        else:
            val = state.get(cfg.get("min_field") or "", 0) or 0
    elif opts:
        if fmt in ("index", "index_or_none"):
            low = [i for _, i in ranks if i in opts]
            val = low[0] if low else opts[0]
        else:
            val = opts[0]
    elif fmt in ("index", "index_or_none"):
        val = None if (fmt == "index_or_none" and cfg.get("optional")) else (ranks[0][1] if ranks else 0)
    elif fmt in ("keep_indices", "discard_indices", "keep_cards", "discard_cards"):
        if cfg["kind"] == "redraw" and cfg.get("optional", False):
            kept = list(range(n))
        else:
            kept = sorted(i for _, i in ranks[n - keep:]) if keep > 0 else []
        idx = kept if fmt.startswith("keep") else [i for i in range(n) if i not in kept]
        val = idx if fmt.endswith("indices") else [hole[i] for i in idx]
    elif fmt == "card":
        val = hole[ranks[0][1]] if ranks else None
    elif fmt == "groups":
        val, i = [], 0
        for s in list(state.get(cfg.get("groups_field") or "groups") or ()):
            val.append(list(range(i, i + s)))
            i += s
    else:
        val = None
    return _answer(state, req, cfg, val)
