"""VTable: the kit's Table with a variant layer (subclass; sim/kit is imported, never edited).

Everything not listed here is inherited from sim.kit.engine.Table: __init__ (stacks, payouts, schedule, seeds,
button), play() (the hand loop and the table end), _finish() (ICM points), _emit() (observe events), strictness
bookkeeping (SeatStats). VTable replaces _play_hand() with a copy of the kit's hand loop that adds:

  state["variant"]       variant name ("plain", "toss", "shortdeck", "bounty", "doubleboard", "antes", "bbante")
  state["request"]       "act" (the normal betting decision) or "discard" (toss: pick one hole card to toss)
  state["boards"]        list of boards (each a list of card strings); state["board"] is boards[0]
  state["bounty_rank"]   your own hidden bounty rank character (bounty variant), else None
  state["discards"]      public list of (seat, card) tossed so far this hand (toss), else []
  state["hole_dealt"]    number of hole cards dealt per seat at the start of the hand

Discard request (toss): legal_actions == ["discard"], state["discard_options"] == [0, 1, 2] (indexes into
state["hole"]); answer {"action": "discard", "index": k}. Strictness is the kit's: an exception, an illegal answer or
a decision over the limit is counted and the engine plays the default (variant.discard_error: "fold" = fold, the
strict reading; "lowest" = toss your lowest-ranked card). Nothing is repaired.

The deck is dealt by position like the kit: seat i gets deck[k*i : k*i+k] (k = hole cards), the board(s) follow,
and the deck of hand h depends only on (table seed, h), so PAIRED comparisons still line up.
"""
import random
import time
from fractions import Fraction

try:
    from sim.kit.agent import validate_action
    from sim.kit.cards import CARD_STR, RANKS
    from sim.kit.engine import EngineError, Table, STREETS, _FC, _FCR, _MAX_EXAMPLES, _X, _XR, _short, _trunc
except ImportError:   # pragma: no cover
    from agent import validate_action
    from cards import CARD_STR, RANKS
    from engine import EngineError, Table, STREETS, _FC, _FCR, _MAX_EXAMPLES, _X, _XR, _short, _trunc

from .spec import Variant, make_variant

_DISCARD = ("discard",)


def validate_discard(out, n_cards):
    """Exact check for a discard answer. Returns (index, None) when legal, else (None, reason)."""
    if type(out) is not dict:
        return None, f"discard answer is {type(out).__name__}, not a dict"
    if out.get("action") != "discard":
        return None, f"action {out.get('action')!r} is not 'discard' (request was discard)"
    idx = out.get("index")
    if isinstance(idx, bool) or not isinstance(idx, int):
        if isinstance(idx, float) and idx.is_integer():
            idx = int(idx)
        else:
            return None, f"discard index {idx!r} is not an integer"
    if not 0 <= idx < n_cards:
        return None, f"discard index {idx} outside 0..{n_cards - 1}"
    return idx, None


def settle_bounty(layers, hits, button, room, mult=0.5, flat=10, order=None):
    """Bounty payouts, pure function (the engine calls it after the normal pot award).

    layers: list of (slice, winners, contrib) one per side-pot layer: slice = chips in the layer, winners = seats
            sharing it, contrib = {seat: chips that seat put into the layer}.
    hits:   set of seats whose hidden bounty rank is in their hole cards or on the board.
    button: dealer seat (a fractional bonus is rounded up for every other seat, down for the dealer, as the MIT 2025
            rules round up out of position and down in position).
    room:   {seat: chips the seat can still pay} (their stack after the normal award); a transfer never exceeds it.
    Rule (MIT 2025, generalised): each winner w of a layer is treated as winning an equal sub-pot slice/len(winners).
    Normal winnings N_w = sum over layers of (slice - contrib_w) / len(winners): what the others put in. A winner
    that hit gets bonus = mult * N_w + flat (so total winnings = (1 + mult) * N_w + flat; 1.5 N + 10 with the
    defaults), taken from the other contributors of those layers in proportion to what they put in, never more than
    their room. Returns {winner: {payer: chips}} (only winners that hit and have N_w > 0).
    """
    n_w = {}
    weights = {}
    for slice_, winners, contrib in layers:
        m = len(winners)
        for w in winners:
            n_w[w] = n_w.get(w, 0) + Fraction(slice_ - contrib.get(w, 0), m)
            wt = weights.setdefault(w, {})
            for j, cj in contrib.items():
                if j != w and cj > 0:
                    wt[j] = wt.get(j, 0) + Fraction(cj, m)
    room = dict(room)
    out = {}
    mult = Fraction(mult).limit_denominator(1000)
    for w in (order if order is not None else sorted(n_w)):
        if w not in hits or w not in n_w or n_w[w] <= 0:
            continue
        raw = n_w[w] * mult + flat
        amt = -((-raw.numerator) // raw.denominator) if w != button else raw.numerator // raw.denominator
        wt = weights[w]
        paid = {}
        remaining = amt
        while remaining > 0:
            cand = [j for j in wt if room.get(j, 0) > 0]
            if not cand:
                break
            tw = sum(wt[j] for j in cand)
            moved = 0
            for j in cand:
                share = int(remaining * wt[j] / tw)
                a = min(share, room[j])
                if a > 0:
                    paid[j] = paid.get(j, 0) + a
                    room[j] -= a
                    moved += a
            if moved == 0:   # remainders smaller than one chip each: one chip from the heaviest payer with room
                j = max(cand, key=lambda x: wt[x])
                paid[j] = paid.get(j, 0) + 1
                room[j] -= 1
                moved = 1
            remaining -= moved
        if paid:
            out[w] = paid
    return out


class VTable(Table):
    """A kit Table running a Variant. VTable(variant, factories, payouts, schedule, seed, **table_kwargs)."""

    def __init__(self, variant, factories, payouts="TOP3", schedule="TURBO", seed=0, **kw):
        if isinstance(variant, str):
            variant = make_variant(variant)
        self.variant = variant
        self.seed_tag = seed
        super().__init__(factories, payouts, schedule, seed, **kw)
        self.schedule = variant.apply_schedule(self.schedule)
        self.bounty_ranks = [None] * self.n
        self._bounty_key = None

    # ------------------------------------------------------------------ deck / config
    def _random_deck(self, hand_no):
        need = self.variant.deck_cards_needed(self.n)
        d = list(self.variant.deck)
        m = len(d)
        rnd = self.deck_rng.random
        for k in range(need):
            j = k + int(rnd() * (m - k))
            d[k], d[j] = d[j], d[k]
        return d[:need]

    def _config(self, seat):
        cfg = super()._config(seat)
        v = self.variant
        cfg["variant"] = v.name
        cfg["variant_options"] = v.public_options()
        cfg["hole_dealt"] = v.hole_dealt
        cfg["n_boards"] = v.n_boards
        cfg["deck"] = [CARD_STR[c] for c in v.deck]
        cfg["ante_mode"] = v.ante_mode
        return cfg

    def _roll_bounties(self, hand_no, level_no):
        v = self.variant.bounty
        key = hand_no if v["mode"] == "hand" else level_no
        if key == self._bounty_key:
            return
        self._bounty_key = key
        rng = random.Random(f"bounty:{self.seed_tag}:{v['mode']}:{key}")
        self.bounty_ranks = [rng.randrange(13) for _ in range(self.n)]

    # ------------------------------------------------------------------ one agent decision
    def _ask(self, i, state, legal, min_to, mx, st, hand_no, street_name, discard_n=0):
        """Run agent i on `state` with the kit's timing/strictness bookkeeping.
        Returns (kind, amount_or_index); kind None means an error was counted (caller plays the default)."""
        pc = time.perf_counter_ns
        tc = time.thread_time_ns
        limit_ns = self.limit_ns
        scale = self.time_scale
        tlim = self.time_limit_ms
        st.decisions += 1
        actf = self.acts[i]
        if actf is None:
            st.exceptions += 1
            st.add_example("exception", "agent failed to construct; every decision errors")
            return None, 0
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
                st.add_example("exception", f"{type(exc).__name__}: {_trunc(exc)} (hand {hand_no} {street_name})")
            return None, 0
        if (c_over if self.timing == "cpu" else w_over):
            st.timeouts += 1
            if len(st.examples.get("timeout", ())) < _MAX_EXAMPLES:
                st.add_example("timeout", f"act() took {el / 1e6:.3f} ms wall, {cel / 1e6:.3f} ms cpu x scale "
                                          f"{scale:g} > {tlim} ms (hand {hand_no} {street_name})")
            return None, 0
        try:
            if discard_n:
                kind, amt, reason = "discard", None, None
                idx, reason = validate_discard(out, discard_n)
                if reason is None:
                    amt = idx
                else:
                    kind = None
            else:
                k = out.get("action") if type(out) is dict else None
                if k is not None and k != "raise" and k in legal:
                    kind, amt = k, 0
                else:
                    kind, amt, reason = validate_action(out, legal, min_to, mx)
        except Exception as e:   # a hostile return value must not crash the engine
            kind, amt, reason = None, 0, f"validation raised {type(e).__name__}"
        if kind is None:
            st.illegal += 1
            if len(st.examples.get("illegal", ())) < _MAX_EXAMPLES:
                st.add_example("illegal", f"{reason}; returned {_short(out)} (hand {hand_no} {street_name})")
            return None, 0
        return kind, amt

    # ------------------------------------------------------------------ hand
    def _play_hand(self, hand_no):
        V = self.variant
        n = self.n
        stacks = self.stacks
        stats = self.stats
        pays = self.pays
        pays_list = self.pays_list
        emitting = bool(self.observers) or self.events is not None
        emit = self._emit
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
        k_hole = V.hole_dealt
        nb = V.n_boards
        toss = V.toss
        bounty_cfg = V.bounty
        if bounty_cfg is not None:
            self._roll_bounties(hand_no, level_no)
        bounty_ranks = self.bounty_ranks

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
            h = tuple(deck[k_hole * i: k_hole * i + k_hole])
            hole[i] = h
            hole_s[i] = tuple(CARD_STR[c] for c in h)
        base = k_hole * n
        L = V.board_len
        bds = [deck[base + bi * L: base + (bi + 1) * L] for bi in range(nb)]
        boards = [[] for _ in range(nb)]
        boards_s = [[] for _ in range(nb)]
        discards = []

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
            payers = order if V.ante_mode == "all" else [bb_seat]
            for i in payers:
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
                  "alive": tuple(alive_list), "posts": tuple(history), "variant": V.name})
            for i in order:
                emit({"type": "deal", "hand_no": hand_no, "seat": i, "hole": hole_s[i]}, only=i)
                if bounty_cfg is not None:
                    emit({"type": "bounty", "hand_no": hand_no, "seat": i, "rank": RANKS[bounty_ranks[i]]}, only=i)
        vpip = [False] * n
        pfr = [False] * n
        n_in = n_alive
        cur = bb
        min_inc = bb
        for street in range(4):
            if street:
                for i in order:
                    bets[i] = 0
                cur = 0
                min_inc = bb
                new = V.street_new[street - 1]
                off = sum(V.street_new[:street - 1])
                for bi in range(nb):
                    cs = bds[bi][off: off + new]
                    boards[bi].extend(cs)
                    boards_s[bi].extend(CARD_STR[c] for c in cs)
                street_starts.append(len(history))
                if emitting:
                    ev = {"type": "street", "hand_no": hand_no, "street": STREETS[street],
                          "board": tuple(boards_s[0]), "pot": pot}
                    if nb > 1:
                        ev["boards"] = tuple(tuple(x) for x in boards_s)
                    emit(ev)
                if toss and street == 1:
                    # discard round: every contender (all-in players too), left of the button first
                    for i in order:
                        if folded[i]:
                            continue
                        h = hole_s[i]
                        st = stats[i]
                        state = {
                            "hand_no": hand_no, "level": level_no, "hands_to_next_level": hands_to_next,
                            "hands_left": hands_left, "sb": sb, "bb": bb, "ante": ante, "button": b,
                            "sb_seat": sb_seat, "bb_seat": bb_seat, "seat": i, "street": "flop",
                            "hole": list(h), "board": boards_s[0][:], "pot": pot, "stacks": stacks[:],
                            "bets": bets[:], "contributed": contrib[:], "active": active[:],
                            "all_in": allin[:], "to_call": 0, "min_raise_to": 0, "max_raise_to": 0,
                            "legal_actions": list(_DISCARD), "history_this_hand": history[:],
                            "street_starts": street_starts[:], "n_alive": n_alive, "alive": alive_list[:],
                            "start_stacks": start_stacks[:], "payouts": pays_list[:], "n_seats": n,
                            "time_limit_ms": tlim, "variant": V.name, "request": "discard",
                            "boards": [boards_s[0][:]], "bounty_rank": None, "discards": list(discards),
                            "hole_dealt": k_hole, "discard_options": list(range(len(h))),
                        }
                        kind, idx = self._ask(i, state, _DISCARD, 0, 0, st, hand_no, "flop", discard_n=len(h))
                        if kind is None:
                            if V.discard_error == "fold" and n_in > 1:   # the last contender cannot fold
                                folded[i] = True
                                active[i] = False
                                n_in -= 1
                                st.n_fold += 1
                                history.append((i, "fold", 0))
                                if emitting:
                                    emit({"type": "action", "hand_no": hand_no, "street": "flop", "seat": i,
                                          "action": "fold", "amount": 0, "to_call": 0, "pot": pot,
                                          "stack": stacks[i], "all_in": bool(allin[i])})
                                continue
                            idx = min(range(len(hole[i])), key=lambda x: hole[i][x] >> 2)
                        card = hole[i][idx]
                        hole[i] = hole[i][:idx] + hole[i][idx + 1:]
                        hole_s[i] = hole_s[i][:idx] + hole_s[i][idx + 1:]
                        boards[0].append(card)
                        boards_s[0].append(CARD_STR[card])
                        discards.append((i, CARD_STR[card]))
                        if emitting:
                            emit({"type": "discard", "hand_no": hand_no, "seat": i, "card": CARD_STR[card],
                                  "board": tuple(boards_s[0])})
                    if n_in == 1:
                        break
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
            can_act = 0
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
                others = can_act > 1
                if to_call <= 0 and not others:
                    acted[i] = True
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
                state = {
                    "hand_no": hand_no, "level": level_no, "hands_to_next_level": hands_to_next,
                    "hands_left": hands_left, "sb": sb, "bb": bb, "ante": ante, "button": b,
                    "sb_seat": sb_seat, "bb_seat": bb_seat, "seat": i, "street": street_name,
                    "hole": list(hole_s[i]), "board": boards_s[0][:], "pot": pot, "stacks": stacks[:],
                    "bets": bets[:], "contributed": contrib[:], "active": active[:],
                    "all_in": allin[:], "to_call": to_call, "min_raise_to": min_to, "max_raise_to": mx,
                    "legal_actions": list(legal), "history_this_hand": history[:],
                    "street_starts": street_starts[:], "n_alive": n_alive, "alive": alive_list[:],
                    "start_stacks": start_stacks[:], "payouts": pays_list[:], "n_seats": n,
                    "time_limit_ms": tlim, "variant": V.name, "request": "act",
                    "boards": [x[:] for x in boards_s],
                    "bounty_rank": RANKS[bounty_ranks[i]] if bounty_cfg is not None else None,
                    "discards": list(discards), "hole_dealt": k_hole,
                }
                kind, amt = self._ask(i, state, legal, min_to, mx, st, hand_no, street_name)
                if kind is None:              # rules s3.7: fold, or check when checking is legal
                    kind = "check" if to_call <= 0 else "fold"
                    amt = 0
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
            # uncalled bet back to its owner
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

        # award
        won = [0] * n
        shown = ()
        layers = []
        if n_in == 1:
            for i in order:
                if not folded[i]:
                    won[i] = pot
                    if bounty_cfg is not None:
                        layers.append((pot, [i], {j: contrib[j] for j in order if contrib[j] > 0}))
                    break
        else:
            contenders = [i for i in order if not folded[i]]
            ev_f = V.evaluate
            scores = []
            for bi in range(nb):
                bt = tuple(boards[bi])
                sc = [0] * n
                for i in contenders:
                    sc[i] = ev_f(hole[i] + bt)
                scores.append(sc)
            # Dead money (a big-blind ante that is not matched, a folded player's chips above every live
            # contribution) joins the top live layer instead of forming a layer nobody can win.
            top_live = max(contrib[i] for i in contenders)
            levels = sorted(set(c for c in contrib if 0 < c <= top_live))
            prev = 0
            for lvl in levels:
                cap = lvl if lvl < top_live else 1 << 60
                slice_ = 0
                for j in range(n):
                    cj = contrib[j]
                    if cj > prev:
                        slice_ += (cj if cj < cap else cap) - prev
                elig = [i for i in contenders if contrib[i] >= lvl]
                if not elig:
                    raise EngineError(f"pot layer {prev}-{lvl} has no eligible player (hand {hand_no})")
                bq, br = divmod(slice_, nb)
                for bi in range(nb):
                    half = bq + (1 if bi < br else 0)
                    sc = scores[bi]
                    best = max(sc[i] for i in elig)
                    winners = [i for i in elig if sc[i] == best]
                    share, rem = divmod(half, len(winners))
                    for t, i in enumerate(winners):
                        won[i] += share + (1 if t < rem else 0)
                    if bounty_cfg is not None:
                        layers.append((half, winners, {j: (min(contrib[j], cap) - prev) for j in range(n)
                                                       if contrib[j] > prev}))
                prev = lvl
            if emitting:
                shown = tuple((i, hole_s[i]) for i in contenders)
                ev = {"type": "showdown", "hand_no": hand_no, "board": tuple(boards_s[0]), "shown": shown}
                if nb > 1:
                    ev["boards"] = tuple(tuple(x) for x in boards_s)
                emit(ev)
        for i in range(n):
            if won[i]:
                stacks[i] += won[i]
        if sum(won) != pot:
            raise EngineError(f"awarded {sum(won)} != pot {pot} (hand {hand_no})")
        bonus_paid = ()
        if bounty_cfg is not None and layers:
            hits = set()
            flat_board = set(c >> 2 for c in boards[0])
            for _s, ws, _c in layers:
                for w in ws:
                    r = bounty_ranks[w]
                    if r in flat_board or any((c >> 2) == r for c in hole[w]):
                        hits.add(w)
            if hits:
                room = {j: stacks[j] for j in range(n)}
                wins_order = [i for i in order if i in hits]
                tr = settle_bounty(layers, hits, b, room, bounty_cfg["mult"], bounty_cfg["flat"], wins_order)
                bp = []
                for w, paid in tr.items():
                    tot_w = 0
                    for j, a in paid.items():
                        stacks[j] -= a
                        stacks[w] += a
                        tot_w += a
                    bp.append((w, tot_w))
                bonus_paid = tuple(bp)
                if emitting and bonus_paid:
                    emit({"type": "bounty_bonus", "hand_no": hand_no, "bonuses": bonus_paid})
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
            ev = {"type": "hand_end", "hand_no": hand_no, "pot": pot,
                  "won": tuple((i, won[i]) for i in range(n) if won[i]), "stacks": tuple(stacks),
                  "board": tuple(boards_s[0])}
            if nb > 1:
                ev["boards"] = tuple(tuple(x) for x in boards_s)
            if bounty_cfg is not None:
                ev["bonuses"] = bonus_paid
            emit(ev)
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


def play_variant_table(variant, factories, payouts="TOP3", schedule="TURBO", seed=0, **kw):
    """Convenience: VTable(variant, factories, ...).play() -> kit TableResult."""
    return VTable(variant, factories, payouts, schedule, seed, **kw).play()


def make_variant_deck(variant, n_seats, holes=None, board=None, fill_seed=0):
    """Rigged deck for tests. holes = {seat: ("Ah", "Kd", ...)}, board = card strings for board 1 then board 2
    (concatenated, up to n_boards * board_len cards). Unspecified positions get the remaining deck cards shuffled."""
    from_str = {s: i for i, s in enumerate(CARD_STR)}
    if isinstance(variant, str):
        variant = make_variant(variant)
    k = variant.hole_dealt
    need = variant.deck_cards_needed(n_seats)
    deck = [None] * need
    used = set()
    for seat, cards in (holes or {}).items():
        for j, c in enumerate(cards):
            deck[k * seat + j] = from_str[c]
    for j, c in enumerate(board or []):
        deck[k * n_seats + j] = from_str[c]
    for c in deck:
        if c is not None:
            if c in used:
                raise ValueError("duplicate card in rigged deck")
            if c not in variant.deck:
                raise ValueError(f"card {CARD_STR[c]} not in this variant's deck")
            used.add(c)
    rest = [c for c in variant.deck if c not in used]
    random.Random(fill_seed).shuffle(rest)
    it = iter(rest)
    return [c if c is not None else next(it) for c in deck]
