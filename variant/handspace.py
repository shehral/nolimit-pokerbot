"""HandSpace: the hand classes of a VariantSpec (variants_deep notes s6.2, R5). Pure standard library.

Replaces the bot's NLHE-only constants (169 labels, combos 6/4/12, the 1,326 denominator, label_of, the 2,704-entry
CLASS_OF) with accessors computed from the spec:

  n_classes, labels[i], combos[i], total_combos      class i holds combos[i] dealt hole-card sets
  class_of(hole) -> i, label_of(hole)               any card order; never raises (class_of_safe)
  members(i)                                        dealt hands (sorted tuples) of class i
  percentiles(order)                                midpoint percentile of each class in a best-first order
  set_strength(eq)                                  order + percentiles by equity vs random (from build_tables)
  sample_deal(rng, n_players), evaluate(hole, board), choose(hole, board)

Classes are suit-isomorphism classes of the dealt hand (2, 3 or 4 cards; NLHE gives exactly the bot's 169 in the bot's
EQ_LABELS order). For "keep k of n" variants a class is "best keepable subset by static score" + discard information:
  discard_info="full"     the full isomorphism class, labelled kept|discards (3 dealt keep 2: 1,755 classes)
  discard_info="summary"  kept class + (discards pairing a kept rank, discards sharing a kept suit, top discard rank)
  discard_info="none"     kept class only
  "auto" = full when the number of dealt hands is at most ENUM_MAX, else summary.
Labels never contain spaces (bot_tables-style space-separated strings stay valid).
"""
import random
from itertools import combinations, permutations
from math import comb

from .spec import static_score

ENUM_MAX = 400_000        # enumerate every dealt hand up to this many (Omaha 270,725 is fine; 5 of 52 is not)
SAMPLE_N = 200_000        # dealt hands sampled to estimate class sizes when the universe is larger


class HandSpace:
    def __init__(self, spec, discard_info="auto", enum_max=ENUM_MAX, sample_n=SAMPLE_N, seed=11):
        self.spec = spec
        cd = spec.codec
        self.slot = cd.slot
        self.rank_chars = spec.rank_chars
        self.n = spec.hole_dealt
        self.k = spec.n_kept
        self.deck = tuple(spec.deck)
        self.N = len(self.deck)
        self.universe_size = comb(self.N, self.n)
        self.total_combos = self.universe_size
        self.choice = self.k < self.n
        if discard_info == "auto":
            discard_info = "full" if self.universe_size <= enum_max else "summary"
        if not self.choice:
            discard_info = "full"
        self.discard_info = discard_info
        self._score = spec.keep_score or (lambda cs: static_score(cs, self.slot))
        # suit permutations that map the deck onto itself (all of them for a suit-symmetric deck)
        suits = sorted(set(c % self.slot for c in self.deck))
        dset = set(self.deck)
        perms = []
        for p in permutations(suits):
            m = list(range(self.slot))
            for a, b in zip(suits, p):
                m[a] = b
            if all((c - c % self.slot + m[c % self.slot]) in dset for c in self.deck):
                perms.append(m)
        self._perms = perms
        self.enumerated = self.universe_size <= enum_max
        keys = {}
        counts = {}
        self._by_sorted = {}
        if self.enumerated:
            self.universe = [tuple(sorted(h)) for h in combinations(self.deck, self.n)]
            ucls_key = []
            for h in self.universe:
                key = self._key(h)
                ucls_key.append(key)
                counts[key] = counts.get(key, 0) + 1
            scale = 1.0
        else:
            self.universe = None
            rng = random.Random(seed)
            ucls_key = None
            for _ in range(sample_n):
                key = self._key(tuple(sorted(rng.sample(self.deck, self.n))))
                counts[key] = counts.get(key, 0) + 1
            scale = self.universe_size / float(sample_n)
        for key in counts:
            keys[key] = self._label_and_sort(key)
        order = sorted(counts, key=lambda kk: keys[kk][1])
        self.keys = order
        self.key_index = {kk: i for i, kk in enumerate(order)}
        self.labels = [keys[kk][0] for kk in order]
        self.label_index = {lab: i for i, lab in enumerate(self.labels)}
        if len(self.label_index) != len(self.labels):
            raise RuntimeError("HandSpace labels are not unique")
        self.combos = [counts[kk] * scale for kk in order]
        if self.enumerated:
            self.combos = [int(round(c)) for c in self.combos]
            self.univ_class = [self.key_index[kk] for kk in ucls_key]
            self._by_sorted = dict(zip(self.universe, self.univ_class))
        else:
            self.univ_class = None
        self.n_classes = len(self.labels)
        self._pair_table = None
        if self.n == 2 and self.enumerated:
            M = max(self.deck) + 1
            t = [-1] * (M * M)
            for (a, b), ci in self._by_sorted.items():
                t[a * M + b] = ci
                t[b * M + a] = ci
            self._pair_table, self._pair_M = t, M
        self.order = None
        self.pct = None
        self.strength = None

    # ------------------------------------------------------------------ canonical forms
    def _canon(self, hand):
        """Canonical representative of a dealt hand under suit permutations: max over permutations of the
        descending-sorted mapped card tuple."""
        slot = self.slot
        best = None
        for m in self._perms:
            t = tuple(sorted((c - c % slot + m[c % slot] for c in hand), reverse=True))
            if best is None or t > best:
                best = t
        return best

    def split(self, hand):
        """(kept, discarded) of a hand by the static keep rule (no board), deterministic on the given order."""
        if not self.choice:
            return tuple(hand), ()
        best, bk = None, None
        for sub in combinations(hand, self.k):
            s = self._score(sub)
            if bk is None or s > bk:
                best, bk = sub, s
        disc = tuple(c for c in hand if c not in best)
        return tuple(best), disc

    def _key(self, hand):
        rep = self._canon(hand)
        if not self.choice or self.discard_info == "full":
            return rep
        kept, disc = self.split(rep)
        kept_rep = self._canon(kept)
        if self.discard_info == "none":
            return (kept_rep,)
        slot = self.slot
        kr = set(c // slot for c in kept)
        ks = set(c % slot for c in kept)
        pair_b = sum(1 for c in disc if c // slot in kr)
        suit_b = sum(1 for c in disc if c % slot in ks)
        top = max(c // slot for c in disc)
        return (kept_rep, (pair_b, suit_b, top))

    def _letters(self, cards, letters):
        out = []
        for c in cards:
            s = c % self.slot
            if s not in letters:
                letters[s] = "abcdefgh"[len(letters)]
            out.append(letters[s])
        return out

    def _label_and_sort(self, key):
        """(label, sort key) of a class key. NLHE-style labels for 2 kept cards: AA, AKs, AKo."""
        rc = self.rank_chars
        slot = self.slot
        if self.choice and self.discard_info != "full":
            kept = key[0]
            disc_part = key[1] if len(key) > 1 else None
            disc = ()
        else:
            kept, disc = self.split(key)
            disc_part = None
        kept = tuple(sorted(kept, reverse=True))
        disc = tuple(sorted(disc, reverse=True))
        letters = {}
        kl = self._letters(kept, letters)
        kr = [c // slot for c in kept]
        if len(kept) == 2:
            if kr[0] == kr[1]:
                klab = rc[kr[0]] * 2
            else:
                klab = rc[kr[0]] + rc[kr[1]] + ("s" if kl[0] == kl[1] else "o")
        else:
            klab = "".join(rc[r] for r in kr) + "-" + "".join(kl)
        kpat = "".join(kl)
        if disc:
            dl = self._letters(disc, letters)
            dr = [c // slot for c in disc]
            lab = klab + "|" + "".join(rc[r] + l for r, l in zip(dr, dl))
            dsort = (tuple(-r for r in dr), "".join(dl))
        elif disc_part is not None:
            pb, sb, top = disc_part
            lab = f"{klab}|p{pb}s{sb}h{rc[top]}"
            dsort = ((-top,), f"{pb}{sb}")
        else:
            lab = klab
            dsort = ((), "")
        return lab, (tuple(-r for r in kr), kpat, dsort)

    # ------------------------------------------------------------------ lookups
    def class_of(self, hole):
        """Class index of a dealt hand (card ints, any order). Raises KeyError only for impossible hands."""
        if self._pair_table is not None:
            return self._pair_table[hole[0] * self._pair_M + hole[1]]
        ci = self._by_sorted.get(tuple(sorted(hole)))
        if ci is not None:
            return ci
        return self.key_index[self._key(tuple(sorted(hole)))]

    def class_of_dealt(self, hole):
        """Class of a hand whose size differs from the dealt size n (raises if it cannot place it):
        more cards than n (e.g. 3 dealt cards against 2-card classes): the best n-subset by the keep score;
        fewer (e.g. the 2 cards kept after a toss against 3-dealt classes): the class of the dealt hand completed
        with the lowest deck card for which the static keep rule keeps exactly these cards (keep|lowest discard).
        Verify 2026-11-03: the post-toss 2-card hole used to fall to class 0, the best class."""
        hole = tuple(hole)
        m = len(hole)
        if m == self.n:
            return self.class_of(hole)
        if m > self.n:
            best, bk = None, None
            for sub in combinations(hole, self.n):
                s = self._score(sub)
                if bk is None or s > bk:
                    best, bk = sub, s
            return self.class_of(best)
        if not self.choice or m != self.k:
            raise KeyError(f"cannot place a {m}-card hand in {self.n}-card classes")
        hs = set(hole)
        fill = sorted((c for c in self.deck if c not in hs), key=lambda c: (c // self.slot, c % self.slot))
        need = self.n - m
        for extra in combinations(fill[:12 + need], need):
            hand = tuple(sorted(hole + extra))
            kept, _ = self.split(hand)
            if set(kept) == hs:
                return self.class_of(hand)
        # no completion keeps these cards (a hand the keep rule never keeps, such as 72o next to any third card):
        # the class whose kept cards score closest to this hand's static score, the weakest such class on ties
        if getattr(self, "_kept_scores", None) is None:
            self._kept_scores = [self._score(self.kept_of(i)) for i in range(self.n_classes)]
        target = self._score(hole)
        rank = self.pct if self.pct is not None else None
        return min(range(self.n_classes),
                   key=lambda i: (abs(self._kept_scores[i] - target), -(rank[i] if rank else i)))

    def class_of_safe(self, hole, default=None):
        """class_of that never raises: a hand of another size goes through class_of_dealt; anything still unknown
        maps to `default` (else the median class by strength, else the median by static score)."""
        try:
            ci = self.class_of(hole) if len(hole) == self.n else self.class_of_dealt(hole)
            if ci is not None and ci >= 0:
                return ci
        except Exception:  # noqa: BLE001
            pass
        try:
            ci = self.class_of_dealt(hole)
            if ci is not None and ci >= 0:
                return ci
        except Exception:  # noqa: BLE001
            pass
        if default is not None:
            return default
        if self.order:
            return self.order[len(self.order) // 2]
        # no strength installed yet: median class by static score (was class 0 = the BEST class in every
        # HandSpace, so an unknown hand, e.g. the 2-card hole after a toss, played as the nuts; verify 2026-11-03)
        fb = getattr(self, "_fallback_cls", None)
        if fb is None:
            try:
                from .spec import static_score
                sc = [static_score(self.kept_of(i), self.slot) for i in range(self.n_classes)]
                fb = sorted(range(self.n_classes), key=lambda i: (-sc[i], i))[self.n_classes // 2]
            except Exception:  # noqa: BLE001
                fb = self.n_classes // 2
            self._fallback_cls = fb
        return fb

    def label_of(self, hole):
        return self.labels[self.class_of(hole)]

    def members(self, i):
        if self.universe is None:
            raise RuntimeError("members() needs an enumerated universe")
        return [h for h, c in zip(self.universe, self.univ_class) if c == i]

    def kept_of(self, i):
        """Kept cards (a representative) of class i, for labels and keep defaults."""
        return self.split(self.keys[i])[0] if (not self.choice or self.discard_info == "full") else self.keys[i][0]

    # ------------------------------------------------------------------ strength and percentiles
    def percentiles(self, order):
        """Midpoint percentile of each class (list indexed by class) for a best-first order given as class indices
        or labels: (combos of better classes + half its own) / total combos. Generalizes brain.percentiles()."""
        idx = [self.label_index[o] if isinstance(o, str) else o for o in order]
        out = [0.0] * self.n_classes
        acc = 0.0
        tot = float(self.total_combos)
        for i in idx:
            c = self.combos[i]
            out[i] = (acc + c / 2.0) / tot
            acc += c
        return out

    def set_strength(self, eq):
        """Install an equity-vs-random vector (indexed by class): order = best first, pct = midpoint percentile."""
        self.strength = list(eq)
        self.order = sorted(range(self.n_classes), key=lambda i: (-eq[i], i))
        self.pct = self.percentiles(self.order)
        return self.order

    def top_range(self, order, J):
        """Weights in [0, 1] per class of the top-J fraction of combos in `order` (partial class at the boundary),
        the same construction as sng_icm fit_eq_vs_range.py and the bot's eq_vs()."""
        idx = [self.label_index[o] if isinstance(o, str) else o for o in order]
        r = [0.0] * self.n_classes
        tot = float(self.total_combos)
        cum = 0.0
        for i in idx:
            c = self.combos[i] / tot
            lo = cum
            cum += c
            if cum <= J + 1e-12:
                r[i] = 1.0
            elif lo < J:
                r[i] = (J - lo) / c
            else:
                break
        return r

    # ------------------------------------------------------------------ dealing and showdown helpers
    def sample_deal(self, rng, n_players):
        """(list of dealt hands, board) for n_players from a fresh deck."""
        need = n_players * self.n + self.spec.board_cards
        cards = rng.sample(self.deck, need)
        hands = [tuple(cards[i * self.n:(i + 1) * self.n]) for i in range(n_players)]
        return hands, tuple(cards[n_players * self.n:])

    def evaluate(self, hole, board):
        return self.spec.hand_value(hole, board)

    def choose(self, hole, board_so_far=()):
        return self.spec.keep(hole, board_so_far)

    def describe(self):
        return (f"{self.spec.name}: {self.n_classes} classes, {self.total_combos} dealt hands "
                f"({'enumerated' if self.enumerated else 'sampled'}), dealt {self.n} keep {self.k}, "
                f"discard_info {self.discard_info}, {len(self._perms)} suit permutations")
