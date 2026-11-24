"""Field mixes and entrant lists built from the archetype zoo.

Archetype keys (one file each in this directory):
  chart_tag  pushfold_maniac  station  nit  lag_aggro  random_legal  dead  buggy  icm_pro

MIXES are seat shares (sum to 100). They are priors: guesses about what a field of bots written in an hour, mostly
with AI help, would look like at the start (T1_like) and the end (T6_like) of an event, plus a worst case (STRONG).
"Broken" = buggy + dead + random_legal.

  T1_like  broken 30% (dead 20, buggy 6, random 4); chart_tag 29, icm_pro 12, station 15, pushfold 6, nit 5, lag 3.
  T6_like  broken 6% (dead 3, buggy 2, random 1); chart_tag 30, icm_pro 28, station 15, pushfold 9, nit 5, lag 7.
  STRONG   worst case: icm_pro 50, chart_tag 35, nit 5, pushfold 4, lag 3, station 2, buggy 1.

The private version of this zoo also had three slow Monte Carlo archetypes modelled on the organizer's public
exhibit bots. They are not published here. Their seats went to the closest public archetype: the one that timed out
on almost every decision (so it auto-folded) to `dead`, the two loose over-callers to `station`.
These are priors, not measurements; read every score as "against this model of the field".

Helpers:
  entrants_from_mix(mix, n)       n distinct-named entrants allocated by largest remainder (for runner seating="event"
                                  with n a multiple of 8, or "sample" with a bigger n)
  sample_opponents(mix, rng, k)   k archetype names drawn i.i.d. from the shares (one table's opponents)
  jittered_variants(n, seed, mix) about n distinct entrants for a leaderboard: archetypes allocated by `mix`,
                                  parameters jittered inside each class's JITTER ranges
"""
import functools
import importlib
import random

ARCHETYPES = {          # key -> (module, class)
    "chart_tag": ("chart_tag", "ChartTag"),
    "pushfold_maniac": ("pushfold_maniac", "PushFoldManiac"),
    "station": ("station", "Station"),
    "nit": ("nit", "Nit"),
    "lag_aggro": ("lag_aggro", "LagAggro"),
    "random_legal": ("random_legal", "RandomLegal"),
    "dead": ("dead", "Dead"),
    "buggy": ("buggy", "Buggy"),
    "icm_pro": ("icm_pro", "IcmPro"),
}
BROKEN = ("buggy", "dead", "random_legal")

T1_like = {"dead": 20, "buggy": 6, "random_legal": 4,
           "chart_tag": 29, "icm_pro": 12, "station": 15, "pushfold_maniac": 6, "nit": 5, "lag_aggro": 3}
T6_like = {"dead": 3, "buggy": 2, "random_legal": 1,
           "chart_tag": 30, "icm_pro": 28, "station": 15, "pushfold_maniac": 9, "nit": 5, "lag_aggro": 7}
STRONG = {"icm_pro": 50, "chart_tag": 35, "nit": 5, "pushfold_maniac": 4, "lag_aggro": 3, "station": 2, "buggy": 1}
# Midpoint of T1_like and T6_like, used for the 20-entrant simulated leaderboard so every archetype can appear.
LEADERBOARD = {k: (T1_like.get(k, 0) + T6_like.get(k, 0)) / 2.0 for k in set(T1_like) | set(T6_like)}
MIXES = {"T1_like": T1_like, "T6_like": T6_like, "STRONG": STRONG}
for _name, _m in MIXES.items():
    assert sum(_m.values()) == 100 and set(_m) <= set(ARCHETYPES), _name


def broken_share(mix):
    return sum(mix.get(k, 0) for k in BROKEN) / sum(mix.values())


def agent_class(key):
    mod, cls = ARCHETYPES[key]
    return getattr(importlib.import_module(f"sim.zoo.{mod}"), cls)


def factory(key, **params):
    """A picklable bot factory (seat, config) -> agent for archetype `key` with parameter overrides."""
    cls = agent_class(key)
    return functools.partial(cls, **params) if params else cls


def allocate(mix, n):
    """Largest-remainder allocation of n slots to the archetypes of `mix` -> {key: count} (deterministic)."""
    total = float(sum(mix.values()))
    raw = {k: v * n / total for k, v in mix.items()}
    out = {k: int(x) for k, x in raw.items()}
    left = n - sum(out.values())
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], mix[k]), reverse=True)[:left]:
        out[k] += 1
    return {k: c for k, c in out.items() if c > 0}


def entrants_from_mix(mix, n=40):
    """n entrants (name, factory) allocated by share, named like 'chart_tag_03' (default parameters)."""
    out = []
    for key, cnt in allocate(mix, n).items():
        out.extend((f"{key}_{i + 1:02d}", factory(key)) for i in range(cnt))
    return out


def sample_opponents(mix, rng, k=7):
    keys = list(mix)
    return rng.choices(keys, [mix[x] for x in keys], k=k)


def jitter_params(key, rng):
    """One jittered parameter set for archetype `key`: uniform within the class's JITTER ranges (ints stay ints)."""
    cls = agent_class(key)
    out = {}
    for name, (lo, hi) in cls.JITTER.items():
        default = cls.PARAMS[name]
        v = rng.uniform(lo, hi)
        out[name] = int(round(v)) if isinstance(default, int) and not isinstance(default, bool) else round(v, 4)
    return out


def jittered_variants(n=20, seed=0, mix=None):
    """About n distinct entrants for a leaderboard: archetypes by `mix` (default LEADERBOARD, the T1/T6 midpoint), jittered parameters.
    Returns [(name, factory, spec)], spec = {'archetype': key, 'params': {...}} for the report."""
    rng = random.Random(seed)
    mix = mix or LEADERBOARD
    out = []
    for key, cnt in allocate(mix, n).items():
        for i in range(cnt):
            params = jitter_params(key, rng)
            out.append((f"{key}_{i + 1:02d}", factory(key, **params), {"archetype": key, "params": params}))
    return out
