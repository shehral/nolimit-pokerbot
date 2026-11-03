"""Blind schedules of the mock starter kit (the real schedule is announced at the event; the rules: blinds start at
10/20 and increase every 25 hands).

A schedule is a tuple of levels; each level is (sb, bb, ante). Level k (0-based) covers hands
k*hands_per_level + 1 .. (k+1)*hands_per_level; after the last level the blinds stay at the last level.
With 25 hands per level and the 200-hand cap, 8 levels cover a whole table.
"""

HANDS_PER_LEVEL = 25

# SPEC section 1, verbatim levels.
TURBO = ((10, 20, 0), (15, 30, 0), (25, 50, 0), (50, 100, 0), (75, 150, 0), (100, 200, 0), (150, 300, 0),
         (200, 400, 0))
# Doubling every level: 10/20 -> 1280/2560.
FAST = ((10, 20, 0), (20, 40, 0), (40, 80, 0), (80, 160, 0), (160, 320, 0), (320, 640, 0), (640, 1280, 0),
        (1280, 2560, 0))
# About 1.25x per level (big blind 20 -> 100 over 7 steps: geometric mean 1.258x).
SLOW = ((10, 20, 0), (12, 25, 0), (15, 30, 0), (20, 40, 0), (25, 50, 0), (30, 60, 0), (40, 80, 0), (50, 100, 0))

SCHEDULES = {"TURBO": TURBO, "FAST": FAST, "SLOW": SLOW}


def get_schedule(spec):
    """Name ('TURBO', 'FAST', 'SLOW', case-insensitive) or a sequence of (sb, bb) / (sb, bb, ante) levels ->
    a tuple of (sb, bb, ante) int triples. Raises ValueError on a malformed schedule."""
    if isinstance(spec, str):
        key = spec.upper()
        if key not in SCHEDULES:
            raise ValueError(f"unknown schedule {spec!r}; known: {sorted(SCHEDULES)}")
        return SCHEDULES[key]
    levels = []
    for lv in spec:
        lv = tuple(lv)
        if len(lv) == 2:
            lv = (lv[0], lv[1], 0)
        if len(lv) != 3 or not all(isinstance(x, int) and not isinstance(x, bool) for x in lv):
            raise ValueError(f"bad level {lv!r}: need (sb, bb) or (sb, bb, ante) ints")
        sb, bb, ante = lv
        if not (0 < sb <= bb) or ante < 0:
            raise ValueError(f"bad level {lv!r}: need 0 < sb <= bb and ante >= 0")
        levels.append(lv)
    if not levels:
        raise ValueError("empty schedule")
    return tuple(levels)


def schedule_name(spec):
    """A short label for reports."""
    if isinstance(spec, str):
        return spec.upper()
    for name, sch in SCHEDULES.items():
        if get_schedule(spec) == sch:
            return name
    return "CUSTOM"


def level_index(hand_no, n_levels, hands_per_level=HANDS_PER_LEVEL):
    """0-based level for hand_no (1-based), clamped to the last level."""
    lv = (hand_no - 1) // hands_per_level
    return lv if lv < n_levels else n_levels - 1


def blinds_for_hand(schedule, hand_no, hands_per_level=HANDS_PER_LEVEL):
    """(level_index, sb, bb, ante) for a 1-based hand number."""
    sch = get_schedule(schedule)
    lv = level_index(hand_no, len(sch), hands_per_level)
    sb, bb, ante = sch[lv]
    return lv, sb, bb, ante
