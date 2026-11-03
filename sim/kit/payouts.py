"""Payout (points by finishing place) structures of the mock starter kit. The real one is announced at the start of
the build (rules s4). Every structure here sums to 100, so the average seat at an 8-seat table earns exactly 12.5
points, and relative score = points / 12.5.
"""

TOP3 = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LINEAR = tuple(x * 100.0 / 36.0 for x in (8, 7, 6, 5, 4, 3, 2, 1))   # [8..1] scaled to sum to 100
WTA = (100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TOP4 = (40.0, 30.0, 20.0, 10.0, 0.0, 0.0, 0.0, 0.0)

PAYOUTS = {"TOP3": TOP3, "LINEAR": LINEAR, "WTA": WTA, "TOP4": TOP4}


def get_payouts(spec, n_seats=8, truncate=False):
    """Name ('TOP3', 'LINEAR', 'WTA', 'TOP4', case-insensitive) or a sequence of points by place (1st first) ->
    a tuple of exactly n_seats floats (padded with zeros). Raises ValueError if a payout is negative, or if a
    non-zero payout falls beyond n_seats places (unless truncate=True: small test tables drop those places)."""
    if isinstance(spec, str):
        key = spec.upper()
        if key not in PAYOUTS:
            raise ValueError(f"unknown payouts {spec!r}; known: {sorted(PAYOUTS)}")
        pays = PAYOUTS[key]
    else:
        pays = tuple(float(x) for x in spec)
    if any(x < 0 for x in pays):
        raise ValueError(f"negative payout in {pays!r}")
    if len(pays) > n_seats:
        if not truncate and any(x != 0 for x in pays[n_seats:]):
            raise ValueError(f"payouts {pays!r} pay more places than the {n_seats} seats")
        pays = pays[:n_seats]
    return tuple(float(x) for x in pays) + (0.0,) * (n_seats - len(pays))


def payouts_name(spec):
    """A short label for reports."""
    if isinstance(spec, str):
        return spec.upper()
    pays = get_payouts(spec, max(8, len(tuple(spec))))
    for name, p in PAYOUTS.items():
        if all(abs(a - b) < 1e-9 for a, b in zip(get_payouts(p, len(pays)), pays)):
            return name
    return "CUSTOM"


def average_seat(payouts, n_seats=8):
    """Points the average seat earns: sum(payouts) / n_seats (12.5 for every built-in structure at 8 seats)."""
    return sum(get_payouts(payouts, n_seats)) / n_seats
