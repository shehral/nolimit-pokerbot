"""Small statistics helpers for runner results (standard library only).

All intervals are normal-approximation 95% intervals: mean +- 1.96 * sd / sqrt(n). Per-table points are bounded
and n is usually in the thousands, so the approximation is fine; for n < 30 treat the interval as rough.
"""
import math

Z95 = 1.959963984540054


def mean_sd(xs):
    """(n, mean, sample sd). sd is 0.0 when n < 2."""
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return n, m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return n, m, math.sqrt(v)


def ci95(xs):
    """dict(n, mean, sd, se, lo, hi) for the mean of xs."""
    n, m, sd = mean_sd(xs)
    se = sd / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": m, "sd": sd, "se": se, "lo": m - Z95 * se, "hi": m + Z95 * se}


def relative(xs, avg_seat):
    """Relative score = mean points / average-seat points, with its 95% interval (1.00 = an average seat)."""
    c = ci95(xs)
    if avg_seat <= 0:
        return {"rel": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": c["n"]}
    return {"rel": c["mean"] / avg_seat, "lo": c["lo"] / avg_seat, "hi": c["hi"] / avg_seat, "n": c["n"]}


def paired(xs, ys):
    """Paired comparison of two equally long, aligned lists (same tables): stats of d = x - y, the correlation,
    and the standard error an unpaired comparison of the same data would have (to show the pairing gain)."""
    if len(xs) != len(ys):
        raise ValueError(f"paired lists differ in length: {len(xs)} vs {len(ys)}")
    d = [a - b for a, b in zip(xs, ys)]
    c = ci95(d)
    nx, mx, sx = mean_sd(xs)
    ny, my, sy = mean_sd(ys)
    n = len(d)
    if n >= 2 and sx > 0 and sy > 0:
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (n - 1)
        corr = cov / (sx * sy)
    else:
        corr = 0.0
    unpaired_se = math.sqrt(sx * sx / nx + sy * sy / ny) if n else 0.0
    return {"n": n, "mean_diff": c["mean"], "sd_diff": c["sd"], "se": c["se"], "lo": c["lo"], "hi": c["hi"],
            "corr": corr, "unpaired_se": unpaired_se,
            "variance_ratio": (unpaired_se / c["se"]) ** 2 if c["se"] > 0 else float("inf")}
