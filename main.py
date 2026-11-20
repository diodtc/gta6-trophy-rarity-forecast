"""GTA 6 trophy rarity forecast for make.fun bounty gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at a case's target time, as five quantiles per trophy.

What the data really is
-----------------------
Exophase prints each trophy's earn rate as 100 * earners / players_tracked,
rounded to two decimals. players_tracked only grows, and the page is a stale
snapshot between Exophase refreshes: reads sit identical for hours, then jump
at a refresh, when players jumps up a lot and earners jump some. Net rates
usually FALL at a refresh (dilution dominates) except for very early trophies
whose earners grow faster than players. So a rate is a ratio of two moving
quantities, and the two move on different clocks: earners creep up steadily
as players progress, players jumps in steps at refreshes.

Model
-----
Time is measured in hours since the game's release (2026-11-19T00:00Z) and
compressed with x = ln(1 + hours), so early launch hours stretch out and the
long tail to the target compresses.

1. History reads are deduplicated into distinct snapshots: identical
   (players_tracked, rates) pairs are one snapshot, kept at its first read
   time. A flat launch night is one snapshot, not dozens of votes for flat.
2. Earners are recovered per snapshot as E = round(rate * N / 100). A rate of
   0.00 means E in [0, 0.005*N/100]; we use 0. A rate of 100 means E = N.
3. Per trophy, an ensemble (median in logit space) of three median models:
   (e) logit(rate) quadratic in x over all distinct snapshots, weighted
       least squares with recency weights, quadratic coefficient clamped
       to <= 0: rates saturate, so the logit curve bends down as a trophy
       approaches its ceiling, and a straight line extrapolated 4.5 days
       out sails past it. Falls back to the linear fit when degenerate.
   (b) logit(rate) linear in x on the last 24h of history only (falls
       back to the full-history linear fit when fewer than 3 distinct
       snapshots sit in that window);
   (lvl) the recent level: mean logit of the last five distinct
       snapshots, the anchor that stops the two trend members from
       running away together.
   The median of three makes the ensemble robust: one wild member cannot
   move it past the other two. The full-history linear fit (a) is kept
   only as the residual-variance source and as the fallback above, and
   the earners/players decomposition (d) was dropped: on synthetic
   final-like cases both extrapolated past ceilings far worse than the
   members above (e.g. a trophy truly at 25% forecast at 51-71%).
4. Flat-history fallback: with fewer than 3 distinct snapshots (a practice-
   like input) there is no trend to fit, so earners are taken as frozen and
   players are assumed to jump 3x by the target (a refresh is assumed before
   it): median rate = 100 * E_last / (3 * N_last). The 100% trophy dilutes
   like the rest here -- at the launch-night refresh it did.
5. Spreads are normal on the logit scale around the ensemble median:
   sigma combines the (a)-fit residual variance, a term growing with the
   gap from the last read to the target, a floor for refresh surprise, and
   the spread between ensemble members. Quantiles use z = [-1.645, -0.674,
   0, 0.674, 1.645], rounded half-up to 4 decimals, clamped to [0.01, 99.99]
   and forced non-decreasing.

Special cases: a trophy at 100% in the last read of a rich history keeps
q = [99.5, 99.9, 100, 100, 100] (it has survived every refresh so far at
100%). A trophy with zero earners in every snapshot keeps its median at
0.01% but gets a wide spread (sigma 1.5, so q95 ~ 0.12%): by the target,
endgame trophies sitting at zero at the checkpoint can pick up their
first earners, and a narrow band would pay heavily for every one that
does.

The program is stdlib only, makes no network calls, is deterministic (no
randomness, no wall-clock dependence -- only case data), and reads
$MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR, else /input and /output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release moment: the origin of the compressed time axis.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Rate clamps before taking a logit (the page prints 0.00..100.00, and
# 0% / 100% have no logit).
RATE_LO = 0.01
RATE_HI = 99.99

# Flat-history fallback: players multiplier assumed by the target.
FLAT_PLAYERS_MULT = 3.0
FLAT_SIGMA = 0.9

# Spread floors / caps (logit units).
RESID_VAR_FLOOR = 0.04
ENSEMBLE_VAR_FLOOR = 0.01
SPREAD_FLOOR = 0.25
SIGMA_MIN = 0.15
SIGMA_MAX = 1.6
# Fixed spread for a trophy with zero earners in every snapshot: a wide
# upper tail, so a first earner by the target does not blow up the loss.
ZERO_SIGMA = 1.5


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def time_axis(moment):
    """Compressed time x = ln(1 + hours since release)."""
    return math.log(max(1.0 + hours_since_release(moment), 0.01))


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, rate))


def logit_of_rate(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(value):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if float(rounded).is_integer() else rounded


def weighted_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x.

    Returns (intercept, slope, weighted mean of x, weighted mean of y).
    With a single point, or zero x-variance, the slope is 0.
    """
    total_w = sum(ws)
    if total_w <= 0:
        return 0.0, 0.0, 0.0, 0.0
    x_bar = sum(w * x for x, w in zip(xs, ws)) / total_w
    y_bar = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return y_bar, 0.0, x_bar, y_bar
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    return y_bar - slope * x_bar, slope, x_bar, y_bar


def recency_weights(count):
    """Weights rising linearly from 0.5 (oldest) to 1.0 (newest).

    Recent snapshots count more, but no single snapshot dominates.
    """
    if count <= 1:
        return [1.0]
    return [0.5 + 0.5 * (k / (count - 1)) for k in range(count)]


def distinct_snapshots(history):
    """Collapse identical reads into one snapshot each, keeping first time.

    A snapshot is keyed by (players_tracked, rates tuple). Returns a list
    of (moment, players_tracked, rates) in time order.
    """
    seen = {}
    order = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key not in seen:
            seen[key] = (parse_time(read["at"]), read["players_tracked"],
                         list(read["rates"]))
            order.append(key)
    return [seen[k] for k in order]


def recover_earners(rate, players):
    """Earners implied by a displayed rate: E = round(rate * N / 100)."""
    if rate is None:
        return None
    if rate >= 100.0:
        return players
    if rate <= 0.0:
        return 0
    return int(round(rate * players / 100.0))


def quantiles_from(median_rate, sigma):
    """Five quantiles around a median rate, normal on the logit scale."""
    centre = logit_of_rate(median_rate)
    out = []
    for z in Z_SCORES:
        q = rate_of_logit(centre + z * sigma)
        out.append(round4(min(RATE_HI, max(RATE_LO, q))))
    # Enforce non-decreasing after rounding.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def solve_3x3(matrix, rhs):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting.

    Returns the solution list, or None when the system is singular.
    """
    rows = [row[:] + [b] for row, b in zip(matrix, rhs)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, 3):
            factor = rows[r][col] / rows[col][col]
            for cc in range(col, 4):
                rows[r][cc] -= factor * rows[col][cc]
    out = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        rest = sum(rows[r][cc] * out[cc] for cc in range(r + 1, 3))
        out[r] = (rows[r][3] - rest) / rows[r][r]
    return out


def quad_member(xs, ys, ws, x_target):
    """Concave quadratic fit of logit(rate) on centred time.

    Fits y = a + b*u + c*u^2 with u = x - x_bar, weighted least squares,
    with the quadratic coefficient clamped to <= 0: earn rates saturate,
    so logit(rate) against log-time bends down, and a straight line
    extrapolated 4.5 days out sails past the ceiling. When the fit is
    degenerate or there are too few snapshots, returns None (caller
    falls back to the linear fit).
    """
    if len(xs) < 6:
        return None
    total_w = sum(ws)
    x_bar = sum(w * x for x, w in zip(xs, ws)) / total_w
    feats = [[1.0, x - x_bar, (x - x_bar) ** 2] for x in xs]
    matrix = [[sum(w * f[i] * f[j] for f, w in zip(feats, ws))
               for j in range(3)] for i in range(3)]
    rhs = [sum(w * f[i] * y for f, y, w in zip(feats, ys, ws))
           for i in range(3)]
    coef = solve_3x3(matrix, rhs)
    if coef is None:
        return None
    a0, b0, c0 = coef
    if c0 > 0:
        # No concavity in the data: refit as a straight line in u.
        matrix2 = [[sum(w * f[i] * f[j] for f, w in zip(feats, ws))
                    for j in range(2)] for i in range(2)]
        rhs2 = [sum(w * f[i] * y for f, y, w in zip(feats, ys, ws))
                for i in range(2)]
        det = matrix2[0][0] * matrix2[1][1] - matrix2[0][1] * matrix2[1][0]
        if abs(det) < 1e-12:
            return None
        a0 = (rhs2[0] * matrix2[1][1] - matrix2[0][1] * rhs2[1]) / det
        b0 = (matrix2[0][0] * rhs2[1] - rhs2[0] * matrix2[1][0]) / det
        c0 = 0.0
    u_target = x_target - x_bar
    return a0 + b0 * u_target + c0 * u_target * u_target


def forecast_trophy(points, x_target):
    """Forecast one trophy.

    points: list of (x, rate, earners, players) over distinct snapshots,
    in time order, rate not None. Returns the five quantiles.
    """
    xs = [p[0] for p in points]
    logits = [logit_of_rate(p[1]) for p in points]
    earners = [p[2] for p in points]
    players = [p[3] for p in points]
    n = len(points)
    x_last = xs[-1]

    # --- Flat history: no trend can be fitted. -------------------------
    if n < 3:
        e_last = earners[-1]
        n_last = players[-1]
        # The 100% trophy dilutes like the rest at a refresh, so no
        # special case here: E_last / (3 * N_last) for every trophy.
        median = 100.0 * e_last / (FLAT_PLAYERS_MULT * n_last)
        median = max(median, RATE_LO)
        return quantiles_from(median, FLAT_SIGMA)

    weights = recency_weights(n)

    # --- Member (a): logit-linear in x, all snapshots. ----------------
    # Kept as the residual-variance source and as the fallback for the
    # other members, not as an ensemble member itself: a straight line
    # fitted over the whole history extrapolates past a trophy's ceiling
    # (see the synthetic validation notes in the README).
    a_int, a_slope, _, _ = weighted_line(xs, logits, weights)
    logit_a = a_int + a_slope * x_target
    resid_var = sum(
        w * (y - (a_int + a_slope * x)) ** 2
        for x, y, w in zip(xs, logits, weights)
    ) / sum(weights)
    resid_var = max(resid_var, RESID_VAR_FLOOR)

    members = []

    # --- Member (e): concave quadratic in x, all snapshots. -----------
    # Captures the bend as a rate approaches its ceiling.
    quad = quad_member(xs, logits, weights, x_target)
    members.append(quad if quad is not None else logit_a)

    # --- Member (b): logit-linear, last 24h of history only. ----------
    # x is ln(1+hours), so a 24h window is not a fixed x gap: convert
    # back to hours and keep snapshots within 24h of the last read.
    recent = []
    hours_last = math.exp(x_last) - 1.0
    for (x, y) in zip(xs, logits):
        if math.exp(x) - 1.0 >= hours_last - 24.0:
            recent.append((x, y))
    if len(recent) >= 3:
        r_xs = [r[0] for r in recent]
        r_ys = [r[1] for r in recent]
        r_int, r_slope, _, _ = weighted_line(
            r_xs, r_ys, recency_weights(len(recent)))
        members.append(r_int + r_slope * x_target)
    else:
        members.append(logit_a)

    # --- Member (lvl): the recent level. -------------------------------
    # Mean logit of the last five distinct snapshots: the anchor that
    # stops the two trend members from running away together.
    last_few = logits[-5:]
    members.append(sum(last_few) / len(last_few))

    # --- Ensemble: median of member logits. ---------------------------
    ordered = sorted(members)
    m = len(ordered)
    median_logit = (ordered[m // 2] if m % 2 == 1
                    else 0.5 * (ordered[m // 2 - 1] + ordered[m // 2]))
    mean_member = sum(members) / m
    ensemble_var = max(
        sum((v - mean_member) ** 2 for v in members) / m,
        ENSEMBLE_VAR_FLOOR)
    median_rate = rate_of_logit(median_logit)

    # --- Spread. -------------------------------------------------------
    gap = max(x_target - x_last, 0.0)
    sigma = math.sqrt(resid_var + (0.40 * gap) ** 2
                      + SPREAD_FLOOR ** 2 + ensemble_var)
    sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))

    # A trophy nobody has earned yet: keep the median at the floor, but
    # give it a wide upper tail -- by the target, endgame trophies that
    # sit at zero at the checkpoint can pick up their first earners, and
    # a narrow band would pay heavily for every one that does.
    if all(e == 0 for e in earners):
        median_rate = max(median_rate, RATE_LO)
        sigma = ZERO_SIGMA

    return quantiles_from(median_rate, sigma)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError("%s asks for levels %s, not %s"
                         % (name, case["levels"], LEVELS))
    if not case["history"]:
        # No history at all: emit a valid, maximally uninformative row.
        return {"trophies": [
            {"id": t["id"], "q": [0.01, 0.01, 0.01, 0.02, 0.05]}
            for t in sorted(case["trophies"], key=lambda t: t["id"])]}

    target = parse_time(case["target"])
    x_target = time_axis(target)
    snaps = distinct_snapshots(case["history"])
    last_rates = snaps[-1][2]

    rows = []
    for pos, trophy in enumerate(case["trophies"]):
        points = []
        for moment, players, rates in snaps:
            rate = rates[pos]
            if rate is None:
                continue
            points.append((time_axis(moment), rate,
                           recover_earners(rate, players), players))
        last_rate = last_rates[pos]

        # Rich history, trophy still at 100% in the last read: it has
        # survived every refresh so far at 100%.
        if len(points) >= 3 and last_rate is not None and last_rate >= 100.0:
            rows.append({"id": trophy["id"],
                         "q": [99.5, 99.9, 100, 100, 100]})
            continue

        if not points:
            rows.append({"id": trophy["id"],
                         "q": [0.01, 0.01, 0.01, 0.02, 0.05]})
            continue

        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(points, x_target)})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
