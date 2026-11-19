"""GTA VI trophy earn-rate forecaster for make.fun bounty gta6-trophy-rarity-forecast-01.

Reads the case's recorded Exophase trophy-page reads and writes, for each trophy,
five quantiles of its earn rate at the case's target.

Model
-----
A trophy's earn rate r = e / P * 100, where P is the players Exophase tracks and
e the players who have earned the trophy, so in logit space

    logit(r) = ln(e / (P - e))

exactly. The forecaster fits both counts on x = ln(1 + hours since launch) with
recency weights, damps each fitted slope toward its recent slope (a launch-week
cohort's growth decelerates, and deep-story trophies accelerate), extrapolates
both to the target, and blends the structural logit with a direct fit of
logit(rate) on the same x. The spread combines the fits' residuals, the slopes'
standard errors over the extrapolation distance, the players fit's uncertainty
(it moves every trophy together) and a horizon-scaled floor.

When the page looks stale inside the launch window (a midnight launch whose
tracked counts have not refreshed for hours), the pending refresh will dilute
every rate: the forecast shifts down and the band widens, more below than above.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

HALF_LIFE_H = 10.0        # recency half-life of an observation, hours
RECENT_WINDOW_H = 8.0     # span of the recent slope, hours
BLEND_RECENT = 0.8        # weight of the recent slope over the overall one
DAMP_SLOPE_MIN = 0.15     # only blend a slope estimated this well
MODEL_ERR = 0.45          # logit units of model error per unit of ln-time moved
SPREAD_BUMP = 1.05        # mild calibration headroom
STALE_TRIGGER_H = 2.0     # page staleness that counts as stale
STALE_WINDOW_H = 24.0     # only inside the launch window
STALE_SHIFT = -0.60       # logit shift for the pending refresh's dilution
STALE_EXTRA = [-0.55, -0.25, 0.0, 0.15, 0.45]


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def expit(l: float) -> float:
    if l >= 0:
        p = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return 100.0 * p


def round4(x: float):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_iso_ms(s: str):
    """Parse an ISO 8601 UTC timestamp to milliseconds since the epoch."""
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def wls(pts, half_life_h, ref_h):
    """Weighted least squares y = a + b x.

    pts: list of (x, y, h), h the observation's hours since launch; weight
    2 ** (-(ref_h - h) / half_life), floored at 1/64.
    Returns (a, b, resid_std, se_b), or None with too few effective points.
    """
    if len(pts) < 3:
        return None
    ws = []
    for _, _, h in pts:
        age = max(0.0, ref_h - h)
        ws.append(max(2.0 ** (-age / half_life_h), 1.0 / 64.0))
    wsum = sum(ws)
    mx = sum(w * p[0] for w, p in zip(ws, pts)) / wsum
    my = sum(w * p[1] for w, p in zip(ws, pts)) / wsum
    sxx = sum(w * (p[0] - mx) ** 2 for w, p in zip(ws, pts))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (p[0] - mx) * (p[1] - my) for w, p in zip(ws, pts))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (p[1] - a - b * p[0]) ** 2 for w, p in zip(ws, pts)) / wsum
    resid_std = math.sqrt(max(var, 0.0))
    se_b = math.sqrt(max(var, 1e-12) / sxx)
    return a, b, resid_std, se_b


def recent_slope(pts, half_life_h, ref_h, window_h=None):
    """Slope over the most recent `window_h` of the window (>= 3 points)."""
    if len(pts) < 6:
        return None
    h_last = max(p[2] for p in pts)
    h_first = min(p[2] for p in pts)
    window = max(window_h if window_h else RECENT_WINDOW_H,
                 0.15 * (h_last - h_first))
    recent = [p for p in pts if p[2] >= h_last - window]
    if len(recent) < 3:
        return None
    fit = wls(recent, half_life_h, ref_h)
    return None if fit is None else fit[1]


def damp(overall, recent):
    """Blend an overall slope with its recent slope for extrapolation: a
    launch-week cohort's growth decelerates, and deep-story trophies accelerate."""
    if overall is None:
        return 0.0
    if recent is None or abs(overall) < DAMP_SLOPE_MIN:
        return overall
    return BLEND_RECENT * recent + (1.0 - BLEND_RECENT) * overall


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    launch_ms = parse_iso_ms(case.get("game", {}).get("release_date", "")
                             + "T00:00:00Z")
    if launch_ms is None:
        launch_ms = parse_iso_ms(case["history"][0].get("slot")
                                 or case["history"][0].get("at")) or 0

    def hours(ts_ms):
        return (ts_ms - launch_ms) / 3600000.0

    times, players, rates = [], [], []
    for read in case["history"]:
        t = parse_iso_ms(read.get("slot") or read.get("at"))
        times.append(hours(t) if t is not None else 0.0)
        players.append(read.get("players_tracked"))
        rates.append(read.get("rates") or [])
    n_reads = len(case["history"])
    h_at = hours(parse_iso_ms(case["at"])) if parse_iso_ms(case["at"]) else 0.0
    h_target = hours(parse_iso_ms(case["target"])) if parse_iso_ms(case["target"]) \
        else times[-1]
    xs = [math.log(max(0.05, 1.0 + h)) for h in times]
    x_target = math.log(max(0.05, 1.0 + h_target))
    dx = x_target - xs[-1]

    # Staleness: hours since the page last changed (players or any rate).
    last_change = 0
    for j in range(1, n_reads):
        if players[j] != players[j - 1] or rates[j] != rates[j - 1]:
            last_change = j
    staleness_h = times[-1] - times[last_change]
    stale = 0.0 <= h_at <= STALE_WINDOW_H and staleness_h >= STALE_TRIGGER_H

    floor_s = 0.22 + 0.16 * math.sqrt(max(horizon_days, 0.0))

    p_pts = [(xs[j], math.log(players[j]), times[j])
             for j in range(n_reads) if players[j] and players[j] > 0]
    p_fit = wls(p_pts, HALF_LIFE_H, h_at)
    beta = damp(p_fit[1], recent_slope(p_pts, HALF_LIFE_H, h_at)) if p_fit else 0.0
    se_beta = p_fit[3] if p_fit else 0.0
    ln_p_last = math.log(players[-1]) if players[-1] else None
    # A tracked pool's growth decays with distance: adoption slows as it works
    # through the players it will ever track.
    beta_decay = 1.0 - min(0.5, 0.40 * abs(dx))
    ln_p_fc = ln_p_last + beta * dx * beta_decay if ln_p_last is not None else None
    rows = []
    for i, trophy in enumerate(trophies):
        series = [(times[j], rates[j][i]) for j in range(n_reads)
                  if i < len(rates[j]) and rates[j][i] is not None]
        if not series:
            rows.append({"id": trophy["id"], "q": [0, 0, 0, 0, 0]})
            continue
        last_r = series[-1][1]

        # A trophy every tracked player holds stays at 100 for good.
        if last_r >= 99.99 and all(r >= 99.99 for _, r in series):
            rows.append({"id": trophy["id"], "q": [99.98, 99.99, 100, 100, 100]})
            continue

        # A trophy nobody tracked has earned yet stays at zero this early.
        if last_r <= 0.005 and all(r <= 0.005 for _, r in series):
            rows.append({"id": trophy["id"], "q": [0, 0, 0, 0, 0.01]})
            continue

        # Direct fit of logit(rate) on x.
        r_pts = [(xs[j], logit(rates[j][i]), times[j]) for j in range(n_reads)
                 if i < len(rates[j]) and rates[j][i] is not None]
        r_fit = wls(r_pts, HALF_LIFE_H, h_at)
        b_r = damp(r_fit[1], recent_slope(r_pts, HALF_LIFE_H, h_at)) if r_fit else 0.0
        se_b_r = r_fit[3] if r_fit else 0.0
        resid_r = r_fit[2] if r_fit else 0.0
        # Extrapolate from the fit's level at the last read (the page's current
        # state, smoothed) plus the damped slope over the remaining distance,
        # decaying with distance: growth that held for a day rarely holds for a
        # week at the same pace.
        decay = 1.0 - min(0.45, 0.30 * abs(dx))
        l_direct = (r_fit[0] + r_fit[1] * xs[-1] + b_r * dx * decay) if r_fit \
            else logit(last_r)
        s_recent = recent_slope(r_pts, HALF_LIFE_H, h_at) if r_fit else None
        s_day = recent_slope(r_pts, HALF_LIFE_H, h_at, window_h=20.0) if r_fit \
            else None

        # Structural fit: e = rate/100 * P and logit(r) = ln(e / (P - e)).
        e_pts = [(xs[j], math.log(rates[j][i] * players[j] / 100.0 + 0.5), times[j])
                 for j in range(n_reads)
                 if i < len(rates[j]) and rates[j][i] is not None and players[j]]
        e_fit = wls(e_pts, HALF_LIFE_H, h_at)
        gamma = damp(e_fit[1], recent_slope(e_pts, HALF_LIFE_H, h_at)) if e_fit else 0.0
        se_gamma = e_fit[3] if e_fit else 0.0
        resid_e = e_fit[2] if e_fit else 0.0

        l_struct = None
        struct_unc = 0.0
        if e_fit and ln_p_fc is not None:
            ln_e_fc = e_fit[0] + e_fit[1] * xs[-1] + gamma * dx
            e_fc = min(math.exp(ln_e_fc), 0.98 * math.exp(ln_p_fc))
            p_fc = math.exp(ln_p_fc)
            if e_fc > 0.05 and p_fc - e_fc > 0.05:
                l_struct = math.log(e_fc) - math.log(p_fc - e_fc)
                struct_unc = math.sqrt(resid_e ** 2 + (se_gamma * dx) ** 2)

        # Regime-aware ensemble. How the trophy has been moving decides whom to
        # trust: a falling rate is mostly the tracked pool's dilution (the
        # structural e/P view), a flat one is the page holding (persistence),
        # a rising one is the cohort still progressing (the local trend).
        l_persist = logit(last_r)
        predictors = [p for p in (l_direct, l_struct, l_persist) if p is not None]
        # Regime signal: half the last 8 hours' slope, half the last day's —
        # a decline can flatten for a while and still be under way.
        if s_recent is not None or s_day is not None:
            vals = [v for v in (s_recent, s_day) if v is not None]
            u = max(-0.5, min(0.5, sum(vals) / len(vals)))
        else:
            u = 0.0
        t = u + 0.5
        w_trend, w_struct, w_persist = 0.15 + 0.35 * t, 0.55 - 0.30 * t, 0.30 - 0.05 * t
        # A tracked pool still growing will dilute every rate at its next
        # refresh: holding the last rate then is a bad anchor, so lean on the
        # structural e/P view instead — but only for a rate that has actually
        # been falling, which is the dilution signature.
        if ln_p_fc is not None and ln_p_last is not None and u < -0.15:
            p_growth = math.exp(ln_p_fc - ln_p_last)
            if p_growth > 1.2:
                shift = min(0.20, w_persist)
                w_persist -= shift
                w_struct += shift
        if l_struct is None:
            w_trend += w_struct / 2.0
            w_persist += w_struct / 2.0
            w_struct = 0.0
        l_med = (w_trend * l_direct + w_struct * (l_struct if l_struct is not None
                 else l_direct) + w_persist * l_persist)

        mean_p = sum(predictors) / len(predictors)
        dis = math.sqrt(sum((p - mean_p) ** 2 for p in predictors)
                        / len(predictors))
        s_parts = [resid_r if r_fit else 0.6, floor_s,
                   min(0.8 * dis, 1.0)]
        if l_struct is not None:
            s_parts.append(struct_unc)
        if p_fit:
            s_parts.append(min(se_beta * abs(dx), 1.0))
        s_parts.append(MODEL_ERR * abs(dx))   # the world is not log-linear
        s = min(math.sqrt(sum(p * p for p in s_parts)) * SPREAD_BUMP, 2.0)

        if stale:
            qs = [expit(l_med + STALE_SHIFT + z * s + extra)
                  for z, extra in zip(Z, STALE_EXTRA)]
        else:
            qs = [expit(l_med + z * s) for z in Z]
        qs = sorted(round4(min(100.0, max(0.0, q))) for q in qs)
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
