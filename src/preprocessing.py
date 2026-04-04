"""
Components 2–5 – Preprocessing Stage  +  Four-layer Bias Mitigation

2. Minimum Sample Size Check
3. Smoothing / Normalisation
4. Demographic Bias Adjustment  (post-stratification weights, Giorgi et al.)
5. CI / Uncertainty Estimate    (classifier-error corrected, Daughton & Paul)

Four-layer bias mitigation (all four layers from the proposal):
  Layer 1 – Platform underrepresentation  : platform_bias_adjustment()
  Layer 2 – Sentiment tool demographic bias : demographic_bias_adjustment()
  Layer 3 – Geographic density bias       : handled by rural adjacency weighting
             in aggregate_track.py (crisis_score → _rural_adjacency_weight)
  Layer 4 – Intervention bias on low-engagement users : intervention_bias_adjustment()

References
----------
Giorgi et al. (2022): Robust Poststratification — +53% accuracy over uncorrected.
Daughton & Paul (2019): standard CIs for social media health estimates are
    inaccurate without correcting for classifier error.
Aguirre & Dredze (2021): depression classifiers perform differently across
    gender and racial groups.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime
from typing import Dict, List, Optional

import src.config as cfg
from src.models import (
    BiasAdjustmentResult,
    ConfidenceResult,
    RegionSignal,
    SampleCheckResult,
    SampleStatus,
    SmoothedSignal,
)


# ---------------------------------------------------------------------------
# Component 2 – Minimum Sample Size Check
# ---------------------------------------------------------------------------

def _density_weight(pop_density: float) -> float:
    """
    log2(1 + pop_density / DENSITY_REF)

    Denser areas produce more posts per-capita so we require proportionally
    more posts before trusting the signal.
    """
    return math.log2(1.0 + pop_density / cfg.DENSITY_REF)


def sample_size_check(region: RegionSignal, date_key: str) -> SampleCheckResult:
    """
    Ensure the region has enough posts for statistical validity.

    Parameters
    ----------
    region   : RegionSignal with daily_counts and pop_density populated
    date_key : "YYYY-MM-DD" string for the day being checked
    """
    density_w = _density_weight(region.pop_density)
    min_n = cfg.BASE_MIN_N * density_w
    count = region.daily_counts.get(date_key, 0)

    if count >= min_n:
        return SampleCheckResult(status=SampleStatus.SUFFICIENT, n=count, min_n=min_n)
    else:
        return SampleCheckResult(
            status=SampleStatus.INSUFFICIENT,
            n=count,
            min_n=min_n,
            fallback="WIDEN_GEO | DEFER",
        )


# ---------------------------------------------------------------------------
# Component 3 – Smoothing / Normalisation
# ---------------------------------------------------------------------------

def smooth_and_normalise(region: RegionSignal, today: Optional[date] = None) -> SmoothedSignal:
    """
    Apply 7-day rolling average, calendar dampening, and per-capita normalisation.

    Parameters
    ----------
    region : RegionSignal
    today  : date to process (defaults to UTC today)
    """
    if today is None:
        today = datetime.utcnow().date()

    today_key = today.strftime("%Y-%m-%d")

    # ---- Step 1: 7-day rolling average ----
    sorted_keys = sorted(region.daily_counts.keys())
    # Keep only the last ROLLING_WINDOW_DAYS days up to and including today
    window_keys = [k for k in sorted_keys if k <= today_key][-cfg.ROLLING_WINDOW_DAYS:]
    window_values = [region.daily_counts[k] for k in window_keys]
    smoothed = statistics.mean(window_values) if window_values else 0.0

    # ---- Step 2: Calendar dampening ----
    month_day = today.strftime("%m-%d")
    calendar_event = cfg.CALENDAR_EVENTS.get(month_day)
    dampened = False
    event_name: Optional[str] = None

    if calendar_event:
        alpha = calendar_event["alpha"]
        smoothed = smoothed * alpha
        dampened = True
        event_name = calendar_event["name"]

    # ---- Step 3: Per-capita normalisation (posts per 100k population) ----
    per_capita = smoothed / (region.population / 100_000) if region.population > 0 else 0.0

    return SmoothedSignal(
        region_id=region.region_id,
        raw_count=region.daily_counts.get(today_key, 0),
        smoothed_count=smoothed,
        per_capita_rate=per_capita,
        calendar_dampened=dampened,
        dampening_event=event_name,
    )


# ---------------------------------------------------------------------------
# Component 4 – Demographic Bias Adjustment
# ---------------------------------------------------------------------------
# Post-stratification corrects for the fact that social media users are not
# representative of the underlying population (Giorgi et al., 2022).
#
# The weight for a stratum s is:
#   w_s = (population_share_s) / (platform_share_s)
#
# If census/platform breakdowns are unavailable we fall back to uniform
# weights (no adjustment). Weights are applied to the smoothed per-capita
# rate before CI estimation.

def demographic_bias_adjustment(
    per_capita_rate: float,
    strata_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Apply post-stratification demographic bias adjustment.

    Parameters
    ----------
    per_capita_rate : smoothed per-capita rate from smooth_and_normalise()
    strata_weights  : mapping of stratum label → post-stratification weight.
                      If None or empty, returns rate unchanged (no adjustment).

    Returns
    -------
    Bias-adjusted per-capita rate.

    Notes
    -----
    strata_weights are computed from:
        w_s = census_proportion_s / platform_proportion_s
    and passed in per region from the pre-computed lookup table.
    The adjusted rate is the weighted mean across strata.
    """
    if not strata_weights:
        return per_capita_rate

    total_weight = sum(strata_weights.values())
    if total_weight <= 0:
        return per_capita_rate

    # Weighted correction: scale rate by the mean post-stratification weight
    # (assumes the raw rate already embeds the platform composition implicitly)
    mean_weight = total_weight / len(strata_weights)
    adjusted = per_capita_rate * mean_weight
    return adjusted


# ---------------------------------------------------------------------------
# Component 5 – CI / Uncertainty Estimate
# ---------------------------------------------------------------------------
# Classifier-error-corrected confidence intervals following Daughton & Paul (2019).
# Standard CIs underestimate true uncertainty because they ignore classifier
# false-positive / false-negative rates.

def confidence_estimate(
    smoothed_history: List[float],
    classifier_error: float = 0.0,
) -> ConfidenceResult:
    """
    Compute confidence interval and derive a confidence score in [0, 1].

    Parameters
    ----------
    smoothed_history : list of per-capita rates for the last W days
    classifier_error : estimated classifier error rate (0.0 = perfect classifier).
                       Widens the CI when > 0 (Daughton & Paul correction).

    Returns
    -------
    ConfidenceResult with conf score, ci_low, ci_high, ci_width, n.
    """
    n = len(smoothed_history)
    if n < 2:
        return ConfidenceResult(conf=0.0, ci_low=float("nan"), ci_high=float("nan"),
                                ci_width=float("nan"), n=n)

    mean = statistics.mean(smoothed_history)
    if mean == 0.0:
        return ConfidenceResult(conf=0.0, ci_low=0.0, ci_high=0.0, ci_width=0.0, n=n)

    sigma = statistics.stdev(smoothed_history)
    se = sigma / math.sqrt(n)

    # Classifier-error correction: inflate SE proportionally to classifier error
    se_corrected = se * (1.0 + classifier_error)

    ci_low = mean - cfg.Z_SCORE * se_corrected
    ci_high = mean + cfg.Z_SCORE * se_corrected
    ci_width = ci_high - ci_low

    raw_conf = 1.0 - (ci_width / mean)
    conf = max(0.0, min(1.0, raw_conf))   # clamp to [0, 1]

    return ConfidenceResult(
        conf=conf,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_width=ci_width,
        n=n,
    )


# ---------------------------------------------------------------------------
# Bias Layer 1 – Platform Underrepresentation
# ---------------------------------------------------------------------------
# Twitter and Reddit users are systematically younger, more urban, and more
# English-speaking than the general population. Raw post rates therefore
# over-represent some demographics and under-represent others.
# Applying a platform weight rescales the signal toward population-level
# representativeness.

def platform_bias_adjustment(
    per_capita_rate: float,
    platform: str = "default",
) -> float:
    """
    Correct for platform-level demographic skew.

    Parameters
    ----------
    per_capita_rate : smoothed per-capita rate (after demographic adjustment)
    platform        : "twitter" | "reddit" | "default"
                      Key into cfg.PLATFORM_WEIGHTS.

    Returns
    -------
    Platform-adjusted per-capita rate.

    Notes
    -----
    PLATFORM_WEIGHTS are derived empirically from:
        w_platform = census_representative_proportion / platform_user_proportion
    per demographic stratum, aggregated to a single platform-level scalar.
    Update values in config.py as new platform demographic data becomes available.
    """
    weight = cfg.PLATFORM_WEIGHTS.get(platform.lower(),
                                       cfg.PLATFORM_WEIGHTS["default"])
    return per_capita_rate * weight


# ---------------------------------------------------------------------------
# Bias Layer 4 – Intervention Bias on Low-Engagement Users
# ---------------------------------------------------------------------------
# Users with low follower counts or low interaction rates are less likely to
# see platform-surfaced resources (banners, overlays, helpline prompts) because
# the platform's own ranking algorithms deprioritise their feeds.
# This creates a systematic intervention gap: the pipeline surfaces resources,
# but those resources may never reach the people who need them most.
#
# Correction: for low-engagement users, lower the severity thresholds so they
# escalate more readily to direct human outreach (HUMAN_REVIEW), bypassing
# passive resource surfacing that they are unlikely to encounter.

def intervention_bias_adjustment(
    avg_daily_interactions: float,
    theta_low: float,
    theta_high: float,
) -> tuple:
    """
    Adjust escalation thresholds for low-engagement users.

    Parameters
    ----------
    avg_daily_interactions : from Post.avg_daily_interactions
    theta_low              : current THETA_LOW value
    theta_high             : current THETA_HIGH value

    Returns
    -------
    (adjusted_theta_low, adjusted_theta_high, discount_applied: bool)

    If the user is low-engagement, both thresholds are lowered by
    LOW_ENGAGEMENT_SEVERITY_DISCOUNT so they escalate to direct human
    review at a lower severity score.
    """
    if avg_daily_interactions < cfg.LOW_ENGAGEMENT_INTERACTION_THRESHOLD:
        discount = cfg.LOW_ENGAGEMENT_SEVERITY_DISCOUNT
        return (
            max(0.05, theta_low  - discount),
            max(0.10, theta_high - discount),
            True,
        )
    return (theta_low, theta_high, False)


# ---------------------------------------------------------------------------
# Convenience: apply all four bias layers in sequence and return a summary
# ---------------------------------------------------------------------------

def apply_all_bias_layers(
    per_capita_rate: float,
    platform: str = "default",
    strata_weights: Optional[Dict[str, float]] = None,
    avg_daily_interactions: float = 99.0,
) -> "BiasAdjustmentResult":
    """
    Apply all four bias mitigation layers in sequence.

    Layer 1: Platform underrepresentation
    Layer 2: Sentiment tool demographic bias (post-stratification)
    Layer 3: Geographic density bias — handled separately in aggregate_track.py
    Layer 4: Intervention bias on low-engagement users

    Returns a BiasAdjustmentResult documenting each step.
    """
    # Layer 1
    after_platform = platform_bias_adjustment(per_capita_rate, platform)
    platform_weight = cfg.PLATFORM_WEIGHTS.get(platform.lower(),
                                                cfg.PLATFORM_WEIGHTS["default"])

    # Layer 2
    after_demographic = demographic_bias_adjustment(after_platform, strata_weights)
    demo_weight = (sum(strata_weights.values()) / len(strata_weights)
                   if strata_weights else 1.0)

    # Layer 4 — threshold adjustment (rate unchanged; thresholds shift)
    adj_low, adj_high, low_eng = intervention_bias_adjustment(
        avg_daily_interactions, cfg.THETA_LOW, cfg.THETA_HIGH
    )

    return BiasAdjustmentResult(
        original_rate=per_capita_rate,
        after_platform_adjustment=after_platform,
        after_demographic_adjustment=after_demographic,
        final_rate=after_demographic,
        platform_weight=platform_weight,
        demographic_weight=demo_weight,
        low_engagement_discount_applied=low_eng,
        adjusted_theta_low=adj_low,
        adjusted_theta_high=adj_high,
    )
