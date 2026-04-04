"""
Campaign Effectiveness Feedback Loop

Enables pre/post measurement of intervention impact and longitudinal
strategy adjustment — as specified in the proposal architecture diagram.

Three outputs:
  1. Pre/post signal comparison per region
  2. Intervention impact score
  3. Strategy adjustment flag

This closes the outer feedback loop: if an intervention (helpline campaign,
resource deployment, public health announcement) was launched in a region,
the framework measures whether crisis signals subsequently dropped, stayed
flat, or worsened, and flags the strategy team accordingly.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.audit import AuditLog


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InterventionRecord:
    """Describes a public health intervention applied to a region."""
    intervention_id: str
    region_id: str
    intervention_type: str           # e.g. "helpline_campaign", "resource_deployment"
    started_at: datetime
    ended_at: Optional[datetime] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalWindow:
    """Aggregated signal stats for a time window."""
    region_id: str
    window_label: str                 # "pre" or "post"
    start: datetime
    end: datetime
    crisis_mean: Optional[float]
    crisis_std: Optional[float]
    severity_mean: Optional[float]
    escalation_count: int
    n_records: int


@dataclass
class CampaignEffectivenessResult:
    intervention_id: str
    region_id: str
    pre_window: SignalWindow
    post_window: SignalWindow
    impact_score: float               # negative = improvement, positive = worsening
    strategy_flag: str                # "EFFECTIVE" | "NEUTRAL" | "INEFFECTIVE" | "INSUFFICIENT_DATA"
    recommendation: str


# ---------------------------------------------------------------------------
# Thresholds for strategy flag
# ---------------------------------------------------------------------------
EFFECTIVE_THRESHOLD: float = -0.10    # impact score below this → intervention worked
INEFFECTIVE_THRESHOLD: float = 0.05  # impact score above this → intervention failed
MIN_RECORDS_FOR_ASSESSMENT: int = 5  # minimum records in each window


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _extract_window(
    audit_log: AuditLog,
    region_id: str,
    start: datetime,
    end: datetime,
    label: str,
) -> SignalWindow:
    records = audit_log.query(region_id=region_id, time_from=start, time_to=end)

    crisis_scores   = [r["crisis_score"]   for r in records if r.get("crisis_score")   is not None]
    severity_scores = [r["severity_score"] for r in records if r.get("severity_score") is not None]
    escalations     = sum(1 for r in records if r.get("action_taken") == "ESCALATE")

    return SignalWindow(
        region_id=region_id,
        window_label=label,
        start=start,
        end=end,
        crisis_mean=statistics.mean(crisis_scores)   if crisis_scores   else None,
        crisis_std =statistics.stdev(crisis_scores)  if len(crisis_scores) > 1 else None,
        severity_mean=statistics.mean(severity_scores) if severity_scores else None,
        escalation_count=escalations,
        n_records=len(records),
    )


# ---------------------------------------------------------------------------
# Impact scoring
# ---------------------------------------------------------------------------

def _impact_score(pre: SignalWindow, post: SignalWindow) -> float:
    """
    Compute intervention impact score.

    impact = (post_crisis_mean - pre_crisis_mean) / pre_crisis_mean

    Negative → signal dropped after intervention (effective).
    Zero / positive → signal unchanged or worsened.
    Returns 0.0 if either window has no data.
    """
    if pre.crisis_mean is None or post.crisis_mean is None:
        return 0.0
    if pre.crisis_mean == 0.0:
        return 0.0
    return (post.crisis_mean - pre.crisis_mean) / pre.crisis_mean


def _strategy_flag(impact: float, pre: SignalWindow, post: SignalWindow) -> tuple:
    """Return (flag_string, recommendation_string)."""
    if pre.n_records < MIN_RECORDS_FOR_ASSESSMENT or post.n_records < MIN_RECORDS_FOR_ASSESSMENT:
        return (
            "INSUFFICIENT_DATA",
            "Not enough records in pre or post window to assess effectiveness. "
            "Extend measurement period or check data pipeline.",
        )
    if impact <= EFFECTIVE_THRESHOLD:
        return (
            "EFFECTIVE",
            f"Crisis signal dropped by {abs(impact)*100:.1f}% post-intervention. "
            "Continue and consider scaling this strategy to adjacent regions.",
        )
    if impact >= INEFFECTIVE_THRESHOLD:
        return (
            "INEFFECTIVE",
            f"Crisis signal increased by {impact*100:.1f}% post-intervention. "
            "Review intervention design; consider alternative approaches.",
        )
    return (
        "NEUTRAL",
        "No significant change detected. May need longer observation window or "
        "higher-intensity intervention.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_campaign_effectiveness(
    audit_log: AuditLog,
    intervention: InterventionRecord,
    pre_window_days: int = 7,
    post_window_days: int = 7,
) -> CampaignEffectivenessResult:
    """
    Compare pre/post crisis signals around an intervention event.

    Parameters
    ----------
    audit_log         : AuditLog instance
    intervention      : the InterventionRecord being evaluated
    pre_window_days   : how many days before intervention to use as baseline
    post_window_days  : how many days after intervention end to measure impact

    Returns
    -------
    CampaignEffectivenessResult with impact_score and strategy_flag.
    """
    pre_end   = intervention.started_at
    pre_start = pre_end - timedelta(days=pre_window_days)

    post_start = intervention.ended_at or intervention.started_at
    post_end   = post_start + timedelta(days=post_window_days)

    pre_window  = _extract_window(audit_log, intervention.region_id,
                                  pre_start, pre_end, "pre")
    post_window = _extract_window(audit_log, intervention.region_id,
                                  post_start, post_end, "post")

    impact = _impact_score(pre_window, post_window)
    flag, recommendation = _strategy_flag(impact, pre_window, post_window)

    return CampaignEffectivenessResult(
        intervention_id=intervention.intervention_id,
        region_id=intervention.region_id,
        pre_window=pre_window,
        post_window=post_window,
        impact_score=impact,
        strategy_flag=flag,
        recommendation=recommendation,
    )


def batch_measure(
    audit_log: AuditLog,
    interventions: List[InterventionRecord],
    pre_window_days: int = 7,
    post_window_days: int = 7,
) -> List[CampaignEffectivenessResult]:
    """
    Evaluate effectiveness for a list of interventions.
    Returns results sorted by impact_score (most effective first).
    """
    results = [
        measure_campaign_effectiveness(audit_log, iv, pre_window_days, post_window_days)
        for iv in interventions
    ]
    return sorted(results, key=lambda r: r.impact_score)
