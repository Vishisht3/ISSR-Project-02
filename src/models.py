"""
Typed data models for the AI4MH crisis detection pipeline.
All inter-component data is passed as instances of these dataclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackType(str, Enum):
    INDIVIDUAL = "INDIVIDUAL"
    AGGREGATE  = "AGGREGATE"


class IndividualAction(str, Enum):
    PASSIVE_RESOURCE = "PASSIVE_RESOURCE"   # banner / sidebar link
    HELPLINE_PROMPT  = "HELPLINE_PROMPT"    # anonymous helpline overlay
    HUMAN_REVIEW     = "HUMAN_REVIEW"       # queue to on-call moderator


class AggregateAction(str, Enum):
    HOLD         = "HOLD"
    FLAG         = "FLAG"
    MONITORING   = "MONITORING"
    ESCALATE     = "ESCALATE"


class BucketType(str, Enum):
    REACTIVE         = "REACTIVE"
    PERSONAL_DISTRESS = "PERSONAL_DISTRESS"


class SampleStatus(str, Enum):
    SUFFICIENT   = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class BotFilterAction(str, Enum):
    PASS    = "PASS"
    DISCARD = "DISCARD"


# ---------------------------------------------------------------------------
# Raw input types
# ---------------------------------------------------------------------------

@dataclass
class Post:
    """A single social media post ingested by the pipeline."""
    post_id: str
    account_id: str
    text: str
    timestamp: datetime
    region_id: str
    latitude: float
    longitude: float
    account_age_days: int
    posts_last_hour: int
    reply_count: int
    avg_daily_interactions: float      # for inactive-graph isolation signal
    is_first_post: bool = False        # set by upstream data pipeline
    local_hour: int = 0                # hour of day in the poster's local timezone
    raw_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionSignal:
    """Aggregated signal for a geographic region over a time window."""
    region_id: str
    centroid_lat: float
    centroid_lon: float
    population: float
    pop_density: float                 # people per km²
    daily_counts: Dict[str, int]       # {"YYYY-MM-DD": count, ...}
    post_corpus: List[str]             # raw post texts for keyword extraction
    spike_start_time: Optional[datetime] = None
    smoothed_count: float = 0.0
    per_capita_rate: float = 0.0
    # Optional engagement-weighted daily counts.
    # Each post contributes log2(1 + upvotes + comments) instead of a flat 1,
    # so high-engagement posts satisfy the sample-size gate with fewer raw
    # posts. When None, sample_size_check falls back to raw daily_counts.
    engagement_weighted_counts: Optional[Dict[str, float]] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preprocessing outputs
# ---------------------------------------------------------------------------

@dataclass
class BotFilterResult:
    action: BotFilterAction
    reason: Optional[str] = None       # why discarded, if applicable


@dataclass
class SampleCheckResult:
    status: SampleStatus
    n: int
    min_n: float
    fallback: Optional[str] = None     # "WIDEN_GEO" | "DEFER"


@dataclass
class SmoothedSignal:
    region_id: str
    raw_count: int
    smoothed_count: float
    per_capita_rate: float
    calendar_dampened: bool = False
    dampening_event: Optional[str] = None


@dataclass
class ConfidenceResult:
    conf: float                        # clamped to [0.0, 1.0]
    ci_low: float
    ci_high: float
    ci_width: float
    n: int


# ---------------------------------------------------------------------------
# Individual track outputs
# ---------------------------------------------------------------------------

@dataclass
class IsolationSignals:
    late_night: int       # 0 or 1
    zero_replies: int     # 0 or 1
    alone_keywords: float # 0.0–1.0 keyword match score
    inactive_graph: int   # 0 or 1


@dataclass
class SeverityResult:
    severity: float                   # composite [0.0, 1.0]
    sentiment_raw: float
    isolation_raw: float
    signals: IsolationSignals


@dataclass
class FirstTimeFlagResult:
    first_time: bool
    adjusted_severity: float


@dataclass
class ThresholdDecisionResult:
    action: IndividualAction
    severity: float
    theta_low: float
    theta_high: float


# ---------------------------------------------------------------------------
# Aggregate track outputs
# ---------------------------------------------------------------------------

@dataclass
class EventClassificationResult:
    event_type: str                    # tier name or "NONE"
    match_confidence: float
    event_weight: float


@dataclass
class MediaCorroborationResult:
    corroborated: bool
    score: float
    articles_found: int
    lag_hours: float


@dataclass
class BucketResult:
    bucket: BucketType
    delta: float


@dataclass
class CrisisScoreResult:
    crisis_raw: float
    crisis_adj: float                  # after bucket dampening
    sentiment_score: float
    volume_score: float
    geo_score: float
    bucket_delta: float


@dataclass
class ConfidenceThresholdResult:
    exceeds: bool
    crisis_adj: float
    effective_threshold: float
    confidence_used: float
    penalty_applied: float


@dataclass
class ConfirmationWindowResult:
    confirmed: bool
    action: AggregateAction
    hours_remaining: float = 0.0


@dataclass
class EscalationResult:
    action: AggregateAction
    region_id: str
    crisis_score: float


# ---------------------------------------------------------------------------
# Audit log record
# ---------------------------------------------------------------------------

@dataclass
class BiasAdjustmentResult:
    """Captures the full four-layer bias correction applied to a signal."""
    original_rate: float
    after_platform_adjustment: float
    after_demographic_adjustment: float
    final_rate: float
    platform_weight: float
    demographic_weight: float
    low_engagement_discount_applied: bool
    adjusted_theta_low: float
    adjusted_theta_high: float


@dataclass
class AuditRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    track: Optional[TrackType] = None
    region_id: Optional[str] = None
    action_taken: Optional[str] = None
    severity_score: Optional[float] = None
    crisis_score: Optional[float] = None
    confidence: Optional[float] = None
    event_type: Optional[str] = None
    bucket: Optional[str] = None
    first_time: Optional[bool] = None
    reviewer_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
