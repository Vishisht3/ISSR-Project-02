"""
Main Pipeline Orchestrator

Wires all components together in the order specified by the proposal
architecture diagram:

Raw data
  → Bot / Coordination Filter
  → Minimum Sample Size Check
  → Smoothing / Normalisation
  → Four-layer Bias Mitigation (platform + demographic + geographic + intervention)
  → CI / Uncertainty Estimate
  → [split]
      Individual Track: Severity → First-time Flag
                        → Intervention Bias threshold adjustment
                        → Threshold Decision
                        → HITL queue (if HUMAN_REVIEW)
      Aggregate Track:  Event Classification + Media Corroboration
                        → Contagion Flagging
                        → Bucket → Crisis Scoring → Confidence Threshold
                        → Confirmation Window → Escalation Decision
                        → HITL queue (if ESCALATE)
  → Audit Log + Dashboard Output
  → Campaign Effectiveness Feedback Loop

The pre-made models (VADER sentiment, volume spike, geographic clustering)
are expected to be injected as callables via the `PipelineConfig` class so
the framework remains testable without the prototype's full stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.audit import AuditLog
from src.bot_filter import bot_filter, BotFilterAction
from src.preprocessing import (
    sample_size_check,
    smooth_and_normalise,
    apply_all_bias_layers,
    confidence_estimate,
    intervention_bias_adjustment,
)
from src.individual_track import run_individual_track, threshold_decision
from src.aggregate_track import run_aggregate_track
from src.contagion_flag import check_contagion
from src.hitl_queue import HITLQueue, QueueTrack
from src.heatmap import generate_heatmap
from src.models import (
    ConfidenceResult,
    IndividualAction,
    AggregateAction,
    Post,
    RegionSignal,
    SampleStatus,
)
import src.config as cfg


# ---------------------------------------------------------------------------
# Type aliases for pre-made model callables
# ---------------------------------------------------------------------------
SentimentModel    = Callable[[str], float]           # text → intensity [0,1]
SentimentAggModel = Callable[[RegionSignal], float]  # region → aggregate intensity [0,1]
VolumeSpikeModel  = Callable[[RegionSignal], float]  # region → spike score [0,1]
GeoClusterModel   = Callable[[RegionSignal], float]  # region → cluster score [0,1]


# ---------------------------------------------------------------------------
# Default model implementations (VADER-based, matching the prototype)
# ---------------------------------------------------------------------------

def _default_sentiment(text: str) -> float:
    """
    VADER-based individual post sentiment intensity.
    Maps the negative compound score to [0, 1]: 1 = most intense distress.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyser = SentimentIntensityAnalyzer()
        scores = _analyser.polarity_scores(text)
        return max(0.0, -scores["compound"])
    except ImportError:
        return 0.5


def _default_sentiment_agg(region: RegionSignal) -> float:
    """Aggregate sentiment intensity across a region's post corpus."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyser = SentimentIntensityAnalyzer()
        if not region.post_corpus:
            return 0.0
        scores = [max(0.0, -_analyser.polarity_scores(t)["compound"])
                  for t in region.post_corpus]
        return sum(scores) / len(scores)
    except ImportError:
        return 0.5


def _default_volume_spike(region: RegionSignal) -> float:
    """
    Volume spike score: ratio of today's count to the 7-day rolling mean,
    normalised to [0, 1]. Scores above 1 are clamped.
    """
    counts = list(region.daily_counts.values())
    if len(counts) < 2:
        return 0.0
    baseline = sum(counts[:-1]) / len(counts[:-1])
    if baseline == 0:
        return 0.0
    return min(1.0, counts[-1] / baseline - 1.0)


def _default_geo_cluster(region: RegionSignal) -> float:
    """
    Geographic cluster score based on population density normalisation.
    Returns a value in [0, 1]; dense clusters score higher.
    """
    import math
    density_norm = min(1.0, region.pop_density / 5000.0)
    return density_norm


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Model callables and runtime settings for the AI4MH pipeline.
    The sentiment, volume spike, and geographic clustering models are
    pre-built in the existing prototype; pass them here to connect the
    governance framework to those implementations.
    """
    sentiment_model:     SentimentModel    = field(default=_default_sentiment)
    sentiment_agg_model: SentimentAggModel = field(default=_default_sentiment_agg)
    volume_spike_model:  VolumeSpikeModel  = field(default=_default_volume_spike)
    geo_cluster_model:   GeoClusterModel   = field(default=_default_geo_cluster)
    audit_log_path:      Optional[str]     = None
    hitl_queue_path:     Optional[str]     = None
    confirm_hours:       float             = cfg.CONFIRM_HOURS
    # Platform source used for platform-underrepresentation bias correction
    platform:            str               = "default"


# ---------------------------------------------------------------------------
# Individual post processing
# ---------------------------------------------------------------------------

def process_post(
    post: Post,
    prior_post_count: int,
    region_conf: ConfidenceResult,
    pipeline_cfg: PipelineConfig,
    audit_log: AuditLog,
    hitl_queue: HITLQueue,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a single post through the individual pipeline track.

    HITL gate: if the recommended action is HUMAN_REVIEW, the item is placed
    in the HITL queue and must be approved by a reviewer before any action is
    executed. The pipeline returns the queue item_id so the caller can poll
    get_final_action() later.

    Returns None if the post was filtered by the bot filter.
    """
    # Step 1: Bot filter
    filter_result = bot_filter(post, now)
    if filter_result.action == BotFilterAction.DISCARD:
        return None

    # Step 2: Bias Layer 4 — intervention bias threshold adjustment
    adj_theta_low, adj_theta_high, low_eng_discount = intervention_bias_adjustment(
        post.avg_daily_interactions, cfg.THETA_LOW, cfg.THETA_HIGH
    )

    # Step 3: Individual track (severity → first-time flag)
    sentiment = pipeline_cfg.sentiment_model(post.text)
    track_results = run_individual_track(post, sentiment, prior_post_count)

    # Step 4: Apply adjusted thresholds (re-run threshold decision with bias-corrected bounds)
    ftf      = track_results["first_time_result"]
    thr      = threshold_decision(ftf.adjusted_severity)

    # If low-engagement discount was applied, re-evaluate with lower thresholds
    if low_eng_discount:
        from src.models import IndividualAction as IA
        sev = ftf.adjusted_severity
        if sev >= adj_theta_high:
            thr_action = IA.HUMAN_REVIEW
        elif sev >= adj_theta_low:
            thr_action = IA.HELPLINE_PROMPT
        else:
            thr_action = IA.PASSIVE_RESOURCE
        from src.models import ThresholdDecisionResult
        thr = ThresholdDecisionResult(
            action=thr_action,
            severity=sev,
            theta_low=adj_theta_low,
            theta_high=adj_theta_high,
        )

    track_results["threshold_result"] = thr
    track_results["low_engagement_discount"] = low_eng_discount

    # Step 5: Audit log
    audit_id = audit_log.write_individual(
        region_id=post.region_id,
        action=thr.action.value,
        severity=ftf.adjusted_severity,
        confidence=region_conf.conf,
        first_time=ftf.first_time,
        metadata={
            "post_id":               post.post_id,
            "sentiment_raw":         track_results["severity_result"].sentiment_raw,
            "isolation_raw":         track_results["severity_result"].isolation_raw,
            "low_engagement_flag":   low_eng_discount,
            "adj_theta_low":         adj_theta_low,
            "adj_theta_high":        adj_theta_high,
        },
    )

    # Step 6: HITL gate — block HUMAN_REVIEW until a reviewer approves
    hitl_item_id = None
    if thr.action == IndividualAction.HUMAN_REVIEW:
        hitl_item_id = hitl_queue.enqueue(
            track=QueueTrack.INDIVIDUAL.value,
            region_id=post.region_id,
            recommended_action=thr.action.value,
            severity_score=ftf.adjusted_severity,
            confidence=region_conf.conf,
            audit_record_id=audit_id,
            metadata={"post_id": post.post_id},
        )

    track_results["audit_record_id"] = audit_id
    track_results["hitl_item_id"]    = hitl_item_id
    return track_results


# ---------------------------------------------------------------------------
# Region aggregate processing
# ---------------------------------------------------------------------------

def process_region(
    region: RegionSignal,
    pipeline_cfg: PipelineConfig,
    audit_log: AuditLog,
    hitl_queue: HITLQueue,
    strata_weights: Optional[Dict[str, float]] = None,
    news_articles: Optional[List[Dict[str, Any]]] = None,
    volume_ratio: float = 1.0,
    geo_concentration: float = 0.0,
    multi_region_spike: bool = False,
    score_history: Optional[List[float]] = None,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process one region through the full preprocessing + aggregate pipeline.

    HITL gate: ESCALATE decisions are queued for human review before acting.
    Contagion flag: checked after media corroboration; if raised, bucket
    dampening is suppressed and the escalation threshold is lowered.

    Returns None if the region fails the sample size check.
    """
    if today is None:
        today = (now or datetime.utcnow()).date()
    date_key = today.strftime("%Y-%m-%d")

    # Step 1: Minimum sample size check
    sample_result = sample_size_check(region, date_key)
    if sample_result.status == SampleStatus.INSUFFICIENT:
        return {"sample_result": sample_result, "skipped": True}

    # Step 2: Smoothing / normalisation
    smoothed = smooth_and_normalise(region, today)

    # Step 3: Full four-layer bias adjustment
    bias_result = apply_all_bias_layers(
        per_capita_rate=smoothed.per_capita_rate,
        platform=pipeline_cfg.platform,
        strata_weights=strata_weights,
    )
    adjusted_rate = bias_result.final_rate

    # Step 4: CI / uncertainty estimate
    sorted_keys = sorted(region.daily_counts.keys())
    history_float = [float(region.daily_counts[k])
                     for k in sorted_keys[-cfg.ROLLING_WINDOW_DAYS:]]
    region_conf = confidence_estimate(history_float)

    # Step 5: Contagion check (before aggregate track so it can influence bucketing)
    contagion_result = check_contagion(region, news_articles, now)

    # Step 6: Aggregate track (event classification → escalation)
    agg_results = run_aggregate_track(
        region=region,
        sentiment_intensity_agg=pipeline_cfg.sentiment_agg_model(region),
        volume_spike_score=pipeline_cfg.volume_spike_model(region),
        geo_cluster_score=pipeline_cfg.geo_cluster_model(region),
        region_conf=region_conf,
        news_articles=news_articles,
        volume_ratio=volume_ratio,
        geo_concentration=geo_concentration,
        multi_region_spike=multi_region_spike,
        score_history=score_history,
        now=now,
    )

    # Step 7: Apply contagion override to bucket and threshold
    bkt = agg_results["bucket_result"]
    thr = agg_results["threshold_result"]
    crs = agg_results["crisis_result"]

    if contagion_result.flagged:
        # Force full weight — do NOT dampen as a normal reactive event
        from src.models import BucketResult, BucketType
        bkt = BucketResult(bucket=BucketType.PERSONAL_DISTRESS, delta=1.0)
        crisis_adj_contagion = min(1.0, crs.crisis_raw * 1.0)

        from src.models import ConfidenceThresholdResult
        new_threshold = max(0.10, thr.effective_threshold - contagion_result.threshold_discount)
        thr = ConfidenceThresholdResult(
            exceeds=crisis_adj_contagion > new_threshold,
            crisis_adj=crisis_adj_contagion,
            effective_threshold=new_threshold,
            confidence_used=thr.confidence_used,
            penalty_applied=thr.penalty_applied,
        )
        agg_results["bucket_result"]    = bkt
        agg_results["threshold_result"] = thr

        # Re-run confirmation window and escalation with updated threshold
        from src.aggregate_track import confirmation_window, escalation_decision
        win = confirmation_window(region, thr, now=now,
                                  confirm_hours=pipeline_cfg.confirm_hours)
        esc = escalation_decision(region, win, crisis_adj_contagion)
        agg_results["window_result"]     = win
        agg_results["escalation_result"] = esc

    esc = agg_results["escalation_result"]
    evt = agg_results["event_result"]
    crs = agg_results["crisis_result"]

    # Step 8: Audit log
    audit_id = audit_log.write_aggregate(
        region_id=region.region_id,
        action=esc.action.value,
        crisis_score=esc.crisis_score,
        confidence=region_conf.conf,
        event_type=evt.event_type,
        bucket=bkt.bucket.value,
        metadata={
            "crisis_raw":            crs.crisis_raw,
            "bucket_delta":          bkt.delta,
            "effective_threshold":   agg_results["threshold_result"].effective_threshold,
            "smoothed_per_capita":   adjusted_rate,
            "platform_weight":       bias_result.platform_weight,
            "demographic_weight":    bias_result.demographic_weight,
            "ci_low":                region_conf.ci_low,
            "ci_high":               region_conf.ci_high,
            "contagion_flagged":     contagion_result.flagged,
            "contagion_grade":       contagion_result.grade,
        },
    )

    # Step 9: HITL gate — block ESCALATE until a human reviewer approves
    hitl_item_id = None
    if esc.action == AggregateAction.ESCALATE:
        hitl_item_id = hitl_queue.enqueue(
            track=QueueTrack.AGGREGATE.value,
            region_id=region.region_id,
            recommended_action=esc.action.value,
            crisis_score=esc.crisis_score,
            confidence=region_conf.conf,
            event_type=evt.event_type,
            contagion_flagged=contagion_result.flagged,
            audit_record_id=audit_id,
        )

    return {
        "sample_result":    sample_result,
        "smoothed":         smoothed,
        "bias_result":      bias_result,
        "adjusted_rate":    adjusted_rate,
        "region_conf":      region_conf,
        "contagion_result": contagion_result,
        "audit_record_id":  audit_id,
        "hitl_item_id":     hitl_item_id,
        **agg_results,
    }


# ---------------------------------------------------------------------------
# Full pipeline class
# ---------------------------------------------------------------------------

class CrisisPipeline:
    """
    High-level interface for the AI4MH framework.

    Usage
    -----
    pipeline = CrisisPipeline(PipelineConfig(
        sentiment_model=real_vader_fn,
        platform="twitter",
    ))

    # Process individual posts
    pipeline.ingest_posts(posts, prior_counts_by_account)

    # Process regional aggregates
    pipeline.ingest_regions(regions, news_by_region)

    # Reviewer checks pending HITL items
    pending = pipeline.hitl_queue.get_pending()
    pipeline.hitl_queue.review(item_id, reviewer_id="dr_jones", decision="APPROVED")

    # Generate heatmap
    summary = pipeline.render_heatmap(time_window_hours=1.0)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        audit_log: Optional[AuditLog] = None,
        hitl_queue: Optional[HITLQueue] = None,
    ) -> None:
        self.config     = config or PipelineConfig()
        self.audit_log  = audit_log  or AuditLog(self.config.audit_log_path)
        self.hitl_queue = hitl_queue or HITLQueue(self.config.hitl_queue_path)

    def ingest_posts(
        self,
        posts: List[Post],
        prior_counts: Optional[Dict[str, int]] = None,
        region_confs: Optional[Dict[str, ConfidenceResult]] = None,
        now: Optional[datetime] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of individual posts.

        Posts that result in HUMAN_REVIEW are placed in the HITL queue.
        The returned dict includes hitl_item_id when applicable — poll
        hitl_queue.get_final_action(item_id) before executing any output.
        """
        prior_counts = prior_counts or {}
        region_confs = region_confs or {}
        default_conf = ConfidenceResult(conf=0.5, ci_low=0.0, ci_high=1.0,
                                        ci_width=1.0, n=0)
        results = []
        for post in posts:
            conf  = region_confs.get(post.region_id, default_conf)
            count = prior_counts.get(post.account_id, 0)
            r = process_post(post, count, conf, self.config,
                             self.audit_log, self.hitl_queue, now)
            results.append(r)
        return results

    def ingest_regions(
        self,
        regions: List[RegionSignal],
        news_by_region: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        strata_by_region: Optional[Dict[str, Dict[str, float]]] = None,
        volume_ratios: Optional[Dict[str, float]] = None,
        score_histories: Optional[Dict[str, List[float]]] = None,
        now: Optional[datetime] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of regional aggregate signals.

        Regions that result in ESCALATE are placed in the HITL queue.
        Contagion check runs automatically for every region.
        All four bias layers are applied in sequence.
        """
        news_by_region   = news_by_region   or {}
        strata_by_region = strata_by_region or {}
        volume_ratios    = volume_ratios    or {}
        score_histories  = score_histories  or {}
        today = (now or datetime.utcnow()).date()

        multi_spike = sum(
            1 for r in regions if volume_ratios.get(r.region_id, 1.0) > 2.0
        ) > 1

        results = []
        for region in regions:
            r = process_region(
                region=region,
                pipeline_cfg=self.config,
                audit_log=self.audit_log,
                hitl_queue=self.hitl_queue,
                strata_weights=strata_by_region.get(region.region_id),
                news_articles=news_by_region.get(region.region_id),
                volume_ratio=volume_ratios.get(region.region_id, 1.0),
                multi_region_spike=multi_spike,
                score_history=score_histories.get(region.region_id),
                today=today,
                now=now,
            )
            results.append(r)
        return results

    def render_heatmap(
        self,
        time_window_hours: float = 1.0,
        region_centroids: Optional[Dict[str, Tuple[float, float]]] = None,
        output_path: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate and return the heatmap summary."""
        return generate_heatmap(
            self.audit_log,
            time_window_hours=time_window_hours,
            region_centroids=region_centroids,
            output_path=output_path,
            now=now,
        )
