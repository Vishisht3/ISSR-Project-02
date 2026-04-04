"""
Aggregate Track – Components 8–14

 8. Event Classification         (5-tier taxonomy + feature extraction)
 9. Media Corroboration          (local news scrape, timing + geo + keyword 3-axis match)
10. Reactive vs Personal Bucket  (dampening weight δ)
11. Crisis Scoring               (weighted composite + rural adjacency weighting)
12. Confidence-modulated Threshold
13. Confirmation Window          (signal persistence requirement — default 24 hr)
14. Escalation Decision          (HOLD / FLAG / ESCALATE)

References
----------
McClellan et al. (2017): ARIMA on 176M tweets — unexpected volume spikes < 2 days.
Gao et al. (2018): KDE smoothing + historical normalisation for spatiotemporal events.
Daughton & Paul (2019): classifier-error-corrected confidence intervals.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import src.config as cfg
from src.models import (
    AggregateAction,
    BucketResult,
    BucketType,
    ConfidenceResult,
    ConfidenceThresholdResult,
    ConfirmationWindowResult,
    CrisisScoreResult,
    EscalationResult,
    EventClassificationResult,
    MediaCorroborationResult,
    RegionSignal,
)


# ---------------------------------------------------------------------------
# In-memory trigger store for confirmation window (swap for DB in production)
# ---------------------------------------------------------------------------
_triggers: Dict[str, datetime] = {}   # {region_id: trigger_timestamp}


# ---------------------------------------------------------------------------
# Component 8 – Event Classification
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _extract_keywords(texts: List[str], top_n: int = 30) -> set:
    """Simple frequency-based keyword extraction (TF proxy)."""
    import re
    from collections import Counter
    stopwords = {
        "the", "a", "an", "is", "it", "in", "of", "and", "or",
        "to", "for", "on", "at", "with", "i", "me", "my", "you",
        "we", "he", "she", "they", "this", "that", "was", "are",
    }
    all_words: List[str] = []
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        all_words.extend(w for w in words if w not in stopwords and len(w) > 2)
    counts = Counter(all_words)
    return {word for word, _ in counts.most_common(top_n)}


def _calendar_match(today: datetime) -> float:
    """Return 1.0 if today matches a known calendar event, else 0.0."""
    key = today.strftime("%m-%d")
    return 1.0 if key in cfg.CALENDAR_EVENTS else 0.0


def _celebrity_match(post_keywords: set) -> float:
    celebrity_kw = set(cfg.EVENT_KEYWORDS.get("CELEBRITY", []))
    return _jaccard(post_keywords, celebrity_kw)


def _local_mh_match(post_keywords: set, geo_concentration: float) -> float:
    """Combine keyword overlap with geographic concentration metric."""
    local_kw = set(cfg.EVENT_KEYWORDS.get("LOCAL_MH", []))
    kw_score = _jaccard(post_keywords, local_kw)
    return 0.5 * kw_score + 0.5 * min(geo_concentration, 1.0)


def _socioeconomic_match(post_keywords: set, volume_ratio: float) -> float:
    """Combine keyword overlap with volume-above-baseline ratio."""
    soc_kw = set(cfg.EVENT_KEYWORDS.get("SOCIOECONOMIC", []))
    kw_score = _jaccard(post_keywords, soc_kw)
    vol_score = min(volume_ratio / 2.0, 1.0)   # normalise; 2× baseline → 1.0
    return 0.5 * kw_score + 0.5 * vol_score


def _global_match(post_keywords: set, multi_region_spike: bool) -> float:
    global_kw = set(cfg.EVENT_KEYWORDS.get("GLOBAL", []))
    kw_score = _jaccard(post_keywords, global_kw)
    spike_score = 1.0 if multi_region_spike else 0.0
    return 0.5 * kw_score + 0.5 * spike_score


def classify_event(
    region: RegionSignal,
    today: Optional[datetime] = None,
    volume_ratio: float = 1.0,
    geo_concentration: float = 0.0,
    multi_region_spike: bool = False,
) -> EventClassificationResult:
    """
    Classify the likely driver behind a regional aggregate signal spike.

    Parameters
    ----------
    region              : RegionSignal with post_corpus populated
    today               : UTC datetime (defaults to now)
    volume_ratio        : current volume / historical baseline (1.0 = normal)
    geo_concentration   : fraction of posts concentrated in a single sub-region
    multi_region_spike  : True if multiple regions spiked simultaneously

    Returns
    -------
    EventClassificationResult with event_type, match_confidence, event_weight.
    """
    if today is None:
        today = datetime.utcnow()

    post_kw = _extract_keywords(region.post_corpus)

    scores: Dict[str, float] = {
        "CALENDAR":      _calendar_match(today),
        "CELEBRITY":     _celebrity_match(post_kw),
        "LOCAL_MH":      _local_mh_match(post_kw, geo_concentration),
        "SOCIOECONOMIC": _socioeconomic_match(post_kw, volume_ratio),
        "GLOBAL":        _global_match(post_kw, multi_region_spike),
    }

    best_type = max(scores, key=scores.__getitem__)
    best_score = scores[best_type]

    if best_score < cfg.TIER_MIN_THRESHOLD:
        best_type = "NONE"
        best_score = 0.0

    event_weight = cfg.TIER_WEIGHTS.get(best_type, 1.0)

    return EventClassificationResult(
        event_type=best_type,
        match_confidence=best_score,
        event_weight=event_weight,
    )


# ---------------------------------------------------------------------------
# Component 9 – Media Corroboration
# ---------------------------------------------------------------------------

def _gaussian_decay(lag_hours: float, sigma: float) -> float:
    """Gaussian decay centred at 0 (news precedes or coincides with spike)."""
    return math.exp(-0.5 * (lag_hours / sigma) ** 2)


def media_corroboration(
    region: RegionSignal,
    event_type: str,
    news_articles: Optional[List[Dict[str, Any]]] = None,
) -> MediaCorroborationResult:
    """
    Cross-reference the detected signal with local news articles.

    Parameters
    ----------
    region        : RegionSignal with post_corpus and spike_start_time
    event_type    : classified event type (from classify_event)
    news_articles : list of dicts with keys: headline, summary, published_at (datetime)
                    If None or empty, returns corroborated=False.

    3-axis match (per proposal): keyword overlap · timing alignment · geo (implicit via region)
    """
    if not news_articles:
        return MediaCorroborationResult(
            corroborated=False, score=0.0, articles_found=0, lag_hours=0.0
        )

    event_kw = set(cfg.EVENT_KEYWORDS.get(event_type, []))
    post_kw  = _extract_keywords(region.post_corpus)

    # Axis 1 — keyword overlap
    news_words: set = set()
    for article in news_articles:
        text = (article.get("headline", "") + " " + article.get("summary", "")).lower()
        import re
        news_words.update(re.findall(r"[a-z]+", text))
    keyword_overlap = _jaccard(post_kw | event_kw, news_words)

    # Axis 2 — timing alignment (news should precede or coincide with spike)
    now = datetime.utcnow()
    published_times: List[datetime] = [
        a["published_at"] for a in news_articles
        if isinstance(a.get("published_at"), datetime)
    ]
    if published_times and region.spike_start_time:
        earliest = min(published_times)
        lag_hours = (region.spike_start_time - earliest).total_seconds() / 3600
    else:
        lag_hours = 0.0

    timing_score = _gaussian_decay(lag_hours, cfg.LAG_SIGMA)

    # Composite corroboration score (3-axis: keyword, timing, geo already scoped via region)
    corr_score = 0.5 * keyword_overlap + 0.5 * timing_score

    return MediaCorroborationResult(
        corroborated=corr_score > cfg.CORR_THRESHOLD,
        score=corr_score,
        articles_found=len(news_articles),
        lag_hours=lag_hours,
    )


# ---------------------------------------------------------------------------
# Component 10 – Reactive vs Personal Distress Bucket
# ---------------------------------------------------------------------------

def bucket_signal(
    event_result: EventClassificationResult,
    media_result: MediaCorroborationResult,
) -> BucketResult:
    """
    Determine whether the signal is a reaction to an external event (REACTIVE)
    or reflects organic personal distress (PERSONAL_DISTRESS).

    REACTIVE → apply dampening weight δ from REACTIVE_DELTA config.
    PERSONAL → δ = 1.0 (full weight, no dampening).
    """
    if event_result.event_type != "NONE" and media_result.corroborated:
        delta = cfg.REACTIVE_DELTA.get(event_result.event_type, 0.50)
        return BucketResult(bucket=BucketType.REACTIVE, delta=delta)
    return BucketResult(bucket=BucketType.PERSONAL_DISTRESS, delta=1.0)


# ---------------------------------------------------------------------------
# Component 11 – Crisis Scoring
# ---------------------------------------------------------------------------

def _rural_adjacency_weight(region: RegionSignal) -> float:
    """
    Rural adjacency weighting: rural regions have fewer posts per-capita,
    so signals from them are upweighted to compensate for platform under-
    representation (geographic density bias correction).

    weight = 1.0 + max(0, 1 - pop_density / DENSITY_REF)
    - Dense urban region (pop_density >= DENSITY_REF) → weight ≈ 1.0
    - Sparse rural region (pop_density → 0)            → weight → 2.0
    """
    weight = 1.0 + max(0.0, 1.0 - region.pop_density / cfg.DENSITY_REF)
    return weight


def crisis_score(
    region: RegionSignal,
    bucket_result: BucketResult,
    sentiment_intensity_agg: float,
    volume_spike_score: float,
    geo_cluster_score: float,
) -> CrisisScoreResult:
    """
    Combine pre-made model outputs into a single weighted crisis score.

    Parameters
    ----------
    region                   : RegionSignal (used for rural adjacency weighting)
    bucket_result            : from bucket_signal()
    sentiment_intensity_agg  : float [0,1] from pre-made aggregate sentiment model
    volume_spike_score       : float [0,1] from pre-made volume spike detector
    geo_cluster_score        : float [0,1] from pre-made geographic clustering model

    Returns
    -------
    CrisisScoreResult with crisis_raw, crisis_adj (after bucket dampening).
    """
    # Weighted combination of pre-made scores
    crisis_raw = (
        cfg.W_S * sentiment_intensity_agg
        + cfg.W_V * volume_spike_score
        + cfg.W_G * geo_cluster_score
    )

    # Rural adjacency bias correction (per proposal diagram)
    rural_adj = _rural_adjacency_weight(region)
    crisis_raw = crisis_raw * rural_adj

    # Apply bucket dampening weight
    crisis_adj = crisis_raw * bucket_result.delta

    # Clamp to [0, 1]
    crisis_adj = min(1.0, crisis_adj)
    crisis_raw = min(1.0, crisis_raw)

    return CrisisScoreResult(
        crisis_raw=crisis_raw,
        crisis_adj=crisis_adj,
        sentiment_score=sentiment_intensity_agg,
        volume_score=volume_spike_score,
        geo_score=geo_cluster_score,
        bucket_delta=bucket_result.delta,
    )


# ---------------------------------------------------------------------------
# Component 12 – Confidence-modulated Threshold
# ---------------------------------------------------------------------------

def _trend_velocity(score_history: List[float]) -> float:
    """
    Early warning: compute the recent rate of change (velocity) of crisis scores.

    Uses simple linear regression slope over the history window.
    A positive velocity indicates an accelerating signal.
    """
    n = len(score_history)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(score_history)
    numerator   = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(score_history))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confidence_threshold(
    crisis_adj: float,
    region_conf: ConfidenceResult,
    score_history: Optional[List[float]] = None,
) -> ConfidenceThresholdResult:
    """
    Raise the escalation threshold when data confidence is low.

    Trend velocity early warning (per proposal diagram): if the crisis score
    is accelerating (positive velocity), lower the effective threshold by a
    small factor to allow earlier detection.

    effective_threshold = BASE * (1 + (1 - conf) * PENALTY_FACTOR)
                          * (1 - velocity_discount)
    """
    penalty = (1.0 - region_conf.conf) * cfg.PENALTY_FACTOR
    effective_threshold = cfg.BASE_CRISIS_THRESHOLD * (1.0 + penalty)

    # Trend velocity early warning
    velocity_discount = 0.0
    if score_history and len(score_history) >= 2:
        velocity = _trend_velocity(score_history)
        # Only discount (lower threshold) for accelerating signals
        if velocity > 0:
            velocity_discount = min(0.10, velocity)   # cap at 10% discount
    effective_threshold = max(0.10, effective_threshold * (1.0 - velocity_discount))

    exceeds = crisis_adj > effective_threshold

    return ConfidenceThresholdResult(
        exceeds=exceeds,
        crisis_adj=crisis_adj,
        effective_threshold=effective_threshold,
        confidence_used=region_conf.conf,
        penalty_applied=penalty,
    )


# ---------------------------------------------------------------------------
# Component 13 – Confirmation Window
# ---------------------------------------------------------------------------

def confirmation_window(
    region: RegionSignal,
    threshold_result: ConfidenceThresholdResult,
    recomputed_crisis_score: Optional[float] = None,
    now: Optional[datetime] = None,
    confirm_hours: float = cfg.CONFIRM_HOURS,
) -> ConfirmationWindowResult:
    """
    Require the signal to persist above threshold for CONFIRM_HOURS before escalating.

    The proposal diagram shows a 24-hour window; the pseudocode uses 6 hours.
    The default here uses CONFIRM_HOURS from config (set to 24.0 to match the
    proposal diagram; adjust in config.py as needed).

    Parameters
    ----------
    recomputed_crisis_score : if provided, used to re-check whether the signal
                              still exceeds the threshold at window end.
    """
    if now is None:
        now = datetime.utcnow()

    if not threshold_result.exceeds:
        _triggers.pop(region.region_id, None)
        return ConfirmationWindowResult(confirmed=False, action=AggregateAction.HOLD)

    # Record trigger if not already set
    if region.region_id not in _triggers:
        _triggers[region.region_id] = now

    trigger_time = _triggers[region.region_id]
    elapsed_hours = (now - trigger_time).total_seconds() / 3600

    if elapsed_hours < confirm_hours:
        remaining = confirm_hours - elapsed_hours
        return ConfirmationWindowResult(
            confirmed=False,
            action=AggregateAction.MONITORING,
            hours_remaining=remaining,
        )

    # Window elapsed — re-check signal
    current_score = recomputed_crisis_score if recomputed_crisis_score is not None \
                    else threshold_result.crisis_adj
    still_above = current_score > threshold_result.effective_threshold

    _triggers.pop(region.region_id, None)   # clear trigger regardless of outcome

    if still_above:
        return ConfirmationWindowResult(confirmed=True, action=AggregateAction.ESCALATE)
    else:
        return ConfirmationWindowResult(confirmed=False, action=AggregateAction.FLAG)


# ---------------------------------------------------------------------------
# Component 14 – Escalation Decision
# ---------------------------------------------------------------------------

def escalation_decision(
    region: RegionSignal,
    confirmation_result: ConfirmationWindowResult,
    crisis_adj: float,
    evidence_bundle: Optional[Dict[str, Any]] = None,
) -> EscalationResult:
    """
    Final routing: translate the confirmation result into HOLD / FLAG / ESCALATE.

    In production:
    - ESCALATE creates a review ticket, notifies on-call team, and attaches
      an evidence bundle (post sample, news articles, scores, timeline, geo map).
    - FLAG adds the region to a watch-list for the next processing cycle.
    - HOLD logs and continues passive monitoring.
    """
    action = confirmation_result.action

    if action == AggregateAction.ESCALATE:
        # Production: create_review_ticket() + notify_on_call_team()
        pass
    elif action == AggregateAction.FLAG:
        # Production: add_to_watchlist(region.region_id, ttl=next_cycle)
        pass
    # HOLD / MONITORING: log only

    return EscalationResult(
        action=action,
        region_id=region.region_id,
        crisis_score=crisis_adj,
    )


# ---------------------------------------------------------------------------
# Convenience: run the full aggregate track for one region
# ---------------------------------------------------------------------------

def run_aggregate_track(
    region: RegionSignal,
    sentiment_intensity_agg: float,
    volume_spike_score: float,
    geo_cluster_score: float,
    region_conf: ConfidenceResult,
    news_articles: Optional[List[Dict[str, Any]]] = None,
    volume_ratio: float = 1.0,
    geo_concentration: float = 0.0,
    multi_region_spike: bool = False,
    score_history: Optional[List[float]] = None,
    recomputed_crisis_score: Optional[float] = None,
    now: Optional[datetime] = None,
) -> dict:
    """
    Execute the full aggregate track pipeline for one region.
    Returns a dict with all intermediate and final results.
    """
    evt_result  = classify_event(region, volume_ratio=volume_ratio,
                                 geo_concentration=geo_concentration,
                                 multi_region_spike=multi_region_spike)
    med_result  = media_corroboration(region, evt_result.event_type, news_articles)
    bkt_result  = bucket_signal(evt_result, med_result)
    crs_result  = crisis_score(region, bkt_result,
                               sentiment_intensity_agg, volume_spike_score, geo_cluster_score)
    thr_result  = confidence_threshold(crs_result.crisis_adj, region_conf, score_history)
    win_result  = confirmation_window(region, thr_result,
                                      recomputed_crisis_score=recomputed_crisis_score, now=now)
    esc_result  = escalation_decision(region, win_result, crs_result.crisis_adj)

    return {
        "event_result":      evt_result,
        "media_result":      med_result,
        "bucket_result":     bkt_result,
        "crisis_result":     crs_result,
        "threshold_result":  thr_result,
        "window_result":     win_result,
        "escalation_result": esc_result,
    }
