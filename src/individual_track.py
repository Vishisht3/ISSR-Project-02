"""
Individual Track – Components 5, 6, 7

5. Severity Scoring
   Combines:
     - Sentiment intensity (VADER — pre-made, already in prototype)
     - Isolation signals  (late-night, zero-replies, alone-keywords, inactive-graph)
     - Ideation signals   (explicit ideation keywords — added per proposal diagram)
     - First-post critical flag

6. First-time Poster Flag
   Boosts severity by FIRST_TIME_BOOST for accounts with no prior post history.

7. Threshold Decision
   Maps final severity to one of three response tiers:
     - PASSIVE_RESOURCE : severity < THETA_LOW
     - HELPLINE_PROMPT  : THETA_LOW <= severity < THETA_HIGH
     - HUMAN_REVIEW     : severity >= THETA_HIGH

References
----------
Swaminathan et al. (2023): keyword filtering + logistic regression reduced
    triage time from 9 hours to 8-13 minutes.
"""

from __future__ import annotations

import re
from typing import List, Optional

import src.config as cfg
from src.models import (
    FirstTimeFlagResult,
    IndividualAction,
    IsolationSignals,
    Post,
    SeverityResult,
    ThresholdDecisionResult,
)

# ---------------------------------------------------------------------------
# Ideation keyword lexicon (explicit suicidal / self-harm ideation signals)
# Reviewed quarterly by clinical advisory board.
# ---------------------------------------------------------------------------
IDEATION_LEXICON: List[str] = [
    "want to die", "don't want to live", "end my life", "kill myself",
    "suicide", "suicidal", "self harm", "self-harm", "cutting myself",
    "no reason to live", "better off dead", "can't go on", "give up on life",
    "not worth living", "goodbye forever", "last post", "take my own life",
]

# Ideation weight: added on top of composite severity when ideation is detected
IDEATION_BOOST: float = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_match_score(text: str, lexicon: List[str]) -> float:
    """Return fraction of lexicon phrases found in text (0.0 – 1.0)."""
    if not lexicon:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for phrase in lexicon if phrase in text_lower)
    return hits / len(lexicon)


def _ideation_detected(text: str) -> bool:
    """Return True if any ideation keyword is found in text."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in IDEATION_LEXICON)


# ---------------------------------------------------------------------------
# Component 5 – Severity Scoring
# ---------------------------------------------------------------------------

def severity_score(post: Post, sentiment_intensity: float) -> SeverityResult:
    """
    Compute a composite severity score for an individual post.

    Parameters
    ----------
    post                : Post dataclass
    sentiment_intensity : float in [0, 1] from the pre-made VADER-based model
                          (1 = most intense negative sentiment)

    Returns
    -------
    SeverityResult with severity in [0, 1] (may be slightly above 1.0 due to
    ideation boost — callers should clamp before display).
    """
    # ---- Isolation signal extraction ----
    late_night     = 1 if post.local_hour in range(0, 6) else 0
    zero_replies   = 1 if post.reply_count == 0 else 0
    alone_keywords = _keyword_match_score(post.text, cfg.ALONE_LEXICON)
    inactive_graph = 1 if post.avg_daily_interactions < cfg.INACTIVITY_THRESHOLD else 0

    signals = IsolationSignals(
        late_night=late_night,
        zero_replies=zero_replies,
        alone_keywords=alone_keywords,
        inactive_graph=inactive_graph,
    )

    # Weighted dot product then normalise to [0, 1]
    signal_values = [late_night, zero_replies, alone_keywords, inactive_graph]
    raw_isolation = sum(w * s for w, s in zip(cfg.ISO_WEIGHTS, signal_values))
    isolation_score = raw_isolation / sum(cfg.ISO_WEIGHTS)

    # ---- Composite severity ----
    severity = cfg.W_SENTIMENT * sentiment_intensity + cfg.W_ISOLATION * isolation_score

    # ---- Ideation boost (per proposal architecture diagram) ----
    if _ideation_detected(post.text):
        severity = min(1.0, severity + IDEATION_BOOST)

    return SeverityResult(
        severity=severity,
        sentiment_raw=sentiment_intensity,
        isolation_raw=isolation_score,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Component 6 – First-time Poster Flag
# ---------------------------------------------------------------------------

def first_time_flag(post: Post, severity: float, prior_post_count: int) -> FirstTimeFlagResult:
    """
    Boost severity for first-time posters.

    Parameters
    ----------
    post             : Post dataclass
    severity         : current severity score
    prior_post_count : number of prior posts from this account in the dataset

    Returns
    -------
    FirstTimeFlagResult with adjusted_severity clamped to [0, 1].
    """
    if prior_post_count == 0:
        adjusted = min(1.0, severity * cfg.FIRST_TIME_BOOST)
        return FirstTimeFlagResult(first_time=True, adjusted_severity=adjusted)
    return FirstTimeFlagResult(first_time=False, adjusted_severity=severity)


# ---------------------------------------------------------------------------
# Component 7 – Threshold Decision
# ---------------------------------------------------------------------------

def threshold_decision(severity: float) -> ThresholdDecisionResult:
    """
    Map a final severity score to one of three response tiers.

    Tier boundaries
    ---------------
    severity < THETA_LOW  → PASSIVE_RESOURCE  (banner / sidebar link)
    THETA_LOW <= s < HIGH → HELPLINE_PROMPT   (non-intrusive overlay)
    severity >= THETA_HIGH → HUMAN_REVIEW     (immediate moderator queue)
    """
    if severity < cfg.THETA_LOW:
        action = IndividualAction.PASSIVE_RESOURCE
    elif severity < cfg.THETA_HIGH:
        action = IndividualAction.HELPLINE_PROMPT
    else:
        action = IndividualAction.HUMAN_REVIEW

    return ThresholdDecisionResult(
        action=action,
        severity=severity,
        theta_low=cfg.THETA_LOW,
        theta_high=cfg.THETA_HIGH,
    )


# ---------------------------------------------------------------------------
# Convenience: run the full individual track for one post
# ---------------------------------------------------------------------------

def run_individual_track(
    post: Post,
    sentiment_intensity: float,
    prior_post_count: int,
) -> dict:
    """
    Execute severity scoring → first-time flag → threshold decision in sequence.

    Returns a dict with all intermediate and final results.
    """
    sev_result   = severity_score(post, sentiment_intensity)
    flag_result  = first_time_flag(post, sev_result.severity, prior_post_count)
    thr_result   = threshold_decision(flag_result.adjusted_severity)

    return {
        "severity_result":   sev_result,
        "first_time_result": flag_result,
        "threshold_result":  thr_result,
    }
