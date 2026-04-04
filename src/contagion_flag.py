"""
Contagion Flagging — Suicide News Co-occurrence Detection

Explicitly listed in the proposal Key Points:
  "contagion flagging for suicide news co-occurence"

And in the coding plan (Week 6–7):
  "Implement reactive vs. personal distress bucketing and contagion flagging pathway"

Background — Werther Effect
----------------------------
Research consistently shows that media coverage of suicide can trigger
imitative suicidal behaviour in vulnerable individuals, particularly when
coverage is prominent, detailed, or involves a celebrity (Niederkrotenthaler
et al., 2010). A spike in suicide-related social media posts that co-occurs
with news coverage of a suicide is qualitatively different from a spike
driven purely by personal distress — it requires a distinct response pathway
that accounts for possible contagion spread rather than treating it as a
simple external event to be dampened.

What this module does
---------------------
1. Checks whether the post corpus for a region contains a high density of
   suicide-specific keywords (not just general distress).
2. Checks whether local/national news within the preceding 48 hours contains
   suicide-specific coverage.
3. If both are present (co-occurrence), raises a ContagionFlag with a
   severity grade.

How it integrates into the pipeline
------------------------------------
- Called from the aggregate track after media corroboration, before or
  alongside the reactive/personal bucket decision.
- When a contagion flag is raised:
    * The signal is NOT dampened as a normal "reactive" event — the
      concern is that the media is amplifying distress, not explaining it away.
    * The bucket delta is forced to 1.0 (full weight) regardless of event type.
    * An additional contagion_flag field is written to the audit record.
    * The escalation decision is biased toward ESCALATE even at lower crisis
      scores (threshold reduced by CONTAGION_THRESHOLD_DISCOUNT).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import src.config as cfg
from src.models import RegionSignal


# ---------------------------------------------------------------------------
# Threshold discount when contagion is flagged
# ---------------------------------------------------------------------------
CONTAGION_THRESHOLD_DISCOUNT: float = 0.15   # lower effective threshold by this amount


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ContagionFlagResult:
    flagged: bool
    grade: str                    # "NONE" | "LOW" | "MEDIUM" | "HIGH"
    post_suicide_density: float   # fraction of posts containing suicide keywords
    news_suicide_score: float     # keyword overlap between news and suicide lexicon
    co_occurrence_score: float    # combined score
    threshold_discount: float     # how much to lower escalation threshold (0 if not flagged)
    details: str                  # human-readable explanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suicide_density(texts: List[str]) -> float:
    """
    Fraction of posts containing at least one suicide keyword.
    Returns a value in [0.0, 1.0].
    """
    if not texts:
        return 0.0
    hits = sum(
        1 for t in texts
        if any(kw in t.lower() for kw in cfg.SUICIDE_KEYWORDS)
    )
    return hits / len(texts)


def _news_suicide_score(news_articles: List[Dict[str, Any]]) -> float:
    """
    Compute how heavily suicide-related the news corpus is.
    Returns a value in [0.0, 1.0] using keyword hit-rate in headlines + summaries.
    """
    if not news_articles:
        return 0.0
    kw_set = set(cfg.SUICIDE_KEYWORDS)
    total_hits = 0
    for article in news_articles:
        text = (article.get("headline", "") + " " + article.get("summary", "")).lower()
        total_hits += sum(1 for kw in kw_set if kw in text)
    # Normalise by number of articles × number of keywords
    max_possible = len(news_articles) * len(kw_set)
    return min(1.0, total_hits / max_possible) if max_possible > 0 else 0.0


def _within_window(
    news_articles: List[Dict[str, Any]],
    spike_time: datetime,
    window_hours: int,
) -> List[Dict[str, Any]]:
    """Filter articles to those published within window_hours before the spike."""
    cutoff = spike_time - timedelta(hours=window_hours)
    return [
        a for a in news_articles
        if isinstance(a.get("published_at"), datetime)
        and cutoff <= a["published_at"] <= spike_time
    ]


def _grade(score: float) -> str:
    if score < 0.25:
        return "NONE"
    if score < 0.45:
        return "LOW"
    if score < 0.65:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_contagion(
    region: RegionSignal,
    news_articles: Optional[List[Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> ContagionFlagResult:
    """
    Detect suicide news co-occurrence (contagion risk) for a region.

    Parameters
    ----------
    region        : RegionSignal with post_corpus and spike_start_time
    news_articles : articles returned by the news API (already fetched for
                    media corroboration — reuse the same list)
    now           : override current time for testing

    Returns
    -------
    ContagionFlagResult. If flagged=True, the caller must:
      - Force bucket delta to 1.0 (do NOT dampen as a normal reactive event)
      - Reduce the effective escalation threshold by threshold_discount
      - Write contagion_flag=True to the audit record
    """
    if now is None:
        now = datetime.utcnow()

    # 1. Post-side: suicide keyword density in the post corpus
    post_density = _suicide_density(region.post_corpus)

    # 2. News-side: suicide keyword score in recent news
    spike_time = region.spike_start_time or now
    recent_news = _within_window(
        news_articles or [], spike_time, cfg.CONTAGION_NEWS_WINDOW_HOURS
    )
    news_score = _news_suicide_score(recent_news)

    # 3. Co-occurrence score: geometric mean of both signals
    #    (both must be present — neither alone is sufficient)
    co_score = math.sqrt(post_density * news_score)

    flagged = co_score >= cfg.CONTAGION_CORR_THRESHOLD
    grade   = _grade(co_score)

    if flagged:
        details = (
            f"Contagion risk detected (grade={grade}): "
            f"{post_density*100:.1f}% of posts contain suicide keywords; "
            f"news suicide score={news_score:.3f} across {len(recent_news)} article(s) "
            f"within {cfg.CONTAGION_NEWS_WINDOW_HOURS}h of spike. "
            "Bucket dampening suppressed. Escalation threshold lowered."
        )
        discount = CONTAGION_THRESHOLD_DISCOUNT
    else:
        details = (
            f"No contagion risk (co_occurrence_score={co_score:.3f} "
            f"< threshold {cfg.CONTAGION_CORR_THRESHOLD})."
        )
        discount = 0.0

    return ContagionFlagResult(
        flagged=flagged,
        grade=grade,
        post_suicide_density=post_density,
        news_suicide_score=news_score,
        co_occurrence_score=co_score,
        threshold_discount=discount,
        details=details,
    )
