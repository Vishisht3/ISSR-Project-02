"""
ISSR Adapter — Yixing Fan Pipeline → Vishisht Magan Framework

This module bridges the data produced by the existing ISSR prototype
(ISSR_AI4MH_Yixing_Fan/pipeline) with the governance framework in this
repository (ISSR Project 02).

It converts:
  • Reddit post rows (dict / DataFrame row) → Post dataclasses
  • Groups of Reddit posts grouped by subreddit → RegionSignal dataclasses
  • Yixing's VADER / BERT model callables → framework-compatible sentiment callables

Nothing in the ISSR_AI4MH_Yixing_Fan folder is modified. All bridging logic
lives here so the two codebases remain independently maintainable.

─────────────────────────────────────────────────────────────────────────────
NOTE ON ACCOUNT METADATA
─────────────────────────────────────────────────────────────────────────────
Instead of fetching just Reddit data, we can collect other metadata from a
user account within privacy laws to further help people — for example:
account creation date (available via Reddit's public API with no login),
posting frequency in the past hour, average daily interactions (upvotes +
comments), and first-post status. Richer signals such as follower graphs,
cross-platform activity, or demographic information must only be collected
with explicit informed consent and in full compliance with applicable
data-protection regulations (e.g. GDPR, CCPA, HIPAA where relevant).
The field defaults below reflect what is safely collectable today without
additional consent; fields marked "# REQUIRES CONSENT" must not be populated
without a proper consent flow in place.
─────────────────────────────────────────────────────────────────────────────

References
----------
Yixing Fan, ISSR AI4MH prototype:
    pipeline/fetch_reddit.py      — PRAW-based Reddit ingestion
    pipeline/sentiment_risk_classifier.py — VADER + keyword risk levels
    pipeline/predict.py           — fine-tuned BERT suicidal ideation classifier
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models import IndividualAction, Post, RegionSignal


# ---------------------------------------------------------------------------
# Risk level bridge — Yixing's classifier → Vishisht's action enum
# ---------------------------------------------------------------------------
# Yixing's sentiment_risk_classifier.py outputs three string labels:
#   "High-Risk"       → maps to HUMAN_REVIEW   (route to on-call clinician)
#   "Moderate Concern"→ maps to HELPLINE_PROMPT (surface anonymous helpline)
#   "Low Concern"     → maps to PASSIVE_RESOURCE(banner / sidebar link)
#
# This mapping is the canonical translation point between the two systems.
# If Yixing's taxonomy changes, update only here.

_RISK_TO_ACTION: Dict[str, IndividualAction] = {
    "High-Risk":        IndividualAction.HUMAN_REVIEW,
    "Moderate Concern": IndividualAction.HELPLINE_PROMPT,
    "Low Concern":      IndividualAction.PASSIVE_RESOURCE,
}


def risk_level_to_action(risk_level: str) -> IndividualAction:
    """
    Convert Yixing's keyword-based risk label to the framework's IndividualAction.

    Parameters
    ----------
    risk_level : one of "High-Risk", "Moderate Concern", "Low Concern"
                 as produced by sentiment_risk_classifier.py.

    Returns
    -------
    The corresponding IndividualAction. Unknown labels fall back to
    PASSIVE_RESOURCE (safest default — never silently drops a signal).
    """
    return _RISK_TO_ACTION.get(risk_level, IndividualAction.PASSIVE_RESOURCE)


# ---------------------------------------------------------------------------
# Subreddit → geographic proxy
# ---------------------------------------------------------------------------
# Reddit posts carry no latitude / longitude. The mapping below assigns a
# representative centroid and population figure to each subreddit so the
# aggregate track can compute population-density-weighted signals. Values
# are broad US / global estimates, not authoritative census data. Replace
# with real geodata if the upstream pipeline gains location tagging.

_SUBREDDIT_GEO: Dict[str, Dict[str, Any]] = {
    # subreddit          lat      lon     population   pop_density (per km²)
    "depression":       {"lat": 37.09, "lon": -95.71, "pop": 5_000_000, "density": 400},
    "SuicideWatch":     {"lat": 37.09, "lon": -95.71, "pop": 3_000_000, "density": 400},
    "mentalhealth":     {"lat": 37.09, "lon": -95.71, "pop": 4_000_000, "density": 400},
    "Anxiety":          {"lat": 37.09, "lon": -95.71, "pop": 4_500_000, "density": 400},
    "addiction":        {"lat": 37.09, "lon": -95.71, "pop": 2_000_000, "density": 400},
    "offmychest":       {"lat": 37.09, "lon": -95.71, "pop": 3_500_000, "density": 350},
    "PTSD":             {"lat": 37.09, "lon": -95.71, "pop": 1_500_000, "density": 400},
    "BPD":              {"lat": 37.09, "lon": -95.71, "pop": 1_000_000, "density": 400},
    "lonely":           {"lat": 37.09, "lon": -95.71, "pop": 2_500_000, "density": 350},
    "grief":            {"lat": 37.09, "lon": -95.71, "pop": 1_200_000, "density": 350},
    "bipolar":          {"lat": 37.09, "lon": -95.71, "pop": 1_800_000, "density": 400},
}

_DEFAULT_GEO = {"lat": 37.09, "lon": -95.71, "pop": 1_000_000, "density": 300}


def _geo_for(subreddit: str) -> Dict[str, Any]:
    return _SUBREDDIT_GEO.get(subreddit, _DEFAULT_GEO)


# ---------------------------------------------------------------------------
# Reddit row → Post
# ---------------------------------------------------------------------------

def reddit_row_to_post(row: Dict[str, Any]) -> Post:
    """
    Convert one row from filtered_reddit_posts.csv (or an equivalent dict
    produced by fetch_reddit.py) into a framework Post dataclass.

    Expected keys in *row*
    ----------------------
    id          : Reddit post ID string
    timestamp   : Unix UTC timestamp (float or int)
    subreddit   : subreddit name (used as region_id proxy)
    raw_text    : original post text (preferred for sentiment analysis)
    cleaned_text: preprocessed text (fallback if raw_text absent)
    upvotes     : post score / upvotes
    comments    : number of comments (used as reply_count)

    Optional keys (populated by an enhanced fetch — see NOTE above)
    ---------------------------------------------------------------
    author          : Reddit username string
    account_age_days: int — days since account was created
    posts_last_hour : int — posts by this author in the past hour
    is_first_post   : bool — True if this is the account's first known post
    """
    ts = float(row.get("timestamp", 0))
    post_dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
    geo = _geo_for(str(row.get("subreddit", "")))

    # Prefer raw_text for analysis; fall back to cleaned_text
    text = str(row.get("raw_text") or row.get("cleaned_text") or "")

    # avg_daily_interactions: use upvotes as a proxy.
    # A richer signal (upvotes + comments / account_age_days) is preferable
    # but requires the enhanced fetch described in the NOTE at the top of
    # this file.
    upvotes = float(row.get("upvotes", 1) or 1)
    comments = int(row.get("comments", 0) or 0)
    account_age = int(row.get("account_age_days", 365) or 365)
    avg_interactions = max(0.0, (upvotes + comments) / max(account_age, 1))

    return Post(
        post_id=str(row.get("id", "")),
        account_id=str(row.get("author", row.get("id", "unknown"))),
        text=text,
        timestamp=post_dt,
        region_id=str(row.get("subreddit", "unknown")),
        latitude=float(geo["lat"]),
        longitude=float(geo["lon"]),
        account_age_days=account_age,
        posts_last_hour=int(row.get("posts_last_hour", 1) or 1),
        reply_count=comments,
        avg_daily_interactions=avg_interactions,
        is_first_post=bool(row.get("is_first_post", False)),
        local_hour=post_dt.hour,
        raw_payload=dict(row),
    )


def dataframe_to_posts(df: Any) -> List[Post]:
    """
    Convert a pandas DataFrame (e.g. filtered_reddit_posts.csv) to a list
    of Post objects. Column names must match the expected keys in
    reddit_row_to_post().
    """
    return [reddit_row_to_post(row) for row in df.to_dict(orient="records")]


# ---------------------------------------------------------------------------
# Reddit posts grouped by subreddit → RegionSignal
# ---------------------------------------------------------------------------

def posts_to_region_signals(
    posts: List[Post],
    today: Optional[datetime] = None,
) -> Tuple[List[RegionSignal], datetime]:
    """
    Aggregate a flat list of Post objects into one RegionSignal per subreddit
    (region_id). The subreddit is used as the geographic region proxy.

    Reference date
    --------------
    When *today* is None (the default) the reference date is inferred from the
    most recent post timestamp in the batch rather than the wall-clock date.
    This is critical for historical datasets (e.g. Yixing's archived Reddit
    CSV) where post timestamps are weeks or years in the past — using the real
    wall-clock date would make every daily_counts lookup return 0 and cause the
    sample-size gate to block every region.

    Engagement-weighted counts
    --------------------------
    Each RegionSignal carries both raw daily_counts and
    engagement_weighted_counts. The latter replaces each post's contribution
    of 1 with log2(1 + upvotes + comments), so a post with 7 000 upvotes and
    200 comments counts as ≈13 effective samples. sample_size_check() uses
    the engagement-weighted total when it is available, allowing a small number
    of high-signal posts to satisfy the minimum-sample gate.

    Returns
    -------
    (signals, reference_date) — pass reference_date as the *now* argument to
    CrisisPipeline.ingest_regions() so the pipeline evaluates the correct day.
    """
    # Infer reference date from data if not supplied
    if today is None:
        latest_ts = max((p.timestamp for p in posts), default=datetime.utcnow())
        today = latest_ts

    today_key = today.strftime("%Y-%m-%d")

    # Group posts by region_id (= subreddit)
    by_region: Dict[str, List[Post]] = defaultdict(list)
    for p in posts:
        by_region[p.region_id].append(p)

    signals = []
    for region_id, region_posts in by_region.items():
        geo = _geo_for(region_id)

        # Build raw daily_counts from post timestamps
        daily_counts: Dict[str, int] = defaultdict(int)
        for p in region_posts:
            day_key = p.timestamp.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1

        # Build engagement-weighted daily counts.
        # Weight = log2(1 + upvotes + comments) so that highly engaged posts
        # count for more than silent ones, while a post with 0 engagement
        # still contributes log2(1) = 0 (the raw count guards provide a floor
        # of 1 per post via the max() in sample_size_check).
        engagement_weighted: Dict[str, float] = defaultdict(float)
        for p in region_posts:
            day_key = p.timestamp.strftime("%Y-%m-%d")
            upvotes  = float(p.raw_payload.get("upvotes",  0) or 0)
            comments = float(p.raw_payload.get("comments", 0) or 0)
            weight = math.log2(1.0 + upvotes + comments)
            # Floor at 1.0 so every post contributes at least 1 effective sample
            engagement_weighted[day_key] += max(1.0, weight)

        # Ensure today_key always has an entry
        if today_key not in daily_counts:
            daily_counts[today_key] = 0
        if today_key not in engagement_weighted:
            engagement_weighted[today_key] = 0.0

        # Detect spike start: most recent day that exceeded 1.5× rolling mean
        sorted_days = sorted(daily_counts.keys())
        counts_list = [daily_counts[d] for d in sorted_days]
        baseline = sum(counts_list[:-1]) / max(len(counts_list) - 1, 1)
        spike_start: Optional[datetime] = None
        if daily_counts.get(today_key, 0) > baseline * 1.5:
            spike_start = today.replace(hour=0, minute=0, second=0, microsecond=0)

        signals.append(RegionSignal(
            region_id=region_id,
            centroid_lat=float(geo["lat"]),
            centroid_lon=float(geo["lon"]),
            population=float(geo["pop"]),
            pop_density=float(geo["density"]),
            daily_counts=dict(daily_counts),
            engagement_weighted_counts=dict(engagement_weighted),
            post_corpus=[p.text for p in region_posts],
            spike_start_time=spike_start,
        ))

    return signals, today


# ---------------------------------------------------------------------------
# Sentiment model callables (framework-compatible wrappers)
# ---------------------------------------------------------------------------

def make_vader_sentiment_model() -> Callable[[str], float]:
    """
    Return a sentiment callable compatible with PipelineConfig.sentiment_model.

    Uses nltk's VADER (same as Yixing's sentiment_risk_classifier.py) so both
    pipelines share an identical underlying model. Maps the VADER compound
    score from [-1, 1] to a distress intensity in [0, 1]: 1 = maximum distress.
    """
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        return lambda text: max(0.0, -sia.polarity_scores(str(text))["compound"])
    except ImportError:
        return lambda text: 0.5  # neutral default if nltk is not installed


def make_bert_enhanced_sentiment_model(model_path: str = "results/final_model") -> Callable[[str], float]:
    """
    Return a sentiment callable that combines VADER with Yixing's fine-tuned
    BERT suicidal ideation classifier (pipeline/predict.py).

    When BERT classifies a post as 'Suicidal', the VADER distress score
    receives a significant uplift so that the individual track escalates
    more readily — reflecting the higher precision of the task-specific model.

    Parameters
    ----------
    model_path : path to the saved BERT model (default: results/final_model,
                 relative to the ISSR_AI4MH_Yixing_Fan directory).
    """
    vader_fn = make_vader_sentiment_model()

    try:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.eval()

        def bert_predict(text: str) -> str:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                padding=True, max_length=128,
            )
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            return "Suicidal" if pred == 1 else "Non-Suicidal"

        def combined(text: str) -> float:
            vader_score = vader_fn(text)
            if bert_predict(text) == "Suicidal":
                # BERT confirms ideation: uplift and clamp to [0, 1]
                return min(1.0, vader_score * 1.4 + 0.25)
            return vader_score

        return combined

    except Exception:
        # BERT unavailable (model not downloaded, no torch, etc.) — fall back
        # to VADER-only, which is still the same model Yixing uses.
        return vader_fn


def make_vader_agg_sentiment_model() -> Callable[[Any], float]:
    """
    Return an aggregate-region sentiment callable compatible with
    PipelineConfig.sentiment_agg_model. Averages VADER distress scores
    across all posts in the region's post_corpus.
    """
    vader_fn = make_vader_sentiment_model()

    def agg(region: Any) -> float:
        corpus = getattr(region, "post_corpus", [])
        if not corpus:
            return 0.0
        scores = [vader_fn(t) for t in corpus]
        return sum(scores) / len(scores)

    return agg


# ---------------------------------------------------------------------------
# Convenience: build a PipelineConfig wired to Yixing's models
# ---------------------------------------------------------------------------

def make_issr_pipeline_config(
    bert_model_path: Optional[str] = None,
    platform: str = "reddit",
) -> Any:
    """
    Return a PipelineConfig pre-wired to Yixing's VADER and (optionally)
    BERT models so that CrisisPipeline uses the same underlying classifiers
    as the existing ISSR prototype.

    Parameters
    ----------
    bert_model_path : path to Yixing's fine-tuned BERT model directory.
                      If None or the model cannot be loaded, falls back to
                      VADER-only sentiment (still identical to the prototype).
    platform        : "reddit" — applies the reddit platform-bias weight from
                      config.py (PLATFORM_WEIGHTS["reddit"] = 1.35).
    """
    from src.pipeline import PipelineConfig

    if bert_model_path:
        sentiment_fn = make_bert_enhanced_sentiment_model(bert_model_path)
    else:
        sentiment_fn = make_vader_sentiment_model()

    return PipelineConfig(
        sentiment_model=sentiment_fn,
        sentiment_agg_model=make_vader_agg_sentiment_model(),
        platform=platform,
    )
