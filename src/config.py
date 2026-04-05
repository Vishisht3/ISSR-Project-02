"""
Pipeline configuration parameters.
All tunable constants live here. Adjust and review quarterly
in consultation with the clinical advisory board.
"""

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Bot / Coordination Filter
# ---------------------------------------------------------------------------
MIN_ACCOUNT_AGE_DAYS: int = 7          # accounts younger than this trigger freq check
MAX_POSTS_PER_HOUR: int = 30           # max allowed posts/hour for young accounts
ENTROPY_FLOOR: float = 2.5            # bits; below this = template/spam
COORD_WINDOW_MINUTES: int = 10        # time window for coordination detection
COORD_K: int = 5                       # min distinct accounts for coordinated burst

# ---------------------------------------------------------------------------
# Minimum Sample Size Check
# ---------------------------------------------------------------------------
BASE_MIN_N: int = 30                   # baseline minimum posts required per region
DENSITY_REF: float = 1000.0           # reference pop/km² used as log normaliser

# ---------------------------------------------------------------------------
# Smoothing / Normalisation
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS: int = 7

# Calendar dampening: key = "MM-DD", value = dampening factor alpha in (0, 1)
# Reviewed quarterly by clinical advisory board
CALENDAR_EVENTS: Dict[str, dict] = {
    "01-01": {"name": "New Year's Day",                     "alpha": 0.60},
    "02-14": {"name": "Valentine's Day",                    "alpha": 0.65},
    "09-10": {"name": "World Suicide Prevention Day",       "alpha": 0.40},
    "10-10": {"name": "World Mental Health Day",            "alpha": 0.40},
    "11-11": {"name": "Veterans Day / Remembrance Day",     "alpha": 0.55},
    "12-25": {"name": "Christmas Day",                      "alpha": 0.60},
    "12-31": {"name": "New Year's Eve",                     "alpha": 0.60},
}

# ---------------------------------------------------------------------------
# CI / Uncertainty Estimate
# ---------------------------------------------------------------------------
Z_SCORE: float = 1.96                 # 95% confidence level

# ---------------------------------------------------------------------------
# Individual Track – Severity Scoring
# ---------------------------------------------------------------------------
W_SENTIMENT: float = 0.65             # weight for sentiment intensity component
W_ISOLATION: float = 0.35             # weight for isolation signal component

# Isolation signal sub-weights (must sum to 1.0 after normalisation)
# Order: [late_night, zero_replies, alone_keywords, inactive_graph]
ISO_WEIGHTS: list = [0.15, 0.30, 0.35, 0.20]

ALONE_LEXICON: list = [
    "no one", "alone", "nobody cares", "all alone", "by myself",
    "no friends", "nobody", "isolated", "no one listens", "invisible",
    "unwanted", "forgotten", "left out", "no support", "on my own",
]

INACTIVITY_THRESHOLD: float = 1.0     # avg daily interactions below this = inactive

# ---------------------------------------------------------------------------
# Individual Track – First-time Poster Flag
# ---------------------------------------------------------------------------
FIRST_TIME_BOOST: float = 1.25        # 25% severity uplift for first-time posters

# ---------------------------------------------------------------------------
# Individual Track – Threshold Decision
# ---------------------------------------------------------------------------
THETA_LOW: float = 0.30               # below → passive resource surfacing
THETA_HIGH: float = 0.70              # above → direct human review

# ---------------------------------------------------------------------------
# Aggregate Track – Event Classification
# ---------------------------------------------------------------------------

# Per-tier minimum match score to be considered a match
TIER_MIN_THRESHOLD: float = 0.30

# Event weights: how much each event type amplifies / dampens the crisis signal
TIER_WEIGHTS: Dict[str, float] = {
    "CALENDAR":      0.30,
    "CELEBRITY":     0.45,
    "LOCAL_MH":      0.90,
    "SOCIOECONOMIC": 0.70,
    "GLOBAL":        0.70,
    "NONE":          1.00,
}

# Keywords associated with each event tier (used in classification & media lookup)
EVENT_KEYWORDS: Dict[str, list] = {
    "CALENDAR": [
        "holiday", "anniversary", "awareness day", "remembrance",
    ],
    "CELEBRITY": [
        "celebrity", "famous", "death", "died", "arrest", "scandal",
        "suicide", "overdose", "passed away",
    ],
    "LOCAL_MH": [
        "school", "campus", "workplace", "community", "neighbourhood",
        "cluster", "tragedy", "crisis",
    ],
    "SOCIOECONOMIC": [
        "layoff", "unemployment", "rent", "eviction", "cost of living",
        "food bank", "bankruptcy", "debt", "poverty",
    ],
    "GLOBAL": [
        "pandemic", "war", "conflict", "earthquake", "flood",
        "disaster", "refugee", "outbreak",
    ],
}

# ---------------------------------------------------------------------------
# Aggregate Track – Media Corroboration
# ---------------------------------------------------------------------------
NEWS_WINDOW_HOURS: int = 24           # ±24 h around signal window
LAG_SIGMA: float = 12.0               # hours; Gaussian decay std for timing score
CORR_THRESHOLD: float = 0.45         # corroboration score above this = confirmed

# ---------------------------------------------------------------------------
# Aggregate Track – Reactive vs Personal Bucket
# ---------------------------------------------------------------------------
REACTIVE_DELTA: Dict[str, float] = {
    "CALENDAR":      0.30,
    "CELEBRITY":     0.45,
    "LOCAL_MH":      0.70,
    "SOCIOECONOMIC": 0.60,
    "GLOBAL":        0.50,
    "NONE":          1.00,
}

# ---------------------------------------------------------------------------
# Aggregate Track – Crisis Scoring
# ---------------------------------------------------------------------------
W_S: float = 0.40                    # sentiment intensity aggregate weight
W_V: float = 0.35                    # volume spike weight
W_G: float = 0.25                    # geographic clustering weight

# ---------------------------------------------------------------------------
# Aggregate Track – Confidence-modulated Threshold
# ---------------------------------------------------------------------------
BASE_CRISIS_THRESHOLD: float = 0.55  # base escalation threshold
PENALTY_FACTOR: float = 0.50         # scales confidence penalty

# ---------------------------------------------------------------------------
# Aggregate Track – Confirmation Window
# ---------------------------------------------------------------------------
CONFIRM_HOURS: float = 24.0          # signal must persist this many hours (proposal diagram: 24hr)

# ---------------------------------------------------------------------------
# Contagion Flagging
# ---------------------------------------------------------------------------
# Keywords indicating suicide-related content in posts or news
SUICIDE_KEYWORDS: list = [
    "suicide", "suicidal", "took their own life", "took his own life",
    "took her own life", "died by suicide", "killed himself", "killed herself",
    "killed themselves", "self-inflicted", "overdose", "hanging",
]

# Minimum keyword overlap score between post corpus and suicide news to flag contagion
CONTAGION_CORR_THRESHOLD: float = 0.40

# Lag window: news must appear within this many hours before the post spike
CONTAGION_NEWS_WINDOW_HOURS: int = 48

# ---------------------------------------------------------------------------
# Bias Mitigation – Platform Underrepresentation
# ---------------------------------------------------------------------------
# Platform weight = expected_population_share / actual_platform_share per platform.
# Reviewed and updated quarterly against current platform demographic data.
PLATFORM_WEIGHTS: dict = {
    "twitter": 1.20,   # over-represents 18-29, urban, college-educated
    "reddit":  1.35,   # over-represents male, 18-34, tech-adjacent
    "default": 1.00,
}

# ---------------------------------------------------------------------------
# Bias Mitigation – Intervention Bias (Low-Engagement Users)
# ---------------------------------------------------------------------------
# Users below this follower / interaction threshold are less likely to see
# platform-surfaced resources (banners, overlays). Severity threshold is
# lowered so they escalate sooner to direct outreach.
LOW_ENGAGEMENT_INTERACTION_THRESHOLD: float = 2.0   # avg daily interactions
LOW_ENGAGEMENT_SEVERITY_DISCOUNT: float = 0.05       # subtract from THETA_LOW and THETA_HIGH

# ---------------------------------------------------------------------------
# Heatmap Output
# ---------------------------------------------------------------------------
ESC_CAP: int = 10                    # escalation count cap for colour normalisation
HOTSPOT_MIN: float = 0.60            # combined score above this → annotate hotspot
DASHBOARD_URL: str = "/api/heatmap/live"
