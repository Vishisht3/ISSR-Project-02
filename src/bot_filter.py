"""
Component 1 – Bot / Coordination Filter

Purpose: Remove automated, spam, and coordinated-inauthentic posts before
they pollute downstream signals. This is a mandatory quality gate; no post
reaches any scoring stage without passing this filter.

References
----------
Gera & Ciampaglia (2022): tweet-health correlations disappear entirely
without bot removal.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import src.config as cfg
from src.models import BotFilterAction, BotFilterResult, Post


# ---------------------------------------------------------------------------
# Fingerprint cache: {fingerprint: [(account_id, timestamp), ...]}
# ---------------------------------------------------------------------------
_fingerprint_cache: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string in bits."""
    if not text:
        return 0.0
    freq: Dict[str, int] = defaultdict(int)
    for ch in text:
        freq[ch] += 1
    n = len(text)
    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values())
    return entropy


def _normalise_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for fingerprinting."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_fingerprint(text: str) -> str:
    normalised = _normalise_text(text)
    return hashlib.md5(normalised.encode()).hexdigest()


def _lookup_fingerprint(fingerprint: str, window_minutes: int, now: datetime) -> List[str]:
    """Return list of distinct account_ids that posted this fingerprint within the window."""
    cutoff = now - timedelta(minutes=window_minutes)
    entries = _fingerprint_cache.get(fingerprint, [])
    recent = [account_id for account_id, ts in entries if ts >= cutoff]
    return recent


def _store_fingerprint(fingerprint: str, account_id: str, timestamp: datetime) -> None:
    _fingerprint_cache[fingerprint].append((account_id, timestamp))


def _flag_coordinated(fingerprint: str, window_minutes: int, now: datetime) -> None:
    """Mark all recent entries for this fingerprint as coordinated."""
    pass


def bot_filter(post: Post, now: Optional[datetime] = None) -> BotFilterResult:
    """
    Run bot / coordination checks on a single post.

    Returns BotFilterResult with action=PASS or action=DISCARD.
    """
    if now is None:
        now = datetime.utcnow()

    # ------------------------------------------------------------------
    # Rule 1: Young account + high posting frequency
    # ------------------------------------------------------------------
    if post.account_age_days < cfg.MIN_ACCOUNT_AGE_DAYS:
        if post.posts_last_hour > cfg.MAX_POSTS_PER_HOUR:
            return BotFilterResult(
                action=BotFilterAction.DISCARD,
                reason="young_account_high_freq",
            )

    # ------------------------------------------------------------------
    # Rule 2: Text entropy check (catches template / copy-paste spam)
    # ------------------------------------------------------------------
    entropy = _shannon_entropy(post.text)
    if entropy < cfg.ENTROPY_FLOOR:
        return BotFilterResult(
            action=BotFilterAction.DISCARD,
            reason="low_entropy",
        )

    # ------------------------------------------------------------------
    # Rule 3: Coordination detection
    # ------------------------------------------------------------------
    fingerprint = _make_fingerprint(post.text)
    recent_accounts = _lookup_fingerprint(fingerprint, cfg.COORD_WINDOW_MINUTES, now)
    distinct_accounts = list(set(recent_accounts))

    if len(distinct_accounts) >= cfg.COORD_K:
        _flag_coordinated(fingerprint, cfg.COORD_WINDOW_MINUTES, now)
        return BotFilterResult(
            action=BotFilterAction.DISCARD,
            reason="coordinated_burst",
        )

    # Store fingerprint for future coordination checks
    _store_fingerprint(fingerprint, post.account_id, now)

    return BotFilterResult(action=BotFilterAction.PASS)


def filter_posts(posts: List[Post], now: Optional[datetime] = None) -> List[Post]:
    """
    Apply bot_filter to a batch of posts.
    Returns only posts that pass the filter.
    """
    if now is None:
        now = datetime.utcnow()
    return [p for p in posts if bot_filter(p, now).action == BotFilterAction.PASS]
