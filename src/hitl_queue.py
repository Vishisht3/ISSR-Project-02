"""
Human-in-the-Loop (HITL) Review Queue

Implements the single most important safeguard stated in the proposal:

    "No automated action that surfaces resources to, flags, or escalates an
     individual should execute without human review. The system recommends;
     humans decide."

     — Proposal, Governance Reflection section

How it works
------------
When the pipeline produces a HUMAN_REVIEW (individual track) or ESCALATE
(aggregate track) decision, the output is NOT immediately acted upon.
Instead it is placed in a pending queue. The automated system is blocked
from taking the action until a human reviewer either:

  - APPROVES  → action proceeds as recommended
  - OVERRIDES → reviewer substitutes a different action
  - DISMISSES → reviewer determines no action is needed

Every review decision is written to the audit log, creating a complete
human-decision trail at every escalation threshold.

Lower tiers (PASSIVE_RESOURCE, HELPLINE_PROMPT, HOLD, FLAG) do not require
HITL review and are executed immediately — they are non-intrusive and
reversible. Only actions that directly surface something to or escalate a
specific individual / community require human sign-off.

Daruna (2026) formalises four HITL escalation patterns; this implementation
follows the blocking-queue pattern.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReviewDecision(str, Enum):
    APPROVED  = "APPROVED"
    OVERRIDDEN = "OVERRIDDEN"
    DISMISSED  = "DISMISSED"


class QueueItemStatus(str, Enum):
    PENDING   = "PENDING"
    REVIEWED  = "REVIEWED"
    EXPIRED   = "EXPIRED"


class QueueTrack(str, Enum):
    INDIVIDUAL = "INDIVIDUAL"
    AGGREGATE  = "AGGREGATE"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QueueItem:
    """A single item awaiting human review."""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    track: Optional[QueueTrack] = None
    region_id: Optional[str] = None
    recommended_action: str = ""          # what the pipeline recommended
    severity_score: Optional[float] = None
    crisis_score: Optional[float] = None
    confidence: Optional[float] = None
    event_type: Optional[str] = None
    contagion_flagged: bool = False
    status: QueueItemStatus = QueueItemStatus.PENDING
    reviewed_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    review_decision: Optional[ReviewDecision] = None
    override_action: Optional[str] = None  # set when decision=OVERRIDDEN
    reviewer_notes: str = ""
    audit_record_id: Optional[str] = None  # links back to audit log entry
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Queue storage backend (JSON-lines, swap for DB in production)
# ---------------------------------------------------------------------------

_DEFAULT_QUEUE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "hitl_queue.jsonl"
)


def _serialise_item(item: QueueItem) -> Dict[str, Any]:
    return {
        "item_id":            item.item_id,
        "created_at":         item.created_at.isoformat(),
        "track":              item.track.value if item.track else None,
        "region_id":          item.region_id,
        "recommended_action": item.recommended_action,
        "severity_score":     item.severity_score,
        "crisis_score":       item.crisis_score,
        "confidence":         item.confidence,
        "event_type":         item.event_type,
        "contagion_flagged":  item.contagion_flagged,
        "status":             item.status.value,
        "reviewed_at":        item.reviewed_at.isoformat() if item.reviewed_at else None,
        "reviewer_id":        item.reviewer_id,
        "review_decision":    item.review_decision.value if item.review_decision else None,
        "override_action":    item.override_action,
        "reviewer_notes":     item.reviewer_notes,
        "audit_record_id":    item.audit_record_id,
        "metadata":           item.metadata,
    }


def _deserialise_item(row: Dict[str, Any]) -> QueueItem:
    item = QueueItem(
        item_id=row["item_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        track=QueueTrack(row["track"]) if row.get("track") else None,
        region_id=row.get("region_id"),
        recommended_action=row.get("recommended_action", ""),
        severity_score=row.get("severity_score"),
        crisis_score=row.get("crisis_score"),
        confidence=row.get("confidence"),
        event_type=row.get("event_type"),
        contagion_flagged=row.get("contagion_flagged", False),
        status=QueueItemStatus(row["status"]),
        reviewed_at=datetime.fromisoformat(row["reviewed_at"]) if row.get("reviewed_at") else None,
        reviewer_id=row.get("reviewer_id"),
        review_decision=ReviewDecision(row["review_decision"]) if row.get("review_decision") else None,
        override_action=row.get("override_action"),
        reviewer_notes=row.get("reviewer_notes", ""),
        audit_record_id=row.get("audit_record_id"),
        metadata=row.get("metadata", {}),
    )
    return item


# ---------------------------------------------------------------------------
# HITLQueue class
# ---------------------------------------------------------------------------

class HITLQueue:
    """
    Blocking HITL review queue.

    Items are enqueued when the pipeline recommends HUMAN_REVIEW or ESCALATE.
    The pipeline checks get_final_action() before executing any output — if
    the item is still PENDING, no action is taken until a reviewer decides.

    Usage
    -----
    queue = HITLQueue()

    # Pipeline enqueues a recommendation
    item_id = queue.enqueue(track="INDIVIDUAL", region_id="R1",
                            recommended_action="HUMAN_REVIEW", severity=0.82)

    # Dashboard surfaces the item to an on-call reviewer
    pending = queue.get_pending()

    # Reviewer approves or overrides
    queue.review(item_id, reviewer_id="dr_jones", decision="APPROVED")

    # Pipeline checks before executing
    action = queue.get_final_action(item_id)  # "HUMAN_REVIEW" (or override)
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path or _DEFAULT_QUEUE_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Internal load/save ----

    def _load_all(self) -> Dict[str, QueueItem]:
        items: Dict[str, QueueItem] = {}
        if not self._path.exists():
            return items
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                item = _deserialise_item(row)
                items[item.item_id] = item
        return items

    def _save_all(self, items: Dict[str, QueueItem]) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            for item in items.values():
                fh.write(json.dumps(_serialise_item(item), default=str) + "\n")

    # ---- Public API ----

    def enqueue(
        self,
        track: str,
        region_id: str,
        recommended_action: str,
        severity_score: Optional[float] = None,
        crisis_score: Optional[float] = None,
        confidence: Optional[float] = None,
        event_type: Optional[str] = None,
        contagion_flagged: bool = False,
        audit_record_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a recommendation to the HITL queue.
        Returns the item_id.
        """
        item = QueueItem(
            track=QueueTrack(track),
            region_id=region_id,
            recommended_action=recommended_action,
            severity_score=severity_score,
            crisis_score=crisis_score,
            confidence=confidence,
            event_type=event_type,
            contagion_flagged=contagion_flagged,
            audit_record_id=audit_record_id,
            metadata=metadata or {},
        )
        items = self._load_all()
        items[item.item_id] = item
        self._save_all(items)
        return item.item_id

    def get_pending(self) -> List[QueueItem]:
        """Return all items currently awaiting review, sorted oldest-first."""
        items = self._load_all()
        return sorted(
            [i for i in items.values() if i.status == QueueItemStatus.PENDING],
            key=lambda i: i.created_at,
        )

    def review(
        self,
        item_id: str,
        reviewer_id: str,
        decision: str,                   # "APPROVED" | "OVERRIDDEN" | "DISMISSED"
        override_action: Optional[str] = None,
        notes: str = "",
    ) -> QueueItem:
        """
        Record a human reviewer's decision on a pending item.

        Parameters
        ----------
        item_id         : returned by enqueue()
        reviewer_id     : identifier of the reviewing clinician / moderator
        decision        : ReviewDecision value
        override_action : required when decision=OVERRIDDEN; the replacement action
        notes           : free-text rationale (encouraged for audit trail)
        """
        items = self._load_all()
        if item_id not in items:
            raise KeyError(f"Queue item {item_id} not found.")

        item = items[item_id]
        if item.status != QueueItemStatus.PENDING:
            raise ValueError(f"Item {item_id} is already {item.status.value}.")

        dec = ReviewDecision(decision)
        if dec == ReviewDecision.OVERRIDDEN and not override_action:
            raise ValueError("override_action must be set when decision=OVERRIDDEN.")

        item.status          = QueueItemStatus.REVIEWED
        item.reviewed_at     = datetime.utcnow()
        item.reviewer_id     = reviewer_id
        item.review_decision = dec
        item.override_action = override_action
        item.reviewer_notes  = notes

        items[item_id] = item
        self._save_all(items)
        return item

    def get_final_action(self, item_id: str) -> Optional[str]:
        """
        Return the action that should actually be executed.

        Returns
        -------
        - None                if item is still PENDING (block execution)
        - override_action     if reviewer chose OVERRIDDEN
        - recommended_action  if reviewer APPROVED
        - None                if reviewer DISMISSED
        """
        items = self._load_all()
        item = items.get(item_id)
        if item is None or item.status == QueueItemStatus.PENDING:
            return None   # blocked — do not proceed
        if item.review_decision == ReviewDecision.DISMISSED:
            return None
        if item.review_decision == ReviewDecision.OVERRIDDEN:
            return item.override_action
        return item.recommended_action   # APPROVED

    def get_item(self, item_id: str) -> Optional[QueueItem]:
        return self._load_all().get(item_id)

    def pending_count(self) -> int:
        return len(self.get_pending())
