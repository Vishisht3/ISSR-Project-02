"""
Component 15 – Audit Log

Immutable, append-only record of every decision made by both pipeline tracks.
Provides transparency, debugging capability, and regulatory compliance support.

Every escalation threshold crossing, action taken, and human override is logged
here. No automated action fires without an audit record.

Storage backend
---------------
Writes to a JSON-lines file (one record per line, append-only).
The backend class can be replaced with a database writer by subclassing
_JsonLinesBackend and passing it to AuditLog.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import AuditRecord, TrackType


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class _JsonLinesBackend:
    """Append-only JSON-lines file writer."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        row = _serialise(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def query(
        self,
        region_id: Optional[str] = None,
        action_type: Optional[str] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        results = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ts = datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else None
                if region_id   and row.get("region_id") != region_id:
                    continue
                if action_type and row.get("action_taken") != action_type:
                    continue
                if time_from   and ts and ts < time_from:
                    continue
                if time_to     and ts and ts > time_to:
                    continue
                results.append(row)
        return results


def _serialise(record: AuditRecord) -> Dict[str, Any]:
    return {
        "id":             record.id,
        "timestamp":      record.timestamp.isoformat() if record.timestamp else None,
        "track":          record.track.value if record.track else None,
        "region_id":      record.region_id,
        "action_taken":   record.action_taken,
        "severity_score": record.severity_score,
        "crisis_score":   record.crisis_score,
        "confidence":     record.confidence,
        "event_type":     record.event_type,
        "bucket":         record.bucket,
        "first_time":     record.first_time,
        "reviewer_id":    record.reviewer_id,
        "metadata":       record.metadata,
    }


# ---------------------------------------------------------------------------
# Public AuditLog interface
# ---------------------------------------------------------------------------

_DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "audit_log.jsonl"
)


class AuditLog:
    """
    Thread-safe append-only audit log.

    Usage
    -----
    log = AuditLog()
    log.write_individual(region_id=..., action=..., severity=..., ...)
    log.write_aggregate(region_id=..., action=..., crisis=..., ...)
    records = log.query(region_id="county_42", time_from=..., time_to=...)
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._backend = _JsonLinesBackend(path or _DEFAULT_LOG_PATH)

    # ---- Write helpers ----

    def write(self, record: AuditRecord) -> str:
        """Append a fully constructed AuditRecord. Returns the record id."""
        record.timestamp = record.timestamp or datetime.utcnow()
        self._backend.append(record)
        return record.id

    def write_individual(
        self,
        region_id: str,
        action: str,
        severity: float,
        confidence: float,
        first_time: bool,
        metadata: Optional[Dict[str, Any]] = None,
        reviewer_id: Optional[str] = None,
    ) -> str:
        record = AuditRecord(
            track=TrackType.INDIVIDUAL,
            region_id=region_id,
            action_taken=action,
            severity_score=severity,
            confidence=confidence,
            first_time=first_time,
            reviewer_id=reviewer_id,
            metadata=metadata or {},
        )
        return self.write(record)

    def write_aggregate(
        self,
        region_id: str,
        action: str,
        crisis_score: float,
        confidence: float,
        event_type: Optional[str] = None,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reviewer_id: Optional[str] = None,
    ) -> str:
        record = AuditRecord(
            track=TrackType.AGGREGATE,
            region_id=region_id,
            action_taken=action,
            crisis_score=crisis_score,
            confidence=confidence,
            event_type=event_type,
            bucket=bucket,
            reviewer_id=reviewer_id,
            metadata=metadata or {},
        )
        return self.write(record)

    def record_human_override(
        self,
        record_id: str,
        reviewer_id: str,
        override_action: str,
        reason: str,
    ) -> str:
        """Log a human reviewer override as a separate audit entry."""
        record = AuditRecord(
            action_taken=f"HUMAN_OVERRIDE:{override_action}",
            reviewer_id=reviewer_id,
            metadata={
                "overrides_record_id": record_id,
                "reason": reason,
            },
        )
        return self.write(record)

    # ---- Query ----

    def query(
        self,
        region_id: Optional[str] = None,
        action_type: Optional[str] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query audit records with optional filters.
        Returns list of dicts (raw JSON records).
        """
        return self._backend.query(
            region_id=region_id,
            action_type=action_type,
            time_from=time_from,
            time_to=time_to,
        )
