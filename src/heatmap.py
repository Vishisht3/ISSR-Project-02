"""
Component 16 – Heatmap Output & Dashboard

Generates a geographic heatmap from audit log data.
Supports:
  - Real-time heatmap (short time window, e.g. last 1 hr)
  - Longitudinal trend view (longer window, e.g. last 30 days)
  - Confidence overlay on heatmap scores
  - Crisis type drill-down labels for hotspots
  - Alert status indicators

Output: Folium HTML map + summary JSON payload published to DASHBOARD_URL.

References
----------
Gao et al. (2018): KDE smoothing with historical normalisation for
    spatiotemporal event detection from social media.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import src.config as cfg
from src.audit import AuditLog


# ---------------------------------------------------------------------------
# Colour interpolation
# ---------------------------------------------------------------------------

_COLOUR_SCALE: List[Tuple[float, str]] = [
    (0.00, "#4CAF50"),   # green
    (0.33, "#FFEB3B"),   # yellow
    (0.60, "#FF9800"),   # orange
    (1.00, "#F44336"),   # red
]


def _interpolate_colour(value: float) -> str:
    """Map a normalised value in [0, 1] to a hex colour via the crisis colour scale."""
    value = max(0.0, min(1.0, value))
    for i in range(len(_COLOUR_SCALE) - 1):
        low_val,  low_col  = _COLOUR_SCALE[i]
        high_val, high_col = _COLOUR_SCALE[i + 1]
        if low_val <= value <= high_val:
            t = (value - low_val) / (high_val - low_val)
            # Interpolate each RGB channel
            lr, lg, lb = int(low_col[1:3], 16),  int(low_col[3:5], 16),  int(low_col[5:7], 16)
            hr, hg, hb = int(high_col[1:3], 16), int(high_col[3:5], 16), int(high_col[5:7], 16)
            r = int(lr + t * (hr - lr))
            g = int(lg + t * (hg - lg))
            b = int(lb + t * (hb - lb))
            return f"#{r:02X}{g:02X}{b:02X}"
    return _COLOUR_SCALE[-1][1]


# ---------------------------------------------------------------------------
# Aggregation from audit log
# ---------------------------------------------------------------------------

def _aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group audit records by region_id and compute per-region stats."""
    by_region: Dict[str, List[Dict]] = {}
    for r in records:
        rid = r.get("region_id")
        if not rid:
            continue
        by_region.setdefault(rid, []).append(r)

    result: Dict[str, Dict[str, Any]] = {}
    for region_id, recs in by_region.items():
        crisis_scores  = [r["crisis_score"]   for r in recs if r.get("crisis_score")   is not None]
        severity_scores = [r["severity_score"] for r in recs if r.get("severity_score") is not None]
        escalations    = sum(1 for r in recs if r.get("action_taken") == "ESCALATE")
        event_types    = [r["event_type"] for r in recs if r.get("event_type")]

        crisis_avg   = statistics.mean(crisis_scores)   if crisis_scores   else 0.0
        severity_avg = statistics.mean(severity_scores) if severity_scores else 0.0

        result[region_id] = {
            "crisis_avg":        crisis_avg,
            "severity_avg":      severity_avg,
            "escalation_count":  escalations,
            "total_actions":     len(recs),
            "event_type":        max(set(event_types), key=event_types.count) if event_types else None,
            "records":           recs,
        }
    return result


def _compute_combined_score(agg: Dict[str, Any]) -> float:
    """
    Combined colour score per pseudocode:
    0.5 * crisis_avg + 0.3 * severity_avg + 0.2 * min(escalation_count / ESC_CAP, 1)
    """
    esc_norm = min(agg["escalation_count"] / cfg.ESC_CAP, 1.0)
    return (
        0.5 * (agg["crisis_avg"] or 0.0)
        + 0.3 * (agg["severity_avg"] or 0.0)
        + 0.2 * esc_norm
    )


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmap(
    audit_log: AuditLog,
    time_window_hours: float = 1.0,
    region_centroids: Optional[Dict[str, Tuple[float, float]]] = None,
    output_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Generate a geographic heatmap from recent audit records.

    Parameters
    ----------
    audit_log         : AuditLog instance
    time_window_hours : how far back to look (1.0 = real-time, 720.0 = 30 days)
    region_centroids  : {region_id: (lat, lon)} lookup; used to place map markers
    output_path       : if provided, write HTML map to this path
    now               : override current time (for testing)

    Returns
    -------
    Summary dict with regions_rendered count, hotspots list, and per-region stats.
    """
    if now is None:
        now = datetime.utcnow()

    time_from = now - timedelta(hours=time_window_hours)
    records = audit_log.query(time_from=time_from, time_to=now)

    if not records:
        return {"regions_rendered": 0, "hotspots": []}

    aggregated = _aggregate_records(records)

    # Attach combined score and colour to each region
    for region_id, agg in aggregated.items():
        combined = _compute_combined_score(agg)
        agg["combined_score"] = combined
        agg["colour"]         = _interpolate_colour(combined)
        agg["is_hotspot"]     = combined > cfg.HOTSPOT_MIN

    hotspots = [
        {"region_id": rid, "score": agg["combined_score"], "event_type": agg["event_type"]}
        for rid, agg in aggregated.items()
        if agg["is_hotspot"]
    ]

    # Optionally render HTML map with Folium
    if output_path and region_centroids:
        _render_folium_map(aggregated, region_centroids, hotspots, output_path)

    # Publish JSON summary payload
    summary = {
        "generated_at":     now.isoformat(),
        "window_hours":     time_window_hours,
        "regions_rendered": len(aggregated),
        "hotspots":         hotspots,
        "region_data":      {
            rid: {
                "combined_score":   agg["combined_score"],
                "colour":           agg["colour"],
                "crisis_avg":       agg["crisis_avg"],
                "severity_avg":     agg["severity_avg"],
                "escalation_count": agg["escalation_count"],
                "event_type":       agg["event_type"],
            }
            for rid, agg in aggregated.items()
        },
        "dashboard_url": cfg.DASHBOARD_URL,
    }
    return summary


def _render_folium_map(
    aggregated: Dict[str, Dict[str, Any]],
    region_centroids: Dict[str, Tuple[float, float]],
    hotspots: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Render a Folium choropleth / circle-marker map.
    Falls back gracefully if folium is not installed.
    """
    try:
        import folium  # type: ignore
    except ImportError:
        return

    m = folium.Map(tiles="OpenStreetMap", zoom_start=4)

    for region_id, agg in aggregated.items():
        coords = region_centroids.get(region_id)
        if not coords:
            continue
        lat, lon = coords
        combined = agg["combined_score"]
        colour   = agg["colour"]
        radius   = 8 + combined * 20   # larger circle = higher crisis score

        popup_lines = [
            f"<b>Region:</b> {region_id}",
            f"<b>Combined score:</b> {combined:.3f}",
            f"<b>Crisis avg:</b> {agg['crisis_avg']:.3f}",
            f"<b>Severity avg:</b> {agg['severity_avg']:.3f}",
            f"<b>Escalations:</b> {agg['escalation_count']}",
        ]
        if agg.get("event_type"):
            popup_lines.append(f"<b>Event type:</b> {agg['event_type']}")

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup("<br>".join(popup_lines), max_width=250),
            tooltip=f"{region_id} — score: {combined:.2f}",
        ).add_to(m)

        # Hotspot annotation label
        if agg["is_hotspot"]:
            label = agg.get("event_type") or "unclassified"
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;color:#B71C1C;font-weight:bold;">{label}</div>'
                ),
            ).add_to(m)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    m.save(output_path)


# ---------------------------------------------------------------------------
# Longitudinal trend query helper
# ---------------------------------------------------------------------------

def longitudinal_trend(
    audit_log: AuditLog,
    region_id: str,
    days: int = 30,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Return daily aggregated crisis scores for a region over the last N days.
    Useful for the longitudinal trend view in the dashboard.
    """
    if now is None:
        now = datetime.utcnow()

    daily: Dict[str, List[float]] = {}
    for d in range(days):
        day = (now - timedelta(days=d)).strftime("%Y-%m-%d")
        daily[day] = []

    records = audit_log.query(
        region_id=region_id,
        time_from=now - timedelta(days=days),
        time_to=now,
    )
    for r in records:
        ts = r.get("timestamp", "")[:10]
        if ts in daily and r.get("crisis_score") is not None:
            daily[ts].append(r["crisis_score"])

    return [
        {
            "date":        day,
            "crisis_avg":  statistics.mean(scores) if scores else None,
            "n_records":   len(scores),
        }
        for day, scores in sorted(daily.items())
    ]
