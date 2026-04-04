# GSoC2026-AI4MH

**AI-Powered Social Media Crisis Detection Pipeline for Mental Health**
GSoC 2026 @ HumanAI

---

## 🧠 Overview

This project builds a production-ready dual-track pipeline for early detection and monitoring of mental health crises using social media data. The system operates at two complementary levels simultaneously:

- **Individual Track** — scores every post for distress severity and routes it to the appropriate response tier (passive resource surfacing, helpline prompt, or human review).
- **Aggregate Track** — monitors regional signal patterns, classifies contextual events, corroborates with news media, and escalates emerging crises through a 24-hour confirmation window.

Both tracks feed into a blocking Human-in-the-Loop (HITL) review queue so that no automated action reaching an individual or community executes without a human reviewer's sign-off.

Key design principles:

- **Bias-aware**: four-layer mitigation covering platform underrepresentation, sentiment tool demographic skew, geographic density, and intervention access gaps for low-engagement users.
- **Uncertainty-quantified**: classifier-error-corrected confidence intervals (Daughton & Paul, 2019) modulate all escalation thresholds.
- **Auditable**: every decision — automated or human override — is written to an immutable append-only audit log.
- **Contagion-sensitive**: Werther Effect detection flags co-occurring suicide news and post volume spikes, with automatic threshold adjustment.

---

## 👥 Team

- **Primary Developer**: Vishisht Magan
- **Organisation**: HumanAI / Institute for Social Science Research, The University of Alabama

---

## 📌 Key Features

- Bot and coordination filter using Shannon entropy and near-duplicate fingerprinting.
- Population-density-weighted minimum sample size check before any signal is trusted.
- 7-day rolling average smoothing with calendar event dampening (7 recognised dates).
- Four-layer bias mitigation: platform weights, post-stratification (Giorgi et al., 2022), rural adjacency weighting, intervention bias correction.
- VADER-based sentiment intensity scoring with isolation signal sub-weights and ideation keyword detection.
- First-time poster severity uplift (1.25×).
- Five-tier contextual event classification: CALENDAR, CELEBRITY, LOCAL\_MH, SOCIOECONOMIC, GLOBAL.
- Three-axis media corroboration: keyword overlap + Gaussian timing decay + geographic scope.
- Reactive vs Personal Distress bucketing with per-event dampening weights.
- Confidence-modulated crisis threshold with trend-velocity early warning (linear regression).
- 24-hour confirmation window before any escalation fires.
- Contagion flagging via geometric mean co-occurrence score (NONE / LOW / MEDIUM / HIGH).
- Blocking HITL review queue — pipeline output is gated until a clinician approves, overrides, or dismisses.
- Append-only audit log with query support by region, action type, and time range.
- Geographic heatmap with green → yellow → orange → red colour interpolation and hotspot annotation.
- Campaign effectiveness feedback loop: pre/post signal comparison with EFFECTIVE / NEUTRAL / INEFFECTIVE strategy flags.

---

## 🚧 Current Status

- [x] Bot / coordination filter
- [x] Minimum sample size check
- [x] Smoothing, normalisation, and calendar dampening
- [x] Four-layer bias mitigation
- [x] CI / uncertainty estimation (classifier-error-corrected)
- [x] Individual track: severity scoring, first-time flag, threshold routing
- [x] Contagion flagging (Werther Effect detection)
- [x] Aggregate track: event classification, media corroboration, bucketing
- [x] Aggregate track: crisis scoring, confidence-modulated threshold, confirmation window
- [x] HITL review queue (blocking pattern)
- [x] Audit log (immutable append-only)
- [x] Geographic heatmap output
- [x] Campaign effectiveness feedback loop
- [x] Pipeline orchestration (`CrisisPipeline` class)
- [x] Pipeline pseudocode document (`Pipeline_Pseudocode.docx`)

Future work: live dashboard integration, multi-language sentiment models, cross-platform demographic data refresh.

---

## 📂 Repo Structure

```
src/
  config.py              # All tunable constants — reviewed quarterly by clinical advisory board
  models.py              # Typed dataclasses for all pipeline inputs and outputs
  bot_filter.py          # Component 1 — Bot / coordination filter
  preprocessing.py       # Components 2–5 — Sample check, smoothing, bias mitigation, CI
  individual_track.py    # Components 6–9 — Severity, first-time flag, threshold decision
  contagion_flag.py      # Component 10a — Werther Effect contagion detection
  aggregate_track.py     # Components 10b–17 — Event classification through escalation
  hitl_queue.py          # Component 17 — Human-in-the-Loop review queue
  audit.py               # Component 15 — Immutable audit log
  heatmap.py             # Component 16 — Geographic heatmap output
  campaign_feedback.py   # Component 18 — Campaign effectiveness feedback loop
  pipeline.py            # Top-level orchestration (CrisisPipeline)

data/                    # Runtime output (gitignored)
  audit_log.jsonl        # Append-only audit records
  hitl_queue.jsonl       # HITL review queue state

Pipeline_Pseudocode.docx # 17-component pseudocode reference document
requirements.txt         # Python dependencies
README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Core runtime dependencies: `vaderSentiment`, `pandas`, `numpy`, `geopandas`, `folium`, `plotly`, `requests`.

---

## 🚀 Quick Start

```python
from src.pipeline import CrisisPipeline, PipelineConfig
from src.models import Post, RegionSignal

config = PipelineConfig()
pipeline = CrisisPipeline(config)

# Individual track
pipeline.ingest_posts(posts, region_conf=0.80)

# Aggregate track
pipeline.ingest_regions(regions, news_articles=articles)

# Heatmap
pipeline.render_heatmap(region_centroids=centroids, output_path="heatmap.html")
```

---

## 📚 References

- Daughton, A. R., & Paul, M. J. (2019). Identifying Protective Factors for Depression on Twitter. *ICWSM*.
- Gera, N., & Ciampaglia, G. L. (2022). Tweet Health Correlations and Bot Removal. *WebSci*.
- Giorgi, S., et al. (2022). Robust Post-stratification for Social Media Health Estimates. *ACL*.
- Aguirre, C., & Dredze, M. (2021). Demographic Bias in Depression Classifiers. *W-NUT*.
- McClellan, C., et al. (2017). Using Social Media to Monitor Mental Health Discussions. *JAMIA*.
- Daruna, A. (2026). Human-in-the-Loop Escalation Patterns for Clinical AI Systems.

---

## 📖 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments

This project was carried out under the **Google Summer of Code 2025** programme, with organisational support from **HumanAI** and the **Institute for Social Science Research, The University of Alabama**.

Pipeline parameters and calendar dampening factors are reviewed quarterly in consultation with the clinical advisory board.

NOTE: THIS README WAS GENERATED USING CLAUDE AI
