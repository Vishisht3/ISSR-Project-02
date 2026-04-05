# GSoC2026-AI4MH

**AI-Powered Social Media Crisis Detection Pipeline for Mental Health**
GSoC 2026 @ HumanAI

---

## 🧠 Overview

This project builds a production-ready dual-track pipeline for early detection and monitoring of mental health crises using social media data. The system operates at two complementary levels simultaneously:

- **Individual Track** — scores every post for distress severity and routes it to the appropriate response tier (passive resource surfacing, helpline prompt, or human review).
- **Aggregate Track** — monitors regional signal patterns, classifies contextual events, corroborates with news media, and escalates emerging crises through a 24-hour confirmation window.

Both tracks feed into a blocking Human-in-the-Loop (HITL) review queue so that no automated action reaching an individual or community executes without a human reviewer's sign-off.

This framework integrates directly with the existing ISSR prototype by Yixing Fan (`ISSR_AI4MH_Yixing_Fan`), which handles Reddit data ingestion and BERT-based suicidal ideation classification. The adapter layer (`src/issr_adapter.py`) bridges the two codebases without modifying any of Yixing's files.

Key design principles:

- **Bias-aware**: four-layer mitigation covering platform underrepresentation, sentiment tool demographic skew, geographic density, and intervention access gaps for low-engagement users.
- **Uncertainty-quantified**: classifier-error-corrected confidence intervals (Daughton & Paul, 2019) modulate all escalation thresholds.
- **Auditable**: every decision — automated or human override — is written to an immutable append-only audit log.
- **Contagion-sensitive**: Werther Effect detection flags co-occurring suicide news and post volume spikes, with automatic threshold adjustment.
- **Engagement-aware**: sample size thresholds scale with both geographic population density and post engagement (upvotes + comments), so a small number of high-signal posts can satisfy the statistical gate that raw post count alone would fail.

---

## 👥 Team

- **Primary Developer**: Vishisht Magan
- **Organisation**: HumanAI / Institute for Social Science Research, The University of Alabama

---

## 📌 Key Features

- Bot and coordination filter using Shannon entropy and near-duplicate fingerprinting.
- Population-density-weighted minimum sample size check, with engagement weighting — each post contributes `log2(1 + upvotes + comments)` effective samples so high-engagement posts count proportionally more.
- 7-day rolling average smoothing with calendar event dampening (7 recognised dates).
- Four-layer bias mitigation: platform weights, post-stratification (Giorgi et al., 2022), rural adjacency weighting, intervention bias correction.
- nltk VADER sentiment intensity scoring (same package as Yixing's prototype) with isolation signal sub-weights and ideation keyword detection.
- First-time poster severity uplift (1.25×).
- Five-tier contextual event classification: CALENDAR, CELEBRITY, LOCAL_MH, SOCIOECONOMIC, GLOBAL.
- Three-axis media corroboration: keyword overlap + Gaussian timing decay + geographic scope.
- Reactive vs Personal Distress bucketing with per-event dampening weights.
- Confidence-modulated crisis threshold with trend-velocity early warning (linear regression).
- 24-hour confirmation window before any escalation fires.
- Contagion flagging via geometric mean co-occurrence score (NONE / LOW / MEDIUM / HIGH).
- Blocking HITL review queue — pipeline output is gated until a clinician approves, overrides, or dismisses.
- Append-only audit log with query support by region, action type, and time range.
- Geographic heatmap with green → yellow → orange → red colour interpolation and hotspot annotation.
- Campaign effectiveness feedback loop: pre/post signal comparison with EFFECTIVE / NEUTRAL / INEFFECTIVE strategy flags.
- Integration adapter for Yixing Fan's pipeline: CSV → Post/RegionSignal conversion, risk-level mapping, VADER and BERT model wrappers, reference-date inference from historical data.

---

## 🚧 Current Status

- [] Bot / coordination filter
- [] Minimum sample size check (density-weighted + engagement-weighted)
- [] Smoothing, normalisation, and calendar dampening
- [] Four-layer bias mitigation
- [] CI / uncertainty estimation (classifier-error-corrected)
- [] Individual track: severity scoring, first-time flag, threshold routing
- [] Contagion flagging (Werther Effect detection)
- [] Aggregate track: event classification, media corroboration, bucketing
- [] Aggregate track: crisis scoring, confidence-modulated threshold, confirmation window
- [] HITL review queue (blocking pattern)
- [] Audit log (immutable append-only)
- [] Geographic heatmap output
- [] Campaign effectiveness feedback loop
- [] Pipeline orchestration (`CrisisPipeline` class)
- [] Pipeline pseudocode document (`Pipeline_Pseudocode.docx`)
- [] Integration adapter for Yixing Fan's ISSR prototype (`src/issr_adapter.py`)

Future work: live dashboard integration, multi-language sentiment models, cross-platform demographic data refresh.

---

## 📂 Repo Structure

```
src/
  __init__.py            # Public API exports
  config.py              # All tunable constants — reviewed quarterly by clinical advisory board
  models.py              # Typed dataclasses for all pipeline inputs and outputs
  bot_filter.py          # Component 1  — Bot / coordination filter
  preprocessing.py       # Components 2–5 — Sample check, smoothing, bias mitigation, CI
  individual_track.py    # Components 6–9 — Severity, first-time flag, threshold decision
  contagion_flag.py      # Component 10a — Werther Effect contagion detection
  aggregate_track.py     # Components 10b–17 — Event classification through escalation
  hitl_queue.py          # Component 17 — Human-in-the-Loop review queue
  audit.py               # Component 15 — Immutable audit log
  heatmap.py             # Component 16 — Geographic heatmap output
  campaign_feedback.py   # Component 18 — Campaign effectiveness feedback loop
  pipeline.py            # Top-level orchestration (CrisisPipeline)
  issr_adapter.py        # Integration bridge → Yixing Fan's ISSR prototype

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

Core runtime dependencies: `nltk`, `pandas`, `numpy`, `geopandas`, `folium`, `plotly`, `requests`.

---

## 🚀 Quick Start

### Standalone (no external prototype)

```python
from src.pipeline import CrisisPipeline, PipelineConfig
from src.models import Post, RegionSignal

pipeline = CrisisPipeline(PipelineConfig())

# Individual track
pipeline.ingest_posts(posts)

# Aggregate track
pipeline.ingest_regions(regions, news_articles=articles)

# Heatmap
pipeline.render_heatmap(region_centroids=centroids, output_path="heatmap.html")
```

### Integrated with Yixing Fan's pipeline

```python
import pandas as pd
from src.issr_adapter import (
    dataframe_to_posts,
    posts_to_region_signals,
    make_issr_pipeline_config,
    risk_level_to_action,
)
from src.pipeline import CrisisPipeline

# Load Yixing's Reddit data
df = pd.read_csv("../ISSR_AI4MH_Yixing_Fan/output/filtered_reddit_posts.csv")

# Convert to framework types (ref_date inferred from post timestamps)
posts = dataframe_to_posts(df)
signals, ref_date = posts_to_region_signals(posts)

# Build pipeline wired to Yixing's models
pipeline = CrisisPipeline(make_issr_pipeline_config(
    bert_model_path="../ISSR_AI4MH_Yixing_Fan/results/final_model",
    platform="reddit",
))

# Run both tracks
pipeline.ingest_posts(posts, now=ref_date)
pipeline.ingest_regions(
    signals,
    geo_concentrations={r.region_id: 0.6 for r in signals},
    now=ref_date,
)

# Review pending HITL items
for item in pipeline.hitl_queue.get_pending():
    print(item)

# Translate Yixing's risk labels to framework actions
action = risk_level_to_action("High-Risk")  # → IndividualAction.HUMAN_REVIEW
```

---

## 🔗 Integration Notes

`src/issr_adapter.py` bridges this framework with Yixing Fan's ISSR prototype without modifying any of his files. Key behaviours:

- **Date inference**: `posts_to_region_signals()` infers the reference date from the most recent post timestamp in the batch, so historical datasets (e.g. archived Reddit CSVs) are evaluated against their own timeline rather than the wall-clock date.
- **Engagement weighting**: each post contributes `log2(1 + upvotes + comments)` effective samples to the daily count. A post with 7,000 upvotes satisfies the sample-size gate on its own; a post with zero engagement contributes a floor of 1.
- **Risk label mapping**: Yixing's `"High-Risk"` / `"Moderate Concern"` / `"Low Concern"` labels map cleanly to `HUMAN_REVIEW` / `HELPLINE_PROMPT` / `PASSIVE_RESOURCE`.
- **VADER consistency**: both codebases use `nltk.sentiment.SentimentIntensityAnalyzer`, eliminating any scoring divergence from package version skew.
- **BERT uplift**: when Yixing's fine-tuned BERT classifies a post as `"Suicidal"`, the VADER distress score receives a 40% uplift before reaching the threshold logic, reflecting the higher precision of the task-specific model.

---

## 📚 References

- Daughton, A. R., & Paul, M. J. (2019). Identifying Protective Factors for Depression on Twitter. _ICWSM_.
- Gera, N., & Ciampaglia, G. L. (2022). Tweet Health Correlations and Bot Removal. _WebSci_.
- Giorgi, S., et al. (2022). Robust Post-stratification for Social Media Health Estimates. _ACL_.
- Aguirre, C., & Dredze, M. (2021). Demographic Bias in Depression Classifiers. _W-NUT_.
- McClellan, C., et al. (2017). Using Social Media to Monitor Mental Health Discussions. _JAMIA_.
- Daruna, A. (2026). Human-in-the-Loop Escalation Patterns for Clinical AI Systems.

---

## 📖 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments

This project was carried out under the **Google Summer of Code 2025** programme, with organisational support from **HumanAI** and the **Institute for Social Science Research, The University of Alabama**.

Pipeline parameters and calendar dampening factors are reviewed quarterly in consultation with the clinical advisory board.

NOTE: README WAS GENERATED USING CLAUDE.AI
