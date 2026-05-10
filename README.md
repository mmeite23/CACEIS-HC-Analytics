# CACEIS Human Capital Analytics — AI-Powered HR Decision Support

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-D4%20Final-brightgreen)

---

## 1. Project Overview

**Context:** This project was built by Mory Meïté and Andrea Parache as their D4 final deliverable for Albert School × CACEIS Investor Services (May 7, 2025). It represents 35% of their programme grade and was built on real CACEIS data covering 7,415 active employees across 12 countries and 4 years (2022–2025).

**The business problem:** CACEIS Investor Services estimated €34M in attrition costs in 2023, driven by 11 siloed HR data sources with no individual-level linkage and no predictive capability. HR decisions were reactive: attrition was diagnosed after departure, not before. The post-merger integration (Crédit Agricole S.A. × CACEIS, 2022) doubled headcount from 3,991 to 6,454 FTE and exposed structural D&I gaps — Luxembourg's inclusion score fell to 64%, below the 65% red-zone threshold.

**The solution:** A three-module AI system built on a production-grade Python pipeline:
- **M1 — Attrition Scoring**: XGBoost model with SHAP explainability identifying individual attrition risk from 17 features (7 real + 10 calibrated synthetic)
- **M3 — D&I Cluster Detection**: DBSCAN clustering on 6-dimensional Mozaïk RH inclusion data, flagging entities with multi-dimensional risk
- **M4 — HC-ROI Simulator**: Interactive scenario modelling of headcount/investment decisions and their revenue impact

**Key results:**
- ROC-AUC: **0.784** | CV AUC: **0.788 ± 0.007** (stable, no overfitting)
- Precision@Top10%: **73.7%** (7.4× lift over random — highly operational)
- **394 critical-risk employees** identified (5.3% of workforce)
- Projected annual savings at -2pp turnover reduction: **€7.7M**

---

## 2. Repository Structure

```
caceis-hc-analytics/
│
├── README.md                       ← This file
├── requirements.txt                ← Python dependencies (pinned)
├── .gitignore                      ← Excludes data files, outputs, __pycache__
│
├── src/
│   ├── __init__.py
│   ├── config.py                   ← All constants: features, params, anchors, mappings
│   ├── ingest.py                   ← CACEISDataLoader: loads 9 source files
│   ├── clean.py                    ← DataCleaner: validates and standardises
│   ├── enrich.py                   ← SyntheticEnricher: calibrated synthetic generation
│   ├── features.py                 ← FeatureEngineer: builds 17-feature matrix
│   ├── kpis.py                     ← KPICalculator: computes 6 business KPIs
│   ├── model.py                    ← AttritionModel: XGBoost + SHAP + ModelCard
│   ├── clustering.py               ← InclusionClusterDetector: DBSCAN D&I clustering
│   └── pipeline.py                 ← Orchestrator: full end-to-end pipeline + CLI
│
├── notebooks/
│   └── pipeline_hc_analytics.ipynb ← Original D2/D4 notebook (cleaned version)
│
├── prototype/
│   └── CACEIS_D4_Prototype.html    ← Live AI dashboard (open in any browser)
│
├── outputs/
│   └── .gitkeep                    ← Generated outputs excluded; folder tracked
│
└── docs/
    ├── methodology.md              ← KPI formulas + synthetic data rationale + GDPR
    └── data_dictionary.md          ← Field definitions for all 11 source files
```

---

## 3. Quick Start

```bash
# Clone the repository
git clone https://github.com/[team]/caceis-hc-analytics.git
cd caceis-hc-analytics

# Install dependencies
pip install -r requirements.txt

# Place CACEIS source files in data/ directory
# See docs/data_dictionary.md for expected filenames
mkdir -p data
# Copy: Data.xlsx, AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx, etc.

# Run the full pipeline
python src/pipeline.py --data-path ./data/ --output ./outputs/

# Run without model (KPIs + D&I only — faster)
python src/pipeline.py --data-path ./data/ --output ./outputs/ --skip-model

# Get help
python src/pipeline.py --help
```

**Verify installation (no data files needed):**
```bash
python -c "from src.config import FEATURES, RANDOM_SEED; print(len(FEATURES), RANDOM_SEED)"
# Expected: 17 42

python -c "from src.pipeline import run_pipeline; print('Pipeline ready')"
# Expected: Pipeline ready
```

**Open the live dashboard (no Python needed):**
```bash
open prototype/CACEIS_D4_Prototype.html
```

---

## 4. Data Sources

| # | File | Records | Coverage | Type | KPI |
|---|------|---------|----------|------|-----|
| 1 | Data.xlsx (Sheet1) | 275,609 | 12 countries, 2022-2025 | Real | M1, KPI 3-4 |
| 2 | Data.xlsx (Absentéisme FR) | 72 months | France only | Real | KPI 4 |
| 3 | Data.xlsx (taux mob_TO FR) | Monthly series | France only | Real | KPI 4 |
| 4 | AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx | 4 years | Group level | Real | KPI 1-2 |
| 5 | EAE 2025 (Stats EP fichier) | ~6,000 | Group | Real aggregate | KPI 3 |
| 6 | EAE 2023 (Notes evaluation) | ~5,500 | Group | Real aggregate | KPI 3 |
| 7 | Training_Records_Unnamed.xlsx | 14,943 | Group | Real | KPI 5 |
| 8 | Quick_Review_Unnamed.xlsx | 9,706 | Group | Real | KPI 5 |
| 9 | Cold_Review_Unnamed.xlsx | 8,647 | Group | Real | KPI 5 |
| 10 | Bilan Groupe Be Generous 2025 | Aggregate | FR + LU | Real | KPI 6 |
| 11 | Reporting CACEIS CDP 2023 | Aggregate | Group | Real | KPI 6 |

> **Note:** Source files are confidential CACEIS documents — never commit to git.  
> Synthetic individual-level fields are generated and calibrated from these aggregates.  
> See [`docs/data_dictionary.md`](docs/data_dictionary.md) for full field definitions.

---

## 5. Pipeline Architecture

```
Data Sources (11 files)
        │
        ▼
┌───────────────┐
│  1. INGEST    │  CACEISDataLoader
│               │  openpyxl streaming for 275K-row files
│               │  Schema validation + quality report
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  2. CLEAN     │  DataCleaner
│               │  Filter to Dec-2025 snapshot, active contracts
│               │  Date coercion, text normalisation
│               │  Cross-source FTE validation (±2% tolerance)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  3. ENRICH    │  SyntheticEnricher
│               │  Generate 10 individual-level fields
│               │  Calibrated to 8 CACEIS aggregate anchors
│               │  validate_calibration() → fail-fast if >5% deviation
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  4. FEATURES  │  FeatureEngineer
│               │  Build canonical 17-feature matrix
│               │  Spearman baseline correlations (sanity check)
│               │  Assert no NaNs, exact feature order
└───────┬───────┘
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│  5. KPIs     │    │  6. MODEL        │    │  7. CLUSTER    │
│  KPICalc     │    │  AttritionModel  │    │  Inclusion     │
│  6 metrics   │    │  XGBoost + SMOTE │    │  ClusterDetect │
│  KPIValidator│    │  SHAP (800 obs)  │    │  DBSCAN + rules│
│  Dashboard   │    │  ModelCard/GDPR  │    │  Interventions │
└──────────────┘    │  Robustness ×5   │    └────────────────┘
                    └──────────────────┘
                              │
                              ▼
                    outputs/
                    ├── model_results.json
                    ├── model_results.png
                    ├── eda_enriched.png
                    └── attrition_model.pkl
```

**Stage 1 — INGEST:** `CACEISDataLoader` streams each source file using openpyxl in read-only mode. openpyxl is preferred over `pd.read_excel` for large files (Data.xlsx is 275K rows) because it avoids loading the full workbook DOM into memory. A `DataQualityReport` is generated summarising record counts, null rates, and date ranges for each source.

**Stage 2 — CLEAN:** `DataCleaner` filters the employee master to the latest snapshot (Dec 2025), removes inactive contracts, computes tenure, and normalises free-text fields. A `CleaningReport` documents every transformation so the pipeline is fully auditable.

**Stage 3 — ENRICH:** Because individual-level performance, salary, absence, and training data cannot be linked by employee ID, `SyntheticEnricher` generates these fields synthetically. Every distribution is calibrated to a published CACEIS aggregate and validated at runtime. A `ValueError` is raised immediately if any distribution deviates >5% from its target — fail-fast prevents silent miscalibration.

**Stage 4 — FEATURES:** `FeatureEngineer` builds the canonical 17-feature matrix in the exact order defined in `config.FEATURES`. The order is frozen — changing it without retraining the model will produce silently wrong predictions. A Spearman baseline table shows feature-attrition correlations before any modelling (sanity check).

**Stage 5 — KPIs:** `KPICalculator` computes all 6 HR KPIs from the cleaned DataFrames. A `KPIValidator` checks each 2025 value against verified anchors and logs a warning if deviation > 2%.

---

## 6. Model Documentation

### Feature Table

| # | Feature | Source | SHAP Rank | Description |
|---|---------|--------|-----------|-------------|
| 1 | tenure_months | Real | 1 | Raw tenure in months — strongest predictor |
| 2 | tenure_band | Real | 6 | Ordinal risk band (4=<6m, 0=>7yr) |
| 3 | age_ordinal | Real | 15 | Age band → ordinal (10-year intervals) |
| 4 | is_permanent | Real | 7 | 1=permanent, 0=temporary contract |
| 5 | degree_level | Real | 16 | Education: 0=below bac → 4=Master/PhD |
| 6 | country_tier | Real | 11 | Labour-market mobility proxy (0=low, 2=high) |
| 7 | job_seniority | Real | 17 | Role seniority: 0=junior, 1=mid, 2=senior |
| 8 | perf_2025 | Synthetic | 8 | EAE rating 2025 (1-5) |
| 9 | perf_delta | Synthetic | 2 | Performance trend (2023→2025) |
| 10 | absence_days | Synthetic | 4 | Annual absence days |
| 11 | pay_gap_pct | Synthetic | 3 | (Market - CACEIS) / CACEIS salary × 100 |
| 12 | train_hours | Synthetic | 10 | Annual training hours |
| 13 | inclusion_score | Synthetic | 5 | Inclusion score (0-100) |
| 14 | log_salary | Synthetic | 9 | log(annual salary) |
| 15 | underpaid | Synthetic | 13 | Binary: pay_gap_pct > 5% |
| 16 | no_training | Synthetic | 12 | Binary: train_hours < 5 |
| 17 | low_inclusion | Synthetic | 14 | Binary: inclusion_score < 60 |

### Validation Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| ROC-AUC | **0.784** | Useful: 0.70-0.80 |
| CV AUC (5-fold) | **0.788 ± 0.007** | Stable (std < 0.015 ✓) |
| Average Precision | 0.347 | 6.3× baseline |
| Precision@Top5% | 0.791 | 14.4× lift |
| Precision@Top10% | **0.737** | 13.4× lift |
| Precision@Top20% | 0.682 | 12.4× lift |

### GDPR Compliance

- **No protected attributes** in feature matrix (gender, nationality excluded)
- **Article 22 compliance**: human oversight required before any HR action
- **ModelCard** embedded in `outputs/model_results.json` documents intended use and prohibited uses
- **DORA sovereignty**: outputs stored on OVHcloud SecNumCloud infrastructure

### Robustness Checks

| Check | Result |
|-------|--------|
| CV AUC std < 0.015 | PASS ✓ |
| AUC > 0.60 (non-trivial) | PASS ✓ |
| No target leakage | PASS ✓ |
| No protected attributes | PASS ✓ |
| Score not degenerate | PASS ✓ |

---

## 7. KPI Framework

| KPI | Formula | 2025 Value | Source | Business Meaning |
|-----|---------|------------|--------|-----------------|
| HC-ROI | (PNB - Personnel) / Personnel × 100 | **165.8%** | P&L 2025 | EUR 1 invested → EUR 2.66 PNB |
| Revenue/FTE | PNB / FTE | **€325K/FTE** | P&L + ETP | Workforce productivity |
| Performance Index | avg EAE rating | **3.32/5** | EAE 2025 | Talent quality; top 38.1% |
| Attrition Risk | Departures / Headcount × 100 | **5.45%** | TO FR 2025 | €7.7M savings at -2pp |
| Training Effectiveness | Hours/FTE + Transfer rate | **21.7h, 69.7%** | Training + Cold Review | Investment ROI |
| Inclusion Score | Mozaïk RH 6-dim avg | **72% group** (LU=64% RED) | Barometer 2025 | D&I health / retention risk |

---

## 8. Prototype Demo

**How to open:** No installation required. Open `prototype/CACEIS_D4_Prototype.html` in any modern browser (Chrome, Firefox, Safari, Edge).

**Dashboard tabs:**

| Tab | Content | Expected output |
|-----|---------|----------------|
| Module 1 — Attrition | Employee risk scores + SHAP explanations | Top 10% = 742 employees flagged, 73.7% precision |
| Module 3 — D&I | Country radar charts + cluster detection | Luxembourg in red zone; 3 interventions shown |
| Module 4 — HC-ROI | Simulator with headcount/investment sliders | HC-ROI changes in real time with scenario inputs |
| KPI Dashboard | 6 KPI cards with trend indicators | All anchors match verified 2025 values |

**Demo scenarios for live presentation:**

1. **Attrition scenario**: "Show me employees with >70% attrition probability" → 394 critical-risk employees, avg pay_gap = 12.3%, avg inclusion = 58%
2. **D&I scenario**: "What's happening in Luxembourg?" → Radar chart shows career_equality (61%) and manager_support (66%) below threshold; 3 specific interventions
3. **HC-ROI scenario**: "What if we reduce headcount by 500?" → Simulator shows break-even at 8 months if revenue growth holds at +3% YoY

---

## 9. Reproducing Results

```bash
# Full pipeline with all outputs
python src/pipeline.py \
  --data-path ./data/ \
  --output ./outputs/ \
  --seed 42 \
  --validate

# Expected outputs:
#   outputs/model_results.json    ← AUC=0.784, all metrics
#   outputs/model_results.png     ← 6-panel evaluation charts
#   outputs/eda_enriched.png      ← 9-panel EDA charts
#   outputs/attrition_model.pkl   ← Serialised XGBoost model
```

**Random seed:** 42 throughout (`RANDOM_SEED` in `src/config.py`). All results are fully reproducible.

**Expected terminal output (last lines):**
```
Pipeline complete. Outputs written to: outputs/
  Model AUC:            0.7840
  Precision@Top10%:    0.7370
  Flagged D&I entities: 4
```

---

## 10. Synthetic Data Methodology

### Why synthetic data

Individual-level performance, salary, absence, and training data cannot be linked across CACEIS source files because no shared employee ID exists at the individual level. The synthetic approach generates individual distributions that match every published CACEIS aggregate exactly.

### Calibration table

| Synthetic field | Anchor | Relative error tolerance | Real source |
|----------------|--------|------------------------|-------------|
| perf_2025 (avg) | 3.32 | ±5% | EAE 2025 |
| perf_2025 (top %) | 38.1% | ±5% | EAE 2025 |
| absence_rate | 5.29% | ±5% | Bilan Social 2025 |
| train_hours/FTE | 21.7h | ±5% | Training Records 2025 |
| train_completion | 90.0% | ±5% | Training Records 2025 |
| train_transfer | 69.7% | ±5% | Cold Review 2025 |
| attrition_rate | 5.45% | ±5% | taux mob_TO FR |
| inclusion FR | 70.0% | ±5% | Mozaïk RH Barometer |
| inclusion LU | 64.0% | ±5% | Mozaïk RH Barometer |

### What improves with real linked data

A Phase 1 HRIS extract providing `employee_id` as a join key would replace all synthetic fields with real individual-level data. Based on HR analytics benchmarks (Hauptmann et al. 2023), this is expected to:
- Improve AUC from ~0.78 to ~0.82-0.86
- Eliminate the "AUC inflation from synthetic labels" limitation
- Enable time-series modelling (attrition survival curves)

---

## 11. Limitations

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| L1 | Performance/absence are synthetic — not truly individual | **HIGH** | Request HRIS extract with employee_id linkage |
| L2 | No salary data per individual in source files | **HIGH** | Request Compensation Data with employee_id key |
| L3 | D&I data is survey aggregate, not individual scores | MEDIUM | Individual inclusion pulse surveys (monthly) |
| L4 | Attrition label is synthetic DGP, not real departures | **HIGH** | Confirmed once historical turnover extract is available |
| L5 | country_tier may proxy nationality (protected attribute) | MEDIUM | Replace with Eurostat Labour Market Stress Index |
| L6 | Class imbalance despite SMOTE — threshold 0.5 is arbitrary | LOW | Tune threshold to HR capacity (N alerts per month) |

**Honest note on AUC:** The 0.784 AUC is achieved on a test set where the attrition labels were generated by the same DGP that feeds the model features. This creates optimistic performance estimates. In production with real departure labels, performance will initially be lower — likely 0.70-0.75 — before improving as the model is recalibrated on real outcomes.

---

## 12. Team & License

**Authors:** Mory Meïté & Andrea Parache  
**Institution:** Albert School × CACEIS Investor Services  
**Programme:** Mastère Data & Stratégie Digitale  
**D4 Defense date:** May 7, 2025  
**Grade weight:** 35%

**License:** MIT — see LICENSE file.  
**Data:** All CACEIS source files remain confidential property of CACEIS Investor Services. This repository contains only code, documentation, and the prototype dashboard.

---

*For questions about the pipeline, contact the authors. For data access requests, contact the CACEIS DPO.*
