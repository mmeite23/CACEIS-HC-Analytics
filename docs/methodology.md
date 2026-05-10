# CACEIS Human Capital Analytics — Methodology

**Authors:** Mory Meïté & Andrea Parache  
**Institution:** Albert School × CACEIS Investor Services  
**Date:** May 7, 2025

---

## 1. KPI Framework

### KPI 1 — HC-ROI (Human Capital Return on Investment)

**Formula:**

$$\text{HC-ROI} = \frac{\text{PNB} - \text{Personnel Costs}}{\text{Personnel Costs}} \times 100$$

**Business rationale:** HC-ROI measures how efficiently CACEIS converts its workforce investment into revenue. For a financial services firm post-merger (Crédit Agricole + CACEIS integration 2022), tracking HC-ROI is critical: the integration doubled headcount (3,991 → 6,454 FTE) while PNB growth needed to match. A declining HC-ROI signals that hiring outpaced revenue growth — a strategic early warning.

**Data source:** `AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx`, sheet `Synthese_PL`  
**Verified 2025 value:** **165.8%** (EUR 1 invested → EUR 2.66 PNB)  
**Trend:** Down from ~180% in 2022 — post-integration normalisation expected  
**Connection to AI module:** M4 HC-ROI simulator models headcount reduction scenarios and their revenue impact before any restructuring decision.

---

### KPI 2 — Revenue per FTE

**Formula:**

$$\text{Rev/FTE} = \frac{\text{PNB}}{\text{FTE headcount}}$$

**Business rationale:** Workforce productivity benchmark. CACEIS target: €325K/FTE. Deviation from this benchmark — either via headcount bloat or revenue shortfall — requires CEO/CFO intervention. Post-merger efficiency synergies should push Rev/FTE upward as integration completes.

**Data source:** `AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx`  
**Verified 2025 value:** **€325K/FTE**  
**Trend:** Up YoY — integration efficiency improving  
**Connection to AI module:** M4 simulator; headcount optimisation scenarios.

---

### KPI 3 — Performance Index

**Formula:**

$$\text{PI} = \bar{x}_{\text{EAE}} = \frac{1}{N}\sum_{i=1}^{N} \text{rating}_i \quad \in [1, 5]$$

Sub-metrics: top-performer rate (rating ≥ 4), low-performer rate (rating ≤ 2), EAE completion rate.

**Business rationale:** CACEIS uses a 5-level EAE scale (Insuffisant → Exceptionnel). The aggregate performance index feeds into calibrated pay-raise pools and succession planning. A low-performer concentration > 10% triggers a talent development review — the current 7.7% is within acceptable bounds.

**Data source:** `2025 - Stats CACEIS EAE EP fichier de travail - Vretraitement.xlsx`  
**Verified 2025 values:** avg **3.32/5** | top (4-5): **38.1%** | low (1-2): **7.7%** | exceptional (5): **1.8%**  
**Trend:** Stable (marginal improvement vs 2023)  
**Connection to AI module:** M1 attrition model — perf_2025 and perf_delta are top-5 SHAP features.

---

### KPI 4 — Attrition Risk

**Formula:**

$$\text{Turnover Rate} = \frac{\text{Departures}}{\text{Avg Headcount}} \times 100$$

$$\text{Attrition Cost} = N_{\text{departures}} \times \bar{S}_{\text{leavers}} \times 1.5$$

The 1.5 multiplier (150% of annual salary) covers: recruitment fees (~15-20% of salary), productivity gap during ramp-up (~6 months), knowledge transfer loss, and manager time cost.

**Business rationale:** CACEIS lost an estimated €34M in attrition costs in 2023 (€790M personnel costs × ~8% average turnover at the time × 150% replacement). Even a 2pp reduction from 5.45% to 3.45% saves €7.7M annually.

**Data source:** `Data.xlsx` sheet `taux mob_TO FR`; `Absentéisme` sheet  
**Verified 2025 values:** turnover FR: **5.45%** | absenteeism: **5.29%** | critical-risk employees: **394**  
**Trend:** Declining from 2023 peak — integration disruption dissipating  
**Connection to AI module:** M1 attrition model is the primary instrument for this KPI.

---

### KPI 5 — Training Effectiveness

**Formula:**

$$\text{Hours/FTE} = \frac{\sum \text{training hours}}{N_{\text{FTE}}}$$

$$\text{Transfer Rate} = \frac{N_{\text{employees applying skills at work}}}{N_{\text{trained}}} \times 100$$

**Business rationale:** Hours/FTE is a lagging input metric. Transfer rate (from Cold Review surveys) is a more meaningful output metric: it measures whether training investment actually changes on-the-job behaviour. CACEIS targets 70% transfer rate; the current 69.7% is marginally below target.

**Data source:** `Training_Records_Unnamed.xlsx` (completion) + `Cold_Review_Unnamed.xlsx` (transfer) + `Quick_Review_Unnamed.xlsx` (satisfaction)  
**Verified 2025 values:** **21.7h/FTE** | completion: **90.0%** | transfer: **69.7%**  
**Trend:** Hours/FTE stable; transfer rate improving from 64% (2023)  
**Connection to AI module:** train_hours and no_training are features 12 and 15 in the M1 attrition model.

---

### KPI 6 — Inclusion Score

**Formula:**

$$\text{Inclusion Score (entity)} = \frac{\sum_{d=1}^{6} w_d \cdot \text{dim}_d}{6} \times 100$$

Where the 6 dimensions are: overall inclusion, leadership identification, discrimination rate (inverted), career equality, work-life balance, manager support.

**Business rationale:** CACEIS's D&I obligation extends beyond legal compliance (loi Rixain, BDES). The Mozaïk RH barometer provides entity-level benchmarks. Luxembourg at 64% is **below the 65% red-zone threshold**, representing a structural inclusion risk — employees in red-zone entities show 1.4× higher attrition probability in the M1 model.

**Data source:** `Bilan Groupe Be Generous CACEIS 2025.xlsx` + `Reporting CACEIS Groupe CDP 2023.xlsx` (Mozaïk RH Barometer)  
**Verified 2025 values:** FR: **70%** | LU: **64%** (RED ZONE) | Group: **72%**  
**Trend:** LU declining (from 66% in 2023); FR stable  
**Connection to AI module:** M3 DBSCAN clustering identifies entity clusters with multi-dimensional D&I risk.

---

## 2. Synthetic Data Rationale

### Why synthetic data was necessary

The 11 CACEIS source files do not share a common employee identifier that would allow individual-level linkage. Specifically:

| Data need | Source available | Problem |
|-----------|-----------------|---------|
| Individual performance score | EAE 2025 database | Aggregate stats only (avg, %, distribution) |
| Individual absence days | Bilan Social 2025 | Monthly aggregate by entity, not per person |
| Individual salary | Compensation Data FR | Role-family averages, not per employee |
| Individual training hours | Training Records | Session-level, not linked to employee master ID |
| Individual inclusion score | Mozaïk RH barometer | Country/entity level only |

Without individual-level linkage, a predictive model cannot be trained. The synthetic enrichment approach:
1. Generates individual-level distributions that **match the real published aggregates exactly**
2. Uses calibration targets drawn from actual CACEIS publications (not assumed)
3. Implements a logistic Data Generating Process (DGP) for the attrition label that encodes validated HR theory

### Calibration methodology

Each synthetic field is generated with a target anchor and validated at pipeline runtime:

| Synthetic field | Generation method | Anchor | Real source |
|----------------|------------------|--------|-------------|
| perf_2025 | Multinomial sample | avg=3.32, top=38.1%, low=7.7% | EAE 2025 |
| perf_delta | Difference of two multinomials | — | EAE 2025 + 2023 |
| absence_days | Zero-inflated Negative Binomial | rate=5.29% | Bilan Social 2025 |
| salary | Role-base × tenure × perf × noise | avg by role family | Compensation Data FR |
| train_hours | Poisson sessions × Lognormal hours | 21.7h/FTE | Training Records 2025 |
| train_done | Bernoulli(0.902) | 90.0% | Training Records 2025 |
| train_transfer | Bernoulli(0.697) | 69.7% | Cold Review 2025 |
| inclusion_score | Country base + adjustments + noise | FR=70%, LU=64% | Mozaïk RH 2025 |
| attrited | Logistic DGP (11 factors) | rate=5.45% | taux mob_TO FR |

### Validation approach

At runtime, `SyntheticEnricher.validate_calibration()` asserts that each synthetic aggregate is within ±5% of its target anchor. The pipeline raises a `ValueError` immediately if any check fails, preventing silent miscalibration.

### Production replacement plan

When CACEIS provides a Phase 1 HRIS extract with `employee_id` as a join key across sources, the synthetic fields should be replaced as follows:

| Current (synthetic) | Replace with |
|--------------------|-------------|
| perf_2025 | Direct join on ANON_ID to EAE database |
| absence_days | Direct join on ANON_ID to Bilan Social detail extract |
| salary | Direct join on ANON_ID to Compensation Data |
| train_hours | Direct join on ANON_ID to Training Records |
| inclusion_score | Direct join on ANON_ID to Mozaïk RH individual survey |

With real linked data, the model AUC is expected to improve from ~0.78 to ~0.82-0.86 based on academic literature on HR analytics (Hauptmann et al. 2023).

---

## 3. Governance & GDPR Framework

### Legal basis

Data processing for the CACEIS HC Analytics project rests on:
- **GDPR Art. 6(1)(f)** — legitimate interest (improving workforce decisions, preventing costly attrition)
- **Proportionality**: only aggregate-level data used in production; individual scores are internal HR tools not shared externally
- **BDES obligation** (Loi El Khomri, France): HR analytics results are disclosed to the Works Council via the BDES annual report

### Anonymisation approach

All employee identifiers in source files are replaced with `ANON_*` IDs before loading into the pipeline. The mapping table is held by the DPO and never enters the analytics environment.

### Article 22 compliance

GDPR Article 22 prohibits automated individual decisions "based solely on automated processing." The CACEIS attrition model is designed as a **decision-support tool**, not an automated decision system:

- HR Business Partners receive a ranked list of at-risk employees, not automated alerts
- No disciplinary, contractual, or salary action may be triggered without human review
- All outputs are marked as "AI-Assisted, Human Decision Required"
- A `ModelCard` is embedded in `model_results.json` documenting prohibited uses

### Bias mitigation

The following features are **explicitly excluded** from the model to prevent discrimination:
- `gender` — protected attribute (GDPR Art. 9)
- Raw `age` — protected attribute; only ordinal age band used as proxy for career stage
- `nationality` — protected attribute; `country` is used only for labour-market mobility proxy
- `disability status`, `union membership`, `political opinion` — never in scope

The `country_tier` feature may proxy nationality. This is flagged in the ModelCard as a known limitation. In Phase 2, it should be replaced with external labour-market mobility indices (Eurostat LMSI).

### DORA sovereignty

All pipeline outputs (model weights, scores) are stored on OVHcloud SecNumCloud-certified infrastructure, compliant with DORA Article 30 requirements for Tier 1 financial firms. Cross-border data transfers follow standard contractual clauses (SCCs) where applicable.
