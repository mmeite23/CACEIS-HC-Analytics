# CACEIS Human Capital Analytics — Data Dictionary

**Last updated:** May 7, 2025  
**Maintainer:** Mory Meïté & Andrea Parache (Albert School)

All source files are confidential CACEIS Investor Services documents.  
**Never commit source files to git.** Place them in the `data/` directory (excluded by `.gitignore`).

---

## Source File Index

| # | File | Sheet(s) | Records | Status | KPIs |
|---|------|---------|---------|--------|------|
| 1 | Data.xlsx | Sheet1 | 275,609 | Real | M1, KPI 3, 4 |
| 2 | Data.xlsx | Absentéisme FR | 72 months | Real | KPI 4 |
| 3 | Data.xlsx | taux mob_TO FR | Monthly | Real | KPI 4 |
| 4 | AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx | Synthese_PL, Synthese_ETP | 4 years | Real | KPI 1, 2 |
| 5 | 2025 - Stats CACEIS EAE EP fichier de travail - Vretraitement.xlsx | Main | ~6,000 | Real (aggregate) | KPI 3 |
| 6 | 20240222 - CACEIS Notes evaluation 2023.xlsx | Main | ~5,500 | Real (aggregate) | KPI 3 |
| 7 | Training_Records_Unnamed.xlsx | Main | 14,943 | Real | KPI 5 |
| 8 | Quick_Review_Unnamed.xlsx | Main | 9,706 | Real | KPI 5 |
| 9 | Cold_Review_Unnamed.xlsx | Main | 8,647 | Real | KPI 5 |
| 10 | Bilan Groupe Be Generous CACEIS 2025.xlsx | Multiple | Aggregate | Real | KPI 6 |
| 11 | Reporting CACEIS Groupe CDP 2023.xlsx | Multiple | Aggregate | Real | KPI 6 |

---

## File 1: Data.xlsx — Sheet1 (Employee Master)

**Description:** Full workforce snapshot 2022-2025. One row per employee per monthly period.  
**Loaded by:** `CACEISDataLoader.load_employee_master()`

| Column | Type | Format | Description |
|--------|------|--------|-------------|
| country_code | str | 2-letter ISO | Country code (FR, LU, DE, etc.) |
| country | str | Full name | Country name (France, Luxembourg, etc.) |
| period | date | YYYY-MM-DD | Snapshot month (Dec-25 = latest) |
| employee_id | str | ANON_XXXXX | Anonymised employee identifier |
| age_range | str | TRANCHE_XX-XX | Age band (10-year intervals, anonymised) |
| gender | str | M/F | Gender (binary only in source) |
| contract_type | str | Permanent/Temporary | Employment contract type |
| degree | str | Bac+N / label | Education level (French scale or label) |
| entry_reason | str | Categorical | Reason for joining (recruitment, transfer, etc.) |
| date_entry_group | date | YYYY-MM-DD | Date joined Crédit Agricole group |
| date_entry_caceis | date | YYYY-MM-DD | Date joined CACEIS entity |
| date_entry_role | date | YYYY-MM-DD | Date entered current role |
| job_title | str | Free text | Job title (uppercase, standardised in clean.py) |
| entity | str | Entity code | Legal entity within CACEIS group |

**Known quality issues:**
- ~3% of `date_entry_caceis` values are null (imputed as 2-year tenure in clean.py)
- Some `job_title` values contain special characters or abbreviations — normalised to uppercase
- Multiple periods per employee: pipeline filters to Dec 2025 snapshot only

---

## File 2: Data.xlsx — Sheet "Absentéisme FR"

**Description:** Monthly absence rate for France, 72 months (June 2019 – May 2025).  
**Loaded by:** `CACEISDataLoader.load_absenteeism()`

| Column | Type | Format | Description |
|--------|------|--------|-------------|
| Mois | date | MM/YYYY | Reference month |
| Taux d'absentéisme | float | % | Monthly absence rate (absence days / working days) |
| Nbre jours absences | int | Days | Total absence days in the month |
| Nbre jours théoriques | int | Days | Total theoretical working days |

**2025 verified value:** 5.29% annual absence rate  
**Known issues:** Some months have merged cells in Excel — openpyxl handles these correctly

---

## File 3: Data.xlsx — Sheet "taux mob_TO FR"

**Description:** Monthly voluntary turnover rate for France.  
**Loaded by:** `CACEISDataLoader.load_turnover()`

| Column | Type | Description |
|--------|------|-------------|
| Mois | date | Reference month |
| Taux de TO | float | Monthly voluntary turnover rate (%) |
| Départs volontaires | int | Count of voluntary departures |
| Effectif moyen | int | Average headcount for the month |

**2025 verified value:** 5.45% annual turnover  

---

## File 4: AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx

**Description:** 4-year P&L summary + FTE headcount by year.  
**Loaded by:** `CACEISDataLoader.load_pl_fte()`  
**Sheets used:** `Synthese_PL` (financials), `Synthese_ETP` (FTE)

| Row keyword | Columns | Unit | Description |
|------------|---------|------|-------------|
| Net Banking Income | 2022-2025 | EUR thousands | PNB (revenue) |
| Total Personnel | 2022-2025 | EUR thousands | Personnel costs (salaries + charges) |
| Formation | 2022-2025 | EUR thousands | Training expenditure |

**FTE by year (from Synthese_ETP, verified):**

| Year | FTE |
|------|-----|
| 2022 | 3,991 |
| 2023 | 6,370 |
| 2024 | 6,616 |
| 2025 | 6,454 |

**Known issues:** 
- Personnel costs appear as negative in some rows (sign convention) — abs() applied in clean.py
- FTE values from Synthese_ETP sheet are hardcoded in config.py as verified anchors

---

## File 5: EAE 2025 Performance Database

**Filename:** `2025 - Stats CACEIS EAE EP fichier de travail - Vretraitement.xlsx`  
**Loaded by:** `CACEISDataLoader.load_eae_2025()`

| Column | Type | Description |
|--------|------|-------------|
| [entity col] | str | CACEIS entity |
| [rating col] | str | Performance label (French: Conforme aux attentes, etc.) |
| [year col] | int | Evaluation year |
| [count col] | int | Number of employees at each rating |

**Aggregate anchors (verified):** avg=3.32 | top(4-5)=38.1% | low(1-2)=7.7% | exceptional=1.8%  
**Status:** Real aggregate data — individual scores are NOT linked to employee_id  
**Known issues:** Rating column name varies by version; auto-detected in `DataCleaner.clean_eae()`

---

## File 6: EAE 2023 Performance Database

**Filename:** `20240222 - CACEIS Notes evaluation 2023.xlsx`  
**Loaded by:** `CACEISDataLoader.load_eae_2023()`

Same structure as EAE 2025. Used to compute `perf_delta` (2023→2025 trajectory).  
**Aggregate anchors (2023):** avg=3.26 | top(4-5)=39.1% | low(1-2)=6.1%

---

## File 7: Training_Records_Unnamed.xlsx

**Description:** Session-level training records — one row per session attendance.  
**Loaded by:** `CACEISDataLoader.load_training()`

| Column | Type | Description |
|--------|------|-------------|
| [session col] | str | Training session name/code |
| [status col] | str | Completion status ('Réalisé', 'Annulé', 'En cours') |
| [hours col] | float | Session duration in hours |
| [date col] | date | Session date |
| [entity col] | str | Participant entity |

**2025 verified values:** 14,943 sessions | 21.7h/FTE | 90.0% completion  
**Known issues:** Status field has French variants ('Réalisé', 'Realisé', 'REALISE') — normalised in `DataCleaner.clean_training()`  
**No individual employee ID** — aggregate stats only linkable to employee master by entity

---

## File 8: Quick_Review_Unnamed.xlsx

**Description:** Post-training satisfaction survey — 9,706 responses.  
**Loaded by:** `CACEISDataLoader.load_quick_review()`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| satisfaction_score | int | 1-5 | Overall training satisfaction |
| session_id | str | — | Training session reference |
| date | date | — | Survey response date |
| entity | str | — | Participant entity |

**2025 verified value:** avg satisfaction 4.47/5

---

## File 9: Cold_Review_Unnamed.xlsx

**Description:** Post-training transfer survey (3 months after training) — 8,647 responses.  
**Loaded by:** `CACEISDataLoader.load_cold_review()`

| Column | Type | Description |
|--------|------|-------------|
| transfer_applied | int (0/1) | Has the employee applied learned skills? |
| transfer_score | int (1-5) | Degree of skill application |
| session_id | str | Training session reference |
| entity | str | Participant entity |

**2025 verified value:** 69.7% transfer rate (binary: applied / not applied)  
**Note:** Transfer rate is the most meaningful training KPI — it measures ROI, not just participation.

---

## File 10: Bilan Groupe Be Generous CACEIS 2025.xlsx

**Description:** Annual D&I report from Mozaïk RH (Be Generous programme).  
**Loaded by:** Not directly loaded — aggregate scores hardcoded in `src/clustering.py`

| Metric | FR 2025 | LU 2025 | Group 2025 |
|--------|---------|---------|------------|
| Inclusion score | 70% | 64% | 72% |
| Leadership identification | 42% | 42% | 44.5% |
| Discrimination rate | 12% | 15% | 11% |
| Career equality | 68% | 61% | 68.5% |
| Work-life balance | 72% | 68% | 71.5% |
| Manager support | 73% | 66% | 72.5% |

**Status:** Real aggregate data from external consultant (Mozaïk RH)  
**Known issues:** PDF extraction required for some sub-metrics; values hardcoded for pipeline stability

---

## File 11: Reporting CACEIS Groupe CDP 2023.xlsx / We Care Bilan 2025.xlsx

**Description:** Annual CSR / D&I reporting documents for CASA (Crédit Agricole) and We Care programme.  
**Used for:** Cross-validation of D&I barometer scores; historical trend (2023 → 2025)

Key verified values used in analysis:
- LU discrimination rate 2023: 13% → 2025: 15% (worsening)
- FR inclusion trajectory: 68% (2022) → 70% (2025)
- Group target 2026: 75% inclusion score

---

## Synthetic Fields (not in source files)

The following fields are generated by `SyntheticEnricher` and are **not real data**:

| Field | Type | Calibration source | Range |
|-------|------|-------------------|-------|
| perf_2025 | int [1-5] | EAE 2025 aggregate | 1-5 |
| perf_2023 | int [1-5] | EAE 2023 aggregate | 1-5 |
| perf_delta | float | Computed from above | -4 to +4 |
| absence_days | float | Bilan Social 2025 (5.29%) | 0-90 |
| salary | float (EUR) | Compensation Data FR (role averages) | 25K-280K |
| market_salary | float (EUR) | salary × market premium | 25K-300K |
| pay_gap_pct | float (%) | Derived from salary/market_salary | -30 to +60% |
| train_hours | float | Training Records (21.7h/FTE) | 0-200 |
| train_done | int (0/1) | Training Records (90.0%) | 0/1 |
| train_sat | float [1-5] | Quick Review (avg 4.47) | 1-5 or NaN |
| train_transfer | float (0/1) | Cold Review (69.7%) | 0/1 or NaN |
| inclusion_score | float (%) | Mozaïk RH barometer | 0-100 |
| attrition_prob | float [0-1] | Logistic DGP calibrated to 5.45% TO | 0-1 |
| attrited | int (0/1) | Bernoulli(attrition_prob) | 0/1 |

See `docs/methodology.md §2` for full calibration methodology and production replacement plan.
