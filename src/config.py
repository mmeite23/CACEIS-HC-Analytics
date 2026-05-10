"""
Configuration module for CACEIS Human Capital Analytics pipeline.

All constants, calibration anchors, model parameters, and domain mappings
are defined here. Every other module imports from this file — do not
scatter magic numbers across the codebase.

Structure:
  - Paths
  - Year/FTE anchors
  - Synthetic calibration targets
  - Feature list (canonical 17-feature order)
  - XGBoost hyperparameters
  - Domain mappings (salary bases, country inclusion, country tiers)
  - KPI thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH: Final[Path] = Path("data")
OUTPUT_DIR: Final[Path] = Path("outputs")

# Source filenames (relative to DATA_PATH)
FILE_EMPLOYEE_MASTER: Final[str] = "Data.xlsx"
FILE_PL_FTE: Final[str] = "AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx"
FILE_EAE_2025: Final[str] = "2025 - Stats CACEIS EAE EP fichier de travail - Vretraitement.xlsx"
FILE_EAE_2023: Final[str] = "20240222 - CACEIS Notes evaluation 2023.xlsx"
FILE_TRAINING: Final[str] = "Training_Records_Unnamed.xlsx"
FILE_QUICK_REVIEW: Final[str] = "Quick_Review_Unnamed.xlsx"
FILE_COLD_REVIEW: Final[str] = "Cold_Review_Unnamed.xlsx"

SHEET_ABSENTEEISM: Final[str] = "Absentéisme FR"
SHEET_TURNOVER: Final[str] = "taux mob_TO FR"
SHEET_PL: Final[str] = "Synthese_PL"

# ── Time / headcount anchors ───────────────────────────────────────────────────

YEARS: Final[list[int]] = [2022, 2023, 2024, 2025]

# Verified FTE by year from Synthese_ETP sheet
FTE_VALS: Final[dict[int, int]] = {
    2022: 3991,
    2023: 6370,
    2024: 6616,
    2025: 6454,
}

ACTIVE_WORKFORCE_DEC2025: Final[int] = 7_415
RANDOM_SEED: Final[int] = 42

# ── Synthetic calibration anchors ─────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationAnchors:
    """
    Published CACEIS aggregate figures used to calibrate synthetic distributions.

    Every synthetic field must have a corresponding anchor here so that the
    validate_calibration() method in SyntheticEnricher can assert it.

    Sources:
      - perf_*: EAE 2025 database / 20250218 Stats EAE report
      - absence_rate_pct: Bilan Social 2025
      - turnover_rate_pct: taux mob_TO FR 2025
      - training_*: Training Records 2025 + Cold/Quick Review
      - inclusion_*: Mozaïk RH Barometer 2025
    """

    # Performance (EAE 2025)
    perf_avg: float = 3.32
    perf_top_pct: float = 38.1          # % rated 4 or 5
    perf_low_pct: float = 7.7           # % rated 1 or 2
    perf_exceptional_pct: float = 1.8   # % rated 5

    # Absence (Bilan Social 2025)
    absence_rate_pct: float = 5.29      # absence days / (N × 230)

    # Turnover (taux mob_TO FR 2025)
    turnover_rate_pct: float = 5.45

    # Training (Training Records + Cold Review 2025)
    training_hours_per_fte: float = 21.7
    training_completion_pct: float = 90.0
    training_transfer_pct: float = 69.7

    # Inclusion (Mozaïk RH Barometer 2025)
    inclusion_fr: float = 70.0
    inclusion_lu: float = 64.0
    inclusion_group: float = 72.0

    # HC-ROI & financials (PL 2025)
    hc_roi_2025: float = 165.8
    revenue_per_fte_k: float = 325.0
    personnel_costs_m: float = 790.0

    # Model performance (validated)
    model_auc: float = 0.784
    precision_at_top10: float = 0.737
    critical_risk_employees: int = 394
    annual_savings_target_m: float = 7.7


ANCHORS: Final[CalibrationAnchors] = CalibrationAnchors()

# Performance distribution by year
PERF_DIST_2025: Final[list[float]] = [0.015, 0.062, 0.543, 0.362, 0.018]
PERF_DIST_2023: Final[list[float]] = [0.012, 0.049, 0.548, 0.373, 0.018]

# ── Features (canonical order — do not change without updating model) ──────────

FEATURES: Final[list[str]] = [
    # Real features (from employee master)
    "tenure_months",
    "tenure_band",
    "age_ordinal",
    "is_permanent",
    "degree_level",
    "country_tier",
    "job_seniority",
    # Synthetic features (calibrated enrichment)
    "perf_2025",
    "perf_delta",
    "absence_days",
    "pay_gap_pct",
    "train_hours",
    "inclusion_score",
    "log_salary",
    "underpaid",
    "no_training",
    "low_inclusion",
]

assert len(FEATURES) == 17, "FEATURES list must contain exactly 17 items"

# ── XGBoost hyperparameters ───────────────────────────────────────────────────

MODEL_PARAMS: Final[dict] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ── Domain mappings ───────────────────────────────────────────────────────────

# Base annual salaries by job-title keyword (EUR, gross)
JOB_BASES: Final[dict[str, int]] = {
    "FUND ACCOUNTANT": 48_000,
    "BUSINESS COORDINATOR": 52_000,
    "TEAM MANAGER": 72_000,
    "HEAD OF UNIT": 95_000,
    "CLIENT RELATIONSHIP": 68_000,
    "SENIOR PROJECT": 78_000,
    "COMPLIANCE": 55_000,
    "RISK": 58_000,
    "TRANSFER": 42_000,
    "DEPOSITARY": 46_000,
    "IT ": 62_000,
    "DATA ANAL": 56_000,
    "GROUP MANAGER": 112_000,
    "DIRECTOR": 145_000,
    "CHIEF": 145_000,
}

DEFAULT_SALARY: Final[int] = 50_000

# Inclusion scores by country (Mozaïk RH Barometer 2025, % scale 0–100)
COUNTRY_INCLUSION: Final[dict[str, int]] = {
    "France": 70,
    "Luxembourg": 64,
    "Germany": 73,
    "Spain": 69,
    "Malaysia": 71,
    "Brazil": 66,
    "Belgium": 72,
    "Ireland": 74,
    "United Kingdom": 73,
    "Netherlands": 75,
    "Switzerland": 74,
    "Colombia": 65,
    "Italy": 68,
    "Portugal": 69,
}

# Country tiers for country_tier feature (labour-market mobility proxy)
#   2 = high-mobility markets, 1 = medium, 0 = low (core European hubs)
COUNTRY_TIER: Final[dict[str, int]] = {
    "Luxembourg": 0,
    "France": 0,
    "Germany": 0,
    "Switzerland": 0,
    "Belgium": 0,
    "Netherlands": 0,
    "Ireland": 1,
    "United Kingdom": 1,
    "Italy": 1,
    "Portugal": 1,
    "Spain": 2,
    "Malaysia": 2,
    "Brazil": 2,
    "Colombia": 2,
}

HIGH_MOBILITY_COUNTRIES: Final[set[str]] = {
    c for c, t in COUNTRY_TIER.items() if t == 2
}

# Age range to ordinal mapping
AGE_ORDINAL: Final[dict[str, int]] = {
    "TRANCHE_10-19": 1,
    "TRANCHE_20-29": 2,
    "TRANCHE_30-39": 3,
    "TRANCHE_40-49": 4,
    "TRANCHE_50-59": 5,
    "TRANCHE_60-69": 6,
    "TRANCHE_70-79": 7,
}

# ── KPI thresholds ────────────────────────────────────────────────────────────

KPI_THRESHOLDS: Final[dict] = {
    "inclusion_red_zone": 65.0,       # below this = red zone
    "inclusion_watch": 70.0,          # below this = watch
    "absence_alert_days": 15,         # individual alert threshold
    "pay_gap_alert_pct": 5.0,         # underpaid flag threshold
    "pay_gap_critical_pct": 10.0,
    "high_risk_prob_threshold": 0.50, # attrition probability alert
    "attrition_cost_multiplier": 1.5, # replacement cost = salary × this
    "training_min_hours": 5.0,        # below = no-training flag
    "precision_at_k_pct": 10,         # operational HR capacity
}

# CACEIS brand colours (for consistent charts across modules)
PALETTE: Final[dict[str, str]] = {
    "navy": "#1B3A6B",
    "blue": "#2E6DA4",
    "gold": "#C8922A",
    "red": "#C0392B",
    "green": "#1A6B3A",
    "muted": "#6B7A99",
    "purple": "#8E44AD",
}
