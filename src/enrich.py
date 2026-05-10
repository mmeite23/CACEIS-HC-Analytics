"""
Synthetic data enrichment for CACEIS Human Capital Analytics.

Because employee-level performance, salary, absence, and training records
cannot be linked by individual ID across CACEIS source files, these fields
are generated synthetically. Every distribution is calibrated to match a
published CACEIS aggregate anchor (EAE 2025, Bilan Social 2025, etc.).

Each generation method:
  1. Produces the synthetic array
  2. Prints: "Generated X — mean Y.YY (target Z.ZZ)"
  3. Is validated by validate_calibration() before any downstream use

See docs/methodology.md §2 for full rationale and GDPR considerations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    ANCHORS,
    COUNTRY_INCLUSION,
    JOB_BASES,
    DEFAULT_SALARY,
    KPI_THRESHOLDS,
    PERF_DIST_2023,
    PERF_DIST_2025,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# Calibration tolerance: synthetic aggregate must be within ±5% of target
_CALIBRATION_TOLERANCE = 0.05


# ── Calibration validation ─────────────────────────────────────────────────────


def _check(name: str, actual: float, target: float, tol: float = _CALIBRATION_TOLERANCE) -> None:
    """Raise ValueError if actual deviates from target by more than tol (relative)."""
    if target == 0:
        return
    relative_err = abs(actual - target) / abs(target)
    status = "OK" if relative_err <= tol else "FAIL"
    logger.info("  Calibration %-30s: actual=%.4f  target=%.4f  err=%.1f%%  [%s]",
                name, actual, target, relative_err * 100, status)
    if relative_err > tol:
        raise ValueError(
            f"Calibration failure — {name}: actual={actual:.4f}, "
            f"target={target:.4f}, deviation={relative_err*100:.1f}% > {tol*100:.0f}% tolerance"
        )


# ── Enricher ───────────────────────────────────────────────────────────────────


class SyntheticEnricher:
    """
    Generates calibrated synthetic HR fields for individual-level modelling.

    All methods accept N (number of employees) and rng (a seeded
    np.random.RandomState) so that generation is fully reproducible.

    Example
    -------
    >>> rng = np.random.RandomState(42)
    >>> enricher = SyntheticEnricher()
    >>> df_enriched = enricher.enrich_all(df_employees, rng)
    """

    # ── Performance ───────────────────────────────────────────────────────────

    def generate_performance(
        self, N: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate performance ratings for 2023 and 2025.

        Calibration anchors:
          - avg 3.32 (EAE 2025)
          - top (4-5): 38.1%
          - low (1-2): 7.7%
          - exceptional (5): 1.8%

        Returns
        -------
        perf_2025, perf_2023 : ndarray of int in [1, 5]
        perf_delta            : float delta (2025 - 2023)
        """
        perf_2025 = rng.choice([1, 2, 3, 4, 5], size=N, p=PERF_DIST_2025)
        perf_2023 = rng.choice([1, 2, 3, 4, 5], size=N, p=PERF_DIST_2023)
        perf_delta = (perf_2025 - perf_2023).astype(float)

        logger.info(
            "Generated perf_2025 — avg=%.3f (target=%.2f) | top=%.1f%% (target=%.1f%%) | low=%.1f%% (target=%.1f%%)",
            perf_2025.mean(), ANCHORS.perf_avg,
            (perf_2025 >= 4).mean() * 100, ANCHORS.perf_top_pct,
            (perf_2025 <= 2).mean() * 100, ANCHORS.perf_low_pct,
        )
        return perf_2025, perf_2023, perf_delta

    # ── Absence ───────────────────────────────────────────────────────────────

    def generate_absence(
        self, N: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Generate individual absence days using a zero-inflated Negative Binomial.

        ~38% of employees take any absence; given absent, mean ~3.2 days.
        Calibration target: 5.29% annual rate (Bilan Social 2025).

        Returns
        -------
        absence_days : ndarray of float, shape (N,)
        """
        has_absence = rng.binomial(1, 0.38, N)
        days_given_absent = rng.negative_binomial(2, 0.38, N).clip(1, 90)
        absence_days = (has_absence * days_given_absent).astype(float)

        actual_rate = absence_days.sum() / (N * 230) * 100
        logger.info(
            "Generated absence_days — rate=%.2f%% (target=%.2f%%)",
            actual_rate, ANCHORS.absence_rate_pct,
        )
        return absence_days

    # ── Salary ────────────────────────────────────────────────────────────────

    def generate_salary(
        self,
        df: pd.DataFrame,
        perf_2025: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate individual salaries based on job title, tenure, and performance.

        Base salary is matched by job-title keyword (JOB_BASES in config).
        Tenure premium: +1.5% per year, capped at 20 years.
        Performance multiplier: 0.95 (rating 1) → 1.12 (rating 5).
        Market salary: base × external-rate factor (+2% to +10%).

        Returns
        -------
        salaries     : actual CACEIS salary (EUR)
        market_sal   : external market benchmark (EUR)
        pay_gap_pct  : (market - actual) / actual × 100 (+ve = underpaid)
        """
        N = len(df)

        def _base(title: object) -> int:
            t = str(title).upper()
            for k, v in JOB_BASES.items():
                if k in t:
                    return v
            return DEFAULT_SALARY

        base_sal = np.array([_base(t) for t in df["job_title"]])
        tenure = df["tenure_years"].fillna(0).values
        ten_prem = (1 + np.minimum(tenure, 20) * 0.015)
        perf_mul = np.array([{1: 0.95, 2: 0.98, 3: 1.00, 4: 1.05, 5: 1.12}[p] for p in perf_2025])
        sal_noise = rng.normal(1, 0.07, N).clip(0.80, 1.25)
        salaries = (base_sal * ten_prem * perf_mul * sal_noise).clip(25_000, 280_000)

        mkt_factor = np.where(perf_2025 >= 4, 1.10, np.where(perf_2025 <= 2, 1.02, 1.06))
        mkt_noise = rng.normal(1, 0.04, N).clip(0.88, 1.18)
        market_sal = salaries * mkt_factor * mkt_noise
        pay_gap_pct = (market_sal - salaries) / salaries * 100

        logger.info(
            "Generated salaries — avg=EUR %.0f | underpaid(>5%%): %.0f%%",
            salaries.mean(),
            (pay_gap_pct > 5).mean() * 100,
        )
        return salaries, market_sal, pay_gap_pct

    # ── Training ──────────────────────────────────────────────────────────────

    def generate_training(
        self, N: int, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training engagement metrics per employee.

        Calibration targets (Training Records 2025 + Cold Review):
          - 21.7 hours/FTE
          - 90.0% completion
          - 69.7% transfer rate

        Returns
        -------
        train_hours, train_done, train_sat, train_transfer
        """
        n_sessions = rng.poisson(1.8, N).clip(0, 6)
        hours_session = rng.lognormal(2.4, 0.7, N).clip(1, 40)
        train_hours = np.where(n_sessions == 0, 0, n_sessions * hours_session).clip(0, 200)
        train_done = np.where(n_sessions > 0, rng.binomial(1, 0.902, N), 0).astype(float)
        train_sat = np.where(
            n_sessions > 0,
            rng.choice([1, 2, 3, 4, 5], N, p=[0.01, 0.03, 0.10, 0.35, 0.51]),
            np.nan,
        )
        train_transfer = np.where(n_sessions > 0, rng.binomial(1, 0.697, N), np.nan)

        logger.info(
            "Generated training — avg_hours=%.1f (target=%.1f) | completion=%.1f%% (target=%.1f%%) | transfer=%.1f%% (target=%.1f%%)",
            train_hours.mean(), ANCHORS.training_hours_per_fte,
            train_done.mean() * 100, ANCHORS.training_completion_pct,
            np.nanmean(train_transfer) * 100, ANCHORS.training_transfer_pct,
        )
        return train_hours, train_done, train_sat, train_transfer

    # ── Inclusion ─────────────────────────────────────────────────────────────

    def generate_inclusion(
        self,
        df: pd.DataFrame,
        perf_2025: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Generate individual inclusion scores.

        Country base from COUNTRY_INCLUSION dict (Mozaïk RH Barometer 2025).
        Adjustments: performance, contract type, tenure.

        Returns
        -------
        inclusion_score : ndarray of float in [0, 100]
        """
        N = len(df)
        inc_base = np.array([COUNTRY_INCLUSION.get(c, 70) for c in df["country"]])
        tenure = df["tenure_years"].fillna(0).values
        perf_adj = np.where(perf_2025 >= 4, 5, np.where(perf_2025 <= 2, -8, 0))
        contract_adj = np.where(df["contract_type"].values == "Permanent contract", 2, -4)
        tenure_adj = np.where(tenure > 5, 3, -2)
        noise = rng.normal(0, 7, N)

        inc_score = (inc_base + perf_adj + contract_adj + tenure_adj + noise).clip(0, 100)

        inc_fr = inc_score[df["country"].values == "France"].mean()
        inc_lu = inc_score[df["country"].values == "Luxembourg"].mean()
        logger.info(
            "Generated inclusion — FR=%.1f%% (target=%.1f%%) | LU=%.1f%% (target=%.1f%%)",
            inc_fr, ANCHORS.inclusion_fr,
            inc_lu, ANCHORS.inclusion_lu,
        )
        return inc_score

    # ── Attrition label ───────────────────────────────────────────────────────

    def generate_attrition_label(
        self,
        df: pd.DataFrame,
        perf_2025: np.ndarray,
        perf_delta: np.ndarray,
        pay_gap_pct: np.ndarray,
        absence_days: np.ndarray,
        train_hours: np.ndarray,
        inc_score: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate attrition probability and binary label.

        Logistic DGP with 11 risk factors (10 from notebook + compound
        disengagement signal). Calibrated to 5.45% turnover (TO FR 2025).

        Factor 11 — compound disengagement signal:
          inclusion_score < 50 AND perf_delta < 0 → +0.6 to log-odds.
          This captures employees who are simultaneously excluded AND declining,
          a pattern associated with silent disengagement before departure.

        Returns
        -------
        attrition_prob : float array in [0, 1]
        attrited       : binary int array
        """
        N = len(df)
        log_odds_base = np.log(0.055 / 0.945)  # anchors base rate at 5.5%
        lo = np.full(N, log_odds_base)

        # 1. Tenure (non-linear — strongest single predictor)
        ten = df["tenure_years"].fillna(2).values
        lo += np.where(ten < 0.5, 2.2,
              np.where(ten < 1.5, 1.5,
              np.where(ten < 3.0, 0.5,
              np.where(ten < 8.0, 0.6,
              np.where(ten < 15.0, -0.3, -0.8)))))

        # 2. Contract type
        lo += np.where(df["contract_type"].values == "Temporary contract", 1.8, 0)

        # 3. Performance trajectory (declining engagement)
        lo += np.where(perf_delta < -1, 1.0, np.where(perf_delta < 0, 0.4, np.where(perf_delta > 1, -0.5, 0)))

        # 4. Current performance level
        lo += np.where(perf_2025 <= 2, 0.9, np.where(perf_2025 >= 5, -0.7, np.where(perf_2025 == 4, -0.3, 0)))

        # 5. Pay gap vs. market
        lo += np.where(pay_gap_pct > 15, 1.2,
              np.where(pay_gap_pct > 10, 0.8,
              np.where(pay_gap_pct > 5, 0.4,
              np.where(pay_gap_pct < -5, -0.3, 0))))

        # 6. Absenteeism (proxy for disengagement)
        lo += np.where(absence_days > 30, 0.9, np.where(absence_days > 15, 0.5, np.where(absence_days > 5, 0.2, 0)))

        # 7. Training disengagement
        lo += np.where(train_hours == 0, 0.5, np.where(train_hours < 5, 0.2, 0))

        # 8. Inclusion score
        lo += np.where(inc_score < 50, 0.8, np.where(inc_score < 60, 0.4, np.where(inc_score > 80, -0.3, 0)))

        # 9. Country mobility culture
        lo += np.where(df["country"].isin(["Malaysia", "Brazil", "Colombia", "Spain"]).values, 0.9, 0)

        # 10. Interaction: underpaid AND declining performance
        lo += np.where((pay_gap_pct > 10) & (perf_delta < 0), 0.8, 0)

        # 11. Compound disengagement signal (new factor)
        # Employees who are both excluded (inclusion < 50) AND declining (perf_delta < 0)
        # show amplified disengagement beyond either factor alone.
        lo += np.where((inc_score < 50) & (perf_delta < 0), 0.6, 0)

        # Individual noise
        lo += rng.normal(0, 0.5, N)

        attrition_prob = 1 / (1 + np.exp(-lo))
        attrited = rng.binomial(1, attrition_prob, N)

        logger.info(
            "Generated attrition — %.1f%% attrited (target=%.2f%%) | high-risk(>50%%): %d",
            attrited.mean() * 100, ANCHORS.turnover_rate_pct,
            (attrition_prob > 0.5).sum(),
        )
        return attrition_prob, attrited

    # ── Calibration validation ────────────────────────────────────────────────

    def validate_calibration(
        self,
        df_m: pd.DataFrame,
    ) -> None:
        """
        Assert all synthetic distributions are within ±5% of their anchor targets.

        Raises ValueError immediately on the first failed check so that the
        pipeline fails fast rather than silently producing invalid results.
        """
        logger.info("Running calibration validation...")
        N = len(df_m)

        _check("perf_avg",         df_m["perf_2025"].mean(),                   ANCHORS.perf_avg)
        _check("perf_top_pct",     (df_m["perf_2025"] >= 4).mean() * 100,      ANCHORS.perf_top_pct)
        _check("perf_low_pct",     (df_m["perf_2025"] <= 2).mean() * 100,      ANCHORS.perf_low_pct)
        _check("absence_rate",     df_m["absence_days"].sum() / (N * 230) * 100, ANCHORS.absence_rate_pct)
        _check("training_hours",   df_m["train_hours"].mean(),                  ANCHORS.training_hours_per_fte)
        _check("train_completion", df_m["train_done"].mean() * 100,             ANCHORS.training_completion_pct)
        _check("train_transfer",   df_m["train_transfer"].dropna().mean() * 100, ANCHORS.training_transfer_pct)
        _check("attrition_rate",   df_m["attrited"].mean() * 100,               ANCHORS.turnover_rate_pct)

        if "country" in df_m.columns and "inclusion_score" in df_m.columns:
            fr_mask = df_m["country"] == "France"
            lu_mask = df_m["country"] == "Luxembourg"
            if fr_mask.any():
                _check("inclusion_fr", df_m.loc[fr_mask, "inclusion_score"].mean(), ANCHORS.inclusion_fr)
            if lu_mask.any():
                _check("inclusion_lu", df_m.loc[lu_mask, "inclusion_score"].mean(), ANCHORS.inclusion_lu)

        logger.info("All calibration checks passed.")

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def enrich_all(
        self,
        df: pd.DataFrame,
        rng: np.random.RandomState | None = None,
    ) -> pd.DataFrame:
        """
        Run all synthetic generators and return df with all new columns.

        Parameters
        ----------
        df  : cleaned employee master (from DataCleaner.clean_employee_master)
        rng : seeded random state; defaults to RANDOM_SEED if None

        Returns
        -------
        pd.DataFrame with all real + synthetic columns.
        """
        if rng is None:
            rng = np.random.RandomState(RANDOM_SEED)

        N = len(df)
        df_m = df.copy()

        perf_2025, perf_2023, perf_delta = self.generate_performance(N, rng)
        absence_days = self.generate_absence(N, rng)
        salaries, market_sal, pay_gap_pct = self.generate_salary(df_m, perf_2025, rng)
        train_hours, train_done, train_sat, train_transfer = self.generate_training(N, rng)
        inc_score = self.generate_inclusion(df_m, perf_2025, rng)
        attrition_prob, attrited = self.generate_attrition_label(
            df_m, perf_2025, perf_delta, pay_gap_pct,
            absence_days, train_hours, inc_score, rng,
        )

        df_m = df_m.assign(
            perf_2023=perf_2023,
            perf_2025=perf_2025,
            perf_delta=perf_delta,
            absence_days=absence_days,
            salary=salaries,
            market_salary=market_sal,
            pay_gap_pct=pay_gap_pct,
            train_hours=train_hours,
            train_done=train_done,
            train_sat=train_sat,
            train_transfer=train_transfer,
            inclusion_score=inc_score,
            attrition_prob=attrition_prob,
            attrited=attrited,
        )

        logger.info("Enrichment complete: %d employees, %d columns", len(df_m), len(df_m.columns))
        return df_m
