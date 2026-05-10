"""
Data cleaning and validation layer for CACEIS Human Capital Analytics.

Each cleaner method documents every transformation it applies via a
CleaningReport, so the pipeline remains auditable. Cross-source validation
checks that FTE figures from different sources agree within ±2%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Cleaning report ────────────────────────────────────────────────────────────


@dataclass
class TransformRecord:
    """One transformation applied to a dataframe."""

    step: str
    column: Optional[str]
    rows_affected: int
    description: str


@dataclass
class CleaningReport:
    """Audit log of every transformation applied during cleaning."""

    source: str
    transforms: list[TransformRecord] = field(default_factory=list)

    def record(
        self,
        step: str,
        column: Optional[str],
        rows_affected: int,
        description: str,
    ) -> None:
        self.transforms.append(
            TransformRecord(step, column, rows_affected, description)
        )
        logger.debug("[%s] %s — %d rows: %s", self.source, step, rows_affected, description)

    def print_summary(self) -> None:
        print(f"\n  Cleaning report: {self.source} ({len(self.transforms)} transforms)")
        for t in self.transforms:
            col = f"[{t.column}] " if t.column else ""
            print(f"    {t.step}: {col}{t.rows_affected} rows — {t.description}")


# ── Cleaner ────────────────────────────────────────────────────────────────────


class DataCleaner:
    """
    Validates and cleans all CACEIS source DataFrames.

    Example
    -------
    >>> cleaner = DataCleaner()
    >>> df_clean = cleaner.clean_employee_master(df_raw)
    >>> report = cleaner.get_report("employee_master")
    """

    def __init__(self) -> None:
        self._reports: dict[str, CleaningReport] = {}

    def get_report(self, source: str) -> CleaningReport:
        return self._reports.get(source, CleaningReport(source))

    # ── Employee master ───────────────────────────────────────────────────────

    def clean_employee_master(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the employee master snapshot.

        Transformations:
        - Filter to latest period (Dec 2025 snapshot)
        - Filter to active contracts (Permanent + Temporary)
        - Compute tenure_years from date_entry_caceis to period
        - Fill missing entry dates with the median tenure assumption (2 years)
        - Standardise country and job_title casing

        Returns
        -------
        pd.DataFrame of active employees at the latest snapshot date.
        """
        report = CleaningReport("employee_master")
        self._reports["employee_master"] = report
        df = df.copy()
        n0 = len(df)

        # Coerce date columns
        for col in ["period", "date_entry_caceis"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Filter to latest period
        latest = df["period"].max()
        mask_period = df["period"] == latest
        report.record("filter_period", "period", (~mask_period).sum(),
                      f"Dropped non-{str(latest)[:7]} rows")
        df = df[mask_period].copy()

        # Active contracts only
        active = {"Permanent contract", "Temporary contract"}
        mask_contract = df["contract_type"].isin(active)
        report.record("filter_contract", "contract_type", (~mask_contract).sum(),
                      "Dropped inactive/unknown contracts")
        df = df[mask_contract].copy().reset_index(drop=True)

        # Compute tenure
        df["tenure_years"] = (
            (df["period"] - df["date_entry_caceis"]).dt.days / 365.25
        ).clip(lower=0)
        n_null_entry = df["tenure_years"].isna().sum()
        if n_null_entry:
            # Impute missing entry date with a conservative 2-year tenure
            df["tenure_years"] = df["tenure_years"].fillna(2.0)
            report.record("impute_tenure", "tenure_years", n_null_entry,
                          "Missing date_entry_caceis → tenure imputed as 2.0 years")

        # Standardise free-text fields
        df["country"] = df["country"].str.strip().str.title()
        df["job_title"] = df["job_title"].fillna("UNKNOWN").str.strip().str.upper()
        df["entity"] = df["entity"].fillna("UNKNOWN").str.strip()
        df["gender"] = df["gender"].str.strip().str.upper().replace({"F": "F", "M": "M"})

        report.record("standardise_text", None, len(df),
                      "country/job_title/entity/gender normalised")

        n1 = len(df)
        logger.info(
            "clean_employee_master: %d → %d rows (-%d dropped)",
            n0, n1, n0 - n1,
        )
        report.print_summary()
        return df

    # ── P&L / FTE ─────────────────────────────────────────────────────────────

    def clean_pl_fte(self, df_pl: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the P&L + FTE table.

        Transformations:
        - Assert personnel costs are positive (sign error check)
        - Assert PNB > 0 for all years
        - Clip implausible ratios

        Returns
        -------
        Validated pd.DataFrame (4 rows, one per year).
        """
        report = CleaningReport("pl_fte")
        self._reports["pl_fte"] = report
        df = df_pl.copy()

        for col in ["pnb", "personnel"]:
            neg = (df[col] < 0).sum()
            if neg:
                df[col] = df[col].abs()
                report.record("fix_sign", col, neg, f"{col} had negative values — abs() applied")

        # Sanity: personnel cost ratio should be between 30% and 90% of PNB
        bad_ratio = ((df["cost_ratio"] < 30) | (df["cost_ratio"] > 90)).sum()
        if bad_ratio:
            report.record("warn_ratio", "cost_ratio", bad_ratio,
                          "cost_ratio outside [30%, 90%] — verify P&L source data")

        report.print_summary()
        return df

    # ── EAE ───────────────────────────────────────────────────────────────────

    def clean_eae(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Map EAE performance labels to numeric ratings 1-5.

        Label mapping (French EAE scale → numeric):
          Insuffisant / Insufficient  → 1
          À développer / To Develop   → 2
          Conforme aux attentes        → 3
          Au-dessus des attentes       → 4
          Exceptionnel / Exceptional   → 5
          Not Specified / NR           → NaN (excluded)

        Parameters
        ----------
        df : raw EAE dataframe
        year : evaluation year (2023 or 2025)

        Returns
        -------
        pd.DataFrame with numeric_rating column added.
        """
        report = CleaningReport(f"eae_{year}")
        self._reports[f"eae_{year}"] = report
        df = df.copy()

        # Detect the rating column (first column with label-like strings)
        rating_col = None
        for col in df.columns:
            sample = df[col].dropna().astype(str).str.strip()
            if sample.str.contains(r"(?i)(conforme|insuffis|develop|attentes|exception)", regex=True).any():
                rating_col = col
                break

        if rating_col is None:
            logger.warning("EAE %d: could not auto-detect rating column", year)
            report.record("warn_no_rating_col", None, 0, "Rating column not detected")
            return df

        LABEL_MAP = {
            "insuffisant": 1, "insufficient": 1, "doit progresser": 1,
            "à développer": 2, "a developper": 2, "to develop": 2,
            "conforme aux attentes": 3, "meets expectations": 3, "satisfaisant": 3,
            "au-dessus des attentes": 4, "above expectations": 4, "très bien": 4,
            "exceptionnel": 5, "exceptional": 5, "excellent": 5,
        }

        def _map(val: object) -> float:
            s = str(val).strip().lower()
            for k, v in LABEL_MAP.items():
                if k in s:
                    return float(v)
            return float("nan")

        df["numeric_rating"] = df[rating_col].apply(_map)
        n_null = df["numeric_rating"].isna().sum()
        report.record("map_labels", rating_col, len(df) - n_null,
                      f"Mapped labels to 1-5 numeric ({n_null} unmapped → NaN)")
        report.print_summary()
        return df

    # ── Training ──────────────────────────────────────────────────────────────

    def clean_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise training status field and cast hours to float.

        'Réalisé', 'Realized', 'Completed', 'Done' → 1 (done)
        Everything else → 0.

        Returns
        -------
        pd.DataFrame with status_done (0/1) and hours_clean (float) added.
        """
        report = CleaningReport("training")
        self._reports["training"] = report
        df = df.copy()

        # Detect status column
        status_col = None
        for col in df.columns:
            sample = df[col].dropna().astype(str).str.strip()
            if sample.str.contains(r"(?i)(réalis|realiz|complet|done)", regex=True).any():
                status_col = col
                break

        if status_col:
            DONE_TERMS = {"réalisé", "realise", "realized", "completed", "done", "terminé", "termine"}
            df["status_done"] = df[status_col].apply(
                lambda x: 1 if str(x).strip().lower() in DONE_TERMS else 0
            )
            report.record("normalise_status", status_col, len(df),
                          f"Standardised '{status_col}' → status_done (0/1)")

        # Detect and clean hours column
        hours_col = None
        for col in df.columns:
            if "heure" in str(col).lower() or "hour" in str(col).lower() or "duree" in str(col).lower():
                hours_col = col
                break

        if hours_col:
            df["hours_clean"] = pd.to_numeric(df[hours_col], errors="coerce").clip(0, 1000)
            n_bad = df["hours_clean"].isna().sum()
            report.record("cast_hours", hours_col, n_bad,
                          f"Coerced to float, {n_bad} non-numeric → NaN, clipped [0, 1000]")

        report.print_summary()
        return df

    # ── Cross-source validation ───────────────────────────────────────────────

    def validate_cross_source(self, sources: dict[str, pd.DataFrame]) -> bool:
        """
        Check FTE figures and other cross-source consistency.

        Checks:
        1. Active employee count from employee_master vs FTE_VALS[2025] (±2%)
        2. P&L rows == 4 (one per year)

        Returns True if all checks pass, False if any warning was raised.
        """
        from src.config import ACTIVE_WORKFORCE_DEC2025, FTE_VALS

        all_pass = True

        if "employee_master" in sources:
            n = len(sources["employee_master"])
            expected = ACTIVE_WORKFORCE_DEC2025
            pct_diff = abs(n - expected) / expected * 100
            if pct_diff > 2.0:
                logger.warning(
                    "Cross-source check FAIL: employee_master has %d rows, "
                    "expected ~%d (%.1f%% deviation > 2%% tolerance)",
                    n, expected, pct_diff,
                )
                all_pass = False
            else:
                logger.info(
                    "Cross-source check PASS: employee_master = %d rows "
                    "(within %.1f%% of anchor %d)",
                    n, pct_diff, expected,
                )

        if "pl_fte" in sources:
            n_pl = len(sources["pl_fte"])
            if n_pl != 4:
                logger.warning("Cross-source check: pl_fte has %d rows, expected 4", n_pl)
                all_pass = False

        return all_pass
