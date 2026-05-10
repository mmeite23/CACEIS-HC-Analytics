"""
Feature engineering for the CACEIS attrition model.

Produces a canonical 17-feature matrix in the exact order defined in
config.FEATURES. The order is frozen — do not modify without retraining
the model and updating model_results.json.

Feature groups:
  Real (7):      tenure_months, tenure_band, age_ordinal, is_permanent,
                 degree_level, country_tier, job_seniority
  Synthetic (10): perf_2025, perf_delta, absence_days, pay_gap_pct,
                  train_hours, inclusion_score, log_salary,
                  underpaid, no_training, low_inclusion
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import AGE_ORDINAL, COUNTRY_TIER, FEATURES, KPI_THRESHOLDS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds the 17-feature matrix for attrition modelling.

    Example
    -------
    >>> fe = FeatureEngineer()
    >>> df_feat = fe.add_tenure_features(df_m)
    >>> df_feat = fe.add_demographic_features(df_feat)
    >>> ... (chain all add_* methods) ...
    >>> X, y, names = fe.build_feature_matrix(df_feat)
    """

    # ── Individual feature builders ───────────────────────────────────────────

    def add_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute tenure_months and tenure_band (0–4 ordinal, 4=highest risk).

        tenure_band encoding:
          4 → < 6 months   (probationary, very high risk)
          3 → 6-18 months  (early, high risk)
          2 → 18m-3 years  (settling)
          1 → 3-7 years    (mid-career restlessness)
          0 → > 7 years    (stable / long-tenured)
        """
        df = df.copy()
        df["tenure_months"] = (df["tenure_years"].fillna(2) * 12).clip(lower=0)
        df["tenure_band"] = df["tenure_months"].apply(
            lambda m: 4 if m < 6 else 3 if m < 18 else 2 if m < 36 else 1 if m < 84 else 0
        )
        return df

    def add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode age_ordinal, is_permanent, degree_level.

        degree_level:
          4 → Master / Bac+5 / PhD
          3 → Bac+4 / Bachelor / Licence
          2 → Bac+2 / Bac+3
          1 → Bac (secondary)
          0 → Unknown / Below bac
        """
        df = df.copy()
        df["age_ordinal"] = df["age_range"].map(AGE_ORDINAL).fillna(3).astype(int)
        df["is_permanent"] = (df["contract_type"] == "Permanent contract").astype(int)

        def _degree(d: object) -> int:
            s = str(d).lower()
            if any(x in s for x in ["master", "bac+5", "phd", "doctor", "ingénieur"]):
                return 4
            if any(x in s for x in ["bac+4", "bachelor", "licence", "license"]):
                return 3
            if any(x in s for x in ["bac+2", "bac+3", "bts", "dut"]):
                return 2
            if "bac" in s:
                return 1
            return 0

        df["degree_level"] = df["degree"].apply(_degree)
        return df

    def add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode country_tier (0=low mobility, 1=medium, 2=high mobility).

        High-mobility countries (tier 2): Malaysia, Brazil, Colombia, Spain.
        Core hubs (tier 0): France, Luxembourg, Germany, Switzerland, Belgium,
        Netherlands.
        """
        df = df.copy()
        df["country_tier"] = df["country"].map(COUNTRY_TIER).fillna(1).astype(int)
        return df

    def add_role_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode job_seniority (0=junior, 1=mid, 2=senior/leadership).

        Senior keywords: head, director, chief, senior, group manager.
        Junior keywords: officer, analyst, assistant, trainee, coordinator.
        """
        df = df.copy()

        def _seniority(t: object) -> int:
            s = str(t).lower()
            if any(k in s for k in ["head", "director", "chief", "senior", "group manager"]):
                return 2
            if any(k in s for k in ["officer", "analyst", "assistant", "trainee", "coordinator"]):
                return 0
            return 1

        df["job_seniority"] = df["job_title"].apply(_seniority)
        return df

    def add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log_salary and underpaid flag (pay_gap_pct > 5%).
        """
        df = df.copy()
        df["log_salary"] = np.log(df["salary"].clip(lower=1))
        df["underpaid"] = (df["pay_gap_pct"] > KPI_THRESHOLDS["pay_gap_alert_pct"]).astype(int)
        return df

    def add_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary risk flags: no_training, low_inclusion, high_absence.

        These binary versions of continuous features help the tree model
        split on the most HR-actionable thresholds.
        """
        df = df.copy()
        df["no_training"] = (df["train_hours"] < KPI_THRESHOLDS["training_min_hours"]).astype(int)
        df["low_inclusion"] = (df["inclusion_score"] < 60).astype(int)
        df["high_absence"] = (df["absence_days"] > KPI_THRESHOLDS["absence_alert_days"]).astype(int)
        return df

    # ── Feature baseline (pre-model sanity check) ─────────────────────────────

    def feature_importance_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Spearman correlation of each FEATURES column with attrited.

        Prints a ranked table and returns a DataFrame for logging.
        This is a sanity check that features actually correlate with
        the outcome before fitting any model.
        """
        if "attrited" not in df.columns:
            logger.warning("feature_importance_baseline: 'attrited' column not found")
            return pd.DataFrame()

        rows = []
        for feat in FEATURES:
            if feat not in df.columns:
                continue
            corr, pval = stats.spearmanr(df[feat].fillna(0), df["attrited"])
            rows.append({"feature": feat, "spearman_r": corr, "p_value": pval})

        result = pd.DataFrame(rows).sort_values("spearman_r", key=abs, ascending=False)

        print("\nFeature baseline (Spearman ρ with attrition):")
        print(f"  {'Feature':<22} {'ρ':>8}   {'p-value':>10}")
        print("  " + "-" * 46)
        for _, row in result.iterrows():
            sig = "*" if row["p_value"] < 0.05 else " "
            print(f"  {row['feature']:<22} {row['spearman_r']:>8.4f} {sig}  {row['p_value']:>10.2e}")

        return result

    # ── Build feature matrix ──────────────────────────────────────────────────

    def build_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Apply all feature builders and return (X, y, feature_names).

        Validates that X has exactly 17 columns in the canonical FEATURES order.

        Returns
        -------
        X            : np.ndarray of shape (N, 17), float64, no NaNs
        y            : np.ndarray of shape (N,), int (0/1 attrition)
        feature_names: list of 17 feature name strings
        """
        df = (
            df.pipe(self.add_tenure_features)
              .pipe(self.add_demographic_features)
              .pipe(self.add_geographic_features)
              .pipe(self.add_role_features)
              .pipe(self.add_financial_features)
              .pipe(self.add_risk_flags)
        )

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns after engineering: {missing}\n"
                "Ensure enrich_all() was called before build_feature_matrix()."
            )

        df_ml = df[FEATURES + ["attrited"]].dropna(subset=FEATURES)
        X = df_ml[FEATURES].values.astype(np.float64)
        y = df_ml["attrited"].values.astype(int)

        assert X.shape[1] == 17, f"Expected 17 features, got {X.shape[1]}"
        assert not np.isnan(X).any(), "NaN values in feature matrix after dropna"

        logger.info(
            "Feature matrix: %d obs × %d features | %.1f%% attrited",
            X.shape[0], X.shape[1], y.mean() * 100,
        )
        return X, y, FEATURES
