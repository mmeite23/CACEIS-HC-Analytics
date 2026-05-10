"""
XGBoost attrition model with SHAP explainability.

Trains, evaluates, and explains the CACEIS attrition model.
Includes a ModelCard dataclass for governance and GDPR Article 22 compliance.
Robustness checks verify that no protected attribute leakage has occurred.

Validated performance (see model_results.json):
  ROC-AUC:          0.784
  CV AUC (5-fold):  0.788 ± 0.007
  Precision@Top10%: 73.7%
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import ANCHORS, FEATURES, MODEL_PARAMS, RANDOM_SEED

logger = logging.getLogger(__name__)


# ── Result containers ──────────────────────────────────────────────────────────


@dataclass
class ModelResults:
    """Structured model evaluation output."""

    auc: float
    avg_precision: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    lift_at_10: float
    cv_auc_mean: float
    cv_auc_std: float
    n_train: int
    n_test: int
    n_attrited_test: int
    confusion_matrix: list
    classification_report_str: str


@dataclass
class ModelCard:
    """
    GDPR Art. 22 governance document for the attrition model.

    Automated individual decisions based on this model are prohibited.
    All risk scores require human oversight before any HR action.
    """

    model_name: str = "CACEIS Attrition Risk Scorer"
    version: str = "1.0.0"
    training_date: str = field(default_factory=lambda: str(date.today()))
    features_used: list[str] = field(default_factory=lambda: FEATURES)
    n_features: int = 17

    # Performance
    auc: float = ANCHORS.model_auc
    precision_at_top10: float = ANCHORS.precision_at_top10

    # Data
    training_data: str = "CACEIS employee master (real) + synthetic enrichment (calibrated)"
    synthetic_fields: list[str] = field(default_factory=lambda: [
        "perf_2025", "perf_delta", "absence_days", "pay_gap_pct",
        "train_hours", "inclusion_score", "log_salary",
        "underpaid", "no_training", "low_inclusion",
    ])

    # Intended use
    intended_use: str = (
        "Prioritise HR manager attention to at-risk employees. "
        "Scores are decision-support inputs, not automated decisions."
    )
    prohibited_uses: list[str] = field(default_factory=lambda: [
        "Automated dismissal or disciplinary action (GDPR Art. 22)",
        "Salary decisions without human review",
        "Any use that discriminates on protected attributes",
        "Use without informing the Works Council (BDES requirement)",
    ])

    # Exclusions (GDPR / bias mitigation)
    excluded_features: list[str] = field(default_factory=lambda: [
        "gender", "nationality", "age (raw)", "disability status",
        "union membership", "political opinion",
    ])

    limitations: list[str] = field(default_factory=lambda: [
        "AUC is inflated because attrition labels are synthetic (DGP), not real departures",
        "Performance/absence are synthetic — not truly individual-level data",
        "Country_tier may proxy nationality — replace with labour-market indicators",
        "Model trained on 2025 snapshot — recalibration needed annually",
        "Class imbalance (~5.5% attrition) means threshold 0.5 is arbitrary",
    ])

    oversight_requirement: str = (
        "CACEIS HR Business Partners must review each flagged employee before any action. "
        "Model output is advisory only. Contact: DPO@caceis.com for data access requests."
    )


# ── Model class ────────────────────────────────────────────────────────────────


class AttritionModel:
    """
    XGBoost attrition risk scorer with SHAP explainability.

    Example
    -------
    >>> model = AttritionModel()
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
    >>> model.train(X_tr, y_tr, X_te, y_te)
    >>> results = model.evaluate(X_te, y_te)
    >>> shap_df = model.compute_shap(X_te)
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = {**MODEL_PARAMS, **(params or {})}
        self._model: Optional[xgb.XGBClassifier] = None
        self._feature_names: list[str] = FEATURES
        self.model_card = ModelCard()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        apply_smote: bool = True,
    ) -> "AttritionModel":
        """
        Fit XGBoost with optional SMOTE oversampling on the training set.

        SMOTE is applied on training data only — never on validation/test.
        Early stopping is disabled to keep the CV stable; n_estimators in
        MODEL_PARAMS is already tuned for this dataset size.

        Returns self for method chaining.
        """
        if apply_smote:
            X_train, y_train = SMOTE(
                random_state=RANDOM_SEED, k_neighbors=5
            ).fit_resample(X_train, y_train)
            logger.info("SMOTE applied: %d training samples (balanced)", len(X_train))

        self._model = xgb.XGBClassifier(**self.params)
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("XGBoost training complete")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_full: Optional[np.ndarray] = None,
        y_full: Optional[np.ndarray] = None,
    ) -> ModelResults:
        """
        Evaluate model on the test set. Optionally run 5-fold CV on full data.

        Returns
        -------
        ModelResults dataclass with all metrics.
        """
        if self._model is None:
            raise RuntimeError("Call train() before evaluate()")

        y_prob = self._model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        sorted_idx = np.argsort(y_prob)[::-1]
        sorted_y = y_test[sorted_idx]

        def prec_at_k(k_pct: int) -> float:
            n_k = max(1, int(len(y_test) * k_pct / 100))
            return float(sorted_y[:n_k].mean())

        p5 = prec_at_k(5)
        p10 = prec_at_k(10)
        p20 = prec_at_k(20)
        lift10 = p10 / y_test.mean() if y_test.mean() > 0 else 0.0

        cv_mean = cv_std = 0.0
        if X_full is not None and y_full is not None:
            cv_scores = cross_val_score(
                self._model, X_full, y_full,
                cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
                scoring="roc_auc",
            )
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

        cm = confusion_matrix(y_test, y_pred).tolist()
        clf_report = classification_report(y_test, y_pred)

        logger.info(
            "Evaluation — AUC=%.4f | AP=%.4f | Precision@10%%=%.4f | Lift=%.1fx",
            auc, ap, p10, lift10,
        )
        logger.info("CV AUC: %.4f ± %.4f", cv_mean, cv_std)

        return ModelResults(
            auc=round(auc, 4),
            avg_precision=round(ap, 4),
            precision_at_5=round(p5, 4),
            precision_at_10=round(p10, 4),
            precision_at_20=round(p20, 4),
            lift_at_10=round(lift10, 2),
            cv_auc_mean=round(cv_mean, 4),
            cv_auc_std=round(cv_std, 4),
            n_train=int((y_test == 0).sum() + (y_test == 1).sum()),
            n_test=len(y_test),
            n_attrited_test=int(y_test.sum()),
            confusion_matrix=cm,
            classification_report_str=clf_report,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return attrition probabilities for input feature matrix."""
        if self._model is None:
            raise RuntimeError("Call train() before predict_proba()")
        return self._model.predict_proba(X)[:, 1]

    # ── SHAP explainability ───────────────────────────────────────────────────

    def compute_shap(
        self,
        X_test: np.ndarray,
        n_samples: int = 800,
    ) -> pd.DataFrame:
        """
        Compute SHAP values for the first n_samples test observations.

        Returns
        -------
        pd.DataFrame of shape (n_samples, 17) with SHAP values.
        Columns match self._feature_names.
        """
        if self._model is None:
            raise RuntimeError("Call train() before compute_shap()")

        import shap

        logger.info("Computing SHAP values for %d samples...", min(n_samples, len(X_test)))
        explainer = shap.TreeExplainer(self._model)
        X_sample = X_test[:n_samples]
        shap_values = explainer.shap_values(X_sample)

        df_shap = pd.DataFrame(shap_values, columns=self._feature_names)
        logger.info("SHAP computation complete")
        return df_shap

    def explain_individual(
        self,
        X_row: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> list[tuple[str, str, float, str]]:
        """
        Translate SHAP values for one employee into plain English.

        Returns
        -------
        list of (feature, direction, magnitude, plain_text) tuples
        sorted by |magnitude| descending. Used by the dashboard.
        """
        if self._model is None:
            raise RuntimeError("Call train() before explain_individual()")

        import shap

        feature_names = feature_names or self._feature_names
        explainer = shap.TreeExplainer(self._model)
        shap_vals = explainer.shap_values(X_row.reshape(1, -1))[0]

        PLAIN_TEXT = {
            "tenure_months":   ("short tenure = higher departure risk",   "long tenure = retention anchor"),
            "perf_delta":      ("declining performance = disengagement signal", "improving performance = engagement"),
            "pay_gap_pct":     ("underpaid vs market = push factor",       "fair/above market pay = retention"),
            "absence_days":    ("high absence = disengagement proxy",      "low absence = engagement"),
            "inclusion_score": ("low inclusion = belonging deficit",       "high inclusion = belonging driver"),
            "tenure_band":     ("high-risk tenure stage",                  "stable tenure stage"),
            "is_permanent":    ("temporary contract = structurally mobile", "permanent contract = stability"),
            "perf_2025":       ("low performance = dual flight risk",      "high performance = retained star"),
            "log_salary":      ("low salary level = retention risk",       "high salary = retention anchor"),
            "train_hours":     ("low training = disengagement",            "high training = investment signal"),
            "country_tier":    ("high-mobility country norms",             "low-mobility country norms"),
            "no_training":     ("no training at all = strong disengagement flag", "training engaged"),
            "underpaid":       ("underpaid vs market benchmark",           "fairly/well paid"),
            "low_inclusion":   ("low inclusion flag",                      "good inclusion environment"),
            "age_ordinal":     ("age context factor",                      "age context factor"),
            "degree_level":    ("education level = external market options", "education context"),
            "job_seniority":   ("junior role = more mobile",               "senior role = less mobile"),
        }

        results = []
        for fname, sv in sorted(zip(feature_names, shap_vals), key=lambda x: -abs(x[1])):
            direction = "increases risk" if sv > 0 else "decreases risk"
            texts = PLAIN_TEXT.get(fname, ("risk factor", "protective factor"))
            plain = texts[0] if sv > 0 else texts[1]
            results.append((fname, direction, round(float(sv), 4), plain))

        return results

    # ── Robustness checks ─────────────────────────────────────────────────────

    def run_robustness_checks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: Optional[pd.DataFrame] = None,
    ) -> dict[str, bool]:
        """
        Run 5 robustness checks and return pass/fail dict.

        Checks:
        1. Stability: CV AUC std < 0.015 (model is not unstable)
        2. Not-trivial: AUC > 0.60 (better than random)
        3. Leakage: attrition_prob not in feature matrix
        4. Protected attributes: gender/nationality not in feature matrix
        5. Score spread: not degenerate (std > 0.01)
        """
        if self._model is None:
            raise RuntimeError("Call train() before run_robustness_checks()")

        checks: dict[str, bool] = {}

        # 1. Stability
        cv_scores = cross_val_score(
            self._model, X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
        )
        checks["cv_stability"] = bool(cv_scores.std() < 0.015)

        # 2. Not-trivial
        checks["auc_above_baseline"] = bool(cv_scores.mean() > 0.60)

        # 3. Target leakage
        protected_leakage_cols = {"attrition_prob", "attrited"}
        checks["no_target_leakage"] = not bool(
            protected_leakage_cols.intersection(set(self._feature_names))
        )

        # 4. Protected attributes
        protected_attr = {"gender", "nationality", "age", "disability", "race", "religion"}
        checks["no_protected_attributes"] = not bool(
            protected_attr.intersection({f.lower() for f in self._feature_names})
        )

        # 5. Score spread
        probs = self.predict_proba(X)
        checks["score_not_degenerate"] = bool(probs.std() > 0.01)

        all_pass = all(checks.values())
        logger.info(
            "Robustness checks: %d/5 passed (%s)",
            sum(checks.values()),
            "ALL PASS" if all_pass else "SOME FAILED",
        )
        for name, result in checks.items():
            logger.info("  %-30s %s", name, "PASS" if result else "FAIL")

        return checks

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Serialise model and feature names to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "features": self._feature_names}, f)
        logger.info("Model saved: %s", path)

    def load(self, path: Path | str) -> "AttritionModel":
        """Load model and feature names from disk. Returns self."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._feature_names = data["features"]
        logger.info("Model loaded: %s", path)
        return self

    def save_results_json(
        self,
        results: ModelResults,
        shap_importance: Optional[pd.Series] = None,
        path: Path | str = "outputs/model_results.json",
    ) -> None:
        """Save model results and SHAP importance to JSON for dashboard consumption."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metrics": asdict(results),
            "model_card": asdict(self.model_card),
            "shap_importance": shap_importance.round(5).to_dict() if shap_importance is not None else {},
            "feature_names": self._feature_names,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Model results saved: %s", path)
