"""
End-to-end pipeline orchestrator for CACEIS Human Capital Analytics.


Runs: INGEST → CLEAN → ENRICH → FEATURES → KPIs → MODEL → CLUSTERING
Saves: model_results.json, eda_enriched.png, model_results.png
Returns a structured dict consumed by the dashboard.

Usage
-----
  # From Python:
  from src.pipeline import run_pipeline
  results = run_pipeline(data_path="./data/", output_dir="./outputs/")

  # From CLI:
  python src/pipeline.py --data-path ./data/ --output ./outputs/
  python src/pipeline.py --help
"""

from __future__ import annotations

import sys
import os

# Allow running as `python src/pipeline.py` from the project root
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server-side rendering

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.clean import DataCleaner
from src.clustering import InclusionClusterDetector
from src.config import (
    DATA_PATH,
    FEATURES,
    OUTPUT_DIR,
    PALETTE,
    RANDOM_SEED,
)
from src.enrich import SyntheticEnricher
from src.features import FeatureEngineer
from src.ingest import CACEISDataLoader
from src.kpis import KPICalculator
from src.model import AttritionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Chart generation ───────────────────────────────────────────────────────────


def _generate_eda_charts(df_m: pd.DataFrame, output_dir: Path) -> None:
    """Produce and save the 9-panel EDA figure to output_dir/eda_enriched.png."""
    NAVY = PALETTE["navy"]
    BLUE = PALETTE["blue"]
    GOLD = PALETTE["gold"]
    RED = PALETTE["red"]
    GREEN = PALETTE["green"]
    MUTED = PALETTE["muted"]

    plt.rcParams.update({
        "figure.dpi": 130, "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FC", "axes.edgecolor": "#DDE5F0",
        "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#E8EDF5", "grid.linestyle": "--", "grid.alpha": 0.7,
    })

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        "CACEIS Human Capital — Enriched EDA (Real + Synthetic Data)",
        fontsize=14, fontweight="bold", color=NAVY, y=1.01,
    )

    labels_p = ["Insufficient", "To Develop", "Meets Exp.", "Above Exp.", "Exceptional"]
    x = np.arange(1, 6); w = 0.38

    # 1. Performance distribution 2023 vs 2025
    ax = axes[0, 0]
    d23 = df_m["perf_2023"].value_counts(normalize=True).sort_index() * 100
    d25 = df_m["perf_2025"].value_counts(normalize=True).sort_index() * 100
    b1 = ax.bar(x - w / 2, [d23.get(i, 0) for i in x], w, label="2023", color=BLUE, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, [d25.get(i, 0) for i in x], w, label="2025", color=GOLD, alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels_p, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("%"); ax.set_title("Performance Distribution 2023 vs 2025", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.5)

    # 2. Tenure × attrition
    ax = axes[0, 1]
    bins_t = [0, 0.5, 1.5, 3, 5, 8, 12, 20, 50]
    labs_t = ["<6m", "6-18m", "18m-3y", "3-5y", "5-8y", "8-12y", "12-20y", ">20y"]
    df_m["ten_bin"] = pd.cut(df_m["tenure_years"], bins=bins_t, labels=labs_t)
    t_agg = df_m.groupby("ten_bin", observed=True)["attrited"].agg(["mean", "count"])
    cols_t = [RED if v > 0.12 else GOLD if v > 0.06 else GREEN for v in t_agg["mean"]]
    ax.bar(t_agg.index, t_agg["mean"] * 100, color=cols_t, edgecolor="white")
    ax.axhline(df_m["attrited"].mean() * 100, color=NAVY, linestyle="--", linewidth=1.5,
               label=f"Avg {df_m['attrited'].mean()*100:.1f}%")
    ax.set_ylabel("Attrition Rate (%)"); ax.set_title("Attrition Rate by Tenure Band", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.5)

    # 3. Pay gap × attrition
    ax = axes[0, 2]
    bins_pg = [-40, -10, -5, 0, 5, 10, 20, 60]
    labs_pg = ["<-10%", "-10 to -5", "-5 to 0", "0-5%", "5-10%", "10-20%", ">20%"]
    df_m["pg_bin"] = pd.cut(df_m["pay_gap_pct"], bins=bins_pg, labels=labs_pg)
    pg_agg = df_m.groupby("pg_bin", observed=True)["attrited"].agg(["mean", "count"])
    cols_pg = [GREEN, GREEN, GREEN, GOLD, GOLD, RED, RED][: len(pg_agg)]
    ax.bar(pg_agg.index, pg_agg["mean"] * 100, color=cols_pg, edgecolor="white")
    ax.set_xlabel("Pay vs. Market (%)"); ax.set_ylabel("Attrition Rate (%)")
    ax.set_title("Attrition Rate by Pay Gap vs. Market", fontweight="bold")
    ax.grid(axis="y", alpha=0.5); plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

    # 4. Inclusion by country
    ax = axes[1, 0]
    inc_c = df_m.groupby("country")["inclusion_score"].agg(["mean", "count"])
    inc_c = inc_c[inc_c["count"] >= 30].sort_values("mean")
    cols_ic = [RED if v < 65 else GOLD if v < 70 else GREEN for v in inc_c["mean"]]
    bars = ax.barh(inc_c.index, inc_c["mean"], color=cols_ic, edgecolor="white")
    ax.axvline(72, color=NAVY, linestyle="--", linewidth=1.5, label="Group avg 72%")
    ax.axvline(65, color=RED, linestyle=":", linewidth=1.5, label="Alert < 65%")
    ax.set_xlabel("Inclusion Score (%)"); ax.set_title("Inclusion Score by Country", fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="x", alpha=0.5); ax.set_xlim(52, 88)

    # 5. Absence × performance heatmap
    ax = axes[1, 1]
    df_m["abs_bin"] = pd.cut(df_m["absence_days"], bins=[-1, 0, 5, 15, 30, 100],
                              labels=["0d", "1-5d", "6-15d", "16-30d", ">30d"])
    heat = df_m.groupby(["abs_bin", "perf_2025"], observed=True).size().unstack(fill_value=0)
    heat_pct = heat.div(heat.sum(axis=1), axis=0) * 100
    sns.heatmap(heat_pct, ax=ax, cmap="RdYlGn", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="white", cbar_kws={"label": "%"})
    ax.set_title("Absence Days vs Performance Rating (%)", fontweight="bold")
    ax.set_xlabel("Performance 2025"); ax.set_ylabel("Absence Days")

    # 6. Training × attrition
    ax = axes[1, 2]
    df_m["tr_bin"] = pd.cut(df_m["train_hours"], bins=[-1, 0, 5, 15, 30, 200],
                             labels=["None", "1-5h", "6-15h", "16-30h", ">30h"])
    tr_agg = df_m.groupby("tr_bin", observed=True)["attrited"].agg(["mean", "count"])
    cols_tr = [RED if v > 0.10 else GOLD if v > 0.06 else GREEN for v in tr_agg["mean"]]
    ax.bar(tr_agg.index, tr_agg["mean"] * 100, color=cols_tr, edgecolor="white")
    ax.set_ylabel("Attrition Rate (%)"); ax.set_title("Attrition Rate by Training Engagement", fontweight="bold")
    ax.grid(axis="y", alpha=0.5)

    # 7. Correlation heatmap
    ax = axes[2, 0]
    corr_cols = ["attrited", "perf_2025", "perf_delta", "absence_days",
                 "pay_gap_pct", "train_hours", "inclusion_score", "log_salary", "tenure_years"]
    corr_labs = ["Attrited", "Perf", "Perf Trend", "Absence", "Pay Gap",
                 "Training", "Inclusion", "Log Salary", "Tenure"]
    available = [c for c in corr_cols if c in df_m.columns]
    cmap_div = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(df_m[available].corr(), ax=ax, cmap=cmap_div, center=0,
                annot=True, fmt=".2f", linewidths=0.5, linecolor="white",
                square=True, annot_kws={"size": 8},
                xticklabels=corr_labs[:len(available)],
                yticklabels=corr_labs[:len(available)])
    ax.set_title("HR Metrics Correlation Matrix", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    # 8. Multi-risk bubble map
    ax = axes[2, 1]
    samp = df_m.sample(min(600, len(df_m)), random_state=RANDOM_SEED)
    sc = ax.scatter(samp["absence_days"], samp["pay_gap_pct"],
                    c=samp["attrition_prob"], s=samp["tenure_band"] * 30 + 15,
                    cmap="RdYlGn_r", alpha=0.55, edgecolors="white", linewidth=0.2,
                    vmin=0, vmax=0.4)
    plt.colorbar(sc, ax=ax, label="Attrition Probability")
    ax.axhline(5, color=GOLD, linestyle="--", linewidth=1.2, label="Pay gap alert (5%)")
    ax.axvline(15, color=GOLD, linestyle=":", linewidth=1.2, label="Absence alert (15d)")
    ax.set_xlabel("Absence Days"); ax.set_ylabel("Pay Gap vs. Market (%)")
    ax.set_title("Multi-Risk Profile Map\n(bubble = tenure risk, color = attrition prob)", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")

    # 9. Attrition by country
    ax = axes[2, 2]
    att_c = (
        df_m.groupby("country")["attrited"]
        .agg(["mean", "count"])
        .query("count >= 30")
        .sort_values("mean", ascending=False)
        .head(12)
    )
    cols_att = [RED if v > 0.10 else GOLD if v > 0.06 else GREEN for v in att_c["mean"]]
    ax.barh(att_c.index[::-1], att_c["mean"][::-1] * 100, color=cols_att[::-1], edgecolor="white")
    ax.axvline(df_m["attrited"].mean() * 100, color=NAVY, linestyle="--", linewidth=1.5,
               label=f"Avg {df_m['attrited'].mean()*100:.1f}%")
    ax.set_xlabel("Attrition Rate (%)"); ax.set_title("Attrition Rate by Country (top 12)", fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="x", alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / "eda_enriched.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()
    logger.info("EDA chart saved: %s", out_path)


def _generate_model_charts(
    model: AttritionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shap_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Produce and save the model evaluation figure to output_dir/model_results.png."""
    from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

    NAVY = PALETTE["navy"]
    BLUE = PALETTE["blue"]
    GOLD = PALETTE["gold"]
    RED = PALETTE["red"]
    GREEN = PALETTE["green"]
    MUTED = PALETTE["muted"]

    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CACEIS Attrition Model — XGBoost + SHAP Results",
                 fontsize=14, fontweight="bold", color=NAVY)

    # ROC curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.fill_between(fpr, tpr, alpha=0.15, color=GOLD)
    ax.plot(fpr, tpr, color=GOLD, linewidth=2.5, label=f"XGBoost (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (0.500)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve", fontweight="bold"); ax.legend(fontsize=10); ax.grid(alpha=0.5)

    # PR curve
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.fill_between(rec, prec, alpha=0.15, color=BLUE)
    ax.plot(rec, prec, color=BLUE, linewidth=2.5, label=f"XGBoost (AP={ap:.3f})")
    ax.axhline(y_test.mean(), linestyle="--", color=MUTED, linewidth=1.5,
               label=f"Baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold"); ax.legend(fontsize=9); ax.grid(alpha=0.5)

    # Precision@K
    ax = axes[0, 2]
    s_idx = np.argsort(y_prob)[::-1]; s_y = y_test[s_idx]
    ks = np.arange(1, 31)
    pk = [s_y[: max(1, int(len(y_test) * k / 100))].mean() for k in ks]
    ax2 = ax.twinx()
    ax.plot(ks, np.array(pk) * 100, color=GOLD, linewidth=2.5, marker="o", markersize=4)
    ax2.plot(ks, [p / y_test.mean() for p in pk], color=BLUE, linewidth=1.5, linestyle="--")
    ax.axhline(y_test.mean() * 100, linestyle=":", color=MUTED, linewidth=1.5, label="Random baseline")
    ax.set_xlabel("Top K% Flagged"); ax.set_ylabel("Precision (%)", color=GOLD)
    ax2.set_ylabel("Lift", color=BLUE)
    ax.set_title("Precision@K & Lift", fontweight="bold")
    ax.tick_params(axis="y", labelcolor=GOLD); ax2.tick_params(axis="y", labelcolor=BLUE)
    ax.grid(alpha=0.5)

    # SHAP bar
    ax = axes[1, 0]
    shap_mean = shap_df.abs().mean().sort_values()
    top12 = shap_mean.tail(12)
    cols_shap = [RED if v > top12.quantile(0.75) else GOLD if v > top12.quantile(0.5) else BLUE for v in top12.values]
    bars = ax.barh(top12.index, top12.values, color=cols_shap, edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|"); ax.set_title("Feature Importance (SHAP)", fontweight="bold")
    ax.grid(axis="x", alpha=0.5)

    # Score distribution
    ax = axes[1, 1]
    ax.hist(y_prob[y_test == 0], bins=50, alpha=0.65, color=GREEN, density=True,
            label=f"Retained (n={int((y_test==0).sum()):,})", edgecolor="white", linewidth=0.2)
    ax.hist(y_prob[y_test == 1], bins=50, alpha=0.75, color=RED, density=True,
            label=f"Attrited (n={int((y_test==1).sum()):,})", edgecolor="white", linewidth=0.2)
    ax.axvline(0.5, color=NAVY, linestyle="--", linewidth=2, label="Threshold 0.5")
    ax.set_xlabel("Predicted Attrition Probability"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Outcome", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.5)

    # SHAP waterfall — highest risk profile
    ax = axes[1, 2]
    top_idx = int(np.argmax(y_prob))
    sv = shap_df.iloc[min(top_idx, len(shap_df) - 1)]
    sv_sorted = sv.sort_values()
    cols_wf = [RED if v > 0 else GREEN for v in sv_sorted.values]
    ax.barh(sv_sorted.index, sv_sorted.values, color=cols_wf, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set_xlabel("SHAP Value (log-odds)")
    ax.set_title(
        f"Risk Decomposition — Highest Risk Profile\n(Predicted prob: {y_prob[top_idx]:.1%})",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / "model_results.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()
    logger.info("Model chart saved: %s", out_path)


# ── Pipeline orchestrator ──────────────────────────────────────────────────────


def run_pipeline(
    data_path: str | Path = DATA_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    validate: bool = True,
    seed: int = RANDOM_SEED,
    skip_model: bool = False,
) -> dict[str, Any]:
    """
    Run the full CACEIS HC Analytics pipeline end-to-end.

    Stages
    ------
    1. INGEST   — load all source files
    2. CLEAN    — validate and standardise
    3. ENRICH   — generate calibrated synthetic fields
    4. FEATURES — build 17-feature matrix
    5. KPIs     — compute 6 KPI metrics
    6. MODEL    — train XGBoost + SHAP
    7. CLUSTER  — D&I risk detection

    Returns
    -------
    dict with keys: kpis, model_metrics, shap_importance, cluster_risks, df_m
    """
    start_total = time.time()
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    timings: dict[str, float] = {}

    # ── Stage 1: INGEST ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — INGEST")
    t0 = time.time()
    loader = CACEISDataLoader(data_path)
    sources = loader.load_all()
    timings["ingest"] = time.time() - t0
    logger.info("Ingest complete in %.1fs", timings["ingest"])

    if "employee_master" not in sources:
        raise RuntimeError(
            "employee_master could not be loaded. "
            "Place Data.xlsx in the data/ directory and retry."
        )

    # ── Stage 2: CLEAN ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — CLEAN")
    t0 = time.time()
    cleaner = DataCleaner()
    df = cleaner.clean_employee_master(sources["employee_master"])
    df_pl = cleaner.clean_pl_fte(sources["pl_fte"]) if "pl_fte" in sources else None

    if validate and df_pl is not None:
        cleaner.validate_cross_source({"employee_master": df, "pl_fte": df_pl})

    timings["clean"] = time.time() - t0
    logger.info("Clean complete in %.1fs | %d active employees", timings["clean"], len(df))

    # ── Stage 3: ENRICH ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — ENRICH (synthetic calibration)")
    t0 = time.time()
    enricher = SyntheticEnricher()
    df_m = enricher.enrich_all(df, rng)

    if validate:
        enricher.validate_calibration(df_m)

    timings["enrich"] = time.time() - t0
    logger.info("Enrich complete in %.1fs", timings["enrich"])

    # ── Stage 4: FEATURES ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — FEATURE ENGINEERING")
    t0 = time.time()
    fe = FeatureEngineer()
    df_m = (
        df_m.pipe(fe.add_tenure_features)
            .pipe(fe.add_demographic_features)
            .pipe(fe.add_geographic_features)
            .pipe(fe.add_role_features)
            .pipe(fe.add_financial_features)
            .pipe(fe.add_risk_flags)
    )
    X, y, feature_names = fe.build_feature_matrix(df_m)
    logger.info("Feature baseline correlations:")
    fe.feature_importance_baseline(df_m)
    timings["features"] = time.time() - t0
    logger.info("Features complete in %.1fs | %d obs × %d features", timings["features"], *X.shape)

    # ── Stage 5: KPIs ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 — KPI COMPUTATION")
    t0 = time.time()
    calc = KPICalculator()
    kpi_inputs: dict = {"df_m": df_m}
    if df_pl is not None:
        kpi_inputs["df_pl"] = df_pl
    else:
        # Fall back to synthetic P&L using verified anchors
        from src.config import ANCHORS, FTE_VALS, YEARS
        df_pl_synthetic = pd.DataFrame({
            "year": YEARS,
            "pnb": [2_100_000, 2_500_000, 2_700_000, 2_097_000],
            "personnel": [750_000, 780_000, 790_000, 790_000],
            "training": [6_000, 8_500, 9_000, 9_200],
            "fte": [FTE_VALS[y] for y in YEARS],
        })
        df_pl_synthetic["hc_roi"] = (df_pl_synthetic["pnb"] - df_pl_synthetic["personnel"]) / df_pl_synthetic["personnel"] * 100
        df_pl_synthetic["rev_per_fte"] = df_pl_synthetic["pnb"] / df_pl_synthetic["fte"]
        df_pl_synthetic["cost_ratio"] = df_pl_synthetic["personnel"] / df_pl_synthetic["pnb"] * 100
        df_pl_synthetic["train_per_fte"] = df_pl_synthetic["training"] / df_pl_synthetic["fte"] * 1000
        kpi_inputs["df_pl"] = df_pl_synthetic
        logger.warning("P&L not loaded — using synthetic fallback values")

    kpi_results = calc.compute_all(
        df_pl=kpi_inputs["df_pl"],
        df_m=kpi_inputs["df_m"],
        df_abs=sources.get("absenteeism"),
        df_to=sources.get("turnover"),
    )
    calc.print_dashboard(kpi_results)
    timings["kpis"] = time.time() - t0
    logger.info("KPIs complete in %.1fs", timings["kpis"])

    # ── Stage 6: MODEL ────────────────────────────────────────────────────────
    model_results_obj = None
    shap_df = None
    shap_importance = None

    if not skip_model:
        logger.info("=" * 60)
        logger.info("STAGE 6 — MODEL TRAINING (XGBoost + SHAP)")
        t0 = time.time()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

        model = AttritionModel()
        model.train(X_tr, y_tr, X_te, y_te)
        model_results_obj = model.evaluate(X_te, y_te, X_full=X, y_full=y)

        shap_df = model.compute_shap(X_te, n_samples=800)
        shap_importance = shap_df.abs().mean().sort_values(ascending=False)

        robustness = model.run_robustness_checks(X, y)

        # Save artefacts
        model.save(output_dir / "attrition_model.pkl")
        model.save_results_json(model_results_obj, shap_importance, output_dir / "model_results.json")

        _generate_model_charts(model, X_te, y_te, shap_df, output_dir)

        timings["model"] = time.time() - t0
        logger.info(
            "Model complete in %.1fs | AUC=%.4f | P@10%%=%.4f",
            timings["model"], model_results_obj.auc, model_results_obj.precision_at_10,
        )

    # ── Stage 7: CLUSTERING ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 7 — D&I CLUSTER DETECTION")
    t0 = time.time()
    detector = InclusionClusterDetector()
    cluster_risks = detector.detect_clusters()
    flagged = detector.auto_flag_risks()
    logger.info("D&I risks: %d entities flagged (watch + critical)", len(flagged))
    if flagged:
        logger.info("  Flagged: %s", ", ".join(flagged))
    timings["clustering"] = time.time() - t0

    # ── EDA charts ────────────────────────────────────────────────────────────
    logger.info("Generating EDA charts...")
    _generate_eda_charts(df_m, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", total_time)
    for stage, t in timings.items():
        logger.info("  %-12s %.1fs", stage, t)

    return {
        "kpis": kpi_results,
        "model_metrics": model_results_obj,
        "shap_importance": shap_importance.to_dict() if shap_importance is not None else {},
        "cluster_risks": {e: {"risk": r.risk_level, "score": r.score, "flagged_dims": r.flagged_dimensions}
                          for e, r in cluster_risks.items()},
        "df_m": df_m,
        "timings": timings,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CACEIS Human Capital Analytics — full pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to CACEIS source data directory")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Path to output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Run calibration and cross-source validation checks")
    parser.add_argument("--no-validate", action="store_false", dest="validate",
                        help="Skip validation checks")
    parser.add_argument("--skip-model", action="store_true", default=False,
                        help="Skip model training (faster, for KPI-only runs)")
    args = parser.parse_args()

    results = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output,
        validate=args.validate,
        seed=args.seed,
        skip_model=args.skip_model,
    )

    print(f"\nPipeline complete. Outputs written to: {args.output}")
    if results["model_metrics"]:
        print(f"  Model AUC:            {results['model_metrics'].auc:.4f}")
        print(f"  Precision@Top10%%:    {results['model_metrics'].precision_at_10:.4f}")
    print(f"  Flagged D&I entities: {sum(1 for r in results['cluster_risks'].values() if r['risk'] in ('critical','watch'))}")


if __name__ == "__main__":
    main()
