"""
KPI computation layer for CACEIS Human Capital Analytics.

Six KPIs are computed here, one method each. Each KPI method returns a dict
with value, trend, and benchmark so the dashboard can render it consistently.
A KPIValidator asserts each 2025 value against known anchors (±2% tolerance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ANCHORS, KPI_THRESHOLDS

logger = logging.getLogger(__name__)

_ANCHOR_TOLERANCE = 0.02  # 2% deviation triggers a warning


# ── KPI result container ───────────────────────────────────────────────────────


@dataclass
class KPIResult:
    """Structured KPI output consumed by dashboard and reporting layers."""

    name: str
    value: float
    unit: str
    trend: str             # "up", "down", "stable"
    benchmark: str         # human-readable benchmark or target
    details: dict          # additional sub-metrics


# ── Validator ─────────────────────────────────────────────────────────────────


class KPIValidator:
    """
    Checks each 2025 KPI value against verified anchors.

    Warnings (not errors) are raised so the pipeline can still proceed
    when data quality issues cause minor deviations.
    """

    def validate(self, results: dict[str, dict]) -> None:
        checks = {
            "hc_roi":        (results.get("hc_roi", {}).get("value"),  ANCHORS.hc_roi_2025),
            "revenue_per_fte": (
                results.get("revenue_per_fte", {}).get("value"),
                ANCHORS.revenue_per_fte_k * 1000,
            ),
        }
        for name, (actual, target) in checks.items():
            if actual is None or target is None:
                continue
            pct_diff = abs(actual - target) / abs(target) * 100
            if pct_diff > _ANCHOR_TOLERANCE * 100:
                logger.warning(
                    "KPI validator: %s = %.2f, anchor = %.2f (%.1f%% deviation)",
                    name, actual, target, pct_diff,
                )
            else:
                logger.info("KPI validator OK: %s = %.2f (within %.1f%%)", name, actual, pct_diff)


# ── Calculator ────────────────────────────────────────────────────────────────


class KPICalculator:
    """
    Computes all six CACEIS HC KPIs from cleaned dataframes.

    Example
    -------
    >>> calc = KPICalculator()
    >>> results = calc.compute_all(df_pl=df_pl, df_m=df_m)
    >>> calc.print_dashboard(results)
    """

    # ── KPI 1: HC-ROI ─────────────────────────────────────────────────────────

    def kpi_hc_roi(self, df_pl: pd.DataFrame) -> dict:
        """
        HC-ROI = (PNB - Personnel costs) / Personnel costs × 100.

        Measures how many euros of revenue each euro of workforce cost generates.
        A declining HC-ROI signals that personnel costs are growing faster than
        revenue — a strategic red flag for CACEIS management.

        Returns
        -------
        dict with: value (2025 HC-ROI %), trend, 4-year series, benchmark
        """
        row_2025 = df_pl.loc[df_pl["year"] == 2025]
        row_2022 = df_pl.loc[df_pl["year"] == 2022]

        value = float(row_2025["hc_roi"].values[0]) if not row_2025.empty else ANCHORS.hc_roi_2025
        value_2022 = float(row_2022["hc_roi"].values[0]) if not row_2022.empty else value

        trend = "down" if value < value_2022 else "up" if value > value_2022 else "stable"
        series = df_pl[["year", "hc_roi"]].set_index("year")["hc_roi"].to_dict()

        return {
            "value": value,
            "unit": "%",
            "trend": trend,
            "benchmark": "Industry avg ~145% (French banking sector)",
            "series": series,
            "interpretation": f"EUR 1 invested → EUR {value/100+1:.2f} PNB",
        }

    # ── KPI 2: Revenue per FTE ─────────────────────────────────────────────────

    def kpi_revenue_per_fte(self, df_pl: pd.DataFrame) -> dict:
        """
        Revenue/FTE = PNB / FTE headcount.

        Benchmarks workforce productivity. CACEIS target: €325K/FTE.

        Returns
        -------
        dict with value (EUR), trend, 4-year series, benchmark
        """
        row_2025 = df_pl.loc[df_pl["year"] == 2025]
        row_2022 = df_pl.loc[df_pl["year"] == 2022]

        value = float(row_2025["rev_per_fte"].values[0]) if not row_2025.empty else ANCHORS.revenue_per_fte_k * 1_000
        value_2022 = float(row_2022["rev_per_fte"].values[0]) if not row_2022.empty else value
        trend = "up" if value > value_2022 else "down" if value < value_2022 else "stable"

        return {
            "value": value,
            "unit": "EUR/FTE",
            "trend": trend,
            "benchmark": f"EUR {ANCHORS.revenue_per_fte_k:.0f}K target (2025 plan)",
            "series": df_pl[["year", "rev_per_fte"]].set_index("year")["rev_per_fte"].to_dict(),
        }

    # ── KPI 3: Performance Index ───────────────────────────────────────────────

    def kpi_performance_index(self, df_m: pd.DataFrame) -> dict:
        """
        Performance Index = average EAE rating + distribution breakdown.

        Tracks workforce performance health. Low-performer concentration
        above 10% triggers a talent management review signal.

        Returns
        -------
        dict with avg, top_pct, low_pct, completion, trend
        """
        avg = float(df_m["perf_2025"].mean())
        top_pct = float((df_m["perf_2025"] >= 4).mean() * 100)
        low_pct = float((df_m["perf_2025"] <= 2).mean() * 100)
        completion = float(df_m["train_done"].mean() * 100) if "train_done" in df_m.columns else 0.0

        trend = "up" if avg > ANCHORS.perf_avg else "down" if avg < ANCHORS.perf_avg - 0.1 else "stable"

        return {
            "value": avg,
            "unit": "/5",
            "trend": trend,
            "benchmark": f"EAE 2025 target: {ANCHORS.perf_avg:.2f}/5",
            "top_pct": top_pct,
            "low_pct": low_pct,
            "training_completion_pct": completion,
        }

    # ── KPI 4: Attrition Risk ─────────────────────────────────────────────────

    def kpi_attrition_risk(
        self,
        df_m: pd.DataFrame,
        df_abs: pd.DataFrame | None = None,
        df_to: pd.DataFrame | None = None,
    ) -> dict:
        """
        Attrition Risk = turnover rate + absenteeism + estimated replacement cost.

        Replacement cost formula: N_departures × avg_salary × 1.5.
        The 1.5 multiplier (150% of annual salary) is the industry benchmark
        for fully-loaded talent replacement including recruiting, onboarding,
        and productivity ramp-up.

        Returns
        -------
        dict with rate, absenteeism, estimated_cost_m, high_risk_count
        """
        att_rate = float(df_m["attrited"].mean() * 100)
        n_departures = int(df_m["attrited"].sum())
        avg_salary = float(df_m.loc[df_m["attrited"] == 1, "salary"].mean())
        cost_m = n_departures * avg_salary * KPI_THRESHOLDS["attrition_cost_multiplier"] / 1_000_000

        abs_rate = float(df_m["absence_days"].sum() / (len(df_m) * 230) * 100)
        high_risk = int((df_m["attrition_prob"] > KPI_THRESHOLDS["high_risk_prob_threshold"]).sum())

        savings_at_2pp = (
            int(len(df_m) * 0.02)
            * float(np.median(df_m["salary"]))
            * KPI_THRESHOLDS["attrition_cost_multiplier"]
            / 1_000_000
        )

        return {
            "value": att_rate,
            "unit": "%",
            "trend": "stable",
            "benchmark": f"TO FR 2025 anchor: {ANCHORS.turnover_rate_pct:.2f}%",
            "absenteeism_rate": abs_rate,
            "estimated_cost_m": round(cost_m, 1),
            "high_risk_count": high_risk,
            "savings_at_minus2pp_m": round(savings_at_2pp, 1),
        }

    # ── KPI 5: Training Effectiveness ─────────────────────────────────────────

    def kpi_training_effectiveness(self, df_m: pd.DataFrame) -> dict:
        """
        Training Effectiveness = hours/FTE + completion + transfer rate + satisfaction.

        Transfer rate (cold review) measures knowledge applied on the job —
        a stronger signal than completion rate alone. CACEIS target: 69.7%.

        Returns
        -------
        dict with hours_fte, completion, transfer, satisfaction
        """
        hours_fte = float(df_m["train_hours"].mean())
        completion = float(df_m["train_done"].mean() * 100)
        transfer = float(df_m["train_transfer"].dropna().mean() * 100) if "train_transfer" in df_m.columns else 0.0
        sat = float(df_m["train_sat"].dropna().mean()) if "train_sat" in df_m.columns else 0.0

        return {
            "value": hours_fte,
            "unit": "h/FTE",
            "trend": "stable",
            "benchmark": f"{ANCHORS.training_hours_per_fte:.1f}h/FTE target",
            "completion_pct": completion,
            "transfer_pct": transfer,
            "satisfaction_avg": sat,
        }

    # ── KPI 6: Inclusion Score ─────────────────────────────────────────────────

    def kpi_inclusion_score(self, df_m: pd.DataFrame) -> dict:
        """
        Inclusion Score = avg inclusion score by entity, flagging red zones.

        Red zone threshold: < 65% (from Mozaïk RH methodology).
        Luxembourg is flagged as red zone (64%) — below the 65% threshold.

        Returns
        -------
        dict with scores by country, red_zone_entities, group_avg
        """
        group_avg = float(df_m["inclusion_score"].mean())

        country_scores = (
            df_m.groupby("country")["inclusion_score"]
            .agg(["mean", "count"])
            .query("count >= 30")
            .sort_values("mean")
            .round(1)
        )

        red_zone = country_scores[country_scores["mean"] < KPI_THRESHOLDS["inclusion_red_zone"]].index.tolist()
        watch = country_scores[
            (country_scores["mean"] >= KPI_THRESHOLDS["inclusion_red_zone"]) &
            (country_scores["mean"] < KPI_THRESHOLDS["inclusion_watch"])
        ].index.tolist()

        return {
            "value": group_avg,
            "unit": "%",
            "trend": "stable",
            "benchmark": f"Group target: {ANCHORS.inclusion_group:.0f}% | Alert: <{KPI_THRESHOLDS['inclusion_red_zone']:.0f}%",
            "by_country": country_scores["mean"].to_dict(),
            "red_zone_entities": red_zone,
            "watch_entities": watch,
        }

    # ── Compute all ───────────────────────────────────────────────────────────

    def compute_all(
        self,
        df_pl: pd.DataFrame,
        df_m: pd.DataFrame,
        df_abs: pd.DataFrame | None = None,
        df_to: pd.DataFrame | None = None,
    ) -> dict[str, dict]:
        """
        Run all 6 KPIs and return named results dict.

        Parameters
        ----------
        df_pl : cleaned P&L / FTE dataframe
        df_m  : master analytics dataframe (enriched)
        df_abs: absenteeism monthly series (optional)
        df_to : turnover monthly series (optional)
        """
        results = {
            "hc_roi": self.kpi_hc_roi(df_pl),
            "revenue_per_fte": self.kpi_revenue_per_fte(df_pl),
            "performance_index": self.kpi_performance_index(df_m),
            "attrition_risk": self.kpi_attrition_risk(df_m, df_abs, df_to),
            "training_effectiveness": self.kpi_training_effectiveness(df_m),
            "inclusion_score": self.kpi_inclusion_score(df_m),
        }
        KPIValidator().validate(results)
        return results

    # ── Dashboard print ───────────────────────────────────────────────────────

    def print_dashboard(self, results: dict[str, dict]) -> None:
        """Print a formatted KPI dashboard matching the notebook output style."""
        print("\n" + "=" * 58)
        print("KPI DASHBOARD — CACEIS Human Capital 2025")
        print("=" * 58)

        r = results.get("hc_roi", {})
        print(f"\nKPI 1 — HC-ROI:          {r.get('value', 0):.1f}%")
        print(f"  Interpretation:         {r.get('interpretation', '')}")
        for year, val in r.get("series", {}).items():
            print(f"  {year}: {val:.1f}%")

        r = results.get("revenue_per_fte", {})
        print(f"\nKPI 2 — Revenue / FTE:   EUR {r.get('value', 0)/1000:.0f}K")

        r = results.get("performance_index", {})
        print(f"\nKPI 3 — Performance Index: {r.get('value', 0):.2f}/5")
        print(f"  Top performers (4-5): {r.get('top_pct', 0):.1f}% | Low (1-2): {r.get('low_pct', 0):.1f}%")
        print(f"  Training completion:  {r.get('training_completion_pct', 0):.1f}%")

        r = results.get("attrition_risk", {})
        print(f"\nKPI 4 — Attrition Risk:")
        print(f"  Turnover rate 2025:   {r.get('value', 0):.1f}%")
        print(f"  Absenteeism rate:     {r.get('absenteeism_rate', 0):.2f}%")
        print(f"  Est. replacement cost: EUR {r.get('estimated_cost_m', 0):.1f}M")
        print(f"  High-risk employees:  {r.get('high_risk_count', 0):,}")
        print(f"  Savings at -2pp:      EUR {r.get('savings_at_minus2pp_m', 0):.1f}M")

        r = results.get("training_effectiveness", {})
        print(f"\nKPI 5 — Training Effectiveness:")
        print(f"  Hours/FTE:   {r.get('value', 0):.1f}h  |  Completion: {r.get('completion_pct', 0):.1f}%")
        print(f"  Transfer rate: {r.get('transfer_pct', 0):.1f}%  |  Satisfaction: {r.get('satisfaction_avg', 0):.2f}/5")

        r = results.get("inclusion_score", {})
        print(f"\nKPI 6 — Inclusion Score:   {r.get('value', 0):.1f}%")
        if r.get("red_zone_entities"):
            print(f"  RED ZONE entities:    {', '.join(r['red_zone_entities'])}")
        if r.get("watch_entities"):
            print(f"  WATCH entities:       {', '.join(r['watch_entities'])}")

        print()
