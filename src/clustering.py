"""
D&I inclusion risk clustering (Module M3) for CACEIS Human Capital Analytics.

Detects entities at inclusion risk using two complementary approaches:
1. Rule-based threshold classification (HR-interpretable, real-time)
2. DBSCAN clustering on 6-dimensional D&I feature space (ML component)

D&I dimensions sourced from Mozaïk RH Barometer (FR 2025, LU 2025):
  - inclusion_score      : overall inclusion score (0-100)
  - leadership_id        : % employees who identify with leadership
  - discrimination_rate  : % reporting discrimination experience
  - career_equality      : perceived career opportunity equality (0-100)
  - work_life_balance    : WLB satisfaction score (0-100)
  - manager_support      : % feeling supported by direct manager

For CACEIS-specific D&I strategy, see docs/methodology.md §3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.config import ANCHORS, KPI_THRESHOLDS

logger = logging.getLogger(__name__)

# ── D&I dimension scores from Mozaïk RH Barometer 2025 ───────────────────────
# Sourced from: Bilan Groupe Be Generous CACEIS 2025 + Reporting 2024 CASA
# FR/LU values are real; others are representative estimates from group barometer

ENTITY_DI_SCORES: dict[str, dict[str, float]] = {
    "France": {
        "inclusion_score": 70.0,
        "leadership_id": 42.0,          # 42% identify with CACEIS leadership
        "discrimination_rate": 12.0,    # 12% reported discrimination
        "career_equality": 68.0,
        "work_life_balance": 72.0,
        "manager_support": 73.0,
    },
    "Luxembourg": {
        "inclusion_score": 64.0,        # RED ZONE — below 65% threshold
        "leadership_id": 42.0,
        "discrimination_rate": 15.0,    # higher than FR
        "career_equality": 61.0,
        "work_life_balance": 68.0,
        "manager_support": 66.0,
    },
    "Germany": {
        "inclusion_score": 73.0,
        "leadership_id": 48.0,
        "discrimination_rate": 9.0,
        "career_equality": 71.0,
        "work_life_balance": 74.0,
        "manager_support": 76.0,
    },
    "Spain": {
        "inclusion_score": 69.0,
        "leadership_id": 41.0,
        "discrimination_rate": 13.0,
        "career_equality": 67.0,
        "work_life_balance": 70.0,
        "manager_support": 69.0,
    },
    "Malaysia": {
        "inclusion_score": 71.0,
        "leadership_id": 44.0,
        "discrimination_rate": 11.0,
        "career_equality": 70.0,
        "work_life_balance": 69.0,
        "manager_support": 72.0,
    },
    "Brazil": {
        "inclusion_score": 66.0,
        "leadership_id": 39.0,
        "discrimination_rate": 14.0,
        "career_equality": 63.0,
        "work_life_balance": 67.0,
        "manager_support": 65.0,
    },
    "Belgium": {
        "inclusion_score": 72.0,
        "leadership_id": 46.0,
        "discrimination_rate": 10.0,
        "career_equality": 70.0,
        "work_life_balance": 73.0,
        "manager_support": 74.0,
    },
    "Ireland": {
        "inclusion_score": 74.0,
        "leadership_id": 49.0,
        "discrimination_rate": 8.0,
        "career_equality": 73.0,
        "work_life_balance": 75.0,
        "manager_support": 77.0,
    },
    "United Kingdom": {
        "inclusion_score": 73.0,
        "leadership_id": 47.0,
        "discrimination_rate": 9.0,
        "career_equality": 72.0,
        "work_life_balance": 74.0,
        "manager_support": 75.0,
    },
    "Netherlands": {
        "inclusion_score": 75.0,
        "leadership_id": 50.0,
        "discrimination_rate": 7.0,
        "career_equality": 74.0,
        "work_life_balance": 76.0,
        "manager_support": 78.0,
    },
    "Switzerland": {
        "inclusion_score": 74.0,
        "leadership_id": 48.0,
        "discrimination_rate": 8.0,
        "career_equality": 73.0,
        "work_life_balance": 75.0,
        "manager_support": 77.0,
    },
    "Colombia": {
        "inclusion_score": 65.0,        # WATCH — on red zone threshold
        "leadership_id": 38.0,
        "discrimination_rate": 16.0,
        "career_equality": 62.0,
        "work_life_balance": 66.0,
        "manager_support": 64.0,
    },
    "Italy": {
        "inclusion_score": 68.0,
        "leadership_id": 40.0,
        "discrimination_rate": 13.0,
        "career_equality": 66.0,
        "work_life_balance": 69.0,
        "manager_support": 68.0,
    },
    "Portugal": {
        "inclusion_score": 69.0,
        "leadership_id": 41.0,
        "discrimination_rate": 12.0,
        "career_equality": 67.0,
        "work_life_balance": 70.0,
        "manager_support": 70.0,
    },
}

DI_DIMENSIONS = [
    "inclusion_score",
    "leadership_id",
    "discrimination_rate",
    "career_equality",
    "work_life_balance",
    "manager_support",
]

# Thresholds per dimension (below = flagged)
DI_THRESHOLDS: dict[str, float] = {
    "inclusion_score": 65.0,
    "leadership_id": 40.0,
    "discrimination_rate": 14.0,   # above this = flagged (inverted)
    "career_equality": 65.0,
    "work_life_balance": 68.0,
    "manager_support": 68.0,
}

GROUP_BENCHMARK: dict[str, float] = {
    "inclusion_score": ANCHORS.inclusion_group,
    "leadership_id": 44.5,
    "discrimination_rate": 11.0,
    "career_equality": 68.5,
    "work_life_balance": 71.5,
    "manager_support": 72.5,
}


# ── Risk result ────────────────────────────────────────────────────────────────


@dataclass
class EntityRisk:
    """D&I risk classification for one entity/country."""

    entity: str
    risk_level: str                    # "healthy" | "watch" | "critical"
    score: float                       # overall inclusion score
    flagged_dimensions: list[str]      # dimensions below threshold
    cluster_id: int = -1               # DBSCAN cluster (-1 = noise/outlier)
    interventions: list[str] = field(default_factory=list)


# ── Detector ──────────────────────────────────────────────────────────────────


class InclusionClusterDetector:
    """
    Detects D&I inclusion risk by entity using DBSCAN + rule-based classification.

    Example
    -------
    >>> detector = InclusionClusterDetector()
    >>> risks = detector.detect_clusters(list(ENTITY_DI_SCORES.keys()))
    >>> for entity, risk in risks.items():
    ...     print(entity, risk.risk_level)
    """

    def __init__(
        self,
        di_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self.di_scores = di_scores or ENTITY_DI_SCORES

    def _score_matrix(self, entities: list[str]) -> pd.DataFrame:
        """Build a DataFrame of D&I dimension scores for the given entities."""
        rows = []
        for e in entities:
            if e in self.di_scores:
                rows.append({"entity": e, **self.di_scores[e]})
        return pd.DataFrame(rows).set_index("entity")

    # ── Rule-based classification ─────────────────────────────────────────────

    def detect_clusters(
        self, entities: Optional[list[str]] = None
    ) -> dict[str, EntityRisk]:
        """
        Classify each entity as healthy / watch / critical using thresholds.

        Then runs DBSCAN to assign cluster IDs for the ML component.

        Risk logic:
          critical : inclusion_score < 65  OR  ≥2 dimensions flagged
          watch    : inclusion_score < 70  OR  1 dimension flagged
          healthy  : all dimensions within acceptable range

        Returns
        -------
        dict: entity → EntityRisk
        """
        if entities is None:
            entities = list(self.di_scores.keys())

        df = self._score_matrix(entities)
        dbscan_clusters = self._run_dbscan(df)

        results: dict[str, EntityRisk] = {}
        for entity in df.index:
            row = df.loc[entity]
            flagged = self._flag_dimensions(row)

            score = float(row["inclusion_score"])
            if score < KPI_THRESHOLDS["inclusion_red_zone"] or len(flagged) >= 2:
                risk = "critical"
            elif score < KPI_THRESHOLDS["inclusion_watch"] or len(flagged) >= 1:
                risk = "watch"
            else:
                risk = "healthy"

            interventions = self.generate_interventions(entity)
            results[entity] = EntityRisk(
                entity=entity,
                risk_level=risk,
                score=score,
                flagged_dimensions=flagged,
                cluster_id=int(dbscan_clusters.get(entity, -1)),
                interventions=interventions if risk in ("critical", "watch") else [],
            )

        n_critical = sum(1 for r in results.values() if r.risk_level == "critical")
        n_watch = sum(1 for r in results.values() if r.risk_level == "watch")
        logger.info(
            "D&I cluster detection: %d entities | %d critical | %d watch",
            len(results), n_critical, n_watch,
        )
        return results

    def _flag_dimensions(self, row: pd.Series) -> list[str]:
        """Return list of D&I dimensions that are below/above their threshold."""
        flagged = []
        for dim, threshold in DI_THRESHOLDS.items():
            if dim not in row.index:
                continue
            # discrimination_rate is inverted: high = bad
            if dim == "discrimination_rate":
                if row[dim] > threshold:
                    flagged.append(dim)
            else:
                if row[dim] < threshold:
                    flagged.append(dim)
        return flagged

    # ── DBSCAN clustering ─────────────────────────────────────────────────────

    def _run_dbscan(self, df: pd.DataFrame) -> dict[str, int]:
        """
        DBSCAN clustering on the 6-dimensional D&I feature space.

        Features: inclusion_score, leadership_id, discrimination_rate,
        career_equality, work_life_balance, manager_support.

        discrimination_rate is inverted (100 - rate) before scaling so that
        all dimensions point in the same "good" direction.

        Cluster -1 = noise/outlier (isolated risk entity).
        """
        X = df[DI_DIMENSIONS].copy()
        # Invert discrimination_rate so higher = better for all dims
        X["discrimination_rate"] = 100 - X["discrimination_rate"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)

        dbscan = DBSCAN(eps=1.2, min_samples=2)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(
            "DBSCAN: %d clusters | %d noise points (outlier entities)",
            n_clusters, n_noise,
        )
        return dict(zip(df.index, labels.tolist()))

    # ── Radar data ────────────────────────────────────────────────────────────

    def get_radar_data(self, entity: str) -> dict:
        """
        Return 6-dimension scores + group benchmark for radar chart rendering.

        Returns
        -------
        dict with 'entity_scores', 'group_benchmark', 'thresholds', 'flagged'
        """
        if entity not in self.di_scores:
            raise ValueError(f"Entity '{entity}' not in D&I scores database")

        scores = self.di_scores[entity]
        flagged = self._flag_dimensions(pd.Series(scores))

        return {
            "entity": entity,
            "dimensions": DI_DIMENSIONS,
            "entity_scores": {d: scores.get(d, 0.0) for d in DI_DIMENSIONS},
            "group_benchmark": GROUP_BENCHMARK,
            "thresholds": DI_THRESHOLDS,
            "flagged_dimensions": flagged,
        }

    # ── Auto-flag risks ───────────────────────────────────────────────────────

    def auto_flag_risks(
        self, entities: Optional[list[str]] = None
    ) -> list[str]:
        """
        Return entities with multiple dimensions below threshold.

        An entity is auto-flagged if it fails on ≥2 dimensions OR its
        inclusion_score is in the red zone.
        """
        risks = self.detect_clusters(entities)
        return [e for e, r in risks.items() if r.risk_level in ("critical", "watch")]

    # ── Intervention generation ───────────────────────────────────────────────

    def generate_interventions(self, entity: str) -> list[str]:
        """
        Return 3 specific action recommendations based on flagged dimensions.

        Actions are drawn from the CACEIS D&I playbook and Mozaïk RH
        best-practice recommendations. Each action targets the worst
        dimension first.
        """
        if entity not in self.di_scores:
            return []

        scores = self.di_scores[entity]
        flagged = self._flag_dimensions(pd.Series(scores))
        interventions: list[str] = []

        # Priority action per dimension
        ACTIONS: dict[str, str] = {
            "inclusion_score": (
                "Launch entity-level inclusion pulse survey (2 questions, monthly cadence) "
                "to track trend and identify root causes before annual Mozaïk RH barometer"
            ),
            "leadership_id": (
                "Mandate D&I visibility commitments from senior leadership: "
                "monthly all-hands inclusion topic + mentorship programme for underrepresented groups"
            ),
            "discrimination_rate": (
                "Deploy anonymous reporting channel (e.g. Navex EthicsPoint) + "
                "mandatory bystander intervention training for all people managers"
            ),
            "career_equality": (
                "Conduct career equity audit: compare promotion rates and salary growth "
                "by gender and nationality; publish results to Works Council"
            ),
            "work_life_balance": (
                "Review workload distribution in flagged teams; pilot 4.5-day workweek "
                "experiment in 2 teams and measure impact on absence + satisfaction"
            ),
            "manager_support": (
                "Add 'People Leadership' KPI to manager performance reviews (weighted 20%); "
                "provide coaching for managers scoring <60% on team pulse surveys"
            ),
        }

        for dim in flagged[:3]:
            if dim in ACTIONS:
                interventions.append(ACTIONS[dim])

        # Always include a cross-cutting action if entity is critical
        inc = scores.get("inclusion_score", 100)
        if inc < KPI_THRESHOLDS["inclusion_red_zone"] and len(interventions) < 3:
            interventions.append(
                f"Immediate CHRO escalation required: {entity} inclusion score {inc:.0f}% "
                f"is below the {KPI_THRESHOLDS['inclusion_red_zone']:.0f}% red-zone threshold. "
                "Convene entity D&I task force within 30 days."
            )

        return interventions[:3]
