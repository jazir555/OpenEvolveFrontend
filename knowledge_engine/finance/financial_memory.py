"""
Financial Memory - Hybrid memory for financial strategy evolution.

Combines multiple memory structures:
- Evolutionary tree (lineage of strategies)
- MAP-Elites archive (diverse strategy niches)
- Crisis-specific lessons (what worked in which crises)
- Feature importance tracking
- Failure patterns

Pure-stdlib implementation (no numpy / pydantic dependency).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .schemas import (
    CrisisLesson,
    CrisisType,
    MarketConditions,
    StrategyFailure,
    StrategyType,
)


_MAP_ELITES_NICHES = [
    "high_volatility",
    "low_volatility",
    "crisis_survivors",
    "bull_market_winners",
    "bear_market_winners",
    "trend_followers",
    "mean_reverters",
]


class FinancialMemory:
    """Multi-structure memory for financial strategy evolution."""

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path

        self.evolutionary_tree = {
            "root": {
                "strategy_id": "root",
                "strategy_type": "initial_population",
                "children": [],
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {},
            }
        }
        self.map_elites_archive: Dict[str, List[CrisisLesson]] = {
            niche: [] for niche in _MAP_ELITES_NICHES
        }
        self.crisis_lessons: Dict[CrisisType, List[CrisisLesson]] = defaultdict(list)
        self.feature_importance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[StrategyFailure]] = defaultdict(list)
        self.performance_history: List[Dict[str, Any]] = []

        if persistence_path:
            self._load_from_disk()

    # -- Lessons ----------------------------------------------------------
    def store_lesson(self, lesson: CrisisLesson) -> None:
        self.crisis_lessons[lesson.crisis].append(lesson)
        for feature, importance in lesson.feature_importance.items():
            self.feature_importance[feature].append({
                "importance": importance,
                "timestamp": datetime.utcnow().isoformat(),
                "crisis": lesson.crisis.value,
            })
        niche = self._classify_strategy_niche(lesson.strategy_type, lesson.conditions_met)
        if niche and lesson.successful:
            self.map_elites_archive[niche].append(lesson)
            self.map_elites_archive[niche].sort(key=lambda l: l.boost_amount, reverse=True)
            self.map_elites_archive[niche] = self.map_elites_archive[niche][:10]
        if self.persistence_path:
            self._save_to_disk()

    def store_failure(self, failure: StrategyFailure) -> None:
        self.failure_patterns[failure.failure_type].append(failure)
        if len(self.failure_patterns[failure.failure_type]) > 100:
            self.failure_patterns[failure.failure_type] = \
                self.failure_patterns[failure.failure_type][-100:]
        if self.persistence_path:
            self._save_to_disk()

    # -- Retrieval --------------------------------------------------------
    def get_relevant_lessons(self, current_conditions: MarketConditions) -> List[CrisisLesson]:
        relevant: List[CrisisLesson] = []
        if current_conditions.resembles_crisis:
            relevant.extend(self.crisis_lessons.get(current_conditions.resembles_crisis, []))
        niche = self._classify_market_regime(current_conditions)
        if niche:
            relevant.extend(self.map_elites_archive.get(niche, []))
        if current_conditions.volatility > 0.30:
            relevant.extend(self.map_elites_archive.get("high_volatility", []))
        elif current_conditions.volatility < 0.15:
            relevant.extend(self.map_elites_archive.get("low_volatility", []))
        unique = {lesson.lesson_id: lesson for lesson in relevant}
        return list(unique.values())

    def get_recent_failures(self, n: int = 5) -> List[StrategyFailure]:
        all_failures: List[StrategyFailure] = []
        for failures in self.failure_patterns.values():
            all_failures.extend(failures)
        all_failures.sort(key=lambda f: f.occurred_at, reverse=True)
        return all_failures[:n]

    def get_niche_representatives(self, niche: str, n: int = 5) -> List[CrisisLesson]:
        if niche not in self.map_elites_archive:
            return []
        return self.map_elites_archive[niche][:n]

    def get_crisis_statistics(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for crisis, lessons in self.crisis_lessons.items():
            successful = [l for l in lessons if l.successful]
            failed = [l for l in lessons if not l.successful]
            boosts = [l.boost_amount for l in lessons]
            stats[crisis.value if isinstance(crisis, CrisisType) else crisis] = {
                "total_lessons": len(lessons),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": (len(successful) / len(lessons)) if lessons else 0.0,
                "avg_boost": statistics.mean(boosts) if boosts else 0.0,
            }
        return stats

    def get_feature_importance(
        self,
        feature: str,
        crisis_type: Optional[CrisisType] = None,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        if feature not in self.feature_importance:
            return []
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        filtered = [
            r for r in self.feature_importance[feature]
            if datetime.fromisoformat(r["timestamp"]) > cutoff
        ]
        if crisis_type is not None:
            ct = crisis_type.value if isinstance(crisis_type, CrisisType) else crisis_type
            filtered = [r for r in filtered if r.get("crisis") == ct]
        return filtered

    def get_average_feature_importance(
        self,
        feature: str,
        crisis_type: Optional[CrisisType] = None,
    ) -> float:
        records = self.get_feature_importance(feature, crisis_type, days_back=365)
        if not records:
            return 0.0
        return statistics.mean([r["importance"] for r in records])

    # -- Lineage ----------------------------------------------------------
    def add_strategy_lineage(
        self,
        parent_id: Optional[str],
        child_id: str,
        strategy_type: StrategyType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        node = {
            "strategy_id": child_id,
            "strategy_type": strategy_type.value if isinstance(strategy_type, StrategyType) else strategy_type,
            "children": [],
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        if parent_id is None:
            self.evolutionary_tree["root"]["children"].append(node)
        else:
            self._add_to_tree(self.evolutionary_tree["root"], parent_id, node)
        if self.persistence_path:
            self._save_to_disk()

    def get_strategy_lineage(self, strategy_id: str) -> List[Dict[str, Any]]:
        lineage: List[Dict[str, Any]] = []
        self._find_lineage(self.evolutionary_tree["root"], strategy_id, lineage)
        return lineage

    # -- Maintenance ------------------------------------------------------
    def clear_old_data(self, days_to_keep: int = 90) -> None:
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
        for feature in self.feature_importance:
            self.feature_importance[feature] = [
                r for r in self.feature_importance[feature]
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]
        for failure_type in self.failure_patterns:
            self.failure_patterns[failure_type] = [
                f for f in self.failure_patterns[failure_type]
                if f.occurred_at > cutoff
            ]
        if self.persistence_path:
            self._save_to_disk()

    def record_performance(self, strategy_id: str, score: float, crisis: Optional[str] = None) -> None:
        self.performance_history.append({
            "strategy_id": strategy_id,
            "score": score,
            "crisis": crisis,
            "timestamp": datetime.utcnow().isoformat(),
        })

    # -- Helpers ----------------------------------------------------------
    def _classify_strategy_niche(self, strategy_type: StrategyType, conditions: Dict[str, Any]) -> Optional[str]:
        st = strategy_type.value if isinstance(strategy_type, StrategyType) else strategy_type
        if st == StrategyType.MOMENTUM.value:
            return "trend_followers"
        if st == StrategyType.MEAN_REVERSION.value:
            return "mean_reverters"
        if conditions.get("high_volatility"):
            return "high_volatility"
        if conditions.get("low_volatility"):
            return "low_volatility"
        if conditions.get("crisis_survived"):
            return "crisis_survivors"
        if conditions.get("bull_market"):
            return "bull_market_winners"
        if conditions.get("bear_market"):
            return "bear_market_winners"
        return None

    def _classify_market_regime(self, conditions: MarketConditions) -> Optional[str]:
        if conditions.resembles_crisis:
            return "crisis_survivors"
        if conditions.trend == "up":
            return "bull_market_winners"
        if conditions.trend == "down":
            return "bear_market_winners"
        return None

    def _add_to_tree(self, node: Dict[str, Any], parent_id: str, child_node: Dict[str, Any]) -> bool:
        if node.get("strategy_id") == parent_id:
            node["children"].append(child_node)
            return True
        for child in node.get("children", []):
            if self._add_to_tree(child, parent_id, child_node):
                return True
        return False

    def _find_lineage(self, node: Dict[str, Any], target_id: str, lineage: List[Dict[str, Any]]) -> bool:
        lineage.append(node)
        if node.get("strategy_id") == target_id:
            return True
        for child in node.get("children", []):
            if self._find_lineage(child, target_id, lineage):
                return True
        lineage.pop()
        return False

    def _save_to_disk(self) -> None:
        if not self.persistence_path:
            return
        try:
            data = {
                "crisis_lessons": {
                    (k.value if isinstance(k, CrisisType) else k): [l.__dict__ for l in v]
                    for k, v in self.crisis_lessons.items()
                },
                "map_elites_archive": {
                    k: [l.__dict__ for l in v] for k, v in self.map_elites_archive.items()
                },
                "feature_importance": dict(self.feature_importance),
                "failure_patterns": {
                    k: [f.__dict__ for f in v] for k, v in self.failure_patterns.items()
                },
            }
            with open(self.persistence_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
        except (OSError, IOError, TypeError):
            pass

    def _load_from_disk(self) -> None:
        if not self.persistence_path:
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, IOError, json.JSONDecodeError):
            return
        for crisis, lessons in data.get("crisis_lessons", {}).items():
            try:
                ct = CrisisType(crisis)
            except ValueError:
                ct = CrisisType.CUSTOM
            self.crisis_lessons[ct] = [CrisisLesson(**l) for l in lessons]
        for niche, lessons in data.get("map_elites_archive", {}).items():
            if niche in self.map_elites_archive:
                self.map_elites_archive[niche] = [CrisisLesson(**l) for l in lessons]
        self.feature_importance = defaultdict(list, data.get("feature_importance", {}))
        for ftype, failures in data.get("failure_patterns", {}).items():
            self.failure_patterns[ftype] = [StrategyFailure(**f) for f in failures]
