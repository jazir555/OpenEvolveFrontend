"""
FinancialEvolutionMemory - Hybrid memory for financial strategy evolution

Combines multiple memory structures:
- Evolutionary tree (lineage of strategies)
- MAP-Elites archive (diverse strategy niches)
- Crisis-specific lessons (what worked in which crises)
- Feature importance tracking
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict

from .schemas import (
    CrisisLesson,
    StrategyFailure,
    MarketConditions,
    StrategyType,
    CrisisType
)


class FinancialEvolutionMemory:
    """
    Multi-structure memory for financial strategy evolution.

    Maintains:
    1. Evolutionary tree: Tracks lineage of strategies
    2. MAP-Elites archive: Diverse strategies across niches
    3. Crisis lessons: What worked in specific crises
    4. Feature importance: Track feature relevance over time
    5. Failure patterns: Common failure modes
    """

    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize financial evolution memory.

        Args:
            persistence_path: Optional path to persist memory to disk
        """
        self.persistence_path = persistence_path

        # Evolutionary tree
        self.evolutionary_tree = {
            "root": {
                "strategy_type": "initial_population",
                "children": [],
                "created_at": datetime.utcnow().isoformat()
            }
        }

        # MAP-Elites archive (diverse niches)
        self.map_elites_archive = {
            "high_volatility": [],      # Strategies that thrive in volatility
            "low_volatility": [],       # Strategies for calm markets
            "crisis_survivors": [],     # Strategies that survived crises
            "bull_market_winners": [],  # Strategies for bull markets
            "bear_market_winners": [],  # Strategies for bear markets
            "trend_followers": [],      # Trend-following strategies
            "mean_reverters": []        # Mean-reversion strategies
        }

        # Crisis-specific lessons
        self.crisis_lessons = defaultdict(list)

        # Feature importance cache
        self.feature_importance = defaultdict(list)

        # Failure patterns
        self.failure_patterns = defaultdict(list)

        # Performance history
        self.performance_history = []

        # Load from disk if available
        if persistence_path:
            self._load_from_disk()

    def store_lesson(self, lesson: CrisisLesson) -> None:
        """
        Store lesson learned from evolution.

        Args:
            lesson: CrisisLesson to store
        """
        # Store in crisis-specific bucket
        self.crisis_lessons[lesson.crisis].append(lesson)

        # Update feature importance
        for feature, importance in lesson.feature_importance.items():
            self.feature_importance[feature].append({
                "importance": importance,
                "timestamp": datetime.utcnow().isoformat(),
                "crisis": lesson.crisis
            })

        # Store in MAP-Elites niche
        niche = self._classify_strategy_niche(lesson.strategy_type, lesson.conditions_met)
        if niche and lesson.successful:
            self.map_elites_archive[niche].append(lesson)
            # Keep only top 10 per niche
            self.map_elites_archive[niche].sort(
                key=lambda l: l.boost_amount,
                reverse=True
            )
            self.map_elites_archive[niche] = self.map_elites_archive[niche][:10]

        # Persist if path specified
        if self.persistence_path:
            self._save_to_disk()

    def store_failure(self, failure: StrategyFailure) -> None:
        """
        Store strategy failure for learning.

        Args:
            failure: StrategyFailure to store
        """
        self.failure_patterns[failure.failure_type].append(failure)

        # Limit failure history
        if len(self.failure_patterns[failure.failure_type]) > 100:
            self.failure_patterns[failure.failure_type] = \
                self.failure_patterns[failure.failure_type][-100:]

        # Persist if path specified
        if self.persistence_path:
            self._save_to_disk()

    def get_relevant_lessons(
        self,
        current_conditions: MarketConditions
    ) -> List[CrisisLesson]:
        """
        Retrieve lessons relevant to current market conditions.

        Args:
            current_conditions: Current market conditions

        Returns:
            List of relevant CrisisLesson objects
        """
        relevant = []

        # Crisis-specific lessons
        if current_conditions.resembles_crisis:
            relevant.extend(
                self.crisis_lessons.get(current_conditions.resembles_crisis, [])
            )

        # Regime-specific lessons
        niche = self._classify_market_regime(current_conditions)
        if niche:
            relevant.extend(self.map_elites_archive.get(niche, []))

        # Volatility-specific lessons
        if current_conditions.volatility > 0.30:  # High volatility threshold
            relevant.extend(self.map_elites_archive.get("high_volatility", []))
        elif current_conditions.volatility < 0.15:  # Low volatility threshold
            relevant.extend(self.map_elites_archive.get("low_volatility", []))

        # Remove duplicates and return
        unique_lessons = list({lesson.lesson_id: lesson for lesson in relevant}.values())
        return unique_lessons

    def get_recent_failures(self, n: int = 5) -> List[StrategyFailure]:
        """
        Get recent failures for learning.

        Args:
            n: Number of recent failures to return

        Returns:
            List of recent StrategyFailure objects
        """
        all_failures = []
        for failures in self.failure_patterns.values():
            all_failures.extend(failures)

        # Sort by time and get most recent
        all_failures.sort(key=lambda f: f.occurred_at, reverse=True)
        return all_failures[:n]

    def get_feature_importance(
        self,
        feature: str,
        crisis_type: Optional[CrisisType] = None,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get feature importance history.

        Args:
            feature: Feature name
            crisis_type: Optional crisis filter
            days_back: Days to look back

        Returns:
            List of importance records
        """
        if feature not in self.feature_importance:
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        filtered = [
            record for record in self.feature_importance[feature]
            if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]

        if crisis_type:
            filtered = [r for r in filtered if r.get("crisis") == crisis_type]

        return filtered

    def get_average_feature_importance(
        self,
        feature: str,
        crisis_type: Optional[CrisisType] = None
    ) -> float:
        """
        Get average feature importance.

        Args:
            feature: Feature name
            crisis_type: Optional crisis filter

        Returns:
            Average importance (0-1)
        """
        records = self.get_feature_importance(feature, crisis_type, days_back=365)

        if not records:
            return 0.0

        return np.mean([r["importance"] for r in records])

    def add_strategy_lineage(
        self,
        parent_id: Optional[str],
        child_id: str,
        strategy_type: StrategyType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add strategy to evolutionary tree.

        Args:
            parent_id: Parent strategy ID (None for root)
            child_id: Child strategy ID
            strategy_type: Type of strategy
            metadata: Optional metadata
        """
        node = {
            "strategy_id": child_id,
            "strategy_type": strategy_type,
            "children": [],
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        if parent_id is None:
            # Add to root
            self.evolutionary_tree["root"]["children"].append(node)
        else:
            # Find parent and add (simplified - in production use proper tree traversal)
            self._add_to_tree(self.evolutionary_tree["root"], parent_id, node)

        # Persist if path specified
        if self.persistence_path:
            self._save_to_disk()

    def get_strategy_lineage(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get lineage of a strategy.

        Args:
            strategy_id: Strategy ID

        Returns:
            List of ancestors from root to strategy
        """
        # Simplified implementation - in production use proper tree traversal
        lineage = []
        self._find_lineage(self.evolutionary_tree["root"], strategy_id, lineage)
        return lineage

    def get_niche_representatives(
        self,
        niche: str,
        n: int = 5
    ) -> List[CrisisLesson]:
        """
        Get representative strategies from a niche.

        Args:
            niche: Niche name
            n: Number of representatives

        Returns:
            List of CrisisLesson objects
        """
        if niche not in self.map_elites_archive:
            return []

        return self.map_elites_archive[niche][:n]

    def get_crisis_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about crisis lessons.

        Returns:
            Dictionary with crisis statistics
        """
        stats = {}
        for crisis, lessons in self.crisis_lessons.items():
            successful = [l for l in lessons if l.successful]
            failed = [l for l in lessons if not l.successful]

            stats[crisis] = {
                "total_lessons": len(lessons),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(lessons) if lessons else 0,
                "avg_boost": np.mean([l.boost_amount for l in lessons]) if lessons else 0
            }

        return stats

    def clear_old_data(self, days_to_keep: int = 90) -> None:
        """
        Clear old data to manage memory.

        Args:
            days_to_keep: Days of data to retain
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        # Clear old feature importance
        for feature in self.feature_importance:
            self.feature_importance[feature] = [
                record for record in self.feature_importance[feature]
                if datetime.fromisoformat(record["timestamp"]) > cutoff_date
            ]

        # Clear old failures
        for failure_type in self.failure_patterns:
            self.failure_patterns[failure_type] = [
                failure for failure in self.failure_patterns[failure_type]
                if failure.occurred_at > cutoff_date
            ]

        # Persist if path specified
        if self.persistence_path:
            self._save_to_disk()

    # Private helper methods

    def _classify_strategy_niche(
        self,
        strategy_type: StrategyType,
        conditions: Dict[str, Any]
    ) -> Optional[str]:
        """Classify strategy into MAP-Elites niche"""
        if strategy_type == StrategyType.MOMENTUM:
            return "trend_followers"
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return "mean_reverters"
        elif conditions.get("high_volatility", False):
            return "high_volatility"
        elif conditions.get("low_volatility", False):
            return "low_volatility"
        elif conditions.get("crisis_survived", False):
            return "crisis_survivors"
        elif conditions.get("bull_market", False):
            return "bull_market_winners"
        elif conditions.get("bear_market", False):
            return "bear_market_winners"
        return None

    def _classify_market_regime(self, conditions: MarketConditions) -> Optional[str]:
        """Classify current market regime"""
        if conditions.resembles_crisis:
            return "crisis_survivors"
        if conditions.trend == "up":
            return "bull_market_winners"
        elif conditions.trend == "down":
            return "bear_market_winners"
        return None

    def _add_to_tree(
        self,
        node: Dict[str, Any],
        parent_id: str,
        child_node: Dict[str, Any]
    ) -> bool:
        """Add child to tree node (recursive)"""
        if node.get("strategy_id") == parent_id:
            node["children"].append(child_node)
            return True

        for child in node.get("children", []):
            if self._add_to_tree(child, parent_id, child_node):
                return True

        return False

    def _find_lineage(
        self,
        node: Dict[str, Any],
        target_id: str,
        lineage: List[Dict[str, Any]]
    ) -> bool:
        """Find lineage to target (recursive)"""
        lineage.append(node)

        if node.get("strategy_id") == target_id:
            return True

        for child in node.get("children", []):
            if self._find_lineage(child, target_id, lineage):
                return True

        lineage.pop()
        return False

    def _save_to_disk(self) -> None:
        """Persist memory to disk"""
        if not self.persistence_path:
            return

        data = {
            "evolutionary_tree": self.evolutionary_tree,
            "map_elites_archive": {
                k: [l.dict() for l in lessons]
                for k, lessons in self.map_elites_archive.items()
            },
            "crisis_lessons": {
                k: [l.dict() for l in lessons]
                for k, lessons in self.crisis_lessons.items()
            },
            "feature_importance": dict(self.feature_importance),
            "failure_patterns": {
                k: [f.dict() for f in failures]
                for k, failures in self.failure_patterns.items()
            }
        }

        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self) -> None:
        """Load memory from disk"""
        if not self.persistence_path:
            return

        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            self.evolutionary_tree = data.get("evolutionary_tree", self.evolutionary_tree)

            # Restore crisis lessons
            for crisis, lessons in data.get("crisis_lessons", {}).items():
                self.crisis_lessons[crisis] = [
                    CrisisLesson(**l) for l in lessons
                ]

            # Restore MAP-Elites archive
            for niche, lessons in data.get("map_elites_archive", {}).items():
                self.map_elites_archive[niche] = [
                    CrisisLesson(**l) for l in lessons
                ]

            self.feature_importance = defaultdict(
                list,
                data.get("feature_importance", {})
            )

            # Restore failure patterns
            for failure_type, failures in data.get("failure_patterns", {}).items():
                self.failure_patterns[failure_type] = [
                    StrategyFailure(**f) for f in failures
                ]

        except (FileNotFoundError, json.JSONDecodeError):
            # Start fresh if file doesn't exist or is corrupted
            pass
