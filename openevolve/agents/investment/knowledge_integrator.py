#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Integrator - Continuous Learning from Investment Outcomes

Extracts knowledge from each review cycle, builds causal models of market dynamics,
tracks which factors actually predict outcomes, updates investment heuristics,
and learns from mistakes and successes.

This module implements the learning component of the investment committee,
enabling continuous improvement over time.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from logging import getLogger
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class CausalFactor:
    """Represents a factor that causally influences investment outcomes."""
    name: str
    category: str
    predictive_power: float  # 0.0 to 1.0, how well it predicts outcomes
    confidence: float  # 0.0 to 1.0, statistical confidence
    sample_count: int  # Number of observations
    last_observed: str  # ISO timestamp
    successful_predictions: int
    total_predictions: int


@dataclass
class InvestmentHeuristic:
    """A learned rule or heuristic for investment decisions."""
    rule: str
    conditions: List[str]
    conclusion: str
    success_rate: float
    confidence: float
    times_applied: int
    last_updated: str


@dataclass
class LessonLearned:
    """A lesson learned from a decision outcome."""
    lesson: str
    context: str
    outcome: str
    generality: str  # "specific", "moderate", "general"
    applicable_scenarios: List[str]
    timestamp: str


class KnowledgeIntegrator:
    """
    Knowledge Integrator for Investment Committee

    Continuously learns from:
    - Decision outcomes
    - Factor performance
    - Pattern recognition
    - Mistakes and successes

    Maintains:
    - Cusal models of market dynamics
    - Investment heuristics
    - Lessons learned
    - Performance tracking
    """

    def __init__(self, database_path: Path):
        """
        Initialize the Knowledge Integrator.

        Args:
            database_path: Path to persistent storage for knowledge
        """
        self.database_path = database_path
        self.logger = getLogger(__name__)

        # Knowledge structures
        self.causal_factors: Dict[str, CausalFactor] = {}
        self.heuristics: List[InvestmentHeuristic] = []
        self.lessons_learned: List[LessonLearned] = []

        # Performance tracking
        self.factor_performance: Dict[str, List[bool]] = defaultdict(list)  # factor -> [success, failure, ...]
        self.heuristic_performance: Dict[str, List[bool]] = defaultdict(list)

        # Scenario database
        self.scenario_database: List[Dict[str, Any]] = []

    def load_knowledge(self):
        """Load knowledge from persistent storage."""
        try:
            # Load causal factors
            factors_file = self.database_path / "causal_factors.json"
            if factors_file.exists():
                with open(factors_file, "r") as f:
                    factors_data = json.load(f)
                    self.causal_factors = {
                        name: CausalFactor(**data)
                        for name, data in factors_data.items()
                    }
                self.logger.info(f"Loaded {len(self.causal_factors)} causal factors")

            # Load heuristics
            heuristics_file = self.database_path / "heuristics.json"
            if heuristics_file.exists():
                with open(heuristics_file, "r") as f:
                    heuristics_data = json.load(f)
                    self.heuristics = [InvestmentHeuristic(**h) for h in heuristics_data]
                self.logger.info(f"Loaded {len(self.heuristics)} heuristics")

            # Load lessons
            lessons_file = self.database_path / "lessons.json"
            if lessons_file.exists():
                with open(lessons_file, "r") as f:
                    lessons_data = json.load(f)
                    self.lessons_learned = [LessonLearned(**l) for l in lessons_data]
                self.logger.info(f"Loaded {len(self.lessons_learned)} lessons")

            # Load scenario database
            scenarios_file = self.database_path / "scenarios.json"
            if scenarios_file.exists():
                with open(scenarios_file, "r") as f:
                    self.scenario_database = json.load(f)
                self.logger.info(f"Loaded {len(self.scenario_database)} scenarios")

        except Exception as e:
            self.logger.error(f"Error loading knowledge: {e}")

    def save_knowledge(self):
        """Save knowledge to persistent storage."""
        try:
            self.database_path.mkdir(parents=True, exist_ok=True)

            # Save causal factors
            factors_file = self.database_path / "causal_factors.json"
            with open(factors_file, "w") as f:
                json.dump(
                    {name: asdict(factor) for name, factor in self.causal_factors.items()},
                    f, indent=2
                )

            # Save heuristics
            heuristics_file = self.database_path / "heuristics.json"
            with open(heuristics_file, "w") as f:
                json.dump([asdict(h) for h in self.heuristics], f, indent=2)

            # Save lessons
            lessons_file = self.database_path / "lessons.json"
            with open(lessons_file, "w") as f:
                json.dump([asdict(l) for l in self.lessons_learned], f, indent=2)

            # Save scenario database
            scenarios_file = self.database_path / "scenarios.json"
            with open(scenarios_file, "w") as f:
                json.dump(self.scenario_database, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving knowledge: {e}")

    async def extract_causal_knowledge(self, decision: Any) -> CausalFactor:
        """
        Extract causal factors from a decision and its context.

        Args:
            decision: Investment decision with context and outcome

        Returns:
            Most significant causal factor identified
        """
        # Extract factors from decision metadata
        metadata = decision.metadata if hasattr(decision, 'metadata') else {}

        if "review_data" in metadata:
            review_data = metadata["review_data"]
            changes = review_data.get("changes", [])

            for change in changes:
                factor_name = change.get("factor", "unknown")
                category = change.get("category", "other")

                # Update or create causal factor
                if factor_name in self.causal_factors:
                    factor = self.causal_factors[factor_name]
                    factor.sample_count += 1
                    factor.last_observed = datetime.utcnow().isoformat()
                else:
                    self.causal_factors[factor_name] = CausalFactor(
                        name=factor_name,
                        category=category,
                        predictive_power=0.5,  # Initial estimate
                        confidence=0.3,  # Low initial confidence
                        sample_count=1,
                        last_observed=datetime.utcnow().isoformat(),
                        successful_predictions=0,
                        total_predictions=0
                    )

        # Return the most recently updated factor
        if self.causal_factors:
            return list(self.causal_factors.values())[-1]

        return None

    async def update_heuristics(self, decisions: List[Any]):
        """
        Update investment heuristics based on historical decisions.

        Args:
            decisions: List of historical decisions with outcomes
        """
        # Only update if we have decisions with outcomes
        decisions_with_outcomes = [
            d for d in decisions
            if hasattr(d, 'actual_outcome') and d.actual_outcome
        ]

        if not decisions_with_outcomes:
            return

        # Analyze successful decisions
        successful_decisions = [
            d for d in decisions_with_outcomes
            if "positive" in d.actual_outcome.lower() or "gain" in d.actual_outcome.lower()
        ]

        if successful_decisions:
            # Extract common patterns
            common_factors = self._find_common_patterns(successful_decisions)

            for factor, frequency in common_factors.items():
                if frequency >= len(successful_decisions) * 0.6:  # Appears in 60%+ of successes
                    heuristic = InvestmentHeuristic(
                        rule=f"Consider {factor} when making investment decisions",
                        conditions=[f"When {factor} is present"],
                        conclusion=f"Higher probability of positive outcome",
                        success_rate=frequency / len(successful_decisions),
                        confidence=min(0.9, 0.5 + frequency / len(successful_decisions) * 0.4),
                        times_applied=len(successful_decisions),
                        last_updated=datetime.utcnow().isoformat()
                    )

                    # Check if similar heuristic exists
                    if not any(h.rule == heuristic.rule for h in self.heuristics):
                        self.heuristics.append(heuristic)
                        self.logger.info(f"Learned new heuristic: {heuristic.rule}")

        # Analyze failed decisions
        failed_decisions = [
            d for d in decisions_with_outcomes
            if "negative" in d.actual_outcome.lower() or "loss" in d.actual_outcome.lower()
        ]

        if failed_decisions:
            # Extract common failure patterns
            failure_factors = self._find_common_patterns(failed_decisions)

            for factor, frequency in failure_factors.items():
                if frequency >= len(failed_decisions) * 0.5:  # Appears in 50%+ of failures
                    heuristic = InvestmentHeuristic(
                        rule=f"Avoid or reduce exposure when {factor} is elevated",
                        conditions=[f"When {factor} is high"],
                        conclusion=f"Higher probability of negative outcome",
                        success_rate=1.0 - (frequency / len(failed_decisions)),
                        confidence=min(0.9, 0.5 + frequency / len(failed_decisions) * 0.4),
                        times_applied=len(failed_decisions),
                        last_updated=datetime.utcnow().isoformat()
                    )

                    if not any(h.rule == heuristic.rule for h in self.heuristics):
                        self.heuristics.append(heuristic)
                        self.logger.info(f"Learned avoidance heuristic: {heuristic.rule}")

    async def analyze_predictive_factors(self, decisions: List[Any]):
        """
        Analyze which factors actually predict outcomes.

        Args:
            decisions: List of historical decisions with outcomes
        """
        decisions_with_outcomes = [
            d for d in decisions
            if hasattr(d, 'actual_outcome') and d.actual_outcome
        ]

        if not decisions_with_outcomes:
            return

        # Extract factors from each decision
        for decision in decisions_with_outcomes:
            outcome_positive = (
                "positive" in decision.actual_outcome.lower() or
                "gain" in decision.actual_outcome.lower() or
                "profit" in decision.actual_outcome.lower()
            )

            metadata = decision.metadata if hasattr(decision, 'metadata') else {}

            if "analysis_results" in metadata:
                analysis = metadata["analysis_results"]

                # Extract factors from RLM decomposition
                if "rlm_decomposition" in analysis:
                    rlm = analysis["rlm_decomposition"]
                    key_factors = rlm.get("key_factors", [])

                    for factor_dict in key_factors:
                        factor_name = factor_dict.get("name", "unknown")

                        # Track performance
                        if factor_name not in self.factor_performance:
                            self.factor_performance[factor_name] = []

                        self.factor_performance[factor_name].append(outcome_positive)

                        # Update causal factor
                        if factor_name in self.causal_factors:
                            factor = self.causal_factors[factor_name]
                            factor.total_predictions += 1
                            if outcome_positive:
                                factor.successful_predictions += 1

                            # Update predictive power
                            if factor.total_predictions > 5:
                                factor.predictive_power = (
                                    factor.successful_predictions / factor.total_predictions
                                )

                                # Increase confidence with more data
                                factor.confidence = min(0.95, 0.3 + factor.total_predictions * 0.05)

        # Remove factors with low predictive power or low sample size
        factors_to_remove = [
            name for name, factor in self.causal_factors.items()
            if (factor.total_predictions > 10 and factor.predictive_power < 0.45) or
               (factor.total_predictions > 20 and factor.predictive_power < 0.50)
        ]

        for name in factors_to_remove:
            del self.causal_factors[name]
            self.logger.info(f"Removed low-predictive-power factor: {name}")

    async def learn_from_outcome(self, decision: Any):
        """
        Learn from a specific decision outcome.

        Args:
            decision: Decision with actual outcome recorded
        """
        if not (hasattr(decision, 'actual_outcome') and decision.actual_outcome):
            return

        # Extract lesson
        lesson = self._extract_lesson(decision)

        if lesson:
            self.lessons_learned.append(lesson)
            self.logger.info(f"Learned lesson: {lesson.lesson}")

        # Store scenario
        scenario = self._create_scenario_record(decision)
        if scenario:
            self.scenario_database.append(scenario)

    async def retrieve_similar_scenarios(
        self,
        changes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical scenarios.

        Args:
            changes: Current market changes

        Returns:
            List of similar scenarios from history
        """
        if not self.scenario_database:
            return []

        # Simple similarity matching based on change types
        similar_scenarios = []

        for scenario in self.scenario_database:
            scenario_changes = scenario.get("changes", [])

            # Calculate similarity
            similarity = self._calculate_scenario_similarity(changes, scenario_changes)

            if similarity > 0.5:  # Threshold for similarity
                similar_scenarios.append({
                    "scenario": scenario,
                    "similarity": similarity
                })

        # Sort by similarity
        similar_scenarios.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top 5
        return similar_scenarios[:5]

    def _find_common_patterns(self, decisions: List[Any]) -> Dict[str, int]:
        """Find common factors/patterns in a set of decisions."""
        factor_counts = defaultdict(int)

        for decision in decisions:
            metadata = decision.metadata if hasattr(decision, 'metadata') else {}

            if "analysis_results" in metadata:
                analysis = metadata["analysis_results"]

                if "rlm_decomposition" in analysis:
                    rlm = analysis["rlm_decomposition"]
                    key_factors = rlm.get("key_factors", [])

                    for factor_dict in key_factors:
                        if factor_dict.get("importance", 0) > 0.6:  # Only high-importance factors
                            factor_name = factor_dict.get("name", "unknown")
                            factor_counts[factor_name] += 1

        return dict(factor_counts)

    def _extract_lesson(self, decision: Any) -> Optional[LessonLearned]:
        """Extract lesson learned from a decision outcome."""
        outcome = decision.actual_outcome if hasattr(decision, 'actual_outcome') else None
        expected = decision.expected_outcome if hasattr(decision, 'expected_outcome') else None

        if not outcome or not expected:
            return None

        # Check if outcome matched expectation
        outcome_matched = (
            ("positive" in outcome.lower() and "positive" in expected.lower()) or
            ("negative" in outcome.lower() and "negative" in expected.lower())
        )

        if not outcome_matched:
            # Unexpected outcome - this is a learning opportunity
            lesson = LessonLearned(
                lesson=f"Expected {expected}, but got {outcome}. Review decision process.",
                context=f"Decision: {decision.reasoning if hasattr(decision, 'reasoning') else 'N/A'}",
                outcome=outcome,
                generality="moderate",
                applicable_scenarios=["investment_decision"],
                timestamp=datetime.utcnow().isoformat()
            )
            return lesson

        # Even when outcome matched, extract general lessons
        if "positive" in outcome.lower():
            lesson = LessonLearned(
                lesson=f"Decision process leading to positive outcome should be reinforced",
                context=f"Confidence: {decision.confidence if hasattr(decision, 'confidence') else 'N/A'}",
                outcome=outcome,
                generality="specific",
                applicable_scenarios=["high_confidence_decisions"],
                timestamp=datetime.utcnow().isoformat()
            )
            return lesson

        return None

    def _create_scenario_record(self, decision: Any) -> Optional[Dict[str, Any]]:
        """Create a scenario record from a decision."""
        metadata = decision.metadata if hasattr(decision, 'metadata') else {}

        if "review_data" not in metadata:
            return None

        review_data = metadata["review_data"]

        scenario = {
            "timestamp": decision.timestamp.isoformat() if hasattr(decision, 'timestamp') else datetime.utcnow().isoformat(),
            "changes": review_data.get("changes", []),
            "market_context": review_data.get("market_context", {}),
            "decision_type": decision.decision_type if hasattr(decision, 'decision_type') else "unknown",
            "outcome": decision.actual_outcome if hasattr(decision, 'actual_outcome') else None,
            "performance": decision.performance_metrics if hasattr(decision, 'performance_metrics') else None
        }

        return scenario

    def _calculate_scenario_similarity(
        self,
        changes1: List[Dict[str, Any]],
        changes2: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity between two sets of changes."""
        if not changes1 or not changes2:
            return 0.0

        # Simple Jaccard similarity based on change types
        types1 = set(c.get("type", c.get("category", "unknown")) for c in changes1)
        types2 = set(c.get("type", c.get("category", "unknown")) for c in changes2)

        intersection = len(types1 & types2)
        union = len(types1 | types2)

        if union == 0:
            return 0.0

        return intersection / union

    def get_top_predictive_factors(self, n: int = 10) -> List[CausalFactor]:
        """Get top N predictive factors."""
        factors = list(self.causal_factors.values())

        # Filter by minimum sample size
        factors = [f for f in factors if f.sample_count >= 5]

        # Sort by predictive power
        factors.sort(key=lambda x: (x.predictive_power, x.confidence), reverse=True)

        return factors[:n]

    def get_applicable_heuristics(self, context: Dict[str, Any]) -> List[InvestmentHeuristic]:
        """Get heuristics applicable to current context."""
        applicable = []

        for heuristic in self.heuristics:
            # Check if conditions match context
            conditions_met = True

            for condition in heuristic.conditions:
                # Simple keyword matching (could be more sophisticated)
                condition_lower = condition.lower()
                context_str = str(context).lower()

                if condition_lower not in context_str:
                    conditions_met = False
                    break

            if conditions_met and heuristic.success_rate > 0.6:
                applicable.append(heuristic)

        # Sort by success rate and confidence
        applicable.sort(key=lambda h: (h.success_rate * h.confidence), reverse=True)

        return applicable

    def get_recent_lessons(self, days: int = 30) -> List[LessonLearned]:
        """Get lessons learned in the past N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        recent = [
            lesson for lesson in self.lessons_learned
            if datetime.fromisoformat(lesson.timestamp) > cutoff
        ]

        # Sort by recency
        recent.sort(key=lambda l: l.timestamp, reverse=True)

        return recent

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of learned knowledge."""
        return {
            "total_causal_factors": len(self.causal_factors),
            "total_heuristics": len(self.heuristics),
            "total_lessons": len(self.lessons_learned),
            "total_scenarios": len(self.scenario_database),
            "top_predictive_factors": [
                {
                    "name": f.name,
                    "predictive_power": f.predictive_power,
                    "confidence": f.confidence,
                    "sample_count": f.sample_count
                }
                for f in self.get_top_predictive_factors(5)
            ],
            "most_reliable_heuristics": [
                {
                    "rule": h.rule,
                    "success_rate": h.success_rate,
                    "confidence": h.confidence,
                    "times_applied": h.times_applied
                }
                for h in sorted(
                    self.heuristics,
                    key=lambda x: x.success_rate * x.confidence * x.times_applied,
                    reverse=True
                )[:5]
            ] if self.heuristics else [],
            "recent_lessons": len(self.get_recent_lessons(30))
        }
