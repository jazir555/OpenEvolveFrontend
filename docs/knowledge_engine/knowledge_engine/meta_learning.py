"""
Meta-Learning System

Learns across workflow instances to extract transferable patterns.
Supports pattern extraction, transfer learning, and strategy recommendation.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, UTC
from collections import defaultdict
import numpy as np
import logging

from .schemas.long_horizon import (
    MetaPattern,
    StrategyRecommendation,
    LearningOutcome,
    OutcomeType
)


logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract problem features for meta-learning

    Features describe problem characteristics that predict success:
    - Domain (finance, science, etc.)
    - Scale (number of variables, constraints)
    - Evaluation cost (time, resources)
    - Complexity (linear, nonlinear, discrete)
    - Data availability (full, partial, none)
    """

    def extract_features(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from problem description

        Args:
            problem: Problem description

        Returns:
            Feature dictionary
        """
        features = {}

        # Domain
        features["domain"] = problem.get("domain", "general")

        # Scale features
        features["num_variables"] = problem.get("num_variables", 0)
        features["num_constraints"] = problem.get("num_constraints", 0)
        features["scale_category"] = self._categorize_scale(features["num_variables"])

        # Evaluation cost
        features["evaluation_cost"] = problem.get("evaluation_cost", "medium")
        features["time_per_eval"] = problem.get("time_per_eval", 1.0)

        # Complexity
        features["problem_type"] = problem.get("problem_type", "unknown")
        features["is_discrete"] = problem.get("is_discrete", False)
        features["is_constrained"] = problem.get("is_constrained", False)

        # Data availability
        features["has_data"] = problem.get("has_data", False)
        features["data_size"] = problem.get("data_size", 0)

        # Solution space
        features["search_space_size"] = problem.get("search_space_size", "large")

        return features

    def _categorize_scale(self, num_variables: int) -> str:
        """Categorize problem scale"""
        if num_variables < 10:
            return "small"
        elif num_variables < 100:
            return "medium"
        else:
            return "large"


class MetaLearner:
    """
    Learn across workflow instances

    Extracts patterns from past workflows, transfers knowledge across domains,
    and recommends strategies for new problems.

    Usage:
        learner = MetaLearner()

        # Extract patterns from past workflows
        await learner.extract_patterns(workflows)

        # Recommend strategy for new problem
        recommendation = await learner.recommend_strategy(
            problem={"domain": "finance", "num_variables": 50}
        )

        # Transfer knowledge
        await learner.transfer_knowledge(
            source_domain="trading",
            target_domain="finance"
        )
    """

    def __init__(
        self,
        min_evidence: int = 3,
        confidence_threshold: float = 0.7,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """
        Initialize meta-learner

        Args:
            min_evidence: Minimum workflows to support a pattern
            confidence_threshold: Minimum confidence for patterns
            feature_extractor: Feature extractor instance
        """
        self.min_evidence = min_evidence
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = feature_extractor or FeatureExtractor()

        # Pattern storage
        self.patterns: List[MetaPattern] = []

        # Workflow database
        # Key: workflow_id -> workflow data
        self.workflows: Dict[str, Dict[str, Any]] = {}

        # Pattern index
        # Key: domain -> List[pattern_ids]
        self.patterns_by_domain: Dict[str, List[str]] = defaultdict(list)

        # Feature similarity cache
        self.similarity_cache: Dict[Tuple[str, str], float] = {}

    async def extract_patterns(
        self,
        workflows: List[Dict[str, Any]]
    ) -> List[MetaPattern]:
        """
        Extract meta-patterns from workflow runs

        Args:
            workflows: List of workflow execution records

        Returns:
            Extracted patterns
        """
        # Store workflows
        for wf in workflows:
            wf_id = wf.get("workflow_id", f"wf_{datetime.now(UTC).isoformat()}")
            self.workflows[wf_id] = wf

        # Extract patterns by domain
        domain_workflows = defaultdict(list)
        for wf in workflows:
            domain = wf.get("domain", "general")
            domain_workflows[domain].append(wf)

        patterns = []

        # Extract patterns for each domain
        for domain, wf_list in domain_workflows.items():
            domain_patterns = await self._extract_domain_patterns(domain, wf_list)
            patterns.extend(domain_patterns)

        # Filter by confidence and evidence
        valid_patterns = [
            p for p in patterns
            if p.confidence >= self.confidence_threshold
            and len(p.evidence) >= self.min_evidence
        ]

        # Add to pattern database
        for pattern in valid_patterns:
            self._add_pattern(pattern)

        logger.info(
            f"Extracted {len(valid_patterns)} patterns from {len(workflows)} workflows"
        )

        return valid_patterns

    async def _extract_domain_patterns(
        self,
        domain: str,
        workflows: List[Dict[str, Any]]
    ) -> List[MetaPattern]:
        """
        Extract patterns for a specific domain
        """
        patterns = []

        # Pattern 1: Successful strategy patterns
        strategy_patterns = await self._extract_strategy_patterns(domain, workflows)
        patterns.extend(strategy_patterns)

        # Pattern 2: Parameter patterns
        param_patterns = await self._extract_parameter_patterns(domain, workflows)
        patterns.extend(param_patterns)

        # Pattern 3: Feature-performance patterns
        feature_patterns = await self._extract_feature_patterns(domain, workflows)
        patterns.extend(feature_patterns)

        return patterns

    async def _extract_strategy_patterns(
        self,
        domain: str,
        workflows: List[Dict[str, Any]]
    ) -> List[MetaPattern]:
        """
        Extract patterns in successful strategies
        """
        patterns = []

        # Group by strategy
        by_strategy = defaultdict(list)
        for wf in workflows:
            strategy = wf.get("strategy", "unknown")
            by_strategy[strategy].append(wf)

        # Find successful strategies
        for strategy, wf_list in by_strategy.items():
            # Calculate success rate
            successful = [
                wf for wf in wf_list
                if wf.get("outcome_type") == OutcomeType.SUCCESS.value
                or wf.get("success", False)
            ]

            if len(successful) < self.min_evidence:
                continue

            success_rate = len(successful) / len(wf_list)

            if success_rate >= 0.7:  # 70% success threshold
                # Extract common features
                features = self._extract_common_features(successful)

                pattern = MetaPattern(
                    pattern_id=f"strategy_{domain}_{strategy}_{datetime.now(UTC).strftime('%Y%m%d')}",
                    description=f"Strategy '{strategy}' works well for {domain}",
                    applicable_domains=[domain],
                    evidence=[wf.get("workflow_id", "") for wf in successful],
                    confidence=success_rate,
                    feature_signature=features,
                    expected_benefit=success_rate * 0.3  # Estimate
                )

                patterns.append(pattern)

        return patterns

    async def _extract_parameter_patterns(
        self,
        domain: str,
        workflows: List[Dict[str, Any]]
    ) -> List[MetaPattern]:
        """
        Extract patterns in effective parameters
        """
        patterns = []

        # Group by configuration
        by_config = defaultdict(list)
        for wf in workflows:
            config = wf.get("config", {})
            # Create config signature
            sig = self._config_signature(config)
            by_config[sig].append(wf)

        # Find effective configs
        for config_sig, wf_list in by_config.items():
            if len(wf_list) < self.min_evidence:
                continue

            # Calculate average performance
            avg_fitness = np.mean([
                wf.get("fitness", wf.get("metrics", {}).get("fitness", 0))
                for wf in wf_list
            ])

            if avg_fitness > 0.7:  # High performance threshold
                pattern = MetaPattern(
                    pattern_id=f"param_{domain}_{config_sig[:20]}_{datetime.now(UTC).strftime('%Y%m%d')}",
                    description=f"Configuration pattern effective for {domain}",
                    applicable_domains=[domain],
                    evidence=[wf.get("workflow_id", "") for wf in wf_list],
                    confidence=0.8,
                    feature_signature={"config_pattern": config_sig},
                    expected_benefit=avg_fitness * 0.2
                )

                patterns.append(pattern)

        return patterns

    async def _extract_feature_patterns(
        self,
        domain: str,
        workflows: List[Dict[str, Any]]
    ) -> List[MetaPattern]:
        """
        Extract patterns linking features to success
        """
        patterns = []

        # Extract features for all workflows
        featured_workflows = []
        for wf in workflows:
            features = self.feature_extractor.extract_features(wf)
            featured_workflows.append({
                "workflow": wf,
                "features": features,
                "success": wf.get("outcome_type") == OutcomeType.SUCCESS.value
            })

        # Find features correlated with success
        feature_success = defaultdict(lambda: {"success": 0, "total": 0})

        for fw in featured_workflows:
            for key, value in fw["features"].items():
                if isinstance(value, (str, bool)):
                    feature_key = f"{key}={value}"
                    feature_success[feature_key]["total"] += 1
                    if fw["success"]:
                        feature_success[feature_key]["success"] += 1

        # Identify high-success features
        for feature, stats in feature_success.items():
            if stats["total"] < self.min_evidence:
                continue

            success_rate = stats["success"] / stats["total"]

            if success_rate >= 0.8:
                pattern = MetaPattern(
                    pattern_id=f"feature_{domain}_{feature}_{datetime.now(UTC).strftime('%Y%m%d')}",
                    description=f"Feature '{feature}' correlates with success in {domain}",
                    applicable_domains=[domain],
                    evidence=[],
                    confidence=success_rate,
                    feature_signature={"feature": feature},
                    expected_benefit=success_rate * 0.15
                )

                patterns.append(pattern)

        return patterns

    async def recommend_strategy(
        self,
        problem: Dict[str, Any]
    ) -> StrategyRecommendation:
        """
        Recommend strategy for a new problem

        Args:
            problem: Problem description

        Returns:
            Strategy recommendation
        """
        # Extract features
        features = self.feature_extractor.extract_features(problem)
        domain = features["domain"]

        # Find relevant patterns
        relevant_patterns = [
            p for p in self.patterns
            if domain in p.applicable_domains
            or "general" in p.applicable_domains
        ]

        if not relevant_patterns:
            # No patterns found, return default
            return StrategyRecommendation(
                problem_id=problem.get("problem_id", "unknown"),
                recommended_strategy="hybrid",
                confidence=0.3,
                rationale="No relevant patterns found. Using default hybrid strategy.",
                expected_performance=0.5,
                alternative_strategies=[("pes", 0.5), ("qd", 0.5)],
                transfer_source=None
            )

        # Score patterns by feature similarity
        scored_patterns = []
        for pattern in relevant_patterns:
            similarity = self._feature_similarity(features, pattern.feature_signature)
            scored_patterns.append((pattern, similarity))

        # Sort by similarity
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        # Get best pattern
        best_pattern, best_similarity = scored_patterns[0]

        # Extract strategy recommendation
        if "strategy_" in best_pattern.pattern_id:
            strategy = best_pattern.pattern_id.split("_")[2]
        else:
            strategy = "hybrid"  # Default

        # Alternative strategies
        alternatives = []
        for pattern, similarity in scored_patterns[1:4]:  # Top 3 alternatives
            if "strategy_" in pattern.pattern_id:
                alt_strategy = pattern.pattern_id.split("_")[2]
                alternatives.append((alt_strategy, pattern.expected_benefit))

        return StrategyRecommendation(
            problem_id=problem.get("problem_id", "unknown"),
            recommended_strategy=strategy,
            confidence=min(0.95, best_similarity * best_pattern.confidence),
            rationale=f"Based on {len(best_pattern.evidence)} similar workflows with {best_similarity:.0%} feature similarity",
            expected_performance=best_pattern.expected_benefit,
            alternative_strategies=alternatives,
            transfer_source=domain
        )

    async def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str
    ) -> List[MetaPattern]:
        """
        Transfer knowledge from source to target domain

        Args:
            source_domain: Domain to transfer from
            target_domain: Domain to transfer to

        Returns:
            Transferred patterns
        """
        # Get source patterns
        source_patterns = [
            p for p in self.patterns
            if source_domain in p.applicable_domains
        ]

        if not source_patterns:
            logger.warning(f"No patterns found for source domain {source_domain}")
            return []

        transferred = []

        for pattern in source_patterns:
            # Check if pattern already exists in target
            exists = any(
                p.pattern_id == pattern.pattern_id.replace(source_domain, target_domain)
                for p in self.patterns
            )

            if exists:
                continue

            # Create transferred pattern with lower confidence
            transferred_pattern = MetaPattern(
                pattern_id=pattern.pattern_id.replace(source_domain, target_domain),
                description=f"Transferred from {source_domain}: {pattern.description}",
                applicable_domains=[target_domain],
                evidence=[f"transferred_from_{source_domain}"],
                confidence=pattern.confidence * 0.7,  # Reduce confidence
                feature_signature=pattern.feature_signature.copy(),
                expected_benefit=pattern.expected_benefit * 0.7
            )

            self._add_pattern(transferred_pattern)
            transferred.append(transferred_pattern)

        logger.info(
            f"Transferred {len(transferred)} patterns from "
            f"{source_domain} to {target_domain}"
        )

        return transferred

    def _extract_common_features(
        self,
        workflows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract common features across workflows"""
        all_features = []

        for wf in workflows:
            features = self.feature_extractor.extract_features(wf)
            all_features.append(features)

        if not all_features:
            return {}

        # Find common values
        common = {}
        for key in all_features[0].keys():
            values = set(f.get(key) for f in all_features)
            if len(values) == 1:
                common[key] = values.pop()

        return common

    def _config_signature(self, config: Dict[str, Any]) -> str:
        """Create configuration signature"""
        # Simplified: just sorted keys
        return ",".join(sorted(config.keys()))

    def _feature_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """
        Calculate feature similarity (Jaccard-like)
        """
        cache_key = (str(features1), str(features2))

        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Calculate similarity
        keys1 = set(features1.keys())
        keys2 = set(features2.keys())

        intersection = keys1 & keys2
        union = keys1 | keys2

        if not union:
            similarity = 0.0
        else:
            similarity = len(intersection) / len(union)

            # Check value matches for intersecting keys
            matches = sum(
                1 for key in intersection
                if features1.get(key) == features2.get(key)
            )
            if intersection:
                similarity = similarity * (matches / len(intersection))

        self.similarity_cache[cache_key] = similarity
        return similarity

    def _add_pattern(self, pattern: MetaPattern) -> None:
        """Add pattern to database"""
        self.patterns.append(pattern)

        # Index by domain
        for domain in pattern.applicable_domains:
            if pattern.pattern_id not in self.patterns_by_domain[domain]:
                self.patterns_by_domain[domain].append(pattern.pattern_id)

    def get_patterns(
        self,
        domain: Optional[str] = None
    ) -> List[MetaPattern]:
        """
        Get patterns, optionally filtered by domain

        Args:
            domain: Optional domain filter

        Returns:
            List of patterns
        """
        if domain:
            pattern_ids = self.patterns_by_domain.get(domain, [])
            return [
                p for p in self.patterns
                if p.pattern_id in pattern_ids
            ]
        return self.patterns.copy()

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "min_evidence": self.min_evidence,
            "confidence_threshold": self.confidence_threshold,
            "num_patterns": len(self.patterns),
            "num_workflows": len(self.workflows),
            "patterns": [p.to_dict() for p in self.patterns],
            "domains": list(self.patterns_by_domain.keys())
        }
