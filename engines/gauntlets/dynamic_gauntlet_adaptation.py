"""
Dynamic Gauntlet Adaptation Module

This module provides dynamic adaptation of gauntlets based on performance metrics,
feedback, context, and resource availability.
"""
from __future__ import annotations


import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from workflow_structures import GauntletDefinition, GauntletRoundRule, PerformanceMetrics


class GauntletAdaptationEngine:
    """Adapts gauntlets dynamically based on various factors."""
    
    def __init__(self, adaptation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptation engine.
        
        Args:
            adaptation_config: Configuration for adaptation behavior
        """
        self.config = adaptation_config or self._get_default_config()
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default adaptation configuration."""
        return {
            "enable_performance_adaptation": True,
            "enable_feedback_adaptation": True,
            "enable_contextual_adaptation": True,
            "enable_resource_adaptation": True,
            "adaptation_sensitivity": 0.5,  # 0.0-1.0
            "min_samples_for_adaptation": 3,
            "adaptation_rate": 0.3,  # How quickly to adapt (0.0-1.0)
            "performance_window_hours": 24
        }
    
    def adapt_gauntlet(
        self,
        gauntlet: GauntletDefinition,
        context: Dict[str, Any]
    ) -> GauntletDefinition:
        """
        Adapt a gauntlet based on all available factors.
        
        Args:
            gauntlet: Original gauntlet definition
            context: Context including performance, feedback, resources
            
        Returns:
            Adapted gauntlet definition
        """
        adapted = gauntlet
        
        # Apply performance-based adaptation
        if self.config["enable_performance_adaptation"]:
            adapted = self._adapt_based_on_performance(adapted, context)
        
        # Apply feedback-driven adaptation
        if self.config["enable_feedback_adaptation"]:
            adapted = self._adapt_based_on_feedback(adapted, context)
        
        # Apply contextual adaptation
        if self.config["enable_contextual_adaptation"]:
            adapted = self._adapt_based_on_context(adapted, context)
        
        # Apply resource-aware adaptation
        if self.config["enable_resource_adaptation"]:
            adapted = self._adapt_based_on_resources(adapted, context)
        
        # Record adaptation
        self._record_adaptation(gauntlet, adapted, context)
        
        return adapted
    
    def _adapt_based_on_performance(
        self,
        gauntlet: GauntletDefinition,
        context: Dict[str, Any]
    ) -> GauntletDefinition:
        """
        Adapt gauntlet based on historical performance metrics.
        
        Performance-based adaptation:
        - If gauntlet is too lenient (high pass rate), increase strictness
        - If gauntlet is too strict (low pass rate), decrease strictness
        - Adjust based on false positive/negative rates
        """
        performance_data = context.get("performance_metrics", {})
        
        if not performance_data or gauntlet.name not in self.performance_history:
            return gauntlet
        
        history = self.performance_history[gauntlet.name]
        
        if len(history) < self.config["min_samples_for_adaptation"]:
            return gauntlet
        
        # Calculate performance statistics
        recent_history = self._get_recent_performance(history)
        
        pass_rate = self._calculate_pass_rate(recent_history)
        avg_score = self._calculate_avg_score(recent_history)
        false_positive_rate = performance_data.get("false_positive_rate", 0.0)
        false_negative_rate = performance_data.get("false_negative_rate", 0.0)
        
        # Determine adaptation direction
        adaptation_factor = 0.0
        
        # Too lenient - increase strictness
        if pass_rate > 0.9 or false_negative_rate > 0.2:
            adaptation_factor = -self.config["adaptation_rate"]
        
        # Too strict - decrease strictness
        elif pass_rate < 0.3 or false_positive_rate > 0.2:
            adaptation_factor = self.config["adaptation_rate"]
        
        # Moderate adjustment based on average score
        if 0.4 < pass_rate < 0.8:
            if avg_score < 0.6:
                adaptation_factor = self.config["adaptation_rate"] * 0.5
            elif avg_score > 0.9:
                adaptation_factor = -self.config["adaptation_rate"] * 0.5
        
        # Apply adaptation if significant
        if abs(adaptation_factor) > 0.01:
            return self._adjust_gauntlet_strictness(gauntlet, adaptation_factor)
        
        return gauntlet
    
    def _adapt_based_on_feedback(
        self,
        gauntlet: GauntletDefinition,
        context: Dict[str, Any]
    ) -> GauntletDefinition:
        """
        Adapt gauntlet based on user and system feedback.
        
        Feedback-driven adaptation:
        - User feedback on gauntlet effectiveness
        - System feedback on solution quality
        - Patterns in critique/verification reports
        """
        feedback = context.get("feedback", {})
        
        if not feedback:
            return gauntlet
        
        user_satisfaction = feedback.get("user_satisfaction", 0.5)
        solution_quality = feedback.get("solution_quality", 0.5)
        feedback_patterns = feedback.get("patterns", [])
        
        adaptation_factor = 0.0
        
        # Low user satisfaction - adjust gauntlet
        if user_satisfaction < 0.4:
            # Check if users think it's too strict or too lenient
            if "too_strict" in feedback_patterns:
                adaptation_factor = self.config["adaptation_rate"]
            elif "too_lenient" in feedback_patterns:
                adaptation_factor = -self.config["adaptation_rate"]
        
        # Low solution quality despite passing - increase strictness
        if solution_quality < 0.5 and feedback.get("pass_rate", 0.5) > 0.7:
            adaptation_factor = -self.config["adaptation_rate"]
        
        # High solution quality but low pass rate - decrease strictness
        if solution_quality > 0.8 and feedback.get("pass_rate", 0.5) < 0.4:
            adaptation_factor = self.config["adaptation_rate"]
        
        if abs(adaptation_factor) > 0.01:
            return self._adjust_gauntlet_strictness(gauntlet, adaptation_factor)
        
        return gauntlet
    
    def _adapt_based_on_context(
        self,
        gauntlet: GauntletDefinition,
        context: Dict[str, Any]
    ) -> GauntletDefinition:
        """
        Adapt gauntlet based on problem context.
        
        Contextual adaptation:
        - Problem domain (medical, legal, etc. need higher strictness)
        - Problem complexity
        - Criticality/importance
        - Time constraints
        """
        problem_context = context.get("problem_context", {})
        
        domain = problem_context.get("domain", "general")
        complexity = problem_context.get("complexity", 5)
        criticality = problem_context.get("criticality", "medium")
        time_constraint = problem_context.get("time_constraint", "normal")
        
        adaptation_factor = 0.0
        
        # High-criticality domains need stricter gauntlets
        critical_domains = ["medical", "legal", "financial", "safety_critical"]
        if domain in critical_domains:
            adaptation_factor -= 0.2
        
        # High complexity needs more thorough evaluation
        if complexity > 7:
            adaptation_factor -= 0.1
        
        # High criticality needs stricter evaluation
        if criticality == "high":
            adaptation_factor -= 0.15
        elif criticality == "low":
            adaptation_factor += 0.1
        
        # Time constraints may require faster evaluation
        if time_constraint == "urgent":
            # Reduce rounds but maintain strictness
            return self._reduce_gauntlet_rounds(gauntlet, factor=0.7)
        elif time_constraint == "relaxed":
            # Can afford more thorough evaluation
            adaptation_factor -= 0.1
        
        if abs(adaptation_factor) > 0.01:
            return self._adjust_gauntlet_strictness(gauntlet, adaptation_factor)
        
        return gauntlet
    
    def _adapt_based_on_resources(
        self,
        gauntlet: GauntletDefinition,
        context: Dict[str, Any]
    ) -> GauntletDefinition:
        """
        Adapt gauntlet based on available resources.
        
        Resource-aware adaptation:
        - API call limits
        - Token budgets
        - Time constraints
        - Cost constraints
        """
        resources = context.get("resources", {})
        
        api_calls_remaining = resources.get("api_calls_remaining", float('inf'))
        tokens_remaining = resources.get("tokens_remaining", float('inf'))
        time_remaining = resources.get("time_remaining_seconds", float('inf'))
        cost_remaining = resources.get("cost_remaining", float('inf'))
        
        # Calculate resource pressure (0.0 = abundant, 1.0 = scarce)
        resource_pressure = 0.0
        
        if api_calls_remaining < 100:
            resource_pressure = max(resource_pressure, 0.8)
        elif api_calls_remaining < 500:
            resource_pressure = max(resource_pressure, 0.5)
        
        if tokens_remaining < 10000:
            resource_pressure = max(resource_pressure, 0.8)
        elif tokens_remaining < 50000:
            resource_pressure = max(resource_pressure, 0.5)
        
        if time_remaining < 300:  # 5 minutes
            resource_pressure = max(resource_pressure, 0.9)
        elif time_remaining < 1800:  # 30 minutes
            resource_pressure = max(resource_pressure, 0.6)
        
        if cost_remaining < 1.0:
            resource_pressure = max(resource_pressure, 0.8)
        elif cost_remaining < 5.0:
            resource_pressure = max(resource_pressure, 0.5)
        
        # Adapt based on resource pressure
        if resource_pressure > 0.7:
            # High pressure - reduce rounds and panel sizes
            return self._optimize_for_resources(gauntlet, pressure=resource_pressure)
        elif resource_pressure < 0.3:
            # Low pressure - can afford more thorough evaluation
            return self._enhance_for_quality(gauntlet)
        
        return gauntlet
    
    def _adjust_gauntlet_strictness(
        self,
        gauntlet: GauntletDefinition,
        factor: float
    ) -> GauntletDefinition:
        """
        Adjust gauntlet strictness by modifying thresholds.
        
        Args:
            gauntlet: Original gauntlet
            factor: Adjustment factor (-1.0 to 1.0)
                   Negative = more strict, Positive = less strict
        """
        adapted_rounds = []
        
        for round_rule in gauntlet.rounds:
            # Adjust confidence threshold
            new_confidence = round_rule.min_overall_confidence - (factor * 0.1)
            new_confidence = max(0.0, min(1.0, new_confidence))
            
            # Adjust quorum requirements
            new_required = round_rule.quorum_required_approvals
            if factor < -0.2:  # More strict
                new_required = min(
                    round_rule.quorum_from_panel_size,
                    new_required + 1
                )
            elif factor > 0.2:  # Less strict
                new_required = max(1, new_required - 1)
            
            # Adjust variance threshold
            new_variance = round_rule.max_score_variance
            if new_variance is not None:
                new_variance = new_variance * (1 - factor * 0.2)
                new_variance = max(0.0, min(1.0, new_variance))
            
            adapted_round = GauntletRoundRule(
                round_number=round_rule.round_number,
                quorum_required_approvals=new_required,
                quorum_from_panel_size=round_rule.quorum_from_panel_size,
                min_overall_confidence=new_confidence,
                max_score_variance=new_variance,
                per_judge_requirements=round_rule.per_judge_requirements,
                collaboration_mode=round_rule.collaboration_mode
            )
            adapted_rounds.append(adapted_round)
        
        return GauntletDefinition(
            name=f"{gauntlet.name} (Adapted)",
            team_name=gauntlet.team_name,
            rounds=adapted_rounds,
            description=f"Adapted version of {gauntlet.name}",
            attack_modes=gauntlet.attack_modes,
            generation_mode=gauntlet.generation_mode,
            gauntlet_type=gauntlet.gauntlet_type,
            gauntlet_config=gauntlet.gauntlet_config
        )
    
    def _reduce_gauntlet_rounds(
        self,
        gauntlet: GauntletDefinition,
        factor: float = 0.7
    ) -> GauntletDefinition:
        """Reduce number of rounds while maintaining quality."""
        num_rounds_to_keep = max(1, int(len(gauntlet.rounds) * factor))
        
        # Keep the most important rounds (typically first and last)
        if len(gauntlet.rounds) <= 2:
            kept_rounds = gauntlet.rounds
        else:
            kept_rounds = [gauntlet.rounds[0]]  # First round
            if num_rounds_to_keep > 1:
                kept_rounds.append(gauntlet.rounds[-1])  # Last round
            if num_rounds_to_keep > 2:
                # Add middle rounds
                middle_indices = list(range(1, len(gauntlet.rounds) - 1))
                for i in middle_indices[:num_rounds_to_keep - 2]:
                    kept_rounds.insert(-1, gauntlet.rounds[i])
        
        # Renumber rounds
        renumbered_rounds = []
        for i, round_rule in enumerate(kept_rounds, 1):
            renumbered_round = GauntletRoundRule(
                round_number=i,
                quorum_required_approvals=round_rule.quorum_required_approvals,
                quorum_from_panel_size=round_rule.quorum_from_panel_size,
                min_overall_confidence=round_rule.min_overall_confidence,
                max_score_variance=round_rule.max_score_variance,
                per_judge_requirements=round_rule.per_judge_requirements,
                collaboration_mode=round_rule.collaboration_mode
            )
            renumbered_rounds.append(renumbered_round)
        
        return GauntletDefinition(
            name=f"{gauntlet.name} (Optimized)",
            team_name=gauntlet.team_name,
            rounds=renumbered_rounds,
            description=f"Resource-optimized version of {gauntlet.name}",
            attack_modes=gauntlet.attack_modes,
            generation_mode=gauntlet.generation_mode,
            gauntlet_type=gauntlet.gauntlet_type,
            gauntlet_config=gauntlet.gauntlet_config
        )
    
    def _optimize_for_resources(
        self,
        gauntlet: GauntletDefinition,
        pressure: float
    ) -> GauntletDefinition:
        """Optimize gauntlet for resource constraints."""
        # Reduce rounds based on pressure
        reduction_factor = 1.0 - (pressure * 0.5)
        gauntlet = self._reduce_gauntlet_rounds(gauntlet, reduction_factor)
        
        # Reduce panel sizes
        adapted_rounds = []
        for round_rule in gauntlet.rounds:
            new_panel_size = max(1, int(round_rule.quorum_from_panel_size * reduction_factor))
            new_required = min(round_rule.quorum_required_approvals, new_panel_size)
            
            adapted_round = GauntletRoundRule(
                round_number=round_rule.round_number,
                quorum_required_approvals=new_required,
                quorum_from_panel_size=new_panel_size,
                min_overall_confidence=round_rule.min_overall_confidence,
                max_score_variance=round_rule.max_score_variance,
                per_judge_requirements=round_rule.per_judge_requirements,
                collaboration_mode=round_rule.collaboration_mode
            )
            adapted_rounds.append(adapted_round)
        
        gauntlet.rounds = adapted_rounds
        return gauntlet
    
    def _enhance_for_quality(self, gauntlet: GauntletDefinition) -> GauntletDefinition:
        """Enhance gauntlet for better quality when resources allow."""
        # Increase panel sizes slightly
        adapted_rounds = []
        for round_rule in gauntlet.rounds:
            new_panel_size = round_rule.quorum_from_panel_size + 1
            
            adapted_round = GauntletRoundRule(
                round_number=round_rule.round_number,
                quorum_required_approvals=round_rule.quorum_required_approvals,
                quorum_from_panel_size=new_panel_size,
                min_overall_confidence=round_rule.min_overall_confidence,
                max_score_variance=round_rule.max_score_variance,
                per_judge_requirements=round_rule.per_judge_requirements,
                collaboration_mode=round_rule.collaboration_mode
            )
            adapted_rounds.append(adapted_round)
        
        return GauntletDefinition(
            name=f"{gauntlet.name} (Enhanced)",
            team_name=gauntlet.team_name,
            rounds=adapted_rounds,
            description=f"Quality-enhanced version of {gauntlet.name}",
            attack_modes=gauntlet.attack_modes,
            generation_mode=gauntlet.generation_mode,
            gauntlet_type=gauntlet.gauntlet_type,
            gauntlet_config=gauntlet.gauntlet_config
        )
    
    def record_performance(self, gauntlet_name: str, metrics: PerformanceMetrics):
        """Record performance metrics for a gauntlet."""
        if gauntlet_name not in self.performance_history:
            self.performance_history[gauntlet_name] = []
        
        self.performance_history[gauntlet_name].append(metrics)
    
    def _get_recent_performance(
        self,
        history: List[PerformanceMetrics]
    ) -> List[PerformanceMetrics]:
        """Get recent performance within the configured window."""
        cutoff_time = datetime.now() - timedelta(
            hours=self.config["performance_window_hours"]
        )
        
        return [
            m for m in history
            if datetime.fromtimestamp(m.timestamp) > cutoff_time
        ]
    
    def _calculate_pass_rate(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate pass rate from metrics."""
        if not metrics:
            return 0.5
        
        total_runs = sum(m.metrics.get("total_runs", 0) for m in metrics)
        passed_runs = sum(m.metrics.get("passed_runs", 0) for m in metrics)
        
        return passed_runs / total_runs if total_runs > 0 else 0.5
    
    def _calculate_avg_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate average score from metrics."""
        if not metrics:
            return 0.5
        
        scores = [m.metrics.get("avg_score", 0.5) for m in metrics]
        return sum(scores) / len(scores) if scores else 0.5
    
    def _record_adaptation(
        self,
        original: GauntletDefinition,
        adapted: GauntletDefinition,
        context: Dict[str, Any]
    ):
        """Record adaptation for analysis."""
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "original_gauntlet": original.name,
            "adapted_gauntlet": adapted.name,
            "context": context,
            "changes": self._calculate_changes(original, adapted)
        })
    
    def _calculate_changes(
        self,
        original: GauntletDefinition,
        adapted: GauntletDefinition
    ) -> Dict[str, Any]:
        """Calculate what changed between original and adapted gauntlet."""
        return {
            "num_rounds_changed": len(original.rounds) != len(adapted.rounds),
            "original_rounds": len(original.rounds),
            "adapted_rounds": len(adapted.rounds),
            "strictness_change": self._estimate_strictness_change(original, adapted)
        }
    
    def _estimate_strictness_change(
        self,
        original: GauntletDefinition,
        adapted: GauntletDefinition
    ) -> str:
        """Estimate if gauntlet became more or less strict."""
        if not original.rounds or not adapted.rounds:
            return "unknown"
        
        orig_avg_confidence = sum(
            r.min_overall_confidence for r in original.rounds
        ) / len(original.rounds)
        
        adapted_avg_confidence = sum(
            r.min_overall_confidence for r in adapted.rounds
        ) / len(adapted.rounds)
        
        diff = adapted_avg_confidence - orig_avg_confidence
        
        if diff > 0.05:
            return "less_strict"
        elif diff < -0.05:
            return "more_strict"
        else:
            return "similar"
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptations."""
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptations": len([
                a for a in self.adaptation_history
                if datetime.fromisoformat(a["timestamp"]) > 
                   datetime.now() - timedelta(hours=24)
            ]),
            "strictness_changes": {
                "more_strict": len([
                    a for a in self.adaptation_history
                    if a["changes"]["strictness_change"] == "more_strict"
                ]),
                "less_strict": len([
                    a for a in self.adaptation_history
                    if a["changes"]["strictness_change"] == "less_strict"
                ]),
                "similar": len([
                    a for a in self.adaptation_history
                    if a["changes"]["strictness_change"] == "similar"
                ])
            }
        }
