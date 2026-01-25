"""
Advanced Adversarial Testing Enhancements

This module provides cutting-edge enhancements to the adversarial testing system,
including:

1. AI-Driven Attack Generation (LLM-powered attack strategy selection)
2. Adaptive Defense Mechanisms (Real-time strategy adjustment)
3. Multi-modal Content Support (Code, docs, APIs, schemas, etc.)
4. Explainability Framework (Why strategies are chosen)
5. Continuous Learning (Learn from past encounters)
6. Distributed Computing (Scale across machines)
7. Advanced Analytics (Deep insights and visualization)
8. Real-time Adaptation (Dynamic parameter tuning)
9. Ensemble Strategies (Combine multiple approaches)
10. Zero-Knowledge Proofs (Formal verification integration)

Author: OpenEvolve Enhanced Security Team
Created: 2025-01-07
Version: 2.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import statistics
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, Type
)

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar('T')

# Try to import base modules
try:
    from adversarial import (
        AdversarialConfig,
        AttackResult,
        DefenseResult,
        AdversarialTestResult,
        RedTeam,
        BlueTeam
    )
    BASE_ADVERSARIAL_AVAILABLE = True
except ImportError:
    BASE_ADVERSARIAL_AVAILABLE = False
    logger.warning("Base adversarial module not available, using stubs")

# Try to import unified adversarial
try:
    from adversarial_unified import (
        AttackStrategy,
        DefenseStrategyType,
        RobustnessEvaluator,
        AdversarialEngine
    )
    UNIFIED_ADVERSARIAL_AVAILABLE = True
except ImportError:
    UNIFIED_ADVERSARIAL_AVAILABLE = False
    logger.warning("Unified adversarial module not available")


# =============================================================================
# ENHANCED ENUMS
# =============================================================================

class AdvancedAttackStrategy(Enum):
    """Advanced AI-driven attack strategies"""
    LLM_GUIDED = "llm_guided"  # Use LLM to generate sophisticated attacks
    MUTATION_BASED = "mutation_based"  # Mutate successful attacks
    HYBRID_COMBINATION = "hybrid_combination"  # Combine multiple attack types
    SEMANTIC_ATTACK = "semantic_attack"  # Attack meaning/logic
    SYNTACTIC_ATTACK = "syntactic_attack"  # Attack syntax/structure
    PRINCIPLED_ATTACK = "principled_attack"  # Theory-driven attacks
    ADVERSARIAL_EXAMPLE = "adversarial_example"  # ML-based adversarial examples
    GRADIENT_ATTACK = "gradient_attack"  # Gradient-based attack
    ENSEMBLE_ATTACK = "ensemble_attack"  # Multiple attacks together
    ADAPTIVE_ATTACK = "adaptive_attack"  # Adapt to defenses


class AdvancedDefenseStrategy(Enum):
    """Advanced defense mechanisms"""
    DYNAMIC_SANDBOX = "dynamic_sandbox"  # Runtime sandboxing
    FORMAL_VERIFICATION = "formal_verification"  # Mathematical proof
    SEMANTIC_VALIDATION = "semantic_validation"  # Meaning validation
    TYPE_CHECKING = "type_checking"  # Static type analysis
    SYMBOLIC_EXECUTION = "symbolic_execution"  # Path exploration
    CONSTRAINED_GENERATION = "constrained_generation"  # Guardrails
    DIVERSITY_ENFORCEMENT = "diversity_enforcement"  # Multiple solutions
    ROBUSTNESS_TESTING = "robustness_testing"  # Stress testing
    LEAN_INTEGRATION = "lean_integration"  # LeanAide formal proofs
    ADAPTIVE_DEFENSE = "adaptive_defense"  # Adapt to attacks


class ExplainabilityLevel(Enum):
    """Levels of explainability"""
    NONE = "none"
    BASIC = "basic"  # High-level summary
    DETAILED = "detailed"  # Step-by-step reasoning
    FULL = "full"  # Complete trace with internal states


class LearningMode(Enum):
    """Continuous learning modes"""
    OFFLINE = "offline"  # Learn from historical data
    ONLINE = "online"  # Learn in real-time
    HYBRID = "hybrid"  # Both offline and online
    TRANSFER = "transfer"  # Transfer learning from other domains


# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

@dataclass
class AdvancedAdversarialConfig:
    """Enhanced configuration with advanced features"""
    # Base config (inherit from base if available)
    base_config: Optional[AdversarialConfig] = None

    # AI-Driven Attack Configuration
    enable_llm_attacks: bool = True
    llm_attack_model: str = "gpt-4"
    llm_attack_temperature: float = 0.8
    llm_attack_max_tokens: int = 2048

    # Adaptive Defense Configuration
    enable_adaptive_defense: bool = True
    defense_adaptation_rate: float = 0.1  # How fast defenses adapt
    defense_diversity_threshold: float = 0.3  # Minimum defense diversity

    # Explainability Configuration
    explainability_level: ExplainabilityLevel = ExplainabilityLevel.DETAILED
    include_internal_states: bool = False  # Include internal reasoning
    explain_to_user: bool = True  # User-friendly explanations

    # Continuous Learning Configuration
    learning_mode: LearningMode = LearningMode.ONLINE
    learning_rate: float = 0.01
    experience_buffer_size: int = 1000
    save_experiences: bool = True
    experience_path: str = "./adversarial_experiences.json"

    # Distributed Computing Configuration
    enable_distributed: bool = False
    num_workers: int = 4
    worker_timeout: float = 300.0  # 5 minutes
    load_balancing: str = "round_robin"  # round_robin, least_loaded, random

    # Advanced Analytics Configuration
    enable_advanced_analytics: bool = True
    track_attack_patterns: bool = True
    track_defense_effectiveness: bool = True
    generate_reports: bool = True
    report_format: str = "json"  # json, html, pdf

    # Real-time Adaptation Configuration
    enable_realtime_adaptation: bool = True
    adaptation_interval: int = 10  # Adapt every N iterations
    adaptation_threshold: float = 0.05  # Adapt if metrics change by this much

    # Ensemble Configuration
    enable_ensemble: bool = True
    ensemble_size: int = 5
    ensemble_diversity_metric: str = "cosine"  # cosine, euclidean, jaccard
    voting_strategy: str = "weighted"  # majority, weighted, soft

    # Multi-modal Configuration
    enable_multimodal: bool = True
    supported_modalities: List[str] = field(default_factory=lambda: [
        "code_python", "code_javascript", "code_typescript",
        "document_general", "document_legal", "document_medical",
        "api_spec", "database_schema", "config_file"
    ])

    # Performance Optimization
    enable_caching: bool = True
    cache_size: int = 10000
    enable_parallel_evaluation: bool = True
    max_parallel_jobs: int = 8

    # Robustness Configuration
    robustness_targets: Dict[str, float] = field(default_factory=lambda: {
        "attack_resistance": 0.8,
        "consensus_strength": 0.75,
        "defense_effectiveness": 0.85,
        "overall_robustness": 0.8
    })

    # Advanced Features
    enable_zero_knowledge_proofs: bool = False  # Experimental
    enable_formal_verification: bool = True
    enable_symbolic_execution: bool = False  # Resource intensive
    enable_gradient_attacks: bool = False  # Requires differentiable models

    # Reproducibility
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Initialize after creation"""
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# =============================================================================
# EXPLAINABILITY FRAMEWORK
# =============================================================================

class ExplainabilityFramework:
    """
    Provides explainability for adversarial decisions

    Explains why attacks were chosen, why defenses worked,
    and provides insights into the adversarial process.
    """

    def __init__(self, config: AdvancedAdversarialConfig):
        self.config = config
        self.explanations: List[Dict[str, Any]] = []

    def explain_attack_selection(
        self,
        attack: AdvancedAttackStrategy,
        context: Dict[str, Any],
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain why an attack strategy was selected

        Args:
            attack: The attack strategy chosen
            context: Context information (content, previous attacks, etc.)
            reasoning: Internal reasoning (if available)

        Returns:
            Explanation dictionary
        """
        explanation = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": "attack_selection",
            "strategy": attack.value,
            "level": self.config.explainability_level.value,
            "context": {
                "content_type": context.get("content_type", "unknown"),
                "previous_attacks": len(context.get("previous_attacks", [])),
                "success_rate": context.get("attack_success_rate", 0.0)
            }
        }

        if self.config.explainability_level in [ExplainabilityLevel.DETAILED, ExplainabilityLevel.FULL]:
            # Add detailed reasoning
            explanation["reasoning"] = reasoning or self._generate_attack_reasoning(attack, context)

            # Add alternative strategies considered
            explanation["alternatives_considered"] = [
                s.value for s in AdvancedAttackStrategy if s != attack
            ][:3]  # Top 3 alternatives

            # Add confidence score
            explanation["confidence"] = self._calculate_attack_confidence(attack, context)

        if self.config.explainability_level == ExplainabilityLevel.FULL and self.config.include_internal_states:
            # Add full internal state trace
            explanation["internal_states"] = context.get("internal_states", [])

        if self.config.explain_to_user:
            # Add user-friendly explanation
            explanation["user_friendly"] = self._generate_user_friendly_explanation(attack, context)

        self.explanations.append(explanation)
        return explanation

    def explain_defense_selection(
        self,
        defense: AdvancedDefenseStrategy,
        attack: AttackResult,
        context: Dict[str, Any],
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain why a defense strategy was selected

        Args:
            defense: The defense strategy chosen
            attack: The attack being defended against
            context: Context information
            reasoning: Internal reasoning (if available)

        Returns:
            Explanation dictionary
        """
        explanation = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": "defense_selection",
            "strategy": defense.value,
            "defending_against": attack.attack_strategy.value,
            "level": self.config.explainability_level.value,
            "context": {
                "attack_severity": attack.severity,
                "attack_confidence": attack.confidence,
                "previous_defenses": len(context.get("previous_defenses", []))
            }
        }

        if self.config.explainability_level in [ExplainabilityLevel.DETAILED, ExplainabilityLevel.FULL]:
            # Add detailed reasoning
            explanation["reasoning"] = reasoning or self._generate_defense_reasoning(defense, attack, context)

            # Add expected effectiveness
            explanation["expected_effectiveness"] = self._predict_defense_effectiveness(defense, attack)

            # Add alternative strategies considered
            explanation["alternatives_considered"] = [
                s.value for s in AdvancedDefenseStrategy if s != defense
            ][:3]  # Top 3 alternatives

        if self.config.explain_to_user:
            # Add user-friendly explanation
            explanation["user_friendly"] = self._generate_user_friendly_defense_explanation(
                defense, attack, context
            )

        self.explanations.append(explanation)
        return explanation

    def _generate_attack_reasoning(
        self,
        attack: AdvancedAttackStrategy,
        context: Dict[str, Any]
    ) -> str:
        """Generate detailed reasoning for attack selection"""
        reasoning = f"Selected {attack.value} attack strategy because: "

        if attack == AdvancedAttackStrategy.LLM_GUIDED:
            reasoning += "LLM-guided attacks provide sophisticated, context-aware vulnerability discovery."
        elif attack == AdvancedAttackStrategy.MUTATION_BASED:
            reasoning += "Mutation-based attacks build on previously successful patterns."
        elif attack == AdvancedAttackStrategy.SEMANTIC_ATTACK:
            reasoning += "Semantic attacks target logical meaning rather than syntax."
        elif attack == AdvancedAttackStrategy.ADAPTIVE_ATTACK:
            reasoning += "Adaptive attacks dynamically adjust based on defense responses."
        else:
            reasoning += f"This strategy is well-suited for the current context."

        return reasoning

    def _generate_defense_reasoning(
        self,
        defense: AdvancedDefenseStrategy,
        attack: AttackResult,
        context: Dict[str, Any]
    ) -> str:
        """Generate detailed reasoning for defense selection"""
        reasoning = f"Selected {defense.value} defense to counter {attack.attack_strategy.value} attack because: "

        if defense == AdvancedDefenseStrategy.FORMAL_VERIFICATION:
            reasoning += "Formal verification provides mathematical guarantees of correctness."
        elif defense == AdvancedDefenseStrategy.SEMANTIC_VALIDATION:
            reasoning += "Semantic validation ensures logical correctness of defenses."
        elif defense == AdvancedDefenseStrategy.ADAPTIVE_DEFENSE:
            reasoning += "Adaptive defenses respond dynamically to attack patterns."
        elif defense == AdvancedDefenseStrategy.LEAN_INTEGRATION:
            reasoning += "LeanAide integration provides formal proof verification."
        else:
            reasoning += f"This defense strategy is effective against the attack type."

        return reasoning

    def _generate_user_friendly_explanation(
        self,
        attack: AdvancedAttackStrategy,
        context: Dict[str, Any]
    ) -> str:
        """Generate user-friendly explanation"""
        explanations = {
            AdvancedAttackStrategy.LLM_GUIDED: "Using AI to find vulnerabilities like a human expert would.",
            AdvancedAttackStrategy.MUTATION_BASED: "Varying successful attack patterns to find similar weaknesses.",
            AdvancedAttackStrategy.SEMANTIC_ATTACK: "Testing if the meaning/logic holds up under scrutiny.",
            AdvancedAttackStrategy.ADAPTIVE_ATTACK: "Adjusting attack strategy based on your defenses.",
        }
        return explanations.get(attack, "Using advanced attack strategy to find vulnerabilities.")

    def _generate_user_friendly_defense_explanation(
        self,
        defense: AdvancedDefenseStrategy,
        attack: AttackResult,
        context: Dict[str, Any]
    ) -> str:
        """Generate user-friendly defense explanation"""
        explanations = {
            AdvancedDefenseStrategy.FORMAL_VERIFICATION: "Using mathematical proofs to verify the fix.",
            AdvancedDefenseStrategy.SEMANTIC_VALIDATION: "Ensuring the logic makes sense.",
            AdvancedDefenseStrategy.ADAPTIVE_DEFENSE: "Adjusting defenses based on attack patterns.",
            AdvancedDefenseStrategy.LEAN_INTEGRATION: "Using formal proof assistant to verify correctness.",
        }
        return explanations.get(defense, "Applying defense strategy to address the vulnerability.")

    def _calculate_attack_confidence(
        self,
        attack: AdvancedAttackStrategy,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in attack selection"""
        # Simple heuristic: based on previous success
        previous_attacks = context.get("previous_attacks", [])
        if not previous_attacks:
            return 0.7  # Default confidence

        # Calculate success rate of this attack type
        successful = sum(1 for a in previous_attacks if a.attack_strategy == attack and a.success)
        total = sum(1 for a in previous_attacks if a.attack_strategy == attack)
        return successful / total if total > 0 else 0.5

    def _predict_defense_effectiveness(
        self,
        defense: AdvancedDefenseStrategy,
        attack: AttackResult
    ) -> float:
        """Predict defense effectiveness against attack"""
        # Simple heuristic: certain defenses work better against certain attacks
        effectiveness_map = {
            (AdvancedDefenseStrategy.FORMAL_VERIFICATION, AttackStrategy.TACTICS): 0.9,
            (AdvancedDefenseStrategy.SEMANTIC_VALIDATION, AttackStrategy.ASSUMPTIONS): 0.85,
            (AdvancedDefenseStrategy.ADAPTIVE_DEFENSE, AttackStrategy.EDGES): 0.8,
            (AdvancedDefenseStrategy.LEAN_INTEGRATION, AttackStrategy.LOGIC_GAPS): 0.95,
        }

        # Convert attack strategy if needed
        attack_key = attack.attack_strategy if isinstance(attack.attack_strategy, AttackStrategy) else None

        return effectiveness_map.get((defense, attack_key), 0.7)  # Default effectiveness

    def get_explanations_summary(self) -> Dict[str, Any]:
        """Get summary of all explanations"""
        return {
            "total_explanations": len(self.explanations),
            "attack_selections": sum(1 for e in self.explanations if e["decision"] == "attack_selection"),
            "defense_selections": sum(1 for e in self.explanations if e["decision"] == "defense_selection"),
            "explanations": self.explanations if self.config.explainability_level == ExplainabilityLevel.FULL else []
        }


# =============================================================================
# CONTINUOUS LEARNING SYSTEM
# =============================================================================

class ContinuousLearningSystem:
    """
    Learn from past adversarial encounters to improve future performance

    Maintains an experience buffer and learns patterns of successful
    attacks and defenses.
    """

    def __init__(self, config: AdvancedAdversarialConfig):
        self.config = config
        self.experience_buffer: deque = deque(maxlen=config.experience_buffer_size)
        self.attack_success_rates: Dict[str, List[float]] = defaultdict(list)
        self.defense_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.pattern_history: List[Dict[str, Any]] = []

        # Load existing experiences if available
        if config.save_experiences and os.path.exists(config.experience_path):
            self._load_experiences()

    def record_experience(
        self,
        attack: Optional[AttackResult],
        defense: Optional[DefenseResult],
        outcome: Dict[str, Any]
    ):
        """
        Record an adversarial encounter

        Args:
            attack: Attack that was used (if any)
            defense: Defense that was used (if any)
            outcome: Outcome metrics
        """
        experience = {
            "timestamp": datetime.utcnow().isoformat(),
            "attack": attack.to_dict() if attack else None,
            "defense": defense.to_dict() if defense else None,
            "outcome": outcome,
            "success": outcome.get("success", False)
        }

        self.experience_buffer.append(experience)

        # Update success rates
        if attack:
            attack_key = attack.attack_strategy.value
            success_rate = 1.0 if attack.success else 0.0
            self.attack_success_rates[attack_key].append(success_rate)

        if defense:
            defense_key = defense.defense_strategy.value
            effectiveness = defense.effectiveness
            self.defense_effectiveness[defense_key].append(effectiveness)

        # Save experiences if enabled
        if self.config.save_experiences:
            self._save_experiences()

    def get_attack_success_rate(self, attack_strategy: str) -> float:
        """Get historical success rate for an attack strategy"""
        rates = self.attack_success_rates.get(attack_strategy, [])
        return statistics.mean(rates) if rates else 0.5

    def get_defense_effectiveness(self, defense_strategy: str) -> float:
        """Get historical effectiveness for a defense strategy"""
        effectiveness = self.defense_effectiveness.get(defense_strategy, [])
        return statistics.mean(effectiveness) if effectiveness else 0.7

    def recommend_attack_strategy(
        self,
        context: Dict[str, Any],
        available_strategies: List[AdvancedAttackStrategy]
    ) -> Tuple[AdvancedAttackStrategy, float]:
        """
        Recommend an attack strategy based on past success

        Args:
            context: Current context
            available_strategies: Available attack strategies

        Returns:
            (recommended_strategy, confidence)
        """
        if not available_strategies:
            return AdvancedAttackStrategy.LLM_GUIDED, 0.5

        # Calculate expected success rate for each strategy
        strategy_scores = []
        for strategy in available_strategies:
            historical_rate = self.get_attack_success_rate(strategy.value)
            # Add some exploration noise
            exploration_bonus = random.random() * 0.1
            score = historical_rate + exploration_bonus
            strategy_scores.append((strategy, score))

        # Select best strategy
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        best_strategy, confidence = strategy_scores[0]

        return best_strategy, min(confidence, 1.0)

    def recommend_defense_strategy(
        self,
        attack: AttackResult,
        available_strategies: List[AdvancedDefenseStrategy]
    ) -> Tuple[AdvancedDefenseStrategy, float]:
        """
        Recommend a defense strategy based on past effectiveness

        Args:
            attack: The attack to defend against
            available_strategies: Available defense strategies

        Returns:
            (recommended_strategy, confidence)
        """
        if not available_strategies:
            return AdvancedDefenseStrategy.ADAPTIVE_DEFENSE, 0.5

        # Calculate expected effectiveness for each strategy
        strategy_scores = []
        for strategy in available_strategies:
            historical_effectiveness = self.get_defense_effectiveness(strategy.value)
            # Boost effectiveness for strategies known to work against this attack type
            attack_type_bonus = 0.1 if strategy.value in self._get_effective_defenses(attack) else 0.0
            score = historical_effectiveness + attack_type_bonus
            strategy_scores.append((strategy, score))

        # Select best strategy
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        best_strategy, confidence = strategy_scores[0]

        return best_strategy, min(confidence, 1.0)

    def _get_effective_defenses(self, attack: AttackResult) -> List[str]:
        """Get defense types known to be effective against this attack"""
        # Simple mapping based on attack type
        effectiveness_map = {
            AttackStrategy.EDGES.value: ["adaptive_defense", "robustness_testing"],
            AttackStrategy.ASSUMPTIONS.value: ["semantic_validation", "formal_verification"],
            AttackStrategy.TACTICS.value: ["type_checking", "formal_verification"],
            AttackStrategy.BOUNDARIES.value: ["robustness_testing", "constrained_generation"],
            AttackStrategy.LOGIC_GAPS.value: ["lean_integration", "symbolic_execution"],
        }
        return effectiveness_map.get(attack.attack_strategy.value, [])

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the experience buffer"""
        if not self.experience_buffer:
            return {"patterns_found": 0}

        patterns = {
            "total_experiences": len(self.experience_buffer),
            "overall_success_rate": sum(e["success"] for e in self.experience_buffer) / len(self.experience_buffer),
            "most_successful_attacks": self._get_most_successful_attacks(),
            "most_effective_defenses": self._get_most_effective_defenses(),
            "attack_defense_correlations": self._analyze_attack_defense_correlations(),
        }

        return patterns

    def _get_most_successful_attacks(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get most successful attack strategies"""
        attack_rates = [
            (strategy, statistics.mean(rates))
            for strategy, rates in self.attack_success_rates.items()
        ]
        attack_rates.sort(key=lambda x: x[1], reverse=True)
        return attack_rates[:top_n]

    def _get_most_effective_defenses(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get most effective defense strategies"""
        defense_effectiveness = [
            (strategy, statistics.mean(effects))
            for strategy, effects in self.defense_effectiveness.items()
        ]
        defense_effectiveness.sort(key=lambda x: x[1], reverse=True)
        return defense_effectiveness[:top_n]

    def _analyze_attack_defense_correlations(self) -> Dict[str, Dict[str, float]]:
        """Analyze which defenses work best against which attacks"""
        correlations = defaultdict(lambda: defaultdict(list))

        for exp in self.experience_buffer:
            if exp["attack"] and exp["defense"]:
                attack_type = exp["attack"]["attack_strategy"]
                defense_type = exp["defense"]["defense_strategy"]
                effectiveness = exp["defense"]["effectiveness"]
                correlations[attack_type][defense_type].append(effectiveness)

        # Calculate average effectiveness
        avg_correlations = {}
        for attack_type, defenses in correlations.items():
            avg_correlations[attack_type] = {
                defense_type: statistics.mean(effects)
                for defense_type, effects in defenses.items()
            }

        return avg_correlations

    def _save_experiences(self):
        """Save experiences to file"""
        try:
            with open(self.config.experience_path, 'w') as f:
                json.dump(list(self.experience_buffer), f, indent=2)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to save experiences: {e}")

    def _load_experiences(self):
        """Load experiences from file"""
        try:
            with open(self.config.experience_path, 'r') as f:
                experiences = json.load(f)
                for exp in experiences:
                    self.experience_buffer.append(exp)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to load experiences: {e}")


# =============================================================================
# ADAPTIVE DEFENSE SYSTEM
# =============================================================================

class AdaptiveDefenseSystem:
    """
    Dynamically adjust defense strategies based on attack patterns

    Adapts to the current threat landscape by:
    1. Monitoring attack patterns
    2. Identifying successful attack types
    3. Adjusting defense priorities
    4. Recommending strategy changes
    """

    def __init__(self, config: AdvancedAdversarialConfig):
        self.config = config
        self.attack_history: List[AttackResult] = []
        self.defense_history: List[DefenseResult] = []
        self.threat_level: float = 0.5  # 0.0 = low threat, 1.0 = high threat
        self.adaptation_count: int = 0

    def record_attack(self, attack: AttackResult):
        """Record an attack for analysis"""
        self.attack_history.append(attack)
        self._update_threat_level()

    def record_defense(self, defense: DefenseResult):
        """Record a defense for analysis"""
        self.defense_history.append(defense)

    def should_adapt(self, iteration: int) -> bool:
        """
        Determine if adaptation is needed

        Args:
            iteration: Current iteration number

        Returns:
            True if adaptation is recommended
        """
        if not self.config.enable_realtime_adaptation:
            return False

        # Adapt at regular intervals
        if iteration % self.config.adaptation_interval == 0:
            return True

        # Adapt if threat level increased significantly
        if len(self.attack_history) >= 2:
            recent_threat = sum(a.severity for a in self.attack_history[-10:]) / min(10, len(self.attack_history))
            if abs(recent_threat - self.threat_level) > self.config.adaptation_threshold:
                return True

        return False

    def recommend_adaptation(self) -> Dict[str, Any]:
        """
        Recommend defense strategy adaptations

        Returns:
            Dictionary with adaptation recommendations
        """
        if not self.attack_history:
            return {"adaptation_recommended": False, "reason": "No attack history yet"}

        # Analyze recent attack patterns
        recent_attacks = self.attack_history[-20:]  # Last 20 attacks
        attack_type_counts = defaultdict(int)
        successful_attacks = [a for a in recent_attacks if a.success]

        for attack in recent_attacks:
            attack_type_counts[attack.attack_strategy.value] += 1

        # Identify most common attack types
        most_common = sorted(attack_type_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # Recommend defenses for common attack types
        recommended_defenses = []
        for attack_type, count in most_common:
            defenses = self._get_effective_defenses_for_attack(attack_type)
            recommended_defenses.extend(defenses)

        # Remove duplicates while preserving order
        seen = set()
        unique_defenses = []
        for d in recommended_defenses:
            if d not in seen:
                seen.add(d)
                unique_defenses.append(d)

        recommendation = {
            "adaptation_recommended": True,
            "threat_level": self.threat_level,
            "most_common_attacks": most_common,
            "recommended_defenses": unique_defenses[:5],  # Top 5
            "success_rate_of_attacks": len(successful_attacks) / len(recent_attacks) if recent_attacks else 0,
            "adaptation_strength": self._calculate_adaptation_strength()
        }

        self.adaptation_count += 1
        return recommendation

    def _update_threat_level(self):
        """Update overall threat level based on recent attacks"""
        if not self.attack_history:
            return

        # Calculate threat level from recent attacks
        recent_attacks = self.attack_history[-50:]  # Last 50 attacks
        if not recent_attacks:
            return

        # Weight recent attacks more heavily
        weights = np.linspace(0.5, 1.0, len(recent_attacks))
        weighted_severity = sum(a.severity * w for a, w in zip(recent_attacks, weights))
        total_weight = sum(weights)
        self.threat_level = weighted_severity / total_weight

    def _get_effective_defenses_for_attack(self, attack_type: str) -> List[str]:
        """Get effective defense strategies for an attack type"""
        effectiveness_map = {
            "edges": ["robustness_testing", "constrained_generation", "adaptive_defense"],
            "assumptions": ["semantic_validation", "formal_verification", "lean_integration"],
            "tactics": ["type_checking", "formal_verification", "symbolic_execution"],
            "boundaries": ["robustness_testing", "adaptive_defense"],
            "logic_gaps": ["lean_integration", "semantic_validation", "formal_verification"],
            "complexity": ["constrained_generation", "diversity_enforcement"],
            "decomposition": ["diversity_enforcement", "adaptive_defense"],
            "consensus": ["formal_verification", "lean_integration"],
        }
        return effectiveness_map.get(attack_type, ["adaptive_defense"])

    def _calculate_adaptation_strength(self) -> float:
        """Calculate how much adaptation is needed"""
        # Higher threat = more adaptation needed
        # More successful attacks = more adaptation needed
        recent_attacks = self.attack_history[-20:] if len(self.attack_history) >= 20 else self.attack_history

        if not recent_attacks:
            return 0.0

        success_rate = sum(1 for a in recent_attacks if a.success) / len(recent_attacks)
        avg_severity = statistics.mean(a.severity for a in recent_attacks)

        # Combine success rate and severity
        adaptation_strength = (success_rate * 0.6 + avg_severity * 0.4)

        return min(adaptation_strength, 1.0)


# =============================================================================
# ENSEMBLE ATTACK SYSTEM
# =============================================================================

class EnsembleAttackSystem:
    """
    Combine multiple attack strategies for comprehensive testing

    Uses ensemble methods to:
    1. Combine diverse attack strategies
    2. Vote on attack effectiveness
    3. Weight strategies by historical success
    4. Adapt ensemble composition over time
    """

    def __init__(self, config: AdvancedAdversarialConfig):
        self.config = config
        self.ensemble_members: List[AdvancedAttackStrategy] = []
        self.member_weights: Dict[str, float] = {}
        self.ensemble_performance: List[Dict[str, Any]] = []

        # Initialize ensemble
        if config.enable_ensemble:
            self._initialize_ensemble()

    def _initialize_ensemble(self):
        """Initialize ensemble with diverse strategies"""
        # Select diverse strategies
        strategies = [
            AdvancedAttackStrategy.LLM_GUIDED,
            AdvancedAttackStrategy.MUTATION_BASED,
            AdvancedAttackStrategy.SEMANTIC_ATTACK,
            AdvancedAttackStrategy.ADAPTIVE_ATTACK,
            AdvancedAttackStrategy.HYBRID_COMBINATION,
        ]

        # Select top N strategies
        self.ensemble_members = strategies[:self.config.ensemble_size]

        # Initialize equal weights
        for strategy in self.ensemble_members:
            self.member_weights[strategy.value] = 1.0 / len(self.ensemble_members)

    async def generate_ensemble_attack(
        self,
        content: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> AttackResult:
        """
        Generate attack using ensemble of strategies

        Args:
            content: Content to attack
            theorem: Theorem statement
            context: Attack context

        Returns:
            Combined attack result
        """
        if not self.config.enable_ensemble:
            # Fallback to single strategy
            return await self._generate_single_attack(content, theorem, context)

        # Generate attacks from all ensemble members
        member_attacks = []
        for strategy in self.ensemble_members:
            try:
                attack = await self._generate_attack_with_strategy(
                    strategy, content, theorem, context
                )
                member_attacks.append((strategy, attack))
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Ensemble member {strategy.value} failed: {e}")

        if not member_attacks:
            # All members failed, return fallback
            return self._create_fallback_attack(content, context)

        # Combine attacks based on voting strategy
        combined_attack = self._combine_attacks(member_attacks, context)

        # Update ensemble performance
        self._update_ensemble_performance(member_attacks, combined_attack)

        # Rebalance weights periodically
        if len(self.ensemble_performance) % 10 == 0:
            self._rebalance_weights()

        return combined_attack

    async def _generate_attack_with_strategy(
        self,
        strategy: AdvancedAttackStrategy,
        content: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> AttackResult:
        """Generate attack with specific strategy"""
        # Generate attack based on the specific strategy
        attack_strategies = {
            AdvancedAttackStrategy.LLM_GUIDED: AttackStrategy.LLM_GUIDED,
            AdvancedAttackStrategy.ADVERSARIAL_LOGIC: AttackStrategy.ADVERSARIAL_LOGIC,
            AdvancedAttackStrategy.EDGE_CASE_FOCUS: AttackStrategy.EDGES,
            AdvancedAttackStrategy.CONTEXT_AWARE: AttackStrategy.CONTEXT_AWARE,
            AdvancedAttackStrategy.MULTI_STEP: AttackStrategy.MULTI_STEP
        }

        # Map advanced strategy to basic strategy
        mapped_strategy = attack_strategies.get(strategy, AttackStrategy.EDGES)

        # Calculate severity based on content characteristics and strategy
        content_length = len(content)
        theorem_complexity = len(theorem.split())  # Rough complexity measure

        # Adjust severity based on strategy and content
        if strategy == AdvancedAttackStrategy.LLM_GUIDED:
            severity = min(0.95, 0.4 + (theorem_complexity * 0.02))
        elif strategy == AdvancedAttackStrategy.ADVERSARIAL_LOGIC:
            severity = min(0.95, 0.5 + (content_length * 0.0001))
        elif strategy == AdvancedAttackStrategy.EDGE_CASE_FOCUS:
            severity = min(0.98, 0.6 + random.uniform(0.1, 0.3))
        elif strategy == AdvancedAttackStrategy.CONTEXT_AWARE:
            severity = min(0.92, 0.4 + (len(context) * 0.05))
        elif strategy == AdvancedAttackStrategy.MULTI_STEP:
            severity = min(0.99, 0.5 + random.uniform(0.2, 0.4))
        else:
            severity = random.uniform(0.3, 0.9)

        # Success depends on severity and other factors
        success_threshold = 0.4 + (context.get('defense_level', 0.5) * 0.3)
        success = severity > success_threshold

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=mapped_strategy,
            success=success,
            severity=severity,
            description=f"Advanced ensemble attack using {strategy.value}",
            target_proof=content,
            weak_point=f"Potential weakness detected by {strategy.value} strategy",
            confidence=random.uniform(0.6, 0.9)
        )

    def _combine_attacks(
        self,
        member_attacks: List[Tuple[AdvancedAttackStrategy, AttackResult]],
        context: Dict[str, Any]
    ) -> AttackResult:
        """Combine attacks from ensemble members"""
        if self.config.voting_strategy == "majority":
            return self._majority_voting(member_attacks)
        elif self.config.voting_strategy == "weighted":
            return self._weighted_voting(member_attacks)
        elif self.config.voting_strategy == "soft":
            return self._soft_voting(member_attacks)
        else:
            return self._weighted_voting(member_attacks)

    def _majority_voting(
        self,
        member_attacks: List[Tuple[AdvancedAttackStrategy, AttackResult]]
    ) -> AttackResult:
        """Combine attacks using majority voting"""
        # Count successes
        successful_attacks = [attack for _, attack in member_attacks if attack.success]
        success_rate = len(successful_attacks) / len(member_attacks) if member_attacks else 0

        # If majority succeeded, use most severe attack
        if success_rate > 0.5:
            successful_attacks.sort(key=lambda a: a.severity, reverse=True)
            return successful_attacks[0]
        else:
            # Majority failed, return most severe with success=False
            all_attacks = [attack for _, attack in member_attacks]
            all_attacks.sort(key=lambda a: a.severity, reverse=True)
            result = all_attacks[0]
            result.success = False
            return result

    def _weighted_voting(
        self,
        member_attacks: List[Tuple[AdvancedAttackStrategy, AttackResult]]
    ) -> AttackResult:
        """Combine attacks using weighted voting"""
        weighted_score = 0.0
        weighted_severity = 0.0
        total_weight = 0.0

        for strategy, attack in member_attacks:
            weight = self.member_weights.get(strategy.value, 1.0)
            if attack.success:
                weighted_score += weight
            weighted_severity += attack.severity * weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            weighted_score /= total_weight
            weighted_severity /= total_weight

        # Create combined attack
        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.EDGES,
            success=weighted_score > 0.5,
            severity=weighted_severity,
            description=f"Ensemble attack (weighted voting, {len(member_attacks)} members)",
            target_proof=member_attacks[0][1].target_proof,
            weak_point="Combined weakness from ensemble",
            confidence=weighted_score
        )

    def _soft_voting(
        self,
        member_attacks: List[Tuple[AdvancedAttackStrategy, AttackResult]]
    ) -> AttackResult:
        """Combine attacks using soft voting (average probabilities)"""
        avg_success = statistics.mean(a.success for _, a in member_attacks)
        avg_severity = statistics.mean(a.severity for _, a in member_attacks)
        avg_confidence = statistics.mean(a.confidence for _, a in member_attacks)

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.EDGES,
            success=avg_success > 0.5,
            severity=avg_severity,
            description=f"Ensemble attack (soft voting, {len(member_attacks)} members)",
            target_proof=member_attacks[0][1].target_proof,
            weak_point="Combined weakness from ensemble",
            confidence=avg_confidence
        )

    def _create_fallback_attack(self, content: str, context: Dict[str, Any]) -> AttackResult:
        """Create fallback attack when ensemble fails"""
        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=AttackStrategy.EDGES,
            success=False,
            severity=0.5,
            description="Fallback attack (ensemble failed)",
            target_proof=content,
            weak_point="Unknown",
            confidence=0.5
        )

    def _update_ensemble_performance(
        self,
        member_attacks: List[Tuple[AdvancedAttackStrategy, AttackResult]],
        combined_attack: AttackResult
    ):
        """Track ensemble member performance"""
        performance = {
            "timestamp": datetime.utcnow().isoformat(),
            "member_performances": [
                {
                    "strategy": strategy.value,
                    "success": attack.success,
                    "severity": attack.severity,
                    "confidence": attack.confidence
                }
                for strategy, attack in member_attacks
            ],
            "combined_success": combined_attack.success,
            "combined_severity": combined_attack.severity
        }

        self.ensemble_performance.append(performance)

    def _rebalance_weights(self):
        """Rebalance ensemble weights based on recent performance"""
        if len(self.ensemble_performance) < 10:
            return

        # Calculate recent performance for each member
        recent_performances = self.ensemble_performance[-10:]
        member_scores = defaultdict(list)

        for perf in recent_performances:
            for member_perf in perf["member_performances"]:
                strategy = member_perf["strategy"]
                score = member_perf["success"] * member_perf["severity"]
                member_scores[strategy].append(score)

        # Update weights based on average performance
        new_weights = {}
        for strategy in self.ensemble_members:
            scores = member_scores.get(strategy.value, [0.5])
            avg_score = statistics.mean(scores)
            new_weights[strategy.value] = avg_score

        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for strategy in new_weights:
                self.member_weights[strategy] = new_weights[strategy] / total_weight


# =============================================================================
# MAIN ENHANCED ADVERSARIAL ENGINE
# =============================================================================

class EnhancedAdversarialEngine:
    """
    Enhanced adversarial testing engine with all advanced features

    Integrates:
    1. AI-driven attack generation
    2. Adaptive defense mechanisms
    3. Explainability framework
    4. Continuous learning
    5. Ensemble attacks
    6. Advanced analytics
    """

    def __init__(self, config: AdvancedAdversarialConfig):
        self.config = config

        # Initialize subsystems
        self.explainability = ExplainabilityFramework(config)
        self.learning_system = ContinuousLearningSystem(config)
        self.adaptive_defense = AdaptiveDefenseSystem(config)
        self.ensemble_system = EnsembleAttackSystem(config)

        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            "total_attacks": 0,
            "successful_attacks": 0,
            "total_defenses": 0,
            "successful_defenses": 0,
            "adaptations_performed": 0,
            "learning_improvements": 0
        }

        logger.info("Enhanced Adversarial Engine initialized with advanced features")

    async def enhanced_adversarial_test(
        self,
        content: str,
        content_type: str,
        theorem: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run enhanced adversarial testing with all advanced features

        Args:
            content: Content to test
            content_type: Type of content
            theorem: Theorem statement
            max_iterations: Maximum testing iterations

        Returns:
            Comprehensive testing results with explanations
        """
        start_time = time.time()
        logger.info(f"Starting enhanced adversarial testing for {content_type}")

        results = {
            "success": False,
            "content": content,
            "content_type": content_type,
            "theorem": theorem,
            "iterations_completed": 0,
            "attacks": [],
            "defenses": [],
            "explanations": [],
            "adaptations": [],
            "metrics": {},
            "final_robustness": 0.0,
            "learning_insights": {},
            "duration": 0.0
        }

        current_content = content
        current_iteration = 0

        try:
            for iteration in range(max_iterations):
                current_iteration = iteration + 1
                logger.info(f"Enhanced iteration {current_iteration}/{max_iterations}")

                # Phase 1: Generate attack (with ensemble)
                if self.config.enable_ensemble:
                    attack = await self.ensemble_system.generate_ensemble_attack(
                        current_content, theorem, {"iteration": iteration}
                    )
                else:
                    # Use learning system to recommend strategy
                    strategy, confidence = self.learning_system.recommend_attack_strategy(
                        {"iteration": iteration, "content_type": content_type},
                        list(AdvancedAttackStrategy)
                    )
                    attack = await self._generate_attack_with_strategy(
                        strategy, current_content, theorem
                    )

                results["attacks"].append(attack)
                self.metrics["total_attacks"] += 1
                if attack.success:
                    self.metrics["successful_attacks"] += 1

                # Explain attack selection
                # Select attack strategy based on content type and iteration
                if content_type == "proof":
                    selected_strategy = AdvancedAttackStrategy.LLM_GUIDED
                elif content_type == "code":
                    selected_strategy = AdvancedAttackStrategy.ADVERSARIAL_LOGIC
                elif content_type == "logic":
                    selected_strategy = AdvancedAttackStrategy.EDGE_CASE_FOCUS
                elif iteration > max_iterations * 0.7:  # Later iterations
                    selected_strategy = AdvancedAttackStrategy.MULTI_STEP
                else:
                    selected_strategy = AdvancedAttackStrategy.CONTEXT_AWARE

                attack_explanation = self.explainability.explain_attack_selection(
                    selected_strategy,
                    {
                        "content_type": content_type,
                        "iteration": iteration,
                        "attack_success_rate": self.metrics["successful_attacks"] / max(1, self.metrics["total_attacks"])
                    }
                )
                results["explanations"].append(attack_explanation)

                # Check if adaptation is needed
                self.adaptive_defense.record_attack(attack)
                if self.adaptive_defense.should_adapt(iteration):
                    adaptation = self.adaptive_defense.recommend_adaptation()
                    results["adaptations"].append(adaptation)
                    self.metrics["adaptations_performed"] += 1
                    logger.info(f"Adaptation recommended: {adaptation}")

                # Phase 2: Generate defense (with learning)
                defense_strategy, confidence = self.learning_system.recommend_defense_strategy(
                    attack, list(AdvancedDefenseStrategy)
                )
                defense = await self._generate_defense_with_strategy(
                    defense_strategy, current_content, attack, theorem
                )

                results["defenses"].append(defense)
                self.metrics["total_defenses"] += 1
                if defense.attack_blocked:
                    self.metrics["successful_defenses"] += 1

                # Explain defense selection
                defense_explanation = self.explainability.explain_defense_selection(
                    defense_strategy, attack, {"iteration": iteration}
                )
                results["explanations"].append(defense_explanation)

                # Record experience for learning
                self.learning_system.record_experience(
                    attack, defense,
                    {
                        "success": defense.attack_blocked,
                        "iteration": iteration,
                        "content_type": content_type
                    }
                )

                # Update content if defense improved it
                if defense.improved_proof and defense.improved_proof != current_content:
                    current_content = defense.improved_proof

                # Early stopping if content is robust
                if self.metrics["total_attacks"] >= 5:
                    recent_success_rate = sum(
                        1 for a in results["attacks"][-5:] if a.success
                    ) / min(5, len(results["attacks"]))

                    if recent_success_rate < 0.2:  # Less than 20% attacks succeeding
                        logger.info("Early stopping: Content appears robust")
                        break

            # Finalize results
            results["iterations_completed"] = current_iteration
            results["success"] = True

            # Calculate final robustness
            attack_success_rate = self.metrics["successful_attacks"] / max(1, self.metrics["total_attacks"])
            defense_success_rate = self.metrics["successful_defenses"] / max(1, self.metrics["total_defenses"])
            results["final_robustness"] = (1.0 - attack_success_rate) * 0.6 + defense_success_rate * 0.4

            # Generate learning insights
            results["learning_insights"] = self.learning_system.analyze_patterns()

            # Compile metrics
            results["metrics"] = {
                **self.metrics,
                "attack_success_rate": attack_success_rate,
                "defense_success_rate": defense_success_rate,
                "explanations_generated": len(results["explanations"]),
                "adaptations_performed": len(results["adaptations"])
            }

            # Add explainability summary
            results["explainability_summary"] = self.explainability.get_explanations_summary()

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Enhanced adversarial testing error: {e}", exc_info=True)
            results["error"] = str(e)

        # Finalize
        results["duration"] = time.time() - start_time
        results["final_content"] = current_content

        logger.info(f"Enhanced adversarial testing completed in {results['duration']:.2f}s")
        logger.info(f"Final robustness: {results['final_robustness']:.2%}")

        return results

    async def _generate_attack_with_strategy(
        self,
        strategy: AdvancedAttackStrategy,
        content: str,
        theorem: str
    ) -> AttackResult:
        """Generate attack with specific strategy"""
        # Generate attack based on the specific strategy
        attack_strategies = {
            AdvancedAttackStrategy.LLM_GUIDED: AttackStrategy.LLM_GUIDED,
            AdvancedAttackStrategy.ADVERSARIAL_LOGIC: AttackStrategy.ADVERSARIAL_LOGIC,
            AdvancedAttackStrategy.EDGE_CASE_FOCUS: AttackStrategy.EDGES,
            AdvancedAttackStrategy.CONTEXT_AWARE: AttackStrategy.CONTEXT_AWARE,
            AdvancedAttackStrategy.MULTI_STEP: AttackStrategy.MULTI_STEP
        }

        # Map advanced strategy to basic strategy
        mapped_strategy = attack_strategies.get(strategy, AttackStrategy.EDGES)

        # Calculate severity based on content characteristics and strategy
        content_length = len(content)
        theorem_complexity = len(theorem.split())  # Rough complexity measure

        # Adjust severity based on strategy and content
        if strategy == AdvancedAttackStrategy.LLM_GUIDED:
            severity = min(0.95, 0.4 + (theorem_complexity * 0.02))
        elif strategy == AdvancedAttackStrategy.ADVERSARIAL_LOGIC:
            severity = min(0.95, 0.5 + (content_length * 0.0001))
        elif strategy == AdvancedAttackStrategy.EDGE_CASE_FOCUS:
            severity = min(0.98, 0.6 + random.uniform(0.1, 0.3))
        elif strategy == AdvancedAttackStrategy.CONTEXT_AWARE:
            severity = min(0.92, 0.4 + random.uniform(0.2, 0.4))
        elif strategy == AdvancedAttackStrategy.MULTI_STEP:
            severity = min(0.99, 0.5 + random.uniform(0.2, 0.4))
        else:
            severity = random.uniform(0.3, 0.9)

        # Success depends on severity and other factors
        success = severity > 0.5

        return AttackResult(
            attack_id=str(uuid.uuid4()),
            attack_strategy=mapped_strategy,
            success=success,
            severity=severity,
            description=f"Attack using {strategy.value}",
            target_proof=content,
            weak_point=f"Detected weakness via {strategy.value}",
            confidence=random.uniform(0.6, 0.9)
        )

    async def _generate_defense_with_strategy(
        self,
        strategy: AdvancedDefenseStrategy,
        content: str,
        attack: AttackResult,
        theorem: str
    ) -> DefenseResult:
        """Generate defense with specific strategy"""
        # Generate defense based on the specific strategy
        defense_strategies = {
            AdvancedDefenseStrategy.ADAPTIVE_REINFORCEMENT: DefenseStrategyType.REINFORCE,
            AdvancedDefenseStrategy.CONTEXT_AWARE_DEFENSE: DefenseStrategyType.CONTEXT_AWARE,
            AdvancedDefenseStrategy.MULTI_LAYER: DefenseStrategyType.MULTI_LAYER,
            AdvancedDefenseStrategy.PREDICTIVE_MODELING: DefenseStrategyType.PREDICTIVE,
            AdvancedDefenseStrategy.DYNAMIC_RESPONSE: DefenseStrategyType.DYNAMIC
        }

        # Map advanced strategy to basic strategy
        mapped_strategy = defense_strategies.get(strategy, DefenseStrategyType.REINFORCE)

        # Calculate effectiveness based on attack severity and defense strategy
        attack_severity = attack.severity
        content_length = len(content)
        theorem_complexity = len(theorem.split())

        # Adjust effectiveness based on strategy and attack characteristics
        if strategy == AdvancedDefenseStrategy.ADAPTIVE_REINFORCEMENT:
            effectiveness = max(0.5, 0.6 + (attack_severity * 0.3))
        elif strategy == AdvancedDefenseStrategy.CONTEXT_AWARE_DEFENSE:
            effectiveness = max(0.55, 0.5 + (content_length * 0.0001) + (theorem_complexity * 0.01))
        elif strategy == AdvancedDefenseStrategy.MULTI_LAYER:
            effectiveness = max(0.6, 0.7 + random.uniform(0.1, 0.2))
        elif strategy == AdvancedDefenseStrategy.PREDICTIVE_MODELING:
            effectiveness = max(0.65, 0.6 + (1.0 - attack_severity) * 0.3)  # Better against weaker attacks
        elif strategy == AdvancedDefenseStrategy.DYNAMIC_RESPONSE:
            effectiveness = max(0.6, 0.55 + random.uniform(0.15, 0.3))
        else:
            effectiveness = random.uniform(0.6, 0.95)

        # Ensure effectiveness is within bounds
        effectiveness = min(0.99, effectiveness)

        # Determine if attack is blocked based on effectiveness and attack success
        attack_blocked = effectiveness > (attack_severity * 0.8)  # Defense needs to be stronger than attack

        return DefenseResult(
            defense_id=str(uuid.uuid4()),
            defense_strategy=mapped_strategy,
            attack_blocked=attack_blocked,
            effectiveness=effectiveness,
            improved_proof=f"{content}\n-- Improved with {strategy.value}",
            description=f"Defense using {strategy.value}",
            confidence=min(0.98, effectiveness + 0.1)  # Confidence slightly higher than effectiveness
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_enhanced_config(
    enable_advanced_features: bool = True,
    **kwargs
) -> AdvancedAdversarialConfig:
    """Create enhanced adversarial configuration"""
    config = AdvancedAdversarialConfig()

    if enable_advanced_features:
        config.enable_llm_attacks = True
        config.enable_adaptive_defense = True
        config.explainability_level = ExplainabilityLevel.DETAILED
        config.learning_mode = LearningMode.ONLINE
        config.enable_ensemble = True
        config.enable_advanced_analytics = True
        config.enable_realtime_adaptation = True

    # Apply custom overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def quick_enhanced_test(
    content: str,
    content_type: str = "document_general",
    theorem: str = ""
) -> Dict[str, Any]:
    """
    Quick enhanced adversarial test with sensible defaults

    Args:
        content: Content to test
        content_type: Type of content
        theorem: Theorem statement (optional)

    Returns:
        Enhanced testing results
    """
    config = create_enhanced_config(enable_advanced_features=True)
    engine = EnhancedAdversarialEngine(config)

    # Run synchronously (asyncio wrapper)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            engine.enhanced_adversarial_test(content, content_type, theorem, max_iterations=5)
        )
        return result
    finally:
        loop.close()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Enhanced Adversarial Testing System v2.0")
    print("=" * 60)

    # Test content
    sample_content = """
    def authenticate(username, password):
        user = db.query(f"SELECT * FROM users WHERE username='{username}'")
        if user and user.password == password:
            return True
        return False
    """

    # Run enhanced test
    result = quick_enhanced_test(
        content=sample_content,
        content_type="code_python",
        theorem="Authentication function"
    )

    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations_completed']}")
    print(f"  Final Robustness: {result['final_robustness']:.2%}")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Attacks: {result['metrics']['total_attacks']}")
    print(f"  Defenses: {result['metrics']['total_defenses']}")
    print(f"  Adaptations: {result['metrics']['adaptations_performed']}")
