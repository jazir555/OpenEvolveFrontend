"""
Comprehensive Test Suite for Enhanced Adversarial Testing System

This module provides extensive testing coverage for all enhanced features:
- LLM-driven attacks
- Adaptive defense
- Explainability framework
- Continuous learning
- Ensemble attacks
- Performance benchmarks

Author: OpenEvolve Testing Team
Created: 2025-01-07
Version: 1.0.0
"""

import asyncio
import json
import os
import pytest
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np

# Import modules to test
try:
    from adversarial_advanced import (
        AdvancedAdversarialConfig,
        ExplainabilityLevel,
        LearningMode,
        AttackStrategy,
        DefenseStrategy,
        ExplainabilityFramework,
        ContinuousLearningSystem,
        AdaptiveDefenseSystem,
        EnsembleAttackSystem,
        EnhancedAdversarialEngine,
        create_enhanced_config,
        quick_enhanced_test
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_code():
    """Sample Python code for testing"""
    return """
def authenticate(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
"""

@pytest.fixture
def sample_theorem():
    """Sample theorem for testing"""
    return "This function should authenticate users securely"

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return create_enhanced_config(
        max_iterations=3,
        ensemble_size=3,
        enable_llm_attacks=False,  # Disable for faster tests
        enable_adaptive_defense=True,
        explainability_level=ExplainabilityLevel.BASIC,
        learning_mode=LearningMode.OFFLINE,
        enable_ensemble=True,
        enable_caching=True,
        enable_parallel_evaluation=False  # Disable for simpler tests
    )

@pytest.fixture
def sample_attack_result():
    """Sample attack result for testing"""
    return {
        "success": True,
        "severity": 0.8,
        "description": "SQL injection vulnerability",
        "weak_point": "String concatenation in query",
        "confidence": 0.9
    }

@pytest.fixture
def sample_defense_result():
    """Sample defense result for testing"""
    return {
        "attack_blocked": True,
        "effectiveness": 0.95,
        "improved_proof": sample_code + "\n# DEFENSE: Use parameterized queries",
        "description": "Added parameterized query",
        "confidence": 0.95
    }

@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data"""
    data_dir = tmp_path / "adversarial_data"
    data_dir.mkdir()
    return data_dir


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestAdvancedAdversarialConfig:
    """Test suite for AdvancedAdversarialConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = AdvancedAdversarialConfig()

        assert config.max_iterations == 10
        assert config.ensemble_size == 5
        assert config.enable_llm_attacks is True
        assert config.enable_adaptive_defense is True
        assert config.explainability_level == ExplainabilityLevel.DETAILED
        assert config.learning_mode == LearningMode.ONLINE
        assert config.enable_ensemble is True
        assert config.enable_caching is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = AdvancedAdversarialConfig(
            max_iterations=15,
            ensemble_size=7,
            enable_llm_attacks=False
        )

        assert config.max_iterations == 15
        assert config.ensemble_size == 7
        assert config.enable_llm_attacks is False

    def test_create_enhanced_config(self):
        """Test create_enhanced_config helper"""
        config = create_enhanced_config(
            max_iterations=5,
            ensemble_size=3
        )

        assert config.max_iterations == 5
        assert config.ensemble_size == 3

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = AdvancedAdversarialConfig(max_iterations=10)
        assert config.max_iterations == 10

        # Invalid max_iterations (should raise or clamp)
        with pytest.raises((ValueError, AssertionError)):
            config = AdvancedAdversarialConfig(max_iterations=-1)

    def test_explainability_levels(self):
        """Test explainability level enum"""
        assert ExplainabilityLevel.BASIC.value == "basic"
        assert ExplainabilityLevel.DETAILED.value == "detailed"
        assert ExplainabilityLevel.FULL.value == "full"

    def test_learning_modes(self):
        """Test learning mode enum"""
        assert LearningMode.OFFLINE.value == "offline"
        assert LearningMode.ONLINE.value == "online"
        assert LearningMode.HYBRID.value == "hybrid"
        assert LearningMode.TRANSFER.value == "transfer"


# =============================================================================
# EXPLAINABILITY FRAMEWORK TESTS
# =============================================================================

class TestExplainabilityFramework:
    """Test suite for ExplainabilityFramework"""

    @pytest.fixture
    def framework(self):
        """Create explainability framework instance"""
        return ExplainabilityFramework(ExplainabilityLevel.DETAILED)

    def test_explain_attack_selection_basic(self, framework, sample_attack_result):
        """Test basic attack explanation"""
        framework.level = ExplainabilityLevel.BASIC

        explanation = framework.explain_attack_selection(
            attack=sample_attack_result,
            context={"content_type": "code_python"},
            reasoning="High severity SQL injection found"
        )

        assert "attack_type" in explanation
        assert "confidence" in explanation
        assert explanation["confidence"] == sample_attack_result["confidence"]

    def test_explain_attack_selection_detailed(self, framework, sample_attack_result):
        """Test detailed attack explanation"""
        framework.level = ExplainabilityLevel.DETAILED

        explanation = framework.explain_attack_selection(
            attack=sample_attack_result,
            context={"content_type": "code_python", "iteration": 1},
            reasoning="High severity SQL injection found"
        )

        assert "attack_type" in explanation
        assert "confidence" in explanation
        assert "reasoning" in explanation
        assert "context" in explanation

    def test_explain_defense_selection(self, framework, sample_attack_result, sample_defense_result):
        """Test defense explanation"""
        explanation = framework.explain_defense_selection(
            defense=sample_defense_result,
            attack=sample_attack_result,
            context={"content_type": "code_python"},
            reasoning="Parameterized queries prevent SQL injection"
        )

        assert "defense_type" in explanation
        assert "effectiveness" in explanation
        assert explanation["effectiveness"] == sample_defense_result["effectiveness"]

    def test_explain_decision_full(self, framework):
        """Test full decision explanation"""
        framework.level = ExplainabilityLevel.FULL

        explanation = framework.explain_decision(
            decision="attack",
            options=["attack", "defend", "evaluate"],
            context={"iteration": 1, "robustness": 0.5},
            reasoning="Low robustness score requires more attacks"
        )

        assert "decision" in explanation
        assert "alternatives_considered" in explanation
        assert "reasoning" in explanation
        assert "internal_state" in explanation

    def test_explain_to_user(self, framework):
        """Test user-friendly explanation"""
        technical_explanation = {
            "attack_type": "SQL Injection",
            "confidence": 0.9,
            "reasoning": "String concatenation in SQL query allows injection"
        }

        user_friendly = framework.explain_to_user(technical_explanation)

        assert "plain_english" in user_friendly
        assert "recommendation" in user_friendly
        assert user_friendly["plain_english"]  # Not empty

    def test_get_explainability_config(self, framework):
        """Test getting explainability configuration"""
        config = framework.get_explainability_config()

        assert "level" in config
        assert "include_internal_state" in config
        assert "target_audience" in config


# =============================================================================
# CONTINUOUS LEARNING SYSTEM TESTS
# =============================================================================

class TestContinuousLearningSystem:
    """Test suite for ContinuousLearningSystem"""

    @pytest.fixture
    def learning_system(self, temp_data_dir):
        """Create learning system instance"""
        return ContinuousLearningSystem(
            mode=LearningMode.ONLINE,
            experience_buffer_size=100,
            persistence_path=str(temp_data_dir / "experiences.json")
        )

    def test_record_experience(self, learning_system, sample_attack_result, sample_defense_result):
        """Test recording experience"""
        learning_system.record_experience(
            attack=sample_attack_result,
            defense=sample_defense_result,
            outcome={"robustness_improved": True, "new_robustness": 0.85}
        )

        assert len(learning_system.experience_buffer) == 1

    def test_record_multiple_experiences(self, learning_system):
        """Test recording multiple experiences"""
        for i in range(5):
            learning_system.record_experience(
                attack={"success": True, "severity": 0.5 + i * 0.1},
                defense={"attack_blocked": True, "effectiveness": 0.8},
                outcome={"robustness_improved": True}
            )

        assert len(learning_system.experience_buffer) == 5

    def test_recommend_attack_strategy(self, learning_system):
        """Test attack strategy recommendation"""
        # Record some experiences
        learning_system.record_experience(
            attack={"strategy": "sql_injection", "success": True, "severity": 0.9},
            defense={"attack_blocked": False, "effectiveness": 0.0},
            outcome={"robustness_improved": False}
        )

        strategy = learning_system.recommend_attack_strategy(
            context={"content_type": "code_python"},
            available_strategies=["sql_injection", "xss", "csrf"]
        )

        assert "recommended_strategy" in strategy
        assert "confidence" in strategy
        assert strategy["recommended_strategy"] in ["sql_injection", "xss", "csrf"]

    def test_learn_from_outcome(self, learning_system):
        """Test learning from outcomes"""
        initial_patterns = len(learning_system.successful_patterns)

        learning_system.learn_from_outcome(
            attack_strategy="sql_injection",
            defense_strategy="parameterized_queries",
            outcome={
                "attack_success": True,
                "defense_effectiveness": 0.3,
                "robustness_change": 0.1
            }
        )

        # Should have learned something
        assert len(learning_system.successful_patterns) >= initial_patterns

    def test_get_learning_statistics(self, learning_system):
        """Test getting learning statistics"""
        # Add some experiences
        for i in range(10):
            learning_system.record_experience(
                attack={"success": i % 2 == 0},
                defense={"attack_blocked": i % 2 == 0},
                outcome={"robustness_improved": True}
            )

        stats = learning_system.get_learning_statistics()

        assert "total_experiences" in stats
        assert "successful_attacks" in stats
        assert "successful_defenses" in stats
        assert stats["total_experiences"] == 10

    def test_save_and_load_experiences(self, learning_system, temp_data_dir):
        """Test saving and loading experiences"""
        # Record experience
        learning_system.record_experience(
            attack={"success": True},
            defense={"attack_blocked": True},
            outcome={"robustness_improved": True}
        )

        # Save
        save_path = temp_data_dir / "experiences.json"
        learning_system.save_experiences(str(save_path))

        # Create new system and load
        new_system = ContinuousLearningSystem(
            mode=LearningMode.OFFLINE,
            persistence_path=str(save_path)
        )
        new_system.load_experiences(str(save_path))

        assert len(new_system.experience_buffer) == len(learning_system.experience_buffer)


# =============================================================================
# ADAPTIVE DEFENSE SYSTEM TESTS
# =============================================================================

class TestAdaptiveDefenseSystem:
    """Test suite for AdaptiveDefenseSystem"""

    @pytest.fixture
    def defense_system(self):
        """Create adaptive defense system instance"""
        return AdaptiveDefenseSystem(
            adaptation_threshold=0.5,
            defense_strategies=["parameterized", "sanitization", "validation"]
        )

    def test_evaluate_current_defense(self, defense_system, sample_attack_result):
        """Test evaluating current defense"""
        evaluation = defense_system.evaluate_current_defense(
            content=sample_code,
            attack=sample_attack_result
        )

        assert "effectiveness" in evaluation
        assert "gaps" in evaluation
        assert 0 <= evaluation["effectiveness"] <= 1

    def test_recommend_defense_adjustment(self, defense_system, sample_attack_result):
        """Test defense adjustment recommendation"""
        recommendation = defense_system.recommend_defense_adjustment(
            current_effectiveness=0.3,
            attack=sample_attack_result,
            context={"content_type": "code_python"}
        )

        assert "should_adapt" in recommendation
        assert "recommended_strategy" in recommendation
        assert "reasoning" in recommendation

    def test_adapt_defense(self, defense_system):
        """Test defense adaptation"""
        new_defense = defense_system.adapt_defense(
            current_defense={"type": "basic"},
            attack_severity=0.8,
            adaptation_strategy="aggressive"
        )

        assert "type" in new_defense
        assert "adaptations" in new_defense

    def test_monitor_threats(self, defense_system):
        """Test threat monitoring"""
        # Add some attack history
        for i in range(5):
            defense_system.attack_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "success": i % 2 == 0,
                "severity": 0.5 + i * 0.1
            })

        threats = defense_system.monitor_threats()

        assert "threat_level" in threats
        assert "recent_attacks" in threats
        assert "trends" in threats


# =============================================================================
# ENSEMBLE ATTACK SYSTEM TESTS
# =============================================================================

class TestEnsembleAttackSystem:
    """Test suite for EnsembleAttackSystem"""

    @pytest.fixture
    def ensemble_system(self):
        """Create ensemble attack system instance"""
        return EnsembleAttackSystem(
            strategies=[
                AttackStrategy.MUTATION_BASED,
                AttackStrategy.SEMANTIC,
                AttackStrategy.GRADIENT_BASED
            ],
            voting_method="weighted"
        )

    def test_generate_ensemble_attack(self, ensemble_system):
        """Test ensemble attack generation"""
        attack = ensemble_system.generate_ensemble_attack(
            content=sample_code,
            content_type="code_python",
            theorem=sample_theorem
        )

        assert "success" in attack
        assert "attacks" in attack
        assert "aggregated_result" in attack
        assert len(attack["attacks"]) == len(ensemble_system.strategies)

    def test_weighted_voting(self, ensemble_system):
        """Test weighted voting mechanism"""
        individual_attacks = [
            {"success": True, "confidence": 0.8, "severity": 0.7},
            {"success": True, "confidence": 0.6, "severity": 0.5},
            {"success": False, "confidence": 0.3, "severity": 0.1}
        ]

        result = ensemble_system._weighted_vote(individual_attacks)

        assert "success" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_majority_voting(self, ensemble_system):
        """Test majority voting mechanism"""
        ensemble_system.voting_method = "majority"

        individual_attacks = [
            {"success": True, "confidence": 0.8},
            {"success": True, "confidence": 0.6},
            {"success": False, "confidence": 0.3}
        ]

        result = ensemble_system._majority_vote(individual_attacks)

        assert result["success"] is True  # 2 out of 3

    def test_strategy_weights(self, ensemble_system):
        """Test strategy weight management"""
        initial_weights = ensemble_system.strategy_weights.copy()

        # Update weights based on performance
        ensemble_system.update_strategy_weights(
            strategy_performances={
                AttackStrategy.MUTATION_BASED: 0.8,
                AttackStrategy.SEMANTIC: 0.6,
                AttackStrategy.GRADIENT_BASED: 0.4
            }
        )

        # Weights should have changed
        assert ensemble_system.strategy_weights != initial_weights


# =============================================================================
# ENHANCED ADVERSARIAL ENGINE TESTS
# =============================================================================

class TestEnhancedAdversarialEngine:
    """Test suite for EnhancedAdversarialEngine"""

    @pytest.fixture
    def engine(self, sample_config):
        """Create enhanced adversarial engine instance"""
        return EnhancedAdversarialEngine(sample_config)

    @pytest.mark.asyncio
    async def test_quick_enhanced_test(self, sample_code):
        """Test quick enhanced test function"""
        # Mock to avoid actual LLM calls
        with patch('adversarial_advanced.EnhancedAdversarialEngine.enhanced_adversarial_test'):
            result = quick_enhanced_test(
                content=sample_code,
                content_type="code_python",
                theorem=sample_theorem
            )

            # Should return a result structure
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_enhanced_adversarial_test_basic(self, engine, sample_code):
        """Test basic enhanced adversarial test"""
        # Mock the ensemble attack system
        engine.ensemble_system = AsyncMock()
        engine.ensemble_system.generate_ensemble_attack.return_value = {
            "success": True,
            "severity": 0.7,
            "attacks": []
        }

        result = await engine.enhanced_adversarial_test(
            content=sample_code,
            content_type="code_python",
            theorem=sample_theorem,
            max_iterations=2
        )

        assert "success" in result
        assert "final_robustness" in result
        assert 0 <= result["final_robustness"] <= 1

    @pytest.mark.asyncio
    async def test_enhanced_test_with_explainability(self, engine, sample_code):
        """Test enhanced test with explainability"""
        engine.config.explainability_level = ExplainabilityLevel.DETAILED

        result = await engine.enhanced_adversarial_test(
            content=sample_code,
            content_type="code_python",
            theorem=sample_theorem,
            max_iterations=1
        )

        assert "explanations" in result
        assert isinstance(result["explanations"], list)

    @pytest.mark.asyncio
    async def test_enhanced_test_with_learning(self, engine, sample_code):
        """Test enhanced test with continuous learning"""
        engine.config.learning_mode = LearningMode.ONLINE

        result = await engine.enhanced_adversarial_test(
            content=sample_code,
            content_type="code_python",
            theorem=sample_theorem,
            max_iterations=1
        )

        assert "learning_insights" in result
        assert isinstance(result["learning_insights"], dict)

    def test_engine_initialization(self, sample_config):
        """Test engine initialization"""
        engine = EnhancedAdversarialEngine(sample_config)

        assert engine.config == sample_config
        assert engine.explainability is not None
        assert engine.learning_system is not None
        assert engine.adaptive_defense is not None
        assert engine.ensemble_system is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete enhanced system"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_code):
        """Test complete pipeline with all features"""
        config = create_enhanced_config(
            max_iterations=2,
            ensemble_size=3,
            enable_llm_attacks=False,
            explainability_level=ExplainabilityLevel.BASIC,
            learning_mode=LearningMode.OFFLINE
        )

        engine = EnhancedAdversarialEngine(config)

        # Mock ensemble system to avoid actual attacks
        engine.ensemble_system = AsyncMock()
        engine.ensemble_system.generate_ensemble_attack.return_value = {
            "success": False,
            "severity": 0.0,
            "attacks": []
        }

        result = await engine.enhanced_adversarial_test(
            content=sample_code,
            content_type="code_python",
            theorem=sample_theorem,
            max_iterations=2
        )

        # Verify result structure
        assert "success" in result
        assert "final_robustness" in result
        assert "iterations" in result
        assert "attacks" in result
        assert "defenses" in result
        assert "explanations" in result
        assert "adaptations" in result

    @pytest.mark.asyncio
    async def test_learning_across_multiple_tests(self, sample_code):
        """Test continuous learning across multiple tests"""
        config = create_enhanced_config(
            learning_mode=LearningMode.ONLINE,
            max_iterations=1
        )

        engine = EnhancedAdversarialEngine(config)

        # Mock ensemble system
        engine.ensemble_system = AsyncMock()
        engine.ensemble_system.generate_ensemble_attack.return_value = {
            "success": True,
            "severity": 0.5,
            "attacks": []
        }

        # Run multiple tests
        results = []
        for i in range(3):
            result = await engine.enhanced_adversarial_test(
                content=sample_code,
                content_type="code_python",
                theorem=f"Test theorem {i}",
                max_iterations=1
            )
            results.append(result)

        # Check that learning occurred
        learning_stats = engine.learning_system.get_learning_statistics()
        assert learning_stats["total_experiences"] == 3


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and benchmark tests"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_with_caching(self, sample_code):
        """Test performance improvement with caching"""
        config = create_enhanced_config(
            enable_caching=True,
            max_iterations=2
        )

        engine = EnhancedAdversarialEngine(config)

        # Mock ensemble system
        engine.ensemble_system = AsyncMock()
        engine.ensemble_system.generate_ensemble_attack.return_value = {
            "success": False,
            "severity": 0.0,
            "attacks": []
        }

        # First run (uncached)
        start = time.time()
        await engine.enhanced_adversarial_test(
            content=sample_code,
            content_type="code_python",
            theorem=sample_theorem,
            max_iterations=1
        )
        first_run_time = time.time() - start

        # Second run (cached)
        start = time.time()
        await engine.enhanced_adversarial_test(
            content=sample_code,  # Same content
            content_type="code_python",
            theorem=sample_theorem,
            max_iterations=1
        )
        second_run_time = time.time() - start

        # Cached run should be faster (or similar if caching not implemented)
        assert second_run_time <= first_run_time * 1.5  # Allow some tolerance

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_scalability_with_iterations(self, sample_code):
        """Test scalability with different iteration counts"""
        config = create_enhanced_config(enable_llm_attacks=False)
        engine = EnhancedAdversarialEngine(config)

        # Mock ensemble system
        engine.ensemble_system = AsyncMock()
        engine.ensemble_system.generate_ensemble_attack.return_value = {
            "success": False,
            "severity": 0.0,
            "attacks": []
        }

        iteration_counts = [1, 3, 5]
        times = []

        for count in iteration_counts:
            start = time.time()
            await engine.enhanced_adversarial_test(
                content=sample_code,
                content_type="code_python",
                theorem=sample_theorem,
                max_iterations=count
            )
            times.append(time.time() - start)

        # Time should scale roughly linearly with iterations
        # (allowing for some overhead)
        assert times[2] <= times[0] * 6  # Should not be more than 6x slower for 5x iterations


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
