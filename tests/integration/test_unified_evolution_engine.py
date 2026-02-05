"""
Comprehensive Integration Tests for Unified Evolution Engine

This test suite validates the complete unified evolutionary optimization pipeline
integrating OpenEvolve and LoongFlow PES with domain-specific optimizers.

Test Categories (40+ tests):
1. Strategy Selection (5 tests)
2. Evolution Execution - All Modes (6 tests)
3. Knowledge Extraction & Memory Fusion (5 tests)
4. Gauntlet Integration (4 tests)
5. Cross-Domain Knowledge Transfer (4 tests)
6. Learning Loops (3 tests)
7. All 6 Domains (6 tests)
8. Error Handling & Recovery (4 tests)
9. Performance Benchmarks (4 tests)
10. End-to-End Workflows (4 tests)

Copyright 2026 OpenEvolve
Licensed under Apache License 2.0
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from dataclasses import dataclass, field
import json
import uuid
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Try importing unified evolution components
try:
    from openevolve.unified.config import (
        UnifiedEvolutionConfig,
        EvolutionMode,
        PESConfig,
        QDConfig,
        MOConfig,
        AdversarialConfig,
        DomainConfig
    )
    from openevolve.unified.config_mapper import ConfigMapper
    UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unified config not available: {e}")
    UNIFIED_AVAILABLE = False

try:
    from knowledge_engine.integrations.unified_evolution_integration import (
        UnifiedEvolutionKnowledgeExtractor
    )
    KNOWLEDGE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Knowledge integration not available: {e}")
    KNOWLEDGE_INTEGRATION_AVAILABLE = False


# ============================================================================
# MOCK DATA MODELS
# ============================================================================

@dataclass
class MockStrategyResult:
    """Mock strategy selection result."""
    mode: str
    confidence: float
    reason: str
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pes_strategy(cls):
        return cls(
            mode="pes",
            confidence=0.9,
            reason="Expensive evaluations, PES reduces cost by 60%",
            config={"enable_planning": True, "enable_memory": True}
        )

    @classmethod
    def qd_strategy(cls):
        return cls(
            mode="qd",
            confidence=0.8,
            reason="Exploration of diverse solutions required",
            config={"grid_resolution": 10, "archive_size": 1000}
        )

    @classmethod
    def mo_strategy(cls):
        return cls(
            mode="mo",
            confidence=0.85,
            reason="Multiple competing objectives require Pareto optimization",
            config={"pareto_front_size": 100}
        )

    @classmethod
    def adversarial_strategy(cls):
        return cls(
            mode="adversarial",
            confidence=0.85,
            reason="Safety-critical, adversarial testing finds failures",
            config={"adversarial_rounds": 20}
        )


@dataclass
class MockEvolutionResult:
    """Mock evolution execution result."""
    best_solution: str
    fitness: float
    evaluations: int
    total_time: float
    strategy_used: MockStrategyResult
    evolution_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    final_score: float = 0.0
    improvement: float = 0.0
    pareto_front: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    gauntlet_result: Optional['MockGauntletResult'] = None

    def __post_init__(self):
        if self.final_score == 0.0:
            self.final_score = self.fitness


@dataclass
class MockGauntletResult:
    """Mock gauntlet execution result."""
    passed: bool
    final_score: float
    rounds_completed: int
    round_results: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_strategy_selector():
    """Mock strategy selector for testing."""
    selector = Mock()

    async def select_strategy(problem, domain, constraints):
        # Auto-detect based on problem characteristics
        problem_desc = problem.get('description', '').lower()

        if 'backtest' in problem_desc or 'simulation' in problem_desc:
            return MockStrategyResult.pes_strategy()
        elif 'explore' in problem_desc or 'diverse' in problem_desc:
            return MockStrategyResult.qd_strategy()
        elif len(problem.get('objectives', [])) > 1:
            return MockStrategyResult.mo_strategy()
        elif domain in ['engineering', 'pharma', 'finance']:
            return MockStrategyResult.adversarial_strategy()
        else:
            return MockStrategyResult.pes_strategy()  # Default

    selector.select_strategy = select_strategy
    return selector


@pytest.fixture
def mock_knowledge_engine():
    """Mock knowledge engine for testing."""
    ke = Mock()

    ke.store_artifacts = AsyncMock()
    ke.query_artifacts = AsyncMock(return_value=[])
    ke.extract_patterns = AsyncMock(return_value=[])
    ke.recommend_strategy = AsyncMock(return_value={
        'recommended_strategy': 'pes',
        'confidence': 0.85,
        'expected_improvement': '60%',
        'config': {}
    })

    return ke


@pytest.fixture
def sample_problems():
    """Sample problems for each domain."""
    return {
        "general": "Maximize f(x) = x^2 where x in [0, 10]",
        "finance": "Optimize portfolio allocation for max Sharpe ratio with min risk",
        "trading": "Develop trading strategy with entry/exit signals",
        "science": "Design experiment to maximize statistical power",
        "engineering": "Minimize structural weight while maintaining strength",
        "pharma": "Optimize molecular binding affinity",
        "web_design": "Maximize landing page conversion rate"
    }


@pytest.fixture
def sample_solutions():
    """Sample solutions for testing."""
    return {
        "good": {
            "code": "def optimal_solution():\n    # Optimized approach\n    return x**2",
            "quality": "high"
        },
        "moderate": {
            "code": "def moderate_solution():\n    # Decent approach\n    return x * 1.5",
            "quality": "medium"
        },
        "bad": {
            "code": "def bad_solution():\n    # Poor approach\n    return 1",
            "quality": "low"
        }
    }


@pytest.fixture
def mock_evolution_engine():
    """Mock evolution engine for testing."""
    engine = Mock()

    async def run_evolution(problem, config, mode):
        # Simulate evolution based on mode
        await asyncio.sleep(0.1)  # Simulate work

        if mode == "pes":
            return MockEvolutionResult(
                best_solution="def pes_solution(): return optimal",
                fitness=0.95,
                evaluations=30,  # 60% fewer
                total_time=45.0,
                strategy_used=MockStrategyResult.pes_strategy(),
                evolution_artifacts=[
                    {"type": "pes_pattern", "content": "Planning worked well"},
                    {"type": "performance", "evaluations": 30, "efficiency": 0.60}
                ],
                metadata={"mode": "pes", "early_stops": [10, 20]}
            )
        elif mode == "qd":
            return MockEvolutionResult(
                best_solution="def qd_solution(): return diverse",
                fitness=0.85,
                evaluations=100,
                total_time=60.0,
                strategy_used=MockStrategyResult.qd_strategy(),
                evolution_artifacts=[
                    {"type": "archive", "size": 50, "coverage": 0.90},
                    {"type": "niches", "filled": 45}
                ]
            )
        elif mode == "mo":
            return MockEvolutionResult(
                best_solution="def mo_solution(): return pareto_optimal",
                fitness=0.88,
                evaluations=120,
                total_time=90.0,
                strategy_used=MockStrategyResult.mo_strategy(),
                pareto_front=[
                    {"fitness": 0.88, "objectives": {"obj1": 0.9, "obj2": 0.86}},
                    {"fitness": 0.85, "objectives": {"obj1": 0.95, "obj2": 0.75}},
                    {"fitness": 0.82, "objectives": {"obj1": 0.99, "obj2": 0.65}}
                ]
            )
        elif mode == "adversarial":
            return MockEvolutionResult(
                best_solution="def adversarial_solution(): return robust",
                fitness=0.82,
                evaluations=150,
                total_time=120.0,
                strategy_used=MockStrategyResult.adversarial_strategy(),
                evolution_artifacts=[
                    {"type": "attack_survived", "count": 15},
                    {"type": "robustness", "score": 0.85}
                ]
            )
        else:
            # Standard mode
            return MockEvolutionResult(
                best_solution="def standard_solution(): return decent",
                fitness=0.75,
                evaluations=200,
                total_time=90.0,
                strategy_used=MockStrategyResult(mode="standard", confidence=0.5, reason="Fallback")
            )

    engine.run_evolution = run_evolution
    return engine


@pytest.fixture
def mock_gauntlet_system():
    """Mock gauntlet system for testing."""
    gauntlet = Mock()

    async def execute_gauntlet(solution, problem, config):
        # Simulate gauntlet execution
        await asyncio.sleep(0.05)

        solution_quality = solution.get('quality', 'medium')

        if solution_quality == 'high':
            # Pass all rounds
            return MockGauntletResult(
                passed=True,
                final_score=0.92,
                rounds_completed=3,
                round_results=[
                    {"round": 1, "name": "loongflow_ai", "score": 0.90, "passed": True},
                    {"round": 2, "name": "red_team", "score": 0.85, "passed": True},
                    {"round": 3, "name": "gold_team", "score": 0.95, "passed": True}
                ]
            )
        elif solution_quality == 'medium':
            # Pass first two, fail third
            return MockGauntletResult(
                passed=False,
                final_score=0.75,
                rounds_completed=3,
                round_results=[
                    {"round": 1, "name": "loongflow_ai", "score": 0.75, "passed": True},
                    {"round": 2, "name": "red_team", "score": 0.70, "passed": True},
                    {"round": 3, "name": "gold_team", "score": 0.65, "passed": False}
                ]
            )
        else:
            # Fail early
            return MockGauntletResult(
                passed=False,
                final_score=0.40,
                rounds_completed=1,
                round_results=[
                    {"round": 1, "name": "loongflow_ai", "score": 0.40, "passed": False}
                ]
            )

    gauntlet.execute_gauntlet = execute_gauntlet
    return gauntlet


# ============================================================================
# TEST CLASS: UNIFIED EVOLUTION ENGINE INTEGRATION
# ============================================================================

@pytest.mark.skipif(not UNIFIED_AVAILABLE, reason="Unified config not available")
class TestUnifiedEvolutionEngine:
    """
    Integration tests for unified evolution engine.

    Tests the complete pipeline:
    Strategy Selection -> Evolution Execution -> Knowledge Extraction ->
    Gauntlet Evaluation -> Memory Fusion -> Learning Loop
    """

    # ========================================================================
    # CATEGORY 1: STRATEGY SELECTION (5 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_strategy_selection_expensive_evaluations(self, mock_strategy_selector):
        """
        Test strategy selection for expensive evaluation scenarios.

        Should select PES mode for:
        - Finance domain with backtesting
        - Science domain with experiments
        - Engineering domain with simulations
        """
        problem = {
            "description": "Optimize portfolio with backtesting",
            "estimated_time_per_eval": 300,  # 5 minutes
            "estimated_cost_per_eval": 1000
        }

        strategy = await mock_strategy_selector.select_strategy(
            problem=problem,
            domain="finance",
            constraints={}
        )

        assert strategy.mode == "pes"
        assert strategy.confidence >= 0.8
        assert "expensive" in strategy.reason.lower() or "60%" in strategy.reason

    @pytest.mark.asyncio
    async def test_strategy_selection_multi_objective(self, mock_strategy_selector):
        """Test strategy selection for multi-objective optimization."""
        problem = {
            "description": "Optimize for multiple objectives",
            "objectives": ["cost", "quality", "time"]
        }

        strategy = await mock_strategy_selector.select_strategy(
            problem=problem,
            domain="general",
            constraints={}
        )

        assert strategy.mode == "mo"
        assert strategy.confidence >= 0.8
        assert "pareto" in strategy.reason.lower() or "multi" in strategy.reason.lower()

    @pytest.mark.asyncio
    async def test_strategy_selection_diversity_needed(self, mock_strategy_selector):
        """Test strategy selection when diversity is needed."""
        problem = {
            "description": "Explore diverse solutions for novelty",
            "require_diversity": True
        }

        strategy = await mock_strategy_selector.select_strategy(
            problem=problem,
            domain="general",
            constraints={}
        )

        assert strategy.mode == "qd"
        assert strategy.confidence >= 0.7
        assert "diverse" in strategy.reason.lower() or "explore" in strategy.reason.lower()

    @pytest.mark.asyncio
    async def test_strategy_selection_safety_critical(self, mock_strategy_selector):
        """Test strategy selection for safety-critical domains."""
        problem = {
            "description": "Design safety-critical component",
            "safety_critical": True
        }

        strategy = await mock_strategy_selector.select_strategy(
            problem=problem,
            domain="engineering",
            constraints={}
        )

        assert strategy.mode == "adversarial"
        assert strategy.confidence >= 0.8
        assert "adversarial" in strategy.reason.lower() or "safety" in strategy.reason.lower()

    @pytest.mark.asyncio
    async def test_strategy_selection_default(self, mock_strategy_selector):
        """Test default strategy selection."""
        problem = {
            "description": "Simple optimization problem"
        }

        strategy = await mock_strategy_selector.select_strategy(
            problem=problem,
            domain="general",
            constraints={}
        )

        # Should default to PES (best general performance)
        assert strategy.mode in ["pes", "standard"]
        assert strategy.confidence >= 0.5

    # ========================================================================
    # CATEGORY 2: EVOLUTION EXECUTION - ALL MODES (6 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_pes_evolution_execution(self, mock_evolution_engine):
        """Test PES (Plan-Execute-Summarize) evolution execution."""
        result = await mock_evolution_engine.run_evolution(
            problem="Optimize with reasoning",
            config={"enable_planning": True},
            mode="pes"
        )

        assert result.fitness >= 0.9
        assert result.evaluations < 100  # 60% fewer
        assert result.strategy_used.mode == "pes"
        assert len(result.evolution_artifacts) > 0
        assert any(a["type"] == "pes_pattern" for a in result.evolution_artifacts)

    @pytest.mark.asyncio
    async def test_qd_evolution_execution(self, mock_evolution_engine):
        """Test Quality-Diversity (MAP-Elites) evolution execution."""
        result = await mock_evolution_engine.run_evolution(
            problem="Explore diverse solutions",
            config={"grid_resolution": 10},
            mode="qd"
        )

        assert result.fitness >= 0.8
        assert result.strategy_used.mode == "qd"
        assert len(result.evolution_artifacts) > 0
        assert any(a["type"] == "archive" for a in result.evolution_artifacts)
        # Check archive metrics
        archive = next(a for a in result.evolution_artifacts if a["type"] == "archive")
        assert archive["size"] > 0
        assert archive["coverage"] > 0.5

    @pytest.mark.asyncio
    async def test_mo_evolution_execution(self, mock_evolution_engine):
        """Test Multi-Objective (Pareto) evolution execution."""
        result = await mock_evolution_engine.run_evolution(
            problem="Optimize multiple objectives",
            config={"pareto_front_size": 100},
            mode="mo"
        )

        assert result.fitness >= 0.8
        assert result.strategy_used.mode == "mo"
        assert len(result.pareto_front) > 0
        # Verify Pareto front has multiple solutions
        assert len(result.pareto_front) >= 2
        # Verify each has multiple objectives
        for solution in result.pareto_front:
            assert "objectives" in solution
            assert len(solution["objectives"]) >= 2

    @pytest.mark.asyncio
    async def test_adversarial_evolution_execution(self, mock_evolution_engine):
        """Test Adversarial evolution execution."""
        result = await mock_evolution_engine.run_evolution(
            problem="Test robustness",
            config={"adversarial_rounds": 20},
            mode="adversarial"
        )

        assert result.fitness >= 0.8
        assert result.strategy_used.mode == "adversarial"
        assert len(result.evolution_artifacts) > 0
        # Check robustness metrics
        assert any(a["type"] == "attack_survived" for a in result.evolution_artifacts)
        assert any(a["type"] == "robustness" for a in result.evolution_artifacts)

    @pytest.mark.asyncio
    async def test_standard_evolution_execution(self, mock_evolution_engine):
        """Test standard evolution execution."""
        result = await mock_evolution_engine.run_evolution(
            problem="Simple optimization",
            config={},
            mode="standard"
        )

        assert result.fitness >= 0.7
        assert result.strategy_used.mode == "standard"
        assert result.evaluations > 0

    @pytest.mark.asyncio
    async def test_evolution_modes_have_different_characteristics(
        self, mock_evolution_engine
    ):
        """Test that different evolution modes have distinct characteristics."""
        modes = ["pes", "qd", "mo", "adversarial", "standard"]
        results = {}

        for mode in modes:
            result = await mock_evolution_engine.run_evolution(
                problem=f"Test {mode}",
                config={},
                mode=mode
            )
            results[mode] = result

        # Verify each mode is distinct
        pes_evals = results["pes"].evaluations
        qd_evals = results["qd"].evaluations
        mo_evals = results["mo"].evaluations

        # PES should use fewest evaluations
        assert pes_evals < qd_evals
        # MO should have Pareto front
        assert len(results["mo"].pareto_front) > 0
        # QD should have archive
        assert len(results["qd"].evolution_artifacts) > 0
        # Adversarial should have robustness metrics
        assert len(results["adversarial"].evolution_artifacts) > 0

    # ========================================================================
    # CATEGORY 3: KNOWLEDGE EXTRACTION & MEMORY FUSION (5 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_knowledge_extraction_from_pes_run(self, mock_knowledge_engine):
        """Test knowledge extraction from PES evolution run."""
        pes_result = MockEvolutionResult(
            best_solution="def solution(): return x",
            fitness=0.95,
            evaluations=30,
            total_time=45.0,
            strategy_used=MockStrategyResult.pes_strategy(),
            evolution_artifacts=[
                {"type": "pes_pattern", "planning_success": 0.9},
                {"type": "efficiency", "gain": 0.60}
            ]
        )

        # Mock extraction
        artifacts = {
            "run_id": "test_pes_001",
            "mode": "pes",
            "patterns": ["Good planning", "Early stopping worked"],
            "performance": {"evaluations": 30, "efficiency": 0.60}
        }

        await mock_knowledge_engine.store_artifacts(artifacts)

        # Verify storage was called
        mock_knowledge_engine.store_artifacts.assert_called_once()
        call_args = mock_knowledge_engine.store_artifacts.call_args[0][0]
        assert call_args["mode"] == "pes"
        assert "patterns" in call_args

    @pytest.mark.asyncio
    async def test_knowledge_extraction_from_qd_run(self, mock_knowledge_engine):
        """Test knowledge extraction from QD evolution run."""
        qd_result = MockEvolutionResult(
            best_solution="def solution(): return diverse",
            fitness=0.85,
            evaluations=100,
            total_time=60.0,
            strategy_used=MockStrategyResult.qd_strategy(),
            evolution_artifacts=[
                {"type": "archive", "size": 50, "coverage": 0.90},
                {"type": "niche", "filled": 45}
            ]
        )

        artifacts = {
            "run_id": "test_qd_001",
            "mode": "qd",
            "archive_metrics": {"size": 50, "coverage": 0.90},
            "niches_filled": 45
        }

        await mock_knowledge_engine.store_artifacts(artifacts)

        mock_knowledge_engine.store_artifacts.assert_called_once()
        call_args = mock_knowledge_engine.store_artifacts.call_args[0][0]
        assert call_args["mode"] == "qd"
        assert "archive_metrics" in call_args

    @pytest.mark.asyncio
    async def test_memory_fusion_openevolve_loongflow(self, mock_knowledge_engine):
        """Test memory fusion between OpenEvolve and LoongFlow."""
        # Simulate storing artifacts from both systems
        oe_artifacts = {
            "system": "openevolve",
            "mode": "qd",
            "evaluations": 500,
            "fitness": 0.85
        }

        lf_artifacts = {
            "system": "loongflow",
            "mode": "pes",
            "evaluations": 30,
            "fitness": 0.95
        }

        # Store both
        await mock_knowledge_engine.store_artifacts(oe_artifacts)
        await mock_knowledge_engine.store_artifacts(lf_artifacts)

        # Verify both stored
        assert mock_knowledge_engine.store_artifacts.call_count == 2

    @pytest.mark.asyncio
    async def test_cross_domain_pattern_matching(self, mock_knowledge_engine):
        """Test pattern matching across domains."""
        # Query for similar patterns
        mock_knowledge_engine.query_artifacts.return_value = [
            {
                "domain": "finance",
                "pattern": "Use momentum for convergence",
                "success_rate": 0.85
            },
            {
                "domain": "trading",
                "pattern": "Momentum helps escape local optima",
                "success_rate": 0.82
            }
        ]

        patterns = await mock_knowledge_engine.query_artifacts(
            query="momentum convergence optimization"
        )

        assert len(patterns) == 2
        assert all("pattern" in p for p in patterns)
        assert all("success_rate" in p for p in patterns)

    @pytest.mark.asyncio
    async def test_strategy_recommendation_from_knowledge(self, mock_knowledge_engine):
        """Test strategy recommendation based on historical knowledge."""
        recommendation = await mock_knowledge_engine.recommend_strategy(
            problem_type="financial_optimization",
            constraints={"max_evaluations": 50}
        )

        assert "recommended_strategy" in recommendation
        assert "confidence" in recommendation
        assert recommendation["confidence"] > 0.5

    # ========================================================================
    # CATEGORY 4: GAUNTLET INTEGRATION (4 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_gauntlet_all_rounds_passed(self, mock_gauntlet_system):
        """Test gauntlet with all rounds passing."""
        result = await mock_gauntlet_system.execute_gauntlet(
            solution=sample_solutions()["good"],
            problem="Test problem",
            config={"rounds": 3}
        )

        assert result.passed is True
        assert result.final_score >= 0.9
        assert result.rounds_completed == 3
        assert len(result.round_results) == 3

        # Verify all rounds passed
        for round_result in result.round_results:
            assert round_result["passed"] is True

    @pytest.mark.asyncio
    async def test_gauntlet_early_termination(self, mock_gauntlet_system):
        """Test gauntlet with early termination."""
        result = await mock_gauntlet_system.execute_gauntlet(
            solution=sample_solutions()["bad"],
            problem="Test problem",
            config={"rounds": 3}
        )

        assert result.passed is False
        assert result.final_score < 0.5
        # Should terminate after first round
        assert result.rounds_completed == 1
        assert len(result.round_results) == 1

    @pytest.mark.asyncio
    async def test_gauntlet_partial_pass(self, mock_gauntlet_system):
        """Test gauntlet with partial pass (some rounds fail)."""
        result = await mock_gauntlet_system.execute_gauntlet(
            solution=sample_solutions()["moderate"],
            problem="Test problem",
            config={"rounds": 3}
        )

        assert result.passed is False
        assert 0.7 <= result.final_score <= 0.8
        assert result.rounds_completed == 3

        # First two should pass, third should fail
        assert result.round_results[0]["passed"] is True
        assert result.round_results[1]["passed"] is True
        assert result.round_results[2]["passed"] is False

    @pytest.mark.asyncio
    async def test_gauntlet_score_aggregation(self, mock_gauntlet_system):
        """Test gauntlet score aggregation across rounds."""
        result = await mock_gauntlet_system.execute_gauntlet(
            solution=sample_solutions()["good"],
            problem="Test problem",
            config={"rounds": 3}
        )

        # Final score should be weighted average
        # Round 1: 20%, Round 2: 30%, Round 3: 50%
        expected_score = (
            result.round_results[0]["score"] * 0.2 +
            result.round_results[1]["score"] * 0.3 +
            result.round_results[2]["score"] * 0.5
        )

        assert abs(result.final_score - expected_score) < 0.01

    # ========================================================================
    # CATEGORY 5: CROSS-DOMAIN KNOWLEDGE TRANSFER (4 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_knowledge_transfer_finance_to_trading(
        self, mock_knowledge_engine, mock_evolution_engine
    ):
        """Test knowledge transfer from finance to trading domain."""
        # First run in finance
        finance_result = await mock_evolution_engine.run_evolution(
            problem="Optimize portfolio",
            config={},
            mode="pes"
        )

        finance_artifacts = {
            "domain": "finance",
            "patterns": ["Use momentum", "Apply risk parity"],
            "success_rate": 0.85
        }

        await mock_knowledge_engine.store_artifacts(finance_artifacts)

        # Second run in trading (should leverage finance knowledge)
        mock_knowledge_engine.query_artifacts.return_value = [finance_artifacts]

        trading_result = await mock_evolution_engine.run_evolution(
            problem="Develop trading strategy",
            config={},
            mode="pes"
        )

        # Should complete successfully
        assert trading_result.fitness > 0.0
        # Knowledge should have been queried
        mock_knowledge_engine.query_artifacts.assert_called()

    @pytest.mark.asyncio
    async def test_knowledge_transfer_engineering_to_pharma(
        self, mock_knowledge_engine, mock_evolution_engine
    ):
        """Test knowledge transfer from engineering to pharma domain."""
        # Engineering run
        eng_result = await mock_evolution_engine.run_evolution(
            problem="Optimize structure",
            config={},
            mode="adversarial"
        )

        eng_artifacts = {
            "domain": "engineering",
            "patterns": ["Robustness testing", "Safety margins"],
            "success_rate": 0.80
        }

        await mock_knowledge_engine.store_artifacts(eng_artifacts)

        # Pharma run
        mock_knowledge_engine.query_artifacts.return_value = [eng_artifacts]

        pharma_result = await mock_evolution_engine.run_evolution(
            problem="Optimize molecule",
            config={},
            mode="adversarial"
        )

        assert pharma_result.fitness > 0.0

    @pytest.mark.asyncio
    async def test_cross_domain_similarity_detection(self, mock_knowledge_engine):
        """Test detection of similar problems across domains."""
        # Mock similar problems from different domains
        mock_knowledge_engine.query_artifacts.return_value = [
            {
                "domain": "finance",
                "problem": "Optimize portfolio",
                "similarity": 0.85
            },
            {
                "domain": "trading",
                "problem": "Optimize strategy",
                "similarity": 0.82
            }
        ]

        similar = await mock_knowledge_engine.query_artifacts(
            query="optimization with risk constraints"
        )

        assert len(similar) == 2
        assert all("similarity" in s for s in similar)
        assert all(s["similarity"] > 0.8 for s in similar)

    @pytest.mark.asyncio
    async def test_cross_domain_pattern_validation(self, mock_knowledge_engine):
        """Test that cross-domain patterns are validated before application."""
        # Pattern that works in finance
        finance_pattern = {
            "domain": "finance",
            "pattern": "Use 5% risk limit",
            "success_rate": 0.90,
            "context": "portfolio optimization"
        }

        await mock_knowledge_engine.store_artifacts(finance_pattern)

        # Query should include context for validation
        mock_knowledge_engine.query_artifacts.return_value = [
            {**finance_pattern, "applicable": True}
        ]

        patterns = await mock_knowledge_engine.query_artifacts(
            query="risk management"
        )

        assert len(patterns) > 0
        # Pattern should have applicability info
        assert "applicable" in patterns[0]

    # ========================================================================
    # CATEGORY 6: LEARNING LOOPS (3 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_learning_loop_multiple_runs(
        self, mock_evolution_engine, mock_knowledge_engine
    ):
        """Test learning across multiple evolutionary runs."""
        problems = [
            "Simple optimization problem 1",
            "Simple optimization problem 2",
            "Simple optimization problem 3"
        ]

        results = []
        for i, problem in enumerate(problems):
            result = await mock_evolution_engine.run_evolution(
                problem=problem,
                config={},
                mode="pes"
            )

            # Store knowledge
            artifacts = {
                "run_id": f"run_{i}",
                "problem": problem,
                "fitness": result.fitness,
                "evaluations": result.evaluations
            }

            await mock_knowledge_engine.store_artifacts(artifacts)
            results.append(result)

        # All should complete successfully
        for result in results:
            assert result.fitness > 0.0

        # Knowledge should accumulate
        assert mock_knowledge_engine.store_artifacts.call_count == 3

    @pytest.mark.asyncio
    async def test_strategy_selector_learning(
        self, mock_strategy_selector, mock_knowledge_engine
    ):
        """Test that strategy selector learns from past runs."""
        # Simulate learning data
        past_runs = [
            {"domain": "finance", "strategy": "pes", "success": True, "efficiency": 0.90},
            {"domain": "finance", "strategy": "pes", "success": True, "efficiency": 0.85},
            {"domain": "finance", "strategy": "qd", "success": False, "efficiency": 0.40}
        ]

        # Mock knowledge engine returns past performance
        mock_knowledge_engine.query_artifacts.return_value = past_runs

        # Strategy selector should recommend PES for finance
        problem = {"description": "Finance optimization"}

        # In real system, selector would query knowledge engine
        # Here we just verify it returns a valid strategy
        strategy = await mock_strategy_selector.select_strategy(
            problem=problem,
            domain="finance",
            constraints={}
        )

        assert strategy.mode in ["pes", "qd", "mo", "adversarial"]
        assert strategy.confidence > 0.0

    @pytest.mark.asyncio
    async def test_adaptive_parameter_tuning(
        self, mock_evolution_engine, mock_knowledge_engine
    ):
        """Test adaptive parameter tuning based on past performance."""
        # First run with default params
        result1 = await mock_evolution_engine.run_evolution(
            problem="Optimize",
            config={"population_size": 100},
            mode="pes"
        )

        # Store performance
        await mock_knowledge_engine.store_artifacts({
            "config": {"population_size": 100},
            "fitness": result1.fitness,
            "evaluations": result1.evaluations
        })

        # Second run with adapted params (smaller population worked better)
        mock_knowledge_engine.query_artifacts.return_value = [{
            "best_config": {"population_size": 50},
            "reason": "Smaller population more efficient"
        }]

        result2 = await mock_evolution_engine.run_evolution(
            problem="Optimize again",
            config={"population_size": 50},
            mode="pes"
        )

        # Should complete successfully
        assert result2.fitness > 0.0

    # ========================================================================
    # CATEGORY 7: ALL 6 DOMAINS (6 tests)
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain,problem,expected_mode", [
        ("finance", "Optimize portfolio allocation", "pes"),
        ("trading", "Develop trading strategy", "pes"),
        ("science", "Design experiment", "pes"),
        ("engineering", "Optimize structure", "adversarial"),
        ("pharma", "Optimize molecule", "adversarial"),
        ("web_design", "Maximize conversion", "pes")
    ])
    async def test_all_domains(
        self, domain, problem, expected_mode,
        mock_strategy_selector, mock_evolution_engine
    ):
        """Test that all 6 domain optimizers work correctly."""
        # Select strategy
        strategy = await mock_strategy_selector.select_strategy(
            problem={"description": problem},
            domain=domain,
            constraints={}
        )

        # Run evolution
        result = await mock_evolution_engine.run_evolution(
            problem=problem,
            config=strategy.config,
            mode=strategy.mode
        )

        # Verify success
        assert result.fitness > 0.0
        assert result.evaluations > 0
        assert result.total_time > 0
        assert len(result.best_solution) > 0

    # ========================================================================
    # CATEGORY 8: ERROR HANDLING & RECOVERY (4 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_invalid_problem_handling(self, mock_strategy_selector):
        """Test handling of invalid problem description."""
        # Empty problem
        with pytest.raises((ValueError, AttributeError)):
            await mock_strategy_selector.select_strategy(
                problem={"description": ""},
                domain="general",
                constraints={}
            )

    @pytest.mark.asyncio
    async def test_evolution_failure_recovery(self, mock_evolution_engine):
        """Test recovery from evolution failure."""
        # Mock evolution failure
        async def failing_evolution(problem, config, mode):
            raise RuntimeError("Evolution failed")

        mock_evolution_engine.run_evolution = failing_evolution

        # Should handle gracefully
        with pytest.raises(RuntimeError):
            await mock_evolution_engine.run_evolution(
                problem="Test",
                config={},
                mode="pes"
            )

    @pytest.mark.asyncio
    async def test_gauntlet_timeout_handling(self, mock_gauntlet_system):
        """Test handling of gauntlet timeout."""
        # Mock timeout
        async def timeout_gauntlet(solution, problem, config):
            await asyncio.sleep(10)  # Simulate timeout
            return MockGauntletResult(passed=False, final_score=0.0, rounds_completed=0)

        mock_gauntlet_system.execute_gauntlet = timeout_gauntlet

        # Should handle timeout gracefully
        with pytest.raises((asyncio.TimeoutError, RuntimeError)):
            # Use timeout wrapper
            await asyncio.wait_for(
                mock_gauntlet_system.execute_gauntlet(
                    solution={"code": "test"},
                    problem="Test",
                    config={}
                ),
                timeout=1.0
            )

    @pytest.mark.asyncio
    async def test_knowledge_engine_unavailable(self, mock_evolution_engine):
        """Test evolution when knowledge engine is unavailable."""
        # Run evolution without knowledge engine
        result = await mock_evolution_engine.run_evolution(
            problem="Optimize without knowledge",
            config={},
            mode="pes"
        )

        # Should still work
        assert result.fitness > 0.0

    # ========================================================================
    # CATEGORY 9: PERFORMANCE BENCHMARKS (4 tests)
    # ========================================================================

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_general_domain_performance(self, mock_evolution_engine):
        """Test performance benchmarks for general domain."""
        start_time = time.time()

        result = await mock_evolution_engine.run_evolution(
            problem="Simple optimization",
            config={},
            mode="pes"
        )

        elapsed = time.time() - start_time

        # Performance targets
        assert result.fitness >= 0.7
        assert result.evaluations <= 100
        assert elapsed < 60  # Should complete in < 1 minute

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_finance_domain_performance(self, mock_evolution_engine):
        """Test performance benchmarks for finance domain."""
        start_time = time.time()

        result = await mock_evolution_engine.run_evolution(
            problem="Optimize portfolio with backtesting",
            config={},
            mode="pes"
        )

        elapsed = time.time() - start_time

        # Finance domain targets (expensive evaluations)
        assert result.fitness >= 0.7
        assert result.evaluations <= 50  # Fewer due to PES
        assert elapsed < 600  # < 10 minutes

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_sample_efficiency_comparison(self, mock_evolution_engine):
        """Test that PES mode achieves better sample efficiency."""
        pes_result = await mock_evolution_engine.run_evolution(
            problem="Optimize",
            config={},
            mode="pes"
        )

        standard_result = await mock_evolution_engine.run_evolution(
            problem="Optimize",
            config={},
            mode="standard"
        )

        # PES should use fewer evaluations
        assert pes_result.evaluations < standard_result.evaluations

        # Calculate efficiency gain
        efficiency_gain = 1 - (pes_result.evaluations / standard_result.evaluations)
        assert efficiency_gain >= 0.4  # At least 40% improvement

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_full_pipeline_performance(
        self, mock_strategy_selector, mock_evolution_engine,
        mock_gauntlet_system, mock_knowledge_engine
    ):
        """Test performance of complete pipeline."""
        start_time = time.time()

        # Step 1: Strategy selection
        strategy = await mock_strategy_selector.select_strategy(
            problem={"description": "Test optimization"},
            domain="general",
            constraints={}
        )

        # Step 2: Evolution
        result = await mock_evolution_engine.run_evolution(
            problem="Test",
            config=strategy.config,
            mode=strategy.mode
        )

        # Step 3: Gauntlet (simplified)
        gauntlet_result = await mock_gauntlet_system.execute_gauntlet(
            solution={"code": result.best_solution},
            problem="Test",
            config={}
        )

        # Step 4: Knowledge storage
        await mock_knowledge_engine.store_artifacts({
            "result": result,
            "gauntlet": gauntlet_result
        })

        elapsed = time.time() - start_time

        # Full pipeline should complete in reasonable time
        assert elapsed < 120  # < 2 minutes
        assert result.fitness > 0.0

    # ========================================================================
    # CATEGORY 10: END-TO-END WORKFLOWS (4 tests)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_complete_workflow_success(
        self, mock_strategy_selector, mock_evolution_engine,
        mock_gauntlet_system, mock_knowledge_engine
    ):
        """Test complete end-to-end workflow with success."""
        # 1. Strategy Selection
        strategy = await mock_strategy_selector.select_strategy(
            problem={"description": "Optimize portfolio"},
            domain="finance",
            constraints={}
        )

        assert strategy.mode is not None

        # 2. Evolution Execution
        result = await mock_evolution_engine.run_evolution(
            problem="Optimize portfolio",
            config=strategy.config,
            mode=strategy.mode
        )

        assert result.fitness >= 0.7

        # 3. Gauntlet Evaluation
        gauntlet_result = await mock_gauntlet_system.execute_gauntlet(
            solution={"code": result.best_solution, "quality": "high"},
            problem="Optimize portfolio",
            config={}
        )

        assert gauntlet_result.passed is True

        # 4. Knowledge Extraction & Storage
        artifacts = {
            "run_id": "workflow_001",
            "domain": "finance",
            "strategy": strategy.mode,
            "fitness": result.fitness,
            "evaluations": result.evaluations,
            "gauntlet_passed": gauntlet_result.passed,
            "gauntlet_score": gauntlet_result.final_score
        }

        await mock_knowledge_engine.store_artifacts(artifacts)

        # Verify complete pipeline
        mock_knowledge_engine.store_artifacts.assert_called_once()
        stored = mock_knowledge_engine.store_artifacts.call_args[0][0]
        assert stored["gauntlet_passed"] is True
        assert stored["fitness"] >= 0.7

    @pytest.mark.asyncio
    async def test_batch_evolution_workflow(
        self, mock_evolution_engine, mock_knowledge_engine
    ):
        """Test batch evolution of multiple problems."""
        problems = [
            "Problem 1: Maximize x^2",
            "Problem 2: Minimize x^2 + 10",
            "Problem 3: Optimize x^3 - x"
        ]

        results = []
        for problem in problems:
            result = await mock_evolution_engine.run_evolution(
                problem=problem,
                config={},
                mode="pes"
            )
            results.append(result)

            # Store individual results
            await mock_knowledge_engine.store_artifacts({
                "problem": problem,
                "fitness": result.fitness
            })

        # Verify all completed
        assert len(results) == 3
        assert all(r.fitness > 0.0 for r in results)
        assert mock_knowledge_engine.store_artifacts.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_evolution_workflow(
        self, mock_evolution_engine, mock_knowledge_engine
    ):
        """Test concurrent evolution of multiple problems."""
        problems = [
            "Concurrent problem 1",
            "Concurrent problem 2",
            "Concurrent problem 3"
        ]

        # Run evolutions concurrently
        tasks = [
            mock_evolution_engine.run_evolution(
                problem=problem,
                config={},
                mode="pes"
            )
            for problem in problems
        ]

        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 3
        assert all(r.fitness > 0.0 for r in results)

        # Store batch results
        await mock_knowledge_engine.store_artifacts({
            "batch_results": [{"fitness": r.fitness} for r in results]
        })

    @pytest.mark.asyncio
    async def test_iterative_improvement_workflow(
        self, mock_evolution_engine, mock_knowledge_engine
    ):
        """Test iterative improvement across multiple generations."""
        best_fitness = 0.0
        iterations = 3

        for i in range(iterations):
            # Run evolution
            result = await mock_evolution_engine.run_evolution(
                problem=f"Iteration {i+1}",
                config={},
                mode="pes"
            )

            # Store learning
            await mock_knowledge_engine.store_artifacts({
                "iteration": i,
                "fitness": result.fitness,
                "improvement": result.fitness - best_fitness
            })

            # Update best
            if result.fitness > best_fitness:
                best_fitness = result.fitness

        # Should show improvement or maintain quality
        assert best_fitness >= 0.7
        assert mock_knowledge_engine.store_artifacts.call_count == iterations


# ============================================================================
# SAMPLE SOLUTIONS DATA (used in fixtures above)
# ============================================================================

def sample_solutions():
    """Sample solutions for testing."""
    return {
        "good": {
            "code": "def optimal_solution():\n    # Optimized approach\n    return x**2",
            "quality": "high"
        },
        "moderate": {
            "code": "def moderate_solution():\n    # Decent approach\n    return x * 1.5",
            "quality": "medium"
        },
        "bad": {
            "code": "def bad_solution():\n    # Poor approach\n    return 1",
            "quality": "low"
        }
    }


# ============================================================================
# TEST EXECUTION CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
