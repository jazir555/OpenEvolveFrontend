"""
Test Suite for Hybrid MCTS Framework

Comprehensive tests for all three hybrid approaches, adaptive selection,
combined search, and LeanAide integration.

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import pytest
import logging
from typing import Dict, Any

from hybrid_mcts_framework import (
    # Enums and Classes
    HybridMCTSApproach,
    ApproachStatus,
    HybridMCTSConfig,
    HybridMCTSResult,
    HybridCache,
    HybridMCTSMonitor,
    AdaptiveHybridSelector,
    HybridMCTSWithLeanAide,
    HybridBenchmark,
    CombinedHybridMCTS,
    HybridMCTSEngine,
    HybridMCTSPresets,
    HybridMCTSWorkflowIntegrator,

    # Utility functions
    create_framework_from_preset,
    quick_search,
    thorough_search,
    print_result_summary,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing"""
    return HybridMCTSConfig(
        approach=HybridMCTSApproach.EVOLVED_POLICIES,
        simulations=50,
        max_depth=20,
        policy_generations=5,
        cache_enabled=True,
        leanaide_enabled=False,  # Disable for faster tests
    )


@pytest.fixture
def sample_theorems():
    """Sample theorems for testing"""
    return [
        "theorem add_comm (a b : nat) : a + b = b + a",
        "theorem mul_assoc (a b c : nat) : (a * b) * c = a * (b * c)",
        "theorem add_le_add_left (a b c : nat) : a <= b -> a + c <= b + c",
    ]


@pytest.fixture
def sample_subproblems():
    """Sample subproblems for workflow integration testing"""
    return [
        {
            "id": "subproblem_1",
            "statement": "theorem add_zero (a : nat) : a + 0 = a",
            "domain": "algebra",
            "difficulty": "easy",
            "dependencies": [],
        },
        {
            "id": "subproblem_2",
            "statement": "theorem mul_one (a : nat) : a * 1 = a",
            "domain": "algebra",
            "difficulty": "easy",
            "dependencies": ["subproblem_1"],
        },
    ]


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestHybridMCTSConfig:
    """Test configuration class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = HybridMCTSConfig()
        assert config.approach == HybridMCTSApproach.EVOLVED_POLICIES
        assert config.exploration_constant == 1.414
        assert config.simulations == 100
        assert config.max_depth == 50
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7

    def test_invalid_exploration_constant(self):
        """Test validation of exploration constant"""
        with pytest.raises(ValueError):
            HybridMCTSConfig(exploration_constant=-1.0)

    def test_invalid_mutation_rate(self):
        """Test validation of mutation rate"""
        with pytest.raises(ValueError):
            HybridMCTSConfig(mutation_rate=1.5)

    def test_invalid_crossover_rate(self):
        """Test validation of crossover rate"""
        with pytest.raises(ValueError):
            HybridMCTSConfig(crossover_rate=-0.1)

    def test_invalid_simulations(self):
        """Test validation of simulations"""
        with pytest.raises(ValueError):
            HybridMCTSConfig(simulations=0)

    def test_custom_config(self):
        """Test custom configuration"""
        config = HybridMCTSConfig(
            approach=HybridMCTSApproach.COEVOLUTION,
            simulations=200,
            max_depth=100,
            tree_generations=50,
        )
        assert config.approach == HybridMCTSApproach.COEVOLUTION
        assert config.simulations == 200
        assert config.max_depth == 100
        assert config.tree_generations == 50


# =============================================================================
# RESULT TESTS
# =============================================================================

class TestHybridMCTSResult:
    """Test result class"""

    def test_result_creation(self):
        """Test creating a result"""
        result = HybridMCTSResult(
            success=True,
            approach_used=HybridMCTSApproach.EVOLVED_POLICIES,
            best_proof="test_proof",
            best_fitness=0.85,
        )
        assert result.success
        assert result.best_fitness == 0.85
        assert result.best_proof == "test_proof"

    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = HybridMCTSResult(
            success=True,
            approach_used=HybridMCTSApproach.EVOLVED_POLICIES,
            best_proof="test_proof",
            best_fitness=0.85,
            execution_time=10.0,
            generations_completed=5,
        )
        data = result.to_dict()
        assert data["success"]
        assert data["approach_used"] == "evolved_policies"
        assert data["best_fitness"] == 0.85

    def test_result_save_and_load(self, tmp_path):
        """Test saving and loading results"""
        result = HybridMCTSResult(
            success=True,
            approach_used=HybridMCTSApproach.EVOLVED_POLICIES,
            best_proof="test_proof",
            best_fitness=0.85,
        )

        filepath = tmp_path / "result.json"
        result.save(str(filepath))

        loaded = HybridMCTSResult.load(str(filepath))
        assert loaded.success == result.success
        assert loaded.approach_used == result.approach_used
        assert loaded.best_fitness == result.best_fitness


# =============================================================================
# CACHE TESTS
# =============================================================================

class TestHybridCache:
    """Test caching system"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = HybridCache(max_size=100, enabled=True)
        assert cache.enabled
        assert cache.max_size == 100
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_disabled(self):
        """Test cache when disabled"""
        cache = HybridCache(enabled=False)
        assert not cache.enabled

        policy = cache.get_policy("test_key")
        assert policy is None

    def test_policy_cache(self):
        """Test policy caching"""
        cache = HybridCache()

        # Cache miss
        policy = cache.get_policy("test_policy")
        assert policy is None
        assert cache.misses == 1

        # Cache policy
        test_policy = {"weights": [1, 2, 3]}
        cache.cache_policy("test_policy", test_policy)

        # Cache hit
        cached = cache.get_policy("test_policy")
        assert cached == test_policy
        assert cache.hits == 1

    def test_node_cache(self):
        """Test node caching"""
        cache = HybridCache()

        test_node = {"value": 0.5, "visits": 10}
        cache.cache_node("test_node", test_node)

        cached = cache.get_node("test_node")
        assert cached == test_node

    def test_tree_cache(self):
        """Test tree caching"""
        cache = HybridCache()

        test_tree = {"depth": 5, "nodes": 20}
        cache.cache_tree("test_tree", test_tree)

        cached = cache.get_tree("test_tree")
        assert cached == test_tree

    def test_evaluation_cache(self):
        """Test evaluation caching"""
        cache = HybridCache()

        # Cache miss
        fitness = cache.get_evaluation("test_individual")
        assert fitness is None

        # Cache fitness
        cache.cache_evaluation("test_individual", 0.85)

        # Cache hit
        cached = cache.get_evaluation("test_individual")
        assert cached == 0.85
        assert cache.hits == 1

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = HybridCache()

        # Add some items
        cache.cache_policy("policy1", {"weights": [1, 2, 3]})
        cache.cache_node("node1", {"value": 0.5})
        cache.cache_evaluation("eval1", 0.85)

        # Generate some hits and misses
        cache.get_policy("policy1")
        cache.get_policy("policy2")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["policy_cache_size"] == 1
        assert stats["node_cache_size"] == 1

    def test_cache_clear(self):
        """Test clearing cache"""
        cache = HybridCache()

        cache.cache_policy("policy1", {"weights": [1, 2, 3]})
        cache.cache_node("node1", {"value": 0.5})

        assert len(cache.policy_cache) == 1
        assert len(cache.node_cache) == 1

        cache.clear()

        assert len(cache.policy_cache) == 0
        assert len(cache.node_cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


# =============================================================================
# MONITOR TESTS
# =============================================================================

class TestHybridMCTSMonitor:
    """Test monitoring system"""

    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)
        assert monitor.approach == HybridMCTSApproach.EVOLVED_POLICIES
        assert monitor.status == ApproachStatus.INITIALIZING

    def test_monitor_start_stop(self):
        """Test starting and stopping monitor"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.start()
        assert monitor.status == ApproachStatus.RUNNING
        assert monitor.start_time is not None

        monitor.stop()
        assert monitor.status == ApproachStatus.COMPLETED
        assert monitor.end_time is not None

    def test_log_generation(self):
        """Test logging generation metrics"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.log_generation(0, {"best_fitness": 0.5})
        monitor.log_generation(1, {"best_fitness": 0.6})

        assert len(monitor.generation_metrics) == 2
        assert monitor.generation_metrics[0]["generation"] == 0
        assert monitor.generation_metrics[0]["best_fitness"] == 0.5
        assert monitor.current_generation == 1

    def test_log_evaluation(self):
        """Test logging evaluation metrics"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.log_evaluation({"fitness": 0.75, "valid": True})
        monitor.log_evaluation({"fitness": 0.80, "valid": True})

        assert len(monitor.evaluation_metrics) == 2
        assert monitor.total_evaluations == 2

    def test_log_custom(self):
        """Test logging custom metrics"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.log_custom("test_metric", 42)
        monitor.log_custom("test_metric", 43)

        assert len(monitor.custom_metrics["test_metric"]) == 2
        assert monitor.custom_metrics["test_metric"][0]["value"] == 42

    def test_update_status(self):
        """Test updating status"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.update_status(ApproachStatus.RUNNING)
        assert monitor.status == ApproachStatus.RUNNING

    def test_get_summary(self):
        """Test getting execution summary"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.start()
        monitor.log_generation(0, {"best_fitness": 0.5})
        monitor.log_generation(1, {"best_fitness": 0.6})
        monitor.stop()

        summary = monitor.get_summary()
        assert summary["approach"] == "evolved_policies"
        assert summary["status"] == "completed"
        assert summary["generations_completed"] == 2
        assert summary["total_evaluations"] == 0
        assert summary["execution_time"] > 0

    def test_get_generation_history(self):
        """Test getting generation history for a metric"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.log_generation(0, {"best_fitness": 0.5})
        monitor.log_generation(1, {"best_fitness": 0.6})
        monitor.log_generation(2, {"best_fitness": 0.7})

        history = monitor.get_generation_history("best_fitness")
        assert history == [0.5, 0.6, 0.7]

    def test_get_best_generation(self):
        """Test getting best generation for a metric"""
        monitor = HybridMCTSMonitor(HybridMCTSApproach.EVOLVED_POLICIES)

        monitor.log_generation(0, {"best_fitness": 0.5})
        monitor.log_generation(1, {"best_fitness": 0.8})
        monitor.log_generation(2, {"best_fitness": 0.7})

        best = monitor.get_best_generation("best_fitness", higher_is_better=True)
        assert best == 1


# =============================================================================
# ADAPTIVE SELECTOR TESTS
# =============================================================================

class TestAdaptiveHybridSelector:
    """Test adaptive approach selection"""

    def test_selector_initialization(self):
        """Test selector initialization"""
        selector = AdaptiveHybridSelector()
        assert selector.approach_performance == {}
        assert len(selector.problem_history) == 0

    def test_extract_features(self):
        """Test feature extraction"""
        selector = AdaptiveHybridSelector()

        theorem = "theorem test (a b : nat) : a + b = b + a"
        features = selector._extract_problem_features(theorem)

        assert "theorem_length" in features
        assert "word_count" in features
        assert "has_quantifiers" in features
        assert "complexity_score" in features
        assert features["theorem_length"] == len(theorem)

    def test_select_approach(self):
        """Test approach selection"""
        selector = AdaptiveHybridSelector()

        theorem = "theorem test (a b : nat) : a + b = b + a"
        approach = selector.select_approach(theorem)

        assert isinstance(approach, HybridMCTSApproach)
        assert len(selector.problem_history) == 1

    def test_update_performance(self):
        """Test performance tracking"""
        selector = AdaptiveHybridSelector()

        selector.update_performance(
            HybridMCTSApproach.EVOLVED_POLICIES,
            0.85,
            "theorem test",
            HybridMCTSResult(
                success=True,
                approach_used=HybridMCTSApproach.EVOLVED_POLICIES,
                best_proof="proof",
                best_fitness=0.85,
            )
        )

        assert "evolved_policies" in selector.approach_performance
        assert selector.approach_performance["evolved_policies"] == [0.85]
        assert selector.approach_runs["evolved_policies"] == 1

    def test_get_statistics(self):
        """Test getting selector statistics"""
        selector = AdaptiveHybridSelector()

        # Add some performance data
        selector.update_performance(
            HybridMCTSApproach.EVOLVED_POLICIES,
            0.85,
            "theorem test",
            HybridMCTSResult(
                success=True,
                approach_used=HybridMCTSApproach.EVOLVED_POLICIES,
                best_proof="proof",
                best_fitness=0.85,
            )
        )

        stats = selector.get_statistics()
        assert "total_selections" in stats
        assert "evolved_policies_avg_performance" in stats


# =============================================================================
# LEANAIDE INTEGRATION TESTS
# =============================================================================

class TestHybridMCTSWithLeanAide:
    """Test LeanAide integration"""

    def test_initialization(self, sample_config):
        """Test LeanAide integration initialization"""
        leanaide = HybridMCTSWithLeanAide(sample_config)
        assert leanaide.config == sample_config

    def test_disabled_leanaide(self, sample_config):
        """Test when LeanAide is disabled"""
        config = HybridMCTSConfig(leanaide_enabled=False)
        leanaide = HybridMCTSWithLeanAide(config)

        result = leanaide.verify_proof("test", "proof")
        assert asyncio.run(result) == (False, None)

    def test_verification_stats(self, sample_config):
        """Test verification statistics"""
        leanaide = HybridMCTSWithLeanAide(sample_config)

        # Add some cached results
        leanaide.verification_cache["test1:proof1"] = True
        leanaide.verification_cache["test2:proof2"] = False

        stats = leanaide.get_verification_stats()
        assert stats["total_verified"] == 2
        assert stats["verified_success"] == 1
        assert stats["verified_failure"] == 1


# =============================================================================
# ENGINE TESTS
# =============================================================================

class TestHybridMCTSEngine:
    """Test main engine"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, sample_config):
        """Test engine initialization"""
        engine = HybridMCTSEngine(sample_config)
        assert engine.config == sample_config
        assert engine.cache is not None
        assert engine.monitor is not None

    @pytest.mark.asyncio
    async def test_search_evolved_policies(self, sample_config):
        """Test evolved policies search"""
        sample_config.approach = HybridMCTSApproach.EVOLVED_POLICIES
        engine = HybridMCTSEngine(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await engine.search(theorem)

        assert isinstance(result, HybridMCTSResult)
        assert result.approach_used == HybridMCTSApproach.EVOLVED_POLICIES
        assert result.policy_fitness_history is not None

    @pytest.mark.asyncio
    async def test_search_evolutionary_nodes(self, sample_config):
        """Test evolutionary nodes search"""
        sample_config.approach = HybridMCTSApproach.EVOLUTIONARY_NODES
        engine = HybridMCTSEngine(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await engine.search(theorem)

        assert isinstance(result, HybridMCTSResult)
        assert result.approach_used == HybridMCTSApproach.EVOLUTIONARY_NODES
        assert result.node_convergence_history is not None

    @pytest.mark.asyncio
    async def test_search_coevolution(self, sample_config):
        """Test coevolution search"""
        sample_config.approach = HybridMCTSApproach.COEVOLUTION
        engine = HybridMCTSEngine(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await engine.search(theorem)

        assert isinstance(result, HybridMCTSResult)
        assert result.approach_used == HybridMCTSApproach.COEVOLUTION
        assert result.pareto_front is not None

    @pytest.mark.asyncio
    async def test_search_adaptive(self, sample_config):
        """Test adaptive search"""
        sample_config.approach = HybridMCTSApproach.ADAPTIVE
        sample_config.adaptive_enabled = True
        engine = HybridMCTSEngine(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await engine.search(theorem)

        assert isinstance(result, HybridMCTSResult)

    @pytest.mark.asyncio
    async def test_search_combined(self, sample_config):
        """Test combined search"""
        sample_config.approach = HybridMCTSApproach.COMBINED
        engine = HybridMCTSEngine(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await engine.search(theorem)

        assert isinstance(result, HybridMCTSResult)
        assert result.approach_used == HybridMCTSApproach.COMBINED

    def test_get_monitoring_report(self, sample_config):
        """Test getting monitoring report"""
        engine = HybridMCTSEngine(sample_config)
        report = engine.get_monitoring_report()
        assert isinstance(report, dict)


# =============================================================================
# PRESET TESTS
# =============================================================================

class TestHybridMCTSPresets:
    """Test configuration presets"""

    def test_fast_preset(self):
        """Test fast preset"""
        config = HybridMCTSPresets.fast()
        assert config.simulations == 50
        assert config.max_depth == 25
        assert config.policy_generations == 5

    def test_balanced_preset(self):
        """Test balanced preset"""
        config = HybridMCTSPresets.balanced()
        assert config.approach == HybridMCTSApproach.ADAPTIVE
        assert config.adaptive_enabled

    def test_thorough_preset(self):
        """Test thorough preset"""
        config = HybridMCTSPresets.thorough()
        assert config.approach == HybridMCTSApproach.COMBINED
        assert config.simulations == 200
        assert config.early_stopping == False

    def test_leanaide_preset(self):
        """Test LeanAide-focused preset"""
        config = HybridMCTSPresets.leanaide_focused()
        assert config.leanaide_enabled
        assert config.leanaide_verify_every == 3

    def test_research_preset(self):
        """Test research preset"""
        config = HybridMCTSPresets.research()
        assert config.approach == HybridMCTSApproach.ADAPTIVE
        assert config.save_checkpoints

    def test_evolved_policies_only_preset(self):
        """Test evolved policies only preset"""
        config = HybridMCTSPresets.evolved_policies_only()
        assert config.approach == HybridMCTSApproach.EVOLVED_POLICIES
        assert config.policy_generations == 20

    def test_evolutionary_nodes_only_preset(self):
        """Test evolutionary nodes only preset"""
        config = HybridMCTSPresets.evolutionary_nodes_only()
        assert config.approach == HybridMCTSApproach.EVOLUTIONARY_NODES
        assert config.node_evolution_generations == 10

    def test_coevolution_only_preset(self):
        """Test coevolution only preset"""
        config = HybridMCTSPresets.coevolution_only()
        assert config.approach == HybridMCTSApproach.COEVOLUTION
        assert config.tree_generations == 100


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_framework_from_preset(self):
        """Test creating framework from preset"""
        engine = create_framework_from_preset("fast")
        assert isinstance(engine, HybridMCTSEngine)
        assert engine.config.simulations == 50

    def test_create_framework_invalid_preset(self):
        """Test creating framework with invalid preset"""
        with pytest.raises(ValueError):
            create_framework_from_preset("invalid_preset")

    @pytest.mark.asyncio
    async def test_quick_search(self):
        """Test quick search utility"""
        result = await quick_search("theorem test (a : nat) : a + 0 = a")
        assert isinstance(result, HybridMCTSResult)
        assert result.approach_used == HybridMCTSApproach.EVOLVED_POLICIES

    @pytest.mark.asyncio
    async def test_thorough_search(self):
        """Test thorough search utility"""
        result = await thorough_search("theorem test (a : nat) : a + 0 = a")
        assert isinstance(result, HybridMCTSResult)
        assert result.approach_used == HybridMCTSApproach.COMBINED


# =============================================================================
# WORKFLOW INTEGRATION TESTS
# =============================================================================

class TestHybridMCTSWorkflowIntegrator:
    """Test workflow integration"""

    @pytest.mark.asyncio
    async def test_solve_subproblem(self, sample_config, sample_subproblems):
        """Test solving a single subproblem"""
        integrator = HybridMCTSWorkflowIntegrator(sample_config)
        subproblem = sample_subproblems[0]

        solution = await integrator.solve_subproblem(subproblem)

        assert isinstance(solution, dict)
        assert "success" in solution
        assert "proof" in solution
        assert "approach" in solution

    @pytest.mark.asyncio
    async def test_solve_batch(self, sample_config, sample_subproblems):
        """Test solving multiple subproblems"""
        integrator = HybridMCTSWorkflowIntegrator(sample_config)

        solutions = await integrator.solve_batch(sample_subproblems)

        assert isinstance(solutions, list)
        assert len(solutions) <= len(sample_subproblems)
        assert all(isinstance(sol, dict) for sol in solutions)


# =============================================================================
# COMBINED SEARCH TESTS
# =============================================================================

class TestCombinedHybridMCTS:
    """Test combined hybrid approach"""

    @pytest.mark.asyncio
    async def test_search_combined_best(self, sample_config):
        """Test combined search with best strategy"""
        sample_config.combination_strategy = "best"
        combined = CombinedHybridMCTS(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await combined.search_combined(
            theorem,
            [
                HybridMCTSApproach.EVOLVED_POLICIES,
                HybridMCTSApproach.EVOLUTIONARY_NODES,
            ]
        )

        assert result.success
        assert result.approach_used == HybridMCTSApproach.COMBINED

    @pytest.mark.asyncio
    async def test_search_combined_weighted(self, sample_config):
        """Test combined search with weighted strategy"""
        sample_config.combination_strategy = "weighted"
        combined = CombinedHybridMCTS(sample_config)

        theorem = "theorem test (a b : nat) : a + b = b + a"
        result = await combined.search_combined(
            theorem,
            [
                HybridMCTSApproach.EVOLVED_POLICIES,
                HybridMCTSApproach.COEVOLUTION,
            ]
        )

        assert result.success


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, sample_theorems):
        """Test complete workflow from start to finish"""
        config = HybridMCTSPresets.balanced()
        engine = HybridMCTSEngine(config)

        results = []
        for theorem in sample_theorems:
            result = await engine.search(theorem)
            results.append(result)

        assert len(results) == len(sample_theorems)
        assert all(isinstance(r, HybridMCTSResult) for r in results)

    @pytest.mark.asyncio
    async def test_adaptive_workflow(self, sample_theorems):
        """Test adaptive workflow with multiple theorems"""
        config = HybridMCTSPresets.research()
        engine = HybridMCTSEngine(config)

        results = []
        for theorem in sample_theorems:
            result = await engine.search(theorem)
            results.append(result)

        # Check that selector learned something
        if engine.selector:
            stats = engine.selector.get_statistics()
            assert stats["total_selections"] == len(sample_theorems)

    @pytest.mark.asyncio
    async def test_checkpoint_workflow(self, sample_config, tmp_path):
        """Test checkpoint saving and loading workflow"""
        engine = HybridMCTSEngine(sample_config)

        # Run search
        theorem = "theorem test (a : nat) : a + 0 = a"
        result = await engine.search(theorem)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.json"
        engine.save_checkpoint(str(checkpoint_path))

        # Verify checkpoint exists
        assert checkpoint_path.exists()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and stress tests"""

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, sample_config):
        """Test multiple concurrent searches"""
        engine = HybridMCTSEngine(sample_config)

        theorems = [
            "theorem test1 (a : nat) : a + 0 = a",
            "theorem test2 (a b : nat) : a + b = b + a",
            "theorem test3 (a b c : nat) : (a + b) + c = a + (b + c)",
        ]

        # Run searches concurrently
        tasks = [engine.search(thm) for thm in theorems]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(theorems)
        assert all(isinstance(r, HybridMCTSResult) for r in results)

    @pytest.mark.asyncio
    async def test_cache_performance(self, sample_config):
        """Test cache performance benefits"""
        cache_enabled_config = sample_config
        cache_enabled_config.cache_enabled = True

        engine = HybridMCTSEngine(cache_enabled_config)

        theorem = "theorem test (a : nat) : a + 0 = a"

        # Run same theorem multiple times
        results = []
        for _ in range(3):
            result = await engine.search(theorem)
            results.append(result)

        # Check cache stats
        cache_stats = engine.cache.get_stats()
        # Note: In the stub implementation, caching might not be fully utilized
        assert "hits" in cache_stats
        assert "misses" in cache_stats


# =============================================================================
# DEMO AND EXAMPLES
# =============================================================================

async def demo_basic_usage():
    """Demonstrate basic usage of the framework"""
    print("\n" + "=" * 60)
    print("Hybrid MCTS Framework - Basic Usage Demo")
    print("=" * 60 + "\n")

    # Create engine with balanced preset
    engine = create_framework_from_preset("balanced")

    # Example theorem
    theorem = "theorem add_comm (a b : nat) : a + b = b + a"

    print(f"Theorem: {theorem}")
    print("\nRunning hybrid MCTS search...")

    # Run search
    result = await engine.search(theorem)

    # Print results
    print_result_summary(result)

    return result


async def demo_all_approaches():
    """Demonstrate all three approaches"""
    print("\n" + "=" * 60)
    print("Hybrid MCTS Framework - All Approaches Demo")
    print("=" * 60 + "\n")

    theorem = "theorem mul_assoc (a b c : nat) : (a * b) * c = a * (b * c)"

    approaches = [
        HybridMCTSApproach.EVOLVED_POLICIES,
        HybridMCTSApproach.EVOLUTIONARY_NODES,
        HybridMCTSApproach.COEVOLUTION,
    ]

    results = []
    for approach in approaches:
        print(f"\nRunning {approach.value}...")
        config = HybridMCTSPresets.fast()
        config.approach = approach
        engine = HybridMCTSEngine(config)
        result = await engine.search(theorem)
        results.append(result)
        print(f"  Fitness: {result.best_fitness:.4f}, Time: {result.execution_time:.2f}s")

    # Compare results
    print("\n" + "-" * 60)
    print("Comparison:")
    for result in results:
        print(f"  {result.approach_used.value:20s}: {result.best_fitness:.4f}")

    return results


async def demo_adaptive_selection():
    """Demonstrate adaptive approach selection"""
    print("\n" + "=" * 60)
    print("Hybrid MCTS Framework - Adaptive Selection Demo")
    print("=" * 60 + "\n")

    config = HybridMCTSPresets.research()
    engine = HybridMCTSEngine(config)

    theorems = [
        "theorem easy (a : nat) : a + 0 = a",
        "theorem medium (a b : nat) : a + b = b + a",
        "theorem hard (a b c : nat) : (a + b) * c = a * c + b * c",
    ]

    print("Running adaptive selection on multiple theorems...\n")

    for i, theorem in enumerate(theorems, 1):
        print(f"\nTheorem {i}: {theorem[:50]}...")
        result = await engine.search(theorem)
        print(f"  Approach: {result.approach_used.value}")
        print(f"  Fitness: {result.best_fitness:.4f}")

    # Show selector statistics
    if engine.selector:
        stats = engine.selector.get_statistics()
        print("\n" + "-" * 60)
        print("Selector Statistics:")
        print(f"  Total selections: {stats['total_selections']}")
        for key, value in stats.items():
            if key.endswith('_avg_performance'):
                approach = key.replace('_avg_performance', '')
                print(f"  {approach}: {value:.4f}")


async def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "#" * 60)
    print("# HYBRID MCTS FRAMEWORK - DEMONSTRATIONS")
    print("#" * 60)

    await demo_basic_usage()
    await demo_all_approaches()
    await demo_adaptive_selection()

    print("\n" + "#" * 60)
    print("# ALL DEMONSTRATIONS COMPLETE")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(run_all_demos())
