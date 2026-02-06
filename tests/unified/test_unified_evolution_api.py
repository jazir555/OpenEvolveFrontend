"""
Comprehensive Tests for Unified Evolution API
============================================

Tests the complete unified evolution pipeline including:
- Strategy selection
- Configuration generation
- Evolution execution
- Knowledge extraction
- Gauntlet evaluation
- Convenience functions
- All 6 domains

Author: Unified Evolution Team
Date: 2026-01-30
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, UTC

# Skip these tests if the unified_evolution_api module is not available
pytest.importorskip("openevolve.unified.unified_evolution_api")

# Import the API
from openevolve.unified.unified_evolution_api import (
    UnifiedEvolutionAPI,
    evolve,
    quick_evolve,
    evolve_no_gauntlet,
    evolve_batch,
    EvolutionResult,
    SystemMode,
    ProgressUpdate
)

# Import dependencies
try:
    from openevolve.unified.config import UnifiedEvolutionConfig, EvolutionMode, DomainType
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from openevolve.gauntlets.three_round_orchestrator import FullGauntletResult
    GAUNTLET_AVAILABLE = True
except ImportError:
    GAUNTLET_AVAILABLE = False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_knowledge_engine():
    """Mock knowledge engine"""
    ke = Mock()
    ke.query = AsyncMock(return_value=[])
    ke.execute_query = AsyncMock()
    return ke


@pytest.fixture
def mock_strategy_recommender():
    """Mock strategy recommender"""
    sr = Mock()
    sr.recommend_strategy = AsyncMock(return_value=Mock(
        recommended_system='openevolve',
        recommended_mode='qd',
        confidence=0.8,
        reasoning=Mock(primary_reason='Quality diversity needed'),
        config_overrides={'qd_enabled': True},
        expected_performance=Mock(
            expected_iterations=100,
            expected_time_seconds=50.0,
            expected_score=0.75,
            success_probability=0.8
        )
    ))
    sr.analyze_problem_characteristics = AsyncMock(return_value=Mock(
        domain='general',
        complexity='medium',
        evaluation_cost='moderate',
        has_multiple_objectives=False,
        requires_diversity=True,
        requires_robustness=False,
        constraint_count=0,
        estimated_iterations=100,
        keywords=['test', 'optimize']
    ))
    sr.learn_from_run = AsyncMock()
    return sr


@pytest.fixture
def api(mock_knowledge_engine, mock_strategy_recommender):
    """Create API instance with mocked dependencies"""
    return UnifiedEvolutionAPI(
        knowledge_engine=mock_knowledge_engine,
        strategy_recommender=mock_strategy_recommender,
        enable_gauntlets=False,  # Disable by default for faster tests
        enable_knowledge_extraction=False
    )


# ============================================================================
# TEST: BASIC EVOLUTION
# ============================================================================

@pytest.mark.asyncio
async def test_basic_evolution(api):
    """Test basic evolution with simple problem"""
    result = await api.evolve(
        problem="Optimize function: f(x) = x^2",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Verify result structure
    assert isinstance(result, EvolutionResult)
    assert result.best_solution is not None
    assert len(result.best_solution) > 0
    assert result.final_score >= 0.0
    assert result.total_time >= 0.0

    # Verify strategy was selected
    assert result.strategy_used is not None
    assert result.strategy_used.system in ['openevolve', 'loongflow']


@pytest.mark.asyncio
async def test_evolution_with_callback(api):
    """Test evolution with progress callback"""
    updates = []

    async def callback(update):
        updates.append(update)

    result = await api.evolve(
        problem="Test problem",
        domain="general",
        callback=callback,
        run_gauntlet=False,
        store_knowledge=False
    )

    # Verify callback was called
    assert len(updates) > 0

    # Verify progress stages
    stages = [u.stage for u in updates]
    assert 'analyzing' in stages
    assert 'selecting_strategy' in stages
    assert 'evolving' in stages
    assert 'complete' in stages

    # Verify final update
    final_update = updates[-1]
    assert final_update.percent_complete == 100
    assert final_update.stage == 'complete'


@pytest.mark.asyncio
async def test_evolution_with_constraints(api):
    """Test evolution with constraints"""
    result = await api.evolve(
        problem="Multi-objective optimization",
        domain="general",
        constraints={
            'objectives': ['objective1', 'objective2'],
            'constraints': ['constraint1'],
            'time_limit_seconds': 60
        },
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.best_solution is not None
    assert result.metadata is not None


# ============================================================================
# TEST: STRATEGY SELECTION
# ============================================================================

@pytest.mark.asyncio
async def test_finance_domain_uses_pes(api):
    """Test finance domain selects PES mode"""
    # Mock recommender to return PES
    api.strategy_recommender.recommend_strategy = AsyncMock(return_value=Mock(
        recommended_system='loongflow',
        recommended_mode='pes',
        confidence=0.9,
        reasoning=Mock(primary_reason='Expensive evaluations favor PES')
    ))

    result = await api.evolve(
        problem="Optimize portfolio allocation",
        domain="finance",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.strategy_used.system == 'loongflow'
    assert result.strategy_used.mode == 'pes'


@pytest.mark.asyncio
async def test_multi_objective_uses_mo(api):
    """Test multi-objective selects MO mode"""
    api.strategy_recommender.recommend_strategy = AsyncMock(return_value=Mock(
        recommended_system='openevolve',
        recommended_mode='mo',
        confidence=0.85,
        reasoning=Mock(primary_reason='Multiple objectives')
    ))

    result = await api.evolve(
        problem="Minimize risk and maximize return",
        domain="finance",
        constraints={'objectives': ['risk', 'return']},
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.strategy_used.mode == 'mo'


@pytest.mark.asyncio
async def test_rules_based_fallback():
    """Test rules-based selection when recommender unavailable"""
    api = UnifiedEvolutionAPI(
        strategy_recommender=None,  # No recommender
        enable_gauntlets=False,
        enable_knowledge_extraction=False
    )

    result = await api.evolve(
        problem="Expensive optimization",
        domain="science",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Should still work with rules-based selection
    assert result.strategy_used is not None
    assert result.best_solution is not None


# ============================================================================
# TEST: CONFIGURATION
# ============================================================================

@pytest.mark.asyncio
async def test_auto_config_generation(api):
    """Test automatic configuration generation"""
    result = await api.evolve(
        problem="Test problem",
        domain="science",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Config should be auto-generated
    assert result.config_used is not None


@pytest.mark.asyncio
async def test_custom_config(api):
    """Test custom configuration"""
    if not CONFIG_AVAILABLE:
        pytest.skip("Config not available")

    custom_config = UnifiedEvolutionConfig(
        max_iterations=50,
        domain=DomainType.SCIENCE
    )

    result = await api.evolve(
        problem="Test problem",
        domain="science",
        config=custom_config,
        run_gauntlet=False,
        store_knowledge=False
    )

    # Custom config should be used
    assert result.config_used.max_iterations == 50


# ============================================================================
# TEST: KNOWLEDGE EXTRACTION
# ============================================================================

@pytest.mark.asyncio
async def test_knowledge_extraction(mock_knowledge_engine, mock_strategy_recommender):
    """Test knowledge extraction from evolution"""
    api = UnifiedEvolutionAPI(
        knowledge_engine=mock_knowledge_engine,
        strategy_recommender=mock_strategy_recommender,
        enable_knowledge_extraction=True,
        enable_gauntlets=False
    )

    result = await api.evolve(
        problem="Test problem",
        domain="general",
        store_knowledge=True,
        run_gauntlet=False
    )

    # Should have artifacts
    assert len(result.evolution_artifacts) > 0


@pytest.mark.asyncio
async def test_no_knowledge_extraction_if_disabled(api):
    """Test no extraction when disabled"""
    result = await api.evolve(
        problem="Test problem",
        domain="general",
        store_knowledge=False,
        run_gauntlet=False
    )

    # Should have no artifacts
    assert len(result.evolution_artifacts) == 0


# ============================================================================
# TEST: GAUNTLET INTEGRATION
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not GAUNTLET_AVAILABLE, reason="Gauntlet not available")
async def test_gauntlet_execution():
    """Test gauntlet execution after evolution"""
    api = UnifiedEvolutionAPI(
        enable_gauntlets=True,
        enable_knowledge_extraction=False
    )

    result = await api.evolve(
        problem="Test problem",
        domain="general",
        run_gauntlet=True,
        store_knowledge=False
    )

    # Should have gauntlet result
    assert result.gauntlet_result is not None
    assert isinstance(result.gauntlet_result, FullGauntletResult)


@pytest.mark.asyncio
@pytest.mark.skipif(not GAUNTLET_AVAILABLE, reason="Gauntlet not available")
async def test_gauntlet_skipped_when_requested():
    """Test gauntlet can be disabled"""
    api = UnifiedEvolutionAPI(
        enable_gauntlets=True,
        enable_knowledge_extraction=False
    )

    result = await api.evolve(
        problem="Test problem",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Should not have gauntlet result
    assert result.gauntlet_result is None


# ============================================================================
# TEST: LEARNING
# ============================================================================

@pytest.mark.asyncio
async def test_learning_from_run(api):
    """Test that strategy recommender learns from runs"""
    result = await api.evolve(
        problem="Test problem",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Verify learn_from_run was called
    api.strategy_recommender.learn_from_run.assert_called_once()

    # Verify the call had correct data
    call_args = api.strategy_recommender.learn_from_run.call_args[0][0]
    assert 'run_id' in call_args
    assert 'domain' in call_args
    assert 'strategy_used' in call_args
    assert 'final_score' in call_args


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

@pytest.mark.asyncio
async def test_evolution_failure_graceful():
    """Test graceful failure handling"""
    api = UnifiedEvolutionAPI(
        enable_gauntlets=False,
        enable_knowledge_extraction=False
    )

    # Mock executor to raise exception
    with patch.object(api, '_execute_openevolve', side_effect=Exception("Test error")):
        result = await api.evolve(
            problem="Test problem",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )

        # Should return error result
        assert result.error is not None
        assert "Test error" in result.error
        assert result.final_score == 0.0


@pytest.mark.asyncio
async def test_loongflow_fallback(api):
    """Test fallback to OpenEvolve when LoongFlow fails"""
    # Set up to use LoongFlow but make it fail
    api.strategy_recommender.recommend_strategy = AsyncMock(return_value=Mock(
        recommended_system='loongflow',
        recommended_mode='pes',
        confidence=0.9,
        reasoning=Mock(primary_reason='PES mode')
    ))

    with patch('openevolve.unified.unified_evolution_api.LOONGFLOW_AVAILABLE', False):
        result = await api.evolve(
            problem="Test problem",
            domain="finance",
            run_gauntlet=False,
            store_knowledge=False
        )

        # Should still succeed with fallback
        assert result.best_solution is not None
        assert result.error is None


# ============================================================================
# TEST: CONVENIENCE FUNCTIONS
# ============================================================================

@pytest.mark.asyncio
async def test_quick_evolve():
    """Test quick_evolve convenience function"""
    solution = await quick_evolve(
        problem="Simple optimization",
        domain="general"
    )

    # Should return just the solution string
    assert isinstance(solution, str)
    assert len(solution) > 0


@pytest.mark.asyncio
async def test_evolve_no_gauntlet():
    """Test evolve_no_gauntlet convenience function"""
    result = await evolve_no_gauntlet(
        problem="Test problem",
        domain="general"
    )

    assert isinstance(result, EvolutionResult)
    assert result.gauntlet_result is None


@pytest.mark.asyncio
async def test_evolve_batch():
    """Test batch evolution"""
    problems = [
        "Problem 1",
        "Problem 2",
        "Problem 3"
    ]

    results = await evolve_batch(
        problems=problems,
        domain="general",
        max_concurrent=2
    )

    # Should return results for all problems
    assert len(results) == len(problems)
    assert all(isinstance(r, EvolutionResult) for r in results)


@pytest.mark.asyncio
async def test_evolve_batch_concurrency_limit():
    """Test that batch respects concurrency limit"""
    problems = [f"Problem {i}" for i in range(5)]

    # Track concurrent executions
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    original_evolve = evolve

    async def tracking_evolve(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent

        async with lock:
            concurrent_count += 1
            if concurrent_count > max_concurrent:
                max_concurrent = concurrent_count

        result = await original_evolve(*args, **kwargs)

        async with lock:
            concurrent_count -= 1

        return result

    with patch('openevolve.unified.unified_evolution_api.evolve', tracking_evolve):
        results = await evolve_batch(
            problems=problems,
            domain="general",
            max_concurrent=2
        )

        # Should not exceed concurrency limit
        assert max_concurrent <= 2


# ============================================================================
# TEST: RESULT SERIALIZATION
# ============================================================================

@pytest.mark.asyncio
async def test_result_to_dict(api):
    """Test EvolutionResult.to_dict()"""
    result = await api.evolve(
        problem="Test problem",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )

    result_dict = result.to_dict()

    # Verify keys
    assert 'best_solution' in result_dict
    assert 'final_score' in result_dict
    assert 'strategy_used' in result_dict
    assert 'total_time' in result_dict
    assert 'iterations' in result_dict
    assert 'evaluations' in result_dict


@pytest.mark.asyncio
async def test_result_save_and_load(api, tmp_path):
    """Test saving and loading EvolutionResult"""
    result = await api.evolve(
        problem="Test problem",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Save result
    filepath = tmp_path / "result.json"
    result.save(str(filepath))

    # Verify file exists
    assert filepath.exists()

    # Load result
    loaded_result = EvolutionResult.load(str(filepath))

    # Verify loaded data matches
    assert loaded_result.best_solution == result.best_solution
    assert loaded_result.final_score == result.final_score
    assert loaded_result.total_time == result.total_time


# ============================================================================
# TEST: DOMAIN-SPECIFIC
# ============================================================================

@pytest.mark.asyncio
async def test_finance_domain(api):
    """Test finance domain optimization"""
    result = await api.evolve(
        problem="Maximize portfolio Sharpe ratio",
        domain="finance",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.metadata['domain'] == 'finance'
    assert result.best_solution is not None


@pytest.mark.asyncio
async def test_trading_domain(api):
    """Test trading domain optimization"""
    result = await api.evolve(
        problem="Optimize trading strategy",
        domain="trading",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.metadata['domain'] == 'trading'


@pytest.mark.asyncio
async def test_science_domain(api):
    """Test science domain optimization"""
    result = await api.evolve(
        problem="Optimize experimental design",
        domain="science",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.metadata['domain'] == 'science'


@pytest.mark.asyncio
async def test_engineering_domain(api):
    """Test engineering domain optimization"""
    result = await api.evolve(
        problem="Design lightweight structure",
        domain="engineering",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.metadata['domain'] == 'engineering'


@pytest.mark.asyncio
async def test_pharma_domain(api):
    """Test pharma domain optimization"""
    result = await api.evolve(
        problem="Optimize drug dosage",
        domain="pharma",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.metadata['domain'] == 'pharma'


@pytest.mark.asyncio
async def test_web_domain(api):
    """Test web domain optimization"""
    result = await api.evolve(
        problem="Optimize page load time",
        domain="web",
        run_gauntlet=False,
        store_knowledge=False
    )

    assert result.metadata['domain'] == 'web'


# ============================================================================
# TEST: PERFORMANCE
# ============================================================================

@pytest.mark.asyncio
async def test_evolution_performance(api):
    """Test evolution completes in reasonable time"""
    import time

    start = time.time()
    result = await api.evolve(
        problem="Test problem",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )
    elapsed = time.time() - start

    # Should complete quickly (mock execution)
    assert elapsed < 5.0
    assert result.total_time >= 0


@pytest.mark.asyncio
async def test_batch_performance():
    """Test batch evolution performance"""
    import time

    problems = [f"Problem {i}" for i in range(5)]

    start = time.time()
    results = await evolve_batch(
        problems=problems,
        domain="general",
        max_concurrent=3
    )
    elapsed = time.time() - start

    # All results should be present
    assert len(results) == len(problems)

    # Should be faster than sequential (though mocked)
    # In real scenario with actual evolutions, this would matter more
    assert elapsed < 10.0


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
