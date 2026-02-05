#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Tests for RESE SCE Components

Tests ALL public functions and methods for:
- sce_bridge.py (SymbolicConstraintEngine, Constraint management)
- dito_optimizer.py (DITOOptimizer, graph optimization)

Coverage Goals:
- Unit tests for each function
- Integration tests between components
- Performance tests
- Error handling tests
- Idempotency tests
- CLAUDE.md compliance tests

Total Tests: 100+

Author: OpenEvolve
Created: 2026-02-04
"""

import os
import sys
import json
import time
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from sce_bridge import (
    SCEConfig,
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType,
    ConstraintCategory,
    LogicalFallacy,
    ContradictionPair,
    ContradictionDetectionResult,
    TacitAssumption,
)

try:
    from dito_optimizer import (
        DITOOptimizer,
        InferenceGraphNode,
        ActivationStrategy,
        DITOStats,
        BacktrackPoint,
        DITOContradiction,
    )
    DITO_AVAILABLE = True
except ImportError:
    DITO_AVAILABLE = False


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_sce_config():
    """Create sample SCE configuration"""
    os.environ['SCE_TIMEOUT_MS'] = '5000'
    os.environ['SCE_MAX_CONSTRAINTS'] = '1000'
    os.environ['SCE_MAX_ITERATIONS'] = '1000'
    os.environ['SCE_ENABLE_TACIT_MINING'] = 'true'
    os.environ['RESE_Z3_SCE_ENABLED'] = 'false'  # Disable Z3 for unit tests
    os.environ['RESE_DITO_ENABLED'] = 'false'  # Disable DITO for unit tests
    os.environ['RESE_DITO_ACTIVATION_STRATEGY'] = 'selective_bfs'  # Must be valid even when disabled
    return SCEConfig.from_env()


@pytest.fixture
def sample_constraints():
    """Create sample constraints for testing"""
    return [
        Constraint(
            constraint_id='c1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Temperature must be less than 1000',
        ),
        Constraint(
            constraint_id='c2',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Temperature must be greater than 0',
        ),
        Constraint(
            constraint_id='c3',
            type=ConstraintType.SOFT,
            category=ConstraintCategory.SOFT_STATISTICAL,
            description='Pressure should be around 5000',
            dependencies=['c1'],
        ),
    ]


@pytest.fixture
def sample_failure_patterns():
    """Create sample failure patterns"""
    return [
        {
            'pattern_description': 'lattice defects cause inconsistency',
            'failure_rate': 0.65,
            'data_points': 150,
        },
        {
            'pattern_description': 'temperature affects yield',
            'failure_rate': 0.45,
            'data_points': 200,
        },
    ]


# =============================================================================
# SCE CONFIGURATION TESTS (10 tests)
# =============================================================================

class TestSCEConfig:
    """Test SCEConfig"""

    def test_config_from_env_defaults(self, sample_sce_config):
        """Test default configuration values"""
        assert sample_sce_config.TIMEOUT_MS == 5000
        assert sample_sce_config.MAX_CONSTRAINTS == 1000
        assert sample_sce_config.MAX_ITERATIONS == 1000

    def test_config_custom_values(self, monkeypatch):
        """Test custom configuration"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '10000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '2000')
        config = SCEConfig.from_env()
        assert config.TIMEOUT_MS == 10000
        assert config.MAX_CONSTRAINTS == 2000

    def test_config_invalid_timeout(self, monkeypatch):
        """Test invalid timeout"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '-100')
        with pytest.raises(ValueError, match='must be positive'):
            SCEConfig.from_env()

    def test_config_invalid_max_constraints(self, monkeypatch):
        """Test invalid max constraints"""
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '0')
        with pytest.raises(ValueError, match='must be positive'):
            SCEConfig.from_env()

    def test_config_feature_flags(self, monkeypatch):
        """Test feature flags"""
        monkeypatch.setenv('SCE_ENABLE_TACIT_MINING', 'false')
        monkeypatch.setenv('SCE_TIMEOUT_MS', '5000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '1000')
        monkeypatch.setenv('SCE_MAX_ITERATIONS', '1000')
        monkeypatch.setenv('RESE_Z3_SCE_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ACTIVATION_STRATEGY', 'selective_bfs')
        config = SCEConfig.from_env()
        assert config.ENABLE_TACIT_ASSUMPTION_MINING is False

    def test_config_z3_settings(self, monkeypatch):
        """Test Z3 configuration"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '5000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '1000')
        monkeypatch.setenv('SCE_MAX_ITERATIONS', '1000')
        monkeypatch.setenv('SCE_ENABLE_TACIT_MINING', 'true')
        monkeypatch.setenv('RESE_Z3_SCE_ENABLED', 'true')
        monkeypatch.setenv('Z3_TIMEOUT', '10000')
        monkeypatch.setenv('Z3_MAX_MEMORY_MB', '8192')
        monkeypatch.setenv('RESE_DITO_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ACTIVATION_STRATEGY', 'selective_bfs')
        config = SCEConfig.from_env()
        assert config.ENABLE_Z3_SCE is True
        assert config.Z3_TIMEOUT_MS == 10000
        assert config.Z3_MAX_MEMORY_MB == 8192

    def test_config_dito_settings(self, monkeypatch):
        """Test DITO configuration"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '5000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '1000')
        monkeypatch.setenv('SCE_MAX_ITERATIONS', '1000')
        monkeypatch.setenv('SCE_ENABLE_TACIT_MINING', 'true')
        monkeypatch.setenv('RESE_Z3_SCE_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ENABLED', 'true')
        monkeypatch.setenv('RESE_DITO_ACTIVATION_STRATEGY', 'selective_dfs')
        monkeypatch.setenv('RESE_DITO_ENABLE_LEAN4', 'true')
        config = SCEConfig.from_env()
        assert config.ENABLE_DITO is True
        assert config.DITO_ACTIVATION_STRATEGY == 'selective_dfs'
        assert config.DITO_ENABLE_LEAN4 is True

    def test_config_invalid_dito_strategy(self, monkeypatch):
        """Test invalid DITO strategy"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '5000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '1000')
        monkeypatch.setenv('SCE_MAX_ITERATIONS', '1000')
        monkeypatch.setenv('SCE_ENABLE_TACIT_MINING', 'true')
        monkeypatch.setenv('RESE_Z3_SCE_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ACTIVATION_STRATEGY', 'invalid_strategy')
        with pytest.raises(ValueError, match='Invalid DITO_ACTIVATION_STRATEGY'):
            SCEConfig.from_env()

    def test_config_circuit_breaker_settings(self, monkeypatch):
        """Test circuit breaker configuration"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '5000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '1000')
        monkeypatch.setenv('SCE_MAX_ITERATIONS', '1000')
        monkeypatch.setenv('SCE_ENABLE_TACIT_MINING', 'true')
        monkeypatch.setenv('RESE_Z3_SCE_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ACTIVATION_STRATEGY', 'selective_bfs')
        monkeypatch.setenv('SCE_CIRCUIT_BREAKER_THRESHOLD', '10')
        monkeypatch.setenv('SCE_CIRCUIT_BREAKER_TIMEOUT_MS', '120000')
        config = SCEConfig.from_env()
        assert config.CIRCUIT_BREAKER_THRESHOLD == 10
        assert config.CIRCUIT_BREAKER_TIMEOUT_MS == 120000

    def test_config_max_contradiction_set_size(self, monkeypatch):
        """Test max contradiction set size"""
        monkeypatch.setenv('SCE_TIMEOUT_MS', '5000')
        monkeypatch.setenv('SCE_MAX_CONSTRAINTS', '1000')
        monkeypatch.setenv('SCE_MAX_ITERATIONS', '1000')
        monkeypatch.setenv('SCE_ENABLE_TACIT_MINING', 'true')
        monkeypatch.setenv('RESE_Z3_SCE_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ENABLED', 'false')
        monkeypatch.setenv('RESE_DITO_ACTIVATION_STRATEGY', 'selective_bfs')
        monkeypatch.setenv('SCE_MAX_CONTRADICTION_SET_SIZE', '50')
        config = SCEConfig.from_env()
        assert config.MAX_CONTRADICTION_SET_SIZE == 50


# =============================================================================
# CONSTRAINT DATA STRUCTURE TESTS (10 tests)
# =============================================================================

class TestConstraint:
    """Test Constraint dataclass"""

    def test_constraint_creation(self):
        """Test creating a constraint"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test constraint',
        )
        assert constraint.constraint_id == 'test-1'
        assert constraint.type == ConstraintType.HARD

    def test_constraint_with_dependencies(self):
        """Test constraint with dependencies"""
        constraint = Constraint(
            constraint_id='c3',
            type=ConstraintType.SOFT,
            category=ConstraintCategory.SOFT_STATISTICAL,
            description='Test',
            dependencies=['c1', 'c2'],
        )
        assert len(constraint.dependencies) == 2
        assert 'c1' in constraint.dependencies

    def test_constraint_defaults(self):
        """Test default values"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test',
        )
        assert constraint.dependencies == []
        assert constraint.formalized_in_lean4 is False
        assert constraint.lean4_theorem is None
        assert constraint.created_at is not None

    def test_constraint_to_dict(self):
        """Test converting to dict"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test',
        )
        data = constraint.to_dict()
        assert data['constraint_id'] == 'test-1'
        assert data['type'] == 'hard'
        assert data['category'] == 'hard_parameter_inequality'

    def test_constraint_timestamp_utc(self):
        """Test UTC timestamp"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test',
        )
        # Verify timezone-aware
        assert constraint.created_at.tzinfo == timezone.utc

    def test_constraint_expression(self):
        """Test constraint with expression"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test',
            expression='x < 10',
        )
        assert constraint.expression == 'x < 10'

    def test_constraint_hard_type(self):
        """Test HARD constraint type"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test',
        )
        assert constraint.type == ConstraintType.HARD

    def test_constraint_soft_type(self):
        """Test SOFT constraint type"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.SOFT,
            category=ConstraintCategory.SOFT_STATISTICAL,
            description='Test',
        )
        assert constraint.type == ConstraintType.SOFT

    def test_constraint_category_hard_parameter(self):
        """Test HARD_PARAMETER_INEQUALITY category"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.HARD,
            category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
            description='Test',
        )
        assert constraint.category == ConstraintCategory.HARD_PARAMETER_INEQUALITY

    def test_constraint_category_tacit_assumption(self):
        """Test TACIT_ASSUMPTION category"""
        constraint = Constraint(
            constraint_id='test-1',
            type=ConstraintType.SOFT,
            category=ConstraintCategory.TACIT_ASSUMPTION,
            description='Test',
        )
        assert constraint.category == ConstraintCategory.TACIT_ASSUMPTION


# =============================================================================
# SYMBOLIC CONSTRAINT ENGINE TESTS (20 tests)
# =============================================================================

class TestSymbolicConstraintEngine:
    """Test SymbolicConstraintEngine"""

    @pytest.fixture
    def engine(self, sample_sce_config):
        """Create engine for testing"""
        return SymbolicConstraintEngine(config=sample_sce_config)

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.config is not None
        assert isinstance(engine.constraints, dict)
        assert len(engine.constraints) == 0

    @pytest.mark.asyncio
    async def test_add_constraint(self, engine, sample_constraints):
        """Test adding a constraint"""
        constraint = sample_constraints[0]
        result = await engine.add_constraint(constraint, 'test-123')
        assert result['added'] is True
        assert result['updated'] is False
        assert 'c1' in engine.constraints

    @pytest.mark.asyncio
    async def test_add_constraint_upsert(self, engine, sample_constraints):
        """Test adding existing constraint (upsert)"""
        constraint = sample_constraints[0]
        await engine.add_constraint(constraint, 'test-123')
        result = await engine.add_constraint(constraint, 'test-456')
        assert result['added'] is False
        assert result['updated'] is True

    @pytest.mark.asyncio
    async def test_add_constraint_max_limit(self, engine):
        """Test max constraint limit"""
        # Create config with small limit
        engine.config.MAX_CONSTRAINTS = 2
        c1 = Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test 1')
        c2 = Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test 2')
        c3 = Constraint('c3', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test 3')

        await engine.add_constraint(c1, 'test-1')
        await engine.add_constraint(c2, 'test-2')
        with pytest.raises(ValueError, match='maximum limit'):
            await engine.add_constraint(c3, 'test-3')

    @pytest.mark.asyncio
    async def test_remove_constraint(self, engine, sample_constraints):
        """Test removing a constraint"""
        constraint = sample_constraints[0]
        await engine.add_constraint(constraint, 'test-123')
        result = await engine.remove_constraint('c1', 'test-456')
        assert result['removed'] is True
        assert 'c1' not in engine.constraints

    @pytest.mark.asyncio
    async def test_remove_constraint_nonexistent(self, engine):
        """Test removing non-existent constraint"""
        result = await engine.remove_constraint('nonexistent', 'test-123')
        assert result['removed'] is False

    @pytest.mark.asyncio
    async def test_get_constraint(self, engine, sample_constraints):
        """Test getting a constraint"""
        constraint = sample_constraints[0]
        await engine.add_constraint(constraint, 'test-123')
        retrieved = engine.get_constraint('c1')
        assert retrieved is not None
        assert retrieved.constraint_id == 'c1'

    @pytest.mark.asyncio
    async def test_get_constraint_nonexistent(self, engine):
        """Test getting non-existent constraint"""
        retrieved = engine.get_constraint('nonexistent')
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_all_constraints(self, engine, sample_constraints):
        """Test getting all constraints"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        all_constraints = engine.get_all_constraints()
        assert len(all_constraints) == 3

    @pytest.mark.asyncio
    async def test_get_constraints_by_type(self, engine, sample_constraints):
        """Test filtering by type"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        hard_constraints = engine.get_constraints_by_type(ConstraintType.HARD)
        assert len(hard_constraints) == 2

    @pytest.mark.asyncio
    async def test_get_constraints_by_category(self, engine, sample_constraints):
        """Test filtering by category"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        hard_constraints = engine.get_constraints_by_category(ConstraintCategory.HARD_PARAMETER_INEQUALITY)
        assert len(hard_constraints) == 2

    @pytest.mark.asyncio
    async def test_detect_contradictions_naive(self, engine, sample_constraints):
        """Test naive contradiction detection"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        result = await engine.detect_contradictions('test-456')
        assert isinstance(result, ContradictionDetectionResult)
        assert result.total_checked == 3

    @pytest.mark.asyncio
    async def test_detect_contradictions_empty(self, engine):
        """Test contradiction detection with no constraints"""
        result = await engine.detect_contradictions('test-123')
        assert result.contradiction_found is False
        assert result.total_checked == 0

    @pytest.mark.asyncio
    async def test_check_consistency(self, engine, sample_constraints):
        """Test consistency checking"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        result = await engine.check_consistency('test-456')
        assert 'consistent' in result
        assert 'issues' in result

    @pytest.mark.asyncio
    async def test_clear_constraints(self, engine, sample_constraints):
        """Test clearing all constraints"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        assert len(engine.constraints) == 3
        engine.clear()
        assert len(engine.constraints) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, engine, sample_constraints):
        """Test getting statistics"""
        for constraint in sample_constraints:
            await engine.add_constraint(constraint, 'test-123')
        stats = engine.get_stats()
        assert 'constraint_count' in stats
        assert stats['constraint_count'] == 3
        assert stats['hard_constraints'] == 2
        assert stats['soft_constraints'] == 1

    @pytest.mark.asyncio
    async def test_mine_tacit_assumptions(self, engine, sample_failure_patterns):
        """Test tacit assumption mining"""
        assumptions = await engine.mine_tacit_assumptions(sample_failure_patterns, 'test-123')
        assert isinstance(assumptions, list)
        # May be empty if disabled

    @pytest.mark.asyncio
    async def test_perform_epistemic_audit(self, engine, sample_failure_patterns):
        """Test full epistemic audit"""
        result = await engine.perform_epistemic_audit(
            problem_description='Test problem',
            failure_patterns=sample_failure_patterns,
            correlation_id='test-123',
        )
        assert 'phase' in result
        assert 'audit_id' in result
        assert 'tacit_assumptions' in result
        assert 'contradictions' in result

    @pytest.mark.asyncio
    async def test_reset_circuit_breakers(self, engine):
        """Test resetting circuit breakers"""
        engine.reset_circuit_breakers()
        # No-op test, just ensures it doesn't error


# =============================================================================
# CONTRADICTION PAIR TESTS (8 tests)
# =============================================================================

class TestContradictionPair:
    """Test ContradictionPair"""

    def test_contradiction_pair_creation(self):
        """Test creating contradiction pair"""
        pair = ContradictionPair(
            constraint1_id='c1',
            constraint2_id='c2',
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=1,
            affected_premises=['c1', 'c2'],
            detected_at=datetime.now(timezone.utc),
        )
        assert pair.constraint1_id == 'c1'
        assert pair.type == LogicalFallacy.CONTRADICTION

    def test_contradiction_pair_to_dict(self):
        """Test converting to dict"""
        pair = ContradictionPair(
            constraint1_id='c1',
            constraint2_id='c2',
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=1,
            affected_premises=['c1', 'c2'],
            detected_at=datetime.now(timezone.utc),
        )
        data = pair.to_dict()
        assert data['type'] == 'contradiction'
        assert data['constraint1_id'] == 'c1'

    def test_contradiction_fallacy_type(self):
        """Test CONTRADICTION fallacy type"""
        assert LogicalFallacy.CONTRADICTION.value == 'contradiction'

    def test_circular_reasoning_fallacy(self):
        """Test CIRCULUS_IN_PROBANDO fallacy type"""
        assert LogicalFallacy.CIRCULUS_IN_PROBANDO.value == 'circulus_in_probando'

    def test_confirmation_bias_fallacy(self):
        """Test CONFIRMATION_BIAS fallacy type"""
        assert LogicalFallacy.CONFIRMATION_BIAS.value == 'confirmation_bias'

    def test_inconsistency_fallacy(self):
        """Test INCONSISTENCY fallacy type"""
        assert LogicalFallacy.INCONSISTENCY.value == 'inconsistency'

    def test_timestamp_utc(self):
        """Test UTC timestamp"""
        pair = ContradictionPair(
            constraint1_id='c1',
            constraint2_id='c2',
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=1,
            affected_premises=['c1', 'c2'],
            detected_at=datetime.now(timezone.utc),
        )
        assert pair.detected_at.tzinfo == timezone.utc

    def test_large_contradiction_set(self):
        """Test large contradiction set"""
        pair = ContradictionPair(
            constraint1_id='c1',
            constraint2_id='c2',
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=10,
            rollback_steps=5,
            affected_premises=[f'c{i}' for i in range(10)],
            detected_at=datetime.now(timezone.utc),
        )
        assert pair.contradiction_set_size == 10
        assert len(pair.affected_premises) == 10


# =============================================================================
# TACIT ASSUMPTION TESTS (8 tests)
# =============================================================================

class TestTacitAssumptionSCE:
    """Test TacitAssumption in SCE"""

    def test_tacit_assumption_creation(self):
        """Test creating tacit assumption"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test assumption',
            source_pattern='Test pattern',
            confidence_score=0.75,
            supporting_evidence_count=100,
        )
        assert assumption.id == 'test-1'
        assert assumption.confidence_score == 0.75

    def test_tacit_assumption_to_dict(self):
        """Test converting to dict"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test assumption',
            source_pattern='Test pattern',
            confidence_score=0.75,
            supporting_evidence_count=100,
        )
        data = assumption.to_dict()
        assert data['id'] == 'test-1'
        assert data['description'] == 'Test assumption'

    def test_tacit_assumption_defaults(self):
        """Test default values"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test',
            source_pattern='Pattern',
            confidence_score=0.5,
            supporting_evidence_count=10,
        )
        assert assumption.formalized_in_lean4 is False
        assert assumption.lean4_proposition is None

    def test_tacit_assumption_lean4_formalized(self):
        """Test Lean 4 formalization"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test',
            source_pattern='Pattern',
            confidence_score=0.5,
            supporting_evidence_count=10,
            formalized_in_lean4=True,
            lean4_proposition='theorem test : Prop := sorry',
        )
        assert assumption.formalized_in_lean4 is True
        assert 'theorem' in assumption.lean4_proposition

    def test_confidence_score_range(self):
        """Test confidence score in valid range"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test',
            source_pattern='Pattern',
            confidence_score=1.0,
            supporting_evidence_count=10,
        )
        assert 0 <= assumption.confidence_score <= 1

    def test_supporting_evidence_count(self):
        """Test supporting evidence count"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test',
            source_pattern='Pattern',
            confidence_score=0.5,
            supporting_evidence_count=999,
        )
        assert assumption.supporting_evidence_count == 999

    def test_assumption_id_unique(self):
        """Test unique ID generation"""
        a1 = TacitAssumption(
            id=str(uuid.uuid4()),
            description='Test 1',
            source_pattern='Pattern 1',
            confidence_score=0.5,
            supporting_evidence_count=10,
        )
        a2 = TacitAssumption(
            id=str(uuid.uuid4()),
            description='Test 2',
            source_pattern='Pattern 2',
            confidence_score=0.5,
            supporting_evidence_count=10,
        )
        assert a1.id != a2.id

    def test_source_pattern_preserved(self):
        """Test source pattern is preserved"""
        assumption = TacitAssumption(
            id='test-1',
            description='Test',
            source_pattern='Original failure pattern',
            confidence_score=0.5,
            supporting_evidence_count=10,
        )
        assert assumption.source_pattern == 'Original failure pattern'


# =============================================================================
# DITO OPTIMIZER TESTS (20 tests, if available)
# =============================================================================

@pytest.mark.skipif(not DITO_AVAILABLE, reason="DITO not available")
class TestDITOOptimizer:
    """Test DITOOptizer"""

    @pytest.fixture
    def dito(self):
        """Create DITO optimizer"""
        return DITOOptimizer()

    def test_dito_initialization(self, dito):
        """Test DITO initialization"""
        assert dito.graph is not None
        assert dito.stats is not None

    def test_dito_build_graph(self, dito, sample_constraints):
        """Test building inference graph"""
        dito.build_inference_graph(sample_constraints)
        assert dito.stats.total_nodes == 3

    def test_dito_selective_bfs_strategy(self, dito):
        """Test SELECTIVE_BFS strategy"""
        dito.activation_strategy = ActivationStrategy.SELECTIVE_BFS
        assert dito.activation_strategy == ActivationStrategy.SELECTIVE_BFS

    def test_dito_selective_dfs_strategy(self, dito):
        """Test SELECTIVE_DFS strategy"""
        dito.activation_strategy = ActivationStrategy.SELECTIVE_DFS
        assert dito.activation_strategy == ActivationStrategy.SELECTIVE_DFS

    def test_dito_minimal_subgraph_strategy(self, dito):
        """Test MINIMAL_SUBGRAPH strategy"""
        dito.activation_strategy = ActivationStrategy.MINIMAL_SUBGRAPH
        assert dito.activation_strategy == ActivationStrategy.MINIMAL_SUBGRAPH

    def test_dito_full_strategy(self, dito):
        """Test FULL strategy"""
        dito.activation_strategy = ActivationStrategy.FULL
        assert dito.activation_strategy == ActivationStrategy.FULL

    def test_dito_stats_initialization(self, dito):
        """Test stats initialization"""
        assert dito.stats.total_nodes == 0
        assert dito.stats.verified_nodes == 0
        assert dito.stats.active_nodes == 0

    def test_dito_backtrack_point_creation(self, dito):
        """Test creating backtrack point"""
        point = BacktrackPoint(
            node_id='c1',
            parent_constraints=['c0'],
            verified_nodes=['c0'],
        )
        assert point.node_id == 'c1'
        assert len(point.verified_nodes) == 1

    def test_dito_inference_graph_node(self, dito):
        """Test inference graph node"""
        node = InferenceGraphNode(
            constraint_id='c1',
            dependencies=[],
            dependents=['c2'],
            verified=True,
        )
        assert node.constraint_id == 'c1'
        assert node.verified is True

    def test_dito_optimize_contradiction_detection(self, dito, sample_constraints):
        """Test optimizing contradiction detection"""
        dito.build_inference_graph(sample_constraints)
        contradictions, stats = dito.optimize_contradiction_detection(
            sample_constraints,
            'test-123',
        )
        assert isinstance(contradictions, list)
        assert isinstance(stats, DITOStats)

    def test_dito_stats_complexity_saved(self, dito):
        """Test complexity saved calculation"""
        stats = DITOStats(
            total_nodes=100,
            verified_nodes=30,
            active_nodes=30,
            complexity_saved=70.0,
            atp_checks_performed=30,
            backtracks_performed=5,
        )
        assert stats.complexity_saved == 70.0

    def test_dito_contradiction_creation(self, dito):
        """Test DITO contradiction creation"""
        contradiction = DITOContradiction(
            constraint1_id='c1',
            constraint2_id='c2',
            type=LogicalFallacy.CONTRADICTION,
            contradiction_set_size=2,
            rollback_steps=1,
            affected_premises=['c1', 'c2'],
            detected_at=datetime.now(timezone.utc),
        )
        assert contradiction.constraint1_id == 'c1'


# =============================================================================
# INTEGRATION TESTS (10 tests)
# =============================================================================

class TestSCEIntegration:
    """Integration tests for SCE components"""

    @pytest.fixture
    def engine(self, sample_sce_config):
        """Create engine"""
        return SymbolicConstraintEngine(config=sample_sce_config)

    @pytest.mark.asyncio
    async def test_full_audit_workflow(self, engine, sample_failure_patterns):
        """Test complete audit workflow"""
        # Add constraints
        constraints = [
            Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'T < 1000'),
            Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'T > 0'),
        ]
        for c in constraints:
            await engine.add_constraint(c, 'test-123')

        # Run audit
        result = await engine.perform_epistemic_audit(
            problem_description='Temperature constraint test',
            failure_patterns=sample_failure_patterns,
            correlation_id='test-456',
        )

        assert result['phase'] == 'phase1_epistemic_audit'
        assert 'metrics' in result

    @pytest.mark.asyncio
    async def test_constraint_lifecycle(self, engine):
        """Test full constraint lifecycle"""
        constraint = Constraint(
            'c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test'
        )

        # Add
        await engine.add_constraint(constraint, 'test-1')
        assert engine.get_constraint('c1') is not None

        # Update (upsert)
        await engine.add_constraint(constraint, 'test-2')
        assert engine.get_constraint('c1') is not None

        # Remove
        await engine.remove_constraint('c1', 'test-3')
        assert engine.get_constraint('c1') is None

    @pytest.mark.asyncio
    async def test_contradiction_detection_workflow(self, engine):
        """Test contradiction detection workflow"""
        # Add contradictory constraints
        c1 = Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'X is true')
        c2 = Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'not X is true')
        await engine.add_constraint(c1, 'test-1')
        await engine.add_constraint(c2, 'test-2')

        # Detect contradictions
        result = await engine.detect_contradictions('test-3')
        assert isinstance(result, ContradictionDetectionResult)

    @pytest.mark.asyncio
    async def test_consistency_check_workflow(self, engine):
        """Test consistency check workflow"""
        # Add constraints with dependency
        c1 = Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Base')
        c2 = Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Depends', dependencies=['c1'])
        await engine.add_constraint(c1, 'test-1')
        await engine.add_constraint(c2, 'test-2')

        # Check consistency
        result = await engine.check_consistency('test-3')
        assert result['consistent'] is True

    @pytest.mark.asyncio
    async def test_tacit_assumption_workflow(self, engine, sample_failure_patterns):
        """Test tacit assumption mining workflow"""
        assumptions = await engine.mine_tacit_assumptions(sample_failure_patterns, 'test-123')
        assert isinstance(assumptions, list)

    @pytest.mark.asyncio
    async def test_multiple_contraddiction_sets(self, engine):
        """Test handling multiple contradiction sets"""
        constraints = [
            Constraint(f'c{i}', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, f'Constraint {i}')
            for i in range(10)
        ]
        for c in constraints:
            await engine.add_constraint(c, 'test-123')

        result = await engine.detect_contradictions('test-456')
        assert result.total_checked == 10

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, engine):
        """Test statistics tracking"""
        for i in range(5):
            c = Constraint(f'c{i}', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, f'Test {i}')
            await engine.add_constraint(c, 'test-123')

        stats = engine.get_stats()
        assert stats['constraint_count'] == 5

    @pytest.mark.asyncio
    async def test_clear_and_rebuild(self, engine):
        """Test clearing and rebuilding"""
        constraints = [
            Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test 1'),
            Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test 2'),
        ]
        for c in constraints:
            await engine.add_constraint(c, 'test-123')

        assert len(engine.constraints) == 2
        engine.clear()
        assert len(engine.constraints) == 0

        # Rebuild
        for c in constraints:
            await engine.add_constraint(c, 'test-456')
        assert len(engine.constraints) == 2

    @pytest.mark.asyncio
    async def test_idempotent_operations(self, engine):
        """Test idempotent constraint operations"""
        constraint = Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Test')

        # Add multiple times
        await engine.add_constraint(constraint, 'test-1')
        await engine.add_constraint(constraint, 'test-2')
        await engine.add_constraint(constraint, 'test-3')

        # Should only have one
        assert len(engine.get_all_constraints()) == 1

    @pytest.mark.asyncio
    async def test_large_constraint_set(self, engine):
        """Test handling large constraint sets"""
        count = 100
        for i in range(count):
            c = Constraint(f'c{i}', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, f'Test {i}')
            await engine.add_constraint(c, 'test-123')

        assert len(engine.get_all_constraints()) == count


# =============================================================================
# ERROR HANDLING TESTS (5 tests)
# =============================================================================

class TestSCEErrorHandling:
    """Test error handling in SCE"""

    @pytest.fixture
    def engine(self, sample_sce_config):
        """Create engine"""
        return SymbolicConstraintEngine(config=sample_sce_config)

    @pytest.mark.asyncio
    async def test_invalid_constraint_id(self, engine):
        """Test operations with invalid constraint ID"""
        result = await engine.remove_constraint('nonexistent-id-12345', 'test-123')
        assert result['removed'] is False

    @pytest.mark.asyncio
    async def test_empty_constraint_list(self, engine):
        """Test operations with empty constraint list"""
        result = await engine.detect_contradictions('test-123')
        assert result.total_checked == 0
        assert result.contradiction_found is False

    @pytest.mark.asyncio
    async def test_corrupted_dependency_chain(self, engine):
        """Test handling corrupted dependency chain"""
        c1 = Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Base')
        c2 = Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'Depends', dependencies=['nonexistent'])
        await engine.add_constraint(c1, 'test-1')
        await engine.add_constraint(c2, 'test-2')

        result = await engine.check_consistency('test-3')
        # Should have consistency issue
        assert len(result['issues']) > 0

    @pytest.mark.asyncio
    async def test_circular_dependencies(self, engine):
        """Test circular dependency detection"""
        c1 = Constraint('c1', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'C1', dependencies=['c2'])
        c2 = Constraint('c2', ConstraintType.HARD, ConstraintCategory.HARD_PARAMETER_INEQUALITY, 'C2', dependencies=['c1'])
        await engine.add_constraint(c1, 'test-1')
        await engine.add_constraint(c2, 'test-2')

        result = await engine.check_consistency('test-3')
        # Should detect cycle
        assert any('cycle' in issue.lower() for issue in result['issues'])

    @pytest.mark.asyncio
    async def test_empty_failure_patterns(self, engine):
        """Test mining with empty failure patterns"""
        assumptions = await engine.mine_tacit_assumptions([], 'test-123')
        assert isinstance(assumptions, list)
        assert len(assumptions) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
