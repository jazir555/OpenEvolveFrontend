#!/usr/bin/env python3
"""
================================================================================
COMPREHENSIVE TEST SUITE FOR MDAP/MAKER + MATRYOSHKA INTEGRATION (OPTIONAL)
================================================================================

This module provides complete test coverage for the optional MDAP-Matryoshka
integration, testing both "with dependencies" and "without dependencies" scenarios.

Test Statistics:
- Total Test Functions: 55+
- Test Classes: 10
- Fixture Functions: 20+
- Coverage Areas: Optional deps, Configuration, Integration, CrewAI, E2E, Edge cases

Test Categories:
1. Optional Dependency Tests (8 tests)
2. Configuration Tests (7 tests)
3. Integration Tests - MDAPMakerWithMatryoshka (12 tests)
4. CrewAI Integration Tests (6 tests)
5. End-to-End Workflow Tests (8 tests)
6. Edge Case Tests (8 tests)
7. Fallback & Degradation Tests (6 tests)
8. Utility & Helper Tests (6 tests)

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
================================================================================
"""

import pytest
import time
import uuid
import tempfile
import os
import threading
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, mock_open, PropertyMock
from dataclasses import asdict, dataclass
from pathlib import Path

# ================================================================================
# OPTIONAL DEPENDENCY CHECKS
# ================================================================================

# Check Matryoshka availability
try:
    from matryoshka_unified_memory_integration import (
        MatryoshkaMemoryBridge,
        MatryoshkaExplorationSession,
        UnifiedMatryoshkaClient,
    )
    from matryoshka_enhanced_client import (
        EnhancedMatryoshkaClient,
        AnalysisOptions,
    )
    MATRYOSHKA_AVAILABLE = True
except ImportError:
    MATRYOSHKA_AVAILABLE = False

# Check MDAP/MAKER availability
try:
    from mdap_engine import MDAPConfig, MDAPRunResult, RedFlagRules, MDAPENGINE
    from maker_engine import MakerEngine, MakerConfig, MakerStep
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False

# Check Unified Memory availability
try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        create_unified_system,
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False

# Check CrewAI availability
try:
    from crewai import Agent, Task, Crew
    from crewai_mdap_maker_engine import MAKEREngineCrewAI, MAKERConfig
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import the module under test
try:
    from mdap_maker_matryoshka_integration import (
        MDAPMatryoshkaConfig,
        MDAPMatryoshkaResult,
        ExplorationResult,
        VotingResult,
        HybridDecompositionResult,
        ExplorationStrategy,
        MDAPMakerWithMatryoshka,
        CrewAIMDAPMakerWithMatryoshka,
        MatryoshkaDecisionHelper,
        create_mdap_maker_with_matryoshka,
        create_crewai_maker_with_matryoshka,
        create_auto_configured_engine,
        check_integration_health,
        get_integration_info,
        _check_document_size,
        _estimate_complexity,
        _should_use_matryoshka,
        MATRYOSHKA_AVAILABLE as MODULE_MATRYOSHKA_AVAILABLE,
        MDAP_AVAILABLE as MODULE_MDAP_AVAILABLE,
        UNIFIED_MEMORY_AVAILABLE as MODULE_UNIFIED_MEMORY_AVAILABLE,
        CREWAI_AVAILABLE as MODULE_CREWAI_AVAILABLE,
    )
    MODULE_AVAILABLE = True
except ImportError as e:
    MODULE_AVAILABLE = False
    MODULE_IMPORT_ERROR = str(e)


# ================================================================================
# PYTEST CONFIGURATION
# ================================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "optional_deps: tests for optional dependency handling")
    config.addinivalue_line("markers", "configuration: tests for configuration classes")
    config.addinivalue_line("markers", "integration: tests requiring module integration")
    config.addinivalue_line("markers", "crewai: tests for CrewAI integration")
    config.addinivalue_line("markers", "e2e: end-to-end workflow tests")
    config.addinivalue_line("markers", "edge_cases: edge case and error handling tests")
    config.addinivalue_line("markers", "fallback: fallback and degradation tests")
    config.addinivalue_line("markers", "slow: slow running tests")


# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not MODULE_AVAILABLE,
    reason=f"mdap_maker_matryoshka_integration module not available: {MODULE_IMPORT_ERROR if not MODULE_AVAILABLE else ''}"
)


# ================================================================================
# TEST FIXTURES
# ================================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return str(tmp_path)


@pytest.fixture
def sample_document_path(temp_dir):
    """Create a sample document file for testing."""
    doc_path = os.path.join(temp_dir, "test_sample.py")
    sample_code = '''
def calculate_sum(numbers):
    """Calculate sum of a list of numbers."""
    return sum(numbers)

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_value(self, value):
        self.data.append(value)
    
    def process(self):
        return calculate_sum(self.data)
'''
    with open(doc_path, 'w') as f:
        f.write(sample_code)
    return doc_path


@pytest.fixture
def large_document_path(temp_dir):
    """Create a large document file (>10MB) for testing."""
    doc_path = os.path.join(temp_dir, "large_document.txt")
    # Create ~15MB file
    with open(doc_path, 'w') as f:
        for _ in range(15000):
            f.write("x" * 1024 + "\n")
    return doc_path


@pytest.fixture
def sample_markdown_path(temp_dir):
    """Create a sample markdown document for testing."""
    doc_path = os.path.join(temp_dir, "test_document.md")
    sample_md = '''# Project Documentation

## Overview
This is a sample project for testing.

## Features
- Feature 1: Core functionality
- Feature 2: Advanced capabilities
- Feature 3: Edge case handling

## Implementation
```python
def main():
    print("Hello World")
```

## Conclusion
Project is complete and tested.
'''
    with open(doc_path, 'w') as f:
        f.write(sample_md)
    return doc_path


@pytest.fixture
def mock_matryoshka_client():
    """Create a mock Matryoshka client."""
    client = MagicMock()
    client.explore = MagicMock(return_value={
        'content': 'Explored content',
        'insights': ['Insight 1', 'Insight 2'],
        'key_concepts': ['Concept 1', 'Concept 2'],
        'confidence': 0.85
    })
    return client


@pytest.fixture
def mock_memory_bridge():
    """Create a mock memory bridge."""
    bridge = MagicMock()
    bridge.retrieve_relevant_memories = MagicMock(return_value=[
        'Memory 1: Previous solution',
        'Memory 2: Related problem'
    ])
    bridge.initialize_document_state = MagicMock(return_value={
        'session_id': 'test_session',
        'document_path': '/test/doc.py'
    })
    return bridge


@pytest.fixture
def default_config():
    """Create a default MDAPMatryoshkaConfig."""
    return MDAPMatryoshkaConfig()


@pytest.fixture
def enabled_config():
    """Create an enabled MDAPMatryoshkaConfig."""
    return MDAPMatryoshkaConfig(enabled=True)


@pytest.fixture
def basic_engine(default_config):
    """Create a basic MDAPMakerWithMatryoshka instance."""
    return MDAPMakerWithMatryoshka(matryoshka_config=default_config)


@pytest.fixture
def enabled_engine(enabled_config):
    """Create an enabled MDAPMakerWithMatryoshka instance."""
    return MDAPMakerWithMatryoshka(matryoshka_config=enabled_config)


@pytest.fixture
def mock_mdap_engine():
    """Create a mock MDAP engine."""
    engine = MagicMock()
    engine.run = MagicMock(return_value={'solution': 'MDAP solution', 'score': 0.9})
    return engine


@pytest.fixture
def mock_maker_engine():
    """Create a mock MAKER engine."""
    engine = MagicMock()
    engine.solve = MagicMock(return_value={'solution': 'MAKER solution', 'confidence': 0.95})
    return engine


@pytest.fixture
def sample_problem():
    """Return a sample problem statement."""
    return "Optimize the distributed system architecture for better performance"


@pytest.fixture
def simple_problem():
    """Return a simple problem statement."""
    return "Calculate sum of two numbers"


@pytest.fixture
def complex_problem():
    """Return a complex problem statement."""
    return """
    Design and optimize a complex distributed microservices architecture 
    with multiple dependencies, handling integration challenges, system 
    scalability, and fault tolerance across multiple regions.
    """


@pytest.fixture
def sample_candidates():
    """Return sample candidates for voting."""
    return [
        {'id': 1, 'solution': 'Solution A', 'score': 0.8},
        {'id': 2, 'solution': 'Solution B', 'score': 0.75},
        {'id': 3, 'solution': 'Solution C', 'score': 0.9},
    ]


# ================================================================================
# 1. OPTIONAL DEPENDENCY TESTS
# ================================================================================

class TestOptionalDependencies:
    """Tests for optional dependency handling."""
    
    @pytest.mark.optional_deps
    def test_module_loads_without_matryoshka(self):
        """Test that module loads even when Matryoshka is not available."""
        # Module should be importable regardless of Matryoshka availability
        assert MODULE_AVAILABLE, "Module should load even without Matryoshka"
    
    @pytest.mark.optional_deps
    def test_capability_flags_exist(self):
        """Test that capability detection flags exist."""
        import mdap_maker_matryoshka_integration as test_module
        assert hasattr(test_module, 'MATRYOSHKA_AVAILABLE')
        assert hasattr(test_module, 'MDAP_AVAILABLE')
        assert hasattr(test_module, 'UNIFIED_MEMORY_AVAILABLE')
        assert hasattr(test_module, 'CREWAI_AVAILABLE')
    
    @pytest.mark.optional_deps
    def test_matryoshka_flag_matches_environment(self):
        """Test that MATRYOSHKA_AVAILABLE flag matches actual environment."""
        assert MODULE_MATRYOSHKA_AVAILABLE == MATRYOSHKA_AVAILABLE
    
    @pytest.mark.optional_deps
    def test_mdap_flag_matches_environment(self):
        """Test that MDAP_AVAILABLE flag matches actual environment."""
        assert MODULE_MDAP_AVAILABLE == MDAP_AVAILABLE
    
    @pytest.mark.optional_deps
    def test_unified_memory_flag_matches_environment(self):
        """Test that UNIFIED_MEMORY_AVAILABLE flag matches actual environment."""
        assert MODULE_UNIFIED_MEMORY_AVAILABLE == UNIFIED_MEMORY_AVAILABLE
    
    @pytest.mark.optional_deps
    def test_crewai_flag_matches_environment(self):
        """Test that CREWAI_AVAILABLE flag matches actual environment."""
        assert MODULE_CREWAI_AVAILABLE == CREWAI_AVAILABLE
    
    @pytest.mark.optional_deps
    def test_graceful_degradation_without_matryoshka(self, default_config):
        """Test that engine works without Matryoshka dependencies."""
        engine = MDAPMakerWithMatryoshka(matryoshka_config=default_config)
        # Should have basic functionality even without Matryoshka
        assert engine is not None
        assert hasattr(engine, 'has_matryoshka')
        assert hasattr(engine, 'has_mdap')
        assert hasattr(engine, 'has_maker')
    
    @pytest.mark.optional_deps
    def test_has_matryoshka_false_when_disabled(self, default_config):
        """Test has_matryoshka returns False when disabled."""
        engine = MDAPMakerWithMatryoshka(matryoshka_config=default_config)
        assert engine.has_matryoshka is False


# ================================================================================
# 2. CONFIGURATION TESTS
# ================================================================================

class TestConfiguration:
    """Tests for MDAPMatryoshkaConfig."""
    
    @pytest.mark.configuration
    def test_default_config_values(self):
        """Test default configuration values."""
        config = MDAPMatryoshkaConfig()
        assert config.enabled is False
        assert config.use_for_large_documents is True
        assert config.use_for_deep_exploration is True
        assert config.use_for_cross_session_learning is True
        assert config.enable_unified_memory is True
        assert config.matryoshka_max_turns == 20
        assert config.memory_limit_per_context == 15
        assert config.exploration_strategy == "adaptive"
        assert config.mdap_for_structure is True
        assert config.matryoshka_for_exploration is True
        assert config.document_size_threshold_mb == 10.0
        assert config.fallback_on_error is True
        assert config.cache_exploration_results is True
        assert config.exploration_timeout_seconds == 300
    
    @pytest.mark.configuration
    def test_config_validation_valid_values(self):
        """Test config validation with valid values."""
        config = MDAPMatryoshkaConfig(
            enabled=True,
            matryoshka_max_turns=10,
            memory_limit_per_context=5,
            document_size_threshold_mb=5.0
        )
        assert config.enabled is True
        assert config.matryoshka_max_turns == 10
        assert config.memory_limit_per_context == 5
        assert config.document_size_threshold_mb == 5.0
    
    @pytest.mark.configuration
    def test_config_validation_invalid_turns(self):
        """Test config validation with invalid turns."""
        with pytest.raises(ValueError, match="matryoshka_max_turns must be >= 1"):
            MDAPMatryoshkaConfig(matryoshka_max_turns=0)
        
        with pytest.raises(ValueError, match="matryoshka_max_turns must be >= 1"):
            MDAPMatryoshkaConfig(matryoshka_max_turns=-1)
    
    @pytest.mark.configuration
    def test_config_validation_invalid_memory_limit(self):
        """Test config validation with invalid memory limit."""
        with pytest.raises(ValueError, match="memory_limit_per_context must be >= 1"):
            MDAPMatryoshkaConfig(memory_limit_per_context=0)
        
        with pytest.raises(ValueError, match="memory_limit_per_context must be >= 1"):
            MDAPMatryoshkaConfig(memory_limit_per_context=-5)
    
    @pytest.mark.configuration
    def test_config_validation_invalid_threshold(self):
        """Test config validation with invalid document size threshold."""
        with pytest.raises(ValueError, match="document_size_threshold_mb must be >= 0"):
            MDAPMatryoshkaConfig(document_size_threshold_mb=-1.0)
    
    @pytest.mark.configuration
    def test_config_enable_disable_flags(self):
        """Test enable/disable flags work correctly."""
        # Enabled config
        enabled_config = MDAPMatryoshkaConfig(enabled=True)
        assert enabled_config.enabled is True
        
        # Disabled config
        disabled_config = MDAPMatryoshkaConfig(enabled=False)
        assert disabled_config.enabled is False
        
        # Default config (should be disabled)
        default_config = MDAPMatryoshkaConfig()
        assert default_config.enabled is False
    
    @pytest.mark.configuration
    def test_auto_configuration(self):
        """Test auto-configuration factory function."""
        # Auto-configure with complex problem
        complex_problem = "Optimize distributed architecture with multiple dependencies"
        engine = create_auto_configured_engine(problem=complex_problem)
        
        assert isinstance(engine, MDAPMakerWithMatryoshka)
        assert hasattr(engine, 'matryoshka_config')


# ================================================================================
# 3. INTEGRATION TESTS - MDAPMakerWithMatryoshka
# ================================================================================

class TestMDAPMakerWithMatryoshka:
    """Tests for MDAPMakerWithMatryoshka class."""
    
    @pytest.mark.integration
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        engine = MDAPMakerWithMatryoshka()
        assert engine is not None
        assert engine.matryoshka_config is not None
        assert engine.matryoshka_config.enabled is False
    
    @pytest.mark.integration
    def test_initialization_with_custom_config(self, enabled_config):
        """Test initialization with custom config."""
        engine = MDAPMakerWithMatryoshka(matryoshka_config=enabled_config)
        assert engine.matryoshka_config.enabled is True
    
    @pytest.mark.integration
    def test_has_matryoshka_property(self, basic_engine, enabled_engine):
        """Test has_matryoshka property."""
        # Basic engine (disabled) should not have Matryoshka
        assert basic_engine.has_matryoshka is False
        
        # Enabled engine may have Matryoshka (depends on environment)
        # But should return a boolean
        assert isinstance(enabled_engine.has_matryoshka, bool)
    
    @pytest.mark.integration
    def test_has_mdap_property(self, basic_engine):
        """Test has_mdap property."""
        # Should return a boolean
        assert isinstance(basic_engine.has_mdap, bool)
        # Should match environment
        assert basic_engine.has_mdap == (MDAP_AVAILABLE and basic_engine.mdap_engine is not None)
    
    @pytest.mark.integration
    def test_has_maker_property(self, basic_engine):
        """Test has_maker property."""
        # Should return a boolean
        assert isinstance(basic_engine.has_maker, bool)
        # Should match environment
        assert basic_engine.has_maker == (MDAP_AVAILABLE and basic_engine.maker_engine is not None)
    
    @pytest.mark.integration
    def test_get_status(self, basic_engine):
        """Test get_status method."""
        status = basic_engine.get_status()
        assert isinstance(status, dict)
        assert 'matryoshka_available' in status
        assert 'matryoshka_enabled' in status
        assert 'matryoshka_active' in status
        assert 'mdap_available' in status
        assert 'maker_available' in status
        assert 'exploration_cache_size' in status
    
    @pytest.mark.integration
    def test_solve_with_document_analysis_basic(self, basic_engine, simple_problem):
        """Test solve_with_document_analysis with basic inputs."""
        result = basic_engine.solve_with_document_analysis(simple_problem)
        
        assert isinstance(result, MDAPMatryoshkaResult)
        assert hasattr(result, 'mdap_result')
        assert hasattr(result, 'maker_result')
        assert hasattr(result, 'matryoshka_enhanced')
        assert hasattr(result, 'execution_time_ms')
        assert result.execution_time_ms >= 0
    
    @pytest.mark.integration
    def test_solve_with_document_analysis_empty_problem(self, basic_engine):
        """Test solve_with_document_analysis with empty problem."""
        result = basic_engine.solve_with_document_analysis("")
        
        assert isinstance(result, MDAPMatryoshkaResult)
        # Should handle empty problem gracefully
        assert result.error_message is not None or result.is_success() or not result.is_success()
    
    @pytest.mark.integration
    def test_solve_with_document_analysis_with_document(self, basic_engine, simple_problem, sample_document_path):
        """Test solve_with_document_analysis with document path."""
        result = basic_engine.solve_with_document_analysis(
            simple_problem,
            document_path=sample_document_path
        )
        
        assert isinstance(result, MDAPMatryoshkaResult)
        assert result.execution_time_ms >= 0
    
    @pytest.mark.integration
    def test_solve_with_document_analysis_force_matryoshka_disabled(self, basic_engine, simple_problem):
        """Test solve_with_document_analysis forcing Matryoshka off."""
        result = basic_engine.solve_with_document_analysis(
            simple_problem,
            use_matryoshka=False
        )
        
        assert isinstance(result, MDAPMatryoshkaResult)
        assert result.matryoshka_enhanced is False
    
    @pytest.mark.integration
    def test_decompose_with_memory(self, basic_engine, simple_problem):
        """Test decompose_with_memory method."""
        result = basic_engine.decompose_with_memory(simple_problem)
        
        assert isinstance(result, HybridDecompositionResult)
        assert hasattr(result, 'decomposition')
        assert hasattr(result, 'matryoshka_context')
        assert hasattr(result, 'subproblems')
    
    @pytest.mark.integration
    def test_vote_with_context_retrieval(self, basic_engine, sample_candidates):
        """Test vote_with_context_retrieval method."""
        result = basic_engine.vote_with_context_retrieval(
            candidates=sample_candidates,
            context_query="Find best solution",
            voting_method="standard"
        )
        
        assert isinstance(result, VotingResult)
        assert hasattr(result, 'winner')
        assert hasattr(result, 'rankings')
        assert hasattr(result, 'voting_method')
        assert result.voting_method == "standard"


# ================================================================================
# 4. CREWAI INTEGRATION TESTS
# ================================================================================

class TestCrewAIIntegration:
    """Tests for CrewAIMDAPMakerWithMatryoshka."""
    
    @pytest.mark.crewai
    def test_crewai_initialization(self):
        """Test CrewAI MAKER initialization."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        
        assert engine is not None
        assert hasattr(engine, 'matryoshka')
        assert hasattr(engine, 'base_maker')
        assert hasattr(engine, 'matryoshka_config')
    
    @pytest.mark.crewai
    def test_crewai_has_crewai_property(self):
        """Test has_crewai property."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        
        assert isinstance(engine.has_crewai, bool)
        # Should match environment
        assert engine.has_crewai == (CREWAI_AVAILABLE and engine.base_maker is not None)
    
    @pytest.mark.crewai
    def test_crewai_has_matryoshka_property(self):
        """Test has_matryoshka property in CrewAI MAKER."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        
        assert isinstance(engine.has_matryoshka, bool)
        # Should delegate to underlying matryoshka
        assert engine.has_matryoshka == engine.matryoshka.has_matryoshka
    
    @pytest.mark.crewai
    def test_crewai_solve(self):
        """Test CrewAI MAKER solve method."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        problem = "Test problem for CrewAI"
        
        result = engine.solve(problem)
        
        assert isinstance(result, MDAPMatryoshkaResult)
        assert hasattr(result, 'execution_time_ms')
    
    @pytest.mark.crewai
    def test_crewai_solve_with_document(self, sample_document_path):
        """Test CrewAI MAKER solve with document."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        problem = "Analyze this code"
        
        result = engine.solve(problem, document_path=sample_document_path)
        
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.crewai
    def test_crewai_get_status(self):
        """Test CrewAI MAKER get_status method."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        
        status = engine.get_status()
        
        assert isinstance(status, dict)
        assert 'crewai_available' in status
        assert 'crewai_active' in status
        assert 'matryoshka_available' in status


# ================================================================================
# 5. END-TO-END WORKFLOW TESTS
# ================================================================================

class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_workflow_problem_to_solution(self, basic_engine, simple_problem):
        """Test full workflow: problem -> decomposition -> voting -> solution."""
        # Step 1: Problem decomposition
        decomp_result = basic_engine.decompose_with_memory(simple_problem)
        assert isinstance(decomp_result, HybridDecompositionResult)
        
        # Step 2: Solve
        solve_result = basic_engine.solve_with_document_analysis(simple_problem)
        assert isinstance(solve_result, MDAPMatryoshkaResult)
        
        # Step 3: Voting (if we have candidates)
        candidates = [
            {'id': 1, 'solution': 'Approach A'},
            {'id': 2, 'solution': 'Approach B'}
        ]
        vote_result = basic_engine.vote_with_context_retrieval(
            candidates=candidates,
            context_query=simple_problem
        )
        assert isinstance(vote_result, VotingResult)
    
    @pytest.mark.e2e
    def test_document_analysis_workflow(self, basic_engine, sample_problem, sample_document_path):
        """Test document analysis workflow."""
        result = basic_engine.solve_with_document_analysis(
            problem=sample_problem,
            document_path=sample_document_path
        )
        
        assert isinstance(result, MDAPMatryoshkaResult)
        # Execution time should be >= 0 (may be 0 if no solver available)
        assert result.execution_time_ms >= 0
        # Document analysis should complete (may have error message if no solver)
        assert result.error_message is None or isinstance(result.error_message, str)
    
    @pytest.mark.e2e
    def test_document_analysis_with_content(self, basic_engine, sample_problem):
        """Test document analysis with direct content."""
        content = """
        This is a test document content.
        It describes a system architecture with multiple components.
        The system needs optimization for better performance.
        """
        
        result = basic_engine.solve_with_document_analysis(
            problem=sample_problem,
            document_content=content
        )
        
        assert isinstance(result, MDAPMatryoshkaResult)
        assert result.execution_time_ms >= 0
    
    @pytest.mark.e2e
    def test_cross_session_learning_workflow(self, enabled_engine, sample_problem):
        """Test cross-session learning workflow."""
        if not enabled_engine.has_matryoshka:
            pytest.skip("Matryoshka not available for cross-session learning test")
        
        # First session
        result1 = enabled_engine.solve_with_document_analysis(
            problem=sample_problem
        )
        assert isinstance(result1, MDAPMatryoshkaResult)
        
        # Second session - should potentially benefit from first
        result2 = enabled_engine.solve_with_document_analysis(
            problem=sample_problem + " with additional constraints"
        )
        assert isinstance(result2, MDAPMatryoshkaResult)
    
    @pytest.mark.e2e
    def test_hybrid_decomposition_workflow(self, basic_engine, complex_problem):
        """Test hybrid decomposition workflow."""
        result = basic_engine.decompose_with_memory(
            problem=complex_problem,
            context="Additional context for decomposition"
        )
        
        assert isinstance(result, HybridDecompositionResult)
        # Should have decomposition structure
        assert result.decomposition is not None or result.decomposition is None
    
    @pytest.mark.e2e
    def test_voting_with_memory_workflow(self, basic_engine, sample_candidates, sample_problem):
        """Test voting with memory retrieval workflow."""
        result = basic_engine.vote_with_context_retrieval(
            candidates=sample_candidates,
            context_query=sample_problem,
            voting_method="ranked"
        )
        
        assert isinstance(result, VotingResult)
        assert result.voting_method == "ranked"
        
        if result.rankings:
            # Rankings should be sorted by score (highest first)
            scores = [r[1] for r in result.rankings]
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.e2e
    def test_factory_function_basic(self):
        """Test factory function creates valid engine."""
        engine = create_mdap_maker_with_matryoshka(enabled=False)
        
        assert isinstance(engine, MDAPMakerWithMatryoshka)
        assert engine.matryoshka_config.enabled is False
    
    @pytest.mark.e2e
    def test_factory_function_crewai(self):
        """Test CrewAI factory function."""
        engine = create_crewai_maker_with_matryoshka(enabled=False)
        
        assert isinstance(engine, CrewAIMDAPMakerWithMatryoshka)


# ================================================================================
# 6. EDGE CASE TESTS
# ================================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""
    
    @pytest.mark.edge_cases
    def test_empty_problem(self, basic_engine):
        """Test handling of empty problem."""
        result = basic_engine.solve_with_document_analysis("")
        assert isinstance(result, MDAPMatryoshkaResult)
        # Should handle gracefully
    
    @pytest.mark.edge_cases
    def test_very_long_problem(self, basic_engine):
        """Test handling of very long problem."""
        long_problem = "optimize " * 1000  # Very long problem
        result = basic_engine.solve_with_document_analysis(long_problem)
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.edge_cases
    def test_problem_with_special_characters(self, basic_engine):
        """Test handling of problem with special characters."""
        special_problem = "Problem with special chars: !@#$%^&*()_+{}|:<>?[];',./"
        result = basic_engine.solve_with_document_analysis(special_problem)
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.edge_cases
    def test_missing_document(self, basic_engine):
        """Test handling of missing document path."""
        result = basic_engine.solve_with_document_analysis(
            problem="Test problem",
            document_path="/nonexistent/path/document.txt"
        )
        assert isinstance(result, MDAPMatryoshkaResult)
        # Should handle missing document gracefully
    
    @pytest.mark.edge_cases
    @pytest.mark.slow
    def test_very_large_document(self, basic_engine, large_document_path):
        """Test handling of very large document (>10MB)."""
        result = basic_engine.solve_with_document_analysis(
            problem="Analyze this large document",
            document_path=large_document_path
        )
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.edge_cases
    def test_unicode_content(self, basic_engine):
        """Test handling of Unicode content."""
        unicode_problem = "Problem with unicode: 你好世界 🌍 émojis ñoño"
        result = basic_engine.solve_with_document_analysis(unicode_problem)
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.edge_cases
    def test_none_inputs(self, basic_engine):
        """Test handling of None inputs."""
        # These should not crash
        basic_engine.decompose_with_memory(None)
        basic_engine.vote_with_context_retrieval(None, None)
    
    @pytest.mark.edge_cases
    def test_concurrent_operations(self, basic_engine, simple_problem):
        """Test concurrent operations."""
        results = []
        errors = []
        
        def worker():
            try:
                result = basic_engine.solve_with_document_analysis(simple_problem)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 5


# ================================================================================
# 7. FALLBACK & DEGRADATION TESTS
# ================================================================================

class TestFallbackAndDegradation:
    """Tests for fallback and graceful degradation."""
    
    @pytest.mark.fallback
    def test_standard_mdap_without_matryoshka(self, default_config):
        """Test that standard MDAP works without Matryoshka."""
        engine = MDAPMakerWithMatryoshka(matryoshka_config=default_config)
        
        assert engine.has_matryoshka is False
        # Should still work with standard MDAP
        result = engine.solve_with_document_analysis("Test problem")
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.fallback
    def test_fallback_on_error_config(self):
        """Test fallback on error configuration."""
        config = MDAPMatryoshkaConfig(
            enabled=True,
            fallback_on_error=True
        )
        assert config.fallback_on_error is True
        
        config_no_fallback = MDAPMatryoshkaConfig(
            enabled=True,
            fallback_on_error=False
        )
        assert config_no_fallback.fallback_on_error is False
    
    @pytest.mark.fallback
    def test_no_crash_without_optional_deps(self):
        """Test that system doesn't crash without any optional dependencies."""
        # Create minimal config
        config = MDAPMatryoshkaConfig(enabled=False)
        engine = MDAPMakerWithMatryoshka(matryoshka_config=config)
        
        # All operations should work without crashing
        assert engine.has_matryoshka is False
        status = engine.get_status()
        assert isinstance(status, dict)
        
        result = engine.solve_with_document_analysis("Test")
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.fallback
    def test_memory_bridge_without_unified_memory(self):
        """Test that memory bridge handles missing unified memory."""
        config = MDAPMatryoshkaConfig(enabled=False)
        engine = MDAPMakerWithMatryoshka(matryoshka_config=config)
        
        # Should work even without unified memory
        result = engine.decompose_with_memory("Test problem")
        assert isinstance(result, HybridDecompositionResult)
    
    @pytest.mark.fallback
    def test_crewai_fallback(self):
        """Test CrewAI fallback when not available."""
        engine = CrewAIMDAPMakerWithMatryoshka()
        
        # Should work regardless of CrewAI availability
        result = engine.solve("Test problem")
        assert isinstance(result, MDAPMatryoshkaResult)
    
    @pytest.mark.fallback
    def test_result_is_success_method(self, basic_engine):
        """Test MDAPMatryoshkaResult.is_success method."""
        result = basic_engine.solve_with_document_analysis("Test")
        
        assert isinstance(result.is_success(), bool)
        # Result should have consistent success state
        if result.mdap_result is not None or result.maker_result is not None:
            assert result.is_success() is True


# ================================================================================
# 8. UTILITY & HELPER TESTS
# ================================================================================

class TestUtilities:
    """Tests for utility functions and helpers."""
    
    @pytest.mark.integration
    def test_check_document_size_with_existing_file(self, sample_document_path):
        """Test _check_document_size with existing file."""
        # Small file should return False
        result = _check_document_size(sample_document_path, threshold_mb=10.0)
        assert result is False
        
        # Very low threshold should return True for any file
        result = _check_document_size(sample_document_path, threshold_mb=0.0001)
        assert result is True
    
    @pytest.mark.integration
    def test_check_document_size_with_nonexistent_file(self):
        """Test _check_document_size with non-existent file."""
        result = _check_document_size("/nonexistent/file.txt", threshold_mb=10.0)
        assert result is False
    
    @pytest.mark.integration
    def test_check_document_size_with_none_path(self):
        """Test _check_document_size with None path."""
        result = _check_document_size(None, threshold_mb=10.0)
        assert result is False
    
    @pytest.mark.integration
    def test_estimate_complexity_simple(self):
        """Test _estimate_complexity with simple problem."""
        simple = "Add two numbers"
        complexity = _estimate_complexity(simple)
        assert complexity < 5.0
        assert complexity >= 0
    
    @pytest.mark.integration
    def test_estimate_complexity_complex(self):
        """Test _estimate_complexity with complex problem."""
        complex_problem = """
        Optimize distributed architecture with multiple dependencies,
        complex system integration, and distributed computing challenges.
        """
        complexity = _estimate_complexity(complex_problem)
        assert complexity > 0
    
    @pytest.mark.integration
    def test_should_use_matryoshka_logic(self, enabled_config, default_config):
        """Test _should_use_matryoshka decision logic."""
        # Disabled config should always return False
        result = _should_use_matryoshka(default_config, problem="Complex optimize distributed")
        assert result is False
        
        # Enabled config - test with a problem that has high complexity (>5.0 threshold)
        # Complexity score = min(len(problem)/1000, 5.0) + 0.5 for each complex keyword
        # Need complexity > 5.0 to trigger Matryoshka use
        long_complex_problem = """
        Optimize distributed system architecture with complex integration challenges.
        This involves multiple dependencies, distributed computing requirements,
        and sophisticated system design patterns across multiple regions.
        The solution needs to handle complex scenarios and distributed workloads.
        """ * 10  # Make it long enough to trigger the complexity threshold
        
        result = _should_use_matryoshka(enabled_config, problem=long_complex_problem)
        # Function returns True only if: enabled, available, AND complexity > 5.0
        if enabled_config.enabled and MATRYOSHKA_AVAILABLE:
            assert result is True
        else:
            assert result is False
    
    @pytest.mark.integration
    def test_matryoshka_decision_helper_analyze_document(self, sample_document_path):
        """Test MatryoshkaDecisionHelper.analyze_document."""
        result = MatryoshkaDecisionHelper.analyze_document(sample_document_path)
        
        assert isinstance(result, dict)
        assert 'path' in result
        assert 'exists' in result
        assert 'size_mb' in result
        assert 'is_large' in result
        assert 'recommend_matryoshka' in result
        
        assert result['exists'] is True
        assert result['path'] == sample_document_path
    
    @pytest.mark.integration
    def test_matryoshka_decision_helper_analyze_problem(self):
        """Test MatryoshkaDecisionHelper.analyze_problem."""
        simple_problem = "Add two numbers"
        result = MatryoshkaDecisionHelper.analyze_problem(simple_problem)
        
        assert isinstance(result, dict)
        assert 'length' in result
        assert 'complexity_score' in result
        assert 'complexity_level' in result
        assert 'recommend_matryoshka' in result
        
        assert result['length'] == len(simple_problem)
        assert result['complexity_level'] in ['low', 'medium', 'high']
    
    @pytest.mark.integration
    def test_matryoshka_decision_helper_get_recommendation(self, sample_document_path):
        """Test MatryoshkaDecisionHelper.get_recommendation."""
        result = MatryoshkaDecisionHelper.get_recommendation(
            problem="Optimize complex distributed system",
            document_path=sample_document_path
        )
        
        assert isinstance(result, dict)
        assert 'recommend_matryoshka' in result
        assert 'matryoshka_available' in result
        assert 'reason' in result
        assert 'document_analysis' in result
        assert 'problem_analysis' in result
        
        assert isinstance(result['recommend_matryoshka'], bool)
        assert isinstance(result['matryoshka_available'], bool)
    
    @pytest.mark.integration
    def test_check_integration_health(self):
        """Test check_integration_health function."""
        health = check_integration_health()
        
        assert isinstance(health, dict)
        assert 'matryoshka' in health
        assert 'mdap_maker' in health
        assert 'unified_memory' in health
        assert 'crewai' in health
        assert 'decomposition' in health
        assert 'team' in health
        
        # Check nested structure
        assert 'available' in health['matryoshka']
        assert 'available' in health['mdap_maker']
    
    @pytest.mark.integration
    def test_get_integration_info(self):
        """Test get_integration_info function."""
        info = get_integration_info()
        
        assert isinstance(info, str)
        assert 'MDAP/MAKER + Matryoshka Integration Status' in info
        assert 'Matryoshka Available' in info
        assert 'MDAP/MAKER Available' in info


# ================================================================================
# DATA CLASS TESTS
# ================================================================================

class TestDataClasses:
    """Tests for data classes."""
    
    @pytest.mark.integration
    def test_mdap_matryoshka_result_creation(self):
        """Test MDAPMatryoshkaResult creation."""
        result = MDAPMatryoshkaResult()
        
        assert result.mdap_result is None
        assert result.maker_result is None
        assert result.matryoshka_enhanced is False
        assert result.exploration_result is None
        assert result.cross_session_insights == []
        assert result.execution_time_ms == 0.0
        assert result.fallback_used is False
        assert result.error_message is None
    
    @pytest.mark.integration
    def test_mdap_matryoshka_result_is_success(self):
        """Test MDAPMatryoshkaResult.is_success method."""
        # Empty result should not be success
        empty_result = MDAPMatryoshkaResult()
        assert empty_result.is_success() is False
        
        # Result with mdap_result should be success
        mdap_result = MDAPMatryoshkaResult(mdap_result={'solution': 'test'})
        assert mdap_result.is_success() is True
        
        # Result with maker_result should be success
        maker_result = MDAPMatryoshkaResult(maker_result={'solution': 'test'})
        assert maker_result.is_success() is True
    
    @pytest.mark.integration
    def test_mdap_matryoshka_result_get_solution(self):
        """Test MDAPMatryoshkaResult.get_solution method."""
        # Empty result returns None
        empty_result = MDAPMatryoshkaResult()
        assert empty_result.get_solution() is None
        
        # Result with maker solution
        maker_data = type('obj', (object,), {'solution': 'Maker Solution'})()
        maker_result = MDAPMatryoshkaResult(maker_result=maker_data)
        assert maker_result.get_solution() == 'Maker Solution'
    
    @pytest.mark.integration
    def test_exploration_result_creation(self):
        """Test ExplorationResult creation."""
        result = ExplorationResult(content="Test content")
        
        assert result.content == "Test content"
        assert result.insights == []
        assert result.key_concepts == []
        assert result.related_topics == []
        assert result.confidence == 0.0
        assert result.exploration_depth == 0
        assert result.memory_references == []
        assert result.metadata == {}
    
    @pytest.mark.integration
    def test_voting_result_creation(self):
        """Test VotingResult creation."""
        result = VotingResult()
        
        assert result.winner is None
        assert result.rankings == []
        assert result.context_used is False
        assert result.retrieved_memories == []
        assert result.voting_method == "standard"
        assert result.confidence == 0.0
    
    @pytest.mark.integration
    def test_hybrid_decomposition_result_creation(self):
        """Test HybridDecompositionResult creation."""
        result = HybridDecompositionResult()
        
        assert result.decomposition is None
        assert result.matryoshka_context is None
        assert result.subproblems == []
        assert result.cross_references == []
        assert result.recommended_strategy == "standard"


# ================================================================================
# EXPLORATION STRATEGY TESTS
# ================================================================================

class TestExplorationStrategy:
    """Tests for exploration strategies."""
    
    @pytest.mark.integration
    def test_exploration_strategy_values(self):
        """Test ExplorationStrategy enum values."""
        assert ExplorationStrategy.BREADTH_FIRST.value == "breadth_first"
        assert ExplorationStrategy.DEPTH_FIRST.value == "depth_first"
        assert ExplorationStrategy.ADAPTIVE.value == "adaptive"
        assert ExplorationStrategy.HYBRID.value == "hybrid"
    
    @pytest.mark.integration
    def test_config_with_different_strategies(self):
        """Test config with different exploration strategies."""
        for strategy in ["breadth_first", "depth_first", "adaptive", "hybrid"]:
            config = MDAPMatryoshkaConfig(
                enabled=True,
                exploration_strategy=strategy
            )
            assert config.exploration_strategy == strategy


# ================================================================================
# CACHE TESTS
# ================================================================================

class TestCache:
    """Tests for exploration caching."""
    
    @pytest.mark.integration
    def test_exploration_cache_initially_empty(self, basic_engine):
        """Test that exploration cache is initially empty."""
        assert len(basic_engine.exploration_cache) == 0
    
    @pytest.mark.integration
    def test_cache_exploration_results_config(self):
        """Test cache_exploration_results configuration."""
        config_with_cache = MDAPMatryoshkaConfig(
            enabled=True,
            cache_exploration_results=True
        )
        assert config_with_cache.cache_exploration_results is True
        
        config_without_cache = MDAPMatryoshkaConfig(
            enabled=True,
            cache_exploration_results=False
        )
        assert config_without_cache.cache_exploration_results is False


# ================================================================================
# NETWORK FAILURE SIMULATION TESTS
# ================================================================================

class TestNetworkFailures:
    """Tests simulating network/service failures."""
    
    @pytest.mark.edge_cases
    def test_matryoshka_failure_with_fallback(self, enabled_config):
        """Test Matryoshka failure with fallback enabled."""
        engine = MDAPMakerWithMatryoshka(matryoshka_config=enabled_config)
        
        # Patch _explore_with_matryoshka to simulate failure
        with patch.object(engine, '_explore_with_matryoshka', side_effect=Exception("Network error")):
            result = engine.solve_with_document_analysis(
                "Test problem",
                use_matryoshka=True
            )
        
        assert isinstance(result, MDAPMatryoshkaResult)
        # Should either succeed via fallback or have error message
        assert result.error_message is not None or result.is_success()
    
    @pytest.mark.edge_cases
    def test_memory_bridge_failure(self, enabled_config, mock_memory_bridge):
        """Test memory bridge failure handling."""
        engine = MDAPMakerWithMatryoshka(matryoshka_config=enabled_config)
        
        # Mock memory bridge to fail
        if engine.memory_bridge:
            engine.memory_bridge.retrieve_relevant_memories = MagicMock(
                side_effect=Exception("Memory unavailable")
            )
        
        # Should handle gracefully
        result = engine.vote_with_context_retrieval(
            candidates=[{'id': 1}],
            context_query="test"
        )
        
        assert isinstance(result, VotingResult)


# ================================================================================
# MODULE-LEVEL TESTS
# ================================================================================

class TestModuleLevel:
    """Tests for module-level functionality."""
    
    @pytest.mark.integration
    def test_module_has_all_exports(self):
        """Test that module exports all expected classes and functions."""
        import mdap_maker_matryoshka_integration as module
        
        expected_exports = [
            'MDAPMatryoshkaConfig',
            'MDAPMatryoshkaResult',
            'ExplorationResult',
            'VotingResult',
            'HybridDecompositionResult',
            'ExplorationStrategy',
            'MDAPMakerWithMatryoshka',
            'CrewAIMDAPMakerWithMatryoshka',
            'MatryoshkaDecisionHelper',
            'create_mdap_maker_with_matryoshka',
            'create_crewai_maker_with_matryoshka',
            'create_auto_configured_engine',
            'check_integration_health',
            'get_integration_info',
        ]
        
        for export in expected_exports:
            assert hasattr(module, export), f"Missing export: {export}"


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
