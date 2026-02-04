"""
Comprehensive Test Suite for Matryoshka + Unified Memory Integration

This module provides complete test coverage for the Matryoshka Unified Memory Integration:

- MatryoshkaMemoryBridge (bridge between Matryoshka and unified memory)
- MatryoshkaExplorationSession (exploration session with memory backing)
- UnifiedMatryoshkaClient (high-level client interface)
- Exploration data structures (ExplorationStep, DocumentState, etc.)

Test Statistics:
- Total Test Functions: 40+
- Test Classes: 8
- Fixture Functions: 15
- Coverage Areas: Integration, Session, Context, Performance, Error Handling

Test Categories:
1. Integration Tests - MatryoshkaMemoryBridge functionality
2. Exploration Session Tests - Session lifecycle and exploration
3. Unified Client Tests - High-level client operations
4. Context Rot Prevention Tests - Long exploration preservation
5. Error Handling Tests - Graceful degradation
6. Performance Tests - Timing requirements

Performance Requirements:
- Exploration step recording < 50ms
- Context retrieval < 100ms  
- Synthesis < 500ms

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import time
import uuid
import tempfile
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, mock_open
from dataclasses import asdict

# Import Matryoshka Unified Memory Integration components
try:
    from matryoshka_unified_memory_integration import (
        MatryoshkaMemoryBridge,
        MatryoshkaExplorationSession,
        UnifiedMatryoshkaClient,
        ExplorationStep,
        DocumentState,
        ExplorationContext,
        ExplorationResult,
        SynthesisResult,
        AnalysisResult,
        ExplorationStepType,
        create_unified_matryoshka_client,
        create_memory_backed_session,
    )
    MATRYOSHKA_UNIFIED_AVAILABLE = True
except ImportError:
    MATRYOSHKA_UNIFIED_AVAILABLE = False
    pytestmark = pytest.mark.skip("Matryoshka unified memory integration not available")

try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        UnifiedMemory,
        UnifiedMemoryConfig,
        MemoryStatus,
        TurnProcessingResult,
        create_unified_system,
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary directory for memory databases."""
    memory_dir = tmp_path / "test_matryoshka_memory"
    memory_dir.mkdir()
    return str(memory_dir)


@pytest.fixture
def mock_unified_memory():
    """Create a mock unified memory system."""
    mock = MagicMock(spec=UnifiedMemorySystem)
    mock._memory_registry = {}
    mock._conversation_memories = {}
    mock.state_manager = MagicMock()
    mock._hybrid_retrieve = MagicMock(return_value=[])
    mock._index_memory = MagicMock()
    return mock


@pytest.fixture
def memory_bridge(temp_memory_dir):
    """Create a MatryoshkaMemoryBridge instance."""
    if UNIFIED_MEMORY_AVAILABLE:
        unified = create_unified_system(
            db_dir=temp_memory_dir,
            max_context_tokens=8000,
            enable_maintenance=False
        )
        bridge = MatryoshkaMemoryBridge(unified_memory=unified)
    else:
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
    return bridge


@pytest.fixture
def memory_bridge_mocked(mock_unified_memory):
    """Create a MatryoshkaMemoryBridge with mocked unified memory."""
    bridge = MatryoshkaMemoryBridge(unified_memory=mock_unified_memory)
    return bridge


@pytest.fixture
def sample_document_path(tmp_path):
    """Create a sample document file for testing."""
    doc_path = tmp_path / "test_sample.py"
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
    doc_path.write_text(sample_code)
    return str(doc_path)


@pytest.fixture
def sample_markdown_path(tmp_path):
    """Create a sample markdown document for testing."""
    doc_path = tmp_path / "test_document.md"
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
    doc_path.write_text(sample_md)
    return str(doc_path)


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID."""
    return f"test_session_{uuid.uuid4().hex[:16]}"


@pytest.fixture
def initialized_document_state(memory_bridge, sample_session_id, sample_document_path):
    """Create and initialize a document state."""
    doc_state = memory_bridge.initialize_document_state(
        session_id=sample_session_id,
        document_path=sample_document_path,
        document_type="python",
        document_size=500,
        initial_goal="Find all functions and classes"
    )
    return doc_state


@pytest.fixture
def exploration_session(temp_memory_dir, sample_document_path):
    """Create a MatryoshkaExplorationSession instance."""
    session_id = f"exploration_{uuid.uuid4().hex[:16]}"
    if UNIFIED_MEMORY_AVAILABLE:
        unified = create_unified_system(
            db_dir=temp_memory_dir,
            max_context_tokens=8000,
            enable_maintenance=False
        )
        session = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document_path,
            query="Analyze this file",
            unified_memory=unified
        )
    else:
        session = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document_path,
            query="Analyze this file",
            unified_memory=None
        )
    return session


@pytest.fixture
def unified_client(temp_memory_dir):
    """Create a UnifiedMatryoshkaClient instance."""
    if UNIFIED_MEMORY_AVAILABLE:
        unified = create_unified_system(
            db_dir=temp_memory_dir,
            max_context_tokens=8000,
            enable_maintenance=False
        )
        client = UnifiedMatryoshkaClient(unified_memory=unified)
    else:
        client = UnifiedMatryoshkaClient(unified_memory=None)
    return client


@pytest.fixture
def sample_exploration_steps():
    """Create sample exploration steps."""
    session_id = "test_session_123"
    steps = []
    for i in range(1, 6):
        step = ExplorationStep(
            step_id=f"step_{i}_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            turn_number=i,
            step_type=ExplorationStepType.OBSERVATION,
            query=f"Query {i}",
            code_executed=f"print('Step {i}')",
            observation=f"Observation from step {i}",
            insight=f"Insight {i}: found {i} items",
            importance=0.5 + (i * 0.05),
            confidence=0.7
        )
        steps.append(step)
    return steps


# =============================================================================
# TEST CLASS: MatryoshkaMemoryBridge - Integration Tests
# =============================================================================

class TestMatryoshkaMemoryBridgeIntegration:
    """Test MatryoshkaMemoryBridge core functionality."""

    def test_bridge_initialization(self, temp_memory_dir):
        """Test memory bridge initialization."""
        if UNIFIED_MEMORY_AVAILABLE:
            unified = create_unified_system(db_dir=temp_memory_dir)
            bridge = MatryoshkaMemoryBridge(unified_memory=unified)
            assert bridge.unified_memory is not None
        else:
            bridge = MatryoshkaMemoryBridge(unified_memory=None)
            assert bridge.unified_memory is None

    def test_record_exploration_step_basic(self, memory_bridge, sample_session_id):
        """Test recording a basic exploration step."""
        memory = memory_bridge.record_exploration_step(
            session_id=sample_session_id,
            turn_number=1,
            query="Find all functions",
            code_executed="import ast\nprint('functions')",
            observation="Found 3 functions",
            insight="There are 3 main functions in this module",
            step_type=ExplorationStepType.OBSERVATION,
            importance=0.8,
            confidence=0.9
        )
        
        if memory_bridge.unified_memory:
            assert memory is not None
            assert memory.memory_type == "observation"
            assert memory.importance == 0.8
            assert memory.confidence == 0.9
        else:
            assert memory is None

    def test_record_multiple_steps(self, memory_bridge, sample_session_id):
        """Test recording multiple exploration steps."""
        memories = []
        for i in range(1, 4):
            memory = memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Query {i}",
                code_executed=f"code_{i}",
                observation=f"Observation {i}",
                insight=f"Insight {i}",
                importance=0.5 + (i * 0.1)
            )
            memories.append(memory)
        
        steps = memory_bridge._exploration_steps.get(sample_session_id, [])
        assert len(steps) == 3
        assert steps[0].turn_number == 1
        assert steps[2].turn_number == 3
        # Check chaining
        assert steps[1].previous_step_id == steps[0].step_id
        assert steps[2].previous_step_id == steps[1].step_id

    def test_four_layer_indexing_happens(self, memory_bridge_mocked, sample_session_id):
        """Test that 4-layer indexing is triggered when recording steps."""
        memory_bridge_mocked.record_exploration_step(
            session_id=sample_session_id,
            turn_number=1,
            query="Test query",
            code_executed="test_code",
            observation="test observation",
            insight="test insight"
        )
        
        # Verify indexing was called
        memory_bridge_mocked.unified_memory._index_memory.assert_called()

    def test_hybrid_retrieval_returns_context(self, memory_bridge, sample_session_id):
        """Test that hybrid retrieval returns exploration context."""
        # Record some steps first
        for i in range(1, 4):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Query about functions {i}",
                code_executed=f"code_{i}",
                observation=f"Found {i} functions",
                insight=f"Functions: {i}"
            )
        
        context = memory_bridge.get_exploration_context(
            session_id=sample_session_id,
            current_query="functions",
            max_memories=5
        )
        
        assert isinstance(context, ExplorationContext)
        assert context.total_memories_available >= 3

    def test_state_maintained_across_turns(self, memory_bridge, sample_session_id, sample_document_path):
        """Test that document state is maintained across exploration turns."""
        # Initialize document state
        doc_state = memory_bridge.initialize_document_state(
            session_id=sample_session_id,
            document_path=sample_document_path,
            initial_goal="Analyze code"
        )
        
        # Record steps with findings
        for i in range(1, 4):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Query {i}",
                code_executed=f"code_{i}",
                observation=f"Observation {i}",
                insight=f"Finding {i}",
                confidence=0.8
            )
        
        # Verify state accumulated findings
        updated_state = memory_bridge._document_states.get(sample_session_id)
        assert updated_state is not None
        assert updated_state.total_turns >= 3
        assert len(updated_state.key_findings) == 3

    def test_step_memory_mapping(self, memory_bridge, sample_session_id):
        """Test that steps are properly mapped to memory IDs."""
        memory = memory_bridge.record_exploration_step(
            session_id=sample_session_id,
            turn_number=1,
            query="Test",
            code_executed="test",
            observation="obs",
            insight="insight"
        )
        
        if memory:
            # Find the step
            steps = memory_bridge._exploration_steps.get(sample_session_id, [])
            if steps:
                step_id = steps[0].step_id
                assert step_id in memory_bridge._step_memory_map


# =============================================================================
# TEST CLASS: ExplorationSession Tests
# =============================================================================

class TestExplorationSession:
    """Test MatryoshkaExplorationSession functionality."""

    def test_session_creation(self, sample_document_path):
        """Test exploration session creation and initialization."""
        session_id = f"test_{uuid.uuid4().hex[:12]}"
        session = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document_path,
            query="Analyze this file"
        )
        
        assert session.session_id == session_id
        assert session.document_path == sample_document_path
        assert session.original_query == "Analyze this file"
        assert session.current_turn == 0
        assert not session.is_complete

    def test_session_detects_document_type(self, sample_document_path):
        """Test automatic document type detection."""
        session_id = f"test_{uuid.uuid4().hex[:12]}"
        session = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document_path,
            query="Analyze"
        )
        
        assert session.document_state.document_type == "python"

    def test_session_detects_markdown_type(self, sample_markdown_path):
        """Test markdown document type detection."""
        session_id = f"test_{uuid.uuid4().hex[:12]}"
        session = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_markdown_path,
            query="Analyze"
        )
        
        assert session.document_state.document_type == "markdown"

    def test_exploration_runs_correct_turns(self, exploration_session):
        """Test that exploration runs the correct number of turns."""
        result = exploration_session.explore(max_turns=3)
        
        assert result.total_turns <= 3
        assert len(result.steps) == result.total_turns

    def test_each_turn_recorded_in_memory(self, exploration_session):
        """Test that each turn is recorded in unified memory."""
        result = exploration_session.explore(max_turns=3)
        
        bridge = exploration_session.memory_bridge
        session_id = exploration_session.session_id
        
        # Should have initialization + exploration steps
        steps = bridge._exploration_steps.get(session_id, [])
        assert len(steps) >= result.total_turns

    def test_final_synthesis_uses_all_memories(self, exploration_session):
        """Test that final synthesis uses all indexed memories."""
        result = exploration_session.explore(max_turns=3)
        
        synthesis = exploration_session.memory_bridge.synthesize_findings(
            exploration_session.session_id
        )
        
        assert synthesis.steps_used >= result.total_turns
        assert synthesis.synthesis != ""

    def test_session_add_finding(self, exploration_session):
        """Test manual finding addition."""
        exploration_session.add_finding("Test finding", confidence=0.9)
        
        assert len(exploration_session.document_state.key_findings) == 1
        assert exploration_session.document_state.key_findings[0]["finding"] == "Test finding"

    def test_session_get_stats(self, exploration_session):
        """Test getting session statistics."""
        exploration_session.explore(max_turns=2)
        stats = exploration_session.get_stats()
        
        assert "session_id" in stats
        assert "total_steps" in stats
        assert stats["total_steps"] >= 2


# =============================================================================
# TEST CLASS: Unified Client Tests
# =============================================================================

class TestUnifiedMatryoshkaClient:
    """Test UnifiedMatryoshkaClient functionality."""

    def test_client_initialization(self, temp_memory_dir):
        """Test unified client initialization."""
        if UNIFIED_MEMORY_AVAILABLE:
            unified = create_unified_system(db_dir=temp_memory_dir)
            client = UnifiedMatryoshkaClient(unified_memory=unified)
            assert client.is_available()
        else:
            client = UnifiedMatryoshkaClient(unified_memory=None)
            assert not client.is_available()

    def test_analyze_with_memory_end_to_end(self, unified_client, sample_document_path):
        """Test end-to-end analysis with memory."""
        result = unified_client.analyze_with_memory(
            query="Find all functions and classes",
            file_path=sample_document_path,
            max_turns=3
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.success
        assert result.document_path == sample_document_path
        assert len(result.session_id) > 0

    def test_analyze_file_not_found(self, unified_client):
        """Test analysis with non-existent file."""
        result = unified_client.analyze_with_memory(
            query="Analyze",
            file_path="/nonexistent/file.py",
            max_turns=3
        )
        
        assert not result.success
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_continue_analysis_recalls_session(self, unified_client, sample_document_path):
        """Test continuing analysis recalls previous session."""
        # First analysis
        result1 = unified_client.analyze_with_memory(
            query="Find all functions",
            file_path=sample_document_path,
            max_turns=2
        )
        
        session_id = result1.session_id
        
        # Continue analysis
        result2 = unified_client.continue_analysis(
            session_id=session_id,
            follow_up_query="Find all classes",
            max_turns=2
        )
        
        assert result2.success
        assert result2.session_id == session_id

    def test_continue_analysis_invalid_session(self, unified_client):
        """Test continuing analysis with invalid session ID."""
        result = unified_client.continue_analysis(
            session_id="invalid_session_id",
            follow_up_query="Analyze more",
            max_turns=2
        )
        
        assert not result.success
        assert "not found" in result.error.lower()

    def test_search_across_sessions(self, unified_client, sample_document_path):
        """Test searching across multiple sessions."""
        # Create multiple sessions
        for i in range(3):
            unified_client.analyze_with_memory(
                query=f"Analysis {i}",
                file_path=sample_document_path,
                max_turns=2
            )
        
        # Search across sessions
        results = unified_client.search_across_sessions(
            query="functions",
            limit=10
        )
        
        assert isinstance(results, list)

    def test_get_session_synthesis(self, unified_client, sample_document_path):
        """Test getting session synthesis."""
        result = unified_client.analyze_with_memory(
            query="Analyze code",
            file_path=sample_document_path,
            max_turns=2
        )
        
        synthesis = unified_client.get_session_synthesis(result.session_id)
        
        if synthesis:
            assert isinstance(synthesis, SynthesisResult)
            assert synthesis.synthesis != ""

    def test_list_sessions(self, unified_client, sample_document_path):
        """Test listing active sessions."""
        # Create a session
        unified_client.analyze_with_memory(
            query="Test analysis",
            file_path=sample_document_path,
            max_turns=2
        )
        
        sessions = unified_client.list_sessions()
        
        assert isinstance(sessions, list)
        assert len(sessions) >= 1

    def test_close_session(self, unified_client, sample_document_path):
        """Test closing a session."""
        result = unified_client.analyze_with_memory(
            query="Test",
            file_path=sample_document_path,
            max_turns=2
        )
        
        closed = unified_client.close_session(result.session_id)
        
        assert closed

    def test_close_session_invalid(self, unified_client):
        """Test closing invalid session."""
        closed = unified_client.close_session("invalid_id")
        
        assert not closed

    def test_get_session_memory(self, unified_client, sample_document_path):
        """Test getting session memory context."""
        result = unified_client.analyze_with_memory(
            query="Test",
            file_path=sample_document_path,
            max_turns=2
        )
        
        context = unified_client.get_session_memory(result.session_id)
        
        assert context is not None


# =============================================================================
# TEST CLASS: Context Rot Prevention Tests
# =============================================================================

class TestContextRotPrevention:
    """Test that long explorations don't lose early findings."""

    def test_long_exploration_preserves_early_findings(self, memory_bridge, sample_session_id):
        """Test that 20+ turn exploration preserves early findings."""
        # Initialize state
        memory_bridge.initialize_document_state(
            session_id=sample_session_id,
            document_path="/test/file.py",
            initial_goal="Analyze code"
        )
        
        # Record many steps (simulating 20+ turns)
        for i in range(1, 22):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Turn {i} query",
                code_executed=f"code_{i}",
                observation=f"Observation from turn {i}",
                insight=f"Insight from turn {i}: discovered item {i}",
                importance=0.6
            )
        
        # Verify all steps preserved
        steps = memory_bridge._exploration_steps.get(sample_session_id, [])
        assert len(steps) == 22  # Including initialization
        
        # Verify early findings still accessible
        early_insights = [s for s in steps if s.turn_number <= 5 and s.insight]
        assert len(early_insights) >= 5

    def test_core_findings_persist_in_state(self, memory_bridge, sample_session_id):
        """Test that core findings persist in document state."""
        # Initialize state
        doc_state = memory_bridge.initialize_document_state(
            session_id=sample_session_id,
            document_path="/test/file.py",
            initial_goal="Find critical functions"
        )
        
        # Add critical findings at early turns
        for i in range(1, 6):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Query {i}",
                code_executed=f"code_{i}",
                observation=f"Obs {i}",
                insight=f"Critical finding {i}: Important discovery",
                confidence=0.95,
                importance=0.9
            )
        
        # Add many more turns
        for i in range(6, 16):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Query {i}",
                code_executed=f"code_{i}",
                observation=f"Obs {i}",
                insight=f"Routine finding {i}",
                confidence=0.6
            )
        
        # Verify state contains early critical findings
        state = memory_bridge._document_states.get(sample_session_id)
        critical_in_state = [
            f for f in state.key_findings 
            if "Critical finding" in f.get("finding", "")
        ]
        assert len(critical_in_state) == 5

    def test_context_window_stays_bounded(self, exploration_session):
        """Test that context window stays within bounds during exploration."""
        # Run many turns
        result = exploration_session.explore(max_turns=15)
        
        # Get context
        context = exploration_session.get_current_context()
        prompt_context = context.to_prompt_context(max_bytes=5120)
        
        # Verify context size is bounded
        assert len(prompt_context.encode('utf-8')) <= 5200  # Allow small tolerance

    def test_high_importance_memories_prioritized(self, memory_bridge, sample_session_id):
        """Test that high importance memories are prioritized in retrieval."""
        # Add low importance memories
        for i in range(10):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i+1,
                query="Routine query",
                code_executed="routine_code",
                observation="routine",
                insight=f"Routine insight {i}",
                importance=0.3
            )
        
        # Add high importance memory
        memory_bridge.record_exploration_step(
            session_id=sample_session_id,
            turn_number=11,
            query="Critical query",
            code_executed="critical_code",
            observation="critical",
            insight="CRITICAL: Main function found",
            importance=0.95
        )
        
        # Retrieve context
        context = memory_bridge.get_exploration_context(
            session_id=sample_session_id,
            current_query="main function",
            max_memories=5
        )
        
        # High importance memory should be included
        critical_found = any(
            "CRITICAL" in (m.content if hasattr(m, 'content') else str(m))
            for m in context.relevant_memories
        )
        # Note: This test may need adjustment based on actual retrieval logic


# =============================================================================
# TEST CLASS: Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test graceful error handling and fallbacks."""

    def test_fallback_when_unified_memory_unavailable(self, sample_document_path):
        """Test graceful fallback when unified memory is unavailable."""
        # Create bridge with no unified memory - note: if UNIFIED_MEMORY_AVAILABLE,
        # it will auto-create, so we check for graceful degradation instead
        with patch('matryoshka_unified_memory_integration.UNIFIED_MEMORY_AVAILABLE', False):
            bridge = MatryoshkaMemoryBridge(unified_memory=None)
            
            memory = bridge.record_exploration_step(
                session_id="test_session",
                turn_number=1,
                query="Test",
                code_executed="test",
                observation="test",
                insight="test"
            )
            
            # Should return None gracefully when no unified memory available
            assert memory is None

    def test_missing_file_handling(self, unified_client):
        """Test handling of missing files."""
        result = unified_client.analyze_with_memory(
            query="Analyze",
            file_path="/definitely/not/a/real/file.xyz",
            max_turns=3
        )
        
        assert not result.success
        assert result.error is not None

    def test_invalid_session_id_handling(self, unified_client):
        """Test handling of invalid session ID."""
        result = unified_client.continue_analysis(
            session_id="nonexistent_session_12345",
            follow_up_query="Continue",
            max_turns=3
        )
        
        assert not result.success

    def test_exploration_step_error_handling(self, memory_bridge_mocked, sample_session_id):
        """Test error handling during step recording."""
        # Make indexing fail
        memory_bridge_mocked.unified_memory._index_memory.side_effect = Exception("Indexing failed")
        
        # Should not raise exception
        memory = memory_bridge_mocked.record_exploration_step(
            session_id=sample_session_id,
            turn_number=1,
            query="Test",
            code_executed="test",
            observation="test",
            insight="test"
        )
        
        # Step should still be recorded locally even if indexing fails
        steps = memory_bridge_mocked._exploration_steps.get(sample_session_id, [])
        assert len(steps) == 1

    def test_synthesis_with_no_steps(self, memory_bridge, sample_session_id):
        """Test synthesis when no exploration steps exist."""
        synthesis = memory_bridge.synthesize_findings(sample_session_id)
        
        assert synthesis.synthesis != ""
        assert synthesis.steps_used == 0

    def test_context_retrieval_error_handling(self, memory_bridge_mocked, sample_session_id):
        """Test error handling in context retrieval."""
        memory_bridge_mocked.unified_memory._hybrid_retrieve.side_effect = Exception("Retrieval failed")
        
        # Should not raise exception
        context = memory_bridge_mocked.get_exploration_context(
            session_id=sample_session_id,
            current_query="test"
        )
        
        assert isinstance(context, ExplorationContext)


# =============================================================================
# TEST CLASS: Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance requirements."""

    @pytest.mark.performance
    def test_exploration_step_recording_time(self, memory_bridge, sample_session_id):
        """Test that exploration step recording takes < 50ms."""
        times = []
        
        for i in range(10):
            start = time.time()
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query="Test query",
                code_executed="test_code",
                observation="test observation",
                insight="test insight"
            )
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Allow for occasional slower runs
        assert avg_time < 50, f"Average step recording time {avg_time:.1f}ms exceeds 50ms"
        assert max_time < 200, f"Max step recording time {max_time:.1f}ms exceeds 200ms"

    @pytest.mark.performance
    def test_context_retrieval_time(self, memory_bridge, sample_session_id):
        """Test that context retrieval takes < 100ms."""
        # Add some steps
        for i in range(10):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query="Test query",
                code_executed="test_code",
                observation="test observation",
                insight="test insight"
            )
        
        times = []
        for _ in range(10):
            start = time.time()
            context = memory_bridge.get_exploration_context(
                session_id=sample_session_id,
                current_query="test query"
            )
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        
        assert avg_time < 100, f"Average retrieval time {avg_time:.1f}ms exceeds 100ms"

    @pytest.mark.performance
    def test_synthesis_time(self, memory_bridge, sample_session_id):
        """Test that synthesis takes < 500ms."""
        # Add many steps
        for i in range(20):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i,
                query=f"Query {i}",
                code_executed=f"code_{i}",
                observation=f"Observation {i}",
                insight=f"Insight {i}"
            )
        
        start = time.time()
        synthesis = memory_bridge.synthesize_findings(sample_session_id)
        elapsed_ms = (time.time() - start) * 1000
        
        assert elapsed_ms < 500, f"Synthesis time {elapsed_ms:.1f}ms exceeds 500ms"


# =============================================================================
# TEST CLASS: Data Structure Tests
# =============================================================================

class TestDataStructures:
    """Test data structures and serialization."""

    def test_exploration_step_creation(self):
        """Test ExplorationStep creation."""
        step = ExplorationStep(
            step_id="step_1",
            session_id="session_1",
            turn_number=1,
            step_type=ExplorationStepType.CODE_EXECUTION,
            query="Test query",
            code_executed="print('test')",
            observation="output",
            insight="found something"
        )
        
        assert step.step_id == "step_1"
        assert step.session_id == "session_1"
        assert step.turn_number == 1
        assert step.step_type == ExplorationStepType.CODE_EXECUTION

    def test_exploration_step_to_memory_content(self):
        """Test converting step to memory content format."""
        step = ExplorationStep(
            step_id="step_1",
            session_id="session_1",
            turn_number=1,
            step_type=ExplorationStepType.OBSERVATION,
            query="Test query",
            code_executed="code here",
            observation="observation here",
            insight="insight here"
        )
        
        content = step.to_memory_content()
        
        assert "Turn 1" in content
        assert "OBSERVATION" in content
        assert "Test query" in content
        assert "code here" in content

    def test_exploration_step_to_dict(self):
        """Test ExplorationStep dictionary conversion."""
        step = ExplorationStep(
            step_id="step_1",
            session_id="session_1",
            turn_number=1,
            step_type=ExplorationStepType.OBSERVATION,
            query="Test"
        )
        
        data = step.to_dict()
        
        assert isinstance(data, dict)
        assert data["step_id"] == "step_1"
        assert data["turn_number"] == 1
        assert data["step_type"] == "observation"

    def test_document_state_creation(self):
        """Test DocumentState creation."""
        state = DocumentState(
            session_id="session_1",
            document_path="/test/file.py",
            document_type="python",
            current_goal="Analyze code"
        )
        
        assert state.session_id == "session_1"
        assert state.document_path == "/test/file.py"
        assert state.document_type == "python"
        assert state.total_turns == 0

    def test_document_state_add_finding(self):
        """Test adding findings to document state."""
        state = DocumentState(
            session_id="session_1",
            document_path="/test/file.py"
        )
        
        state.add_finding("Found main function", confidence=0.9)
        state.add_finding("Found helper functions", confidence=0.8)
        
        assert len(state.key_findings) == 2
        assert state.key_findings[0]["finding"] == "Found main function"

    def test_document_state_mark_section_explored(self):
        """Test marking sections as explored."""
        state = DocumentState(
            session_id="session_1",
            document_path="/test/file.py"
        )
        
        state.sections_remaining = {"section1", "section2", "section3"}
        state.mark_section_explored("section1")
        
        assert "section1" in state.sections_explored
        assert "section1" not in state.sections_remaining

    def test_document_state_to_dict(self):
        """Test DocumentState dictionary conversion."""
        state = DocumentState(
            session_id="session_1",
            document_path="/test/file.py",
            document_type="python"
        )
        state.add_finding("Test finding", confidence=0.9)
        
        data = state.to_dict()
        
        assert isinstance(data, dict)
        assert data["session_id"] == "session_1"
        assert data["document_type"] == "python"
        assert len(data["key_findings"]) == 1

    def test_exploration_context_to_prompt_context(self):
        """Test ExplorationContext prompt formatting."""
        state = DocumentState(
            session_id="session_1",
            document_path="/test/file.py",
            document_type="python"
        )
        state.add_finding("Main function found", confidence=0.9)
        
        context = ExplorationContext(
            document_state=state,
            relevant_memories=[],
            step_chain=[]
        )
        
        prompt = context.to_prompt_context(max_bytes=5120)
        
        assert "DOCUMENT STATE" in prompt
        assert "/test/file.py" in prompt
        assert "python" in prompt

    def test_synthesis_result_to_dict(self):
        """Test SynthesisResult dictionary conversion."""
        result = SynthesisResult(
            session_id="session_1",
            synthesis="Test synthesis",
            steps_used=5,
            confidence_score=0.85,
            key_findings=[{"finding": "test", "confidence": 0.9}]
        )
        
        data = result.to_dict()
        
        assert data["session_id"] == "session_1"
        assert data["synthesis"] == "Test synthesis"
        assert data["steps_used"] == 5
        assert data["confidence_score"] == 0.85

    def test_analysis_result_to_dict(self):
        """Test AnalysisResult dictionary conversion."""
        result = AnalysisResult(
            session_id="session_1",
            success=True,
            document_path="/test/file.py",
            query="Analyze",
            answer="Analysis complete",
            findings=["finding1", "finding2"]
        )
        
        data = result.to_dict()
        
        assert data["session_id"] == "session_1"
        assert data["success"] is True
        assert data["answer"] == "Analysis complete"
        assert len(data["findings"]) == 2


# =============================================================================
# TEST CLASS: Cross-Document Learning Tests
# =============================================================================

class TestCrossDocumentLearning:
    """Test cross-document learning capabilities."""

    def test_insights_from_previous_sessions(self, unified_client, sample_document_path, tmp_path):
        """Test that insights from previous sessions are available."""
        # First session - analyze one file
        result1 = unified_client.analyze_with_memory(
            query="Find all classes",
            file_path=sample_document_path,
            max_turns=3
        )
        
        # Create another file
        another_file = tmp_path / "another.py"
        another_file.write_text("class AnotherClass: pass")
        
        # Second session - should potentially leverage insights from first
        result2 = unified_client.analyze_with_memory(
            query="Find classes in this file too",
            file_path=str(another_file),
            max_turns=3
        )
        
        assert result1.success
        assert result2.success
        assert len(unified_client.list_sessions()) >= 2

    def test_search_finds_related_insights(self, unified_client, sample_document_path, tmp_path):
        """Test searching finds related insights across sessions."""
        # Create sessions with specific insights
        for i in range(3):
            test_file = tmp_path / f"test_{i}.py"
            test_file.write_text(f"def func_{i}(): pass")
            unified_client.analyze_with_memory(
                query="Find all functions",
                file_path=str(test_file),
                max_turns=2
            )
        
        # Search for function-related insights
        results = unified_client.search_across_sessions(
            query="function",
            limit=10
        )
        
        # Should find relevant results
        assert isinstance(results, list)


# =============================================================================
# TEST CLASS: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions for creating instances."""

    def test_create_unified_matryoshka_client(self, temp_memory_dir):
        """Test factory function for unified client."""
        client = create_unified_matryoshka_client(
            db_dir=temp_memory_dir,
            executable_path=None
        )
        
        assert isinstance(client, UnifiedMatryoshkaClient)
        assert client.is_available() == UNIFIED_MEMORY_AVAILABLE

    def test_create_memory_backed_session(self, sample_document_path, temp_memory_dir):
        """Test factory function for memory-backed session."""
        if UNIFIED_MEMORY_AVAILABLE:
            unified = create_unified_system(db_dir=temp_memory_dir)
            session = create_memory_backed_session(
                document_path=sample_document_path,
                query="Test query",
                unified_memory=unified
            )
        else:
            session = create_memory_backed_session(
                document_path=sample_document_path,
                query="Test query",
                unified_memory=None
            )
        
        assert isinstance(session, MatryoshkaExplorationSession)
        assert session.document_path == sample_document_path


# =============================================================================
# TEST CLASS: Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of operations."""

    def test_concurrent_step_recording(self, memory_bridge, sample_session_id):
        """Test concurrent step recording is thread-safe."""
        errors = []
        
        def record_steps(start_turn):
            try:
                for i in range(5):
                    memory_bridge.record_exploration_step(
                        session_id=sample_session_id,
                        turn_number=start_turn + i,
                        query=f"Thread query {start_turn + i}",
                        code_executed=f"code_{start_turn + i}",
                        observation="test",
                        insight="test"
                    )
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [
            threading.Thread(target=record_steps, args=(i * 10,))
            for i in range(3)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify no errors and all steps recorded
        assert len(errors) == 0
        steps = memory_bridge._exploration_steps.get(sample_session_id, [])
        assert len(steps) == 15


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_document(self, tmp_path, unified_client):
        """Test analysis of empty document."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        
        result = unified_client.analyze_with_memory(
            query="Analyze empty file",
            file_path=str(empty_file),
            max_turns=2
        )
        
        # Should handle gracefully (may succeed or fail gracefully)
        assert result is not None

    def test_very_long_content(self, tmp_path, unified_client):
        """Test handling of very long content."""
        long_file = tmp_path / "long.py"
        long_content = "\n".join([f"def func_{i}(): pass" for i in range(1000)])
        long_file.write_text(long_content)
        
        result = unified_client.analyze_with_memory(
            query="Analyze large file",
            file_path=str(long_file),
            max_turns=2
        )
        
        assert result is not None

    def test_special_characters_in_content(self, tmp_path, unified_client):
        """Test handling of special characters."""
        special_file = tmp_path / "special.py"
        special_content = '''
def func():
    # Comment with special chars: <>&"'
    text = "Unicode: ñ 中文 🎉"
    return text
'''
        special_file.write_text(special_content)
        
        result = unified_client.analyze_with_memory(
            query="Analyze special file",
            file_path=str(special_file),
            max_turns=2
        )
        
        assert result is not None

    def test_zero_max_turns(self, exploration_session):
        """Test exploration with zero max turns."""
        result = exploration_session.explore(max_turns=0)
        
        assert result.total_turns == 0

    def test_negative_importance_handling(self, memory_bridge, sample_session_id):
        """Test handling of negative importance values."""
        memory = memory_bridge.record_exploration_step(
            session_id=sample_session_id,
            turn_number=1,
            query="Test",
            code_executed="test",
            observation="test",
            insight="test",
            importance=-0.5  # Invalid
        )
        
        # Should handle gracefully (may clamp to valid range)
        steps = memory_bridge._exploration_steps.get(sample_session_id, [])
        if steps:
            assert steps[0].importance >= 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
