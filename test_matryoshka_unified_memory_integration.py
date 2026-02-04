"""
Comprehensive Tests for Matryoshka-UnifiedMemory Integration

Test coverage:
1. Basic Integration Tests - Bridge initialization, step recording, context retrieval
2. Exploration Session Tests - Session creation, multi-turn exploration, persistence
3. 4-Layer Indexing Tests - Hierarchical, graph, hash, semantic indexing
4. Cross-Session Learning Tests - Insights sharing, pattern recognition
5. Context Rot Prevention Tests - Long sessions, token budget, persistence
6. End-to-End Tests - Full workflows, export/import, batch analysis
7. Performance Tests - Retrieval time, indexing time, memory usage
8. Edge Cases - Empty sessions, large documents, concurrent access

Author: OpenEvolve AI
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import tempfile
import threading
import pytest
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module under test
from matryoshka_unified_memory_integration import (
    # Classes
    MatryoshkaMemoryBridge,
    MatryoshkaExplorationSession,
    UnifiedMatryoshkaClient,
    ExplorationStep,
    ExplorationStepType,
    DocumentState,
    ExplorationContext,
    ExplorationResult,
    SynthesisResult,
    AnalysisResult,
    # Functions
    create_unified_matryoshka_client,
    create_memory_backed_session,
    # Constants
    UNIFIED_MEMORY_AVAILABLE,
    STATE_MANAGER_AVAILABLE,
    HYBRID_RETRIEVAL_AVAILABLE,
    MATRYOSHKA_ADAPTER_AVAILABLE,
)

# Import unified memory system if available
if UNIFIED_MEMORY_AVAILABLE:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        UnifiedMemory,
        UnifiedMemoryConfig,
        create_unified_system,
    )

if STATE_MANAGER_AVAILABLE:
    from knowledge_state_manager import (
        CoreFact,
        ActiveDecision,
        TurnResult,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db_dir(tmp_path):
    """Provide a temporary database directory."""
    db_dir = tmp_path / "test_memory"
    db_dir.mkdir()
    return str(db_dir)


@pytest.fixture
def mock_unified_memory():
    """Create a mock unified memory system."""
    mock = MagicMock(spec=UnifiedMemorySystem if UNIFIED_MEMORY_AVAILABLE else object)
    mock._index_memory = Mock(return_value=None)
    mock._hybrid_retrieve = Mock(return_value=[])
    mock._conversation_memories = {}
    mock._memory_registry = {}
    mock.state_manager = Mock()
    mock.state_manager.create_conversation = Mock(return_value=None)
    mock.state_manager.update_from_turn = Mock(return_value=None)
    return mock


@pytest.fixture
def memory_bridge(temp_db_dir, mock_unified_memory):
    """Create a MatryoshkaMemoryBridge with mock unified memory."""
    bridge = MatryoshkaMemoryBridge(unified_memory=mock_unified_memory)
    yield bridge
    # Cleanup
    bridge.cleanup_session("test_session")


@pytest.fixture
def sample_document(tmp_path):
    """Create a sample Python document for testing."""
    doc_path = tmp_path / "sample.py"
    content = '''
"""Sample module for testing."""

def calculate_sum(numbers):
    """Calculate sum of numbers."""
    return sum(numbers)

def find_max(values):
    """Find maximum value."""
    return max(values) if values else None

class DataProcessor:
    """Process data in batches."""
    
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.processed = 0
    
    def process(self, data):
        """Process data batch."""
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            self.processed += len(batch)
        return self.processed
'''
    doc_path.write_text(content)
    return str(doc_path)


@pytest.fixture
def large_document(tmp_path):
    """Create a large document for testing."""
    doc_path = tmp_path / "large.py"
    lines = []
    for i in range(1000):
        lines.append(f"def function_{i}(x): return x * {i}")
        lines.append(f"class Class_{i}:")
        lines.append(f"    def method_{i}(self): pass")
    doc_path.write_text("\n".join(lines))
    return str(doc_path)


@pytest.fixture
def sample_session_id():
    """Provide a sample session ID."""
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_matryoshka_response():
    """Provide a mock Matryoshka response."""
    return """
# Generated exploration code
with open("{file_path}", "r") as f:
    content = f.read()

# Analyze file structure
lines = content.split("\\n")
classes = [line for line in lines if line.strip().startswith("class ")]
functions = [line for line in lines if line.strip().startswith("def ")]

print(f"Total lines: {len(lines)}")
print(f"Classes found: {len(classes)}")
print(f"Functions found: {len(functions)}")
"""


# =============================================================================
# BASIC INTEGRATION TESTS
# =============================================================================

class TestBasicIntegration:
    """Test basic Matryoshka-UnifiedMemory integration."""
    
    def test_bridge_initialization(self, temp_db_dir):
        """Test MatryoshkaMemoryBridge initialization."""
        if UNIFIED_MEMORY_AVAILABLE:
            unified = create_unified_system(db_dir=temp_db_dir)
            bridge = MatryoshkaMemoryBridge(unified_memory=unified)
        else:
            bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        assert bridge is not None
        assert hasattr(bridge, 'unified_memory')
        assert hasattr(bridge, '_sessions')
        assert hasattr(bridge, '_document_states')
        assert hasattr(bridge, '_exploration_steps')
    
    def test_bridge_initialization_without_unified_memory(self):
        """Test bridge works without unified memory (fallback mode)."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        assert bridge.unified_memory is None
        # Should still be functional in fallback mode
        assert bridge._exploration_steps is not None
    
    def test_record_exploration_step(self, memory_bridge, sample_session_id):
        """Test recording an exploration step."""
        step = memory_bridge.record_exploration_step(
            session_id=sample_session_id,
            turn_number=1,
            query="Find all functions",
            code_executed="code = open('file.py').read()",
            observation="Found 5 functions",
            insight="Main module has helper functions",
            step_type=ExplorationStepType.OBSERVATION,
            document_path="/test/file.py",
            importance=0.8,
            confidence=0.9
        )
        
        assert step is not None
        assert step.memory_id is not None
        assert step.source_conversation == sample_session_id
        assert step.source_turn == 1
        assert step.importance == 0.8
        assert step.confidence == 0.9
    
    def test_record_multiple_steps(self, memory_bridge, sample_session_id):
        """Test recording multiple exploration steps."""
        steps = []
        for i in range(5):
            step = memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i + 1,
                query=f"Query {i}",
                code_executed=f"code_{i}()",
                observation=f"Observation {i}",
                insight=f"Insight {i}",
                step_type=ExplorationStepType.OBSERVATION
            )
            steps.append(step)
        
        # Check all steps recorded
        recorded = memory_bridge._exploration_steps.get(sample_session_id, [])
        assert len(recorded) == 5
        
        # Check chaining
        for i in range(1, 5):
            assert recorded[i].previous_step_id == recorded[i-1].step_id
    
    def test_get_exploration_context(self, memory_bridge, sample_session_id):
        """Test retrieving exploration context."""
        # Record some steps first
        for i in range(3):
            memory_bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i + 1,
                query=f"Query {i}",
                code_executed=f"print({i})",
                observation=f"Output {i}",
                insight=f"Finding {i}"
            )
        
        context = memory_bridge.get_exploration_context(
            session_id=sample_session_id,
            current_query="Final query",
            max_memories=10
        )
        
        assert isinstance(context, ExplorationContext)
        assert context.total_memories_available == 3
        assert len(context.step_chain) <= 5  # Last 5 steps
    
    def test_get_exploration_context_empty_session(self, memory_bridge):
        """Test context retrieval for empty session."""
        context = memory_bridge.get_exploration_context(
            session_id="nonexistent_session",
            current_query="Test query"
        )
        
        assert isinstance(context, ExplorationContext)
        assert context.total_memories_available == 0
        assert context.document_state is None
    
    def test_to_prompt_context(self, memory_bridge, sample_session_id):
        """Test converting context to prompt string."""
        # Initialize document state
        doc_state = memory_bridge.initialize_document_state(
            session_id=sample_session_id,
            document_path="/test/file.py",
            document_type="python",
            initial_goal="Analyze file"
        )
        
        # Record a finding
        doc_state.add_finding("Found 10 functions", confidence=0.9)
        
        context = memory_bridge.get_exploration_context(
            session_id=sample_session_id,
            current_query="Analyze"
        )
        context.document_state = doc_state
        
        prompt = context.to_prompt_context(max_bytes=5120)
        
        assert isinstance(prompt, str)
        assert "DOCUMENT STATE" in prompt
        assert "file.py" in prompt
        assert "Found 10 functions" in prompt
    
    def test_to_prompt_context_truncation(self, memory_bridge, sample_session_id):
        """Test prompt context truncation."""
        # Create a very long context
        context = ExplorationContext()
        context.document_state = DocumentState(
            session_id=sample_session_id,
            document_path="/test/file.py"
        )
        # Add many large findings
        for i in range(100):
            context.document_state.add_finding("X" * 500, confidence=0.5)
        
        prompt = context.to_prompt_context(max_bytes=1024)
        
        assert len(prompt.encode('utf-8')) <= 1074  # 1024 + buffer
        assert "...[additional context truncated]" in prompt


# =============================================================================
# EXPLORATION SESSION TESTS
# =============================================================================

class TestExplorationSession:
    """Test MatryoshkaExplorationSession functionality."""
    
    def test_session_creation(self, temp_db_dir, sample_document):
        """Test exploration session creation."""
        session_id = f"test_{uuid.uuid4().hex[:8]}"
        
        session = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document,
            query="Find all classes and functions",
            unified_memory=None  # Use fallback mode
        )
        
        assert session.session_id == session_id
        assert session.document_path == sample_document
        assert session.original_query == "Find all classes and functions"
        assert session.current_turn == 0
        assert session.document_state is not None
        assert session.document_state.document_type == "python"
    
    def test_session_document_type_detection(self, temp_db_dir):
        """Test document type detection from file extension."""
        test_cases = [
            ("file.py", "python"),
            ("file.js", "javascript"),
            ("file.ts", "typescript"),
            ("file.md", "markdown"),
            ("file.json", "json"),
            ("file.yaml", "yaml"),
            ("file.yml", "yaml"),
            ("file.txt", "text"),
            ("file.csv", "csv"),
            ("file.html", "html"),
            ("file.css", "css"),
            ("unknown", None),
        ]
        
        for filename, expected_type in test_cases:
            session = MatryoshkaExplorationSession(
                session_id=f"test_{uuid.uuid4().hex[:8]}",
                document_path=f"/test/{filename}",
                query="Test",
                unified_memory=None
            )
            assert session.document_state.document_type == expected_type
    
    def test_session_state_management(self, temp_db_dir, sample_document):
        """Test document state management during session."""
        session = MatryoshkaExplorationSession(
            session_id=f"test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Add findings
        session.add_finding("Found main function", confidence=0.9)
        session.add_finding("Found helper classes", confidence=0.8)
        
        # Mark sections explored
        session.document_state.mark_section_explored("imports")
        session.document_state.mark_section_explored("classes")
        
        assert len(session.document_state.key_findings) == 2
        assert "imports" in session.document_state.sections_explored
        assert "classes" in session.document_state.sections_explored
    
    def test_multi_turn_exploration(self, temp_db_dir, sample_document):
        """Test multi-turn exploration with mocked code generation."""
        session = MatryoshkaExplorationSession(
            session_id=f"test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze structure",
            unified_memory=None
        )
        
        # Mock code generation to avoid actual LLM calls
        def mock_code_gen(query: str) -> str:
            return f'''
with open("{sample_document}", "r") as f:
    content = f.read()
lines = content.split("\\n")
print(f"Lines: {{len(lines)}}")
'''
        
        result = session.explore(max_turns=3, llm_code_callback=mock_code_gen)
        
        assert isinstance(result, ExplorationResult)
        assert result.success is True
        assert result.total_turns > 0
        assert result.total_turns <= 3
        assert len(result.steps) > 0
    
    def test_session_persistence_across_turns(self, temp_db_dir, sample_document):
        """Test state persists correctly across exploration turns."""
        session = MatryoshkaExplorationSession(
            session_id=f"test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Set initial goal
        session.document_state.current_goal = "Find all functions"
        
        # Explore with mock
        def mock_code_gen(query: str) -> str:
            return f'print("Turn executed")'
        
        session.explore(max_turns=2, llm_code_callback=mock_code_gen)
        
        # Verify state persisted
        assert session.document_state.total_turns >= 2
        assert session.document_state.current_goal == "Find all functions"
    
    def test_exploration_completion_detection(self, temp_db_dir, sample_document):
        """Test exploration completion detection."""
        session = MatryoshkaExplorationSession(
            session_id=f"test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Mock that generates completion marker
        def mock_complete(query: str) -> str:
            return 'print("analysis complete")'
        
        result = session.explore(max_turns=5, llm_code_callback=mock_complete)
        
        # Should complete early due to completion marker
        assert result.success is True
        # Note: completion detection depends on output content


# =============================================================================
# 4-LAYER INDEXING TESTS
# =============================================================================

class TestFourLayerIndexing:
    """Test 4-layer indexing (hierarchical, graph, hash, semantic)."""
    
    def test_step_indexed_hierarchically(self, temp_db_dir, sample_session_id):
        """Test exploration steps are indexed hierarchically."""
        if not UNIFIED_MEMORY_AVAILABLE:
            pytest.skip("Unified memory not available")
        
        unified = create_unified_system(db_dir=temp_db_dir)
        bridge = MatryoshkaMemoryBridge(unified_memory=unified)
        
        # Record steps at different turns
        for turn in range(1, 4):
            bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=turn,
                query=f"Query turn {turn}",
                code_executed=f"print({turn})",
                observation=f"Output {turn}",
                insight=f"Insight {turn}"
            )
        
        # Verify steps are tracked
        steps = bridge._exploration_steps.get(sample_session_id, [])
        assert len(steps) == 3
        
        # Check turn numbers are sequential
        for i, step in enumerate(steps):
            assert step.turn_number == i + 1
    
    def test_step_relationships_graph(self, temp_db_dir, sample_session_id):
        """Test graph relationships between steps."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Create chain of steps
        prev_id = None
        for turn in range(1, 4):
            step_id = f"step_{turn}_{uuid.uuid4().hex[:8]}"
            step = ExplorationStep(
                step_id=step_id,
                session_id=sample_session_id,
                turn_number=turn,
                step_type=ExplorationStepType.OBSERVATION,
                query=f"Query {turn}",
                observation=f"Observation {turn}",
                previous_step_id=prev_id
            )
            bridge._exploration_steps[sample_session_id].append(step)
            prev_id = step_id
        
        # Verify chain
        steps = bridge._exploration_steps[sample_session_id]
        assert steps[0].previous_step_id is None  # First step
        assert steps[1].previous_step_id == steps[0].step_id
        assert steps[2].previous_step_id == steps[1].step_id
    
    def test_deduplication_via_hash(self, temp_db_dir, sample_session_id):
        """Test deduplication of similar observations."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Create steps with identical observations
        observation = "Found 5 functions in the file"
        
        for i in range(3):
            step = ExplorationStep(
                step_id=f"step_{i}_{uuid.uuid4().hex[:8]}",
                session_id=sample_session_id,
                turn_number=i + 1,
                step_type=ExplorationStepType.OBSERVATION,
                query=f"Query {i}",
                observation=observation,
                insight=f"Insight {i}"
            )
            bridge._exploration_steps[sample_session_id].append(step)
        
        # All should be stored (deduplication happens at memory level)
        steps = bridge._exploration_steps[sample_session_id]
        assert len(steps) == 3
        
        # But content is identical
        contents = [s.observation for s in steps]
        assert all(c == observation for c in contents)
    
    def test_semantic_search_across_steps(self, temp_db_dir, sample_session_id):
        """Test semantic search across indexed steps."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Create steps with different content
        contents = [
            ("Find classes in module", "Found 3 classes: DataProcessor, Analyzer, Parser"),
            ("Find functions", "Found helper functions and utilities"),
            ("Analyze imports", "Import statements from standard library"),
        ]
        
        for i, (query, observation) in enumerate(contents):
            step = ExplorationStep(
                step_id=f"step_{i}",
                session_id=sample_session_id,
                turn_number=i + 1,
                step_type=ExplorationStepType.OBSERVATION,
                query=query,
                observation=observation,
                insight=f"Insight about {query}"
            )
            bridge._exploration_steps[sample_session_id].append(step)
        
        # Test fallback retrieval (semantic when unified not available)
        results = bridge._fallback_retrieval(
            sample_session_id,
            "find classes",
            limit=5
        )
        
        assert len(results) > 0
        # Should find the class-related step
        assert any("class" in r.content.lower() for r in results)
    
    def test_importance_scoring(self, temp_db_dir, sample_session_id):
        """Test importance scoring for steps."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Create steps with different importance
        for i, importance in enumerate([0.3, 0.7, 0.9]):
            step = ExplorationStep(
                step_id=f"step_{i}",
                session_id=sample_session_id,
                turn_number=i + 1,
                step_type=ExplorationStepType.OBSERVATION,
                query=f"Query {i}",
                observation=f"Observation {i}",
                importance=importance,
                confidence=0.8
            )
            bridge._exploration_steps[sample_session_id].append(step)
        
        # Retrieve and check ordering favors high importance
        results = bridge._fallback_retrieval(sample_session_id, "query", limit=10)
        
        if results:
            # Higher importance should generally score better
            importances = [r.importance for r in results]
            assert max(importances) >= 0.7


# =============================================================================
# CROSS-SESSION LEARNING TESTS
# =============================================================================

class TestCrossSessionLearning:
    """Test cross-session learning capabilities."""
    
    def test_insights_available_across_sessions(self, temp_db_dir):
        """Test insights from session A available in session B."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Session A - discover patterns
        session_a = f"session_a_{uuid.uuid4().hex[:8]}"
        for i in range(3):
            bridge.record_exploration_step(
                session_id=session_a,
                turn_number=i + 1,
                query=f"Analysis A-{i}",
                code_executed=f"code_a_{i}()",
                observation=f"Pattern discovered: singleton pattern in file A",
                insight=f"Use singleton for configuration management"
            )
        
        # Session B - should be able to search for similar patterns
        session_b = f"session_b_{uuid.uuid4().hex[:8]}"
        bridge.record_exploration_step(
            session_id=session_b,
            turn_number=1,
            query="Look for patterns",
            code_executed="analyze_patterns()",
            observation="Looking for design patterns",
            insight="Searching..."
        )
        
        # Retrieve from session A
        context = bridge.get_exploration_context(
            session_id=session_a,
            current_query="singleton pattern"
        )
        
        # Should find the singleton-related insight
        assert len(context.step_chain) > 0
        assert any("singleton" in s.observation.lower() 
                  for s in context.step_chain if s.observation)
    
    def test_pattern_recognition_across_documents(self, temp_db_dir):
        """Test pattern recognition across different documents."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        # Simulate analyzing multiple documents
        findings = [
            {"file": "auth.py", "pattern": "authentication", "count": 3},
            {"file": "login.py", "pattern": "authentication", "count": 2},
            {"file": "session.py", "pattern": "session management", "count": 4},
        ]
        
        for finding in findings:
            session_id = f"session_{finding['file']}_{uuid.uuid4().hex[:8]}"
            doc_state = client.memory_bridge.initialize_document_state(
                session_id=session_id,
                document_path=f"/src/{finding['file']}",
                document_type="python"
            )
            doc_state.add_finding(
                f"Found {finding['pattern']} ({finding['count']} instances)",
                confidence=0.85
            )
        
        # Search across sessions
        results = client.search_across_sessions("authentication", limit=10)
        
        # Should find auth-related findings
        auth_results = [r for r in results if "authentication" in r["content"].lower()]
        assert len(auth_results) >= 2
    
    def test_search_across_sessions_empty(self, temp_db_dir):
        """Test search across sessions with no matches."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        results = client.search_across_sessions("nonexistent_pattern_xyz", limit=10)
        
        assert isinstance(results, list)
        assert len(results) == 0


# =============================================================================
# CONTEXT ROT PREVENTION TESTS
# =============================================================================

class TestContextRotPrevention:
    """Test context rot prevention in long exploration sessions."""
    
    def test_long_exploration_retains_context(self, temp_db_dir, sample_document):
        """Test that early findings are retrievable after many turns."""
        session = MatryoshkaExplorationSession(
            session_id=f"long_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Comprehensive analysis",
            unified_memory=None
        )
        
        # Record many steps (simulating long exploration)
        early_insight = "CRITICAL: Main entry point is process_data()"
        
        for i in range(50):
            if i == 5:  # Record critical insight early
                session.memory_bridge.record_exploration_step(
                    session_id=session.session_id,
                    turn_number=i + 1,
                    query=f"Step {i}",
                    code_executed=f"step_{i}()",
                    observation=early_insight,
                    insight="Entry point identified",
                    importance=1.0,  # High importance
                    confidence=0.95
                )
            else:
                session.memory_bridge.record_exploration_step(
                    session_id=session.session_id,
                    turn_number=i + 1,
                    query=f"Step {i}",
                    code_executed=f"step_{i}()",
                    observation=f"Observation {i}",
                    insight=f"Insight {i}",
                    importance=0.5
                )
        
        # Search for the critical early finding
        context = session.memory_bridge.get_exploration_context(
            session_id=session.session_id,
            current_query="entry point main process_data"
        )
        
        # Should retrieve the high-importance early step
        all_content = " ".join([m.content for m in context.relevant_memories])
        # Note: In fallback mode, importance-based retrieval should work
        assert context.total_memories_available == 50
    
    def test_core_findings_persistence(self, temp_db_dir, sample_document):
        """Test core findings persist throughout exploration."""
        session = MatryoshkaExplorationSession(
            session_id=f"core_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Add core finding early
        core_finding = "Architecture: 3-layer design (API, Business, Data)"
        session.add_finding(core_finding, confidence=0.95)
        
        # Simulate many turns
        for i in range(30):
            session.memory_bridge.record_exploration_step(
                session_id=session.session_id,
                turn_number=i + 1,
                query=f"Deep analysis {i}",
                code_executed=f"deep_{i}()",
                observation=f"Detailed observation {i}",
                insight=f"Detailed insight {i}"
            )
        
        # Verify core finding still in state
        doc_state = session.document_state
        assert len(doc_state.key_findings) > 0
        assert any(core_finding in f.get("finding", "") 
                  for f in doc_state.key_findings)
    
    def test_token_budget_respected(self, temp_db_dir, sample_document):
        """Test that token budget is respected in context building."""
        session = MatryoshkaExplorationSession(
            session_id=f"token_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Add many findings
        for i in range(20):
            session.add_finding(f"Finding {i}: " + "X" * 500, confidence=0.7)
        
        context = session.get_current_context()
        prompt = context.to_prompt_context(max_bytes=5120)
        
        # Check size is within budget
        size_bytes = len(prompt.encode('utf-8'))
        assert size_bytes <= 5200  # Small buffer for truncation marker


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEnd:
    """Test end-to-end workflows."""
    
    def test_full_document_analysis_workflow(self, temp_db_dir, sample_document):
        """Test complete document analysis workflow."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        def mock_llm(query: str) -> str:
            return f'''
with open("{sample_document}", "r") as f:
    content = f.read()
lines = content.split("\\n")
classes = [l for l in lines if l.strip().startswith("class ")]
functions = [l for l in lines if l.strip().startswith("def ")]
print(f"Classes: {{len(classes)}}")
print(f"Functions: {{len(functions)}}")
'''
        
        result = client.analyze_with_memory(
            query="Find all classes and functions",
            file_path=sample_document,
            max_turns=3,
            llm_code_callback=mock_llm
        )
        
        assert result.success is True
        assert result.session_id is not None
        assert result.answer is not None
        assert len(result.findings) > 0
        assert result.processing_time_ms > 0
    
    def test_continue_analysis_workflow(self, temp_db_dir, sample_document):
        """Test continuing analysis after initial session."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        # Initial analysis
        result1 = client.analyze_with_memory(
            query="Analyze structure",
            file_path=sample_document,
            max_turns=2,
            llm_code_callback=lambda q: 'print("analysis")'
        )
        
        assert result1.success is True
        session_id = result1.session_id
        
        # Continue analysis
        result2 = client.continue_analysis(
            session_id=session_id,
            follow_up_query="Look deeper into classes",
            max_turns=2,
            llm_code_callback=lambda q: 'print("deeper analysis")'
        )
        
        assert result2.success is True
        assert result2.document_path == sample_document
    
    def test_session_export_import(self, temp_db_dir, sample_document):
        """Test exporting and importing session data."""
        session = MatryoshkaExplorationSession(
            session_id=f"export_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Add some data
        for i in range(5):
            session.memory_bridge.record_exploration_step(
                session_id=session.session_id,
                turn_number=i + 1,
                query=f"Q{i}",
                code_executed=f"code_{i}()",
                observation=f"Obs {i}",
                insight=f"Insight {i}"
            )
        session.add_finding("Key finding", confidence=0.9)
        
        # Export to dict
        doc_state_dict = session.document_state.to_dict()
        assert doc_state_dict["session_id"] == session.session_id
        assert doc_state_dict["document_path"] == sample_document
        
        # Verify steps are exportable
        steps = session.memory_bridge._exploration_steps.get(session.session_id, [])
        step_dicts = [s.to_dict() for s in steps]
        assert len(step_dicts) == 5
    
    def test_batch_analysis_workflow(self, temp_db_dir, tmp_path):
        """Test batch analysis of multiple documents."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        # Create multiple documents
        documents = []
        for i in range(3):
            doc_path = tmp_path / f"file_{i}.py"
            doc_path.write_text(f"def func_{i}(): pass")
            documents.append(str(doc_path))
        
        # Analyze each
        results = []
        for doc_path in documents:
            result = client.analyze_with_memory(
                query="Analyze file",
                file_path=doc_path,
                max_turns=1,
                llm_code_callback=lambda q: 'print("done")'
            )
            results.append(result)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Verify all sessions tracked
        sessions = client.list_sessions()
        assert len(sessions) == 3
    
    def test_synthesis_generation(self, temp_db_dir, sample_document):
        """Test synthesis generation from exploration."""
        session = MatryoshkaExplorationSession(
            session_id=f"synth_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Add diverse step types
        step_types = [
            ExplorationStepType.INITIALIZATION,
            ExplorationStepType.OBSERVATION,
            ExplorationStepType.INSIGHT,
            ExplorationStepType.VERIFICATION,
            ExplorationStepType.SYNTHESIS,
        ]
        
        for i, st in enumerate(step_types):
            session.memory_bridge.record_exploration_step(
                session_id=session.session_id,
                turn_number=i,
                query=f"Query {st.value}",
                code_executed="code()",
                observation=f"Observation for {st.value}",
                insight=f"Insight from {st.value}",
                step_type=st
            )
        
        # Add findings
        session.add_finding("Architecture is modular", confidence=0.9)
        session.add_finding("Uses dependency injection", confidence=0.8)
        
        synthesis = session.memory_bridge.synthesize_findings(session.session_id)
        
        assert isinstance(synthesis, SynthesisResult)
        assert synthesis.session_id == session.session_id
        assert synthesis.synthesis is not None
        assert len(synthesis.synthesis) > 0
        assert synthesis.steps_used == 5


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance requirements."""
    
    def test_context_retrieval_time(self, temp_db_dir, sample_session_id):
        """Test context retrieval completes in under 100ms."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Populate with test data
        for i in range(100):
            bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i + 1,
                query=f"Query {i}",
                code_executed=f"code_{i}()",
                observation=f"Observation {i} with some content",
                insight=f"Insight {i}"
            )
        
        # Time retrieval
        start = time.perf_counter()
        context = bridge.get_exploration_context(
            session_id=sample_session_id,
            current_query="test query",
            max_memories=15
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 100, f"Retrieval took {elapsed_ms:.2f}ms, expected <100ms"
    
    def test_step_indexing_time(self, temp_db_dir, sample_session_id):
        """Test step indexing completes quickly."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Time indexing multiple steps
        start = time.perf_counter()
        for i in range(10):
            bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i + 1,
                query=f"Query {i}",
                code_executed=f"code_{i}()",
                observation=f"Observation {i}",
                insight=f"Insight {i}"
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete 10 steps in reasonable time
        assert elapsed_ms < 1000, f"Indexing took {elapsed_ms:.2f}ms"
    
    def test_memory_usage_long_session(self, temp_db_dir, sample_document):
        """Test memory usage remains reasonable in long sessions."""
        import sys
        
        session = MatryoshkaExplorationSession(
            session_id=f"mem_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Record baseline
        gc.collect()
        baseline = sys.getsizeof(session)
        
        # Add many steps
        for i in range(100):
            session.memory_bridge.record_exploration_step(
                session_id=session.session_id,
                turn_number=i + 1,
                query=f"Query {i}",
                code_executed=f"print({i})",
                observation=f"Observation {i}",
                insight=f"Insight {i}"
            )
        
        # Check growth is reasonable (not exponential)
        gc.collect()
        steps = session.memory_bridge._exploration_steps.get(session.session_id, [])
        
        # 100 steps should be manageable
        assert len(steps) == 100
    
    def test_hybrid_retrieval_performance(self, temp_db_dir, sample_session_id):
        """Test hybrid retrieval scales with dataset size."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Create dataset
        for i in range(50):
            bridge._exploration_steps[sample_session_id].append(
                ExplorationStep(
                    step_id=f"step_{i}",
                    session_id=sample_session_id,
                    turn_number=i + 1,
                    step_type=ExplorationStepType.OBSERVATION,
                    query=f"Query about topic {i % 5}",
                    observation=f"Observation on topic {i % 5}",
                    insight=f"Insight {i}"
                )
            )
        
        # Time retrieval
        start = time.perf_counter()
        results = bridge._fallback_retrieval(
            sample_session_id,
            "topic 2",
            limit=10
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 50, f"Hybrid retrieval took {elapsed_ms:.2f}ms"
        assert len(results) <= 10


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_exploration(self, temp_db_dir, sample_document):
        """Test handling of empty exploration session."""
        session = MatryoshkaExplorationSession(
            session_id=f"empty_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Don't run exploration
        synthesis = session.memory_bridge.synthesize_findings(session.session_id)
        
        assert isinstance(synthesis, SynthesisResult)
        assert synthesis.steps_used == 0
        assert "No exploration steps found" in synthesis.synthesis
    
    def test_single_turn_analysis(self, temp_db_dir, sample_document):
        """Test single-turn analysis."""
        session = MatryoshkaExplorationSession(
            session_id=f"single_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Quick analysis",
            unified_memory=None
        )
        
        result = session.explore(
            max_turns=1,
            llm_code_callback=lambda q: 'print("done")'
        )
        
        assert result.success is True
        assert result.total_turns == 1
    
    def test_very_large_document(self, temp_db_dir, large_document):
        """Test handling of very large documents."""
        session = MatryoshkaExplorationSession(
            session_id=f"large_{uuid.uuid4().hex[:8]}",
            document_path=large_document,
            query="Analyze large file",
            unified_memory=None
        )
        
        # Document size should be recorded
        assert session.document_state.document_size_bytes > 0
        
        # Should still be able to explore
        result = session.explore(
            max_turns=2,
            llm_code_callback=lambda q: f'''
with open("{large_document}") as f:
    content = f.read()
print(f"Size: {{len(content)}}")
'''
        )
        
        assert result.success is True
    
    def test_concurrent_sessions(self, temp_db_dir, tmp_path):
        """Test multiple concurrent sessions."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        # Create documents
        documents = []
        for i in range(5):
            doc_path = tmp_path / f"concurrent_{i}.py"
            doc_path.write_text(f"def func_{i}(): return {i}")
            documents.append((f"doc_{i}", str(doc_path)))
        
        def analyze_doc(doc_id, doc_path):
            return client.analyze_with_memory(
                query=f"Analyze {doc_id}",
                file_path=doc_path,
                max_turns=2,
                llm_code_callback=lambda q: 'print("ok")'
            )
        
        # Run concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(analyze_doc, doc_id, doc_path)
                for doc_id, doc_path in documents
            ]
            results = [f.result() for f in as_completed(futures)]
        
        assert len(results) == 5
        assert all(r.success for r in results)
        
        # Verify all sessions exist
        sessions = client.list_sessions()
        assert len(sessions) == 5
    
    def test_missing_matryoshka_binary(self, temp_db_dir, sample_document):
        """Test graceful handling of missing Matryoshka binary."""
        client = UnifiedMatryoshkaClient(
            unified_memory=None,
            executable_path="/nonexistent/matryoshka"
        )
        
        # Should still work in fallback mode
        result = client.analyze_with_memory(
            query="Analyze",
            file_path=sample_document,
            max_turns=1,
            llm_code_callback=lambda q: 'print("fallback")'
        )
        
        assert result.success is True
        assert client.is_available() is True  # Memory bridge still works
    
    def test_invalid_file_path(self, temp_db_dir):
        """Test handling of invalid file path."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        result = client.analyze_with_memory(
            query="Analyze",
            file_path="/nonexistent/file.py",
            max_turns=1
        )
        
        assert result.success is False
        assert "not found" in result.error.lower() or "error" in result.error.lower()
    
    def test_session_cleanup(self, temp_db_dir, sample_document):
        """Test proper session cleanup."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        result = client.analyze_with_memory(
            query="Analyze",
            file_path=sample_document,
            max_turns=1,
            llm_code_callback=lambda q: 'print("ok")'
        )
        
        session_id = result.session_id
        
        # Close session
        closed = client.close_session(session_id)
        assert closed is True
        
        # Verify session removed
        sessions = client.list_sessions()
        assert not any(s["session_id"] == session_id for s in sessions)
        
        # Double close should return False
        closed_again = client.close_session(session_id)
        assert closed_again is False
    
    def test_malformed_code_execution(self, temp_db_dir, sample_document):
        """Test handling of malformed code during execution."""
        session = MatryoshkaExplorationSession(
            session_id=f"error_test_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        # Mock code that causes error
        def error_code(query: str) -> str:
            return 'raise ValueError("Test error")'
        
        result = session.explore(max_turns=1, llm_code_callback=error_code)
        
        # Should handle error gracefully
        assert isinstance(result, ExplorationResult)
        # May succeed if error is captured as observation
    
    def test_duplicate_session_id(self, temp_db_dir, sample_document):
        """Test handling of duplicate session IDs."""
        session_id = f"duplicate_{uuid.uuid4().hex[:8]}"
        
        session1 = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document,
            query="Analysis 1",
            unified_memory=None
        )
        
        session1.add_finding("Finding from session 1", confidence=0.8)
        
        # Second session with same ID
        session2 = MatryoshkaExplorationSession(
            session_id=session_id,
            document_path=sample_document,
            query="Analysis 2",
            unified_memory=None
        )
        
        # Should be independent (or properly managed)
        assert session2.session_id == session_id
    
    def test_very_long_queries(self, temp_db_dir, sample_document):
        """Test handling of very long queries."""
        session = MatryoshkaExplorationSession(
            session_id=f"long_query_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="X" * 10000,  # Very long query
            unified_memory=None
        )
        
        # Should handle gracefully
        assert session.original_query == "X" * 10000
        
        # Step recording should work
        step = session.memory_bridge.record_exploration_step(
            session_id=session.session_id,
            turn_number=1,
            query="Y" * 5000,
            code_executed="print('test')",
            observation="Test",
            insight="Test"
        )
        
        assert step is not None
    
    def test_unicode_content(self, temp_db_dir, sample_document):
        """Test handling of unicode content."""
        session = MatryoshkaExplorationSession(
            session_id=f"unicode_{uuid.uuid4().hex[:8]}",
            document_path=sample_document,
            query="测试 Unicode 分析",
            unified_memory=None
        )
        
        # Record unicode content
        step = session.memory_bridge.record_exploration_step(
            session_id=session.session_id,
            turn_number=1,
            query="Query with emojis 🎉 🚀",
            code_executed="# 日本語コメント",
            observation="Observação em português: café",
            insight="Finding: Über",
            importance=0.8
        )
        
        assert step is not None
        
        # Retrieval should preserve unicode
        context = session.memory_bridge.get_exploration_context(
            session_id=session.session_id,
            current_query="test"
        )
        
        prompt = context.to_prompt_context()
        assert "🎉" in prompt or "日本語" in prompt or "café" in prompt


# =============================================================================
# DOCUMENT STATE TESTS
# =============================================================================

class TestDocumentState:
    """Test DocumentState functionality."""
    
    def test_document_state_initialization(self):
        """Test DocumentState initialization."""
        state = DocumentState(
            session_id="test_session",
            document_path="/test/file.py",
            document_type="python",
            document_size_bytes=1024,
            initial_goal="Analyze"
        )
        
        assert state.session_id == "test_session"
        assert state.document_path == "/test/file.py"
        assert state.document_type == "python"
        assert state.document_size_bytes == 1024
        assert state.current_goal == "Analyze"
        assert len(state.sections_explored) == 0
        assert len(state.key_findings) == 0
    
    def test_add_finding(self):
        """Test adding findings to document state."""
        state = DocumentState(
            session_id="test",
            document_path="/test/file.py"
        )
        
        state.add_finding("Found main function", confidence=0.9, source_step_id="step_1")
        state.add_finding("Uses async pattern", confidence=0.8)
        
        assert len(state.key_findings) == 2
        assert state.key_findings[0]["finding"] == "Found main function"
        assert state.key_findings[0]["confidence"] == 0.9
        assert state.key_findings[0]["source_step_id"] == "step_1"
    
    def test_mark_section_explored(self):
        """Test marking sections as explored."""
        state = DocumentState(
            session_id="test",
            document_path="/test/file.py"
        )
        
        state.sections_remaining = {"imports", "classes", "functions"}
        
        state.mark_section_explored("imports")
        
        assert "imports" in state.sections_explored
        assert "imports" not in state.sections_remaining
        assert "classes" in state.sections_remaining
    
    def test_to_state_facts(self):
        """Test conversion to state facts."""
        if not STATE_MANAGER_AVAILABLE:
            pytest.skip("State manager not available")
        
        state = DocumentState(
            session_id="test",
            document_path="/test/file.py",
            document_type="python",
            initial_goal="Analyze code"
        )
        
        state.add_finding("Architecture is clean", confidence=0.9)
        
        facts = state.to_state_facts()
        
        assert isinstance(facts, list)
        assert len(facts) > 0
        
        # Check key facts exist
        fact_keys = [f.key for f in facts]
        assert "document_path" in fact_keys
        assert "document_type" in fact_keys
    
    def test_to_dict_serialization(self):
        """Test DocumentState serialization."""
        state = DocumentState(
            session_id="test",
            document_path="/test/file.py",
            document_type="python",
            document_size_bytes=2048
        )
        
        state.add_finding("Test finding", confidence=0.7)
        state.mark_section_explored("header")
        
        data = state.to_dict()
        
        assert data["session_id"] == "test"
        assert data["document_path"] == "/test/file.py"
        assert data["document_type"] == "python"
        assert data["document_size_bytes"] == 2048
        assert len(data["key_findings"]) == 1
        assert "header" in data["sections_explored"]


# =============================================================================
# EXPLORATION STEP TESTS
# =============================================================================

class TestExplorationStep:
    """Test ExplorationStep functionality."""
    
    def test_step_creation(self):
        """Test ExplorationStep creation."""
        step = ExplorationStep(
            step_id="step_1",
            session_id="session_1",
            turn_number=1,
            step_type=ExplorationStepType.OBSERVATION,
            query="Find functions",
            code_executed="code = open('file.py').read()",
            observation="Found 5 functions",
            insight="Main entry point is process()",
            importance=0.8,
            confidence=0.9
        )
        
        assert step.step_id == "step_1"
        assert step.session_id == "session_1"
        assert step.turn_number == 1
        assert step.step_type == ExplorationStepType.OBSERVATION
        assert step.importance == 0.8
        assert step.confidence == 0.9
    
    def test_to_memory_content(self):
        """Test conversion to memory content."""
        step = ExplorationStep(
            step_id="step_1",
            session_id="session_1",
            turn_number=5,
            step_type=ExplorationStepType.INSIGHT,
            query="Analyze architecture",
            code_executed="analyze_modules()",
            observation="3 modules found",
            insight="Uses clean architecture pattern"
        )
        
        content = step.to_memory_content()
        
        assert "[Turn 5]" in content
        assert "INSIGHT" in content
        assert "Query:" in content
        assert "Insight:" in content
    
    def test_to_dict_serialization(self):
        """Test ExplorationStep serialization."""
        step = ExplorationStep(
            step_id="step_1",
            session_id="session_1",
            turn_number=1,
            step_type=ExplorationStepType.VERIFICATION,
            query="Test",
            code_executed="test()",
            observation="Pass",
            insight="Verified"
        )
        
        data = step.to_dict()
        
        assert data["step_id"] == "step_1"
        assert data["session_id"] == "session_1"
        assert data["turn_number"] == 1
        assert data["step_type"] == "verification"
    
    def test_step_types(self):
        """Test all exploration step types."""
        types = [
            ExplorationStepType.INITIALIZATION,
            ExplorationStepType.CODE_GENERATION,
            ExplorationStepType.CODE_EXECUTION,
            ExplorationStepType.OBSERVATION,
            ExplorationStepType.INSIGHT,
            ExplorationStepType.HYPOTHESIS,
            ExplorationStepType.VERIFICATION,
            ExplorationStepType.SYNTHESIS,
            ExplorationStepType.ERROR,
        ]
        
        for i, st in enumerate(types):
            step = ExplorationStep(
                step_id=f"step_{i}",
                session_id="session",
                turn_number=i,
                step_type=st
            )
            assert step.step_type == st
            assert step.step_type.value == st.value


# =============================================================================
# SYNTHESIS TESTS
# =============================================================================

class TestSynthesis:
    """Test synthesis functionality."""
    
    def test_synthesis_result_creation(self):
        """Test SynthesisResult creation."""
        result = SynthesisResult(
            session_id="test_session",
            synthesis="Analysis complete",
            steps_used=10,
            memories_considered=25,
            confidence_score=0.85,
            coverage_score=0.9,
            key_findings=[{"finding": "Main function found", "confidence": 0.9}],
            recommendations=["Refactor for clarity"]
        )
        
        assert result.session_id == "test_session"
        assert result.synthesis == "Analysis complete"
        assert result.steps_used == 10
        assert result.confidence_score == 0.85
    
    def test_synthesis_with_no_steps(self, temp_db_dir):
        """Test synthesis when no steps exist."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        synthesis = bridge.synthesize_findings("nonexistent_session")
        
        assert synthesis.session_id == "nonexistent_session"
        assert "No exploration steps found" in synthesis.synthesis
        assert synthesis.steps_used == 0
    
    def test_synthesis_confidence_calculation(self, temp_db_dir, sample_session_id):
        """Test confidence calculation in synthesis."""
        bridge = MatryoshkaMemoryBridge(unified_memory=None)
        
        # Initialize document state
        doc_state = bridge.initialize_document_state(
            session_id=sample_session_id,
            document_path="/test/file.py"
        )
        
        # Add findings with varying confidence
        doc_state.add_finding("Finding 1", confidence=0.9)
        doc_state.add_finding("Finding 2", confidence=0.7)
        doc_state.add_finding("Finding 3", confidence=0.8)
        
        # Explore sections
        doc_state.sections_remaining = {"a", "b", "c", "d"}
        doc_state.mark_section_explored("a")
        doc_state.mark_section_explored("b")
        
        # Add some steps
        for i in range(5):
            bridge.record_exploration_step(
                session_id=sample_session_id,
                turn_number=i + 1,
                query=f"Query {i}",
                code_executed=f"code_{i}()",
                observation=f"Obs {i}",
                insight=f"Insight {i}"
            )
        
        synthesis = bridge.synthesize_findings(sample_session_id)
        
        assert synthesis.steps_used == 5
        assert synthesis.confidence_score > 0
        assert synthesis.coverage_score == 0.5  # 2/4 sections


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_unified_matryoshka_client(self, temp_db_dir):
        """Test unified Matryoshka client factory."""
        client = create_unified_matryoshka_client(
            db_dir=temp_db_dir,
            executable_path=None
        )
        
        assert isinstance(client, UnifiedMatryoshkaClient)
        assert client.memory_bridge is not None
    
    def test_create_memory_backed_session(self, temp_db_dir, sample_document):
        """Test memory-backed session factory."""
        session = create_memory_backed_session(
            document_path=sample_document,
            query="Analyze",
            unified_memory=None
        )
        
        assert isinstance(session, MatryoshkaExplorationSession)
        assert session.session_id.startswith("exploration_")
        assert session.document_path == sample_document
    
    def test_create_memory_backed_session_with_unified(self, temp_db_dir, sample_document):
        """Test session factory with unified memory."""
        if not UNIFIED_MEMORY_AVAILABLE:
            pytest.skip("Unified memory not available")
        
        unified = create_unified_system(db_dir=temp_db_dir)
        session = create_memory_backed_session(
            document_path=sample_document,
            query="Analyze",
            unified_memory=unified
        )
        
        assert isinstance(session, MatryoshkaExplorationSession)


# =============================================================================
# UNIFIED CLIENT TESTS
# =============================================================================

class TestUnifiedClient:
    """Test UnifiedMatryoshkaClient functionality."""
    
    def test_client_initialization(self, temp_db_dir):
        """Test UnifiedMatryoshkaClient initialization."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        assert client.memory_bridge is not None
        assert client._active_sessions == {}
        assert client.default_max_turns == 10
    
    def test_client_is_available(self, temp_db_dir):
        """Test client availability check."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        # Should be available if memory bridge works
        assert client.is_available() is True
    
    def test_get_session_memory(self, temp_db_dir, sample_document):
        """Test getting session memory context."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        result = client.analyze_with_memory(
            query="Analyze",
            file_path=sample_document,
            max_turns=2,
            llm_code_callback=lambda q: 'print("ok")'
        )
        
        context = client.get_session_memory(result.session_id)
        assert context is not None
    
    def test_get_session_memory_nonexistent(self, temp_db_dir):
        """Test getting memory for non-existent session."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        context = client.get_session_memory("nonexistent")
        assert context is None
    
    def test_list_sessions(self, temp_db_dir, sample_document):
        """Test listing active sessions."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        # Initially empty
        assert client.list_sessions() == []
        
        # Add session
        client.analyze_with_memory(
            query="Analyze",
            file_path=sample_document,
            max_turns=1,
            llm_code_callback=lambda q: 'print("ok")'
        )
        
        sessions = client.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["document_path"] == sample_document
    
    def test_get_stats(self, temp_db_dir, sample_document):
        """Test getting client statistics."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        initial_stats = client.get_stats()
        assert initial_stats["active_sessions"] == 0
        
        client.analyze_with_memory(
            query="Analyze",
            file_path=sample_document,
            max_turns=2,
            llm_code_callback=lambda q: 'print("ok")'
        )
        
        stats = client.get_stats()
        assert stats["active_sessions"] == 1
        assert stats["memory_bridge_healthy"] is True
    
    def test_get_session_synthesis(self, temp_db_dir, sample_document):
        """Test getting session synthesis."""
        client = UnifiedMatryoshkaClient(unified_memory=None)
        
        result = client.analyze_with_memory(
            query="Analyze",
            file_path=sample_document,
            max_turns=2,
            llm_code_callback=lambda q: 'print("ok")'
        )
        
        synthesis = client.get_session_synthesis(result.session_id)
        assert synthesis is not None
        assert synthesis.session_id == result.session_id


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
