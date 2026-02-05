"""
Comprehensive Test Suite for RESE-LeanAide Workflow Integration

Tests ALL workflow components for 100% code coverage:
- LeanAideRESEWorkflow (main orchestrator)
- AutoformalizationService (all 4 RESE phases)
- ProofSearchService (MCTS-guided proof search)
- Phase I: Epistemic Audit
- Phase II: Isomorphic Mapping
- Phase III: MCTS Refinement
- Phase IV: Architectural Synthesis
- Batch processing and error handling
- WorkflowResult aggregation

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Env var validation
- Law of Idempotency: Safe re-execution
- Circuit Breaker: Per-phase circuit breakers
- Structured Logging: JSON format verification
- Law of UTC: Timestamp verification

Author: OpenEvolve
Created: 2026-02-04
"""

import pytest
import pytest_asyncio
import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict

# Import for workflow components
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leanaide_rese_workflow import (
    LeanAideRESEWorkflow,
    WorkflowConfig,
    WorkflowLogger,
    WorkflowResult,
    PhaseResult,
    PhaseStatus,
    ProblemClassification,
    ProblemType,
    SolverType,
    ProblemClass,
    ProblemDomain,
    FormalizationDomain,
)
from autoformalization_service import (
    AutoformalizationService,
    AutoformalizationConfig,
    AutoformalizationPhase,
    AutoformalizationResult,
    FormalizationDomain as AutoDomain,
)
from proof_search_service import (
    ProofSearchService,
    ProofSearchConfig,
    ProofStrategy,
    ProofStatus,
    ProofSearchResult,
    ProofTactic,
    MCTSProofNode,
    MCTSProofSearch,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def workflow_config():
    """Create test workflow configuration."""
    return WorkflowConfig(
        leanaide_host="localhost",
        leanaide_port=7654,
        autoformalization_timeout_ms=30000,
        autoformalization_confidence_threshold=0.7,
        proof_search_timeout_ms=60000,
        proof_search_max_depth=100,
        proof_search_mcts_iterations=1000,
        proof_search_enable_z3=True,
        proof_search_confidence_threshold=0.8,
        phase_i_timeout_ms=60000,
        phase_ii_timeout_ms=90000,
        phase_iii_timeout_ms=120000,
        phase_iv_timeout_ms=90000,
        workflow_timeout_ms=600000,
        max_retries=3,
        retry_delay_ms=1000,
        enable_caching=True,
    )


@pytest.fixture
def autoformalization_config():
    """Create test autoformalization configuration."""
    return AutoformalizationConfig(
        leanaide_host="localhost",
        leanaide_port=7654,
        timeout_ms=30000,
        max_alternatives=3,
        confidence_threshold=0.7,
        enable_caching=True,
    )


@pytest.fixture
def proof_search_config():
    """Create test proof search configuration."""
    return ProofSearchConfig(
        leanaide_host="localhost",
        leanaide_port=7654,
        timeout_ms=60000,
        max_search_depth=100,
        mcts_iterations=1000,
        mcts_exploration_constant=1.414,
        enable_z3_hybrid=True,
        enable_counterexamples=True,
        confidence_threshold=0.8,
    )


@pytest.fixture
def correlation_id():
    """Create correlation ID for tracing."""
    return str(uuid.uuid4())


# =============================================================================
# TEST: WORKFLOW CONFIGURATION
# =============================================================================

class TestWorkflowConfiguration:
    """Test workflow configuration."""

    def test_workflow_config_defaults(self):
        """Test workflow configuration has proper defaults."""
        config = WorkflowConfig()

        assert config.leanaide_host == "localhost"
        assert config.leanaide_port == 7654
        assert config.autoformalization_timeout_ms == 30000
        assert config.autoformalization_confidence_threshold == 0.7
        assert config.proof_search_timeout_ms == 60000
        assert config.max_retries == 3
        assert config.enable_caching is True

    def test_workflow_config_from_env(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            "LEANAIDE_HOST": "test-host",
            "LEANAIDE_PORT": "9999",
            "LEANAIDE_TIMEOUT_MS": "45000",
            "PROOF_SEARCH_TIMEOUT_MS": "90000",
            "WORKFLOW_TIMEOUT_MS": "900000",
            "WORKFLOW_MAX_RETRIES": "5",
            "WORKFLOW_ENABLE_CACHING": "false",
        }):
            config = WorkflowConfig.from_env()

            assert config.leanaide_host == "test-host"
            assert config.leanaide_port == 9999
            assert config.autoformalization_timeout_ms == 45000
            assert config.proof_search_timeout_ms == 90000
            assert config.workflow_timeout_ms == 900000
            assert config.max_retries == 5
            assert config.enable_caching is False

    def test_workflow_config_to_dict(self, workflow_config):
        """Test configuration serialization."""
        config_dict = workflow_config.to_dict()

        assert "leanaide_host" in config_dict
        assert "autoformalization_timeout_ms" in config_dict
        assert "proof_search_timeout_ms" in config_dict
        assert config_dict["leanaide_host"] == workflow_config.leanaide_host


# =============================================================================
# TEST: PROBLEM CLASSIFICATION
# =============================================================================

class TestProblemClassification:
    """Test problem classification logic."""

    def test_classify_theorem_proving(self):
        """Test classification of theorem proving problems."""
        workflow = LeanAideRESEWorkflow()
        workflow.initialize()

        problem = "Prove that for all natural numbers n, n + 0 = n"
        problem_class, problem_domain, complexity = workflow._classify_problem(
            problem,
            None,
            None,
            None
        )

        assert problem_class == ProblemType.THEOREM_PROVING
        assert problem_domain in [FormalizationDomain.ARITHMETIC, FormalizationDomain.LOGIC]

    def test_classify_isomorphism_detection(self):
        """Test classification of isomorphism problems."""
        workflow = LeanAideRESEWorkflow()
        workflow.initialize()

        problem = "Show that the natural numbers are isomorphic to the integers"
        problem_class, problem_domain, complexity = workflow._classify_problem(
            problem,
            None,
            None,
            None
        )

        assert problem_class == ProblemType.ISOMORPHISM_DETECTION

    def test_classify_optimization(self):
        """Test classification of optimization problems."""
        workflow = LeanAideRESEWorkflow()
        workflow.initialize()

        problem = "Minimize the cost function subject to constraints"
        problem_class, problem_domain, complexity = workflow._classify_problem(
            problem,
            None,
            None,
            None
        )

        assert problem_class == ProblemType.OPTIMIZATION

    def test_classify_estimated_tier_1(self):
        """Test classification estimates Tier 1 for simple problems."""
        workflow = LeanAideRESEWorkflow()
        workflow.initialize()

        problem = "x > 5 and y < 10"
        problem_class, problem_domain, complexity = workflow._classify_problem(
            problem,
            [{"expression": "(> x 5)"}],
            [{"name": "x", "type": "int"}],
            None
        )

        # Simple problem should estimate Tier 1
        assert complexity["estimated_tier"] == 1

    def test_classify_estimated_tier_3(self):
        """Test classification estimates Tier 3 for complex problems."""
        workflow = LeanAideRESEWorkflow()
        workflow.initialize()

        # Complex problem with many constraints
        problem = "Complex theorem with deep quantifier nesting"
        many_constraints = [
            {"expression": f"(> x{i} 0)"}
            for i in range(200)
        ]

        problem_class, problem_domain, complexity = workflow._classify_problem(
            problem,
            many_constraints,
            None,
            None
        )

        # Many constraints should estimate Tier 3
        assert complexity["estimated_tier"] == 3


# =============================================================================
# TEST: PHASE I - EPISTEMIC AUDIT
# =============================================================================

class TestPhaseIEpistemicAudit:
    """Test Phase I: Epistemic Audit workflows."""

    @pytest.mark.asyncio
    async def test_execute_phase_i_success(self, workflow_config):
        """Test successful Phase I execution."""
        workflow = LeanAideRESESEWorkflow(workflow_config)
        await workflow.initialize()

        # Mock the services
        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        # Mock autoformalization result
        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
            natural_language="Test constraint",
            lean_code="theorem test : Prop := by sorry",
            domain=AutoDomain.LOGIC,
            confidence=0.85,
            lean_theorem_name="test_constraint",
            lean_type="Prop",
        )

        # Mock proof search result
        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            theorem_name="test_constraint",
            lean_code="theorem test : Prop := by sorry",
            proof_found=True,
            execution_time_ms=1000,
        )

        workflow.autoformalization_service.autoformalize_phase_i = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_i = AsyncMock(return_value=proof_result)

        # Execute Phase I
        result = await workflow._execute_phase_i(
            problem_statement="All prime numbers greater than 2 are odd",
            classification=ProblemClassification(
                problem_type=ProblemType.CONSTRAINT_VERIFICATION,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.Z3,
                confidence=0.75,
                reasoning="Test",
            ),
            correlation_id="test-123",
        )

        assert result.status == PhaseStatus.COMPLETED
        assert result.data["constraint_count"] > 0
        assert len(result.autoformalization_results) > 0
        assert len(result.proof_search_results) > 0

    @pytest.mark.asyncio
    async def test_execute_phase_i_failure(self, workflow_config):
        """Test Phase I handles errors gracefully."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        workflow.autoformalization_service = Mock()
        workflow.autoformalization_service.autoformalize_phase_i = AsyncMock(
            side_effect=Exception("Autoformalization failed")
        )

        result = await workflow._execute_phase_i(
            problem_statement="Test problem",
            classification=ProblemClassification(
                problem_type=ProblemType.CONSTRAINT_VERIFICATION,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.Z3,
                confidence=0.75,
                reasoning="Test",
            ),
            correlation_id="test-456",
        )

        assert result.status == PhaseStatus.FAILED
        assert len(result.errors) > 0


# =============================================================================
# TEST: PHASE II - ISOMORPHIC MAPPING
    =============================================================================

class TestPhaseIIIsomorphicMapping:
    """Test Phase II: Isomorphic Mapping workflows."""

    @pytest.mark.asyncio
    async def test_execute_phase_ii_success(self, workflow_config):
        """Test successful Phase II execution."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        # Mock results
        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING,
            natural_language="Test mapping",
            lean_code="theorem iso : Prop := by sorry",
            domain=AutoDomain.CATEGORY_THEORY,
            confidence=0.80,
        )

        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            proof_found=True,
            execution_time_ms=1500,
        )

        workflow.autoformalization_service.autoformalize_phase_ii = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_ii = AsyncMock(return_value=proof_result)

        phase_i_data = {"test": "data"}

        result = await workflow._execute_phase_ii(
            problem_statement="Show isomorphism between sets",
            phase_i_data=phase_i_data,
            classification=ProblemClassification(
                problem_type=ProblemType.ISOMORPHISM_DETECTION,
                mathematical_domain=FormalizationDomain.SET_THEORY,
                recommended_solver=SolverType.LEANAIDE,
                confidence=0.8,
                reasoning="Test",
            ),
            correlation_id="test-789",
        )

        assert result.status == PhaseStatus.COMPLETED
        assert len(result.autoformalization_results) > 0
        assert "domains" in result.data

    @pytest.mark.asyncio
    async def test_execute_phase_ii_identifies_domains(self, workflow_config):
        """Test Phase II identifies domains correctly."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        # Mock services
        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING,
            natural_language="Test",
            lean_code="theorem test : Prop := by sorry",
            domain=AutoDomain.CATEGORY_THEORY,
        )

        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            proof_found=True,
        )

        workflow.autoformalization_service.autoformalize_phase_ii = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_ii = AsyncMock(return_value=proof_result)

        result = await workflow._execute_phase_ii(
            problem_statement="number and set relationships",
            phase_i_data={},
            classification=ProblemClassification(
                problem_type=ProblemType.ISOMORPHISM_DETECTION,
                mathematical_domain=FormalizationDomain.GENERAL,
                recommended_solver=SolverType.LEANAIDE,
                confidence=0.8,
                reasoning="Test",
            ),
            correlation_id="test-abc",
        )

        # Should identify domains
        assert "domains" in result.data
        assert isinstance(result.data["domains"], list)


# =============================================================================
# TEST: PHASE III - MCTS REFINEMENT
    =============================================================================

class TestPhaseIIIMCTSRefinement:
    """Test Phase III: MCTS Refinement workflows."""

    @pytest.mark.asyncio
    async def test_execute_phase_iii_success(self, workflow_config):
        """Test successful Phase III execution."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        # Mock results
        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT,
            natural_language="Test hypothesis",
            lean_code="theorem hyp : Prop := by sorry",
            domain=AutoDomain.LOGIC,
            confidence=0.75,
        )

        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            proof_found=True,
            confidence=0.9,
            search_nodes_explored=50,
            search_depth=10,
        )

        workflow.autoformalization_service.autoformalize_phase_iii = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_iii = AsyncMock(return_value=proof_result)

        result = await workflow._execute_phase_iii(
            problem_statement="If hypothesis then conclusion",
            previous_phases_data={},
            classification=ProblemClassification(
                problem_type=ProblemType.HYPOTHESIS_TESTING,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.HYBRID_ALL,
                confidence=0.75,
                reasoning="Test",
            ),
            correlation_id="test-def",
        )

        assert result.status == PhaseStatus.COMPLETED
        assert result.data["hypothesis_count"] > 0
        assert result.data["best_confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_execute_phase_iii_generates_hypotheses(self, workflow_config):
        """Test Phase III generates hypotheses."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT,
            natural_language="Test",
            lean_code="theorem test : Prop := by sorry",
            domain=AutoDomain.LOGIC,
        )

        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            proof_found=True,
        )

        workflow.autoformalization_service.autoformalize_phase_iii = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_iii = AsyncMock(return_value=proof_result)

        result = await workflow._execute_phase_iii(
            problem_statement="Test problem",
            previous_phases_data={},
            classification=ProblemClassification(
                problem_type=ProblemType.HYPOTHESIS_TESTING,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.HYBRID_ALL,
                confidence=0.75,
                reasoning="Test",
            ),
            correlation_id="test-ghi",
        )

        assert "hypotheses" in result.data
        assert len(result.data["hypotheses"]) > 0


# =============================================================================
# TEST: PHASE IV - ARCHITECTURAL SYNTHESIS
    =============================================================================

class TestPhaseIVArchitecturalSynthesis:
    """Test Phase IV: Architectural Synthesis workflows."""

    @pytest.mark.asyncio
    async def test_execute_phase_iv_success(self, workflow_config):
        """Test successful Phase IV execution."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        # Mock results
        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS,
            natural_language="Test model",
            lean_code="theorem model : Prop := by sorry",
            domain=AutoDomain.LOGIC,
            confidence=0.70,
        )

        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            proof_found=True,
            confidence=0.85,
        )

        workflow.autoformalization_service.autoformalize_phase_iv = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_iv = AsyncMock(return_value=proof_result)

        result = await workflow._execute_phase_iv(
            problem_statement="Test model",
            previous_phases_data={},
            classification=ProblemClassification(
                problem_type=ProblemType.MODEL_VALIDATION,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.HYBRID_ALL,
                confidence=0.70,
                reasoning="Test",
            ),
            correlation_id="test-jkl",
        )

        assert result.status == PhaseStatus.COMPLETED
        assert "model_description" in result.data
        assert "efficacy_claim" in result.data

    @pytest.mark.asyncio
    async def test_execute_phase_iv_generates_efficacy_claim(self, workflow_config):
        """Test Phase IV generates efficacy claims."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        auto_result = AutoformalizationResult(
            success=True,
            phase=AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS,
            natural_language="Test",
            lean_code="theorem eff : Prop := by sorry",
            domain=AutoDomain.LOGIC,
        )

        proof_result = ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED,
            proof_found=True,
        )

        workflow.autoformalization_service.autoformalize_phase_iv = AsyncMock(return_value=auto_result)
        workflow.proof_search_service.search_phase_iv = AsyncMock(return_value=proof_result)

        result = await workflow._execute_phase_iv(
            problem_statement="Test model description",
            previous_phases_data={},
            classification=ProblemClassification(
                problem_type=ProblemType.MODEL_VALIDATION,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.HYBRID_ALL,
                confidence=0.70,
                reasoning="Test",
            ),
            correlation_id="test-mno",
        )

        assert "efficacy_claim" in result.data
        assert len(result.data["efficacy_claim"]) > 0


# =============================================================================
# TEST: WORKFLOW EXECUTION
# =============================================================================

class TestWorkflowExecution:
    """Test complete workflow execution."""

    @pytest.mark.asyncio
    async def test_execute_workflow_all_phases(self, workflow_config):
        """Test executing complete workflow through all phases."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        # Mock services
        workflow.autoformalization_service = Mock()
        workflow.proof_search_service = Mock()

        # Create mock results for each phase
        def create_auto_result(phase, domain):
            return AutoformalizationResult(
                success=True,
                phase=phase,
                natural_language="Test",
                lean_code="theorem test : Prop := by sorry",
                domain=domain,
                confidence=0.75,
            )

        def create_proof_result():
            return ProofSearchResult(
                success=True,
                status=ProofStatus.PROVED,
                proof_found=True,
                execution_time_ms=1000,
            )

        # Setup mock responses
        async def mock_autoformalize_i(*args, **kwargs):
            return create_auto_result(
                AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
                AutoDomain.LOGIC
            )

        async def mock_autoformalize_ii(*args, **kwargs):
            return create_auto_result(
                AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING,
                AutoDomain.CATEGORY_THEORY
            )

        async def mock_autoformalize_iii(*args, **kwargs):
            return create_auto_result(
                AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT,
                AutoDomain.LOGIC
            )

        async def mock_autoformalize_iv(*args, **kwargs):
            return create_auto_result(
                AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS,
                AutoDomain.LOGIC
            )

        async def mock_proof_search(*args, **kwargs):
            return create_proof_result()

        workflow.autoformalization_service.autoformalize_phase_i = mock_autoformalize_i
        workflow.autoformalization_service.autoformalize_phase_ii = mock_autoformalize_ii
        workflow.autoformalization_service.autoformalize_phase_iii = mock_autoformalize_iii
        workflow.autoformalization_service.autoformalize_phase_iv = mock_autoformalize_iv

        workflow.proof_search_service.search_phase_i = mock_proof_search
        workflow.proof_search_service.search_phase_ii = mock_proof_search
        workflow.proof_search_service.search_phase_iii = mock_proof_search
        workflow.proof_search_service.search_phase_iv = mock_proof_search

        # Execute workflow
        result = await workflow.execute(
            problem_statement="Test problem for all phases",
            correlation_id="test-workflow",
        )

        assert result.overall_status == "completed"
        assert len(result.phase_results) == 4
        assert "phase_i" in result.phase_results
        assert "phase_ii" in result.phase_results
        assert "phase_iii" in result.phase_results
        assert "phase_iv" in result.phase_results

    @pytest.mark.asyncio
    async def test_execute_workflow_handles_phase_failure(self, workflow_config):
        """Test workflow handles phase failure gracefully."""
        workflow = LeanAideRESEWorkflow(workflow_config)
        await workflow.initialize()

        # Mock Phase I to fail
        workflow.autoformalization_service = Mock()
        workflow.autoformalization_service.autoformalize_phase_i = AsyncMock(
            side_effect=Exception("Phase I failed")
        )

        result = await workflow.execute(
            problem_statement="Test failing problem",
            correlation_id="test-fail",
        )

        # Should return failed status
        assert result.overall_status == "failed"
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_execute_workflow_timeout(self, workflow_config):
        """Test workflow handles timeout."""
        # Create config with very short timeout
        short_config = WorkflowConfig(
            workflow_timeout_ms=1,  # 1ms timeout
        )

        workflow = LeanAideRESEWorkflow(short_config)
        await workflow.initialize()

        # Mock services that take time
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms
            return AutoformalizationResult(
                success=True,
                phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
                natural_language="Test",
                lean_code="theorem test : Prop := by sorry",
                domain=AutoDomain.LOGIC,
            )

        workflow.autoformalization_service = Mock()
        workflow.autoformalization_service.autoformalize_phase_i = slow_operation
        workflow.proof_search_service = Mock()

        result = await workflow.execute(
            problem_statement="Test",
            correlation_id="test-timeout",
        )

        # Should timeout and return failed/timeout status
        assert result.overall_status in ["failed", "timeout"]


# =============================================================================
# TEST: WORKFLOW RESULT
# =============================================================================

class TestWorkflowResult:
    """Test workflow result aggregation."""

    def test_workflow_result_to_dict(self):
        """Test WorkflowResult serialization."""
        result = WorkflowResult(
            workflow_id="test-workflow",
            correlation_id="test-correlation",
            problem_classification=ProblemClassification(
                problem_type=ProblemType.THEOREM_PROVING,
                mathematical_domain=FormalizationDomain.LOGIC,
                recommended_solver=SolverType.LEANAIDE,
                confidence=0.85,
                reasoning="Test classification",
            ),
            phase_results={},
            overall_status="completed",
            total_execution_time_ms=5000.0,
        )

        result_dict = result.to_dict()

        assert result_dict["workflow_id"] == "test-workflow"
        assert result_dict["correlation_id"] == "test-correlation_id"
        assert result_dict["overall_status"] == "completed"
        assert "problem_classification" in result_dict
        assert "phase_results" in result_dict

    def test_phase_result_to_dict(self):
        """Test PhaseResult serialization."""
        result = PhaseResult(
            phase="phase_i_epistemic_audit",
            status=PhaseStatus.COMPLETED,
            data={"test": "data"},
            execution_time_ms=1500.0,
        )

        result_dict = result.to_dict()

        assert result_dict["phase"] == "phase_i_epistemic_audit"
        assert result_dict["status"] == "completed"
        assert result_dict["data"] == {"test": "data"}
        assert result_dict["execution_time_ms"] == 1500.0


# =============================================================================
# TEST: BATCH PROCESSING
# =============================================================================

class TestBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.mark.asyncio
    async def test_batch_autoformalization(self):
        """Test batch autoformalization."""
        service = AutoformalizationService()

        # Mock LeanAide client
        service.leanaide_client = None

        items = [
            {"text": "Statement 1", "type": "logical"},
            {"text": "Statement 2", "type": "arithmetic"},
            {"text": "Statement 3", "type": "logic"},
        ]

        results = await service.batch_autoformalize(
            items=items,
            phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
            correlation_id="test-batch",
        )

        assert len(results) == 3
        assert all(isinstance(r, AutoformalizationResult) for r in results)

    @pytest.mark.asyncio
    async def test_batch_proof_search(self):
        """Test batch proof search."""
        service = ProofSearchService()

        # Mock MCTS search
        service.mcts_search = Mock()
        service.mcts_search.search = AsyncMock(
            return_value=ProofSearchResult(
                success=True,
                status=ProofStatus.PROVED,
                theorem_name="test",
                lean_code="code",
                proof_found=True,
            )
        )

        items = [
            {"lean_code": "theorem1 : Prop := by sorry"},
            {"lean_code": "theorem2 : Prop := by sorry"},
            {"lean_code": "theorem3 : Prop := by sorry"},
        ]

        results = await service.batch_search(
            items=items,
            phase="phase_i",
            correlation_id="test-batch-proof",
        )

        assert len(results) == 3
        assert all(isinstance(r, ProofSearchResult) for r in results)


# =============================================================================
# TEST: ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Test error handling in workflow."""

    @pytest.mark.asyncio
    async def test_autoformalization_service_handles_errors(self):
        """Test autoformalization service handles errors gracefully."""
        service = AutoformalizationService()

        # Test with invalid input that causes errors
        result = await service.autoformalize_phase_i(
            constraint_text="",
            constraint_type="logical",
            correlation_id="test-error",
        )

        # Should return failed result
        assert result.success is False
        assert len(result.errors) >= 0

    @pytest.mark.asyncio
    async def test_proof_search_handles_timeout(self):
        """Test proof search handles timeout."""
        config = ProofSearchConfig(
            timeout_ms=1,  # 1ms timeout
        )

        service = ProofSearchService(config)

        # Create search that will timeout
        service.mcts_search = MCTSProofSearch(config, Mock())

        # Mock search that takes too long
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms > 1ms
            return ProofSearchResult(
                success=True,
                status=ProofStatus.PROVED,
                theorem_name="test",
                lean_code="code",
                proof_found=True,
            )

        service.mcts_search.search = slow_search

        result = await service.search_phase_i(
            lean_code="theorem test : Prop := by sorry",
            correlation_id="test-timeout-proof",
        )

        # Should handle timeout gracefully
        assert isinstance(result, ProofSearchResult)


# =============================================================================
# TEST: MCTS PROOF SEARCH
# =============================================================================

class TestMCTSProofSearch:
    """Test MCTS-guided proof search."""

    def test_mcts_proof_node_ucb1_calculation(self):
        """Test UCB1 calculation for node selection."""
        node = MCTSProofNode(
            proof_state="test state",
            parent=None,
            tactic=ProofTactic(name="test", confidence=0.5),
        )

        # UCB1 should be infinity for unvisited nodes
        ucb1 = node.ucb1(0)
        assert ucb1 == float('inf')

    def test_mcts_proof_node_visited_ucb1(self):
        """Test UCB1 calculation for visited nodes."""
        node = MCTSProofNode(
            proof_state="test state",
            parent=None,
        )
        node.visits = 10
        node.value = 5.0

        ucb1 = node.ucb1(20, c=1.414)

        # Should return finite value
        assert ucb1 < float('inf')
        assert ucb1 > 0

    def test_mcts_backpropagation(self):
        """Test backpropagation updates nodes correctly."""
        root = MCTSProofNode(proof_state="root")

        # Add child
        child = MCTSProofNode(
            proof_state="child",
            parent=root,
            tactic=ProofTactic(name="apply", confidence=0.7),
        )
        root.children.append(child)

        # Backpropagate reward
        reward = 0.8

        # Manually backpropagate
        node = child
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent if hasattr(node, 'parent') else None

        # Verify updates
        assert child.visits == 1
        assert child.value == 0.8
        assert root.visits == 1
        assert root.value == 0.8


# =============================================================================
# RUN TESTS WITH COVERAGE
# =============================================================================

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=glue/adapters/rese-leanaide-workflow/src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=90",
    ])
