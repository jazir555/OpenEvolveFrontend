"""
Test suite for LeanAide Evolutionary Workflow Integration

Tests the comprehensive integration of evolutionary LeanAide with the workflow system.
"""

import asyncio
import pytest
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Import the module to test
from leanaide_evolutionary_workflow import (
    LeanEvolutionaryWorkflowStage,
    LeanEvolutionarySubProblemSolver,
    LeanEvolutionaryReassembler,
    EvolutionaryConfig,
    EvolutionaryProgress,
    EvolutionStrategy,
    MathematicalDomain,
    add_evolutionary_config_to_workflow_state,
    extract_evolutionary_config_from_workflow_state,
    is_subproblem_mathematical,
    solve_with_evolutionary_approach,
    verify_sub_problem_with_leanaide_evolutionary,
    verify_final_solution_with_leanaide_evolutionary,
    LEANAIDE_AVAILABLE,
    EVOLUTION_AVAILABLE,
    ADVERSARIAL_AVAILABLE,
    SELFPLAY_AVAILABLE,
    WORKFLOW_AVAILABLE
)

# Import workflow structures if available
if WORKFLOW_AVAILABLE:
    from workflow_structures import (
        WorkflowState,
        SubProblem,
        SolutionAttempt,
        VerificationReport
    )
else:
    # Create minimal stubs for testing
    @dataclass
    class SubProblem:
        id: str
        description: str
        dependencies: List[str] = field(default_factory=list)
        ai_suggested_complexity_score: int = 5
        solution_requirements: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class SolutionAttempt:
        sub_problem_id: str
        content: str
        generated_by_model: str
        timestamp: float
        status: str = "generated"
        solution_approach: Optional[str] = None
        openevolve_metrics: Optional[Dict[str, Any]] = None
        metadata: Optional[Dict[str, Any]] = None

    @dataclass
    class WorkflowState:
        workflow_id: str
        workflow_type: Any
        problem_statement: str
        current_stage: str
        decomposition_plan: Optional[Any] = None
        sub_problem_solutions: Dict[str, SolutionAttempt] = field(default_factory=dict)
        openevolve_parameters: Dict[str, Any] = field(default_factory=dict)


class TestEvolutionaryConfig:
    """Test EvolutionaryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionaryConfig()

        assert config.lean_evolution_enabled == True
        assert config.lean_evolution_strategy == EvolutionStrategy.HYBRID
        assert config.lean_evolution_generations == 50
        assert config.lean_evolution_population_size == 20
        assert config.lean_adversarial_rounds == 10
        assert config.lean_self_play_games == 20

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvolutionaryConfig(
            lean_evolution_enabled=False,
            lean_evolution_strategy=EvolutionStrategy.ADVERSARIAL,
            lean_evolution_generations=100,
            lean_evolution_population_size=50
        )

        assert config.lean_evolution_enabled == False
        assert config.lean_evolution_strategy == EvolutionStrategy.ADVERSARIAL
        assert config.lean_evolution_generations == 100
        assert config.lean_evolution_population_size == 50


class TestEvolutionaryWorkflowStage:
    """Test LeanEvolutionaryWorkflowStage class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EvolutionaryConfig(
            lean_evolution_enabled=True,
            lean_evolution_strategy=EvolutionStrategy.EVOLUTION,
            lean_evolution_generations=5,
            lean_evolution_population_size=10
        )

    @pytest.fixture
    def workflow_state(self):
        """Create test workflow state."""
        return WorkflowState(
            workflow_id="test_workflow",
            workflow_type="decomposition",
            problem_statement="Test problem requiring mathematical proof",
            current_stage="Stage 3"
        )

    @pytest.fixture
    def workflow_stage(self, config, workflow_state):
        """Create workflow stage instance."""
        return LeanEvolutionaryWorkflowStage(
            config=config,
            workflow_state=workflow_state
        )

    def test_initialization(self, workflow_stage):
        """Test workflow stage initialization."""
        assert workflow_stage is not None
        assert workflow_stage.config is not None
        assert workflow_stage.evolution_progress == {}

    def test_is_mathematical_subproblem(self, workflow_stage):
        """Test mathematical sub-problem detection."""
        # Mathematical sub-problem
        math_subproblem = SubProblem(
            id="sp_math",
            description="Prove that for all natural numbers n, n + 0 = n",
            dependencies=[]
        )

        is_math, confidence, domain = workflow_stage.is_mathematical_subproblem(math_subproblem)

        # Should detect as mathematical
        assert isinstance(is_math, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_is_not_mathematical_subproblem(self, workflow_stage):
        """Test non-mathematical sub-problem detection."""
        # Non-mathematical sub-problem
        non_math_subproblem = SubProblem(
            id="sp_ui",
            description="Create a user interface for the dashboard",
            dependencies=[]
        )

        is_math, confidence, domain = workflow_stage.is_mathematical_subproblem(non_math_subproblem)

        # Should detect as non-mathematical or very low confidence
        assert isinstance(is_math, bool)
        assert isinstance(confidence, float)

    def test_mathematical_domain_classification(self, workflow_stage):
        """Test mathematical domain classification."""
        # Algebra problem
        algebra_problem = SubProblem(
            id="sp_algebra",
            description="Prove that every finite field has prime power order",
            dependencies=[]
        )

        _, _, domain = workflow_stage.is_mathematical_subproblem(algebra_problem)

        # Should classify into some domain
        assert domain is None or isinstance(domain, MathematicalDomain)

    @pytest.mark.asyncio
    async def test_solve_subproblem_evolutionary_fallback(
        self, workflow_stage, workflow_state
    ):
        """Test evolutionary solution with fallback to standard."""
        sub_problem = SubProblem(
            id="sp_test",
            description="Create a web API endpoint",
            dependencies=[]
        )

        # This should fall back to standard approach
        solution = await workflow_stage.solve_subproblem_evolutionary(
            sub_problem, workflow_state
        )

        assert solution is not None
        assert solution.sub_problem_id == "sp_test"
        assert solution.content is not None

    def test_get_progress(self, workflow_stage):
        """Test progress tracking."""
        # Get progress for non-existent sub-problem
        progress = workflow_stage.get_progress("sp_nonexistent")
        assert progress is None

    def test_get_statistics(self, workflow_stage):
        """Test statistics retrieval."""
        stats = workflow_stage.get_statistics()
        assert isinstance(stats, dict)


class TestEvolutionarySubProblemSolver:
    """Test LeanEvolutionarySubProblemSolver class."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        config = EvolutionaryConfig(lean_evolution_enabled=True)
        workflow_stage = LeanEvolutionaryWorkflowStage(config=config)
        return LeanEvolutionarySubProblemSolver(workflow_stage)

    @pytest.fixture
    def workflow_state(self):
        """Create test workflow state."""
        return WorkflowState(
     workflow_type="test",
     workflow_id="test_solver",
     problem_statement="Test problem",
            current_stage="Stage 3"
        )

    @pytest.mark.asyncio
    async def test_solve_mathematical_subproblem(self, solver, workflow_state):
        """Test solving a mathematical sub-problem."""
        sub_problem = SubProblem(
            id="sp_math_001",
            description="Prove that the square root of 2 is irrational",
            dependencies=[]
        )

        solution = await solver.solve(sub_problem, workflow_state)

        assert solution is not None
        assert solution.sub_problem_id == "sp_math_001"

    @pytest.mark.asyncio
    async def test_solve_non_mathematical_subproblem(self, solver, workflow_state):
        """Test solving a non-mathematical sub-problem."""
        sub_problem = SubProblem(
            id="sp_ui_001",
            description="Design a responsive login form",
            dependencies=[]
        )

        solution = await solver.solve(sub_problem, workflow_state)

        assert solution is not None
        assert solution.sub_problem_id == "sp_ui_001"

    def test_get_solution_metadata(self, solver):
        """Test retrieving solution metadata."""
        # Get metadata for non-existent solution
        metadata = solver.get_solution_metadata("sp_nonexistent")
        assert metadata is None


class TestEvolutionaryReassembler:
    """Test LeanEvolutionaryReassembler class."""

    @pytest.fixture
    def reassembler(self):
        """Create reassembler instance."""
        config = EvolutionaryConfig()
        workflow_stage = LeanEvolutionaryWorkflowStage(config=config)
        return LeanEvolutionaryReassembler(workflow_stage)

    @pytest.fixture
    def workflow_state(self):
        """Create test workflow state."""
        return WorkflowState(
     workflow_type="test",
     workflow_id="test_reassembly",
     problem_statement="Test reassembly problem",
            current_stage="Stage 4"
        )

    @pytest.mark.asyncio
    async def test_reassemble_empty_solutions(self, reassembler, workflow_state):
        """Test reassembling with no sub-problems."""
        solutions = {}
        final = await reassembler.reassemble(solutions, workflow_state)

        assert final is not None
        assert final.sub_problem_id == "final_solution"

    @pytest.mark.asyncio
    async def test_reassemble_with_solutions(self, reassembler, workflow_state):
        """Test reassembling with sub-problem solutions."""
        solutions = {
            "sp_001": SolutionAttempt(
                sub_problem_id="sp_001",
                content="-- Sub-proof 1",
                generated_by_model="Test",
                timestamp=time.time()
            ),
            "sp_002": SolutionAttempt(
                sub_problem_id="sp_002",
                content="-- Sub-proof 2",
                generated_by_model="Test",
                timestamp=time.time()
            )
        }

        final = await reassembler.reassemble(solutions, workflow_state)

        assert final is not None
        assert final.sub_problem_id == "final_solution"
        assert "sp_001" in final.content or "Sub-proof 1" in final.content


class TestWorkflowIntegration:
    """Test workflow integration functions."""

    def test_add_evolutionary_config_to_workflow_state(self):
        """Test adding configuration to workflow state."""
        workflow_state = WorkflowState(
     workflow_type="test",
     workflow_id="test_config",
     problem_statement="Test",
            current_stage="Stage 3"
        )

        config = EvolutionaryConfig(
            lean_evolution_enabled=True,
            lean_evolution_strategy=EvolutionStrategy.ADVERSARIAL,
            lean_evolution_generations=100
        )

        updated_state = add_evolutionary_config_to_workflow_state(
            workflow_state, config
        )

        assert updated_state.openevolve_parameters is not None
        assert updated_state.openevolve_parameters["lean_evolution_enabled"] == True
        assert updated_state.openevolve_parameters["lean_evolution_strategy"] == "adversarial"
        assert updated_state.openevolve_parameters["lean_evolution_generations"] == 100

    def test_extract_evolutionary_config_from_workflow_state(self):
        """Test extracting configuration from workflow state."""
        workflow_state = WorkflowState(
     workflow_type="test",
     workflow_id="test_extract",
     problem_statement="Test",
            current_stage="Stage 3",
            openevolve_parameters={
                "lean_evolution_enabled": False,
                "lean_evolution_strategy": "self_play",
                "lean_evolution_generations": 75
            }
        )

        config = extract_evolutionary_config_from_workflow_state(workflow_state)

        assert config.lean_evolution_enabled == False
        assert config.lean_evolution_strategy == EvolutionStrategy.SELF_PLAY
        assert config.lean_evolution_generations == 75

    def test_extract_default_config(self):
        """Test extracting config when no parameters set."""
        workflow_state = WorkflowState(
     workflow_type="test",
     workflow_id="test_default",
     problem_statement="Test",
            current_stage="Stage 3",
            openevolve_parameters=None
        )

        config = extract_evolutionary_config_from_workflow_state(workflow_state)

        # Should use defaults
        assert config.lean_evolution_enabled == True
        assert config.lean_evolution_strategy == EvolutionStrategy.HYBRID


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_solve_with_evolutionary_approach(self):
        """Test convenience solve function."""
        sub_problem = SubProblem(
            id="sp_conv_test",
            description="Test sub-problem",
            dependencies=[]
        )

        workflow_state = WorkflowState(
     workflow_type="test",
     workflow_id="test_conv",
     problem_statement="Test",
            current_stage="Stage 3"
        )

        solution = await solve_with_evolutionary_approach(
            sub_problem, workflow_state
        )

        assert solution is not None

    def test_is_subproblem_mathematical(self):
        """Test mathematical detection convenience function."""
        config = EvolutionaryConfig()
        workflow_stage = LeanEvolutionaryWorkflowStage(config=config)

        sub_problem = SubProblem(
            id="sp_math_check",
            description="Prove a theorem about natural numbers",
            dependencies=[]
        )

        is_math, confidence = is_subproblem_mathematical(sub_problem, workflow_stage)

        assert isinstance(is_math, bool)
        assert isinstance(confidence, float)


class TestAvailabilityFlags:
    """Test component availability flags."""

    def test_availability_flags_exist(self):
        """Test that availability flags are defined."""
        assert isinstance(LEANAIDE_AVAILABLE, bool)
        assert isinstance(EVOLUTION_AVAILABLE, bool)
        assert isinstance(ADVERSARIAL_AVAILABLE, bool)
        assert isinstance(SELFPLAY_AVAILABLE, bool)
        assert isinstance(WORKFLOW_AVAILABLE, bool)


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.fixture
    def workflow_stage(self):
        """Create workflow stage for error testing."""
        config = EvolutionaryConfig(
            lean_evolution_enabled=True,
            lean_fallback_to_standard=True
        )
        return LeanEvolutionaryWorkflowStage(config=config)

    @pytest.fixture
    def workflow_state(self):
        """Create workflow state for error testing."""
        return WorkflowState(
     workflow_type="test",
     workflow_id="test_errors",
     problem_statement="Test error handling",
            current_stage="Stage 3"
        )

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, workflow_stage, workflow_state):
        """Test graceful fallback when evolution fails."""
        # Create a sub-problem that might cause issues
        sub_problem = SubProblem(
            id="sp_error_test",
            description="",  # Empty description
            dependencies=[]
        )

        # Should handle gracefully and fall back to standard
        solution = await workflow_stage.solve_subproblem_evolutionary(
            sub_problem, workflow_state
        )

        assert solution is not None


class TestIntegrationWithDecompositionWorkflow:
    """Test integration with Decomposition Workflow stages."""

    @pytest.fixture
    def workflow_state_with_decomposition(self):
        """Create workflow state with decomposition plan."""
        # Create mock sub-problems
        sub_problems = [
            SubProblem(
                id="sp_001",
                description="Prove that addition is commutative",
                dependencies=[],
                ai_suggested_complexity_score=3
            ),
            SubProblem(
                id="sp_002",
                description="Design API endpoint",
                dependencies=["sp_001"],
                ai_suggested_complexity_score=5
            )
        ]

        # Create mock decomposition plan
        @dataclass
        class MockDecompositionPlan:
            sub_problems: List[SubProblem]

        return WorkflowState(
     workflow_type="test",
     workflow_id="test_decomposition",
     problem_statement="Test with decomposition",
            current_stage="Stage 3",
            decomposition_plan=MockDecompositionPlan(sub_problems=sub_problems)
        )

    def test_mathematical_detection_in_decomposition(self, workflow_state_with_decomposition):
        """Test mathematical detection within decomposition context."""
        config = EvolutionaryConfig()
        stage = LeanEvolutionaryWorkflowStage(config=config)

        for sp in workflow_state_with_decomposition.decomposition_plan.sub_problems:
            is_math, confidence, domain = stage.is_mathematical_subproblem(sp)
            # Just verify it runs without error
            assert isinstance(is_math, bool)


class TestProgressTracking:
    """Test evolutionary progress tracking."""

    @pytest.fixture
    def workflow_stage(self):
        """Create workflow stage."""
        config = EvolutionaryConfig(
            lean_evolution_strategy=EvolutionStrategy.EVOLUTION
        )
        return LeanEvolutionaryWorkflowStage(config=config)

    def test_progress_initialization(self, workflow_stage):
        """Test progress tracker initialization."""
        progress = EvolutionaryProgress(
            sub_problem_id="sp_progress",
            strategy=EvolutionStrategy.EVOLUTION
        )

        assert progress.sub_problem_id == "sp_progress"
        assert progress.strategy == EvolutionStrategy.EVOLUTION
        assert progress.generation == 0
        assert progress.best_fitness == 0.0
        assert progress.status == "in_progress"

    def test_progress_to_dict(self):
        """Test progress serialization."""
        progress = EvolutionaryProgress(
            sub_problem_id="sp_serialize",
            strategy=EvolutionStrategy.ADVERSARIAL,
            generation=10,
            best_fitness=0.85
        )

        d = progress.to_dict()

        assert d["sub_problem_id"] == "sp_serialize"
        assert d["strategy"] == "adversarial"
        assert d["generation"] == 10
        assert d["best_fitness"] == 0.85


class TestStageIntegration:
    """Test integration with specific workflow stages."""

    @pytest.fixture
    def workflow_stage(self):
        """Create workflow stage."""
        config = EvolutionaryConfig()
        return LeanEvolutionaryWorkflowStage(config=config)

    @pytest.fixture
    def workflow_state(self):
        """Create workflow state."""
        return WorkflowState(
     workflow_type="test",
     workflow_id="test_stages",
     problem_statement="Test stage integration",
            current_stage="Stage 3"
        )

    @pytest.mark.asyncio
    async def test_stage3a_evolution(self, workflow_stage, workflow_state):
        """Test Stage 3A evolutionary solution generation."""
        solution = SolutionAttempt(
            sub_problem_id="sp_3a",
            content="Initial solution",
            generated_by_model="Test",
            timestamp=time.time()
        )

        evolved = await workflow_stage.evolve_solution_stage3a(
            solution, workflow_state
        )

        assert evolved is not None
        assert evolved.sub_problem_id == "sp_3a"

    @pytest.mark.asyncio
    async def test_stage3b_adversarial(self, workflow_stage, workflow_state):
        """Test Stage 3B adversarial evolution."""
        solution = SolutionAttempt(
            sub_problem_id="sp_3b",
            content="Solution to critique",
            generated_by_model="Test",
            timestamp=time.time()
        )

        evolved = await workflow_stage.adversarial_evolution_stage3b(
            solution, workflow_state
        )

        assert evolved is not None

    @pytest.mark.asyncio
    async def test_stage3c_verification(self, workflow_stage, workflow_state):
        """Test Stage 3C verification."""
        solution = SolutionAttempt(
            sub_problem_id="sp_3c",
            content="Solution to verify",
            generated_by_model="Test",
            timestamp=time.time()
        )

        report = await workflow_stage.verify_evolved_proof_stage3c(
            solution, workflow_state
        )

        assert report is not None
        assert report.solution_attempt_id == "sp_3c"

    @pytest.mark.asyncio
    async def test_stage5_final_verification(self, workflow_stage, workflow_state):
        """Test Stage 5 final verification."""
        solution = SolutionAttempt(
            sub_problem_id="final",
            content="Final integrated solution",
            generated_by_model="Integrated",
            timestamp=time.time()
        )

        report = await workflow_stage.evolutionary_final_verification_stage5(
            solution, workflow_state
        )

        assert report is not None
        assert report.solution_attempt_id == "final_solution"


# Run tests if executed directly
if __name__ == "__main__":
    print("Running LeanAide Evolutionary Workflow Integration Tests...\n")

    # Run a simple subset of tests
    import sys

    print("Testing Configuration...")
    config_test = TestEvolutionaryConfig()
    config_test.test_default_config()
    config_test.test_custom_config()
    print("  Configuration tests: PASSED")

    print("\nTesting Workflow Stage...")
    ws_test = TestEvolutionaryWorkflowStage()
    config = EvolutionaryConfig()
    workflow_state = WorkflowState(
     workflow_type="test",
     workflow_id="test",
     problem_statement="Test",
        current_stage="Stage 3"
    )
    stage = LeanEvolutionaryWorkflowStage(config=config)
    ws_test.test_initialization(stage)
    ws_test.test_get_progress(stage)
    ws_test.test_get_statistics(stage)
    print("  Workflow Stage tests: PASSED")

    print("\nTesting Integration Functions...")
    it_test = TestWorkflowIntegration()
    it_test.test_add_evolutionary_config_to_workflow_state()
    it_test.test_extract_evolutionary_config_from_workflow_state()
    it_test.test_extract_default_config()
    print("  Integration function tests: PASSED")

    print("\nTesting Progress Tracking...")
    pt_test = TestProgressTracking()
    pt_test.test_progress_initialization(stage)
    pt_test.test_progress_to_dict()
    print("  Progress tracking tests: PASSED")

    print("\n" + "="*60)
    print("Basic Tests Completed Successfully!")
    print("="*60)

    print("\nAvailability Status:")
    print(f"  LeanAide: {LEANAIDE_AVAILABLE}")
    print(f"  Evolution: {EVOLUTION_AVAILABLE}")
    print(f"  Adversarial: {ADVERSARIAL_AVAILABLE}")
    print(f"  Self-Play: {SELFPLAY_AVAILABLE}")
    print(f"  Workflow: {WORKFLOW_AVAILABLE}")

    print("\nTo run full test suite with pytest:")
    print("  pytest test_leanaide_evolutionary_workflow.py -v")
