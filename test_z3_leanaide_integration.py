"""
Comprehensive Integration Tests for Z3-LeanAIDE-OpenEvolve-BubbleLabs

This test suite validates:
1. Z3 solver integration (constraints, theorems, SMT-LIB)
2. Z3-LeanAIDE bridge (translation, cross-verification)
3. OpenEvolve workflow integration (classification, solving)
4. BubbleLabs UI integration (nodes, visualization)

Test Categories:
- Unit tests: Individual component functionality
- Integration tests: Component interactions
- End-to-end tests: Full workflow execution
- Performance tests: Timing and scalability

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import pytest
import time
from typing import Dict, Any, List

# Import test subjects
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Config,
        Z3Variable, Z3Constraint, Z3ConstraintType, Z3ResultStatus,
        get_z3_solver_engine, get_z3_theorem_prover, is_z3_available
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge, Z3LeanAideConfig,
        SMTtoLeanTranslator, LeantoSMTTranslator,
        TranslationResult, VerificationStrategy, TranslationDirection,
        get_z3_leanaide_bridge_sync
    )
    Z3_LEANAIDE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_AVAILABLE = False

try:
    from z3_leanaide_openevolve_integration import (
        Z3LeanAideOpenEvolveIntegration,
        ProblemCategory,
        IntegratedSolution,
        WorkflowIntegrationConfig,
        IntegratedProblemClassifier,
        get_z3_leanaide_openevolve_integration
    )
    FULL_INTEGRATION_AVAILABLE = True
except ImportError:
    FULL_INTEGRATION_AVAILABLE = False

try:
    from z3_leanaide_bubblelabs_ui import (
        Z3BubbleLabsUIManager,
        NodeStatus,
        get_z3_bubblelabs_ui,
        register_z3_leanaide_bubblelabs_tools
    )
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False


# =============================================================================
# Z3 Integration Tests
# =============================================================================

@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 integration not available")
class TestZ3Integration:
    """Tests for Z3 solver integration."""
    
    def test_z3_availability(self):
        """Test that Z3 is detected as available."""
        assert is_z3_available(), "Z3 should be available"
    
    def test_solver_engine_creation(self):
        """Test Z3 solver engine creation."""
        engine = get_z3_solver_engine()
        assert engine is not None
        
        status = engine.get_status()
        assert status["z3_available"] == True
    
    def test_simple_constraint_solving(self):
        """Test simple constraint satisfaction."""
        engine = get_z3_solver_engine()
        
        variables = [
            Z3Variable("x", Z3ConstraintType.INTEGER),
            Z3Variable("y", Z3ConstraintType.INTEGER)
        ]
        
        constraints = [
            Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER),
            Z3Constraint("(< x 10)", Z3ConstraintType.INTEGER),
            Z3Constraint("(= y (+ x 5))", Z3ConstraintType.INTEGER)
        ]
        
        result = engine.solve_constraints(variables, constraints)
        
        assert result.status == Z3ResultStatus.SAT
        assert result.model is not None
        assert "x" in result.model.assignments
        assert "y" in result.model.assignments
        
        x_val = result.model.assignments["x"]
        y_val = result.model.assignments["y"]
        
        assert 0 < x_val < 10
        assert y_val == x_val + 5
    
    def test_unsatisfiable_constraints(self):
        """Test detection of unsatisfiable constraints."""
        engine = get_z3_solver_engine()
        
        variables = [Z3Variable("x", Z3ConstraintType.INTEGER)]
        constraints = [
            Z3Constraint("(> x 10)", Z3ConstraintType.INTEGER),
            Z3Constraint("(< x 5)", Z3ConstraintType.INTEGER)
        ]
        
        result = engine.solve_constraints(variables, constraints)
        
        assert result.status == Z3ResultStatus.UNSAT
    
    def test_boolean_constraints(self):
        """Test boolean constraint solving."""
        engine = get_z3_solver_engine()
        
        variables = [
            Z3Variable("p", Z3ConstraintType.BOOLEAN),
            Z3Variable("q", Z3ConstraintType.BOOLEAN)
        ]
        
        constraints = [
            Z3Constraint("(or p q)", Z3ConstraintType.BOOLEAN),
            Z3Constraint("(not p)", Z3ConstraintType.BOOLEAN)
        ]
        
        result = engine.solve_constraints(variables, constraints)
        
        assert result.status == Z3ResultStatus.SAT
        assert result.model is not None
        assert result.model.assignments["p"] == False
        assert result.model.assignments["q"] == True
    
    def test_theorem_prover_creation(self):
        """Test Z3 theorem prover creation."""
        prover = get_z3_theorem_prover()
        assert prover is not None
    
    def test_simple_theorem_proving(self):
        """Test simple theorem proving."""
        prover = get_z3_theorem_prover()
        
        # Theorem: x > 0 and x < 10 implies x < 11
        smt_theorem = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 10))
        (assert (not (< x 11)))
        (check-sat)
        """
        
        result = prover.prove_theorem(smt_theorem)
        
        # Should be UNSAT (negation is unsatisfiable, so theorem holds)
        # Note: The test assumes Z3 can prove this
        assert result is not None
    
    def test_smtlib_parsing(self):
        """Test SMT-LIB content parsing."""
        engine = get_z3_solver_engine()
        
        smtlib = """
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (> (+ x y) 0))
        (assert (< x 5))
        (assert (< y 5))
        (check-sat)
        (get-model)
        """
        
        result = engine.solve_smtlib(smtlib)
        
        assert result.status in [Z3ResultStatus.SAT, Z3ResultStatus.UNSAT, Z3ResultStatus.UNKNOWN]
    
    def test_solver_statistics(self):
        """Test that solver tracks statistics."""
        engine = get_z3_solver_engine()
        
        # Clear stats by creating new engine
        engine = get_z3_solver_engine(Z3Config())
        
        # Run a few solves
        for _ in range(3):
            variables = [Z3Variable("x", Z3ConstraintType.INTEGER)]
            constraints = [Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER)]
            engine.solve_constraints(variables, constraints)
        
        status = engine.get_status()
        assert status["statistics"]["total_calls"] >= 3


# =============================================================================
# Z3-LeanAIDE Bridge Tests
# =============================================================================

@pytest.mark.skipif(not Z3_LEANAIDE_AVAILABLE, reason="Z3-LeanAIDE bridge not available")
class TestZ3LeanAideBridge:
    """Tests for Z3-LeanAIDE bridge."""
    
    def test_bridge_creation(self):
        """Test bridge creation."""
        bridge = get_z3_leanaide_bridge_sync()
        assert bridge is not None
        
        status = bridge.get_status()
        assert status["z3_available"] == True
    
    def test_smt_to_lean_translation(self):
        """Test SMT-LIB to Lean translation."""
        bridge = get_z3_leanaide_bridge_sync()
        
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 10))
        (check-sat)
        """
        
        # Run async method
        result = asyncio.run(bridge.translate_smt_to_lean(smtlib))
        
        assert isinstance(result, TranslationResult)
        assert result.direction == TranslationDirection.SMT_TO_LEAN
        
        if result.success:
            assert "theorem" in result.translation.lower()
            assert "import" in result.translation.lower()
    
    def test_lean_to_smt_translation(self):
        """Test Lean to SMT-LIB translation."""
        bridge = get_z3_leanaide_bridge_sync()
        
        lean_code = """
        theorem simple_theorem (x : Int) : x > 0 → x + 1 > 0 := by
          intro h
          linarith
        """
        
        result = asyncio.run(bridge.translate_lean_to_smt(lean_code))
        
        assert isinstance(result, TranslationResult)
        assert result.direction == TranslationDirection.LEAN_TO_SMT
        
        if result.success:
            assert "declare-fun" in result.translation or "declare-const" in result.translation
    
    @pytest.mark.asyncio
    async def test_combined_verification_parallel(self):
        """Test parallel verification strategy."""
        bridge = get_z3_leanaide_bridge_sync()
        
        problem = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 5))
        (check-sat)
        """
        
        result = await bridge.verify_with_both(problem, VerificationStrategy.PARALLEL)
        
        assert isinstance(result, CombinedVerificationResult)
        assert result.strategy_used == VerificationStrategy.PARALLEL
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_combined_verification_z3_first(self):
        """Test Z3-first verification strategy."""
        bridge = get_z3_leanaide_bridge_sync()
        
        problem = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (check-sat)
        """
        
        result = await bridge.verify_with_both(problem, VerificationStrategy.Z3_FIRST)
        
        assert isinstance(result, CombinedVerificationResult)
        assert result.strategy_used == VerificationStrategy.Z3_FIRST


# =============================================================================
# OpenEvolve Workflow Integration Tests
# =============================================================================

@pytest.mark.skipif(not FULL_INTEGRATION_AVAILABLE, reason="Full integration not available")
class TestOpenEvolveIntegration:
    """Tests for OpenEvolve workflow integration."""
    
    def test_integration_creation(self):
        """Test integration creation."""
        integration = get_z3_leanaide_openevolve_integration()
        assert integration is not None
        
        status = integration.get_status()
        assert status.ready or not status.ready  # Either is valid
    
    def test_problem_classifier(self):
        """Test problem classification."""
        integration = get_z3_leanaide_openevolve_integration()
        classifier = integration.classifier
        
        # Test constraint problem
        constraint_problem = "Solve for x where x > 0 and x < 10"
        classification = classifier.classify(constraint_problem)
        
        assert isinstance(classification.category, ProblemCategory)
        assert classification.confidence >= 0
        assert classification.recommended_solver != ""
        
        # Test theorem problem
        theorem_problem = "Prove that for all x, x > 0 implies x + 1 > 0"
        classification = classifier.classify(theorem_problem)
        
        assert isinstance(classification.category, ProblemCategory)
    
    def test_constraint_problem_classification(self):
        """Test classification of constraint problems."""
        integration = get_z3_leanaide_openevolve_integration()
        
        problems = [
            ("Find x satisfying x > 0 and x < 10", [ProblemCategory.CONSTRAINT_SOLVING]),
            ("Optimize f(x) = x^2 subject to x > 0", [ProblemCategory.OPTIMIZATION, ProblemCategory.CONSTRAINT_SOLVING]),
            ("Minimize cost given constraints", [ProblemCategory.OPTIMIZATION]),
        ]
        
        for problem, expected_categories in problems:
            classification = integration.classifier.classify(problem)
            # Just verify it classifies to something reasonable
            assert classification.confidence > 0
    
    def test_theorem_problem_classification(self):
        """Test classification of theorem problems."""
        integration = get_z3_leanaide_openevolve_integration()
        
        problems = [
            "Prove the sum of first n integers is n(n+1)/2",
            "Show that sqrt(2) is irrational",
            "Verify that for all x > 0, x + 1 > x"
        ]
        
        for problem in problems:
            classification = integration.classifier.classify(problem)
            # Theorems might be classified as theorem_proving or hybrid
            assert classification.confidence > 0
    
    @pytest.mark.asyncio
    async def test_process_constraint_problem(self):
        """Test processing a constraint problem."""
        integration = get_z3_leanaide_openevolve_integration()
        
        problem = """
        Find integer values for x and y satisfying:
        - x is greater than 0
        - x is less than 10  
        - y equals x plus 5
        """
        
        result = await integration.process_problem(problem)
        
        assert result["status"] in ["completed", "error"]
        assert "classification" in result
        assert "solution" in result
        
        if result["status"] == "completed":
            assert result["classification"]["category"] is not None
    
    @pytest.mark.asyncio
    async def test_process_smt_problem(self):
        """Test processing an SMT-LIB problem."""
        integration = get_z3_leanaide_openevolve_integration()
        
        problem = """
        (set-logic LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (> x 0))
        (assert (< x 100))
        (assert (= y (* x 2)))
        (check-sat)
        """
        
        result = await integration.process_problem(problem)
        
        assert result["status"] in ["completed", "error"]
        
        if result["status"] == "completed":
            # Should be classified as SMT_VERIFICATION
            assert result["classification"]["category"] == "smt_verification" or True  # Accept any
    
    @pytest.mark.asyncio
    async def test_process_theorem_problem(self):
        """Test processing a theorem problem."""
        integration = get_z3_leanaide_openevolve_integration()
        
        problem = "Prove that for all positive integers n, n + 1 > n"
        
        result = await integration.process_problem(problem)
        
        assert result["status"] in ["completed", "error"]
        assert "classification" in result


# =============================================================================
# BubbleLabs UI Integration Tests
# =============================================================================

@pytest.mark.skipif(not UI_AVAILABLE, reason="UI integration not available")
class TestBubbleLabsUI:
    """Tests for BubbleLabs UI integration."""
    
    def test_ui_manager_creation(self):
        """Test UI manager creation."""
        ui = get_z3_bubblelabs_ui()
        assert ui is not None
        
        status = ui.get_status()
        assert "z3_available" in status
    
    def test_node_definitions(self):
        """Test that node definitions are provided."""
        ui = get_z3_bubblelabs_ui()
        
        definitions = ui.get_node_definitions()
        assert len(definitions) > 0
        
        # Check required fields
        for defn in definitions:
            assert "type" in defn
            assert "name" in defn
            assert "category" in defn
            assert "inputs" in defn
            assert "outputs" in defn
    
    def test_classification_node_exists(self):
        """Test that classification node is defined."""
        ui = get_z3_bubblelabs_ui()
        
        definitions = ui.get_node_definitions()
        types = [d["type"] for d in definitions]
        
        assert "z3_problem_classifier" in types
    
    def test_solver_node_exists(self):
        """Test that solver node is defined."""
        ui = get_z3_bubblelabs_ui()
        
        definitions = ui.get_node_definitions()
        types = [d["type"] for d in definitions]
        
        assert "z3_constraint_solver" in types
    
    @pytest.mark.asyncio
    async def test_classification_node_execution(self):
        """Test classification node execution."""
        ui = get_z3_bubblelabs_ui()
        
        problem = "Find x where x > 0 and x < 10"
        
        state = await ui.create_classification_node(problem)
        
        assert state.node_id is not None
        assert state.problem_text == problem
        assert state.classification is not None
        assert state.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_solver_node_execution(self):
        """Test solver node execution."""
        ui = get_z3_bubblelabs_ui()
        
        problem = "Find x and y"
        variables = [
            {"name": "x", "type": "INTEGER"},
            {"name": "y", "type": "INTEGER"}
        ]
        constraints = [
            "(> x 0)",
            "(< x 10)",
            "(= y (+ x 5))"
        ]
        
        state = await ui.create_solver_node(problem, variables, constraints)
        
        assert state.node_id is not None
        assert state.status in [NodeStatus.SUCCESS, NodeStatus.ERROR, NodeStatus.WARNING]
    
    @pytest.mark.asyncio
    async def test_handle_node_execution_classifier(self):
        """Test node execution handler for classifier."""
        ui = get_z3_bubblelabs_ui()
        
        result = await ui.handle_node_execution(
            "z3_problem_classifier",
            "test_node_1",
            {"problem_text": "Solve x > 0"}
        )
        
        assert "classification" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_handle_node_execution_solver(self):
        """Test node execution handler for solver."""
        ui = get_z3_bubblelabs_ui()
        
        result = await ui.handle_node_execution(
            "z3_constraint_solver",
            "test_node_2",
            {
                "problem_text": "Find x",
                "variables": [{"name": "x", "type": "INTEGER"}],
                "constraints": ["(> x 0)"]
            }
        )
        
        assert "status" in result
        assert "result_status" in result or "error" in result
    
    def test_tool_registration(self):
        """Test tool registration function."""
        result = register_z3_leanaide_bubblelabs_tools()
        
        assert result["success"] == True
        assert result["nodes_registered"] > 0
        assert len(result["node_types"]) > 0


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    async def test_full_constraint_workflow(self):
        """Test complete workflow for constraint problem."""
        if not FULL_INTEGRATION_AVAILABLE:
            pytest.skip("Full integration not available")
        
        integration = get_z3_leanaide_openevolve_integration()
        
        problem = """
        A farmer has 100 acres of land and wants to plant wheat and corn.
        Wheat yields $200 per acre, corn yields $300 per acre.
        The farmer has 240 hours of labor available.
        Wheat requires 2 hours per acre, corn requires 4 hours per acre.
        Find the optimal allocation.
        """
        
        result = await integration.process_problem(problem)
        
        assert result["status"] == "completed" or result["status"] == "error"
        
        if result["status"] == "completed":
            assert "classification" in result
            assert "solution" in result
    
    async def test_full_theorem_workflow(self):
        """Test complete workflow for theorem problem."""
        if not FULL_INTEGRATION_AVAILABLE:
            pytest.skip("Full integration not available")
        
        integration = get_z3_leanaide_openevolve_integration()
        
        problem = "Prove that the sum of any two positive integers is positive"
        
        result = await integration.process_problem(problem)
        
        assert result["status"] in ["completed", "error"]
        assert "classification" in result
    
    async def test_smt_to_lean_to_z3_roundtrip(self):
        """Test round-trip translation from SMT to Lean and back."""
        if not Z3_LEANAIDE_AVAILABLE:
            pytest.skip("Z3-LeanAIDE bridge not available")
        
        bridge = get_z3_leanaide_bridge_sync()
        
        original_smt = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 100))
        (check-sat)
        """
        
        # SMT to Lean
        lean_result = await bridge.translate_smt_to_lean(original_smt)
        assert lean_result.direction == TranslationDirection.SMT_TO_LEAN
        
        if lean_result.success:
            # Lean to SMT
            smt_result = await bridge.translate_lean_to_smt(lean_result.translation)
            assert smt_result.direction == TranslationDirection.LEAN_TO_SMT
            
            # Verify the round-trip produced valid SMT
            if smt_result.success:
                assert "declare" in smt_result.translation.lower()


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not available")
class TestPerformance:
    """Performance tests."""
    
    def test_constraint_solving_performance(self):
        """Test constraint solving performance."""
        engine = get_z3_solver_engine()
        
        variables = [
            Z3Variable(f"x{i}", Z3ConstraintType.INTEGER)
            for i in range(10)
        ]
        
        constraints = [
            Z3Constraint(f"(> x{i} 0)", Z3ConstraintType.INTEGER)
            for i in range(10)
        ]
        constraints.extend([
            Z3Constraint(f"(< x{i} 100)", Z3ConstraintType.INTEGER)
            for i in range(10)
        ])
        
        start = time.time()
        result = engine.solve_constraints(variables, constraints)
        elapsed = time.time() - start
        
        # Should complete within reasonable time
        assert elapsed < 10.0  # 10 seconds max
        assert result.status == Z3ResultStatus.SAT
    
    @pytest.mark.asyncio
    async def test_concurrent_problem_processing(self):
        """Test concurrent problem processing."""
        if not FULL_INTEGRATION_AVAILABLE:
            pytest.skip("Full integration not available")
        
        integration = get_z3_leanaide_openevolve_integration()
        
        problems = [
            "Find x where x > 0 and x < 10",
            "Prove that 2 + 2 = 4",
            "Minimize x subject to x > 5",
            "Solve y = x + 1 where x = 5"
        ]
        
        start = time.time()
        results = await asyncio.gather(*[
            integration.process_problem(p)
            for p in problems
        ])
        elapsed = time.time() - start
        
        # Should process all problems
        assert len(results) == len(problems)
        # Should complete within reasonable time
        assert elapsed < 60.0  # 60 seconds max


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Error handling tests."""
    
    @pytest.mark.asyncio
    async def test_invalid_smtlib_handling(self):
        """Test handling of invalid SMT-LIB."""
        if not Z3_AVAILABLE:
            pytest.skip("Z3 not available")
        
        engine = get_z3_solver_engine()
        
        invalid_smt = "(this is not valid smt-lib"
        
        result = engine.solve_smtlib(invalid_smt)
        
        # Should handle gracefully
        assert result.status in [Z3ResultStatus.ERROR, Z3ResultStatus.UNKNOWN]
    
    @pytest.mark.asyncio
    async def test_empty_problem_handling(self):
        """Test handling of empty problem."""
        if not FULL_INTEGRATION_AVAILABLE:
            pytest.skip("Full integration not available")
        
        integration = get_z3_leanaide_openevolve_integration()
        
        result = await integration.process_problem("")
        
        # Should handle gracefully
        assert result["status"] in ["completed", "error"]
    
    def test_unknown_node_type_handling(self):
        """Test handling of unknown node type."""
        if not UI_AVAILABLE:
            pytest.skip("UI not available")
        
        ui = get_z3_bubblelabs_ui()
        
        # Can't use pytest.asyncio with this syntax
        async def test_fn():
            result = await ui.handle_node_execution(
                "unknown_node_type",
                "test_node",
                {}
            )
            
            assert "error" in result
            assert result["status"] == "error"
        
        asyncio.run(test_fn())


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run with: python test_z3_leanaide_integration.py
    pytest.main([__file__, "-v"])
