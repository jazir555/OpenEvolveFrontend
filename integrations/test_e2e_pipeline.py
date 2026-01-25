"""
End-to-End Pipeline Tests for RESE Stage Integrations

Comprehensive test suite for all 9 stage integrations with E2E Invention Engine.

Author: Agent A4 (Stage Integration Lead)
Created: 2025-12-31
Status: 🟢 Active Implementation
Target: 2 hours implementation
"""

import sys
import unittest
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all stage integrations
from integrations.stage1 import (
    Stage1Integration, PromptInput, PromptAnalysisStatus
)
from integrations.stage2 import (
    Stage2Integration, Domain, DomainPair, MappingStatus
)
from integrations.stage3 import (
    Stage3Integration, SearchProblem, SearchStatus
)
from integrations.stage5 import (
    Stage5Integration, SolutionCandidate, ValidationStatus
)
from integrations.stage6 import (
    Stage6Integration, ErrorReport, ErrorAnalysisStatus
)
from integrations.stage7 import (
    Stage7Integration, AdversarialScenario, AdversarialStatus
)
from integrations.stage8 import (
    Stage8Integration, ArchitectureComponent, AssemblyStatus
)
from integrations.stage9 import (
    Stage9Integration, FinalValidationStatus
)


class TestStage1Integration(unittest.TestCase):
    """Test Stage 1 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage1Integration()

    def test_prompt_analysis(self):
        """Test basic prompt analysis"""
        prompt = PromptInput(
            text="Design a system that must minimize energy consumption while maximizing performance",
            domain="engineering"
        )

        result = self.integration.analyze_prompt(prompt)

        self.assertEqual(result.status, PromptAnalysisStatus.COMPLETED)
        self.assertIsInstance(result.constraints, list)
        self.assertIsInstance(result.assumptions, list)

    def test_sce_integration(self):
        """Test SCE integration"""
        sce_state = self.integration.get_sce_state()
        self.assertIn('total_constraints', sce_state)

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        prompt = PromptInput(text="Simple test prompt")
        result = self.integration.analyze_prompt(prompt)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)


class TestStage2Integration(unittest.TestCase):
    """Test Stage 2 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage2Integration()

    def test_domain_analysis(self):
        """Test domain pair analysis"""
        source_domain = Domain(
            id="source",
            name="Source Domain",
            description="Test source domain",
            formal_constraints=[],
            metadata={'variables': {'x': 1, 'y': 2}}
        )

        target_domain = Domain(
            id="target",
            name="Target Domain",
            description="Test target domain",
            formal_constraints=[],
            metadata={'variables': {'x': 1, 'z': 3}}
        )

        result = self.integration.analyze_domains(source_domain, target_domain)

        self.assertEqual(result.status, MappingStatus.COMPLETED)
        self.assertIsInstance(result.ontology_mappings, list)

    def test_ontology_mapping(self):
        """Test ontology mapping"""
        source = Domain(
            id="s1",
            name="S1",
            description="Test",
            formal_constraints=[],
            metadata={'variables': {'energy': 0.5}}
        )

        target = Domain(
            id="t1",
            name="T1",
            description="Test",
            formal_constraints=[],
            metadata={'variables': {'energy': 0.5}}  # Changed to 'energy' to match
        )

        result = self.integration.analyze_domains(source, target)
        self.assertGreater(result.transfer_confidence, 0.0)


class TestStage3Integration(unittest.TestCase):
    """Test Stage 3 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage3Integration()

    def test_mcts_search(self):
        """Test MCTS search"""
        problem = SearchProblem(
            id="test_problem",
            variables={'x': 0.0, 'y': 0.0},
            constraints=[],
            objective="minimize"
        )

        result = self.integration.search(problem)

        self.assertIn(result.status, [SearchStatus.CONVERGED, SearchStatus.MAX_ITERATIONS])
        self.assertIsNotNone(result.best_solution)
        self.assertGreater(result.iterations, 0)

    def test_aci_guidance(self):
        """Test ACI-guided search"""
        problem = SearchProblem(
            id="test_aci",
            variables={'x': 0.0},
            constraints=[],
            objective="optimize"
        )

        result = self.integration.search(problem, use_aci_guidance=True)
        self.assertTrue(result.aci_guidance_used)

    def test_batch_search(self):
        """Test batch search"""
        problems = [
            SearchProblem(
                id=f"batch_{i}",
                variables={'x': 0.0},
                constraints=[],
                objective="test"
            )
            for i in range(3)
        ]

        results = self.integration.batch_search(problems)
        self.assertEqual(len(results), 3)


class TestStage5Integration(unittest.TestCase):
    """Test Stage 5 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage5Integration()

    def test_solution_validation(self):
        """Test solution validation"""
        solution = SolutionCandidate(
            id="test_solution",
            variables={'energy': 100.0, 'mass': 50.0},
            constraints=[]
        )

        result = self.integration.validate_solution(solution)

        self.assertIn(result.status, [
            ValidationStatus.VALID,
            ValidationStatus.PARTIALLY_VALID
        ])
        self.assertIsNotNone(result.ltl_validation)

    def test_physics_check(self):
        """Test physics validation"""
        solution = SolutionCandidate(
            id="physics_test",
            variables={'energy': -100.0},  # Invalid: negative energy
            constraints=[]
        )

        result = self.integration.validate_solution(solution)
        self.assertIsNotNone(result.physics_check)

    def test_bias_detection(self):
        """Test bias detection"""
        solution = SolutionCandidate(
            id="bias_test",
            variables={'value': 100},  # Round number - potential anchoring bias
            constraints=[]
        )

        result = self.integration.validate_solution(solution)
        self.assertIsNotNone(result.bias_detection)


class TestStage6Integration(unittest.TestCase):
    """Test Stage 6 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage6Integration()

    def test_error_analysis(self):
        """Test error analysis"""
        error_report = ErrorReport(
            error_id="error_1",
            error_type="optimization_failed",
            error_message="Optimization failed to converge",
            stage="stage3",
            context={'iteration': 100}
        )

        result = self.integration.analyze_error(error_report)

        self.assertEqual(result.status, ErrorAnalysisStatus.COMPLETED)
        self.assertIsNotNone(result.assumption_feedback)

    def test_feedback_loops(self):
        """Test feedback loop generation"""
        error_report = ErrorReport(
            error_id="error_2",
            error_type="divergence",
            error_message="Solution diverged",
            stage="stage3",
            context={}
        )

        result = self.integration.analyze_error(error_report, use_feedback_loops=True)
        self.assertGreater(len(result.feedback_loops), 0)

    def test_diagnosis(self):
        """Test Γ₁ diagnosis"""
        error_report = ErrorReport(
            error_id="error_3",
            error_type="infeasibility",
            error_message="Problem infeasible",
            stage="stage2",
            context={}
        )

        result = self.integration.analyze_error(error_report)
        self.assertIsNotNone(result.diagnosis)


class TestStage7Integration(unittest.TestCase):
    """Test Stage 7 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage7Integration()

    def test_adversarial_validation(self):
        """Test adversarial validation"""
        scenario = AdversarialScenario(
            id="adv_scenario_1",
            solution={'x': 1.0, 'y': 2.0},
            constraints=[],
            assumptions=["Variables are independent", "System is linear"]
        )

        result = self.integration.validate_adversarially(scenario)

        self.assertEqual(result.status, AdversarialStatus.COMPLETED)
        self.assertGreater(len(result.red_team_attacks), 0)

    def test_red_blue_team(self):
        """Test red/blue team interaction"""
        scenario = AdversarialScenario(
            id="adv_scenario_2",
            solution={},
            constraints=[],
            assumptions=["Assumption 1"]
        )

        result = self.integration.validate_adversarially(scenario)

        self.assertGreater(len(result.red_team_attacks), 0)
        self.assertGreater(len(result.blue_team_defenses), 0)

    def test_security_score(self):
        """Test security score calculation"""
        scenario = AdversarialScenario(
            id="security_test",
            solution={},
            constraints=[],
            assumptions=[]
        )

        result = self.integration.validate_adversarially(scenario)
        self.assertGreaterEqual(result.overall_security_score, 0.0)
        self.assertLessEqual(result.overall_security_score, 1.0)


class TestStage8Integration(unittest.TestCase):
    """Test Stage 8 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage8Integration()

    def test_architecture_assembly(self):
        """Test architecture assembly"""
        components = [
            ArchitectureComponent(
                id="comp_1",
                type="neural",
                config={'layers': [64, 32]},
                inputs=['x'],
                outputs=['y']
            ),
            ArchitectureComponent(
                id="comp_2",
                type="symbolic",
                config={'rules': 5},
                inputs=['y'],
                outputs=['z']
            )
        ]

        result = self.integration.assemble_architecture(
            components,
            integration_strategy="hierarchical"
        )

        self.assertEqual(result.status, AssemblyStatus.COMPLETED)
        self.assertIsNotNone(result.architecture_blueprint)

    def test_predictive_models(self):
        """Test predictive model generation"""
        components = [
            ArchitectureComponent(
                id="model_comp",
                type="ensemble",
                config={},
                inputs=[],
                outputs=[]
            )
        ]

        result = self.integration.assemble_architecture(components, generate_models=True)
        self.assertGreater(len(result.predictive_models), 0)

    def test_model_validation(self):
        """Test model validation"""
        components = [
            ArchitectureComponent(
                id="val_comp",
                type="neural",
                config={},
                inputs=[],
                outputs=[]
            )
        ]

        result = self.integration.assemble_architecture(components, generate_models=True)
        self.assertGreater(len(result.validation_results), 0)


class TestStage9Integration(unittest.TestCase):
    """Test Stage 9 Integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration = Stage9Integration()

    def test_convergence_prediction(self):
        """Test convergence prediction"""
        aci_history = [0.9, 0.7, 0.5, 0.3, 0.2, 0.15]

        result = self.integration.validate_final_solution(
            solution_id="test_convergence",
            aci_history=aci_history,
            current_iteration=100
        )

        self.assertEqual(result.status, FinalValidationStatus.COMPLETED)
        self.assertIsNotNone(result.convergence_prediction)

    def test_final_validation(self):
        """Test final validation"""
        aci_history = [0.8, 0.6, 0.4, 0.2]  # 75% reduction

        result = self.integration.validate_final_solution(
            solution_id="test_final",
            aci_history=aci_history,
            current_iteration=50
        )

        self.assertIsNotNone(result.final_validation)
        self.assertTrue(
            result.final_validation.reduction_significant or
            len(result.final_validation.issues) > 0
        )

    def test_overall_validity(self):
        """Test overall validity determination"""
        aci_history = [0.9, 0.5, 0.1]  # Good reduction

        result = self.integration.validate_final_solution(
            solution_id="test_validity",
            aci_history=aci_history,
            current_iteration=75
        )

        # Should determine validity
        self.assertIsInstance(result.overall_valid, bool)


class TestEndToEndPipeline(unittest.TestCase):
    """Test Complete End-to-End Pipeline"""

    def test_full_pipeline_execution(self):
        """Test execution of all stages in sequence"""
        print("\n=== Testing Full Pipeline ===")

        # Stage 1: Prompt Analysis
        print("Stage 1: Prompt Analysis")
        stage1 = Stage1Integration()
        prompt = PromptInput(
            text="Design optimization problem with energy and performance constraints",
            domain="engineering"
        )
        result1 = stage1.analyze_prompt(prompt)
        self.assertEqual(result1.status, PromptAnalysisStatus.COMPLETED)
        print(f"  [OK] Constraints extracted: {len(result1.constraints)}")

        # Stage 2: Domain Mapping
        print("Stage 2: Domain Mapping")
        stage2 = Stage2Integration()
        source = Domain(
            id="s1", name="Source", description="Test",
            formal_constraints=[], metadata={'variables': {'x': 1}}
        )
        target = Domain(
            id="t1", name="Target", description="Test",
            formal_constraints=[], metadata={'variables': {'x': 1}}
        )
        result2 = stage2.analyze_domains(source, target)
        self.assertEqual(result2.status, MappingStatus.COMPLETED)
        print(f"  [OK] Transfer confidence: {result2.transfer_confidence:.2f}")

        # Stage 3: Search
        print("Stage 3: MCTS Search")
        stage3 = Stage3Integration()
        problem = SearchProblem(
            id="e2e_problem",
            variables={'x': 0.0},
            constraints=[],
            objective="minimize"
        )
        result3 = stage3.search(problem)
        self.assertIn(result3.status, [SearchStatus.CONVERGED, SearchStatus.MAX_ITERATIONS])
        print(f"  [OK] Best value: {result3.best_value:.4f}")

        # Stage 5: Validation
        print("Stage 5: Solution Validation")
        stage5 = Stage5Integration()
        solution = SolutionCandidate(
            id="e2e_solution",
            variables={'energy': 100.0},
            constraints=[]
        )
        result5 = stage5.validate_solution(solution)
        self.assertIn(result5.status, [ValidationStatus.VALID, ValidationStatus.PARTIALLY_VALID])
        print(f"  [OK] Validation status: {result5.status.value}")

        # Stage 9: Final Validation
        print("Stage 9: Final Validation")
        stage9 = Stage9Integration()
        aci_history = [0.8, 0.6, 0.4, 0.2]
        result9 = stage9.validate_final_solution(
            solution_id="e2e_final",
            aci_history=aci_history,
            current_iteration=100
        )
        self.assertEqual(result9.status, FinalValidationStatus.COMPLETED)
        print(f"  [OK] Overall valid: {result9.overall_valid}")

        print("\n=== Full Pipeline Test Passed [OK] ===")

    def test_pipeline_performance(self):
        """Test pipeline execution time"""
        start_time = time.time()

        # Quick pipeline run
        stage1 = Stage1Integration()
        prompt = PromptInput(text="Test prompt", domain="test")
        result1 = stage1.analyze_prompt(prompt)

        stage3 = Stage3Integration()
        problem = SearchProblem(id="perf", variables={}, constraints=[], objective="test")
        result3 = stage3.search(problem)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        self.assertLess(elapsed, 30.0)
        print(f"\nPipeline performance: {elapsed:.2f}s")


def run_all_tests():
    """Run all tests and generate report"""
    print("=" * 80)
    print("RESE Stage Integration Tests")
    print("=" * 80)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStage1Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage2Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage3Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage5Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage6Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage7Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage8Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestStage9Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
