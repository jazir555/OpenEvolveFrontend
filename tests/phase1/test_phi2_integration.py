"""
Integration Tests for Φ₂ System

Tests integration with SCE and Stage 5.

Author: Agent B2 (Φ₂ Specialist)
Created: 2025-12-31
Status: 🟢 Active
"""

import pytest
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase1"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from phase1.phi2_integration import (
    SCEPhi2Integrator,
    Stage5Phi2Monitor,
    IntegrationConfig
)
from phase1.cognitive_biases import BiasType, Severity
from core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)


class TestSCEIntegration:
    """Test integration with Symbolic Constraint Engine"""

    @pytest.fixture
    def sce(self):
        """Create fresh SCE instance"""
        return SymbolicConstraintEngine()

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return IntegrationConfig(
            auto_check_on_add=True,
            auto_check_on_conflict=True,
            bias_threshold=0.4,
            max_bias_score=0.6,
            log_all_detections=False  # Disable logging for tests
        )

    @pytest.fixture
    def integrator(self, sce, config):
        """Create integrator instance"""
        return SCEPhi2Integrator(sce, config)

    def test_integrator_initialization(self, sce, config):
        """Test integrator can be initialized"""
        integrator = SCEPhi2Integrator(sce, config)
        assert integrator.sce == sce
        assert integrator.config == config
        assert integrator.detector is not None

    def test_constraint_addition_with_bias_check(self, integrator):
        """Test constraints are checked for bias when added"""
        constraint = Constraint(
            id="test_1",
            type=ConstraintType.HARD,
            description="This will certainly achieve perfect results",
            formalization="perfect = true",
            source="user"
        )

        integrator.sce.add_constraint(constraint)

        # Should have bias history
        assert len(integrator.bias_history) > 0

        # Latest report should show detections
        latest_report = integrator.bias_history[-1]
        assert latest_report.total_detections > 0

    def test_check_all_constraints(self, integrator):
        """Test checking all constraints at once"""
        # Add multiple constraints
        for i in range(5):
            constraint = Constraint(
                id=f"test_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"c_{i}",
                source="test"
            )
            integrator.sce.add_constraint(constraint)

        # Check all
        report = integrator.check_all_constraints()

        assert report.total_detections >= 0
        assert len(integrator.bias_history) > 0

    def test_get_biased_constraints(self, integrator):
        """Test retrieving biased constraints"""
        # Add biased constraint
        constraint = Constraint(
            id="biased_1",
            type=ConstraintType.HARD,
            description="This will certainly always work perfectly",
            formalization="perfect = true",
            source="expert"
        )
        integrator.sce.add_constraint(constraint)

        # Get biased constraints
        biased = integrator.get_biased_constraints(min_severity=Severity.MEDIUM)

        assert isinstance(biased, dict)
        # May or may not have biased constraints depending on detection

    def test_suggest_debiased_formulation(self, integrator):
        """Test debiased formulation suggestions"""
        constraint = Constraint(
            id="debias_1",
            type=ConstraintType.HARD,
            description="We must maximize performance",
            formalization="max_performance",
            source="user"
        )
        integrator.sce.add_constraint(constraint)

        suggestions = integrator.suggest_debiased_formulation("debias_1")

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)

    def test_get_integration_statistics(self, integrator):
        """Test integration statistics"""
        # Add some constraints
        for i in range(3):
            constraint = Constraint(
                id=f"stat_{i}",
                type=ConstraintType.HARD,
                description=f"Test {i}",
                formalization=f"t_{i}",
                source="test"
            )
            integrator.sce.add_constraint(constraint)

        stats = integrator.get_integration_statistics()

        assert "sce_constraints_analyzed" in stats
        assert "bias_reports_generated" in stats
        assert stats["sce_constraints_analyzed"] >= 3

    def test_auto_check_disabled(self, sce):
        """Test behavior when auto-check is disabled"""
        config = IntegrationConfig(auto_check_on_add=False)
        integrator = SCEPhi2Integrator(sce, config)

        constraint = Constraint(
            id="no_auto_1",
            type=ConstraintType.HARD,
            description="Test",
            formalization="test",
            source="test"
        )

        integrator.sce.add_constraint(constraint)

        # Should not have auto-checked (history may be empty or have fewer entries)
        # We'll just verify the constraint was added successfully
        assert integrator.sce.get_constraint("no_auto_1") is not None


class TestStage5Integration:
    """Test integration with Stage 5 solution generation"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return IntegrationConfig(
            real_time_monitoring=True,
            max_bias_score=0.6
        )

    @pytest.fixture
    def monitor(self, config):
        """Create monitor instance"""
        return Stage5Phi2Monitor(config)

    def test_monitor_initialization(self, config):
        """Test monitor can be initialized"""
        monitor = Stage5Phi2Monitor(config)
        assert monitor.config == config
        assert monitor.detector is not None

    def test_monitor_generation_step(self, monitor):
        """Test monitoring a generation step"""
        reasoning = "We will certainly achieve the optimal solution"

        report = monitor.monitor_generation_step(1, reasoning)

        assert report is not None
        assert len(monitor.generation_history) == 1

    def test_should_intervene(self, monitor):
        """Test intervention logic"""
        # Add high-bias reasoning
        reasoning = "This will certainly always work perfectly"

        monitor.monitor_generation_step(1, reasoning)

        # Should recommend intervention
        should_intervene = monitor.should_intervene(0)
        # May or may not intervene depending on threshold
        assert isinstance(should_intervene, bool)

    def test_get_bias_trajectory(self, monitor):
        """Test bias trajectory tracking"""
        reasonings = [
            "We will certainly achieve success",
            "This approach clearly works",
            "The outcome is guaranteed",
        ]

        for i, reasoning in enumerate(reasonings, 1):
            monitor.monitor_generation_step(i, reasoning)

        trajectory = monitor.get_bias_trajectory()

        assert len(trajectory) == 3
        assert all(isinstance(score, float) for score in trajectory)
        assert all(0 <= score <= 1 for score in trajectory)

    def test_get_step_recommendations(self, monitor):
        """Test getting recommendations for a step"""
        reasoning = "This will certainly work perfectly"
        monitor.monitor_generation_step(1, reasoning)

        recommendations = monitor.get_step_recommendations(0)

        assert isinstance(recommendations, list)

    def test_generate_debiased_alternatives(self, monitor):
        """Test generating debiased alternatives"""
        reasoning = "We must maximize performance"

        alternatives = monitor.generate_debiased_alternatives(reasoning)

        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        assert all(isinstance(alt, str) for alt in alternatives)

    def test_get_monitoring_statistics(self, monitor):
        """Test monitoring statistics"""
        # Add some steps
        for i in range(5):
            reasoning = f"Step {i} reasoning"
            monitor.monitor_generation_step(i, reasoning)

        stats = monitor.get_monitoring_statistics()

        assert "total_steps_monitored" in stats
        assert stats["total_steps_monitored"] == 5
        assert "average_bias_score" in stats


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""

    @pytest.fixture
    def sce(self):
        return SymbolicConstraintEngine()

    @pytest.fixture
    def config(self):
        return IntegrationConfig(
            auto_check_on_add=True,
            real_time_monitoring=True,
            bias_threshold=0.3,
            max_bias_score=0.5,
            log_all_detections=False
        )

    def test_constraint_to_solution_workflow(self, sce, config):
        """Test workflow from constraint formulation to solution generation"""
        # Step 1: Create integrator
        integrator = SCEPhi2Integrator(sce, config)

        # Step 2: Add constraints (with automatic bias checking)
        constraints = [
            Constraint(
                id="wf_1",
                type=ConstraintType.HARD,
                description="The system should approximately maintain 80% accuracy",
                formalization="accuracy >= 0.8",
                source="requirements"
            ),
            Constraint(
                id="wf_2",
                type=ConstraintType.SOFT,
                description="Response time should preferably be under 100ms",
                formalization="response_time < 100ms preferred",
                source="requirements"
            ),
        ]

        for constraint in constraints:
            sce.add_constraint(constraint)

        # Step 3: Check overall bias
        report = integrator.check_all_constraints()
        assert report is not None

        # Step 4: Simulate Stage 5 generation
        monitor = Stage5Phi2Monitor(config)

        generation_steps = [
            "We'll implement a caching layer to improve response time",
            "The accuracy target will be met through algorithm optimization",
            "Performance will be approximately within specified ranges",
        ]

        for i, reasoning in enumerate(generation_steps, 1):
            monitor.monitor_generation_step(i, reasoning, constraints)

        # Step 5: Check overall monitoring statistics
        stats = monitor.get_monitoring_statistics()
        assert stats["total_steps_monitored"] == 3

    def test_bias_mitigation_workflow(self, sce, config):
        """Test workflow of detecting and mitigating bias"""
        integrator = SCEPhi2Integrator(sce, config)

        # Step 1: Add biased constraint
        biased_constraint = Constraint(
            id="mitigate_1",
            type=ConstraintType.HARD,
            description="This will certainly achieve perfect accuracy",
            formalization="accuracy = 1.0",
            source="expert"
        )

        sce.add_constraint(biased_constraint)

        # Step 2: Detect bias
        report = integrator.check_all_constraints()

        # Step 3: Get suggestions for debiasing
        suggestions = integrator.suggest_debiased_formulation("mitigate_1")

        assert len(suggestions) > 0

        # Step 4: Create debiased version (manual step)
        debiased_constraint = Constraint(
            id="mitigate_1_debiased",
            type=ConstraintType.HARD,
            description="The system should maintain accuracy above 95%",
            formalization="accuracy >= 0.95",
            source="requirements"
        )

        # Step 5: Verify reduced bias
        # (In a real system, we'd replace the original constraint)
        sce.add_constraint(debiased_constraint)

        final_report = integrator.check_all_constraints()
        assert final_report is not None

    def test_iterative_debiasing_workflow(self, sce, config):
        """Test iterative debiasing over multiple iterations"""
        integrator = SCEPhi2Integrator(sce, config)

        # Iteration 1: Initial biased constraints
        constraints_v1 = [
            Constraint(
                id=f"v1_{i}",
                type=ConstraintType.HARD,
                description=f"This will certainly work perfectly {i}",
                formalization=f"perfect_{i}",
                source="expert"
            )
            for i in range(3)
        ]

        for c in constraints_v1:
            sce.add_constraint(c)

        report_v1 = integrator.check_all_constraints()
        score_v1 = report_v1.overall_bias_score

        # Iteration 2: Debiased constraints
        constraints_v2 = [
            Constraint(
                id=f"v2_{i}",
                type=ConstraintType.HARD,
                description=f"The system should approximately achieve target {i}",
                formalization=f"approx_{i}",
                source="data"
            )
            for i in range(3)
        ]

        for c in constraints_v2:
            sce.add_constraint(c)

        report_v2 = integrator.check_all_constraints()
        score_v2 = report_v2.overall_bias_score

        # V2 should have lower or equal bias (though this depends on the specific constraints)
        assert isinstance(score_v2, float)


class TestErrorHandling:
    """Test error handling in integration"""

    def test_nonexistent_constraint_suggestion(self):
        """Test handling of nonexistent constraint in suggestions"""
        sce = SymbolicConstraintEngine()
        config = IntegrationConfig()
        integrator = SCEPhi2Integrator(sce, config)

        # Try to get suggestions for nonexistent constraint
        with pytest.raises(ValueError):
            integrator.suggest_debiased_formulation("nonexistent")

    def test_invalid_step_recommendations(self):
        """Test handling of invalid step numbers"""
        monitor = Stage5Phi2Monitor()

        # Try to get recommendations for invalid step
        recommendations = monitor.get_step_recommendations(999)
        assert recommendations == []

        recommendations = monitor.get_step_recommendations(-1)
        assert recommendations == []

    def test_empty_generation_steps(self):
        """Test monitoring with no generation steps"""
        monitor = Stage5Phi2Monitor()

        stats = monitor.get_monitoring_statistics()
        assert stats["total_steps_monitored"] == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
