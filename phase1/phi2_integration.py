"""
Phi 2 Integration Module

Integrates the Metacognitive Debiasing System with:
- SCE (Symbolic Constraint Engine)
- Stage 5 (Solution Generation)
- Real-time bias monitoring

Author: Agent B2 (Phi 2 Specialist)
Created: 2025-12-31
Status: Green - Active Implementation
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path

# Import Φ₂ and SCE
from .cognitive_biases import (
    CognitiveBiasDetector,
    BiasReport,
    BiasDetection,
    BiasType,
    Severity,
    DebiasingStrategy
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)


@dataclass
class IntegrationConfig:
    """Configuration for Φ₂ integration"""
    # SCE Integration
    auto_check_on_add: bool = True  # Check bias when adding constraints
    auto_check_on_conflict: bool = True  # Check bias when conflicts detected
    bias_threshold: float = 0.5  # Confidence threshold for alerts

    # Stage 5 Integration
    real_time_monitoring: bool = True  # Monitor during solution generation
    max_bias_score: float = 0.7  # Maximum allowed bias score
    auto_debias: bool = False  # Automatically apply debiasing (experimental)

    # Logging
    log_all_detections: bool = True
    log_path: Optional[Path] = None


class SCEPhi2Integrator:
    """
    Integrates Φ₂ with the Symbolic Constraint Engine.

    This class hooks into SCE events to provide real-time bias detection
    and mitigation for constraint management.
    """

    def __init__(
        self,
        sce: SymbolicConstraintEngine,
        config: Optional[IntegrationConfig] = None
    ):
        """
        Initialize the SCE-Φ₂ integration.

        Args:
            sce: The Symbolic Constraint Engine instance
            config: Optional configuration for the integration
        """
        self.sce = sce
        self.config = config or IntegrationConfig()
        self.detector = CognitiveBiasDetector()
        self.bias_history: List[BiasReport] = []

        # Register hooks with SCE
        if self.config.auto_check_on_add:
            self._register_add_hooks()

        if self.config.auto_check_on_conflict:
            self._register_conflict_hooks()

    def _register_add_hooks(self) -> None:
        """Register hooks for constraint addition events"""
        # Store original method
        original_add = self.sce.add_constraint

        # Wrap with bias checking
        def add_with_bias_check(constraint: Constraint) -> None:
            # First, run bias detection
            if self.config.auto_check_on_add:
                report = self.detector.analyze_constraints([constraint])

                # Log if above threshold
                if report.overall_bias_score >= self.config.bias_threshold:
                    self._log_bias_detection(
                        "constraint_add",
                        constraint.id,
                        report
                    )

                    # Optionally, warn user
                    if report.overall_bias_score >= self.config.max_bias_score:
                        print(
                            f"[Φ₂ WARNING] High bias detected ({report.overall_bias_score:.2f}) "
                            f"in constraint '{constraint.id}'"
                        )
                        print(f"  Top issues: {report.recommendations[:2]}")

            # Add to history (always, not just when logging)
            self.bias_history.append(report)

            # Call original method
            original_add(constraint)

        # Replace method
        self.sce.add_constraint = add_with_bias_check

    def _register_conflict_hooks(self) -> None:
        """Register hooks for conflict detection events"""
        # Note: This would require modifying SCE to emit events
        # For now, we'll provide a manual check method
        pass

    def check_constraint_bias(self, constraint: Constraint) -> BiasReport:
        """
        Manually check a constraint for bias.

        Args:
            constraint: Constraint to check

        Returns:
            BiasReport with analysis results
        """
        report = self.detector.analyze_constraints([constraint])
        self.bias_history.append(report)
        return report

    def check_all_constraints(self) -> BiasReport:
        """
        Check all constraints in the SCE for bias.

        Returns:
            BiasReport with comprehensive analysis
        """
        all_constraints = self.sce.get_all_constraints()
        report = self.detector.analyze_constraints(all_constraints)
        self.bias_history.append(report)
        return report

    def get_biased_constraints(
        self,
        min_severity: Severity = Severity.MEDIUM
    ) -> Dict[str, List[BiasDetection]]:
        """
        Get all constraints with biases above minimum severity.

        Args:
            min_severity: Minimum severity level to include

        Returns:
            Dict mapping constraint IDs to lists of detections
        """
        if not self.bias_history:
            return {}

        # Get most recent report
        latest_report = self.bias_history[-1]

        # Group detections by constraint
        by_constraint: Dict[str, List[BiasDetection]] = {}
        for detection in latest_report.detections:
            if detection.severity.value >= min_severity.value:
                for constraint_id in detection.affected_elements:
                    if constraint_id not in by_constraint:
                        by_constraint[constraint_id] = []
                    by_constraint[constraint_id].append(detection)

        return by_constraint

    def suggest_debiased_formulation(
        self,
        constraint_id: str
    ) -> List[str]:
        """
        Suggest debiased formulations for a constraint.

        Args:
            constraint_id: ID of constraint to debias

        Returns:
            List of suggested reformulations
        """
        constraint = self.sce.get_constraint(constraint_id)
        if not constraint:
            raise ValueError(f"Constraint {constraint_id} not found")

        suggestions = []

        # Consider the opposite
        suggestions.append(DebiasingStrategy.consider_the_opposite(constraint))

        # Devil's advocate challenges
        challenges = DebiasingStrategy.devils_advocate(constraint)
        suggestions.extend([f"Challenge: {c}" for c in challenges])

        # Forced reformulations
        reformulations = DebiasingStrategy.forced_reformulation(constraint)
        suggestions.extend(reformulations)

        return suggestions

    def _log_bias_detection(
        self,
        event_type: str,
        constraint_id: str,
        report: BiasReport
    ) -> None:
        """Log bias detection to file or console"""
        log_entry = {
            "event_type": event_type,
            "constraint_id": constraint_id,
            "bias_score": report.overall_bias_score,
            "detections": len(report.detections),
            "recommendations": report.recommendations[:3]
        }

        if self.config.log_path:
            # Write to file
            log_file = Path(self.config.log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        else:
            # Print to console
            print(f"[Φ₂ LOG] {log_entry}")

    def get_integration_statistics(self) -> Dict:
        """Get statistics about the integration"""
        stats = self.detector.get_statistics()

        # Add SCE-specific stats
        stats["sce_constraints_analyzed"] = len(self.sce.get_all_constraints())
        stats["bias_reports_generated"] = len(self.bias_history)

        if self.bias_history:
            latest = self.bias_history[-1]
            stats["latest_bias_score"] = latest.overall_bias_score
            stats["latest_detections"] = latest.total_detections

        return stats


class Stage5Phi2Monitor:
    """
    Monitors solution generation for cognitive biases (Stage 5 integration).

    This class provides real-time bias monitoring during the solution
    generation phase, detecting biased reasoning patterns and suggesting
    debiasing interventions.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        Initialize the Stage 5 monitor.

        Args:
            config: Optional configuration for monitoring
        """
        self.config = config or IntegrationConfig()
        self.detector = CognitiveBiasDetector()
        self.generation_history: List[Dict] = []
        self.current_bias_score: float = 0.0

    def monitor_generation_step(
        self,
        step: int,
        reasoning: str,
        constraints: Optional[List[Constraint]] = None
    ) -> BiasReport:
        """
        Monitor a generation step for bias.

        Args:
            step: Step number
            reasoning: The reasoning text for this step
            constraints: Optional list of constraints being used

        Returns:
            BiasReport with analysis
        """
        # Create temporary constraint from reasoning for analysis
        temp_constraint = Constraint(
            id=f"generation_step_{step}",
            type=ConstraintType.SOFT,
            description=reasoning,
            formalization=f"step_{step}_reasoning",
            source="stage5_generation"
        )

        # Analyze for bias
        constraints_to_check = [temp_constraint]
        if constraints:
            constraints_to_check.extend(constraints)

        report = self.detector.analyze_constraints(constraints_to_check)

        # Update current bias score
        self.current_bias_score = report.overall_bias_score

        # Log
        log_entry = {
            "step": step,
            "bias_score": report.overall_bias_score,
            "detections": len(report.detections),
            "critical_biases": [
                d for d in report.detections
                if d.severity in [Severity.HIGH, Severity.CRITICAL]
            ]
        }
        self.generation_history.append(log_entry)

        # Warn if bias too high
        if report.overall_bias_score >= self.config.max_bias_score:
            print(
                f"[Φ₂ STAGE 5 WARNING] Step {step}: "
                f"Bias score {report.overall_bias_score:.2f} exceeds threshold"
            )
            print(f"  Critical biases: {len(log_entry['critical_biases'])}")

        return report

    def get_bias_trajectory(self) -> List[float]:
        """
        Get the bias score trajectory over generation steps.

        Returns:
            List of bias scores for each step
        """
        return [entry["bias_score"] for entry in self.generation_history]

    def get_step_recommendations(self, step: int) -> List[str]:
        """
        Get debiasing recommendations for a specific step.

        Args:
            step: Step number

        Returns:
            List of recommendations
        """
        if step < 0 or step >= len(self.generation_history):
            return []

        entry = self.generation_history[step]

        recommendations = []

        # Add general recommendations based on bias score
        if entry["bias_score"] > 0.7:
            recommendations.append(
                f"Step {step}: CRITICAL - Apply immediate debiasing intervention"
            )
        elif entry["bias_score"] > 0.4:
            recommendations.append(
                f"Step {step}: WARNING - Consider debiasing strategies"
            )

        # Add specific recommendations from detections
        for detection in entry["critical_biases"]:
            recommendations.append(
                f"Step {step}: {detection.bias_type.value} - {detection.suggestion}"
            )

        return recommendations

    def generate_debiased_alternatives(
        self,
        reasoning: str
    ) -> List[str]:
        """
        Generate debiased alternative reasoning.

        Args:
            reasoning: Original reasoning text

        Returns:
            List of debiased alternatives
        """
        # Create temporary constraint
        temp_constraint = Constraint(
            id="temp_reasoning",
            type=ConstraintType.SOFT,
            description=reasoning,
            formalization="temp_reasoning_formalization",
            source="stage5_generation"
        )

        # Apply debiasing strategies
        alternatives = []

        # Consider the opposite
        opposite = DebiasingStrategy.consider_the_opposite(temp_constraint)
        alternatives.append(opposite)

        # Devil's advocate
        challenges = DebiasingStrategy.devils_advocate(temp_constraint)
        alternatives.extend([f"Reconsider: {c}" for c in challenges])

        # Pre-mortem
        failure_modes = DebiasingStrategy.pre_mortem_analysis([temp_constraint])
        if failure_modes:
            alternatives.append(
                f"Pre-mortem: Address potential failures: {'; '.join(failure_modes[:2])}"
            )

        return alternatives

    def should_intervene(self, current_step: int) -> bool:
        """
        Determine if intervention is needed at current step.

        Args:
            current_step: Current generation step

        Returns:
            True if intervention is recommended
        """
        if current_step < 0 or current_step >= len(self.generation_history):
            return False

        entry = self.generation_history[current_step]

        # Intervene if:
        # 1. Bias score exceeds threshold
        # 2. Critical biases detected
        # 3. Bias score increasing over last few steps

        if entry["bias_score"] >= self.config.max_bias_score:
            return True

        if entry["critical_biases"]:
            return True

        # Check trend (last 3 steps)
        if current_step >= 3:
            recent_scores = [
                self.generation_history[i]["bias_score"]
                for i in range(current_step - 3, current_step + 1)
            ]
            # Check if monotonically increasing
            if all(recent_scores[i] < recent_scores[i+1] for i in range(len(recent_scores)-1)):
                if recent_scores[-1] > 0.4:  # Only if meaningful increase
                    return True

        return False

    def get_monitoring_statistics(self) -> Dict:
        """Get statistics about monitoring"""
        if not self.generation_history:
            return {
                "total_steps_monitored": 0,
                "current_bias_score": 0.0,
                "average_bias_score": 0.0
            }

        scores = [entry["bias_score"] for entry in self.generation_history]
        critical_count = sum(
            len(entry["critical_biases"]) for entry in self.generation_history
        )

        return {
            "total_steps_monitored": len(self.generation_history),
            "current_bias_score": self.current_bias_score,
            "average_bias_score": sum(scores) / len(scores),
            "max_bias_score": max(scores),
            "min_bias_score": min(scores),
            "total_critical_biases": critical_count,
            "interventions_recommended": sum(
                1 for i in range(len(self.generation_history))
                if self.should_intervene(i)
            )
        }


# ========================================
# DEMONSTRATION
# ========================================

if __name__ == "__main__":
    print("=" * 80)
    print("Φ₂ Integration Module - Demonstration")
    print("=" * 80)

    # Create SCE instance
    sce = SymbolicConstraintEngine()
    print("\n[OK] SCE initialized")

    # Create integration config
    config = IntegrationConfig(
        auto_check_on_add=True,
        auto_check_on_conflict=True,
        bias_threshold=0.4,
        max_bias_score=0.6,
        log_all_detections=True
    )

    # Create integrator
    integrator = SCEPhi2Integrator(sce, config)
    print("[OK] SCE-Φ₂ integrator initialized")

    # Add some test constraints (will be auto-checked for bias)
    print("\n" + "-" * 80)
    print("Adding test constraints (with automatic bias checking)...")
    print("-" * 80)

    test_constraints = [
        Constraint(
            id="c1",
            type=ConstraintType.HARD,
            description="The system will certainly achieve perfect accuracy",
            formalization="accuracy = 1.0",
            source="user_prompt"
        ),
        Constraint(
            id="c2",
            type=ConstraintType.HARD,
            description="This is clearly the best approach",
            formalization="best_approach = current",
            source="expert_opinion"
        ),
    ]

    for constraint in test_constraints:
        print(f"\nAdding: {constraint.id}")
        sce.add_constraint(constraint)

    # Check all constraints
    print("\n" + "-" * 80)
    print("Comprehensive bias check of all constraints...")
    print("-" * 80)

    report = integrator.check_all_constraints()
    print(f"Overall bias score: {report.overall_bias_score:.2f}")
    print(f"Total detections: {report.total_detections}")

    # Get biased constraints
    print("\n" + "-" * 80)
    print("Biased constraints (MEDIUM severity or higher)...")
    print("-" * 80)

    biased = integrator.get_biased_constraints(min_severity=Severity.MEDIUM)
    for constraint_id, detections in biased.items():
        print(f"\n{constraint_id}: {len(detections)} detections")
        for detection in detections[:2]:
            print(f"  - {detection.bias_type.value} [{detection.severity.name}]")

    # Suggest debiased formulations
    print("\n" + "-" * 80)
    print("Debiased formulation suggestions...")
    print("-" * 80)

    for constraint_id in ["c1", "c2"]:
        print(f"\n{constraint_id}:")
        suggestions = integrator.suggest_debiased_formulation(constraint_id)
        for suggestion in suggestions[:3]:
            print(f"  {suggestion}")

    # Demonstrate Stage 5 monitoring
    print("\n" + "=" * 80)
    print("Stage 5 Monitoring Demonstration")
    print("=" * 80)

    monitor = Stage5Phi2Monitor(config)
    print("\n[OK] Stage 5 monitor initialized")

    # Simulate generation steps
    print("\n" + "-" * 80)
    print("Simulating generation steps with bias monitoring...")
    print("-" * 80)

    generation_steps = [
        "We will definitely achieve the optimal solution",
        "Clearly, this approach is superior to alternatives",
        "In hindsight, our initial assumptions were correct",
        "This certainly leads to the best outcome",
    ]

    for step, reasoning in enumerate(generation_steps, 1):
        print(f"\nStep {step}: {reasoning[:60]}...")
        report = monitor.monitor_generation_step(step, reasoning)

        # Check if intervention needed
        if monitor.should_intervene(step - 1):  # step is 1-indexed
            print(f"  [⚠️] INTERVENTION RECOMMENDED")
            recommendations = monitor.get_step_recommendations(step - 1)
            for rec in recommendations[:2]:
                print(f"    - {rec}")

            # Generate debiased alternatives
            alternatives = monitor.generate_debiased_alternatives(reasoning)
            print(f"  [💡] Debiased alternatives:")
            for alt in alternatives[:2]:
                print(f"    - {alt[:80]}...")

    # Show monitoring statistics
    print("\n" + "-" * 80)
    print("Monitoring Statistics:")
    print("-" * 80)

    stats = monitor.get_monitoring_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show bias trajectory
    print("\n" + "-" * 80)
    print("Bias Score Trajectory:")
    print("-" * 80)

    trajectory = monitor.get_bias_trajectory()
    for step, score in enumerate(trajectory, 1):
        bar = "█" * int(score * 50)
        print(f"  Step {step}: {score:.2f} {bar}")

    # Integration statistics
    print("\n" + "=" * 80)
    print("Integration Statistics:")
    print("=" * 80)

    stats = integrator.get_integration_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value[:5]:
                print(f"  - {item}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("[OK] Φ₂ integration demonstration complete")
    print("=" * 80)
