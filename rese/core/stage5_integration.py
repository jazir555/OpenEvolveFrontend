"""
Stage 5 Integration: Real-time Loss Feedback

Integrates LLTL with End-to-End Stage 5 (Physics/Logic Validation) to provide
real-time constraint loss feedback during generation.

Enables:
- Real-time loss monitoring during generation
- Constraint violation detection
- Backpropagation to generator
- Adaptive generation based on constraint satisfaction

Author: Agent A2 (LLTL Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
Dependencies: LLTL, E2E Stage 5, PyTorch
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Import LLTL
from core.logic_to_loss_translation import (
    LogicToLossTranslator,
    LossFunction,
    LossAggregationMethod,
    create_lltl_from_sce,
)

from core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine,
)

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackMode(Enum):
    """Modes for providing feedback to the generator"""
    REALTIME = "realtime"           # Continuous feedback during generation
    BATCH = "batch"                 # Feedback after each batch
    ON_VIOLATION = "on_violation"   # Feedback only when violations occur
    ADAPTIVE = "adaptive"           # Adaptively provide feedback


class FeedbackStrategy(Enum):
    """Strategies for handling constraint violations"""
    STOP_ON_HARD = "stop_on_hard"               # Stop generation on hard constraint violation
    BACKPROPAGATE = "backpropagate"             # Backpropagate loss to generator
    REGENERATE = "regenerate"                   # Regenerate violating portions
    ADJUST_WEIGHTS = "adjust_weights"           # Adjust constraint weights
    IGNORE_PREFERENCE = "ignore_preference"     # Ignore preference violations


@dataclass
class GenerationState:
    """
    Represents the state of generation at a point in time.

    Attributes:
        step: Current generation step
        variables: Dictionary of generated variable values
        loss: Current total loss
        violations: Dictionary of constraint violations
        timestamp: Time of state capture
    """
    step: int
    variables: Dict[str, Union[torch.Tensor, np.ndarray]]
    loss: Union[torch.Tensor, np.ndarray, float]
    violations: Dict[str, Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step": self.step,
            "variables": {
                k: v.tolist() if isinstance(v, np.ndarray) else
                   v.detach().cpu().tolist() if (PYTORCH_AVAILABLE and isinstance(v, torch.Tensor)) else
                   v
                for k, v in self.variables.items()
            },
            "loss": float(self.loss) if not isinstance(self.loss, dict) else self.loss,
            "violations": self.violations,
            "timestamp": self.timestamp,
        }


@dataclass
class FeedbackSignal:
    """
    A feedback signal sent to the generator.

    Attributes:
        should_stop: Whether generation should stop
        should_adjust: Whether generator should adjust its output
        should_backpropagate: Whether to backpropagate the loss
        adjustment_hints: Hints for how to adjust
        loss_gradients: Gradient information for backpropagation
    """
    should_stop: bool = False
    should_adjust: bool = False
    should_backpropagate: bool = False
    adjustment_hints: Dict[str, Any] = field(default_factory=dict)
    loss_gradients: Optional[Dict[str, torch.Tensor]] = None


class Stage5Integration:
    """
    Integrates LLTL with Stage 5 for real-time constraint validation.

    This class monitors generation in real-time, computes constraint losses,
    and provides feedback to the generator to ensure constraint satisfaction.

    Key features:
    1. Real-time loss monitoring
    2. Constraint violation detection
    3. Feedback signal generation
    4. Backpropagation support
    5. Adaptive constraint weighting
    """

    def __init__(
        self,
        lltl: LogicToLossTranslator,
        sce: SymbolicConstraintEngine,
        feedback_mode: FeedbackMode = FeedbackMode.REALTIME,
        feedback_strategy: FeedbackStrategy = FeedbackStrategy.BACKPROPAGATE,
        violation_threshold: float = 0.01,
        max_violations: int = 3,
    ):
        """
        Initialize Stage 5 integration.

        Args:
            lltl: The Logic-to-Loss Translator
            sce: The Symbolic Constraint Engine
            feedback_mode: How often to provide feedback
            feedback_strategy: How to handle violations
            violation_threshold: Loss threshold for considering something violated
            max_violations: Maximum number of violations before stopping
        """
        self.lltl = lltl
        self.sce = sce
        self.feedback_mode = feedback_mode
        self.feedback_strategy = feedback_strategy
        self.violation_threshold = violation_threshold
        self.max_violations = max_violations

        # State tracking
        self.generation_history: List[GenerationState] = []
        self.current_step = 0
        self.violation_count = 0
        self.hard_violation_count = 0

        # Adaptive weighting
        self.adaptive_weights: Dict[str, float] = {}
        self.violation_history: Dict[str, List[float]] = {}

        # Statistics
        self._stats = {
            "total_steps": 0,
            "violations_detected": 0,
            "feedback_signals_sent": 0,
            "backpropagations": 0,
            "stops_triggered": 0,
        }

        logger.info(
            f"Stage 5 Integration initialized: mode={feedback_mode.value}, "
            f"strategy={feedback_strategy.value}"
        )

    def monitor_generation(
        self,
        variables: Dict[str, Union[torch.Tensor, np.ndarray]],
        step: Optional[int] = None,
    ) -> GenerationState:
        """
        Monitor a generation step and compute losses.

        Args:
            variables: Dictionary of generated variable values
            step: Optional step number (auto-incremented if None)

        Returns:
            GenerationState with current loss and violations
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Compute total loss
        total_loss = self.lltl.compute_total_loss(variables)

        # Get violations
        violations = self.lltl.get_loss_violations(variables)

        # Create state
        state = GenerationState(
            step=self.current_step,
            variables=variables,
            loss=total_loss,
            violations=violations,
        )

        # Store history
        self.generation_history.append(state)
        self._stats["total_steps"] += 1

        # Update violation history
        for cid, viol in violations.items():
            if viol.get("violated", False):
                if cid not in self.violation_history:
                    self.violation_history[cid] = []
                self.violation_history[cid].append(viol["loss"])
                self._stats["violations_detected"] += 1

        return state

    def generate_feedback(
        self,
        state: GenerationState,
    ) -> FeedbackSignal:
        """
        Generate feedback signal for the generator.

        Args:
            state: Current generation state

        Returns:
            FeedbackSignal with instructions for generator
        """
        signal = FeedbackSignal()

        # Check for hard constraint violations
        hard_violations = [
            (cid, viol)
            for cid, viol in state.violations.items()
            if viol.get("violated") and viol.get("type") == "hard"
        ]

        # Check for soft constraint violations
        soft_violations = [
            (cid, viol)
            for cid, viol in state.violations.items()
            if viol.get("violated") and viol.get("type") == "soft"
        ]

        # Determine if we should stop
        if hard_violations:
            self.hard_violation_count += 1
            self.violation_count += 1

            if self.feedback_strategy == FeedbackStrategy.STOP_ON_HARD:
                signal.should_stop = True
                signal.should_adjust = True
                signal.adjustment_hints = {
                    "reason": "hard_constraint_violation",
                    "violations": [cid for cid, _ in hard_violations],
                }
                self._stats["stops_triggered"] += 1

        # Check if we've exceeded max violations
        if self.violation_count >= self.max_violations:
            signal.should_stop = True
            signal.adjustment_hints = {
                "reason": "max_violations_exceeded",
                "violation_count": self.violation_count,
            }
            self._stats["stops_triggered"] += 1

        # Determine if we should backpropagate
        if self.feedback_strategy == FeedbackStrategy.BACKPROPAGATE:
            if hard_violations or soft_violations:
                signal.should_backpropagate = True

                # Compute gradients if using PyTorch
                if PYTORCH_AVAILABLE:
                    signal.loss_gradients = self._compute_gradients(state)

                self._stats["backpropagations"] += 1

        # Determine if we should regenerate
        if self.feedback_strategy == FeedbackStrategy.REGENERATE:
            if hard_violations:
                signal.should_adjust = True
                signal.adjustment_hints = {
                    "action": "regenerate",
                    "reason": "hard_constraint_violation",
                    "violations": [cid for cid, _ in hard_violations],
                }

        # Determine if we should adjust weights
        if self.feedback_strategy == FeedbackStrategy.ADJUST_WEIGHTS:
            self._update_adaptive_weights(state)
            signal.should_adjust = True
            signal.adjustment_hints = {
                "action": "adjust_weights",
                "new_weights": self.adaptive_weights.copy(),
            }

        # Send feedback signal if needed
        if signal.should_stop or signal.should_adjust or signal.should_backpropagate:
            self._stats["feedback_signals_sent"] += 1

        return signal

    def _compute_gradients(
        self,
        state: GenerationState,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for backpropagation.

        Args:
            state: Current generation state

        Returns:
            Dictionary mapping variable names to gradients
        """
        if not PYTORCH_AVAILABLE:
            return {}

        gradients = {}

        for var_name, var_value in state.variables.items():
            if isinstance(var_value, torch.Tensor) and var_value.requires_grad:
                # Compute gradient of loss with respect to this variable
                loss = state.loss
                if isinstance(loss, torch.Tensor):
                    # Compute gradient
                    grad = torch.autograd.grad(
                        loss,
                        var_value,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]

                    if grad is not None:
                        gradients[var_name] = grad

        return gradients

    def _update_adaptive_weights(
        self,
        state: GenerationState,
    ):
        """
        Update adaptive weights based on violation history.

        Constraints with frequent violations get higher weights.
        """
        for cid, viol in state.violations.items():
            if viol.get("violated"):
                # Increase weight for violated constraints
                current_weight = self.adaptive_weights.get(cid, 1.0)
                violation_count = len(self.violation_history.get(cid, []))

                # Exponential increase
                new_weight = current_weight * (1.0 + 0.1 * violation_count)
                self.adaptive_weights[cid] = new_weight

                # Update LLTL weight
                if cid in self.lltl.loss_functions:
                    self.lltl.loss_functions[cid].weight = new_weight

    def get_generation_summary(self) -> Dict[str, Any]:
        """
        Get summary of generation process.

        Returns:
            Dictionary with generation statistics and violation summary
        """
        # Count violations by type
        hard_violations = 0
        soft_violations = 0
        pref_violations = 0

        for state in self.generation_history:
            for viol in state.violations.values():
                if viol.get("violated"):
                    vtype = viol.get("type", "unknown")
                    if vtype == "hard":
                        hard_violations += 1
                    elif vtype == "soft":
                        soft_violations += 1
                    elif vtype == "preference":
                        pref_violations += 1

        # Get final loss
        final_loss = None
        if self.generation_history:
            final_loss = self.generation_history[-1].loss

        return {
            "total_steps": self._stats["total_steps"],
            "final_loss": float(final_loss) if final_loss is not None else None,
            "violations_by_type": {
                "hard": hard_violations,
                "soft": soft_violations,
                "preference": pref_violations,
            },
            "total_violations": self._stats["violations_detected"],
            "feedback_signals_sent": self._stats["feedback_signals_sent"],
            "backpropagations": self._stats["backpropagations"],
            "stops_triggered": self._stats["stops_triggered"],
            "constraint_weights": self.adaptive_weights.copy(),
        }

    def export_history(self, filepath: str):
        """
        Export generation history to file.

        Args:
            filepath: Path to save history
        """
        import json

        history_data = [state.to_dict() for state in self.generation_history]

        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Exported {len(history_data)} generation steps to {filepath}")

    def reset(self):
        """Reset generation state and history"""
        self.generation_history.clear()
        self.current_step = 0
        self.violation_count = 0
        self.hard_violation_count = 0
        self.adaptive_weights.clear()
        self.violation_history.clear()

        self._stats = {
            "total_steps": 0,
            "violations_detected": 0,
            "feedback_signals_sent": 0,
            "backpropagations": 0,
            "stops_triggered": 0,
        }

        logger.info("Stage 5 Integration reset")


class GeneratorValidator:
    """
    High-level API for validating generator output with constraints.

    Provides a simple interface for integrating constraint validation
    into any generator pipeline.
    """

    def __init__(
        self,
        sce: SymbolicConstraintEngine,
        feedback_mode: FeedbackMode = FeedbackMode.BATCH,
        stop_on_violation: bool = False,
    ):
        """
        Initialize generator validator.

        Args:
            sce: Symbolic Constraint Engine
            feedback_mode: When to provide feedback
            stop_on_violation: Whether to stop on hard constraint violations
        """
        # Create LLTL
        self.lltl = create_lltl_from_sce(sce)

        # Create Stage 5 integration
        strategy = (
            FeedbackStrategy.STOP_ON_HARD if stop_on_violation
            else FeedbackStrategy.BACKPROPAGATE
        )

        self.integration = Stage5Integration(
            lltl=self.lltl,
            sce=sce,
            feedback_mode=feedback_mode,
            feedback_strategy=strategy,
        )

    def validate_step(
        self,
        variables: Dict[str, Union[torch.Tensor, np.ndarray]],
        step: Optional[int] = None,
    ) -> Tuple[bool, GenerationState, FeedbackSignal]:
        """
        Validate a single generation step.

        Args:
            variables: Generated variables
            step: Optional step number

        Returns:
            Tuple of (should_continue, state, feedback_signal)
        """
        # Monitor generation
        state = self.integration.monitor_generation(variables, step)

        # Generate feedback
        signal = self.integration.generate_feedback(state)

        # Determine if we should continue
        should_continue = not signal.should_stop

        return should_continue, state, signal

    def validate_batch(
        self,
        batch_variables: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
    ) -> List[Tuple[bool, GenerationState, FeedbackSignal]]:
        """
        Validate a batch of generation steps.

        Args:
            batch_variables: List of variable dictionaries

        Returns:
            List of (should_continue, state, feedback_signal) tuples
        """
        results = []

        for i, variables in enumerate(batch_variables):
            should_continue, state, signal = self.validate_step(variables, step=i)
            results.append((should_continue, state, signal))

            # Stop if signal says so
            if not should_continue:
                break

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return self.integration.get_generation_summary()

    def reset(self):
        """Reset validator state"""
        self.integration.reset()


# Convenience functions

def create_validator_from_constraints(
    constraints: List[Constraint],
    feedback_mode: FeedbackMode = FeedbackMode.BATCH,
) -> GeneratorValidator:
    """
    Create a GeneratorValidator from a list of constraints.

    Args:
        constraints: List of constraints
        feedback_mode: When to provide feedback

    Returns:
        Configured GeneratorValidator
    """
    # Create SCE
    sce = SymbolicConstraintEngine()

    for constraint in constraints:
        sce.add_constraint(constraint)

    # Create validator
    validator = GeneratorValidator(
        sce=sce,
        feedback_mode=feedback_mode,
    )

    return validator


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 5 Integration - Demonstration")
    print("=" * 70)

    # Check PyTorch availability
    print(f"\nPyTorch Available: {PYTORCH_AVAILABLE}")

    # Create test SCE
    from symbolic_constraint_engine import Constraint, ConstraintType, SymbolicConstraintEngine

    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall (T : Temperature), T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="min_pressure",
        type=ConstraintType.SOFT,
        description="Pressure should be above 5 bar",
        formalization="forall (P : Pressure), P > 5",
        source="system_inferred"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)

    print("\n[OK] SCE created with 2 constraints")

    # Create Stage 5 integration
    print("\n" + "=" * 70)
    print("Creating Stage 5 Integration...")
    print("=" * 70)

    lltl = create_lltl_from_sce(sce)

    integration = Stage5Integration(
        lltl=lltl,
        sce=sce,
        feedback_mode=FeedbackMode.REALTIME,
        feedback_strategy=FeedbackStrategy.BACKPROPAGATE,
    )

    print("[OK] Integration created")

    # Simulate generation
    print("\n" + "=" * 70)
    print("Simulating Generation...")
    print("=" * 70)

    if PYTORCH_AVAILABLE:
        # Step 1: Valid values
        print("\nStep 1: Valid values")
        variables = {
            "temperature": torch.tensor([750.0], requires_grad=True),
            "pressure": torch.tensor([8.0], requires_grad=True),
        }

        state = integration.monitor_generation(variables, step=1)
        signal = integration.generate_feedback(state)

        print(f"  Loss: {state.loss.item():.4f}")
        print(f"  Should stop: {signal.should_stop}")
        print(f"  Should adjust: {signal.should_adjust}")

        # Step 2: Violation
        print("\nStep 2: Temperature violation")
        variables = {
            "temperature": torch.tensor([1200.0], requires_grad=True),
            "pressure": torch.tensor([8.0], requires_grad=True),
        }

        state = integration.monitor_generation(variables, step=2)
        signal = integration.generate_feedback(state)

        print(f"  Loss: {state.loss.item():.4f}")
        print(f"  Should stop: {signal.should_stop}")
        print(f"  Should adjust: {signal.should_adjust}")

        if state.violations:
            print("  Violations:")
            for cid, viol in state.violations.items():
                if viol.get("violated"):
                    print(f"    {cid}: {viol['description']}")

    # Display summary
    print("\n" + "=" * 70)
    print("Generation Summary:")
    print("=" * 70)

    summary = integration.get_generation_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] Stage 5 Integration demonstration complete")
    print("=" * 70)
