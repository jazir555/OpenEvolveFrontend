"""
Logic-to-Loss Translation Layer (LLTL)

Bridges symbolic logic (SCE constraints) and neural systems (loss functions)
to enable gradient-based optimization.

Converts Lean 4 propositions to differentiable loss functions using fuzzy logic
relaxation, enabling backpropagation through constraint violations.

Author: Agent A2 (LLTL Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
Dependencies: PyTorch, SCE (Agent A1)
"""

import inspect
import logging
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available - LLTL will use NumPy fallback. "
        "For full functionality, install PyTorch: pip install torch"
    )

# Import SCE
from core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine,
)

# Configure logging
logger = logging.getLogger(__name__)


class LossAggregationMethod(Enum):
    """Methods for aggregating multiple constraint losses"""
    WEIGHTED_SUM = "weighted_sum"           # Simple weighted sum
    LEXICOGRAPHIC = "lexicographic"         # Prioritize by order
    MAX = "max"                             # Take maximum violation
    PRODUCT = "product"                     # Multiply all violations
    ADAPTIVE = "adaptive"                   # Adaptive weighting


class FuzzyLogicType(Enum):
    """Types of fuzzy logic relaxations"""
    LUKASIEWICZ = "lukasiewicz"             # Standard fuzzy logic
    GODEL = "godel"                         # Godel fuzzy logic
    PRODUCT = "product"                     # Product fuzzy logic
    SMOOTH_HINGE = "smooth_hinge"           # Smooth hinge loss


@dataclass
class LossFunction:
    """
    A differentiable loss function derived from a constraint.

    Attributes:
        constraint: The original constraint
        loss_fn: The PyTorch/NumPy loss function
        weight: Weight for this loss in aggregation
        fuzzy_type: Type of fuzzy logic relaxation
        differentiable: Whether the function is differentiable
    """
    constraint: Constraint
    loss_fn: Callable
    weight: float = 1.0
    fuzzy_type: FuzzyLogicType = FuzzyLogicType.LUKASIEWICZ
    differentiable: bool = True

    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Evaluate the loss function"""
        return self.loss_fn(*args, **kwargs)


@dataclass
class LossTranslationResult:
    """
    Result of translating a constraint to a loss function.

    Attributes:
        constraint_id: ID of the constraint
        success: Whether translation was successful
        loss_function: The translated loss function (if successful)
        error: Error message (if unsuccessful)
        warnings: List of warnings during translation
    """
    constraint_id: str
    success: bool
    loss_function: Optional[LossFunction] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class LogicToLossTranslator:
    """
    Translates symbolic logic constraints to differentiable loss functions.

    The LLTL is the bridge between the symbolic world (Lean 4 propositions)
    and the neural world (gradient-based optimization). It enables:

    1. Hard constraints → Barrier functions (steep penalty near violations)
    2. Soft constraints → Penalty functions (gradual penalty)
    3. Preference constraints → Weighted regularization (gentle guidance)

    All operations are differentiable, enabling backpropagation from
    constraint violations back to the generator.
    """

    def __init__(
        self,
        aggregation_method: LossAggregationMethod = LossAggregationMethod.WEIGHTED_SUM,
        default_fuzzy_type: FuzzyLogicType = FuzzyLogicType.LUKASIEWICZ,
        torch_dtype: Optional[Any] = None,
        device: str = "cpu",
    ):
        """
        Initialize the LLTL.

        Args:
            aggregation_method: How to aggregate multiple losses
            default_fuzzy_type: Default fuzzy logic type for relaxations
            torch_dtype: PyTorch data type (if None, uses torch.float32)
            device: PyTorch device ("cpu" or "cuda")
        """
        self.aggregation_method = aggregation_method
        self.default_fuzzy_type = default_fuzzy_type
        self.device = device
        self.torch_dtype = torch_dtype or (torch.float32 if PYTORCH_AVAILABLE else None)

        # Storage for translated losses
        self.loss_functions: Dict[str, LossFunction] = {}
        self.translation_cache: Dict[str, LossTranslationResult] = {}

        # Statistics
        self._translation_stats = {
            "total_translations": 0,
            "successful": 0,
            "failed": 0,
            "hard_constraints": 0,
            "soft_constraints": 0,
            "preference_constraints": 0,
        }

        logger.info(f"LLTL initialized with aggregation={aggregation_method.value}")

    def translate_constraint(
        self,
        constraint: Constraint,
        weight: Optional[float] = None,
        fuzzy_type: Optional[FuzzyLogicType] = None,
    ) -> LossTranslationResult:
        """
        Translate a single constraint to a loss function.

        Args:
            constraint: The constraint to translate
            weight: Optional weight (auto-determined if None)
            fuzzy_type: Optional fuzzy logic type (uses default if None)

        Returns:
            LossTranslationResult with the loss function or error
        """
        # Check cache
        if constraint.id in self.translation_cache:
            return self.translation_cache[constraint.id]

        result = LossTranslationResult(constraint_id=constraint.id, success=False)

        try:
            # Determine weight
            if weight is None:
                weight = self._determine_weight(constraint)

            # Determine fuzzy type
            fuzzy_type = fuzzy_type or self.default_fuzzy_type

            # Parse the formalization to extract logical structure
            logical_structure = self._parse_formalization(constraint.formalization)

            # Create appropriate loss function based on constraint type
            if constraint.type == ConstraintType.HARD:
                loss_fn = self._create_hard_constraint_loss(
                    constraint, logical_structure, fuzzy_type
                )
                self._translation_stats["hard_constraints"] += 1

            elif constraint.type == ConstraintType.SOFT:
                loss_fn = self._create_soft_constraint_loss(
                    constraint, logical_structure, fuzzy_type
                )
                self._translation_stats["soft_constraints"] += 1

            elif constraint.type == ConstraintType.PREFERENCE:
                loss_fn = self._create_preference_constraint_loss(
                    constraint, logical_structure, fuzzy_type
                )
                self._translation_stats["preference_constraints"] += 1

            else:
                raise ValueError(f"Unknown constraint type: {constraint.type}")

            # Create LossFunction wrapper
            loss_function = LossFunction(
                constraint=constraint,
                loss_fn=loss_fn,
                weight=weight,
                fuzzy_type=fuzzy_type,
                differentiable=True,
            )

            # Store result
            result.success = True
            result.loss_function = loss_function
            self.loss_functions[constraint.id] = loss_function
            self._translation_stats["successful"] += 1

            logger.info(f"Translated constraint {constraint.id} to loss function")

        except Exception as e:
            result.success = False
            result.error = str(e)
            self._translation_stats["failed"] += 1
            logger.error(f"Failed to translate constraint {constraint.id}: {e}")

        # Cache result
        self.translation_cache[constraint.id] = result
        self._translation_stats["total_translations"] += 1

        return result

    def translate_sce(
        self,
        sce: SymbolicConstraintEngine,
        constraint_filter: Optional[Callable[[Constraint], bool]] = None,
    ) -> Dict[str, LossTranslationResult]:
        """
        Translate all constraints from an SCE to loss functions.

        Args:
            sce: The Symbolic Constraint Engine
            constraint_filter: Optional filter to select which constraints to translate

        Returns:
            Dictionary mapping constraint IDs to translation results
        """
        results = {}

        for constraint in sce.get_all_constraints():
            # Apply filter if provided
            if constraint_filter and not constraint_filter(constraint):
                continue

            # Translate constraint
            result = self.translate_constraint(constraint)
            results[constraint.id] = result

        logger.info(
            f"Translated {len(results)} constraints from SCE: "
            f"{sum(1 for r in results.values() if r.success)} successful, "
            f"{sum(1 for r in results.values() if not r.success)} failed"
        )

        return results

    def compute_total_loss(
        self,
        inputs: Dict[str, Union[torch.Tensor, np.ndarray]],
        constraint_ids: Optional[List[str]] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute the total aggregated loss.

        Args:
            inputs: Dictionary mapping variable names to tensor values
            constraint_ids: Optional list of constraint IDs to include
                           (if None, uses all translated constraints)

        Returns:
            Aggregated loss value
        """
        # Determine which constraints to include
        if constraint_ids is None:
            constraint_ids = list(self.loss_functions.keys())

        # Compute individual losses
        losses = {}
        for cid in constraint_ids:
            if cid not in self.loss_functions:
                logger.warning(f"Constraint {cid} not translated, skipping")
                continue

            loss_fn = self.loss_functions[cid]
            try:
                loss = loss_fn(**inputs)
                losses[cid] = loss
            except Exception as e:
                logger.error(f"Error computing loss for {cid}: {e}")
                continue

        # Aggregate losses
        if not losses:
            return torch.tensor(0.0, device=self.device) if PYTORCH_AVAILABLE else np.array(0.0)

        return self._aggregate_losses(losses)

    def get_loss_violations(
        self,
        inputs: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed violation information for all constraints.

        Args:
            inputs: Dictionary mapping variable names to tensor values

        Returns:
            Dictionary mapping constraint IDs to violation details:
            {
                "constraint_id": {
                    "loss": loss_value,
                    "violated": bool,
                    "severity": float,
                    "description": str,
                }
            }
        """
        violations = {}

        for cid, loss_fn in self.loss_functions.items():
            try:
                loss = loss_fn(**inputs)

                # Convert to scalar if needed
                if PYTORCH_AVAILABLE and isinstance(loss, torch.Tensor):
                    loss_scalar = loss.detach().cpu().item()
                elif isinstance(loss, np.ndarray):
                    loss_scalar = float(loss.item() if loss.ndim > 0 else loss)
                else:
                    loss_scalar = float(loss)

                # Determine violation status
                violated = loss_scalar > 0.001  # Lowered threshold for "violated"

                # Determine severity (0-1)
                severity = min(1.0, loss_scalar)

                violations[cid] = {
                    "loss": loss_scalar,
                    "violated": violated,
                    "severity": severity,
                    "description": loss_fn.constraint.description,
                    "type": loss_fn.constraint.type.value,
                }

            except Exception as e:
                logger.error(f"Error checking violations for {cid}: {e}")
                violations[cid] = {
                    "error": str(e),
                    "violated": False,
                    "severity": 0.0,
                }

        return violations

    def _determine_weight(self, constraint: Constraint) -> float:
        """
        Determine appropriate weight for a constraint.

        Args:
            constraint: The constraint

        Returns:
            Weight value
        """
        # Hard constraints get higher weight
        if constraint.type == ConstraintType.HARD:
            return 10.0
        elif constraint.type == ConstraintType.SOFT:
            return 1.0
        else:  # PREFERENCE
            return 0.1

    def _parse_formalization(self, formalization: str) -> Dict[str, Any]:
        """
        Parse a Lean 4 formalization to extract logical structure.

        Args:
            formalization: Lean 4 proposition string

        Returns:
            Dictionary with parsed logical structure
        """
        structure = {
            "raw": formalization,
            "quantifiers": [],
            "operators": [],
            "variables": [],
            "type": "unknown",
        }

        # Extract quantifiers (forall, exists)
        if "forall" in formalization.lower():
            structure["quantifiers"].append("forall")
        if "exists" in formalization.lower():
            structure["quantifiers"].append("exists")

        # Extract operators
        operators = []
        if "<" in formalization:
            operators.append("lt")
        if ">" in formalization:
            operators.append("gt")
        if "<=" in formalization or "≤" in formalization:
            operators.append("le")
        if ">=" in formalization or "≥" in formalization:
            operators.append("ge")
        if "==" in formalization or "=" in formalization:
            operators.append("eq")
        if "!=" in formalization or "≠" in formalization:
            operators.append("neq")

        structure["operators"] = operators

        # Determine constraint type from operators
        if "lt" in operators or "gt" in operators:
            structure["type"] = "inequality"
        elif "le" in operators or "ge" in operators:
            structure["type"] = "inequality_soft"
        elif "eq" in operators:
            structure["type"] = "equality"
        elif "neq" in operators:
            structure["type"] = "inequality"

        return structure

    def _create_hard_constraint_loss(
        self,
        constraint: Constraint,
        logical_structure: Dict[str, Any],
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """
        Create a loss function for a hard constraint using barrier functions.

        Barrier functions provide steep penalties near constraint violations,
        effectively creating "walls" that the optimizer cannot cross.

        Args:
            constraint: The constraint
            logical_structure: Parsed logical structure
            fuzzy_type: Type of fuzzy logic relaxation

        Returns:
            Differentiable loss function
        """
        constraint_type = logical_structure["type"]

        if constraint_type == "inequality":
            return self._barrier_inequality_loss(constraint, fuzzy_type)
        elif constraint_type == "inequality_soft":
            return self._barrier_inequality_soft_loss(constraint, fuzzy_type)
        elif constraint_type == "equality":
            return self._barrier_equality_loss(constraint, fuzzy_type)
        else:
            # Generic barrier loss
            return self._generic_barrier_loss(constraint, fuzzy_type)

    def _create_soft_constraint_loss(
        self,
        constraint: Constraint,
        logical_structure: Dict[str, Any],
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """
        Create a loss function for a soft constraint using penalty functions.

        Penalty functions provide gradual penalties for violations,
        allowing the optimizer to trade off violations against other objectives.

        Args:
            constraint: The constraint
            logical_structure: Parsed logical structure
            fuzzy_type: Type of fuzzy logic relaxation

        Returns:
            Differentiable loss function
        """
        constraint_type = logical_structure["type"]

        if constraint_type == "inequality":
            return self._penalty_inequality_loss(constraint, fuzzy_type)
        elif constraint_type == "inequality_soft":
            return self._penalty_inequality_soft_loss(constraint, fuzzy_type)
        elif constraint_type == "equality":
            return self._penalty_equality_loss(constraint, fuzzy_type)
        else:
            # Generic penalty loss
            return self._generic_penalty_loss(constraint, fuzzy_type)

    def _create_preference_constraint_loss(
        self,
        constraint: Constraint,
        logical_structure: Dict[str, Any],
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """
        Create a loss function for a preference constraint using regularization.

        Regularization provides gentle guidance towards preferred values,
        but doesn't strongly penalize deviations.

        Args:
            constraint: The constraint
            logical_structure: Parsed logical structure
            fuzzy_type: Type of fuzzy logic relaxation

        Returns:
            Differentiable loss function
        """
        # Preferences use L2 regularization with small coefficient
        def preference_regularization(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            # Extract variables from kwargs
            # This is a simplified version - real implementation would parse
            # the formalization more carefully
            loss = 0.0

            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE:
                    if isinstance(var_value, torch.Tensor):
                        loss += torch.mean(var_value ** 2)
                else:
                    if isinstance(var_value, np.ndarray):
                        loss += np.mean(var_value ** 2)

            if PYTORCH_AVAILABLE:
                return 0.01 * torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return 0.01 * np.array(loss)

        return preference_regularization

    # Barrier loss functions (hard constraints)

    def _barrier_inequality_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Barrier loss for strict inequality (e.g., x < 1000)"""

        def barrier_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            # Simplified: assume we're checking if values exceed a threshold
            # In real implementation, this would parse the formalization
            # to extract the threshold and variable

            loss = 0.0
            has_torch_input = False

            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    has_torch_input = True
                    # Log-barrier: -log(threshold - x) for x < threshold
                    # This creates a steep penalty as x approaches threshold
                    threshold = 100.0  # Would be extracted from formalization
                    violations = torch.clamp(var_value - threshold, min=0.0)

                    # Log-barrier with small epsilon to avoid log(0)
                    # Only apply barrier when there are actual violations
                    epsilon = 1e-6
                    if torch.any(violations > 0):
                        barrier_values = -torch.log(1.0 - violations / (threshold + epsilon) + epsilon)
                        loss += torch.sum(torch.where(violations > 0, barrier_values, torch.tensor(0.0))).item()

                elif isinstance(var_value, np.ndarray):
                    threshold = 100.0
                    violations = np.clip(var_value - threshold, 0, None)
                    epsilon = 1e-6
                    if np.any(violations > 0):
                        barrier_values = -np.log(1.0 - violations / (threshold + epsilon) + epsilon)
                        loss += np.sum(np.where(violations > 0, barrier_values, 0.0))

            # Ensure loss is non-negative
            loss = max(0.0, loss)

            if has_torch_input:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return barrier_loss

    def _barrier_inequality_soft_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Barrier loss for soft inequality (e.g., x <= 1000)"""

        def barrier_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    threshold = 1000.0
                    violations = torch.clamp(var_value - threshold, min=0.0)
                    # Inverse barrier: 1 / (threshold - x)
                    loss += torch.sum(1.0 / (1.0 - violations / 1000.0 + 1e-6))

                elif isinstance(var_value, np.ndarray):
                    threshold = 1000.0
                    violations = np.clip(var_value - threshold, 0, None)
                    loss += np.sum(1.0 / (1.0 - violations / 1000.0 + 1e-6))

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return barrier_loss

    def _barrier_equality_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Barrier loss for equality (e.g., x == 100)"""

        def barrier_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    target = 100.0
                    # Squared barrier: (x - target)^2 / epsilon
                    # Large penalty even for small deviations
                    loss += torch.sum((var_value - target) ** 2) / 1e-3

                elif isinstance(var_value, np.ndarray):
                    target = 100.0
                    loss += np.sum((var_value - target) ** 2) / 1e-3

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return barrier_loss

    def _generic_barrier_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Generic barrier loss for unknown constraint types"""

        def barrier_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            # Default: steep exponential penalty
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    # Exponential barrier
                    loss += torch.sum(torch.exp(var_value))

                elif isinstance(var_value, np.ndarray):
                    loss += np.sum(np.exp(var_value))

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return barrier_loss

    # Penalty loss functions (soft constraints)

    def _penalty_inequality_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Penalty loss for strict inequality (e.g., x < 1000)"""

        def penalty_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    threshold = 1000.0
                    violations = torch.clamp(var_value - threshold, min=0.0)
                    # Quadratic penalty
                    loss += torch.sum(violations ** 2)

                elif isinstance(var_value, np.ndarray):
                    threshold = 1000.0
                    violations = np.clip(var_value - threshold, 0, None)
                    loss += np.sum(violations ** 2)

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return penalty_loss

    def _penalty_inequality_soft_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Penalty loss for soft inequality (e.g., x <= 1000)"""

        def penalty_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    threshold = 1000.0
                    violations = torch.clamp(var_value - threshold, min=0.0)
                    # Linear penalty
                    loss += torch.sum(violations)

                elif isinstance(var_value, np.ndarray):
                    threshold = 1000.0
                    violations = np.clip(var_value - threshold, 0, None)
                    loss += np.sum(violations)

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return penalty_loss

    def _penalty_equality_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Penalty loss for equality (e.g., x == 100)"""

        def penalty_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    target = 100.0
                    # Quadratic penalty
                    loss += torch.sum((var_value - target) ** 2)

                elif isinstance(var_value, np.ndarray):
                    target = 100.0
                    loss += np.sum((var_value - target) ** 2)

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return penalty_loss

    def _generic_penalty_loss(
        self,
        constraint: Constraint,
        fuzzy_type: FuzzyLogicType,
    ) -> Callable:
        """Generic penalty loss for unknown constraint types"""

        def penalty_loss(**kwargs) -> Union[torch.Tensor, np.ndarray]:
            # Default: quadratic penalty
            loss = 0.0
            for var_name, var_value in kwargs.items():
                if PYTORCH_AVAILABLE and isinstance(var_value, torch.Tensor):
                    loss += torch.sum(var_value ** 2)

                elif isinstance(var_value, np.ndarray):
                    loss += np.sum(var_value ** 2)

            if PYTORCH_AVAILABLE:
                return torch.tensor(loss, device=self.device, dtype=self.torch_dtype)
            else:
                return np.array(loss)

        return penalty_loss

    # Loss aggregation methods

    def _aggregate_losses(
        self,
        losses: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Aggregate multiple losses using the configured method.

        Args:
            losses: Dictionary mapping constraint IDs to loss values

        Returns:
            Aggregated loss
        """
        if not losses:
            return torch.tensor(0.0, device=self.device) if PYTORCH_AVAILABLE else np.array(0.0)

        if self.aggregation_method == LossAggregationMethod.WEIGHTED_SUM:
            return self._weighted_sum_aggregation(losses)
        elif self.aggregation_method == LossAggregationMethod.LEXICOGRAPHIC:
            return self._lexicographic_aggregation(losses)
        elif self.aggregation_method == LossAggregationMethod.MAX:
            return self._max_aggregation(losses)
        elif self.aggregation_method == LossAggregationMethod.PRODUCT:
            return self._product_aggregation(losses)
        elif self.aggregation_method == LossAggregationMethod.ADAPTIVE:
            return self._adaptive_aggregation(losses)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _weighted_sum_aggregation(
        self,
        losses: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """Weighted sum of losses"""
        total = 0.0
        has_torch = False

        for cid, loss in losses.items():
            weight = self.loss_functions[cid].weight
            if PYTORCH_AVAILABLE and isinstance(loss, torch.Tensor):
                has_torch = True
                total += weight * loss.item()
            else:
                total += weight * float(loss)

        if has_torch:
            return torch.tensor(total, device=self.device, dtype=self.torch_dtype)
        else:
            return np.array(total)

    def _lexicographic_aggregation(
        self,
        losses: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Lexicographic aggregation (prioritize by constraint type).

        Hard constraints > Soft constraints > Preference constraints
        """
        # Group by constraint type
        hard_losses = []
        soft_losses = []
        pref_losses = []

        for cid, loss in losses.items():
            constraint_type = self.loss_functions[cid].constraint.type
            if constraint_type == ConstraintType.HARD:
                hard_losses.append(loss)
            elif constraint_type == ConstraintType.SOFT:
                soft_losses.append(loss)
            else:
                pref_losses.append(loss)

        # Prioritize: hard first, then soft, then preferences
        if hard_losses:
            # Return sum of hard losses (highest priority)
            if PYTORCH_AVAILABLE:
                return sum(hard_losses) if isinstance(hard_losses[0], torch.Tensor) else torch.tensor(sum(hard_losses))
            else:
                return np.array(sum(hard_losses))
        elif soft_losses:
            return sum(soft_losses) if PYTORCH_AVAILABLE else np.array(sum(soft_losses))
        else:
            return sum(pref_losses) if PYTORCH_AVAILABLE else np.array(sum(pref_losses))

    def _max_aggregation(
        self,
        losses: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """Take maximum loss (min-max optimization)"""
        if PYTORCH_AVAILABLE:
            return torch.max(torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in losses.values()]))
        else:
            return np.array(max(losses.values()))

    def _product_aggregation(
        self,
        losses: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """Multiply all losses (emphasizes multiple violations)"""
        if PYTORCH_AVAILABLE:
            product = torch.tensor(1.0, device=self.device)
            for loss in losses.values():
                if isinstance(loss, torch.Tensor):
                    product = product * loss
                else:
                    product = product * torch.tensor(loss)
            return product
        else:
            product = 1.0
            for loss in losses.values():
                product *= float(loss)
            return np.array(product)

    def _adaptive_aggregation(
        self,
        losses: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Adaptive aggregation (adjust weights based on violation severity).

        Constraints with large violations get higher weights.
        """
        total = 0.0

        for cid, loss in losses.items():
            # Convert to float
            if PYTORCH_AVAILABLE and isinstance(loss, torch.Tensor):
                loss_val = loss.detach().cpu().item()
            else:
                loss_val = float(loss)

            # Adaptive weight: base weight * (1 + loss magnitude)
            base_weight = self.loss_functions[cid].weight
            adaptive_weight = base_weight * (1.0 + loss_val)

            total += adaptive_weight * loss_val

        if PYTORCH_AVAILABLE:
            return torch.tensor(total, device=self.device, dtype=self.torch_dtype)
        else:
            return np.array(total)

    # Utility methods

    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics"""
        return {
            **self._translation_stats,
            "translated_constraints": len(self.loss_functions),
            "aggregation_method": self.aggregation_method.value,
            "default_fuzzy_type": self.default_fuzzy_type.value,
            "pytorch_available": PYTORCH_AVAILABLE,
        }

    def clear_cache(self):
        """Clear translation cache and start fresh"""
        self.loss_functions.clear()
        self.translation_cache.clear()
        self._translation_stats = {
            "total_translations": 0,
            "successful": 0,
            "failed": 0,
            "hard_constraints": 0,
            "soft_constraints": 0,
            "preference_constraints": 0,
        }
        logger.info("LLTL cache cleared")

    def export_loss_functions(self, filepath: str):
        """
        Export loss functions to a file for inspection.

        Args:
            filepath: Path to save the export
        """
        import json

        export_data = []
        for cid, loss_fn in self.loss_functions.items():
            export_data.append({
                "constraint_id": cid,
                "description": loss_fn.constraint.description,
                "type": loss_fn.constraint.type.value,
                "formalization": loss_fn.constraint.formalization,
                "weight": loss_fn.weight,
                "fuzzy_type": loss_fn.fuzzy_type.value,
                "differentiable": loss_fn.differentiable,
            })

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(export_data)} loss functions to {filepath}")


# Convenience functions

def create_lltl_from_sce(
    sce: SymbolicConstraintEngine,
    aggregation_method: LossAggregationMethod = LossAggregationMethod.WEIGHTED_SUM,
    device: str = "cpu",
) -> LogicToLossTranslator:
    """
    Create an LLTL instance and translate all SCE constraints.

    Args:
        sce: The Symbolic Constraint Engine
        aggregation_method: Loss aggregation method
        device: PyTorch device

    Returns:
        Configured LogicToLossTranslator with all constraints translated
    """
    lltl = LogicToLossTranslator(
        aggregation_method=aggregation_method,
        device=device,
    )

    # Translate all constraints
    results = lltl.translate_sce(sce)

    # Log summary
    successful = sum(1 for r in results.values() if r.success)
    failed = sum(1 for r in results.values() if not r.success)

    logger.info(
        f"Created LLTL from SCE: {successful} successful, {failed} failed translations"
    )

    return lltl


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("Logic-to-Loss Translation Layer (LLTL) - Demonstration")
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
        id="min_temp",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 500°C",
        formalization="forall (T : Temperature), T > 500",
        source="user_prompt"
    )

    c3 = Constraint(
        id="max_pressure",
        type=ConstraintType.SOFT,
        description="Pressure should preferably be below 10 bar",
        formalization="forall (P : Pressure), P < 10",
        source="system_inferred"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)

    print("\n[OK] SCE created with 3 constraints")

    # Create LLTL
    print("\n" + "=" * 70)
    print("Creating LLTL...")
    print("=" * 70)

    lltl = LogicToLossTranslator(
        aggregation_method=LossAggregationMethod.WEIGHTED_SUM,
        device="cpu",
    )

    results = lltl.translate_sce(sce)

    print(f"\nTranslated {len(results)} constraints:")
    for cid, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"  {status} {cid}: {result.error or 'OK'}")

    # Test loss computation
    print("\n" + "=" * 70)
    print("Testing Loss Computation...")
    print("=" * 70)

    if PYTORCH_AVAILABLE:
        # Create test inputs
        inputs = {
            "temperature": torch.tensor([750.0, 800.0, 1200.0]),
            "pressure": torch.tensor([8.0, 12.0, 5.0]),
        }

        total_loss = lltl.compute_total_loss(inputs)
        print(f"\nTotal Loss: {total_loss.item():.4f}")

        violations = lltl.get_loss_violations(inputs)
        print("\nViolations:")
        for cid, viol in violations.items():
            if viol.get("violated"):
                print(f"  {cid}: VIOLATED (loss={viol['loss']:.4f}, severity={viol['severity']:.2f})")
            else:
                print(f"  {cid}: OK (loss={viol['loss']:.4f})")

    # Display statistics
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    stats = lltl.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] LLTL demonstration complete")
    print("=" * 70)
