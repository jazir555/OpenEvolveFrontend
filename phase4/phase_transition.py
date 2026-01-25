"""
Phase Transition Detector for Δ₃
==================================

Detects phase transitions in ACI history, specifically chaos → control transformations.

This module provides:
- Phase transition detection
- Discontinuity measurement
- Chaos-to-control identification

Author: Agent E3 (Δ₃ Specialist)
Date: 2025-12-31
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .aci_reduction_validator import (
    PhaseTransitionResult,
    Delta3Config
)


# =============================================================================
# PHASE TRANSITION DETECTOR
# =============================================================================

class PhaseTransitionDetector:
    """
    Detect phase transitions in ACI history.

    Implements statistical tests to identify significant phase transitions,
    particularly chaos → control transformations.
    """

    def __init__(self, config: Delta3Config):
        """
        Initialize phase transition detector.

        Args:
            config: Δ₃ configuration
        """
        self.config = config

    def detect(self, aci_history: List[float]) -> PhaseTransitionResult:
        """
        Detect phase transition in ACI history.

        Args:
            aci_history: ACI values through RESE stages

        Returns:
            PhaseTransitionResult
        """
        if len(aci_history) < 3:
            # Not enough data points
            return PhaseTransitionResult(
                phase_transition_detected=False,
                transition_point=None,
                aci_change=None,
                chaos_to_control=False,
                discontinuity_magnitude=None
            )

        # Calculate differences
        changes = self._calculate_changes(aci_history)

        # Find maximum change
        max_change_idx, max_change = self._find_maximum_change(changes)

        # Test if change is significant
        is_significant = self._test_significance(aci_history, max_change_idx, max_change)

        if not is_significant:
            return PhaseTransitionResult(
                phase_transition_detected=False,
                transition_point=None,
                aci_change=None,
                chaos_to_control=False,
                discontinuity_magnitude=None
            )

        # Check if chaos → control
        chaos_to_control = self._is_chaos_to_control(aci_history, max_change_idx)

        # Calculate discontinuity magnitude
        discontinuity = self._calculate_discontinuity(aci_history, max_change_idx)

        return PhaseTransitionResult(
            phase_transition_detected=True,
            transition_point=max_change_idx,
            aci_change=max_change,
            chaos_to_control=chaos_to_control,
            discontinuity_magnitude=discontinuity
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_changes(self, aci_history: List[float]) -> List[float]:
        """
        Calculate ACI changes between consecutive stages.

        Args:
            aci_history: ACI values

        Returns:
            List of changes (differences)
        """
        changes = []
        for i in range(len(aci_history) - 1):
            change = aci_history[i] - aci_history[i + 1]
            changes.append(change)

        return changes

    def _find_maximum_change(self, changes: List[float]) -> Tuple[int, float]:
        """
        Find index and value of maximum change.

        Args:
            changes: List of changes

        Returns:
            Tuple of (index, change_value)
        """
        max_idx = 0
        max_change = changes[0]

        for i, change in enumerate(changes[1:], start=1):
            if abs(change) > abs(max_change):
                max_change = change
                max_idx = i

        return max_idx, max_change

    def _test_significance(
        self,
        aci_history: List[float],
        change_idx: int,
        change: float
    ) -> bool:
        """
        Test if change is statistically significant.

        Uses z-test: is change > threshold * std(aci_history)?

        Args:
            aci_history: ACI values
            change_idx: Index of maximum change
            change: Maximum change value

        Returns:
            True if significant
        """
        # Calculate standard deviation
        std = np.std(aci_history)

        # Avoid division by zero
        if std < 1e-10:
            return abs(change) > 0.1

        # Calculate z-score
        z_score = abs(change) / std

        # Test against threshold
        return z_score > self.config.phase_transition_threshold

    def _is_chaos_to_control(
        self,
        aci_history: List[float],
        transition_idx: int
    ) -> bool:
        """
        Check if transition is chaos → control.

        Chaos → control is characterized by:
        - High initial ACI (chaos)
        - Low final ACI (control)
        - Significant decrease at transition

        Args:
            aci_history: ACI values
            transition_idx: Index of transition

        Returns:
            True if chaos → control
        """
        # Split at transition
        before = aci_history[:transition_idx + 1]
        after = aci_history[transition_idx + 1:]

        if len(before) == 0 or len(after) == 0:
            return False

        # Check means
        mean_before = np.mean(before)
        mean_after = np.mean(after)

        # Chaos → control: high ACI → low ACI
        return mean_before > mean_after

    def _calculate_discontinuity(
        self,
        aci_history: List[float],
        transition_idx: int
    ) -> float:
        """
        Calculate discontinuity magnitude at transition.

        Discontinuity = |ACI[transition] - ACI[transition+1]|

        Args:
            aci_history: ACI values
            transition_idx: Index of transition

        Returns:
            Discontinuity magnitude
        """
        if transition_idx < len(aci_history) - 1:
            return abs(aci_history[transition_idx] - aci_history[transition_idx + 1])
        else:
            return 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_chaos_to_control(aci_history: List[float]) -> bool:
    """
    Detect chaos → control transition.

    Args:
        aci_history: ACI values

    Returns:
        True if chaos → control detected
    """
    # Simple heuristic: monotonic decrease
    if len(aci_history) < 2:
        return False

    # Check if generally decreasing
    n_decreases = sum(1 for i in range(len(aci_history) - 1)
                     if aci_history[i] > aci_history[i + 1])

    ratio = n_decreases / (len(aci_history) - 1)

    return ratio > 0.7  # 70% of transitions are decreases


def calculate_transition_magnitude(aci_history: List[float]) -> float:
    """
    Calculate magnitude of largest transition.

    Args:
        aci_history: ACI values

    Returns:
        Magnitude of largest transition
    """
    if len(aci_history) < 2:
        return 0.0

    max_change = 0.0
    for i in range(len(aci_history) - 1):
        change = abs(aci_history[i] - aci_history[i + 1])
        max_change = max(max_change, change)

    return max_change


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PhaseTransitionDetector',
    'detect_chaos_to_control',
    'calculate_transition_magnitude',
]
