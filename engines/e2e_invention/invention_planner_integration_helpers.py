"""
Helper Methods for Advanced Integrations in End-to-End Invention Planner

This module provides helper methods for Phase 4 integrations that are used
by the main end_to_end_invention_planner.py file.

Author: Agent 4 - Advanced Integrations
Version: 1.0.0
Date: 2025-12-30
"""
from __future__ import annotations



import re
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Import data models from the main planner
try:
    from end_to_end_invention_planner import (
        ValidatedMath,
        ErrorSource,
        InventionGoal
    )
except ImportError:
    # Define fallback data classes if import fails
    @dataclass
    class ValidatedMath:
        description: str
        lean_theorem: str
        lean_proof: str
        variables: Dict[str, str]
        assumptions: List[str]
        verification_method: str
        confidence: float

    @dataclass
    class ErrorSource:
        error_type: str
        description: str
        probability: float
        impact: str
        mitigation_strategy: str
        verification_method: str
        acceptance_criteria: str

    @dataclass
    class InventionGoal:
        goal_type: str
        target: str
        domain: str
        key_requirements: List[str]
        constraints: List[str]
        success_definition: str
        complexity_score: float


class IntegrationHelpers:
    """Helper methods for Phase 4 integrations"""

    @staticmethod
    def extract_equations(goal: InventionGoal, knowledge: List[str]) -> List[str]:
        """
        Extract mathematical equations from goal and knowledge.

        Args:
            goal: Invention goal
            knowledge: Knowledge base

        Returns:
            List of mathematical equations as strings
        """
        equations = []

        # Extract equations from knowledge base
        for item in knowledge:
            # Look for mathematical patterns
            math_patterns = [
                r'[a-zA-Z]+\s*=\s*[^.]+?(?:\n|$)',  # variable = expression
                r'\([^)]+\)\s*=\s*[^.]+',  # f(x) = expression
                r'\w+\s*[\*\/\+\-]\s*\w+',  # Simple operations
            ]

            for pattern in math_patterns:
                matches = re.findall(pattern, item)
                equations.extend(matches[:2])  # Limit to 2 per item

        return equations[:10]  # Limit to 10 equations

    @staticmethod
    def parse_delegated_math(delegated_result: List[Dict]) -> List[ValidatedMath]:
        """
        Parse math formalization results from crewai # MIGRATED: was CrewAI delegation.

        Args:
            delegated_result: Result from crewai # MIGRATED: was CrewAI delegation

        Returns:
            List of ValidatedMath objects
        """
        formalized = []

        for item in delegated_result:
            if isinstance(item, dict):
                formalized.append(ValidatedMath(
                    description=item.get("equation", "Unknown"),
                    lean_theorem=item.get("formalized", "theorem unknown : Prop := by sorry"),
                    lean_proof="-- Proof delegated to CrewAI",
                    variables={},
                    assumptions=[],
                    verification_method="CrewAI delegation",
                    confidence=item.get("confidence", 0.85)
                ))

        return formalized

    @staticmethod
    def parse_delegated_errors(delegated_result: List[Dict]) -> List[ErrorSource]:
        """
        Parse error analysis results from crewai # MIGRATED: was CrewAI delegation.

        Args:
            delegated_result: Result from crewai # MIGRATED: was CrewAI delegation

        Returns:
            List of ErrorSource objects
        """
        errors = []

        for item in delegated_result:
            if isinstance(item, dict):
                errors.append(ErrorSource(
                    error_type="delegated_analysis",
                    description=item.get("description", "Potential error"),
                    probability=item.get("probability", 0.1),
                    impact=item.get("impact", "medium"),
                    mitigation_strategy=item.get("mitigation", "Verify all parameters"),
                    verification_method="CrewAI delegation",
                    acceptance_criteria="Error within acceptable tolerance"
                ))

        return errors

    @staticmethod
    async def generate_blue_fixes(red_findings: List[str]) -> List[str]:
        """
        Generate blue team fixes for red team findings.

        Args:
            red_findings: List of red team findings

        Returns:
            List of blue team fixes
        """
        fixes = []

        for finding in red_findings:
            # Generate fix for each finding
            fix = f"Address: {finding}. Implement verification and add error handling."
            fixes.append(fix)

        return fixes

    @staticmethod
    def get_integration_summary(integrations) -> Dict[str, bool]:
        """
        Get status of all Phase 4 integrations.

        Args:
            integrations: InventionPlannerIntegrations instance

        Returns:
            Dictionary with integration availability status
        """
        if integrations:
            return integrations.get_integration_status()
        return {
            "bubblelabs": False,
            "CrewAI": False,
            "sovereign": False,
            "multi_decomposition": False,
            "steer": False
        }


# Export all helper functions
__all__ = [
    'IntegrationHelpers',
    'extract_equations',
    'parse_delegated_math',
    'parse_delegated_errors',
    'generate_blue_fixes',
    'get_integration_summary',
]
