"""
Stage 1 Integration for Symbolic Constraint Engine

Integrates SCE with E2E Stage 1 (Prompt Analysis).
Converts natural language invention prompts to formal constraints.

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine
)


@dataclass
class PromptAnalysis:
    """
    Result of analyzing an invention prompt.

    Attributes:
        raw_prompt: Original prompt text
        extracted_constraints: List of constraints extracted
        confidence: Confidence score (0-1)
        missing_info: Information that couldn't be extracted
    """
    raw_prompt: str
    extracted_constraints: List[Constraint]
    confidence: float
    missing_info: List[str]

    def __post_init__(self):
        if self.missing_info is None:
            self.missing_info = []


class Stage1Integrator:
    """
    Integrates SCE with Stage 1 prompt analysis.

    Features:
    - Natural language parsing
    - Constraint extraction from prompts
    - Constraint type inference
    - Dependency detection
    """

    # Patterns for constraint extraction
    PATTERNS = {
        # Hard constraints (must, required, shall)
        'hard': [
            r'must\s+be\s+(.+?)(?:\.|$)',
            r'required\s+to\s+be\s+(.+?)(?:\.|$)',
            r'shall\s+be\s+(.+?)(?:\.|$)',
            r'must\s+not\s+(.+?)(?:\.|$)',
            r'cannot\s+(.+?)(?:\.|$)'
        ],

        # Soft constraints (should, preferably, desirable)
        'soft': [
            r'should\s+be\s+(.+?)(?:\.|$)',
            r'preferably\s+(.+?)(?:\.|$)',
            r'it\s+is\s+desirable\s+to\s+(.+?)(?:\.|$)',
            r'would\s+prefer\s+(.+?)(?:\.|$)'
        ],

        # Preference constraints (nice to have, optional)
        'preference': [
            r'optional\s+(?:to\s+)?(.+?)(?:\.|$)',
            r'nice\s+to\s+have\s+(.+?)(?:\.|$)',
            r'if\s+possible\s+(.+?)(?:\.|$)',
            r'ideally\s+(.+?)(?:\.|$)'
        ]
    }

    # Domain-specific terminology
    DOMAIN_TERMS = {
        'temperature': ['temperature', 'temp', 'heat', 'thermal', 'celsius', 'fahrenheit', 'kelvin'],
        'pressure': ['pressure', 'bar', 'psi', 'pascal'],
        'time': ['time', 'duration', 'delay', 'latency', 'seconds', 'minutes', 'hours'],
        'cost': ['cost', 'price', 'expense', 'budget', 'dollar', 'euro'],
        'quality': ['quality', 'accuracy', 'precision', 'reliability', 'error'],
        'performance': ['performance', 'speed', 'throughput', 'efficiency', 'fast', 'slow']
    }

    def __init__(self, sce: Optional[SymbolicConstraintEngine] = None):
        """
        Initialize Stage 1 integrator.

        Args:
            sce: Optional existing constraint engine
        """
        self.sce = sce or SymbolicConstraintEngine()
        self._constraint_counter = 0

    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Analyze an invention prompt and extract constraints.

        Args:
            prompt: Natural language invention prompt

        Returns:
            PromptAnalysis with extracted constraints

        Example:
            Input: "The system must operate at temperatures below 1000°C
                   and should preferably cost less than $1000."

            Output: Two constraints:
                   - HARD: Temperature < 1000°C
                   - SOFT: Cost < $1000
        """
        self._constraint_counter = 0
        constraints = []
        missing_info = []

        # Extract constraints by type
        for constraint_type, patterns in self.PATTERNS.items():
            type_constraints = self._extract_constraints_of_type(
                prompt, patterns, ConstraintType(constraint_type)
            )
            constraints.extend(type_constraints)

        # Detect dependencies between constraints
        constraints = self._detect_dependencies(constraints, prompt)

        # Calculate confidence
        confidence = self._calculate_confidence(prompt, constraints)

        # Identify missing information
        missing_info = self._identify_missing_info(prompt, constraints)

        # Add constraints to SCE
        for constraint in constraints:
            try:
                self.sce.add_constraint(constraint)
            except ValueError as e:
                # Skip if constraint already exists
                pass

        return PromptAnalysis(
            raw_prompt=prompt,
            extracted_constraints=constraints,
            confidence=confidence,
            missing_info=missing_info
        )

    def _extract_constraints_of_type(
        self,
        prompt: str,
        patterns: List[str],
        constraint_type: ConstraintType
    ) -> List[Constraint]:
        """
        Extract constraints of a specific type from prompt.

        Args:
            prompt: Prompt text
            patterns: Regex patterns for this type
            constraint_type: Type of constraint

        Returns:
            List of extracted constraints
        """
        constraints = []

        for pattern in patterns:
            matches = re.finditer(pattern, prompt, re.IGNORECASE | re.MULTILINE)

            for match in matches:
                constraint_text = match.group(1).strip()

                # Convert to formal constraint
                formal = self._text_to_formal(constraint_text)

                # Create constraint
                constraint = Constraint(
                    id=self._generate_constraint_id(),
                    type=constraint_type,
                    description=constraint_text,
                    formalization=formal,
                    source="stage1_prompt_analysis"
                )

                constraints.append(constraint)

        return constraints

    def _text_to_formal(self, text: str) -> str:
        """
        Convert natural language constraint to formal representation.

        Args:
            text: Natural language constraint

        Returns:
            Formal constraint representation
        """
        # Detect domain and extract variables
        domain, variables = self._detect_domain(text)

        # Convert to logical form
        formal = text

        # Replace "less than" with "<"
        formal = re.sub(r'less\s+than\s+(\d+)', r'< \1', formal, flags=re.IGNORECASE)

        # Replace "greater than" with ">"
        formal = re.sub(r'greater\s+than\s+(\d+)', r'> \1', formal, flags=re.IGNORECASE)

        # Replace "equal to" with "="
        formal = re.sub(r'equal\s+to\s+(\d+)', r'= \1', formal, flags=re.IGNORECASE)

        # Replace "at most" with "<="
        formal = re.sub(r'at\s+most\s+(\d+)', r'≤ \1', formal, flags=re.IGNORECASE)

        # Replace "at least" with ">="
        formal = re.sub(r'at\s+least\s+(\d+)', r'≥ \1', formal, flags=re.IGNORECASE)

        # Add quantifier if variable detected
        if variables:
            var = variables[0]
            if "forall" not in formal.lower() and "∀" not in formal:
                formal = f"∀ {var} : {domain}, {formal}"

        return formal

    def _detect_domain(self, text: str) -> Tuple[str, List[str]]:
        """
        Detect domain and variables from constraint text.

        Args:
            text: Constraint text

        Returns:
            Tuple of (domain_type, variable_names)
        """
        text_lower = text.lower()

        for domain, terms in self.DOMAIN_TERMS.items():
            for term in terms:
                if term in text_lower:
                    # Extract variable name
                    if domain == "temperature":
                        return "Temperature", ["T"]
                    elif domain == "pressure":
                        return "Pressure", ["P"]
                    elif domain == "time":
                        return "Time", ["t"]
                    elif domain == "cost":
                        return "Cost", ["C"]
                    elif domain == "quality":
                        return "Real", ["Q"]
                    elif domain == "performance":
                        return "Performance", ["Perf"]

        # Default to Real domain
        return "Real", ["x"]

    def _generate_constraint_id(self) -> str:
        """Generate unique constraint ID"""
        self._constraint_counter += 1
        return f"stage1_constraint_{self._constraint_counter}"

    def _detect_dependencies(
        self,
        constraints: List[Constraint],
        prompt: str
    ) -> List[Constraint]:
        """
        Detect dependencies between constraints.

        Args:
            constraints: List of constraints
            prompt: Original prompt

        Returns:
            Constraints with dependencies added
        """
        # Look for dependency indicators in prompt
        dependency_patterns = [
            r'however',
            r'but',
            r'also',
            r'and',
            r'furthermore'
        ]

        # Simple heuristic: constraints mentioned later depend on earlier ones
        # if they're in the same domain
        for i, later_constraint in enumerate(constraints):
            for earlier_constraint in constraints[:i]:
                # Check if they're in the same domain
                if self._same_domain(later_constraint, earlier_constraint):
                    # Add dependency
                    if earlier_constraint.id not in later_constraint.dependencies:
                        later_constraint.dependencies.append(earlier_constraint.id)

        return constraints

    def _same_domain(self, c1: Constraint, c2: Constraint) -> bool:
        """Check if two constraints are in the same domain"""
        # Extract domain from formalization
        domain1 = self._extract_domain_from_formal(c1.formalization)
        domain2 = self._extract_domain_from_formal(c2.formalization)

        return domain1 == domain2

    def _extract_domain_from_formal(self, formal: str) -> str:
        """Extract domain type from formal constraint"""
        # Look for type annotations like ": Real", ": Temperature", etc.
        match = re.search(r':\s*(\w+)', formal)
        if match:
            return match.group(1)
        return "Real"  # Default

    def _calculate_confidence(
        self,
        prompt: str,
        constraints: List[Constraint]
    ) -> float:
        """
        Calculate confidence score for constraint extraction.

        Args:
            prompt: Original prompt
            constraints: Extracted constraints

        Returns:
            Confidence score (0-1)
        """
        if not constraints:
            return 0.0

        # Base score: number of constraints found
        score = min(len(constraints) * 0.2, 0.8)

        # Boost for clear language
        if any(word in prompt.lower() for word in ['must', 'required', 'shall']):
            score += 0.1

        # Boost for specific values (numbers, units)
        if re.search(r'\d+', prompt):
            score += 0.05

        # Boost for domain terminology
        domain_hits = sum(
            1 for terms in self.DOMAIN_TERMS.values()
            for term in terms
            if term in prompt.lower()
        )
        score += min(domain_hits * 0.02, 0.1)

        return min(score, 1.0)

    def _identify_missing_info(
        self,
        prompt: str,
        constraints: List[Constraint]
    ) -> List[str]:
        """
        Identify information that couldn't be extracted.

        Args:
            prompt: Original prompt
            constraints: Extracted constraints

        Returns:
            List of missing information items
        """
        missing = []

        # Check for vague requirements
        vague_patterns = [
            r'better',
            r'faster',
            r'improved',
            r'optimized',
            r'enhanced'
        ]

        for pattern in vague_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                if not any(pattern in c.description.lower() for c in constraints):
                    missing.append(f"Specific metric for '{pattern}' not quantified")

        # Check for missing units
        if re.search(r'\d+(?!\s*(degrees?|°|C|F|K|%|\$|€|£|seconds?|minutes?|hours?))', prompt):
            missing.append("Numerical values without units detected")

        return missing

    def get_constraints(self) -> List[Constraint]:
        """Get all constraints from the SCE"""
        return self.sce.get_all_constraints()

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about constraint extraction"""
        stats = self.sce.get_statistics()
        return stats


# Convenience functions

def analyze_invention_prompt(prompt: str) -> PromptAnalysis:
    """
    Analyze an invention prompt (convenience function).

    Args:
        prompt: Invention prompt text

    Returns:
        PromptAnalysis result
    """
    integrator = Stage1Integrator()
    return integrator.analyze_prompt(prompt)


def batch_analyze_prompts(prompts: List[str]) -> List[PromptAnalysis]:
    """
    Analyze multiple invention prompts.

    Args:
        prompts: List of prompt texts

    Returns:
        List of PromptAnalysis results
    """
    integrator = Stage1Integrator()
    results = []

    for prompt in prompts:
        result = integrator.analyze_prompt(prompt)
        results.append(result)

    return results


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 1 Integration - Demonstration")
    print("=" * 70)

    # Test prompts
    test_prompts = [
        """
        The thermal management system must operate at temperatures below 1000°C
        and shall maintain a pressure greater than 5 bar. The system should
        preferably cost less than $5000 to manufacture.
        """,

        """
        Our invention processes data in under 10 seconds. It must achieve
        an accuracy of at least 95% and would prefer to use less than 100W
        of power if possible.
        """,

        """
        The device needs to be faster than traditional methods while
        maintaining reliability. It cannot exceed the budget constraints.
        """
    ]

    integrator = Stage1Integrator()
    print("\n[OK] Stage 1 Integrator initialized")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"Analyzing Prompt {i}:")
        print("=" * 70)
        print(prompt.strip())

        result = integrator.analyze_prompt(prompt)

        print(f"\nExtracted {len(result.extracted_constraints)} constraints:")
        for j, constraint in enumerate(result.extracted_constraints, 1):
            print(f"\n  {j}. {constraint.id}")
            print(f"     Type: {constraint.type.value}")
            print(f"     Description: {constraint.description}")
            print(f"     Formalization: {constraint.formalization}")
            if constraint.dependencies:
                print(f"     Dependencies: {constraint.dependencies}")

        print(f"\nConfidence: {result.confidence:.2f}")

        if result.missing_info:
            print(f"\nMissing information:")
            for item in result.missing_info:
                print(f"  - {item}")

    # Overall statistics
    print(f"\n{'=' * 70}")
    print("Overall Statistics:")
    print("=" * 70)
    stats = integrator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] Stage 1 Integration demonstration complete")
    print("=" * 70)
