"""
LLTL Handoff Module for Symbolic Constraint Engine

Prepares constraints for handoff to Agent A2 (LLTL Specialist).
Provides translation from constraints to LLTL specifications.

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine
)


class LLTLTemplate(Enum):
    """LLTL template types"""
    SAFETY = "safety"  # []P (always P)
    LIVENESS = "liveness"  # <>P (eventually P)
    REACTIVITY = "reactivity"  # P -> <>Q (if P then eventually Q)
    BOUNDED_RESPONSE = "bounded_response"  # P ~> Q (P leads to Q within bound)
    PERSISTENCE = "persistence"  # <>[]P (eventually always P)


@dataclass
class LLTLSpecification:
    """
    LLTL specification derived from constraints.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        template: LLTL template type
        formula: LLTL formula string
        source_constraint: ID of source constraint
        priority: Priority level (1-3)
        variables: Variables used in formula
        assumptions: Assumptions about the system
        guarantees: Guarantees provided
    """
    id: str
    name: str
    template: LLTLTemplate
    formula: str
    source_constraint: str
    priority: int
    variables: List[str]
    assumptions: List[str]
    guarantees: List[str]

    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.assumptions is None:
            self.assumptions = []
        if self.guarantees is None:
            self.guarantees = []


@dataclass
class HandoffPackage:
    """
    Complete package for LLTL handoff.

    Attributes:
        constraints: All constraints
        ltl_specifications: Generated LLTL specifications
        translation_map: Mapping from constraints to LLTL specs
        metadata: Additional metadata
    """
    constraints: List[Constraint]
    ltl_specifications: List[LLTLSpecification]
    translation_map: Dict[str, str]  # constraint_id -> ltl_spec_id
    metadata: Dict[str, any]

    def __post_init__(self):
        if self.translation_map is None:
            self.translation_map = {}
        if self.metadata is None:
            self.metadata = {}


class LLTLHandoff:
    """
    Prepares constraints for LLTL handoff.

    Features:
    - Constraint → LLTL translation
    - Template selection
    - Variable extraction
    - Assumption/guarantee generation
    - Handoff package creation
    """

    # Mapping of constraint patterns to LLTL templates
    TEMPLATE_PATTERNS = {
        LLTLTemplate.SAFETY: [
            r'must\s+always',
            r'never\s+(?:be|occur|happen)',
            r'cannot\s+(?:be|occur|happen)',
            r'forbidden',
            r'required\s+to\s+always'
        ],

        LLTLTemplate.LIVENESS: [
            r'eventually',
            r'sooner\s+or\s+later',
            r'will\s+.*\s+(?:eventually|finally)',
            r'must\s+.*\s+(?:eventually|reach|achieve)'
        ],

        LLTLTemplate.REACTIVITY: [
            r'whenever\s+.*\s+then',
            r'if\s+.*\s+then\s+.*\s+eventually',
            r'leads\s+to',
            r'triggers'
        ],

        LLTLTemplate.BOUNDED_RESPONSE: [
            r'within\s+\d+\s+(?:seconds?|minutes?|steps?)',
            r'in\s+less\s+than',
            r'respond\s+within',
            r'bounded\s+by'
        ],

        LLTLTemplate.PERSISTENCE: [
            r'once\s+.*\s+always',
            r'stays\s+(?:forever|permanently)',
            r'eventually\s+always',
            r'converges\s+to'
        ]
    }

    # LLTL operator mappings
    OPERATOR_MAP = {
        "always": "[]",
        "eventually": "<>",
        "next": "X",
        "until": "U",
        "implies": "->",
        "and": "&&",
        "or": "||",
        "not": "!"
    }

    def __init__(self, sce: Optional[SymbolicConstraintEngine] = None):
        """
        Initialize LLTL handoff module.

        Args:
            sce: Optional existing constraint engine
        """
        self.sce = sce or SymbolicConstraintEngine()
        self._spec_counter = 0

    def prepare_handoff(self) -> HandoffPackage:
        """
        Prepare complete handoff package for LLTL.

        Returns:
            HandoffPackage with all necessary information
        """
        # Get all constraints
        constraints = self.sce.get_all_constraints()

        # Generate LLTL specifications
        ltl_specs = []
        translation_map = {}

        for constraint in constraints:
            specs = self.constraint_to_lltl(constraint)
            ltl_specs.extend(specs)

            # Map constraint to first generated spec
            if specs:
                translation_map[constraint.id] = specs[0].id

        # Create metadata
        metadata = {
            "total_constraints": len(constraints),
            "total_ltl_specs": len(ltl_specs),
            "hard_constraints": len([c for c in constraints if c.is_hard()]),
            "verified_constraints": len([c for c in constraints if c.is_verified()]),
            "has_dependencies": len([c for c in constraints if c.dependencies]),
            "template_distribution": self._get_template_distribution(ltl_specs)
        }

        return HandoffPackage(
            constraints=constraints,
            ltl_specifications=ltl_specs,
            translation_map=translation_map,
            metadata=metadata
        )

    def constraint_to_lltl(self, constraint: Constraint) -> List[LLTLSpecification]:
        """
        Convert a constraint to one or more LLTL specifications.

        Args:
            constraint: Python constraint

        Returns:
            List of LLTL specifications (can be multiple for complex constraints)
        """
        # Determine template
        template = self._select_template(constraint)

        # Generate LLTL formula
        formula = self._generate_lltl_formula(constraint, template)

        # Extract variables
        variables = self._extract_variables(constraint)

        # Generate assumptions and guarantees
        assumptions, guarantees = self._generate_assumptions_guarantees(constraint)

        # Calculate priority
        priority = self._calculate_priority(constraint)

        spec = LLTLSpecification(
            id=self._generate_spec_id(),
            name=f"LLTL for {constraint.id}",
            template=template,
            formula=formula,
            source_constraint=constraint.id,
            priority=priority,
            variables=variables,
            assumptions=assumptions,
            guarantees=guarantees
        )

        return [spec]

    def _select_template(self, constraint: Constraint) -> LLTLTemplate:
        """
        Select appropriate LLTL template for constraint.

        Args:
            constraint: Constraint to analyze

        Returns:
            LLTL template type
        """
        description = constraint.description.lower()

        # Score each template
        template_scores = {}
        for template, patterns in self.TEMPLATE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, description):
                    score += 1
            template_scores[template] = score

        # Return highest-scoring template (default to SAFETY)
        if template_scores:
            best_template = max(template_scores, key=template_scores.get)
            if template_scores[best_template] > 0:
                return best_template

        # Default selection based on constraint type
        if constraint.type == ConstraintType.HARD:
            return LLTLTemplate.SAFETY
        elif constraint.type == ConstraintType.SOFT:
            return LLTLTemplate.REACTIVITY
        else:
            return LLTLTemplate.LIVENESS

    def _generate_lltl_formula(
        self,
        constraint: Constraint,
        template: LLTLTemplate
    ) -> str:
        """
        Generate LLTL formula from constraint.

        Args:
            constraint: Source constraint
            template: LLTL template

        Returns:
            LLTL formula string
        """
        description = constraint.description
        formalization = constraint.formalization

        # Extract propositions
        proposition = self._extract_proposition(description)

        # Build formula based on template
        if template == LLTLTemplate.SAFETY:
            return f"[] ({proposition})"

        elif template == LLTLTemplate.LIVENESS:
            return f"<> ({proposition})"

        elif template == LLTLTemplate.REACTIVITY:
            # Pattern: "if P then eventually Q"
            trigger, response = self._split_reactivity(description)
            if trigger and response:
                return f"({trigger}) -> <> ({response})"
            else:
                return f"<> ({proposition})"

        elif template == LLTLTemplate.BOUNDED_RESPONSE:
            # Pattern: "P ~> Q" (P leads to Q within bound)
            trigger, response, bound = self._split_bounded_response(description)
            if trigger and response and bound:
                return f"({trigger}) ~>_{bound} ({response})"
            else:
                return f"<> ({proposition})"

        elif template == LLTLTemplate.PERSISTENCE:
            return f"<> [] ({proposition})"

        else:
            return proposition

    def _extract_proposition(self, description: str) -> str:
        """
        Extract atomic proposition from description.

        Args:
            description: Constraint description

        Returns:
            Proposition string
        """
        # Remove qualifiers
        prop = description

        # Remove common prefixes
        prefixes = [
            r'must\s+',
            r'should\s+',
            r'must\s+not\s+',
            r'cannot\s+',
            r'will\s+',
            r'the\s+system\s+'
        ]

        for prefix in prefixes:
            prop = re.sub(prefix, '', prop, flags=re.IGNORECASE)

        # Convert to logical form
        prop = prop.strip()

        # Replace spaces with underscores for variable names
        prop = prop.replace(' ', '_')

        return prop

    def _split_reactivity(self, description: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Split reactivity constraint into trigger and response.

        Args:
            description: Constraint description

        Returns:
            Tuple of (trigger, response)
        """
        # Look for patterns like "if X then Y" or "whenever X then Y"
        patterns = [
            r'if\s+(.+?)\s+then\s+(.+)',
            r'whenever\s+(.+?)\s+then\s+(.+)',
            r'when\s+(.+?)\s+then\s+(.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                trigger = self._extract_proposition(match.group(1))
                response = self._extract_proposition(match.group(2))
                return trigger, response

        return None, None

    def _split_bounded_response(
        self,
        description: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Split bounded response into trigger, response, and bound.

        Args:
            description: Constraint description

        Returns:
            Tuple of (trigger, response, bound)
        """
        # Look for patterns like "respond to X within Y seconds"
        pattern = r'(?:respond\s+to|when)\s+(.+?)\s+within\s+(\d+)\s+(\w+)'

        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            trigger = self._extract_proposition(match.group(1))
            bound = match.group(2)
            unit = match.group(3)
            response = f"response_{trigger}"
            return trigger, response, f"{bound}_{unit}"

        return None, None, None

    def _extract_variables(self, constraint: Constraint) -> List[str]:
        """
        Extract variables from constraint.

        Args:
            constraint: Constraint to analyze

        Returns:
            List of variable names
        """
        variables = []

        # Extract from formalization
        formal = constraint.formalization

        # Look for quantified variables
        matches = re.findall(r'(?:forall|∀|exists|∃)\s+(\w+)', formal)
        variables.extend(matches)

        # Look for single-letter variables (x, y, T, P, etc.)
        matches = re.findall(r'\b([a-zA-Z])\b', formal)
        variables.extend(matches)

        # Remove duplicates and filter
        variables = list(set([v for v in variables if len(v) <= 2]))

        return variables

    def _generate_assumptions_guarantees(
        self,
        constraint: Constraint
    ) -> Tuple[List[str], List[str]]:
        """
        Generate assumptions and guarantees from constraint.

        Args:
            constraint: Constraint to analyze

        Returns:
            Tuple of (assumptions, guarantees)
        """
        assumptions = []
        guarantees = []

        # Extract assumptions from dependencies
        for dep_id in constraint.dependencies:
            dep = self.sce.get_constraint(dep_id)
            if dep:
                assumptions.append(f"Constraint {dep_id}: {dep.description}")

        # The constraint itself is a guarantee
        guarantees.append(constraint.description)

        # Add formalization as guarantee
        if constraint.formalization:
            guarantees.append(f"Formal: {constraint.formalization}")

        return assumptions, guarantees

    def _calculate_priority(self, constraint: Constraint) -> int:
        """
        Calculate priority for LLTL spec (1-3).

        Args:
            constraint: Source constraint

        Returns:
            Priority level
        """
        if constraint.type == ConstraintType.HARD:
            return 3
        elif constraint.type == ConstraintType.SOFT:
            return 2
        else:
            return 1

    def _generate_spec_id(self) -> str:
        """Generate unique specification ID"""
        self._spec_counter += 1
        return f"ltl_spec_{self._spec_counter}"

    def _get_template_distribution(
        self,
        specs: List[LLTLSpecification]
    ) -> Dict[str, int]:
        """
        Get distribution of template usage.

        Args:
            specs: List of LLTL specifications

        Returns:
            Dictionary of template -> count
        """
        distribution = {template.value: 0 for template in LLTLTemplate}

        for spec in specs:
            distribution[spec.template.value] += 1

        return distribution

    def export_to_json(self, filepath: str, package: HandoffPackage) -> None:
        """
        Export handoff package to JSON.

        Args:
            filepath: Output file path
            package: Handoff package to export
        """
        import json

        data = {
            "metadata": package.metadata,
            "constraints": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "description": c.description,
                    "formalization": c.formalization,
                    "source": c.source,
                    "dependencies": c.dependencies,
                    "verified": c.verified
                }
                for c in package.constraints
            ],
            "ltl_specifications": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "template": spec.template.value,
                    "formula": spec.formula,
                    "source_constraint": spec.source_constraint,
                    "priority": spec.priority,
                    "variables": spec.variables,
                    "assumptions": spec.assumptions,
                    "guarantees": spec.guarantees
                }
                for spec in package.ltl_specifications
            ],
            "translation_map": package.translation_map
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def create_example_translations(self) -> Dict[str, List[str]]:
        """
        Create example translations for documentation.

        Returns:
            Dictionary of constraint_type -> example LLTL formulas
        """
        examples = {
            "safety": [
                "[] (temperature < 1000)",
                "[] (pressure <= 10)",
                "! (system_failure)",
                "[] (request -> acknowledged)"
            ],
            "liveness": [
                "<> (request_processed)",
                "<> (system_ready)",
                "<> (goal_reached)"
            ],
            "reactivity": [
                "(request_sent) -> <> (request_received)",
                "(button_pressed) -> <> (action_executed)",
                "(error_detected) -> <> (error_handled)"
            ],
            "bounded_response": [
                "(request) ~>_5s (response)",
                "(alarm) ~>_1s (shutdown)",
                "(input) ~>_100ms (output)"
            ],
            "persistence": [
                "<> [] (system_stable)",
                "<> [] (connection_established)",
                "<> [] (temperature_regulated)"
            ]
        }

        return examples


# Convenience functions

def prepare_lltl_handoff(sce: SymbolicConstraintEngine) -> HandoffPackage:
    """
    Prepare LLTL handoff package (convenience function).

    Args:
        sce: Constraint engine with constraints

    Returns:
        HandoffPackage
    """
    handoff = LLTLHandoff(sce)
    return handoff.prepare_handoff()


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("LLTL Handoff Module - Demonstration")
    print("=" * 70)

    from symbolic_constraint_engine import SymbolicConstraintEngine

    # Create test constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_safety",
        type=ConstraintType.HARD,
        description="Temperature must always be below 1000°C",
        formalization="forall T : Real, T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="request_liveness",
        type=ConstraintType.HARD,
        description="Every request must eventually be processed",
        formalization="forall r : Request, eventually processed(r)",
        source="system_requirement"
    )

    c3 = Constraint(
        id="response_reactivity",
        type=ConstraintType.SOFT,
        description="When a request is received, it must be acknowledged within 5 seconds",
        formalization="forall r : Request, received(r) -> acknowledged(r) within 5",
        source="performance_requirement"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)

    # Create handoff module
    handoff = LLTLHandoff(sce)
    print("\n[OK] LLTL Handoff Module initialized")

    # Prepare handoff package
    print("\n" + "=" * 70)
    print("Preparing Handoff Package:")
    print("=" * 70)

    package = handoff.prepare_handoff()

    print(f"\nTotal constraints: {package.metadata['total_constraints']}")
    print(f"Total LLTL specs: {package.metadata['total_ltl_specs']}")
    print(f"Hard constraints: {package.metadata['hard_constraints']}")

    # Display LLTL specifications
    print("\n" + "=" * 70)
    print("Generated LLTL Specifications:")
    print("=" * 70)

    for spec in package.ltl_specifications:
        print(f"\n{spec.id}:")
        print(f"  Name: {spec.name}")
        print(f"  Template: {spec.template.value}")
        print(f"  Formula: {spec.formula}")
        print(f"  Priority: {spec.priority}")
        print(f"  Source: {spec.source_constraint}")

        if spec.variables:
            print(f"  Variables: {', '.join(spec.variables)}")

        if spec.assumptions:
            print(f"  Assumptions:")
            for assumption in spec.assumptions:
                print(f"    - {assumption}")

        if spec.guarantees:
            print(f"  Guarantees:")
            for guarantee in spec.guarantees:
                print(f"    - {guarantee}")

    # Show example translations
    print("\n" + "=" * 70)
    print("Example Translations:")
    print("=" * 70)

    examples = handoff.create_example_translations()
    for template_type, formulas in examples.items():
        print(f"\n{template_type.upper()}:")
        for formula in formulas[:3]:  # Show first 3
            print(f"  {formula}")

    # Export to JSON
    import tempfile
    temp_file = tempfile.mktemp(suffix='.json')
    handoff.export_to_json(temp_file, package)
    print(f"\n[OK] Exported handoff package to {temp_file}")

    print("\n" + "=" * 70)
    print("[OK] LLTL Handoff Module demonstration complete")
    print("=" * 70)
    print("\nReady for handoff to Agent A2 (LLTL Specialist)")
