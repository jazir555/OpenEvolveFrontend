"""
Enhanced Continuous Math Detector (Phase 3)

Improvements over base detector:
- Ambiguity resolution using context analysis
- Multi-equation detection and parsing
- Context-aware classification
- Improved confidence scoring
- Equation relationship detection

Author: OpenEvolve
Created: 2026-01-09
Phase: 3 - Enhanced Detection
"""

import re
import logging
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# Import base detector
from continuous_math_detector import (
    ContinuousMathDetector,
    MathType,
    ProblemType,
    ScientificDomain,
    MathDetectionResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Data Structures
# ============================================================================

@dataclass
class EquationStructure:
    """Structure of a parsed equation"""
    dependent_var: str
    independent_vars: List[str]
    order: int
    is_linear: bool
    raw_equation: str
    equation_type: str


@dataclass
class EquationRelation:
    """Relationship between multiple equations"""
    relation_type: str  # "system", "sequential", "coupled", "independent"
    variables_shared: List[str]
    coupling_strength: float  # 0-1
    dependencies: List[str] = field(default_factory=list)


@dataclass
class EnhancedDetectionResult(MathDetectionResult):
    """Extended detection result with enhanced features"""
    equations_found: List[EquationStructure] = field(default_factory=list)
    equation_relations: Optional[EquationRelation] = None
    ambiguity_score: float = 0.0  # 0=unambiguous, 1=highly ambiguous
    context_keywords: List[str] = field(default_factory=list)
    alternative_interpretations: List[Dict[str, any]] = field(default_factory=list)


# ============================================================================
# Enhanced Detector
# ============================================================================

class EnhancedContinuousMathDetector(ContinuousMathDetector):
    """
    Enhanced math detector with ambiguity resolution and multi-equation support.

    Improvements:
    1. Detects multiple equations in text
    2. Analyzes relationships between equations
    3. Resolves ambiguities using context
    4. Provides alternative interpretations
    5. Better confidence scoring
    """

    def __init__(self):
        super().__init__()
        self._init_ambiguity_patterns()
        self._init_multi_equation_patterns()
        self._init_context_keywords()

    def _init_ambiguity_patterns(self):
        """Initialize patterns for detecting ambiguous expressions"""

        self.ambiguity_patterns = {
            'ode_pde_ambiguous': [
                # Could be ODE or PDE depending on context
                r'partial.*d\w+.*dt',  # "partial derivative with respect to t"
                r'd\w+/dt.*partial',  # Mix of notations
            ],
            'integral_derivative_ambiguous': [
                # Integral of derivative or derivative of integral
                r'integral.*derivative',
                r'd/dx.*integral',
            ],
            'domain_ambiguous': [
                # Could be multiple domains
                r'growth.*decay',  # Could be biology, physics, economics
                r'oscillation',  # Could be physics, engineering, biology
            ],
        }

        self.resolution_heuristics = {
            'physics_indicators': [
                'energy', 'momentum', 'force', 'velocity', 'acceleration',
                'wave', 'heat', 'temperature', 'quantum', 'schrodinger',
                'lagrangian', 'hamiltonian', 'newton'
            ],
            'biology_indicators': [
                'population', 'species', 'predator', 'prey', 'cell',
                'bacteria', 'virus', 'infection', 'epidemic', 'sir',
                'lotka', 'volterra', 'growth', 'decay', 'biological'
            ],
            'chemistry_indicators': [
                'reaction', 'concentration', 'rate', 'catalyst',
                'equilibrium', 'stoichiometry', 'molecule', 'atom'
            ],
            'engineering_indicators': [
                'control', 'feedback', 'stability', 'circuit', 'rlc',
                'signal', 'transfer', 'function'
            ],
            'economics_indicators': [
                'price', 'cost', 'demand', 'supply', 'utility',
                'profit', 'market', 'stock', 'option', 'black', 'scholes'
            ],
        }

    def _init_multi_equation_patterns(self):
        """Initialize patterns for detecting multiple equations"""

        self.multi_equation_separators = [
            r'\n\s*-\s*',  # Newline with dash
            r'\n\s*\n',  # Double newline
            r',\s*',  # Comma (for inline systems)
            r';\s*',  # Semicolon
            r'where',  # "where" keyword
            r'with',  # "with" keyword
            r'and',  # "and" keyword
            r's\.t\.',  # "such that"
        ]

        self.system_keywords = [
            'system', 'coupled', 'simultaneous', 'together',
            'satisfies', 'subject to'
        ]

    def _init_context_keywords(self):
        """Initialize context-aware keyword classifications"""

        self.context_domains = {
            'thermodynamics': ['heat', 'temperature', 'entropy', 'enthalpy', 'thermal'],
            'quantum_mechanics': ['quantum', 'wave', 'function', 'schrodinger', 'operator'],
            'electromagnetism': ['electric', 'magnetic', 'field', 'maxwell', 'charge'],
            'fluid_dynamics': ['fluid', 'flow', 'navier', 'stokes', 'reynolds'],
            'population_dynamics': ['population', 'growth', 'carrying', 'capacity', 'species'],
            'epidemiology': ['epidemic', 'infection', 'susceptible', 'infectious', 'recovered'],
            'chemical_kinetics': ['reaction', 'rate', 'concentration', 'catalyst', 'enzyme'],
            'control_theory': ['control', 'feedback', 'stability', 'pole', 'zero'],
            'finance': ['price', 'volatility', 'portfolio', 'option', 'derivative'],
        }

    # ==========================================================================
    # Enhanced Detection Methods
    # ==========================================================================

    def detect(self, text: str) -> EnhancedDetectionResult:
        """
        Enhanced detection with ambiguity resolution and multi-equation support.

        Args:
            text: Input text containing mathematics

        Returns:
            EnhancedDetectionResult with additional context and alternatives
        """
        # Step 1: Detect multiple equations
        equations = self._detect_equations(text)

        # Step 2: Base detection (from parent)
        base_result = super().detect(text)

        # Step 3: Analyze equation relationships
        relation = None
        if len(equations) > 1:
            relation = self._analyze_equation_relations(equations, text)

        # Step 4: Calculate ambiguity score
        ambiguity = self._calculate_ambiguity_score(text, base_result)

        # Step 5: Extract context keywords
        context = self._extract_context_keywords(text)

        # Step 6: Generate alternative interpretations
        alternatives = self._generate_alternatives(text, base_result, context)

        # Step 7: Enhance confidence based on context
        enhanced_confidence = self._enhance_confidence(
            base_result.confidence,
            context,
            ambiguity
        )

        # Step 8: Resolve domain with context
        enhanced_domain = self._resolve_domain_with_context(
            base_result.domain,
            context,
            text
        )

        # Create enhanced result
        return EnhancedDetectionResult(
            math_type=base_result.math_type,
            problem_type=base_result.problem_type,
            domain=enhanced_domain,
            confidence=enhanced_confidence,
            equations=base_result.equations,
            variables=base_result.variables,
            notation=base_result.notation,
            keywords=base_result.keywords,
            metadata=base_result.metadata,
            # Enhanced fields
            equations_found=equations,
            equation_relations=relation,
            ambiguity_score=ambiguity,
            context_keywords=context,
            alternative_interpretations=alternatives
        )

    # ==========================================================================
    # Multi-Equation Detection
    # ==========================================================================

    def _detect_equations(self, text: str) -> List[EquationStructure]:
        """Detect multiple equations in text"""

        equations = []

        # Try different separators
        for pattern in self.multi_equation_separators:
            parts = re.split(pattern, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Found multiple equations
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        # Extract equation structure
                        eq_structure = self._parse_equation_structure(part, i)
                        if eq_structure:
                            equations.append(eq_structure)
                break

        # If no separator found, try detecting multiple equations in one line
        if not equations:
            equations = self._detect_inline_equations(text)

        # Special handling: check for newline-separated derivatives
        # This catches cases like "dx/dt = x\ndy/dt = y" that weren't caught above
        if len(equations) <= 1:
            # Split on single newlines when followed by derivative pattern
            derivative_pattern = r'\n(d[a-zA-Z]+/d[a-zA-Z]+\s*=\s*[^$\n]+)'
            matches = re.findall(derivative_pattern, text, re.MULTILINE)

            if len(matches) > 1:
                equations = []
                for i, match in enumerate(matches):
                    eq_structure = self._parse_equation_structure(match.strip(), i)
                    if eq_structure:
                        equations.append(eq_structure)

        return equations

    def _parse_equation_structure(
        self,
        text: str,
        index: int
    ) -> Optional[EquationStructure]:
        """Parse structure of a single equation"""

        # Extract dependent variable
        dependent_var = self._extract_dependent_variable(text)

        # Extract independent variables
        independent_vars = self._extract_independent_variables(text)

        # Extract order
        order = self._extract_equation_order(text)

        # Extract linearity
        is_linear = self._check_linearity(text)

        return EquationStructure(
            dependent_var=dependent_var or f"y_{index}",
            independent_vars=independent_vars or ["x"],
            order=order,
            is_linear=is_linear,
            raw_equation=text.strip(),
            equation_type="unknown"
        )

    def _detect_inline_equations(self, text: str) -> List[EquationStructure]:
        """Detect multiple equations on a single line"""

        equations = []

        # Look for patterns like "dx/dt = ..., dy/dt = ..."
        ode_patterns = [
            r'(d\w+/dt\s*=\s*[^,;\n]+)',  # Match up to comma, semicolon, or newline
            r'(d\w+/dx\s*=\s*[^,;\n]+)',
            r'(d²\w+/dt²\s*=\s*[^,;\n]+)',
            r'(d\w+/dt\s*=\s*[^\n]+)',  # Match entire line for derivative
        ]

        for pattern in ode_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for i, match in enumerate(matches):
                eq_structure = self._parse_equation_structure(match.strip(), i)
                if eq_structure:
                    equations.append(eq_structure)

        return equations

    # ==========================================================================
    # Equation Relationship Analysis
    # ==========================================================================

    def _analyze_equation_relations(
        self,
        equations: List[EquationStructure],
        text: str
    ) -> EquationRelation:
        """Analyze relationships between multiple equations"""

        # Find shared variables
        all_vars = set()
        for eq in equations:
            all_vars.add(eq.dependent_var)
            all_vars.update(eq.independent_vars)

        shared_vars = []
        for var in all_vars:
            count = sum(1 for eq in equations if var in eq.independent_vars or var == eq.dependent_var)
            if count > 1:
                shared_vars.append(var)

        # Determine relation type
        relation_type = "independent"
        coupling_strength = 0.0

        # Check for system keywords
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in self.system_keywords):
            relation_type = "system"
            coupling_strength = 0.8
        elif any(word in text_lower for word in ['then', 'next', 'follows']):
            relation_type = "sequential"
            coupling_strength = 0.5
        elif len(shared_vars) > 0:
            # Check if equations truly depend on each other
            # by checking if one equation's dependent var appears in another
            truly_coupled = False
            for eq in equations:
                for other_eq in equations:
                    if eq != other_eq:
                        # Check if this eq's dependent var appears in other eq
                        if eq.dependent_var in other_eq.independent_vars:
                            truly_coupled = True
                            break

            if truly_coupled:
                relation_type = "coupled"
                coupling_strength = min(1.0, len(shared_vars) * 0.3)
            else:
                # Variables are shared but equations are independent
                relation_type = "independent"
                coupling_strength = 0.1

        # Dependencies
        dependencies = []
        if relation_type == "sequential":
            for i in range(len(equations) - 1):
                dependencies.append(f"eq{i} -> eq{i+1}")

        return EquationRelation(
            relation_type=relation_type,
            variables_shared=shared_vars,
            coupling_strength=coupling_strength,
            dependencies=dependencies
        )

    # ==========================================================================
    # Ambiguity Resolution
    # ==========================================================================

    def _calculate_ambiguity_score(
        self,
        text: str,
        result: MathDetectionResult
    ) -> float:
        """Calculate ambiguity score (0=clear, 1=ambiguous)"""

        ambiguity_score = 0.0

        # Check against ambiguity patterns
        for category, patterns in self.ambiguity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    ambiguity_score += 0.3

        # Low confidence increases ambiguity
        if result.confidence < 0.5:
            ambiguity_score += 0.4
        elif result.confidence < 0.7:
            ambiguity_score += 0.2

        # Unknown math type adds ambiguity
        if result.math_type == MathType.UNKNOWN:
            ambiguity_score += 0.3

        # Clear domain reduces ambiguity
        if result.domain != ScientificDomain.GENERAL:
            ambiguity_score *= 0.7

        # Very short text is more ambiguous
        if len(text.strip()) < 20:
            ambiguity_score += 0.2

        # Question marks indicate uncertainty
        if '?' in text:
            ambiguity_score += 0.2

        return min(1.0, ambiguity_score)

    def _resolve_domain_with_context(
        self,
        detected_domain: ScientificDomain,
        context: List[str],
        text: str
    ) -> ScientificDomain:
        """Resolve domain using context keywords"""

        # Count domain indicators regardless of current domain
        # (context can override low-confidence detection)
        domain_scores = {
            ScientificDomain.PHYSICS: 0,
            ScientificDomain.BIOLOGY: 0,
            ScientificDomain.CHEMISTRY: 0,
            ScientificDomain.ENGINEERING: 0,
            ScientificDomain.ECONOMICS: 0,
        }

        text_lower = text.lower()

        # Check resolution heuristics
        for domain, indicators in self.resolution_heuristics.items():
            domain_enum = getattr(ScientificDomain, domain.upper() + '_INDICATORS', ScientificDomain.GENERAL)
            if domain_enum == ScientificDomain.GENERAL:
                continue

            score = 0
            for indicator in indicators:
                if indicator in text_lower:
                    score += 1

            # Map to correct domain
            if domain == 'physics':
                domain_scores[ScientificDomain.PHYSICS] = score
            elif domain == 'biology':
                domain_scores[ScientificDomain.BIOLOGY] = score
            elif domain == 'chemistry':
                domain_scores[ScientificDomain.CHEMISTRY] = score
            elif domain == 'engineering':
                domain_scores[ScientificDomain.ENGINEERING] = score
            elif domain == 'economics':
                domain_scores[ScientificDomain.ECONOMICS] = score

        # Return domain with highest score
        # Priority: Biology > Physics > Chemistry > Engineering > Economics
        domain_priority = [
            ScientificDomain.BIOLOGY,
            ScientificDomain.PHYSICS,
            ScientificDomain.CHEMISTRY,
            ScientificDomain.ENGINEERING,
            ScientificDomain.ECONOMICS,
        ]

        max_score = max(domain_scores.values())
        if max_score > 0:
            # Return highest priority domain with max score
            for domain in domain_priority:
                if domain_scores[domain] == max_score:
                    return domain

        # Only use detected domain if no context found
        return detected_domain

    def _enhance_confidence(
        self,
        base_confidence: float,
        context: List[str],
        ambiguity: float
    ) -> float:
        """Enhance confidence score based on context"""

        # Context keywords increase confidence
        context_boost = len(context) * 0.02

        # Ambiguity decreases confidence
        ambiguity_penalty = ambiguity * 0.3

        enhanced = base_confidence + context_boost - ambiguity_penalty

        return max(0.0, min(1.0, enhanced))

    # ==========================================================================
    # Context Extraction
    # ==========================================================================

    def _extract_context_keywords(self, text: str) -> List[str]:
        """Extract domain-relevant context keywords"""

        context = []
        text_lower = text.lower()

        # Check context domain patterns
        for domain_name, keywords in self.context_domains.items():
            for keyword in keywords:
                if keyword in text_lower:
                    context.append(f"{domain_name}:{keyword}")

        # Check resolution heuristics
        for domain, indicators in self.resolution_heuristics.items():
            for indicator in indicators:
                if indicator in text_lower:
                    context.append(f"{domain}:{indicator}")

        return list(set(context))  # Remove duplicates

    # ==========================================================================
    # Alternative Interpretations
    # ==========================================================================

    def _generate_alternatives(
        self,
        text: str,
        result: MathDetectionResult,
        context: List[str]
    ) -> List[Dict[str, any]]:
        """Generate alternative interpretations for ambiguous cases"""

        alternatives = []

        # Alternative math types - check for ambiguous cases even if type is known
        if result.math_type == MathType.ODE:
            # Could be a PDE if multiple variables
            if len(result.variables) > 2:
                alternatives.append({
                    'math_type': MathType.PDE,
                    'reason': 'Multiple independent variables detected',
                    'confidence': 0.6
                })

        if result.math_type == MathType.INTEGRAL:
            # Could also be studying integral applications
            alternatives.append({
                'math_type': MathType.INTEGRAL,
                'reason': 'Integral with application context',
                'confidence': 0.5
            })

        if result.math_type == MathType.UNKNOWN:
            # Try specific types
            if 'integral' in text.lower():
                alternatives.append({
                    'math_type': MathType.INTEGRAL,
                    'reason': 'Integral notation detected',
                    'confidence': 0.7
                })

            if 'd/d' in text or 'derivative' in text.lower():
                alternatives.append({
                    'math_type': MathType.DERIVATIVE,
                    'reason': 'Derivative notation detected',
                    'confidence': 0.7
                })

        # Alternative domains
        if result.domain == ScientificDomain.GENERAL and context:
            # Suggest domains based on context
            domain_counts = {}
            for ctx in context:
                domain = ctx.split(':')[0]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Top 2 suggested domains
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            for domain, count in sorted_domains[:2]:
                alternatives.append({
                    'domain': domain,
                    'reason': f'Context indicators: {count}',
                    'confidence': min(0.8, 0.3 + count * 0.1)
                })

        # Always provide alternatives if confidence is low
        if result.confidence < 0.5 and not alternatives:
            alternatives.append({
                'math_type': MathType.UNKNOWN,
                'reason': 'Low confidence detection',
                'confidence': 0.3
            })

        return alternatives

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _extract_dependent_variable(self, text: str) -> Optional[str]:
        """Extract dependent variable from equation"""

        # Look for patterns like "dy/dt", "d²x/dt²"
        patterns = [
            r'd([a-zA-Z]+)/d[a-zA-Z]+',
            r'd²([a-zA-Z]+)/d[a-zA-Z]+²',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_independent_variables(self, text: str) -> Optional[List[str]]:
        """Extract independent variables from equation"""

        # Look for variables after d/d
        matches = re.findall(r'd/d([a-zA-Z]+)', text)
        if matches:
            return list(set(matches))

        return None

    def _extract_equation_order(self, text: str) -> int:
        """Extract order of differential equation"""

        if 'd²' in text or 'd2' in text or "''.*" in text:
            return 2
        elif 'd³' in text or 'd3' in text:
            return 3
        elif 'd/d' in text or "'" in text:
            return 1
        else:
            return 0

    def _check_linearity(self, text: str) -> bool:
        """Check if equation appears linear"""

        # Simple heuristic: no products of dependent variable or its derivatives
        # This is a basic check

        # Remove powers and products
        clean_text = re.sub(r'\^2|\^3|\*\*', '', text)

        # Check for common non-linear patterns
        non_linear_patterns = [
            r'\by\b\s*\*\s*\by',  # y * y
            r'\by\b\s*\^',  # y^
            r'sin\s*\(\s*y',  # sin(y)
            r'cos\s*\(\s*y',  # cos(y)
            r'exp\s*\(\s*y',  # exp(y)
            r'log\s*\(\s*y',  # log(y)
        ]

        for pattern in non_linear_patterns:
            if re.search(pattern, text):
                return False

        return True


# ============================================================================
# Convenience Functions
# ============================================================================

def detect_continuous_math_enhanced(text: str) -> EnhancedDetectionResult:
    """
    Convenience function for enhanced detection.

    Args:
        text: Input text containing mathematics

    Returns:
        EnhancedDetectionResult with ambiguity resolution and multi-equation support
    """
    detector = EnhancedContinuousMathDetector()
    return detector.detect(text)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Solve dy/dt = y, this is a simple exponential growth",
        "System: dx/dt = x - xy, dy/dt = xy - y",
        "Is this a physics or biology problem: growth model",
    ]

    detector = EnhancedContinuousMathDetector()

    for text in test_texts:
        print(f"\nText: {text}")
        result = detector.detect(text)
        print(f"Math Type: {result.math_type}")
        print(f"Domain: {result.domain}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Ambiguity: {result.ambiguity_score:.2f}")
        print(f"Equations Found: {len(result.equations_found)}")
        if result.equation_relations:
            print(f"Relation Type: {result.equation_relations.relation_type}")
        if result.alternative_interpretations:
            print("Alternatives:")
            for alt in result.alternative_interpretations:
                print(f"  - {alt}")
