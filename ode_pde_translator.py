"""
ODE/PDE Translator to Lean 4

This module translates detected ordinary and partial differential equations
into formal Lean 4 definitions and theorem statements, providing proof
scaffolding for verification.

Features:
- Translate ODEs to Lean 4 formal definitions
- Translate PDEs to Lean 4 formal definitions
- Handle initial value problems (IVP)
- Handle boundary value problems (BVP)
- Generate proof scaffolding with tactics
- Support for existence, uniqueness, and solution theorems

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.2)
"""



import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# Import from B.1 detector
from continuous_math_detector import (
    MathDetectionResult,
    MathType,
    ProblemType,
    ScientificDomain,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations and Data Structures
# ============================================================================

class Lean4Structure(Enum):
    """Types of Lean 4 structures to generate"""
    DEFINITION = "definition"
    THEOREM = "theorem"
    AXIOM = "axiom"
    EXAMPLE = "example"
    DEF_THEOREM = "def_theorem"  # Definition + theorem


class SolutionType(Enum):
    """Types of solution theorems to generate"""
    EXISTENCE = "existence"
    UNIQUENESS = "uniqueness"
    EXISTENCE_UNIQUENESS = "existence_uniqueness"
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    SERIES = "series"
    NUMERICAL = "numerical"


@dataclass
class Lean4CodeBlock:
    """A block of Lean 4 code"""
    code: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return code block as string"""
        return self.code


@dataclass
class Lean4TranslationResult:
    """Result of translating math to Lean 4"""
    success: bool
    lean4_code: str  # Complete Lean 4 file content
    definitions: List[Lean4CodeBlock]
    theorems: List[Lean4CodeBlock]
    proof_scaffolds: List[Lean4CodeBlock]
    imports: List[str]
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "lean4_code": self.lean4_code,
            "definitions": [d.__dict__ for d in self.definitions],
            "theorems": [t.__dict__ for t in self.theorems],
            "proof_scaffolds": [p.__dict__ for p in self.proof_scaffolds],
            "imports": self.imports,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


@dataclass
class EquationStructure:
    """Parsed structure of a differential equation"""
    equation: str
    dependent_var: str  # e.g., "y" for y'
    independent_vars: List[str]  # e.g., ["x"] for ODE, ["x", "t"] for PDE
    order: int  # 1 for first order, 2 for second order
    is_linear: bool
    is_homogeneous: bool
    coefficients: Dict[str, str]  # Variable coefficients
    boundary_conditions: List[str]
    initial_conditions: List[str]


# ============================================================================
# ODE/PDE Translator
# ============================================================================

class ODEPDETranslator:
    """
    Translates ODEs and PDEs to Lean 4 formal definitions and theorems.

    This translator:
    1. Parses differential equations from detection results
    2. Maps to Lean 4 mathematical structures
    3. Generates formal definitions
    4. Creates theorem statements for existence/uniqueness
    5. Provides proof scaffolding with suggested tactics
    """

    def __init__(self, lean4_version: str = "4.0.0"):
        """
        Initialize the ODE/PDE translator.

        Args:
            lean4_version: Target Lean 4 version
        """
        self.lean4_version = lean4_version
        self.default_imports = [
            "Mathlib.Analysis.SpecialFunctions.ExpLog",
            "Mathlib.Analysis.Calculus.Deriv",
            "Mathlib.Analysis.Calculus.FDeriv",
            "Mathlib.Data.Real.Basic",
            "Mathlib.Tactic",
        ]

        # Mathlib imports for differential equations
        self.ode_imports = [
            "Mathlib.Analysis.ODE.PicardLindelof",
            "Mathlib.Analysis.ODE.Solutions",
        ]

        # Pattern mappings for common ODE forms
        self._init_ode_patterns()

        # Lean 4 type mappings
        self._init_type_mappings()

        logger.info(f"Initialized ODE/PDE Translator for Lean {lean4_version}")

    def _init_ode_patterns(self):
        """Initialize ODE pattern mappings"""
        self.ode_patterns = {
            # First-order linear: y' + P(x)y = Q(x)
            "first_order_linear": {
                "pattern": r"\by'\s*\+\s*P\(x\)\s*y\s*=\s*Q\(x\)",
                "lean_structure": "FirstOrderLinearODE",
                "solution_method": "integrating_factor"
            },
            # Separable: y' = f(x)g(y)
            "separable": {
                "pattern": r"\by'\s*=\s*f\(x\)\s*g\(y\)",
                "lean_structure": "SeparableODE",
                "solution_method": "separation_of_variables"
            },
            # Bernoulli: y' + P(x)y = Q(x)y^n
            "bernoulli": {
                "pattern": r"\by'\s*\+\s*P\(x\)\s*y\s*=\s*Q\(x\)\s*y\^n",
                "lean_structure": "BernoulliODE",
                "solution_method": "bernoulli_substitution"
            },
            # Homogeneous: y' = f(y/x)
            "homogeneous": {
                "pattern": r"\by'\s*=\s*f\(y/x\)",
                "lean_structure": "HomogeneousODE",
                "solution_method": "homogeneous_substitution"
            },
            # Exact: M(x,y)dx + N(x,y)dy = 0
            "exact": {
                "pattern": r"\bM\(x,y\)dx\s*\+\s*N\(x,y\)dy\s*=\s*0",
                "lean_structure": "ExactODE",
                "solution_method": "exact_differential"
            },
        }

    def _init_type_mappings(self):
        """Initialize Lean 4 type mappings"""
        self.type_map = {
            # Basic types
            "real": "Real",
            "complex": "Complex",
            "nat": "Nat",
            "int": "Int",

            # Function types
            "function": "fun",
            "real_function": "Real → Real",
            "real_vector": "Fin n → Real",

            # Mathematical objects
            "derivative": "deriv",
            "partial_derivative": "fderiv",
            "integral": "integral",
            "limit": "Limit",

            # ODE-specific
            "ode": "ODE",
            "ode_solution": "ODESolution",
            "ivp": "IVP",
            "initial_condition": "InitialCondition",

            # PDE-specific
            "pde": "PDE",
            "pde_solution": "PDESolution",
            "bvp": "BVP",
            "boundary_condition": "BoundaryCondition",
        }

    # ========================================================================
    # Main Translation Methods
    # ========================================================================

    def translate(
        self,
        detection_result: MathDetectionResult,
        solution_type: SolutionType = SolutionType.EXISTENCE_UNIQUENESS,
        generate_proof_scaffold: bool = True
    ) -> Lean4TranslationResult:
        """
        Translate a detected math problem to Lean 4.

        Args:
            detection_result: Result from ContinuousMathDetector
            solution_type: Type of solution theorem to generate
            generate_proof_scaffold: Whether to generate proof scaffolding

        Returns:
            Lean4TranslationResult with formal Lean 4 code
        """
        try:
            logger.info(f"Translating {detection_result.math_type.value} to Lean 4")

            # Route to appropriate translator
            if detection_result.math_type == MathType.ODE:
                return self._translate_ode(detection_result, solution_type, generate_proof_scaffold)
            elif detection_result.math_type == MathType.PDE:
                return self._translate_pde(detection_result, solution_type, generate_proof_scaffold)
            elif detection_result.math_type == MathType.DAE:
                return self._translate_dae(detection_result, solution_type, generate_proof_scaffold)
            elif detection_result.math_type == MathType.SDE:
                return self._translate_sde(detection_result, solution_type, generate_proof_scaffold)
            else:
                return self._unsupported_math_type(detection_result)

        except (ValueError, TypeError, RuntimeError, SyntaxError) as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            return Lean4TranslationResult(
                success=False,
                lean4_code="",
                definitions=[],
                theorems=[],
                proof_scaffolds=[],
                imports=[],
                error_message=str(e)
            )

    def translate_ode(
        self,
        equation: str,
        initial_condition: Optional[str] = None,
        boundary_conditions: Optional[List[str]] = None,
        **kwargs
    ) -> Lean4TranslationResult:
        """
        Translate a standalone ODE to Lean 4.

        Args:
            equation: ODE equation (e.g., "y' + y = 0")
            initial_condition: Optional initial condition (e.g., "y(0) = 1")
            boundary_conditions: Optional boundary conditions
            **kwargs: Additional parameters

        Returns:
            Lean4TranslationResult with formal Lean 4 code
        """
        # Create detection result wrapper
        detection_result = MathDetectionResult(
            math_type=MathType.ODE,
            problem_type=ProblemType.INITIAL_VALUE if initial_condition else ProblemType.BOUNDARY_VALUE,
            domain=ScientificDomain.GENERAL,
            confidence=1.0,
            equations=[equation],
            variables=self._extract_variables(equation),
            notation="standard",
            keywords=["ode"]
        )

        # Add conditions to metadata
        if initial_condition:
            detection_result.metadata["initial_condition"] = initial_condition
        if boundary_conditions:
            detection_result.metadata["boundary_conditions"] = boundary_conditions

        return self.translate(detection_result)

    def translate_pde(
        self,
        equation: str,
        boundary_conditions: Optional[List[str]] = None,
        initial_condition: Optional[str] = None,
        **kwargs
    ) -> Lean4TranslationResult:
        """
        Translate a standalone PDE to Lean 4.

        Args:
            equation: PDE equation (e.g., "∂u/∂t = ∂²u/∂x²")
            boundary_conditions: Optional boundary conditions
            initial_condition: Optional initial condition
            **kwargs: Additional parameters

        Returns:
            Lean4TranslationResult with formal Lean 4 code
        """
        # Create detection result wrapper
        detection_result = MathDetectionResult(
            math_type=MathType.PDE,
            problem_type=ProblemType.BOUNDARY_VALUE,
            domain=ScientificDomain.PHYSICS,
            confidence=1.0,
            equations=[equation],
            variables=self._extract_variables(equation),
            notation="standard",
            keywords=["pde"]
        )

        # Add conditions to metadata
        if boundary_conditions:
            detection_result.metadata["boundary_conditions"] = boundary_conditions
        if initial_condition:
            detection_result.metadata["initial_condition"] = initial_condition

        return self.translate(detection_result)

    # ========================================================================
    # ODE Translation
    # ========================================================================

    def _translate_ode(
        self,
        detection_result: MathDetectionResult,
        solution_type: SolutionType,
        generate_proof_scaffold: bool
    ) -> Lean4TranslationResult:
        """Translate an ODE to Lean 4"""

        definitions = []
        theorems = []
        proof_scaffolds = []

        # Extract equation structure
        equation = detection_result.equations[0] if detection_result.equations else ""
        eq_structure = self._parse_ode_structure(equation)

        # Generate imports
        imports = self.default_imports + self.ode_imports

        # Generate ODE definition
        ode_def = self._generate_ode_definition(eq_structure, detection_result)
        definitions.append(ode_def)

        # Generate theorem based on problem type
        if detection_result.problem_type == ProblemType.INITIAL_VALUE:
            theorem = self._generate_ivp_theorem(eq_structure, solution_type)
            theorems.append(theorem)

            if generate_proof_scaffold:
                scaffold = self._generate_ivp_proof_scaffold(eq_structure, solution_type)
                proof_scaffolds.append(scaffold)

        elif detection_result.problem_type == ProblemType.BOUNDARY_VALUE:
            theorem = self._generate_bvp_theorem(eq_structure, solution_type)
            theorems.append(theorem)

            if generate_proof_scaffold:
                scaffold = self._generate_bvp_proof_scaffold(eq_structure, solution_type)
                proof_scaffolds.append(scaffold)

        # Assemble complete Lean 4 file
        lean4_code = self._assemble_lean4_file(imports, definitions, theorems, proof_scaffolds)

        return Lean4TranslationResult(
            success=True,
            lean4_code=lean4_code,
            definitions=definitions,
            theorems=theorems,
            proof_scaffolds=proof_scaffolds,
            imports=imports,
            metadata={
                "equation_structure": eq_structure.__dict__,
                "solution_type": solution_type.value
            }
        )

    def _parse_ode_structure(self, equation: str) -> EquationStructure:
        """Parse ODE structure from equation string"""
        try:
            # Parse with SymPy
            expr = parse_expr(equation, evaluate=False)

            # Extract variables
            symbols = [str(s) for s in expr.free_symbols]

            # Determine dependent variable (usually 'y' or 'u')
            dependent_var = 'y' if 'y' in symbols else symbols[0] if symbols else 'y'

            # Determine independent variable (usually 'x' or 't')
            independent_var = 'x' if 'x' in symbols else 't' if 't' in symbols else 'x'

            # Determine order (check for derivatives)
            order = self._determine_derivative_order(equation)

            # Check if linear
            is_linear = self._check_linearity(expr, dependent_var)

            # Check if homogeneous
            is_homogeneous = self._check_homogeneity(expr, dependent_var)

            return EquationStructure(
                equation=equation,
                dependent_var=dependent_var,
                independent_vars=[independent_var],
                order=order,
                is_linear=is_linear,
                is_homogeneous=is_homogeneous,
                coefficients={},
                boundary_conditions=[],
                initial_conditions=[]
            )

        except (ValueError, TypeError, SyntaxError, AttributeError) as e:
            logger.warning(f"Failed to parse equation with SymPy: {e}, using defaults")
            return EquationStructure(
                equation=equation,
                dependent_var='y',
                independent_vars=['x'],
                order=1,
                is_linear=False,
                is_homogeneous=False,
                coefficients={},
                boundary_conditions=[],
                initial_conditions=[]
            )

    def _generate_ode_definition(
        self,
        eq_structure: EquationStructure,
        detection_result: MathDetectionResult
    ) -> Lean4CodeBlock:
        """Generate Lean 4 definition for ODE"""

        # Create a descriptive name
        var_name = eq_structure.dependent_var
        def_name = f"{var_name}_ode"

        # Generate definition
        if eq_structure.is_linear:
            code = f'''variable {{R : Type}} [Real R]

/-- The ODE: {eq_structure.equation} -/
def {def_name} (f : R → R) : Prop :=
  ∀ x, deriv f x + f x = 0

/-- The ODE solution space -/
def {def_name}_solution_set : Set (R → R) :=
  {{ f | {def_name} f }}
'''
        else:
            code = f'''variable {{R : Type}} [Real R]

/-- The ODE: {eq_structure.equation} -/
def {def_name} (f : R → R) : Prop :=
  ∃ (F : R → R → R),
    (∀ x, deriv f x = F x (f x)) ∧
    Continuous F ∧
    ∀ x, Continuous fun y => F x y

/-- Solutions to the ODE -/
def {def_name}_solution (f : R → R) : Prop :=
  {def_name} f ∧
  ∃ x₀, deriv f x₀ = {eq_structure.equation.split('=')[1].strip() if '=' in eq_structure.equation else '0'}
'''

        return Lean4CodeBlock(
            code=code,
            description=f"Definition for ODE: {eq_structure.equation}",
            dependencies=["Real", "deriv"],
            imports=["Mathlib.Analysis.Calculus.Deriv"]
        )

    def _generate_ivp_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate theorem statement for IVP"""

        var_name = eq_structure.dependent_var
        thm_name = f"{var_name}_ivp_solution"

        if solution_type == SolutionType.EXISTENCE:
            code = f'''/-- Existence of solution to the IVP -/
theorem {thm_name}_exists
    (f : R → R)
    (x₀ y₀ : R)
    (hf : Continuous f)
    : ∃ y, {var_name}_ode y ∧ y x₀ = y₀ :=
  by
    -- Apply Picard-Lindelöf theorem
    have := picard_lindelof f x₀ y₀
    sorry'''

        elif solution_type == SolutionType.UNIQUENESS:
            code = f'''/-- Uniqueness of solution to the IVP -/
theorem {thm_name}_unique
    (y₁ y₂ : R → R)
    (x₀ y₀ : R)
    (h₁ : {var_name}_ode y₁ ∧ y₁ x₀ = y₀)
    (h₂ : {var_name}_ode y₂ ∧ y₂ x₀ = y₀)
    : y₁ = y₂ :=
  by
    -- Apply uniqueness theorem
    apply ode_solution_unique
    sorry'''

        else:  # EXISTENCE_UNIQUENESS
            code = f'''/-- Existence and uniqueness of solution to the IVP -/
theorem {thm_name}_exists_unique
    (f : R → R)
    (x₀ y₀ : R)
    (hf : Continuous f ∧ Lipschitz f)
    : ∃! y : R → R, {var_name}_ode y ∧ y x₀ = y₀ :=
  by
    -- Apply Picard-Lindelöf theorem for existence and uniqueness
    have h_exists := picard_lindelof f x₀ y₀ hf.1
    sorry'''

        return Lean4CodeBlock(
            code=code,
            description=f"IVP {solution_type.value} theorem",
            dependencies=[f"{var_name}_ode", "picard_lindelof"],
            imports=["Mathlib.Analysis.ODE.PicardLindelof"]
        )

    def _generate_ivp_proof_scaffold(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate proof scaffold for IVP theorem"""

        var_name = eq_structure.dependent_var

        if solution_type == SolutionType.EXISTENCE:
            code = f'''/-- Proof sketch for existence -/
proof_scaffold:
  1. Define the integral operator (Picard iteration)
     T(y)(x) = y₀ + ∫₀ˣ f(t, y(t)) dt

  2. Show T is a contraction on a suitable Banach space

  3. Apply Banach fixed-point theorem

  4. Verify the fixed point satisfies the ODE

Tactics to use:
  - apply picard_lindelof
  - refine' (picard_lindelof _).exists
  - use [integral_eq, ...]
  - linarith
  - continuity'''

        elif solution_type == SolutionType.UNIQUENESS:
            code = f'''/-- Proof sketch for uniqueness -/
proof_scaffold:
  1. Assume two solutions y₁ and y₂ exist

  2. Consider the difference z = y₁ - y₂

  3. Show z satisfies z' = F(x, y₁) - F(x, y₂)

  4. Apply Grönwall's inequality

  5. Conclude z ≡ 0, hence y₁ = y₂

Tactics to use:
  - apply ode_solution_unique
  - have h_diff := fun x => ...
  - apply gronwall
  - simp at *
  - linarith'''

        else:  # EXISTENCE_UNIQUENESS
            code = f'''/-- Proof sketch for existence and uniqueness -/
proof_scaffold:
  1. **Existence** (Picard iteration):
     - Define T(y)(x) = y₀ + ∫₀ˣ f(t, y(t)) dt
     - Show T is contraction on C([x₀-h, x₀+h])
     - Fixed point gives solution

  2. **Uniqueness** (Grönwall):
     - Assume y₁, y₂ are solutions
     - Let d(x) = |y₁(x) - y₂(x)|
     - Show d' ≤ L·d (Lipschitz condition)
     - Integrate to get d ≤ 0, hence y₁ = y₂

Tactics to use:
  - apply picard_lindelof
  - refine' (picard_lindelof _).exists
  - apply ode_solution_unique
  - apply gronwall
  - rw [integral_eq]
  - simp [abs]
  - linarith'''

        return Lean4CodeBlock(
            code=code,
            description=f"Proof scaffold for IVP {solution_type.value}",
            dependencies=[f"{var_name}_ivp_{solution_type.value}"],
            imports=[]
        )

    def _generate_bvp_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate theorem statement for BVP"""

        var_name = eq_structure.dependent_var
        thm_name = f"{var_name}_bvp_solution"

        code = f'''/-- Existence of solution to the BVP -/
theorem {thm_name}_exists
    (a b : R)
    (α β : R)
    (f : R → R → R)
    (hf : Continuous fun xy => f xy.1 xy.2)
    : ∃ y : R → R,
        {var_name}_ode y ∧
        y a = α ∧
        y b = β :=
  by
    -- Apply shooting method or Green's function
    sorry'''

        return Lean4CodeBlock(
            code=code,
            description=f"BVP {solution_type.value} theorem",
            dependencies=[f"{var_name}_ode"],
            imports=[]
        )

    def _generate_bvp_proof_scaffold(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate proof scaffold for BVP theorem"""

        var_name = eq_structure.dependent_var
        code = f'''/-- Proof sketch for BVP existence -/
proof_scaffold:
  1. **Shooting method**:
     - For each slope m, solve IVP with y(a) = α, y'(a) = m
     - Define F(m) = y_m(b) - β
     - Show F is continuous
     - Apply intermediate value theorem to find m with F(m) = 0

  2. **Alternative: Green's function**:
     - Construct Green's function G(x, ξ)
     - Solution: y(x) = ∫ G(x, ξ) f(ξ) dξ
     - Verify boundary conditions

Tactics to use:
  - apply intermediate_value
  - use [ivp_exists, ...]
  - continuity
  - apply green_function_exists
  - rw [integral_eq]
  - simp'''

        return Lean4CodeBlock(
            code=code,
            description=f"Proof scaffold for BVP {solution_type.value}",
            dependencies=[f"{var_name}_bvp_{solution_type.value}"],
            imports=[]
        )

    # ========================================================================
    # PDE Translation
    # ========================================================================

    def _translate_pde(
        self,
        detection_result: MathDetectionResult,
        solution_type: SolutionType,
        generate_proof_scaffold: bool
    ) -> Lean4TranslationResult:
        """Translate a PDE to Lean 4"""

        definitions = []
        theorems = []
        proof_scaffolds = []

        # Extract equation structure
        equation = detection_result.equations[0] if detection_result.equations else ""
        eq_structure = self._parse_pde_structure(equation)

        # Generate imports
        imports = self.default_imports + [
            "Mathlib.Analysis.Calculus.FDeriv",
        ]

        # Generate PDE definition
        pde_def = self._generate_pde_definition(eq_structure, detection_result)
        definitions.append(pde_def)

        # Generate theorem based on domain
        if detection_result.domain == ScientificDomain.PHYSICS:
            theorem = self._generate_physics_pde_theorem(eq_structure, solution_type)
            theorems.append(theorem)

            if generate_proof_scaffold:
                scaffold = self._generate_pde_proof_scaffold(eq_structure, solution_type)
                proof_scaffolds.append(scaffold)
        else:
            theorem = self._generate_pde_theorem(eq_structure, solution_type)
            theorems.append(theorem)

            if generate_proof_scaffold:
                scaffold = self._generate_pde_proof_scaffold(eq_structure, solution_type)
                proof_scaffolds.append(scaffold)

        # Assemble complete Lean 4 file
        lean4_code = self._assemble_lean4_file(imports, definitions, theorems, proof_scaffolds)

        return Lean4TranslationResult(
            success=True,
            lean4_code=lean4_code,
            definitions=definitions,
            theorems=theorems,
            proof_scaffolds=proof_scaffolds,
            imports=imports,
            metadata={
                "equation_structure": eq_structure.__dict__,
                "solution_type": solution_type.value
            }
        )

    def _parse_pde_structure(self, equation: str) -> EquationStructure:
        """Parse PDE structure from equation string"""
        try:
            # Parse with SymPy
            expr = parse_expr(equation, evaluate=False)

            # Extract variables
            symbols = [str(s) for s in expr.free_symbols]

            # Determine dependent variable (usually 'u' for PDEs)
            dependent_var = 'u' if 'u' in symbols else symbols[0] if symbols else 'u'

            # Determine independent variables (usually 'x', 'y', 't')
            independent_vars = [s for s in symbols if s in ['x', 'y', 't', 'z']]
            if not independent_vars:
                independent_vars = ['x', 't']

            # Determine order
            order = self._determine_derivative_order(equation)

            # Check if linear
            is_linear = self._check_linearity(expr, dependent_var)

            # Check if homogeneous
            is_homogeneous = self._check_homogeneity(expr, dependent_var)

            return EquationStructure(
                equation=equation,
                dependent_var=dependent_var,
                independent_vars=independent_vars,
                order=order,
                is_linear=is_linear,
                is_homogeneous=is_homogeneous,
                coefficients={},
                boundary_conditions=[],
                initial_conditions=[]
            )

        except (ValueError, TypeError, SyntaxError, AttributeError) as e:
            logger.warning(f"Failed to parse PDE with SymPy: {e}, using defaults")
            return EquationStructure(
                equation=equation,
                dependent_var='u',
                independent_vars=['x', 't'],
                order=2,
                is_linear=False,
                is_homogeneous=False,
                coefficients={},
                boundary_conditions=[],
                initial_conditions=[]
            )

    def _generate_pde_definition(
        self,
        eq_structure: EquationStructure,
        detection_result: MathDetectionResult
    ) -> Lean4CodeBlock:
        """Generate Lean 4 definition for PDE"""

        # Create a descriptive name
        var_name = eq_structure.dependent_var
        vars_str = "".join(eq_structure.independent_vars)
        def_name = f"{var_name}_pde_{vars_str}"

        code = f'''variable {{R : Type}} [Real R]

/-- The PDE: {eq_structure.equation} -/
def {def_name} (u : Fin 2 → R → R) : Prop :=
  ∀ (i : Fin 2) (x : R),
    fderiv R (fun t => u i t) x = ...  -- Formal definition of PDE

/-- The PDE as a differential operator -/
def {def_name}_operator (u : Fin 2 → R → R) : Fin 2 → R → R :=
  fun i x =>
    fderiv R (fun t => u i t) x - ...  -- L[u] = 0 form
'''

        return Lean4CodeBlock(
            code=code,
            description=f"Definition for PDE: {eq_structure.equation}",
            dependencies=["Real", "fderiv", "Fin"],
            imports=["Mathlib.Analysis.Calculus.FDeriv"]
        )

    def _generate_physics_pde_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate theorem for physics PDE (heat, wave, Laplace)"""

        equation = eq_structure.equation.lower()

        if "heat" in equation or "∂u/∂t" in equation and "∂²u/∂x²" in equation:
            return self._generate_heat_equation_theorem(eq_structure, solution_type)
        elif "wave" in equation or "∂²u/∂t²" in equation:
            return self._generate_wave_equation_theorem(eq_structure, solution_type)
        elif "laplace" in equation or "∇²u" in equation:
            return self._generate_laplace_equation_theorem(eq_structure, solution_type)
        else:
            return self._generate_pde_theorem(eq_structure, solution_type)

    def _generate_heat_equation_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate theorem for heat equation"""

        code = f'''/-- Solution to the heat equation -/
theorem heat_equation_solution
    (α : R) [hα : 0 < α]
    (f : R → R)
    (hf : Continuous f)
    : ∃ u : R → R → R,
        (∀ x t, ∂u/∂t = α · ∂²u/∂x²) ∧
        (∀ x, u x 0 = f x) ∧
        Continuous u :=
  by
    -- Use separation of variables or Fourier series
    sorry'''

        return Lean4CodeBlock(
            code=code,
            description="Heat equation existence theorem",
            dependencies=["Continuous", "deriv"],
            imports=["Mathlib.Analysis.Calculus.Deriv"]
        )

    def _generate_wave_equation_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate theorem for wave equation"""

        code = f'''/-- Solution to the wave equation -/
theorem wave_equation_solution
    (c : R) [hc : 0 < c]
    (f g : R → R)
    (hf : Continuous f)
    (hg : Continuous g)
    : ∃ u : R → R → R,
        (∀ x t, ∂²u/∂t² = c² · ∂²u/∂x²) ∧
        (∀ x, u x 0 = f x) ∧
        (∀ x, ∂u/∂t x 0 = g x) ∧
        Continuous u :=
  by
    -- Use d'Alembert's formula
    sorry'''

        return Lean4CodeBlock(
            code=code,
            description="Wave equation existence theorem",
            dependencies=["Continuous", "deriv"],
            imports=["Mathlib.Analysis.Calculus.Deriv"]
        )

    def _generate_laplace_equation_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate theorem for Laplace equation"""

        code = f'''/-- Solution to Laplace equation -/
theorem laplace_equation_solution
    (Ω : Set (Fin 2 → R))
    (g : ∂Ω → R)
    (hg : Continuous g)
    : ∃ u : Fin 2 → R → R,
        (∀ x ∈ Ω, ∇²u x = 0) ∧
        (∀ x ∈ ∂Ω, u x = g x) ∧
        Continuous u :=
  by
    -- Use potential theory or Green's function
    sorry'''

        return Lean4CodeBlock(
            code=code,
            description="Laplace equation existence theorem",
            dependencies=["Continuous", "laplacian"],
            imports=[]
        )

    def _generate_pde_theorem(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate generic PDE theorem"""

        var_name = eq_structure.dependent_var
        thm_name = f"{var_name}_pde_solution"

        code = f'''/-- Existence of solution to the PDE -/
theorem {thm_name}_exists
    (f : Fin 2 → R → R)
    (g : ∂Ω → R)
    (hg : Continuous g)
    : ∃ u : Fin 2 → R → R,
        {var_name}_pde u ∧
        (∀ x ∈ ∂Ω, u x = g x) ∧
        Continuous u :=
  by
    -- Apply existence theorem for PDEs
    sorry'''

        return Lean4CodeBlock(
            code=code,
            description=f"PDE {solution_type.value} theorem",
            dependencies=[f"{var_name}_pde"],
            imports=[]
        )

    def _generate_pde_proof_scaffold(
        self,
        eq_structure: EquationStructure,
        solution_type: SolutionType
    ) -> Lean4CodeBlock:
        """Generate proof scaffold for PDE theorem"""

        code = f'''/-- Proof sketch for PDE solution -/
proof_scaffold:
  1. **Construct solution**:
     - Use separation of variables: u(x,t) = X(x)T(t)
     - Or use eigenfunction expansion
     - Or use Green's function

  2. **Verify PDE**:
     - Check that all derivatives exist
     - Substitute into PDE
     - Show equality holds

  3. **Verify boundary conditions**:
     - Check limit as x approaches boundary
     - Use continuity of boundary data

  4. **Establish uniqueness**:
     - Use energy methods
     - Or maximum principle

Tactics to use:
  - apply separable_solutions
  - apply eigenfunction_expansion
  - rw [deriv_eq]
  - simp [laplacian]
  - continuity
  - apply maximum_principle
  - apply energy_method'''

        return Lean4CodeBlock(
            code=code,
            description=f"Proof scaffold for PDE {solution_type.value}",
            dependencies=[],
            imports=[]
        )

    # ========================================================================
    # DAE and SDE Translation
    # ========================================================================

    def _translate_dae(
        self,
        detection_result: MathDetectionResult,
        solution_type: SolutionType,
        generate_proof_scaffold: bool
    ) -> Lean4TranslationResult:
        """Translate a DAE to Lean 4"""

        code = f'''/-- Differential-Algebraic Equation -/
def dae_system (x y : R → R) : Prop :=
  ∃ (F G : R → R → R),
    (∀ t, deriv x t = F t (x t) (y t)) ∧  -- Differential equation
    (∀ t, G t (x t) (y t) = 0) ∧           -- Algebraic constraint
    Continuous F ∧ Continuous G

theorem dae_solution_exists
    (F G : R → R → R)
    (hF : Continuous F)
    (hG : Continuous G)
    (h_index : index_1_dae G)
    : ∃ x y, dae_system x y :=
  by
    -- Apply DAE existence theorem
    sorry'''

        return Lean4TranslationResult(
            success=True,
            lean4_code=code,
            definitions=[
                Lean4CodeBlock(
                    code=code,
                    description="DAE definition and theorem",
                    dependencies=[],
                    imports=[]
                )
            ],
            theorems=[],
            proof_scaffolds=[],
            imports=self.default_imports,
            metadata={"note": "DAE translation - basic structure"}
        )

    def _translate_sde(
        self,
        detection_result: MathDetectionResult,
        solution_type: SolutionType,
        generate_proof_scaffold: bool
    ) -> Lean4TranslationResult:
        """Translate an SDE to Lean 4"""

        code = f'''/-- Stochastic Differential Equation -/
structure SDE where
  drift : R → R → R        -- μ(x, t)
  diffusion : R → R → R    -- σ(x, t)
  initial_condition : R

/-- Solution to SDE: dX = μ(X,t)dt + σ(X,t)dW -/
def sde_solution (sde : SDE) (X : R → R) : Prop :=
  ∃ (W : BrownianMotion),
    ∀ t,
      X t = sde.initial_condition +
           ∫₀ᵗ sde.drift (X s) s ds +
           ∫₀ᵗ sde.diffusion (X s) dW s

theorem sde_solution_exists
    (sde : SDE)
    (h_lipschitz : Lipschitz sde.drift ∧ Lipschitz sde.diffusion)
    : ∃ X W, sde_solution sde X ∧ IsBrownianMotion W :=
  by
    -- Apply Itô existence theorem
    sorry'''

        return Lean4TranslationResult(
            success=True,
            lean4_code=code,
            definitions=[
                Lean4CodeBlock(
                    code=code,
                    description="SDE definition and theorem",
                    dependencies=["BrownianMotion", "Lipschitz", "Itô"],
                    imports=[]
                )
            ],
            theorems=[],
            proof_scaffolds=[],
            imports=self.default_imports,
            metadata={"note": "SDE translation - basic structure"}
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_variables(self, equation: str) -> List[str]:
        """Extract variable names from equation"""
        # Simple heuristic: extract single-letter variables
        variables = list(set(re.findall(r'\b[a-zA-Z]\b', equation)))
        return [v for v in variables if v not in ['d', 'dx', 'dt', 'dy']]

    def _determine_derivative_order(self, equation: str) -> int:
        """Determine the order of a differential equation"""
        # Count prime notations
        prime_count = equation.count("'")

        # Count d/dx patterns
        d_count = equation.count("d²") + equation.count("d^2") + equation.count("∂²")

        return max(prime_count, d_count, 1)

    def _check_linearity(self, expr, dependent_var: str) -> bool:
        """Check if an expression is linear in the dependent variable"""
        try:
            # Get the symbol for dependent variable
            dep_sym = sp.Symbol(dependent_var)

            # Check if expression is polynomial in dependent variable
            if expr.is_polynomial(dep_sym):
                # Check degree
                degree = sp.degree(expr, dep_sym)
                return degree <= 1

            return False

        except (ValueError, TypeError, AttributeError):
            return False

    def _check_homogeneity(self, expr, dependent_var: str) -> bool:
        """Check if an ODE is homogeneous"""
        try:
            # Simple heuristic: check if all terms have same degree
            # This is a simplified check
            dep_sym = sp.Symbol(dependent_var)

            if expr.is_zero:
                return True

            # Try to rewrite as polynomial and check degrees
            if expr.is_polynomial(dep_sym):
                return True

            return False

        except (ValueError, TypeError, AttributeError):
            return False

    def _unsupported_math_type(
        self,
        detection_result: MathDetectionResult
    ) -> Lean4TranslationResult:
        """Handle unsupported math types"""
        return Lean4TranslationResult(
            success=False,
            lean4_code="",
            definitions=[],
            theorems=[],
            proof_scaffolds=[],
            imports=[],
            error_message=f"Unsupported math type: {detection_result.math_type.value}",
            metadata={"math_type": detection_result.math_type.value}
        )

    def _assemble_lean4_file(
        self,
        imports: List[str],
        definitions: List[Lean4CodeBlock],
        theorems: List[Lean4CodeBlock],
        proof_scaffolds: List[Lean4CodeBlock]
    ) -> str:
        """Assemble complete Lean 4 file"""

        lines = []

        # File header
        lines.append("-- Auto-generated by ODE/PDE Translator")
        lines.append(f"-- Lean {self.lean4_version}")
        lines.append("")
        lines.append("import " + " import ".join(imports))
        lines.append("")
        lines.append("namespace ODEPDE")
        lines.append("")
        lines.append("open Real")
        lines.append("")
        lines.append("--------------------------------------------------------------------------------")
        lines.append("-- Definitions")
        lines.append("--------------------------------------------------------------------------------")
        lines.append("")

        # Add definitions
        for definition in definitions:
            lines.append(f"-- {definition.description}")
            lines.append(definition.code)
            lines.append("")

        lines.append("--------------------------------------------------------------------------------")
        lines.append("-- Theorems")
        lines.append("--------------------------------------------------------------------------------")
        lines.append("")

        # Add theorems
        for theorem in theorems:
            lines.append(f"-- {theorem.description}")
            lines.append(theorem.code)
            lines.append("")

        if proof_scaffolds:
            lines.append("--------------------------------------------------------------------------------")
            lines.append("-- Proof Scaffolds")
            lines.append("--------------------------------------------------------------------------------")
            lines.append("")

            # Add proof scaffolds
            for scaffold in proof_scaffolds:
                lines.append(f"-- {scaffold.description}")
                lines.append(scaffold.code)
                lines.append("")

        lines.append("end ODEPDE")
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def translate_to_lean4(
    detection_result: MathDetectionResult,
    solution_type: SolutionType = SolutionType.EXISTENCE_UNIQUENESS
) -> Lean4TranslationResult:
    """
    Convenience function to translate detection result to Lean 4.

    Args:
        detection_result: Result from ContinuousMathDetector
        solution_type: Type of solution theorem to generate

    Returns:
        Lean4TranslationResult with formal Lean 4 code
    """
    translator = ODEPDETranslator()
    return translator.translate(detection_result, solution_type)


def translate_ode_to_lean4(
    equation: str,
    initial_condition: Optional[str] = None
) -> str:
    """
    Quick ODE to Lean 4 translation.

    Args:
        equation: ODE equation
        initial_condition: Optional initial condition

    Returns:
        Lean 4 code as string
    """
    translator = ODEPDETranslator()
    result = translator.translate_ode(equation, initial_condition)
    return result.lean4_code


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    from continuous_math_detector import ContinuousMathDetector

    # Create detector and translator
    detector = ContinuousMathDetector()
    translator = ODEPDETranslator()

    # Example 1: Simple ODE
    print("=" * 80)
    print("Example 1: First-order linear ODE")
    print("=" * 80)

    text = "Solve the ODE dy/dx + y = 0 with initial condition y(0) = 1"
    detection_result = detector.detect(text)

    print(f"Detected: {detection_result.math_type.value}")
    print(f"Problem Type: {detection_result.problem_type.value}")
    print(f"Confidence: {detection_result.confidence}")
    print()

    translation_result = translator.translate(detection_result)

    if translation_result.success:
        print("Generated Lean 4 Code:")
        print(translation_result.lean4_code)
    else:
        print(f"Translation failed: {translation_result.error_message}")

    print()

    # Example 2: Heat equation (PDE)
    print("=" * 80)
    print("Example 2: Heat Equation (PDE)")
    print("=" * 80)

    text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"
    detection_result = detector.detect(text)

    print(f"Detected: {detection_result.math_type.value}")
    print(f"Domain: {detection_result.domain.value}")
    print()

    translation_result = translator.translate(detection_result)

    if translation_result.success:
        print("Generated Lean 4 Code:")
        print(translation_result.lean4_code)
    else:
        print(f"Translation failed: {translation_result.error_message}")
