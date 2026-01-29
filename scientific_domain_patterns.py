"""
Scientific Domain Patterns for Continuous Mathematics

This module provides domain-specific patterns, knowledge bases, and templates
for continuous mathematics across scientific domains including Physics, Chemistry,
Biology, Engineering, and Economics.

Features:
- Domain-specific equation patterns and templates
- Parameter conventions and units
- Boundary/initial condition patterns
- Solution method recommendations
- Verification patterns for each domain

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.3)
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

# Import from B.1 detector
from continuous_math_detector import ScientificDomain, MathType, ProblemType

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Domain-Specific Data Structures
# ============================================================================

class DomainPattern(Enum):
    """Types of domain-specific patterns"""
    EQUATION_TEMPLATE = "equation_template"
    PARAMETER_CONVENTION = "parameter_convention"
    BOUNDARY_CONDITION = "boundary_condition"
    SOLUTION_METHOD = "solution_method"
    VERIFICATION_PATTERN = "verification_pattern"
    NAMED_PROBLEM = "named_problem"


@dataclass
class EquationTemplate:
    """Template for domain-specific equations"""
    name: str
    domain: ScientificDomain
    category: str  # e.g., "mechanics", "electromagnetism", "quantum_mechanics"
    equation_pattern: str
    description: str
    parameters: Dict[str, str]  # parameter name -> description
    typical_conditions: List[str]
    solution_method: str
    lean4_template: Optional[str] = None


@dataclass
class ParameterConvention:
    """Domain-specific parameter conventions"""
    domain: ScientificDomain
    parameter: str
    symbol: str
    description: str
    typical_values: List[str]
    units: Optional[str] = None
    related_parameters: List[str] = field(default_factory=list)


@dataclass
class DomainKnowledge:
    """Knowledge base for a specific domain"""
    domain: ScientificDomain
    equation_templates: List[EquationTemplate]
    parameter_conventions: List[ParameterConvention]
    common_boundary_conditions: List[str]
    typical_solution_methods: List[str]
    verification_patterns: List[str]
    named_problems: Dict[str, str]  # name -> description


# ============================================================================
# Scientific Domain Patterns Manager
# ============================================================================

class ScientificDomainPatterns:
    """
    Manager for scientific domain-specific patterns and knowledge.

    Provides:
    - Domain-specific equation templates
    - Parameter conventions
    - Boundary/initial condition patterns
    - Solution method recommendations
    - Verification patterns
    """

    def __init__(self):
        """Initialize domain patterns with knowledge bases"""
        self.domain_knowledge: Dict[ScientificDomain, DomainKnowledge] = {}
        self._init_physics_patterns()
        self._init_chemistry_patterns()
        self._init_biology_patterns()
        self._init_engineering_patterns()
        self._init_economics_patterns()

        logger.info("Initialized Scientific Domain Patterns for 5 domains")

    # ========================================================================
    # Physics Domain Patterns
    # ========================================================================

    def _init_physics_patterns(self):
        """Initialize physics domain patterns"""

        equation_templates = [
            EquationTemplate(
                name="Newton's Second Law",
                domain=ScientificDomain.PHYSICS,
                category="mechanics",
                equation_pattern="F = m*a or F = m*d²x/dt²",
                description="Newton's second law of motion",
                parameters={
                    "F": "Force",
                    "m": "Mass",
                    "a": "Acceleration",
                    "x": "Position"
                },
                typical_conditions=[
                    "x(0) = x₀",
                    "v(0) = v₀",
                    "x(t) → 0 as t → ∞"
                ],
                solution_method="Direct integration or energy methods",
                lean4_template='''
/-- Newton's Second Law -/
def newtons_second_law (F : Real → Real) (m : Real) (x : Real → Real) : Prop :=
  ∀ t, m * deriv (deriv x) t = F t

theorem newtons_second_law_solution
    (F : Real → Real)
    (m : Real) [hm : 0 < m]
    (x₀ v₀ : Real)
    : ∃ x : Real → Real,
        newtons_second_law F m x ∧
        x 0 = x₀ ∧
        deriv x 0 = v₀
'''
            ),
            EquationTemplate(
                name="Heat Equation",
                domain=ScientificDomain.PHYSICS,
                category="thermodynamics",
                equation_pattern="∂u/∂t = α∇²u",
                description="Heat diffusion equation",
                parameters={
                    "u": "Temperature",
                    "α": "Thermal diffusivity",
                    "t": "Time",
                    "x": "Position"
                },
                typical_conditions=[
                    "u(x,0) = f(x)",
                    "u(0,t) = u(L,t) = 0",
                    "∂u/∂n|∂Ω = 0"
                ],
                solution_method="Separation of variables, Fourier series",
                lean4_template='''
/-- Heat Equation -/
def heat_equation (α : Real) (u : Real → Real → Real) : Prop :=
  ∀ x t, deriv (fun t => u x t) t = α * laplacian (fun x => u x t) x

theorem heat_equation_solution_exists
    (α : Real) [hα : 0 < α]
    (f : Real → Real)
    : ∃ u : Real → Real → Real,
        heat_equation α u ∧
        (∀ x, u x 0 = f x)
'''
            ),
            EquationTemplate(
                name="Wave Equation",
                domain=ScientificDomain.PHYSICS,
                category="waves",
                equation_pattern="∂²u/∂t² = c²∇²u",
                description="Wave propagation equation",
                parameters={
                    "u": "Wave amplitude",
                    "c": "Wave speed",
                    "t": "Time",
                    "x": "Position"
                },
                typical_conditions=[
                    "u(x,0) = f(x)",
                    "∂u/∂t(x,0) = g(x)",
                    "u(0,t) = u(L,t) = 0"
                ],
                solution_method="d'Alembert's formula, separation of variables",
                lean4_template='''
/-- Wave Equation -/
def wave_equation (c : Real) (u : Real → Real → Real) : Prop :=
  ∀ x t, deriv (deriv (fun t => u x t)) t = c² * laplacian (fun x => u x t) x

theorem wave_equation_solution
    (c : Real) [hc : 0 < c]
    (f g : Real → Real)
    : ∃ u : Real → Real → Real,
        wave_equation c u ∧
        (∀ x, u x 0 = f x) ∧
        (∀ x, deriv (fun t => u x t) 0 = g x)
'''
            ),
            EquationTemplate(
                name="Schrödinger Equation",
                domain=ScientificDomain.PHYSICS,
                category="quantum_mechanics",
                equation_pattern="iħ∂ψ/∂t = Ĥψ",
                description="Time-dependent Schrödinger equation",
                parameters={
                    "ψ": "Wave function",
                    "ħ": "Reduced Planck constant",
                    "Ĥ": "Hamiltonian operator",
                    "t": "Time"
                },
                typical_conditions=[
                    "ψ(x,0) = ψ₀(x)",
                    "ψ → 0 as |x| → ∞",
                    "∫|ψ|²dx = 1"
                ],
                solution_method="Spectral methods, perturbation theory",
                lean4_template='''
/-- Schrödinger Equation -/
def schrödinger_equation (Ĥ : (Real → Complex) → (Real → Complex)) (ψ : Real → Real → Complex) : Prop :=
  ∀ x t, I * ħ * deriv (fun t => ψ x t) t = Ĥ (fun x => ψ x t) x

theorem schrödinger_solution_exists
    (Ĥ : _)
    (ψ₀ : Real → Complex)
    (h_normalized : ∫ x, |ψ₀ x|² = 1)
    : ∃ ψ : Real → Real → Complex,
        schrödinger_equation Ĥ ψ ∧
        (∀ x, ψ x 0 = ψ₀ x) ∧
        (∀ t, ∫ x, |ψ x t|² = 1)
'''
            ),
            EquationTemplate(
                name="Laplace Equation",
                domain=ScientificDomain.PHYSICS,
                category="electrostatics",
                equation_pattern="∇²φ = 0",
                description="Laplace equation for potential",
                parameters={
                    "φ": "Electric potential",
                    "∇²": "Laplacian operator"
                },
                typical_conditions=[
                    "φ = f on ∂Ω",
                    "φ → 0 as |x| → ∞",
                    "∂φ/∂n = g on ∂Ω"
                ],
                solution_method="Potential theory, Green's functions",
                lean4_template='''
/-- Laplace Equation -/
def laplace_equation (φ : Real → Real) : Prop :=
  ∀ x, laplacian φ x = 0

theorem laplace_solution_exists
    (Ω : Set (Fin n → Real))
    (f : ∂Ω → Real)
    : ∃ φ : Fin n → Real → Real,
        laplace_equation (fun x => φ x) ∧
        (∀ x ∈ ∂Ω, φ x = f x)
'''
            ),
            EquationTemplate(
                name="Navier-Stokes Equation",
                domain=ScientificDomain.PHYSICS,
                category="fluid_dynamics",
                equation_pattern="∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u",
                description="Navier-Stokes equation for fluid flow",
                parameters={
                    "u": "Velocity field",
                    "p": "Pressure",
                    "ρ": "Density",
                    "ν": "Kinematic viscosity"
                },
                typical_conditions=[
                    "u = 0 on ∂Ω (no-slip)",
                    "∇·u = 0 (incompressibility)",
                    "u(x,0) = u₀(x)"
                ],
                solution_method="Numerical methods (CFD)",
                lean4_template='''
/-- Navier-Stokes Equation -/
def navier_stokes
    (u p : Real → Real → Real)
    (ρ ν : Real)
    : Prop :=
  (∀ x t, div u x = 0) ∧
  (∀ x t,
    deriv (fun t => u x t) t +
    grad (u x t) · u x t =
    -grad p x / ρ +
    ν * laplacian u x)
'''
            )
        ]

        parameter_conventions = [
            ParameterConvention(
                domain=ScientificDomain.PHYSICS,
                parameter="Reduced Planck constant",
                symbol="ħ or hbar",
                description="Quantum of action",
                typical_values=["1.055×10⁻³⁴ J·s"],
                units="J·s"
            ),
            ParameterConvention(
                domain=ScientificDomain.PHYSICS,
                parameter="Speed of light",
                symbol="c",
                description="Speed of light in vacuum",
                typical_values=["2.998×10⁸ m/s"],
                units="m/s"
            ),
            ParameterConvention(
                domain=ScientificDomain.PHYSICS,
                parameter="Gravitational constant",
                symbol="G",
                description="Gravitational constant",
                typical_values=["6.674×10⁻¹¹ m³·kg⁻¹·s⁻²"],
                units="m³·kg⁻¹·s⁻²"
            )
        ]

        named_problems = {
            "Kepler Problem": "Two-body gravitational problem",
            "Harmonic Oscillator": "Simple harmonic motion",
            "Anharmonic Oscillator": "Perturbed harmonic oscillator",
            "Double Well Potential": "Quantum tunneling problem",
            "Particle in a Box": "Quantum confinement",
            "Hydrogen Atom": "Coulomb potential problem"
        }

        self.domain_knowledge[ScientificDomain.PHYSICS] = DomainKnowledge(
            domain=ScientificDomain.PHYSICS,
            equation_templates=equation_templates,
            parameter_conventions=parameter_conventions,
            common_boundary_conditions=[
                "Dirichlet: u = f on boundary",
                "Neumann: ∂u/∂n = g on boundary",
                "Robin: αu + β∂u/∂n = h on boundary",
                "Periodic: u(x) = u(x+L)"
            ],
            typical_solution_methods=[
                "Separation of variables",
                "Fourier series",
                "Laplace transform",
                "Green's functions",
                "Variational methods",
                "Perturbation theory"
            ],
            verification_patterns=[
                "Energy conservation",
                "Momentum conservation",
                "Angular momentum conservation",
                "Unitarity (quantum mechanics)",
                "Gauge invariance"
            ],
            named_problems=named_problems
        )

    # ========================================================================
    # Chemistry Domain Patterns
    # ========================================================================

    def _init_chemistry_patterns(self):
        """Initialize chemistry domain patterns"""

        equation_templates = [
            EquationTemplate(
                name="Rate Equation",
                domain=ScientificDomain.CHEMISTRY,
                category="kinetics",
                equation_pattern="d[A]/dt = -k[A]ⁿ",
                description="Chemical reaction rate equation",
                parameters={
                    "A": "Concentration of species A",
                    "k": "Rate constant",
                    "n": "Reaction order"
                },
                typical_conditions=[
                    "[A](0) = [A]₀",
                    "[A] → 0 as t → ∞",
                    "Temperature T fixed"
                ],
                solution_method="Analytical integration (simple orders), numerical (complex)",
                lean4_template='''
/-- Chemical Rate Equation -/
def rate_equation (n : Nat) (k : Real) (A : Real → Real) : Prop :=
  ∀ t, deriv A t = -k * (A t) ^ n

theorem rate_equation_solution
    (n : Nat)
    (k : Real) [hk : 0 < k]
    (A₀ : Real)
    : ∃ A : Real → Real,
        rate_equation n k A ∧
        A 0 = A₀
'''
            ),
            EquationTemplate(
                name="Michaelis-Menten Kinetics",
                domain=ScientificDomain.CHEMISTRY,
                category="enzyme_kinetics",
                equation_pattern="d[P]/dt = V_max[S]/(K_m + [S])",
                description="Enzyme-catalyzed reaction rate",
                parameters={
                    "P": "Product concentration",
                    "S": "Substrate concentration",
                    "V_max": "Maximum reaction rate",
                    "K_m": "Michaelis constant"
                },
                typical_conditions=[
                    "[S](0) = [S]₀",
                    "[P](0) = 0",
                    "[E] << [S]"
                ],
                solution_method="Quasi-steady-state approximation",
                lean4_template='''
/-- Michaelis-Menten Kinetics -/
def michaelis_menten (V_max K_m : Real) (S P : Real → Real) : Prop :=
  ∀ t,
    deriv P t = V_max * S t / (K_m + S t) ∧
    deriv S t = -deriv P t

theorem michaelis_menten_solution
    (V_max K_m : Real)
    (S₀ : Real) [hV : 0 < V_max] [hK : 0 < K_m]
    : ∃ S P : Real → Real,
        michaelis_menten V_max K_m S P ∧
        S 0 = S₀ ∧
        P 0 = 0
'''
            ),
            EquationTemplate(
                name="Diffusion Equation",
                domain=ScientificDomain.CHEMISTRY,
                category="transport",
                equation_pattern="∂C/∂t = D∇²C",
                description="Molecular diffusion equation",
                parameters={
                    "C": "Concentration",
                    "D": "Diffusion coefficient",
                    "t": "Time",
                    "x": "Position"
                },
                typical_conditions=[
                    "C(x,0) = C₀(x)",
                    "C(0,t) = C_s (surface concentration)",
                    "∂C/∂x|_(L,t) = 0 (no flux)"
                ],
                solution_method="Separation of variables, error function solutions",
                lean4_template='''
/-- Diffusion Equation -/
def diffusion_equation (D : Real) (C : Real → Real → Real) : Prop :=
  ∀ x t, deriv (fun t => C x t) t = D * laplacian (fun x => C x t) x

theorem diffusion_solution
    (D : Real) [hD : 0 < D]
    (C₀ : Real → Real)
    : ∃ C : Real → Real → Real,
        diffusion_equation D C ∧
        (∀ x, C x 0 = C₀ x)
'''
            )
        ]

        parameter_conventions = [
            ParameterConvention(
                domain=ScientificDomain.CHEMISTRY,
                parameter="Avogadro's number",
                symbol="N_A",
                description="Number of particles per mole",
                typical_values=["6.022×10²³ mol⁻¹"],
                units="mol⁻¹"
            ),
            ParameterConvention(
                domain=ScientificDomain.CHEMISTRY,
                parameter="Gas constant",
                symbol="R",
                description="Universal gas constant",
                typical_values=["8.314 J·mol⁻¹·K⁻¹"],
                units="J·mol⁻¹·K⁻¹"
            ),
            ParameterConvention(
                domain=ScientificDomain.CHEMISTRY,
                parameter="Faraday constant",
                symbol="F",
                description="Charge per mole of electrons",
                typical_values=["96485 C·mol⁻¹"],
                units="C·mol⁻¹"
            )
        ]

        named_problems = {
            "First-order kinetics": "Exponential decay",
            "Second-order kinetics": "Reciprocal concentration dependence",
            "Autocatalysis": "Product catalyzes its own formation",
            "Oscillating reactions": "Belousov-Zhabotinsky reaction",
            "Chain reactions": "Combustion, polymerization"
        }

        self.domain_knowledge[ScientificDomain.CHEMISTRY] = DomainKnowledge(
            domain=ScientificDomain.CHEMISTRY,
            equation_templates=equation_templates,
            parameter_conventions=parameter_conventions,
            common_boundary_conditions=[
                "Fixed concentration at boundary",
                "No-flux boundary condition",
                "Periodic boundary conditions",
                "Infinite boundary (C → 0 as x → ∞)"
            ],
            typical_solution_methods=[
                "Steady-state approximation",
                "Equilibrium approximation",
                "Numerical integration",
                "Monte Carlo methods"
            ],
            verification_patterns=[
                "Mass conservation",
                "Charge conservation",
                "Thermodynamic consistency",
                "Detailed balance (equilibrium)"
            ],
            named_problems=named_problems
        )

    # ========================================================================
    # Biology Domain Patterns
    # ========================================================================

    def _init_biology_patterns(self):
        """Initialize biology domain patterns"""

        equation_templates = [
            EquationTemplate(
                name="Lotka-Volterra Equations",
                domain=ScientificDomain.BIOLOGY,
                category="ecology",
                equation_pattern="dx/dt = αx - βxy, dy/dt = δxy - γy",
                description="Predator-prey dynamics",
                parameters={
                    "x": "Prey population",
                    "y": "Predator population",
                    "α": "Prey growth rate",
                    "β": "Predation rate",
                    "δ": "Predator growth rate",
                    "γ": "Predator death rate"
                },
                typical_conditions=[
                    "x(0) = x₀ > 0",
                    "y(0) = y₀ > 0",
                    "x, y ≥ 0"
                ],
                solution_method="Phase plane analysis, numerical integration",
                lean4_template='''
/-- Lotka-Volterra Predator-Prey Model -/
def lotka_volterra (α β δ γ : Real) (x y : Real → Real) : Prop :=
  (∀ t, deriv x t = α * x t - β * x t * y t) ∧
  (∀ t, deriv y t = δ * x t * y t - γ * y t) ∧
  (∀ t, 0 ≤ x t ∧ 0 ≤ y t)

theorem lotka_volterra_solution
    (α β δ γ : Real)
    (x₀ y₀ : Real)
    [hα : 0 < α] [hβ : 0 < β] [hδ : 0 < δ] [hγ : 0 < γ]
    : ∃ x y : Real → Real,
        lotka_volterra α β δ γ x y ∧
        x 0 = x₀ ∧ y 0 = y₀
'''
            ),
            EquationTemplate(
                name="SIR Model",
                domain=ScientificDomain.BIOLOGY,
                category="epidemiology",
                equation_pattern="dS/dt = -βSI, dI/dt = βSI - γI, dR/dt = γI",
                description="Epidemic disease spread model",
                parameters={
                    "S": "Susceptible population",
                    "I": "Infected population",
                    "R": "Recovered population",
                    "β": "Infection rate",
                    "γ": "Recovery rate"
                },
                typical_conditions=[
                    "S(0) = S₀",
                    "I(0) = I₀",
                    "R(0) = 0",
                    "S + I + R = N (constant)"
                ],
                solution_method="Phase plane analysis, basic reproduction number R₀",
                lean4_template='''
/-- SIR Epidemic Model -/
def sir_model (β γ : Real) (S I R : Real → Real) : Prop :=
  (∀ t, deriv S t = -β * S t * I t) ∧
  (∀ t, deriv I t = β * S t * I t - γ * I t) ∧
  (∀ t, deriv R t = γ * I t) ∧
  (∀ t, S t + I t + R t = N)

theorem sir_solution
    (β γ : Real)
    (S₀ I₀ N : Real)
    : ∃ S I R : Real → Real,
        sir_model β γ S I R ∧
        S 0 = S₀ ∧ I 0 = I₀ ∧ R 0 = 0
'''
            ),
            EquationTemplate(
                name="FitzHugh-Nagumo Equation",
                domain=ScientificDomain.BIOLOGY,
                category="neuroscience",
                equation_pattern="∂v/∂t = v - v³/3 + w + I, ∂w/∂t = (v - a + bw)/τ",
                description="Neuron excitation model",
                parameters={
                    "v": "Membrane potential",
                    "w": "Recovery variable",
                    "I": "Input current",
                    "a": "Threshold parameter",
                    "b": "Recovery parameter",
                    "τ": "Time constant"
                },
                typical_conditions=[
                    "v(x,0) = v₀(x)",
                    "w(x,0) = w₀(x)",
                    "Periodic boundary conditions"
                ],
                solution_method="Numerical integration, bifurcation analysis",
                lean4_template='''
/-- FitzHugh-Nagumo Model -/
def fitzhugh_nagumo (a b τ I : Real) (v w : Real → Real) : Prop :=
  (∀ t, deriv v t = v t - (v t)³ / 3 + w t + I) ∧
  (∀ t, deriv w t = (v t - a + b * w t) / τ)

theorem fitzhugh_nagumo_solution
    (a b τ I : Real)
    (v₀ w₀ : Real)
    : ∃ v w : Real → Real,
        fitzhugh_nagumo a b τ I v w ∧
        v 0 = v₀ ∧ w 0 = w₀
'''
            ),
            EquationTemplate(
                name="Logistic Growth",
                domain=ScientificDomain.BIOLOGY,
                category="population_dynamics",
                equation_pattern="dN/dt = rN(1 - N/K)",
                description="Population growth with carrying capacity",
                parameters={
                    "N": "Population size",
                    "r": "Growth rate",
                    "K": "Carrying capacity"
                },
                typical_conditions=[
                    "N(0) = N₀",
                    "0 ≤ N ≤ K",
                    "N → K as t → ∞"
                ],
                solution_method="Analytical solution (logistic function)",
                lean4_template='''
/-- Logistic Growth Model -/
def logistic_growth (r K : Real) (N : Real → Real) : Prop :=
  (∀ t, deriv N t = r * N t * (1 - N t / K)) ∧
  (∀ t, 0 ≤ N t ∧ N t ≤ K)

theorem logistic_solution
    (r K : Real)
    (N₀ : Real) [hr : 0 < r] [hK : 0 < K]
    : ∃ N : Real → Real,
        logistic_growth r K N ∧
        N 0 = N₀
'''
            )
        ]

        parameter_conventions = [
            ParameterConvention(
                domain=ScientificDomain.BIOLOGY,
                parameter="Carrying capacity",
                symbol="K",
                description="Maximum sustainable population",
                typical_values=["Environment-specific"],
                units="individuals"
            ),
            ParameterConvention(
                domain=ScientificDomain.BIOLOGY,
                parameter="Basic reproduction number",
                symbol="R₀",
                description="Average number of secondary infections",
                typical_values=["> 1 for epidemic, < 1 for disease die-out"],
                units="dimensionless"
            )
        ]

        named_problems = {
            "Competitive exclusion": "Two species competing for same resource",
            "Mutualism": "Both species benefit",
            "Commensalism": "One benefits, other unaffected",
            "Parasitism": "One benefits, other harmed",
            "Trophic cascade": "Predator effects propagate through food web"
        }

        self.domain_knowledge[ScientificDomain.BIOLOGY] = DomainKnowledge(
            domain=ScientificDomain.BIOLOGY,
            equation_templates=equation_templates,
            parameter_conventions=parameter_conventions,
            common_boundary_conditions=[
                "Non-negative populations",
                "Fixed initial populations",
                "Conservation of total population (closed system)",
                "Influx/outflux boundary conditions (open system)"
            ],
            typical_solution_methods=[
                "Phase plane analysis",
                "Stability analysis",
                "Bifurcation theory",
                "Numerical simulation",
                "Markov chain models"
            ],
            verification_patterns=[
                "Non-negativity of populations",
                "Conservation laws",
                "Stability of equilibria",
                "Positivity of solutions"
            ],
            named_problems=named_problems
        )

    # ========================================================================
    # Engineering Domain Patterns
    # ========================================================================

    def _init_engineering_patterns(self):
        """Initialize engineering domain patterns"""

        equation_templates = [
            EquationTemplate(
                name="Control System ODE",
                domain=ScientificDomain.ENGINEERING,
                category="control_theory",
                equation_pattern="dx/dt = Ax + Bu",
                description="State-space control system",
                parameters={
                    "x": "State vector",
                    "u": "Control input",
                    "A": "System matrix",
                    "B": "Input matrix"
                },
                typical_conditions=[
                    "x(0) = x₀",
                    "u bounded",
                    "System stable (Re(λ) < 0)"
                ],
                solution_method="Matrix exponential, Laplace transform",
                lean4_template='''
/-- State-Space Control System -/
def state_space (A B : Matrix) (x u : Real → Real) : Prop :=
  ∀ t, deriv x t = A * x t + B * u t

theorem state_space_response
    (A B : Matrix)
    (x₀ : Real)
    (u : Real → Real)
    : ∃ x : Real → Real,
        state_space A B x u ∧
        x 0 = x₀
'''
            ),
            EquationTemplate(
                name="RLC Circuit",
                domain=ScientificDomain.ENGINEERING,
                category="electrical_engineering",
                equation_pattern="L*d²q/dt² + R*dq/dt + q/C = V(t)",
                description="RLC circuit equation",
                parameters={
                    "q": "Charge",
                    "L": "Inductance",
                    "R": "Resistance",
                    "C": "Capacitance",
                    "V": "Voltage source"
                },
                typical_conditions=[
                    "q(0) = q₀",
                    "dq/dt(0) = I₀",
                    "Overdamped/critically damped/underdamped"
                ],
                solution_method="Characteristic equation, Laplace transform",
                lean4_template='''
/-- RLC Circuit -/
def rlc_circuit (L R C : Real) (V q : Real → Real) : Prop :=
  L * deriv (deriv q) t + R * deriv q t + q t / C = V t

theorem rlc_solution
    (L R C : Real)
    (V : Real → Real)
    (q₀ I₀ : Real)
    : ∃ q : Real → Real,
        rlc_circuit L R C V q ∧
        q 0 = q₀ ∧
        deriv q 0 = I₀
'''
            ),
            EquationTemplate(
                name="Beam Equation",
                domain=ScientificDomain.ENGINEERING,
                category="mechanical_engineering",
                equation_pattern="EI*d⁴w/dx⁴ = q(x)",
                description="Euler-Bernoulli beam equation",
                parameters={
                    "w": "Deflection",
                    "E": "Young's modulus",
                    "I": "Moment of inertia",
                    "q": "Distributed load"
                },
                typical_conditions=[
                    "w(0) = w(L) = 0 (simply supported)",
                    "w'(0) = w'(L) = 0 (clamped)",
                    "w''(0) = w''(L) = 0 (free)"
                ],
                solution_method="Green's functions, superposition",
                lean4_template='''
/-- Euler-Bernoulli Beam Equation -/
def beam_equation (E I : Real) (q w : Real → Real) : Prop :=
  ∀ x, E * I * deriv (deriv (deriv (deriv w))) x = q x

theorem beam_solution
    (E I : Real)
    (q : Real → Real)
    (boundary_conditions : List BoundaryCondition)
    : ∃ w : Real → Real,
        beam_equation E I q w ∧
        boundary_conditions_satisfied
'''
            )
        ]

        parameter_conventions = [
            ParameterConvention(
                domain=ScientificDomain.ENGINEERING,
                parameter="Transfer function",
                symbol="H(s)",
                description="Laplace domain system response",
                typical_values=["System-specific"],
                units="varies"
            )
        ]

        named_problems = {
            "PID control": "Proportional-Integral-Derivative controller",
            "LQR control": "Linear Quadratic Regulator",
            "Kalman filter": "Optimal state estimation",
            "Bode plot": "Frequency response analysis",
            "Nyquist criterion": "Stability criterion"
        }

        self.domain_knowledge[ScientificDomain.ENGINEERING] = DomainKnowledge(
            domain=ScientificDomain.ENGINEERING,
            equation_templates=equation_templates,
            parameter_conventions=parameter_conventions,
            common_boundary_conditions=[
                "Initial rest conditions",
                "Causality",
                "Bounded-input bounded-output (BIBO) stability",
                "Controllability and observability"
            ],
            typical_solution_methods=[
                "Laplace transform",
                "Z-transform (discrete-time)",
                "State-space methods",
                "Frequency response analysis",
                "Numerical control design"
            ],
            verification_patterns=[
                "Stability (BIBO, Lyapunov)",
                "Controllability",
                "Observability",
                "Performance specifications"
            ],
            named_problems=named_problems
        )

    # ========================================================================
    # Economics Domain Patterns
    # ========================================================================

    def _init_economics_patterns(self):
        """Initialize economics domain patterns"""

        equation_templates = [
            EquationTemplate(
                name="Black-Scholes Equation",
                domain=ScientificDomain.ECONOMICS,
                category="financial_mathematics",
                equation_pattern="∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0",
                description="Option pricing PDE",
                parameters={
                    "V": "Option value",
                    "S": "Stock price",
                    "σ": "Volatility",
                    "r": "Risk-free rate",
                    "t": "Time"
                },
                typical_conditions=[
                    "V(S,T) = max(S-K, 0) (payoff)",
                    "V(0,t) = 0",
                    "V(S,t) → S as S → ∞"
                ],
                solution_method="Analytical (Black-Scholes formula), numerical (finite differences)",
                lean4_template='''
/-- Black-Scholes Equation -/
def black_scholes (σ r : Real) (V : Real → Real → Real) : Prop :=
  ∀ S t,
    deriv (fun t => V S t) t +
    (1/2) * σ² * S² * deriv (deriv (fun S => V S t)) S +
    r * S * deriv (fun S => V S t) S -
    r * V S t = 0

theorem black_scholes_solution
    (σ r K T : Real)
    : ∃ V : Real → Real → Real,
        black_scholes σ r V ∧
        (∀ S, V S T = max (S - K) 0)
'''
            ),
            EquationTemplate(
                name="Geometric Brownian Motion",
                domain=ScientificDomain.ECONOMICS,
                category="financial_mathematics",
                equation_pattern="dS = μSdt + σSdW",
                description="Stock price dynamics (SDE)",
                parameters={
                    "S": "Stock price",
                    "μ": "Drift (expected return)",
                    "σ": "Volatility",
                    "W": "Wiener process"
                },
                typical_conditions=[
                    "S(0) = S₀",
                    "S > 0"
                ],
                solution_method="Itô calculus (analytical solution: log-normal)",
                lean4_template='''
/-- Geometric Brownian Motion -/
def gbm (μ σ : Real) (S : Real → Real) : Prop :=
  ∃ W : BrownianMotion,
    ∀ t,
      S t = S₀ * exp ((μ - σ²/2) * t + σ * W t)

theorem gbm_solution
    (μ σ : Real)
    (S₀ : Real) [hS₀ : 0 < S₀]
    : ∃ S W : Real → Real,
        gbm μ σ S W ∧
        S 0 = S₀
'''
            ),
            EquationTemplate(
                name="Solow Growth Model",
                domain=ScientificDomain.ECONOMICS,
                category="macroeconomics",
                equation_pattern="dk/dt = s*f(k) - (n+g+δ)k",
                description="Economic growth model",
                parameters={
                    "k": "Capital per worker",
                    "s": "Savings rate",
                    "n": "Population growth",
                    "g": "Technological progress",
                    "δ": "Depreciation rate"
                },
                typical_conditions=[
                    "k(0) = k₀",
                    "k → k* (steady state) as t → ∞"
                ],
                solution_method="Steady-state analysis, phase diagram",
                lean4_template='''
/-- Solow Growth Model -/
def solow_model (s n g δ : Real) (f k : Real → Real) : Prop :=
  ∀ t,
    deriv k t = s * f (k t) - (n + g + δ) * k t

theorem solow_steady_state
    (s n g δ : Real)
    (k₀ : Real)
    : ∃ k : Real → Real,
        solow_model s n g δ f k ∧
        k 0 = k₀ ∧
        (∃ k* : Real, limit k t = k* as t → ∞)
'''
            )
        ]

        parameter_conventions = [
            ParameterConvention(
                domain=ScientificDomain.ECONOMICS,
                parameter="Risk-free rate",
                symbol="r",
                description="Risk-free interest rate",
                typical_values=["0.01-0.05 (1-5% annually)"],
                units="1/time"
            ),
            ParameterConvention(
                domain=ScientificDomain.ECONOMICS,
                parameter="Volatility",
                symbol="σ",
                description="Price volatility",
                typical_values=["0.1-0.5 (10-50% annually)"],
                units="1/√time"
            )
        ]

        named_problems = {
            "Option pricing": "European/American options",
            "Portfolio optimization": "Markowitz mean-variance",
            "Capital Asset Pricing Model": "Risk-return tradeoff",
            "Arbitrage Pricing Theory": "Multi-factor asset pricing",
            "Rational expectations": "Forward-looking agents"
        }

        self.domain_knowledge[ScientificDomain.ECONOMICS] = DomainKnowledge(
            domain=ScientificDomain.ECONOMICS,
            equation_templates=equation_templates,
            parameter_conventions=parameter_conventions,
            common_boundary_conditions=[
                "Terminal conditions (option maturity)",
                "No-arbitrage conditions",
                "Budget constraints",
                "Transversality conditions"
            ],
            typical_solution_methods=[
                "Itô calculus",
                "Risk-neutral pricing",
                "Dynamic programming",
                "Stochastic optimal control",
                "Numerical methods (Monte Carlo, finite differences)"
            ],
            verification_patterns=[
                "No-arbitrage condition",
                "Risk-neutral valuation",
                "Martingale property",
                "Budget balance"
            ],
            named_problems=named_problems
        )

    # ========================================================================
    # Public API Methods
    # ========================================================================

    def get_domain_knowledge(self, domain: ScientificDomain) -> Optional[DomainKnowledge]:
        """
        Get knowledge base for a specific domain.

        Args:
            domain: Scientific domain

        Returns:
            DomainKnowledge if available, None otherwise
        """
        return self.domain_knowledge.get(domain)

    def get_equation_templates(
        self,
        domain: ScientificDomain,
        category: Optional[str] = None
    ) -> List[EquationTemplate]:
        """
        Get equation templates for a domain.

        Args:
            domain: Scientific domain
            category: Optional category filter

        Returns:
            List of equation templates
        """
        knowledge = self.get_domain_knowledge(domain)
        if not knowledge:
            return []

        templates = knowledge.equation_templates
        if category:
            templates = [t for t in templates if t.category == category]

        return templates

    def get_parameter_conventions(
        self,
        domain: ScientificDomain
    ) -> List[ParameterConvention]:
        """
        Get parameter conventions for a domain.

        Args:
            domain: Scientific domain

        Returns:
            List of parameter conventions
        """
        knowledge = self.get_domain_knowledge(domain)
        return knowledge.parameter_conventions if knowledge else []

    def get_solution_methods(self, domain: ScientificDomain) -> List[str]:
        """
        Get typical solution methods for a domain.

        Args:
            domain: Scientific domain

        Returns:
            List of solution method names
        """
        knowledge = self.get_domain_knowledge(domain)
        return knowledge.typical_solution_methods if knowledge else []

    def get_boundary_conditions(
        self,
        domain: ScientificDomain
    ) -> List[str]:
        """
        Get common boundary conditions for a domain.

        Args:
            domain: Scientific domain

        Returns:
            List of boundary condition descriptions
        """
        knowledge = self.get_domain_knowledge(domain)
        return knowledge.common_boundary_conditions if knowledge else []

    def get_verification_patterns(
        self,
        domain: ScientificDomain
    ) -> List[str]:
        """
        Get verification patterns for a domain.

        Args:
            domain: Scientific domain

        Returns:
            List of verification pattern names
        """
        knowledge = self.get_domain_knowledge(domain)
        return knowledge.verification_patterns if knowledge else []

    def find_named_problem(
        self,
        domain: ScientificDomain,
        name: str
    ) -> Optional[str]:
        """
        Find description of a named problem.

        Args:
            domain: Scientific domain
            name: Problem name

        Returns:
            Problem description if found, None otherwise
        """
        knowledge = self.get_domain_knowledge(domain)
        if not knowledge:
            return None

        return knowledge.named_problems.get(name)

    def match_equation_to_template(
        self,
        equation: str,
        domain: ScientificDomain
    ) -> Optional[EquationTemplate]:
        """
        Match an equation to a domain-specific template.

        Args:
            equation: Equation string
            domain: Scientific domain

        Returns:
            Matching EquationTemplate if found, None otherwise
        """
        templates = self.get_equation_templates(domain)

        for template in templates:
            # Check if equation contains key pattern
            if any(symbol in equation for symbol in template.equation_pattern.split()):
                return template

        return None

    def recommend_solution_method(
        self,
        domain: ScientificDomain,
        math_type: MathType,
        problem_type: ProblemType
    ) -> List[str]:
        """
        Recommend solution methods based on domain and problem type.

        Args:
            domain: Scientific domain
            math_type: Type of mathematics (ODE, PDE, etc.)
            problem_type: Type of problem (IVP, BVP, etc.)

        Returns:
            List of recommended solution methods
        """
        methods = self.get_solution_methods(domain)

        # Filter based on problem type
        if problem_type == ProblemType.INITIAL_VALUE:
            methods = [m for m in methods if "integration" in m.lower() or "transform" in m.lower()]
        elif problem_type == ProblemType.BOUNDARY_VALUE:
            methods = [m for m in methods if "separation" in m.lower() or "green" in m.lower()]

        return methods if methods else ["Numerical methods"]

    def get_domain_summary(self, domain: ScientificDomain) -> Dict[str, Any]:
        """
        Get summary of domain patterns and knowledge.

        Args:
            domain: Scientific domain

        Returns:
            Dictionary with domain summary
        """
        knowledge = self.get_domain_knowledge(domain)

        if not knowledge:
            return {"domain": domain.value, "status": "not_available"}

        return {
            "domain": domain.value,
            "num_equation_templates": len(knowledge.equation_templates),
            "num_parameter_conventions": len(knowledge.parameter_conventions),
            "categories": list(set(t.category for t in knowledge.equation_templates)),
            "solution_methods": knowledge.typical_solution_methods,
            "verification_patterns": knowledge.verification_patterns
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def get_domain_patterns() -> ScientificDomainPatterns:
    """
    Get the global domain patterns instance.

    Returns:
        ScientificDomainPatterns instance
    """
    return ScientificDomainPatterns()


def get_equation_template(
    domain: ScientificDomain,
    name: str
) -> Optional[EquationTemplate]:
    """
    Get a specific equation template by name.

    Args:
        domain: Scientific domain
        name: Template name

    Returns:
        EquationTemplate if found, None otherwise
    """
    patterns = get_domain_patterns()
    templates = patterns.get_equation_templates(domain)

    for template in templates:
        if template.name == name:
            return template

    return None


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    patterns = ScientificDomainPatterns()

    print("=" * 80)
    print("Scientific Domain Patterns - Domain Summaries")
    print("=" * 80)

    for domain in ScientificDomain:
        if domain == ScientificDomain.GENERAL:
            continue

        summary = patterns.get_domain_summary(domain)
        print(f"\n{domain.value.upper()}")
        print(f"  Equation Templates: {summary['num_equation_templates']}")
        print(f"  Categories: {', '.join(summary['categories'])}")
        print(f"  Solution Methods: {', '.join(summary['solution_methods'][:3])}...")

    print("\n" + "=" * 80)
    print("Example: Physics - Heat Equation Template")
    print("=" * 80)

    heat_template = get_equation_template(ScientificDomain.PHYSICS, "Heat Equation")
    if heat_template:
        print(f"Name: {heat_template.name}")
        print(f"Category: {heat_template.category}")
        print(f"Equation: {heat_template.equation_pattern}")
        print(f"Parameters: {heat_template.parameters}")
        print(f"Solution Method: {heat_template.solution_method}")
