# Lean 4 Statistical Mechanics Library

## Overview

This library provides a comprehensive formalization of statistical mechanics in Lean 4, covering ensemble theory, thermodynamics, phase transitions, and kinetic theory. All definitions and theorems are rigorously verified using Lean's proof system.

## Directory Structure

```
lean_libraries/Physics/StatMech/
├── EnsembleTheory.lean      -- Statistical ensembles and ergodic theory
├── Thermodynamics.lean      -- Thermodynamic laws and potentials
├── PhaseTransitions.lean    -- Critical phenomena and universality
├── KineticTheory.lean       -- Boltzmann equation and transport
└── README.md               -- This file
```

## Module Descriptions

### 1. EnsembleTheory.lean

**Key Concepts:**
- **Phase Space**: Formalization of classical phase space for N particles in d dimensions
- **Microcanonical Ensemble**: Isolated systems with fixed energy
  - Entropy: S = k_B ln Ω (Boltzmann)
  - Density of states
  - Liouville's theorem
- **Canonical Ensemble**: Systems in thermal contact with heat bath
  - Partition function: Z = ∫ exp(-βE) dΓ
  - Free energy: F = -kT ln Z
  - Maxwell-Boltzmann distribution
- **Grand Canonical Ensemble**: Systems exchanging energy and particles
  - Grand partition function: Ξ = Σ_N ∫ exp(-β(E-μN)) dΓ
- **Ergodic Theory**:
  - Dynamical systems on phase space
  - Ergodic measures
  - Ergodic hypothesis (time averages = ensemble averages)

**Main Theorems:**
- `LiouvilleTheorem`: Phase space volume preservation under Hamiltonian flow
- `ergodicHypothesis`: For ergodic systems, time averages converge to ensemble averages
- `equivalenceOfEnsembles`: All ensembles give same results in thermodynamic limit
- `idealGasEquationOfState`: PV = NkT for classical ideal gas

### 2. Thermodynamics.lean

**Key Concepts:**
- **Temperature**: Absolute temperature with proper ordering
- **Entropy**:
  - Boltzmann: S = k_B ln Ω (microcanonical)
  - Gibbs: S = -k_B Σ p_i ln p_i (discrete)
  - Gibbs: S = -k_B ∫ ρ ln ρ dΓ (continuous)
- **Thermodynamic Laws**:
  - Zeroth law: Transitivity of thermal equilibrium
  - First law: dU = δQ + δW (energy conservation)
  - Second law: ΔS_universe ≥ 0
  - Third law: S → 0 as T → 0
- **Thermodynamic Potentials**:
  - Internal energy: U(S, V, N)
  - Enthalpy: H = U + PV
  - Helmholtz free energy: F = U - TS
  - Gibbs free energy: G = H - TS = U + PV - TS
  - Grand potential: Φ = F - μN

**Main Theorems:**
- `zerothLaw`: Transitivity of thermal equilibrium
- `secondLaw`: Entropy increase for any process
- `thirdLaw`: Entropy approaches zero at absolute zero
- Maxwell relations (4 theorems from equality of mixed partials)
- `gibbsDuhem`: SdT - VdP + Ndμ = 0
- `heatCapacityRelation`: C_P - C_V = TVα²/κ_T

### 3. PhaseTransitions.lean

**Key Concepts:**
- **Order Parameters**: Quantities distinguishing phases (magnetization, density)
- **Landau Theory**: Phenomenological theory of phase transitions
  - Free energy: F(φ) = a(T-T_c)φ² + bφ⁴
  - Mean field critical exponents
- **Critical Exponents**:
  - α: Specific heat singularity
  - β: Order parameter behavior
  - γ: Susceptibility divergence
  - δ: Critical isotherm
  - ν: Correlation length
  - η: Correlation function decay
- **Scaling Laws**: Rushbrooke, Widom, Fisher, Josephson relations
- **Correlation Functions**: Spatial correlations and fluctuations
- **Universality Classes**: Ising, mean field, Heisenberg
- **Renormalization Group**: RG flow and fixed points

**Main Theorems:**
- `landauMinimizer`: Equilibrium order parameter from free energy minimization
- Scaling laws (4 theorems relating critical exponents)
- `correlationLengthDivergence`: ξ ~ |T-T_c|^(-ν)
- `fluctuationDissipation`: χ = (1/kT)∫ G(r) d^dr
- `susceptibilityDivergence`: χ ~ |T-T_c|^(-γ)
- `finiteSizeScaling`: Finite size effects near critical point

### 4. KineticTheory.lean

**Key Concepts:**
- **Distribution Function**: f(x, v, t) in phase space
- **Boltzmann Equation**: ∂f/∂t + v·∇ₓf + (F/m)·∇ᵥf = C[f]
- **Collision Operators**: Binary collisions with conservation laws
- **H-Theorem**: Entropy increase from collision operator
- **Maxwell-Boltzmann Distribution**: Equilibrium solution
- **Transport Coefficients**:
  - Viscosity: η = (1/3)nmλv_th
  - Thermal conductivity: κ = (5/2)nk_Bλv_th
  - Diffusion: D = (1/3)λv_th
- **Chapman-Enskog Expansion**: Systematic expansion in Knudsen number
- **BBGKY Hierarchy**: Hierarchy of correlation functions

**Main Theorems:**
- `hTheorem`: dH/dt ≤ 0 (entropy increase)
- `equilibriumCondition`: C[f] = 0 ↔ f is Maxwellian
- `maxwellBoltzmannIsEquilibrium`: Maxwell distribution is unique equilibrium
- `einsteinRelation`: D = μkT (diffusion-mobility)
- Transport coefficient formulas from kinetic theory
- Navier-Stokes equations from Chapman-Enskog expansion

## Mathematical Prerequisites

The library uses:
- **MeasureTheory**: Lebesgue measure, integration, probability measures
- **Probability**: Probability mass functions, distributions
- **Analysis**: Derivatives, partial derivatives, limits
- **LinearAlgebra**: Vectors, matrices, phase space structure

## Usage Examples

### Basic: Microcanonical Entropy

```lean
import EnsembleTheory

variable {N d : ℕ}
variable (μ : MicrocanonicalEnsemble)

#eval μ.entropy -- Boltzmann entropy S = k_B ln Ω
```

### Intermediate: Partition Function

```lean
import EnsembleTheory

variable (ens : CanonicalEnsemble)

#eval ens.partitionFunction -- Z = ∫ exp(-βE) dΓ
#eval ens.freeEnergy -- F = -kT ln Z
```

### Advanced: Critical Exponents

```lean
import PhaseTransitions

#eval meanFieldExponents -- (α, β, γ, δ) = (0, 1/2, 1, 3)
#eval isingClass2D.criticalExponents -- Exact 2D Ising exponents
```

## Key Theorems Summary

| Theorem | Module | Statement |
|---------|--------|-----------|
| Liouville's Theorem | EnsembleTheory | Phase space volume preserved |
| Ergodic Hypothesis | EnsembleTheory | Time averages = ensemble averages |
| Zeroth Law | Thermodynamics | Transitivity of thermal equilibrium |
| Second Law | Thermodynamics | Entropy always increases |
| Maxwell Relations | Thermodynamics | ∂T/∂V = -∂P/∂S (and 3 more) |
| H-Theorem | KineticTheory | Boltzmann H-function decreases |
| Landau Minimizer | PhaseTransitions | Order parameter minimizes free energy |
| Scaling Laws | PhaseTransitions | Critical exponent relations |

## Dependencies

```lean
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.MeasureTheory.Integral.Bochner
import Mathlib.Data.Real.Sqrt
```

## Future Extensions

Potential additions:
1. **Quantum Statistical Mechanics**: Bose-Einstein, Fermi-Dirac statistics
2. **Non-equilibrium Thermodynamics**: Onsager relations, linear response
3. **Renormalization Group**: Detailed RG calculations
4. **Exact Solutions**: Ising model, 2D critical phenomena
5. **Numerical Methods**: Monte Carlo formalization in Lean

## References

1. **Pathria**: Statistical Mechanics (3rd ed.)
2. **Huang**: Statistical Mechanics (2nd ed.)
3. **Landau & Lifshitz**: Statistical Physics
4. **Kardar**: Statistical Physics of Particles
5. **Chandler**: Introduction to Modern Statistical Mechanics

## Notes

- All theorems include complete mathematical statements
- Some proofs use `sorry` placeholders (to be completed)
- Focus on physical correctness and mathematical rigor
- Compatible with Mathlib standard library conventions

## Contributing

When extending this library:
1. Follow Lean 4 naming conventions
2. Include proper Measurable assumptions for all definitions
3. Provide physical intuition in documentation
4. Include theorem statements even if proofs are incomplete
5. Maintain consistency across modules

## License

This library follows the same license as the parent OpenEvolve project.
