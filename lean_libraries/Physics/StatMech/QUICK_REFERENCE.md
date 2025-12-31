# Statistical Mechanics Library - Quick Reference

## File Overview

| File | Size | Lines | Topics |
|------|------|-------|--------|
| EnsembleTheory.lean | 10.6 KB | ~350 | Ensembles, Ergodic Theory, Liouville's Theorem |
| Thermodynamics.lean | 9.9 KB | ~320 | Thermodynamic Laws, Potentials, Maxwell Relations |
| PhaseTransitions.lean | 11.5 KB | ~360 | Critical Exponents, Scaling Laws, Universality |
| KineticTheory.lean | 11.3 KB | ~350 | Boltzmann Equation, H-Theorem, Transport |
| StatMech.lean | 9.9 KB | ~280 | Unified Interface, Classic Results |

## Key Definitions by Module

### EnsembleTheory.lean

```lean
-- Phase space for N particles in d dimensions
def PhaseSpace (N d : ℕ) := ℝ^(2 * N * d)

-- Microcanonical ensemble (isolated system)
structure MicrocanonicalEnsemble where
  N : ℕ
  d : ℕ
  E : ℝ  -- Energy
  ΔE : ℝ -- Energy width
  energyFunction : PhaseSpace N d → ℝ

-- Canonical ensemble (thermal bath)
structure CanonicalEnsemble where
  N : ℕ
  d : ℕ
  T : ℝ -- Temperature
  β : ℝ := 1/T
  partitionFunction : ℝ

-- Grand canonical ensemble (energy + particle exchange)
structure GrandCanonicalEnsemble where
  d : ℕ
  T : ℝ
  μ_chem : ℝ -- Chemical potential
```

### Thermodynamics.lean

```lean
-- Absolute temperature
structure Temperature where
  kelvin : ℝ
  pos : kelvin > 0

-- Boltzmann entropy
def boltzmannEntropy (Ω : ℝ) (k_B : ℝ) : ℝ := k_B * log Ω

-- Gibbs entropy (discrete)
def gibbsEntropy {Ω : Type*} [Fintype Ω] (p : Ω → ℝ) : ℝ :=
  -k_B * Σ i, p i * log (p i)

-- Thermodynamic potentials
def helmholtzFreeEnergy (U T S : ℝ) : ℝ := U - T * S
def gibbsFreeEnergy (U P V T S : ℝ) : ℝ := U + P * V - T * S
def grandPotential (U T S μ N : ℝ) : ℝ := U - T * S - μ * N
```

### PhaseTransitions.lean

```lean
-- Order parameter
structure OrderParameter where
  value : ℝ
  symmetricPhase : value = 0
  brokenSymmetryPhase : value ≠ 0

-- Landau free energy
def landaeuFreeEnergy (a b T T_c φ : ℝ) : ℝ :=
  a * (T - T_c) * φ^2 + b * φ^4

-- Critical exponents
structure CriticalExponents where
  α : ℝ -- Specific heat
  β : ℝ -- Order parameter
  γ : ℝ -- Susceptibility
  δ : ℝ -- Critical isotherm
  ν : ℝ -- Correlation length
  η : ℝ -- Anomalous dimension

-- Universality class
structure UniversalityClass where
  spatialDimension : ℕ
  symmetryGroup : Type
  orderParameterDimension : ℕ
  criticalExponents : CriticalExponents
```

### KineticTheory.lean

```lean
-- Distribution function
structure DistributionFunction where
  position : ℝ → ℝ → ℝ → ℝ -- f(x, v, t)
  nonnegative : ∀ x v t, position x v t ≥ 0

-- Boltzmann equation
structure BoltzmannEquation where
  f : DistributionFunction
  forceField : ℝ → ℝ → ℝ
  collisionTerm : DistributionFunction → DistributionFunction

-- Transport coefficients
def viscosity (n m λ v_th : ℝ) : ℝ := (1/3) * n * m * λ * v_th
def thermalConductivity (n k_B λ v_th : ℝ) : ℝ := (5/2) * n * k_B * λ * v_th
def diffusionCoefficient (λ v_th : ℝ) : ℝ := (1/3) * λ * v_th
```

## Main Theorems

### Fundamental Laws

| Theorem | Module | Statement |
|---------|--------|-----------|
| `liouvilleTheorem` | EnsembleTheory | dΓ/dt = 0 (phase space volume conserved) |
| `ergodicHypothesis` | EnsembleTheory | ⟨f⟩_time = ⟨f⟩_ensemble |
| `zerothLaw` | Thermodynamics | Thermal equilibrium is transitive |
| `secondLaw` | Thermodynamics | ΔS_universe ≥ 0 |
| `thirdLaw` | Thermodynamics | S → 0 as T → 0 |
| `hTheorem` | KineticTheory | dH/dt ≤ 0 (entropy increase) |

### Critical Phenomena

| Theorem | Statement |
|---------|-----------|
| `rushbrookeScaling` | α + 2β + γ = 2 |
| `widomScaling` | γ = β(δ - 1) |
| `fisherScaling` | γ = ν(2 - η) |
| `josephsonScaling` | νd = 2 - α |
| `correlationLengthDivergence` | ξ ~ |T-T_c|^(-ν) |

### Transport

| Theorem | Statement |
|---------|-----------|
| `einsteinRelation` | D = μkT |
| `greenKuboViscosity` | η = (V/kT)∫⟨P_xy(t)P_xy(0)⟩dt |
| `greenKuboThermalConductivity` | κ = (V/kT²)∫⟨J_Q(t)J_Q(0)⟩dt |
| `greenKuboDiffusion` | D = (1/3)∫⟨v(t)·v(0)⟩dt |

## Critical Exponents Reference

| System | α | β | γ | δ | ν | η |
|--------|---|---|---|---|---|---|
| Mean Field | 0 | 1/2 | 1 | 3 | 1/2 | 0 |
| 2D Ising (exact) | 0 | 1/8 | 7/4 | 15 | 1 | 1/4 |
| 3D Ising | 0.11 | 0.326 | 1.237 | 4.79 | 0.630 | 0.036 |
| 3D Heisenberg | -0.12 | 0.365 | 1.386 | 4.80 | 0.709 | 0.037 |

## Usage Patterns

### 1. Calculate Thermodynamic Properties

```lean
example (ens : CanonicalEnsemble) : ℝ :=
  ens.freeEnergy  -- F = -kT ln Z
```

### 2. Check for Phase Transition

```lean
example (T T_c : ℝ) (φ : ℝ) :
    T < T_c → φ ≠ 0 :=  -- Spontaneous symmetry breaking
  landaeuMinimizer a b T T_c
```

### 3. Compute Transport Coefficients

```lean
example (n m λ v_th : ℝ) : ℝ :=
  viscosity n m λ v_th  -- η = (1/3)nmλv_th
```

### 4. Apply Scaling Laws

```lean
example (α β γ : ℝ) :
    α + 2*β + γ = 2 :=  -- Rushbrooke
  rushbrookeScaling α β γ
```

## Common Constants

```lean
-- Physical constants
def k_B : ℝ := 1.380649e-23  -- Boltzmann constant (J/K)
def ℏ : ℝ := 1.0545718e-34  -- Reduced Planck constant (J·s)
def N_A : ℝ := 6.02214076e23 -- Avogadro's number
def R : ℝ := 8.314462618     -- Gas constant (J/(mol·K))

-- Conversions
def eV_to_Joule : ℝ := 1.602176634e-19
def atm_to_Pa : ℝ := 101325
def cal_to_J : ℝ := 4.184
```

## Mathematical Dependencies

```lean
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic  -- Integration
import Mathlib.Probability.ProbabilityMassFunction   -- Probability
import Mathlib.Analysis.Calculus.FDeriv.Basic        -- Derivatives
import Mathlib.MeasureTheory.Integral.Bochner        -- Bochner integral
import Mathlib.Data.Real.Sqrt                         -- Square roots
```

## Proof Strategies

### For Ensemble Theory
1. Use Liouville's theorem for phase space arguments
2. Apply saddle-point approximation for partition functions
3. Use Laplace method for thermodynamic limit

### For Thermodynamics
1. Legendre transforms for different ensembles
2. Equality of mixed partials for Maxwell relations
3. Concavity/convexity arguments for stability

### For Phase Transitions
1. Landau theory for mean-field results
2. Scaling hypothesis for critical exponents
3. Renormalization group for universality

### For Kinetic Theory
1. Moment expansion for fluid equations
2. Chapman-Enskog for transport coefficients
3. Detailed balance for equilibrium solutions

## Tips and Tricks

1. **Always check measurability**: All integrals require `Measurable` assumptions
2. **Use positivity**: Entropy, temperature must be positive
3. **Beware of limits**: Many results require thermodynamic limit N→∞
4. **Symmetry matters**: Order parameters relate to symmetry breaking
5. **Conservation laws**: Energy, momentum, particle number constrain dynamics

## Future Work

- [ ] Complete all `sorry` proofs
- [ ] Add quantum statistical mechanics
- [ ] Implement exact solutions (Ising model)
- [ ] Add renormalization group calculations
- [ ] Include Monte Carlo methods
- [ ] Formalize fluctuation theorems
- [ ] Add non-equilibrium steady states

## Related Libraries

- `lean_libraries/Physics/ClassicalMechanics.lean` - Hamiltonian dynamics
- `lean_libraries/Physics/QuantumMechanics.lean` - Quantum foundations
- `lean_libraries/Probability/StochasticProcesses.lean` - Random processes
- `lean_libraries/Analysis/ConvexAnalysis.lean` - Thermodynamic potentials

## References

1. R.K. Pathria, "Statistical Mechanics" (3rd ed.)
2. K. Huang, "Statistical Mechanics" (2nd ed.)
3. L.D. Landau & E.M. Lifshitz, "Statistical Physics"
4. M. Kardar, "Statistical Physics of Particles"
5. D. Chandler, "Introduction to Modern Statistical Mechanics"

## Contributing

When adding new content:
1. Follow the existing structure
2. Include physical motivation
3. Provide theorem statements
4. Add relevant examples
5. Update this quick reference

---

**Total Lines of Code**: ~1,660
**Total Theorems**: 80+
**Total Definitions**: 120+
**Modules**: 4 + 1 unified interface
