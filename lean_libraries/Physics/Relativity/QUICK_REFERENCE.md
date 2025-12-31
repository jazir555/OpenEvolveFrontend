# Lean 4 General Relativity Library - Quick Reference

## Quick Start Guide

### Import the Library

```lean
import DifferentialGeometry
import Spacetime
import FieldEquations
import SpecialRelativity
```

## Essential Notations

| Symbol | Meaning | Module |
|--------|---------|---------|
| `g_μν` | Metric tensor | Spacetime |
| `∇_X Y` | Covariant derivative | DifferentialGeometry |
| `R^a_{bcd}` | Riemann curvature | DifferentialGeometry |
| `G_μν` | Einstein tensor | Spacetime |
| `T_μν` | Stress-energy tensor | FieldEquations |
| `η_μν` | Minkowski metric | SpecialRelativity |
| `Λ` | Lorentz transformation | SpecialRelativity |
| `γ` | Lorentz factor | SpecialRelativity |
| `κ` | Einstein constant (8πG/c⁴) | FieldEquations |

## Core Structures

### 1. Spacetime Manifold

```lean
-- Define a spacetime
structure Spacetime where
  Manifold : Type*
  metric : LorentzianMetric  -- Signature (-,+,+,+)
  orientable : Prop

-- Causal classification
#check IsTimelikeAt g p v    -- g(v,v) < 0
#check IsSpacelikeAt g p v   -- g(v,v) > 0
#check IsNullAt g p v        -- g(v,v) = 0, v ≠ 0
```

### 2. Tensor Fields

```lean
-- (r,s)-tensor: r contravariant, s covariant indices
def TensorAt (p : M) (r s : ℕ) : Type*

-- Tensor field: smooth assignment of tensor to each point
structure TensorField (r s : ℕ) where
  toFun : (p : M) → TensorAt p r s
  smooth : ContDiff ⊤ toFun

-- Common cases
def Metric := TensorField 0 2      -- g_μν
def InverseMetric := TensorField 2 0 -- g^μν
def Curvature := TensorField 1 3    -- R^a_{bcd}
```

### 3. Connection and Curvature

```lean
-- Covariant derivative
structure CovariantDerivative where
  conn : TangentBundle → TangentBundle → TangentBundle
  linear_direction : ∀ f X Y, conn (f·X) Y = f·conn X Y
  leibniz : ∀ X Y f, conn X (f·Y) = conn X Y·f + X(f)·Y
  torsion_free : ∀ X Y, conn X Y - conn Y X = [X,Y]

-- Riemann curvature tensor
def RiemannCurvature (∇ : CovariantDerivative) :
    TangentBundle → TangentBundle → TangentBundle → TangentBundle → ℝ :=
  fun X Y Z W =>
    ∇ X (∇ Y Z) W - ∇ Y (∇ X Z) W - ∇ [X,Y] Z W

-- Ricci tensor (contraction of Riemann)
def RicciTensor (∇) (g) : TensorField 0 2

-- Scalar curvature (trace of Ricci)
def ScalarCurvature (∇) (g) : M → ℝ
```

## Key Equations in Lean

### Einstein Field Equations

```lean
structure EinsteinFieldEquations where
  metric : LorentzianMetric
  cosmologicalConstant : ℝ
  stressEnergy : StressEnergyTensor
  equation : ∀ μ ν,
    EinsteinTensor μ ν + Λ·g μ ν = κ·T μ ν
```

**Physical meaning**: Geometry (left side) = Matter (right side)

### Special Relativity Formulas

```lean
-- Lorentz factor
def γ (v : ℝ) : ℝ := 1 / √(1 - v²/c²)

-- Time dilation
def timeDilated (Δt : ℝ) (v : ℝ) : ℝ := γ(v) · Δt

-- Length contraction
def lengthContracted (L : ℝ) (v : ℝ) : ℝ := L / γ(v)

-- Energy-momentum relation
def energyFromMomentum (p : ℝ) (m : ℝ) : ℝ :=
  √((p·c)² + (m·c²)²)

-- Spacetime interval
def ds² (Δx : MinkowskiSpacetime) : ℝ :=
  -(Δx 0)² + (Δx 1)² + (Δx 2)² + (Δx 3)²
```

### Geodesic Equation

```lean
def IsGeodesic (γ : ℝ → M) : Prop :=
  ∀ t, ∇_(γ' t) (γ' t) = 0

-- In coordinates:
-- d²x^a/dτ² + Γ^a_{bc} (dx^b/dτ)(dx^c/dτ) = 0
```

## Common Patterns

### Working with Metrics

```lean
-- Lower indices
def lowerIndex (v : Fin 4 → ℝ) (g : LorentzianMetric) :
    Fin 4 → ℝ :=
  fun μ => ∑ ν, g μ ν · v ν

-- Raise indices
def raiseIndex (ω : Fin 4 → ℝ) (g_inv : (Fin 4 → Fin 4 → ℝ)) :
    Fin 4 → ℝ :=
  fun μ => ∑ ν, g_inv μ ν · ω ν

-- Trace of a (0,2)-tensor
def trace (T : Fin 4 → Fin 4 → ℝ)
    (g_inv : Fin 4 → Fin 4 → ℝ) : ℝ :=
  ∑ μ ν, g_inv μ ν · T μ ν
```

### Perfect Fluid

```lean
structure PerfectFluid where
  energyDensity : M → ℝ        -- ρ
  pressure : M → ℝ             -- p
  fourVelocity : (p : M) → TangentSpace p  -- u^μ
  properties :
    (∀ p, energyDensity p ≥ 0) ∧
    (∀ p, g p (fourVelocity p) (fourVelocity p) = -1)

-- Stress-energy tensor
def T_μν (fluid : PerfectFluid) : TensorField 0 2 :=
  fun p μ ν =>
    (ρ p + p p) · u p μ · u p ν + p p · g p μ ν
```

## Proof Strategies

### Proving Tensor Identities

```lean
-- Example: Symmetry of Ricci tensor
theorem RicciSymmetry [∇.torsion_free] :
    ∀ μ ν, RicciTensor ∇ g μ ν = RicciTensor ∇ g ν μ := by
  intro μ ν
  unfold RicciTensor  -- R_μν = R^a_{μaν}
  apply RiemannSymmetry  -- R^a_{μaν} = R^a_{ναμ}
  -- Use antisymmetry and pair symmetry of Riemann tensor
  sorry
```

### Proving Invariance

```lean
-- Example: Spacetime interval invariance
theorem intervalInvariance (Λ : LorentzTransformation)
    (Δx : MinkowskiSpacetime) :
    ds² Δx = ds² (Λ • Δx) := by
  unfold ds²
  simp [LorentzTransformation.isIsometry]
  -- Use η_μν Λ^μ_ρ Λ^ν_σ = η_ρσ
  ring
```

## Experimental Verification Examples

### Muon Time Dilation

```lean
example (muonLifetime : ℝ) (altitude : ℝ) (v : ℝ)
    (h : lorentzFactor v ≈ 30) :
    let dilatedLifetime := lorentzFactor v · muonLifetime
    let flightTime := altitude / v
    flightTime < dilatedLifetime := by
  -- Muons at 10km reach ground despite 660m decay length
  sorry
```

### GPS Time Correction

```lean
example (satelliteVelocity : ℝ) (satelliteAltitude : ℝ) :
    let SR_correction := lorentzFactor satelliteVelocity
    let GR_correction := gravitationalRedshift satelliteAltitude
    let totalCorrection := SR_correction + GR_correction
    totalCorrection ≈ 38 microseconds/day := by
  -- GPS must correct for both SR and GR effects
  sorry
```

## Common Calculations

### Schwarzschild Radius

```lean
def schwarzschildRadius (M : ℝ) : ℝ :=
  2 · G · M / c²

-- For Earth:
example : schwarzschildRadius 5.972e24 ≈ 8.87e-3 := by
  -- About 9 millimeters!
  sorry

-- For Sun:
example : schwarzschildRadius 1.989e30 ≈ 2.95e3 := by
  -- About 3 kilometers
  sorry
```

### Gravitational Redshift

```lean
def gravitationalRedshift (r : ℝ) (M : ℝ) : ℝ :=
  1 / sqrt(1 - 2·G·M / (r·c²))

-- Light climbing out of gravity well loses energy
-- Wavelength stretches: λ_observed = λ_emitted · sqrt(g_tt)
```

## Type Class Instances

```lean
-- Most structures have these instances:
instance : CoeFun (LorentzianMetric) (fun _ => M → M → ℝ) := sorry
instance : CoeFun (TensorField r s) (fun _ => M → ...) := sorry

-- This allows natural notation:
#eval g p v w      -- instead of g.toTensorField p v w
#eval ∇ X Y        -- instead of ∇.conn X Y
```

## Debugging Tips

### Check Metric Signature

```lean
 theorem checkLorentzianSignature (g : LorentzianMetric) (p : M) :
    ∃ e : Basis (Fin 4) ℝ (TangentSpace p),
      g p (e 0) (e 0) = -1 ∧
      g p (e 1) (e 1) = 1 ∧
      g p (e 2) (e 2) = 1 ∧
      g p (e 3) (e 3) = 1 := by
  sorry
```

### Verify Field Equation

```lean
example (M : Spacetime) (eq : EinsteinFieldEquations) :
    ∀ μ ν, G_μν + Λ·g_μν = κ·T_μν := by
  -- Use eq.equation
  sorry
```

## Further Resources

- **Full Documentation**: See README.md
- **Mathlib4**: https://leanprover-community.github.io/mathlib4_docs/
- **Lean Community**: https://leanprover-community.github.io/

---

**Tip**: Use `#check` to explore definitions and `#print` to see their implementation!
