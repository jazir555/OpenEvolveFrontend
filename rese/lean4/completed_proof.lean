import Mathlib
import Mathlib.MeasureTheory.Integral.ProbabilityMass

/-!
# Statistical Mechanics: Partition Function

This file contains standalone proofs about the partition function
and its properties in statistical mechanics.

**Theorems**:
- Boltzmann distribution from partition function
- Thermodynamic quantities from Z
- Partition function factorizes for independent systems

**Task**: Formalize these proofs in Lean 4

**Proof Goals**:
1. Define partition function Z = Σ exp(-βE_i)
2. Derive probability P_i = exp(-βE_i)/Z
3. Express energy, entropy, free energy in terms of Z
-/

namespace StatisticalMechanics

universe u

open MeasureTheory BigOperators Real

variable {Ω : Type*} [Fintype Ω] [DecidableEq Ω]


/-!
## Basic Definitions
-/

/-- Inverse temperature β = 1/(k_B T) -/
structure InverseTemperature where
  value : ℝ
  positive : 0 < value


/-- Boltzmann constant (in natural units where k_B = 1) -/
def boltzmannConstant : ℝ := 1  -- Natural units: k_B = 1


/-!
## Partition Function
-/

/-- Partition function Z = Σ_i exp(-βE_i) -/
def partitionFunction (β : InverseTemperature) (E : Ω → ℝ) : ℝ :=
  ∑ i, Real.exp (-β.value * E i)


/-- **Theorem**: Partition function is positive

Z = Σ exp(-βE_i) > 0 for finite temperature (β > 0)
-/
theorem partition_function_positive
    (β : InverseTemperature) (E : Ω → ℝ) :
    0 < partitionFunction β E := by
  unfold partitionFunction
  apply Finset.sum_pos
  · intro i _
    apply Real.exp_pos
  · cases Fintype.elems_nonempty with
    | intro i => exists i


/-!
## Boltzmann Distribution
-/

/-- Probability of state i: P_i = exp(-βE_i) / Z -/
def boltzmannProbability
    (β : InverseTemperature) (E : Ω → ℝ) (i : Ω) : ℝ :=
  Real.exp (-β.value * E i) / partitionFunction β E


/-- **Theorem**: Boltzmann distribution is normalized

Σ_i P_i = 1
-/
theorem boltzmann_distribution_normalized
    (β : InverseTemperature) (E : Ω → ℝ) :
    ∑ i, boltzmannProbability β E i = 1 := by
  unfold boltzmannProbability partitionFunction
  rw [← Finset.sum_div]
  apply div_self
  apply ne_of_gt
  apply partition_function_positive


/-- **Theorem**: Probabilities are non-negative

P_i ≥ 0 for all i
-/
theorem boltzmann_probability_nonneg
    (β : InverseTemperature) (E : Ω → ℝ) (i : Ω) :
    0 ≤ boltzmannProbability β E i := by
  unfold boltzmannProbability
  apply div_nonneg
  · apply Real.exp_pos.le
  · apply (partition_function_positive β E).le


/-!
## Thermodynamic Quantities
-/

/-- Expected energy: ⟨E⟩ = Σ P_i E_i = -∂lnZ/∂β -/
def expectedEnergy (β : InverseTemperature) (E : Ω → ℝ) : ℝ :=
  ∑ i, boltzmannProbability β E i * E i


/-- Helper lemma for derivative of log -/
lemma deriv_of_log
    (f : ℝ → ℝ) (x : ℝ) (h_pos : 0 < f x) (h_diff : DifferentiableAt ℝ f x) :
    deriv (Real.log ∘ f) x = deriv f x / f x := by
  have h_log_diff : DifferentiableAt ℝ Real.log (f x) := by
    apply Real.differentiableAt_log.mpr
    exact h_pos
  rw [deriv_comp x h_log_diff h_diff]
  rw [Real.deriv_log (f x) h_pos]
  ring


/-- **Theorem**: Energy can be computed from partition function

⟨E⟩ = -d(ln Z)/dβ
-/
theorem energy_from_partition_function
    (β : InverseTemperature) (E : Ω → ℝ) :
    expectedEnergy β E = - deriv (fun t => Real.log (∑ i, Real.exp (-t * E i))) β.value := by
  let Z := fun (t : ℝ) => ∑ i, Real.exp (-t * E i)
  have hZ_pos : ∀ t, 0 < Z t := by
    intro t
    apply Finset.sum_pos
    · intro i _; apply Real.exp_pos
    · cases Fintype.elems_nonempty with | intro i => exists i
  have hZ_diff : ∀ t, DifferentiableAt ℝ Z t := by
    intro t
    apply DifferentiableAt.sum
    intro i _
    apply DifferentiableAt.exp
    apply DifferentiableAt.neg
    apply differentiableAt_id.mul_const
  rw [deriv_of_log Z β.value (hZ_pos β.value) (hZ_diff β.value)]
  have h_deriv_Z : deriv Z β.value = ∑ i, -E i * Real.exp (-β.value * E i) := by
    rw [deriv_sum (fun i _ => hZ_diff β.value i)]
    apply Finset.sum_congr rfl
    intro i _
    rw [deriv_exp, deriv_neg, deriv_mul_const, deriv_id'']
    · ring
    · apply DifferentiableAt.neg
      apply differentiableAt_id.mul_const
  rw [h_deriv_Z]
  unfold expectedEnergy boltzmannProbability partitionFunction
  simp only [neg_div, Finset.sum_neg_distrib, neg_neg]
  rw [← Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  ring


/-- Entropy: S = -k_B Σ P_i ln P_i -/
def entropy (β : InverseTemperature) (E : Ω → ℝ) : ℝ :=
  -boltzmannConstant * ∑ i,
    let p := boltzmannProbability β E i
    p * Real.log p


/-- **Theorem**: Entropy in terms of partition function

S = k_B (ln Z + β⟨E⟩)
-/
theorem entropy_partition_function_form
    (β : InverseTemperature) (E : Ω → ℝ) :
    let Z := partitionFunction β E
    let lnZ := Real.log Z
    let avgE := expectedEnergy β E
    entropy β E = boltzmannConstant * (lnZ + β.value * avgE) := by
  intro Z lnZ avgE
  unfold entropy expectedEnergy boltzmannProbability partitionFunction
  have hZ_pos : 0 < ∑ j, Real.exp (-β.value * E j) := partition_function_positive β E
  have log_simp : ∀ i, Real.log (Real.exp (-β.value * E i) / (∑ j, Real.exp (-β.value * E j))) =
      -β.value * E i - Real.log (∑ j, Real.exp (-β.value * E j)) := by
    intro i
    rw [Real.log_div (Real.exp_pos (-β.value * E i)).ne' hZ_pos.ne']
    rw [Real.log_exp]
    ring
  conv => lhs; arg 2; arg 2; ext; rw [log_simp]
  rw [show ∀ i, (Real.exp (-β.value * E i) / (∑ j, Real.exp (-β.value * E j))) *
        (-β.value * E i - Real.log (∑ j, Real.exp (-β.value * E j))) =
        - (β.value * E i * (Real.exp (-β.value * E i) / (∑ j, Real.exp (-β.value * E j)))) -
        (Real.log (∑ j, Real.exp (-β.value * E j)) * (Real.exp (-β.value * E i) / (∑ j, Real.exp (-β.value * E j)))) from by
        intro i; ring]
  rw [Finset.sum_sub, ← Finset.sum_mul]
  rw [show (∑ i, Real.exp (-β.value * E i) / ∑ j, Real.exp (-β.value * E j)) = 1 by
    rw [← Finset.sum_div]
    apply div_self hZ_pos.ne']
  rw [boltzmannConstant]
  simp only [one_mul, neg_mul, neg_sub, mul_one]
  rw [expectedEnergy]
  unfold boltzmannProbability partitionFunction
  simp only [mul_comm]
  ring


/-- Helmholtz free energy: F = -k_B T ln Z -/
def helmholtzFreeEnergy (β : InverseTemperature) (E : Ω → ℝ) : ℝ :=
  -boltzmannConstant / β.value * Real.log (partitionFunction β E)


/-- **Theorem**: F = ⟨E⟩ - TS

Proof: Use F = -kT ln Z and S = k(ln Z + β⟨E⟩)
-/
theorem free_energy_relation
    (β : InverseTemperature) (E : Ω → ℝ) :
    helmholtzFreeEnergy β E =
    expectedEnergy β E -
    (1 / β.value) * entropy β E := by
  let Z := partitionFunction β E
  let lnZ := Real.log Z
  let avgE := expectedEnergy β E
  have h_entropy_formula : entropy β E = boltzmannConstant * (lnZ + β.value * avgE) := by
    exact entropy_partition_function_form β E
  unfold helmholtzFreeEnergy
  rw [h_entropy_formula]
  rw [boltzmannConstant]
  field_simp [β.positive.ne']
  ring


/-!
## Factorization
-/

/-- **Theorem**: Partition function factorizes for independent systems

If system = A ⊗ B with no interaction, then Z_total = Z_A * Z_B
-/
theorem partition_function_factorizes
    {Ω_A Ω_B : Type*} [Fintype Ω_A] [Fintype Ω_B]
    [DecidableEq Ω_A] [DecidableEq Ω_B]
    (β : InverseTemperature)
    (E_A : Ω_A → ℝ) (E_B : Ω_B → ℝ)
    (E_total : Ω_A × Ω_B → ℝ)
    (h_independent : ∀ p, E_total p = E_A p.1 + E_B p.2) :
    partitionFunction β E_total = partitionFunction β E_A * partitionFunction β E_B := by
  unfold partitionFunction
  rw [Finset.sum_product]
  rw [Finset.sum_mul]
  congr
  ext i
  rw [Finset.mul_sum]
  congr
  ext j
  rw [h_independent]
  rw [neg_mul, add_mul, neg_add, Real.exp_add]
  ring

end StatisticalMechanics
