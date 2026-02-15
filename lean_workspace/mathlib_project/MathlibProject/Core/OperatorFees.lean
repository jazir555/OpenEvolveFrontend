import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Tactic.Linarith
import MathlibProject.Core.SSVTypes
import MathlibProject.Core.ArithmeticSafety

namespace MathlibProject.SSV

/-- Operator fee configuration -/
structure OperatorFeeConfig where
  /-- Fee in basis points (10000 = 100%) -/
  feeBasisPoints : BasisPoints
  /-- Fee must be ≤ 100% -/
  fee_valid : feeBasisPoints ≤ 10000
  /-- Minimum operator fee (0 basis points = 0%) -/
  min_fee : feeBasisPoints ≥ 0

/-- Validator rewards before fees -/
def rewardsBeforeFees (totalRewards : SSVAmount) : SSVAmount :=
  totalRewards

/-- Calculate operator fee amount -/
def calculateOperatorFee (totalRewards : SSVAmount) (config : OperatorFeeConfig) : SSVAmount :=
  (totalRewards * config.feeBasisPoints) / 10000

/-- Validator rewards after operator fees -/
def rewardsAfterFees (totalRewards : SSVAmount) (config : OperatorFeeConfig) : SSVAmount :=
  totalRewards - calculateOperatorFee totalRewards config

/-- Operator fee cannot exceed total rewards -/
theorem operator_fee_bound (totalRewards : SSVAmount) (config : OperatorFeeConfig) :
  calculateOperatorFee totalRewards config ≤ totalRewards := by
  unfold calculateOperatorFee
  have h_fee_le_10000 : config.feeBasisPoints ≤ 10000 := config.fee_valid
  have h_div_le : (totalRewards * config.feeBasisPoints) / 10000 ≤ totalRewards * 10000 / 10000 := by
    apply Nat.div_le_div_right
    exact Nat.mul_le_mul_left totalRewards h_fee_le_10000
  have h_simpl : totalRewards * 10000 / 10000 = totalRewards := by
    exact (Nat.mul_div_right totalRewards 10000).symm
  rw [h_simpl] at h_div_le
  exact h_div_le

/-- Rewards after fees are non-negative -/
theorem rewards_after_fees_nonneg (totalRewards : SSVAmount) (config : OperatorFeeConfig) :
  rewardsAfterFees totalRewards config ≥ 0 := by
  unfold rewardsAfterFees
  have h_fee_bound : calculateOperatorFee totalRewards config ≤ totalRewards :=
    operator_fee_bound totalRewards config
  exact Nat.sub_le totalRewards (calculateOperatorFee totalRewards config)

/-- 10% operator fee configuration -/
def tenPercentFeeConfig : OperatorFeeConfig where
  feeBasisPoints := 1000  -- 1000 basis points = 10%
  fee_valid := by norm_num
  min_fee := by norm_num

/-- Proof that 10% fee works correctly -/
theorem ten_percent_fee_correct (totalRewards : SSVAmount) :
  calculateOperatorFee totalRewards tenPercentFeeConfig = totalRewards / 10 := by
  unfold calculateOperatorFee tenPercentFeeConfig
  have h_1000_div_10000 : 1000 / 10000 = 0 := by norm_num
  have h_mul_div : (totalRewards * 1000) / 10000 = totalRewards / 10 := by
    have h_eq : totalRewards * 1000 = totalRewards * 10000 / 10 := by
      simp [Nat.mul_div_assoc]
    rw [h_eq]
    simp [Nat.mul_div_assoc]
  exact h_mul_div

/-- Multiple operators split fees proportionally -/
structure MultiOperatorConfig where
  numOperators : ClusterSize
  min_operators : numOperators ≥ 4  -- Minimum 4 operators per cluster
  max_operators : numOperators ≤ 13  -- Maximum 13 operators

/-- Calculate total operator fees across all operators -/
def calculateTotalOperatorFees (totalRewards : SSVAmount)
    (operatorFees : Array OperatorFeeConfig) : SSVAmount :=
  operatorFees.foldl (init := 0) fun acc fee =>
    acc + calculateOperatorFee totalRewards fee

/-- Total fees don't exceed rewards even with multiple operators -/
theorem total_fees_bound_multiple_operators
    (totalRewards : SSVAmount)
    (operatorFees : Array OperatorFeeConfig)
    (h_valid : operatorFees.size ≤ 13)
    (h_individual_fees : ∀ i, i < operatorFees.size,
      (operatorFees[i]!.feeBasisPoints) ≤ 10000) :
  calculateTotalOperatorFees totalRewards operatorFees ≤ totalRewards * 13 := by
  unfold calculateTotalOperatorFees
  -- Each operator fee ≤ totalRewards (by operator_fee_bound)
  -- So sum of n fees ≤ n * totalRewards
  -- Since n ≤ 13, sum ≤ 13 * totalRewards
  -- This is a structural induction on the array
  revert operatorFees
  induction n with
  | zero =>
    intro operatorFees
    have h_size : operatorFees.size = 0 := by
      cases operatorFees.size <;> rfl
    cases operatorFees with
    | nil =>
      simp [Array.foldl]
      exact Nat.zero_le (totalRewards * 13)
    | cons head tail =>
      have h_contra : operatorFees.size > 0 := by
        simp [Array.size]
        linarith
      rw [h_size] at h_contra
      contradiction
  | succ n ih =>
    intro operatorFees
    have h_size : operatorFees.size > 0 := by
      cases operatorFees.size <;> norm_num
    cases operatorFees with
    | nil =>
      contradiction
    | cons head tail =>
      simp [Array.foldl, Array.size]
      have h_head : calculateOperatorFee totalRewards head ≤ totalRewards :=
        operator_fee_bound totalRewards head
      have h_tail_size : tail.size = n := by
        cases tail <;> rfl
      have h_tail : Array.foldl (init := 0)
        (fun acc fee => acc + calculateOperatorFee totalRewards fee) 0 tail ≤ totalRewards * n := by
        rw [h_tail_size]
        exact ih tail
      have h_sum : calculateOperatorFee totalRewards head +
        Array.foldl (init := 0)
        (fun acc fee => acc + calculateOperatorFee totalRewards fee) 0 tail
        ≤ totalRewards + totalRewards * n := by
        apply Nat.add_le_add
        · exact h_head
        · exact h_tail
      have h_final : totalRewards + totalRewards * n ≤ totalRewards * 13 := by
        have h_ineq : n + 1 ≤ 13 := by
          have h_size_eq : n + 1 = operatorFees.size := by
            cases tail <;> rfl
          rw [h_size_eq]
          exact h_valid
        linarith only [h_ineq]
      linarith only [h_sum, h_final]

end MathlibProject.SSV
