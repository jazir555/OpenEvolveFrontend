import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Tactic.Linarith
import MathlibProject.Core.SSVTypes
import MathlibProject.Core.ArithmeticSafety
import MathlibProject.Core.OperatorFees

namespace MathlibProject.SSV

/-- Cluster state -/
inductive ClusterState where
  | active : ClusterState
  | liquidated : ClusterState
  deriving Repr

/-- Cluster configuration -/
structure ClusterConfig where
  /-- Number of operators in the cluster -/
  operatorCount : ClusterSize
  /-- Minimum operators for consensus -/
  minOperators : operatorCount ≥ 4
  /-- Maximum operators per cluster -/
  maxOperators : operatorCount ≤ 13
  /-- Liquidation threshold in basis points -/
  liquidationThreshold : LiquidationThreshold
  /-- Threshold must be reasonable (e.g., 80% = 8000 basis points) -/
  threshold_valid : liquidationThreshold ≥ 8000

/-- Cluster state with balances -/
structure ClusterState where
  /-- Current balance of SSV tokens -/
  balance : SSVAmount
  /-- Virtual debt (accrued but not yet withdrawn) -/
  virtualDebt : SSVAmount
  /-- Number of blocks since last update -/
  blocksElapsed : BlockNumber
  /-- Whether cluster is liquidated -/
  isLiquidated : Bool

/-- Calculate health ratio (balance / total_liabilities) -/
def calculateHealthRatio (state : ClusterState) : Nat :=
  if state.virtualDebt = 0 then
    10000  -- Healthy: no debt
  else
    (state.balance * 10000) / (state.balance + state.virtualDebt)

/-- Check if cluster should be liquidated -/
def shouldLiquidate (config : ClusterConfig) (state : ClusterState) : Bool :=
  state.isLiquidated = false ∧
  calculateHealthRatio state < config.liquidationThreshold

/-- Liquidation reduces virtual debt to zero -/
def liquidateCluster (state : ClusterState) : ClusterState :=
  { state with
    virtualDebt := 0
    isLiquidated := true
  }

/-- Health ratio is bounded between 0 and 10000 -/
theorem health_ratio_bounded (state : ClusterState) :
  0 ≤ calculateHealthRatio state ∧ calculateHealthRatio state ≤ 10000 := by
  unfold calculateHealthRatio
  split
  · next h_zero =>
    constructor
    · exact Nat.zero_le 10000
    · exact Nat.le_refl 10000
  · next h_not_zero =>
    constructor
    · exact Nat.zero_le _
    · have h_div_le : (state.balance * 10000) / (state.balance + state.virtualDebt) ≤ 10000 := by
        have h_denom_pos : state.balance + state.virtualDebt > 0 := by
          have h_balance_pos : state.balance ≥ 0 := Nat.zero_le state.balance
          have h_debt_pos : state.virtualDebt > 0 := by
            have h_gt : state.virtualDebt ≠ 0 := h_not_zero
            exact Nat.pos_of_ne_zero h_gt
          linarith
        apply Nat.div_le_of_le_mul
        linarith
      exact h_div_le

/-- Liquidation zero virtual debt -/
theorem liquidation_zeros_debt (state : ClusterState) :
  (liquidateCluster state).virtualDebt = 0 := by
  unfold liquidateCluster
  rfl

/-- Liquidation sets isLiquidated flag -/
theorem liquidation_sets_flag (state : ClusterState) :
  (liquidateCluster state).isLiquidated = true := by
  unfold liquidateCluster
  rfl

/-- Once liquidated, cluster remains liquidated -/
theorem liquidation_persistent (state : ClusterState) :
  (liquidateCluster (liquidateCluster state)).isLiquidated = true := by
  unfold liquidateCluster
  rfl

/-- Insolvency occurs when virtual debt exceeds balance -/
def isInsolvent (state : ClusterState) : Bool :=
  state.virtualDebt > state.balance

/-- Insolvency implies health ratio < 5000 (50%) -/
theorem insolvency_implies_low_health (state : ClusterState)
    (h_insolvent : isInsolvent state = true) :
  calculateHealthRatio state < 5000 := by
  unfold isInsolvent at h_insolvent
  unfold calculateHealthRatio
  split
  · next h_debt_zero =>
    have h_contra : state.virtualDebt = 0 := h_debt_zero
    have h_insolvent' : state.virtualDebt > state.balance := by
      rw [h_contra] at h_insolvent
      exact h_insolvent
    have h_balance_neg : state.balance < 0 := by
      have h_zero_gt : 0 > state.balance := by
        have h_eq : state.virtualDebt = 0 := h_debt_zero
        rw [h_eq] at h_insolvent'
        exact h_insolvent'
      exact h_zero_gt
    have h_balance_nonneg : state.balance ≥ 0 := Nat.zero_le state.balance
    linarith
  · next h_debt_not_zero =>
    have h_insolvent' : state.virtualDebt > state.balance := by
      rw [isInsolvent] at h_insolvent
      exact h_insolvent
    have h_ratio_lt_5000 :
      (state.balance * 10000) / (state.balance + state.virtualDebt) < 5000 := by
      have h_denom : state.balance + state.virtualDebt > 0 := by
        have h_debt_pos : state.virtualDebt > 0 := by
          have h_ne : state.virtualDebt ≠ 0 := h_debt_not_zero
          exact Nat.pos_of_ne_zero h_ne
        linarith
      -- We need to show: (balance * 10000) / (balance + virtualDebt) < 5000
      -- This is equivalent to: balance * 10000 < 5000 * (balance + virtualDebt)
      -- Which simplifies to: 2 * balance < balance + virtualDebt
      -- Which is: balance < virtualDebt (true by insolvency)
      have h_equiv : ((state.balance * 10000) / (state.balance + state.virtualDebt) < 5000) ↔
        (state.balance * 10000 < 5000 * (state.balance + state.virtualDebt)) := by
        apply Nat.div_lt_iff_lt_mul (state.balance + state.virtualDebt) 5000
        · exact h_denom
        · norm_num
      apply h_equiv.mpr
      -- Simplify: balance * 10000 < 5000 * (balance + virtualDebt)
      -- This is: 2 * balance * 5000 < 5000 * balance + 5000 * virtualDebt
      -- Which is: 2 * balance < balance + virtualDebt
      -- Which is: balance < virtualDebt
      have h_simpl : state.balance * 10000 < 5000 * (state.balance + state.virtualDebt) ↔
        state.balance < state.virtualDebt := by
        constructor
        · intro h_lt
          have h_expand : 5000 * (state.balance + state.virtualDebt) =
            5000 * state.balance + 5000 * state.virtualDebt := by
            ring
          rw [h_expand] at h_lt
          have h_balance_simpl : state.balance * 10000 = 2 * state.balance * 5000 := by
            ring
          rw [h_balance_simpl] at h_lt
          -- Now: 2 * balance * 5000 < balance * 5000 + virtualDebt * 5000
          have h_sub : 2 * state.balance * 5000 - state.balance * 5000 < state.virtualDebt * 5000 := by
            linarith only [h_lt]
          have h_simpl2 : 2 * state.balance * 5000 - state.balance * 5000 = state.balance * 5000 := by
            ring
          rw [h_simpl2] at h_sub
          have h_final : state.balance * 5000 < state.virtualDebt * 5000 := by
            exact h_sub
          have h_cancel_5000 : state.balance < state.virtualDebt := by
            have h_pos : 5000 > 0 := by norm_num
            exact (Nat.mul_lt_mul_right h_pos).mp h_final
          exact h_cancel_5000
        · intro h_lt_balance
          have h_expand : 5000 * (state.balance + state.virtualDebt) =
            5000 * state.balance + 5000 * state.virtualDebt := by
            ring
          rw [h_expand]
          have h_balance_simpl : state.balance * 10000 = 2 * state.balance * 5000 := by
            ring
          rw [h_balance_simpl]
          have h_mul : state.balance * 5000 < state.virtualDebt * 5000 := by
            have h_pos : 5000 > 0 := by norm_num
            exact Nat.mul_lt_mul_right h_pos h_lt_balance
          linarith only [h_mul]
      exact h_simpl.mpr h_insolvent'
      exact h_ratio_lt_5000

/-- Safe liquidation theorem: liquidation prevents debt growth -/
theorem liquidation_prevents_debt_growth (state : ClusterState) :
  (liquidateCluster state).virtualDebt ≤ state.virtualDebt := by
  unfold liquidateCluster
  exact Nat.zero_le state.virtualDebt

end MathlibProject.SSV
