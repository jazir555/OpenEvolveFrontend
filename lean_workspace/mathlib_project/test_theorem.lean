import Mathlib.Tactic.Linarith
import MathlibProject.Core.SSVTypes
import MathlibProject.Core.ArithmeticSafety
import MathlibProject.Core.OperatorFees
import MathlibProject.Core.ClusterLiquidation
import MathlibProject.Core.InsolvencyTheorem

/--
  Comprehensive SSV Insolvency Theorem Test

  This file demonstrates the complete insolvency proof with realistic parameters.
-/

namespace SSVInsolvencyTest

/-- Realistic SSV network parameters -/
def INITIAL_BALANCE : Nat := 32 * 10^18  -- 32 SSV tokens (1 validator)
def PER_BLOCK_FEE : Nat := 1000 * 10^12  -- 0.001 SSV per block
def BLOCKS_PER_YEAR : Nat := 2_628_000
def YEARS_UNTIL_INSOLVENCY : Nat := 10

/-- Calculate blocks until insolvency -/
def blocks_until_insolvency : Nat :=
  INITIAL_BALANCE / PER_BLOCK_FEE + 1

/-- Verify insolvency occurs -/
theorem insolvency_occurs_realistic :
  let virtualDebt := MathlibProject.SSV.calculateVirtualDebt blocks_until_insolvency PER_BLOCK_FEE
  let totalLiabilities := MathlibProject.SSV.calculateTotalLiabilities INITIAL_BALANCE virtualDebt
  totalLiabilities > INITIAL_BALANCE := by
  intro virtualDebt totalLiabilities
  unfold blocks_until_insolvency INITIAL_BALANCE PER_BLOCK_FEE
  have h_balance_pos : (32 * 10^18) > 0 := by linarith
  have h_blocks_pos : ((32 * 10^18) / (1000 * 10^12) + 1) > 0 := by linarith
  have h_fee_pos : (1000 * 10^12) > 0 := by linarith
  exact MathlibProject.SSV.ssv_insolvency_possible
    (32 * 10^18)
    ((32 * 10^18) / (1000 * 10^12) + 1)
    (1000 * 10^12)
    h_balance_pos
    h_blocks_pos
    h_fee_pos

/-- Verify operator fee is bounded -/
theorem operator_fee_safe :
  let rewards := 100 * 10^18
  let fee := MathlibProject.SSV.calculateOperatorFee rewards MathlibProject.SSV.tenPercentFeeConfig
  fee ≤ rewards ∧ fee = 10 * 10^18 := by
  intro rewards fee
  constructor
  · exact MathlibProject.SSV.operator_fee_bound (100 * 10^18) MathlibProject.SSV.tenPercentFeeConfig
  · have h_calc : (rewards * 1000) / 10000 = rewards / 10 := by
      have h_eq : (100 * 10^18 * 1000) / 10000 = 10 * 10^18 := by
        have h_simpl : (100 * 10^18 * 1000) / 10000 = (100 * 1000 / 10000) * 10^18 := by
          rw [Nat.mul_assoc, Nat.mul_div_assoc]
        rw [h_simpl]
        have h_div : 100 * 1000 / 10000 = 10 := by
          norm_num
        rw [h_div]
        norm_num
      unfold rewards
      exact h_eq
    unfold fee calculateOperatorFee MathlibProject.SSV.tenPercentFeeConfig
    unfold rewards
    exact h_calc

/-- Verify liquidation stops debt growth -/
theorem liquidation_works :
  let state : MathlibProject.SSV.ClusterState := {
    balance := 10 * 10^18
    virtualDebt := 50 * 10^18
    blocksElapsed := 1000000
    isLiquidated := false
  }
  let liquidated := MathlibProject.SSV.liquidateCluster state
  liquidated.virtualDebt = 0 ∧ liquidated.isLiquidated = true := by
  intro state liquidated
  constructor
  · exact MathlibProject.SSV.liquidation_zeros_debt state
  · exact MathlibProject.SSV.liquidation_sets_flag state

/-- Verify cluster size constraints -/
theorem cluster_size_constraints :
  let config : MathlibProject.SSV.ClusterConfig := {
    operatorCount := 4
    minOperators := by norm_num
    maxOperators := by norm_num
    liquidationThreshold := 8000
    threshold_valid := by norm_num
  }
  config.operatorCount ≥ 4 ∧ config.operatorCount ≤ 13 := by
  intro config
  constructor
  · exact config.minOperators
  · exact config.maxOperators

/-- Verify arithmetic safety -/
theorem arithmetic_safe :
  let max := MathlibProject.SSV.SSV_MAX_SUPPLY
  let a : Nat := 1000 * 10^18
  let b : Nat := 2000 * 10^18
  h : a + b < max := by
    intro a b h
    unfold max a
    have h_sum : 3000 * 10^18 = 3 * 10^21 := by norm_num
    have h_max : 10_000_000 * 10^18 = 10^25 := by norm_num
    linarith

end SSVInsolvencyTest
