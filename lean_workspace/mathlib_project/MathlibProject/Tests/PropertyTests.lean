import Mathlib.Data.Nat.Basic
import Mathlib.Tactic.Linarith
import MathlibProject.Core.SSVTypes
import MathlibProject.Core.ArithmeticSafety
import MathlibProject.Core.OperatorFees
import MathlibProject.Core.ClusterLiquidation
import MathlibProject.Core.InsolvencyTheorem

namespace MathlibProject.SSV.Tests

/-- Test: Verify SSV supply is within Ethereum bounds -/
#eval ssv_supply_within_eth_bounds

/-- Test: 10% operator fee calculation -/
theorem test_ten_percent_fee :
  let rewards := 1000 * 10^18  -- 1000 SSV tokens
  let fee := calculateOperatorFee rewards tenPercentFeeConfig
  fee = 100 * 10^18 := by
  intro rewards fee
  unfold calculateOperatorFee tenPercentFeeConfig
  have h_calc : (rewards * 1000) / 10000 = rewards / 10 := by
    have h_mul_div : ∀ (r : Nat), (r * 1000) / 10000 = r / 10 := by
      intro r
      have h_fact : r * 1000 = r * 10000 / 10 := by
        have h_simpl : r * 1000 * 10 = r * 10000 := by
          ring_nf
        have h_div : r * 1000 = (r * 10000) / 10 := by
          have h_eq : r * 1000 * 10 = r * 10000 := by ring_nf
          have h_pos : 10 > 0 := by norm_num
          exact Nat.eq_mul_of_div_eq h_pos (Eq.symm h_eq)
        exact h_div
      rw [h_fact]
      have h_div_cancel : (r * 10000 / 10) / 10000 = r / 10 := by
        have h_assoc : (r * 10000 / 10) / 10000 = r * (10000 / 10) / 10000 := by
          rw [Nat.mul_div_assoc _ _ (by norm_num)]
        rw [h_assoc]
        have h_simpl : 10000 / 10 = 1000 := by norm_num
        rw [h_simpl]
        have h_final : (r * 1000) / 10000 = r / 10 := by
          have h_fact2 : r * 1000 = r * 10000 / 10 := by
            have h_eq : r * 1000 * 10 = r * 10000 := by ring_nf
            have h_pos : 10 > 0 := by norm_num
            exact Nat.eq_mul_of_div_eq h_pos (Eq.symm h_eq)
          rw [h_fact2]
          rw [Nat.mul_div_assoc _ _ (by norm_num)]
          rfl
        exact h_final
      exact h_div_cancel
    exact h_mul_div rewards
  have h_rew : rewards = 1000 * 10^18 := by rfl
  rw [h_rew]
  have h_simpl : (1000 * 10^18 * 1000) / 10000 = 100 * 10^18 := by
    have h_assoc : (1000 * 10^18 * 1000) / 10000 = (1000 * 1000 / 10000) * 10^18 := by
      rw [Nat.mul_assoc, Nat.mul_div_assoc]
    rw [h_assoc]
    have h_div : 1000 * 1000 / 10000 = 100 := by
      norm_num
    rw [h_div]
    rfl
  rw [h_simpl]
  exact h_calc

/-- Test: Insolvency occurs with positive blocks and fees -/
theorem test_insolvency_occurs :
  let balance := 1000 * 10^18
  let blocks := 1000000
  let fee := 1000
  let virtualDebt := calculateVirtualDebt blocks fee
  let totalLiabilities := calculateTotalLiabilities balance virtualDebt
  totalLiabilities > balance := by
  intro balance blocks fee virtualDebt totalLiabilities
  have h_balance_pos : balance > 0 := by
    unfold balance
    linarith
  have h_blocks_pos : blocks > 0 := by
    unfold blocks
    linarith
  have h_fee_pos : fee > 0 := by
    unfold fee
    linarith
  exact ssv_insolvency_possible balance blocks fee h_balance_pos h_blocks_pos h_fee_pos

/-- Test: Health ratio is bounded -/
theorem test_health_ratio_bounded :
  let state : ClusterState := {
    balance := 1000 * 10^18
    virtualDebt := 500 * 10^18
    blocksElapsed := 1000000
    isLiquidated := false
  }
  let ratio := calculateHealthRatio state
  ratio ≥ 0 ∧ ratio ≤ 10000 := by
  intro state ratio
  unfold ratio
  exact health_ratio_bounded state

/-- Test: Liquidation zeros debt -/
theorem test_liquidation_zeros_debt :
  let state : ClusterState := {
    balance := 100 * 10^18
    virtualDebt := 200 * 10^18
    blocksElapsed := 500000
    isLiquidated := false
  }
  let liquidated := liquidateCluster state
  liquidated.virtualDebt = 0 := by
  intro state liquidated
  unfold liquidated
  exact liquidation_zeros_debt state

/-- Test: Operator fee doesn't exceed rewards -/
theorem test_operator_fee_bound :
  let rewards := 500 * 10^18
  let config := tenPercentFeeConfig
  calculateOperatorFee rewards config ≤ rewards := by
  intro rewards config
  unfold rewards config
  exact operator_fee_bound (500 * 10^18) tenPercentFeeConfig

/-- Test: Blocks until insolvency calculation -/
theorem test_blocks_until_insolvency :
  let balance := 1000 * 10^18
  let perBlockFee := 1000 * 10^12  -- 0.001 SSV per block
  let blocks := blocksUntilInsolvency balance perBlockFee
  blocks > 0 := by
  intro balance perBlockFee blocks
  unfold blocks balance perBlockFee
  split
  · next h_zero =>
    have h_contra : perBlockFee = 0 := h_zero
    have h_actual : perBlockFee = 1000 * 10^12 := by rfl
    contradiction
  · next h_not_zero =>
    have h_pos : (1000 * 10^18) / (1000 * 10^12) + 1 > 0 := by
      linarith
    exact h_pos

/-- Property: Adding operator fee reduces rewards -/
theorem property_fee_reduces_rewards (totalRewards : SSVAmount) (config : OperatorFeeConfig) :
  rewardsAfterFees totalRewards config ≤ totalRewards := by
  unfold rewardsAfterFees
  have h_fee_bound : calculateOperatorFee totalRewards config ≤ totalRewards :=
    operator_fee_bound totalRewards config
  exact Nat.sub_le totalRewards (calculateOperatorFee totalRewards config)

/-- Property: Virtual debt is monotonic in blocks -/
theorem property_virtual_debt_monotonic
    (blocks1 blocks2 : BlockNumber) (fee : SSVAmount)
    (h_gt : blocks1 < blocks2) (h_fee_pos : fee > 0) :
  calculateVirtualDebt blocks1 fee < calculateVirtualDebt blocks2 fee := by
  unfold calculateVirtualDebt
  exact Nat.mul_lt_mul_of_pos_left h_gt h_fee_pos

/-- Property: Total liabilities monotonic in virtual debt -/
theorem property_liabilities_monotonic
    (balance : SSVAmount) (debt1 debt2 : SSVAmount)
    (h_gt : debt1 < debt2) :
  calculateTotalLiabilities balance debt1 < calculateTotalLiabilities balance debt2 := by
  unfold calculateTotalLiabilities
  exact Nat.add_lt_add_left h_gt balance

/-- Property: Liquidation is idempotent -/
theorem property_liquidation_idempotent (state : ClusterState) :
  liquidateCluster (liquidateCluster state) = liquidateCluster state := by
  unfold liquidateCluster
  rfl

end MathlibProject.SSV.Tests
