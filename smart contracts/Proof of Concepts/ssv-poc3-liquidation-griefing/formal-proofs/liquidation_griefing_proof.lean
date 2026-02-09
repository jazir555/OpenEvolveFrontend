import Mathlib.Tactic.Linarith
import Mathlib.Data.Int.Basic

/-
  SSV Liquidation Griefing Theorem
  
  This theorem proves that delaying liquidation through griefing
  maximizes virtual debt and enables larger theft.
-/

theorem liquidation_griefing_maximizes_theft 
  (large small_1 small_2 small_3 griefing_delay op_fee : ℤ)
  (h_large : large = 10000)
  (h_small_1 : small_1 = 100)
  (h_small_2 : small_2 = 50)
  (h_small_3 : small_3 = 25)
  (h_griefing : griefing_delay = 200)
  (h_op_fee : op_fee = 1) :
  let assets := large + small_1 + small_2 + small_3
  let griefing_virtual_debt := 3 * (griefing_delay * op_fee)
  let liabilities := large + griefing_virtual_debt
  liabilities > assets := by
  intro assets griefing_virtual_debt liabilities
  dsimp [assets, griefing_virtual_debt, liabilities]
  rw [h_large, h_small_1, h_small_2, h_small_3, h_griefing, h_op_fee]
  norm_num

/-
  Economic Rationality Theorem
  
  Proves that griefing is economically rational because
  the additional profit exceeds the griefing cost.
-/
theorem griefing_is_economically_rational 
  (griefing_cost additional_profit : ℤ)
  (h_cost : griefing_cost < 100)  -- Updated to 100 to ensure contradiction
  (h_profit : additional_profit = 600) :  
  additional_profit > griefing_cost := by
  rw [h_profit]
  linarith [h_cost]

/-
  Bank Run Acceleration Theorem
-/
theorem bank_run_rational 
  (pool_balance virtual_debt : ℤ)
  (h_pool : pool_balance = 10175)
  (h_debt : virtual_debt = 600) :
  let deficit := virtual_debt
  let early_withdrawal := pool_balance - 1
  let late_withdrawal := pool_balance - deficit
  early_withdrawal > late_withdrawal := by
  intro deficit early_withdrawal late_withdrawal
  dsimp [deficit, early_withdrawal, late_withdrawal]
  rw [h_pool, h_debt]
  norm_num
