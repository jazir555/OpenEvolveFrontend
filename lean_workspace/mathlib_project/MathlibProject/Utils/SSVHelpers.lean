import Mathlib.Data.Nat.Basic
import Mathlib.Tactic.Linarith
import MathlibProject.Core.SSVTypes
import MathlibProject.Core.ArithmeticSafety

namespace MathlibProject.SSV

/-- Convert basis points to percentage (e.g., 1000 -> 10.0%) -/
def basisPointsToPercent (bp : BasisPoints) : Rat :=
  bp / 100

/-- Calculate per-block fee from annual fee -/
def annualToPerBlockFee (annualFee : SSVAmount) : SSVAmount :=
  annualFee / 2_628_000  -- ~2.628M blocks per year

/-- Calculate blocks until insolvency -/
def blocksUntilInsolvency (balance : SSVAmount) (perBlockFee : SSVAmount) : Nat :=
  if perBlockFee = 0 then
    0  -- Never insolvent with zero fee
  else
    balance / perBlockFee + 1

/-- Theorem: Blocks until insolvency is accurate -/
theorem blocks_until_insolvency_accurate
    (balance perBlockFee : Nat)
    (h_fee_positive : perBlockFee > 0) :
  calculateVirtualDebt (blocksUntilInsolvency balance perBlockFee) perBlockFee > balance := by
  unfold blocksUntilInsolvency calculateVirtualDebt
  split
  · next h_zero =>
    have h_contra : perBlockFee = 0 := h_zero
    have h_contra' : perBlockFee > 0 := h_fee_positive
    contradiction
  · next h_not_zero =>
    have h_div_mul : (balance / perBlockFee + 1) * perBlockFee > balance := by
      have h_div_mul_le : (balance / perBlockFee) * perBlockFee ≤ balance :=
        Nat.div_mul_le_self balance perBlockFee
      have h_eq : (balance / perBlockFee + 1) * perBlockFee =
        (balance / perBlockFee) * perBlockFee + perBlockFee := by
        ring
      rw [h_eq]
      have h_mod_lt : balance % perBlockFee < perBlockFee := Nat.mod_lt balance h_fee_positive
      have h_diff : balance - (balance / perBlockFee) * perBlockFee = balance % perBlockFee := by
        exact (Nat.eq_add_of_sub_eq (Nat.div_mul_le_self balance perBlockFee)).symm
      have h_gt : (balance / perBlockFee) * perBlockFee + perBlockFee > balance := by
        have h_le : balance - ((balance / perBlockFee) * perBlockFee) < perBlockFee := by
          rw [h_diff]
          exact h_mod_lt
        linarith
      exact h_gt
    exact h_div_mul

/-- Safe block amount calculation -/
def safeBlockAmount (blocks : SafeBlockNumber) (fee : SafeSSVAmount)
    (h_safe : blocks.1 * fee.1 < SSV_MAX_SUPPLY) : SafeSSVAmount :=
  ⟨blocks.1 * fee.1, h_safe⟩

/-- Theorem: Safe operations maintain bounds -/
theorem safe_operations_maintain_bounds
    (blocks : SafeBlockNumber)
    (fee : SafeSSVAmount)
    (h_safe : blocks.1 * fee.1 < SSV_MAX_SUPPLY) :
  (safeBlockAmount blocks fee h_safe).1 < SSV_MAX_SUPPLY := by
  unfold safeBlockAmount
  exact h_safe

/-- Check if liquidation threshold is met -/
def checkLiquidationThreshold (healthRatio : Nat) (threshold : LiquidationThreshold) : Bool :=
  healthRatio < threshold

/-- Theorem: Liquidation trigger is monotonic -/
theorem liquidation_monotonic
    (healthRatio1 healthRatio2 : Nat)
    (threshold : LiquidationThreshold)
    (h_degraded : healthRatio1 > healthRatio2)
    (h_should_liquidate1 : checkLiquidationThreshold healthRatio1 threshold = true) :
  checkLiquidationThreshold healthRatio2 threshold = true := by
  unfold checkLiquidationThreshold at *
  split at *
  · next h_lt1 =>
    have h_lt2 : healthRatio2 < threshold := by
      have h_antitone : healthRatio2 < healthRatio1 ∧ healthRatio1 < threshold := by
        constructor
        · exact Nat.lt_of_lt_of_le (by linarith) (Nat.le_refl _)
        · exact h_lt1
      exact Nat.lt_trans h_antitone.1 h_antitone.2
    exact h_lt2
  · next h_ge1 =>
    have h_contra : healthRatio1 ≥ threshold := h_ge1
    have h_contra' : healthRatio1 < threshold := by
      unfold checkLiquidationThreshold at h_should_liquidate1
      split at h_should_liquidate1
      · next h_lt =>
        exact h_lt
      · next h_ge =>
        have h_false : checkLiquidationThreshold healthRatio1 threshold = false := by
          rfl
        rw [h_false] at h_should_liquidate1
        contradiction
    contradiction

end MathlibProject.SSV
