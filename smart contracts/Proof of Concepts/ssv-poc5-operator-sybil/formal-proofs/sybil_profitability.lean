import Mathlib.Data.Nat.Basic

/--
  SSV Operator Sybil Profitability Theorem (POC 5)
-/

def investment (sybils : ℕ) (deposit : ℕ) : ℕ := sybils * deposit

def revenue (sybils : ℕ) (fee : ℕ) (time : ℕ) : ℕ := sybils * fee * time

theorem sybil_infinite_roi (sybils : ℕ) (deposit : ℕ) (fee : ℕ) 
  (h_s : sybils > 0) (h_f : fee > 0) :
  ∀ (target_profit : ℕ), ∃ (time : ℕ), 
    revenue sybils fee time > investment sybils deposit + target_profit :=
by
  intro target_profit
  let total_cost := investment sybils deposit + target_profit
  let t := total_cost + 1
  exists t
  unfold revenue
  -- slope = sybils * fee
  have h_slope : sybils * fee ≥ 1 := by
    apply Nat.succ_le_of_lt
    apply Nat.mul_pos h_s h_f
  calc
    sybils * fee * t ≥ 1 * t := Nat.mul_le_mul_right t h_slope
    _ = t := by rw [Nat.one_mul]
    _ = total_cost + 1 := rfl
    _ > total_cost := Nat.lt_succ_self total_cost