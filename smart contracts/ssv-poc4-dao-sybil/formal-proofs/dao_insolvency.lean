import Mathlib.Data.Nat.Basic

/--
  SSV DAO Insolvency Theorem (POC 4)
-/

def dao_earnings (blocks : ℕ) (clusters : ℕ) (fee : ℕ) : ℕ :=
  blocks * clusters * fee

def total_assets : ℕ := 10000

theorem dao_insolvency_unbounded (clusters : ℕ) (fee : ℕ) (h_c : clusters > 0) (h_f : fee > 0) :
  ∃ (blocks : ℕ), dao_earnings blocks clusters fee > total_assets :=
by
  let b := total_assets + 1
  exists b
  unfold dao_earnings
  have h_prod : clusters * fee ≥ 1 := by
    apply Nat.succ_le_of_lt
    apply Nat.mul_pos h_c h_f
  calc
    b * clusters * fee = b * (clusters * fee) := by rw [Nat.mul_assoc]
    _ ≥ b * 1 := Nat.mul_le_mul_left b h_prod
    _ = total_assets + 1 := by rw [Nat.mul_one]
    _ > total_assets := Nat.lt_succ_self total_assets