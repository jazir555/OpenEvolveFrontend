import Lake
open Lake DSL

package «ssv-insolvency-poc» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «SsvInsolvencyPoC» where
  srcDir := "formal-proofs"
  roots := #[`ssv_global_insolvency_proof, `ssv_insolvency_mathlib_proof]
