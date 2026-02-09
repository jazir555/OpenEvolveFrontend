import Lake
open Lake DSL

package «ssv-insolvency-proofs» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «SsvInsolvencyProofs» where
  roots := #[`ssv_global_insolvency_proof, `ssv_insolvency_mathlib_proof]
