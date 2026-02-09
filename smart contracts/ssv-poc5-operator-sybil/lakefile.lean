import Lake
open Lake DSL

package «ssv-poc5-operator-sybil» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «SsvOperatorSybilPoC» where
  srcDir := "formal-proofs"
  roots := #[`sybil_profitability]
