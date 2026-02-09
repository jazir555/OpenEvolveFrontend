import Lake
open Lake DSL

package «ssv-poc3-liquidation-griefing» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «SsvLiquidationGriefingPoC» where
  srcDir := "formal-proofs"
  roots := #[`liquidation_griefing_proof]
