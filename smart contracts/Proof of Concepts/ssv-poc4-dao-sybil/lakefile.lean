import Lake
open Lake DSL

package «ssv-poc4-dao-sybil» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «SsvDaoSybilPoC» where
  srcDir := "formal-proofs"
  roots := #[`dao_insolvency]
