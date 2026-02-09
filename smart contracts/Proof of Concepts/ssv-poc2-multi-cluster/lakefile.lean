import Lake
open Lake DSL

package «ssv-poc2-multi-cluster» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «SsvMultiClusterPoC» where
  srcDir := "formal-proofs"
  roots := #[`multi_cluster_insolvency_proof]
