import Lake
open Lake DSL

package rese {
  -- add package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib RESE {
  -- add library configuration options here
}

lean_lib Physics {
  srcDir := "physics_infrastructure"
}
