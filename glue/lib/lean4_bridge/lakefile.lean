/-
Lake Build Configuration for RESE Lean 4 Library

This is the Lake build configuration file for the RESE formal verification
library. Lake is Lean 4's standard build system.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All dependencies explicit
- Law of Runtime Truth: Verify all dependencies exist

Usage:
  lake build           # Build the library
  lake test            # Run tests
  lake serve           # Start interactive server
-/

import Lake
open Lake DSL

package RESE {
  -- Add package configuration options

  -- Add more package configuration options here
}

-- Lean 4 library for RESE formal verification
lean_lib RESE {
  -- add library configuration options here
}

-- Test suite for RESE library
lean_lib RESE.Tests {
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.11.0"

meta if get_config? env = some "dev" then
-- dev dependencies so tests can support
-- running the linter itself
require «doc-gen» from git
  "https://github.com/leanprover/doc-gen" @ "main"

-- ============================================================================
-- BUILD TARGETS
-- ============================================================================

target «RESE.olean» (pkg : Package) : FilePath := do
  let leanFile := pkg.dir / "lean4" / "RESE.lean"
  let oleanFile := pkg.buildDir / "lean4" / "RESE.olean"
  buildOlean leanFile oleanFile pkg.root

target «Constraints.olean» (pkg : Package) : FilePath := do
  let leanFile := pkg.dir / "lean4" / "Constraints.lean"
  let oleanFile := pkg.buildDir / "lean4" / "Constraints.olean"
  buildOlean leanFile oleanFile pkg.root

target «FDG.olean» (pkg : Package) : FilePath := do
  let leanFile := pkg.dir / "lean4" / "FDG.lean"
  let oleanFile := pkg.buildDir / "lean4" / "FDG.olean"
  buildOlean leanFile oleanFile pkg.root

-- ============================================================================
-- BUILD HELPERS
-- ============================================================================

def buildOlean (leanFile : FilePath) (oleanFile : FilePath) (root : FilePath) : BuildM (BuildJob FilePath) := do
  let leanCmd :=_PKG.config.buildLean
  let leanExe : FilePath := pkg.targetLeanExe.get
  buildFileAfterDepends oleanFile fun _ => do
    Proc.out {
      cmd := leanExe.toString
      args := #["--make", leanFile.toString]
      env := #[("LEAN_PATH", root.toString :: libraryPaths)]
    }

-- ============================================================================
-- LIBRARY PATHS
-- ============================================================================

def libraryPaths : List String := #[
  "--leanpath=" ++ (← IO.getCurrentDir).toString ++ "/lean4"
]
