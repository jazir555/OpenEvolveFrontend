import Lake
open Lake DSL

package «mathlib_project» where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

-- Require Mathlib from the official repository
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «MathlibProject» where
  -- Specify the library structure
  globs := #[.submodules `MathlibProject]

-- Additional targets for specific proof modules
lean_lib «MathlibProject.Core» where
  globs := #[.submodules `MathlibProject.Core]

lean_lib «MathlibProject.Utils» where
  globs := #[.submodules `MathlibProject.Utils]

lean_lib «MathlibProject.Tests» where
  globs := #[.submodules `MathlibProject.Tests]
