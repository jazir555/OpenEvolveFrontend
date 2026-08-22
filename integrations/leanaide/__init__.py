"""LeanAIDE integrations package.

Collection of standalone modules for Lean 4 proof generation / verification,
MCTS-MDAP planning, evolutionary strategies, and assorted adapters.

NOTE: This package is used as a flat set of importable scripts (modules import
each other and `lean4_integration*` by bare module name). Several modules are
stubs or require external services (Lean 4 toolchain, OpenAI API, CrewAI,
FastAPI) that are NOT satisfied in this repo. See ACTUAL_STATUS.md.

Wiring: many modules import sibling modules and ``lean4_integration`` by bare
module name. Neither this directory nor ``engines/other`` is a package, so we
add both to ``sys.path`` here (once) so those bare imports resolve when any
module in this package is imported.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OTHER_ENGINES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)), "engines", "other"
)

for _p in (_THIS_DIR, _OTHER_ENGINES_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
