"""
Shim package to expose OpenEvolve gauntlets from core-projects/openevolve.
"""

from __future__ import annotations

import os
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_core_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "core-projects", "openevolve", "openevolve", "gauntlets")
)

if os.path.isdir(_core_path) and _core_path not in __path__:
    __path__.append(_core_path)

# Re-export public gauntlet components
try:
    from openevolve.gauntlets import *  # noqa: F401,F403
except Exception:
    # Allow import of this shim even if core package is not yet available
    pass
