"""
Shim module to allow imports with underscores.

This module redirects imports from:
    glue.adapters.rese_leanaide_workflow
To the actual location:
    glue/adapters/rese-leanaide-workflow/
"""

import sys
from pathlib import Path

# Get the actual directory with hyphens
_current_dir = Path(__file__).parent
_actual_dir = _current_dir / "rese-leanaide-workflow"

# Add the src directory to path if it exists
_src_dir = _actual_dir / "src"
if _src_dir.exists():
    sys.path.insert(0, str(_src_dir))

# Import and re-export all test modules
try:
    import test_leanaide_rese_workflow
    __all__ = ['test_leanaide_rese_workflow']
except ImportError:
    __all__ = []
