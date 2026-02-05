"""
Pytest configuration for Phase II tests.

Sets up paths before importing modules.
"""

import os
import sys

# Get paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_current_dir, "..", "src")
_schemas_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "schemas"))
_lib_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "lib"))
_adapter_dir = os.path.dirname(_current_dir)

# Add to sys.path
sys.path.insert(0, _src_dir)
sys.path.insert(0, _schemas_dir)
sys.path.insert(0, _lib_dir)
sys.path.insert(0, _adapter_dir)
