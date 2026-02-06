"""
Compatibility alias for curie-globalchem-integration

Python package names cannot contain hyphens. This module provides
a compatibility alias to the curie-globalchem-integration package.
"""

import sys
import importlib.util
import os

# Load the actual module from the hyphenated directory
_dir = os.path.join(os.path.dirname(__file__), 'curie-globalchem-integration', 'src')
_spec = importlib.util.spec_from_file_location(
    'curie_globalchem_adapter',
    os.path.join(_dir, 'curie_globalchem_adapter.py')
)
_curie_module = importlib.util.module_from_spec(_spec)
_sys_modules_before = set(sys.modules.keys())
sys.modules['curie_globalchem_adapter'] = _curie_module
_spec.loader.exec_module(_curie_module)
CurieGlobalChemAdapter = _curie_module.CurieGlobalChemAdapter

__version__ = "1.0.0"

__all__ = ['CurieGlobalChemAdapter', '__version__']
