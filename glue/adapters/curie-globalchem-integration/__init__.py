"""
Curie-GlobalChem Integration Adapter

This package provides an adapter that allows Curie to leverage GlobalChem's
chemical knowledge for conducting chemistry-related experiments.
"""

from .src.curie_globalchem_adapter import CurieGlobalChemAdapter

__version__ = "1.0.0"

__all__ = ['CurieGlobalChemAdapter', '__version__']
