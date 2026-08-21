"""
GlobalChem - Chemical Knowledge Graph compatibility stub.

GlobalChem is a chemical knowledge graph that stores organic molecules
and their SMILES/SMARTS representations. This stub provides minimal
implementations to allow imports to succeed.

Note: This is NOT a functional replacement for GlobalChem.
Install global-chem package for full functionality: pip install global-chem

Repository: https://github.com/Sulstice/global-chem
"""
from __future__ import annotations


import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

warnings.warn(
    "Using global_chem stub module. This is not a functional replacement. "
    "Install global-chem package for full functionality: pip install global-chem",
    RuntimeWarning,
    stacklevel=2
)

__version__ = "1.0.0-stub"


@dataclass
class Molecule:
    """Represents a chemical molecule."""
    name: str
    smiles: str
    smart: Optional[str] = None
    category: Optional[str] = None
    
    def __post_init__(self):
        if self.smart is None:
            self.smart = self.smiles  # Simplified


class GlobalChem:
    """
    Stub implementation of GlobalChem main class.
    
    GlobalChem is a chemical knowledge graph containing organic molecules
    organized by categories.
    """
    
    # Sample chemical data for stub
    _SAMPLE_DATA: Dict[str, Dict[str, str]] = {
        "methane": {"smiles": "C", "category": "alkanes"},
        "ethane": {"smiles": "CC", "category": "alkanes"},
        "propane": {"smiles": "CCC", "category": "alkanes"},
        "ethene": {"smiles": "C=C", "category": "alkenes"},
        "ethanol": {"smiles": "CCO", "category": "alcohols"},
        "benzene": {"smiles": "c1ccccc1", "category": "aromatics"},
        "toluene": {"smiles": "Cc1ccccc1", "category": "aromatics"},
        "water": {"smiles": "O", "category": "inorganic"},
        "carbon_dioxide": {"smiles": "O=C=O", "category": "inorganic"},
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize GlobalChem instance.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.molecules: Dict[str, Molecule] = {}
        self.network = None  # NetworkX graph would go here
        self._build_network()
    
    def _build_network(self) -> None:
        """Build the chemical knowledge network."""
        # In real implementation, builds NetworkX graph
        self.network = {"nodes": [], "edges": []}
        
        for name, data in self._SAMPLE_DATA.items():
            molecule = Molecule(
                name=name,
                smiles=data["smiles"],
                category=data["category"]
            )
            self.molecules[name] = molecule
            self.network["nodes"].append(name)
    
    def get_smiles(self, name: str) -> Optional[str]:
        """
        Get SMILES string for a molecule.
        
        Args:
            name: Molecule name
            
        Returns:
            SMILES string or None if not found
        """
        molecule = self.molecules.get(name.lower())
        return molecule.smiles if molecule else None
    
    def get_smart(self, name: str) -> Optional[str]:
        """
        Get SMARTS pattern for a molecule.
        
        Args:
            name: Molecule name
            
        Returns:
            SMARTS string or None if not found
        """
        molecule = self.molecules.get(name.lower())
        return molecule.smart if molecule else None
    
    def get_node(self, name: str) -> Optional[Dict]:
        """
        Get node data for a molecule.
        
        Args:
            name: Molecule name
            
        Returns:
            Node data dictionary or None
        """
        molecule = self.molecules.get(name.lower())
        if molecule:
            return {
                "name": molecule.name,
                "smiles": molecule.smiles,
                "smart": molecule.smart,
                "category": molecule.category
            }
        return None
    
    def get_all_nodes(self) -> List[str]:
        """
        Get all molecule names in the network.
        
        Returns:
            List of molecule names
        """
        return list(self.molecules.keys())
    
    def get_all_smiles(self) -> List[str]:
        """
        Get all SMILES strings.
        
        Returns:
            List of SMILES strings
        """
        return [m.smiles for m in self.molecules.values()]
    
    def search_by_category(self, category: str) -> List[Molecule]:
        """
        Search molecules by category.
        
        Args:
            category: Category name
            
        Returns:
            List of molecules in category
        """
        return [m for m in self.molecules.values() if m.category == category.lower()]
    
    def add_molecule(self, name: str, smiles: str, smart: Optional[str] = None, category: Optional[str] = None) -> None:
        """
        Add a new molecule to the network.
        
        Args:
            name: Molecule name
            smiles: SMILES representation
            smart: SMARTS pattern (optional)
            category: Category (optional)
        """
        self.molecules[name.lower()] = Molecule(
            name=name,
            smiles=smiles,
            smart=smart or smiles,
            category=category
        )
        self.network["nodes"].append(name.lower())


class CypherQuery:
    """Stub for Cypher query interface."""
    
    def __init__(self, global_chem: GlobalChem):
        self.gc = global_chem
    
    def query(self, cypher_string: str) -> List[Dict]:
        """
        Execute a Cypher-like query on the chemical graph.
        
        Args:
            cypher_string: Query string
            
        Returns:
            List of result dictionaries
        """
        # Stub implementation - just return all molecules
        return [
            {"name": m.name, "smiles": m.smiles, "category": m.category}
            for m in self.gc.molecules.values()
        ]


def get_common_solvents() -> Dict[str, str]:
    """
    Get dictionary of common solvents.
    
    Returns:
        Dictionary mapping solvent names to SMILES
    """
    return {
        "water": "O",
        "methanol": "CO",
        "ethanol": "CCO",
        "acetone": "CC(=O)C",
        "dmso": "CS(=O)C",
        "dmf": "CN(C)C=O",
        "thf": "C1CCOC1",
        "acetonitrile": "CC#N",
        "dcm": "ClCCl",
        "chloroform": "ClC(Cl)Cl",
        "hexane": "CCCCCC",
        "toluene": "Cc1ccccc1",
    }


def get_common_reagents() -> Dict[str, str]:
    """
    Get dictionary of common reagents.
    
    Returns:
        Dictionary mapping reagent names to SMILES
    """
    return {
        "hcl": "Cl",
        "sulfuric_acid": "O=S(=O)(O)O",
        "naoh": "[Na+].[OH-]",
        "nahco3": "[Na+].[Na+].[O-]C(=O)[O-]",
    }


__all__ = [
    'GlobalChem', 'Molecule', 'CypherQuery',
    'get_common_solvents', 'get_common_reagents',
    '__version__'
]
