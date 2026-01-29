"""
GlobalChem Integration Module for OpenEvolve Knowledge Engine

This module integrates GlobalChem's chemical knowledge graph capabilities
for chemistry-aware knowledge extraction and entity recognition.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Add GlobalChem to path
global_chem_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'global-chem', 'global_chem'
)
if global_chem_path not in sys.path:
    sys.path.insert(0, global_chem_path)


class GlobalChemKnowledgeAdapter:
    """
    Adapter for GlobalChem chemical knowledge graph.
    
    Provides:
    - Chemical entity recognition
    - SMILES/SMARTS parsing
    - Chemical knowledge retrieval
    - Molecular property queries
    """
    
    def __init__(self):
        """Initialize GlobalChem adapter."""
        self._global_chem_available = False
        self._gc = None
        self._chemical_cache = {}
        self._initialize_global_chem()
    
    def _initialize_global_chem(self):
        """Initialize GlobalChem with error handling."""
        try:
            from global_chem import GlobalChem
            self._gc = GlobalChem()
            self._gc.build_global_chem_network()
            self._global_chem_available = True
            logger.info("GlobalChem initialized successfully")
        except ImportError as e:
            logger.warning(f"GlobalChem not available: {e}")
            self._global_chem_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize GlobalChem: {e}")
            self._global_chem_available = False
    
    def is_available(self) -> bool:
        """Check if GlobalChem is available."""
        return self._global_chem_available
    
    def get_chemical_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get chemical information by name.
        
        Args:
            name: Chemical name
            
        Returns:
            Chemical information dictionary
        """
        if not self.is_available():
            return None
        
        try:
            # Search in GlobalChem network
            smiles = self._gc.get_node_smiles(name.lower())
            if smiles:
                return {
                    'name': name,
                    'smiles': smiles,
                    'source': 'global_chem',
                    'timestamp': datetime.now().isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving chemical {name}: {e}")
            return None
    
    def get_chemical_list(self, category: str) -> Dict[str, Any]:
        """
        Get a list of chemicals by category.
        
        Args:
            category: Category name (e.g., 'vitamins', 'amino_acids')
            
        Returns:
            Dictionary with chemical list
        """
        if not self.is_available():
            return {'status': 'error', 'message': 'GlobalChem not available'}
        
        try:
            smiles_dict = self._gc.get_node_smiles(category)
            if smiles_dict:
                return {
                    'status': 'success',
                    'category': category,
                    'chemicals': [
                        {'name': name, 'smiles': smiles}
                        for name, smiles in smiles_dict.items()
                    ],
                    'count': len(smiles_dict)
                }
            return {
                'status': 'error',
                'message': f'Category {category} not found'
            }
        except Exception as e:
            logger.error(f"Error retrieving category {category}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def recognize_chemical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Recognize chemical entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of recognized chemical entities
        """
        if not self.is_available():
            return []
        
        entities = []
        
        # Get all available nodes (chemical lists)
        try:
            all_nodes = list(self._gc.get_all_nodes())
            
            for node_name in all_nodes:
                if node_name.lower() in text.lower():
                    smiles_dict = self._gc.get_node_smiles(node_name)
                    if smiles_dict:
                        entities.append({
                            'entity': node_name,
                            'type': 'chemical_category',
                            'chemicals_count': len(smiles_dict),
                            'source': 'global_chem'
                        })
        except Exception as e:
            logger.error(f"Error recognizing entities: {e}")
        
        return entities
    
    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Validate SMILES notation.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Validation result
        """
        if not self.is_available():
            return {'valid': False, 'error': 'GlobalChem not available'}
        
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return {
                    'valid': True,
                    'smiles': smiles,
                    'canonical_smiles': Chem.MolToSmiles(mol),
                    'molecular_weight': self._calculate_molecular_weight(mol)
                }
            else:
                return {'valid': False, 'error': 'Invalid SMILES'}
        except ImportError:
            # Fallback without RDKit
            return {'valid': True, 'smiles': smiles, 'note': 'RDKit not available'}
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _calculate_molecular_weight(self, mol) -> float:
        """Calculate molecular weight using RDKit."""
        try:
            from rdkit.Chem import Descriptors
            return Descriptors.MolWt(mol)
        except:
            return 0.0
    
    def get_available_categories(self) -> List[str]:
        """Get list of available chemical categories."""
        if not self.is_available():
            return []
        
        try:
            return list(self._gc.get_all_nodes())
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def search_chemicals(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for chemicals by name or pattern.
        
        Args:
            query: Search query
            
        Returns:
            List of matching chemicals
        """
        if not self.is_available():
            return []
        
        results = []
        query_lower = query.lower()
        
        try:
            categories = self.get_available_categories()
            
            for category in categories:
                smiles_dict = self._gc.get_node_smiles(category)
                if smiles_dict:
                    for name, smiles in smiles_dict.items():
                        if query_lower in name.lower():
                            results.append({
                                'name': name,
                                'smiles': smiles,
                                'category': category
                            })
        except Exception as e:
            logger.error(f"Error searching chemicals: {e}")
        
        return results
    
    def get_chemical_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Get chemical properties from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Chemical properties
        """
        if not self.is_available():
            return {'status': 'error', 'message': 'GlobalChem not available'}
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'status': 'error', 'message': 'Invalid SMILES'}
            
            properties = {
                'status': 'success',
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'h_bond_donors': Lipinski.NumHDonors(mol),
                'h_bond_acceptors': Lipinski.NumHAcceptors(mol),
                'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol)
            }
            
            # Lipinski's Rule of Five
            violations = 0
            if properties['molecular_weight'] > 500:
                violations += 1
            if properties['logp'] > 5:
                violations += 1
            if properties['h_bond_donors'] > 5:
                violations += 1
            if properties['h_bond_acceptors'] > 10:
                violations += 1
            
            properties['lipinski_violations'] = violations
            properties['drug_like'] = violations <= 1
            
            return properties
            
        except ImportError:
            return {'status': 'error', 'message': 'RDKit not available'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def enrich_knowledge_graph_with_chemistry(
        self,
        graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich knowledge graph with chemical information.
        
        Args:
            graph_data: Knowledge graph data
            
        Returns:
            Enriched graph data
        """
        if not self.is_available():
            return graph_data
        
        enriched = graph_data.copy()
        chemical_nodes = []
        chemical_edges = []
        
        # Identify chemical entities in nodes
        for node in graph_data.get('nodes', []):
            node_name = node.get('id', '')
            chemical_info = self.get_chemical_by_name(node_name)
            
            if chemical_info:
                node['chemical_info'] = chemical_info
                node['type'] = 'chemical_entity'
                chemical_nodes.append(node)
                
                # Add chemical properties
                properties = self.get_chemical_properties(chemical_info.get('smiles', ''))
                if properties.get('status') == 'success':
                    node['chemical_properties'] = properties
        
        enriched['chemical_entities_count'] = len(chemical_nodes)
        enriched['chemical_nodes'] = chemical_nodes
        
        return enriched
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            'available': self.is_available(),
            'categories_count': len(self.get_available_categories()) if self.is_available() else 0,
            'timestamp': datetime.now().isoformat()
        }
