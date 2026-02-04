"""
Curie-GlobalChem Integration Adapter

This module provides an adapter that allows Curie to leverage GlobalChem's
chemical knowledge for conducting chemistry-related experiments.

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs and outputs
- ANTI-HALLUCINATION: Verify data integrity
- READ-ONLY STATE: Don't modify GlobalChem's data
- IDEMPOTENCY: Safe to run multiple times
- CONFIGURATION EXPLICITNESS: All parameters configurable
- UTC: All timestamps in UTC
"""

import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import GlobalChem
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "core-projects", "global-chem"))
    from global_chem import GlobalChem
except ImportError as e:
    print(f"Error importing GlobalChem: {e}")
    raise

class CurieGlobalChemAdapter:
    """
    Adapter class that bridges Curie and GlobalChem systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with configuration
        
        Args:
            config: Configuration dictionary with optional parameters
        """
        self.config = config or {}
        self.global_chem = GlobalChem()
        
        # Initialize logging
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Build the GlobalChem network
        self.global_chem.build_global_chem_network()
        
        self.logger.info("Curie-GlobalChem Adapter initialized successfully")
    
    def search_chemical_by_name(self, chemical_name: str) -> Optional[Dict[str, str]]:
        """
        Search for a chemical compound by name in GlobalChem
        
        Args:
            chemical_name: Name of the chemical to search for
            
        Returns:
            Dictionary with chemical info (name, smiles, etc.) or None if not found
        """
        self.logger.info(f"Searching for chemical: {chemical_name}")
        
        # Normalize the input
        normalized_name = chemical_name.strip().lower()
        
        # Get all names from GlobalChem
        all_names = self.global_chem.get_all_names()
        
        # Find exact match first
        for name in all_names:
            if name.lower() == normalized_name:
                # Find which node contains this name
                for node_key, node_class in self.global_chem._GlobalChem__NODES__.items():
                    if node_key != 'global_chem' and node_key != 'common_regex_patterns':
                        smiles_dict = node_class().get_smiles()
                        if name in smiles_dict:
                            result = {
                                'name': name,
                                'smiles': smiles_dict[name],
                                'node': node_key,
                                'timestamp': datetime.utcnow().isoformat() + 'Z'  # UTC timestamp
                            }
                            self.logger.info(f"Found chemical: {result}")
                            return result
        
        # If no exact match, try fuzzy matching
        for name in all_names:
            if normalized_name in name.lower() or name.lower() in normalized_name:
                # Find which node contains this name
                for node_key, node_class in self.global_chem._GlobalChem__NODES__.items():
                    if node_key != 'global_chem' and node_key != 'common_regex_patterns':
                        smiles_dict = node_class().get_smiles()
                        if name in smiles_dict:
                            result = {
                                'name': name,
                                'smiles': smiles_dict[name],
                                'node': node_key,
                                'match_type': 'partial',
                                'timestamp': datetime.utcnow().isoformat() + 'Z'  # UTC timestamp
                            }
                            self.logger.info(f"Found partial match for chemical: {result}")
                            return result
        
        self.logger.warning(f"No chemical found matching: {chemical_name}")
        return None
    
    def get_chemical_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Get chemical properties for a given SMILES string
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary with calculated chemical properties
        """
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        self.logger.info(f"Calculating properties for SMILES: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.logger.error(f"Invalid SMILES string: {smiles}")
            return {}
        
        # Calculate molecular descriptors
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_heavy_atoms': rdMolDescriptors.CalcNumHeavyAtoms(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'smiles': smiles,
            'formula': rdMolDescriptors.CalculateMolFormula(mol),
            'timestamp': datetime.utcnow().isoformat() + 'Z'  # UTC timestamp
        }
        
        self.logger.info(f"Calculated properties: {properties}")
        return properties
    
    def get_related_chemicals(self, chemical_name: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Find related chemicals based on shared categories or properties
        
        Args:
            chemical_name: Name of the reference chemical
            max_results: Maximum number of related chemicals to return
            
        Returns:
            List of related chemicals with their SMILES
        """
        self.logger.info(f"Finding related chemicals for: {chemical_name}")
        
        # First find the reference chemical
        ref_chem = self.search_chemical_by_name(chemical_name)
        if not ref_chem:
            self.logger.warning(f"Reference chemical not found: {chemical_name}")
            return []
        
        # Get all chemicals from the same node/category
        related_chemicals = []
        target_node = ref_chem['node']
        
        node_class = self.global_chem._GlobalChem__NODES__.get(target_node)
        if node_class:
            smiles_dict = node_class().get_smiles()
            
            for name, smiles in smiles_dict.items():
                if name.lower() != chemical_name.lower():  # Exclude the reference chemical itself
                    related_chemicals.append({
                        'name': name,
                        'smiles': smiles,
                        'node': target_node,
                        'relationship': 'same_category',
                        'timestamp': datetime.utcnow().isoformat() + 'Z'  # UTC timestamp
                    })
                    
                    if len(related_chemicals) >= max_results:
                        break
        
        self.logger.info(f"Found {len(related_chemicals)} related chemicals")
        return related_chemicals
    
    def run_chemistry_experiment(self, question: str) -> Dict[str, Any]:
        """
        Run a chemistry-focused experiment using Curie with GlobalChem knowledge
        
        Args:
            question: The chemistry question to investigate
            
        Returns:
            Dictionary with experiment results
        """
        self.logger.info(f"Running chemistry experiment for question: {question}")
        
        # Parse the question to identify chemical entities
        import re
        
        # Simple pattern to extract potential chemical names
        # In a real implementation, this would be more sophisticated
        potential_chemicals = re.findall(r'\b[A-Za-z]+\b', question)
        
        results = {
            'question': question,
            'identified_chemicals': [],
            'related_chemicals': [],
            'properties_calculated': [],
            'timestamp': datetime.utcnow().isoformat() + 'Z',  # UTC timestamp
            'adapter_version': '1.0.0'
        }
        
        # Look up each potential chemical in GlobalChem
        for chem_name in potential_chemicals:
            if len(chem_name) > 2:  # Skip short words that are unlikely to be chemical names
                chem_info = self.search_chemical_by_name(chem_name)
                if chem_info:
                    results['identified_chemicals'].append(chem_info)
                    
                    # Calculate properties if SMILES is available
                    if 'smiles' in chem_info:
                        props = self.get_chemical_properties(chem_info['smiles'])
                        if props:
                            results['properties_calculated'].append(props)
                    
                    # Find related chemicals
                    related = self.get_related_chemicals(chem_name, max_results=5)
                    results['related_chemicals'].extend(related)
        
        self.logger.info(f"Chemistry experiment completed: {results}")
        return results


def create_curie_interface(adapter: CurieGlobalChemAdapter):
    """
    Creates an interface function that Curie can use to access GlobalChem data
    
    Args:
        adapter: Instance of CurieGlobalChemAdapter
        
    Returns:
        Function that Curie can call to access chemistry knowledge
    """
    def curie_chemistry_query(query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Interface function for Curie to query chemistry data
        
        Args:
            query_type: Type of query ('search', 'properties', 'related', 'experiment')
            **kwargs: Query-specific parameters
            
        Returns:
            Query results
        """
        if query_type == 'search':
            chemical_name = kwargs.get('chemical_name')
            if not chemical_name:
                raise ValueError("chemical_name is required for search queries")
            return adapter.search_chemical_by_name(chemical_name)
        
        elif query_type == 'properties':
            smiles = kwargs.get('smiles')
            if not smiles:
                raise ValueError("smiles is required for properties queries")
            return adapter.get_chemical_properties(smiles)
        
        elif query_type == 'related':
            chemical_name = kwargs.get('chemical_name')
            if not chemical_name:
                raise ValueError("chemical_name is required for related queries")
            max_results = kwargs.get('max_results', 10)
            return adapter.get_related_chemicals(chemical_name, max_results)
        
        elif query_type == 'experiment':
            question = kwargs.get('question')
            if not question:
                raise ValueError("question is required for experiment queries")
            return adapter.run_chemistry_experiment(question)
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    return curie_chemistry_query


# Example usage
if __name__ == "__main__":
    # Initialize the adapter
    adapter = CurieGlobalChemAdapter()
    
    # Create the interface for Curie
    chemistry_interface = create_curie_interface(adapter)
    
    # Example: Search for a known chemical
    result = chemistry_interface('search', chemical_name='aspirin')
    print("Search result:", result)
    
    # Example: Get related chemicals
    if result:
        related = chemistry_interface('related', chemical_name='aspirin', max_results=5)
        print("Related chemicals:", related)
    
    # Example: Calculate properties
    if result and 'smiles' in result:
        props = chemistry_interface('properties', smiles=result['smiles'])
        print("Properties:", props)
    
    # Example: Run a chemistry experiment
    experiment_result = chemistry_interface('experiment', question='What are the properties of aspirin and related compounds?')
    print("Experiment result:", experiment_result)