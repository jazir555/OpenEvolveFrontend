"""
GlobalChem Integration for OpenEvolve

This package provides integration with GlobalChem, a community-curated
chemical knowledge graph with SMILES/SMARTS support.

Components:
- adapter: GlobalChemAdapter for chemical knowledge access
- bridge: GlobalChemBridge for knowledge base integration
- config: Configuration settings

Key Features:
- Chemical list queries (organic compounds, biomolecules, etc.)
- SMILES/SMARTS parsing and validation
- Chemical property prediction
- Entity recognition and relationship extraction
- Integration with OneKE for enhanced knowledge extraction
"""

from .adapter import (
    GlobalChemAdapter,
    ChemicalKnowledgeError,
    SMILESParsingError,
    SMARTSParsingError,
)

from .bridge import (
    GlobalChemBridge,
    ChemicalEntity,
    ChemicalRelationship,
    ChemicalEntityType,
)

__version__ = '0.1.0'
__author__ = 'Agent 7 (GlobalChem Integration Specialist)'

# Export main classes
__all__ = [
    # Adapter
    'GlobalChemAdapter',
    'ChemicalKnowledgeError',
    'SMILESParsingError',
    'SMARTSParsingError',

    # Bridge
    'GlobalChemBridge',
    'ChemicalEntity',
    'ChemicalRelationship',
    'ChemicalEntityType',
]

# Module metadata
INTEGRATION_INFO = {
    'name': 'GlobalChem Integration',
    'version': __version__,
    'description': 'Community-curated chemical knowledge graph with SMILES/SMARTS support',
    'gaps_filled': ['GAP-13 (Chemical/Biological Knowledge)', 'GAP-2 (Domain Knowledge)'],
    'priority': 'P4 (OPTIONAL)',
    'effort': '1 week',
    'repository': 'https://github.com/Sulstice/global-chem',
    'features': [
        'Chemical list queries',
        'SMILES/SMARTS parsing',
        'Chemical property prediction',
        'Entity recognition',
        'Relationship extraction',
        'OneKE integration',
    ],
    'chemical_lists': [
        # Organic Chemistry
        'organic_and_inorganic_bronsted_acids',
        'common_organic_solvents',
        'common_monomer_repeating_units',

        # Biomolecules
        'amino_acids',
        'vitamins',
        'phytocannabinoids',
        'cannabis_compounds',

        # Medicinal Chemistry
        'drugs_from_snake_venom',
        'electrophilic_warheads',
        'privileged_scaffolds',
        'kinase_inhibitors',

        # Food Chemistry
        'salt',
        'fda_color_additives',
        'mango_compounds',

        # Environmental Chemistry
        'alternative_jet_fuels',
        'chemicals_from_biomass',
        'emerging_perfluoroalkyls',
        'interstellar_space',

        # Controlled Substances
        'narcotics_schedules',
        'pihkal',

        # Warfare Agents
        'nerve_agents',
    ],
    'status': 'Complete',
    'created': '2026-01-02',
}

# Configuration template
DEFAULT_CONFIG = {
    'project': {
        'name': 'global-chem',
        'version': '0.1.0',
        'enabled': True,
    },
    'features': {
        'chemical_lists': True,
        'smiles_parsing': True,
        'smarts_parsing': True,
        'property_prediction': True,
        'oneke_integration': True,
    },
    'integration': {
        'auto_start': True,
        'oneke_integration': True,
        'cache_enabled': True,
        'cache_ttl': 3600,
    },
    'entity_recognition': {
        'enabled': True,
        'confidence_threshold': 0.7,
    },
    'performance': {
        'max_workers': 4,
        'timeout': 30,
        'batch_size': 100,
    },
}


def get_integration_info():
    """Get integration information."""
    return INTEGRATION_INFO


def get_default_config():
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()
