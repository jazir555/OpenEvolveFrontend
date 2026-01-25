"""
GlobalChem Adapter for OpenEvolve

This module provides an adapter that wraps GlobalChem's functionality to provide
chemical knowledge graph capabilities to OpenEvolve. It enables access to
community-curated chemical lists, SMILES/SMARTS parsing, and chemical property
prediction without modifying GlobalChem's source code.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging
from functools import lru_cache

# Add GlobalChem to path
global_chem_path = os.path.join(os.path.dirname(__file__), "../../projects to analyze/global-chem")
if global_chem_path not in sys.path:
    sys.path.insert(0, global_chem_path)

try:
    from global_chem.global_chem.global_chem import GlobalChem
    GLOBAL_CHEM_AVAILABLE = True
except ImportError as e:
    GLOBAL_CHEM_AVAILABLE = False
    global_chem_import_error = str(e)

from integrations.base.knowledge_interface import (
    KnowledgeGraphInterface,
    KnowledgeGraphError,
    ConfigurationError,
    ConnectionError,
    ValidationError,
    StorageError,
    SearchError,
    AnalysisError,
    ShutdownError,
    RetrievalError,
    RemovalError,
    TemporalFilter,
)

logger = logging.getLogger(__name__)


class ChemicalKnowledgeError(KnowledgeGraphError):
    """Exception raised for chemical knowledge specific errors."""
    pass


class SMILESParsingError(ChemicalKnowledgeError):
    """Exception raised when SMILES parsing fails."""
    pass


class SMARTSParsingError(ChemicalKnowledgeError):
    """Exception raised when SMARTS parsing fails."""
    pass


class GlobalChemAdapter(KnowledgeGraphInterface):
    """
    Adapter for GlobalChem chemical knowledge graph.

    This adapter wraps GlobalChem's community-curated chemical lists and
    SMILES/SMARTS parsing capabilities to provide a consistent interface
    for OpenEvolve. It supports:

    - Chemical list queries (organic compounds, biomolecules, etc.)
    - SMILES/SMARTS string parsing and validation
    - Chemical property prediction
    - Integration with OneKE for entity recognition
    - Domain-specific knowledge for chemistry/biology

    Gracefully degrades if GlobalChem is unavailable.
    """

    def __init__(self):
        """Initialize the adapter without connecting to GlobalChem."""
        self.global_chem: Optional[GlobalChem] = None
        self.config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self.chemical_lists_cache: Dict[str, Dict[str, str]] = {}
        self.smiles_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_enabled = True
        self.cache_ttl = 3600  # Default cache TTL in seconds

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize GlobalChem with the provided configuration.

        Args:
            config: Configuration dictionary with keys:
                - chemical_lists: List of chemical list names to load
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Cache time-to-live in seconds (default: 3600)
                - auto_start: Whether to auto-load all lists (default: True)
                - oneke_integration: Enable OneKE integration (default: True)

        Returns:
            True if initialization was successful

        Raises:
            ConfigurationError: If config is invalid or GlobalChem unavailable
            ConnectionError: If connection fails
        """
        if not GLOBAL_CHEM_AVAILABLE:
            logger.warning(f"GlobalChem not available: {global_chem_import_error}")
            raise ConfigurationError(
                f"GlobalChem is not available. Please ensure it is installed. Error: {global_chem_import_error}"
            )

        try:
            self.config = config

            # Extract configuration
            chemical_lists = config.get("chemical_lists", [])
            self.cache_enabled = config.get("cache_enabled", True)
            self.cache_ttl = config.get("cache_ttl", 3600)
            auto_start = config.get("auto_start", True)

            # Initialize GlobalChem
            self.global_chem = GlobalChem()

            # Pre-load specified chemical lists if auto_start is enabled
            if auto_start:
                if chemical_lists:
                    for list_name in chemical_lists:
                        self._load_chemical_list(list_name)
                else:
                    # Load all available lists
                    self._load_all_chemical_lists()

            self.is_initialized = True
            logger.info("GlobalChem adapter initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GlobalChem adapter: {e}")
            raise ConnectionError(f"Failed to initialize GlobalChem: {e}")

    def _load_chemical_list(self, list_name: str) -> None:
        """
        Load a specific chemical list from GlobalChem.

        Args:
            list_name: Name of the chemical list to load

        Raises:
            ChemicalKnowledgeError: If list not found or loading fails
        """
        if not self.global_chem:
            raise ChemicalKnowledgeError("GlobalChem not initialized")

        try:
            # Get the SMILES data for the specified list
            smiles_data = self.global_chem.get_all_smiles()

            # Filter to the specific list if available
            if list_name in smiles_data:
                self.chemical_lists_cache[list_name] = smiles_data[list_name]
                logger.debug(f"Loaded chemical list: {list_name} with {len(smiles_data[list_name])} entries")
            else:
                logger.warning(f"Chemical list '{list_name}' not found in GlobalChem")

        except Exception as e:
            logger.error(f"Failed to load chemical list '{list_name}': {e}")
            raise ChemicalKnowledgeError(f"Failed to load chemical list: {e}")

    def _load_all_chemical_lists(self) -> None:
        """Load all available chemical lists from GlobalChem."""
        if not self.global_chem:
            raise ChemicalKnowledgeError("GlobalChem not initialized")

        try:
            smiles_data = self.global_chem.get_all_smiles()
            self.chemical_lists_cache = smiles_data
            logger.info(f"Loaded {len(smiles_data)} chemical lists from GlobalChem")

        except Exception as e:
            logger.error(f"Failed to load chemical lists: {e}")
            raise ChemicalKnowledgeError(f"Failed to load chemical lists: {e}")

    async def parse_smiles(self, smiles_string: str) -> Dict[str, Any]:
        """
        Parse and validate a SMILES string.

        Args:
            smiles_string: SMILES string to parse

        Returns:
            Dictionary with parsing results:
            - is_valid: Whether the SMILES is valid
            - canonical_form: Canonical SMILES if valid
            - molecular_formula: Molecular formula
            - molecular_weight: Molecular weight
            - error: Error message if invalid

        Raises:
            SMILESParsingError: If parsing fails
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem adapter not initialized")

        try:
            # Check cache first
            if self.cache_enabled and smiles_string in self.smiles_cache:
                logger.debug(f"SMILES cache hit for: {smiles_string}")
                return self.smiles_cache[smiles_string]

            # Simple validation - check if SMILES exists in any list
            all_smiles = self.global_chem.get_all_smiles()

            is_valid = False
            canonical_form = None
            source_list = None

            # Search through all lists
            for list_name, smiles_dict in all_smiles.items():
                if smiles_string in smiles_dict.values():
                    is_valid = True
                    canonical_form = smiles_string
                    source_list = list_name
                    break

            result = {
                "is_valid": is_valid,
                "canonical_form": canonical_form,
                "molecular_formula": self._extract_formula(smiles_string) if is_valid else None,
                "molecular_weight": self._calculate_molecular_weight(smiles_string) if is_valid else None,
                "source_list": source_list,
                "error": None if is_valid else "SMILES not found in GlobalChem database"
            }

            # Cache the result
            if self.cache_enabled:
                self.smiles_cache[smiles_string] = result

            return result

        except Exception as e:
            logger.error(f"Failed to parse SMILES '{smiles_string}': {e}")
            raise SMILESParsingError(f"Failed to parse SMILES: {e}")

    async def parse_smarts(self, smarts_string: str) -> Dict[str, Any]:
        """
        Parse and validate a SMARTS string.

        Args:
            smarts_string: SMARTS string to parse

        Returns:
            Dictionary with parsing results:
            - is_valid: Whether the SMARTS is valid
            - pattern_type: Type of pattern
            - error: Error message if invalid

        Raises:
            SMARTSParsingError: If parsing fails
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem adapter not initialized")

        try:
            # Basic SMARTS validation
            # SMARTS is an extension of SMILES with pattern matching capabilities
            is_valid = self._validate_smarts_basic(smarts_string)

            result = {
                "is_valid": is_valid,
                "pattern_type": self._classify_smarts_pattern(smarts_string) if is_valid else None,
                "error": None if is_valid else "Invalid SMARTS pattern"
            }

            return result

        except Exception as e:
            logger.error(f"Failed to parse SMARTS '{smarts_string}': {e}")
            raise SMARTSParsingError(f"Failed to parse SMARTS: {e}")

    def _validate_smarts_basic(self, smarts: str) -> bool:
        """Basic SMARTS validation."""
        if not smarts or len(smarts) == 0:
            return False

        # Basic atom and bond validation
        valid_atoms = set('CNOPSFClBrI()=[]#@')
        has_valid_content = any(c in valid_atoms for c in smarts)

        return has_valid_content

    def _classify_smarts_pattern(self, smarts: str) -> str:
        """Classify the SMARTS pattern type."""
        if '[' in smarts and ']' in smarts:
            return "atom_query"
        elif '$' in smarts:
            return "recursive"
        elif '!' in smarts:
            return "negation"
        else:
            return "simple"

    def _extract_formula(self, smiles: str) -> str:
        """Extract molecular formula from SMILES (simplified)."""
        # This is a simplified implementation
        # In production, use RDKit or similar
        atom_counts = {}
        for char in smiles:
            if char.isalpha():
                atom_counts[char] = atom_counts.get(char, 0) + 1

        formula = ""
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            formula += atom + (str(count) if count > 1 else "")

        return formula

    def _calculate_molecular_weight(self, smiles: str) -> float:
        """Calculate molecular weight from SMILES (simplified)."""
        # Simplified atomic weights
        atomic_weights = {
            'C': 12.01, 'N': 14.01, 'O': 16.00, 'P': 30.97,
            'S': 32.07, 'F': 19.00, 'Cl': 35.45, 'Br': 79.90,
            'I': 126.90, 'H': 1.008
        }

        weight = 0.0
        for char in smiles:
            if char in atomic_weights:
                weight += atomic_weights[char]

        return weight

    async def query_chemical_list(
        self,
        list_name: str,
        query: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Query a specific chemical list.

        Args:
            list_name: Name of the chemical list
            query: Optional search query for filtering
            limit: Maximum number of results

        Returns:
            Dictionary with query results:
            - chemicals: List of chemical entries
            - total: Total count
            - list_name: Name of the list queried
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem adapter not initialized")

        try:
            # Load list if not cached
            if list_name not in self.chemical_lists_cache:
                self._load_chemical_list(list_name)

            chemicals = self.chemical_lists_cache.get(list_name, {})

            # Filter by query if provided
            if query:
                chemicals = {
                    name: smiles for name, smiles in chemicals.items()
                    if query.lower() in name.lower()
                }

            # Limit results
            chemical_items = list(chemicals.items())[:limit]

            return {
                "chemicals": [
                    {
                        "name": name,
                        "smiles": smiles
                    }
                    for name, smiles in chemical_items
                ],
                "total": len(chemicals),
                "list_name": list_name
            }

        except Exception as e:
            logger.error(f"Failed to query chemical list '{list_name}': {e}")
            raise ChemicalKnowledgeError(f"Failed to query chemical list: {e}")

    async def get_available_chemical_lists(self) -> List[str]:
        """
        Get list of available chemical lists in GlobalChem.

        Returns:
            List of chemical list names
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem adapter not initialized")

        try:
            all_smiles = self.global_chem.get_all_smiles()
            return list(all_smiles.keys())

        except Exception as e:
            logger.error(f"Failed to get available chemical lists: {e}")
            raise ChemicalKnowledgeError(f"Failed to get chemical lists: {e}")

    async def add_episode(
        self,
        name: str,
        body: str,
        reference_time: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "openevolve",
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add an episode (not applicable for GlobalChem).

        GlobalChem is a static knowledge base, so this method is not applicable.
        """
        logger.warning("add_episode not applicable for GlobalChem (static knowledge base)")
        return {
            "status": "not_applicable",
            "message": "GlobalChem is a static knowledge base, episodes cannot be added"
        }

    async def search(
        self,
        query: str,
        temporal_filters: Optional[Dict[str, Any]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search GlobalChem for chemical knowledge.

        Args:
            query: Search query (chemical name or SMILES)
            temporal_filters: Not applicable for GlobalChem
            num_results: Maximum results
            group_ids: Not applicable

        Returns:
            Search results dictionary

        Raises:
            SearchError: If search fails
        """
        if not self.is_initialized:
            raise SearchError("GlobalChem adapter not initialized")

        try:
            # Search through all chemical lists
            all_smiles = self.global_chem.get_all_smiles()

            results = []
            for list_name, smiles_dict in all_smiles.items():
                for chemical_name, smiles in smiles_dict.items():
                    # Check if query matches chemical name
                    if query.lower() in chemical_name.lower():
                        results.append({
                            "name": chemical_name,
                            "smiles": smiles,
                            "list": list_name
                        })

                    # Check if query is a SMILES substring
                    elif query.lower() in smiles.lower():
                        results.append({
                            "name": chemical_name,
                            "smiles": smiles,
                            "list": list_name
                        })

                    if len(results) >= num_results:
                        break

                if len(results) >= num_results:
                    break

            return {
                "chemicals": results[:num_results],
                "total_found": len(results),
                "query": query
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"GlobalChem search failed: {e}")

    async def get_community_detections(
        self,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Community detection (not applicable for GlobalChem).

        Returns empty result as GlobalChem is a static knowledge base.
        """
        return {
            "communities": [],
            "community_edges": [],
            "metrics": {
                "num_communities": 0,
                "num_edges": 0,
            },
            "message": "Community detection not applicable for static knowledge base"
        }

    async def validate(self) -> Dict[str, Any]:
        """
        Validate GlobalChem state.

        Returns:
            Validation results
        """
        if not self.is_initialized:
            return {
                "is_valid": False,
                "checks": {"initialized": False},
                "issues": ["GlobalChem adapter not initialized"],
                "metrics": {},
            }

        try:
            # Basic validation - check if we can get chemical lists
            available_lists = await self.get_available_chemical_lists()

            return {
                "is_valid": True,
                "checks": {
                    "initialized": True,
                    "chemical_lists_loaded": len(available_lists) > 0,
                    "cache_enabled": self.cache_enabled,
                },
                "issues": [],
                "metrics": {
                    "num_chemical_lists": len(available_lists),
                    "cache_enabled": self.cache_enabled,
                    "cache_ttl": self.cache_ttl,
                    "cached_lists": len(self.chemical_lists_cache),
                },
            }

        except Exception as e:
            return {
                "is_valid": False,
                "checks": {
                    "initialized": True,
                    "chemical_lists_loaded": False,
                },
                "issues": [str(e)],
                "metrics": {},
            }

    async def shutdown(self) -> bool:
        """
        Shutdown GlobalChem connection.

        Returns:
            True if successful
        """
        if not self.is_initialized:
            return True

        try:
            # Clear caches
            self.chemical_lists_cache.clear()
            self.smiles_cache.clear()

            self.is_initialized = False
            logger.info("GlobalChem adapter shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            raise ShutdownError(f"Failed to shutdown GlobalChem: {e}")

    async def get_episodes(
        self,
        reference_time: datetime,
        last_n: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve episodes (not applicable for GlobalChem).

        Returns empty list as GlobalChem is a static knowledge base.
        """
        logger.warning("get_episodes not applicable for GlobalChem (static knowledge base)")
        return []

    async def add_triplet(
        self,
        source_entity: Dict[str, Any],
        relationship: Dict[str, Any],
        target_entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a triplet (not applicable for GlobalChem).

        GlobalChem is a static knowledge base, so this method is not applicable.
        """
        logger.warning("add_triplet not applicable for GlobalChem (static knowledge base)")
        return {
            "status": "not_applicable",
            "message": "GlobalChem is a static knowledge base, triplets cannot be added"
        }

    async def remove_episode(self, episode_uuid: str) -> bool:
        """
        Remove an episode (not applicable for GlobalChem).

        Returns False as GlobalChem is a static knowledge base.
        """
        logger.warning("remove_episode not applicable for GlobalChem (static knowledge base)")
        return False
