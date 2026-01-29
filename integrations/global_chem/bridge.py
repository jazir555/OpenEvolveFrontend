"""
GlobalChem Bridge for OpenEvolve

This module provides a bridge that connects GlobalChem's chemical knowledge
with OpenEvolve's knowledge base and OneKE integration. It enables chemical
entity recognition, property prediction, and knowledge extraction.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Set, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

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

from integrations.global_chem.adapter import GlobalChemAdapter, ChemicalKnowledgeError

logger = logging.getLogger(__name__)


class ChemicalEntityType(Enum):
    """Types of chemical entities that can be recognized."""
    ORGANIC_COMPOUND = "organic_compound"
    INORGANIC_COMPOUND = "inorganic_compound"
    BIOMOLECULE = "biomolecule"
    DRUG = "drug"
    POLYMER = "polymer"
    SOLVENT = "solvent"
    NARCOTIC = "narcotic"
    FOOD_ADDITIVE = "food_additive"
    ENVIRONMENTAL_CHEMICAL = "environmental_chemical"
    WARFARE_AGENT = "warfare_agent"
    UNKNOWN = "unknown"


@dataclass
class ChemicalEntity:
    """Represents a recognized chemical entity."""
    name: str
    smiles: Optional[str]
    entity_type: ChemicalEntityType
    source_list: str
    properties: Dict[str, Any]
    confidence: float


@dataclass
class ChemicalRelationship:
    """Represents a relationship between chemical entities."""
    source_entity: str
    relationship_type: str
    target_entity: str
    confidence: float
    metadata: Dict[str, Any]


class GlobalChemBridge:
    """
    Bridge for integrating GlobalChem with OpenEvolve's knowledge base.

    This bridge provides:
    - Chemical entity recognition from text
    - Integration with OneKE for enhanced entity extraction
    - Chemical property prediction
    - Knowledge graph generation for chemical concepts
    - Relationship extraction between chemical entities
    """

    def __init__(self, adapter: GlobalChemAdapter):
        """
        Initialize the bridge with a GlobalChem adapter.

        Args:
            adapter: GlobalChemAdapter instance
        """
        self.adapter = adapter
        self.global_chem: Optional[GlobalChem] = None
        self.entity_cache: Dict[str, ChemicalEntity] = {}
        self.relationship_cache: List[ChemicalRelationship] = []
        self.oneke_integration_enabled = False
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the bridge with configuration.

        Args:
            config: Configuration dictionary with keys:
                - oneke_integration: Enable OneKE integration (default: False)
                - cache_entities: Cache recognized entities (default: True)
                - entity_recognition_threshold: Minimum confidence for entities (default: 0.7)

        Returns:
            True if initialization was successful
        """
        try:
            # Initialize the adapter if not already initialized
            if not self.adapter.is_initialized:
                await self.adapter.initialize(config)

            # Get GlobalChem instance from adapter
            if hasattr(self.adapter, 'global_chem') and self.adapter.global_chem:
                self.global_chem = self.adapter.global_chem

            # Extract configuration
            self.oneke_integration_enabled = config.get("oneke_integration", False)
            cache_entities = config.get("cache_entities", True)

            if not cache_entities:
                self.entity_cache.clear()

            self.is_initialized = True
            logger.info("GlobalChem bridge initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GlobalChem bridge: {e}")
            raise ChemicalKnowledgeError(f"Failed to initialize bridge: {e}")

    async def recognize_chemical_entities(
        self,
        text: str,
        threshold: float = 0.7
    ) -> List[ChemicalEntity]:
        """
        Recognize chemical entities in text.

        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold

        Returns:
            List of recognized ChemicalEntity objects
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem bridge not initialized")

        try:
            recognized_entities = []

            # Get all available chemical lists
            all_smiles = self.global_chem.get_all_smiles()

            # Search for chemical names in text
            for list_name, smiles_dict in all_smiles.items():
                entity_type = self._classify_list_type(list_name)

                for chemical_name, smiles in smiles_dict.items():
                    # Check if chemical name appears in text
                    if chemical_name.lower() in text.lower():
                        # Calculate confidence based on match quality
                        confidence = self._calculate_confidence(chemical_name, text)

                        if confidence >= threshold:
                            entity = ChemicalEntity(
                                name=chemical_name,
                                smiles=smiles,
                                entity_type=entity_type,
                                source_list=list_name,
                                properties=await self._predict_properties(smiles),
                                confidence=confidence
                            )

                            recognized_entities.append(entity)

                            # Cache the entity
                            self.entity_cache[chemical_name] = entity

            # Sort by confidence
            recognized_entities.sort(key=lambda e: e.confidence, reverse=True)

            logger.info(f"Recognized {len(recognized_entities)} chemical entities in text")
            return recognized_entities

        except Exception as e:
            logger.error(f"Failed to recognize chemical entities: {e}")
            raise ChemicalKnowledgeError(f"Entity recognition failed: {e}")

    def _classify_list_type(self, list_name: str) -> ChemicalEntityType:
        """
        Classify the chemical list type.

        Args:
            list_name: Name of the chemical list

        Returns:
            ChemicalEntityType enum value
        """
        list_name_lower = list_name.lower()

        if any(keyword in list_name_lower for keyword in ['organic', 'cannabis', 'cannabinoid']):
            return ChemicalEntityType.ORGANIC_COMPOUND
        elif any(keyword in list_name_lower for keyword in ['inorganic', 'metal', 'mineral']):
            return ChemicalEntityType.INORGANIC_COMPOUND
        elif any(keyword in list_name_lower for keyword in ['amino', 'protein', 'peptide', 'vitamin']):
            return ChemicalEntityType.BIOMOLECULE
        elif any(keyword in list_name_lower for keyword in ['drug', 'medicine', 'pharmaceutical', 'kinase']):
            return ChemicalEntityType.DRUG
        elif 'polymer' in list_name_lower or 'monomer' in list_name_lower:
            return ChemicalEntityType.POLYMER
        elif 'solvent' in list_name_lower:
            return ChemicalEntityType.SOLVENT
        elif any(keyword in list_name_lower for keyword in ['narcotic', 'schedule', 'pihkal']):
            return ChemicalEntityType.NARCOTIC
        elif 'food' in list_name_lower or 'color' in list_name_lower:
            return ChemicalEntityType.FOOD_ADDITIVE
        elif 'environment' in list_name_lower or 'interstellar' in list_name_lower:
            return ChemicalEntityType.ENVIRONMENTAL_CHEMICAL
        elif 'warfare' in list_name_lower or 'nerve' in list_name_lower:
            return ChemicalEntityType.WARFARE_AGENT
        else:
            return ChemicalEntityType.UNKNOWN

    def _calculate_confidence(self, chemical_name: str, text: str) -> float:
        """
        Calculate confidence score for entity recognition.

        Args:
            chemical_name: Recognized chemical name
            text: Input text

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from exact match
        if chemical_name.lower() in text.lower():
            base_confidence = 0.8

            # Boost confidence for exact word boundaries
            import re
            pattern = r'\b' + re.escape(chemical_name.lower()) + r'\b'
            if re.search(pattern, text.lower()):
                base_confidence = 0.95

            # Reduce confidence for very short names
            if len(chemical_name) < 4:
                base_confidence *= 0.7

            return min(base_confidence, 1.0)

        return 0.0

    async def _predict_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Predict chemical properties from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of predicted properties
        """
        try:
            # Use adapter to parse SMILES
            parse_result = await self.adapter.parse_smiles(smiles)

            if parse_result['is_valid']:
                return {
                    "molecular_formula": parse_result.get('molecular_formula'),
                    "molecular_weight": parse_result.get('molecular_weight'),
                    "canonical_smiles": parse_result.get('canonical_form'),
                }
            else:
                return {}

        except Exception as e:
            logger.warning(f"Failed to predict properties for SMILES '{smiles}': {e}")
            return {}

    async def extract_chemical_relationships(
        self,
        entities: List[ChemicalEntity],
        text: str
    ) -> List[ChemicalRelationship]:
        """
        Extract relationships between chemical entities.

        Args:
            entities: List of recognized chemical entities
            text: Input text

        Returns:
            List of ChemicalRelationship objects
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem bridge not initialized")

        try:
            relationships = []

            # Extract relationships based on text patterns
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # Look for co-occurrence patterns
                    if self._are_related(entity1, entity2, text):
                        relationship = self._infer_relationship(entity1, entity2, text)
                        if relationship:
                            relationships.append(relationship)

            self.relationship_cache.extend(relationships)
            logger.info(f"Extracted {len(relationships)} chemical relationships")
            return relationships

        except Exception as e:
            logger.error(f"Failed to extract chemical relationships: {e}")
            raise ChemicalKnowledgeError(f"Relationship extraction failed: {e}")

    def _are_related(
        self,
        entity1: ChemicalEntity,
        entity2: ChemicalEntity,
        text: str
    ) -> bool:
        """
        Determine if two entities are related in the text.

        Args:
            entity1: First chemical entity
            entity2: Second chemical entity
            text: Input text

        Returns:
            True if entities appear to be related
        """
        # Simple heuristic: entities are related if they appear in the same sentence
        # or within a certain word distance

        # Find positions
        pos1 = text.lower().find(entity1.name.lower())
        pos2 = text.lower().find(entity2.name.lower())

        if pos1 == -1 or pos2 == -1:
            return False

        # Check if within 100 characters
        distance = abs(pos1 - pos2)
        return distance < 100

    def _infer_relationship(
        self,
        entity1: ChemicalEntity,
        entity2: ChemicalEntity,
        text: str
    ) -> Optional[ChemicalRelationship]:
        """
        Infer relationship type between two entities.

        Args:
            entity1: First chemical entity
            entity2: Second chemical entity
            text: Input text

        Returns:
            ChemicalRelationship if relationship inferred, None otherwise
        """
        # Simple relationship inference based on keywords
        text_lower = text.lower()

        # Look for relationship keywords
        if any(word in text_lower for word in ['reacts with', 'react', 'reaction']):
            relationship_type = "reacts_with"
        elif any(word in text_lower for word in ['derivative', 'derived from']):
            relationship_type = "derivative_of"
        elif any(word in text_lower for word in ['similar to', 'analog']):
            relationship_type = "analog_of"
        elif any(word in text_lower for word in ['inhibits', 'inhibition']):
            relationship_type = "inhibits"
        elif any(word in text_lower for word in ['binds to', 'binding']):
            relationship_type = "binds_to"
        else:
            relationship_type = "co_occurs_with"

        return ChemicalRelationship(
            source_entity=entity1.name,
            relationship_type=relationship_type,
            target_entity=entity2.name,
            confidence=0.7,  # Default confidence
            metadata={
                "source_list_1": entity1.source_list,
                "source_list_2": entity2.source_list,
            }
        )

    async def generate_knowledge_graph(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Generate a knowledge graph from chemical entities in text.

        Args:
            text: Input text

        Returns:
            Dictionary with nodes (entities) and edges (relationships)
        """
        try:
            # Recognize entities
            entities = await self.recognize_chemical_entities(text)

            # Extract relationships
            relationships = await self.extract_chemical_relationships(entities, text)

            # Format as knowledge graph
            nodes = [
                {
                    "id": entity.name,
                    "type": entity.entity_type.value,
                    "properties": {
                        "smiles": entity.smiles,
                        "source_list": entity.source_list,
                        **entity.properties
                    },
                    "confidence": entity.confidence
                }
                for entity in entities
            ]

            edges = [
                {
                    "source": rel.source_entity,
                    "target": rel.target_entity,
                    "relationship": rel.relationship_type,
                    "confidence": rel.confidence,
                    "metadata": rel.metadata
                }
                for rel in relationships
            ]

            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "num_entities": len(entities),
                    "num_relationships": len(relationships),
                    "source": "global_chem"
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate knowledge graph: {e}")
            raise ChemicalKnowledgeError(f"Knowledge graph generation failed: {e}")

    async def query_chemical_knowledge(
        self,
        query: str,
        entity_type: Optional[ChemicalEntityType] = None
    ) -> List[Dict[str, Any]]:
        """
        Query chemical knowledge base.

        Args:
            query: Search query
            entity_type: Optional entity type filter

        Returns:
            List of matching chemical entities
        """
        if not self.is_initialized:
            raise ChemicalKnowledgeError("GlobalChem bridge not initialized")

        try:
            # Use adapter to search
            search_results = await self.adapter.search(query, num_results=50)

            # Filter by entity type if specified
            if entity_type:
                filtered_results = []
                for chemical in search_results['chemicals']:
                    list_type = self._classify_list_type(chemical['list'])
                    if list_type == entity_type:
                        filtered_results.append(chemical)
                return filtered_results

            return search_results['chemicals']

        except Exception as e:
            logger.error(f"Failed to query chemical knowledge: {e}")
            raise ChemicalKnowledgeError(f"Query failed: {e}")

    async def integrate_with_oneke(
        self,
        text: str,
        oneke_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate with OneKE for enhanced entity recognition.

        Args:
            text: Input text
            oneke_results: Optional pre-computed OneKE results

        Returns:
            Integrated results with chemical entities
        """
        if not self.oneke_integration_enabled:
            logger.warning("OneKE integration not enabled")
            return {"chemical_entities": [], "oneke_integration": False}

        try:
            # Recognize chemical entities
            chemical_entities = await self.recognize_chemical_entities(text)

            # If OneKE results provided, merge them
            if oneke_results:
                # Merge chemical entities with OneKE entities
                # This is a simplified implementation
                return {
                    "chemical_entities": [e.__dict__ for e in chemical_entities],
                    "oneke_entities": oneke_results.get("entities", []),
                    "integrated": True,
                    "oneke_integration": True
                }

            return {
                "chemical_entities": [e.__dict__ for e in chemical_entities],
                "integrated": False,
                "oneke_integration": True
            }

        except Exception as e:
            logger.error(f"Failed to integrate with OneKE: {e}")
            raise ChemicalKnowledgeError(f"OneKE integration failed: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the bridge.

        Returns:
            Dictionary with statistics
        """
        if not self.is_initialized:
            return {"initialized": False}

        try:
            available_lists = await self.adapter.get_available_chemical_lists()

            return {
                "initialized": True,
                "oneke_integration_enabled": self.oneke_integration_enabled,
                "cached_entities": len(self.entity_cache),
                "cached_relationships": len(self.relationship_cache),
                "available_chemical_lists": len(available_lists),
                "lists": available_lists
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    async def shutdown(self) -> bool:
        """
        Shutdown the bridge.

        Returns:
            True if successful
        """
        try:
            # Clear caches
            self.entity_cache.clear()
            self.relationship_cache.clear()

            # Shutdown adapter
            if self.adapter.is_initialized:
                await self.adapter.shutdown()

            self.is_initialized = False
            logger.info("GlobalChem bridge shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False
