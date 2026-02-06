"""
ROMA Entity Knowledge Graph Module

Entity knowledge graph for ROMA framework with advanced semantic analysis,
relationship mapping, and entanglement detection.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import hashlib
import json
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class ROMAEntityKG:
    """ROMA Entity Knowledge Graph class with full implementation"""
    
    def __init__(self):
        logger.info("ROMA Entity KG initialized")
        self.entities = {}  # entity_id -> entity_data
        self.entity_types = defaultdict(set)  # type -> set of entity_ids
        self.relationships = defaultdict(list)  # entity_id -> list of (relationship_type, target_entity_id)
        self.inverse_relationships = defaultdict(list)  # entity_id -> list of (relationship_type, source_entity_id)
        self.entity_attributes = defaultdict(dict)  # entity_id -> {attribute: value}
        self.attribute_index = defaultdict(set)  # attribute_value -> set of entity_ids
        self.semantic_vectors = {}  # entity_id -> list of semantic features
        self.entanglement_matrix = defaultdict(dict)  # entity_id -> {other_entity_id: entanglement_strength}
        self.timestamp = datetime.utcnow()
        
    def add_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Add entity to knowledge graph with full relationship and attribute indexing.
        
        Args:
            entity: Entity dictionary with id, type, attributes, and relationships
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            # Validate entity structure
            if not isinstance(entity, dict):
                logger.error("Entity must be a dictionary")
                return False
                
            if "id" not in entity:
                # Generate ID if not provided
                entity_id = hashlib.md5(json.dumps(entity, sort_keys=True).encode()).hexdigest()[:12]
                entity["id"] = entity_id
            else:
                entity_id = entity["id"]
                
            # Validate required fields
            entity_type = entity.get("type", "general")
            attributes = entity.get("attributes", {})
            relationships = entity.get("relationships", [])
            
            # Store entity
            self.entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "attributes": attributes,
                "relationships": relationships,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Index by type
            self.entity_types[entity_type].add(entity_id)
            
            # Index attributes
            for attr_name, attr_value in attributes.items():
                attr_key = f"{attr_name}:{str(attr_value)}"
                self.attribute_index[attr_key].add(entity_id)
                
                # Also index by attribute name for broader searches
                self.attribute_index[attr_name].add(entity_id)
            
            # Process relationships
            for rel in relationships:
                if isinstance(rel, dict) and "type" in rel and "target" in rel:
                    target_id = rel["target"]
                    rel_type = rel["type"]
                    
                    # Add forward relationship
                    self.relationships[entity_id].append((rel_type, target_id))
                    
                    # Add inverse relationship
                    self.inverse_relationships[target_id].append((rel_type, entity_id))
                    
                    # Calculate entanglement based on relationship type
                    entanglement_strength = self._calculate_entanglement_strength(rel_type)
                    self.entanglement_matrix[entity_id][target_id] = entanglement_strength
                    self.entanglement_matrix[target_id][entity_id] = entanglement_strength
            
            # Generate semantic vector for similarity calculations
            self.semantic_vectors[entity_id] = self._generate_semantic_vector(entity)
            
            logger.info(f"Entity '{entity_id}' of type '{entity_type}' added to knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entity to knowledge graph: {e}")
            return False
    
    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query knowledge graph with advanced filtering and relationship traversal.
        
        Args:
            query: Query dictionary with filters, relationships, and semantic search
            
        Returns:
            List of matching entities with relationship information
        """
        try:
            # Extract query parameters
            entity_type = query.get("type")
            attributes = query.get("attributes", {})
            relationships = query.get("relationships", [])
            text_search = query.get("text", "")
            semantic_similarity_threshold = query.get("semantic_threshold", 0.7)
            max_results = query.get("max_results", 100)
            include_relationships = query.get("include_relationships", True)
            include_attributes = query.get("include_attributes", True)
            
            # Start with all entities if no type specified, otherwise filter by type
            if entity_type:
                candidate_entities = self.entity_types.get(entity_type, set())
            else:
                candidate_entities = set(self.entities.keys())
            
            # Filter by attributes
            for attr_name, attr_value in attributes.items():
                attr_key = f"{attr_name}:{str(attr_value)}"
                attr_candidates = self.attribute_index.get(attr_key, set())
                candidate_entities &= attr_candidates
            
            # If text search is specified, perform semantic matching
            if text_search:
                text_candidates = self._semantic_search(text_search, semantic_similarity_threshold)
                candidate_entities &= set(text_candidates)
            
            # Apply relationship filters
            if relationships:
                for rel_filter in relationships:
                    if isinstance(rel_filter, dict) and "type" in rel_filter and "target" in rel_filter:
                        rel_type = rel_filter["type"]
                        target = rel_filter["target"]
                        
                        # Find entities that have this relationship with the target
                        matching_entities = set()
                        for entity_id in candidate_entities:
                            for rel_type_found, target_found in self.relationships[entity_id]:
                                if rel_type_found == rel_type and target_found == target:
                                    matching_entities.add(entity_id)
                        candidate_entities &= matching_entities
            
            # Convert to list of entity objects
            results = []
            for entity_id in list(candidate_entities)[:max_results]:
                entity = self.entities[entity_id].copy()
                
                # Conditionally include relationships
                if include_relationships:
                    entity["relationships"] = self._get_entity_relationships(entity_id)
                else:
                    entity.pop("relationships", None)
                
                # Conditionally include attributes
                if not include_attributes:
                    entity.pop("attributes", None)
                
                results.append(entity)
            
            logger.info(f"Query returned {len(results)} entities from {len(candidate_entities)} candidates")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def _calculate_entanglement_strength(self, relationship_type: str) -> float:
        """Calculate entanglement strength based on relationship type."""
        # Define relationship type strengths
        strength_map = {
            "dependency": 0.9,
            "part_of": 0.8,
            "contains": 0.8,
            "inherits": 0.7,
            "uses": 0.6,
            "related_to": 0.5,
            "similar_to": 0.4,
            "connected_to": 0.3,
            "associated_with": 0.2
        }
        
        return strength_map.get(relationship_type, 0.1)
    
    def _generate_semantic_vector(self, entity: Dict[str, Any]) -> List[float]:
        """Generate semantic vector for entity based on its content."""
        # Create a text representation of the entity
        text_parts = []
        
        # Add type
        text_parts.append(entity.get("type", ""))
        
        # Add attribute values
        for attr_value in entity.get("attributes", {}).values():
            text_parts.append(str(attr_value))
        
        # Add relationship targets
        for rel in entity.get("relationships", []):
            if isinstance(rel, dict) and "target" in rel:
                text_parts.append(rel["target"])
        
        # Combine all text
        full_text = " ".join(text_parts).lower()
        
        # Simple semantic vector based on character frequencies
        # In a real implementation, this would use embeddings
        vector = [0.0] * 256  # ASCII characters
        for char in full_text:
            if ord(char) < 256:
                vector[ord(char)] += 1
        
        # Normalize vector
        total = sum(vector)
        if total > 0:
            vector = [v / total for v in vector]
        
        return vector
    
    def _semantic_search(self, query_text: str, threshold: float) -> List[str]:
        """Perform semantic similarity search."""
        query_vector = self._generate_semantic_vector({"type": "", "attributes": {"text": query_text}, "relationships": []})
        
        matches = []
        for entity_id, entity_vector in self.semantic_vectors.items():
            similarity = self._cosine_similarity(query_vector, entity_vector)
            if similarity >= threshold:
                matches.append((entity_id, similarity))
        
        # Sort by similarity and return entity IDs
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        relationships = []
        
        # Forward relationships
        for rel_type, target_id in self.relationships.get(entity_id, []):
            relationships.append({
                "type": rel_type,
                "target": target_id,
                "direction": "forward"
            })
        
        # Inverse relationships (incoming)
        for rel_type, source_id in self.inverse_relationships.get(entity_id, []):
            relationships.append({
                "type": rel_type,
                "target": source_id,
                "direction": "inverse"
            })
        
        return relationships
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by its ID."""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_types.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_related_entities(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get entities related to a given entity."""
        related_ids = set()
        
        # Get forward relationships
        for rel_type, target_id in self.relationships.get(entity_id, []):
            if relationship_type is None or rel_type == relationship_type:
                related_ids.add(target_id)
        
        # Get inverse relationships
        for rel_type, source_id in self.inverse_relationships.get(entity_id, []):
            if relationship_type is None or rel_type == relationship_type:
                related_ids.add(source_id)
        
        return [self.entities[eid] for eid in related_ids if eid in self.entities]
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove entity from knowledge graph."""
        if entity_id not in self.entities:
            return False
        
        # Remove from type index
        entity_type = self.entities[entity_id]["type"]
        self.entity_types[entity_type].discard(entity_id)
        
        # Remove from attribute index
        for attr_name, attr_value in self.entities[entity_id]["attributes"].items():
            attr_key = f"{attr_name}:{str(attr_value)}"
            self.attribute_index[attr_key].discard(entity_id)
            self.attribute_index[attr_name].discard(entity_id)
        
        # Remove relationships
        del self.relationships[entity_id]
        del self.inverse_relationships[entity_id]
        
        # Remove semantic vector
        if entity_id in self.semantic_vectors:
            del self.semantic_vectors[entity_id]
        
        # Remove from entanglement matrix
        if entity_id in self.entanglement_matrix:
            del self.entanglement_matrix[entity_id]
        # Remove references to this entity in other entanglement entries
        for other_id in self.entanglement_matrix:
            if entity_id in self.entanglement_matrix[other_id]:
                del self.entanglement_matrix[other_id][entity_id]
        
        # Remove entity
        del self.entities[entity_id]
        
        logger.info(f"Entity '{entity_id}' removed from knowledge graph")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_entities": len(self.entities),
            "total_relationships": sum(len(rels) for rels in self.relationships.values()),
            "entity_types": {etype: len(ids) for etype, ids in self.entity_types.items()},
            "timestamp": self.timestamp.isoformat()
        }
    
    def clear(self):
        """Clear the entire knowledge graph."""
        self.entities.clear()
        self.entity_types.clear()
        self.relationships.clear()
        self.inverse_relationships.clear()
        self.entity_attributes.clear()
        self.attribute_index.clear()
        self.semantic_vectors.clear()
        self.entanglement_matrix.clear()
        self.timestamp = datetime.utcnow()
        logger.info("Knowledge graph cleared")
