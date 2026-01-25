"""
Deduplication Module for KG-Gen Pipeline

This module provides functionality for deduplicating entities and relationships
in knowledge graphs using semantic hashing and clustering techniques.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of a deduplication operation."""
    success: bool
    original_count: int
    deduplicated_count: int
    entity_clusters: Dict[str, List[str]]  # canonical -> [duplicates]
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class DeduplicationEngine:
    """
    Deduplication engine for knowledge graphs.
    
    Provides methods for identifying and merging duplicate entities using:
    - Semantic hashing (SEMHASH)
    - Language model clustering
    - Hybrid approaches
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the deduplication engine.
        
        Args:
            config: Configuration for deduplication methods
        """
        self.config = config or self._get_default_config()
        
        # Initialize embedding model
        model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {model_name}: {e}")
            self.embedding_model = None
        
        logger.info({
            "msg": "DeduplicationEngine initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for deduplication."""
        return {
            'semhash_threshold': 0.95,
            'lm_cluster_size': 128,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'clustering_algorithm': 'hdbscan',  # 'hdbscan', 'dbscan', 'agglomerative'
            'min_cluster_size': 2,
            'cluster_selection_epsilon': 0.1
        }
    
    async def deduplicate_entities(
        self,
        entities: List[str],
        method: str = 'full',
        correlation_id: Optional[str] = None
    ) -> DeduplicationResult:
        """
        Deduplicate a list of entities using the specified method.
        
        Args:
            entities: List of entity names to deduplicate
            method: Deduplication method ('semhash', 'lm_cluster', 'full')
            correlation_id: Correlation ID for tracking
            
        Returns:
            DeduplicationResult with clusters and counts
        """
        correlation_id = correlation_id or f"dedup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting entity deduplication",
            "entity_count": len(entities),
            "method": method,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if method == 'semhash':
                result = await self._semhash_deduplication(entities, correlation_id)
            elif method == 'lm_cluster':
                result = await self._lm_cluster_deduplication(entities, correlation_id)
            elif method == 'full':
                # Apply both methods: first semhash, then LM clustering on canonicals
                semhash_result = await self._semhash_deduplication(entities, correlation_id)
                
                # Get canonical entities from semhash result
                canonical_entities = list(semhash_result.entity_clusters.keys())
                
                # Apply LM clustering to canonicals
                lm_result = await self._lm_cluster_deduplication(canonical_entities, correlation_id)
                
                # Combine results
                result = self._combine_deduplication_results(semhash_result, lm_result)
            else:
                raise ValueError(f"Unknown deduplication method: {method}")
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time_ms
            
            logger.info({
                "msg": "Entity deduplication completed",
                "correlation_id": correlation_id,
                "original_count": result.original_count,
                "deduplicated_count": result.deduplicated_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Entity deduplication failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return DeduplicationResult(
                success=False,
                original_count=len(entities),
                deduplicated_count=0,
                entity_clusters={},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def _semhash_deduplication(
        self,
        entities: List[str],
        correlation_id: str
    ) -> DeduplicationResult:
        """
        Perform semantic hash-based deduplication.
        
        Process:
        1. Create embeddings for each entity
        2. Generate semantic hashes
        3. Cluster similar hashes
        4. Merge duplicates
        """
        logger.debug({
            "msg": "Starting SEMHASH deduplication",
            "entity_count": len(entities),
            "correlation_id": correlation_id
        })
        
        if not entities:
            return DeduplicationResult(
                success=True,
                original_count=0,
                deduplicated_count=0,
                entity_clusters={}
            )
        
        # Create entity clusters based on semantic similarity
        entity_clusters = {}
        
        # For each entity, compute a normalized form for comparison
        for entity in entities:
            # Normalize the entity name
            normalized = entity.lower().strip()
            
            # Find if this entity is similar to any existing canonical
            found_match = False
            for canonical, duplicates in entity_clusters.items():
                if self._entities_are_similar_semhash(normalized, canonical.lower().strip()):
                    # Add to existing cluster
                    duplicates.append(entity)
                    found_match = True
                    break
            
            if not found_match:
                # Create new cluster with this entity as canonical
                entity_clusters[entity] = []
        
        result = DeduplicationResult(
            success=True,
            original_count=len(entities),
            deduplicated_count=len(entity_clusters),
            entity_clusters=entity_clusters
        )
        
        logger.debug({
            "msg": "SEMHASH deduplication completed",
            "correlation_id": correlation_id,
            "original_count": result.original_count,
            "deduplicated_count": result.deduplicated_count
        })
        
        return result
    
    def _entities_are_similar_semhash(self, entity1: str, entity2: str) -> bool:
        """
        Check if two entities are similar using semantic hashing approach.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            True if entities are similar enough to be considered duplicates
        """
        # Simple approach: check if one is contained in the other or vice versa
        # Also check for common variations
        e1, e2 = entity1.lower().strip(), entity2.lower().strip()
        
        # Direct match
        if e1 == e2:
            return True
        
        # One contains the other
        if e1 in e2 or e2 in e1:
            # Check if the shorter one is a significant portion of the longer one
            shorter, longer = (e1, e2) if len(e1) < len(e2) else (e2, e1)
            if len(shorter) / len(longer) > 0.7:  # At least 70% overlap
                return True
        
        # Check for common patterns like "Inc." vs "Incorporated", etc.
        variations_map = {
            'inc.': 'incorporated',
            'inc': 'incorporated',
            'corp.': 'corporation',
            'corp': 'corporation',
            'ltd.': 'limited',
            'ltd': 'limited',
            'co.': 'company',
            'co': 'company',
        }
        
        # Normalize by replacing common abbreviations
        norm1, norm2 = e1, e2
        for abbr, full in variations_map.items():
            norm1 = norm1.replace(abbr, full)
            norm2 = norm2.replace(abbr, full)
        
        if norm1 == norm2:
            return True
        
        # Check if they differ only by common suffixes/prefixes
        common_suffixes = [' inc', ' corp', ' ltd', ' co', ', inc', ', corp', ', ltd', ', co']
        for suffix in common_suffixes:
            if (norm1.endswith(suffix) and norm2 == norm1[:-len(suffix)]) or \
               (norm2.endswith(suffix) and norm1 == norm2[:-len(suffix)]):
                return True
        
        return False
    
    async def _lm_cluster_deduplication(
        self,
        entities: List[str],
        correlation_id: str
    ) -> DeduplicationResult:
        """
        Perform language model clustering deduplication.
        
        Process:
        1. Generate embeddings for all entities
        2. Perform clustering (DBSCAN/HDBSCAN)
        3. Merge entities within clusters
        """
        logger.debug({
            "msg": "Starting LM clustering deduplication",
            "entity_count": len(entities),
            "correlation_id": correlation_id
        })
        
        if not entities:
            return DeduplicationResult(
                success=True,
                original_count=0,
                deduplicated_count=0,
                entity_clusters={}
            )
        
        if not self.embedding_model:
            logger.warning({
                "msg": "Embedding model not available, skipping LM clustering",
                "correlation_id": correlation_id
            })
            # Return entities as-is in individual clusters
            entity_clusters = {entity: [] for entity in entities}
            return DeduplicationResult(
                success=True,
                original_count=len(entities),
                deduplicated_count=len(entities),
                entity_clusters=entity_clusters
            )
        
        try:
            # Generate embeddings for all entities
            embeddings = self.embedding_model.encode(entities)
            
            # Perform clustering based on configuration
            clustering_algorithm = self.config.get('clustering_algorithm', 'hdbscan')
            
            if clustering_algorithm == 'hdbscan':
                clusterer = HDBSCAN(
                    min_cluster_size=self.config.get('min_cluster_size', 2),
                    metric='euclidean',
                    cluster_selection_epsilon=self.config.get('cluster_selection_epsilon', 0.1)
                )
            elif clustering_algorithm == 'dbscan':
                clusterer = DBSCAN(
                    eps=self.config.get('cluster_selection_epsilon', 0.5),
                    min_samples=self.config.get('min_cluster_size', 2),
                    metric='euclidean'
                )
            else:
                logger.warning({
                    "msg": f"Unknown clustering algorithm {clustering_algorithm}, using HDBSCAN",
                    "correlation_id": correlation_id
                })
                clusterer = HDBSCAN(
                    min_cluster_size=self.config.get('min_cluster_size', 2),
                    metric='euclidean',
                    cluster_selection_epsilon=self.config.get('cluster_selection_epsilon', 0.1)
                )
            
            # Fit the clustering model
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Organize entities into clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise point, treat as individual cluster
                    clusters[entities[i]] = []
                else:
                    # Find the first entity in this cluster to be the canonical
                    canonical_found = False
                    for canonical, duplicates in clusters.items():
                        if canonical in entities and cluster_labels[entities.index(canonical)] == label:
                            # Add to existing cluster
                            if entities[i] != canonical:
                                duplicates.append(entities[i])
                            canonical_found = True
                            break
                    
                    if not canonical_found:
                        # Create new cluster with this entity as canonical
                        clusters[entities[i]] = []
            
            result = DeduplicationResult(
                success=True,
                original_count=len(entities),
                deduplicated_count=len(clusters),
                entity_clusters=clusters
            )
            
            logger.debug({
                "msg": "LM clustering deduplication completed",
                "correlation_id": correlation_id,
                "original_count": result.original_count,
                "deduplicated_count": result.deduplicated_count
            })
            
            return result
            
        except Exception as e:
            logger.error({
                "msg": f"LM clustering deduplication failed: {e}",
                "correlation_id": correlation_id
            })
            # Return entities as-is in individual clusters
            entity_clusters = {entity: [] for entity in entities}
            return DeduplicationResult(
                success=False,
                original_count=len(entities),
                deduplicated_count=len(entities),
                entity_clusters=entity_clusters,
                error=str(e)
            )
    
    def _combine_deduplication_results(
        self,
        result1: DeduplicationResult,
        result2: DeduplicationResult
    ) -> DeduplicationResult:
        """
        Combine results from two deduplication passes.
        
        Args:
            result1: First deduplication result
            result2: Second deduplication result (on canonicals from result1)
            
        Returns:
            Combined deduplication result
        """
        # Take the canonical entities from result2 and map back to original entities
        final_clusters = {}
        
        for canonical2, duplicates2 in result2.entity_clusters.items():
            # canonical2 was a canonical from result1, so it should be in result1's keys
            if canonical2 in result1.entity_clusters:
                # This becomes a new canonical, with all its original duplicates
                original_duplicates = result1.entity_clusters[canonical2]
                final_clusters[canonical2] = original_duplicates[:]
                
                # Add the duplicates of canonical2 (which were also canonical in result1)
                for dup in duplicates2:
                    if dup in result1.entity_clusters:
                        # Add all the original duplicates of this duplicate
                        final_clusters[canonical2].extend(result1.entity_clusters[dup])
                    else:
                        # This duplicate wasn't a canonical in result1, so just add it
                        final_clusters[canonical2].append(dup)
            else:
                # This shouldn't happen in normal operation, but handle gracefully
                final_clusters[canonical2] = []
        
        return DeduplicationResult(
            success=True,
            original_count=result1.original_count,
            deduplicated_count=len(final_clusters),
            entity_clusters=final_clusters
        )
    
    async def deduplicate_knowledge_graph(
        self,
        entities: List[str],
        relations: List[Tuple[str, str, str]],
        method: str = 'full',
        correlation_id: Optional[str] = None
    ) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[str, List[str]]]:
        """
        Deduplicate entities and update relationships accordingly.
        
        Args:
            entities: List of entity names
            relations: List of (subject, predicate, object) relationships
            method: Deduplication method to use
            correlation_id: Correlation ID for tracking
            
        Returns:
            Tuple of (deduplicated_entities, updated_relations, entity_clusters)
        """
        correlation_id = correlation_id or f"graph_dedup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting knowledge graph deduplication",
            "entity_count": len(entities),
            "relation_count": len(relations),
            "method": method,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # First, deduplicate entities
            dedup_result = await self.deduplicate_entities(
                entities=entities,
                method=method,
                correlation_id=correlation_id
            )
            
            if not dedup_result.success:
                logger.error({
                    "msg": "Entity deduplication failed in graph deduplication",
                    "correlation_id": correlation_id,
                    "error": dedup_result.error
                })
                return entities, relations, {}
            
            # Create a mapping from original entities to canonical entities
            entity_mapping = {}
            for canonical, duplicates in dedup_result.entity_clusters.items():
                entity_mapping[canonical] = canonical
                for duplicate in duplicates:
                    entity_mapping[duplicate] = canonical
            
            # Update relations to use canonical entities
            updated_relations = []
            for subj, pred, obj in relations:
                canonical_subj = entity_mapping.get(subj, subj)
                canonical_obj = entity_mapping.get(obj, obj)
                updated_relations.append((canonical_subj, pred, canonical_obj))
            
            # Get the canonical entities
            canonical_entities = list(dedup_result.entity_clusters.keys())
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Knowledge graph deduplication completed",
                "correlation_id": correlation_id,
                "original_entities": len(entities),
                "deduplicated_entities": len(canonical_entities),
                "original_relations": len(relations),
                "updated_relations": len(updated_relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return canonical_entities, updated_relations, dedup_result.entity_clusters
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge graph deduplication failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return original data in case of failure
            return entities, relations, {}
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the deduplication engine.
        
        Returns:
            Dictionary with engine statistics
        """
        return {
            "embedding_model_loaded": self.embedding_model is not None,
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }