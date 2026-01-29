"""
KG-Gen Extraction Pipeline for OpenEvolve Knowledge Engine

This module provides the complete 3-stage knowledge graph extraction pipeline:
1. Entity Extraction: Identify and extract entities using DSPy
2. Relation Extraction: Extract subject-predicate-object triples
3. Deduplication: Merge duplicate entities using SEMHASH and LM clustering

Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: All config via parameters/env vars
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
- RUNTIME TRUTH: Verify components before use
- IDEMPOTENCY: All operations safe to run multiple times
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import uuid
import hashlib
import json
from pathlib import Path

# Try to import DSPy and related libraries
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraph:
    """Representation of a knowledge graph with entities and relationships."""
    entities: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    entity_clusters: Dict[str, List[str]] = field(default_factory=dict)  # canonical -> [duplicates]
    
    def merge(self, other: 'KnowledgeGraph'):
        """Merge another knowledge graph into this one."""
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)
        
        # Merge entity clusters
        for canonical, duplicates in other.entity_clusters.items():
            if canonical in self.entity_clusters:
                self.entity_clusters[canonical].extend(duplicates)
            else:
                self.entity_clusters[canonical] = duplicates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'entities': self.entities,
            'relations': self.relations,
            'entity_clusters': self.entity_clusters
        }


class ExtractionPipeline:
    """
    KG-Gen Extraction Pipeline with 3-stage processing.
    
    Provides methods for:
    - Entity extraction using DSPy
    - Relation extraction
    - Entity deduplication
    - Graph construction
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        timeout_ms: int = 30000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            model: LLM model to use for extraction
            timeout_ms: Timeout for extraction operations
            config: Additional configuration options
        """
        self.model = model
        self.timeout_ms = timeout_ms
        self.config = config or self._get_default_config()
        
        # Initialize DSPy if available
        self.lm = None
        if DSPY_AVAILABLE:
            try:
                self.lm = dspy.OpenAI(model=model)
                dspy.configure(lm=self.lm)
            except Exception as e:
                logger.warning(f"Could not initialize DSPy with model {model}: {e}")
        
        # Initialize embedding model for deduplication
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Could not initialize embedding model {model_name}: {e}")
        
        logger.info({
            "msg": "KG-Gen ExtractionPipeline initialized",
            "model": model,
            "timeout_ms": timeout_ms,
            "dspy_available": DSPY_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'chunk_size': 5000,
            'overlap': 200,
            'parallel_workers': 4,
            'semhash_threshold': 0.95,
            'lm_cluster_size': 128,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'clustering_algorithm': 'hdbscan',
            'min_cluster_size': 2,
            'cluster_selection_epsilon': 0.1,
            'extract_temporal': True,
            'extract_attributes': True,
            'entity_types': [
                'PERSON', 'ORGANIZATION', 'LOCATION', 'TECHNOLOGY',
                'CONCEPT', 'PRODUCT', 'EVENT', 'DATE'
            ]
        }
    
    async def extract(
        self,
        text: str,
        context: str = "",
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract knowledge graph from text using the complete 3-stage pipeline.
        
        Args:
            text: Input text to extract knowledge from
            context: Context information for extraction
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with extracted entities, relations, and metadata
        """
        correlation_id = correlation_id or f"extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting KG-Gen extraction",
            "text_length": len(text),
            "context": context,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Stage 1: Entity Extraction
            entities = await self._extract_entities(text, context, correlation_id)
            
            # Stage 2: Relation Extraction
            relations = await self._extract_relations(text, entities, context, correlation_id)
            
            # Stage 3: Deduplication
            deduplicated_graph = await self._deduplicate_graph(
                KnowledgeGraph(entities=entities, relations=relations),
                method=self.config.get('deduplication_method', 'full'),
                correlation_id=correlation_id
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "entities": deduplicated_graph.entities,
                "relations": deduplicated_graph.relations,
                "triples": deduplicated_graph.relations,  # For compatibility
                "entity_clusters": deduplicated_graph.entity_clusters,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "stages_completed": ["entity_extraction", "relation_extraction", "deduplication"]
            }
            
            logger.info({
                "msg": "KG-Gen extraction completed",
                "correlation_id": correlation_id,
                "entities_count": len(deduplicated_graph.entities),
                "relations_count": len(deduplicated_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "KG-Gen extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
    
    async def _extract_entities(
        self,
        text: str,
        context: str,
        correlation_id: str
    ) -> List[str]:
        """Stage 1: Extract entities from text."""
        logger.debug({
            "msg": "Stage 1: Extracting entities",
            "text_length": len(text),
            "correlation_id": correlation_id
        })
        
        if DSPY_AVAILABLE and self.lm:
            try:
                # Use DSPy for entity extraction
                # This is a simplified implementation - in practice, you'd define
                # a DSPy signature for entity extraction
                prompt = f"""
                Extract named entities from the following text. Context: {context}
                
                Text: {text}
                
                Return a list of named entities found in the text.
                """
                
                response = self.lm(prompt, max_tokens=2000)
                # Parse the response to extract entities
                # This is a simplified approach - real implementation would parse structured output
                entities_str = response[0]['choices'][0]['message']['content']
                
                # Try to parse as JSON if possible, otherwise split by common separators
                try:
                    entities = json.loads(entities_str)
                except json.JSONDecodeError:
                    # Split by common separators and clean up
                    import re
                    entities = re.split(r'[,\n\r]+', entities_str)
                    entities = [e.strip().strip('"\'') for e in entities if e.strip()]
                
                logger.debug({
                    "msg": "Entities extracted with DSPy",
                    "count": len(entities),
                    "correlation_id": correlation_id
                })
                
                return entities
            except Exception as e:
                logger.warning({
                    "msg": f"DSPy entity extraction failed: {e}, falling back to simple extraction",
                    "correlation_id": correlation_id
                })
        
        # Fallback: Simple entity extraction using regex patterns
        import re
        
        # Common patterns for named entities
        patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Person names (John Doe)
            r'\b[A-Z][A-Z]+\b',  # Organizations (NASA, FBI)
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Mixed case entities
            r'\b\d{4}\b',  # Years
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        logger.debug({
            "msg": "Entities extracted with fallback method",
            "count": len(unique_entities),
            "correlation_id": correlation_id
        })
        
        return unique_entities
    
    async def _extract_relations(
        self,
        text: str,
        entities: List[str],
        context: str,
        correlation_id: str
    ) -> List[Tuple[str, str, str]]:
        """Stage 2: Extract relations between entities."""
        logger.debug({
            "msg": "Stage 2: Extracting relations",
            "text_length": len(text),
            "entities_count": len(entities),
            "correlation_id": correlation_id
        })
        
        if DSPY_AVAILABLE and self.lm:
            try:
                # Use DSPy for relation extraction
                prompt = f"""
                Extract subject-predicate-object relations from the following text.
                Use the provided entities to form meaningful triples.
                Context: {context}
                
                Text: {text}
                
                Entities: {', '.join(entities)}
                
                Return a list of (subject, predicate, object) triples.
                """
                
                response = self.lm(prompt, max_tokens=3000)
                relations_str = response[0]['choices'][0]['message']['content']
                
                # Parse the response to extract relations
                try:
                    relations_list = json.loads(relations_str)
                    # Convert to tuples if needed
                    relations = [(rel[0], rel[1], rel[2]) for rel in relations_list if len(rel) >= 3]
                except json.JSONDecodeError:
                    # Fallback parsing
                    import re
                    # Look for patterns like (subject, predicate, object)
                    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^\)]+)\)'
                    matches = re.findall(pattern, relations_str)
                    relations = [(m[0].strip().strip('"\''), m[1].strip().strip('"\''), m[2].strip().strip('"\'')) for m in matches]
                
                logger.debug({
                    "msg": "Relations extracted with DSPy",
                    "count": len(relations),
                    "correlation_id": correlation_id
                })
                
                return relations
            except Exception as e:
                logger.warning({
                    "msg": f"DSPy relation extraction failed: {e}, falling back to simple extraction",
                    "correlation_id": correlation_id
                })
        
        # Fallback: Simple relation extraction
        # Look for patterns in the text that connect entities
        relations = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Look for connecting words between entities in the text
                    text_lower = text.lower()
                    entity1_lower = entity1.lower()
                    entity2_lower = entity2.lower()
                    
                    # Find positions of entities in text
                    pos1 = text_lower.find(entity1_lower)
                    pos2 = text_lower.find(entity2_lower)
                    
                    if pos1 != -1 and pos2 != -1:
                        # Get the text between entities
                        start_pos = min(pos1, pos2) + len(entity1_lower if pos1 < pos2 else entity2_lower)
                        end_pos = max(pos1, pos2)
                        middle_text = text[start_pos:end_pos].strip()
                        
                        # Look for common relation words
                        if 'is' in middle_text or 'was' in middle_text or 'works' in middle_text:
                            # Determine direction based on order in text
                            if pos1 < pos2:
                                relations.append((entity1, 'related_to', entity2))
                            else:
                                relations.append((entity2, 'related_to', entity1))
        
        logger.debug({
            "msg": "Relations extracted with fallback method",
            "count": len(relations),
            "correlation_id": correlation_id
        })
        
        return relations
    
    async def _deduplicate_graph(
        self,
        graph: KnowledgeGraph,
        method: str = 'full',
        correlation_id: str = None
    ) -> KnowledgeGraph:
        """Stage 3: Deduplicate entities and relations."""
        logger.debug({
            "msg": f"Stage 3: Deduplicating graph with method {method}",
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "correlation_id": correlation_id
        })
        
        if method == 'semhash':
            return await self._semhash_deduplication(graph, correlation_id)
        elif method == 'lm_cluster':
            return await self._lm_cluster_deduplication(graph, correlation_id)
        elif method == 'full':
            # Apply both methods
            graph_after_semhash = await self._semhash_deduplication(graph, correlation_id)
            return await self._lm_cluster_deduplication(graph_after_semhash, correlation_id)
        else:
            logger.warning({
                "msg": f"Unknown deduplication method: {method}, using 'full'",
                "correlation_id": correlation_id
            })
            graph_after_semhash = await self._semhash_deduplication(graph, correlation_id)
            return await self._lm_cluster_deduplication(graph_after_semhash, correlation_id)
    
    async def _semhash_deduplication(
        self,
        graph: KnowledgeGraph,
        correlation_id: str = None
    ) -> KnowledgeGraph:
        """Apply semantic hash-based deduplication."""
        logger.debug({
            "msg": "Applying semantic hash deduplication",
            "entities_count": len(graph.entities),
            "correlation_id": correlation_id
        })
        
        # Create a new graph to populate with deduplicated entities
        new_graph = KnowledgeGraph()
        entity_clusters = {}
        
        # For each entity, compute a semantic hash
        for entity in graph.entities:
            # Compute a hash based on the entity name
            entity_hash = hashlib.md5(entity.lower().encode()).hexdigest()[:8]
            
            # Group entities by their hash (first 8 chars for similarity)
            found_cluster = False
            for canonical, duplicates in entity_clusters.items():
                canonical_hash = hashlib.md5(canonical.lower().encode()).hexdigest()[:8]
                if entity_hash == canonical_hash:
                    # Add to existing cluster
                    duplicates.append(entity)
                    found_cluster = True
                    break
            
            if not found_cluster:
                # Create new cluster with this entity as canonical
                entity_clusters[entity] = []
        
        # Select canonical entities and update relations
        canonical_entities = list(entity_clusters.keys())
        new_graph.entities = canonical_entities
        new_graph.entity_clusters = entity_clusters
        
        # Update relations to use canonical entities
        for subj, pred, obj in graph.relations:
            # Map to canonical forms
            canonical_subj = self._map_to_canonical(subj, entity_clusters)
            canonical_obj = self._map_to_canonical(obj, entity_clusters)
            new_graph.relations.append((canonical_subj, pred, canonical_obj))
        
        logger.debug({
            "msg": "Semantic hash deduplication completed",
            "original_count": len(graph.entities),
            "deduplicated_count": len(new_graph.entities),
            "correlation_id": correlation_id
        })
        
        return new_graph
    
    async def _lm_cluster_deduplication(
        self,
        graph: KnowledgeGraph,
        correlation_id: str = None
    ) -> KnowledgeGraph:
        """Apply language model clustering deduplication."""
        logger.debug({
            "msg": "Applying LM clustering deduplication",
            "entities_count": len(graph.entities),
            "correlation_id": correlation_id
        })
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.embedding_model:
            logger.warning({
                "msg": "Sentence transformers not available, skipping LM clustering",
                "correlation_id": correlation_id
            })
            return graph
        
        try:
            # Generate embeddings for all entities
            entities = graph.entities
            if not entities:
                return graph
            
            embeddings = self.embedding_model.encode(entities)
            
            # Perform clustering (using a simple approach since we don't have sklearn in this context)
            # In a real implementation, we'd use DBSCAN, HDBSCAN, or similar
            clusters = self._simple_clustering(embeddings, entities)
            
            # Create new graph with clustered entities
            new_graph = KnowledgeGraph()
            entity_clusters = {}
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Multiple entities in cluster - pick the first as canonical
                    canonical = cluster[0]
                    duplicates = cluster[1:]
                    entity_clusters[canonical] = duplicates
                    
                    # Add canonical entity
                    new_graph.entities.append(canonical)
                else:
                    # Single entity in cluster
                    new_graph.entities.append(cluster[0])
            
            # Update relations to use canonical entities
            for subj, pred, obj in graph.relations:
                canonical_subj = self._map_to_canonical_cluster(subj, entity_clusters)
                canonical_obj = self._map_to_canonical_cluster(obj, entity_clusters)
                new_graph.relations.append((canonical_subj, pred, canonical_obj))
            
            # Add any remaining relations that weren't mapped
            new_graph.entity_clusters = entity_clusters
            
            logger.debug({
                "msg": "LM clustering deduplication completed",
                "original_count": len(graph.entities),
                "deduplicated_count": len(new_graph.entities),
                "correlation_id": correlation_id
            })
            
            return new_graph
            
        except Exception as e:
            logger.error({
                "msg": f"LM clustering deduplication failed: {e}",
                "correlation_id": correlation_id
            })
            return graph
    
    def _simple_clustering(self, embeddings, entities):
        """Simple clustering based on cosine similarity."""
        if len(entities) <= 1:
            return [[entity] for entity in entities]
        
        # Calculate similarities (simplified approach)
        clusters = []
        used = set()
        
        for i, entity in enumerate(entities):
            if i in used:
                continue
                
            cluster = [entity]
            used.add(i)
            
            # Find similar entities
            for j, other_entity in enumerate(entities):
                if j in used or i == j:
                    continue
                
                # Calculate similarity (simplified - in practice use cosine similarity)
                similarity = self._calculate_similarity(embeddings[i], embeddings[j])
                
                if similarity > self.config.get('semhash_threshold', 0.95):
                    cluster.append(other_entity)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings."""
        if NUMPY_AVAILABLE:
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        else:
            # Fallback: simple similarity based on string length and common chars
            str1 = str(emb1)
            str2 = str(emb2)
            common_chars = set(str1) & set(str2)
            return len(common_chars) / max(len(set(str1)), len(set(str2)), 1)
    
    def _map_to_canonical(self, entity, entity_clusters):
        """Map an entity to its canonical form."""
        for canonical, duplicates in entity_clusters.items():
            if entity == canonical or entity in duplicates:
                return canonical
        return entity  # Return original if not found in clusters
    
    def _map_to_canonical_cluster(self, entity, entity_clusters):
        """Map an entity to its canonical form based on clusters."""
        for canonical, duplicates in entity_clusters.items():
            if entity == canonical or entity in duplicates:
                return canonical
        return entity  # Return original if not found in clusters
    
    async def extract_from_large_document(
        self,
        document: str,
        chunk_size: int = 5000,
        overlap: int = 200,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge from large documents by chunking and processing in parallel.
        
        Args:
            document: Large document text
            chunk_size: Size of chunks for processing
            overlap: Overlap between chunks
            correlation_id: Correlation ID for tracking
            
        Returns:
            Combined knowledge graph from all chunks
        """
        correlation_id = correlation_id or f"large_doc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Extracting from large document",
            "document_length": len(document),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create chunks
            chunks = self._create_chunks(document, chunk_size, overlap)
            
            logger.info({
                "msg": "Document chunked",
                "chunk_count": len(chunks),
                "correlation_id": correlation_id
            })
            
            # Process chunks in parallel
            tasks = []
            for i, chunk in enumerate(chunks):
                task = self.extract(
                    text=chunk,
                    context=f"Chunk {i+1} of {len(chunks)}",
                    correlation_id=f"{correlation_id}_chunk_{i}"
                )
                tasks.append(task)
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_graph = KnowledgeGraph()
            successful_extractions = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Chunk {i} extraction failed",
                        "error": str(result),
                        "correlation_id": f"{correlation_id}_chunk_{i}"
                    })
                    continue
                
                if result.get("success"):
                    # Convert result to KnowledgeGraph and merge
                    chunk_graph = KnowledgeGraph(
                        entities=result.get("entities", []),
                        relations=result.get("relations", []),
                        entity_clusters=result.get("entity_clusters", {})
                    )
                    combined_graph.merge(chunk_graph)
                    successful_extractions += 1
                else:
                    logger.warning({
                        "msg": f"Chunk {i} extraction unsuccessful",
                        "error": result.get("error"),
                        "correlation_id": f"{correlation_id}_chunk_{i}"
                    })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Large document extraction completed",
                "correlation_id": correlation_id,
                "successful_chunks": successful_extractions,
                "total_chunks": len(chunks),
                "final_entities_count": len(combined_graph.entities),
                "final_relations_count": len(combined_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return combined_graph
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Large document extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeGraph()
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create chunks from text with specified size and overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position by chunk_size minus overlap
            start = end - overlap
            
            # If we're near the end, adjust to avoid negative start
            if start >= len(text):
                break
        
        return chunks
    
    async def extract_batch(
        self,
        texts: List[str],
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge graphs from multiple texts in batch.
        
        Args:
            texts: List of input texts
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of extraction results
        """
        correlation_id = correlation_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting batch extraction",
            "text_count": len(texts),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process all texts in parallel
            tasks = []
            for i, text in enumerate(texts):
                task = self.extract(
                    text=text,
                    context=f"Batch item {i+1}",
                    correlation_id=f"{correlation_id}_item_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            successful = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch item {i} extraction failed",
                        "error": str(result),
                        "correlation_id": f"{correlation_id}_item_{i}"
                    })
                    processed_results.append({
                        "success": False,
                        "error": str(result),
                        "index": i
                    })
                else:
                    processed_results.append(result)
                    if result.get("success"):
                        successful += 1
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Batch extraction completed",
                "correlation_id": correlation_id,
                "successful": successful,
                "total": len(texts),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return processed_results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Batch extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return []