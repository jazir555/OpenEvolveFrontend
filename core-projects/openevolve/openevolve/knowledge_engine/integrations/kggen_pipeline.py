"""
KG-Gen Graph Generation Pipeline Integration

This module integrates kg-gen's advanced 3-stage pipeline (Entity Extraction →
Relation Extraction → Deduplication) with the Knowledge Engine, including parallel
chunk processing and Neo4j auto-upload.
"""

import asyncio
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, Callable

import yaml

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Represents a knowledge graph with entities, relationships, and metadata.
    """

    def __init__(
        self,
        entities: Optional[List[str]] = None,
        relationships: Optional[List[Tuple[str, str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a knowledge graph.

        Args:
            entities: List of entity names
            relationships: List of (subject, predicate, object) triples
            metadata: Optional metadata dictionary
        """
        self.entities = entities or []
        self.relationships = relationships or []
        self.metadata = metadata or {
            "created_at": datetime.now().isoformat(),
            "source": "kg-gen-pipeline"
        }
        self.entity_clusters: Dict[str, List[str]] = {}

    def add_entity(self, entity: str):
        """Add an entity to the graph."""
        if entity not in self.entities:
            self.entities.append(entity)

    def add_relationship(self, subject: str, predicate: str, obj: str):
        """Add a relationship to the graph."""
        triple = (subject, predicate, obj)
        if triple not in self.relationships:
            self.relationships.append(triple)

    def merge(self, other: 'KnowledgeGraph'):
        """Merge another knowledge graph into this one."""
        for entity in other.entities:
            self.add_entity(entity)
        for rel in other.relationships:
            self.add_relationship(*rel)
        # Merge clusters
        for cluster_id, entities in other.entity_clusters.items():
            if cluster_id not in self.entity_clusters:
                self.entity_clusters[cluster_id] = []
            for entity in entities:
                if entity not in self.entity_clusters[cluster_id]:
                    self.entity_clusters[cluster_id].append(entity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entities": self.entities,
            "relationships": [
                {"subject": s, "predicate": p, "object": o}
                for s, p, o in self.relationships
            ],
            "metadata": self.metadata,
            "entity_clusters": self.entity_clusters
        }


class UploadResult:
    """Result of Neo4j upload operation."""

    def __init__(
        self,
        success: bool,
        entities_uploaded: int = 0,
        relationships_uploaded: int = 0,
        error: Optional[str] = None
    ):
        self.success = success
        self.entities_uploaded = entities_uploaded
        self.relationships_uploaded = relationships_uploaded
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "entities_uploaded": self.entities_uploaded,
            "relationships_uploaded": self.relationships_uploaded,
            "error": self.error,
            "timestamp": self.timestamp
        }


class KGGenPipelineIntegration:
    """
    Integrates kg-gen's 3-stage pipeline with Knowledge Engine.

    Stages:
    1. Entity Extraction - Extract entities with DSPy
    2. Relation Extraction - Extract SPO triples
    3. Deduplication - SEMHASH + LM clustering
    """

    def __init__(self, kggen_client=None, neo4j_backend=None):
        """
        Initialize the kg-gen pipeline integration.

        Args:
            kggen_client: Optional kg-gen client instance
            neo4j_backend: Optional Neo4j backend instance
        """
        self.kggen = kggen_client
        self.neo4j = neo4j_backend
        self.pipeline_config = self._load_config()
        self.executor = ThreadPoolExecutor(
            max_workers=self.pipeline_config.get('parallel_workers', 4)
        )
        logger.info("KGGenPipelineIntegration initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration from YAML file."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'config',
            'kggen_pipeline.yaml'
        )

        default_config = {
            'enabled': True,
            'default_chunk_size': 5000,
            'default_overlap': 200,
            'parallel_workers': 4,
            'stages': {
                'entity_extraction': {
                    'model': 'openai/gpt-4o',
                    'temperature': 0.0,
                    'max_tokens': 4000
                },
                'relation_extraction': {
                    'model': 'openai/gpt-4o',
                    'temperature': 0.0,
                    'max_tokens': 8000
                },
                'deduplication': {
                    'method': 'full',
                    'semhash_threshold': 0.95,
                    'lm_cluster_size': 128
                }
            },
            'neo4j_upload': {
                'enabled': True,
                'batch_size': 100,
                'create_indices': True,
                'verify_upload': True
            },
            'progress_tracking': {
                'enabled': True,
                'log_interval': 10
            }
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge configs
                    if user_config:
                        self._deep_merge(default_config, user_config)
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")

        return default_config

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    async def extract_knowledge_graph(
        self,
        text: str,
        context: str = "",
        chunk_size: int = 5000
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from text using 3-stage pipeline.

        Args:
            text: Input text to extract knowledge from
            context: Optional context information
            chunk_size: Maximum chunk size for processing

        Returns:
            KnowledgeGraph object with extracted entities and relationships
        """
        logger.info(f"Starting knowledge graph extraction for text length: {len(text)}")

        # Stage 1: Entity Extraction
        entities = await self._extract_entities(text, context)
        logger.info(f"Extracted {len(entities)} entities")

        # Stage 2: Relation Extraction
        relationships = await self._extract_relations(text, entities, context)
        logger.info(f"Extracted {len(relationships)} relationships")

        # Create initial knowledge graph
        graph = KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            metadata={
                "context": context,
                "text_length": len(text),
                "extraction_method": "3-stage-pipeline"
            }
        )

        # Stage 3: Deduplication
        deduped_graph = await self._deduplicate_graph(
            graph,
            method=self.pipeline_config['stages']['deduplication']['method']
        )
        logger.info(f"Deduplication complete. {len(deduped_graph.entities)} unique entities")

        return deduped_graph

    async def _extract_entities(
        self,
        text: str,
        context: str
    ) -> List[str]:
        """
        Stage 1: Extract entities from text.

        Args:
            text: Input text
            context: Context information

        Returns:
            List of extracted entities
        """
        try:
            if self.kggen:
                # Use kg-gen's entity extraction
                result = await self._call_kggen_stage('entities', text, context)
                return result.get('entities', [])
            else:
                # Fallback: Simple entity extraction using regex/NER
                return await self._fallback_entity_extraction(text)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return await self._fallback_entity_extraction(text)

    async def _extract_relations(
        self,
        text: str,
        entities: List[str],
        context: str
    ) -> List[Tuple[str, str, str]]:
        """
        Stage 2: Extract relationships from text.

        Args:
            text: Input text
            entities: List of entities to find relationships for
            context: Context information

        Returns:
            List of (subject, predicate, object) triples
        """
        try:
            if self.kggen:
                # Use kg-gen's relation extraction
                result = await self._call_kggen_stage(
                    'relations',
                    text,
                    context,
                    entities=entities
                )
                return [
                    (r['subject'], r['predicate'], r['object'])
                    for r in result.get('relationships', [])
                ]
            else:
                # Fallback: Simple pattern-based extraction
                return await self._fallback_relation_extraction(text, entities)
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return await self._fallback_relation_extraction(text, entities)

    async def _deduplicate_graph(
        self,
        graph: KnowledgeGraph,
        method: str = 'full'
    ) -> KnowledgeGraph:
        """
        Stage 3: Deduplicate entities and relationships.

        Args:
            graph: Input knowledge graph
            method: Deduplication method ('semhash', 'lm_cluster', 'full')

        Returns:
            Deduplicated knowledge graph
        """
        logger.info(f"Starting deduplication with method: {method}")

        try:
            if method == 'semhash' or method == 'full':
                # Semantic hash-based deduplication
                graph = await self._semhash_deduplication(graph)

            if method == 'lm_cluster' or method == 'full':
                # LM-based clustering deduplication
                graph = await self._lm_cluster_deduplication(graph)

            return graph

        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return graph

    async def _semhash_deduplication(
        self,
        graph: KnowledgeGraph
    ) -> KnowledgeGraph:
        """
        Semantic hash-based deduplication.

        Args:
            graph: Input knowledge graph

        Returns:
            Deduplicated graph
        """
        threshold = self.pipeline_config['stages']['deduplication']['semhash_threshold']
        seen_hashes = {}
        unique_entities = []

        for entity in graph.entities:
            # Create semantic hash (simplified - in production use embeddings)
            entity_hash = self._create_entity_hash(entity)

            # Check for similar entities
            is_duplicate = False
            for seen_hash, seen_entity in seen_hashes.items():
                similarity = self._hash_similarity(entity_hash, seen_hash)
                if similarity > threshold:
                    is_duplicate = True
                    # Add to clusters
                    if seen_entity not in graph.entity_clusters:
                        graph.entity_clusters[seen_entity] = []
                    graph.entity_clusters[seen_entity].append(entity)
                    break

            if not is_duplicate:
                seen_hashes[entity_hash] = entity
                unique_entities.append(entity)

        graph.entities = unique_entities
        return graph

    async def _lm_cluster_deduplication(
        self,
        graph: KnowledgeGraph
    ) -> KnowledgeGraph:
        """
        Language model-based clustering deduplication.

        Args:
            graph: Input knowledge graph

        Returns:
            Deduplicated graph with clusters
        """
        # Simplified clustering - in production use actual embeddings
        # Group entities by string similarity
        cluster_size = self.pipeline_config['stages']['deduplication']['lm_cluster_size']

        for i, entity in enumerate(graph.entities):
            # Create cluster ID based on first few words
            words = entity.lower().split()[:3]
            cluster_id = '_'.join(words)

            if cluster_id not in graph.entity_clusters:
                graph.entity_clusters[cluster_id] = []

            if len(graph.entity_clusters[cluster_id]) < cluster_size:
                graph.entity_clusters[cluster_id].append(entity)

        return graph

    def _create_entity_hash(self, entity: str) -> str:
        """Create a hash for an entity."""
        # Simplified - in production use semantic embeddings
        normalized = entity.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes."""
        # Simplified - in production use cosine similarity of embeddings
        return 1.0 if hash1 == hash2 else 0.0

    async def _call_kggen_stage(
        self,
        stage: str,
        text: str,
        context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Call kg-gen for a specific stage."""
        # This would call the actual kg-gen API
        # For now, return mock results
        if stage == 'entities':
            return {'entities': await self._fallback_entity_extraction(text)}
        elif stage == 'relations':
            entities = kwargs.get('entities', [])
            return {
                'relationships': await self._fallback_relation_extraction(text, entities)
            }
        return {}

    async def _fallback_entity_extraction(self, text: str) -> List[str]:
        """Fallback entity extraction using simple patterns."""
        import re

        # Extract capitalized phrases as potential entities
        pattern = r'\b[A-Z][a-zA-Z]+\b'
        matches = re.findall(pattern, text)

        # Deduplicate
        unique_entities = list(set(matches))
        return unique_entities[:50]  # Limit to top 50

    async def _fallback_relation_extraction(
        self,
        text: str,
        entities: List[str]
    ) -> List[Tuple[str, str, str]]:
        """Fallback relation extraction using simple patterns."""
        import re

        relationships = []

        # Simple pattern: "Entity1 verb Entity2"
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:i+6]:  # Look at nearby entities
                # Look for sentences containing both entities
                pattern = f"{entity1}.*(is|has|was|contains|includes).*{entity2}"
                if re.search(pattern, text, re.IGNORECASE):
                    relationships.append((entity1, "related_to", entity2))

        return relationships[:100]  # Limit to top 100

    async def extract_from_large_document(
        self,
        document: str,
        chunk_size: int = 5000,
        parallel_chunks: int = 4
    ) -> KnowledgeGraph:
        """
        Process large documents with parallel chunking.

        Args:
            document: Large document text
            chunk_size: Size of each chunk
            parallel_chunks: Number of chunks to process in parallel

        Returns:
            Combined knowledge graph from all chunks
        """
        from .kggen_chunking import DocumentChunker
        from .kggen_parallel import ParallelChunkProcessor

        logger.info(f"Processing large document ({len(document)} chars)")

        # Chunk the document
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=200)
        chunks = chunker.chunk_document(document)
        logger.info(f"Split into {len(chunks)} chunks")

        # Process chunks in parallel
        processor = ParallelChunkProcessor(max_workers=parallel_chunks)
        results = await processor.process_chunks_parallel(
            chunks,
            lambda chunk: self.extract_knowledge_graph(chunk.text)
        )

        # Merge results
        combined_graph = KnowledgeGraph()
        for result in results:
            if result:
                combined_graph.merge(result)

        logger.info(
            f"Combined graph: {len(combined_graph.entities)} entities, "
            f"{len(combined_graph.relationships)} relationships"
        )

        return combined_graph

    async def upload_to_neo4j(
        self,
        graph: KnowledgeGraph,
        batch_size: int = 100
    ) -> UploadResult:
        """
        Upload knowledge graph to Neo4j.

        Args:
            graph: Knowledge graph to upload
            batch_size: Batch size for uploads

        Returns:
            UploadResult with upload statistics
        """
        if not self.neo4j:
            logger.warning("Neo4j backend not configured")
            return UploadResult(
                success=False,
                error="Neo4j backend not configured"
            )

        try:
            from .kggen_neo4j import Neo4jGraphUploader

            uploader = Neo4jGraphUploader(self.neo4j)
            result = await uploader.upload_graph(graph, batch_size)

            return result

        except Exception as e:
            logger.error(f"Neo4j upload failed: {e}")
            return UploadResult(success=False, error=str(e))

    async def extract_and_upload(
        self,
        text: str,
        context: str = "",
        upload_to_neo4j: bool = True
    ) -> KnowledgeGraph:
        """
        Complete pipeline: extract → dedup → upload.

        Args:
            text: Input text
            context: Optional context
            upload_to_neo4j: Whether to upload to Neo4j

        Returns:
            Extracted knowledge graph
        """
        # Extract knowledge graph
        graph = await self.extract_knowledge_graph(text, context)

        # Upload to Neo4j if requested
        if upload_to_neo4j:
            upload_result = await self.upload_to_neo4j(graph)
            if upload_result.success:
                logger.info(
                    f"Successfully uploaded {upload_result.entities_uploaded} entities "
                    f"and {upload_result.relationships_uploaded} relationships to Neo4j"
                )
            else:
                logger.error(f"Neo4j upload failed: {upload_result.error}")

        return graph

    @lru_cache(maxsize=100)
    async def extract_cached(self, text_hash: str) -> KnowledgeGraph:
        """
        Extract knowledge graph with caching.

        Args:
            text_hash: Hash of input text for cache key

        Returns:
            Cached knowledge graph
        """
        # This is a placeholder - actual implementation would deserialize
        logger.info(f"Cache hit for hash: {text_hash}")
        return KnowledgeGraph()

    async def extract_batch(
        self,
        texts: List[str]
    ) -> List[KnowledgeGraph]:
        """
        Extract knowledge graphs from multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            List of knowledge graphs
        """
        tasks = [self.extract_knowledge_graph(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("KGGenPipelineIntegration closed")
