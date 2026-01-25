"""
KG-Gen Pipeline Integration for OpenEvolve Knowledge Engine

This module provides the complete 3-stage knowledge graph generation pipeline
integrating kg-gen with OpenEvolve, including chunking, parallel processing,
deduplication, and Neo4j integration.

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
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import uuid
import yaml
import os

from .extraction_pipeline import ExtractionPipeline, KnowledgeGraph
from .chunking import DocumentChunker
from .parallel_processing import ParallelChunkProcessor
from .deduplication import DeduplicationEngine
from .neo4j_integration import Neo4jGraphUploader

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the KG-Gen pipeline."""
    enabled: bool = True
    default_chunk_size: int = 5000
    default_overlap: int = 200
    parallel_workers: int = 4
    
    # Stage configurations
    stages: Dict[str, Any] = field(default_factory=dict)
    
    # Neo4j upload configuration
    neo4j_upload: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    progress_tracking: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.stages:
            self.stages = {
                'entity_extraction': {
                    'model': os.getenv('KGGEN_ENTITY_MODEL', 'gpt-4o'),
                    'temperature': 0.0,
                    'max_tokens': 4000,
                    'prompt_template': None
                },
                'relation_extraction': {
                    'model': os.getenv('KGGEN_RELATION_MODEL', 'gpt-4o'),
                    'temperature': 0.0,
                    'max_tokens': 8000,
                    'extract_temporal': True,
                    'extract_attributes': True
                },
                'deduplication': {
                    'method': 'full',  # semhash, lm_cluster, full
                    'semhash_threshold': 0.95,
                    'lm_cluster_size': 128,
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
                }
            }
        
        if not self.neo4j_upload:
            self.neo4j_upload = {
                'enabled': True,
                'batch_size': 100,
                'create_indices': True,
                'verify_upload': True
            }
        
        if not self.progress_tracking:
            self.progress_tracking = {
                'enabled': True,
                'log_interval': 10,
                'save_intermediate': True
            }


class KGGenPipelineIntegration:
    """
    Complete KG-Gen Pipeline Integration with OpenEvolve.
    
    Provides methods for:
    - Complete 3-stage knowledge extraction
    - Document chunking and parallel processing
    - Advanced deduplication
    - Neo4j upload
    - Progress tracking
    """
    
    def __init__(
        self,
        kggen_client: Optional[ExtractionPipeline] = None,
        neo4j_backend = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the KG-Gen pipeline integration.
        
        Args:
            kggen_client: Pre-initialized ExtractionPipeline (optional)
            neo4j_backend: Neo4j driver instance (optional)
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.pipeline_config = self._load_config(config_path)
        
        # Initialize components
        self.kggen_client = kggen_client or ExtractionPipeline(
            model=self.pipeline_config.stages['entity_extraction']['model']
        )
        
        self.neo4j_backend = neo4j_backend
        self.chunker = DocumentChunker(
            chunk_size=self.pipeline_config.default_chunk_size,
            overlap=self.pipeline_config.default_overlap
        )
        self.parallel_processor = ParallelChunkProcessor(
            max_workers=self.pipeline_config.parallel_workers
        )
        self.deduplication_engine = DeduplicationEngine(
            config=self.pipeline_config.stages['deduplication']
        )
        
        # Neo4j uploader
        self.neo4j_uploader = None
        if self.neo4j_backend:
            self.neo4j_uploader = Neo4jGraphUploader(self.neo4j_backend)
        
        logger.info({
            "msg": "KG-Gen Pipeline Integration initialized",
            "config": {
                "enabled": self.pipeline_config.enabled,
                "chunk_size": self.pipeline_config.default_chunk_size,
                "workers": self.pipeline_config.parallel_workers
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _load_config(self, config_path: Optional[str]) -> PipelineConfig:
        """Load pipeline configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return PipelineConfig(**config_data)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
        
        return PipelineConfig()
    
    async def extract_knowledge_graph(
        self,
        text: str,
        context: str = "",
        dedup_method: str = 'full',
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from text using the complete pipeline.
        
        Args:
            text: Input text to extract knowledge from
            context: Context information for extraction
            dedup_method: Deduplication method to use
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeGraph with extracted entities and relationships
        """
        correlation_id = correlation_id or f"kg_extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting knowledge graph extraction",
            "text_length": len(text),
            "context": context,
            "dedup_method": dedup_method,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Use the underlying extraction pipeline
            result = await self.kggen_client.extract(
                text=text,
                context=context,
                correlation_id=correlation_id
            )
            
            if not result.get("success"):
                raise RuntimeError(f"Extraction failed: {result.get('error')}")
            
            # Create KnowledgeGraph from result
            graph = KnowledgeGraph(
                entities=result.get("entities", []),
                relations=result.get("relations", []),
                entity_clusters=result.get("entity_clusters", {})
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Knowledge graph extraction completed",
                "correlation_id": correlation_id,
                "entities_count": len(graph.entities),
                "relations_count": len(graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return graph
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge graph extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeGraph()
    
    async def extract_from_large_document(
        self,
        document: str,
        chunk_size: Optional[int] = None,
        parallel_chunks: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge from large documents with parallel chunking.
        
        Args:
            document: Large document text
            chunk_size: Size of chunks (uses default if None)
            parallel_chunks: Number of chunks to process in parallel (uses default if None)
            correlation_id: Correlation ID for tracking
            
        Returns:
            Combined KnowledgeGraph from all chunks
        """
        correlation_id = correlation_id or f"large_doc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Extracting from large document",
            "document_length": len(document),
            "chunk_size": chunk_size or self.pipeline_config.default_chunk_size,
            "parallel_chunks": parallel_chunks or self.pipeline_config.parallel_workers,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Use the underlying extraction pipeline's large document method
            graph = await self.kggen_client.extract_from_large_document(
                document=document,
                chunk_size=chunk_size or self.pipeline_config.default_chunk_size,
                overlap=self.pipeline_config.default_overlap,
                correlation_id=correlation_id
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Large document extraction completed",
                "correlation_id": correlation_id,
                "entities_count": len(graph.entities),
                "relations_count": len(graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return graph
            
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
    
    async def extract_and_upload(
        self,
        text: str,
        context: str = "",
        upload_to_neo4j: bool = True,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph and upload to Neo4j.
        
        Args:
            text: Input text to extract knowledge from
            context: Context information for extraction
            upload_to_neo4j: Whether to upload to Neo4j
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeGraph with extracted entities and relationships
        """
        correlation_id = correlation_id or f"extract_upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting extraction and upload",
            "text_length": len(text),
            "context": context,
            "upload_to_neo4j": upload_to_neo4j,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Extract knowledge graph
            graph = await self.extract_knowledge_graph(
                text=text,
                context=context,
                correlation_id=correlation_id
            )
            
            # Upload to Neo4j if requested and available
            if upload_to_neo4j and self.neo4j_uploader:
                upload_result = await self.neo4j_uploader.upload_graph(
                    graph=graph,
                    batch_size=self.pipeline_config.neo4j_upload.get('batch_size', 100),
                    correlation_id=correlation_id
                )
                
                logger.info({
                    "msg": "Neo4j upload completed",
                    "correlation_id": correlation_id,
                    "entities_uploaded": upload_result.get('entities_uploaded', 0),
                    "relationships_uploaded": upload_result.get('relationships_uploaded', 0),
                    "upload_success": upload_result.get('success', False)
                })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Extraction and upload completed",
                "correlation_id": correlation_id,
                "entities_count": len(graph.entities),
                "relations_count": len(graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return graph
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Extraction and upload failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeGraph()
    
    async def upload_to_neo4j(
        self,
        graph: KnowledgeGraph,
        batch_size: int = 100,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload knowledge graph to Neo4j.
        
        Args:
            graph: KnowledgeGraph to upload
            batch_size: Size of batches for uploading
            correlation_id: Correlation ID for tracking
            
        Returns:
            Upload result with success status and counts
        """
        if not self.neo4j_uploader:
            logger.error({
                "msg": "Neo4j uploader not available",
                "correlation_id": correlation_id
            })
            return {"success": False, "error": "Neo4j uploader not initialized"}
        
        correlation_id = correlation_id or f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Uploading knowledge graph to Neo4j",
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "batch_size": batch_size,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            result = await self.neo4j_uploader.upload_graph(
                graph=graph,
                batch_size=batch_size,
                correlation_id=correlation_id
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result["processing_time_ms"] = processing_time_ms
            
            logger.info({
                "msg": "Neo4j upload completed",
                "correlation_id": correlation_id,
                "entities_uploaded": result.get('entities_uploaded', 0),
                "relationships_uploaded": result.get('relationships_uploaded', 0),
                "success": result.get('success', False),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Neo4j upload failed",
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
    
    async def extract_batch(
        self,
        texts: List[str],
        correlation_id: Optional[str] = None
    ) -> List[KnowledgeGraph]:
        """
        Extract knowledge graphs from multiple texts in batch.
        
        Args:
            texts: List of input texts
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of KnowledgeGraph objects
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
            # Use the underlying extraction pipeline's batch method
            results = await self.kggen_client.extract_batch(
                texts=texts,
                correlation_id=correlation_id
            )
            
            # Convert results to KnowledgeGraph objects
            graphs = []
            for result in results:
                if result.get("success"):
                    graph = KnowledgeGraph(
                        entities=result.get("entities", []),
                        relations=result.get("relations", []),
                        entity_clusters=result.get("entity_clusters", {})
                    )
                    graphs.append(graph)
                else:
                    graphs.append(KnowledgeGraph())  # Empty graph for failed extractions
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Batch extraction completed",
                "correlation_id": correlation_id,
                "successful_extractions": len([g for g in graphs if g.entities or g.relations]),
                "total_extractions": len(graphs),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return graphs
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Batch extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return [KnowledgeGraph() for _ in texts]
    
    async def close(self):
        """Close resources used by the pipeline."""
        logger.info({
            "msg": "Closing KG-Gen pipeline resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close Neo4j connection if available
        if self.neo4j_backend:
            try:
                self.neo4j_backend.close()
                logger.info("Neo4j backend closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j backend: {e}")
        
        logger.info({
            "msg": "KG-Gen pipeline resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline."""
        return {
            "enabled": self.pipeline_config.enabled,
            "components": {
                "kggen_client": self.kggen_client is not None,
                "neo4j_backend": self.neo4j_backend is not None,
                "chunker": self.chunker is not None,
                "parallel_processor": self.parallel_processor is not None,
                "deduplication_engine": self.deduplication_engine is not None,
                "neo4j_uploader": self.neo4j_uploader is not None
            },
            "config": {
                "chunk_size": self.pipeline_config.default_chunk_size,
                "overlap": self.pipeline_config.default_overlap,
                "workers": self.pipeline_config.parallel_workers
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }