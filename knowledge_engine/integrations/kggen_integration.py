"""
KG-Gen Integration for OpenEvolve Knowledge Engine

This module provides integration with the KG-Gen knowledge extraction pipeline,
enabling entity and relationship extraction with advanced deduplication.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
import json
from pathlib import Path

try:
    from kg_gen.kg_gen import KGGen
    from kg_gen.models import Graph
    from kg_gen.steps._3_deduplicate import DeduplicateMethod
    KG_GEN_AVAILABLE = True
except ImportError:
    KG_GEN_AVAILABLE = False
    # Define mock classes for when kg-gen is not available
    class KGGen:
        pass
    class Graph:
        def __init__(self, entities=None, relations=None, edges=None):
            self.entities = entities or set()
            self.relations = relations or set()
            self.edges = edges or set()
    class DeduplicateMethod:
        SEMHASH = "semhash"
        LM_CLUSTER = "lm_cluster"
        FULL = "full"

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraph:
    """Representation of a knowledge graph with entities and relationships."""
    entities: List[str] = None
    relations: List[Tuple[str, str, str]] = None
    entity_clusters: Dict[str, List[str]] = None  # canonical -> [duplicates]
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relations is None:
            self.relations = []
        if self.entity_clusters is None:
            self.entity_clusters = {}
    
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


class KGGenIntegration:
    """
    Integration with KG-Gen knowledge extraction pipeline.
    
    Provides methods for:
    - Entity extraction using DSPy
    - Relation extraction
    - Entity deduplication
    - Graph construction
    - Batch processing
    """
    
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the KG-Gen integration.
        
        Args:
            model: LLM model to use for extraction
            max_tokens: Maximum tokens for model
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: API base for model access
            config: Additional configuration options
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.config = config or self._get_default_config()
        
        # Initialize KG-Gen if available
        self.kggen_client = None
        self._initialize_kggen()
        
        logger.info({
            "msg": "KGGenIntegration initialized",
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kggen_available": KG_GEN_AVAILABLE,
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
    
    def _initialize_kggen(self):
        """Initialize the KG-Gen client."""
        if not KG_GEN_AVAILABLE:
            logger.warning("KG-Gen not available, using mock implementation")
            return
        
        try:
            self.kggen_client = KGGen(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base
            )
            logger.info("KG-Gen client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KG-Gen client: {e}")
            self.kggen_client = None
    
    async def extract_knowledge_graph(
        self,
        text: str,
        context: str = "",
        deduplication_method: str = 'FULL',
        chunk_size: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from text using KG-Gen.
        
        Args:
            text: Input text to extract knowledge from
            context: Context information for extraction
            deduplication_method: Method for deduplication ('SEMHASH', 'LM_CLUSTER', 'FULL')
            chunk_size: Size of text chunks for processing
            correlation_id: Correlation ID for tracking
            
        Returns:
            KnowledgeGraph with extracted entities and relationships
        """
        correlation_id = correlation_id or f"kggen_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting KG-Gen extraction",
            "text_length": len(text),
            "context": context,
            "deduplication_method": deduplication_method,
            "chunk_size": chunk_size,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.kggen_client:
                raise RuntimeError("KG-Gen client not available")
            
            # Map deduplication method to KG-Gen enum
            method_map = {
                'SEMHASH': DeduplicateMethod.SEMHASH,
                'LM_CLUSTER': DeduplicateMethod.LM_CLUSTER,
                'FULL': DeduplicateMethod.FULL
            }
            dedup_method = method_map.get(deduplication_method.upper(), DeduplicateMethod.FULL)
            
            # Extract knowledge graph using KG-Gen
            graph = self.kggen_client.generate(
                input_data=text,
                context=context,
                chunk_size=chunk_size or self.config.get('chunk_size'),
                deduplication_method=dedup_method
            )
            
            # Convert KG-Gen Graph to our KnowledgeGraph format
            knowledge_graph = self._convert_kggen_to_knowledge_graph(graph)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "KG-Gen extraction completed",
                "correlation_id": correlation_id,
                "entities_count": len(knowledge_graph.entities),
                "relations_count": len(knowledge_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return knowledge_graph
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "KG-Gen extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return KnowledgeGraph()
    
    def _convert_kggen_to_knowledge_graph(self, kggen_graph: Graph) -> KnowledgeGraph:
        """Convert KG-Gen Graph to our KnowledgeGraph format."""
        entities = list(kggen_graph.entities) if kggen_graph.entities else []
        
        # Convert relations from KG-Gen format to our format
        relations = []
        if kggen_graph.relations:
            for rel in kggen_graph.relations:
                if isinstance(rel, tuple) and len(rel) >= 3:
                    relations.append((rel[0], rel[1], rel[2]))
                elif isinstance(rel, str):
                    # If it's a string, try to parse it
                    parts = rel.split()
                    if len(parts) >= 3:
                        relations.append((parts[0], parts[1], parts[2]))
        
        # Get entity clusters if available
        entity_clusters = {}
        if hasattr(kggen_graph, 'entity_clusters') and kggen_graph.entity_clusters:
            entity_clusters = {k: list(v) for k, v in kggen_graph.entity_clusters.items()}
        
        return KnowledgeGraph(
            entities=entities,
            relations=relations,
            entity_clusters=entity_clusters
        )
    
    async def extract_from_large_document(
        self,
        document: str,
        chunk_size: int = 5000,
        overlap: int = 200,
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Extract knowledge from large documents by chunking and processing.
        
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
            "msg": "Extracting from large document with KG-Gen",
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
                "msg": "Document chunked for processing",
                "chunk_count": len(chunks),
                "correlation_id": correlation_id
            })
            
            # Process each chunk and collect results
            combined_graph = KnowledgeGraph()
            successful_extractions = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_graph = await self.extract_knowledge_graph(
                        text=chunk,
                        context=f"Chunk {i+1} of {len(chunks)}",
                        correlation_id=f"{correlation_id}_chunk_{i}"
                    )
                    
                    # Merge the chunk graph into the combined graph
                    combined_graph.entities.extend(chunk_graph.entities)
                    combined_graph.relations.extend(chunk_graph.relations)
                    
                    # Merge entity clusters
                    for canonical, duplicates in chunk_graph.entity_clusters.items():
                        if canonical in combined_graph.entity_clusters:
                            combined_graph.entity_clusters[canonical].extend(duplicates)
                        else:
                            combined_graph.entity_clusters[canonical] = duplicates
                    
                    successful_extractions += 1
                    
                except Exception as e:
                    logger.warning({
                        "msg": f"Chunk {i} extraction failed",
                        "error": str(e),
                        "correlation_id": f"{correlation_id}_chunk_{i}"
                    })
                    continue
            
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
            "msg": "Starting batch extraction with KG-Gen",
            "text_count": len(texts),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process all texts in parallel using asyncio
            tasks = []
            for i, text in enumerate(texts):
                task = self.extract_knowledge_graph(
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
                    processed_results.append(KnowledgeGraph())
                else:
                    processed_results.append(result)
                    if result.entities or result.relations:
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
            
            return [KnowledgeGraph() for _ in texts]
    
    async def deduplicate_graph(
        self,
        graph: KnowledgeGraph,
        method: str = 'FULL',
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Apply deduplication to a knowledge graph using KG-Gen's capabilities.
        
        Args:
            graph: KnowledgeGraph to deduplicate
            method: Deduplication method ('SEMHASH', 'LM_CLUSTER', 'FULL')
            correlation_id: Correlation ID for tracking
            
        Returns:
            Deduplicated KnowledgeGraph
        """
        correlation_id = correlation_id or f"dedup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting graph deduplication with KG-Gen",
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "method": method,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.kggen_client:
                raise RuntimeError("KG-Gen client not available")
            
            # Convert our KnowledgeGraph to KG-Gen Graph format
            kggen_graph = Graph(
                entities=set(graph.entities),
                relations=set(graph.relations),
                edges=set([rel[1] for rel in graph.relations if len(rel) >= 3])  # relation type
            )
            
            # Map deduplication method
            method_map = {
                'SEMHASH': DeduplicateMethod.SEMHASH,
                'LM_CLUSTER': DeduplicateMethod.LM_CLUSTER,
                'FULL': DeduplicateMethod.FULL
            }
            dedup_method = method_map.get(method.upper(), DeduplicateMethod.FULL)
            
            # Apply deduplication using KG-Gen
            deduplicated_graph = self.kggen_client.deduplicate(
                graph=kggen_graph,
                method=dedup_method
            )
            
            # Convert back to our KnowledgeGraph format
            result_graph = self._convert_kggen_to_knowledge_graph(deduplicated_graph)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Graph deduplication completed",
                "correlation_id": correlation_id,
                "original_entities": len(graph.entities),
                "deduplicated_entities": len(result_graph.entities),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result_graph
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Graph deduplication failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return graph  # Return original graph if deduplication fails
    
    async def aggregate_graphs(
        self,
        graphs: List[KnowledgeGraph],
        correlation_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Aggregate multiple knowledge graphs into a single graph.
        
        Args:
            graphs: List of KnowledgeGraph objects to aggregate
            correlation_id: Correlation ID for tracking
            
        Returns:
            Aggregated KnowledgeGraph
        """
        correlation_id = correlation_id or f"aggregate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting graph aggregation",
            "graph_count": len(graphs),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.kggen_client:
                raise RuntimeError("KG-Gen client not available")
            
            # Convert our KnowledgeGraphs to KG-Gen Graph format
            kggen_graphs = []
            for graph in graphs:
                kggen_graph = Graph(
                    entities=set(graph.entities),
                    relations=set(graph.relations),
                    edges=set([rel[1] for rel in graph.relations if len(rel) >= 3])
                )
                kggen_graphs.append(kggen_graph)
            
            # Aggregate using KG-Gen
            aggregated_graph = self.kggen_client.aggregate(kggen_graphs)
            
            # Convert back to our KnowledgeGraph format
            result_graph = self._convert_kggen_to_knowledge_graph(aggregated_graph)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Graph aggregation completed",
                "correlation_id": correlation_id,
                "original_graphs": len(graphs),
                "aggregated_entities": len(result_graph.entities),
                "aggregated_relations": len(result_graph.relations),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result_graph
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Graph aggregation failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return empty graph if aggregation fails
            return KnowledgeGraph()
    
    async def export_graph(
        self,
        graph: KnowledgeGraph,
        output_path: str,
        format: str = 'json'
    ) -> bool:
        """
        Export knowledge graph to file.
        
        Args:
            graph: KnowledgeGraph to export
            output_path: Path for output file
            format: Export format ('json', 'graphml', 'gexf')
            
        Returns:
            True if export successful
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting graph export",
            "output_path": output_path,
            "format": format,
            "entities_count": len(graph.entities),
            "relations_count": len(graph.relations),
            "timestamp": start_time.isoformat()
        })
        
        try:
            if format.lower() == 'json':
                # Export as JSON
                graph_data = {
                    "entities": graph.entities,
                    "relations": graph.relations,
                    "entity_clusters": graph.entity_clusters,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "export_format": "json"
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2)
                
                logger.info({
                    "msg": "Graph exported to JSON successfully",
                    "output_path": output_path,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return True
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error({
                "msg": "Graph export failed",
                "output_path": output_path,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return False
    
    def get_kggen_status(self) -> Dict[str, Any]:
        """
        Get the status of the KG-Gen integration.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "available": KG_GEN_AVAILABLE,
            "client_initialized": self.kggen_client is not None,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing KG-Gen integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # No specific cleanup needed for KG-Gen at the moment
        logger.info({
            "msg": "KG-Gen integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })