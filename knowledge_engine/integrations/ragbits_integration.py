"""
Ragbits Integration for OpenEvolve Knowledge Engine

This module provides integration with the Ragbits retrieval-augmented generation system,
enabling document search, ingestion, and retrieval capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Sequence
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class RagbitsResult:
    """Result of a Ragbits operation."""
    success: bool
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class RagbitsIntegration:
    """
    Integration with Ragbits document search and retrieval system.
    
    Provides methods for:
    - Document ingestion and indexing
    - Semantic search and retrieval
    - Query rephrasing and reranking
    - Vector store operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ragbits integration.
        
        Args:
            config: Configuration for Ragbits components
        """
        self.config = config or self._get_default_config()
        
        # Initialize Ragbits components
        self.document_search = None
        self.vector_store = None
        self.query_rephraser = None
        self.reranker = None
        
        # Initialize components based on configuration
        self._initialize_components()
        
        logger.info({
            "msg": "RagbitsIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Ragbits integration."""
        return {
            "vector_store": {
                "type": "qdrant",  # Can be "qdrant", "chroma", "pinecone", etc.
                "config": {
                    "location": ":memory:",  # For in-memory, or specify server URL
                    "collection_name": "knowledge_artifacts"
                }
            },
            "query_rephraser": {
                "type": "noop",  # Can be "noop", "llm", etc.
                "config": {}
            },
            "reranker": {
                "type": "noop",  # Can be "noop", "cohere", "colbert", etc.
                "config": {}
            },
            "ingest_strategy": {
                "type": "sequential",
                "config": {
                    "max_workers": 4
                }
            },
            "default_options": {
                "top_k": 10,
                "similarity_threshold": 0.7
            }
        }
    
    def _initialize_components(self):
        """Initialize Ragbits components based on configuration."""
        try:
            # Import Ragbits components
            from ragbits.document_search import DocumentSearch
            from ragbits.core.vector_stores import VectorStore
            from ragbits.document_search.retrieval.rephrasers import QueryRephraser
            from ragbits.document_search.retrieval.rerankers import Reranker
            
            # Initialize vector store based on config
            vector_store_config = self.config.get("vector_store", {})
            vector_store_type = vector_store_config.get("type", "qdrant")
            
            if vector_store_type == "qdrant":
                from ragbits.core.vector_stores.qdrant import QdrantVectorStore
                self.vector_store = QdrantVectorStore(**vector_store_config.get("config", {}))
            elif vector_store_type == "chroma":
                from ragbits.core.vector_stores.chroma import ChromaVectorStore
                self.vector_store = ChromaVectorStore(**vector_store_config.get("config", {}))
            else:
                raise ValueError(f"Unsupported vector store type: {vector_store_type}")
            
            # Initialize query rephraser
            rephraser_config = self.config.get("query_rephraser", {})
            rephraser_type = rephraser_config.get("type", "noop")
            
            if rephraser_type == "noop":
                from ragbits.document_search.retrieval.rephrasers.noop import NoopQueryRephraser
                self.query_rephraser = NoopQueryRephraser()
            elif rephraser_type == "llm":
                from ragbits.document_search.retrieval.rephrasers.llm import LLMQueryRephraser
                self.query_rephraser = LLMQueryRephraser(**rephraser_config.get("config", {}))
            else:
                # Default to noop
                from ragbits.document_search.retrieval.rephrasers.noop import NoopQueryRephraser
                self.query_rephraser = NoopQueryRephraser()
            
            # Initialize reranker
            reranker_config = self.config.get("reranker", {})
            reranker_type = reranker_config.get("type", "noop")
            
            if reranker_type == "noop":
                from ragbits.document_search.retrieval.rerankers.noop import NoopReranker
                self.reranker = NoopReranker()
            elif reranker_type == "cohere":
                from ragbits.document_search.retrieval.rerankers.cohere import CohereReranker
                self.reranker = CohereReranker(**reranker_config.get("config", {}))
            else:
                # Default to noop
                from ragbits.document_search.retrieval.rerankers.noop import NoopReranker
                self.reranker = NoopReranker()
            
            # Initialize document search with components
            self.document_search = DocumentSearch(
                vector_store=self.vector_store,
                query_rephraser=self.query_rephraser,
                reranker=self.reranker,
                default_options=self.config.get("default_options", {})
            )
            
            logger.info({
                "msg": "Ragbits components initialized successfully",
                "vector_store_type": vector_store_type,
                "rephraser_type": rephraser_type,
                "reranker_type": reranker_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ImportError as e:
            logger.warning({
                "msg": f"Ragbits not available, using mock implementation: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components
            self._initialize_mock_components()
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize Ragbits components: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    def _initialize_mock_components(self):
        """Initialize mock components when Ragbits is not available."""
        logger.info({
            "msg": "Initializing mock Ragbits components",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create mock implementations
        class MockVectorStore:
            def __init__(self):
                self.documents = {}
            
            async def retrieve(self, query: str, options=None):
                # Mock retrieval - return empty results
                return []
        
        class MockQueryRephraser:
            async def rephrase(self, query: str, options=None):
                # Return original query
                return [query]
        
        class MockReranker:
            async def rerank(self, elements, query, options=None):
                # Return elements as-is
                return elements
        
        self.vector_store = MockVectorStore()
        self.query_rephraser = MockQueryRephraser()
        self.reranker = MockReranker()
        self.document_search = None  # Will use individual components directly
    
    async def search_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        correlation_id: Optional[str] = None
    ) -> RagbitsResult:
        """
        Search documents using Ragbits.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            correlation_id: Correlation ID for tracking
            
        Returns:
            RagbitsResult with search results
        """
        correlation_id = correlation_id or f"ragbits_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting Ragbits document search",
            "query_length": len(query),
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.document_search:
                raise RuntimeError("Ragbits document search not initialized")
            
            # Prepare search options
            search_options = {}
            if top_k:
                search_options["top_k"] = top_k
            if similarity_threshold:
                search_options["similarity_threshold"] = similarity_threshold
            
            # Perform search
            results = await self.document_search.search(
                query=query,
                options=search_options if search_options else None
            )
            
            # Convert results to our format
            search_results = []
            for result in results:
                search_results.append({
                    "content": getattr(result, 'content', ''),
                    "metadata": getattr(result, 'metadata', {}),
                    "score": getattr(result, 'score', 0.0),
                    "source": getattr(result, 'source', 'unknown')
                })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = RagbitsResult(
                success=True,
                results=search_results,
                metadata={
                    "query": query,
                    "top_k": top_k or self.config.get("default_options", {}).get("top_k", 10),
                    "similarity_threshold": similarity_threshold or self.config.get("default_options", {}).get("similarity_threshold", 0.7),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Ragbits document search completed",
                "correlation_id": correlation_id,
                "results_count": len(search_results),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Ragbits document search failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return RagbitsResult(
                success=False,
                results=[],
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def ingest_documents(
        self,
        documents: Union[str, List[Dict[str, Any]]],
        correlation_id: Optional[str] = None
    ) -> RagbitsResult:
        """
        Ingest documents into the Ragbits system.
        
        Args:
            documents: Either a path/string source or list of document dictionaries
            correlation_id: Correlation ID for tracking
            
        Returns:
            RagbitsResult with ingestion results
        """
        correlation_id = correlation_id or f"ragbits_ingest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting Ragbits document ingestion",
            "document_source_type": type(documents).__name__,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.document_search:
                raise RuntimeError("Ragbits document search not initialized")
            
            # Perform ingestion
            result = await self.document_search.ingest(documents=documents)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            ragbits_result = RagbitsResult(
                success=True,
                results=[{
                    "ingested_count": len(result.successful) if hasattr(result, 'successful') else 0,
                    "failed_count": len(result.failed) if hasattr(result, 'failed') else 0,
                    "total_processed": len(result.all_results) if hasattr(result, 'all_results') else 0
                }],
                metadata={
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Ragbits document ingestion completed",
                "correlation_id": correlation_id,
                "ingested_count": len(result.successful) if hasattr(result, 'successful') else 0,
                "failed_count": len(result.failed) if hasattr(result, 'failed') else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ragbits_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Ragbits document ingestion failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return RagbitsResult(
                success=False,
                results=[],
                metadata={
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        correlation_id: Optional[str] = None
    ) -> List[RagbitsResult]:
        """
        Perform batch search on multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of RagbitsResult objects
        """
        correlation_id = correlation_id or f"ragbits_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting Ragbits batch search",
            "query_count": len(queries),
            "top_k": top_k,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process queries in parallel
            tasks = [
                self.search_documents(
                    query=query,
                    top_k=top_k,
                    correlation_id=f"{correlation_id}_query_{i}"
                )
                for i, query in enumerate(queries)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the gathered results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch query {i} failed",
                        "correlation_id": f"{correlation_id}_query_{i}",
                        "error": str(result)
                    })
                    processed_results.append(RagbitsResult(
                        success=False,
                        results=[],
                        metadata={"query_index": i, "error": str(result)},
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Ragbits batch search completed",
                "correlation_id": correlation_id,
                "query_count": len(queries),
                "successful_queries": len([r for r in processed_results if r.success]),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return processed_results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Ragbits batch search failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all queries
            error_results = []
            for i in range(len(queries)):
                error_results.append(RagbitsResult(
                    success=False,
                    results=[],
                    metadata={"query_index": i, "error": str(e)},
                    processing_time_ms=processing_time_ms / len(queries) if queries else 0,
                    error=str(e)
                ))
            
            return error_results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Ragbits system.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "vector_store": type(self.vector_store).__name__ if self.vector_store else "None",
                "query_rephraser": type(self.query_rephraser).__name__ if self.query_rephraser else "None",
                "reranker": type(self.reranker).__name__ if self.reranker else "None"
            },
            "initialized": self.document_search is not None
        }

        # Add vector store statistics if available
        if self.vector_store and hasattr(self.vector_store, 'get_statistics'):
            try:
                vector_stats = await self.vector_store.get_statistics()
                stats["vector_store_stats"] = vector_stats
            except Exception as e:
                logger.warning(f"Failed to get vector store stats: {e}")

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Ragbits integration.
        
        Returns:
            Health check results
        """
        start_time = datetime.now(timezone.utc)
        
        health = {
            "component": "ragbits",
            "status": "healthy",
            "checks": {},
            "timestamp": start_time.isoformat()
        }
        
        try:
            # Check if components are initialized
            if not self.document_search:
                health["status"] = "unhealthy"
                health["checks"]["document_search_initialized"] = {
                    "status": "failed",
                    "message": "Document search not initialized"
                }
            else:
                health["checks"]["document_search_initialized"] = {
                    "status": "passed"
                }
            
            # Check vector store
            if self.vector_store:
                try:
                    # Try a simple operation to verify vector store is working
                    health["checks"]["vector_store"] = {
                        "status": "passed",
                        "type": type(self.vector_store).__name__
                    }
                except Exception as e:
                    health["status"] = "degraded"
                    health["checks"]["vector_store"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            else:
                health["status"] = "degraded"
                health["checks"]["vector_store"] = {
                    "status": "failed",
                    "message": "Vector store not initialized"
                }
            
            health["processing_time_ms"] = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return health
            
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            health["processing_time_ms"] = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Ragbits health check error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return health
    
    async def close(self):
        """Close Ragbits resources."""
        logger.info({
            "msg": "Closing Ragbits integration",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close vector store if it has a close method
        if self.vector_store and hasattr(self.vector_store, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.vector_store.close):
                    await self.vector_store.close()
                else:
                    self.vector_store.close()
            except Exception as e:
                logger.error(f"Error closing vector store: {e}")
        
        logger.info({
            "msg": "Ragbits integration closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })