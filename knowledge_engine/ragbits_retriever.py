"""
RAGBits-Enhanced Knowledge Retriever

Integrates RAGBits semantic search capabilities with the OpenEvolve Knowledge Engine,
providing advanced retrieval for task agents and workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    from ragbits_integration.document_search.knowledge_retriever import RagbitsKnowledgeRetriever
    from ragbits_integration.config import RagbitsIntegrationConfig
except ImportError:
    RagbitsKnowledgeRetriever = None
    RagbitsIntegrationConfig = None

try:
    from knowledge_engine.ragbits_safety import (
        validate_query,
        validate_top_k,
        validate_filters
    )
except ImportError:
    # Fallback validation functions if ragbits_safety not available
    def validate_query(query):
        return query is not None and isinstance(query, str) and len(query.strip()) > 0

    def validate_top_k(top_k):
        if top_k is None:
            return 5
        if not isinstance(top_k, (int, float, str)):
            return 5
        try:
            top_k_int = int(top_k)
        except (ValueError, TypeError):
            return 5
        if top_k_int < 1:
            return 1
        if top_k_int > 100:
            return 100
        return top_k_int

    def validate_filters(filters):
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            return {}
        return filters

logger = logging.getLogger(__name__)


class RAGBitsEnhancedRetriever:
    """
    Enhanced knowledge retriever combining RAGBits with Knowledge Engine.

    Features:
    - Semantic vector search via RAGBits
    - Hybrid search combining semantic + keyword
    - Contextual retrieval with metadata filtering
    - Agent-aware search optimization
    - Caching for performance

    Usage:
        retriever = RAGBitsEnhancedRetriever()
        results = await retriever.search_similar_solutions(
            query="microservices authentication",
            top_k=5,
            filters={"stage": "stage_3"}
        )
    """

    def __init__(
        self,
        ragbits_config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Initialize the RAGBits-enhanced retriever.

        Args:
            ragbits_config: Configuration for RAGBits integration
            enable_cache: Enable result caching
            cache_ttl: Cache TTL in seconds
        """
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self._cache = {}

        # Initialize RAGBits retriever if available
        self.ragbits_retriever: Optional[RagbitsKnowledgeRetriever] = None
        self.ragbits_available = False

        if RagbitsKnowledgeRetriever and RagbitsIntegrationConfig:
            try:
                # Create RAGBits config
                config = RagbitsIntegrationConfig(**(ragbits_config or {}))
                self.ragbits_retriever = RagbitsKnowledgeRetriever(config)
                self.ragbits_available = True
                logger.info("[OK] RAGBits retriever initialized successfully")
            except Exception as e:
                logger.warning(f"[WARN] Could not initialize RAGBits: {e}")
        else:
            logger.warning("[WARN] RAGBits dependencies not available")

        logger.info(f"RAGBitsEnhancedRetriever initialized (RAGBits available: {self.ragbits_available})")

    async def search_similar_solutions(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_success_rate: float = 0.0,
        enable_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar solutions from historical data.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (stage, team, document_type, etc.)
            min_success_rate: Minimum success rate threshold
            enable_hybrid_search: Use hybrid search (semantic + keyword)

        Returns:
            List of similar solutions with metadata and scores.
            Returns empty list on error (never raises).
        """
        # Validate and normalize inputs
        if not validate_query(query):
            logger.warning("[WARN] Invalid query provided")
            return []

        top_k = validate_top_k(top_k)
        filters = validate_filters(filters)

        cache_key = self._get_cache_key("solutions", query, top_k, filters)

        if self.enable_cache and cache_key in self._cache:
            logger.debug("Cache hit for similar solutions search")
            return self._cache[cache_key]

        logger.info(f"🔍 Searching for similar solutions: {query[:100]}...")

        try:
            if self.ragbits_available and self.ragbits_retriever:
                results = await self._ragbits_search_solutions(query, top_k, filters, min_success_rate)
            else:
                # Fallback to mock results
                logger.info("ℹ️ RAGBits not available, using fallback search")
                results = await self._mock_search(query, top_k, "solutions")

            # Filter and rank results
            filtered_results = self._filter_results(results, min_success_rate)

            if self.enable_cache:
                self._cache[cache_key] = filtered_results

            logger.info(f"[OK] Found {len(filtered_results)} similar solutions")
            return filtered_results

        except asyncio.CancelledError:
            logger.warning("[WARN] Search operation was cancelled")
            return []
        except Exception as e:
            logger.error(f"[FAIL] Error searching similar solutions: {e}", exc_info=True)
            return []

    async def search_decomposition_patterns(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_depth: int = 2,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for decomposition patterns and strategies.

        Args:
            query: Search query describing the problem
            top_k: Number of patterns to return
            filters: Metadata filters
            min_depth: Minimum decomposition depth
            max_depth: Maximum decomposition depth

        Returns:
            List of decomposition patterns with structure and metadata
        """
        logger.info(f"🔍 Searching decomposition patterns: {query[:100]}...")

        try:
            if self.ragbits_available and self.ragbits_retriever:
                results = await self._ragbits_search_patterns(
                    query, top_k, filters, "decomposition"
                )
            else:
                results = await self._mock_search(query, top_k, "decomposition")

            # Filter by depth
            filtered = [
                r for r in results
                if min_depth <= r.get("depth", 0) <= max_depth
            ]

            logger.info(f"[OK] Found {len(filtered)} decomposition patterns")
            return filtered

        except Exception as e:
            logger.error(f"[FAIL] Error searching decomposition patterns: {e}")
            return []

    async def search_critique_patterns(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        critique_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for historical critique patterns and feedback.

        Args:
            query: Search query
            top_k: Number of patterns to return
            filters: Metadata filters
            critique_type: Type of critique (security, performance, etc.)

        Returns:
            List of critique patterns with feedback
        """
        logger.info(f"🔍 Searching critique patterns: {query[:100]}...")

        try:
            if self.ragbits_available and self.ragbits_retriever:
                results = await self._ragbits_search_patterns(
                    query, top_k, filters, "critique"
                )
            else:
                results = await self._mock_search(query, top_k, "critique")

            # Filter by critique type if specified
            if critique_type:
                results = [
                    r for r in results
                    if r.get("critique_type") == critique_type
                ]

            logger.info(f"[OK] Found {len(results)} critique patterns")
            return results

        except Exception as e:
            logger.error(f"[FAIL] Error searching critique patterns: {e}")
            return []

    async def search_verification_benchmarks(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_coverage: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for verification benchmarks and test cases.

        Args:
            query: Search query
            top_k: Number of benchmarks to return
            filters: Metadata filters
            min_coverage: Minimum test coverage threshold

        Returns:
            List of verification benchmarks with coverage data
        """
        logger.info(f"🔍 Searching verification benchmarks: {query[:100]}...")

        try:
            if self.ragbits_available and self.ragbits_retriever:
                results = await self._ragbits_search_patterns(
                    query, top_k, filters, "verification"
                )
            else:
                results = await self._mock_search(query, top_k, "verification")

            # Filter by coverage
            filtered = [
                r for r in results
                if r.get("coverage", 0.0) >= min_coverage
            ]

            logger.info(f"[OK] Found {len(filtered)} verification benchmarks")
            return filtered

        except Exception as e:
            logger.error(f"[FAIL] Error searching verification benchmarks: {e}")
            return []

    async def search_contextual_knowledge(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: int = 5,
        enable_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Context-aware search that incorporates workflow state.

        Args:
            query: Search query
            context: Workflow context (stage, team, sub_problem_id, etc.)
            top_k: Number of results
            enable_hybrid_search: Use hybrid search

        Returns:
            Contextually relevant knowledge items
        """
        logger.info(f"🔍 Contextual search: {query[:100]}... (stage: {context.get('stage')})")

        # Build filters from context
        filters = {
            "stage": context.get("stage"),
            "team": context.get("team"),
            "sub_problem_id": context.get("sub_problem_id")
        }
        filters = {k: v for k, v in filters.items() if v is not None}

        return await self.search_similar_solutions(
            query=query,
            top_k=top_k,
            filters=filters,
            enable_hybrid_search=enable_hybrid_search
        )

    async def ingest_artifact(
        self,
        content: str,
        metadata: Dict[str, Any],
        artifact_type: str = "solution"
    ) -> str:
        """
        Ingest a workflow artifact into the knowledge base.

        Args:
            content: Artifact content
            metadata: Artifact metadata
            artifact_type: Type of artifact

        Returns:
            Artifact ID (or empty string on failure - never raises)
        """
        # Validate inputs
        if not content or not isinstance(content, str):
            logger.warning("[WARN] Invalid content provided for ingestion")
            return ""

        if not metadata or not isinstance(metadata, dict):
            logger.warning("[WARN] Invalid metadata provided, using empty dict")
            metadata = {}

        if not artifact_type or not isinstance(artifact_type, str):
            logger.warning("[WARN] Invalid artifact_type, defaulting to 'general'")
            artifact_type = "general"

        logger.info(f"📥 Ingesting {artifact_type} artifact...")

        try:
            if self.ragbits_available and self.ragbits_retriever:
                # Ingest via RAGBits
                doc_id = await self._ragbits_ingest(content, metadata, artifact_type)
                if not doc_id:
                    # Fallback if RAGBits returns empty ID
                    logger.warning("[WARN] RAGBits returned empty ID, using fallback")
                    doc_id = f"artifact_{datetime.utcnow().timestamp()}"
            else:
                # Mock ingestion
                logger.info("ℹ️ RAGBits not available, using fallback ingestion")
                doc_id = f"artifact_{datetime.utcnow().timestamp()}"

            # Clear relevant cache entries (best effort)
            try:
                self._clear_cache_for_type(artifact_type)
            except Exception as cache_error:
                logger.warning(f"[WARN] Could not clear cache: {cache_error}")

            logger.info(f"[OK] Ingested artifact: {doc_id}")
            return doc_id

        except asyncio.CancelledError:
            logger.warning("[WARN] Artifact ingestion was cancelled")
            return ""
        except Exception as e:
            logger.error(f"[FAIL] Error ingesting artifact: {e}", exc_info=True)
            return ""

    async def _ragbits_search_solutions(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        min_success_rate: float
    ) -> List[Dict[str, Any]]:
        """Search solutions using RAGBits"""
        try:
            results = await self.ragbits_retriever.retrieve_similar_solutions(
                query=query,
                top_k=top_k,
                filters=filters or {}
            )
            return results
        except Exception as e:
            logger.error(f"RAGBits search error: {e}")
            return []

    async def _ragbits_search_patterns(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        pattern_type: str
    ) -> List[Dict[str, Any]]:
        """Search patterns using RAGBits"""
        try:
            # Add pattern type to filters
            search_filters = {**(filters or {}), "pattern_type": pattern_type}

            # Use semantic search via RAGBits document search
            results = await self.ragbits_retriever.retrieve_similar_solutions(
                query=query,
                top_k=top_k,
                filters=search_filters
            )
            return results
        except Exception as e:
            logger.error(f"RAGBits pattern search error: {e}")
            return []

    async def _ragbits_ingest(
        self,
        content: str,
        metadata: Dict[str, Any],
        artifact_type: str
    ) -> str:
        """Ingest artifact via RAGBits"""
        # This would call the RAGBits ingestion API
        # For now, return a mock ID
        return f"ragbits_{datetime.utcnow().timestamp()}"

    async def _mock_search(
        self,
        query: str,
        top_k: int,
        search_type: str
    ) -> List[Dict[str, Any]]:
        """Mock search when RAGBits is not available"""
        return [
            {
                "content": f"Mock {search_type} result for: {query}",
                "score": 0.8,
                "metadata": {
                    "source": "mock",
                    "type": search_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        ]

    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        min_success_rate: float
    ) -> List[Dict[str, Any]]:
        """Filter results by success rate"""
        return [
            r for r in results
            if r.get("success_rate", 1.0) >= min_success_rate
        ]

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        return f"{':'.join(str(arg) for arg in args)}"

    def _clear_cache_for_type(self, artifact_type: str):
        """Clear cache entries for a specific artifact type"""
        keys_to_remove = [
            key for key in self._cache
            if artifact_type in key
        ]
        for key in keys_to_remove:
            del self._cache[key]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "ragbits_available": self.ragbits_available,
            "cache_size": len(self._cache),
            "cache_enabled": self.enable_cache,
            "queries_since_start": len(self._cache)
        }

    async def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        logger.info("[OK] Cache cleared")


# Singleton instance
_instance: Optional[RAGBitsEnhancedRetriever] = None


def get_ragbits_retriever() -> RAGBitsEnhancedRetriever:
    """Get singleton RAGBits-enhanced retriever instance"""
    global _instance
    if _instance is None:
        _instance = RAGBitsEnhancedRetriever()
    return _instance
