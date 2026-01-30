"""
Advanced RAG Engine

Advanced Retrieval-Augmented Generation engine with hybrid search,
reranking, and query expansion.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search strategies"""
    SEMANTIC = "semantic"                # Pure vector similarity
    KEYWORD = "keyword"                  # Keyword-based search
    HYBRID = "hybrid"                    # Combined semantic + keyword
    RERANKED = "reranked"                # Search with reranking
    EXPANDED = "expanded"                # Query expansion


@dataclass
class RAGQuery:
    """A RAG query"""
    query_text: str
    search_type: SearchType = SearchType.HYBRID
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.0
    expand_query: bool = False
    rerank: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query_text": self.query_text,
            "search_type": self.search_type.value,
            "top_k": self.top_k,
            "filters": self.filters,
            "min_similarity": self.min_similarity,
            "expand_query": self.expand_query,
            "rerank": self.rerank,
            "metadata": self.metadata
        }


@dataclass
class RAGResult:
    """Result from RAG query"""
    query: RAGQuery
    retrieved_documents: List[Dict[str, Any]]
    ranked_documents: List[Dict[str, Any]]
    query_expansion: Optional[List[str]] = None
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query.to_dict(),
            "retrieved_documents": self.retrieved_documents,
            "ranked_documents": self.ranked_documents,
            "query_expansion": self.query_expansion,
            "search_metadata": self.search_metadata,
            "retrieval_time_ms": self.retrieval_time_ms,
            "timestamp": self.timestamp
        }


class AdvancedRAGEngine:
    """
    Advanced RAG engine with hybrid search and reranking.

    Usage:
        engine = AdvancedRAGEngine(document_search, crewai_client)

        # Simple query
        result = await engine.query(
            query_text="How to implement JWT authentication?",
            top_k=5
        )

        # Advanced query with filtering
        result = await engine.query(
            query_text="Authentication best practices",
            search_type=SearchType.RERANKED,
            filters={"artifact_type": "solution"},
            top_k=10
        )
    """

    def __init__(self, document_search=None, crewai_client=None):
        """
        Initialize advanced RAG engine.

        Args:
            document_search: Document search instance
            crewai_client: Optional LLM client for reranking/expansion
        """
        self.document_search = document_search
        self.crewai_client = crewai_client

        # Search statistics
        self.search_stats = {
            "total_queries": 0,
            "by_type": {search_type.value: 0 for search_type in SearchType},
            "average_retrieval_time": 0.0
        }

        logger.info("AdvancedRAGEngine initialized")

    async def query(
        self,
        query_text: str,
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        expand_query: bool = False,
        rerank: bool = True,
        **kwargs
    ) -> RAGResult:
        """
        Execute a RAG query.

        Args:
            query_text: Query text
            search_type: Type of search to perform
            top_k: Number of results to retrieve
            filters: Optional metadata filters
            expand_query: Whether to expand query
            rerank: Whether to rerank results

        Returns:
            RAG result
        """
        import time
        start_time = time.time()

        # Create query object
        query = RAGQuery(
            query_text=query_text,
            search_type=search_type,
            top_k=top_k,
            filters=filters,
            expand_query=expand_query,
            rerank=rerank,
            metadata=kwargs
        )

        logger.info(
            f"Executing RAG query: {query_text[:50]}... "
            f"(type={search_type.value}, k={top_k})"
        )

        # Query expansion
        expanded_queries = []
        if expand_query and self.crewai_client:
            expanded_queries = await self._expand_query(query_text)

        # Execute search based on type
        if search_type == SearchType.SEMANTIC:
            documents = await self._semantic_search(query, top_k)
        elif search_type == SearchType.KEYWORD:
            documents = await self._keyword_search(query, top_k)
        elif search_type == SearchType.HYBRID:
            documents = await self._hybrid_search(query, top_k)
        elif search_type == SearchType.EXPANDED:
            documents = await self._expanded_search(
                query,
                expanded_queries,
                top_k
            )
        else:  # RERANKED
            documents = await self._reranked_search(query, top_k)

        # Apply filters
        if filters:
            documents = self._apply_filters(documents, filters)

        # Rerank if requested
        if rerank and search_type != SearchType.RERANKED:
            documents = await self._rerank_documents(query, documents)

        retrieval_time = (time.time() - start_time) * 1000

        # Update statistics
        self.search_stats["total_queries"] += 1
        self.search_stats["by_type"][search_type.value] += 1

        result = RAGResult(
            query=query,
            retrieved_documents=documents,
            ranked_documents=documents,
            query_expansion=expanded_queries if expand_query else None,
            search_metadata={
                "total_retrieved": len(documents),
                "search_type": search_type.value
            },
            retrieval_time_ms=retrieval_time
        )

        logger.info(
            f"Retrieved {len(documents)} documents in {retrieval_time:.0f}ms"
        )

        return result

    async def _semantic_search(
        self,
        query: RAGQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Pure semantic search"""
        if not self.document_search:
            return []

        try:
            results = await self.document_search.search(
                query_text=query.query_text,
                top_k=top_k,
                filters=query.filters
            )

            return results

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    async def _keyword_search(
        self,
        query: RAGQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Keyword-based search"""
        if not self.document_search:
            return []

        # Extract keywords from query
        keywords = self._extract_keywords(query.query_text)

        # Simple implementation: return documents with keyword matches
        # In production, would use proper keyword search (BM25, etc.)
        try:
            results = await self.document_search.search(
                query_text=" ".join(keywords),
                top_k=top_k * 2,  # Get more results for filtering
                filters=query.filters
            )

            # Filter by keyword presence
            filtered = [
                r for r in results
                if any(kw.lower() in r.get("content", "").lower()
                      for kw in keywords)
            ]

            return filtered[:top_k]

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

    async def _hybrid_search(
        self,
        query: RAGQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Hybrid semantic + keyword search"""
        # Get results from both methods
        semantic_results = await self._semantic_search(query, top_k * 2)
        keyword_results = await self._keyword_search(query, top_k * 2)

        # Combine and deduplicate
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight=0.7,
            keyword_weight=0.3
        )

        return combined[:top_k]

    async def _expanded_search(
        self,
        query: RAGQuery,
        expanded_queries: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search with query expansion"""
        # Search with original query
        original_results = await self._semantic_search(query, top_k)

        if not expanded_queries:
            return original_results

        # Search with expanded queries
        all_results = original_results.copy()

        for expanded_query in expanded_queries[:2]:  # Limit expansions
            expanded_results = await self._semantic_search(
                RAGQuery(query_text=expanded_query),
                top_k
            )
            all_results.extend(expanded_results)

        # Combine and deduplicate
        combined = self._deduplicate_results(all_results)

        # Rerank combined results
        reranked = await self._rerank_documents(query, combined)

        return reranked[:top_k]

    async def _reranked_search(
        self,
        query: RAGQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search with reranking"""
        # Get initial results
        initial_results = await self._hybrid_search(query, top_k * 2)

        # Rerank
        reranked = await self._rerank_documents(query, initial_results)

        return reranked[:top_k]

    async def _expand_query(self, query_text: str) -> List[str]:
        """Expand query using LLM"""
        if not self.crewai_client:
            return []

        try:
            prompt = f"""Generate 3 alternative queries for the following question.
Each query should explore different aspects or use different wording.

Original query: {query_text}

Generate only the queries, one per line:
1.
2.
3."""

            response = await self.crewai_client.generate(
                prompt,
                temperature=0.5
            )

            # Parse expanded queries
            expanded = self._parse_expanded_queries(
                response.get("text", "")
            )

            return expanded

        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return []

    def _parse_expanded_queries(self, response_text: str) -> List[str]:
        """Parse expanded queries from LLM response"""
        queries = []

        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Remove numbering
            line = line.lstrip("0123456789.-) ")
            if len(line) > 10:
                queries.append(line)

        return queries[:3]

    async def _rerank_documents(
        self,
        query: RAGQuery,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using LLM or scoring"""
        if not documents:
            return documents

        if self.crewai_client:
            return await self._llm_rerank(query, documents)
        else:
            return self._score_rerank(query, documents)

    async def _llm_rerank(
        self,
        query: RAGQuery,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using LLM"""
        try:
            # Prepare documents for reranking
            doc_texts = [
                f"Doc {i+1}: {doc.get('content', '')[:200]}"
                for i, doc in enumerate(documents[:10])  # Limit to 10
            ]

            prompt = f"""Rank the following documents by relevance to the query.
Query: {query.query_text}

Documents:
{chr(10).join(doc_texts)}

Return ranking as comma-separated numbers (e.g., 3,1,2,4):"""

            response = await self.crewai_client.generate(
                prompt,
                temperature=0.3
            )

            # Parse ranking
            ranking = self._parse_ranking(
                response.get("text", ""),
                len(documents)
            )

            # Reorder documents
            reranked = [documents[i-1] for i in ranking if i <= len(documents)]

            return reranked

        except Exception as e:
            logger.error(f"LLM reranking error: {e}")
            return documents

    def _parse_ranking(self, response_text: str, num_docs: int) -> List[int]:
        """Parse ranking from LLM response"""
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', response_text)

        ranking = [int(n) for n in numbers if int(n) <= num_docs]

        return ranking if ranking else list(range(1, num_docs + 1))

    def _score_rerank(
        self,
        query: RAGQuery,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using scoring"""
        query_words = set(query.query_text.lower().split())

        scored_docs = []

        for doc in documents:
            content = doc.get("content", "").lower()

            # Calculate score based on word overlap
            overlap = len(query_words & set(content.split()))
            score = overlap / len(query_words) if query_words else 0

            scored_docs.append((doc, score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs]

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        # Remove stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "how", "what", "when", "where", "why"
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _combine_results(
        self,
        results1: List[Dict[str, Any]],
        results2: List[Dict[str, Any]],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Combine results from multiple searches"""
        combined = {}

        # Add first set with weights
        for i, doc in enumerate(results1):
            doc_id = doc.get("id", f"doc_{i}")
            combined[doc_id] = {
                **doc,
                "combined_score": doc.get("score", 0.5) * semantic_weight
            }

        # Add second set
        for i, doc in enumerate(results2):
            doc_id = doc.get("id", f"doc_{i}")
            if doc_id in combined:
                combined[doc_id]["combined_score"] += (
                    doc.get("score", 0.5) * keyword_weight
                )
            else:
                combined[doc_id] = {
                    **doc,
                    "combined_score": doc.get("score", 0.5) * keyword_weight
                }

        # Sort by combined score
        sorted_docs = sorted(
            combined.values(),
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )

        return sorted_docs

    def _deduplicate_results(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate documents"""
        seen = set()
        deduplicated = []

        for doc in documents:
            # Use content hash as identifier
            content = doc.get("content", "")
            content_hash = hash(content[:200])

            if content_hash not in seen:
                seen.add(content_hash)
                deduplicated.append(doc)

        return deduplicated

    def _apply_filters(
        self,
        documents: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters to documents"""
        filtered = []

        for doc in documents:
            metadata = doc.get("metadata", {})

            # Check all filters
            matches = True
            for key, value in filters.items():
                if metadata.get(key) != value:
                    matches = False
                    break

            if matches:
                filtered.append(doc)

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            **self.search_stats,
            "average_retrieval_time": (
                sum(s.search_metadata.get("retrieval_time_ms", 0)
                    for s in self.search_stats.get("recent_queries", [])) /
                len(self.search_stats.get("recent_queries", []))
                if self.search_stats.get("recent_queries") else 0
            )
        }
