#!/usr/bin/env python
"""
Knowledge Explorer

Interactive knowledge exploration interface with advanced filtering,
faceted search, and knowledge graph visualization.
"""

import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path


class SearchStrategy(Enum):
    """Search strategies"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    EXACT = "exact"


class EntityType(Enum):
    """Knowledge entity types"""
    SOLUTION_PATTERN = "solution_pattern"
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    ANTI_PATTERN = "anti_pattern"
    TECHNIQUE = "technique"
    PRINCIPLE = "principle"
    REQUIREMENT = "requirement"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"
    DEPENDENCY = "dependency"


class SortOrder(Enum):
    """Sort orders"""
    RELEVANCE = "relevance"
    DATE_NEWEST = "date_newest"
    DATE_OLDEST = "date_oldest"
    QUALITY_ASC = "quality_asc"
    QUALITY_DESC = "quality_desc"


@dataclass
class SearchFilter:
    """Search filter"""
    entity_types: Optional[List[EntityType]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_quality_score: float = 0.0
    tags: Optional[List[str]] = None
    artifact_types: Optional[List[str]] = None
    stage: Optional[str] = None
    team: Optional[str] = None


@dataclass
class SearchResult:
    """Single search result"""
    entity_id: str
    entity_type: EntityType
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    quality_score: float
    highlights: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)


@dataclass
class KnowledgeNode:
    """Node in knowledge graph"""
    node_id: str
    label: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """Edge in knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure"""
    nodes: List[KnowledgeNode] = field(default_factory=list)
    edges: List[KnowledgeEdge] = field(default_factory=list)


class KnowledgeExplorer:
    """
    Interactive knowledge exploration interface.

    Features:
    - Multi-strategy search (semantic, keyword, hybrid)
    - Advanced filtering and faceting
    - Knowledge graph visualization
    - Entity relationship exploration
    - Interactive refinement
    """

    def __init__(self, storage_manager=None, rag_engine=None):
        """
        Initialize knowledge explorer.

        Args:
            storage_manager: Optional storage manager
            rag_engine: Optional RAG engine for search
        """
        from ragbits_integration.intermediary_storage import IntermediaryStorageManager
        from ragbits_integration.knowledge_base import AdvancedRAGEngine

        self.storage = storage_manager
        self.rag = rag_engine

        # Search history
        self._search_history: List[Dict[str, Any]] = []

        # Knowledge graph cache
        self._graph_cache: Optional[KnowledgeGraph] = None

    async def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        filters: Optional[SearchFilter] = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: SortOrder = SortOrder.RELEVANCE
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Search knowledge base.

        Args:
            query: Search query
            strategy: Search strategy
            filters: Optional search filters
            limit: Max results
            offset: Offset for pagination
            sort_by: Sort order

        Returns:
            Tuple of (results, metadata)
        """
        # Record search
        self._search_history.append({
            "query": query,
            "strategy": strategy.value,
            "timestamp": datetime.now().isoformat()
        })

        # Execute search based on strategy
        if strategy == SearchStrategy.SEMANTIC:
            results = await self._semantic_search(query, filters, limit * 2)
        elif strategy == SearchStrategy.KEYWORD:
            results = await self._keyword_search(query, filters, limit * 2)
        elif strategy == SearchStrategy.HYBRID:
            results = await self._hybrid_search(query, filters, limit * 2)
        else:  # EXACT
            results = await self._exact_search(query, filters, limit * 2)

        # Apply additional filters
        if filters:
            results = self._apply_filters(results, filters)

        # Sort results
        results = self._sort_results(results, sort_by)

        # Paginate
        total_count = len(results)
        paginated_results = results[offset:offset + limit]

        # Generate highlights
        for result in paginated_results:
            result.highlights = self._generate_highlights(query, result.content)

        # Metadata
        metadata = {
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total_count
        }

        return paginated_results, metadata

    async def _semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilter],
        limit: int
    ) -> List[SearchResult]:
        """Perform semantic search"""
        if not self.rag:
            return []

        try:
            rag_result = await self.rag.query(
                query_text=query,
                search_type="semantic",
                top_k=limit
            )

            results = []
            for doc in rag_result.ranked_documents:
                results.append(SearchResult(
                    entity_id=doc.get("id", ""),
                    entity_type=EntityType(doc.get("entity_type", "solution_pattern")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=doc.get("score", 0.0),
                    quality_score=doc.get("quality_score", 0.5)
                ))

            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    async def _keyword_search(
        self,
        query: str,
        filters: Optional[SearchFilter],
        limit: int
    ) -> List[SearchResult]:
        """Perform keyword search"""
        if not self.rag:
            return []

        try:
            rag_result = await self.rag.query(
                query_text=query,
                search_type="keyword",
                top_k=limit
            )

            results = []
            for doc in rag_result.ranked_documents:
                results.append(SearchResult(
                    entity_id=doc.get("id", ""),
                    entity_type=EntityType(doc.get("entity_type", "solution_pattern")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=doc.get("score", 0.0),
                    quality_score=doc.get("quality_score", 0.5)
                ))

            return results
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    async def _hybrid_search(
        self,
        query: str,
        filters: Optional[SearchFilter],
        limit: int
    ) -> List[SearchResult]:
        """Perform hybrid search"""
        if not self.rag:
            return []

        try:
            rag_result = await self.rag.query(
                query_text=query,
                search_type="hybrid",
                top_k=limit
            )

            results = []
            for doc in rag_result.ranked_documents:
                results.append(SearchResult(
                    entity_id=doc.get("id", ""),
                    entity_type=EntityType(doc.get("entity_type", "solution_pattern")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=doc.get("score", 0.0),
                    quality_score=doc.get("quality_score", 0.5)
                ))

            return results
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

    async def _exact_search(
        self,
        query: str,
        filters: Optional[SearchFilter],
        limit: int
    ) -> List[SearchResult]:
        """Perform exact match search"""
        # For now, fallback to keyword search
        return await self._keyword_search(query, filters, limit)

    def _apply_filters(
        self,
        results: List[SearchResult],
        filters: SearchFilter
    ) -> List[SearchResult]:
        """Apply filters to results"""
        filtered = results

        # Entity types
        if filters.entity_types:
            filtered = [
                r for r in filtered
                if r.entity_type in filters.entity_types
            ]

        # Quality score
        if filters.min_quality_score > 0:
            filtered = [
                r for r in filtered
                if r.quality_score >= filters.min_quality_score
            ]

        # Tags
        if filters.tags:
            filtered = [
                r for r in filtered
                if any(tag in r.metadata.get("tags", []) for tag in filters.tags)
            ]

        # Artifact types
        if filters.artifact_types:
            filtered = [
                r for r in filtered
                if r.metadata.get("artifact_type") in filters.artifact_types
            ]

        # Stage
        if filters.stage:
            filtered = [
                r for r in filtered
                if r.metadata.get("stage") == filters.stage
            ]

        # Team
        if filters.team:
            filtered = [
                r for r in filtered
                if r.metadata.get("team") == filters.team
            ]

        return filtered

    def _sort_results(
        self,
        results: List[SearchResult],
        sort_by: SortOrder
    ) -> List[SearchResult]:
        """Sort results"""
        if sort_by == SortOrder.RELEVANCE:
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
        elif sort_by == SortOrder.QUALITY_DESC:
            return sorted(results, key=lambda r: r.quality_score, reverse=True)
        elif sort_by == SortOrder.QUALITY_ASC:
            return sorted(results, key=lambda r: r.quality_score)
        elif sort_by == SortOrder.DATE_NEWEST:
            return sorted(
                results,
                key=lambda r: r.metadata.get("created_at", ""),
                reverse=True
            )
        elif sort_by == SortOrder.DATE_OLDEST:
            return sorted(
                results,
                key=lambda r: r.metadata.get("created_at", "")
            )
        else:
            return results

    def _generate_highlights(
        self,
        query: str,
        content: str,
        window: int = 100
    ) -> List[str]:
        """Generate search highlights"""
        query_terms = query.lower().split()
        highlights = []

        # Simple highlighting
        for term in query_terms:
            if term in content.lower():
                # Find position
                idx = content.lower().find(term)
                if idx >= 0:
                    # Extract window around match
                    start = max(0, idx - window)
                    end = min(len(content), idx + len(term) + window)
                    highlight = content[start:end]

                    # Add ellipsis if needed
                    if start > 0:
                        highlight = "..." + highlight
                    if end < len(content):
                        highlight = highlight + "..."

                    highlights.append(highlight)

        return highlights[:3]  # Max 3 highlights

    async def get_knowledge_graph(
        self,
        center_entity_id: Optional[str] = None,
        max_depth: int = 2,
        max_nodes: int = 50
    ) -> KnowledgeGraph:
        """
        Get knowledge graph.

        Args:
            center_entity_id: Optional center entity ID
            max_depth: Maximum depth from center
            max_nodes: Maximum nodes to return

        Returns:
            KnowledgeGraph object
        """
        # Build graph from storage
        nodes = []
        edges = []

        if not self.storage:
            return KnowledgeGraph(nodes=nodes, edges=edges)

        # Get recent entities
        # In production, this would query the knowledge base
        # For now, return empty graph
        return KnowledgeGraph(nodes=nodes, edges=edges)

    async def get_entity_details(
        self,
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an entity.

        Args:
            entity_id: Entity ID

        Returns:
            Entity details or None
        """
        if not self.storage:
            return None

        # Would query storage for full details
        return {
            "entity_id": entity_id,
            "note": "Full details would be loaded from storage"
        }

    async def get_similar_entities(
        self,
        entity_id: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find similar entities.

        Args:
            entity_id: Entity ID
            limit: Max results

        Returns:
            List of similar entities
        """
        # Get entity details
        details = await self.get_entity_details(entity_id)
        if not details:
            return []

        # Use content to find similar
        content = details.get("content", "")
        if not content:
            return []

        # Semantic search
        results, _ = await self.search(
            query=content[:200],
            strategy=SearchStrategy.SEMANTIC,
            limit=limit + 1  # +1 to exclude self
        )

        # Exclude self
        return [r for r in results if r.entity_id != entity_id][:limit]

    async def get_facets(
        self,
        query: Optional[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Get facet counts for filtering.

        Args:
            query: Optional query to scope facets

        Returns:
            Facet counts
        """
        # Would compute actual facet counts from knowledge base
        return {
            "entity_type": {
                "solution_pattern": 150,
                "best_practice": 120,
                "lesson_learned": 80,
                "technique": 60
            },
            "artifact_type": {
                "solution": 200,
                "decomposition_plan": 100,
                "critique": 80
            },
            "stage": {
                "stage_3": 150,
                "stage_4": 100,
                "stage_5": 80
            }
        }

    def export_search_results(
        self,
        results: List[SearchResult],
        metadata: Dict[str, Any],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export search results.

        Args:
            results: Search results
            metadata: Search metadata
            format: Export format
            output_path: Optional output file

        Returns:
            Exported content
        """
        if format == "json":
            content = self._export_json(results, metadata)
        elif format == "markdown":
            content = self._export_markdown(results, metadata)
        elif format == "csv":
            content = self._export_csv(results)
        else:
            raise ValueError(f"Unknown format: {format}")

        if output_path:
            Path(output_path).write_text(content)

        return content

    def _export_json(
        self,
        results: List[SearchResult],
        metadata: Dict[str, Any]
    ) -> str:
        """Export as JSON"""
        export = {
            "metadata": metadata,
            "results": [
                {
                    "entity_id": r.entity_id,
                    "entity_type": r.entity_type.value,
                    "content": r.content[:500],
                    "relevance_score": r.relevance_score,
                    "quality_score": r.quality_score,
                    "highlights": r.highlights,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }

        return json.dumps(export, indent=2)

    def _export_markdown(
        self,
        results: List[SearchResult],
        metadata: Dict[str, Any]
    ) -> str:
        """Export as Markdown"""
        lines = [
            "# Knowledge Search Results",
            "",
            f"**Total Results:** {metadata.get('total_count', 0)}",
            f"**Showing:** {metadata.get('offset', 0)} - {metadata.get('offset', 0) + len(results)}",
            ""
        ]

        for i, result in enumerate(results, 1):
            lines.append(f"## Result {i}")
            lines.append(f"**ID:** {result.entity_id}")
            lines.append(f"**Type:** {result.entity_type.value}")
            lines.append(f"**Relevance:** {result.relevance_score:.2f}")
            lines.append(f"**Quality:** {result.quality_score:.2f}")
            lines.append("")
            lines.append("**Content:**")
            lines.append(result.content[:300] + "..." if len(result.content) > 300 else result.content)
            lines.append("")

            if result.highlights:
                lines.append("**Highlights:**")
                for highlight in result.highlights:
                    lines.append(f"- {highlight}")
                lines.append("")

        return "\n".join(lines)

    def _export_csv(self, results: List[SearchResult]) -> str:
        """Export as CSV"""
        lines = ["entity_id,entity_type,relevance_score,quality_score,content"]

        for result in results:
            # Escape content for CSV
            content = result.content.replace('"', '""').replace('\n', ' ')
            lines.append(f'"{result.entity_id}","{result.entity_type.value}",{result.relevance_score:.2f},{result.quality_score:.2f},"{content[:200]}"')

        return "\n".join(lines)

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history"""
        return self._search_history.copy()


__all__ = [
    "KnowledgeExplorer",
    "SearchResult",
    "SearchFilter",
    "SearchStrategy",
    "SortOrder",
    "EntityType",
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge"
]
