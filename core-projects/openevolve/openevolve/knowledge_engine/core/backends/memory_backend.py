"""
In-Memory Backend Adapter for Unified Knowledge Graph Manager.

Simple in-memory storage for testing and development.
Follows CLAUDE.md principles: Configuration Explicitness, UTC.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
from collections import defaultdict

from .base import (
    KnowledgeGraphBackend,
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)

logger = logging.getLogger(__name__)


class MemoryBackend(KnowledgeGraphBackend):
    """
    In-memory backend for knowledge graph storage.

    This is a simple implementation that stores everything in memory.
    Useful for testing and development. Data is lost on restart.

    Environment Variables:
        None required - runs entirely in memory
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.MEMORY

        # In-memory storage
        self.knowledge_store: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []

        logger.info("Memory backend initialized (data will be lost on restart)")

    async def connect(self) -> bool:
        """Initialize in-memory backend"""
        self.is_healthy = True
        logger.info("Memory backend connected")
        return True

    async def disconnect(self) -> None:
        """Cleanup in-memory backend"""
        self.knowledge_store.clear()
        self.entities.clear()
        self.relationships.clear()
        self.is_healthy = False
        logger.info("Memory backend disconnected and cleared")

    async def health_check(self) -> bool:
        """Memory backend is always healthy if initialized"""
        return self.is_healthy

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add knowledge to memory"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        start_time = datetime.utcnow()

        try:
            # Generate ID
            entry_id = str(uuid.uuid4())

            # Store knowledge
            self.knowledge_store[entry_id] = {
                "id": entry_id,
                "source": entry.source,
                "content": entry.content,
                "metadata": entry.metadata or {},
                "embedding": entry.embedding,
                "timestamp": entry.timestamp
            }

            # Extract entities (simple word extraction)
            words = entry.content.split()
            extracted_entities = list(set([w for w in words if len(w) > 3 and w.isalnum()]))

            for entity_name in extracted_entities[:5]:
                # Create entity if doesn't exist
                if entity_name not in self.entities:
                    self.entities[entity_name] = {
                        "name": entity_name,
                        "mentions": 0
                    }

                # Increment mention count
                self.entities[entity_name]["mentions"] += 1

                # Add relationship
                self.relationships.append({
                    "source": entry_id,
                    "relation": "MENTIONS",
                    "target": entity_name
                })

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Added knowledge to memory in {elapsed_ms:.2f}ms: {entry_id}")

            return entry_id

        except Exception as e:
            logger.error(f"Failed to add knowledge to memory: {e}")
            raise ConnectionError(f"Memory add_knowledge failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Search knowledge in memory"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        start_time = datetime.utcnow()

        try:
            query_lower = query.lower()
            results = []

            # Search through knowledge store
            for entry_id, entry in self.knowledge_store.items():
                # Apply filters
                if filters and "source" in filters:
                    if entry["source"] != filters["source"]:
                        continue

                # Content search
                if query_lower in entry["content"].lower() or query_lower in entry["source"].lower():
                    # Find related entities
                    related_entities = [
                        r["target"] for r in self.relationships
                        if r["source"] == entry_id
                    ]

                    results.append({
                        "id": entry_id,
                        "source": entry["source"],
                        "content": entry["content"],
                        "metadata": entry["metadata"],
                        "timestamp": entry["timestamp"],
                        "entities": related_entities
                    })

            # Apply pagination
            paginated_results = results[offset:offset + limit]

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SearchResults(
                query=query,
                results=paginated_results,
                total_count=len(results),
                backend_used="memory",
                search_time_ms=elapsed_ms,
                metadata={"filters": filters}
            )

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise ConnectionError(f"Memory search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze in-memory knowledge graph"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        start_time = datetime.utcnow()

        try:
            if analysis_type == "entity_analysis":
                # Analyze entities
                sorted_entities = sorted(
                    self.entities.items(),
                    key=lambda x: x[1]["mentions"],
                    reverse=True
                )[:20]

                results = {
                    "total_entities": len(self.entities),
                    "top_entities": [
                        {"name": name, "mentions": data["mentions"]}
                        for name, data in sorted_entities
                    ]
                }

            elif analysis_type == "source_distribution":
                # Analyze by source
                source_counts = defaultdict(int)
                for entry in self.knowledge_store.values():
                    source_counts[entry["source"]] += 1

                results = {
                    "by_source": [
                        {"source": source, "count": count}
                        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
                    ]
                }

            elif analysis_type == "relationship_analysis":
                # Analyze relationships
                relationship_types = defaultdict(int)
                for rel in self.relationships:
                    relationship_types[rel["relation"]] += 1

                results = {
                    "total_relationships": len(self.relationships),
                    "by_type": dict(relationship_types)
                }

            elif analysis_type == "graph_overview":
                # Overall graph statistics
                results = {
                    "knowledge_entries": len(self.knowledge_store),
                    "entities": len(self.entities),
                    "relationships": len(self.relationships)
                }

            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AnalysisResult(
                analysis_type=analysis_type,
                target=target or "graph",
                results=results,
                backend_used="memory",
                analysis_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            raise ConnectionError(f"Memory analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get in-memory graph statistics"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        try:
            return GraphStatistics(
                node_count=len(self.knowledge_store) + len(self.entities),
                edge_count=len(self.relationships),
                backend="memory",
                metadata={
                    "knowledge_entries": len(self.knowledge_store),
                    "entities": len(self.entities),
                    "relationships": len(self.relationships)
                },
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            raise ConnectionError(f"Memory statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from memory"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        try:
            if output_format == 'json':
                # Export as JSON
                data = {
                    "knowledge": list(self.knowledge_store.values()),
                    "entities": list(self.entities.values()),
                    "relationships": self.relationships
                }
                return json.dumps(data, indent=2)

            elif output_format == 'html':
                # Generate HTML visualization
                stats = await self.get_statistics()

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Memory Knowledge Graph</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .stats {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                        .stat-item {{ margin: 10px 0; }}
                        .section {{ margin: 20px 0; }}
                        .entry {{ background: white; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <h1>In-Memory Knowledge Graph</h1>
                    <div class="stats">
                        <div class="stat-item"><strong>Total Nodes:</strong> {stats.node_count}</div>
                        <div class="stat-item"><strong>Total Edges:</strong> {stats.edge_count}</div>
                        <div class="stat-item"><strong>Knowledge Entries:</strong> {stats.metadata['knowledge_entries']}</div>
                        <div class="stat-item"><strong>Entities:</strong> {stats.metadata['entities']}</div>
                    </div>

                    <h2>Top Entities</h2>
                    <div class="section">
                """

                # Show top 10 entities
                sorted_entities = sorted(
                    self.entities.items(),
                    key=lambda x: x[1]["mentions"],
                    reverse=True
                )[:10]

                for entity_name, entity_data in sorted_entities:
                    html += f"""
                        <div class="entry">
                            <strong>Name:</strong> {entity_name}<br>
                            <strong>Mentions:</strong> {entity_data['mentions']}
                        </div>
                    """

                html += """
                    </div>
                    <p><em>Note: This is an in-memory backend. Data will be lost on restart.</em></p>
                </body>
                </html>
                """
                return html

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Memory visualization failed: {e}")
            raise ConnectionError(f"Memory visualization failed: {e}")

    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete knowledge from memory"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        try:
            if entry_id in self.knowledge_store:
                del self.knowledge_store[entry_id]

                # Remove associated relationships
                self.relationships = [
                    r for r in self.relationships if r["source"] != entry_id
                ]

                return True
            return False

        except Exception as e:
            logger.error(f"Memory delete failed: {e}")
            raise ConnectionError(f"Memory delete failed: {e}")

    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update knowledge in memory"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        try:
            if entry_id in self.knowledge_store:
                self.knowledge_store[entry_id].update(updates)
                return True
            return False

        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            raise ConnectionError(f"Memory update failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from memory"""
        if not self.is_healthy:
            raise ConnectionError("Memory backend not healthy")

        try:
            count = len(self.knowledge_store)

            self.knowledge_store.clear()
            self.entities.clear()
            self.relationships.clear()

            logger.warning(f"Cleared {count} entries from memory")
            return count

        except Exception as e:
            logger.error(f"Memory clear failed: {e}")
            raise ConnectionError(f"Memory clear failed: {e}")
