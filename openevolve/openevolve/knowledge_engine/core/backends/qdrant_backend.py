"""
Qdrant Backend Adapter for Unified Knowledge Graph Manager.

Provides vector similarity search using Qdrant.
Follows CLAUDE.md principles: Runtime Truth, Configuration Explicitness, UTC.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid

from .base import (
    KnowledgeGraphBackend,
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)

logger = logging.getLogger(__name__)


class QdrantBackend(KnowledgeGraphBackend):
    """
    Qdrant backend adapter for vector similarity search.

    Environment Variables Required:
        QDRANT_HOST: Qdrant host (default: localhost)
        QDRANT_PORT: Qdrant port (default: 6333)
        QDRANT_COLLECTION: Collection name (default: knowledge_graph)
        QDRANT_API_KEY: Optional API key for authentication
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.QDRANT
        self.client = None
        self.collection_name = None
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration"""
        if 'host' not in self.config:
            raise ValueError("Qdrant backend requires 'host' in config")
        if 'port' not in self.config:
            raise ValueError("Qdrant backend requires 'port' in config")

        self.host = self.config['host']
        self.port = self.config['port']
        self.collection_name = self.config.get('collection', 'knowledge_graph')
        self.api_key = self.config.get('api_key')
        self.vector_size = self.config.get('vector_size', 1536)  # OpenAI default

        logger.info(f"Qdrant backend configured for collection: {self.collection_name}")

    async def connect(self) -> bool:
        """Establish connection to Qdrant - Runtime Truth"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.async_client import AsyncQdrantClient

            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=30
            )

            # Verify connection - Runtime Truth
            collections = await self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")

            # Create collection if it doesn't exist
            await self._ensure_collection_exists()

            self.is_healthy = True
            return True

        except ImportError:
            logger.error("qdrant-client package not installed. Install with: pip install qdrant-client")
            raise ImportError("qdrant-client package required for QdrantBackend")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Qdrant connection failed: {e}")

    async def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct

            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")

                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )

                logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Qdrant connection"""
        if self.client:
            await self.client.close()
            self.is_healthy = False
            logger.info("Disconnected from Qdrant")

    async def health_check(self) -> bool:
        """Check Qdrant health"""
        if not self.client:
            return False

        try:
            await self.client.get_collections()
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            self.is_healthy = False
            return False

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        # In production, use actual embedding service
        # For now, return dummy embedding
        import hashlib
        import numpy as np

        # Create deterministic pseudo-embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float array
        embedding = []
        for i in range(self.vector_size):
            byte_val = hash_bytes[i % len(hash_bytes)]
            embedding.append((byte_val - 128) / 128.0)

        return embedding

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add knowledge to Qdrant with vector embedding"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        from qdrant_client.models import PointStruct

        start_time = datetime.utcnow()

        try:
            # Generate or use provided embedding
            if entry.embedding is None:
                embedding = await self._generate_embedding(entry.content)
            else:
                embedding = entry.embedding

            # Create point ID
            point_id = str(uuid.uuid4())

            # Create payload
            payload = {
                "source": entry.source,
                "content": entry.content,
                "metadata": entry.metadata or {},
                "timestamp": entry.timestamp
            }

            # Insert point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Added knowledge to Qdrant in {elapsed_ms:.2f}ms: {point_id}")

            return point_id

        except Exception as e:
            logger.error(f"Failed to add knowledge to Qdrant: {e}")
            raise ConnectionError(f"Qdrant add_knowledge failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Vector similarity search in Qdrant"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        start_time = datetime.utcnow()

        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Build filter
            search_filter = None
            if filters and "source" in filters:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=filters["source"])
                        )
                    ]
                )

            # Search
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "source": hit.payload.get("source"),
                    "content": hit.payload.get("content"),
                    "metadata": hit.payload.get("metadata", {}),
                    "timestamp": hit.payload.get("timestamp")
                })

            # Get total count
            collection_info = await self.client.get_collection(self.collection_name)
            total_count = collection_info.points_count

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SearchResults(
                query=query,
                results=results,
                total_count=total_count,
                backend_used="qdrant",
                search_time_ms=elapsed_ms,
                metadata={"filters": filters, "search_type": "vector_similarity"}
            )

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise ConnectionError(f"Qdrant search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze Qdrant collection"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        start_time = datetime.utcnow()

        try:
            if analysis_type == "distribution":
                # Analyze knowledge distribution
                collection_info = await self.client.get_collection(self.collection_name)

                results = {
                    "total_points": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                }

            elif analysis_type == "source_breakdown":
                # Analyze by source
                # Note: This would require scroll API for full analysis
                results = {
                    "message": "Source breakdown requires full collection scan",
                    "hint": "Use scroll API for detailed analysis"
                }

            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AnalysisResult(
                analysis_type=analysis_type,
                target=target or "collection",
                results=results,
                backend_used="qdrant",
                analysis_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Qdrant analysis failed: {e}")
            raise ConnectionError(f"Qdrant analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get Qdrant collection statistics"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        try:
            collection_info = await self.client.get_collection(self.collection_name)

            return GraphStatistics(
                node_count=collection_info.points_count,
                edge_count=0,  # Qdrant doesn't have edges
                backend="qdrant",
                metadata={
                    "collection": self.collection_name,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value,
                    "status": collection_info.status.value
                },
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get Qdrant statistics: {e}")
            raise ConnectionError(f"Qdrant statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from Qdrant"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        try:
            if output_format == 'json':
                # Export sample as JSON
                from qdrant_client.models import ScrollRequest, PointSelector

                points, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )

                results = []
                for point in points:
                    results.append({
                        "id": point.id,
                        "payload": point.payload
                    })

                return json.dumps({
                    "points": results,
                    "collection": self.collection_name
                }, indent=2)

            elif output_format == 'html':
                # Simple HTML visualization
                collection_info = await self.client.get_collection(self.collection_name)

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Qdrant Knowledge Collection</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .stats {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                        .stat-item {{ margin: 10px 0; }}
                    </style>
                </head>
                <body>
                    <h1>Qdrant Knowledge Collection</h1>
                    <div class="stats">
                        <div class="stat-item"><strong>Collection:</strong> {self.collection_name}</div>
                        <div class="stat-item"><strong>Total Points:</strong> {collection_info.points_count}</div>
                        <div class="stat-item"><strong>Vector Size:</strong> {collection_info.config.params.vectors.size}</div>
                        <div class="stat-item"><strong>Distance:</strong> {collection_info.config.params.vectors.distance.value}</div>
                        <div class="stat-item"><strong>Status:</strong> {collection_info.status.value}</div>
                    </div>
                    <p>Vector similarity search is available through the API.</p>
                </body>
                </html>
                """
                return html

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Qdrant visualization failed: {e}")
            raise ConnectionError(f"Qdrant visualization failed: {e}")

    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete knowledge from Qdrant"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[entry_id]
            )
            return True

        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            raise ConnectionError(f"Qdrant delete failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from Qdrant - Destructive operation"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        try:
            # Get count before deletion
            collection_info = await self.client.get_collection(self.collection_name)
            count = collection_info.points_count

            # Delete all points
            await self.client.delete_collection(self.collection_name)

            # Recreate collection
            await self._ensure_collection_exists()

            logger.warning(f"Cleared {count} points from Qdrant")
            return count

        except Exception as e:
            logger.error(f"Qdrant clear failed: {e}")
            raise ConnectionError(f"Qdrant clear failed: {e}")

    async def batch_add_knowledge(
        self,
        entries: List[KnowledgeEntry]
    ) -> List[str]:
        """Batch add knowledge to Qdrant efficiently"""
        if not self.is_healthy:
            raise ConnectionError("Qdrant backend not healthy")

        from qdrant_client.models import PointStruct

        start_time = datetime.utcnow()

        try:
            points = []
            ids = []

            for entry in entries:
                # Generate embedding
                if entry.embedding is None:
                    embedding = await self._generate_embedding(entry.content)
                else:
                    embedding = entry.embedding

                point_id = str(uuid.uuid4())
                ids.append(point_id)

                payload = {
                    "source": entry.source,
                    "content": entry.content,
                    "metadata": entry.metadata or {},
                    "timestamp": entry.timestamp
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

            # Batch upsert
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Batch added {len(ids)} entries to Qdrant in {elapsed_ms:.2f}ms")

            return ids

        except Exception as e:
            logger.error(f"Qdrant batch add failed: {e}")
            raise ConnectionError(f"Qdrant batch add failed: {e}")
