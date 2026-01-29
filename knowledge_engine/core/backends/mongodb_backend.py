"""
MongoDB Backend Adapter for Unified Knowledge Graph Manager.

Provides document storage using MongoDB.
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


class MongoDBBackend(KnowledgeGraphBackend):
    """
    MongoDB backend adapter for document storage.

    Environment Variables Required:
        MONGODB_URI: MongoDB connection URI (e.g., mongodb://localhost:27017)
        MONGODB_DATABASE: Database name (default: knowledge_graph)
        MONGODB_COLLECTION: Collection name (default: knowledge)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.MONGODB
        self.client = None
        self.db = None
        self.collection = None
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration"""
        if 'uri' not in self.config:
            raise ValueError("MongoDB backend requires 'uri' in config")

        self.uri = self.config['uri']
        self.database_name = self.config.get('database', 'knowledge_graph')
        self.collection_name = self.config.get('collection', 'knowledge')

        logger.info(f"MongoDB backend configured for: {self.database_name}.{self.collection_name}")

    async def connect(self) -> bool:
        """Establish connection to MongoDB - Runtime Truth"""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self.client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=30000
            )

            # Verify connection - Runtime Truth
            await self.client.admin.command('ping')

            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]

            # Create indexes for better search performance
            await self._ensure_indexes()

            self.is_healthy = True
            logger.info(f"Successfully connected to MongoDB: {self.database_name}")

            return True

        except ImportError:
            logger.error("motor package not installed. Install with: pip install motor")
            raise ImportError("motor package required for MongoDBBackend")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionError(f"MongoDB connection failed: {e}")

    async def _ensure_indexes(self):
        """Create indexes for better performance"""
        try:
            # Text index for full-text search
            await self.collection.create_index([("content", "text"), ("source", "text")])

            # Single field indexes
            await self.collection.create_index([("source", 1)])
            await self.collection.create_index([("timestamp", -1)])
            await self.collection.create_index([("metadata.tags", 1)])

            logger.info("Created MongoDB indexes")

        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    async def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.is_healthy = False
            logger.info("Disconnected from MongoDB")

    async def health_check(self) -> bool:
        """Check MongoDB health"""
        if not self.client:
            return False

        try:
            await self.client.admin.command('ping')
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            self.is_healthy = False
            return False

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add knowledge to MongoDB"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        start_time = datetime.utcnow()

        try:
            # Create document
            document = {
                "_id": str(uuid.uuid4()),
                "source": entry.source,
                "content": entry.content,
                "metadata": entry.metadata or {},
                "embedding": entry.embedding,
                "timestamp": entry.timestamp,
                "created_at": datetime.utcnow()
            }

            # Insert document
            result = await self.collection.insert_one(document)

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Added knowledge to MongoDB in {elapsed_ms:.2f}ms: {result.inserted_id}")

            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to add knowledge to MongoDB: {e}")
            raise ConnectionError(f"MongoDB add_knowledge failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Search knowledge in MongoDB"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        start_time = datetime.utcnow()

        try:
            # Build query
            mongo_query = {}

            # Text search
            if query:
                mongo_query["$text"] = {"$search": query}

            # Apply filters
            if filters:
                if "source" in filters:
                    mongo_query["source"] = filters["source"]
                if "date_after" in filters:
                    mongo_query["timestamp"] = {"$gte": filters["date_after"]}
                if "tags" in filters:
                    mongo_query["metadata.tags"] = {"$in": filters["tags"]}

            # Execute search
            cursor = self.collection.find(mongo_query).sort("timestamp", -1).skip(offset).limit(limit)

            results = []
            async for doc in cursor:
                results.append({
                    "id": str(doc["_id"]),
                    "source": doc["source"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "timestamp": doc["timestamp"],
                    "created_at": doc.get("created_at", datetime.utcnow()).isoformat()
                })

            # Get total count
            total_count = await self.collection.count_documents(mongo_query)

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SearchResults(
                query=query,
                results=results,
                total_count=total_count,
                backend_used="mongodb",
                search_time_ms=elapsed_ms,
                metadata={"filters": filters}
            )

        except Exception as e:
            logger.error(f"MongoDB search failed: {e}")
            raise ConnectionError(f"MongoDB search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze MongoDB knowledge collection"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        start_time = datetime.utcnow()

        try:
            if analysis_type == "source_distribution":
                # Analyze distribution by source
                pipeline = [
                    {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ]

                results = {"by_source": []}
                async for doc in self.collection.aggregate(pipeline):
                    results["by_source"].append({
                        "source": doc["_id"],
                        "count": doc["count"]
                    })

            elif analysis_type == "tag_distribution":
                # Analyze distribution by tags
                pipeline = [
                    {"$unwind": "$metadata.tags"},
                    {"$group": {"_id": "$metadata.tags", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 20}
                ]

                results = {"by_tag": []}
                async for doc in self.collection.aggregate(pipeline):
                    results["by_tag"].append({
                        "tag": doc["_id"],
                        "count": doc["count"]
                    })

            elif analysis_type == "temporal_analysis":
                # Analyze knowledge over time
                pipeline = [
                    {
                        "$group": {
                            "_id": {
                                "year": {"$year": "$created_at"},
                                "month": {"$month": "$created_at"},
                                "day": {"$dayOfMonth": "$created_at"}
                            },
                            "count": {"$sum": 1}
                        }
                    },
                    {"$sort": {"_id": 1}},
                    {"$limit": 30}
                ]

                results = {"timeline": []}
                async for doc in self.collection.aggregate(pipeline):
                    results["timeline"].append({
                        "date": f"{doc['_id']['year']}-{doc['_id']['month']}-{doc['_id']['day']}",
                        "count": doc["count"]
                    })

            elif analysis_type == "content_statistics":
                # Content length statistics
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "avg_length": {"$avg": {"$strLenCP": "$content"}},
                            "max_length": {"$max": {"$strLenCP": "$content"}},
                            "min_length": {"$min": {"$strLenCP": "$content"}},
                            "total_docs": {"$sum": 1}
                        }
                    }
                ]

                async for doc in self.collection.aggregate(pipeline):
                    results = {
                        "average_content_length": int(doc["avg_length"]),
                        "max_content_length": doc["max_length"],
                        "min_content_length": doc["min_length"],
                        "total_documents": doc["total_docs"]
                    }

            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AnalysisResult(
                analysis_type=analysis_type,
                target=target or "collection",
                results=results,
                backend_used="mongodb",
                analysis_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"MongoDB analysis failed: {e}")
            raise ConnectionError(f"MongoDB analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get MongoDB collection statistics"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        try:
            # Get collection stats
            total_count = await self.collection.count_documents({})

            # Get stats by source
            pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]

            source_stats = {}
            async for doc in self.collection.aggregate(pipeline):
                source_stats[doc["_id"]] = doc["count"]

            return GraphStatistics(
                node_count=total_count,
                edge_count=0,  # MongoDB doesn't have native edges
                backend="mongodb",
                metadata={
                    "database": self.database_name,
                    "collection": self.collection_name,
                    "by_source": source_stats
                },
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get MongoDB statistics: {e}")
            raise ConnectionError(f"MongoDB statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from MongoDB"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        try:
            if output_format == 'json':
                # Export sample as JSON
                cursor = self.collection.find().limit(100)

                results = []
                async for doc in cursor:
                    doc["_id"] = str(doc["_id"])
                    if "created_at" in doc:
                        doc["created_at"] = doc["created_at"].isoformat()
                    results.append(doc)

                return json.dumps({
                    "documents": results,
                    "database": self.database_name,
                    "collection": self.collection_name
                }, indent=2)

            elif output_format == 'html':
                # Generate HTML visualization
                stats = await self.get_statistics()

                # Get recent documents
                cursor = self.collection.find().sort("timestamp", -1).limit(10)
                recent_docs = []
                async for doc in cursor:
                    recent_docs.append({
                        "id": str(doc["_id"]),
                        "source": doc["source"],
                        "content": doc["content"][:100] + "...",
                        "timestamp": doc["timestamp"]
                    })

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>MongoDB Knowledge Collection</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .stats {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                        .stat-item {{ margin: 10px 0; }}
                        .docs {{ margin-top: 20px; }}
                        .doc {{ background: white; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <h1>MongoDB Knowledge Collection</h1>
                    <div class="stats">
                        <div class="stat-item"><strong>Database:</strong> {self.database_name}</div>
                        <div class="stat-item"><strong>Collection:</strong> {self.collection_name}</div>
                        <div class="stat-item"><strong>Total Documents:</strong> {stats.node_count}</div>
                        <div class="stat-item"><strong>Sources:</strong> {len(stats.metadata.get('by_source', {}))}</div>
                    </div>
                    <h2>Recent Documents</h2>
                    <div class="docs">
                """

                for doc in recent_docs:
                    html += f"""
                        <div class="doc">
                            <strong>ID:</strong> {doc['id']}<br>
                            <strong>Source:</strong> {doc['source']}<br>
                            <strong>Content:</strong> {doc['content']}<br>
                            <strong>Timestamp:</strong> {doc['timestamp']}
                        </div>
                    """

                html += """
                    </div>
                </body>
                </html>
                """
                return html

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"MongoDB visualization failed: {e}")
            raise ConnectionError(f"MongoDB visualization failed: {e}")

    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete knowledge from MongoDB"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        try:
            result = await self.collection.delete_one({"_id": entry_id})
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"MongoDB delete failed: {e}")
            raise ConnectionError(f"MongoDB delete failed: {e}")

    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update knowledge in MongoDB"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        try:
            # Don't allow updating _id
            if "_id" in updates:
                del updates["_id"]

            # Add updated timestamp
            updates["updated_at"] = datetime.utcnow()

            result = await self.collection.update_one(
                {"_id": entry_id},
                {"$set": updates}
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"MongoDB update failed: {e}")
            raise ConnectionError(f"MongoDB update failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from MongoDB - Destructive operation"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        try:
            # Get count before deletion
            count = await self.collection.count_documents({})

            # Delete all documents
            result = await self.collection.delete_many({})

            logger.warning(f"Cleared {result.deleted_count} documents from MongoDB")
            return result.deleted_count

        except Exception as e:
            logger.error(f"MongoDB clear failed: {e}")
            raise ConnectionError(f"MongoDB clear failed: {e}")

    async def batch_add_knowledge(
        self,
        entries: List[KnowledgeEntry]
    ) -> List[str]:
        """Batch add knowledge to MongoDB efficiently"""
        if not self.is_healthy:
            raise ConnectionError("MongoDB backend not healthy")

        start_time = datetime.utcnow()

        try:
            documents = []
            ids = []

            for entry in entries:
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)

                document = {
                    "_id": doc_id,
                    "source": entry.source,
                    "content": entry.content,
                    "metadata": entry.metadata or {},
                    "embedding": entry.embedding,
                    "timestamp": entry.timestamp,
                    "created_at": datetime.utcnow()
                }

                documents.append(document)

            # Batch insert
            result = await self.collection.insert_many(documents)

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Batch added {len(ids)} entries to MongoDB in {elapsed_ms:.2f}ms")

            return ids

        except Exception as e:
            logger.error(f"MongoDB batch add failed: {e}")
            raise ConnectionError(f"MongoDB batch add failed: {e}")
