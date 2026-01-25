"""
OpenEvolve Knowledge Engine - Data Storage Layer

This module provides the data storage infrastructure for the knowledge engine,
including database connections, vector stores, caching, and knowledge artifact persistence.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import uuid
import json
from pathlib import Path
import aiosqlite
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
import redis.asyncio as redis
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeArtifact:
    """Represents a knowledge artifact with metadata."""
    id: str
    content: str
    artifact_type: str  # 'entity', 'relation', 'triple', 'document', 'pattern', etc.
    source: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "artifact_type": self.artifact_type,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding,
            "confidence": self.confidence,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeArtifact':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            content=data["content"],
            artifact_type=data["artifact_type"],
            source=data["source"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data["metadata"],
            embedding=data.get("embedding"),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", "1.0.0")
        )


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database manager.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.pool = None
        self.connection_type = config.get("type", "sqlite")
        
        logger.info({
            "msg": "DatabaseManager initialized",
            "connection_type": self.connection_type,
            "host": config.get("host", "localhost"),
            "database": config.get("database", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self):
        """Initialize database connections."""
        if self.connection_type == "postgresql":
            await self._initialize_postgresql()
        elif self.connection_type == "sqlite":
            await self._initialize_sqlite()
        else:
            raise ValueError(f"Unsupported database type: {self.connection_type}")
        
        # Create tables if they don't exist
        await self._create_tables()
        
        logger.info({
            "msg": "Database connections initialized",
            "connection_type": self.connection_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["username"],
                password=self.config["password"],
                database=self.config["database"],
                min_size=1,
                max_size=self.config.get("connection_pool_size", 10),
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _initialize_sqlite(self):
        """Initialize SQLite connection."""
        try:
            db_path = Path(self.config.get("database", "./knowledge_engine.db"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For SQLite, we'll use aiosqlite for async operations
            self.db_path = db_path
            logger.info(f"SQLite database initialized at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables for knowledge storage."""
        if self.connection_type == "postgresql":
            await self._create_postgresql_tables()
        elif self.connection_type == "sqlite":
            await self._create_sqlite_tables()
    
    async def _create_postgresql_tables(self):
        """Create PostgreSQL tables."""
        async with self.pool.acquire() as conn:
            # Create knowledge artifacts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_artifacts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    source TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB,
                    embedding VECTOR(1536),
                    confidence FLOAT DEFAULT 1.0,
                    version TEXT DEFAULT '1.0.0'
                );
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_artifact_type ON knowledge_artifacts(artifact_type);
                CREATE INDEX IF NOT EXISTS idx_source ON knowledge_artifacts(source);
                CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_artifacts(created_at);
                CREATE INDEX IF NOT EXISTS idx_embedding ON knowledge_artifacts USING IVFFLAT (embedding vector_cosine_ops) WITH (lists = 100);
            """)
            
            logger.info("PostgreSQL knowledge artifacts table created")
    
    async def _create_sqlite_tables(self):
        """Create SQLite tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable JSON support
            await db.execute("CREATE TABLE IF NOT EXISTS knowledge_artifacts ("
                           "id TEXT PRIMARY KEY, "
                           "content TEXT NOT NULL, "
                           "artifact_type TEXT NOT NULL, "
                           "source TEXT, "
                           "created_at TEXT DEFAULT (datetime('now')), "
                           "updated_at TEXT DEFAULT (datetime('now')), "
                           "metadata TEXT, "  # Store as JSON string
                           "embedding TEXT, "  # Store as JSON array string
                           "confidence REAL DEFAULT 1.0, "
                           "version TEXT DEFAULT '1.0.0')")
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_artifact_type ON knowledge_artifacts(artifact_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge_artifacts(source)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_artifacts(created_at)")
            
            await db.commit()
            logger.info("SQLite knowledge artifacts table created")
    
    async def store_artifact(self, artifact: KnowledgeArtifact) -> bool:
        """Store a knowledge artifact in the database."""
        try:
            if self.connection_type == "postgresql":
                async with self.pool.acquire() as conn:
                    embedding_str = f"[{','.join(map(str, artifact.embedding))}]" if artifact.embedding else None
                    await conn.execute(
                        "INSERT INTO knowledge_artifacts (id, content, artifact_type, source, created_at, updated_at, metadata, embedding, confidence, version) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) "
                        "ON CONFLICT (id) DO UPDATE SET "
                        "content = EXCLUDED.content, "
                        "artifact_type = EXCLUDED.artifact_type, "
                        "source = EXCLUDED.source, "
                        "updated_at = EXCLUDED.updated_at, "
                        "metadata = EXCLUDED.metadata, "
                        "embedding = EXCLUDED.embedding, "
                        "confidence = EXCLUDED.confidence, "
                        "version = EXCLUDED.version",
                        artifact.id,
                        artifact.content,
                        artifact.artifact_type,
                        artifact.source,
                        artifact.created_at,
                        artifact.updated_at,
                        json.dumps(artifact.metadata),
                        embedding_str,
                        artifact.confidence,
                        artifact.version
                    )
            elif self.connection_type == "sqlite":
                async with aiosqlite.connect(self.db_path) as db:
                    embedding_str = json.dumps(artifact.embedding) if artifact.embedding else None
                    metadata_str = json.dumps(artifact.metadata)
                    
                    await db.execute(
                        "INSERT OR REPLACE INTO knowledge_artifacts (id, content, artifact_type, source, created_at, updated_at, metadata, embedding, confidence, version) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            artifact.id,
                            artifact.content,
                            artifact.artifact_type,
                            artifact.source,
                            artifact.created_at.isoformat(),
                            artifact.updated_at.isoformat(),
                            metadata_str,
                            embedding_str,
                            artifact.confidence,
                            artifact.version
                        )
                    )
                    await db.commit()
            
            logger.info({
                "msg": "Knowledge artifact stored",
                "artifact_id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to store knowledge artifact: {e}",
                "artifact_id": artifact.id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False
    
    async def retrieve_artifact(self, artifact_id: str) -> Optional[KnowledgeArtifact]:
        """Retrieve a knowledge artifact by ID."""
        try:
            if self.connection_type == "postgresql":
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT id, content, artifact_type, source, created_at, updated_at, metadata, embedding, confidence, version FROM knowledge_artifacts WHERE id = $1",
                        artifact_id
                    )
            elif self.connection_type == "sqlite":
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        "SELECT id, content, artifact_type, source, created_at, updated_at, metadata, embedding, confidence, version FROM knowledge_artifacts WHERE id = ?",
                        (artifact_id,)
                    )
                    row = await cursor.fetchone()
            
            if row:
                metadata = json.loads(row['metadata'] if self.connection_type == "postgresql" else row[6])
                embedding = json.loads(row['embedding'] if self.connection_type == "postgresql" else row[7]) if row['embedding'] if self.connection_type == "postgresql" else row[7] else None
                
                return KnowledgeArtifact(
                    id=row['id'] if self.connection_type == "postgresql" else row[0],
                    content=row['content'] if self.connection_type == "postgresql" else row[1],
                    artifact_type=row['artifact_type'] if self.connection_type == "postgresql" else row[2],
                    source=row['source'] if self.connection_type == "postgresql" else row[3],
                    created_at=datetime.fromisoformat(row['created_at'] if self.connection_type == "postgresql" else row[4]),
                    updated_at=datetime.fromisoformat(row['updated_at'] if self.connection_type == "postgresql" else row[5]),
                    metadata=metadata,
                    embedding=embedding,
                    confidence=row['confidence'] if self.connection_type == "postgresql" else row[8],
                    version=row['version'] if self.connection_type == "postgresql" else row[9]
                )
            
            return None
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to retrieve knowledge artifact: {e}",
                "artifact_id": artifact_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
    
    async def search_artifacts(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[KnowledgeArtifact]:
        """Search for knowledge artifacts."""
        try:
            conditions = ["TRUE"]
            params = []
            param_index = 1
            
            if artifact_type:
                if self.connection_type == "postgresql":
                    conditions.append(f"artifact_type = ${param_index}")
                else:
                    conditions.append(f"artifact_type = ?")
                params.append(artifact_type)
                param_index += 1
            
            if query:
                if self.connection_type == "postgresql":
                    conditions.append(f"content ILIKE ${param_index}")
                else:
                    conditions.append(f"content LIKE ?")
                params.append(f"%{query}%")
                param_index += 1
            
            where_clause = " AND ".join(conditions)
            
            if self.connection_type == "postgresql":
                query_sql = f"""
                    SELECT id, content, artifact_type, source, created_at, updated_at, metadata, embedding, confidence, version
                    FROM knowledge_artifacts
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """
                params.extend([limit, offset])
                
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(query_sql, *params)
            elif self.connection_type == "sqlite":
                query_sql = f"""
                    SELECT id, content, artifact_type, source, created_at, updated_at, metadata, embedding, confidence, version
                    FROM knowledge_artifacts
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])
                
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(query_sql, params)
                    rows = await cursor.fetchall()
            
            artifacts = []
            for row in rows:
                metadata = json.loads(row['metadata'] if self.connection_type == "postgresql" else row[6])
                embedding = json.loads(row['embedding'] if self.connection_type == "postgresql" else row[7]) if (row['embedding'] if self.connection_type == "postgresql" else row[7]) else None
                
                artifact = KnowledgeArtifact(
                    id=row['id'] if self.connection_type == "postgresql" else row[0],
                    content=row['content'] if self.connection_type == "postgresql" else row[1],
                    artifact_type=row['artifact_type'] if self.connection_type == "postgresql" else row[2],
                    source=row['source'] if self.connection_type == "postgresql" else row[3],
                    created_at=datetime.fromisoformat(row['created_at'] if self.connection_type == "postgresql" else row[4]),
                    updated_at=datetime.fromisoformat(row['updated_at'] if self.connection_type == "postgresql" else row[5]),
                    metadata=metadata,
                    embedding=embedding,
                    confidence=row['confidence'] if self.connection_type == "postgresql" else row[8],
                    version=row['version'] if self.connection_type == "postgresql" else row[9]
                )
                artifacts.append(artifact)
            
            return artifacts
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to search knowledge artifacts: {e}",
                "query": query,
                "artifact_type": artifact_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []
    
    async def close(self):
        """Close database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
        logger.info("Database connections closed")


class VectorStoreManager:
    """Manages vector store operations for embeddings and similarity search."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store manager.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.client = None
        self.collection_name = config.get("collection_name", "knowledge_artifacts")
        
        logger.info({
            "msg": "VectorStoreManager initialized",
            "vector_store_type": config.get("type", "qdrant"),
            "host": config.get("host", "localhost"),
            "collection_name": self.collection_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self):
        """Initialize vector store connection."""
        if self.config.get("type", "qdrant") == "qdrant":
            await self._initialize_qdrant()
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.get('type')}")
        
        # Create collection if it doesn't exist
        await self._create_collection()
        
        logger.info({
            "msg": "Vector store initialized",
            "collection_name": self.collection_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def _initialize_qdrant(self):
        """Initialize Qdrant client."""
        try:
            self.client = AsyncQdrantClient(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6333),
                prefer_grpc=True
            )
            
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def _create_collection(self):
        """Create vector collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                # Create collection
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.config.get("vector_size", 1536),
                        distance=models.Distance.COSINE
                    )
                )
                
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    async def store_embedding(
        self,
        artifact_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store an embedding in the vector store."""
        try:
            # Prepare point for Qdrant
            points = [
                models.PointStruct(
                    id=artifact_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "artifact_id": artifact_id,
                        "metadata": metadata,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            ]
            
            # Upload to Qdrant
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info({
                "msg": "Embedding stored in vector store",
                "artifact_id": artifact_id,
                "vector_size": len(embedding),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to store embedding: {e}",
                "artifact_id": artifact_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False
    
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        try:
            # Prepare filters if provided
            qdrant_filters = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchText(text=value)
                            )
                        )
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if filter_conditions:
                    qdrant_filters = models.Filter(must=filter_conditions)
            
            # Search in Qdrant
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filters
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "artifact_id": result.payload.get("artifact_id", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score,
                    "timestamp": result.payload.get("timestamp")
                })
            
            logger.info({
                "msg": "Similarity search completed",
                "query_vector_size": len(query_embedding),
                "results_count": len(results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return results
            
        except Exception as e:
            logger.error({
                "msg": f"Similarity search failed: {e}",
                "query_vector_size": len(query_embedding),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []
    
    async def delete_embedding(self, artifact_id: str) -> bool:
        """Delete an embedding from the vector store."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[artifact_id])
            )
            
            logger.info({
                "msg": "Embedding deleted from vector store",
                "artifact_id": artifact_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to delete embedding: {e}",
                "artifact_id": artifact_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False
    
    async def close(self):
        """Close vector store connections."""
        if self.client:
            # Qdrant client doesn't have a specific close method
            logger.info("Vector store connections closed")


class CacheManager:
    """Manages caching for performance optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.client = None
        self.cache_ttl = config.get("ttl_seconds", 3600)
        
        logger.info({
            "msg": "CacheManager initialized",
            "cache_type": config.get("type", "redis"),
            "host": config.get("host", "localhost"),
            "port": config.get("port", 6379),
            "ttl_seconds": self.cache_ttl,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self):
        """Initialize cache connection."""
        if self.config.get("type", "redis") == "redis":
            await self._initialize_redis()
        else:
            raise ValueError(f"Unsupported cache type: {self.config.get('type')}")
        
        logger.info({
            "msg": "Cache initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def _initialize_redis(self):
        """Initialize Redis client."""
        try:
            self.client = redis.Redis(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                db=self.config.get("db", 0),
                decode_responses=True
            )
            
            # Test connection
            await self.client.ping()
            
            logger.info("Redis client initialized and connection tested")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            value = await self.client.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key}")
            else:
                logger.debug(f"Cache miss for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.cache_ttl
            await self.client.setex(key, ttl, value)
            
            logger.debug(f"Cache set for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            result = await self.client.delete(key)
            logger.debug(f"Cache delete for key: {key}, result: {result}")
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            keys = await self.client.keys(pattern)
            if keys:
                result = await self.client.delete(*keys)
                logger.info(f"Cleared {result} keys matching pattern: {pattern}")
                return result
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern failed for pattern {pattern}: {e}")
            return 0
    
    async def close(self):
        """Close cache connections."""
        if self.client:
            await self.client.close()
            logger.info("Cache connections closed")


class KnowledgeStorageEngine:
    """
    Main storage engine that coordinates database, vector store, and caching operations.
    
    Provides unified interface for storing, retrieving, and searching knowledge artifacts
    with proper consistency between all storage layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge storage engine.
        
        Args:
            config: Complete storage configuration
        """
        self.config = config
        
        # Initialize storage components
        self.database = DatabaseManager(config.get("database", {}))
        self.vector_store = VectorStoreManager(config.get("vector_store", {}))
        self.cache = CacheManager(config.get("cache", {}))
        
        logger.info({
            "msg": "KnowledgeStorageEngine initialized",
            "config": {
                "database_type": config.get("database", {}).get("type"),
                "vector_store_type": config.get("vector_store", {}).get("type"),
                "cache_type": config.get("cache", {}).get("type")
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self):
        """Initialize all storage components."""
        # Initialize in order: database, vector store, cache
        await self.database.initialize()
        await self.vector_store.initialize()
        await self.cache.initialize()
        
        logger.info({
            "msg": "Knowledge storage engine initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def store_knowledge_artifact(
        self,
        content: str,
        artifact_type: str,
        source: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Store a knowledge artifact with all associated data.
        
        Args:
            content: Content of the knowledge artifact
            artifact_type: Type of artifact ('entity', 'relation', 'triple', etc.)
            source: Source of the knowledge
            embedding: Optional embedding vector
            metadata: Optional metadata
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            Artifact ID of the stored artifact
        """
        artifact_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        # Create knowledge artifact
        artifact = KnowledgeArtifact(
            id=artifact_id,
            content=content,
            artifact_type=artifact_type,
            source=source,
            created_at=created_at,
            updated_at=created_at,
            metadata=metadata or {},
            embedding=embedding,
            confidence=confidence
        )
        
        # Store in database
        db_success = await self.database.store_artifact(artifact)
        
        # Store embedding in vector store if provided
        embedding_success = True
        if embedding:
            embedding_success = await self.vector_store.store_embedding(
                artifact_id=artifact_id,
                embedding=embedding,
                content=content,
                metadata=artifact.metadata
            )
        
        # Invalidate any related cache entries
        await self.cache.clear_pattern(f"search:*:{artifact_type}")
        
        success = db_success and embedding_success
        
        logger.info({
            "msg": "Knowledge artifact stored",
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "content_length": len(content),
            "embedding_provided": embedding is not None,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        if success:
            return artifact_id
        else:
            raise RuntimeError("Failed to store knowledge artifact in one or more storage systems")
    
    async def retrieve_knowledge_artifact(self, artifact_id: str) -> Optional[KnowledgeArtifact]:
        """
        Retrieve a knowledge artifact by ID.
        
        Args:
            artifact_id: ID of the artifact to retrieve
            
        Returns:
            KnowledgeArtifact if found, None otherwise
        """
        # First check cache
        cached_data = await self.cache.get(f"artifact:{artifact_id}")
        if cached_data:
            try:
                artifact_dict = json.loads(cached_data)
                return KnowledgeArtifact.from_dict(artifact_dict)
            except json.JSONDecodeError:
                logger.warning(f"Invalid cached data for artifact {artifact_id}, retrieving from database")
        
        # Retrieve from database
        artifact = await self.database.retrieve_artifact(artifact_id)
        
        if artifact:
            # Cache the result
            await self.cache.set(
                f"artifact:{artifact_id}",
                json.dumps(artifact.to_dict()),
                ttl=3600  # Cache for 1 hour
            )
        
        return artifact
    
    async def search_knowledge_artifacts(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeArtifact]:
        """
        Search for knowledge artifacts using text and semantic search.
        
        Args:
            query: Search query
            artifact_type: Optional type filter
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching KnowledgeArtifacts
        """
        # Create cache key
        cache_key = f"search:{hash(query)}:{artifact_type}:{top_k}:{hash(str(filters))}"
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            try:
                result_data = json.loads(cached_result)
                artifacts = [KnowledgeArtifact.from_dict(data) for data in result_data]
                logger.debug(f"Cache hit for search: {query}")
                return artifacts
            except json.JSONDecodeError:
                logger.warning(f"Invalid cached search result for query: {query}")
        
        # Perform search in database
        db_results = await self.database.search_artifacts(
            query=query,
            artifact_type=artifact_type,
            limit=top_k
        )
        
        # If we have an embedding for the query, also search vector store
        embedding_results = []
        if len(query) > 10:  # Only embed if query is substantial
            # In a real implementation, we would generate an embedding for the query
            # For now, we'll skip vector search
            pass
        
        # Combine and deduplicate results
        all_results = db_results  # Add vector results when available
        
        # Remove duplicates based on ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        # Limit to top_k
        final_results = unique_results[:top_k]
        
        # Cache the results
        result_dicts = [artifact.to_dict() for artifact in final_results]
        await self.cache.set(
            cache_key,
            json.dumps(result_dicts),
            ttl=300  # Cache for 5 minutes
        )
        
        logger.info({
            "msg": "Knowledge artifact search completed",
            "query_length": len(query),
            "artifact_type": artifact_type,
            "results_count": len(final_results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return final_results
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        artifact_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query_embedding: Embedding vector to search for
            top_k: Number of results to return
            artifact_type: Optional type filter
            filters: Optional metadata filters
            
        Returns:
            List of search results with similarity scores
        """
        # Add type filter to filters if specified
        search_filters = filters or {}
        if artifact_type:
            search_filters["artifact_type"] = artifact_type
        
        # Search in vector store
        vector_results = await self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=search_filters
        )
        
        # Get full artifact details from database for each result
        detailed_results = []
        for result in vector_results:
            artifact = await self.database.retrieve_artifact(result["artifact_id"])
            if artifact:
                detailed_results.append({
                    "artifact": artifact,
                    "similarity_score": result["score"],
                    "vector_id": result["id"]
                })
        
        logger.info({
            "msg": "Semantic search completed",
            "query_vector_size": len(query_embedding),
            "results_count": len(detailed_results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return detailed_results
    
    async def close(self):
        """Close all storage connections."""
        logger.info({
            "msg": "Closing knowledge storage engine connections",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close in reverse order
        await self.cache.close()
        await self.vector_store.close()
        await self.database.close()
        
        logger.info({
            "msg": "Knowledge storage engine connections closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })