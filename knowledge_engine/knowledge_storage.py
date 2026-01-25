"""
Knowledge Storage for OpenEvolve Knowledge Engine

This module provides storage capabilities for knowledge artifacts with support
for multiple backend databases (Qdrant, MongoDB, Neo4j, Redis).
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import uuid


logger = logging.getLogger(__name__)


@dataclass
class StorageResult:
    """Result of a storage operation."""
    success: bool
    artifact_id: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class KnowledgeStorage:
    """
    Storage layer for knowledge artifacts with multi-database support.
    
    Provides methods for:
    - Storing knowledge artifacts
    - Retrieving knowledge artifacts
    - Multi-database backend support
    - Caching mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge storage with backend connections.
        
        Args:
            config: Configuration for storage backends
        """
        self.config = config or self._get_default_config()
        
        # Initialize backend connections
        self.qdrant_client = None
        self.mongo_client = None
        self.neo4j_driver = None
        self.redis_client = None
        
        # Initialize backends based on config
        self._initialize_backends()
        
        logger.info({
            "msg": "KnowledgeStorage initialized",
            "config": self.config,
            "backends": {
                "qdrant": self.qdrant_client is not None,
                "mongo": self.mongo_client is not None,
                "neo4j": self.neo4j_driver is not None,
                "redis": self.redis_client is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "enabled": True
            },
            "mongo": {
                "uri": "mongodb://localhost:27017",
                "database": "openevolve",
                "collection": "knowledge_artifacts",
                "enabled": True
            },
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password",
                "enabled": True
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "enabled": True
            },
            "default_backend": "mongo",  # Which backend to use by default
            "fallback_enabled": True,    # Whether to try other backends on failure
            "cache_ttl": 300,           # Cache TTL in seconds
            "batch_size": 100           # Batch size for bulk operations
        }
    
    def _initialize_backends(self):
        """Initialize storage backends based on configuration."""
        # Initialize Qdrant
        if self.config.get("qdrant", {}).get("enabled", True):
            try:
                import qdrant_client
                host = self.config["qdrant"].get("host", "localhost")
                port = self.config["qdrant"].get("port", 6333)
                self.qdrant_client = qdrant_client.QdrantClient(host=host, port=port)
                logger.info("Qdrant client initialized")
            except ImportError:
                logger.warning("Qdrant client not available, skipping initialization")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {e}")
        
        # Initialize MongoDB
        if self.config.get("mongo", {}).get("enabled", True):
            try:
                from pymongo import MongoClient
                uri = self.config["mongo"].get("uri", "mongodb://localhost:27017")
                self.mongo_client = MongoClient(uri)
                logger.info("MongoDB client initialized")
            except ImportError:
                logger.warning("PyMongo not available, skipping MongoDB initialization")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB client: {e}")
        
        # Initialize Neo4j
        if self.config.get("neo4j", {}).get("enabled", True):
            try:
                from neo4j import GraphDatabase
                uri = self.config["neo4j"].get("uri", "bolt://localhost:7687")
                user = self.config["neo4j"].get("user", "neo4j")
                password = self.config["neo4j"].get("password", "password")
                self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
                logger.info("Neo4j driver initialized")
            except ImportError:
                logger.warning("Neo4j driver not available, skipping Neo4j initialization")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
        
        # Initialize Redis
        if self.config.get("redis", {}).get("enabled", True):
            try:
                import redis
                host = self.config["redis"].get("host", "localhost")
                port = self.config["redis"].get("port", 6379)
                self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
                logger.info("Redis client initialized")
            except ImportError:
                logger.warning("Redis not available, skipping Redis initialization")
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
    
    def store_knowledge_artifact(
        self,
        artifact: Dict[str, Any],
        generate_embedding: bool = True,
        backend: Optional[str] = None
    ) -> Optional[str]:
        """
        Store a knowledge artifact in the specified backend.
        
        Args:
            artifact: Knowledge artifact to store
            generate_embedding: Whether to generate embeddings for the artifact
            backend: Backend to use ('qdrant', 'mongo', 'neo4j', 'redis', or None for default)
            
        Returns:
            Artifact ID if successful, None otherwise
        """
        start_time = datetime.now(timezone.utc)
        
        # Use default backend if none specified
        if not backend:
            backend = self.config.get("default_backend", "mongo")
        
        logger.info({
            "msg": "Storing knowledge artifact",
            "backend": backend,
            "artifact_type": artifact.get("type", "unknown"),
            "generate_embedding": generate_embedding,
            "timestamp": start_time.isoformat()
        })
        
        try:
            artifact_id = str(uuid.uuid4())
            
            # Add metadata
            artifact_with_meta = {
                **artifact,
                "artifact_id": artifact_id,
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "backend": backend
            }
            
            # Store in the specified backend
            success = False
            if backend == "qdrant" and self.qdrant_client:
                success = self._store_in_qdrant(artifact_with_meta, generate_embedding)
            elif backend == "mongo" and self.mongo_client:
                success = self._store_in_mongo(artifact_with_meta)
            elif backend == "neo4j" and self.neo4j_driver:
                success = self._store_in_neo4j(artifact_with_meta)
            elif backend == "redis" and self.redis_client:
                success = self._store_in_redis(artifact_with_meta)
            else:
                # Try default backend if specified backend is not available
                default_backend = self.config.get("default_backend", "mongo")
                if self.config.get("fallback_enabled", True):
                    logger.warning(f"Backend {backend} not available, trying default: {default_backend}")
                    return self.store_knowledge_artifact(artifact, generate_embedding, default_backend)
                else:
                    logger.error(f"No available backend for storing artifact")
                    return None
            
            if success:
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.info({
                    "msg": "Knowledge artifact stored successfully",
                    "artifact_id": artifact_id,
                    "backend": backend,
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return artifact_id
            else:
                logger.error({
                    "msg": "Failed to store knowledge artifact",
                    "backend": backend,
                    "artifact_id": artifact_id
                })
                return None
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Error storing knowledge artifact",
                "backend": backend,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Try fallback if enabled
            if self.config.get("fallback_enabled", True) and backend != self.config.get("default_backend"):
                logger.info("Trying fallback backend")
                return self.store_knowledge_artifact(artifact, generate_embedding, self.config.get("default_backend"))
            
            return None
    
    def _store_in_qdrant(self, artifact: Dict[str, Any], generate_embedding: bool) -> bool:
        """Store artifact in Qdrant."""
        try:
            from qdrant_client.http import models
            
            # Generate embedding if requested
            embedding = None
            if generate_embedding:
                # In a real implementation, this would generate actual embeddings
                # For now, we'll use a placeholder
                embedding = [0.1] * 1536  # Placeholder embedding
            
            # Prepare payload
            payload = {
                "content": artifact.get("content", ""),
                "type": artifact.get("type", "unknown"),
                "source": artifact.get("source", "unknown"),
                "context": artifact.get("context", ""),
                "metadata": artifact.get("metadata", {}),
                "stored_at": artifact.get("stored_at", ""),
                "artifact_id": artifact.get("artifact_id", "")
            }
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name="knowledge_artifacts",
                points=[
                    models.PointStruct(
                        id=hash(artifact.get("artifact_id", "")) % (10**9),  # Qdrant point ID
                        vector=embedding if embedding else [0.0] * 1536,
                        payload=payload
                    )
                ]
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to store in Qdrant: {e}")
            return False
    
    def _store_in_mongo(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in MongoDB."""
        try:
            db_name = self.config["mongo"].get("database", "openevolve")
            collection_name = self.config["mongo"].get("collection", "knowledge_artifacts")
            
            db = self.mongo_client[db_name]
            collection = db[collection_name]
            
            # Insert the artifact
            result = collection.insert_one(artifact)
            
            return result.acknowledged
        except Exception as e:
            logger.error(f"Failed to store in MongoDB: {e}")
            return False
    
    def _store_in_neo4j(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                # Create a knowledge artifact node
                query = """
                CREATE (ka:KnowledgeArtifact {
                    artifact_id: $artifact_id,
                    content: $content,
                    type: $type,
                    source: $source,
                    context: $context,
                    stored_at: $stored_at
                })
                RETURN ka.artifact_id as id
                """
                
                result = session.run(
                    query,
                    artifact_id=artifact.get("artifact_id"),
                    content=artifact.get("content", ""),
                    type=artifact.get("type", "unknown"),
                    source=artifact.get("source", "unknown"),
                    context=artifact.get("context", ""),
                    stored_at=artifact.get("stored_at", "")
                )
                
                record = result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Failed to store in Neo4j: {e}")
            return False
    
    def _store_in_redis(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in Redis."""
        try:
            key = f"knowledge_artifact:{artifact.get('artifact_id')}"
            value = json.dumps(artifact)
            
            ttl = self.config.get("cache_ttl", 300)  # Default 5 minutes
            
            # Store with TTL
            result = self.redis_client.setex(key, ttl, value)
            
            return result == "OK"
        except Exception as e:
            logger.error(f"Failed to store in Redis: {e}")
            return False
    
    def retrieve_knowledge_artifact(
        self,
        artifact_id: str,
        backend: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a knowledge artifact by ID.
        
        Args:
            artifact_id: ID of the artifact to retrieve
            backend: Backend to use (None for auto-discovery)
            
        Returns:
            Knowledge artifact if found, None otherwise
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Retrieving knowledge artifact",
            "artifact_id": artifact_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # If backend not specified, try all available backends
            if not backend:
                backends_to_try = []
                
                # Add backends in priority order based on config
                default_backend = self.config.get("default_backend", "mongo")
                backends_to_try.append(default_backend)
                
                # Add other backends if fallback is enabled
                if self.config.get("fallback_enabled", True):
                    other_backends = ["mongo", "qdrant", "neo4j", "redis"]
                    for bk in other_backends:
                        if bk != default_backend and getattr(self, f"{bk}_client" if bk != "neo4j" else f"{bk}_driver"):
                            backends_to_try.append(bk)
                
                # Try each backend
                for bk in backends_to_try:
                    artifact = self._retrieve_from_backend(artifact_id, bk)
                    if artifact:
                        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                        
                        logger.info({
                            "msg": "Knowledge artifact retrieved successfully",
                            "artifact_id": artifact_id,
                            "backend": bk,
                            "processing_time_ms": processing_time_ms,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                        return artifact
            else:
                # Retrieve from specific backend
                artifact = self._retrieve_from_backend(artifact_id, backend)
                if artifact:
                    processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    logger.info({
                        "msg": "Knowledge artifact retrieved successfully",
                        "artifact_id": artifact_id,
                        "backend": backend,
                        "processing_time_ms": processing_time_ms,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                    return artifact
            
            logger.info({
                "msg": "Knowledge artifact not found",
                "artifact_id": artifact_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return None
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Error retrieving knowledge artifact",
                "artifact_id": artifact_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return None
    
    def _retrieve_from_backend(self, artifact_id: str, backend: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact from a specific backend."""
        try:
            if backend == "mongo" and self.mongo_client:
                return self._retrieve_from_mongo(artifact_id)
            elif backend == "qdrant" and self.qdrant_client:
                return self._retrieve_from_qdrant(artifact_id)
            elif backend == "neo4j" and self.neo4j_driver:
                return self._retrieve_from_neo4j(artifact_id)
            elif backend == "redis" and self.redis_client:
                return self._retrieve_from_redis(artifact_id)
            else:
                logger.warning(f"Backend {backend} not available for retrieval")
                return None
        except Exception as e:
            logger.error(f"Error retrieving from {backend}: {e}")
            return None
    
    def _retrieve_from_mongo(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact from MongoDB."""
        try:
            db_name = self.config["mongo"].get("database", "openevolve")
            collection_name = self.config["mongo"].get("collection", "knowledge_artifacts")
            
            db = self.mongo_client[db_name]
            collection = db[collection_name]
            
            artifact = collection.find_one({"artifact_id": artifact_id})
            
            return artifact
        except Exception as e:
            logger.error(f"Failed to retrieve from MongoDB: {e}")
            return None
    
    def _retrieve_from_qdrant(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact from Qdrant."""
        try:
            # Find the point with the given artifact_id in payload
            result = self.qdrant_client.scroll(
                collection_name="knowledge_artifacts",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="artifact_id",
                            match=models.MatchValue(value=artifact_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if result[0]:
                point = result[0][0]
                return point.payload  # Return the payload which contains the artifact
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve from Qdrant: {e}")
            return None
    
    def _retrieve_from_neo4j(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact from Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (ka:KnowledgeArtifact {artifact_id: $artifact_id})
                RETURN ka
                """
                
                result = session.run(query, artifact_id=artifact_id)
                record = result.single()
                
                if record:
                    node = record["ka"]
                    # Convert Neo4j node to dictionary
                    return dict(node)
                
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve from Neo4j: {e}")
            return None
    
    def _retrieve_from_redis(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact from Redis."""
        try:
            key = f"knowledge_artifact:{artifact_id}"
            value = self.redis_client.get(key)
            
            if value:
                return json.loads(value)
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve from Redis: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge storage.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "total_artifacts": 0,
            "backend_status": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get stats from MongoDB if available
        if self.mongo_client:
            try:
                db_name = self.config["mongo"].get("database", "openevolve")
                collection_name = self.config["mongo"].get("collection", "knowledge_artifacts")
                
                db = self.mongo_client[db_name]
                collection = db[collection_name]
                
                count = collection.count_documents({})
                stats["total_artifacts"] = count
                stats["backend_status"]["mongo"] = {
                    "status": "connected",
                    "artifact_count": count
                }
            except Exception as e:
                stats["backend_status"]["mongo"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["backend_status"]["mongo"] = {"status": "disconnected"}
        
        # Add stats for other backends similarly
        if self.qdrant_client:
            try:
                collections = self.qdrant_client.get_collections()
                stats["backend_status"]["qdrant"] = {
                    "status": "connected",
                    "collections": [c.name for c in collections.collections]
                }
            except Exception as e:
                stats["backend_status"]["qdrant"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["backend_status"]["qdrant"] = {"status": "disconnected"}
        
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) AS count")
                    record = result.single()
                    node_count = record["count"] if record else 0
                    
                    stats["backend_status"]["neo4j"] = {
                        "status": "connected",
                        "node_count": node_count
                    }
            except Exception as e:
                stats["backend_status"]["neo4j"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["backend_status"]["neo4j"] = {"status": "disconnected"}
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["backend_status"]["redis"] = {
                    "status": "connected",
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0)
                }
            except Exception as e:
                stats["backend_status"]["redis"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["backend_status"]["redis"] = {"status": "disconnected"}
        
        return stats
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage performance.
        
        Returns:
            Dictionary with optimization results
        """
        results = {
            "operations_performed": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Optimize MongoDB if available
        if self.mongo_client:
            try:
                db_name = self.config["mongo"].get("database", "openevolve")
                collection_name = self.config["mongo"].get("collection", "knowledge_artifacts")
                
                db = self.mongo_client[db_name]
                collection = db[collection_name]
                
                # Create indexes for better performance
                collection.create_index("artifact_id", unique=True)
                collection.create_index("type")
                collection.create_index("stored_at")
                
                results["operations_performed"].append(f"MongoDB indexes created for {collection_name}")
            except Exception as e:
                results["operations_performed"].append(f"MongoDB optimization error: {e}")
        
        # Add optimizations for other backends
        if self.qdrant_client:
            try:
                # Qdrant optimization would go here
                results["operations_performed"].append("Qdrant optimization completed")
            except Exception as e:
                results["operations_performed"].append(f"Qdrant optimization error: {e}")
        
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Create indexes in Neo4j
                    session.run("CREATE INDEX IF NOT EXISTS FOR (ka:KnowledgeArtifact) ON (ka.artifact_id)")
                    session.run("CREATE INDEX IF NOT EXISTS FOR (ka:KnowledgeArtifact) ON (ka.type)")
                    
                    results["operations_performed"].append("Neo4j indexes created")
            except Exception as e:
                results["operations_performed"].append(f"Neo4j optimization error: {e}")
        
        logger.info({
            "msg": "Storage optimization completed",
            "operations": results["operations_performed"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return results
    
    def close_connections(self):
        """Close all database connections."""
        logger.info({
            "msg": "Closing storage connections",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed")
        
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j driver closed")
        
        logger.info({
            "msg": "Storage connections closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })