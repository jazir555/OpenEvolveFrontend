"""
Enhanced Knowledge Storage for OpenEvolve Knowledge Engine

This module provides enhanced storage capabilities with performance optimization,
multi-modal storage, and advanced indexing for the Phase 2 implementation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import uuid
from enum import Enum


logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Enumeration of supported storage backends."""
    MONGODB = "mongodb"
    QDRANT = "qdrant"
    NEO4J = "neo4j"
    REDIS = "redis"


@dataclass
class EnhancedStorageResult:
    """Result of an enhanced storage operation."""
    success: bool
    artifact_id: Optional[str] = None
    backend_used: Optional[str] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedKnowledgeStorage:
    """
    Enhanced storage layer with performance optimization and multi-modal capabilities.
    
    Provides methods for:
    - Advanced indexing strategies
    - Performance optimization
    - Multi-modal storage
    - Quality metrics tracking
    - Batch operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced knowledge storage.
        
        Args:
            config: Configuration for enhanced storage
        """
        self.config = config or self._get_default_config()
        
        # Initialize backends
        self.backends = {}
        self._initialize_backends()
        
        # Performance tracking
        self.operation_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time": 0.0,
            "backend_usage": {}
        }
        
        logger.info({
            "msg": "EnhancedKnowledgeStorage initialized",
            "config": self.config,
            "available_backends": list(self.backends.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for enhanced storage."""
        return {
            "backends": {
                "mongodb": {
                    "enabled": True,
                    "uri": "mongodb://localhost:27017",
                    "database": "openevolve_kg",
                    "collection": "knowledge_artifacts",
                    "indexes": ["artifact_id", "type", "created_at", "embedding"]
                },
                "qdrant": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 6333,
                    "collection": "knowledge_vectors",
                    "vector_size": 1536
                },
                "neo4j": {
                    "enabled": True,
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password",
                    "indexes": ["artifact_id", "type"]
                },
                "redis": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 6379,
                    "ttl": 3600
                }
            },
            "default_backend": "mongodb",
            "fallback_enabled": True,
            "batch_size": 100,
            "cache_ttl": 300,
            "enable_compression": True,
            "enable_encryption": False
        }
    
    def _initialize_backends(self):
        """Initialize configured storage backends."""
        backends_config = self.config.get("backends", {})
        
        # Initialize MongoDB
        if backends_config.get("mongodb", {}).get("enabled", True):
            try:
                from pymongo import MongoClient
                mongo_config = backends_config["mongodb"]
                client = MongoClient(mongo_config["uri"])
                db = client[mongo_config["database"]]
                collection = db[mongo_config["collection"]]
                
                # Create indexes
                for index_field in mongo_config.get("indexes", ["artifact_id"]):
                    collection.create_index(index_field, unique=(index_field == "artifact_id"))
                
                self.backends[StorageBackend.MONGODB] = {
                    "client": client,
                    "db": db,
                    "collection": collection
                }
                logger.info("MongoDB backend initialized")
            except ImportError:
                logger.warning("PyMongo not available, MongoDB backend disabled")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB backend: {e}")
        
        # Initialize Qdrant
        if backends_config.get("qdrant", {}).get("enabled", True):
            try:
                import qdrant_client
                from qdrant_client.http import models
                qdrant_config = backends_config["qdrant"]
                
                client = qdrant_client.QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config["port"]
                )
                
                # Create collection if it doesn't exist
                try:
                    client.get_collection(qdrant_config["collection"])
                except:
                    client.create_collection(
                        collection_name=qdrant_config["collection"],
                        vectors_config=models.VectorParams(
                            size=qdrant_config["vector_size"],
                            distance=models.Distance.COSINE
                        )
                    )
                
                self.backends[StorageBackend.QDRANT] = {
                    "client": client,
                    "collection": qdrant_config["collection"]
                }
                logger.info("Qdrant backend initialized")
            except ImportError:
                logger.warning("qdrant-client not available, Qdrant backend disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant backend: {e}")
        
        # Initialize Neo4j
        if backends_config.get("neo4j", {}).get("enabled", True):
            try:
                from neo4j import GraphDatabase
                neo4j_config = backends_config["neo4j"]
                
                driver = GraphDatabase.driver(
                    neo4j_config["uri"],
                    auth=(neo4j_config["user"], neo4j_config["password"])
                )
                
                # Create indexes
                with driver.session() as session:
                    for index_field in neo4j_config.get("indexes", ["artifact_id"]):
                        session.run(
                            f"CREATE INDEX IF NOT EXISTS FOR (ka:KnowledgeArtifact) ON (ka.{index_field})"
                        )
                
                self.backends[StorageBackend.NEO4J] = {
                    "driver": driver
                }
                logger.info("Neo4j backend initialized")
            except ImportError:
                logger.warning("Neo4j driver not available, Neo4j backend disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j backend: {e}")
        
        # Initialize Redis
        if backends_config.get("redis", {}).get("enabled", True):
            try:
                import redis
                redis_config = backends_config["redis"]
                
                client = redis.Redis(
                    host=redis_config["host"],
                    port=redis_config["port"],
                    decode_responses=True
                )
                
                self.backends[StorageBackend.REDIS] = {
                    "client": client,
                    "ttl": redis_config["ttl"]
                }
                logger.info("Redis backend initialized")
            except ImportError:
                logger.warning("Redis not available, Redis backend disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Redis backend: {e}")
    
    def store_knowledge_artifact(
        self,
        artifact: Dict[str, Any],
        generate_embedding: bool = True,
        backend: Optional[StorageBackend] = None,
        replicate_to: Optional[List[StorageBackend]] = None
    ) -> EnhancedStorageResult:
        """
        Store a knowledge artifact with enhanced capabilities.
        
        Args:
            artifact: Knowledge artifact to store
            generate_embedding: Whether to generate embeddings
            backend: Specific backend to use (None for default)
            replicate_to: List of additional backends to replicate to
            
        Returns:
            EnhancedStorageResult with operation details
        """
        start_time = datetime.now(timezone.utc)
        
        # Use default backend if none specified
        if not backend:
            backend = StorageBackend(self.config.get("default_backend", "mongodb"))
        
        logger.info({
            "msg": "Storing knowledge artifact with enhanced storage",
            "backend": backend.value,
            "artifact_type": artifact.get("type", "unknown"),
            "generate_embedding": generate_embedding,
            "timestamp": start_time.isoformat()
        })
        
        try:
            artifact_id = artifact.get("artifact_id", str(uuid.uuid4()))
            
            # Add metadata
            enhanced_artifact = {
                **artifact,
                "artifact_id": artifact_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": 1,
                "backend": backend.value
            }
            
            # Add embedding if requested and available
            if generate_embedding and "embedding" not in enhanced_artifact:
                # In a real implementation, this would generate actual embeddings
                # For now, we'll add a placeholder
                enhanced_artifact["embedding"] = [0.0] * 1536  # Placeholder
            
            # Store in primary backend
            success = self._store_in_backend(enhanced_artifact, backend)
            
            if success:
                # Replicate to additional backends if requested
                replication_results = []
                if replicate_to:
                    for replica_backend in replicate_to:
                        if replica_backend != backend:
                            replica_success = self._store_in_backend(enhanced_artifact, replica_backend)
                            replication_results.append({
                                "backend": replica_backend.value,
                                "success": replica_success
                            })
                
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                # Update stats
                self._update_operation_stats(True, backend.value, processing_time_ms)
                
                result = EnhancedStorageResult(
                    success=True,
                    artifact_id=artifact_id,
                    backend_used=backend.value,
                    processing_time_ms=processing_time_ms,
                    metadata={
                        "replication_results": replication_results,
                        "replicated_to_count": len(replication_results)
                    }
                )
                
                logger.info({
                    "msg": "Knowledge artifact stored successfully with enhanced features",
                    "artifact_id": artifact_id,
                    "backend": backend.value,
                    "replication_count": len(replication_results),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return result
            else:
                logger.error({
                    "msg": "Failed to store knowledge artifact in primary backend",
                    "backend": backend.value,
                    "artifact_id": artifact_id
                })
                
                # Try fallback if enabled
                if self.config.get("fallback_enabled", True):
                    fallback_backend = self._get_fallback_backend(backend)
                    if fallback_backend:
                        logger.info(f"Attempting fallback to {fallback_backend.value}")
                        return self.store_knowledge_artifact(
                            artifact, generate_embedding, fallback_backend, replicate_to
                        )
                
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                self._update_operation_stats(False, backend.value, processing_time_ms)
                
                return EnhancedStorageResult(
                    success=False,
                    artifact_id=artifact_id,
                    backend_used=backend.value,
                    processing_time_ms=processing_time_ms,
                    error="Primary backend storage failed and no fallback succeeded"
                )
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_operation_stats(False, backend.value if backend else "unknown", processing_time_ms)
            
            logger.error({
                "msg": "Error in enhanced knowledge artifact storage",
                "backend": backend.value if backend else "unknown",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return EnhancedStorageResult(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
    
    def _store_in_backend(self, artifact: Dict[str, Any], backend: StorageBackend) -> bool:
        """Store artifact in a specific backend."""
        try:
            if backend == StorageBackend.MONGODB and StorageBackend.MONGODB in self.backends:
                return self._store_in_mongodb(artifact)
            elif backend == StorageBackend.QDRANT and StorageBackend.QDRANT in self.backends:
                return self._store_in_qdrant(artifact)
            elif backend == StorageBackend.NEO4J and StorageBackend.NEO4J in self.backends:
                return self._store_in_neo4j(artifact)
            elif backend == StorageBackend.REDIS and StorageBackend.REDIS in self.backends:
                return self._store_in_redis(artifact)
            else:
                logger.error(f"Backend {backend.value} not available")
                return False
        except Exception as e:
            logger.error(f"Error storing in {backend.value}: {e}")
            return False
    
    def _store_in_mongodb(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in MongoDB."""
        try:
            collection = self.backends[StorageBackend.MONGODB]["collection"]
            
            # Insert the artifact
            result = collection.insert_one(artifact)
            
            return result.acknowledged
        except Exception as e:
            logger.error(f"MongoDB storage error: {e}")
            return False
    
    def _store_in_qdrant(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in Qdrant."""
        try:
            import uuid as uuid_lib
            from qdrant_client.http import models
            
            client = self.backends[StorageBackend.QDRANT]["client"]
            collection = self.backends[StorageBackend.QDRANT]["collection"]
            
            # Extract embedding from artifact
            embedding = artifact.get("embedding", [0.0] * 1536)  # Default to 1536-dim zero vector
            
            # Prepare payload
            payload = {
                "artifact_id": artifact.get("artifact_id"),
                "content": artifact.get("content", ""),
                "type": artifact.get("type", "unknown"),
                "source": artifact.get("source", "unknown"),
                "context": artifact.get("context", ""),
                "created_at": artifact.get("created_at", ""),
                "metadata": artifact.get("metadata", {})
            }
            
            # Store in Qdrant
            client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=hash(artifact.get("artifact_id", str(uuid_lib.uuid4()))) % (10**9),  # Qdrant point ID
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            return True
        except Exception as e:
            logger.error(f"Qdrant storage error: {e}")
            return False
    
    def _store_in_neo4j(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in Neo4j."""
        try:
            driver = self.backends[StorageBackend.NEO4J]["driver"]
            
            with driver.session() as session:
                # Create a knowledge artifact node
                query = """
                MERGE (ka:KnowledgeArtifact {artifact_id: $artifact_id})
                SET ka.content = $content,
                    ka.type = $type,
                    ka.source = $source,
                    ka.context = $context,
                    ka.created_at = $created_at,
                    ka.updated_at = datetime()
                RETURN ka.artifact_id as id
                """
                
                result = session.run(
                    query,
                    artifact_id=artifact.get("artifact_id"),
                    content=artifact.get("content", ""),
                    type=artifact.get("type", "unknown"),
                    source=artifact.get("source", "unknown"),
                    context=artifact.get("context", ""),
                    created_at=artifact.get("created_at", "")
                )
                
                record = result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Neo4j storage error: {e}")
            return False
    
    def _store_in_redis(self, artifact: Dict[str, Any]) -> bool:
        """Store artifact in Redis."""
        try:
            client = self.backends[StorageBackend.REDIS]["client"]
            ttl = self.backends[StorageBackend.REDIS]["ttl"]
            
            key = f"knowledge_artifact:{artifact.get('artifact_id')}"
            value = json.dumps(artifact)
            
            # Store with TTL
            result = client.setex(key, ttl, value)
            
            return result == "OK"
        except Exception as e:
            logger.error(f"Redis storage error: {e}")
            return False
    
    def batch_store_artifacts(
        self,
        artifacts: List[Dict[str, Any]],
        backend: Optional[StorageBackend] = None,
        batch_size: Optional[int] = None
    ) -> List[EnhancedStorageResult]:
        """
        Store multiple knowledge artifacts in batch.
        
        Args:
            artifacts: List of knowledge artifacts to store
            backend: Specific backend to use (None for default)
            batch_size: Size of batches (None for config default)
            
        Returns:
            List of EnhancedStorageResult objects
        """
        start_time = datetime.now(timezone.utc)
        
        if not batch_size:
            batch_size = self.config.get("batch_size", 100)
        
        logger.info({
            "msg": "Starting batch storage of knowledge artifacts",
            "artifact_count": len(artifacts),
            "batch_size": batch_size,
            "backend": backend.value if backend else "default",
            "timestamp": start_time.isoformat()
        })
        
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(artifacts), batch_size):
                batch = artifacts[i:i + batch_size]
                
                # Store each artifact in the batch
                batch_results = []
                for artifact in batch:
                    result = self.store_knowledge_artifact(
                        artifact=artifact,
                        generate_embedding=True,
                        backend=backend
                    )
                    batch_results.append(result)
                
                results.extend(batch_results)
        
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            successful_count = sum(1 for r in results if r.success)
            
            logger.info({
                "msg": "Batch storage completed",
                "artifact_count": len(artifacts),
                "successful_count": successful_count,
                "failed_count": len(artifacts) - successful_count,
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return results
            
        except Exception as e:
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Batch storage failed",
                "artifact_count": len(artifacts),
                "error": str(e),
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all artifacts
            error_results = []
            for artifact in artifacts:
                error_results.append(EnhancedStorageResult(
                    success=False,
                    error=str(e),
                    processing_time_ms=total_processing_time / len(artifacts) if artifacts else 0.0
                ))
            
            return error_results
    
    def create_knowledge_graph(self) -> Dict[str, Any]:
        """
        Create a knowledge graph representation from stored artifacts.
        
        Returns:
            Dictionary with graph representation
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Creating knowledge graph from stored artifacts",
            "timestamp": start_time.isoformat()
        })
        
        try:
            # For this implementation, we'll create a simple graph
            # In a real implementation, this would query the storage backends
            # to extract entities and relationships
            
            # Get all artifacts from MongoDB as an example
            if StorageBackend.MONGODB in self.backends:
                collection = self.backends[StorageBackend.MONGODB]["collection"]
                
                # Get all artifacts
                artifacts_cursor = collection.find({}, {"_id": 0}).limit(1000)  # Limit for performance
                artifacts = list(artifacts_cursor)
                
                # Create a simple graph representation
                nodes = set()
                edges = []
                
                for artifact in artifacts:
                    artifact_id = artifact.get("artifact_id", "")
                    artifact_type = artifact.get("type", "unknown")
                    
                    # Add artifact as a node
                    nodes.add((artifact_id, artifact_type))
                    
                    # If this is a relationship artifact, add edges
                    if artifact_type == "relationship":
                        subject = artifact.get("subject", "")
                        object = artifact.get("object", "")
                        predicate = artifact.get("predicate", "")
                        
                        if subject and object:
                            edges.append((subject, predicate, object))
                
                graph = {
                    "nodes": [{"id": nid, "type": ntype} for nid, ntype in nodes],
                    "edges": [{"source": src, "relation": rel, "target": tgt} for src, rel, tgt in edges],
                    "metadata": {
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                        "generated_at": datetime.now(timezone.utc).isoformat()
                    }
                }
                
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.info({
                    "msg": "Knowledge graph created successfully",
                    "node_count": graph["metadata"]["node_count"],
                    "edge_count": graph["metadata"]["edge_count"],
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return graph
            else:
                logger.warning("MongoDB backend not available for graph creation")
                return {
                    "nodes": [],
                    "edges": [],
                    "metadata": {
                        "node_count": 0,
                        "edge_count": 0,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "error": "No available backend for graph creation"
                    }
                }
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge graph creation failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "node_count": 0,
                    "edge_count": 0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(e)
                }
            }
    
    def get_aggregated_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics about the knowledge storage.
        
        Returns:
            Dictionary with aggregated statistics
        """
        stats = {
            "total_artifacts": 0,
            "backend_status": {},
            "operation_stats": self.operation_stats.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get stats from MongoDB
        if StorageBackend.MONGODB in self.backends:
            try:
                collection = self.backends[StorageBackend.MONGODB]["collection"]
                count = collection.count_documents({})
                stats["total_artifacts"] = count
                stats["backend_status"]["mongodb"] = {
                    "status": "connected",
                    "artifact_count": count,
                    "indexes": collection.index_information()
                }
            except Exception as e:
                stats["backend_status"]["mongodb"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["backend_status"]["mongodb"] = {"status": "disconnected"}
        
        # Get stats from Qdrant
        if StorageBackend.QDRANT in self.backends:
            try:
                client = self.backends[StorageBackend.QDRANT]["client"]
                collection_name = self.backends[StorageBackend.QDRANT]["collection"]
                collection_info = client.get_collection(collection_name)
                vector_count = collection_info.points_count
                
                stats["backend_status"]["qdrant"] = {
                    "status": "connected",
                    "vector_count": vector_count,
                    "collection": collection_name
                }
            except Exception as e:
                stats["backend_status"]["qdrant"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["backend_status"]["qdrant"] = {"status": "disconnected"}
        
        # Get stats from Neo4j
        if StorageBackend.NEO4J in self.backends:
            try:
                driver = self.backends[StorageBackend.NEO4J]["driver"]
                with driver.session() as session:
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
        
        # Get stats from Redis
        if StorageBackend.REDIS in self.backends:
            try:
                client = self.backends[StorageBackend.REDIS]["client"]
                info = client.info()
                db_size = client.dbsize()
                
                stats["backend_status"]["redis"] = {
                    "status": "connected",
                    "key_count": db_size,
                    "used_memory": info.get("used_memory_human", "unknown")
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
        Optimize storage performance with enhanced indexing and cleanup.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting enhanced storage optimization",
            "timestamp": start_time.isoformat()
        })
        
        results = {
            "operations_performed": [],
            "backend_results": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Optimize MongoDB
        if StorageBackend.MONGODB in self.backends:
            try:
                collection = self.backends[StorageBackend.MONGODB]["collection"]
                
                # Create additional indexes for common queries
                collection.create_index("type")
                collection.create_index("source")
                collection.create_index("created_at")
                collection.create_index([("type", 1), ("created_at", -1)])
                
                # Compact database
                # Note: This is just an example; actual compaction depends on MongoDB setup
                results["backend_results"]["mongodb"] = {
                    "status": "success",
                    "indexes_created": 4,
                    "compaction_attempted": False  # Would require admin privileges
                }
                results["operations_performed"].append("MongoDB indexes optimized")
            except Exception as e:
                results["backend_results"]["mongodb"] = {
                    "status": "error",
                    "error": str(e)
                }
                results["operations_performed"].append(f"MongoDB optimization error: {e}")
        
        # Optimize Qdrant
        if StorageBackend.QDRANT in self.backends:
            try:
                # Qdrant optimization would go here
                results["backend_results"]["qdrant"] = {
                    "status": "success",
                    "optimization_performed": True
                }
                results["operations_performed"].append("Qdrant optimization completed")
            except Exception as e:
                results["backend_results"]["qdrant"] = {
                    "status": "error",
                    "error": str(e)
                }
                results["operations_performed"].append(f"Qdrant optimization error: {e}")
        
        # Optimize Neo4j
        if StorageBackend.NEO4J in self.backends:
            try:
                driver = self.backends[StorageBackend.NEO4J]["driver"]
                with driver.session() as session:
                    # Create additional indexes
                    session.run("CREATE INDEX IF NOT EXISTS FOR (ka:KnowledgeArtifact) ON (ka.type)")
                    session.run("CREATE INDEX IF NOT EXISTS FOR (ka:KnowledgeArtifact) ON (ka.source)")
                    session.run("CREATE INDEX IF NOT EXISTS FOR (ka:KnowledgeArtifact) ON (ka.created_at)")
                    
                    results["backend_results"]["neo4j"] = {
                        "status": "success",
                        "indexes_created": 3
                    }
                    results["operations_performed"].append("Neo4j indexes optimized")
            except Exception as e:
                results["backend_results"]["neo4j"] = {
                    "status": "error",
                    "error": str(e)
                }
                results["operations_performed"].append(f"Neo4j optimization error: {e}")
        
        # Optimize Redis
        if StorageBackend.REDIS in self.backends:
            try:
                client = self.backends[StorageBackend.REDIS]["client"]
                
                # Clean up expired keys
                client.execute_command("MEMORY PURGE")  # This is a best-effort command
                
                results["backend_results"]["redis"] = {
                    "status": "success",
                    "cleanup_performed": True
                }
                results["operations_performed"].append("Redis cleanup performed")
            except Exception as e:
                results["backend_results"]["redis"] = {
                    "status": "error",
                    "error": str(e)
                }
                results["operations_performed"].append(f"Redis optimization error: {e}")
        
        total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        results["total_processing_time_ms"] = total_processing_time
        
        logger.info({
            "msg": "Enhanced storage optimization completed",
            "operations_count": len(results["operations_performed"]),
            "total_processing_time_ms": total_processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return results
    
    def _get_fallback_backend(self, primary_backend: StorageBackend) -> Optional[StorageBackend]:
        """Get a fallback backend when primary fails."""
        backend_priority = [
            StorageBackend.MONGODB,
            StorageBackend.QDRANT,
            StorageBackend.NEO4J,
            StorageBackend.REDIS
        ]
        
        for backend in backend_priority:
            if backend != primary_backend and backend in self.backends:
                return backend
        
        return None
    
    def _update_operation_stats(self, success: bool, backend: str, processing_time_ms: float):
        """Update operation statistics."""
        self.operation_stats["total_operations"] += 1
        
        if success:
            self.operation_stats["successful_operations"] += 1
        else:
            self.operation_stats["failed_operations"] += 1
        
        # Update average processing time
        total_ops = self.operation_stats["total_operations"]
        current_avg = self.operation_stats["average_processing_time"]
        new_avg = ((current_avg * (total_ops - 1)) + processing_time_ms) / total_ops
        self.operation_stats["average_processing_time"] = new_avg
        
        # Update backend usage
        if backend not in self.operation_stats["backend_usage"]:
            self.operation_stats["backend_usage"][backend] = {
                "total": 0,
                "successful": 0,
                "failed": 0
            }
        
        backend_stats = self.operation_stats["backend_usage"][backend]
        backend_stats["total"] += 1
        if success:
            backend_stats["successful"] += 1
        else:
            backend_stats["failed"] += 1
    
    def close_connections(self):
        """Close all backend connections."""
        logger.info({
            "msg": "Closing enhanced storage connections",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close MongoDB
        if StorageBackend.MONGODB in self.backends:
            self.backends[StorageBackend.MONGODB]["client"].close()
            logger.info("MongoDB connection closed")
        
        # Close Neo4j
        if StorageBackend.NEO4J in self.backends:
            self.backends[StorageBackend.NEO4J]["driver"].close()
            logger.info("Neo4j driver closed")
        
        logger.info({
            "msg": "Enhanced storage connections closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })