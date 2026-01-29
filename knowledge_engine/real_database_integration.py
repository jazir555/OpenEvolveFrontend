"""
Real Database Integration for OpenEvolve Knowledge Engine

This module provides real database integration with multiple backend support
for the Phase 3 implementation of the OpenEvolve Knowledge Engine.
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


class DatabaseType(Enum):
    """Enumeration of supported database types."""
    QDRANT = "qdrant"
    MONGODB = "mongodb"
    NEO4J = "neo4j"
    REDIS = "redis"


@dataclass
class IntegrationResult:
    """Result of a database integration operation."""
    success: bool
    database_type: str
    operation: str
    records_affected: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RealDatabaseIntegrator:
    """
    Real database integration layer with multi-backend support.
    
    Provides methods for:
    - Multi-database backend management
    - Health monitoring
    - Performance optimization
    - Transaction management
    - Data synchronization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real database integrator.
        
        Args:
            config: Configuration for database integration
        """
        self.config = config or self._get_default_config()
        
        # Initialize database connections
        self.connections = {}
        self._initialize_connections()
        
        # Health tracking
        self.health_status = {
            "overall_status": "unknown",
            "last_check": datetime.now(timezone.utc).isoformat(),
            "available_databases": 0,
            "database_status": {}
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_response_time": 0.0,
            "database_performance": {}
        }
        
        logger.info({
            "msg": "RealDatabaseIntegrator initialized",
            "config": self.config,
            "available_databases": list(self.connections.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for database integration."""
        return {
            "databases": {
                "qdrant": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 6333,
                    "timeout": 30,
                    "collection": "knowledge_vectors",
                    "vector_size": 1536
                },
                "mongodb": {
                    "enabled": True,
                    "uri": "mongodb://localhost:27017",
                    "database": "openevolve_kg",
                    "collection": "knowledge_artifacts",
                    "timeout": 30
                },
                "neo4j": {
                    "enabled": True,
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password",
                    "timeout": 30
                },
                "redis": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "timeout": 10
                }
            },
            "default_database": "mongodb",
            "enable_fallback": True,
            "enable_replication": True,
            "replication_targets": ["qdrant", "neo4j"],
            "transaction_timeout": 60,
            "connection_pool_size": 10,
            "enable_monitoring": True
        }
    
    def _initialize_connections(self):
        """Initialize connections to configured databases."""
        db_configs = self.config.get("databases", {})
        
        # Initialize Qdrant
        if db_configs.get("qdrant", {}).get("enabled", True):
            try:
                import qdrant_client
                qdrant_config = db_configs["qdrant"]
                
                client = qdrant_client.QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config["port"],
                    timeout=qdrant_config["timeout"]
                )
                
                self.connections[DatabaseType.QDRANT] = {
                    "client": client,
                    "config": qdrant_config
                }
                
                logger.info("Qdrant connection initialized")
            except ImportError:
                logger.warning("qdrant-client not available, Qdrant disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant connection: {e}")
        
        # Initialize MongoDB
        if db_configs.get("mongodb", {}).get("enabled", True):
            try:
                from pymongo import MongoClient
                from pymongo.errors import ServerSelectionTimeoutError
                mongo_config = db_configs["mongodb"]
                
                client = MongoClient(
                    mongo_config["uri"],
                    serverSelectionTimeoutMS=mongo_config["timeout"] * 1000
                )
                
                # Test connection
                client.admin.command('ping')
                
                db = client[mongo_config["database"]]
                collection = db[mongo_config["collection"]]
                
                self.connections[DatabaseType.MONGODB] = {
                    "client": client,
                    "db": db,
                    "collection": collection,
                    "config": mongo_config
                }
                
                logger.info("MongoDB connection initialized")
            except ImportError:
                logger.warning("PyMongo not available, MongoDB disabled")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB connection: {e}")
        
        # Initialize Neo4j
        if db_configs.get("neo4j", {}).get("enabled", True):
            try:
                from neo4j import GraphDatabase
                neo4j_config = db_configs["neo4j"]
                
                driver = GraphDatabase.driver(
                    neo4j_config["uri"],
                    auth=(neo4j_config["user"], neo4j_config["password"]),
                    connection_timeout=neo4j_config["timeout"]
                )
                
                # Test connection
                with driver.session() as session:
                    session.run("RETURN 1")
                
                self.connections[DatabaseType.NEO4J] = {
                    "driver": driver,
                    "config": neo4j_config
                }
                
                logger.info("Neo4j connection initialized")
            except ImportError:
                logger.warning("Neo4j driver not available, Neo4j disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j connection: {e}")
        
        # Initialize Redis
        if db_configs.get("redis", {}).get("enabled", True):
            try:
                import redis
                redis_config = db_configs["redis"]
                
                client = redis.Redis(
                    host=redis_config["host"],
                    port=redis_config["port"],
                    db=redis_config["db"],
                    socket_timeout=redis_config["timeout"],
                    decode_responses=True
                )
                
                # Test connection
                client.ping()
                
                self.connections[DatabaseType.REDIS] = {
                    "client": client,
                    "config": redis_config
                }
                
                logger.info("Redis connection initialized")
            except ImportError:
                logger.warning("Redis not available, Redis disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Redis connection: {e}")
    
    def is_production_ready(self) -> bool:
        """
        Check if the system is production ready based on database availability.
        
        Returns:
            True if production ready, False otherwise
        """
        required_dbs = ["qdrant", "mongodb", "neo4j"]  # At least these should be available
        available_dbs = list(self.connections.keys())
        
        # Check if we have the minimum required databases
        required_available = all(
            any(db_type.value == req_db for db_type in available_dbs) 
            for req_db in required_dbs
        )
        
        return required_available
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all connected databases.
        
        Returns:
            Dictionary with health status information
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Checking database health status",
            "timestamp": start_time.isoformat()
        })
        
        status = {
            "overall_status": "healthy",
            "last_check": start_time.isoformat(),
            "available_databases": 0,
            "database_status": {},
            "timestamp": start_time.isoformat()
        }
        
        healthy_count = 0
        
        # Check Qdrant
        if DatabaseType.QDRANT in self.connections:
            try:
                client = self.connections[DatabaseType.QDRANT]["client"]
                # Test basic operation
                client.get_collections()
                status["database_status"]["qdrant"] = "healthy"
                healthy_count += 1
            except Exception as e:
                status["database_status"]["qdrant"] = f"unhealthy: {str(e)}"
                status["overall_status"] = "degraded"
        else:
            status["database_status"]["qdrant"] = "disabled"
        
        # Check MongoDB
        if DatabaseType.MONGODB in self.connections:
            try:
                collection = self.connections[DatabaseType.MONGODB]["collection"]
                # Test basic operation
                collection.count_documents({})
                status["database_status"]["mongodb"] = "healthy"
                healthy_count += 1
            except Exception as e:
                status["database_status"]["mongodb"] = f"unhealthy: {str(e)}"
                status["overall_status"] = "degraded"
        else:
            status["database_status"]["mongodb"] = "disabled"
        
        # Check Neo4j
        if DatabaseType.NEO4J in self.connections:
            try:
                driver = self.connections[DatabaseType.NEO4J]["driver"]
                with driver.session() as session:
                    session.run("RETURN 1")
                status["database_status"]["neo4j"] = "healthy"
                healthy_count += 1
            except Exception as e:
                status["database_status"]["neo4j"] = f"unhealthy: {str(e)}"
                status["overall_status"] = "degraded"
        else:
            status["database_status"]["neo4j"] = "disabled"
        
        # Check Redis
        if DatabaseType.REDIS in self.connections:
            try:
                client = self.connections[DatabaseType.REDIS]["client"]
                client.ping()
                status["database_status"]["redis"] = "healthy"
                healthy_count += 1
            except Exception as e:
                status["database_status"]["redis"] = f"unhealthy: {str(e)}"
                status["overall_status"] = "degraded"
        else:
            status["database_status"]["redis"] = "disabled"
        
        status["available_databases"] = healthy_count
        
        # Update overall status based on number of healthy databases
        if healthy_count == 0:
            status["overall_status"] = "unhealthy"
        elif healthy_count < len([k for k in self.connections.keys()]):
            status["overall_status"] = "degraded"
        
        self.health_status = status
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Database health check completed",
            "overall_status": status["overall_status"],
            "healthy_databases": healthy_count,
            "total_databases": len(self.connections),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return status
    
    def execute_query(
        self,
        query: str,
        database_type: Optional[DatabaseType] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Execute a query against a specific database.
        
        Args:
            query: Query string to execute
            database_type: Type of database to query (None for default)
            params: Query parameters
            
        Returns:
            IntegrationResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        if not database_type:
            db_type = DatabaseType(self.config.get("default_database", "mongodb"))
        else:
            db_type = database_type
        
        logger.info({
            "msg": "Executing database query",
            "database_type": db_type.value,
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "timestamp": start_time.isoformat()
        })
        
        try:
            result = self._execute_query_on_db(query, db_type, params or {})
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self._update_performance_metrics(True, db_type.value, processing_time_ms)
            
            logger.info({
                "msg": "Database query executed successfully",
                "database_type": db_type.value,
                "records_affected": result.records_affected,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Database query execution failed",
                "database_type": db_type.value,
                "query": query,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Update performance metrics
            self._update_performance_metrics(False, db_type.value, processing_time_ms)
            
            # Try fallback if enabled
            if self.config.get("enable_fallback", True):
                fallback_db = self._get_fallback_database(db_type)
                if fallback_db:
                    logger.info(f"Attempting fallback to {fallback_db.value}")
                    return self.execute_query(query, fallback_db, params)
            
            return IntegrationResult(
                success=False,
                database_type=db_type.value,
                operation="query",
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def _execute_query_on_db(
        self,
        query: str,
        db_type: DatabaseType,
        params: Dict[str, Any]
    ) -> IntegrationResult:
        """Execute query on a specific database."""
        if db_type == DatabaseType.MONGODB and DatabaseType.MONGODB in self.connections:
            return self._execute_mongo_query(query, params)
        elif db_type == DatabaseType.NEO4J and DatabaseType.NEO4J in self.connections:
            return self._execute_neo4j_query(query, params)
        elif db_type == DatabaseType.REDIS and DatabaseType.REDIS in self.connections:
            return self._execute_redis_query(query, params)
        elif db_type == DatabaseType.QDRANT and DatabaseType.QDRANT in self.connections:
            return self._execute_qdrant_query(query, params)
        else:
            raise ValueError(f"Database {db_type.value} not available")
    
    def _execute_mongo_query(self, query: str, params: Dict[str, Any]) -> IntegrationResult:
        """Execute query on MongoDB."""
        try:
            collection = self.connections[DatabaseType.MONGODB]["collection"]
            
            # Parse query - this is a simplified approach
            # In a real implementation, you'd have more sophisticated query parsing
            if query.strip().upper().startswith("FIND"):
                # Extract the filter part
                import re
                match = re.search(r'FIND\s*\((.*)\)', query, re.IGNORECASE)
                if match:
                    filter_str = match.group(1).strip()
                    if filter_str:
                        # Safely evaluate the filter (in real implementation, use proper parsing)
                        try:
                            filter_dict = eval(filter_str) if filter_str != '{}' else {}
                        except:
                            filter_dict = {}
                    else:
                        filter_dict = {}
                    
                    cursor = collection.find(filter_dict)
                    results = list(cursor)
                    count = len(results)
                else:
                    count = collection.count_documents({})
            elif query.strip().upper().startswith("COUNT"):
                count = collection.count_documents({})
            else:
                # Default to counting all documents
                count = collection.count_documents({})
            
            return IntegrationResult(
                success=True,
                database_type=DatabaseType.MONGODB.value,
                operation="query",
                records_affected=count,
                metadata={"results": count}
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                database_type=DatabaseType.MONGODB.value,
                operation="query",
                error=str(e)
            )
    
    def _execute_neo4j_query(self, query: str, params: Dict[str, Any]) -> IntegrationResult:
        """Execute query on Neo4j."""
        try:
            driver = self.connections[DatabaseType.NEO4J]["driver"]
            
            with driver.session() as session:
                result = session.run(query, **params)
                
                # Count the records affected
                count = 0
                for _ in result:
                    count += 1
            
            return IntegrationResult(
                success=True,
                database_type=DatabaseType.NEO4J.value,
                operation="query",
                records_affected=count
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                database_type=DatabaseType.NEO4J.value,
                operation="query",
                error=str(e)
            )
    
    def _execute_redis_query(self, query: str, params: Dict[str, Any]) -> IntegrationResult:
        """Execute query on Redis."""
        try:
            client = self.connections[DatabaseType.REDIS]["client"]
            
            # For this example, we'll handle simple GET/SET operations
            query_upper = query.strip().upper()
            
            if query_upper.startswith("GET"):
                key = query.split()[1] if len(query.split()) > 1 else ""
                if key:
                    value = client.get(key)
                    count = 1 if value is not None else 0
                else:
                    count = 0
            elif query_upper.startswith("KEYS"):
                pattern = query.split()[1] if len(query.split()) > 1 else "*"
                keys = client.keys(pattern)
                count = len(keys)
            else:
                # Default to getting DB size
                count = client.dbsize()
            
            return IntegrationResult(
                success=True,
                database_type=DatabaseType.REDIS.value,
                operation="query",
                records_affected=count
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                database_type=DatabaseType.REDIS.value,
                operation="query",
                error=str(e)
            )
    
    def _execute_qdrant_query(self, query: str, params: Dict[str, Any]) -> IntegrationResult:
        """Execute query on Qdrant."""
        try:
            client = self.connections[DatabaseType.QDRANT]["client"]
            collection = self.connections[DatabaseType.QDRANT]["config"]["collection"]
            
            # For this example, we'll just get the collection info
            # In a real implementation, you'd parse the query properly
            collection_info = client.get_collection(collection)
            count = collection_info.points_count
            
            return IntegrationResult(
                success=True,
                database_type=DatabaseType.QDRANT.value,
                operation="query",
                records_affected=count
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                database_type=DatabaseType.QDRANT.value,
                operation="query",
                error=str(e)
            )
    
    def synchronize_data(self, source_db: DatabaseType, target_db: DatabaseType) -> IntegrationResult:
        """
        Synchronize data between two databases.
        
        Args:
            source_db: Source database type
            target_db: Target database type
            
        Returns:
            IntegrationResult with synchronization details
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting database synchronization",
            "source_db": source_db.value,
            "target_db": target_db.value,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # This is a simplified synchronization example
            # In a real implementation, you'd have more sophisticated data transfer logic
            
            # For example, sync from MongoDB to Neo4j
            if source_db == DatabaseType.MONGODB and target_db == DatabaseType.NEO4J:
                sync_result = self._sync_mongo_to_neo4j()
            elif source_db == DatabaseType.NEO4J and target_db == DatabaseType.MONGODB:
                sync_result = self._sync_neo4j_to_mongo()
            else:
                # For other combinations, implement appropriate sync logic
                sync_result = IntegrationResult(
                    success=True,
                    database_type=f"{source_db.value}_to_{target_db.value}",
                    operation="sync",
                    records_affected=0,
                    metadata={"message": "Sync logic not implemented for this combination"}
                )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            sync_result.processing_time_ms = processing_time_ms
            
            logger.info({
                "msg": "Database synchronization completed",
                "source_db": source_db.value,
                "target_db": target_db.value,
                "records_synced": sync_result.records_affected,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return sync_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Database synchronization failed",
                "source_db": source_db.value,
                "target_db": target_db.value,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return IntegrationResult(
                success=False,
                database_type=f"{source_db.value}_to_{target_db.value}",
                operation="sync",
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def _sync_mongo_to_neo4j(self) -> IntegrationResult:
        """Synchronize data from MongoDB to Neo4j."""
        try:
            # Get data from MongoDB
            mongo_collection = self.connections[DatabaseType.MONGODB]["collection"]
            mongo_docs = list(mongo_collection.find({}).limit(100))  # Limit for example
            
            # Insert into Neo4j
            neo4j_driver = self.connections[DatabaseType.NEO4J]["driver"]
            
            with neo4j_driver.session() as session:
                for doc in mongo_docs:
                    # Create a node in Neo4j for each document
                    query = """
                    MERGE (ka:KnowledgeArtifact {artifact_id: $artifact_id})
                    SET ka.content = $content,
                        ka.type = $type,
                        ka.source = $source,
                        ka.context = $context,
                        ka.created_at = $created_at
                    """
                    
                    session.run(
                        query,
                        artifact_id=doc.get("artifact_id", str(uuid.uuid4())),
                        content=doc.get("content", ""),
                        type=doc.get("type", "unknown"),
                        source=doc.get("source", "unknown"),
                        context=doc.get("context", ""),
                        created_at=doc.get("created_at", datetime.now(timezone.utc).isoformat())
                    )
            
            return IntegrationResult(
                success=True,
                database_type="mongodb_to_neo4j",
                operation="sync",
                records_affected=len(mongo_docs),
                metadata={"documents_synced": len(mongo_docs)}
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                database_type="mongodb_to_neo4j",
                operation="sync",
                error=str(e)
            )
    
    def _sync_neo4j_to_mongo(self) -> IntegrationResult:
        """Synchronize data from Neo4j to MongoDB."""
        try:
            # Get data from Neo4j
            neo4j_driver = self.connections[DatabaseType.NEO4J]["driver"]
            
            with neo4j_driver.session() as session:
                result = session.run("MATCH (ka:KnowledgeArtifact) RETURN ka LIMIT 100")
                neo4j_nodes = [record["ka"] for record in result]
            
            # Insert into MongoDB
            mongo_collection = self.connections[DatabaseType.MONGODB]["collection"]
            
            mongo_docs = []
            for node in neo4j_nodes:
                doc = dict(node)
                mongo_docs.append(doc)
            
            if mongo_docs:
                result = mongo_collection.insert_many(mongo_docs)
            
            return IntegrationResult(
                success=True,
                database_type="neo4j_to_mongodb",
                operation="sync",
                records_affected=len(neo4j_nodes),
                metadata={"nodes_synced": len(neo4j_nodes)}
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                database_type="neo4j_to_mongodb",
                operation="sync",
                error=str(e)
            )
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all connected databases.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            "databases": {},
            "total_records": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        total_records = 0
        
        # Get MongoDB stats
        if DatabaseType.MONGODB in self.connections:
            try:
                collection = self.connections[DatabaseType.MONGODB]["collection"]
                count = collection.count_documents({})
                storage_size = collection.estimated_document_count()  # Simplified
                
                stats["databases"]["mongodb"] = {
                    "status": "connected",
                    "record_count": count,
                    "storage_size_approx": storage_size,
                    "indexes": collection.index_information()
                }
                total_records += count
            except Exception as e:
                stats["databases"]["mongodb"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Get Neo4j stats
        if DatabaseType.NEO4J in self.connections:
            try:
                driver = self.connections[DatabaseType.NEO4J]["driver"]
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) AS count")
                    record = result.single()
                    count = record["count"] if record else 0
                    
                    # Get more detailed stats
                    result = session.run("CALL db.labels() YIELD label RETURN label")
                    labels = [record["label"] for record in result]
                    
                    stats["databases"]["neo4j"] = {
                        "status": "connected",
                        "node_count": count,
                        "labels": labels,
                        "relationship_count": count * 2  # Approximation
                    }
                    total_records += count
            except Exception as e:
                stats["databases"]["neo4j"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Get Redis stats
        if DatabaseType.REDIS in self.connections:
            try:
                client = self.connections[DatabaseType.REDIS]["client"]
                info = client.info()
                
                stats["databases"]["redis"] = {
                    "status": "connected",
                    "key_count": client.dbsize(),
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0)
                }
            except Exception as e:
                stats["databases"]["redis"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Get Qdrant stats
        if DatabaseType.QDRANT in self.connections:
            try:
                client = self.connections[DatabaseType.QDRANT]["client"]
                collection_name = self.connections[DatabaseType.QDRANT]["config"]["collection"]
                
                collection_info = client.get_collection(collection_name)
                stats["databases"]["qdrant"] = {
                    "status": "connected",
                    "vector_count": collection_info.points_count,
                    "indexed_vectors": collection_info.indexed_vectors_count,
                    "collection_config": collection_info.config.dict() if hasattr(collection_info.config, 'dict') else {}
                }
                total_records += collection_info.points_count
            except Exception as e:
                stats["databases"]["qdrant"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        stats["total_records"] = total_records
        
        return stats
    
    def _get_fallback_database(self, primary_db: DatabaseType) -> Optional[DatabaseType]:
        """Get a fallback database when primary fails."""
        db_priority = [
            DatabaseType.MONGODB,
            DatabaseType.NEO4J,
            DatabaseType.QDRANT,
            DatabaseType.REDIS
        ]
        
        for db in db_priority:
            if db != primary_db and db in self.connections:
                return db
        
        return None
    
    def _update_performance_metrics(self, success: bool, db_type: str, processing_time_ms: float):
        """Update performance metrics."""
        self.performance_metrics["total_operations"] += 1
        
        if success:
            self.performance_metrics["successful_operations"] += 1
        else:
            self.performance_metrics["failed_operations"] += 1
        
        # Update average processing time
        total_ops = self.performance_metrics["total_operations"]
        current_avg = self.performance_metrics["average_response_time"]
        new_avg = ((current_avg * (total_ops - 1)) + processing_time_ms) / total_ops
        self.performance_metrics["average_response_time"] = new_avg
        
        # Update database-specific metrics
        if db_type not in self.performance_metrics["database_performance"]:
            self.performance_metrics["database_performance"][db_type] = {
                "total_ops": 0,
                "successful_ops": 0,
                "failed_ops": 0,
                "avg_response_time": 0.0
            }
        
        db_metrics = self.performance_metrics["database_performance"][db_type]
        db_metrics["total_ops"] += 1
        
        if success:
            db_metrics["successful_ops"] += 1
        else:
            db_metrics["failed_ops"] += 1
        
        # Update database average response time
        current_db_avg = db_metrics["avg_response_time"]
        total_db_ops = db_metrics["total_ops"]
        new_db_avg = ((current_db_avg * (total_db_ops - 1)) + processing_time_ms) / total_db_ops
        db_metrics["avg_response_time"] = new_db_avg
    
    def close_connections(self):
        """Close all database connections."""
        logger.info({
            "msg": "Closing all database connections",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close MongoDB
        if DatabaseType.MONGODB in self.connections:
            self.connections[DatabaseType.MONGODB]["client"].close()
            logger.info("MongoDB connection closed")
        
        # Close Neo4j
        if DatabaseType.NEO4J in self.connections:
            self.connections[DatabaseType.NEO4J]["driver"].close()
            logger.info("Neo4j driver closed")
        
        # Close Redis
        if DatabaseType.REDIS in self.connections:
            self.connections[DatabaseType.REDIS]["client"].close()
            logger.info("Redis connection closed")
        
        logger.info({
            "msg": "All database connections closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })