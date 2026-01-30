"""
Memgraph Backend Adapter for Unified Knowledge Graph Manager.

Provides graph storage using Memgraph - an Apache 2.0 licensed
graph database compatible with Neo4j's Bolt protocol and Cypher.

License: Apache 2.0 (permissive)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
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


class MemgraphBackend(KnowledgeGraphBackend):
    """
    Memgraph backend adapter for graph storage.
    
    Memgraph is a high-performance graph database that is fully compatible
    with Neo4j's Bolt protocol and Cypher query language.
    
    License: Apache 2.0 (permissive, commercially friendly)
    
    Advantages over Neo4j:
    - Apache 2.0 license (vs Neo4j's GPL/Commercial)
    - In-memory performance with durability
    - Compatible with existing Neo4j Cypher queries
    - Works with standard Neo4j Python driver
    
    Environment Variables Required:
        MEMGRAPH_URI: Memgraph connection URI (e.g., bolt://localhost:7687)
        MEMGRAPH_USER: Username (default: "")
        MEMGRAPH_PASSWORD: Password (default: "")
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.MEMGRAPH
        self.driver = None
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration"""
        self.uri = self.config.get('uri', 'bolt://localhost:7687')
        self.user = self.config.get('user', '')
        self.password = self.config.get('password', '')
        
        logger.info(f"Memgraph backend configured for: {self.uri}")

    async def connect(self) -> bool:
        """Establish connection to Memgraph - Runtime Truth"""
        try:
            # Memgraph uses the same Bolt protocol as Neo4j
            # so we can use the neo4j driver
            from neo4j import AsyncGraphDatabase

            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password) if self.user else None
            )

            # Verify connection - Runtime Truth
            await self.driver.verify_connectivity()

            # Ensure schema constraints exist
            await self._ensure_schema()

            self.is_healthy = True
            logger.info("Successfully connected to Memgraph (Apache 2.0)")

            return True

        except ImportError:
            logger.error("neo4j package not installed. Install with: pip install neo4j")
            raise ImportError("neo4j package required for MemgraphBackend")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph: {e}")
            raise ConnectionError(f"Memgraph connection failed: {e}")

    async def _ensure_schema(self):
        """Create constraints and indexes"""
        try:
            async with self.driver.session() as session:
                # Create constraints for KnowledgeEntry nodes
                try:
                    await session.run("""
                        CREATE CONSTRAINT knowledge_entry_id 
                        FOR (k:KnowledgeEntry) 
                        REQUIRE k.id IS UNIQUE
                    """)
                except Exception:
                    # Constraint may already exist
                    pass
                
                # Create indexes for better performance
                try:
                    await session.run("""
                        CREATE INDEX knowledge_source_index 
                        FOR (k:KnowledgeEntry) 
                        ON (k.source)
                    """)
                except Exception:
                    pass
                    
                try:
                    await session.run("""
                        CREATE INDEX knowledge_timestamp_index 
                        FOR (k:KnowledgeEntry) 
                        ON (k.timestamp)
                    """)
                except Exception:
                    pass

                logger.info("Memgraph schema initialized")
        except Exception as e:
            logger.warning(f"Schema initialization warning: {e}")

    async def disconnect(self) -> None:
        """Close Memgraph connection"""
        if self.driver:
            await self.driver.close()
            self.is_healthy = False
            logger.info("Disconnected from Memgraph")

    async def health_check(self) -> bool:
        """Check Memgraph health"""
        if not self.driver:
            return False

        try:
            await self.driver.verify_connectivity()
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Memgraph health check failed: {e}")
            self.is_healthy = False
            return False

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add knowledge to Memgraph"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        start_time = datetime.now(timezone.utc)
        entry_id = entry.id or str(uuid.uuid4())

        try:
            async with self.driver.session() as session:
                # Create KnowledgeEntry node
                result = await session.run("""
                    CREATE (k:KnowledgeEntry {
                        id: $id,
                        source: $source,
                        content: $content,
                        metadata: $metadata,
                        timestamp: $timestamp,
                        created_at: $created_at
                    })
                    RETURN k.id as node_id
                """, {
                    'id': entry_id,
                    'source': entry.source,
                    'content': entry.content,
                    'metadata': json.dumps(entry.metadata or {}),
                    'timestamp': entry.timestamp or datetime.now(timezone.utc).isoformat(),
                    'created_at': datetime.now(timezone.utc).isoformat()
                })
                
                record = await result.single()
                node_id = record['node_id'] if record else entry_id

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Added knowledge to Memgraph in {elapsed_ms:.2f}ms: {node_id}")

            return node_id

        except Exception as e:
            logger.error(f"Failed to add knowledge to Memgraph: {e}")
            raise ConnectionError(f"Memgraph add_knowledge failed: {e}")

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add relationship between knowledge entries"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        try:
            async with self.driver.session() as session:
                result = await session.run(f"""
                    MATCH (source:KnowledgeEntry {{id: $source_id}})
                    MATCH (target:KnowledgeEntry {{id: $target_id}})
                    CREATE (source)-[r:{relationship_type} $props]->(target)
                    RETURN id(r) as rel_id
                """, {
                    'source_id': source_id,
                    'target_id': target_id,
                    'props': json.dumps(properties or {})
                })
                
                record = await result.single()
                return str(record['rel_id']) if record else None

        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            raise ConnectionError(f"Memgraph add_relationship failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Search knowledge in Memgraph"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        start_time = datetime.now(timezone.utc)

        try:
            # Build Cypher query
            where_clauses = []
            params = {'query': f'(?i).*{query}.*', 'limit': limit, 'offset': offset}
            
            if query:
                where_clauses.append("(k.content =~ $query OR k.source =~ $query)")
            
            if filters:
                if 'source' in filters:
                    where_clauses.append("k.source = $source")
                    params['source'] = filters['source']
                if 'date_after' in filters:
                    where_clauses.append("k.timestamp >= $date_after")
                    params['date_after'] = filters['date_after']
            
            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            async with self.driver.session() as session:
                # Get results
                result = await session.run(f"""
                    MATCH (k:KnowledgeEntry)
                    {where_clause}
                    RETURN k.id as id, k.source as source, k.content as content,
                           k.metadata as metadata, k.timestamp as timestamp,
                           k.created_at as created_at
                    ORDER BY k.timestamp DESC
                    SKIP $offset
                    LIMIT $limit
                """, params)
                
                records = await result.data()
                
                # Get total count
                count_result = await session.run(f"""
                    MATCH (k:KnowledgeEntry)
                    {where_clause}
                    RETURN count(k) as total
                """, params)
                
                count_record = await count_result.single()
                total_count = count_record['total'] if count_record else 0

            results = []
            for record in records:
                results.append({
                    "id": record["id"],
                    "source": record["source"],
                    "content": record["content"],
                    "metadata": json.loads(record["metadata"]) if record["metadata"] else {},
                    "timestamp": record["timestamp"],
                    "created_at": record["created_at"]
                })

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return SearchResults(
                query=query,
                results=results,
                total_count=total_count,
                backend_used="memgraph",
                search_time_ms=elapsed_ms,
                metadata={"filters": filters}
            )

        except Exception as e:
            logger.error(f"Memgraph search failed: {e}")
            raise ConnectionError(f"Memgraph search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze Memgraph knowledge graph"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        start_time = datetime.now(timezone.utc)

        try:
            async with self.driver.session() as session:
                if analysis_type == "source_distribution":
                    result = await session.run("""
                        MATCH (k:KnowledgeEntry)
                        RETURN k.source as source, count(k) as count
                        ORDER BY count DESC
                    """)
                    records = await result.data()
                    results = {"by_source": [{"source": r["source"], "count": r["count"]} for r in records]}

                elif analysis_type == "node_count":
                    result = await session.run("""
                        MATCH (k:KnowledgeEntry)
                        RETURN count(k) as node_count
                    """)
                    record = await result.single()
                    results = {"node_count": record["node_count"] if record else 0}

                elif analysis_type == "relationship_count":
                    result = await session.run("""
                        MATCH ()-[r]->()
                        RETURN count(r) as edge_count
                    """)
                    record = await result.single()
                    results = {"edge_count": record["edge_count"] if record else 0}

                elif analysis_type == "centrality":
                    # Simple degree centrality
                    result = await session.run("""
                        MATCH (k:KnowledgeEntry)
                        OPTIONAL MATCH (k)-[r]-()
                        RETURN k.id as id, k.source as source, count(r) as degree
                        ORDER BY degree DESC
                        LIMIT 10
                    """)
                    records = await result.data()
                    results = {"top_nodes": [{"id": r["id"], "source": r["source"], "degree": r["degree"]} for r in records]}

                else:
                    raise ValueError(f"Unsupported analysis type: {analysis_type}")

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return AnalysisResult(
                analysis_type=analysis_type,
                target=target or "graph",
                results=results,
                backend_used="memgraph",
                analysis_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Memgraph analysis failed: {e}")
            raise ConnectionError(f"Memgraph analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get Memgraph graph statistics"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        try:
            async with self.driver.session() as session:
                # Get node count
                node_result = await session.run("""
                    MATCH (k:KnowledgeEntry)
                    RETURN count(k) as node_count
                """)
                node_record = await node_result.single()
                node_count = node_record["node_count"] if node_record else 0

                # Get edge count
                edge_result = await session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as edge_count
                """)
                edge_record = await edge_result.single()
                edge_count = edge_record["edge_count"] if edge_record else 0

                # Get source distribution
                source_result = await session.run("""
                    MATCH (k:KnowledgeEntry)
                    RETURN k.source as source, count(k) as count
                    ORDER BY count DESC
                """)
                source_records = await source_result.data()
                source_stats = {r["source"]: r["count"] for r in source_records}

            return GraphStatistics(
                node_count=node_count,
                edge_count=edge_count,
                backend="memgraph",
                metadata={
                    "by_source": source_stats,
                    "license": "Apache 2.0"
                },
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get Memgraph statistics: {e}")
            raise ConnectionError(f"Memgraph statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from Memgraph"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        try:
            async with self.driver.session() as session:
                if output_format == 'json':
                    # Get nodes
                    node_result = await session.run("""
                        MATCH (k:KnowledgeEntry)
                        RETURN k.id as id, k.source as source, k.content as content,
                               k.metadata as metadata, k.timestamp as timestamp
                        LIMIT 100
                    """)
                    node_records = await node_result.data()
                    
                    # Get relationships
                    rel_result = await session.run("""
                        MATCH (a)-[r]->(b)
                        RETURN a.id as source, b.id as target, type(r) as type
                        LIMIT 100
                    """)
                    rel_records = await rel_result.data()
                    
                    return json.dumps({
                        "nodes": [{"id": r["id"], "source": r["source"], "content": r["content"][:100]} for r in node_records],
                        "edges": [{"source": r["source"], "target": r["target"], "type": r["type"]} for r in rel_records],
                        "backend": "memgraph",
                        "license": "Apache 2.0"
                    }, indent=2)

                elif output_format == 'html':
                    stats = await self.get_statistics()
                    
                    # Get recent nodes
                    recent_result = await session.run("""
                        MATCH (k:KnowledgeEntry)
                        RETURN k.id as id, k.source as source, k.content as content, k.timestamp as timestamp
                        ORDER BY k.timestamp DESC
                        LIMIT 10
                    """)
                    recent_records = await recent_result.data()
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Memgraph Knowledge Graph</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .stats {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                            .stat-item {{ margin: 10px 0; }}
                            .license {{ color: green; font-weight: bold; }}
                            .nodes {{ margin-top: 20px; }}
                            .node {{ background: white; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }}
                        </style>
                    </head>
                    <body>
                        <h1>Memgraph Knowledge Graph</h1>
                        <div class="stats">
                            <div class="stat-item"><strong>Backend:</strong> Memgraph</div>
                            <div class="stat-item license"><strong>License:</strong> Apache 2.0 (Permissive)</div>
                            <div class="stat-item"><strong>Nodes:</strong> {stats.node_count}</div>
                            <div class="stat-item"><strong>Edges:</strong> {stats.edge_count}</div>
                            <div class="stat-item"><strong>Sources:</strong> {len(stats.metadata.get('by_source', {}))}</div>
                        </div>
                        <h2>Recent Nodes</h2>
                        <div class="nodes">
                    """
                    
                    for record in recent_records:
                        html += f"""
                            <div class="node">
                                <strong>ID:</strong> {record['id']}<br>
                                <strong>Source:</strong> {record['source']}<br>
                                <strong>Content:</strong> {record['content'][:100]}...<br>
                                <strong>Timestamp:</strong> {record['timestamp']}
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
            logger.error(f"Memgraph visualization failed: {e}")
            raise ConnectionError(f"Memgraph visualization failed: {e}")

    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete knowledge from Memgraph"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        try:
            async with self.driver.session() as session:
                result = await session.run("""
                    MATCH (k:KnowledgeEntry {id: $id})
                    DETACH DELETE k
                    RETURN count(k) as deleted
                """, {'id': entry_id})
                
                record = await result.single()
                return record["deleted"] > 0 if record else False

        except Exception as e:
            logger.error(f"Memgraph delete failed: {e}")
            raise ConnectionError(f"Memgraph delete failed: {e}")

    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update knowledge in Memgraph"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        try:
            # Build dynamic SET clause
            set_clauses = []
            params = {'id': entry_id}
            
            for key, value in updates.items():
                if key != 'id':  # Don't update id
                    set_clauses.append(f"k.{key} = ${key}")
                    params[key] = json.dumps(value) if isinstance(value, dict) else value
            
            if not set_clauses:
                return False
            
            set_clause = ", ".join(set_clauses)
            
            async with self.driver.session() as session:
                result = await session.run(f"""
                    MATCH (k:KnowledgeEntry {{id: $id}})
                    SET {set_clause}, k.updated_at = $updated_at
                    RETURN count(k) as updated
                """, {**params, 'updated_at': datetime.now(timezone.utc).isoformat()})
                
                record = await result.single()
                return record["updated"] > 0 if record else False

        except Exception as e:
            logger.error(f"Memgraph update failed: {e}")
            raise ConnectionError(f"Memgraph update failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from Memgraph - Destructive operation"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        try:
            async with self.driver.session() as session:
                # Get count before deletion
                count_result = await session.run("""
                    MATCH (k:KnowledgeEntry)
                    RETURN count(k) as count
                """)
                count_record = await count_result.single()
                count = count_record["count"] if count_record else 0

                # Delete all KnowledgeEntry nodes and their relationships
                await session.run("""
                    MATCH (k:KnowledgeEntry)
                    DETACH DELETE k
                """)

                logger.warning(f"Cleared {count} nodes from Memgraph")
                return count

        except Exception as e:
            logger.error(f"Memgraph clear failed: {e}")
            raise ConnectionError(f"Memgraph clear failed: {e}")

    async def batch_add_knowledge(
        self,
        entries: List[KnowledgeEntry]
    ) -> List[str]:
        """Batch add knowledge to Memgraph efficiently"""
        if not self.is_healthy:
            raise ConnectionError("Memgraph backend not healthy")

        start_time = datetime.now(timezone.utc)
        ids = []

        try:
            async with self.driver.session() as session:
                for entry in entries:
                    entry_id = entry.id or str(uuid.uuid4())
                    ids.append(entry_id)
                    
                    await session.run("""
                        CREATE (k:KnowledgeEntry {
                            id: $id,
                            source: $source,
                            content: $content,
                            metadata: $metadata,
                            timestamp: $timestamp,
                            created_at: $created_at
                        })
                    """, {
                        'id': entry_id,
                        'source': entry.source,
                        'content': entry.content,
                        'metadata': json.dumps(entry.metadata or {}),
                        'timestamp': entry.timestamp or datetime.now(timezone.utc).isoformat(),
                        'created_at': datetime.now(timezone.utc).isoformat()
                    })

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Batch added {len(ids)} entries to Memgraph in {elapsed_ms:.2f}ms")

            return ids

        except Exception as e:
            logger.error(f"Memgraph batch add failed: {e}")
            raise ConnectionError(f"Memgraph batch add failed: {e}")
