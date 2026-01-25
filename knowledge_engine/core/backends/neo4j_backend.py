"""
Neo4j Backend Adapter for Unified Knowledge Graph Manager.

Provides graph database operations using Neo4j.
Follows CLAUDE.md principles: Runtime Truth, Configuration Explicitness, UTC.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .base import (
    KnowledgeGraphBackend,
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)

logger = logging.getLogger(__name__)


class Neo4jBackend(KnowledgeGraphBackend):
    """
    Neo4j backend adapter for knowledge graph storage and retrieval.

    Environment Variables Required:
        NEO4J_URI: Neo4j connection URI (e.g., bolt://localhost:7687)
        NEO4J_USER: Neo4j username
        NEO4J_PASSWORD: Neo4j password
        NEO4J_DATABASE: Database name (default: neo4j)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.NEO4J
        self.driver = None
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration - Law of Configuration Explicitness"""
        required = ['uri', 'user', 'password']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Neo4j backend requires '{key}' in config")

        self.uri = self.config['uri']
        self.user = self.config['user']
        self.password = self.config['password']
        self.database = self.config.get('database', 'neo4j')

        logger.info(f"Neo4j backend configured for database: {self.database}")

    async def connect(self) -> bool:
        """
        Establish connection to Neo4j - Runtime Truth principle.

        Returns:
            bool: True if connection successful
        """
        try:
            from neo4j import AsyncGraphDatabase

            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                connection_timeout=30.0
            )

            # Verify connection - Runtime Truth
            verified = await self._verify_connection()
            if verified:
                self.is_healthy = True
                logger.info(f"Successfully connected to Neo4j at {self.uri}")
            else:
                await self.disconnect()
                raise ConnectionError("Neo4j connection verification failed")

            return True

        except ImportError:
            logger.error("neo4j package not installed. Install with: pip install neo4j")
            raise ImportError("neo4j package required for Neo4jBackend")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Neo4j connection failed: {e}")

    async def _verify_connection(self) -> bool:
        """Verify connection is working - Runtime Truth"""
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 AS num")
                record = await result.single()
                return record is not None and record["num"] == 1
        except Exception as e:
            logger.error(f"Neo4j connection verification failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.is_healthy = False
            logger.info("Disconnected from Neo4j")

    async def health_check(self) -> bool:
        """Check Neo4j health"""
        if not self.driver:
            return False

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run("RETURN 1")
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            self.is_healthy = False
            return False

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """
        Add knowledge to Neo4j graph.

        Creates:
        - A Knowledge node with the content
        - Entity nodes extracted from content (simple extraction)
        - Relationships between entities and knowledge
        """
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        start_time = datetime.utcnow()

        try:
            async with self.driver.session(database=self.database) as session:
                # Create Knowledge node
                query = """
                CREATE (k:Knowledge {
                    id: randomUUID(),
                    source: $source,
                    content: $content,
                    metadata: $metadata,
                    created_at: datetime($timestamp)
                })
                RETURN k.id AS id
                """

                result = await session.run(
                    query,
                    source=entry.source,
                    content=entry.content,
                    metadata=json.dumps(entry.metadata or {}),
                    timestamp=entry.timestamp
                )

                record = await result.single()
                entry_id = record["id"]

                # Extract and create entity nodes (simple word extraction)
                # In production, use NLP for proper entity extraction
                words = entry.content.split()
                entities = list(set([w for w in words if len(w) > 3 and w.isalnum()]))

                for entity_name in entities[:5]:  # Limit to top 5 for demo
                    await session.run("""
                        MERGE (e:Entity {name: $name})
                        WITH e, k
                        MATCH (k:Knowledge {id: $entry_id})
                        CREATE (k)-[:MENTIONS]->(e)
                    """, name=entity_name, entry_id=entry_id)

                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.info(f"Added knowledge to Neo4j in {elapsed_ms:.2f}ms: {entry_id}")

                return entry_id

        except Exception as e:
            logger.error(f"Failed to add knowledge to Neo4j: {e}")
            raise ConnectionError(f"Neo4j add_knowledge failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Search knowledge in Neo4j using Cypher"""
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        start_time = datetime.utcnow()

        try:
            async with self.driver.session(database=self.database) as session:
                # Full-text search query
                cypher = """
                MATCH (k:Knowledge)
                WHERE k.content CONTAINS $query
                   OR k.source CONTAINS $query
                """

                params = {"query": query, "limit": limit, "skip": offset}

                # Apply filters
                if filters:
                    if "source" in filters:
                        cypher += " AND k.source = $source"
                        params["source"] = filters["source"]
                    if "date_after" in filters:
                        cypher += " AND k.created_at >= datetime($date_after)"
                        params["date_after"] = filters["date_after"]

                cypher += f"""
                RETURN k, [(k)-[:MENTIONS]->(e) | e.name] AS entities
                ORDER BY k.created_at DESC
                SKIP $skip LIMIT $limit
                """

                result = await session.run(cypher, **params)

                results = []
                async for record in result:
                    node = record["k"]
                    results.append({
                        "id": node["id"],
                        "source": node["source"],
                        "content": node["content"],
                        "metadata": json.loads(node.get("metadata", "{}")),
                        "created_at": node["created_at"].isoformat(),
                        "entities": record["entities"]
                    })

                # Get total count
                count_query = "MATCH (k:Knowledge) WHERE k.content CONTAINS $query RETURN count(k) AS total"
                if filters and "source" in filters:
                    count_query = count_query.replace(
                        "RETURN",
                        f"AND k.source = $source RETURN"
                    )

                count_result = await session.run(
                    count_query,
                    query=query,
                    source=filters.get("source") if filters else None
                )
                count_record = await count_result.single()
                total_count = count_record["total"]

                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                return SearchResults(
                    query=query,
                    results=results,
                    total_count=total_count,
                    backend_used="neo4j",
                    search_time_ms=elapsed_ms,
                    metadata={"filters": filters}
                )

        except Exception as e:
            logger.error(f"Neo4j search failed: {e}")
            raise ConnectionError(f"Neo4j search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze the Neo4j knowledge graph"""
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        start_time = datetime.utcnow()

        try:
            async with self.driver.session(database=self.database) as session:
                if analysis_type == "connected_components":
                    # Find connected components in the graph
                    query = """
                    MATCH (k:Knowledge)
                    WITH count(k) AS total_knowledge
                    MATCH (e:Entity)
                    WITH total_knowledge, count(e) AS total_entities
                    MATCH ()-[r:MENTIONS]->()
                    RETURN total_knowledge, total_entities, count(r) AS total_mentions
                    """
                    result = await session.run(query)
                    record = await result.single()

                    results = {
                        "total_knowledge_nodes": record["total_knowledge"],
                        "total_entity_nodes": record["total_entities"],
                        "total_mention_edges": record["total_mentions"]
                    }

                elif analysis_type == "entity_connections":
                    # Find most connected entities
                    query = """
                    MATCH (e:Entity)<-[r:MENTIONS]-()
                    RETURN e.name AS entity, count(r) AS connections
                    ORDER BY connections DESC LIMIT 10
                    """
                    result = await session.run(query)
                    results = {"top_entities": []}
                    async for record in result:
                        results["top_entities"].append({
                            "entity": record["entity"],
                            "connections": record["connections"]
                        })

                elif analysis_type == "knowledge_by_source":
                    # Analyze knowledge distribution by source
                    query = """
                    MATCH (k:Knowledge)
                    RETURN k.source AS source, count(k) AS count
                    ORDER BY count DESC
                    """
                    result = await session.run(query)
                    results = {"by_source": []}
                    async for record in result:
                        results["by_source"].append({
                            "source": record["source"],
                            "count": record["count"]
                        })

                else:
                    raise ValueError(f"Unsupported analysis type: {analysis_type}")

                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                return AnalysisResult(
                    analysis_type=analysis_type,
                    target=target or "graph",
                    results=results,
                    backend_used="neo4j",
                    analysis_time_ms=elapsed_ms
                )

        except Exception as e:
            logger.error(f"Neo4j analysis failed: {e}")
            raise ConnectionError(f"Neo4j analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get Neo4j graph statistics"""
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        try:
            async with self.driver.session(database=self.database) as session:
                query = """
                MATCH (k:Knowledge)
                WITH count(k) AS knowledge_count
                MATCH (e:Entity)
                WITH knowledge_count, count(e) AS entity_count
                MATCH ()-[r:MENTIONS]->()
                RETURN knowledge_count, entity_count, count(r) AS edge_count
                """

                result = await session.run(query)
                record = await result.single()

                return GraphStatistics(
                    node_count=record["knowledge_count"] + record["entity_count"],
                    edge_count=record["edge_count"],
                    backend="neo4j",
                    metadata={
                        "knowledge_nodes": record["knowledge_count"],
                        "entity_nodes": record["entity_count"],
                        "mention_edges": record["edge_count"],
                        "database": self.database
                    },
                    timestamp=datetime.utcnow().isoformat()
                )

        except Exception as e:
            logger.error(f"Failed to get Neo4j statistics: {e}")
            raise ConnectionError(f"Neo4j statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from Neo4j"""
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        try:
            async with self.driver.session(database=self.database) as session:
                if output_format == 'json':
                    # Export graph as JSON
                    query = """
                    MATCH (k:Knowledge)
                    OPTIONAL MATCH (k)-[r:MENTIONS]->(e:Entity)
                    RETURN {
                        id: k.id,
                        type: 'knowledge',
                        properties: {content: k.content, source: k.source}
                    } AS node,
                    CASE WHEN e IS NOT NULL THEN [{
                        from: k.id,
                        to: e.name,
                        type: 'MENTIONS'
                    }] ELSE [] END AS edges
                    LIMIT 100
                    """

                    result = await session.run(query)

                    nodes = []
                    edges = []

                    async for record in result:
                        nodes.append(record["node"])
                        edges.extend(record["edges"])

                    return json.dumps({
                        "nodes": nodes,
                        "edges": edges
                    }, indent=2)

                elif output_format == 'html':
                    # Generate HTML visualization using vis.js
                    query = """
                    MATCH (k:Knowledge)
                    OPTIONAL MATCH (k)-[r:MENTIONS]->(e:Entity)
                    RETURN k.id AS id, k.content AS label, 'knowledge' AS type,
                           collect({id: e.name, label: e.name, type: 'entity'}) AS entities
                    LIMIT 50
                    """

                    result = await session.run(query)

                    nodes = []
                    edges_set = set()

                    async for record in result:
                        nodes.append({
                            "id": record["id"],
                            "label": record["label"][:50] + "...",
                            "title": record["label"],
                            "group": record["type"]
                        })

                        for entity in record["entities"]:
                            if entity["id"]:
                                nodes.append(entity)
                                edge_id = f"{record['id']}-{entity['id']}"
                                if edge_id not in edges_set:
                                    edges_set.add(edge_id)
                                    nodes.append({
                                        "from": record["id"],
                                        "to": entity["id"],
                                        "label": "MENTIONS"
                                    })

                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Neo4j Knowledge Graph</title>
                        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                        <style>
                            body {{ margin: 0; padding: 0; }}
                            #mynetwork {{ width: 100vw; height: 100vh; border: 1px solid lightgray; }}
                        </style>
                    </head>
                    <body>
                        <div id="mynetwork"></div>
                        <script>
                            var nodes = new vis.DataSet({json.dumps(nodes)});
                            var data = {{
                                nodes: nodes
                            }};
                            var options = {{
                                nodes: {{ shape: 'box' }},
                                physics: {{ enabled: true }}
                            }};
                            var network = new vis.Network(document.getElementById('mynetwork'), data, options);
                        </script>
                    </body>
                    </html>
                    """
                    return html

                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Neo4j visualization failed: {e}")
            raise ConnectionError(f"Neo4j visualization failed: {e}")

    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete knowledge from Neo4j"""
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        try:
            async with self.driver.session(database=self.database) as session:
                query = """
                MATCH (k:Knowledge {id: $entry_id})
                DETACH DELETE k
                RETURN count(k) AS deleted
                """
                result = await session.run(query, entry_id=entry_id)
                record = await result.single()
                return record["deleted"] > 0

        except Exception as e:
            logger.error(f"Neo4j delete failed: {e}")
            raise ConnectionError(f"Neo4j delete failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from Neo4j - Destructive operation"""
        if not self.is_healthy:
            raise ConnectionError("Neo4j backend not healthy")

        try:
            async with self.driver.session(database=self.database) as session:
                # Count before deletion
                count_result = await session.run("MATCH (k:Knowledge) RETURN count(k) AS count")
                count_record = await count_result.single()
                count = count_record["count"]

                # Delete all
                await session.run("MATCH (n) DETACH DELETE n")

                logger.warning(f"Cleared {count} nodes from Neo4j")
                return count

        except Exception as e:
            logger.error(f"Neo4j clear failed: {e}")
            raise ConnectionError(f"Neo4j clear failed: {e}")
