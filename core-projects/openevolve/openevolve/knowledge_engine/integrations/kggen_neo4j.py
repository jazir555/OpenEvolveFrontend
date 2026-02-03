"""
Neo4j Integration for KG-Gen Pipeline

This module provides Neo4j upload capabilities for knowledge graphs,
including batch operations, index management, and progress tracking.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .kggen_pipeline import KnowledgeGraph, UploadResult

logger = logging.getLogger(__name__)


class Neo4jGraphUploader:
    """
    Upload kg-gen knowledge graphs to Neo4j.

    Features:
    - Batch upload
    - Entity/relationship creation
    - Index management
    - Progress tracking
    """

    def __init__(self, neo4j_driver):
        """
        Initialize Neo4j uploader.

        Args:
            neo4j_driver: Neo4j driver instance
        """
        self.driver = neo4j_driver
        self._indices_created = False
        logger.info("Neo4jGraphUploader initialized")

    async def upload_graph(
        self,
        graph: KnowledgeGraph,
        batch_size: int = 100,
        create_indices: bool = True,
        verify_upload: bool = True
    ) -> UploadResult:
        """
        Upload knowledge graph to Neo4j in batches.

        Args:
            graph: Knowledge graph to upload
            batch_size: Batch size for uploads
            create_indices: Whether to create indices
            verify_upload: Whether to verify upload after completion

        Returns:
            UploadResult with upload statistics
        """
        logger.info(
            f"Starting upload: {len(graph.entities)} entities, "
            f"{len(graph.relationships)} relationships"
        )

        try:
            # Create indices if requested
            if create_indices and not self._indices_created:
                await self._create_indices()
                self._indices_created = True

            # Upload entities
            entities_uploaded = await self.create_entities(
                graph.entities,
                batch_size=batch_size
            )

            # Upload relationships
            relationships_uploaded = await self.create_relationships(
                graph.relationships,
                batch_size=batch_size
            )

            # Create entity clusters if available
            if graph.entity_clusters:
                await self.create_entity_clusters(graph.entity_clusters)

            # Verify upload if requested
            if verify_upload:
                verification = await self._verify_upload(
                    entities_uploaded,
                    relationships_uploaded
                )
                logger.info(f"Upload verification: {verification}")

            logger.info(
                f"Upload complete: {entities_uploaded} entities, "
                f"{relationships_uploaded} relationships"
            )

            return UploadResult(
                success=True,
                entities_uploaded=entities_uploaded,
                relationships_uploaded=relationships_uploaded
            )

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return UploadResult(
                success=False,
                error=str(e)
            )

    async def create_entities(
        self,
        entities: List[str],
        batch_size: int = 100
    ) -> int:
        """
        Create entity nodes in Neo4j.

        Args:
            entities: List of entity names
            batch_size: Batch size for uploads

        Returns:
            Number of entities created
        """
        logger.info(f"Creating {len(entities)} entity nodes")

        created_count = 0

        # Process in batches
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {name: entity})
            ON CREATE SET e.created_at = datetime()
            RETURN count(e) as created
            """

            try:
                with self.driver.session() as session:
                    result = session.run(
                        query,
                        entities=batch
                    )
                    record = result.single()
                    batch_created = record['created']
                    created_count += batch_created

                    logger.debug(f"Created {batch_created} entities in batch {i//batch_size + 1}")

            except Exception as e:
                logger.error(f"Error creating entity batch {i//batch_size + 1}: {e}")
                # Continue with next batch

        logger.info(f"Created {created_count} entity nodes")
        return created_count

    async def create_relationships(
        self,
        relationships: List[Tuple[str, str, str]],
        batch_size: int = 100
    ) -> int:
        """
        Create relationship edges in Neo4j.

        Args:
            relationships: List of (subject, predicate, object) triples
            batch_size: Batch size for uploads

        Returns:
            Number of relationships created
        """
        logger.info(f"Creating {len(relationships)} relationship edges")

        created_count = 0

        # Convert tuples to dicts for Neo4j
        rel_data = [
            {
                'subject': s,
                'predicate': p,
                'object': o
            }
            for s, p, o in relationships
        ]

        # Process in batches
        for i in range(0, len(rel_data), batch_size):
            batch = rel_data[i:i + batch_size]

            query = """
            UNWIND $relationships AS rel
            MATCH (s:Entity {name: rel.subject})
            MATCH (o:Entity {name: rel.object})
            MERGE (s)-[r:RELATES_TO {predicate: rel.predicate}]->(o)
            ON CREATE SET r.created_at = datetime()
            RETURN count(r) as created
            """

            try:
                with self.driver.session() as session:
                    result = session.run(
                        query,
                        relationships=batch
                    )
                    record = result.single()
                    batch_created = record['created']
                    created_count += batch_created

                    logger.debug(
                        f"Created {batch_created} relationships in batch {i//batch_size + 1}"
                    )

            except Exception as e:
                logger.error(f"Error creating relationship batch {i//batch_size + 1}: {e}")
                # Continue with next batch

        logger.info(f"Created {created_count} relationship edges")
        return created_count

    async def create_entity_clusters(
        self,
        clusters: Dict[str, List[str]]
    ) -> int:
        """
        Create entity cluster relationships.

        Args:
            clusters: Dictionary mapping cluster IDs to entity lists

        Returns:
            Number of cluster relationships created
        """
        logger.info(f"Creating {len(clusters)} entity clusters")

        created_count = 0

        for cluster_id, entities in clusters.items():
            query = """
            UNWIND $entities AS entity_name
            MATCH (e:Entity {name: entity_name})
            MERGE (c:Cluster {id: $cluster_id})
            MERGE (e)-[r:IN_CLUSTER]->(c)
            ON CREATE SET r.created_at = datetime()
            RETURN count(r) as created
            """

            try:
                with self.driver.session() as session:
                    result = session.run(
                        query,
                        cluster_id=cluster_id,
                        entities=entities
                    )
                    record = result.single()
                    batch_created = record['created']
                    created_count += batch_created

            except Exception as e:
                logger.error(f"Error creating cluster {cluster_id}: {e}")

        logger.info(f"Created {created_count} cluster relationships")
        return created_count

    async def _create_indices(self):
        """
        Create indices for better performance.
        """
        queries = [
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX rel_predicate_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.predicate)",
            "CREATE INDEX cluster_id_index IF NOT EXISTS FOR (c:Cluster) ON (c.id)"
        ]

        for query in queries:
            try:
                with self.driver.session() as session:
                    session.run(query)
                    logger.info(f"Created index: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

        logger.info("Index creation complete")

    async def _verify_upload(
        self,
        expected_entities: int,
        expected_relationships: int
    ) -> Dict[str, Any]:
        """
        Verify that entities and relationships were uploaded correctly.

        Args:
            expected_entities: Expected number of entities
            expected_relationships: Expected number of relationships

        Returns:
            Verification results
        """
        verification = {
            'expected_entities': expected_entities,
            'expected_relationships': expected_relationships,
            'actual_entities': 0,
            'actual_relationships': 0,
            'entity_match': False,
            'relationship_match': False
        }

        try:
            # Count entities
            with self.driver.session() as session:
                result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                verification['actual_entities'] = result.single()['count']

            # Count relationships
            with self.driver.session() as session:
                result = session.run(
                    "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
                )
                verification['actual_relationships'] = result.single()['count']

            # Check matches
            verification['entity_match'] = (
                verification['actual_entities'] >= expected_entities
            )
            verification['relationship_match'] = (
                verification['actual_relationships'] >= expected_relationships
            )

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification['error'] = str(e)

        return verification

    async def delete_graph(self) -> int:
        """
        Delete all knowledge graph data from Neo4j.

        Returns:
            Number of nodes deleted
        """
        logger.warning("Deleting all knowledge graph data")

        query = """
        MATCH (e:Entity)
        DETACH DELETE e
        RETURN count(e) as deleted
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)
                deleted = result.single()['deleted']
                logger.info(f"Deleted {deleted} nodes")
                return deleted

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0

    async def query_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Query a single entity from Neo4j.

        Args:
            entity_name: Name of entity to query

        Returns:
            Entity data or None if not found
        """
        query = """
        MATCH (e:Entity {name: $entity_name})
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(related)
        RETURN e as entity, collect({predicate: r.predicate, target: related.name}) as relationships
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, entity_name=entity_name)
                record = result.single()

                if record:
                    return {
                        'entity': record['entity'],
                        'relationships': record['relationships']
                    }
                else:
                    return None

        except Exception as e:
            logger.error(f"Query failed for entity {entity_name}: {e}")
            return None

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph in Neo4j.

        Returns:
            Dictionary with graph statistics
        """
        stats = {}

        try:
            with self.driver.session() as session:
                # Count entities
                result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                stats['entity_count'] = result.single()['count']

                # Count relationships
                result = session.run(
                    "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
                )
                stats['relationship_count'] = result.single()['count']

                # Count clusters
                result = session.run("MATCH (c:Cluster) RETURN count(c) as count")
                stats['cluster_count'] = result.single()['count']

                # Get relationship types
                result = session.run(
                    "MATCH ()-[r:RELATES_TO]->() RETURN DISTINCT r.predicate as type LIMIT 100"
                )
                stats['relationship_types'] = [
                    record['type'] for record in result
                ]

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            stats['error'] = str(e)

        return stats

    async def export_graph(self, format: str = 'json') -> str:
        """
        Export knowledge graph from Neo4j.

        Args:
            format: Export format ('json', 'csv', 'graphml')

        Returns:
            Exported graph data as string
        """
        logger.info(f"Exporting graph in {format} format")

        if format == 'json':
            return await self._export_json()
        elif format == 'csv':
            return await self._export_csv()
        elif format == 'graphml':
            return await self._export_graphml()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _export_json(self) -> str:
        """Export graph as JSON."""
        import json

        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(o:Entity)
        RETURN {
            entity: e.name,
            relationships: collect({
                predicate: r.predicate,
                target: o.name
            })
        } as node_data
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)
                nodes = [record['node_data'] for record in result]

            return json.dumps({
                'nodes': nodes,
                'exported_at': datetime.now().isoformat()
            }, indent=2)

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return json.dumps({'error': str(e)})

    async def _export_csv(self) -> str:
        """Export relationships as CSV."""
        query = """
        MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
        RETURN s.name as subject, r.predicate as predicate, o.name as object
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)

                lines = ['subject,predicate,object']
                for record in result:
                    lines.append(
                        f'"{record["subject"]}","{record["predicate"]}","{record["object"]}"'
                    )

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return f'error: {str(e)}'

    async def _export_graphml(self) -> str:
        """Export graph as GraphML (simplified)."""
        # This is a simplified GraphML export
        # A full implementation would use a proper GraphML library

        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(o:Entity)
        RETURN e.name as entity, r.predicate as predicate, o.name as object
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)

                graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
                graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
                graphml.append('  <graph id="G" edgedefault="directed">')

                entities = set()

                for record in result:
                    if record['entity']:
                        entity_id = record['entity'].replace(' ', '_')
                        if entity_id not in entities:
                            graphml.append(
                                f'    <node id="{entity_id}"><data key="name">{record["entity"]}</data></node>'
                            )
                            entities.add(entity_id)

                    if record['predicate'] and record['object']:
                        source_id = record['entity'].replace(' ', '_')
                        target_id = record['object'].replace(' ', '_')
                        graphml.append(
                            f'    <edge source="{source_id}" target="{target_id}">'
                            f'<data key="predicate">{record["predicate"]}</data></edge>'
                        )

                graphml.append('  </graph>')
                graphml.append('</graphml>')

                return '\n'.join(graphml)

        except Exception as e:
            logger.error(f"GraphML export failed: {e}")
            return f'<?xml version="1.0"?><error>{str(e)}</error>'
