"""
Knowledge Graph CRUD Operations

Create, Read, Update, Delete operations for the knowledge graph.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import KnowledgeNode, KnowledgeEdge, KnowledgeGraph
from .schema import NodeType, EdgeType
from .connection import ConnectionPool

logger = logging.getLogger(__name__)


class GraphCRUD:
    """CRUD operations for the knowledge graph"""
    
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
    
    # ===== Node Operations =====
    
    async def create_node(self, node: KnowledgeNode) -> str:
        """Create a new node in the graph"""
        labels = node.get_label_string()
        props = node.to_cypher_properties()
        
        query = f"""
        CREATE (n{labels} {props})
        RETURN n.id as id
        """
        
        result = await self.pool.run_cypher_write(query)
        logger.debug(f"Created node {node.id}")
        return node.id
    
    async def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID"""
        query = """
        MATCH (n {id: $node_id})
        RETURN n
        """
        
        results = await self.pool.run_cypher(query, {"node_id": node_id})
        
        if not results:
            return None
        
        # Convert result to KnowledgeNode
        node_data = results[0].get('n', {})
        return self._dict_to_node(node_data)
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update a node's properties"""
        # Add updated_at timestamp
        properties['updated_at'] = datetime.utcnow().isoformat()
        
        query = """
        MATCH (n {id: $node_id})
        SET n += $properties
        RETURN n.id as id
        """
        
        result = await self.pool.run_cypher_write(
            query, 
            {"node_id": node_id, "properties": properties}
        )
        
        return result.get("updated", 0) > 0
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships"""
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        """
        
        result = await self.pool.run_cypher_write(query, {"node_id": node_id})
        logger.debug(f"Deleted node {node_id}")
        return result.get("deleted", 0) > 0
    
    async def find_nodes(
        self,
        node_type: Optional[NodeType] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[KnowledgeNode]:
        """Find nodes by type and/or properties"""
        
        # Build query
        label = f":{node_type.value}" if node_type else ""
        
        if properties:
            # Build property filters
            prop_filters = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (n{label})
            WHERE {prop_filters}
            RETURN n
            LIMIT $limit
            """
            params = {**properties, "limit": limit}
        else:
            query = f"""
            MATCH (n{label})
            RETURN n
            LIMIT $limit
            """
            params = {"limit": limit}
        
        results = await self.pool.run_cypher(query, params)
        return [self._dict_to_node(r.get('n', {})) for r in results]
    
    # ===== Edge Operations =====
    
    async def create_edge(self, edge: KnowledgeEdge) -> str:
        """Create a new edge between nodes"""
        edge_type = edge.edge_type.value
        props = edge.to_cypher_pattern("a", "b")
        
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        CREATE (a)-[r:{edge_type} {{id: $edge_id, weight: $weight, confidence: $confidence}}]->(b)
        SET r += $properties
        RETURN r.id as id
        """
        
        params = {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_id": edge.id,
            "weight": edge.properties.weight,
            "confidence": edge.properties.confidence,
            "properties": edge.properties.metadata
        }
        
        result = await self.pool.run_cypher_write(query, params)
        logger.debug(f"Created edge {edge.id}")
        return edge.id
    
    async def get_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """Get an edge by ID"""
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        RETURN r, startNode(r) as source, endNode(r) as target
        """
        
        results = await self.pool.run_cypher(query, {"edge_id": edge_id})
        
        if not results:
            return None
        
        return self._dict_to_edge(results[0])
    
    async def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> bool:
        """Update an edge's properties"""
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        SET r += $properties
        RETURN r.id as id
        """
        
        result = await self.pool.run_cypher_write(
            query,
            {"edge_id": edge_id, "properties": properties}
        )
        
        return result.get("updated", 0) > 0
    
    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge"""
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        DELETE r
        """
        
        result = await self.pool.run_cypher_write(query, {"edge_id": edge_id})
        logger.debug(f"Deleted edge {edge_id}")
        return True
    
    async def find_edges(
        self,
        edge_type: Optional[EdgeType] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        limit: int = 100
    ) -> List[KnowledgeEdge]:
        """Find edges by type and/or node IDs"""
        
        conditions = []
        params = {"limit": limit}
        
        if source_id:
            conditions.append("source.id = $source_id")
            params["source_id"] = source_id
        
        if target_id:
            conditions.append("target.id = $target_id")
            params["target_id"] = target_id
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        if edge_type:
            query = f"""
            MATCH (source)-[r:{edge_type.value}]->(target)
            {where_clause}
            RETURN r, source, target
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (source)-[r]->(target)
            {where_clause}
            RETURN r, source, target
            LIMIT $limit
            """
        
        results = await self.pool.run_cypher(query, params)
        return [self._dict_to_edge(r) for r in results]
    
    # ===== Relationship Queries =====
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both"  # "out", "in", "both"
    ) -> List[KnowledgeNode]:
        """Get neighboring nodes"""
        
        edge_filter = f":{edge_type.value}" if edge_type else ""
        
        if direction == "out":
            query = f"""
            MATCH (n {{id: $node_id}})-[r{edge_filter}]->(neighbor)
            RETURN neighbor
            """
        elif direction == "in":
            query = f"""
            MATCH (n {{id: $node_id}})<-[r{edge_filter}]-(neighbor)
            RETURN neighbor
            """
        else:  # both
            query = f"""
            MATCH (n {{id: $node_id}})-[r{edge_filter}]-(neighbor)
            RETURN neighbor
            """
        
        results = await self.pool.run_cypher(query, {"node_id": node_id})
        return [self._dict_to_node(r.get('neighbor', {})) for r in results]
    
    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[KnowledgeNode]:
        """Find path between two nodes"""
        query = """
        MATCH path = shortestPath(
            (source {id: $source_id})-[*..$max_depth]-(target {id: $target_id})
        )
        RETURN [node in nodes(path) | node] as path_nodes
        """
        
        results = await self.pool.run_cypher(
            query,
            {"source_id": source_id, "target_id": target_id, "max_depth": max_depth}
        )
        
        if not results:
            return []
        
        path_nodes = results[0].get('path_nodes', [])
        return [self._dict_to_node(n) for n in path_nodes]
    
    # ===== Bulk Operations =====
    
    async def create_nodes_bulk(self, nodes: List[KnowledgeNode]) -> List[str]:
        """Create multiple nodes in batch"""
        node_ids = []
        for node in nodes:
            try:
                nid = await self.create_node(node)
                node_ids.append(nid)
            except Exception as e:
                logger.error(f"Failed to create node {node.id}: {e}")
        return node_ids
    
    async def create_edges_bulk(self, edges: List[KnowledgeEdge]) -> List[str]:
        """Create multiple edges in batch"""
        edge_ids = []
        for edge in edges:
            try:
                eid = await self.create_edge(edge)
                edge_ids.append(eid)
            except Exception as e:
                logger.error(f"Failed to create edge {edge.id}: {e}")
        return edge_ids
    
    # ===== Helper Methods =====
    
    def _dict_to_node(self, data: Dict[str, Any]) -> KnowledgeNode:
        """Convert dictionary to KnowledgeNode"""
        # Extract known properties
        node_type_str = data.get('node_type', 'CONCEPT')
        node_type = NodeType(node_type_str) if isinstance(node_type_str, str) else NodeType.CONCEPT
        
        from .models import NodeProperties
        
        props_data = {k: v for k, v in data.items() if k not in ['id', 'node_type', 'labels']}
        
        # Parse datetime strings
        for key in ['created_at', 'updated_at']:
            if key in props_data and isinstance(props_data[key], str):
                try:
                    props_data[key] = datetime.fromisoformat(props_data[key])
                except:
                    pass
        
        properties = NodeProperties(**props_data)
        
        return KnowledgeNode(
            id=data.get('id', ''),
            node_type=node_type,
            properties=properties,
            labels=data.get('labels', [node_type.value])
        )
    
    def _dict_to_edge(self, data: Dict[str, Any]) -> KnowledgeEdge:
        """Convert dictionary to KnowledgeEdge"""
        r_data = data.get('r', {})
        source_data = data.get('source', {})
        target_data = data.get('target', {})
        
        edge_type_str = r_data.get('edge_type', 'RELATED_TO')
        edge_type = EdgeType(edge_type_str) if isinstance(edge_type_str, str) else EdgeType.RELATED_TO
        
        from .models import EdgeProperties
        
        props_data = {k: v for k, v in r_data.items() if k not in ['id', 'edge_type']}
        properties = EdgeProperties(**props_data)
        
        return KnowledgeEdge(
            id=r_data.get('id', ''),
            edge_type=edge_type,
            source_id=source_data.get('id', ''),
            target_id=target_data.get('id', ''),
            properties=properties
        )
