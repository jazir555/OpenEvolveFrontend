"""
Cypher Query Builder

Builds Cypher queries programmatically with type safety.

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

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from .schema import NodeType, EdgeType


class ComparisonOperator(Enum):
    """Comparison operators for WHERE clauses"""
    EQUALS = "="
    NOT_EQUALS = "<>"
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
    IN = "IN"
    REGEX = "=~"


@dataclass
class Condition:
    """A condition for WHERE clauses"""
    field: str
    operator: ComparisonOperator
    value: Any
    
    def to_cypher(self, param_name: str) -> str:
        """Convert to Cypher syntax"""
        if self.operator in (ComparisonOperator.CONTAINS, 
                             ComparisonOperator.STARTS_WITH,
                             ComparisonOperator.ENDS_WITH):
            return f"{self.field} {self.operator.value} ${param_name}"
        elif self.operator == ComparisonOperator.IN:
            return f"{self.field} {self.operator.value} ${param_name}"
        else:
            return f"{self.field} {self.operator.value} ${param_name}"


@dataclass
class NodePattern:
    """Pattern for matching nodes"""
    variable: str = "n"
    node_type: Optional[NodeType] = None
    labels: List[str] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)
    
    def to_cypher(self, param_offset: int = 0) -> tuple:
        """Convert to Cypher pattern string and parameters"""
        # Build labels
        labels = []
        if self.node_type:
            labels.append(self.node_type.value)
        labels.extend(self.labels)
        
        label_str = ":" + ":".join(labels) if labels else ""
        
        # Build conditions
        params = {}
        if self.conditions:
            conditions_str = []
            for i, cond in enumerate(self.conditions):
                param_name = f"{self.variable}_p{i}"
                conditions_str.append(cond.to_cypher(param_name))
                params[param_name] = cond.value
            
            where_clause = " AND ".join(conditions_str)
            return f"({self.variable}{label_str} {{{where_clause}}})", params
        
        return f"({self.variable}{label_str})", {}


@dataclass
class EdgePattern:
    """Pattern for matching edges"""
    variable: str = "r"
    edge_type: Optional[EdgeType] = None
    direction: str = "->"  # "->", "<-", "-"
    conditions: List[Condition] = field(default_factory=list)
    min_hops: Optional[int] = None
    max_hops: Optional[int] = None
    
    def to_cypher(self, param_offset: int = 0) -> tuple:
        """Convert to Cypher pattern string and parameters"""
        # Build type
        type_str = f":{self.edge_type.value}" if self.edge_type else ""
        
        # Build hop range
        if self.min_hops is not None or self.max_hops is not None:
            min_h = self.min_hops if self.min_hops is not None else ""
            max_h = self.max_hops if self.max_hops is not None else ""
            hop_str = f"*{min_h}..{max_h}"
        else:
            hop_str = ""
        
        # Build conditions
        params = {}
        if self.conditions:
            conditions_str = []
            for i, cond in enumerate(self.conditions):
                param_name = f"{self.variable}_p{i}"
                conditions_str.append(cond.to_cypher(param_name))
                params[param_name] = cond.value
            
            where_clause = "{" + ", ".join(conditions_str) + "}"
            edge_def = f"[{self.variable}{type_str}{hop_str} {where_clause}]"
        else:
            edge_def = f"[{self.variable}{type_str}{hop_str}]"
        
        # Build direction
        if self.direction == "->":
            return f"-{edge_def}->", params
        elif self.direction == "<-":

            return f"<-{edge_def}-", params
        else:
            return f"-{edge_def}-", params


class CypherQueryBuilder:
    """Builder for Cypher queries"""
    
    def __init__(self):
        self.match_clauses: List[str] = []
        self.where_conditions: List[str] = []
        self.return_clauses: List[str] = []
        self.order_by: List[str] = []
        self.limit_val: Optional[int] = None
        self.skip_val: Optional[int] = None
        self.parameters: Dict[str, Any] = {}
        self.param_counter = 0
    
    def match_node(
        self,
        variable: str = "n",
        node_type: Optional[NodeType] = None,
        labels: Optional[List[str]] = None,
        **properties
    ) -> 'CypherQueryBuilder':
        """Add a node match clause"""
        pattern = NodePattern(
            variable=variable,
            node_type=node_type,
            labels=labels or []
        )
        
        cypher, params = pattern.to_cypher()
        
        # Add property conditions
        if properties:
            prop_conditions = []
            for key, value in properties.items():
                param_name = f"{variable}_{key}"
                prop_conditions.append(f"{variable}.{key} = ${param_name}")
                self.parameters[param_name] = value
            
            if prop_conditions:
                cypher = cypher.replace(")", f" {{ {', '.join(prop_conditions)} }})")
        
        self.match_clauses.append(f"MATCH {cypher}")
        self.parameters.update(params)
        return self
    
    def match_path(
        self,
        source_var: str = "a",
        edge_pattern: Optional[EdgePattern] = None,
        target_var: str = "b",
        source_type: Optional[NodeType] = None,
        target_type: Optional[NodeType] = None
    ) -> 'CypherQueryBuilder':
        """Add a path match clause"""
        # Source node
        source_label = f":{source_type.value}" if source_type else ""
        
        # Edge
        edge = edge_pattern or EdgePattern()
        edge_cypher, edge_params = edge.to_cypher()
        
        # Target node
        target_label = f":{target_type.value}" if target_type else ""
        
        pattern = f"({source_var}{source_label}){edge_cypher}({target_var}{target_label})"
        
        self.match_clauses.append(f"MATCH {pattern}")
        self.parameters.update(edge_params)
        return self
    
    def where(self, condition: str, **params) -> 'CypherQueryBuilder':
        """Add a WHERE condition"""
        self.where_conditions.append(condition)
        self.parameters.update(params)
        return self
    
    def where_equals(self, field: str, value: Any) -> 'CypherQueryBuilder':
        """Add equality condition"""
        param_name = f"p{self.param_counter}"
        self.param_counter += 1
        self.where_conditions.append(f"{field} = ${param_name}")
        self.parameters[param_name] = value
        return self
    
    def where_contains(self, field: str, value: str) -> 'CypherQueryBuilder':
        """Add CONTAINS condition"""
        param_name = f"p{self.param_counter}"
        self.param_counter += 1
        self.where_conditions.append(f"{field} CONTAINS ${param_name}")
        self.parameters[param_name] = value
        return self
    
    def where_in(self, field: str, values: List[Any]) -> 'CypherQueryBuilder':
        """Add IN condition"""
        param_name = f"p{self.param_counter}"
        self.param_counter += 1
        self.where_conditions.append(f"{field} IN ${param_name}")
        self.parameters[param_name] = values
        return self
    
    def return_(self, *clauses) -> 'CypherQueryBuilder':
        """Add RETURN clause"""
        self.return_clauses.extend(clauses)
        return self
    
    def return_all(self) -> 'CypherQueryBuilder':
        """Return all nodes"""
        self.return_clauses.append("*")
        return self
    
    def return_node(self, variable: str, include_properties: bool = True) -> 'CypherQueryBuilder':
        """Return a node variable"""
        if include_properties:
            self.return_clauses.append(f"{variable}")
        else:
            self.return_clauses.append(f"id({variable}) as {variable}_id")
        return self
    
    def order_by(self, field: str, descending: bool = False) -> 'CypherQueryBuilder':
        """Add ORDER BY clause"""
        direction = "DESC" if descending else "ASC"
        self.order_by.append(f"{field} {direction}")
        return self
    
    def limit(self, n: int) -> 'CypherQueryBuilder':
        """Add LIMIT clause"""
        self.limit_val = n
        return self
    
    def skip(self, n: int) -> 'CypherQueryBuilder':
        """Add SKIP clause"""
        self.skip_val = n
        return self
    
    def paginate(self, page: int, per_page: int) -> 'CypherQueryBuilder':
        """Add pagination (SKIP and LIMIT)"""
        self.skip_val = (page - 1) * per_page
        self.limit_val = per_page
        return self
    
    def build(self) -> tuple:
        """Build the complete Cypher query and parameters"""
        parts = []
        
        # MATCH clauses
        parts.extend(self.match_clauses)
        
        # WHERE clause
        if self.where_conditions:
            parts.append("WHERE " + " AND ".join(self.where_conditions))
        
        # RETURN clause
        if self.return_clauses:
            parts.append("RETURN " + ", ".join(self.return_clauses))
        
        # ORDER BY clause
        if self.order_by:
            parts.append("ORDER BY " + ", ".join(self.order_by))
        
        # SKIP clause
        if self.skip_val is not None:
            parts.append(f"SKIP {self.skip_val}")
        
        # LIMIT clause
        if self.limit_val is not None:
            parts.append(f"LIMIT {self.limit_val}")
        
        query = "\n".join(parts)
        return query, self.parameters
    
    def reset(self) -> 'CypherQueryBuilder':
        """Reset the builder for a new query"""
        self.match_clauses = []
        self.where_conditions = []
        self.return_clauses = []
        self.order_by = []
        self.limit_val = None
        self.skip_val = None
        self.parameters = {}
        self.param_counter = 0
        return self
    
    # ===== Common Query Patterns =====
    
    @staticmethod
    def find_by_id(node_id: str) -> tuple:
        """Query to find node by ID"""
        return (
            "MATCH (n {id: $node_id}) RETURN n",
            {"node_id": node_id}
        )
    
    @staticmethod
    def find_by_name(name: str, node_type: Optional[NodeType] = None) -> tuple:
        """Query to find node by name"""
        label = f":{node_type.value}" if node_type else ""
        return (
            f"MATCH (n{label} {{name: $name}}) RETURN n",
            {"name": name}
        )
    
    @staticmethod
    def find_neighbors(
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both"
    ) -> tuple:
        """Query to find neighbors"""
        edge_filter = f":{edge_type.value}" if edge_type else ""
        
        if direction == "out":
            query = f"""
                MATCH (n {{id: $node_id}})-[r{edge_filter}]->(neighbor)
                RETURN neighbor, r
            """
        elif direction == "in":
            query = f"""
                MATCH (n {{id: $node_id}})<-[r{edge_filter}]-(neighbor)
                RETURN neighbor, r
            """
        else:
            query = f"""
                MATCH (n {{id: $node_id}})-[r{edge_filter}]-(neighbor)
                RETURN neighbor, r
            """
        
        return query, {"node_id": node_id}
    
    @staticmethod
    def search_nodes(
        search_term: str,
        node_type: Optional[NodeType] = None,
        limit: int = 10
    ) -> tuple:
        """Query to search nodes by name"""
        label = f":{node_type.value}" if node_type else ""
        return (
            f"""
            MATCH (n{label})
            WHERE n.name CONTAINS $search_term
            RETURN n
            LIMIT $limit
            """,
            {"search_term": search_term, "limit": limit}
        )
    
    @staticmethod
    def create_node(
        node_type: NodeType,
        properties: Dict[str, Any]
    ) -> tuple:
        """Query to create a node"""
        label = node_type.value
        return (
            f"""
            CREATE (n:{label} $properties)
            RETURN n
            """,
            {"properties": properties}
        )
    
    @staticmethod
    def create_relationship(
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        properties: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """Query to create a relationship"""
        return (
            f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[r:{edge_type.value} $properties]->(b)
            RETURN r
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "properties": properties or {}
            }
        )
    
    @staticmethod
    def shortest_path(source_id: str, target_id: str, max_depth: int = 10) -> tuple:
        """Query to find shortest path"""
        return (
            """
            MATCH path = shortestPath(
                (a {id: $source_id})-[*..$max_depth]-(b {id: $target_id})
            )
            RETURN path
            """,
            {"source_id": source_id, "target_id": target_id, "max_depth": max_depth}
        )
    
    @staticmethod
    def delete_node(node_id: str) -> tuple:
        """Query to delete a node"""
        return (
            "MATCH (n {id: $node_id}) DETACH DELETE n",
            {"node_id": node_id}
        )
