"""
FDG Extractor

Extract Functional Dependency Graphs from domain constraints.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import List, Dict, Any, Set
import re
from .fdg import FunctionalDependencyGraph, Node, Edge, EdgeType
from .domain import Domain


class FDGExtractor:
    """
    Extract Functional Dependency Graphs from domain data
    """

    def __init__(self, use_causal_discovery: bool = True):
        self.use_causal_discovery = use_causal_discovery

    def extract(self, domain: Domain) -> FunctionalDependencyGraph:
        """
        Extract FDG from domain

        Args:
            domain: Domain with constraints

        Returns:
            Extracted FDG
        """
        fdg = FunctionalDependencyGraph()

        # Step 1: Parse constraints
        constraints = self._parse_constraints(domain)

        # Step 2: Extract variables as nodes
        nodes = self._extract_variables(constraints)
        for var in nodes:
            node = Node(
                id=var['name'],
                variable=var['name'],
                constraint_type=var['type'],
                metadata=var.get('metadata', {})
            )
            fdg.add_node(node)

        # Step 3: Build edges from dependencies
        edges = self._build_edges(constraints, nodes)
        for edge_data in edges:
            edge = Edge(
                source=edge_data['source'],
                target=edge_data['target'],
                edge_type=EdgeType(edge_data['type']),
                weight=edge_data.get('weight', 1.0),
                metadata=edge_data.get('metadata', {})
            )
            fdg.add_edge(edge)

        # Step 4: Apply causal discovery (if historical data available)
        if self.use_causal_discovery and domain.historical_data is not None:
            self._apply_causal_discovery(fdg, domain.historical_data)

        # Step 5: Store metadata
        fdg.metadata = {
            'domain_id': domain.id,
            'extraction_method': 'FDGExtractor',
            'causal_discovery': self.use_causal_discovery
        }

        return fdg

    def _parse_constraints(self, domain: Domain) -> List[Dict]:
        """Parse constraints from domain representation"""
        constraints = []

        # Formal constraints
        for constraint in domain.formal_constraints:
            constraints.append({
                'type': 'formal',
                'constraint': constraint
            })

        # Natural language constraints
        for text in domain.natural_language_constraints:
            constraints.append({
                'type': 'natural_language',
                'text': text
            })

        return constraints

    def _extract_variables(self, constraints: List[Dict]) -> List[Dict]:
        """Extract variables from constraints"""
        variables = []

        for constraint in constraints:
            if constraint['type'] == 'formal':
                vars_in_constraint = self._extract_formal_variables(constraint['constraint'])
                variables.extend(vars_in_constraint)
            else:
                vars_in_constraint = self._extract_nlp_variables(constraint['text'])
                variables.extend(vars_in_constraint)

        # Deduplicate
        seen = set()
        unique_vars = []
        for var in variables:
            if var['name'] not in seen:
                seen.add(var['name'])
                unique_vars.append(var)

        return unique_vars

    def _extract_formal_variables(self, constraint: Any) -> List[Dict]:
        """
        Extract variables from formal constraint

        Implementation depends on constraint format
        """
        variables = []

        # If constraint is a string, parse variable names
        if isinstance(constraint, str):
            # Extract variable names (simplified regex)
            # Matches: x, y, var_name, etc.
            var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            matches = re.findall(var_pattern, constraint)

            # Filter out keywords
            keywords = {'if', 'then', 'and', 'or', 'not', 'True', 'False'}
            for match in matches:
                if match not in keywords:
                    variables.append({
                        'name': match,
                        'type': 'continuous',  # Default type
                        'metadata': {'source': 'formal'}
                    })

        # If constraint is a dict/object, extract from structure
        elif isinstance(constraint, dict):
            if 'variables' in constraint:
                for var in constraint['variables']:
                    variables.append({
                        'name': var.get('name', var),
                        'type': var.get('type', 'continuous'),
                        'metadata': var.get('metadata', {})
                    })

        return variables

    def _extract_nlp_variables(self, text: str) -> List[Dict]:
        """
        Extract variables from natural language

        Simplified implementation - can be enhanced with NLP
        """
        variables = []

        # Extract capitalized words as potential variables
        var_pattern = r'\b[A-Z][a-zA-Z0-9_]*\b'
        matches = re.findall(var_pattern, text)

        for match in matches:
            variables.append({
                'name': match,
                'type': 'continuous',
                'metadata': {'source': 'natural_language'}
            })

        return variables

    def _build_edges(self, constraints: List[Dict], nodes: List[Dict]) -> List[Dict]:
        """Build edges from constraints"""
        edges = []
        node_names = {n['name'] for n in nodes}

        for constraint in constraints:
            dependencies = self._analyze_dependencies(constraint, node_names)

            for dep in dependencies:
                edges.append({
                    'source': dep['source'],
                    'target': dep['target'],
                    'type': dep['type'],
                    'weight': dep.get('weight', 1.0)
                })

        return edges

    def _analyze_dependencies(
        self,
        constraint: Dict,
        node_names: Set[str]
    ) -> List[Dict]:
        """Analyze dependencies in constraint"""
        dependencies = []

        if constraint['type'] == 'formal':
            formal = constraint['constraint']

            if isinstance(formal, str):
                # Find dependencies: "x depends on y" or "f(x, y)"
                # Simplified: look for patterns like "x + y" or "f(x, y)"

                # Arithmetic operations
                ops_pattern = r'(\w+)\s*[+\-*/]\s*(\w+)'
                matches = re.findall(ops_pattern, formal)

                for m1, m2 in matches:
                    if m1 in node_names and m2 in node_names:
                        dependencies.append({
                            'source': m2,
                            'target': m1,
                            'type': 'causal',
                            'weight': 1.0
                        })

                # Function calls
                func_pattern = r'(\w+)\s*\(\s*([^)]+)\)'
                func_matches = re.findall(func_pattern, formal)

                for func_name, args in func_matches:
                    # Extract arguments
                    arg_names = [a.strip() for a in args.split(',')]

                    for arg in arg_names:
                        if arg in node_names:
                            dependencies.append({
                                'source': arg,
                                'target': func_name,
                                'type': 'causal',
                                'weight': 1.0
                            })

        elif constraint['type'] == 'natural_language':
            text = constraint['text']

            # Look for dependency indicators
            # "X depends on Y", "X influenced by Y", etc.
            dependency_patterns = [
                r'(\w+)\s+(?:depends on|influenced by|caused by)\s+(\w+)',
                r'(\w+)\s+(?:affects|impacts|influences)\s+(\w+)'
            ]

            for pattern in dependency_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)

                for target, source in matches:
                    if source in node_names and target in node_names:
                        dependencies.append({
                            'source': source,
                            'target': target,
                            'type': 'causal',
                            'weight': 1.0
                        })

        return dependencies

    def _apply_causal_discovery(
        self,
        fdg: FunctionalDependencyGraph,
        data: Any
    ) -> None:
        """
        Apply causal discovery algorithm to historical data

        Uses correlation-based analysis for causal inference.
        For advanced causal discovery, integrate causal-learn PC algorithm.
        """
        # Note: For production use with causal discovery, consider:
        # - pip install causal-learn
        # - from causallearn.search.ConstraintBased.PC import pc
        # - pc_algorithm = pc(data, alpha=0.05)

        try:
            import numpy as np

            # If data is a pandas DataFrame
            if hasattr(data, 'corr'):
                corr_matrix = data.corr()

                # Add edges for strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]

                        corr = corr_matrix.iloc[i, j]

                        if abs(corr) > 0.7:  # Strong correlation threshold
                            # Determine direction (heuristic)
                            # In practice, use proper causal discovery
                            if col1 in fdg.nodes and col2 in fdg.nodes:
                                # Add bidirectional edge for correlation
                                edge1 = Edge(
                                    source=col1,
                                    target=col2,
                                    edge_type=EdgeType.CORRELATION,
                                    weight=abs(corr)
                                )
                                fdg.add_edge(edge1)

        except ImportError:
            print("Warning: causal discovery requires pandas/numpy")
        except Exception as e:
            print(f"Warning: causal discovery failed: {e}")
