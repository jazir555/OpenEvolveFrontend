"""
Causal-Learn Integration Module for OpenEvolve Knowledge Engine

This module provides causal discovery capabilities by integrating
causal-learn's state-of-the-art algorithms including PC, FCI, GES, LiNGAM, etc.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

# Add causal-learn to Python path for import
causal_learn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'causal-learn')
if causal_learn_path not in sys.path:
    sys.path.insert(0, causal_learn_path)


class CausalLearnIntegration:
    """
    Main Causal-Learn Integration class for the Knowledge Engine.
    
    Provides causal discovery and structure learning capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Causal-Learn Integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._engine = CausalDiscoveryEngine()
    
    def is_available(self) -> bool:
        """Check if Causal-Learn is available."""
        return self._engine.is_available()
    
    def discover_structure(self, data: np.ndarray, algorithm: str = 'pc') -> Dict[str, Any]:
        """
        Discover causal structure from data.
        
        Args:
            data: Data matrix (n_samples x n_features)
            algorithm: Algorithm to use ('pc', 'fci', 'ges', 'lingam')
            
        Returns:
            Dictionary with discovered structure
        """
        return self._engine.discover_causal_structure(data, algorithm)
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        return self._engine.get_available_algorithms()


class CausalDiscoveryEngine:
    """
    Causal discovery engine using causal-learn algorithms.
    
    This class integrates various causal discovery methods including:
    - Constraint-based: PC, FCI
    - Score-based: GES
    - Functional causal models: LiNGAM, ANM
    """
    
    ALGORITHMS = {
        'pc': {
            'description': 'Peter-Clark algorithm for causal discovery',
            'requires_independence_test': True,
            'handles_latent': False
        },
        'fci': {
            'description': 'Fast Causal Inference for latent variables',
            'requires_independence_test': True,
            'handles_latent': True
        },
        'ges': {
            'description': 'Greedy Equivalence Search',
            'requires_score': True,
            'handles_latent': False
        },
        'lingam': {
            'description': 'Linear Non-Gaussian Acyclic Model',
            'requires_independence_test': False,
            'handles_latent': False
        },
        'direct_lingam': {
            'description': 'Direct LiNGAM algorithm',
            'requires_independence_test': False,
            'handles_latent': False
        },
        'ica_lingam': {
            'description': 'ICA-based LiNGAM',
            'requires_independence_test': False,
            'handles_latent': False
        },
        'granger': {
            'description': 'Granger causality for time series',
            'requires_independence_test': False,
            'handles_latent': False,
            'time_series': True
        }
    }
    
    INDEPENDENCE_TESTS = {
        'fisherz': 'Fisher\'s Z conditional independence test',
        'chisq': 'Chi-squared conditional independence test',
        'gsq': 'G-squared conditional independence test',
        'kci': 'Kernel-based conditional independence test'
    }
    
    def __init__(self):
        """Initialize causal-learn modules."""
        self._causal_learn_available = False
        self._algorithms_available = {}
        self._initialize_causal_learn()
    
    def _initialize_causal_learn(self):
        """Initialize causal-learn with proper error handling."""
        try:
            # Try to import causal-learn modules
            try:
                from causallearn.search.ConstraintBased import PC, FCI
                self._algorithms_available.update({
                    'pc': True,
                    'fci': True
                })
            except ImportError as e:
                print(f"Note: Constraint-based algorithms not available: {e}")
            
            try:
                from causallearn.search.ScoreBased import GES
                self._algorithms_available['ges'] = True
            except ImportError as e:
                print(f"Note: Score-based algorithms not available: {e}")
            
            try:
                from causallearn.search.FCMBased.lingam import ICA_LiNGAM, DirectLiNGAM
                self._algorithms_available.update({
                    'ica_lingam': True,
                    'direct_lingam': True,
                    'lingam': True
                })
            except ImportError as e:
                print(f"Note: LiNGAM algorithms not available: {e}")
            
            try:
                from causallearn.search.Granger import Granger
                self._algorithms_available['granger'] = True
            except ImportError as e:
                print(f"Note: Granger causality not available: {e}")
            
            # Check if any algorithms are available
            if self._algorithms_available:
                self._causal_learn_available = True
                print(f"Causal-learn initialized with algorithms: {list(self._algorithms_available.keys())}")
            else:
                print("Warning: No causal-learn algorithms could be loaded")
                
        except ImportError as e:
            print(f"Warning: Could not import causal-learn modules: {e}")
            print("Causal-learn integration will be disabled.")
    
    def is_available(self) -> bool:
        """Check if causal-learn integration is available."""
        return self._causal_learn_available
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available causal discovery algorithms."""
        return list(self._algorithms_available.keys())
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        algo_key = algorithm.lower()
        if algo_key in self.ALGORITHMS:
            return {
                'name': algorithm,
                'available': algo_key in self.get_available_algorithms(),
                **self.ALGORITHMS[algo_key]
            }
        return {'name': algorithm, 'available': False, 'error': 'Unknown algorithm'}
    
    def discover_causal_structure(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        algorithm: str = 'pc',
        alpha: float = 0.05,
        independence_test: str = 'fisherz',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover causal structure from data.
        
        Args:
            data: Data matrix (n_samples x n_variables)
            variable_names: Names of variables (optional)
            algorithm: Algorithm to use ('pc', 'fci', 'ges', 'lingam', etc.)
            alpha: Significance level for independence tests
            independence_test: Independence test to use ('fisherz', 'chisq', 'gsq', 'kci')
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Dictionary containing causal graph and metadata
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'Causal-learn integration not available',
                'graph': None
            }
        
        try:
            algorithm = algorithm.lower()
            
            if algorithm not in self.get_available_algorithms():
                return {
                    'status': 'error',
                    'message': f'Algorithm {algorithm} not available. Available: {self.get_available_algorithms()}',
                    'graph': None
                }
            
            # Set default variable names if not provided
            if variable_names is None:
                variable_names = [f'X{i}' for i in range(data.shape[1])]
            
            # Run the appropriate algorithm
            if algorithm == 'pc':
                result = self._run_pc(data, variable_names, alpha, independence_test, **kwargs)
            elif algorithm == 'fci':
                result = self._run_fci(data, variable_names, alpha, independence_test, **kwargs)
            elif algorithm == 'ges':
                result = self._run_ges(data, variable_names, **kwargs)
            elif algorithm in ['lingam', 'ica_lingam']:
                result = self._run_ica_lingam(data, variable_names, **kwargs)
            elif algorithm == 'direct_lingam':
                result = self._run_direct_lingam(data, variable_names, **kwargs)
            elif algorithm == 'granger':
                result = self._run_granger(data, variable_names, **kwargs)
            else:
                return {
                    'status': 'error',
                    'message': f'Algorithm {algorithm} not yet implemented',
                    'graph': None
                }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Causal discovery failed: {str(e)}',
                'graph': None
            }
    
    def _run_pc(
        self,
        data: np.ndarray,
        variable_names: List[str],
        alpha: float,
        independence_test: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run PC algorithm."""
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz, chisq, gsq, kci
            
            # Map independence test name to function
            test_map = {
                'fisherz': fisherz,
                'chisq': chisq,
                'gsq': gsq,
                'kci': kci
            }
            
            indep_test = test_map.get(independence_test, fisherz)
            
            # Run PC algorithm
            cg = pc(
                data=data,
                alpha=alpha,
                indep_test=indep_test,
                node_names=variable_names,
                **kwargs
            )
            
            # Convert to adjacency matrix format
            graph_matrix = cg.G.graph if hasattr(cg, 'G') else None
            
            # Extract edges
            edges = self._extract_edges_from_matrix(graph_matrix, variable_names)
            
            return {
                'status': 'success',
                'graph': {
                    'nodes': variable_names,
                    'edges': edges,
                    'adjacency_matrix': graph_matrix.tolist() if graph_matrix is not None else None
                },
                'algorithm': 'pc',
                'parameters': {
                    'alpha': alpha,
                    'independence_test': independence_test
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'PC algorithm failed: {str(e)}',
                'graph': None
            }
    
    def _run_fci(
        self,
        data: np.ndarray,
        variable_names: List[str],
        alpha: float,
        independence_test: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run FCI algorithm."""
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz
            
            # Run FCI algorithm
            g, edges = fci(
                data=data,
                alpha=alpha,
                indep_test=fisherz,
                **kwargs
            )
            
            # Extract PAG (Partial Ancestral Graph)
            graph_matrix = g.graph if hasattr(g, 'graph') else None
            
            return {
                'status': 'success',
                'graph': {
                    'nodes': variable_names,
                    'edges': edges if edges else [],
                    'pag_matrix': graph_matrix.tolist() if graph_matrix is not None else None,
                    'type': 'PAG'
                },
                'algorithm': 'fci',
                'parameters': {
                    'alpha': alpha,
                    'independence_test': independence_test
                },
                'note': 'FCI handles latent confounders'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'FCI algorithm failed: {str(e)}',
                'graph': None
            }
    
    def _run_ges(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Run GES algorithm."""
        try:
            from causallearn.search.ScoreBased.GES import ges
            
            # Run GES algorithm
            record = ges(data, **kwargs)
            
            # Extract graph from record
            graph_matrix = record['G'].graph if 'G' in record else None
            edges = self._extract_edges_from_matrix(graph_matrix, variable_names)
            
            return {
                'status': 'success',
                'graph': {
                    'nodes': variable_names,
                    'edges': edges,
                    'adjacency_matrix': graph_matrix.tolist() if graph_matrix is not None else None
                },
                'algorithm': 'ges',
                'score': record.get('score'),
                'parameters': kwargs
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'GES algorithm failed: {str(e)}',
                'graph': None
            }
    
    def _run_ica_lingam(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Run ICA-LiNGAM algorithm."""
        try:
            from causallearn.search.FCMBased.lingam import ICA_LiNGAM
            
            model = ICA_LiNGAM(**kwargs)
            model.fit(data)
            
            # Extract adjacency matrix
            adjacency_matrix = model.adjacency_matrix_
            
            # Convert to edges
            edges = []
            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if adjacency_matrix[i, j] != 0:
                        edges.append({
                            'source': variable_names[i],
                            'target': variable_names[j],
                            'weight': float(adjacency_matrix[i, j]),
                            'type': 'directed'
                        })
            
            return {
                'status': 'success',
                'graph': {
                    'nodes': variable_names,
                    'edges': edges,
                    'adjacency_matrix': adjacency_matrix.tolist()
                },
                'algorithm': 'ica_lingam',
                'parameters': kwargs,
                'causal_order': [variable_names[i] for i in model.causal_order_]
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'ICA-LiNGAM algorithm failed: {str(e)}',
                'graph': None
            }
    
    def _run_direct_lingam(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Run DirectLiNGAM algorithm."""
        try:
            from causallearn.search.FCMBased.lingam import DirectLiNGAM
            
            model = DirectLiNGAM(**kwargs)
            model.fit(data)
            
            # Extract adjacency matrix
            adjacency_matrix = model.adjacency_matrix_
            
            # Convert to edges
            edges = []
            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if adjacency_matrix[i, j] != 0:
                        edges.append({
                            'source': variable_names[i],
                            'target': variable_names[j],
                            'weight': float(adjacency_matrix[i, j]),
                            'type': 'directed'
                        })
            
            return {
                'status': 'success',
                'graph': {
                    'nodes': variable_names,
                    'edges': edges,
                    'adjacency_matrix': adjacency_matrix.tolist()
                },
                'algorithm': 'direct_lingam',
                'parameters': kwargs,
                'causal_order': [variable_names[i] for i in model.causal_order_]
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'DirectLiNGAM algorithm failed: {str(e)}',
                'graph': None
            }
    
    def _run_granger(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Run Granger causality test."""
        try:
            from causallearn.search.Granger.Granger import granger_lasso
            
            # Run Granger causality
            coeff = granger_lasso(data, **kwargs)
            
            # Convert to edges
            edges = []
            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if i != j and coeff[i, j] != 0:
                        edges.append({
                            'source': variable_names[i],
                            'target': variable_names[j],
                            'weight': float(coeff[i, j]),
                            'type': 'granger_causal'
                        })
            
            return {
                'status': 'success',
                'graph': {
                    'nodes': variable_names,
                    'edges': edges,
                    'coefficient_matrix': coeff.tolist()
                },
                'algorithm': 'granger',
                'parameters': kwargs,
                'note': 'Time series causality'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Granger causality failed: {str(e)}',
                'graph': None
            }
    
    def _extract_edges_from_matrix(
        self,
        matrix: np.ndarray,
        variable_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract edges from adjacency matrix."""
        edges = []
        
        if matrix is None:
            return edges
        
        n = len(variable_names)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] != 0:
                    # Determine edge type based on matrix values
                    if matrix[i, j] == 1 and matrix[j, i] == -1:
                        edge_type = 'directed'
                        direction = f'{variable_names[i]} -> {variable_names[j]}'
                    elif matrix[i, j] == matrix[j, i] == 1:
                        edge_type = 'bidirected'
                        direction = f'{variable_names[i]} <-> {variable_names[j]}'
                    elif matrix[i, j] == matrix[j, i] == -1:
                        edge_type = 'undirected'
                        direction = f'{variable_names[i]} -- {variable_names[j]}'
                    else:
                        edge_type = 'unknown'
                        direction = f'{variable_names[i]} ? {variable_names[j]}'
                    
                    edges.append({
                        'source': variable_names[i],
                        'target': variable_names[j],
                        'type': edge_type,
                        'matrix_value': float(matrix[i, j]),
                        'description': direction
                    })
        
        return edges
    
    def analyze_causal_graph(
        self,
        graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a discovered causal graph.
        
        Args:
            graph_data: Graph data with nodes and edges
            
        Returns:
            Dictionary containing graph analysis
        """
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            # Build adjacency list
            adjacency = {node: [] for node in nodes}
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source in adjacency:
                    adjacency[source].append(target)
            
            # Calculate statistics
            in_degrees = {node: 0 for node in nodes}
            out_degrees = {node: 0 for node in nodes}
            
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source in out_degrees:
                    out_degrees[source] += 1
                if target in in_degrees:
                    in_degrees[target] += 1
            
            # Find root nodes (no parents)
            roots = [node for node in nodes if in_degrees[node] == 0]
            
            # Find leaf nodes (no children)
            leaves = [node for node in nodes if out_degrees[node] == 0]
            
            # Calculate graph metrics
            analysis = {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
                'roots': roots,
                'leaves': leaves,
                'in_degrees': in_degrees,
                'out_degrees': out_degrees,
                'avg_in_degree': sum(in_degrees.values()) / len(nodes) if nodes else 0,
                'avg_out_degree': sum(out_degrees.values()) / len(nodes) if nodes else 0,
                'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
                'max_out_degree': max(out_degrees.values()) if out_degrees else 0
            }
            
            return {
                'status': 'success',
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Graph analysis failed: {str(e)}',
                'analysis': {}
            }
    
    def identify_confounders(
        self,
        graph_data: Dict[str, Any],
        target_x: str,
        target_y: str
    ) -> Dict[str, Any]:
        """
        Identify potential confounders between two variables.
        
        Args:
            graph_data: Causal graph data
            target_x: First target variable
            target_y: Second target variable
            
        Returns:
            Dictionary containing confounders
        """
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            # Build parent relationships
            parents = {node: set() for node in nodes}
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if target in parents:
                    parents[target].add(source)
            
            # Find common causes (parents of both X and Y)
            parents_x = parents.get(target_x, set())
            parents_y = parents.get(target_y, set())
            common_causes = parents_x & parents_y
            
            # Find mediators (on causal path from X to Y or Y to X)
            mediators = self._find_mediators(edges, target_x, target_y)
            
            # Find colliders (common children)
            children = {node: set() for node in nodes}
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source in children:
                    children[source].add(target)
            
            children_x = children.get(target_x, set())
            children_y = children.get(target_y, set())
            colliders = children_x & children_y
            
            return {
                'status': 'success',
                'target_pair': (target_x, target_y),
                'confounders': {
                    'common_causes': list(common_causes),
                    'mediators': mediators,
                    'colliders': list(colliders),
                    'adjustment_set': list(common_causes)  # Variables to adjust for
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Confounder identification failed: {str(e)}',
                'confounders': {}
            }
    
    def _find_mediators(
        self,
        edges: List[Dict[str, Any]],
        source: str,
        target: str
    ) -> List[str]:
        """Find mediators on paths from source to target."""
        # Build adjacency list
        adjacency = {}
        for edge in edges:
            s = edge.get('source')
            t = edge.get('target')
            if s not in adjacency:
                adjacency[s] = []
            adjacency[s].append(t)
        
        # Find all paths
        mediators = set()
        visited = set()
        
        def dfs(current, path):
            if current == target and len(path) > 1:
                # Add all intermediate nodes as potential mediators
                for node in path[1:-1]:
                    mediators.add(node)
                return
            
            if current in visited:
                return
            
            visited.add(current)
            
            for neighbor in adjacency.get(current, []):
                dfs(neighbor, path + [neighbor])
            
            visited.remove(current)
        
        dfs(source, [source])
        
        return list(mediators)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of causal-learn integration."""
        return {
            'available': self.is_available(),
            'algorithms': self.get_available_algorithms(),
            'algorithm_info': self.ALGORITHMS,
            'independence_tests': self.INDEPENDENCE_TESTS,
            'timestamp': datetime.now().isoformat()
        }
