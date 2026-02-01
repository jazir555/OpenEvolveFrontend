"""
Causal Analysis Node for BubbleLabs Integration

Provides causal discovery and analysis capabilities using Causal-Learn:
- Discover causal relationships from data
- Build causal graphs
- Identify confounding variables
- Estimate causal effects
- Validate causal hypotheses

Integrates with the Knowledge Engine's Causal-Learn integration for
state-of-the-art causal discovery algorithms including PC, FCI, GES, and LiNGAM.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from .base_node import BubbleLabsNode, NodeExecutionError


class CausalAnalysisNode(BubbleLabsNode):
    """
    Discover causal relationships and build causal graphs from knowledge.

    Supports multiple causal discovery algorithms:
    - PC: Peter-Clark algorithm for causal discovery
    - FCI: Fast Causal Inference (handles latent variables)
    - GES: Greedy Equivalence Search
    - LiNGAM: Linear Non-Gaussian Acyclic Model

    Operations:
    - discover: Discover causal structure from data
    - build_graph: Build causal graph from relationships
    - identify_confounders: Find confounding variables
    - estimate_effect: Estimate causal effects
    - validate: Validate causal hypotheses
    """

    # Node metadata
    DISPLAY_NAME = "Causal Analysis"
    DESCRIPTION = "Discover causal relationships and build causal graphs from knowledge"
    ICON = "causal-analysis"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of CausalLearnIntegration
        causal_module = self.safe_import(
            'knowledge_engine.integrations.causal_learn_integration',
            fallback_value=None,
            error_msg="Causal-Learn integration not available"
        )

        self.CausalLearnIntegration = None
        self.CausalDiscoveryEngine = None

        if causal_module:
            self.CausalLearnIntegration = getattr(causal_module, 'CausalLearnIntegration', None)
            self.CausalDiscoveryEngine = getattr(causal_module, 'CausalDiscoveryEngine', None)

        # Safe import of UnifiedKGIntegrationHub
        hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available"
        )

        self.UnifiedKGIntegrationHub = None
        if hub_module:
            self.UnifiedKGIntegrationHub = getattr(hub_module, 'UnifiedKGIntegrationHub', None)

        # Initialize causal integration
        self.causal_integration = None
        if self.CausalLearnIntegration:
            try:
                self.causal_integration = self.CausalLearnIntegration()
                self.logger.info("Causal-Learn integration initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize Causal-Learn integration: {e}")
                self.causal_integration = None

        # Valid operations and algorithms
        self.valid_operations = ['discover', 'build_graph', 'identify_confounders', 'estimate_effect', 'validate']
        self.valid_algorithms = ['pc', 'fci', 'ges', 'lingam', 'ica_lingam', 'direct_lingam', 'granger']

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (depending on operation):
            - data: Dict with variable values or knowledge graph data
            - variables: List[str] - Variables to analyze

        Optional:
            - operation: str - Override the configured operation
            - target_variable: str - Target/outcome variable
            - treatment_variable: str - Intervention variable
            - algorithm: str - Causal discovery algorithm
        """
        errors = []

        # Get operation from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'discover'))

        # Validate operation
        if operation not in self.valid_operations:
            errors.append(f"Invalid operation: '{operation}'. Must be one of: {', '.join(self.valid_operations)}")

        # Check for data or knowledge graph
        if 'data' not in inputs and 'knowledge_graph_id' not in inputs:
            # Some operations may work with just variable definitions
            if operation in ['discover', 'estimate_effect', 'validate']:
                errors.append("Missing required input: 'data' or 'knowledge_graph_id'")

        # Validate data format if provided
        if 'data' in inputs:
            data = inputs['data']
            if not isinstance(data, dict):
                errors.append("'data' must be a dictionary with variable names as keys")

        # Validate variables
        if 'variables' in inputs:
            if not isinstance(inputs['variables'], list):
                errors.append("'variables' must be a list of strings")
            elif len(inputs['variables']) < 2:
                errors.append("At least 2 variables are required for causal analysis")

        # Validate target_variable for certain operations
        if operation in ['identify_confounders', 'estimate_effect']:
            target = inputs.get('target_variable') or inputs.get('target') or self.config.get('target_variable')
            if not target:
                errors.append(f"Operation '{operation}' requires 'target_variable' or 'target' in inputs")

        # Validate treatment_variable for estimate_effect
        if operation == 'estimate_effect':
            treatment = inputs.get('treatment_variable') or self.config.get('treatment_variable')
            if not treatment:
                errors.append("Operation 'estimate_effect' requires 'treatment_variable' in inputs")

        # Validate algorithm if provided
        if 'algorithm' in inputs:
            if inputs['algorithm'] not in self.valid_algorithms:
                errors.append(
                    f"Invalid algorithm: '{inputs['algorithm']}'. "
                    f"Must be one of: {', '.join(self.valid_algorithms)}"
                )

        # Validate significance_level
        if 'significance_level' in inputs:
            try:
                sl = float(inputs['significance_level'])
                if not 0.0 <= sl <= 1.0:
                    errors.append("'significance_level' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'significance_level' must be a number")

        # Validate max_path_length
        if 'max_path_length' in inputs:
            try:
                mpl = int(inputs['max_path_length'])
                if mpl < 1:
                    errors.append("'max_path_length' must be at least 1")
            except (TypeError, ValueError):
                errors.append("'max_path_length' must be an integer")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute causal analysis operation.

        Args:
            inputs: Must contain data/variables based on operation type
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - causal_graph: Graph structure with nodes and edges
                - relationships: List of causal relationships
                - effects: List of estimated causal effects
                - confounders: List of identified confounding variables
                - validation_results: Results of hypothesis validation

        Raises:
            NodeExecutionError: If analysis fails
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'discover'))
        variables = inputs.get('variables', self.config.get('variables', []))
        target_variable = inputs.get('target_variable') or inputs.get('target') or self.config.get('target_variable')
        treatment_variable = inputs.get('treatment_variable') or self.config.get('treatment_variable')
        algorithm = inputs.get('algorithm', self.config.get('algorithm', 'pc'))
        significance_level = inputs.get('significance_level', self.config.get('significance_level', 0.05))
        max_path_length = inputs.get('max_path_length', self.config.get('max_path_length', 5))
        knowledge_graph_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Starting causal {operation} with algorithm: {algorithm}")

        try:
            # Load data from knowledge graph if specified
            data = inputs.get('data', {})
            if knowledge_graph_id and not data:
                context.update_progress(20, f"Loading data from knowledge graph: {knowledge_graph_id}")
                data = self._load_from_knowledge_graph(knowledge_graph_id, variables)

            # Execute based on operation type
            if operation == 'discover':
                result = self._discover_causal_structure(
                    data, variables, algorithm, significance_level, context
                )
            elif operation == 'build_graph':
                result = self._build_causal_graph(
                    data, variables, algorithm, context
                )
            elif operation == 'identify_confounders':
                result = self._identify_confounders(
                    data, variables, target_variable, treatment_variable, context
                )
            elif operation == 'estimate_effect':
                result = self._estimate_causal_effect(
                    data, variables, target_variable, treatment_variable, algorithm, context
                )
            elif operation == 'validate':
                result = self._validate_hypothesis(
                    data, variables, target_variable, treatment_variable, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'operation': operation}
                )

            # Add operation metadata
            result['operation'] = operation
            result['algorithm'] = algorithm
            result['variables_analyzed'] = variables
            result['timestamp'] = datetime.now().isoformat()

            # Store artifact in context
            context.add_artifact('causal_analysis', {
                'operation': operation,
                'algorithm': algorithm,
                'variables': variables,
                'result_summary': self._summarize_result(result)
            })

            context.update_progress(100, f"Causal {operation} complete")
            self.logger.info(f"Causal analysis completed: operation={operation}, algorithm={algorithm}")

            return result

        except Exception as e:
            self.logger.error(f"Causal analysis failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Causal analysis failed: {str(e)}",
                details={
                    'operation': operation,
                    'algorithm': algorithm,
                    'variables': variables,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _discover_causal_structure(
        self,
        data: Dict[str, Any],
        variables: List[str],
        algorithm: str,
        significance_level: float,
        context
    ) -> Dict[str, Any]:
        """Discover causal structure from data using specified algorithm."""
        context.update_progress(30, f"Preparing data for {algorithm} algorithm")

        # Convert data to numpy array
        data_array, var_names = self._prepare_data_array(data, variables)

        if data_array is None or len(var_names) < 2:
            return self._fallback_discover(data, variables, context)

        context.update_progress(50, f"Running {algorithm} causal discovery")

        if self.causal_integration and self.causal_integration.is_available():
            try:
                result = self.causal_integration.discover_structure(
                    data=data_array,
                    variable_names=var_names,
                    algorithm=algorithm,
                    alpha=significance_level
                )

                # Extract relationships from graph
                relationships = self._extract_relationships(result.get('graph', {}))

                return {
                    'causal_graph': result.get('graph', {}),
                    'relationships': relationships,
                    'effects': [],
                    'confounders': [],
                    'status': result.get('status', 'success'),
                    'message': result.get('message', 'Causal discovery completed'),
                    'parameters': result.get('parameters', {}),
                    'using_fallback': False
                }
            except Exception as e:
                self.logger.warning(f"Causal-Learn discovery failed: {e}, using fallback")
                return self._fallback_discover(data, variables, context)
        else:
            return self._fallback_discover(data, variables, context)

    def _build_causal_graph(
        self,
        data: Dict[str, Any],
        variables: List[str],
        algorithm: str,
        context
    ) -> Dict[str, Any]:
        """Build causal graph from relationships."""
        context.update_progress(30, "Building causal graph structure")

        # First discover the structure
        discovery_result = self._discover_causal_structure(
            data, variables, algorithm, 0.05, context
        )

        context.update_progress(70, "Analyzing graph structure")

        # Analyze the graph
        graph = discovery_result.get('causal_graph', {})
        analysis = self._analyze_graph_structure(graph)

        return {
            'causal_graph': graph,
            'relationships': discovery_result.get('relationships', []),
            'effects': [],
            'confounders': [],
            'graph_analysis': analysis,
            'status': 'success',
            'message': 'Causal graph built successfully'
        }

    def _identify_confounders(
        self,
        data: Dict[str, Any],
        variables: List[str],
        target_variable: str,
        treatment_variable: Optional[str],
        context
    ) -> Dict[str, Any]:
        """Identify confounding variables between treatment and target."""
        context.update_progress(30, "Discovering causal structure for confounder analysis")

        # Discover causal structure first
        discovery_result = self._discover_causal_structure(
            data, variables, 'pc', 0.05, context
        )

        graph = discovery_result.get('causal_graph', {})

        context.update_progress(60, "Identifying confounders")

        # Use CausalLearnIntegration's confounder identification if available
        if (self.causal_integration and 
            hasattr(self.causal_integration, '_engine') and 
            treatment_variable):
            try:
                engine = self.causal_integration._engine
                confounder_result = engine.identify_confounders(
                    graph_data=graph,
                    target_x=treatment_variable,
                    target_y=target_variable
                )

                confounders = confounder_result.get('confounders', {})

                return {
                    'causal_graph': graph,
                    'relationships': discovery_result.get('relationships', []),
                    'effects': [],
                    'confounders': confounders,
                    'confounder_summary': {
                        'common_causes': confounders.get('common_causes', []),
                        'mediators': confounders.get('mediators', []),
                        'colliders': confounders.get('colliders', []),
                        'adjustment_set': confounders.get('adjustment_set', [])
                    },
                    'status': 'success',
                    'message': f"Identified {len(confounders.get('common_causes', []))} confounders"
                }
            except Exception as e:
                self.logger.warning(f"Confounder identification failed: {e}, using fallback")

        # Fallback confounder identification
        confounders = self._fallback_identify_confounders(
            graph, treatment_variable, target_variable
        )

        return {
            'causal_graph': graph,
            'relationships': discovery_result.get('relationships', []),
            'effects': [],
            'confounders': confounders,
            'status': 'success',
            'message': 'Confounder identification completed (fallback)',
            'using_fallback': True
        }

    def _estimate_causal_effect(
        self,
        data: Dict[str, Any],
        variables: List[str],
        target_variable: str,
        treatment_variable: str,
        algorithm: str,
        context
    ) -> Dict[str, Any]:
        """Estimate causal effect of treatment on target."""
        context.update_progress(30, "Preparing data for causal effect estimation")

        data_array, var_names = self._prepare_data_array(data, variables)

        if data_array is None or treatment_variable not in var_names or target_variable not in var_names:
            return self._fallback_estimate_effect(data, target_variable, treatment_variable, context)

        context.update_progress(50, "Estimating causal effect")

        # Try to use LiNGAM for effect estimation if available
        if self.causal_integration and self.causal_integration.is_available():
            try:
                # Use LiNGAM to get causal effects
                result = self.causal_integration.discover_structure(
                    data=data_array,
                    variable_names=var_names,
                    algorithm='lingam'
                )

                graph = result.get('graph', {})
                adjacency = graph.get('adjacency_matrix', [])

                # Extract effect from adjacency matrix
                effects = []
                if adjacency:
                    treatment_idx = var_names.index(treatment_variable)
                    target_idx = var_names.index(target_variable)
                    effect_value = adjacency[target_idx][treatment_idx]

                    effects.append({
                        'treatment': treatment_variable,
                        'target': target_variable,
                        'effect': effect_value,
                        'type': 'direct',
                        'confidence': 0.8
                    })

                return {
                    'causal_graph': graph,
                    'relationships': self._extract_relationships(graph),
                    'effects': effects,
                    'confounders': [],
                    'status': 'success',
                    'message': f"Estimated effect: {effect_value:.4f}" if effects else "No effect found"
                }
            except Exception as e:
                self.logger.warning(f"Causal effect estimation failed: {e}, using fallback")

        return self._fallback_estimate_effect(data, target_variable, treatment_variable, context)

    def _validate_hypothesis(
        self,
        data: Dict[str, Any],
        variables: List[str],
        target_variable: str,
        treatment_variable: Optional[str],
        context
    ) -> Dict[str, Any]:
        """Validate a causal hypothesis."""
        context.update_progress(30, "Discovering causal structure for validation")

        discovery_result = self._discover_causal_structure(
            data, variables, 'pc', 0.05, context
        )

        graph = discovery_result.get('causal_graph', {})
        relationships = discovery_result.get('relationships', [])

        context.update_progress(60, "Validating causal hypothesis")

        validation_results = []

        if treatment_variable and target_variable:
            # Check if there's a path from treatment to target
            path_exists = self._check_causal_path(
                graph, treatment_variable, target_variable
            )

            validation_results.append({
                'hypothesis': f"{treatment_variable} causes {target_variable}",
                'valid': path_exists,
                'evidence': 'direct_path' if path_exists else 'no_path_found',
                'confidence': 0.85 if path_exists else 0.5
            })

        # Validate graph structure
        validation_results.append({
            'hypothesis': "Causal graph is acyclic",
            'valid': self._is_acyclic(graph),
            'evidence': 'graph_analysis',
            'confidence': 0.9
        })

        return {
            'causal_graph': graph,
            'relationships': relationships,
            'effects': [],
            'confounders': [],
            'validation_results': validation_results,
            'status': 'success',
            'message': f"Validated {len(validation_results)} hypotheses"
        }

    def _prepare_data_array(
        self,
        data: Dict[str, Any],
        variables: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Convert data dictionary to numpy array."""
        if not data:
            return None, []

        # Extract variable names from data if not provided
        if not variables:
            variables = list(data.keys())

        # Filter to only include variables that exist in data
        available_vars = [v for v in variables if v in data]

        if not available_vars:
            return None, []

        # Get data lengths
        data_lengths = [len(data[v]) for v in available_vars if isinstance(data[v], (list, np.ndarray))]

        if not data_lengths:
            return None, []

        min_length = min(data_lengths)

        # Build data array
        data_matrix = []
        for var in available_vars:
            values = data[var]
            if isinstance(values, (list, np.ndarray)):
                data_matrix.append(values[:min_length])
            else:
                # Constant value - expand to array
                data_matrix.append([values] * min_length)

        try:
            data_array = np.array(data_matrix).T
            return data_array, available_vars
        except Exception as e:
            self.logger.warning(f"Could not create data array: {e}")
            return None, []

    def _load_from_knowledge_graph(self, kg_id: str, variables: List[str]) -> Dict[str, Any]:
        """Load data from knowledge graph."""
        self.logger.info(f"Loading data from knowledge graph: {kg_id}")

        # Placeholder for KG integration
        # In a real implementation, this would query the knowledge graph
        # and extract variable data for causal analysis

        return {}

    def _extract_relationships(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract causal relationships from graph."""
        relationships = []
        edges = graph.get('edges', [])

        for edge in edges:
            relationship = {
                'source': edge.get('source'),
                'target': edge.get('target'),
                'type': edge.get('type', 'unknown'),
                'weight': edge.get('weight', 1.0),
                'confidence': 0.8  # Default confidence
            }
            relationships.append(relationship)

        return relationships

    def _analyze_graph_structure(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causal graph structure."""
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        # Build adjacency list
        adjacency = {node: [] for node in nodes}
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in adjacency:
                adjacency[source].append(target)

        # Calculate degrees
        in_degrees = {node: 0 for node in nodes}
        out_degrees = {node: 0 for node in nodes}

        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in out_degrees:
                out_degrees[source] += 1
            if target in in_degrees:
                in_degrees[target] += 1

        # Find root and leaf nodes
        roots = [node for node in nodes if in_degrees[node] == 0]
        leaves = [node for node in nodes if out_degrees[node] == 0]

        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
            'roots': roots,
            'leaves': leaves,
            'in_degrees': in_degrees,
            'out_degrees': out_degrees,
            'avg_in_degree': sum(in_degrees.values()) / len(nodes) if nodes else 0,
            'avg_out_degree': sum(out_degrees.values()) / len(nodes) if nodes else 0
        }

    def _check_causal_path(self, graph: Dict[str, Any], source: str, target: str) -> bool:
        """Check if there's a causal path from source to target."""
        edges = graph.get('edges', [])

        # Build adjacency list
        adjacency = {}
        for edge in edges:
            s = edge.get('source')
            t = edge.get('target')
            if s not in adjacency:
                adjacency[s] = []
            adjacency[s].append(t)

        # BFS to find path
        visited = set()
        queue = [source]

        while queue:
            current = queue.pop(0)
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adjacency.get(current, []))

        return False

    def _is_acyclic(self, graph: Dict[str, Any]) -> bool:
        """Check if graph is acyclic."""
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        # Build adjacency list
        adjacency = {node: [] for node in nodes}
        for edge in edges:
            s = edge.get('source')
            t = edge.get('target')
            if s in adjacency:
                adjacency[s].append(t)

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in nodes:
            if node not in visited:
                if has_cycle(node):
                    return False

        return True

    def _fallback_discover(
        self,
        data: Dict[str, Any],
        variables: List[str],
        context
    ) -> Dict[str, Any]:
        """Fallback causal discovery using correlation analysis."""
        context.update_progress(40, "Using fallback correlation analysis (Causal-Learn not available)")

        if not data:
            return {
                'causal_graph': {'nodes': variables, 'edges': []},
                'relationships': [],
                'effects': [],
                'confounders': [],
                'status': 'warning',
                'message': 'No data available for analysis',
                'using_fallback': True
            }

        # Extract available variables with data
        available_vars = [v for v in variables if v in data and isinstance(data[v], (list, np.ndarray))]

        if len(available_vars) < 2:
            return {
                'causal_graph': {'nodes': available_vars, 'edges': []},
                'relationships': [],
                'effects': [],
                'confounders': [],
                'status': 'warning',
                'message': 'Insufficient data for correlation analysis',
                'using_fallback': True
            }

        # Calculate correlations
        edges = []
        relationships = []

        try:
            for i, var1 in enumerate(available_vars):
                for var2 in available_vars[i+1:]:
                    # Get data arrays
                    arr1 = np.array(data[var1])
                    arr2 = np.array(data[var2])

                    # Ensure same length
                    min_len = min(len(arr1), len(arr2))
                    if min_len < 2:
                        continue

                    arr1 = arr1[:min_len]
                    arr2 = arr2[:min_len]

                    # Calculate correlation
                    corr = np.corrcoef(arr1, arr2)[0, 1]

                    if abs(corr) > 0.5:  # Threshold for relationship
                        edge = {
                            'source': var1,
                            'target': var2,
                            'type': 'correlated',
                            'weight': float(abs(corr)),
                            'correlation': float(corr)
                        }
                        edges.append(edge)

                        relationships.append({
                            'source': var1,
                            'target': var2,
                            'type': 'correlated',
                            'strength': float(abs(corr)),
                            'note': 'Correlation-based (causal direction not determined)'
                        })
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")

        graph = {
            'nodes': available_vars,
            'edges': edges,
            'note': 'Fallback correlation-based graph (not causal)'
        }

        context.update_progress(90, f"Fallback analysis complete: {len(edges)} correlations found")

        return {
            'causal_graph': graph,
            'relationships': relationships,
            'effects': [],
            'confounders': [],
            'status': 'success',
            'message': f'Found {len(edges)} correlations (fallback mode)',
            'using_fallback': True,
            'warning': 'Causal-Learn not available. Using correlation analysis instead.'
        }

    def _fallback_identify_confounders(
        self,
        graph: Dict[str, Any],
        treatment: Optional[str],
        target: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback confounder identification."""
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        # Build adjacency
        parents = {node: set() for node in nodes}
        for edge in edges:
            source = edge.get('source')
            target_node = edge.get('target')
            if target_node in parents:
                parents[target_node].add(source)

        # Find common causes
        common_causes = []
        if treatment and target:
            parents_treatment = parents.get(treatment, set())
            parents_target = parents.get(target, set())
            common_causes = list(parents_treatment & parents_target)

        return {
            'common_causes': common_causes,
            'mediators': [],
            'colliders': [],
            'adjustment_set': common_causes,
            'note': 'Fallback identification (may not be complete)'
        }

    def _fallback_estimate_effect(
        self,
        data: Dict[str, Any],
        target: str,
        treatment: str,
        context
    ) -> Dict[str, Any]:
        """Fallback causal effect estimation using simple difference."""
        context.update_progress(60, "Using fallback effect estimation")

        effects = []

        try:
            if treatment in data and target in data:
                treatment_vals = np.array(data[treatment])
                target_vals = np.array(data[target])

                # Simple correlation as effect estimate
                min_len = min(len(treatment_vals), len(target_vals))
                if min_len > 1:
                    corr = np.corrcoef(treatment_vals[:min_len], target_vals[:min_len])[0, 1]
                    effects.append({
                        'treatment': treatment,
                        'target': target,
                        'effect': float(corr),
                        'type': 'correlation_based',
                        'confidence': 0.5,
                        'note': 'Fallback estimation using correlation'
                    })
        except Exception as e:
            self.logger.warning(f"Fallback effect estimation failed: {e}")

        return {
            'causal_graph': {},
            'relationships': [],
            'effects': effects,
            'confounders': [],
            'status': 'warning',
            'message': 'Effect estimation completed (fallback mode)',
            'using_fallback': True,
            'warning': 'Causal-Learn not available. Using simple correlation-based estimation.'
        }

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the result for artifact storage."""
        return {
            'graph_nodes': len(result.get('causal_graph', {}).get('nodes', [])),
            'graph_edges': len(result.get('causal_graph', {}).get('edges', [])),
            'relationships_count': len(result.get('relationships', [])),
            'effects_count': len(result.get('effects', [])),
            'confounders_count': len(result.get('confounders', {}).get('common_causes', [])),
            'status': result.get('status', 'unknown'),
            'using_fallback': result.get('using_fallback', False)
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Causal Analysis Configuration",
            "description": "Configure causal discovery and analysis parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of causal analysis to perform",
                    "enum": ["discover", "build_graph", "identify_confounders", "estimate_effect", "validate"],
                    "enumNames": [
                        "Discover - Discover causal structure from data",
                        "Build Graph - Build causal graph from relationships",
                        "Identify Confounders - Find confounding variables",
                        "Estimate Effect - Estimate causal effects",
                        "Validate - Validate causal hypotheses"
                    ],
                    "default": "discover"
                },
                "variables": {
                    "type": "array",
                    "title": "Variables",
                    "description": "Variables to analyze for causal relationships",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "target_variable": {
                    "type": "string",
                    "title": "Target Variable",
                    "description": "Outcome variable (required for identify_confounders, estimate_effect, validate)",
                    "default": ""
                },
                "treatment_variable": {
                    "type": "string",
                    "title": "Treatment Variable",
                    "description": "Intervention variable (required for identify_confounders, estimate_effect)",
                    "default": ""
                },
                "algorithm": {
                    "type": "string",
                    "title": "Algorithm",
                    "description": "Causal discovery algorithm to use",
                    "enum": ["pc", "fci", "ges", "lingam"],
                    "enumNames": [
                        "PC - Peter-Clark algorithm",
                        "FCI - Fast Causal Inference (handles latent variables)",
                        "GES - Greedy Equivalence Search",
                        "LiNGAM - Linear Non-Gaussian Acyclic Model"
                    ],
                    "default": "pc"
                },
                "significance_level": {
                    "type": "number",
                    "title": "Significance Level",
                    "description": "Significance level for independence tests (alpha)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.05
                },
                "max_path_length": {
                    "type": "integer",
                    "title": "Maximum Path Length",
                    "description": "Maximum length of causal paths to analyze",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                },
                "knowledge_graph_id": {
                    "type": "string",
                    "title": "Knowledge Graph ID",
                    "description": "Optional ID of knowledge graph to load data from",
                    "default": ""
                }
            },
            "required": ["operation"]
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least fallback analysis is available
        """
        # Node can work with or without Causal-Learn (has fallback)
        return True

    def get_available_algorithms(self) -> List[str]:
        """
        Get list of available causal discovery algorithms.

        Returns:
            List of algorithm names that are currently available
        """
        if self.causal_integration and self.causal_integration.is_available():
            return self.causal_integration.get_available_algorithms()
        else:
            return ['correlation_fallback']

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the node."""
        return {
            'available': True,
            'causal_learn_available': (
                self.causal_integration is not None and 
                self.causal_integration.is_available()
            ),
            'algorithms_available': self.get_available_algorithms(),
            'valid_operations': self.valid_operations,
            'version': self.VERSION
        }
