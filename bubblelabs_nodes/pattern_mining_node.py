"""
Pattern Mining Node for BubbleLabs Integration

Discovers patterns, associations, and anomalies in knowledge graphs using PAMI
(Pattern Mining) algorithms. Supports frequent pattern mining, association rule
discovery, sequential pattern mining, and anomaly detection.

Features:
- Mine frequent patterns in knowledge graphs
- Discover association rules between entities
- Find sequential patterns (temporal)
- Detect anomalous patterns
- Generate comprehensive pattern reports
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import Counter
import itertools
from .base_node import BubbleLabsNode, NodeExecutionError


class PatternMiningNode(BubbleLabsNode):
    """
    Pattern Mining Node for discovering patterns in knowledge graphs.

    Uses PAMI (Pattern Mining) algorithms to analyze knowledge graphs and discover:
    - Frequent patterns: Recurring combinations of entities/relationships
    - Association rules: If-then rules between entities with confidence metrics
    - Sequential patterns: Temporal patterns in entity interactions
    - Anomalies: Unusual patterns that deviate from normal behavior

    The node can work with knowledge graph IDs from the UnifiedKGIntegrationHub
    or process knowledge graph data directly from the workflow context.
    """

    # Node metadata
    DISPLAY_NAME = "Pattern Mining"
    DESCRIPTION = "Discover patterns, associations, and anomalies in knowledge graphs using PAMI"
    ICON = "pattern-mining"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of PAMIIntegration
        pami_module = self.safe_import(
            'knowledge_engine.integrations.pami_integration',
            fallback_value=None,
            error_msg="PAMI integration not available for PatternMiningNode"
        )

        self.PAMIIntegration = None
        self.pami_miner = None

        if pami_module:
            self.PAMIIntegration = getattr(pami_module, 'PAMIIntegration', None)
            if self.PAMIIntegration:
                try:
                    self.pami_miner = self.PAMIIntegration()
                    self.logger.info("PAMI integration initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Could not initialize PAMI integration: {e}")
                    self.pami_miner = None

        # Safe import of UnifiedKGIntegrationHub
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for PatternMiningNode"
        )

        self.UnifiedKGIntegrationHub = None
        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)

        # Initialize hub instance if available
        self.hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Track availability
        self.pami_available = self.pami_miner is not None and (
            self.pami_miner.is_available() if hasattr(self.pami_miner, 'is_available') else True
        )

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (one of):
            - knowledge_graph_id: str - ID of the knowledge graph to analyze
            - knowledge_graph: dict - Knowledge graph data directly

        Optional:
            - mining_type: str - Override the configured mining type
            - entity_types: list - Filter by specific entity types
            - time_window: str - Time range for temporal mining
        """
        errors = []

        # Check that we have either knowledge_graph_id or knowledge_graph
        has_kg_id = 'knowledge_graph_id' in inputs and inputs['knowledge_graph_id']
        has_kg = 'knowledge_graph' in inputs and inputs['knowledge_graph']

        if not has_kg_id and not has_kg:
            errors.append("Missing required input: either 'knowledge_graph_id' or 'knowledge_graph' must be provided")

        # Validate mining_type if provided
        if 'mining_type' in inputs:
            valid_types = ['frequent_patterns', 'association_rules', 'sequential', 'anomaly_detection']
            if inputs['mining_type'] not in valid_types:
                errors.append(f"Invalid mining_type: '{inputs['mining_type']}'. Must be one of: {', '.join(valid_types)}")

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("'entity_types' must be a list of strings")
            elif not all(isinstance(et, str) for et in inputs['entity_types']):
                errors.append("All items in 'entity_types' must be strings")

        # Validate time_window format if provided
        if 'time_window' in inputs and inputs['time_window']:
            time_window = inputs['time_window']
            if not isinstance(time_window, str):
                errors.append("'time_window' must be a string")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute pattern mining on the knowledge graph.

        Args:
            inputs: Contains knowledge_graph_id or knowledge_graph, plus optional parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - patterns: List of discovered patterns
                - rules: List of association rules (if applicable)
                - anomalies: List of anomalous patterns
                - statistics: Mining statistics and metadata

        Raises:
            NodeExecutionError: If mining fails
        """
        # Get configuration
        mining_type = inputs.get('mining_type', self.config.get('mining_type', 'frequent_patterns'))
        min_support = inputs.get('min_support', self.config.get('min_support', 0.1))
        min_confidence = inputs.get('min_confidence', self.config.get('min_confidence', 0.7))
        max_pattern_length = inputs.get('max_pattern_length', self.config.get('max_pattern_length', 5))
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        time_window = inputs.get('time_window', self.config.get('time_window', None))

        context.update_progress(10, f"Initializing {mining_type} pattern mining")
        self.logger.info(f"Starting pattern mining: type={mining_type}, min_support={min_support}")

        try:
            # Retrieve knowledge graph data
            kg_data = self._get_knowledge_graph_data(inputs, context)

            if not kg_data:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message="Could not retrieve knowledge graph data",
                    details={'inputs': list(inputs.keys())}
                )

            context.update_progress(30, "Knowledge graph retrieved, preparing data for mining")

            # Filter by entity types if specified
            if entity_types:
                kg_data = self._filter_by_entity_types(kg_data, entity_types)
                context.update_progress(35, f"Filtered to {len(kg_data.get('nodes', []))} nodes of specified types")

            # Filter by time window if specified
            if time_window:
                kg_data = self._filter_by_time_window(kg_data, time_window)
                context.update_progress(40, f"Filtered by time window: {time_window}")

            context.update_progress(50, f"Executing {mining_type} mining")

            # Execute mining based on type
            if mining_type == 'frequent_patterns':
                result = self._mine_frequent_patterns(
                    kg_data, min_support, max_pattern_length, context
                )
            elif mining_type == 'association_rules':
                result = self._mine_association_rules(
                    kg_data, min_support, min_confidence, max_pattern_length, context
                )
            elif mining_type == 'sequential':
                result = self._mine_sequential_patterns(
                    kg_data, min_support, max_pattern_length, time_window, context
                )
            elif mining_type == 'anomaly_detection':
                result = self._detect_anomalies(
                    kg_data, min_support, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown mining type: {mining_type}",
                    details={'valid_types': ['frequent_patterns', 'association_rules', 'sequential', 'anomaly_detection']}
                )

            context.update_progress(90, "Processing and formatting results")

            # Add execution metadata
            result['metadata'] = {
                'mining_type': mining_type,
                'min_support': min_support,
                'min_confidence': min_confidence if mining_type == 'association_rules' else None,
                'max_pattern_length': max_pattern_length,
                'entity_types': entity_types,
                'time_window': time_window,
                'executed_at': datetime.now().isoformat(),
                'execution_id': self.execution_id,
                'pami_available': self.pami_available,
                'knowledge_graph_stats': {
                    'node_count': len(kg_data.get('nodes', [])),
                    'edge_count': len(kg_data.get('edges', [])),
                    'triple_count': len(kg_data.get('triples', []))
                }
            }

            # Store artifacts in context
            context.add_artifact('pattern_mining', {
                'mining_type': mining_type,
                'patterns_count': len(result.get('patterns', [])),
                'rules_count': len(result.get('rules', [])),
                'anomalies_count': len(result.get('anomalies', [])),
                'statistics': result.get('statistics', {})
            })

            context.update_progress(100, f"Pattern mining complete: {result.get('statistics', {}).get('total_patterns', 0)} patterns found")

            self.logger.info(
                f"Pattern mining completed: type={mining_type}, "
                f"patterns={len(result.get('patterns', []))}, "
                f"rules={len(result.get('rules', []))}, "
                f"anomalies={len(result.get('anomalies', []))}"
            )

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Pattern mining failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Pattern mining failed: {str(e)}",
                details={
                    'mining_type': mining_type,
                    'min_support': min_support,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_knowledge_graph_data(self, inputs: Dict, context) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge graph data from inputs or hub.

        Priority:
        1. knowledge_graph from inputs (direct data)
        2. knowledge_graph_id from inputs (fetch from hub)
        3. knowledge_graph from context (workflow context)
        """
        # Check for direct KG data
        if 'knowledge_graph' in inputs and inputs['knowledge_graph']:
            return inputs['knowledge_graph']

        # Check for KG in context
        if 'knowledge_graph' in context.artifacts:
            return context.artifacts['knowledge_graph']

        # Try to fetch from hub using knowledge_graph_id
        kg_id = inputs.get('knowledge_graph_id')
        if kg_id and self.hub:
            try:
                # Try different methods to retrieve the KG
                if hasattr(self.hub, 'get_knowledge_graph'):
                    return self.hub.get_knowledge_graph(kg_id)
                elif hasattr(self.hub, 'export_knowledge_graph'):
                    return self.hub.export_knowledge_graph(kg_id)
                elif hasattr(self.hub, 'query'):
                    return self.hub.query(kg_id)
            except Exception as e:
                self.logger.warning(f"Could not fetch knowledge graph from hub: {e}")

        # Return empty structure if nothing found
        return None

    def _filter_by_entity_types(self, kg_data: Dict[str, Any], entity_types: List[str]) -> Dict[str, Any]:
        """Filter knowledge graph to only include specified entity types."""
        if not entity_types:
            return kg_data

        entity_type_set = set(entity_types)

        # Filter nodes
        filtered_nodes = [
            node for node in kg_data.get('nodes', [])
            if node.get('type', 'unknown') in entity_type_set
        ]

        # Get IDs of filtered nodes
        filtered_node_ids = {node.get('id') for node in filtered_nodes}

        # Filter edges to only include connections between filtered nodes
        filtered_edges = [
            edge for edge in kg_data.get('edges', [])
            if edge.get('source') in filtered_node_ids and edge.get('target') in filtered_node_ids
        ]

        # Filter triples similarly
        filtered_triples = [
            triple for triple in kg_data.get('triples', [])
            if triple.get('subject') in filtered_node_ids and triple.get('object') in filtered_node_ids
        ]

        return {
            'nodes': filtered_nodes,
            'edges': filtered_edges,
            'triples': filtered_triples
        }

    def _filter_by_time_window(self, kg_data: Dict[str, Any], time_window: str) -> Dict[str, Any]:
        """
        Filter knowledge graph by time window.

        Time window format examples:
        - "2024-01-01/2024-12-31" (date range)
        - "last_30_days"
        - "last_7_days"
        - "last_24_hours"
        """
        from datetime import datetime, timedelta

        now = datetime.now()

        if time_window == 'last_24_hours':
            cutoff = now - timedelta(days=1)
        elif time_window == 'last_7_days':
            cutoff = now - timedelta(days=7)
        elif time_window == 'last_30_days':
            cutoff = now - timedelta(days=30)
        elif '/' in time_window:
            # Date range format
            try:
                start_str, end_str = time_window.split('/')
                start_date = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))

                filtered_nodes = [
                    node for node in kg_data.get('nodes', [])
                    if self._is_in_date_range(node, start_date, end_date)
                ]
                filtered_edges = [
                    edge for edge in kg_data.get('edges', [])
                    if self._is_in_date_range(edge, start_date, end_date)
                ]
                filtered_triples = [
                    triple for triple in kg_data.get('triples', [])
                    if self._is_in_date_range(triple, start_date, end_date)
                ]

                return {
                    'nodes': filtered_nodes,
                    'edges': filtered_edges,
                    'triples': filtered_triples
                }
            except Exception as e:
                self.logger.warning(f"Could not parse time window '{time_window}': {e}")
                return kg_data
        else:
            # Unknown format, return original
            return kg_data

        # Filter by cutoff date for relative time windows
        filtered_nodes = [
            node for node in kg_data.get('nodes', [])
            if self._is_after_date(node, cutoff)
        ]
        filtered_edges = [
            edge for edge in kg_data.get('edges', [])
            if self._is_after_date(edge, cutoff)
        ]
        filtered_triples = [
            triple for triple in kg_data.get('triples', [])
            if self._is_after_date(triple, cutoff)
        ]

        return {
            'nodes': filtered_nodes,
            'edges': filtered_edges,
            'triples': filtered_triples
        }

    def _is_in_date_range(self, item: Dict, start: datetime, end: datetime) -> bool:
        """Check if item timestamp is within date range."""
        timestamp = self._get_timestamp(item)
        if timestamp is None:
            return True  # Include items without timestamps
        return start <= timestamp <= end

    def _is_after_date(self, item: Dict, cutoff: datetime) -> bool:
        """Check if item timestamp is after cutoff."""
        timestamp = self._get_timestamp(item)
        if timestamp is None:
            return True  # Include items without timestamps
        return timestamp >= cutoff

    def _get_timestamp(self, item: Dict) -> Optional[datetime]:
        """Extract timestamp from item."""
        ts = item.get('timestamp')
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                return None
        return None

    def _mine_frequent_patterns(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        max_pattern_length: int,
        context
    ) -> Dict[str, Any]:
        """Mine frequent patterns from the knowledge graph."""
        if self.pami_available and self.pami_miner:
            return self._mine_with_pami(kg_data, min_support, max_pattern_length, context)
        else:
            return self._mine_fallback(kg_data, min_support, max_pattern_length, context)

    def _mine_with_pami(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        max_pattern_length: int,
        context
    ) -> Dict[str, Any]:
        """Mine patterns using PAMI integration."""
        context.update_progress(60, "Mining with PAMI")

        # Convert KG to transaction format
        transactions = self._kg_to_transactions(kg_data)

        # Use PAMI to mine patterns
        try:
            result = self.pami_miner.mine_frequent_patterns(
                data=transactions,
                min_support=min_support
            )

            patterns = result.get('patterns', [])
            statistics = result.get('statistics', {})

            return {
                'patterns': patterns,
                'rules': [],
                'anomalies': [],
                'statistics': {
                    'total_patterns': len(patterns),
                    'patterns_by_length': statistics.get('patterns_by_length', {}),
                    'average_support': statistics.get('average_support', 0.0),
                    'max_support': statistics.get('max_support', 0.0),
                    'min_support': statistics.get('min_support', 0.0),
                    'method': 'pami'
                }
            }
        except Exception as e:
            self.logger.warning(f"PAMI mining failed: {e}, using fallback")
            return self._mine_fallback(kg_data, min_support, max_pattern_length, context)

    def _mine_fallback(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        max_pattern_length: int,
        context
    ) -> Dict[str, Any]:
        """Fallback pattern mining when PAMI is not available."""
        context.update_progress(60, "Mining with fallback algorithm")

        patterns = []

        # Extract entity type patterns
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        triples = kg_data.get('triples', [])

        # Mine frequent entity types
        entity_types = [node.get('type', 'unknown') for node in nodes]
        type_counts = Counter(entity_types)
        abs_min_support = max(1, int(len(nodes) * min_support)) if nodes else 1

        for entity_type, count in type_counts.items():
            if count >= abs_min_support:
                patterns.append({
                    'pattern': [entity_type],
                    'support': count,
                    'support_ratio': count / len(nodes) if nodes else 0,
                    'length': 1,
                    'type': 'entity_type'
                })

        # Mine frequent relationship types
        rel_types = [edge.get('type', 'unknown') for edge in edges]
        rel_counts = Counter(rel_types)
        abs_min_support_edges = max(1, int(len(edges) * min_support)) if edges else 1

        for rel_type, count in rel_counts.items():
            if count >= abs_min_support_edges:
                patterns.append({
                    'pattern': [rel_type],
                    'support': count,
                    'support_ratio': count / len(edges) if edges else 0,
                    'length': 1,
                    'type': 'relationship_type'
                })

        # Mine triple patterns (subject_type - predicate - object_type)
        node_type_map = {node.get('id'): node.get('type', 'unknown') for node in nodes}
        triple_patterns = []

        for triple in triples:
            subject_type = node_type_map.get(triple.get('subject'), 'unknown')
            predicate = triple.get('predicate', 'unknown')
            object_type = node_type_map.get(triple.get('object'), 'unknown')
            triple_patterns.append(f"{subject_type}-{predicate}-{object_type}")

        triple_counts = Counter(triple_patterns)
        abs_min_support_triples = max(1, int(len(triples) * min_support)) if triples else 1

        for triple_pattern, count in triple_counts.items():
            if count >= abs_min_support_triples:
                patterns.append({
                    'pattern': triple_pattern.split('-'),
                    'support': count,
                    'support_ratio': count / len(triples) if triples else 0,
                    'length': 3,
                    'type': 'triple_pattern'
                })

        # Sort by support
        patterns.sort(key=lambda x: x['support'], reverse=True)

        # Calculate statistics
        lengths = {}
        supports = []
        for pattern in patterns:
            length = pattern['length']
            lengths[length] = lengths.get(length, 0) + 1
            supports.append(pattern['support_ratio'])

        return {
            'patterns': patterns,
            'rules': [],
            'anomalies': [],
            'statistics': {
                'total_patterns': len(patterns),
                'patterns_by_length': lengths,
                'average_support': sum(supports) / len(supports) if supports else 0.0,
                'max_support': max(supports) if supports else 0.0,
                'min_support': min(supports) if supports else 0.0,
                'method': 'fallback'
            }
        }

    def _mine_association_rules(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        min_confidence: float,
        max_pattern_length: int,
        context
    ) -> Dict[str, Any]:
        """Discover association rules from the knowledge graph."""
        context.update_progress(60, "Discovering association rules")

        # First mine frequent patterns
        if self.pami_available and self.pami_miner and hasattr(self.pami_miner, 'discover_association_rules'):
            try:
                transactions = self._kg_to_transactions(kg_data)
                result = self.pami_miner.discover_association_rules(
                    transactions=transactions,
                    min_support=min_support,
                    min_confidence=min_confidence
                )

                return {
                    'patterns': [],
                    'rules': result.get('rules', []),
                    'anomalies': [],
                    'statistics': {
                        'total_rules': result.get('statistics', {}).get('total_rules', 0),
                        'average_confidence': result.get('statistics', {}).get('average_confidence', 0.0),
                        'average_support': result.get('statistics', {}).get('average_support', 0.0),
                        'method': 'pami'
                    }
                }
            except Exception as e:
                self.logger.warning(f"PAMI association rule mining failed: {e}, using fallback")

        # Fallback association rule mining
        return self._mine_association_rules_fallback(kg_data, min_support, min_confidence, context)

    def _mine_association_rules_fallback(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        min_confidence: float,
        context
    ) -> Dict[str, Any]:
        """Fallback association rule discovery."""
        rules = []

        # Get frequent patterns first
        patterns_result = self._mine_fallback(kg_data, min_support, 3, context)
        patterns = patterns_result.get('patterns', [])

        # Generate rules from patterns with length >= 2
        multi_item_patterns = [p for p in patterns if len(p.get('pattern', [])) >= 2]

        for pattern in multi_item_patterns:
            items = pattern['pattern']
            pattern_support = pattern['support_ratio']

            # Generate all possible antecedent -> consequent combinations
            for i in range(1, len(items)):
                for antecedent_indices in itertools.combinations(range(len(items)), i):
                    antecedent = [items[j] for j in antecedent_indices]
                    consequent = [items[j] for j in range(len(items)) if j not in antecedent_indices]

                    if not consequent:
                        continue

                    # Calculate confidence: support(pattern) / support(antecedent)
                    antecedent_support = self._get_itemset_support(kg_data, antecedent)

                    if antecedent_support > 0:
                        confidence = pattern_support / antecedent_support

                        if confidence >= min_confidence:
                            # Calculate lift
                            consequent_support = self._get_itemset_support(kg_data, consequent)
                            lift = confidence / consequent_support if consequent_support > 0 else 0

                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': pattern_support,
                                'confidence': confidence,
                                'lift': lift
                            })

        # Sort by confidence
        rules.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'patterns': [],
            'rules': rules,
            'anomalies': [],
            'statistics': {
                'total_rules': len(rules),
                'average_confidence': sum(r['confidence'] for r in rules) / len(rules) if rules else 0,
                'average_support': sum(r['support'] for r in rules) / len(rules) if rules else 0,
                'average_lift': sum(r['lift'] for r in rules) / len(rules) if rules else 0,
                'method': 'fallback'
            }
        }

    def _mine_sequential_patterns(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        max_pattern_length: int,
        time_window: Optional[str],
        context
    ) -> Dict[str, Any]:
        """Mine sequential patterns from temporal knowledge graph data."""
        context.update_progress(60, "Mining sequential patterns")

        # Extract temporal sequences from the knowledge graph
        sequences = self._extract_temporal_sequences(kg_data)

        if not sequences:
            return {
                'patterns': [],
                'rules': [],
                'anomalies': [],
                'statistics': {
                    'total_patterns': 0,
                    'message': 'No temporal sequences found in knowledge graph',
                    'method': 'none'
                }
            }

        if self.pami_available and self.pami_miner and hasattr(self.pami_miner, 'mine_sequences'):
            try:
                result = self.pami_miner.mine_sequences(
                    sequences=sequences,
                    min_support=min_support
                )

                patterns = result.get('patterns', [])

                return {
                    'patterns': patterns,
                    'rules': [],
                    'anomalies': [],
                    'statistics': {
                        'total_patterns': len(patterns),
                        'patterns_by_length': result.get('statistics', {}).get('patterns_by_length', {}),
                        'average_support': result.get('statistics', {}).get('average_support', 0.0),
                        'method': 'pami'
                    }
                }
            except Exception as e:
                self.logger.warning(f"PAMI sequence mining failed: {e}, using fallback")

        # Fallback sequential pattern mining
        return self._mine_sequential_fallback(sequences, min_support, max_pattern_length, context)

    def _mine_sequential_fallback(
        self,
        sequences: List[List[List[str]]],
        min_support: float,
        max_pattern_length: int,
        context
    ) -> Dict[str, Any]:
        """Fallback sequential pattern mining using simple approach."""
        patterns = []

        abs_min_support = max(1, int(len(sequences) * min_support)) if sequences else 1

        # Flatten sequences to find frequent items
        all_items = []
        for sequence in sequences:
            for itemset in sequence:
                all_items.extend(itemset)

        item_counts = Counter(all_items)
        frequent_items = {item for item, count in item_counts.items() if count >= abs_min_support}

        # Find frequent 1-sequences
        for item in frequent_items:
            support = sum(1 for seq in sequences if any(item in itemset for itemset in seq))
            if support >= abs_min_support:
                patterns.append({
                    'pattern': [item],
                    'support': support,
                    'support_ratio': support / len(sequences),
                    'length': 1
                })

        # Find frequent 2-sequences (simplified)
        item_pairs = list(itertools.combinations(frequent_items, 2))
        for pair in item_pairs:
            support = 0
            for sequence in sequences:
                # Check if first item appears before second
                found_first = False
                for itemset in sequence:
                    if pair[0] in itemset:
                        found_first = True
                    elif found_first and pair[1] in itemset:
                        support += 1
                        break

            if support >= abs_min_support:
                patterns.append({
                    'pattern': list(pair),
                    'support': support,
                    'support_ratio': support / len(sequences),
                    'length': 2
                })

        # Sort by support
        patterns.sort(key=lambda x: x['support'], reverse=True)

        # Calculate statistics
        lengths = {}
        supports = []
        for pattern in patterns:
            length = pattern['length']
            lengths[length] = lengths.get(length, 0) + 1
            supports.append(pattern['support_ratio'])

        return {
            'patterns': patterns,
            'rules': [],
            'anomalies': [],
            'statistics': {
                'total_patterns': len(patterns),
                'patterns_by_length': lengths,
                'average_support': sum(supports) / len(supports) if supports else 0.0,
                'method': 'fallback'
            }
        }

    def _detect_anomalies(
        self,
        kg_data: Dict[str, Any],
        min_support: float,
        context
    ) -> Dict[str, Any]:
        """Detect anomalous patterns in the knowledge graph."""
        context.update_progress(60, "Detecting anomalies")

        anomalies = []

        # Get normal patterns first
        patterns_result = self._mine_fallback(kg_data, min_support, 3, context)
        patterns = patterns_result.get('patterns', [])

        # Create a set of frequent patterns for quick lookup
        frequent_patterns = set()
        for pattern in patterns:
            pattern_tuple = tuple(pattern['pattern']) if isinstance(pattern['pattern'], list) else (pattern['pattern'],)
            frequent_patterns.add(pattern_tuple)

        # Detect anomalies:
        # 1. Rare entity types (below min_support)
        nodes = kg_data.get('nodes', [])
        entity_types = [node.get('type', 'unknown') for node in nodes]
        type_counts = Counter(entity_types)
        abs_min_support = max(1, int(len(nodes) * min_support)) if nodes else 1

        rare_types = [
            {'entity_type': et, 'count': count, 'anomaly_type': 'rare_entity_type'}
            for et, count in type_counts.items()
            if count < abs_min_support and count > 0
        ]
        anomalies.extend(rare_types)

        # 2. Unusual triple patterns (not in frequent patterns)
        node_type_map = {node.get('id'): node.get('type', 'unknown') for node in nodes}
        triples = kg_data.get('triples', [])

        unusual_triples = []
        for triple in triples:
            subject_type = node_type_map.get(triple.get('subject'), 'unknown')
            predicate = triple.get('predicate', 'unknown')
            object_type = node_type_map.get(triple.get('object'), 'unknown')

            pattern_tuple = (subject_type, predicate, object_type)

            if pattern_tuple not in frequent_patterns:
                unusual_triples.append({
                    'triple': {
                        'subject': triple.get('subject'),
                        'predicate': predicate,
                        'object': triple.get('object')
                    },
                    'pattern': list(pattern_tuple),
                    'anomaly_type': 'unusual_triple_pattern'
                })

        # Limit unusual triples to avoid overwhelming results
        anomalies.extend(unusual_triples[:100])

        # 3. Entities with unusual connectivity
        edge_counts = Counter()
        for edge in kg_data.get('edges', []):
            edge_counts[edge.get('source')] += 1
            edge_counts[edge.get('target')] += 1

        if edge_counts:
            avg_edges = sum(edge_counts.values()) / len(edge_counts)
            std_edges = (sum((count - avg_edges) ** 2 for count in edge_counts.values()) / len(edge_counts)) ** 0.5

            for node_id, count in edge_counts.items():
                if std_edges > 0 and abs(count - avg_edges) > 2 * std_edges:
                    anomalies.append({
                        'entity_id': node_id,
                        'edge_count': count,
                        'average_edges': avg_edges,
                        'deviation': abs(count - avg_edges) / std_edges if std_edges > 0 else 0,
                        'anomaly_type': 'unusual_connectivity'
                    })

        return {
            'patterns': [],
            'rules': [],
            'anomalies': anomalies,
            'statistics': {
                'total_anomalies': len(anomalies),
                'rare_entity_types': len(rare_types),
                'unusual_triple_patterns': len(unusual_triples),
                'unusual_connectivity': len([a for a in anomalies if a.get('anomaly_type') == 'unusual_connectivity']),
                'method': 'statistical'
            }
        }

    def _kg_to_transactions(self, kg_data: Dict[str, Any]) -> List[List[str]]:
        """
        Convert knowledge graph to transaction format for pattern mining.

        Each transaction represents an entity and its attributes/relationships.
        """
        transactions = []

        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        triples = kg_data.get('triples', [])

        # Create a map of entity relationships
        entity_relationships = {}
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            rel_type = edge.get('type', 'related_to')

            if source not in entity_relationships:
                entity_relationships[source] = []
            if target not in entity_relationships:
                entity_relationships[target] = []

            entity_relationships[source].append(f"{rel_type}:{target}")
            entity_relationships[target].append(f"inverse_{rel_type}:{source}")

        # Convert each node to a transaction
        for node in nodes:
            transaction = []

            # Add entity type
            entity_type = node.get('type', 'unknown')
            transaction.append(f"type:{entity_type}")

            # Add entity attributes
            for key, value in node.items():
                if key not in ['id', 'type'] and value is not None:
                    transaction.append(f"{key}:{value}")

            # Add relationships
            entity_id = node.get('id')
            if entity_id in entity_relationships:
                transaction.extend(entity_relationships[entity_id])

            if transaction:
                transactions.append(transaction)

        # If no nodes, try using triples
        if not transactions and triples:
            for triple in triples:
                transaction = [
                    f"subject:{triple.get('subject')}",
                    f"predicate:{triple.get('predicate')}",
                    f"object:{triple.get('object')}"
                ]
                transactions.append(transaction)

        return transactions

    def _extract_temporal_sequences(self, kg_data: Dict[str, Any]) -> List[List[List[str]]]:
        """
        Extract temporal sequences from knowledge graph.

        Returns sequences ordered by timestamp, where each sequence is a list
        of itemsets (events that occurred at the same time).
        """
        sequences = []

        # Group items by timestamp
        timestamped_items = {}

        # Process nodes
        for node in kg_data.get('nodes', []):
            timestamp = self._get_timestamp(node)
            if timestamp:
                if timestamp not in timestamped_items:
                    timestamped_items[timestamp] = []
                timestamped_items[timestamp].append(f"node:{node.get('type', 'unknown')}")

        # Process edges
        for edge in kg_data.get('edges', []):
            timestamp = self._get_timestamp(edge)
            if timestamp:
                if timestamp not in timestamped_items:
                    timestamped_items[timestamp] = []
                timestamped_items[timestamp].append(f"edge:{edge.get('type', 'unknown')}")

        # Sort by timestamp and create sequences
        sorted_timestamps = sorted(timestamped_items.keys())

        for timestamp in sorted_timestamps:
            items = timestamped_items[timestamp]
            if items:
                sequences.append([items])

        return sequences

    def _get_itemset_support(self, kg_data: Dict[str, Any], itemset: List[str]) -> float:
        """Calculate support for an itemset in the knowledge graph."""
        nodes = kg_data.get('nodes', [])
        if not nodes:
            return 0.0

        count = 0
        itemset_set = set(itemset)

        for node in nodes:
            node_items = {f"type:{node.get('type', 'unknown')}"}
            for key, value in node.items():
                if key not in ['id'] and value is not None:
                    node_items.add(f"{key}:{value}")

            if itemset_set.issubset(node_items):
                count += 1

        return count / len(nodes)

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Pattern Mining Configuration",
            "description": "Configure pattern mining on knowledge graphs using PAMI algorithms",
            "properties": {
                "mining_type": {
                    "type": "string",
                    "title": "Mining Type",
                    "description": "Type of pattern mining to perform",
                    "enum": ["frequent_patterns", "association_rules", "sequential", "anomaly_detection"],
                    "enumNames": [
                        "Frequent Patterns - Find recurring patterns",
                        "Association Rules - Discover if-then rules",
                        "Sequential Patterns - Find temporal patterns",
                        "Anomaly Detection - Find unusual patterns"
                    ],
                    "default": "frequent_patterns"
                },
                "min_support": {
                    "type": "number",
                    "title": "Minimum Support",
                    "description": "Minimum pattern frequency threshold (0.0-1.0). Higher values return fewer, more common patterns.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "description": "Minimum confidence threshold for association rules (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                },
                "max_pattern_length": {
                    "type": "integer",
                    "title": "Maximum Pattern Length",
                    "description": "Maximum length of patterns to discover",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Filter mining to specific entity types (empty = all types)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "time_window": {
                    "type": "string",
                    "title": "Time Window",
                    "description": "Time range for temporal mining (e.g., 'last_7_days', 'last_30_days', '2024-01-01/2024-12-31')",
                    "default": ""
                }
            },
            "required": ["mining_type"],
            "dependencies": {
                "mining_type": {
                    "oneOf": [
                        {
                            "properties": {
                                "mining_type": {"enum": ["frequent_patterns"]}
                            },
                            "description": "Find frequently occurring patterns in the knowledge graph"
                        },
                        {
                            "properties": {
                                "mining_type": {"enum": ["association_rules"]}
                            },
                            "required": ["min_confidence"],
                            "description": "Discover association rules with confidence metrics"
                        },
                        {
                            "properties": {
                                "mining_type": {"enum": ["sequential"]}
                            },
                            "description": "Find sequential/temporal patterns"
                        },
                        {
                            "properties": {
                                "mining_type": {"enum": ["anomaly_detection"]}
                            },
                            "description": "Detect anomalous or unusual patterns"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if the node can execute (fallback is always available)
        """
        # Node can work with or without PAMI (has fallback)
        return True

    def get_available_mining_methods(self) -> Dict[str, bool]:
        """
        Get available mining methods and their availability status.

        Returns:
            Dictionary mapping method names to availability (True/False)
        """
        return {
            'frequent_patterns': True,  # Always available via fallback
            'association_rules': True,  # Always available via fallback
            'sequential': True,         # Always available via fallback
            'anomaly_detection': True,  # Always available via fallback
            'pami_backend': self.pami_available
        }

    def generate_pattern_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a human-readable pattern report.

        Args:
            result: The result dictionary from execute()

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "PATTERN MINING REPORT",
            "=" * 60,
            "",
            f"Mining Type: {result.get('metadata', {}).get('mining_type', 'unknown')}",
            f"Executed At: {result.get('metadata', {}).get('executed_at', 'unknown')}",
            f"Method: {result.get('statistics', {}).get('method', 'unknown')}",
            "",
            "-" * 40,
            "STATISTICS",
            "-" * 40,
        ]

        stats = result.get('statistics', {})
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        patterns = result.get('patterns', [])
        if patterns:
            lines.extend([
                "",
                "-" * 40,
                f"TOP PATTERNS (showing up to 10 of {len(patterns)})",
                "-" * 40
            ])
            for i, pattern in enumerate(patterns[:10], 1):
                pattern_items = pattern.get('pattern', [])
                support = pattern.get('support_ratio', 0)
                lines.append(f"  {i}. {' → '.join(str(p) for p in pattern_items)} (support: {support:.3f})")

        rules = result.get('rules', [])
        if rules:
            lines.extend([
                "",
                "-" * 40,
                f"TOP ASSOCIATION RULES (showing up to 10 of {len(rules)})",
                "-" * 40
            ])
            for i, rule in enumerate(rules[:10], 1):
                antecedent = rule.get('antecedent', [])
                consequent = rule.get('consequent', [])
                confidence = rule.get('confidence', 0)
                support = rule.get('support', 0)
                lines.append(f"  {i}. {' ∧ '.join(str(a) for a in antecedent)} → {' ∧ '.join(str(c) for c in consequent)}")
                lines.append(f"     (confidence: {confidence:.3f}, support: {support:.3f})")

        anomalies = result.get('anomalies', [])
        if anomalies:
            lines.extend([
                "",
                "-" * 40,
                f"ANOMALIES DETECTED ({len(anomalies)} total)",
                "-" * 40
            ])
            for i, anomaly in enumerate(anomalies[:10], 1):
                anomaly_type = anomaly.get('anomaly_type', 'unknown')
                lines.append(f"  {i}. Type: {anomaly_type}")
                if 'entity_type' in anomaly:
                    lines.append(f"     Entity Type: {anomaly['entity_type']} (count: {anomaly.get('count', 0)})")
                elif 'entity_id' in anomaly:
                    lines.append(f"     Entity ID: {anomaly['entity_id']} (edges: {anomaly.get('edge_count', 0)})")

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])

        return "\n".join(lines)
