"""
Contradiction Detection Node for BubbleLabs Integration

Provides comprehensive contradiction detection and resolution capabilities:
- Detect logical contradictions (A says X, B says not X)
- Identify temporal conflicts (X at T1 and T2 where T1 != T2)
- Find factual inconsistencies (different values for same property)
- Discover semantic contradictions (contradictory relationships)
- Assess severity and suggest resolutions
- Generate detailed contradiction reports
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
import re
from .base_node import BubbleLabsNode, NodeExecutionError


class ContradictionType(Enum):
    """Types of contradictions that can be detected."""
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    FACTUAL = "factual"
    SEMANTIC = "semantic"


class SeverityLevel(Enum):
    """Severity levels for contradictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContradictionDetectionNode(BubbleLabsNode):
    """
    Detect and resolve conflicting knowledge in the knowledge graph.

    This node performs comprehensive contradiction analysis:
    - Detect: Find all contradictions in the knowledge graph
    - Analyze: Deep analysis of specific contradictions
    - Resolve: Apply resolution strategies to contradictions
    - Report: Generate detailed contradiction reports

    Supports multiple contradiction types including logical, temporal,
    factual, and semantic contradictions with configurable severity
    thresholds and resolution strategies.
    """

    # Node metadata
    DISPLAY_NAME = "Contradiction Detection"
    DESCRIPTION = "Detect and resolve conflicting knowledge in the graph"
    ICON = "contradiction-detection"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    # Common contradictory terms for semantic analysis
    CONTRADICTORY_PAIRS = [
        ("is", "is_not"),
        ("has", "has_not"),
        ("can", "cannot"),
        ("will", "will_not"),
        ("true", "false"),
        ("yes", "no"),
        ("exists", "does_not_exist"),
        ("equal", "not_equal"),
        ("greater_than", "less_than_or_equal"),
        ("less_than", "greater_than_or_equal"),
        ("increases", "decreases"),
        ("causes", "prevents"),
        ("enables", "disables"),
        ("contains", "excludes"),
        ("starts", "ends"),
        ("creates", "destroys"),
        ("approves", "rejects"),
        ("accepts", "denies"),
        ("positive", "negative"),
        ("active", "inactive"),
        ("valid", "invalid"),
        ("enabled", "disabled"),
        ("open", "closed"),
        ("success", "failure"),
        ("present", "absent"),
    ]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # CAV-NLP configuration option
        self.use_cav_nlp = self.config.get('use_cav_nlp', True)

        # Safe imports for optional dependencies (CAV-NLP)
        self.cav_nlp_bridge = self.safe_import(
            'cav_nlp.cav_nlp_math_bridge.CAVNLPMathBridge',
            fallback_value=None,
            error_msg="CAV-NLP bridge not available for ContradictionDetectionNode"
        )
        if self.cav_nlp_bridge is None:
            self.cav_nlp_bridge = self.safe_import(
                'cav_nlp_math_bridge.CAVNLPMathBridge',
                fallback_value=None,
                error_msg="CAV-NLP bridge not found in alternate path"
            )

        # Import CAV-NLP enhanced solver
        self.EnhancedSolver = self.safe_import(
            'cav_nlp.cav_nlp_math_bridge.EnhancedSolver',
            fallback_value=None,
            error_msg="CAV-NLP EnhancedSolver not available"
        )
        if self.EnhancedSolver is None:
            self.EnhancedSolver = self.safe_import(
                'cav_nlp_math_bridge.EnhancedSolver',
                fallback_value=None,
                error_msg="EnhancedSolver not found in alternate path"
            )

        # Initialize CAV-NLP enhanced solver
        self.enhanced_solver = None
        if self.use_cav_nlp and self.EnhancedSolver:
            try:
                self.enhanced_solver = self.EnhancedSolver()
                self.logger.info("CAV-NLP EnhancedSolver initialized for ContradictionDetectionNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize CAV-NLP EnhancedSolver: {e}")
                self.enhanced_solver = None

        # Safe imports for optional dependencies
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for ContradictionDetectionNode"
        )

        # Alternative import path
        if self.UnifiedKGIntegrationHub is None:
            self.UnifiedKGIntegrationHub = self.safe_import(
                'unified_kg_integration_hub.UnifiedKGIntegrationHub',
                fallback_value=None,
                error_msg="UnifiedKGIntegrationHub not found in alternate path"
            )

        # Import Z3 integration for logical reasoning (optional)
        self.z3_integration = self.safe_import(
            'knowledge_engine.reasoning.z3_integration',
            fallback_value=None,
            error_msg="Z3 integration not available for logical reasoning"
        )

        if self.z3_integration is None:
            self.z3_integration = self.safe_import(
                'z3_integration',
                fallback_value=None,
                error_msg="Z3 integration not found in alternate path"
            )

        # Initialize KG hub instance
        self.kg_hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for ContradictionDetectionNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub = None

        # Initialize Z3 solver if available
        self.z3_solver = None
        if self.z3_integration:
            try:
                self.z3_solver = getattr(self.z3_integration, 'Z3SolverEngine', None)
                self.logger.info("Z3 integration available for logical contradiction detection")
            except Exception as e:
                self.logger.warning(f"Could not initialize Z3 solver: {e}")

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - detect/analyze/resolve/report: Either 'knowledge_graph_id' or 'triples'
        - analyze/resolve: 'contradiction_id' or specific contradiction to analyze/resolve
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'detect'))

        valid_operations = ['detect', 'analyze', 'resolve', 'report']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Check for required knowledge source
        has_kg_id = inputs.get('knowledge_graph_id') or self.config.get('knowledge_graph_id')
        has_triples = inputs.get('triples') or self.config.get('triples')

        if not has_kg_id and not has_triples:
            errors.append("Either 'knowledge_graph_id' or 'triples' must be provided")

        # Validate triples structure if provided
        if 'triples' in inputs:
            triples = inputs['triples']
            if not isinstance(triples, list):
                errors.append("'triples' must be a list")
            else:
                for i, triple in enumerate(triples):
                    if not isinstance(triple, dict):
                        errors.append(f"triple[{i}] must be an object with subject, predicate, object")
                    else:
                        if 'subject' not in triple and 's' not in triple:
                            errors.append(f"triple[{i}] missing 'subject' field")
                        if 'predicate' not in triple and 'p' not in triple:
                            errors.append(f"triple[{i}] missing 'predicate' field")
                        if 'object' not in triple and 'o' not in triple:
                            errors.append(f"triple[{i}] missing 'object' field")

        # Validate check_types if provided
        if 'check_types' in inputs:
            check_types = inputs['check_types']
            valid_types = ['logical', 'temporal', 'factual', 'semantic']
            if not isinstance(check_types, list):
                errors.append("'check_types' must be an array")
            else:
                for ct in check_types:
                    if ct not in valid_types:
                        errors.append(f"Invalid check_type: {ct}. Must be one of: {', '.join(valid_types)}")

        # Validate severity_threshold if provided
        if 'severity_threshold' in inputs:
            threshold = inputs['severity_threshold']
            valid_thresholds = ['low', 'medium', 'high', 'critical']
            if threshold not in valid_thresholds:
                errors.append(f"Invalid severity_threshold: {threshold}. Must be one of: {', '.join(valid_thresholds)}")

        # Validate resolution_strategy if provided
        if 'resolution_strategy' in inputs:
            strategy = inputs['resolution_strategy']
            valid_strategies = ['keep_highest_confidence', 'keep_newest', 'manual_review', 'flag_only']
            if strategy not in valid_strategies:
                errors.append(f"Invalid resolution_strategy: {strategy}. Must be one of: {', '.join(valid_strategies)}")

        # Validate time_range format if provided
        if 'time_range' in inputs and inputs['time_range']:
            time_range = inputs['time_range']
            if not isinstance(time_range, str):
                errors.append("'time_range' must be a string")
            # Basic ISO 8601 period or timestamp validation
            elif not (re.match(r'\d{4}-\d{2}-\d{2}', time_range) or 
                      re.match(r'P\d+', time_range) or
                      '/' in time_range):
                self.logger.warning(f"time_range '{time_range}' may not be in a recognized format")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute contradiction detection based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - contradictions: List of detected contradictions
                - severity_counts: Count of contradictions by severity level
                - resolutions: List of suggested resolutions
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'detect'))
        severity_threshold = inputs.get('severity_threshold', self.config.get('severity_threshold', 'low'))
        check_types = inputs.get('check_types', self.config.get('check_types', ['logical', 'temporal', 'factual', 'semantic']))
        entity_scope = inputs.get('entity_scope', self.config.get('entity_scope', []))
        time_range = inputs.get('time_range', self.config.get('time_range'))
        resolution_strategy = inputs.get('resolution_strategy', self.config.get('resolution_strategy', 'flag_only'))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Executing contradiction {operation} with check_types={check_types}")

        try:
            # Load triples from input or knowledge graph
            triples = self._load_triples(inputs, context)

            if not triples:
                return {
                    'contradictions': [],
                    'severity_counts': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                    'resolutions': [],
                    'warning': 'No triples found for analysis',
                    'metadata': {
                        'operation': operation,
                        'triples_analyzed': 0,
                        'execution_time': 0.0
                    }
                }

            # Filter by entity scope if specified
            if entity_scope:
                triples = self._filter_by_entities(triples, entity_scope)
                context.update_progress(20, f"Filtered to {len(triples)} triples matching entity scope")

            # Filter by time range if specified
            if time_range:
                triples = self._filter_by_time_range(triples, time_range)
                context.update_progress(25, f"Filtered to {len(triples)} triples in time range")

            context.update_progress(30, f"Analyzing {len(triples)} triples for contradictions")

            # Execute based on operation type
            if operation == 'detect':
                result = self._execute_detect(triples, check_types, severity_threshold, context)
            elif operation == 'analyze':
                result = self._execute_analyze(triples, check_types, context)
            elif operation == 'resolve':
                result = self._execute_resolve(triples, check_types, severity_threshold, resolution_strategy, context)
            elif operation == 'report':
                result = self._execute_report(triples, check_types, severity_threshold, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['detect', 'analyze', 'resolve', 'report']}
                )

            # Add execution metadata
            result['metadata'] = {
                'operation': operation,
                'severity_threshold': severity_threshold,
                'check_types': check_types,
                'entity_scope': entity_scope,
                'time_range': time_range,
                'resolution_strategy': resolution_strategy,
                'triples_analyzed': len(triples),
                'execution_id': self.execution_id,
                'executed_at': datetime.now().isoformat()
            }

            # Store result in context
            context.add_artifact('contradiction_detection', {
                'operation': operation,
                'contradictions_found': len(result.get('contradictions', [])),
                'severity_counts': result.get('severity_counts', {})
            })

            context.update_progress(100, f"Contradiction {operation} complete: {len(result.get('contradictions', []))} found")
            self.logger.info(f"Contradiction detection completed: {len(result.get('contradictions', []))} contradictions found")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Contradiction detection failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Contradiction detection failed: {str(e)}",
                details={
                    'operation': operation,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _load_triples(self, inputs: Dict, context) -> List[Dict[str, Any]]:
        """Load triples from inputs or knowledge graph."""
        # First check for direct triples input
        if 'triples' in inputs and inputs['triples']:
            context.update_progress(15, f"Loading {len(inputs['triples'])} triples from input")
            return self._normalize_triples(inputs['triples'])

        if self.config.get('triples'):
            context.update_progress(15, f"Loading {len(self.config['triples'])} triples from config")
            return self._normalize_triples(self.config['triples'])

        # Try to load from knowledge graph
        kg_id = inputs.get('knowledge_graph_id') or self.config.get('knowledge_graph_id')
        if kg_id and self.kg_hub:
            context.update_progress(15, f"Loading triples from knowledge graph: {kg_id}")
            try:
                kg_data = self.kg_hub.get_knowledge_graph(kg_id)
                if kg_data and 'triples' in kg_data:
                    return self._normalize_triples(kg_data['triples'])
            except Exception as e:
                self.logger.warning(f"Could not load from KG hub: {e}")

        # Try kg_instance from inputs (from KnowledgeQueryNode)
        if 'kg_instance' in inputs:
            try:
                kg = inputs['kg_instance']
                if hasattr(kg, 'get_triples'):
                    triples = kg.get_triples()
                    return self._normalize_triples(triples)
            except Exception as e:
                self.logger.warning(f"Could not load from kg_instance: {e}")

        return []

    def _normalize_triples(self, triples: List[Any]) -> List[Dict[str, Any]]:
        """Normalize triples to standard format."""
        normalized = []
        for t in triples:
            if isinstance(t, dict):
                norm = {
                    'subject': t.get('subject') or t.get('s'),
                    'predicate': t.get('predicate') or t.get('p'),
                    'object': t.get('object') or t.get('o'),
                    'confidence': t.get('confidence', 1.0),
                    'source': t.get('source', 'unknown'),
                    'timestamp': t.get('timestamp'),
                    'metadata': t.get('metadata', {})
                }
                normalized.append(norm)
            elif hasattr(t, 'subject') and hasattr(t, 'predicate') and hasattr(t, 'object'):
                # Object-style triple
                norm = {
                    'subject': t.subject,
                    'predicate': t.predicate,
                    'object': t.object,
                    'confidence': getattr(t, 'confidence', 1.0),
                    'source': getattr(t, 'source', 'unknown'),
                    'timestamp': getattr(t, 'timestamp', None),
                    'metadata': getattr(t, 'metadata', {})
                }
                normalized.append(norm)
        return normalized

    def _filter_by_entities(self, triples: List[Dict], entity_scope: List[str]) -> List[Dict]:
        """Filter triples to only include those involving specified entities."""
        entity_set = set(entity_scope)
        return [
            t for t in triples
            if t.get('subject') in entity_set or t.get('object') in entity_set
        ]

    def _filter_by_time_range(self, triples: List[Dict], time_range: str) -> List[Dict]:
        """Filter triples by time range."""
        # Simple implementation - can be enhanced with proper date parsing
        # time_range can be:
        # - "2024-01-01/2024-12-31" (ISO 8601 interval)
        # - "P30D" (ISO 8601 duration, last 30 days)
        # - "2024-01-01" (single date)
        try:
            if '/' in time_range:
                # Interval
                start_str, end_str = time_range.split('/')
                start_date = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                return [
                    t for t in triples
                    if self._triple_in_range(t, start_date, end_date)
                ]
            elif time_range.startswith('P'):
                # Duration - calculate start date from now
                # Parse duration (simplified)
                import re
                days_match = re.search(r'(\d+)D', time_range)
                if days_match:
                    days = int(days_match.group(1))
                    from datetime import timedelta
                    start_date = datetime.now() - timedelta(days=days)
                    return [
                        t for t in triples
                        if self._triple_after_date(t, start_date)
                    ]
            else:
                # Single date
                date = datetime.fromisoformat(time_range.replace('Z', '+00:00'))
                return [
                    t for t in triples
                    if self._triple_on_date(t, date)
                ]
        except Exception as e:
            self.logger.warning(f"Could not parse time_range '{time_range}': {e}")
            return triples

    def _triple_in_range(self, triple: Dict, start: datetime, end: datetime) -> bool:
        """Check if triple timestamp is within range."""
        ts = triple.get('timestamp')
        if not ts:
            return True  # Include triples without timestamps
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return start <= ts <= end
        except:
            return True

    def _triple_after_date(self, triple: Dict, date: datetime) -> bool:
        """Check if triple timestamp is after date."""
        ts = triple.get('timestamp')
        if not ts:
            return True
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return ts >= date
        except:
            return True

    def _triple_on_date(self, triple: Dict, date: datetime) -> bool:
        """Check if triple timestamp is on specific date."""
        ts = triple.get('timestamp')
        if not ts:
            return True
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return ts.date() == date.date()
        except:
            return True

    def _execute_detect(self, triples: List[Dict], check_types: List[str],
                        severity_threshold: str, context) -> Dict[str, Any]:
        """Execute detect operation to find all contradictions."""
        context.update_progress(40, "Detecting contradictions")

        all_contradictions = []

        if 'logical' in check_types:
            context.update_progress(45, "Checking logical contradictions")
            logical = self._detect_logical_contradictions(triples)
            all_contradictions.extend(logical)

        if 'temporal' in check_types:
            context.update_progress(55, "Checking temporal contradictions")
            temporal = self._detect_temporal_contradictions(triples)
            all_contradictions.extend(temporal)

        if 'factual' in check_types:
            context.update_progress(65, "Checking factual contradictions")
            factual = self._detect_factual_contradictions(triples)
            all_contradictions.extend(factual)

        if 'semantic' in check_types:
            context.update_progress(75, "Checking semantic contradictions")
            semantic = self._detect_semantic_contradictions(triples)
            all_contradictions.extend(semantic)

        # Assess severity for all contradictions
        context.update_progress(80, "Assessing contradiction severity")
        for contradiction in all_contradictions:
            contradiction['severity'] = self._assess_severity(contradiction, triples)

        # Filter by severity threshold
        threshold_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_severity = threshold_order.get(severity_threshold, 0)
        filtered_contradictions = [
            c for c in all_contradictions
            if threshold_order.get(c['severity'], 0) >= min_severity
        ]

        # Calculate severity counts
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for c in filtered_contradictions:
            sev = c.get('severity', 'low')
            if sev in severity_counts:
                severity_counts[sev] += 1

        # Generate resolutions
        context.update_progress(90, "Generating resolution suggestions")
        resolutions = self._generate_resolutions(filtered_contradictions, 'flag_only')

        return {
            'contradictions': filtered_contradictions,
            'severity_counts': severity_counts,
            'resolutions': resolutions,
            'total_analyzed': len(triples),
            'contradictions_found': len(filtered_contradictions)
        }

    def _execute_analyze(self, triples: List[Dict], check_types: List[str],
                         context) -> Dict[str, Any]:
        """Execute analyze operation for deep contradiction analysis."""
        context.update_progress(40, "Analyzing contradictions in detail")

        # First detect contradictions
        all_contradictions = []

        if 'logical' in check_types:
            all_contradictions.extend(self._detect_logical_contradictions(triples))
        if 'temporal' in check_types:
            all_contradictions.extend(self._detect_temporal_contradictions(triples))
        if 'factual' in check_types:
            all_contradictions.extend(self._detect_factual_contradictions(triples))
        if 'semantic' in check_types:
            all_contradictions.extend(self._detect_semantic_contradictions(triples))

        # Perform deep analysis on each contradiction
        context.update_progress(60, "Performing deep analysis")
        analyzed_contradictions = []
        for contradiction in all_contradictions:
            analyzed = self._analyze_contradiction_deep(contradiction, triples)
            analyzed_contradictions.append(analyzed)

        # Sort by severity
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        analyzed_contradictions.sort(
            key=lambda x: severity_order.get(x.get('severity', 'low'), 0),
            reverse=True
        )

        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for c in analyzed_contradictions:
            sev = c.get('severity', 'low')
            if sev in severity_counts:
                severity_counts[sev] += 1

        return {
            'contradictions': analyzed_contradictions,
            'severity_counts': severity_counts,
            'resolutions': [],
            'analysis_summary': {
                'total_contradictions': len(analyzed_contradictions),
                'most_common_type': self._get_most_common_type(analyzed_contradictions),
                'most_affected_entities': self._get_most_affected_entities(analyzed_contradictions, 5),
                'confidence_impact': self._calculate_confidence_impact(analyzed_contradictions)
            }
        }

    def _execute_resolve(self, triples: List[Dict], check_types: List[str],
                         severity_threshold: str, resolution_strategy: str,
                         context) -> Dict[str, Any]:
        """Execute resolve operation to apply resolutions."""
        context.update_progress(40, "Detecting contradictions for resolution")

        # Detect all contradictions
        all_contradictions = []

        if 'logical' in check_types:
            all_contradictions.extend(self._detect_logical_contradictions(triples))
        if 'temporal' in check_types:
            all_contradictions.extend(self._detect_temporal_contradictions(triples))
        if 'factual' in check_types:
            all_contradictions.extend(self._detect_factual_contradictions(triples))
        if 'semantic' in check_types:
            all_contradictions.extend(self._detect_semantic_contradictions(triples))

        # Assess severity
        for contradiction in all_contradictions:
            contradiction['severity'] = self._assess_severity(contradiction, triples)

        # Filter by severity threshold
        threshold_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_severity = threshold_order.get(severity_threshold, 0)
        filtered_contradictions = [
            c for c in all_contradictions
            if threshold_order.get(c['severity'], 0) >= min_severity
        ]

        context.update_progress(60, f"Applying resolution strategy: {resolution_strategy}")

        # Generate and apply resolutions
        resolutions = self._generate_resolutions(filtered_contradictions, resolution_strategy)

        # Apply resolutions if not flag_only
        resolved_count = 0
        if resolution_strategy != 'flag_only':
            context.update_progress(80, "Applying resolutions")
            for resolution in resolutions:
                if self._apply_resolution(resolution, triples):
                    resolved_count += 1

        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for c in filtered_contradictions:
            sev = c.get('severity', 'low')
            if sev in severity_counts:
                severity_counts[sev] += 1

        return {
            'contradictions': filtered_contradictions,
            'severity_counts': severity_counts,
            'resolutions': resolutions,
            'resolution_summary': {
                'strategy': resolution_strategy,
                'total_resolutions': len(resolutions),
                'applied': resolved_count,
                'pending': len(resolutions) - resolved_count
            }
        }

    def _execute_report(self, triples: List[Dict], check_types: List[str],
                        severity_threshold: str, context) -> Dict[str, Any]:
        """Execute report operation to generate comprehensive report."""
        context.update_progress(40, "Generating contradiction report")

        # Detect all contradictions
        all_contradictions = []

        if 'logical' in check_types:
            all_contradictions.extend(self._detect_logical_contradictions(triples))
        if 'temporal' in check_types:
            all_contradictions.extend(self._detect_temporal_contradictions(triples))
        if 'factual' in check_types:
            all_contradictions.extend(self._detect_factual_contradictions(triples))
        if 'semantic' in check_types:
            all_contradictions.extend(self._detect_semantic_contradictions(triples))

        # Assess severity
        for contradiction in all_contradictions:
            contradiction['severity'] = self._assess_severity(contradiction, triples)

        # Filter by severity threshold
        threshold_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_severity = threshold_order.get(severity_threshold, 0)
        filtered_contradictions = [
            c for c in all_contradictions
            if threshold_order.get(c['severity'], 0) >= min_severity
        ]

        context.update_progress(70, "Building report structure")

        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for c in filtered_contradictions:
            sev = c.get('severity', 'low')
            if sev in severity_counts:
                severity_counts[sev] += 1

        report = {
            'contradictions': filtered_contradictions,
            'severity_counts': severity_counts,
            'resolutions': self._generate_resolutions(filtered_contradictions, 'flag_only'),
            'report': {
                'title': 'Contradiction Detection Report',
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_triples_analyzed': len(triples),
                    'total_contradictions_found': len(all_contradictions),
                    'filtered_contradictions': len(filtered_contradictions),
                    'contradiction_rate': len(all_contradictions) / len(triples) if triples else 0,
                    'severity_breakdown': severity_counts,
                    'type_breakdown': self._get_type_breakdown(filtered_contradictions)
                },
                'detailed_findings': self._generate_detailed_findings(filtered_contradictions),
                'recommendations': self._generate_recommendations(filtered_contradictions),
                'statistics': {
                    'most_contradicted_entities': self._get_most_affected_entities(filtered_contradictions, 10),
                    'most_common_predicates': self._get_most_common_predicates(filtered_contradictions, 10),
                    'confidence_distribution': self._get_confidence_distribution(filtered_contradictions)
                }
            }
        }

        context.update_progress(90, "Report generation complete")

        return report

    def _detect_logical_contradictions(self, triples: List[Dict]) -> List[Dict]:
        """Detect logical contradictions (A -> B and A -> not B)."""
        contradictions = []

        # Group triples by subject-predicate
        sp_groups: Dict[Tuple[str, str], List[Dict]] = {}
        for t in triples:
            key = (t.get('subject', ''), t.get('predicate', ''))
            if key not in sp_groups:
                sp_groups[key] = []
            sp_groups[key].append(t)

        # Check for direct negations in same group
        for (subject, predicate), group in sp_groups.items():
            if len(group) < 2:
                continue

            for i, t1 in enumerate(group):
                for t2 in group[i+1:]:
                    if self._are_logically_opposite(t1.get('object'), t2.get('object')):
                        contradictions.append({
                            'id': f"logical_{len(contradictions)}",
                            'type': 'logical',
                            'description': f"Logical contradiction: {subject} {predicate} both '{t1.get('object')}' and '{t2.get('object')}'",
                            'involved_triples': [t1, t2],
                            'subject': subject,
                            'predicate': predicate,
                            'values': [t1.get('object'), t2.get('object')],
                            'sources': list(set([t1.get('source', 'unknown'), t2.get('source', 'unknown')])),
                            'detection_method': 'direct_negation'
                        })

        # Check for transitive contradictions using Z3 if available
        if self.z3_solver and len(triples) > 0:
            try:
                z3_contradictions = self._detect_with_z3(triples)
                contradictions.extend(z3_contradictions)
            except Exception as e:
                self.logger.warning(f"Z3 contradiction detection failed: {e}")

        return contradictions

    def _are_logically_opposite(self, obj1: Any, obj2: Any) -> bool:
        """Check if two objects are logically opposite."""
        if obj1 == obj2:
            return False

        s1 = str(obj1).strip().lower()
        s2 = str(obj2).strip().lower()

        # Direct negation
        if s1 == f"not {s2}" or s2 == f"not {s1}":
            return True
        if s1 == f"!{s2}" or s2 == f"!{s1}":
            return True

        # Boolean opposites
        if (s1 == 'true' and s2 == 'false') or (s1 == 'false' and s2 == 'true'):
            return True

        # Numeric range contradictions
        # e.g., x > 5 and x <= 5
        num_pattern = r'([<>]=?|!=|=)\s*(-?\d+(?:\.\d+)?)'
        matches1 = re.findall(num_pattern, s1)
        matches2 = re.findall(num_pattern, s2)

        if matches1 and matches2:
            op1, val1 = matches1[0]
            op2, val2 = matches2[0]
            val1, val2 = float(val1), float(val2)

            # Same value, contradictory operators
            if val1 == val2:
                contradictory_pairs = [
                    ('>', '<='), ('<', '>='),
                    ('>=', '<'), ('<=', '>'),
                    ('=', '!='), ('!=', '=')
                ]
                if (op1, op2) in contradictory_pairs:
                    return True

        return False

    def _detect_with_z3(self, triples: List[Dict]) -> List[Dict]:
        """Use Z3 to detect complex logical contradictions."""
        contradictions = []
        # This is a placeholder for Z3 integration
        # Real implementation would convert triples to Z3 constraints
        # and check for unsatisfiable subsets
        return contradictions

    def _detect_temporal_contradictions(self, triples: List[Dict]) -> List[Dict]:
        """Detect temporal contradictions (X at T1 and X at T2 where incompatible)."""
        contradictions = []

        # Group triples by subject-predicate-object (same fact, different times)
        spo_groups: Dict[Tuple[str, str, str], List[Dict]] = {}
        for t in triples:
            key = (t.get('subject', ''), t.get('predicate', ''), t.get('object', ''))
            if key not in spo_groups:
                spo_groups[key] = []
            spo_groups[key].append(t)

        # Check for overlapping timestamps for mutually exclusive facts
        for (subject, predicate, obj), group in spo_groups.items():
            if len(group) < 2:
                continue

            # Check for time overlaps where the same fact is stated differently
            for i, t1 in enumerate(group):
                for t2 in group[i+1:]:
                    ts1 = t1.get('timestamp')
                    ts2 = t2.get('timestamp')

                    if ts1 and ts2 and ts1 != ts2:
                        # Check for contradictory timestamps
                        if self._are_temporally_incompatible(t1, t2):
                            contradictions.append({
                                'id': f"temporal_{len(contradictions)}",
                                'type': 'temporal',
                                'description': f"Temporal contradiction: {subject} {predicate} {obj} at conflicting times",
                                'involved_triples': [t1, t2],
                                'subject': subject,
                                'predicate': predicate,
                                'object': obj,
                                'timestamps': [ts1, ts2],
                                'sources': list(set([t1.get('source', 'unknown'), t2.get('source', 'unknown')])),
                                'detection_method': 'timestamp_conflict'
                            })

        # Check for event ordering contradictions
        event_triples = [t for t in triples if 'happened' in t.get('predicate', '') or
                        'occurred' in t.get('predicate', '') or
                        'started' in t.get('predicate', '') or
                        'ended' in t.get('predicate', '')]

        for i, t1 in enumerate(event_triples):
            for t2 in event_triples[i+1:]:
                if self._are_events_contradictory(t1, t2):
                    contradictions.append({
                        'id': f"temporal_event_{len(contradictions)}",
                        'type': 'temporal',
                        'description': f"Event ordering contradiction between '{t1.get('subject')}' and '{t2.get('subject')}'",
                        'involved_triples': [t1, t2],
                        'events': [t1.get('subject'), t2.get('subject')],
                        'sources': list(set([t1.get('source', 'unknown'), t2.get('source', 'unknown')])),
                        'detection_method': 'event_ordering'
                    })

        return contradictions

    def _are_temporally_incompatible(self, t1: Dict, t2: Dict) -> bool:
        """Check if two triples have temporally incompatible timestamps."""
        # Check if metadata contains validity periods
        meta1 = t1.get('metadata', {})
        meta2 = t2.get('metadata', {})

        valid_from_1 = meta1.get('valid_from')
        valid_until_1 = meta1.get('valid_until')
        valid_from_2 = meta2.get('valid_from')
        valid_until_2 = meta2.get('valid_until')

        if valid_from_1 and valid_until_2:
            # Check if t1 started after t2 ended
            try:
                from datetime import datetime
                vf1 = datetime.fromisoformat(str(valid_from_1).replace('Z', '+00:00'))
                vu2 = datetime.fromisoformat(str(valid_until_2).replace('Z', '+00:00'))
                if vf1 < vu2:
                    return True
            except:
                pass

        return False

    def _are_events_contradictory(self, t1: Dict, t2: Dict) -> bool:
        """Check if two event triples contradict each other."""
        # Simple check: same event with different timestamps
        if t1.get('subject') == t2.get('subject'):
            pred1 = t1.get('predicate', '')
            pred2 = t2.get('predicate', '')
            ts1 = t1.get('timestamp')
            ts2 = t2.get('timestamp')

            # Started vs ended at same time
            if 'started' in pred1 and 'ended' in pred2 and ts1 == ts2:
                return True
            if 'ended' in pred1 and 'started' in pred2 and ts1 == ts2:
                return True

        return False

    def _detect_factual_contradictions(self, triples: List[Dict]) -> List[Dict]:
        """Detect factual contradictions (different values for same property)."""
        contradictions = []

        # Group by subject-predicate
        sp_groups: Dict[Tuple[str, str], List[Dict]] = {}
        for t in triples:
            key = (t.get('subject', ''), t.get('predicate', ''))
            if key not in sp_groups:
                sp_groups[key] = []
            sp_groups[key].append(t)

        # Find subjects with multiple different values for same predicate
        for (subject, predicate), group in sp_groups.items():
            if len(group) < 2:
                continue

            # Group by object value
            objects: Dict[str, List[Dict]] = {}
            for t in group:
                obj = str(t.get('object', ''))
                if obj not in objects:
                    objects[obj] = []
                objects[obj].append(t)

            # If multiple different values, it's a factual contradiction
            if len(objects) > 1:
                values = list(objects.keys())
                sources = list(set(t.get('source', 'unknown') for t in group))

                contradictions.append({
                    'id': f"factual_{len(contradictions)}",
                    'type': 'factual',
                    'description': f"Factual contradiction: {subject} has multiple values for {predicate}: {values}",
                    'involved_triples': group,
                    'subject': subject,
                    'predicate': predicate,
                    'values': values,
                    'sources': sources,
                    'detection_method': 'multiple_values',
                    'value_confidences': {
                        v: max(t.get('confidence', 0) for t in objects[v])
                        for v in values
                    }
                })

        return contradictions

    def _detect_semantic_contradictions(self, triples: List[Dict]) -> List[Dict]:
        """Detect semantic contradictions (contradictory relationships)."""
        contradictions = []

        # Check for contradictory predicates
        for i, t1 in enumerate(triples):
            for t2 in triples[i+1:]:
                # Same subject and object, contradictory predicates
                if (t1.get('subject') == t2.get('subject') and
                    t1.get('object') == t2.get('object')):

                    pred1 = t1.get('predicate', '').lower()
                    pred2 = t2.get('predicate', '').lower()

                    if self._are_predicates_contradictory(pred1, pred2):
                        contradictions.append({
                            'id': f"semantic_{len(contradictions)}",
                            'type': 'semantic',
                            'description': f"Semantic contradiction: {t1.get('subject')} {pred1} and {pred2} the same {t1.get('object')}",
                            'involved_triples': [t1, t2],
                            'subject': t1.get('subject'),
                            'object': t1.get('object'),
                            'predicates': [pred1, pred2],
                            'sources': list(set([t1.get('source', 'unknown'), t2.get('source', 'unknown')])),
                            'detection_method': 'contradictory_predicates'
                        })

                # Check for antonym objects
                if (t1.get('subject') == t2.get('subject') and
                    t1.get('predicate') == t2.get('predicate')):

                    obj1 = str(t1.get('object', '')).lower()
                    obj2 = str(t2.get('object', '')).lower()

                    if self._are_antonyms(obj1, obj2):
                        contradictions.append({
                            'id': f"semantic_antonym_{len(contradictions)}",
                            'type': 'semantic',
                            'description': f"Semantic contradiction: {t1.get('subject')} {t1.get('predicate')} antonyms '{obj1}' and '{obj2}'",
                            'involved_triples': [t1, t2],
                            'subject': t1.get('subject'),
                            'predicate': t1.get('predicate'),
                            'objects': [obj1, obj2],
                            'sources': list(set([t1.get('source', 'unknown'), t2.get('source', 'unknown')])),
                            'detection_method': 'antonym_objects'
                        })

        return contradictions

    def _are_predicates_contradictory(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are semantically contradictory."""
        if pred1 == pred2:
            return False

        # Check against known contradictory pairs
        for pair in self.CONTRADICTORY_PAIRS:
            if (pred1 in pair and pred2 in pair) or (pred2 in pair and pred1 in pair):
                return True

        # Check for negation patterns
        if pred2 == f"not_{pred1}" or pred1 == f"not_{pred2}":
            return True
        if pred2 == f"{pred1}_not" or pred1 == f"{pred2}_not":
            return True

        return False

    def _are_antonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are antonyms."""
        if word1 == word2:
            return False

        # Check against known contradictory pairs
        for pair in self.CONTRADICTORY_PAIRS:
            if (word1 in pair and word2 in pair) or (word2 in pair and word1 in pair):
                return True

        # Simple negation check
        if word1 == f"not {word2}" or word2 == f"not {word1}":
            return True

        return False

    def _assess_severity(self, contradiction: Dict, all_triples: List[Dict]) -> str:
        """Assess the severity of a contradiction."""
        severity_score = 0

        # Base severity by type
        type_weights = {
            'logical': 3,
            'factual': 2,
            'temporal': 2,
            'semantic': 1
        }
        severity_score += type_weights.get(contradiction.get('type'), 1)

        # Consider confidence of involved triples
        involved = contradiction.get('involved_triples', [])
        avg_confidence = sum(t.get('confidence', 0.5) for t in involved) / len(involved) if involved else 0.5
        # Higher confidence contradictions are more severe
        severity_score += avg_confidence * 2

        # Consider number of sources (more sources = more complex = higher severity)
        sources = contradiction.get('sources', [])
        if len(sources) > 1:
            severity_score += 1

        # Consider if it involves central entities (entities with many connections)
        subject = contradiction.get('subject')
        if subject:
            connections = sum(1 for t in all_triples if t.get('subject') == subject or t.get('object') == subject)
            if connections > 10:
                severity_score += 1

        # Map score to severity level
        if severity_score >= 6:
            return 'critical'
        elif severity_score >= 4:
            return 'high'
        elif severity_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _analyze_contradiction_deep(self, contradiction: Dict, all_triples: List[Dict]) -> Dict:
        """Perform deep analysis on a contradiction."""
        analyzed = contradiction.copy()

        # Add impact analysis
        subject = contradiction.get('subject')
        if subject:
            related_triples = [t for t in all_triples
                             if t.get('subject') == subject or t.get('object') == subject]
            analyzed['impact'] = {
                'affected_triples_count': len(related_triples),
                'affected_entities': list(set(
                    [t.get('subject') for t in related_triples] +
                    [t.get('object') for t in related_triples]
                )),
                'propagation_risk': 'high' if len(related_triples) > 20 else 'medium' if len(related_triples) > 10 else 'low'
            }

        # Add confidence analysis
        involved = contradiction.get('involved_triples', [])
        confidences = [t.get('confidence', 0.5) for t in involved]
        analyzed['confidence_analysis'] = {
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'confidence_gap': max(confidences) - min(confidences) if confidences else 0
        }

        # Add source analysis
        sources = contradiction.get('sources', [])
        analyzed['source_analysis'] = {
            'source_count': len(sources),
            'sources': sources,
            'is_cross_source': len(sources) > 1
        }

        # Add suggested resolution based on confidence
        if analyzed['confidence_analysis']['confidence_gap'] > 0.3:
            analyzed['suggested_resolution'] = 'keep_highest_confidence'
        elif len(sources) > 1:
            analyzed['suggested_resolution'] = 'manual_review'
        else:
            analyzed['suggested_resolution'] = 'flag_only'

        return analyzed

    def _generate_resolutions(self, contradictions: List[Dict], strategy: str) -> List[Dict]:
        """Generate resolution suggestions for contradictions."""
        resolutions = []

        for contradiction in contradictions:
            resolution = {
                'contradiction_id': contradiction.get('id'),
                'strategy': strategy,
                'applicable': True,
                'actions': []
            }

            involved = contradiction.get('involved_triples', [])

            if strategy == 'keep_highest_confidence':
                # Find highest confidence triple
                if involved:
                    highest = max(involved, key=lambda t: t.get('confidence', 0))
                    resolution['actions'] = [
                        {
                            'action': 'keep',
                            'triple': highest,
                            'reason': f'Highest confidence ({highest.get("confidence", 0)})'
                        },
                        {
                            'action': 'deprecate',
                            'triples': [t for t in involved if t != highest],
                            'reason': 'Lower confidence than alternative'
                        }
                    ]

            elif strategy == 'keep_newest':
                # Find newest triple by timestamp
                if involved:
                    def get_timestamp(t):
                        ts = t.get('timestamp')
                        if ts:
                            try:
                                return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
                            except:
                                return datetime.min
                        return datetime.min

                    newest = max(involved, key=get_timestamp)
                    resolution['actions'] = [
                        {
                            'action': 'keep',
                            'triple': newest,
                            'reason': f'Newest timestamp ({newest.get("timestamp")})'
                        },
                        {
                            'action': 'deprecate',
                            'triples': [t for t in involved if t != newest],
                            'reason': 'Older than preferred alternative'
                        }
                    ]

            elif strategy == 'manual_review':
                resolution['actions'] = [
                    {
                        'action': 'flag_for_review',
                        'contradiction': contradiction,
                        'reason': 'Requires human judgment to resolve'
                    }
                ]

            elif strategy == 'flag_only':
                resolution['actions'] = [
                    {
                        'action': 'flag',
                        'contradiction_id': contradiction.get('id'),
                        'severity': contradiction.get('severity'),
                        'reason': 'Contradiction detected and logged'
                    }
                ]

            resolutions.append(resolution)

        return resolutions

    def _apply_resolution(self, resolution: Dict, triples: List[Dict]) -> bool:
        """Apply a resolution to the triples list."""
        try:
            for action in resolution.get('actions', []):
                action_type = action.get('action')

                if action_type == 'deprecate':
                    # Mark triples as deprecated
                    to_deprecate = action.get('triples', [])
                    for t in to_deprecate:
                        t['deprecated'] = True
                        t['deprecation_reason'] = action.get('reason', '')

                elif action_type == 'flag' or action_type == 'flag_for_review':
                    # Add flag to triples
                    target = action.get('contradiction') or action.get('triple')
                    if target:
                        target['flagged'] = True
                        target['flag_reason'] = action.get('reason', '')

            return True
        except Exception as e:
            self.logger.warning(f"Failed to apply resolution: {e}")
            return False

    def _get_most_common_type(self, contradictions: List[Dict]) -> str:
        """Get the most common contradiction type."""
        type_counts: Dict[str, int] = {}
        for c in contradictions:
            t = c.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none'

    def _get_most_affected_entities(self, contradictions: List[Dict], limit: int) -> List[Dict]:
        """Get entities most involved in contradictions."""
        entity_counts: Dict[str, int] = {}
        for c in contradictions:
            subject = c.get('subject')
            if subject:
                entity_counts[subject] = entity_counts.get(subject, 0) + 1

        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'entity': e, 'contradiction_count': c} for e, c in sorted_entities[:limit]]

    def _calculate_confidence_impact(self, contradictions: List[Dict]) -> Dict[str, float]:
        """Calculate the impact on overall confidence."""
        if not contradictions:
            return {'before': 1.0, 'after': 1.0, 'impact': 0.0}

        # Simplified calculation
        total_contradictions = len(contradictions)
        critical_count = sum(1 for c in contradictions if c.get('severity') == 'critical')
        high_count = sum(1 for c in contradictions if c.get('severity') == 'high')

        impact = (critical_count * 0.3 + high_count * 0.15 + (total_contradictions - critical_count - high_count) * 0.05) / max(total_contradictions, 1)

        return {
            'confidence_impact_score': impact,
            'critical_contradictions': critical_count,
            'high_severity_contradictions': high_count
        }

    def _get_type_breakdown(self, contradictions: List[Dict]) -> Dict[str, int]:
        """Get breakdown of contradictions by type."""
        breakdown = {'logical': 0, 'temporal': 0, 'factual': 0, 'semantic': 0}
        for c in contradictions:
            t = c.get('type')
            if t in breakdown:
                breakdown[t] += 1
        return breakdown

    def _generate_detailed_findings(self, contradictions: List[Dict]) -> List[Dict]:
        """Generate detailed findings for report."""
        findings = []
        for c in contradictions:
            finding = {
                'id': c.get('id'),
                'type': c.get('type'),
                'severity': c.get('severity'),
                'description': c.get('description'),
                'subject': c.get('subject'),
                'sources': c.get('sources', []),
                'detection_method': c.get('detection_method')
            }

            # Add type-specific details
            if c.get('type') == 'factual':
                finding['conflicting_values'] = c.get('values', [])
                finding['value_confidences'] = c.get('value_confidences', {})
            elif c.get('type') == 'temporal':
                finding['conflicting_timestamps'] = c.get('timestamps', [])

            findings.append(finding)
        return findings

    def _generate_recommendations(self, contradictions: List[Dict]) -> List[str]:
        """Generate recommendations based on contradictions."""
        recommendations = []

        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for c in contradictions:
            sev = c.get('severity', 'low')
            if sev in severity_counts:
                severity_counts[sev] += 1

        if severity_counts['critical'] > 0:
            recommendations.append(
                f"URGENT: {severity_counts['critical']} critical contradictions detected. "
                "Immediate manual review required."
            )

        if severity_counts['high'] > 5:
            recommendations.append(
                f"High number of high-severity contradictions ({severity_counts['high']}). "
                "Consider reviewing data sources for quality issues."
            )

        type_breakdown = self._get_type_breakdown(contradictions)
        if type_breakdown['logical'] > 0:
            recommendations.append(
                f"{type_breakdown['logical']} logical contradictions found. "
                "Consider using Z3 solver for formal verification."
            )

        if type_breakdown['temporal'] > 0:
            recommendations.append(
                f"{type_breakdown['temporal']} temporal contradictions found. "
                "Review timestamp metadata and validity periods."
            )

        if not recommendations:
            recommendations.append("No critical issues detected. Regular monitoring recommended.")

        return recommendations

    def _get_most_common_predicates(self, contradictions: List[Dict], limit: int) -> List[Dict]:
        """Get predicates most involved in contradictions."""
        predicate_counts: Dict[str, int] = {}
        for c in contradictions:
            pred = c.get('predicate')
            if pred:
                predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

        sorted_preds = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'predicate': p, 'contradiction_count': c} for p, c in sorted_preds[:limit]]

    def _get_confidence_distribution(self, contradictions: List[Dict]) -> Dict[str, int]:
        """Get distribution of contradictions by confidence level."""
        distribution = {'high': 0, 'medium': 0, 'low': 0}

        for c in contradictions:
            involved = c.get('involved_triples', [])
            avg_confidence = sum(t.get('confidence', 0.5) for t in involved) / len(involved) if involved else 0.5

            if avg_confidence >= 0.7:
                distribution['high'] += 1
            elif avg_confidence >= 0.4:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1

        return distribution

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns schema for UI configuration with all operation types and parameters.
        """
        return {
            "type": "object",
            "title": "Contradiction Detection Configuration",
            "description": "Configure contradiction detection and resolution parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The contradiction detection operation to perform",
                    "enum": ["detect", "analyze", "resolve", "report"],
                    "enumNames": [
                        "Detect - Find all contradictions",
                        "Analyze - Deep analysis of contradictions",
                        "Resolve - Apply resolution strategies",
                        "Report - Generate comprehensive report"
                    ],
                    "default": "detect"
                },
                "severity_threshold": {
                    "type": "string",
                    "title": "Severity Threshold",
                    "description": "Minimum severity level to include in results",
                    "enum": ["low", "medium", "high", "critical"],
                    "enumNames": [
                        "Low - Include all contradictions",
                        "Medium - Exclude low severity",
                        "High - Only high and critical",
                        "Critical - Critical only"
                    ],
                    "default": "low"
                },
                "check_types": {
                    "type": "array",
                    "title": "Contradiction Types",
                    "description": "Types of contradictions to detect",
                    "items": {
                        "type": "string",
                        "enum": ["logical", "temporal", "factual", "semantic"]
                    },
                    "default": ["logical", "temporal", "factual", "semantic"]
                },
                "entity_scope": {
                    "type": "array",
                    "title": "Entity Scope",
                    "description": "Specific entities to check (empty = all entities)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "time_range": {
                    "type": "string",
                    "title": "Time Range",
                    "description": "Temporal window for checks (ISO 8601 format, e.g., '2024-01-01/2024-12-31' or 'P30D')",
                    "default": ""
                },
                "resolution_strategy": {
                    "type": "string",
                    "title": "Resolution Strategy",
                    "description": "Strategy for resolving contradictions (used in 'resolve' operation)",
                    "enum": ["keep_highest_confidence", "keep_newest", "manual_review", "flag_only"],
                    "enumNames": [
                        "Keep Highest Confidence - Retain most confident triple",
                        "Keep Newest - Retain most recent triple",
                        "Manual Review - Flag for human review",
                        "Flag Only - Just flag without resolving"
                    ],
                    "default": "flag_only"
                },
                "knowledge_graph_id": {
                    "type": "string",
                    "title": "Knowledge Graph ID",
                    "description": "ID of knowledge graph to analyze (optional if triples provided)",
                    "default": ""
                },
                "triples": {
                    "type": "array",
                    "title": "Triples",
                    "description": "Direct triples input (optional if knowledge_graph_id provided)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "source": {"type": "string"},
                            "timestamp": {"type": "string"},
                            "metadata": {"type": "object"}
                        }
                    },
                    "default": []
                }
            },
            "required": ["operation"]
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least one knowledge source is available, False otherwise
        """
        try:
            # Check if UnifiedKGIntegrationHub is available
            kg_available = self.UnifiedKGIntegrationHub is not None

            # Node is healthy if KG hub is available or we can work with direct triples
            return True  # Can always work with direct triples input
        except Exception:
            return False
