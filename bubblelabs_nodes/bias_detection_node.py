"""
Bias Detection Node for BubbleLabs Integration

Detects and analyzes biases in knowledge representation and coverage:
- Representation biases (under/over-representation of groups)
- Association biases (stereotypical associations)
- Coverage gaps (missing knowledge for certain demographics)
- Temporal biases (knowledge skewed toward certain time periods)
- Demographic balance analysis
- Mitigation strategy generation

Features:
- Multiple bias detection algorithms
- Configurable sensitivity thresholds
- Comprehensive bias reporting
- Mitigation suggestions
- Statistical analysis fallback
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import re
from collections import defaultdict
from .base_node import BubbleLabsNode, NodeExecutionError


class BiasType(Enum):
    """Types of biases that can be detected."""
    REPRESENTATION = "representation"
    ASSOCIATION = "association"
    COVERAGE = "coverage"
    TEMPORAL = "temporal"


class SeverityLevel(Enum):
    """Severity levels for detected biases."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BiasDetectionNode(BubbleLabsNode):
    """
    Detect and analyze biases in knowledge representation and coverage.

    This node performs comprehensive bias analysis:
    - detect: Find all biases in the knowledge graph
    - analyze: Deep analysis of specific bias patterns
    - report: Generate comprehensive bias reports with visualizations
    - mitigate: Generate and apply mitigation strategies

    Supports multiple bias types including representation, association,
    coverage, and temporal biases with configurable sensitivity thresholds.
    """

    # Node metadata
    DISPLAY_NAME = "Bias Detection"
    DESCRIPTION = "Detect and analyze biases in knowledge representation and coverage"
    ICON = "bias-detection"
    CATEGORY = "intelligence"
    VERSION = "1.0.0"

    # Default stereotypical association patterns
    STEREOTYPE_PATTERNS = {
        "gender": {
            "career": ["doctor", "engineer", "nurse", "teacher"],
            "family": ["parent", "caregiver", "breadwinner"],
        },
        "race": {
            "achievement": ["academic", "athletic", "artistic"],
            "socioeconomic": ["wealthy", "poor", "educated"],
        },
        "age": {
            "capability": ["innovative", "experienced", "tech-savvy"],
            "role": ["leader", "learner", "mentor"],
        },
        "nationality": {
            "characteristics": ["hardworking", "intelligent", "creative"],
        }
    }

    # Protected attribute values for analysis
    PROTECTED_ATTRIBUTE_VALUES = {
        "gender": ["male", "female", "non-binary", "other"],
        "race": ["asian", "black", "hispanic", "white", "other"],
        "age": ["child", "young_adult", "adult", "senior"],
        "nationality": []  # Populated dynamically based on data
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports for optional dependencies
        bias_detector_module = self.safe_import(
            'knowledge_engine.bias_detection',
            fallback_value=None,
            error_msg="BiasDetector module not available"
        )

        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for BiasDetectionNode"
        )

        # Store module references
        self.BiasDetector = None
        self.UnifiedKGIntegrationHub = None
        self.UnifiedKGConfig = None

        if bias_detector_module:
            self.BiasDetector = getattr(bias_detector_module, 'BiasDetector', None)
            self.logger.info("BiasDetector module loaded successfully")

        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)
            self.UnifiedKGConfig = getattr(unified_hub_module, 'UnifiedKGConfig', None)

        # Initialize hub instance
        self.hub = None
        if self.UnifiedKGIntegrationHub and self.UnifiedKGConfig:
            try:
                config_obj = self.UnifiedKGConfig(
                    enable_deepke=True,
                    enable_oneke=True,
                    enable_kg_gen=True
                )
                self.hub = self.UnifiedKGIntegrationHub(config=config_obj)
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Initialize bias detector if available
        self.bias_detector = None
        if self.BiasDetector:
            try:
                self.bias_detector = self.BiasDetector()
                self.logger.info("BiasDetector initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize BiasDetector: {e}")
                self.bias_detector = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - detect/analyze/report/mitigate: Either 'knowledge_graph_id' or 'entities'

        Optional:
        - bias_types: List of bias types to check
        - protected_attributes: List of protected attributes to analyze
        - sensitivity_threshold: Float between 0.0 and 1.0
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'detect'))

        valid_operations = ['detect', 'analyze', 'report', 'mitigate']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Check for required knowledge source
        has_kg_id = inputs.get('knowledge_graph_id') or self.config.get('knowledge_graph_id')
        has_entities = inputs.get('entities') or self.config.get('entities')

        if not has_kg_id and not has_entities:
            errors.append("Either 'knowledge_graph_id' or 'entities' must be provided")

        # Validate entities structure if provided
        if 'entities' in inputs:
            entities = inputs['entities']
            if not isinstance(entities, list):
                errors.append("'entities' must be a list")
            elif len(entities) == 0:
                errors.append("'entities' list cannot be empty")
            else:
                for i, entity in enumerate(entities):
                    if not isinstance(entity, dict):
                        errors.append(f"Entity at index {i} must be a dictionary")
                        break

        # Validate bias_types if provided
        if 'bias_types' in inputs:
            bias_types = inputs['bias_types']
            valid_types = ['representation', 'association', 'coverage', 'temporal']
            if not isinstance(bias_types, list):
                errors.append("'bias_types' must be an array")
            else:
                for bt in bias_types:
                    if bt not in valid_types:
                        errors.append(f"Invalid bias_type: {bt}. Must be one of: {', '.join(valid_types)}")

        # Validate protected_attributes if provided
        if 'protected_attributes' in inputs:
            attrs = inputs['protected_attributes']
            valid_attrs = ['gender', 'race', 'age', 'nationality', 'religion', 'disability']
            if not isinstance(attrs, list):
                errors.append("'protected_attributes' must be an array")
            else:
                for attr in attrs:
                    if attr not in valid_attrs:
                        errors.append(f"Invalid protected_attribute: {attr}. Must be one of: {', '.join(valid_attrs)}")

        # Validate sensitivity_threshold if provided
        if 'sensitivity_threshold' in inputs:
            threshold = inputs['sensitivity_threshold']
            try:
                threshold_float = float(threshold)
                if not 0.0 <= threshold_float <= 1.0:
                    errors.append("'sensitivity_threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'sensitivity_threshold' must be a number")

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("'entity_types' must be a list of strings")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute bias detection based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - biases_detected: List of detected biases
                - severity_scores: Dict of severity scores by bias type
                - demographic_analysis: Analysis of demographic representation
                - coverage_gaps: Identified coverage gaps
                - mitigation_suggestions: List of suggested mitigation strategies
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'detect'))
        bias_types = inputs.get('bias_types', self.config.get('bias_types', ['representation', 'association', 'coverage', 'temporal']))
        protected_attributes = inputs.get('protected_attributes', self.config.get('protected_attributes', ['gender', 'race', 'age', 'nationality']))
        sensitivity_threshold = float(inputs.get('sensitivity_threshold', self.config.get('sensitivity_threshold', 0.7)))
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        comparison_baseline = inputs.get('comparison_baseline', self.config.get('comparison_baseline', 'population'))
        generate_visualizations = inputs.get('generate_visualizations', self.config.get('generate_visualizations', True))
        include_mitigation = inputs.get('include_mitigation', self.config.get('include_mitigation', True))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Executing bias {operation} with bias_types={bias_types}")

        try:
            # Retrieve entities to analyze
            context.update_progress(20, "Retrieving entities for analysis")
            entities = self._get_entities(inputs)

            if entity_types:
                entities = [e for e in entities if e.get('type') in entity_types]

            if not entities:
                return {
                    'biases_detected': [],
                    'severity_scores': {},
                    'demographic_analysis': {},
                    'coverage_gaps': [],
                    'mitigation_suggestions': [],
                    'warning': 'No entities found for analysis',
                    'metadata': {
                        'operation': operation,
                        'entities_analyzed': 0,
                        'execution_time': 0.0
                    }
                }

            context.update_progress(30, f"Analyzing {len(entities)} entities for bias")

            # Execute based on operation type
            if operation == 'detect':
                result = self._execute_detect(
                    entities, bias_types, protected_attributes,
                    sensitivity_threshold, comparison_baseline, context
                )
            elif operation == 'analyze':
                result = self._execute_analyze(
                    entities, bias_types, protected_attributes,
                    sensitivity_threshold, context
                )
            elif operation == 'report':
                result = self._execute_report(
                    entities, bias_types, protected_attributes,
                    sensitivity_threshold, comparison_baseline,
                    generate_visualizations, context
                )
            elif operation == 'mitigate':
                result = self._execute_mitigate(
                    entities, bias_types, protected_attributes,
                    sensitivity_threshold, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['detect', 'analyze', 'report', 'mitigate']}
                )

            # Add execution metadata
            result['metadata'] = {
                'operation': operation,
                'bias_types': bias_types,
                'protected_attributes': protected_attributes,
                'sensitivity_threshold': sensitivity_threshold,
                'entity_types': entity_types,
                'comparison_baseline': comparison_baseline,
                'entities_analyzed': len(entities),
                'execution_id': self.execution_id,
                'executed_at': datetime.now().isoformat()
            }

            # Store result in context
            context.add_artifact('bias_detection', {
                'operation': operation,
                'biases_found': len(result.get('biases_detected', [])),
                'severity_scores': result.get('severity_scores', {}),
                'entities_analyzed': len(entities)
            })

            status_msg = f"Found {len(result.get('biases_detected', []))} biases"
            context.update_progress(100, f"Bias {operation} complete: {status_msg}")
            self.logger.info(f"Bias detection completed: {status_msg}")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Bias detection failed: {str(e)}",
                details={
                    'operation': operation,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_entities(self, inputs: Dict) -> List[Dict[str, Any]]:
        """Retrieve entities from inputs or knowledge graph."""
        if 'entities' in inputs and inputs['entities']:
            return inputs['entities']

        if 'knowledge_graph_id' in inputs and self.hub:
            try:
                kg_id = inputs['knowledge_graph_id']
                if hasattr(self.hub, 'get_entities'):
                    return self.hub.get_entities(kg_id)
                elif hasattr(self.hub, 'entities'):
                    return [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.hub.entities]
            except Exception as e:
                self.logger.warning(f"Could not retrieve entities from hub: {e}")

        return []

    def _execute_detect(
        self,
        entities: List[Dict],
        bias_types: List[str],
        protected_attributes: List[str],
        sensitivity_threshold: float,
        comparison_baseline: str,
        context
    ) -> Dict[str, Any]:
        """Execute detect operation to find all biases."""
        context.update_progress(40, "Detecting biases")

        all_biases = []
        severity_scores = {}

        if 'representation' in bias_types:
            context.update_progress(45, "Checking representation biases")
            representation_biases = self._detect_representation_bias(
                entities, protected_attributes, sensitivity_threshold, comparison_baseline
            )
            all_biases.extend(representation_biases)
            severity_scores['representation'] = self._calculate_severity_score(representation_biases)

        if 'association' in bias_types:
            context.update_progress(55, "Checking association biases")
            association_biases = self._detect_association_bias(
                entities, protected_attributes, sensitivity_threshold
            )
            all_biases.extend(association_biases)
            severity_scores['association'] = self._calculate_severity_score(association_biases)

        if 'coverage' in bias_types:
            context.update_progress(65, "Checking coverage gaps")
            coverage_biases = self._detect_coverage_bias(
                entities, protected_attributes, sensitivity_threshold
            )
            all_biases.extend(coverage_biases)
            severity_scores['coverage'] = self._calculate_severity_score(coverage_biases)

        if 'temporal' in bias_types:
            context.update_progress(75, "Checking temporal biases")
            temporal_biases = self._detect_temporal_bias(
                entities, sensitivity_threshold
            )
            all_biases.extend(temporal_biases)
            severity_scores['temporal'] = self._calculate_severity_score(temporal_biases)

        # Calculate demographic analysis
        context.update_progress(85, "Calculating demographic analysis")
        demographic_analysis = self._calculate_demographic_analysis(entities, protected_attributes)

        # Extract coverage gaps
        coverage_gaps = [b for b in all_biases if b.get('bias_type') == 'coverage']

        # Generate mitigation suggestions
        context.update_progress(90, "Generating mitigation suggestions")
        mitigation_suggestions = self._generate_mitigation_suggestions(all_biases)

        # Assess overall severity
        overall_severity = self._assess_overall_severity(severity_scores)

        return {
            'biases_detected': all_biases,
            'severity_scores': severity_scores,
            'overall_severity': overall_severity,
            'demographic_analysis': demographic_analysis,
            'coverage_gaps': coverage_gaps,
            'mitigation_suggestions': mitigation_suggestions,
            'total_entities_analyzed': len(entities),
            'biases_found': len(all_biases)
        }

    def _execute_analyze(
        self,
        entities: List[Dict],
        bias_types: List[str],
        protected_attributes: List[str],
        sensitivity_threshold: float,
        context
    ) -> Dict[str, Any]:
        """Execute analyze operation for deep bias analysis."""
        context.update_progress(40, "Analyzing biases in detail")

        # First detect biases
        all_biases = []

        if 'representation' in bias_types:
            all_biases.extend(self._detect_representation_bias(
                entities, protected_attributes, sensitivity_threshold, 'population'
            ))
        if 'association' in bias_types:
            all_biases.extend(self._detect_association_bias(
                entities, protected_attributes, sensitivity_threshold
            ))
        if 'coverage' in bias_types:
            all_biases.extend(self._detect_coverage_bias(
                entities, protected_attributes, sensitivity_threshold
            ))
        if 'temporal' in bias_types:
            all_biases.extend(self._detect_temporal_bias(entities, sensitivity_threshold))

        # Perform deep analysis on each bias
        context.update_progress(60, "Performing deep analysis")
        analyzed_biases = []
        for bias in all_biases:
            analyzed = self._analyze_bias_deep(bias, entities)
            analyzed_biases.append(analyzed)

        # Sort by severity
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        analyzed_biases.sort(
            key=lambda x: severity_order.get(x.get('severity', 'low'), 0),
            reverse=True
        )

        severity_scores = {}
        for bias_type in bias_types:
            type_biases = [b for b in analyzed_biases if b.get('bias_type') == bias_type]
            severity_scores[bias_type] = self._calculate_severity_score(type_biases)

        return {
            'biases_detected': analyzed_biases,
            'severity_scores': severity_scores,
            'overall_severity': self._assess_overall_severity(severity_scores),
            'demographic_analysis': self._calculate_demographic_analysis(entities, protected_attributes),
            'coverage_gaps': [b for b in analyzed_biases if b.get('bias_type') == 'coverage'],
            'mitigation_suggestions': [],
            'analysis_summary': {
                'total_biases': len(analyzed_biases),
                'most_common_type': self._get_most_common_bias_type(analyzed_biases),
                'most_affected_attributes': self._get_most_affected_attributes(analyzed_biases, 5),
                'confidence_impact': self._calculate_bias_impact(analyzed_biases)
            }
        }

    def _execute_report(
        self,
        entities: List[Dict],
        bias_types: List[str],
        protected_attributes: List[str],
        sensitivity_threshold: float,
        comparison_baseline: str,
        generate_visualizations: bool,
        context
    ) -> Dict[str, Any]:
        """Execute report operation to generate comprehensive report."""
        context.update_progress(40, "Generating bias report")

        # Detect all biases
        all_biases = []

        if 'representation' in bias_types:
            all_biases.extend(self._detect_representation_bias(
                entities, protected_attributes, sensitivity_threshold, comparison_baseline
            ))
        if 'association' in bias_types:
            all_biases.extend(self._detect_association_bias(
                entities, protected_attributes, sensitivity_threshold
            ))
        if 'coverage' in bias_types:
            all_biases.extend(self._detect_coverage_bias(
                entities, protected_attributes, sensitivity_threshold
            ))
        if 'temporal' in bias_types:
            all_biases.extend(self._detect_temporal_bias(entities, sensitivity_threshold))

        context.update_progress(60, "Building report structure")

        severity_scores = {}
        for bias_type in bias_types:
            type_biases = [b for b in all_biases if b.get('bias_type') == bias_type]
            severity_scores[bias_type] = self._calculate_severity_score(type_biases)

        demographic_analysis = self._calculate_demographic_analysis(entities, protected_attributes)

        report = {
            'biases_detected': all_biases,
            'severity_scores': severity_scores,
            'overall_severity': self._assess_overall_severity(severity_scores),
            'demographic_analysis': demographic_analysis,
            'coverage_gaps': [b for b in all_biases if b.get('bias_type') == 'coverage'],
            'mitigation_suggestions': self._generate_mitigation_suggestions(all_biases),
            'report': {
                'title': 'Knowledge Bias Detection Report',
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_entities_analyzed': len(entities),
                    'total_biases_found': len(all_biases),
                    'severity_breakdown': self._get_severity_breakdown(all_biases),
                    'type_breakdown': self._get_type_breakdown(all_biases)
                },
                'demographic_distribution': demographic_analysis,
                'bias_details': self._generate_bias_details(all_biases),
                'recommendations': self._generate_recommendations(all_biases),
                'visualizations': self._generate_visualizations(all_biases, demographic_analysis) if generate_visualizations else [],
                'statistics': {
                    'most_biased_attributes': self._get_most_affected_attributes(all_biases, 10),
                    'bias_distribution_by_type': self._get_bias_distribution_by_type(all_biases),
                    'confidence_metrics': self._calculate_confidence_metrics(all_biases)
                }
            }
        }

        context.update_progress(90, "Report generation complete")

        return report

    def _execute_mitigate(
        self,
        entities: List[Dict],
        bias_types: List[str],
        protected_attributes: List[str],
        sensitivity_threshold: float,
        context
    ) -> Dict[str, Any]:
        """Execute mitigate operation to generate and apply mitigation strategies."""
        context.update_progress(40, "Detecting biases for mitigation")

        # Detect all biases
        all_biases = []

        if 'representation' in bias_types:
            all_biases.extend(self._detect_representation_bias(
                entities, protected_attributes, sensitivity_threshold, 'population'
            ))
        if 'association' in bias_types:
            all_biases.extend(self._detect_association_bias(
                entities, protected_attributes, sensitivity_threshold
            ))
        if 'coverage' in bias_types:
            all_biases.extend(self._detect_coverage_bias(
                entities, protected_attributes, sensitivity_threshold
            ))
        if 'temporal' in bias_types:
            all_biases.extend(self._detect_temporal_bias(entities, sensitivity_threshold))

        context.update_progress(60, "Generating mitigation strategies")

        mitigation_strategies = self._generate_detailed_mitigation_strategies(all_biases)

        context.update_progress(80, "Applying mitigation strategies")

        # Apply mitigations (in this implementation, we generate recommendations)
        # In a real implementation, this might modify entities or suggest specific actions
        applied_count = 0
        for strategy in mitigation_strategies:
            if strategy.get('auto_applicable', False):
                applied_count += 1

        severity_scores = {}
        for bias_type in bias_types:
            type_biases = [b for b in all_biases if b.get('bias_type') == bias_type]
            severity_scores[bias_type] = self._calculate_severity_score(type_biases)

        return {
            'biases_detected': all_biases,
            'severity_scores': severity_scores,
            'overall_severity': self._assess_overall_severity(severity_scores),
            'demographic_analysis': self._calculate_demographic_analysis(entities, protected_attributes),
            'coverage_gaps': [b for b in all_biases if b.get('bias_type') == 'coverage'],
            'mitigation_suggestions': mitigation_strategies,
            'mitigation_summary': {
                'total_strategies': len(mitigation_strategies),
                'applied': applied_count,
                'pending': len(mitigation_strategies) - applied_count,
                'estimated_impact': self._estimate_mitigation_impact(mitigation_strategies)
            }
        }

    def _detect_representation_bias(
        self,
        entities: List[Dict],
        protected_attributes: List[str],
        sensitivity_threshold: float,
        comparison_baseline: str
    ) -> List[Dict]:
        """Detect representation bias in entity distribution."""
        biases = []

        for attribute in protected_attributes:
            # Count distribution of attribute values
            distribution = self._get_attribute_distribution(entities, attribute)

            if not distribution:
                continue

            # Compare against baseline
            expected_distribution = self._get_expected_distribution(attribute, comparison_baseline)

            # Calculate representation disparity
            for value, count in distribution.items():
                expected = expected_distribution.get(value, 1.0 / len(distribution))
                actual = count / sum(distribution.values())

                disparity = abs(actual - expected)

                if disparity > (1 - sensitivity_threshold):
                    severity = self._calculate_disparity_severity(disparity, actual, expected)

                    biases.append({
                        'id': f"rep_{attribute}_{value}",
                        'bias_type': 'representation',
                        'attribute': attribute,
                        'value': value,
                        'description': f"Representation bias: {attribute}='{value}' is {actual:.1%} vs expected {expected:.1%}",
                        'actual_representation': actual,
                        'expected_representation': expected,
                        'disparity': disparity,
                        'severity': severity,
                        'affected_count': count,
                        'detection_method': 'distribution_comparison'
                    })

        return biases

    def _detect_association_bias(
        self,
        entities: List[Dict],
        protected_attributes: List[str],
        sensitivity_threshold: float
    ) -> List[Dict]:
        """Detect stereotypical associations in knowledge."""
        biases = []

        for attribute in protected_attributes:
            # Get stereotype patterns for this attribute
            patterns = self.STEREOTYPE_PATTERNS.get(attribute, {})

            for category, terms in patterns.items():
                # Check for stereotypical associations
                associations = self._find_stereotypical_associations(entities, attribute, terms)

                for association in associations:
                    confidence = association.get('confidence', 0.5)

                    if confidence > sensitivity_threshold:
                        biases.append({
                            'id': f"assoc_{attribute}_{association['term']}",
                            'bias_type': 'association',
                            'attribute': attribute,
                            'category': category,
                            'term': association['term'],
                            'description': f"Potential stereotypical association: {attribute} strongly associated with '{association['term']}'",
                            'confidence': confidence,
                            'severity': self._calculate_association_severity(confidence, sensitivity_threshold),
                            'affected_entities': association.get('entity_count', 0),
                            'detection_method': 'pattern_matching'
                        })

        # Check for co-occurrence patterns
        cooccurrence_biases = self._detect_cooccurrence_bias(entities, protected_attributes, sensitivity_threshold)
        biases.extend(cooccurrence_biases)

        return biases

    def _detect_coverage_bias(
        self,
        entities: List[Dict],
        protected_attributes: List[str],
        sensitivity_threshold: float
    ) -> List[Dict]:
        """Detect coverage gaps in knowledge."""
        biases = []

        for attribute in protected_attributes:
            # Find entities missing this attribute
            missing_count = sum(1 for e in entities if attribute not in e or not e.get(attribute))

            if len(entities) > 0:
                missing_ratio = missing_count / len(entities)

                if missing_ratio > (1 - sensitivity_threshold):
                    biases.append({
                        'id': f"coverage_{attribute}_missing",
                        'bias_type': 'coverage',
                        'attribute': attribute,
                        'description': f"Coverage gap: {missing_count} entities ({missing_ratio:.1%}) missing {attribute} information",
                        'missing_count': missing_count,
                        'missing_ratio': missing_ratio,
                        'severity': self._calculate_coverage_severity(missing_ratio),
                        'detection_method': 'missing_attribute_analysis'
                    })

            # Check for attribute value coverage
            distribution = self._get_attribute_distribution(entities, attribute)
            if distribution:
                # Find underrepresented values
                total = sum(distribution.values())
                for value, count in distribution.items():
                    ratio = count / total if total > 0 else 0
                    if ratio < 0.05:  # Less than 5% representation
                        biases.append({
                            'id': f"coverage_{attribute}_{value}_underrep",
                            'bias_type': 'coverage',
                            'attribute': attribute,
                            'value': value,
                            'description': f"Coverage gap: {attribute}='{value}' severely underrepresented ({ratio:.1%})",
                            'representation_ratio': ratio,
                            'severity': 'high' if ratio < 0.01 else 'medium',
                            'detection_method': 'underrepresentation_detection'
                        })

        return biases

    def _detect_temporal_bias(
        self,
        entities: List[Dict],
        sensitivity_threshold: float
    ) -> List[Dict]:
        """Detect temporal biases in knowledge."""
        biases = []

        # Group entities by timestamp
        temporal_distribution = defaultdict(int)
        for entity in entities:
            timestamp = entity.get('timestamp') or entity.get('created_at')
            if timestamp:
                try:
                    # Extract year from timestamp
                    if isinstance(timestamp, str):
                        year = timestamp[:4]
                        if year.isdigit():
                            temporal_distribution[year] += 1
                except:
                    pass

        if temporal_distribution:
            # Check for temporal skew
            total = sum(temporal_distribution.values())
            years = sorted(temporal_distribution.keys())

            if len(years) > 1:
                # Check if knowledge is skewed toward recent years
                recent_years = years[-3:]  # Last 3 years
                recent_count = sum(temporal_distribution[y] for y in recent_years)
                recent_ratio = recent_count / total if total > 0 else 0

                if recent_ratio > sensitivity_threshold:
                    biases.append({
                        'id': "temporal_recency_skew",
                        'bias_type': 'temporal',
                        'description': f"Temporal bias: {recent_ratio:.1%} of knowledge from recent {len(recent_years)} years",
                        'recent_ratio': recent_ratio,
                        'time_span': f"{years[0]}-{years[-1]}",
                        'severity': self._calculate_temporal_severity(recent_ratio),
                        'detection_method': 'temporal_distribution_analysis'
                    })

                # Check for decade gaps
                for i in range(len(years) - 1):
                    year_gap = int(years[i + 1]) - int(years[i])
                    if year_gap > 5:
                        biases.append({
                            'id': f"temporal_gap_{years[i]}_{years[i+1]}",
                            'bias_type': 'temporal',
                            'description': f"Temporal coverage gap: No knowledge between {years[i]} and {years[i+1]}",
                            'gap_start': years[i],
                            'gap_end': years[i+1],
                            'gap_years': year_gap,
                            'severity': 'medium' if year_gap < 10 else 'high',
                            'detection_method': 'temporal_gap_detection'
                        })

        return biases

    def _find_stereotypical_associations(
        self,
        entities: List[Dict],
        attribute: str,
        terms: List[str]
    ) -> List[Dict]:
        """Find stereotypical associations between attributes and terms."""
        associations = []

        for term in terms:
            count = 0
            for entity in entities:
                # Check if entity has the attribute and mentions the term
                if attribute in entity:
                    entity_text = str(entity.get('description', '')) + ' ' + str(entity.get('name', ''))
                    if re.search(r'\b' + re.escape(term) + r'\b', entity_text, re.IGNORECASE):
                        count += 1

            if count > 0:
                confidence = min(1.0, count / max(len(entities) * 0.1, 1))
                associations.append({
                    'term': term,
                    'entity_count': count,
                    'confidence': confidence
                })

        return associations

    def _detect_cooccurrence_bias(
        self,
        entities: List[Dict],
        protected_attributes: List[str],
        sensitivity_threshold: float
    ) -> List[Dict]:
        """Detect biased co-occurrence patterns."""
        biases = []

        # Check for attribute co-occurrence patterns
        for attr1 in protected_attributes:
            for attr2 in protected_attributes:
                if attr1 >= attr2:
                    continue

                # Build co-occurrence matrix
                cooccurrence = defaultdict(lambda: defaultdict(int))
                for entity in entities:
                    val1 = entity.get(attr1)
                    val2 = entity.get(attr2)
                    if val1 and val2:
                        cooccurrence[str(val1)][str(val2)] += 1

                # Check for skewed co-occurrence
                for val1, val2_counts in cooccurrence.items():
                    total = sum(val2_counts.values())
                    for val2, count in val2_counts.items():
                        ratio = count / total if total > 0 else 0
                        if ratio > sensitivity_threshold:
                            biases.append({
                                'id': f"cooc_{attr1}_{val1}_{attr2}_{val2}",
                                'bias_type': 'association',
                                'attribute1': attr1,
                                'value1': val1,
                                'attribute2': attr2,
                                'value2': val2,
                                'description': f"Strong co-occurrence: {attr1}='{val1}' with {attr2}='{val2}' ({ratio:.1%})",
                                'cooccurrence_ratio': ratio,
                                'count': count,
                                'severity': self._calculate_cooccurrence_severity(ratio),
                                'detection_method': 'cooccurrence_analysis'
                            })

        return biases

    def _get_attribute_distribution(self, entities: List[Dict], attribute: str) -> Dict[str, int]:
        """Get the distribution of values for a given attribute."""
        distribution = defaultdict(int)
        for entity in entities:
            value = entity.get(attribute)
            if value:
                distribution[str(value)] += 1
        return dict(distribution)

    def _get_expected_distribution(self, attribute: str, baseline: str) -> Dict[str, float]:
        """Get expected distribution for an attribute based on baseline."""
        if baseline == 'population':
            # Simplified population demographics (US-based as example)
            if attribute == 'gender':
                return {'male': 0.49, 'female': 0.51, 'non-binary': 0.001, 'other': 0.001}
            elif attribute == 'race':
                return {
                    'asian': 0.06, 'black': 0.13, 'hispanic': 0.19,
                    'white': 0.60, 'other': 0.02
                }
            elif attribute == 'age':
                return {
                    'child': 0.22, 'young_adult': 0.18, 'adult': 0.40, 'senior': 0.20
                }
            else:
                return {}
        elif baseline == 'uniform':
            values = self.PROTECTED_ATTRIBUTE_VALUES.get(attribute, [])
            if values:
                uniform = 1.0 / len(values)
                return {v: uniform for v in values}
            return {}
        else:
            return {}

    def _calculate_demographic_analysis(
        self,
        entities: List[Dict],
        protected_attributes: List[str]
    ) -> Dict[str, Any]:
        """Calculate demographic analysis for all protected attributes."""
        analysis = {}

        for attribute in protected_attributes:
            distribution = self._get_attribute_distribution(entities, attribute)
            total = sum(distribution.values())

            if total > 0:
                percentages = {k: v / total for k, v in distribution.items()}
                analysis[attribute] = {
                    'total_with_attribute': total,
                    'total_entities': len(entities),
                    'coverage': total / len(entities) if entities else 0,
                    'distribution': distribution,
                    'percentages': percentages,
                    'diversity_index': self._calculate_diversity_index(distribution),
                    'dominant_value': max(distribution.items(), key=lambda x: x[1])[0] if distribution else None
                }

        return analysis

    def _calculate_diversity_index(self, distribution: Dict[str, int]) -> float:
        """Calculate Simpson's Diversity Index."""
        total = sum(distribution.values())
        if total <= 1:
            return 0.0

        sum_proportions = sum((count / total) ** 2 for count in distribution.values())
        return 1 - sum_proportions

    def _calculate_disparity_severity(self, disparity: float, actual: float, expected: float) -> str:
        """Calculate severity based on representation disparity."""
        if disparity > 0.5:
            return 'critical'
        elif disparity > 0.3:
            return 'high'
        elif disparity > 0.15:
            return 'medium'
        else:
            return 'low'

    def _calculate_association_severity(self, confidence: float, threshold: float) -> str:
        """Calculate severity based on association confidence."""
        excess = confidence - threshold
        if excess > 0.3:
            return 'critical'
        elif excess > 0.2:
            return 'high'
        elif excess > 0.1:
            return 'medium'
        else:
            return 'low'

    def _calculate_coverage_severity(self, missing_ratio: float) -> str:
        """Calculate severity based on coverage gap."""
        if missing_ratio > 0.7:
            return 'critical'
        elif missing_ratio > 0.5:
            return 'high'
        elif missing_ratio > 0.3:
            return 'medium'
        else:
            return 'low'

    def _calculate_temporal_severity(self, recent_ratio: float) -> str:
        """Calculate severity based on temporal skew."""
        if recent_ratio > 0.9:
            return 'critical'
        elif recent_ratio > 0.8:
            return 'high'
        elif recent_ratio > 0.7:
            return 'medium'
        else:
            return 'low'

    def _calculate_cooccurrence_severity(self, ratio: float) -> str:
        """Calculate severity based on co-occurrence strength."""
        if ratio > 0.9:
            return 'critical'
        elif ratio > 0.8:
            return 'high'
        elif ratio > 0.7:
            return 'medium'
        else:
            return 'low'

    def _calculate_severity_score(self, biases: List[Dict]) -> float:
        """Calculate aggregate severity score from biases."""
        if not biases:
            return 0.0

        severity_weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        total_weight = sum(severity_weights.get(b.get('severity', 'low'), 1) for b in biases)
        max_weight = len(biases) * 4

        return round(total_weight / max_weight, 4) if max_weight > 0 else 0.0

    def _assess_overall_severity(self, severity_scores: Dict[str, float]) -> str:
        """Assess overall severity from type-specific scores."""
        if not severity_scores:
            return 'low'

        avg_score = sum(severity_scores.values()) / len(severity_scores)

        if avg_score > 0.75:
            return 'critical'
        elif avg_score > 0.5:
            return 'high'
        elif avg_score > 0.25:
            return 'medium'
        else:
            return 'low'

    def _analyze_bias_deep(self, bias: Dict, entities: List[Dict]) -> Dict:
        """Perform deep analysis on a specific bias."""
        analyzed = bias.copy()

        # Add impact assessment
        analyzed['impact_assessment'] = {
            'affected_entities_count': bias.get('affected_count', 0),
            'impact_level': bias.get('severity', 'low'),
            'potential_consequences': self._assess_consequences(bias)
        }

        # Add root cause analysis
        analyzed['root_causes'] = self._analyze_root_causes(bias, entities)

        # Add confidence metrics
        analyzed['confidence_metrics'] = {
            'detection_confidence': 0.8 if bias.get('detection_method') == 'pattern_matching' else 0.9,
            'evidence_strength': 'strong' if bias.get('severity') in ['high', 'critical'] else 'moderate'
        }

        return analyzed

    def _assess_consequences(self, bias: Dict) -> List[str]:
        """Assess potential consequences of a bias."""
        consequences = []

        severity = bias.get('severity', 'low')
        bias_type = bias.get('bias_type', '')

        if bias_type == 'representation':
            consequences.append("May lead to skewed decision-making based on underrepresented groups")
            if severity in ['high', 'critical']:
                consequences.append("Risk of perpetuating systemic inequalities")

        elif bias_type == 'association':
            consequences.append("May reinforce harmful stereotypes")
            if severity in ['high', 'critical']:
                consequences.append("Potential for discriminatory outcomes")

        elif bias_type == 'coverage':
            consequences.append("Incomplete understanding due to missing data")
            if severity in ['high', 'critical']:
                consequences.append("Critical blind spots in knowledge base")

        elif bias_type == 'temporal':
            consequences.append("Outdated or overly recent perspectives")

        return consequences

    def _analyze_root_causes(self, bias: Dict, entities: List[Dict]) -> List[str]:
        """Analyze root causes of a bias."""
        causes = []

        bias_type = bias.get('bias_type', '')

        if bias_type == 'representation':
            causes.append("Sampling bias in data collection")
            causes.append("Historical imbalances in source data")

        elif bias_type == 'association':
            causes.append("Societal stereotypes reflected in training data")
            causes.append("Co-occurrence patterns in source texts")

        elif bias_type == 'coverage':
            causes.append("Data collection gaps")
            causes.append("Privacy constraints limiting attribute collection")

        elif bias_type == 'temporal':
            causes.append("Availability of recent digital data")
            causes.append("Focus on current events in data sources")

        return causes

    def _generate_mitigation_suggestions(self, biases: List[Dict]) -> List[Dict]:
        """Generate mitigation suggestions for detected biases."""
        suggestions = []

        for bias in biases:
            bias_type = bias.get('bias_type', '')
            severity = bias.get('severity', 'low')

            if bias_type == 'representation':
                suggestions.append({
                    'target_bias_id': bias.get('id'),
                    'strategy': 'augment_underrepresented',
                    'description': f"Actively collect more data for underrepresented {bias.get('attribute')}={bias.get('value')}",
                    'priority': 'high' if severity in ['critical', 'high'] else 'medium',
                    'estimated_effort': 'medium',
                    'expected_impact': f"Reduce disparity from {bias.get('disparity', 0):.1%} to <10%"
                })

            elif bias_type == 'association':
                suggestions.append({
                    'target_bias_id': bias.get('id'),
                    'strategy': 'rebalance_associations',
                    'description': f"Review and rebalance associations between {bias.get('attribute')} and {bias.get('term', 'related terms')}",
                    'priority': 'high' if severity in ['critical', 'high'] else 'medium',
                    'estimated_effort': 'high',
                    'expected_impact': 'Reduce stereotypical associations by 50%'
                })

            elif bias_type == 'coverage':
                suggestions.append({
                    'target_bias_id': bias.get('id'),
                    'strategy': 'fill_coverage_gaps',
                    'description': f"Implement mandatory collection of {bias.get('attribute')} information",
                    'priority': 'high',
                    'estimated_effort': 'low',
                    'expected_impact': f"Reduce missing data from {bias.get('missing_ratio', 0):.1%} to <5%"
                })

            elif bias_type == 'temporal':
                suggestions.append({
                    'target_bias_id': bias.get('id'),
                    'strategy': 'temporal_balancing',
                    'description': 'Collect historical data to balance temporal distribution',
                    'priority': 'medium',
                    'estimated_effort': 'high',
                    'expected_impact': 'Achieve more uniform temporal coverage'
                })

        return suggestions

    def _generate_detailed_mitigation_strategies(self, biases: List[Dict]) -> List[Dict]:
        """Generate detailed mitigation strategies with action plans."""
        strategies = []

        # Group biases by type for consolidated strategies
        by_type = defaultdict(list)
        for bias in biases:
            by_type[bias.get('bias_type')].append(bias)

        for bias_type, type_biases in by_type.items():
            if bias_type == 'representation':
                strategies.append({
                    'strategy_id': f"mitigate_{bias_type}",
                    'bias_type': bias_type,
                    'title': 'Representation Balancing Strategy',
                    'description': f'Address underrepresentation in {len(type_biases)} demographic categories',
                    'actions': [
                        'Identify and partner with diverse data sources',
                        'Implement stratified sampling for data collection',
                        'Set representation targets based on population demographics',
                        'Regular monitoring and reporting of representation metrics'
                    ],
                    'auto_applicable': False,
                    'estimated_timeline': '3-6 months',
                    'success_criteria': 'Representation within 10% of target demographics'
                })

            elif bias_type == 'association':
                strategies.append({
                    'strategy_id': f"mitigate_{bias_type}",
                    'bias_type': bias_type,
                    'title': 'Stereotype Mitigation Strategy',
                    'description': f'Address {len(type_biases)} stereotypical association patterns',
                    'actions': [
                        'Review and flag stereotypical training examples',
                        'Augment with counter-stereotypical examples',
                        'Implement debiasing techniques in embeddings',
                        'Human review of high-confidence associations'
                    ],
                    'auto_applicable': False,
                    'estimated_timeline': '6-12 months',
                    'success_criteria': 'Association confidence reduced to baseline levels'
                })

            elif bias_type == 'coverage':
                strategies.append({
                    'strategy_id': f"mitigate_{bias_type}",
                    'bias_type': bias_type,
                    'title': 'Coverage Gap Remediation Strategy',
                    'description': f'Fill {len(type_biases)} identified coverage gaps',
                    'actions': [
                        'Audit data collection processes',
                        'Implement mandatory field validation',
                        'Backfill missing attribute data where possible',
                        'Establish data quality SLAs'
                    ],
                    'auto_applicable': True,
                    'estimated_timeline': '1-3 months',
                    'success_criteria': '>95% attribute coverage across all entities'
                })

            elif bias_type == 'temporal':
                strategies.append({
                    'strategy_id': f"mitigate_{bias_type}",
                    'bias_type': bias_type,
                    'title': 'Temporal Balance Strategy',
                    'description': f'Address temporal skew across {len(type_biases)} time periods',
                    'actions': [
                        'Acquire historical archives and datasets',
                        'Weight training data by temporal distribution',
                        'Implement time-aware validation splits',
                        'Monitor for temporal drift'
                    ],
                    'auto_applicable': False,
                    'estimated_timeline': '6-12 months',
                    'success_criteria': 'Uniform distribution within 20% across decades'
                })

        return strategies

    def _estimate_mitigation_impact(self, strategies: List[Dict]) -> Dict[str, Any]:
        """Estimate the impact of mitigation strategies."""
        total_strategies = len(strategies)
        auto_applicable = sum(1 for s in strategies if s.get('auto_applicable', False))

        return {
            'total_strategies': total_strategies,
            'auto_applicable': auto_applicable,
            'requires_manual_intervention': total_strategies - auto_applicable,
            'estimated_overall_improvement': f"{min(90, len(strategies) * 15)}%",
            'priority_order': [s['strategy_id'] for s in sorted(
                strategies,
                key=lambda x: 0 if x.get('auto_applicable') else 1
            )]
        }

    def _get_most_common_bias_type(self, biases: List[Dict]) -> str:
        """Get the most common bias type."""
        if not biases:
            return 'none'

        type_counts = defaultdict(int)
        for bias in biases:
            type_counts[bias.get('bias_type', 'unknown')] += 1

        return max(type_counts.items(), key=lambda x: x[1])[0]

    def _get_most_affected_attributes(self, biases: List[Dict], limit: int) -> List[Dict]:
        """Get the most affected protected attributes."""
        attribute_impact = defaultdict(lambda: {'count': 0, 'severity_sum': 0})
        severity_weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}

        for bias in biases:
            attr = bias.get('attribute', 'unknown')
            attribute_impact[attr]['count'] += 1
            attribute_impact[attr]['severity_sum'] += severity_weights.get(
                bias.get('severity', 'low'), 1
            )

        sorted_attrs = sorted(
            attribute_impact.items(),
            key=lambda x: x[1]['severity_sum'],
            reverse=True
        )

        return [
            {'attribute': attr, 'bias_count': data['count'], 'impact_score': data['severity_sum']}
            for attr, data in sorted_attrs[:limit]
        ]

    def _calculate_bias_impact(self, biases: List[Dict]) -> Dict[str, float]:
        """Calculate the impact of biases on knowledge quality."""
        if not biases:
            return {'overall_impact': 0.0, 'confidence_reduction': 0.0}

        severity_weights = {'low': 0.1, 'medium': 0.2, 'high': 0.4, 'critical': 0.6}
        total_weight = sum(severity_weights.get(b.get('severity', 'low'), 0.1) for b in biases)

        return {
            'overall_impact': min(1.0, total_weight / max(len(biases), 1)),
            'confidence_reduction': min(0.5, total_weight / max(len(biases) * 2, 1)),
            'reliability_score': max(0.0, 1.0 - (total_weight / max(len(biases) * 3, 1)))
        }

    def _get_severity_breakdown(self, biases: List[Dict]) -> Dict[str, int]:
        """Get breakdown of biases by severity."""
        breakdown = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for bias in biases:
            sev = bias.get('severity', 'low')
            if sev in breakdown:
                breakdown[sev] += 1
        return breakdown

    def _get_type_breakdown(self, biases: List[Dict]) -> Dict[str, int]:
        """Get breakdown of biases by type."""
        breakdown = defaultdict(int)
        for bias in biases:
            breakdown[bias.get('bias_type', 'unknown')] += 1
        return dict(breakdown)

    def _generate_bias_details(self, biases: List[Dict]) -> List[Dict]:
        """Generate detailed descriptions for each bias."""
        return [
            {
                'id': bias.get('id'),
                'type': bias.get('bias_type'),
                'description': bias.get('description'),
                'severity': bias.get('severity'),
                'affected_attribute': bias.get('attribute'),
                'detection_method': bias.get('detection_method'),
                'metrics': {k: v for k, v in bias.items() if k not in [
                    'id', 'bias_type', 'description', 'severity', 'attribute',
                    'detection_method'
                ]}
            }
            for bias in biases
        ]

    def _generate_recommendations(self, biases: List[Dict]) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []

        severity_breakdown = self._get_severity_breakdown(biases)
        type_breakdown = self._get_type_breakdown(biases)

        if severity_breakdown['critical'] > 0:
            recommendations.append(
                f"URGENT: Address {severity_breakdown['critical']} critical bias issues immediately"
            )

        if type_breakdown.get('representation', 0) > 3:
            recommendations.append(
                "Implement diverse data sourcing to address representation imbalances"
            )

        if type_breakdown.get('association', 0) > 0:
            recommendations.append(
                "Review and audit training data for stereotypical patterns"
            )

        if type_breakdown.get('coverage', 0) > 0:
            recommendations.append(
                "Improve data collection processes to fill coverage gaps"
            )

        if type_breakdown.get('temporal', 0) > 0:
            recommendations.append(
                "Acquire historical data to balance temporal distribution"
            )

        recommendations.append(
            "Establish ongoing bias monitoring and reporting processes"
        )

        return recommendations

    def _generate_visualizations(
        self,
        biases: List[Dict],
        demographic_analysis: Dict
    ) -> List[Dict]:
        """Generate visualization specifications."""
        visualizations = []

        # Demographic distribution charts
        for attribute, analysis in demographic_analysis.items():
            visualizations.append({
                'type': 'pie_chart',
                'title': f'{attribute.title()} Distribution',
                'data': analysis.get('distribution', {}),
                'description': f'Shows distribution of {attribute} values'
            })

        # Bias severity chart
        severity_breakdown = self._get_severity_breakdown(biases)
        visualizations.append({
            'type': 'bar_chart',
            'title': 'Bias Severity Distribution',
            'data': severity_breakdown,
            'description': 'Shows count of biases by severity level'
        })

        # Bias type chart
        type_breakdown = self._get_type_breakdown(biases)
        visualizations.append({
            'type': 'bar_chart',
            'title': 'Bias Type Distribution',
            'data': type_breakdown,
            'description': 'Shows count of biases by type'
        })

        # Timeline for temporal biases
        temporal_biases = [b for b in biases if b.get('bias_type') == 'temporal']
        if temporal_biases:
            visualizations.append({
                'type': 'timeline',
                'title': 'Temporal Bias Timeline',
                'data': {
                    b.get('id'): {'start': b.get('gap_start'), 'end': b.get('gap_end')}
                    for b in temporal_biases if 'gap_start' in b
                },
                'description': 'Shows temporal coverage gaps'
            })

        return visualizations

    def _get_bias_distribution_by_type(self, biases: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Get distribution of biases by type and severity."""
        distribution = defaultdict(lambda: defaultdict(int))
        for bias in biases:
            bias_type = bias.get('bias_type', 'unknown')
            severity = bias.get('severity', 'low')
            distribution[bias_type][severity] += 1
        return {k: dict(v) for k, v in distribution.items()}

    def _calculate_confidence_metrics(self, biases: List[Dict]) -> Dict[str, float]:
        """Calculate confidence metrics for bias detection."""
        if not biases:
            return {'average_confidence': 0.0, 'high_confidence_ratio': 0.0}

        confidences = []
        high_confidence_count = 0

        for bias in biases:
            # Estimate confidence based on detection method and severity
            if bias.get('detection_method') == 'distribution_comparison':
                confidence = 0.9
            elif bias.get('detection_method') == 'pattern_matching':
                confidence = 0.75
            elif bias.get('detection_method') == 'cooccurrence_analysis':
                confidence = 0.85
            else:
                confidence = 0.8

            if bias.get('severity') in ['high', 'critical']:
                confidence = min(1.0, confidence + 0.1)

            confidences.append(confidence)
            if confidence > 0.8:
                high_confidence_count += 1

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            'average_confidence': round(avg_confidence, 4),
            'high_confidence_ratio': round(high_confidence_count / len(biases), 4),
            'low_confidence_count': sum(1 for c in confidences if c < 0.7)
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Bias Detection Configuration",
            "description": "Configure bias detection parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of bias operation to perform",
                    "enum": ["detect", "analyze", "report", "mitigate"],
                    "enumNames": [
                        "Detect - Find all biases in knowledge",
                        "Analyze - Deep analysis of bias patterns",
                        "Report - Generate comprehensive bias report",
                        "Mitigate - Generate mitigation strategies"
                    ],
                    "default": "detect"
                },
                "bias_types": {
                    "type": "array",
                    "title": "Bias Types",
                    "description": "Types of biases to detect",
                    "items": {
                        "type": "string",
                        "enum": ["representation", "association", "coverage", "temporal"]
                    },
                    "default": ["representation", "association", "coverage", "temporal"]
                },
                "protected_attributes": {
                    "type": "array",
                    "title": "Protected Attributes",
                    "description": "Demographic attributes to analyze for bias",
                    "items": {
                        "type": "string",
                        "enum": ["gender", "race", "age", "nationality", "religion", "disability"]
                    },
                    "default": ["gender", "race", "age", "nationality"]
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Limit analysis to specific entity types (empty = all types)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "sensitivity_threshold": {
                    "type": "number",
                    "title": "Sensitivity Threshold",
                    "description": "Threshold for bias detection sensitivity (0.0-1.0). Higher values detect more subtle biases.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                },
                "comparison_baseline": {
                    "type": "string",
                    "title": "Comparison Baseline",
                    "description": "Baseline for comparing representation",
                    "enum": ["population", "uniform", "historical"],
                    "enumNames": [
                        "Population - Compare to real-world demographics",
                        "Uniform - Compare to uniform distribution",
                        "Historical - Compare to historical data"
                    ],
                    "default": "population"
                },
                "generate_visualizations": {
                    "type": "boolean",
                    "title": "Generate Visualizations",
                    "description": "Generate visualization specifications for reports",
                    "default": True
                },
                "include_mitigation": {
                    "type": "boolean",
                    "title": "Include Mitigation Suggestions",
                    "description": "Include mitigation strategies in output",
                    "default": True
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (can run analysis without external dependencies)
        """
        try:
            # Node can work without external dependencies (has internal analysis logic)
            return True
        except Exception:
            return False

    def get_supported_bias_types(self) -> List[str]:
        """
        Get list of supported bias types.

        Returns:
            List of bias type names
        """
        return [bt.value for bt in BiasType]

    def get_supported_operations(self) -> List[str]:
        """
        Get list of supported operations.

        Returns:
            List of operation names
        """
        return ['detect', 'analyze', 'report', 'mitigate']
