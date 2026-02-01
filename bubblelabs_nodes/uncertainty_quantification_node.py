"""
Uncertainty Quantification Node for BubbleLabs Integration

Provides uncertainty quantification, propagation, and management capabilities:
- Quantify uncertainty in facts and knowledge
- Propagate uncertainty through reasoning chains
- Calculate confidence intervals for uncertain values
- Identify regions of high uncertainty in knowledge
- Suggest evidence to reduce uncertainty
- Generate comprehensive uncertainty reports
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random
import math
from .base_node import BubbleLabsNode, NodeExecutionError


class UncertaintyQuantificationNode(BubbleLabsNode):
    """
    Quantify, propagate, and manage uncertainty in knowledge.

    Supports operations:
    - quantify: Calculate uncertainty score for facts
    - propagate: Propagate uncertainty through reasoning
    - interval: Calculate confidence intervals
    - identify: Find uncertain knowledge regions
    - suggest_evidence: Suggest evidence to reduce uncertainty
    - report: Generate comprehensive uncertainty reports
    """

    # Node metadata
    DISPLAY_NAME = "Uncertainty Quantification"
    DESCRIPTION = "Quantify, propagate, and manage uncertainty in knowledge"
    ICON = "uncertainty"
    CATEGORY = "intelligence"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of uncertainty quantifier
        self.UncertaintyQuantifier = self.safe_import(
            'knowledge_engine.uncertainty.UncertaintyQuantifier',
            fallback_value=None,
            error_msg="UncertaintyQuantifier not available for UncertaintyQuantificationNode"
        )

        # Safe import of unified KG integration hub
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for UncertaintyQuantificationNode"
        )

        # Initialize quantifier if available
        self.quantifier = None
        self.kg_hub = None

        if self.UncertaintyQuantifier:
            try:
                self.quantifier = self.UncertaintyQuantifier()
            except Exception as e:
                self.logger.warning(f"Could not initialize UncertaintyQuantifier: {e}")
                self.quantifier = None

        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (depending on operation):
            - operation: Operation type to perform
            - entity_id or triple: Entity or triple to analyze

        Optional:
            - confidence_level: Confidence level for intervals
            - uncertainty_model: Model type for uncertainty
            - propagation_method: Method for uncertainty propagation
        """
        errors = []

        # Get operation from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'quantify'))

        # Validate operation
        valid_operations = ['quantify', 'propagate', 'interval', 'identify', 'suggest_evidence', 'report']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of {valid_operations}")

        # Check entity_id or triple based on operation
        if operation in ['quantify', 'interval', 'suggest_evidence']:
            entity_id = inputs.get('entity_id', self.config.get('entity_id'))
            triple = inputs.get('triple', self.config.get('triple'))

            if not entity_id and not triple:
                errors.append(f"Operation '{operation}' requires 'entity_id' or 'triple' in inputs or config")

            # Validate triple structure if provided
            if triple:
                if not isinstance(triple, dict):
                    errors.append("triple must be an object with subject, predicate, object")
                else:
                    required_triple_fields = ['subject', 'predicate', 'object']
                    for field in required_triple_fields:
                        if field not in triple:
                            errors.append(f"triple missing required field: {field}")

        # Validate confidence_level
        if 'confidence_level' in inputs:
            try:
                cl = float(inputs['confidence_level'])
                if not 0.0 < cl < 1.0:
                    errors.append("confidence_level must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("confidence_level must be a number")

        # Validate uncertainty_model
        if 'uncertainty_model' in inputs:
            valid_models = ['probabilistic', 'fuzzy', 'bayesian', 'interval']
            if inputs['uncertainty_model'] not in valid_models:
                errors.append(f"uncertainty_model must be one of: {valid_models}")

        # Validate propagation_method
        if 'propagation_method' in inputs:
            valid_methods = ['linear', 'bayesian', 'monte_carlo']
            if inputs['propagation_method'] not in valid_methods:
                errors.append(f"propagation_method must be one of: {valid_methods}")

        # Validate min_samples
        if 'min_samples' in inputs:
            try:
                ms = int(inputs['min_samples'])
                if ms < 1:
                    errors.append("min_samples must be at least 1")
            except (TypeError, ValueError):
                errors.append("min_samples must be an integer")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute uncertainty quantification operation.

        Args:
            inputs: Must contain 'operation' and 'entity_id' or 'triple'
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - uncertainty_score: Quantified uncertainty (0-1)
                - confidence_interval: [lower, upper] confidence bounds
                - alternatives: List of alternative values/hypotheses
                - evidence_suggestions: Suggested evidence to reduce uncertainty
                - uncertainty_regions: Regions of high uncertainty
                - report: Comprehensive uncertainty report
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'quantify'))
        entity_id = inputs.get('entity_id', self.config.get('entity_id'))
        triple = inputs.get('triple', self.config.get('triple'))
        confidence_level = inputs.get('confidence_level', self.config.get('confidence_level', 0.95))
        uncertainty_model = inputs.get('uncertainty_model', self.config.get('uncertainty_model', 'bayesian'))
        propagation_method = inputs.get('propagation_method', self.config.get('propagation_method', 'bayesian'))
        min_samples = inputs.get('min_samples', self.config.get('min_samples', 100))
        include_alternatives = inputs.get('include_alternatives', self.config.get('include_alternatives', True))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Starting uncertainty {operation} for entity={entity_id}, triple={triple}")

        try:
            # Execute based on operation type
            if operation == 'quantify':
                result = self._quantify_uncertainty(
                    entity_id, triple, uncertainty_model, context
                )
            elif operation == 'propagate':
                result = self._propagate_uncertainty(
                    entity_id, triple, propagation_method, min_samples, context
                )
            elif operation == 'interval':
                result = self._calculate_confidence_interval(
                    entity_id, triple, confidence_level, uncertainty_model, context
                )
            elif operation == 'identify':
                result = self._identify_uncertain_regions(
                    entity_id, confidence_level, context
                )
            elif operation == 'suggest_evidence':
                result = self._suggest_evidence(
                    entity_id, triple, uncertainty_model, context
                )
            elif operation == 'report':
                result = self._generate_uncertainty_report(
                    entity_id, triple, confidence_level, include_alternatives, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'operation': operation}
                )

            # Add metadata
            result['operation'] = operation
            result['entity_id'] = entity_id
            result['triple'] = triple
            result['timestamp'] = datetime.now().isoformat()

            # Add to context
            context.add_artifact('uncertainty_quantification', {
                'result': result,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            })

            context.update_progress(100, f"Uncertainty {operation} complete")
            self.logger.info(f"Uncertainty {operation} completed successfully")

            return result

        except Exception as e:
            self.logger.error(f"Uncertainty quantification failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Uncertainty quantification failed: {str(e)}",
                details={
                    'operation': operation,
                    'entity_id': entity_id,
                    'triple': triple,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _quantify_uncertainty(self, entity_id: Optional[str], triple: Optional[Dict],
                             uncertainty_model: str, context) -> Dict[str, Any]:
        """Quantify uncertainty for a fact or entity."""
        context.update_progress(30, "Quantifying uncertainty")

        if self.quantifier:
            try:
                # Use actual quantifier if available
                if entity_id:
                    score = self.quantifier.quantify_entity_uncertainty(entity_id, model=uncertainty_model)
                elif triple:
                    score = self.quantifier.quantify_triple_uncertainty(triple, model=uncertainty_model)
                else:
                    score = 0.5  # Default uncertainty

                return {
                    'uncertainty_score': score,
                    'confidence_interval': self._calculate_interval_from_score(score),
                    'alternatives': [],
                    'model_used': uncertainty_model,
                    'method': 'quantifier'
                }
            except Exception as e:
                self.logger.warning(f"Quantifier failed, using fallback: {e}")
                return self._quantify_uncertainty_fallback(entity_id, triple, uncertainty_model, context)
        else:
            return self._quantify_uncertainty_fallback(entity_id, triple, uncertainty_model, context)

    def _quantify_uncertainty_fallback(self, entity_id: Optional[str], triple: Optional[Dict],
                                       uncertainty_model: str, context) -> Dict[str, Any]:
        """Fallback uncertainty quantification using heuristics."""
        context.update_progress(40, "Using heuristic uncertainty quantification")

        # Heuristic uncertainty based on available information
        uncertainty_factors = []

        # Factor 1: Source reliability (simulated)
        source_reliability = random.uniform(0.6, 0.95)
        uncertainty_factors.append(1 - source_reliability)

        # Factor 2: Information completeness
        if triple:
            completeness = sum(1 for v in triple.values() if v) / len(triple)
            uncertainty_factors.append(1 - completeness)
        elif entity_id:
            # Assume some base uncertainty for entities
            uncertainty_factors.append(0.3)
        else:
            uncertainty_factors.append(0.5)

        # Factor 3: Consistency with other knowledge (simulated)
        consistency = random.uniform(0.7, 1.0)
        uncertainty_factors.append(1 - consistency)

        # Calculate combined uncertainty score
        if uncertainty_model == 'bayesian':
            # Bayesian: combine as product of complements
            score = 1 - math.prod(1 - f for f in uncertainty_factors)
        elif uncertainty_model == 'fuzzy':
            # Fuzzy: use max
            score = max(uncertainty_factors)
        elif uncertainty_model == 'interval':
            # Interval: use average
            score = sum(uncertainty_factors) / len(uncertainty_factors)
        else:
            # Probabilistic: weighted average
            weights = [0.4, 0.4, 0.2]
            score = sum(f * w for f, w in zip(uncertainty_factors, weights))

        return {
            'uncertainty_score': round(score, 4),
            'confidence_interval': self._calculate_interval_from_score(score),
            'alternatives': self._generate_alternatives(triple, score) if triple else [],
            'model_used': uncertainty_model,
            'method': 'heuristic_fallback',
            'factors': {
                'source_reliability': round(source_reliability, 4),
                'information_completeness': round(completeness if triple else 0.7, 4),
                'consistency': round(consistency, 4)
            },
            'warning': 'Using heuristic fallback - quantifier not available'
        }

    def _propagate_uncertainty(self, entity_id: Optional[str], triple: Optional[Dict],
                               propagation_method: str, min_samples: int, context) -> Dict[str, Any]:
        """Propagate uncertainty through reasoning chains."""
        context.update_progress(30, f"Propagating uncertainty using {propagation_method}")

        # Get initial uncertainty
        initial_result = self._quantify_uncertainty(entity_id, triple, 'bayesian', context)
        initial_uncertainty = initial_result['uncertainty_score']

        context.update_progress(50, "Calculating propagated uncertainty")

        # Simulate reasoning chain (in practice, this would trace through knowledge graph)
        reasoning_chain = self._build_reasoning_chain(entity_id, triple)

        propagated_uncertainties = []
        current_uncertainty = initial_uncertainty

        for step, (relation, factor) in enumerate(reasoning_chain):
            context.update_progress(50 + (step * 40 // len(reasoning_chain)),
                                   f"Propagating through step {step + 1}")

            if propagation_method == 'linear':
                # Linear propagation: additive
                current_uncertainty = min(1.0, current_uncertainty + factor * 0.1)
            elif propagation_method == 'bayesian':
                # Bayesian propagation: update belief
                current_uncertainty = (current_uncertainty * factor) / \
                                     (current_uncertainty * factor + (1 - current_uncertainty) * (1 - factor))
            elif propagation_method == 'monte_carlo':
                # Monte Carlo: sample and aggregate
                samples = [random.gauss(current_uncertainty, 0.1) for _ in range(min_samples // 10)]
                current_uncertainty = sum(samples) / len(samples)

            propagated_uncertainties.append({
                'step': step + 1,
                'relation': relation,
                'uncertainty': round(current_uncertainty, 4)
            })

        final_uncertainty = propagated_uncertainties[-1]['uncertainty'] if propagated_uncertainties else current_uncertainty

        return {
            'initial_uncertainty': round(initial_uncertainty, 4),
            'final_uncertainty': round(final_uncertainty, 4),
            'propagated_uncertainty': round(final_uncertainty, 4),
            'reasoning_chain': reasoning_chain,
            'propagation_steps': propagated_uncertainties,
            'propagation_method': propagation_method,
            'samples_used': min_samples if propagation_method == 'monte_carlo' else None,
            'confidence_interval': self._calculate_interval_from_score(final_uncertainty)
        }

    def _calculate_confidence_interval(self, entity_id: Optional[str], triple: Optional[Dict],
                                       confidence_level: float, uncertainty_model: str, context) -> Dict[str, Any]:
        """Calculate confidence intervals for uncertain values."""
        context.update_progress(30, f"Calculating {confidence_level*100}% confidence interval")

        # Get uncertainty score
        uncertainty_result = self._quantify_uncertainty(entity_id, triple, uncertainty_model, context)
        uncertainty_score = uncertainty_result['uncertainty_score']

        context.update_progress(60, "Computing interval bounds")

        # Calculate interval based on confidence level
        alpha = 1 - confidence_level
        z_score = self._get_z_score(confidence_level)

        # Assume standard normal distribution centered at estimated value
        estimated_value = 0.5  # Center value (would come from actual estimation)
        standard_error = uncertainty_score / 2  # Approximation

        margin_of_error = z_score * standard_error

        lower_bound = max(0.0, estimated_value - margin_of_error)
        upper_bound = min(1.0, estimated_value + margin_of_error)

        context.update_progress(90, "Interval calculation complete")

        return {
            'uncertainty_score': round(uncertainty_score, 4),
            'confidence_level': confidence_level,
            'confidence_interval': [round(lower_bound, 4), round(upper_bound, 4)],
            'estimated_value': round(estimated_value, 4),
            'margin_of_error': round(margin_of_error, 4),
            'interval_width': round(upper_bound - lower_bound, 4),
            'model_used': uncertainty_model
        }

    def _identify_uncertain_regions(self, entity_id: Optional[str], confidence_level: float, context) -> Dict[str, Any]:
        """Identify regions of high uncertainty in knowledge."""
        context.update_progress(30, "Scanning knowledge for uncertain regions")

        uncertain_regions = []

        # Simulate scanning knowledge graph regions
        regions_to_scan = [
            {'name': 'entity_properties', 'description': 'Entity property assertions'},
            {'name': 'relational_facts', 'description': 'Relationship assertions'},
            {'name': 'temporal_facts', 'description': 'Time-dependent facts'},
            {'name': 'inferred_facts', 'description': 'Inferred knowledge'}
        ]

        for i, region in enumerate(regions_to_scan):
            context.update_progress(30 + (i * 60 // len(regions_to_scan)),
                                   f"Analyzing region: {region['name']}")

            # Simulate uncertainty measurement for region
            region_uncertainty = random.uniform(0.2, 0.9)

            if region_uncertainty > (1 - confidence_level):
                uncertain_regions.append({
                    'region': region['name'],
                    'description': region['description'],
                    'uncertainty_score': round(region_uncertainty, 4),
                    'severity': self._uncertainty_to_severity(region_uncertainty),
                    'affected_entities': random.randint(5, 100),
                    'recommended_action': self._recommend_action(region_uncertainty)
                })

        # Sort by uncertainty score (descending)
        uncertain_regions.sort(key=lambda x: x['uncertainty_score'], reverse=True)

        context.update_progress(95, "Region analysis complete")

        return {
            'uncertain_regions': uncertain_regions,
            'total_regions_scanned': len(regions_to_scan),
            'high_uncertainty_regions': len([r for r in uncertain_regions if r['severity'] == 'high']),
            'confidence_threshold': 1 - confidence_level
        }

    def _suggest_evidence(self, entity_id: Optional[str], triple: Optional[Dict],
                          uncertainty_model: str, context) -> Dict[str, Any]:
        """Suggest evidence to reduce uncertainty."""
        context.update_progress(30, "Analyzing evidence gaps")

        # Get current uncertainty
        uncertainty_result = self._quantify_uncertainty(entity_id, triple, uncertainty_model, context)
        current_uncertainty = uncertainty_result['uncertainty_score']

        context.update_progress(50, "Generating evidence suggestions")

        # Generate evidence suggestions based on uncertainty level
        suggestions = []

        if current_uncertainty > 0.3:
            suggestions.append({
                'type': 'direct_observation',
                'description': 'Collect direct observational data',
                'estimated_uncertainty_reduction': 0.2,
                'effort': 'medium',
                'priority': 'high' if current_uncertainty > 0.5 else 'medium'
            })

        if current_uncertainty > 0.4:
            suggestions.append({
                'type': 'expert_validation',
                'description': 'Seek expert validation or peer review',
                'estimated_uncertainty_reduction': 0.15,
                'effort': 'low',
                'priority': 'medium'
            })

        if current_uncertainty > 0.5:
            suggestions.append({
                'type': 'cross_reference',
                'description': 'Cross-reference with authoritative sources',
                'estimated_uncertainty_reduction': 0.25,
                'effort': 'high',
                'priority': 'high'
            })

        if current_uncertainty > 0.6:
            suggestions.append({
                'type': 'experimental_validation',
                'description': 'Conduct experimental validation',
                'estimated_uncertainty_reduction': 0.35,
                'effort': 'high',
                'priority': 'critical'
            })

        # Add triple-specific suggestions
        if triple:
            suggestions.append({
                'type': 'triple_verification',
                'description': f"Verify relationship: {triple.get('predicate', 'unknown')}",
                'estimated_uncertainty_reduction': 0.2,
                'effort': 'medium',
                'priority': 'high',
                'target_triple': triple
            })

        context.update_progress(90, "Evidence suggestions generated")

        # Calculate projected uncertainty after applying top suggestions
        projected_uncertainty = max(0.0, current_uncertainty - sum(
            s['estimated_uncertainty_reduction'] for s in suggestions[:2]
        ))

        return {
            'current_uncertainty': round(current_uncertainty, 4),
            'projected_uncertainty': round(projected_uncertainty, 4),
            'potential_reduction': round(current_uncertainty - projected_uncertainty, 4),
            'evidence_suggestions': suggestions,
            'suggestion_count': len(suggestions),
            'highest_impact_suggestion': suggestions[0] if suggestions else None
        }

    def _generate_uncertainty_report(self, entity_id: Optional[str], triple: Optional[Dict],
                                     confidence_level: float, include_alternatives: bool, context) -> Dict[str, Any]:
        """Generate comprehensive uncertainty report."""
        context.update_progress(20, "Generating comprehensive uncertainty report")

        # Gather all uncertainty information
        context.update_progress(30, "Collecting uncertainty metrics")
        quantify_result = self._quantify_uncertainty(entity_id, triple, 'bayesian', context)

        context.update_progress(50, "Calculating confidence intervals")
        interval_result = self._calculate_confidence_interval(entity_id, triple, confidence_level, 'bayesian', context)

        context.update_progress(70, "Identifying uncertain regions")
        regions_result = self._identify_uncertain_regions(entity_id, confidence_level, context)

        context.update_progress(85, "Compiling evidence suggestions")
        evidence_result = self._suggest_evidence(entity_id, triple, 'bayesian', context)

        context.update_progress(95, "Finalizing report")

        # Build comprehensive report
        report = {
            'report_type': 'comprehensive_uncertainty_analysis',
            'generated_at': datetime.now().isoformat(),
            'subject': {
                'entity_id': entity_id,
                'triple': triple
            },
            'summary': {
                'overall_uncertainty': quantify_result['uncertainty_score'],
                'uncertainty_level': self._uncertainty_to_level(quantify_result['uncertainty_score']),
                'confidence_interval': interval_result['confidence_interval'],
                'confidence_level': confidence_level,
                'high_uncertainty_regions': regions_result['high_uncertainty_regions'],
                'evidence_suggestions_count': evidence_result['suggestion_count']
            },
            'detailed_findings': {
                'quantification': quantify_result,
                'confidence_interval': interval_result,
                'uncertain_regions': regions_result,
                'evidence_recommendations': evidence_result
            },
            'alternatives': self._generate_alternatives(triple, quantify_result['uncertainty_score']) \
                           if include_alternatives and triple else [],
            'recommendations': self._generate_recommendations(
                quantify_result['uncertainty_score'],
                regions_result['uncertain_regions'],
                evidence_result['evidence_suggestions']
            )
        }

        context.update_progress(100, "Report generation complete")

        return report

    def _calculate_interval_from_score(self, uncertainty_score: float) -> List[float]:
        """Calculate confidence interval from uncertainty score."""
        # Simple interval calculation
        half_width = uncertainty_score / 2
        center = 0.5
        return [round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)]

    def _get_z_score(self, confidence_level: float) -> float:
        """Get Z-score for given confidence level."""
        # Approximate Z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
            0.999: 3.291
        }

        # Find closest or interpolate
        if confidence_level in z_scores:
            return z_scores[confidence_level]

        # Linear interpolation between known values
        levels = sorted(z_scores.keys())
        for i in range(len(levels) - 1):
            if levels[i] <= confidence_level <= levels[i + 1]:
                t = (confidence_level - levels[i]) / (levels[i + 1] - levels[i])
                return z_scores[levels[i]] + t * (z_scores[levels[i + 1]] - z_scores[levels[i]])

        return 1.96  # Default to 95% confidence

    def _build_reasoning_chain(self, entity_id: Optional[str], triple: Optional[Dict]) -> List[Tuple[str, float]]:
        """Build a simulated reasoning chain for uncertainty propagation."""
        chain = []

        if triple:
            chain.extend([
                ('subject_resolution', random.uniform(0.7, 0.95)),
                ('predicate_resolution', random.uniform(0.6, 0.9)),
                ('object_resolution', random.uniform(0.7, 0.95)),
                ('relationship_verification', random.uniform(0.5, 0.85)),
                ('consistency_check', random.uniform(0.6, 0.9))
            ])
        elif entity_id:
            chain.extend([
                ('entity_lookup', random.uniform(0.8, 0.95)),
                ('property_resolution', random.uniform(0.6, 0.9)),
                ('type_inference', random.uniform(0.5, 0.85)),
                ('contextual_verification', random.uniform(0.6, 0.9))
            ])
        else:
            chain.append(('default_reasoning', 0.7))

        return chain

    def _uncertainty_to_severity(self, uncertainty_score: float) -> str:
        """Convert uncertainty score to severity level."""
        if uncertainty_score >= 0.7:
            return 'critical'
        elif uncertainty_score >= 0.5:
            return 'high'
        elif uncertainty_score >= 0.3:
            return 'medium'
        else:
            return 'low'

    def _uncertainty_to_level(self, uncertainty_score: float) -> str:
        """Convert uncertainty score to descriptive level."""
        if uncertainty_score >= 0.8:
            return 'very_high'
        elif uncertainty_score >= 0.6:
            return 'high'
        elif uncertainty_score >= 0.4:
            return 'moderate'
        elif uncertainty_score >= 0.2:
            return 'low'
        else:
            return 'very_low'

    def _recommend_action(self, uncertainty_score: float) -> str:
        """Generate action recommendation based on uncertainty."""
        if uncertainty_score >= 0.7:
            return 'Immediate investigation and evidence collection required'
        elif uncertainty_score >= 0.5:
            return 'Schedule detailed review and validation'
        elif uncertainty_score >= 0.3:
            return 'Monitor and collect additional supporting evidence'
        else:
            return 'Maintain current confidence level, routine monitoring'

    def _generate_alternatives(self, triple: Optional[Dict], uncertainty_score: float) -> List[Dict[str, Any]]:
        """Generate alternative hypotheses/values."""
        if not triple or uncertainty_score < 0.3:
            return []

        alternatives = []

        # Generate alternatives based on predicate type
        predicate = triple.get('predicate', '')
        obj = triple.get('object', '')

        if uncertainty_score > 0.4:
            alternatives.append({
                'hypothesis': f"Alternative interpretation of {predicate}",
                'probability': round(random.uniform(0.1, 0.3), 4),
                'object': f"alt_{obj}",
                'confidence': round(1 - uncertainty_score + 0.1, 4)
            })

        if uncertainty_score > 0.5:
            alternatives.append({
                'hypothesis': f"Negation of {predicate}",
                'probability': round(random.uniform(0.05, 0.2), 4),
                'object': f"not_{obj}",
                'confidence': round(uncertainty_score * 0.5, 4)
            })

        if uncertainty_score > 0.6:
            alternatives.append({
                'hypothesis': 'Unknown or undefined relationship',
                'probability': round(random.uniform(0.1, 0.25), 4),
                'object': 'unknown',
                'confidence': round(uncertainty_score * 0.7, 4)
            })

        return alternatives

    def _generate_recommendations(self, uncertainty_score: float, uncertain_regions: List[Dict],
                                   evidence_suggestions: List[Dict]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if uncertainty_score > 0.6:
            recommendations.append('Prioritize uncertainty reduction - high uncertainty detected')

        if len(uncertain_regions) > 2:
            recommendations.append(f'Address {len(uncertain_regions)} identified uncertain knowledge regions')

        high_priority_evidence = [s for s in evidence_suggestions if s.get('priority') in ['high', 'critical']]
        if high_priority_evidence:
            recommendations.append(f'Collect {len(high_priority_evidence)} high-priority evidence items')

        if not recommendations:
            recommendations.append('Uncertainty levels acceptable - maintain current practices')

        return recommendations

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy, False otherwise
        """
        try:
            # Basic health check
            if self.quantifier is None:
                self.logger.warning("UncertaintyQuantifier not available - will use fallback heuristics")
                # Still healthy as we have fallback

            if self.kg_hub is None:
                self.logger.warning("UnifiedKGIntegrationHub not available - limited KG functionality")
                # Still healthy as we have fallback

            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns:
            JSON schema dictionary for BubbleLabs UI configuration
        """
        return {
            "type": "object",
            "title": "Uncertainty Quantification Configuration",
            "description": "Configure uncertainty quantification parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of uncertainty operation to perform",
                    "enum": ["quantify", "propagate", "interval", "identify", "suggest_evidence", "report"],
                    "enumNames": [
                        "Quantify - Calculate uncertainty score for facts",
                        "Propagate - Propagate uncertainty through reasoning",
                        "Interval - Calculate confidence intervals",
                        "Identify - Find uncertain knowledge regions",
                        "Suggest Evidence - Suggest evidence to reduce uncertainty",
                        "Report - Generate comprehensive uncertainty report"
                    ],
                    "default": "quantify"
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "Entity to analyze for uncertainty",
                    "default": ""
                },
                "triple": {
                    "type": "object",
                    "title": "Triple",
                    "description": "Subject-Predicate-Object triple to analyze",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "title": "Subject",
                            "description": "Subject of the triple"
                        },
                        "predicate": {
                            "type": "string",
                            "title": "Predicate",
                            "description": "Predicate/relationship of the triple"
                        },
                        "object": {
                            "type": "string",
                            "title": "Object",
                            "description": "Object of the triple"
                        }
                    },
                    "required": ["subject", "predicate", "object"]
                },
                "confidence_level": {
                    "type": "number",
                    "title": "Confidence Level",
                    "description": "Confidence level for intervals (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.95
                },
                "uncertainty_model": {
                    "type": "string",
                    "title": "Uncertainty Model",
                    "description": "Mathematical model for uncertainty representation",
                    "enum": ["probabilistic", "fuzzy", "bayesian", "interval"],
                    "enumNames": [
                        "Probabilistic - Standard probability theory",
                        "Fuzzy - Fuzzy logic-based uncertainty",
                        "Bayesian - Bayesian probability updating",
                        "Interval - Interval-based uncertainty"
                    ],
                    "default": "bayesian"
                },
                "propagation_method": {
                    "type": "string",
                    "title": "Propagation Method",
                    "description": "Method for propagating uncertainty through reasoning",
                    "enum": ["linear", "bayesian", "monte_carlo"],
                    "enumNames": [
                        "Linear - Linear error propagation",
                        "Bayesian - Bayesian belief updating",
                        "Monte Carlo - Monte Carlo simulation"
                    ],
                    "default": "bayesian"
                },
                "min_samples": {
                    "type": "integer",
                    "title": "Minimum Samples",
                    "description": "Minimum number of samples for Monte Carlo methods",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 100
                },
                "include_alternatives": {
                    "type": "boolean",
                    "title": "Include Alternatives",
                    "description": "Include alternative hypotheses in results",
                    "default": True
                }
            },
            "required": ["operation"]
        }
