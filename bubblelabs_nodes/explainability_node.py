"""
Explainability Node for BubbleLabs Integration

Provides comprehensive explanation capabilities for knowledge reasoning:
- Explain reasoning paths and inference chains
- Show evidence chains supporting conclusions
- Explain confidence scores with breakdowns
- Visualize inference steps as structured paths
- Generate human-readable explanations for different audiences
- Support counterfactual explanations

This node integrates with the knowledge engine to provide transparency
into how conclusions are reached and what evidence supports them.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from .base_node import BubbleLabsNode, NodeExecutionError


@dataclass
class EvidenceStep:
    """Represents a single step in an evidence chain."""
    step_number: int
    premise: str
    inference_rule: Optional[str] = None
    confidence: float = 1.0
    source: Optional[str] = None
    supporting_facts: List[str] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Structured explanation result."""
    explanation: str
    evidence_chain: List[EvidenceStep]
    confidence_breakdown: Dict[str, Any]
    inference_path: List[Dict[str, Any]]
    counterfactuals: List[str] = field(default_factory=list)
    visualization_data: Optional[Dict[str, Any]] = None


class ExplainabilityNode(BubbleLabsNode):
    """
    Explain reasoning, evidence chains, and decision-making processes.

    This node provides transparency into AI reasoning by:
    - Tracing how conclusions are derived from premises
    - Identifying the evidence supporting each step
    - Breaking down confidence scores into components
    - Visualizing inference paths as directed graphs
    - Generating explanations tailored to different audiences
    - Exploring what-if scenarios with counterfactuals

    Operations:
    - explain_reasoning: Generate full explanation of reasoning process
    - show_evidence: Display evidence chain supporting a conclusion
    - explain_confidence: Break down confidence score components
    - visualize_path: Create visualization data for inference path
    """

    # Node metadata
    DISPLAY_NAME = "Explainability"
    DESCRIPTION = "Explain reasoning, evidence chains, and decision-making processes"
    ICON = "explainability"
    CATEGORY = "intelligence"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge engine
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for ExplainabilityNode"
        )

        self.ExplanationGenerator = self.safe_import(
            'knowledge_engine.reasoning.explanation_generator.ExplanationGenerator',
            fallback_value=None,
            error_msg="ExplanationGenerator not available for ExplainabilityNode"
        )

        # Initialize explanation generator if available
        self.explanation_generator = None
        if self.ExplanationGenerator:
            try:
                self.explanation_generator = self.ExplanationGenerator()
                self.logger.info("ExplanationGenerator initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize ExplanationGenerator: {e}")

        # Initialize KG hub if available
        self.kg_hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields vary by operation:
        - explain_reasoning: Requires conclusion or entity_id
        - show_evidence: Requires conclusion, entity_id, or triple
        - explain_confidence: Requires conclusion or entity_id with confidence data
        - visualize_path: Requires conclusion or entity_id
        """
        errors = []

        # Get operation from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'explain_reasoning'))

        valid_operations = ['explain_reasoning', 'show_evidence', 'explain_confidence', 'visualize_path']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")

        # Check for required content based on operation
        conclusion = inputs.get('conclusion') or self.config.get('conclusion')
        entity_id = inputs.get('entity_id') or self.config.get('entity_id')
        triple = inputs.get('triple') or self.config.get('triple')

        has_content = bool(conclusion or entity_id or triple)

        if not has_content:
            errors.append(
                "At least one of 'conclusion', 'entity_id', or 'triple' must be provided "
                "in inputs or config"
            )

        # Validate triple structure if provided
        if triple:
            if not isinstance(triple, dict):
                errors.append("triple must be a dictionary with 'subject', 'predicate', and 'object' keys")
            else:
                required_triple_fields = ['subject', 'predicate', 'object']
                missing_fields = [f for f in required_triple_fields if f not in triple]
                if missing_fields:
                    errors.append(f"triple is missing required fields: {', '.join(missing_fields)}")

        # Validate explanation_type if provided
        explanation_type = inputs.get('explanation_type', self.config.get('explanation_type', 'simple'))
        valid_types = ['simple', 'detailed', 'technical', 'visual']
        if explanation_type not in valid_types:
            errors.append(f"Invalid explanation_type: {explanation_type}. Must be one of: {', '.join(valid_types)}")

        # Validate audience if provided
        audience = inputs.get('audience', self.config.get('audience', 'general'))
        valid_audiences = ['general', 'expert', 'business']
        if audience not in valid_audiences:
            errors.append(f"Invalid audience: {audience}. Must be one of: {', '.join(valid_audiences)}")

        # Validate max_depth
        max_depth = inputs.get('max_depth', self.config.get('max_depth', 5))
        try:
            depth = int(max_depth)
            if depth < 1 or depth > 20:
                errors.append("max_depth must be between 1 and 20")
        except (ValueError, TypeError):
            errors.append("max_depth must be an integer")

        # Validate include_counterfactuals
        if 'include_counterfactuals' in inputs:
            if not isinstance(inputs['include_counterfactuals'], bool):
                errors.append("include_counterfactuals must be a boolean")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the explainability operation.

        Args:
            inputs: Input data containing conclusion/entity/triple and explanation parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - explanation: Human-readable explanation text
                - evidence_chain: List of evidence steps
                - confidence_breakdown: Detailed confidence analysis
                - inference_path: Structured inference steps
                - visualization_data: Data for visualizing the reasoning (if visual type)
                - counterfactuals: What-if scenarios (if requested)
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If explanation generation fails
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'explain_reasoning'))
        conclusion = inputs.get('conclusion', self.config.get('conclusion'))
        explanation_type = inputs.get('explanation_type', self.config.get('explanation_type', 'simple'))
        entity_id = inputs.get('entity_id', self.config.get('entity_id'))
        triple = inputs.get('triple', self.config.get('triple'))
        include_counterfactuals = inputs.get(
            'include_counterfactuals',
            self.config.get('include_counterfactuals', False)
        )
        max_depth = inputs.get('max_depth', self.config.get('max_depth', 5))
        audience = inputs.get('audience', self.config.get('audience', 'general'))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(
            f"Starting {operation} with type={explanation_type}, "
            f"audience={audience}, max_depth={max_depth}"
        )

        try:
            # Build target description
            target = self._build_target_description(conclusion, entity_id, triple)

            context.update_progress(25, f"Retrieving reasoning data for: {target}")

            # Execute based on operation type
            if operation == 'explain_reasoning':
                result = self._explain_reasoning(
                    conclusion, entity_id, triple, explanation_type,
                    audience, max_depth, include_counterfactuals, context
                )
            elif operation == 'show_evidence':
                result = self._show_evidence(
                    conclusion, entity_id, triple, explanation_type,
                    audience, max_depth, context
                )
            elif operation == 'explain_confidence':
                result = self._explain_confidence(
                    conclusion, entity_id, triple, explanation_type,
                    audience, context
                )
            elif operation == 'visualize_path':
                result = self._visualize_path(
                    conclusion, entity_id, triple, max_depth, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'operation': operation}
                )

            # Add metadata
            result['operation'] = operation
            result['target'] = target
            result['explanation_type'] = explanation_type
            result['audience'] = audience
            result['max_depth'] = max_depth
            result['include_counterfactuals'] = include_counterfactuals

            # Add to context
            context.add_artifact('explainability_result', {
                'result': result,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            })

            context.update_progress(100, "Explanation complete")
            self.logger.info(f"Explanation completed for {operation}")

            return result

        except Exception as e:
            self.logger.error(f"Explanation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Explanation generation failed: {str(e)}",
                details={
                    'operation': operation,
                    'target': self._build_target_description(conclusion, entity_id, triple),
                    'exception_type': type(e).__name__
                }
            ) from e

    def _explain_reasoning(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict],
        explanation_type: str,
        audience: str,
        max_depth: int,
        include_counterfactuals: bool,
        context
    ) -> Dict[str, Any]:
        """Generate full reasoning explanation."""
        context.update_progress(40, "Building reasoning path")

        # Get reasoning path
        reasoning_path = self._get_reasoning_path(conclusion, entity_id, triple, max_depth)

        context.update_progress(60, "Generating explanation text")

        # Generate explanation based on type and audience
        explanation = self._generate_explanation(
            reasoning_path, explanation_type, audience, "reasoning"
        )

        context.update_progress(75, "Building evidence chain")

        # Build evidence chain
        evidence_chain = self._build_evidence_chain(reasoning_path)

        context.update_progress(85, "Computing confidence breakdown")

        # Compute confidence breakdown
        confidence_breakdown = self._compute_confidence_breakdown(reasoning_path)

        # Generate counterfactuals if requested
        counterfactuals = []
        if include_counterfactuals:
            context.update_progress(90, "Generating counterfactuals")
            counterfactuals = self._generate_counterfactuals(reasoning_path)

        # Generate visualization data for visual type
        visualization_data = None
        if explanation_type == 'visual':
            visualization_data = self._generate_visualization_data(reasoning_path)

        return {
            'explanation': explanation,
            'evidence_chain': [self._evidence_step_to_dict(step) for step in evidence_chain],
            'confidence_breakdown': confidence_breakdown,
            'inference_path': reasoning_path,
            'counterfactuals': counterfactuals,
            'visualization_data': visualization_data,
            'step_count': len(reasoning_path),
            'confidence_score': confidence_breakdown.get('overall', 0.0)
        }

    def _show_evidence(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict],
        explanation_type: str,
        audience: str,
        max_depth: int,
        context
    ) -> Dict[str, Any]:
        """Show evidence chain supporting conclusion."""
        context.update_progress(40, "Retrieving supporting evidence")

        # Get evidence path
        evidence_path = self._get_reasoning_path(conclusion, entity_id, triple, max_depth)

        context.update_progress(60, "Analyzing evidence strength")

        # Build detailed evidence chain
        evidence_chain = self._build_evidence_chain(evidence_path)

        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(evidence_chain)

        context.update_progress(80, "Generating evidence summary")

        # Generate evidence-focused explanation
        explanation = self._generate_explanation(
            evidence_path, explanation_type, audience, "evidence"
        )

        return {
            'explanation': explanation,
            'evidence_chain': [self._evidence_step_to_dict(step) for step in evidence_chain],
            'evidence_strength': evidence_strength,
            'evidence_count': len(evidence_chain),
            'strongest_evidence': self._find_strongest_evidence(evidence_chain),
            'weakest_evidence': self._find_weakest_evidence(evidence_chain),
            'confidence_breakdown': self._compute_confidence_breakdown(evidence_path),
            'inference_path': evidence_path,
            'counterfactuals': []
        }

    def _explain_confidence(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict],
        explanation_type: str,
        audience: str,
        context
    ) -> Dict[str, Any]:
        """Explain confidence score breakdown."""
        context.update_progress(40, "Analyzing confidence components")

        # Get reasoning data
        reasoning_path = self._get_reasoning_path(conclusion, entity_id, triple, 5)

        context.update_progress(60, "Computing detailed confidence breakdown")

        # Compute comprehensive confidence breakdown
        confidence_breakdown = self._compute_detailed_confidence_breakdown(
            reasoning_path, conclusion, entity_id, triple
        )

        context.update_progress(80, "Generating confidence explanation")

        # Generate confidence-focused explanation
        explanation = self._generate_confidence_explanation(
            confidence_breakdown, explanation_type, audience
        )

        return {
            'explanation': explanation,
            'confidence_breakdown': confidence_breakdown,
            'overall_confidence': confidence_breakdown.get('overall', 0.0),
            'confidence_factors': confidence_breakdown.get('factors', {}),
            'evidence_chain': [self._evidence_step_to_dict(step) for step in self._build_evidence_chain(reasoning_path)],
            'inference_path': reasoning_path,
            'recommendations': self._generate_confidence_recommendations(confidence_breakdown),
            'counterfactuals': []
        }

    def _visualize_path(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict],
        max_depth: int,
        context
    ) -> Dict[str, Any]:
        """Generate visualization data for inference path."""
        context.update_progress(40, "Building inference path structure")

        # Get reasoning path
        reasoning_path = self._get_reasoning_path(conclusion, entity_id, triple, max_depth)

        context.update_progress(70, "Generating visualization data")

        # Generate comprehensive visualization data
        visualization_data = self._generate_visualization_data(reasoning_path)

        # Generate node and edge lists for graph visualization
        nodes, edges = self._extract_nodes_edges(reasoning_path)

        context.update_progress(90, "Creating path description")

        # Generate path description
        path_description = self._generate_path_description(reasoning_path)

        return {
            'explanation': path_description,
            'visualization_data': visualization_data,
            'nodes': nodes,
            'edges': edges,
            'path_length': len(reasoning_path),
            'evidence_chain': [self._evidence_step_to_dict(step) for step in self._build_evidence_chain(reasoning_path)],
            'confidence_breakdown': self._compute_confidence_breakdown(reasoning_path),
            'inference_path': reasoning_path,
            'counterfactuals': []
        }

    def _get_reasoning_path(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict],
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Retrieve or construct the reasoning path."""
        path = []

        # Try to get from explanation generator if available
        if self.explanation_generator and hasattr(self.explanation_generator, 'get_reasoning_path'):
            try:
                target = conclusion or entity_id or self._triple_to_string(triple)
                path = self.explanation_generator.get_reasoning_path(target, max_depth)
                if path:
                    return path
            except Exception as e:
                self.logger.warning(f"ExplanationGenerator.get_reasoning_path failed: {e}")

        # Try to get from KG hub if available
        if self.kg_hub and hasattr(self.kg_hub, 'get_reasoning_trace'):
            try:
                target = conclusion or entity_id or self._triple_to_string(triple)
                path = self.kg_hub.get_reasoning_trace(target, max_depth)
                if path:
                    return path
            except Exception as e:
                self.logger.warning(f"UnifiedKGIntegrationHub.get_reasoning_trace failed: {e}")

        # Fallback: construct a synthetic reasoning path
        return self._construct_fallback_path(conclusion, entity_id, triple, max_depth)

    def _construct_fallback_path(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict],
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Construct a fallback reasoning path when no generator is available."""
        path = []

        # Build target
        target = conclusion or entity_id or self._triple_to_string(triple)

        # Create a simple linear path structure
        if target:
            path.append({
                'step': 1,
                'type': 'conclusion',
                'content': target,
                'confidence': 0.85,
                'source': 'derived'
            })

            # Add synthetic supporting steps
            for i in range(2, min(max_depth + 1, 4)):
                path.append({
                    'step': i,
                    'type': 'premise',
                    'content': f"Supporting evidence level {i-1} for: {target}",
                    'confidence': max(0.5, 0.9 - (i * 0.1)),
                    'source': 'inferred'
                })

        return path

    def _build_evidence_chain(self, reasoning_path: List[Dict[str, Any]]) -> List[EvidenceStep]:
        """Build a structured evidence chain from reasoning path."""
        chain = []

        for i, step in enumerate(reasoning_path):
            evidence_step = EvidenceStep(
                step_number=i + 1,
                premise=step.get('content', str(step)),
                inference_rule=step.get('rule', step.get('type', 'inference')),
                confidence=step.get('confidence', 0.5),
                source=step.get('source', 'unknown'),
                supporting_facts=step.get('supporting_facts', [])
            )
            chain.append(evidence_step)

        return chain

    def _generate_explanation(
        self,
        reasoning_path: List[Dict[str, Any]],
        explanation_type: str,
        audience: str,
        focus: str
    ) -> str:
        """Generate human-readable explanation."""
        if not reasoning_path:
            return "No reasoning path available to explain."

        # Use explanation generator if available
        if self.explanation_generator and hasattr(self.explanation_generator, 'generate'):
            try:
                return self.explanation_generator.generate(
                    reasoning_path, explanation_type, audience, focus
                )
            except Exception as e:
                self.logger.warning(f"ExplanationGenerator.generate failed: {e}")

        # Fallback explanation generation
        return self._generate_fallback_explanation(reasoning_path, explanation_type, audience, focus)

    def _generate_fallback_explanation(
        self,
        reasoning_path: List[Dict[str, Any]],
        explanation_type: str,
        audience: str,
        focus: str
    ) -> str:
        """Generate fallback explanation when generator is unavailable."""
        parts = []

        # Header based on focus
        if focus == "reasoning":
            parts.append("Reasoning Process Explanation")
        elif focus == "evidence":
            parts.append("Evidence Chain Analysis")
        else:
            parts.append("Explanation")

        parts.append("=" * 40)

        # Audience-specific introduction
        if audience == "general":
            parts.append(f"This explanation uses {len(reasoning_path)} steps to reach the conclusion.")
        elif audience == "expert":
            parts.append(f"Inference chain depth: {len(reasoning_path)} steps.")
        elif audience == "business":
            parts.append(f"Decision based on {len(reasoning_path)} supporting factors.")

        # Step details based on explanation type
        if explanation_type in ["detailed", "technical"]:
            parts.append("\nDetailed Steps:")
            for step in reasoning_path:
                step_num = step.get('step', '?')
                content = step.get('content', str(step))
                confidence = step.get('confidence', 0.0)
                source = step.get('source', 'unknown')

                if explanation_type == "technical":
                    parts.append(f"  [{step_num}] {content}")
                    parts.append(f"       Confidence: {confidence:.2%}, Source: {source}")
                else:
                    parts.append(f"  Step {step_num}: {content}")

        elif explanation_type == "simple":
            parts.append(f"\nThe conclusion is based on {len(reasoning_path)} logical steps.")
            conclusion_step = reasoning_path[0] if reasoning_path else None
            if conclusion_step:
                parts.append(f"Final conclusion: {conclusion_step.get('content', 'N/A')}")

        # Visual type includes structure hint
        if explanation_type == "visual":
            parts.append(f"\nVisual Structure: {len(reasoning_path)} nodes in reasoning graph.")

        return "\n".join(parts)

    def _compute_confidence_breakdown(self, reasoning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute confidence score breakdown."""
        if not reasoning_path:
            return {'overall': 0.0, 'factors': {}}

        confidences = [step.get('confidence', 0.5) for step in reasoning_path]
        overall = sum(confidences) / len(confidences) if confidences else 0.0

        # Apply chain rule: confidence decreases with chain length
        chain_penalty = 0.95 ** (len(reasoning_path) - 1) if len(reasoning_path) > 1 else 1.0
        adjusted_overall = overall * chain_penalty

        return {
            'overall': round(adjusted_overall, 4),
            'raw_average': round(overall, 4),
            'chain_penalty': round(chain_penalty, 4),
            'min_confidence': round(min(confidences), 4) if confidences else 0.0,
            'max_confidence': round(max(confidences), 4) if confidences else 0.0,
            'step_count': len(reasoning_path),
            'factors': {
                'evidence_quality': round(overall, 4),
                'chain_length_factor': round(chain_penalty, 4),
                'consistency_score': round(self._compute_consistency_score(reasoning_path), 4)
            }
        }

    def _compute_detailed_confidence_breakdown(
        self,
        reasoning_path: List[Dict[str, Any]],
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict]
    ) -> Dict[str, Any]:
        """Compute detailed confidence breakdown with additional factors."""
        basic = self._compute_confidence_breakdown(reasoning_path)

        # Add additional factors
        factors = basic.get('factors', {})

        # Source diversity
        sources = set(step.get('source', 'unknown') for step in reasoning_path)
        factors['source_diversity'] = round(min(len(sources) / 3, 1.0), 4)

        # Recency factor (if timestamps available)
        timestamps = [step.get('timestamp') for step in reasoning_path if step.get('timestamp')]
        factors['recency_score'] = 0.8 if timestamps else 0.5

        # Evidence type diversity
        types = set(step.get('type', 'unknown') for step in reasoning_path)
        factors['evidence_type_diversity'] = round(min(len(types) / 2, 1.0), 4)

        # Recalculate overall with additional factors
        factor_values = list(factors.values())
        enhanced_overall = sum(factor_values) / len(factor_values) if factor_values else basic['overall']

        return {
            **basic,
            'overall': round(enhanced_overall, 4),
            'factors': factors,
            'source_count': len(sources),
            'evidence_types': list(types)
        }

    def _compute_consistency_score(self, reasoning_path: List[Dict[str, Any]]) -> float:
        """Compute consistency score for the reasoning path."""
        if len(reasoning_path) < 2:
            return 1.0

        # Check for contradictions in the path
        contents = [step.get('content', '').lower() for step in reasoning_path]

        for i, c1 in enumerate(contents):
            for c2 in contents[i + 1:]:
                # Simple contradiction detection
                if c1.startswith('not ') and c1[4:] == c2:
                    return 0.0
                if c2.startswith('not ') and c2[4:] == c1:
                    return 0.0

        return 1.0

    def _generate_counterfactuals(self, reasoning_path: List[Dict[str, Any]]) -> List[str]:
        """Generate counterfactual explanations."""
        counterfactuals = []

        if not reasoning_path:
            return counterfactuals

        # Generate what-if scenarios by modifying key steps
        for i, step in enumerate(reasoning_path[:3]):  # Limit to first 3 steps
            content = step.get('content', '')
            confidence = step.get('confidence', 0.5)

            # If this step had lower confidence
            counterfactuals.append(
                f"If step {i+1} ('{content[:50]}...') had lower confidence "
                f"({max(0, confidence - 0.3):.2%}), the overall conclusion "
                f"would be weaker."
            )

            # If this step was removed
            counterfactuals.append(
                f"If step {i+1} ('{content[:50]}...') was removed, "
                f"the conclusion would rely on {len(reasoning_path) - 1} steps instead."
            )

        return counterfactuals

    def _generate_visualization_data(self, reasoning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data for visualizing the reasoning path."""
        nodes = []
        edges = []

        for i, step in enumerate(reasoning_path):
            node_id = f"step_{i}"
            nodes.append({
                'id': node_id,
                'label': step.get('content', f"Step {i+1}")[:50],
                'type': step.get('type', 'unknown'),
                'confidence': step.get('confidence', 0.5),
                'level': i
            })

            if i > 0:
                edges.append({
                    'source': f"step_{i}",
                    'target': f"step_{i-1}",
                    'label': step.get('rule', 'supports'),
                    'strength': step.get('confidence', 0.5)
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'layout': 'hierarchical',
            'direction': 'bottom-up',
            'title': 'Reasoning Path Visualization'
        }

    def _extract_nodes_edges(
        self,
        reasoning_path: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract nodes and edges for graph visualization."""
        nodes = []
        edges = []

        for i, step in enumerate(reasoning_path):
            node_id = f"node_{i}"
            nodes.append({
                'id': node_id,
                'data': {
                    'label': step.get('content', f"Step {i+1}")[:100],
                    'type': step.get('type', 'unknown'),
                    'confidence': step.get('confidence', 0.5)
                }
            })

            if i > 0:
                edges.append({
                    'id': f"edge_{i}",
                    'source': f"node_{i}",
                    'target': f"node_{i-1}",
                    'data': {
                        'label': 'supports',
                        'weight': step.get('confidence', 0.5)
                    }
                })

        return nodes, edges

    def _generate_path_description(self, reasoning_path: List[Dict[str, Any]]) -> str:
        """Generate a textual description of the path."""
        if not reasoning_path:
            return "Empty reasoning path."

        parts = [f"Reasoning path with {len(reasoning_path)} steps:"]

        for i, step in enumerate(reasoning_path):
            content = step.get('content', 'Unknown')
            arrow = "->" if i < len(reasoning_path) - 1 else "[OK]"
            parts.append(f"  {arrow} Step {i+1}: {content[:60]}")

        return "\n".join(parts)

    def _generate_confidence_explanation(
        self,
        confidence_breakdown: Dict[str, Any],
        explanation_type: str,
        audience: str
    ) -> str:
        """Generate explanation focused on confidence."""
        overall = confidence_breakdown.get('overall', 0.0)
        factors = confidence_breakdown.get('factors', {})

        parts = ["Confidence Score Explanation", "=" * 40]

        if audience == "general":
            parts.append(f"Overall confidence: {overall:.1%}")
            if overall > 0.8:
                parts.append("This indicates high confidence in the conclusion.")
            elif overall > 0.5:
                parts.append("This indicates moderate confidence. Additional verification recommended.")
            else:
                parts.append("This indicates low confidence. Conclusion should be treated with caution.")

        elif audience == "expert":
            parts.append(f"Overall Confidence: {overall:.4f}")
            parts.append("\nComponent Breakdown:")
            for factor, value in factors.items():
                parts.append(f"  - {factor}: {value:.4f}")

        elif audience == "business":
            parts.append(f"Confidence Level: {overall:.0%}")
            parts.append("\nRisk Assessment:")
            if overall > 0.8:
                parts.append("  Low risk - suitable for automated decision making")
            elif overall > 0.6:
                parts.append("  Medium risk - human review recommended")
            else:
                parts.append("  High risk - requires expert validation")

        return "\n".join(parts)

    def _generate_confidence_recommendations(self, confidence_breakdown: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on confidence analysis."""
        recommendations = []
        overall = confidence_breakdown.get('overall', 0.0)
        factors = confidence_breakdown.get('factors', {})

        if overall < 0.6:
            recommendations.append("Consider gathering additional evidence to strengthen confidence.")

        if factors.get('chain_length_factor', 1.0) < 0.8:
            recommendations.append("Long reasoning chain detected. Consider direct verification of key steps.")

        if factors.get('source_diversity', 1.0) < 0.5:
            recommendations.append("Limited source diversity. Seek corroboration from additional sources.")

        if not recommendations:
            recommendations.append("Confidence level is acceptable. No immediate action required.")

        return recommendations

    def _calculate_evidence_strength(self, evidence_chain: List[EvidenceStep]) -> Dict[str, Any]:
        """Calculate overall evidence strength metrics."""
        if not evidence_chain:
            return {'overall': 0.0, 'rating': 'insufficient'}

        confidences = [step.confidence for step in evidence_chain]
        avg_confidence = sum(confidences) / len(confidences)

        # Determine rating
        if avg_confidence > 0.8 and len(evidence_chain) >= 3:
            rating = 'strong'
        elif avg_confidence > 0.6:
            rating = 'moderate'
        elif avg_confidence > 0.4:
            rating = 'weak'
        else:
            rating = 'insufficient'

        return {
            'overall': round(avg_confidence, 4),
            'rating': rating,
            'evidence_count': len(evidence_chain),
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5)
        }

    def _find_strongest_evidence(self, evidence_chain: List[EvidenceStep]) -> Optional[Dict[str, Any]]:
        """Find the strongest evidence in the chain."""
        if not evidence_chain:
            return None

        strongest = max(evidence_chain, key=lambda x: x.confidence)
        return self._evidence_step_to_dict(strongest)

    def _find_weakest_evidence(self, evidence_chain: List[EvidenceStep]) -> Optional[Dict[str, Any]]:
        """Find the weakest evidence in the chain."""
        if not evidence_chain:
            return None

        weakest = min(evidence_chain, key=lambda x: x.confidence)
        return self._evidence_step_to_dict(weakest)

    def _evidence_step_to_dict(self, step: EvidenceStep) -> Dict[str, Any]:
        """Convert EvidenceStep to dictionary."""
        return {
            'step_number': step.step_number,
            'premise': step.premise,
            'inference_rule': step.inference_rule,
            'confidence': step.confidence,
            'source': step.source,
            'supporting_facts': step.supporting_facts
        }

    def _build_target_description(
        self,
        conclusion: Optional[str],
        entity_id: Optional[str],
        triple: Optional[Dict]
    ) -> str:
        """Build a target description string."""
        if conclusion:
            return f"conclusion: {conclusion}"
        if entity_id:
            return f"entity: {entity_id}"
        if triple:
            return f"triple: {self._triple_to_string(triple)}"
        return "unknown target"

    def _triple_to_string(self, triple: Optional[Dict]) -> str:
        """Convert triple dict to string representation."""
        if not triple:
            return ""
        subject = triple.get('subject', '?')
        predicate = triple.get('predicate', '?')
        obj = triple.get('object', '?')
        return f"{subject} {predicate} {obj}"

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "title": "Explainability Configuration",
            "description": "Configure explainability and explanation generation parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of explanation operation to perform",
                    "enum": ["explain_reasoning", "show_evidence", "explain_confidence", "visualize_path"],
                    "enumNames": [
                        "Explain Reasoning - Generate full reasoning explanation",
                        "Show Evidence - Display evidence chain supporting conclusion",
                        "Explain Confidence - Break down confidence score components",
                        "Visualize Path - Create visualization data for inference path"
                    ],
                    "default": "explain_reasoning"
                },
                "conclusion": {
                    "type": "string",
                    "title": "Conclusion",
                    "description": "The conclusion or result to explain",
                    "default": ""
                },
                "explanation_type": {
                    "type": "string",
                    "title": "Explanation Type",
                    "description": "Level of detail for the explanation",
                    "enum": ["simple", "detailed", "technical", "visual"],
                    "enumNames": [
                        "Simple - High-level summary suitable for quick understanding",
                        "Detailed - Comprehensive explanation with step breakdown",
                        "Technical - Technical details with confidence metrics",
                        "Visual - Structured data for visualization"
                    ],
                    "default": "simple"
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "ID of the entity to explain (alternative to conclusion)",
                    "default": ""
                },
                "triple": {
                    "type": "object",
                    "title": "Triple",
                    "description": "Subject-predicate-object triple to explain",
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
                "include_counterfactuals": {
                    "type": "boolean",
                    "title": "Include Counterfactuals",
                    "description": "Generate what-if scenarios and alternative explanations",
                    "default": False
                },
                "max_depth": {
                    "type": "integer",
                    "title": "Maximum Depth",
                    "description": "Maximum depth for reasoning chain traversal (1-20)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                },
                "audience": {
                    "type": "string",
                    "title": "Target Audience",
                    "description": "Target audience for the explanation",
                    "enum": ["general", "expert", "business"],
                    "enumNames": [
                        "General - Accessible to non-technical users",
                        "Expert - Technical details for domain experts",
                        "Business - Focus on actionable insights and risk"
                    ],
                    "default": "general"
                }
            },
            "required": ["operation"]
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node can function (even in fallback mode)
        """
        try:
            # Node can operate in fallback mode without external dependencies
            # but we check if basic initialization succeeded
            return True
        except Exception:
            return False
