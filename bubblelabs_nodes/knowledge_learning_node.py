"""
Knowledge Learning Node for BubbleLabs Integration

Learns from user feedback and adapts knowledge extraction, querying, and recommendations.

Features:
- Learn from user feedback on extraction results
- Adapt confidence scores based on historical accuracy
- Update entity profiles from usage patterns
- Improve query results based on user preferences
- Track learning history and improvements
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from collections import deque, defaultdict
import json
import asyncio
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeLearningNode(BubbleLabsNode):
    """
    Learn from feedback and adapt knowledge extraction, querying, and recommendations.

    Supports five operations:
    - feedback: Record user feedback on extraction/query results
    - adapt: Apply learned adaptations to system behavior
    - improve: Trigger active learning improvements
    - analyze_learning: Analyze learning history and metrics
    - reset: Reset learning state

    The node integrates with the AdaptationEngine for advanced learning capabilities
    and falls back to internal learning mechanisms when unavailable.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Learning"
    DESCRIPTION = "Learn from feedback and adapt knowledge extraction, querying, and recommendations"
    ICON = "knowledge-learning"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports with fallback support
        self.AdaptationEngine = self.safe_import(
            'knowledge_engine.learning.adaptation_engine.AdaptationEngine',
            fallback_value=None,
            error_msg="AdaptationEngine not available"
        )
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available"
        )
        self.KnowledgeGraphModels = self.safe_import(
            'knowledge_engine.graph.kg_models.KnowledgeGraphModels',
            fallback_value=None,
            error_msg="KnowledgeGraphModels not available"
        )
        self.EntityProfile = self.safe_import(
            'knowledge_engine.graph.kg_models.EntityProfile',
            fallback_value=None,
            error_msg="EntityProfile not available"
        )

        # Initialize adaptation engine if available
        self.adaptation_engine = None
        self._init_adaptation_engine()

        # Initialize fallback learning storage
        self._init_fallback_learning()

    def _init_adaptation_engine(self):
        """Initialize the adaptation engine if available."""
        if self.AdaptationEngine:
            try:
                learning_rate = self.config.get('learning_rate', 0.1)
                feedback_history_limit = self.config.get('feedback_history_limit', 1000)
                
                self.adaptation_engine = self.AdaptationEngine(
                    learning_rate=learning_rate,
                    experience_buffer_size=feedback_history_limit,
                    enable_auto_adaptation=True
                )
                self.logger.info("AdaptationEngine initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize AdaptationEngine: {e}")
                self.adaptation_engine = None

    def _init_fallback_learning(self):
        """Initialize fallback learning storage when AdaptationEngine is unavailable."""
        max_history = self.config.get('feedback_history_limit', 1000)
        
        # Feedback history storage
        self._feedback_history: deque = deque(maxlen=max_history)
        
        # Entity confidence tracking
        self._entity_confidence: Dict[str, Dict[str, Any]] = {}
        
        # Triple confidence tracking
        self._triple_confidence: Dict[str, Dict[str, Any]] = {}
        
        # User preferences storage
        self._user_preferences: Dict[str, Any] = {
            'preferred_sources': [],
            'confidence_threshold': 0.7,
            'excluded_entities': set(),
            'preferred_relationships': []
        }
        
        # Learning metrics
        self._learning_metrics = {
            'total_feedback_received': 0,
            'positive_feedback_count': 0,
            'negative_feedback_count': 0,
            'corrections_applied': 0,
            'confidence_adjustments': 0,
            'entity_updates': 0,
            'last_learning_at': None
        }
        
        # Adaptations applied
        self._adaptations: List[Dict[str, Any]] = []
        
        self.logger.info("Fallback learning storage initialized")

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - operation: One of ["feedback", "adapt", "improve", "analyze_learning", "reset"]

        Optional (depending on operation):
            - feedback_type: One of ["positive", "negative", "correction", "confirmation"]
            - entity_id: Entity being rated
            - triple_id: Triple being rated
            - correction: Corrected data if feedback is correction
        """
        errors = []

        # Check required fields
        if 'operation' not in inputs:
            errors.append("Missing required field: operation")
        else:
            valid_operations = ['feedback', 'adapt', 'improve', 'analyze_learning', 'reset']
            if inputs['operation'] not in valid_operations:
                errors.append(
                    f"Invalid operation: '{inputs['operation']}'. "
                    f"Must be one of: {', '.join(valid_operations)}"
                )

        operation = inputs.get('operation')

        # Validate feedback-specific inputs
        if operation == 'feedback':
            if 'feedback_type' not in inputs:
                errors.append("Missing required field for feedback operation: feedback_type")
            else:
                valid_feedback_types = ['positive', 'negative', 'correction', 'confirmation']
                if inputs['feedback_type'] not in valid_feedback_types:
                    errors.append(
                        f"Invalid feedback_type: '{inputs['feedback_type']}'. "
                        f"Must be one of: {', '.join(valid_feedback_types)}"
                    )

            # Validate correction data if provided
            if inputs.get('feedback_type') == 'correction':
                if 'correction' not in inputs:
                    errors.append("Missing required field for correction feedback: correction")
                elif not isinstance(inputs['correction'], dict):
                    errors.append("correction must be an object/dictionary")

        # Validate entity_id if provided
        if 'entity_id' in inputs:
            if not isinstance(inputs['entity_id'], str):
                errors.append("entity_id must be a string")

        # Validate triple_id if provided
        if 'triple_id' in inputs:
            if not isinstance(inputs['triple_id'], str):
                errors.append("triple_id must be a string")

        # Validate learning_rate if provided
        if 'learning_rate' in inputs:
            try:
                lr = float(inputs['learning_rate'])
                if not 0.0 <= lr <= 1.0:
                    errors.append("learning_rate must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("learning_rate must be a number")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the knowledge learning operation.

        Args:
            inputs: Operation parameters including:
                - operation: Type of learning operation
                - feedback_type: Type of feedback (for feedback operation)
                - entity_id: Entity being rated
                - triple_id: Triple being rated
                - correction: Corrected data if applicable
                - notes: Additional notes
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - learning_applied: Whether learning was applied
                - adaptations: List of adaptations made
                - confidence_adjustments: Confidence score adjustments
                - metrics: Learning metrics

        Raises:
            NodeExecutionError: If operation fails
        """
        operation = inputs['operation']
        
        context.update_progress(10, f"Starting knowledge learning operation: {operation}")
        self.logger.info(f"Executing knowledge learning: {operation}")

        try:
            if operation == 'feedback':
                result = self._process_feedback(inputs, context)
            elif operation == 'adapt':
                result = self._apply_adaptations(inputs, context)
            elif operation == 'improve':
                result = self._trigger_improvements(inputs, context)
            elif operation == 'analyze_learning':
                result = self._analyze_learning(inputs, context)
            elif operation == 'reset':
                result = self._reset_learning(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['feedback', 'adapt', 'improve', 'analyze_learning', 'reset']}
                )

            context.update_progress(100, f"Knowledge learning '{operation}' completed successfully")
            
            # Store result in context
            context.add_artifact('knowledge_learning', {
                'operation': operation,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'result_summary': {
                    'learning_applied': result.get('learning_applied', False),
                    'adaptations_count': len(result.get('adaptations', [])),
                    'metrics': result.get('metrics', {})
                }
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge learning failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge learning failed: {str(e)}",
                details={
                    'operation': operation,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _process_feedback(self, inputs: Dict, context) -> Dict[str, Any]:
        """Process user feedback and update learning state."""
        context.update_progress(20, "Processing user feedback")

        feedback_type = inputs['feedback_type']
        entity_id = inputs.get('entity_id')
        triple_id = inputs.get('triple_id')
        correction = inputs.get('correction')
        notes = inputs.get('notes', '')
        source = inputs.get('source', 'user')

        learning_rate = float(inputs.get('learning_rate', self.config.get('learning_rate', 0.1)))
        adaptation_target = inputs.get('adaptation_target', self.config.get('adaptation_target', 'all'))

        # Record feedback
        feedback_record = {
            'id': f"fb_{datetime.now(timezone.utc).timestamp()}",
            'feedback_type': feedback_type,
            'entity_id': entity_id,
            'triple_id': triple_id,
            'correction': correction,
            'notes': notes,
            'source': source,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'learning_rate': learning_rate,
            'adaptation_target': adaptation_target
        }

        adaptations = []
        confidence_adjustments = {}

        # Use AdaptationEngine if available
        if self.adaptation_engine:
            try:
                # Record experience with the adaptation engine
                components = []
                if adaptation_target in ['extraction', 'all']:
                    components.append('extraction')
                if adaptation_target in ['query', 'all']:
                    components.append('query')
                if adaptation_target in ['reasoning', 'all']:
                    components.append('reasoning')

                success = feedback_type in ['positive', 'confirmation']
                
                # Run async operation
                asyncio.run(self.adaptation_engine.record_experience(
                    query=notes or f"feedback_{entity_id or triple_id}",
                    success=success,
                    processing_time_ms=0.0,
                    components_used=components,
                    error_type=None if success else feedback_type,
                    metadata=feedback_record
                ))

                adaptations.append({
                    'type': 'adaptation_engine',
                    'component': adaptation_target,
                    'action': 'record_experience',
                    'success': True
                })

            except Exception as e:
                self.logger.warning(f"AdaptationEngine feedback recording failed: {e}")
                adaptations.append({
                    'type': 'adaptation_engine',
                    'component': adaptation_target,
                    'action': 'record_experience',
                    'success': False,
                    'error': str(e)
                })

        # Fallback learning
        self._feedback_history.append(feedback_record)
        self._learning_metrics['total_feedback_received'] += 1

        if feedback_type == 'positive':
            self._learning_metrics['positive_feedback_count'] += 1
        elif feedback_type == 'negative':
            self._learning_metrics['negative_feedback_count'] += 1
        elif feedback_type == 'correction':
            self._learning_metrics['corrections_applied'] += 1

        # Update entity confidence
        if entity_id:
            confidence_adjustments[entity_id] = self._update_entity_confidence(
                entity_id, feedback_type, learning_rate
            )

        # Update triple confidence
        if triple_id:
            confidence_adjustments[triple_id] = self._update_triple_confidence(
                triple_id, feedback_type, learning_rate
            )

        # Apply correction if provided
        if correction and feedback_type == 'correction':
            self._apply_correction(correction, entity_id, triple_id)

        # Update user preferences based on feedback patterns
        self._update_user_preferences(feedback_record)

        self._learning_metrics['last_learning_at'] = datetime.now(timezone.utc).isoformat()

        context.update_progress(80, f"Feedback processed: {feedback_type}")

        return {
            'learning_applied': True,
            'operation': 'feedback',
            'feedback_record': feedback_record,
            'adaptations': adaptations,
            'confidence_adjustments': confidence_adjustments,
            'metrics': self._get_learning_metrics(),
            'fallback_used': self.adaptation_engine is None
        }

    def _update_entity_confidence(self, entity_id: str, feedback_type: str, learning_rate: float) -> Dict[str, Any]:
        """Update confidence score for an entity based on feedback."""
        if entity_id not in self._entity_confidence:
            self._entity_confidence[entity_id] = {
                'confidence': 0.7,
                'feedback_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'history': []
            }

        entity_data = self._entity_confidence[entity_id]
        old_confidence = entity_data['confidence']

        # Adjust confidence based on feedback type
        if feedback_type == 'positive':
            adjustment = learning_rate * (1.0 - old_confidence)
            entity_data['positive_count'] += 1
        elif feedback_type == 'negative':
            adjustment = -learning_rate * old_confidence
            entity_data['negative_count'] += 1
        elif feedback_type == 'confirmation':
            adjustment = learning_rate * 0.5 * (1.0 - old_confidence)
            entity_data['positive_count'] += 1
        else:  # correction
            adjustment = -learning_rate * 0.5 * old_confidence
            entity_data['negative_count'] += 1

        new_confidence = max(0.1, min(1.0, old_confidence + adjustment))
        entity_data['confidence'] = new_confidence
        entity_data['feedback_count'] += 1
        entity_data['history'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'feedback_type': feedback_type,
            'old_confidence': old_confidence,
            'new_confidence': new_confidence,
            'adjustment': adjustment
        })

        # Keep history manageable
        if len(entity_data['history']) > 100:
            entity_data['history'] = entity_data['history'][-100:]

        self._learning_metrics['confidence_adjustments'] += 1
        self._learning_metrics['entity_updates'] += 1

        return {
            'entity_id': entity_id,
            'old_confidence': old_confidence,
            'new_confidence': new_confidence,
            'adjustment': adjustment,
            'feedback_count': entity_data['feedback_count']
        }

    def _update_triple_confidence(self, triple_id: str, feedback_type: str, learning_rate: float) -> Dict[str, Any]:
        """Update confidence score for a triple based on feedback."""
        if triple_id not in self._triple_confidence:
            self._triple_confidence[triple_id] = {
                'confidence': 0.7,
                'feedback_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'history': []
            }

        triple_data = self._triple_confidence[triple_id]
        old_confidence = triple_data['confidence']

        # Adjust confidence based on feedback type
        if feedback_type == 'positive':
            adjustment = learning_rate * (1.0 - old_confidence)
            triple_data['positive_count'] += 1
        elif feedback_type == 'negative':
            adjustment = -learning_rate * old_confidence
            triple_data['negative_count'] += 1
        elif feedback_type == 'confirmation':
            adjustment = learning_rate * 0.5 * (1.0 - old_confidence)
            triple_data['positive_count'] += 1
        else:  # correction
            adjustment = -learning_rate * 0.5 * old_confidence
            triple_data['negative_count'] += 1

        new_confidence = max(0.1, min(1.0, old_confidence + adjustment))
        triple_data['confidence'] = new_confidence
        triple_data['feedback_count'] += 1
        triple_data['history'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'feedback_type': feedback_type,
            'old_confidence': old_confidence,
            'new_confidence': new_confidence,
            'adjustment': adjustment
        })

        # Keep history manageable
        if len(triple_data['history']) > 100:
            triple_data['history'] = triple_data['history'][-100:]

        self._learning_metrics['confidence_adjustments'] += 1

        return {
            'triple_id': triple_id,
            'old_confidence': old_confidence,
            'new_confidence': new_confidence,
            'adjustment': adjustment,
            'feedback_count': triple_data['feedback_count']
        }

    def _apply_correction(self, correction: Dict, entity_id: Optional[str], triple_id: Optional[str]):
        """Apply a correction to entity or triple data."""
        correction_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entity_id': entity_id,
            'triple_id': triple_id,
            'correction_data': correction
        }

        # Store correction for future reference
        if 'corrections' not in self._user_preferences:
            self._user_preferences['corrections'] = []
        self._user_preferences['corrections'].append(correction_record)

        # Update entity profile if correction is for an entity
        if entity_id and 'entity_data' in correction:
            self._update_entity_from_correction(entity_id, correction['entity_data'])

        self.logger.info(f"Correction applied for entity={entity_id}, triple={triple_id}")

    def _update_entity_from_correction(self, entity_id: str, entity_data: Dict):
        """Update entity profile based on correction data."""
        if entity_id in self._entity_confidence:
            self._entity_confidence[entity_id].update(entity_data)
            self._entity_confidence[entity_id]['last_corrected'] = datetime.now(timezone.utc).isoformat()

    def _update_user_preferences(self, feedback_record: Dict):
        """Update user preferences based on feedback patterns."""
        # Track preferred sources based on positive feedback
        if feedback_record['feedback_type'] in ['positive', 'confirmation']:
            source = feedback_record.get('source')
            if source and source not in self._user_preferences['preferred_sources']:
                self._user_preferences['preferred_sources'].append(source)

        # Update confidence threshold based on feedback patterns
        if self._learning_metrics['total_feedback_received'] > 10:
            pos_ratio = (
                self._learning_metrics['positive_feedback_count'] /
                self._learning_metrics['total_feedback_received']
            )
            # Adjust threshold based on positive feedback ratio
            if pos_ratio > 0.8:
                self._user_preferences['confidence_threshold'] = min(
                    0.9, self._user_preferences['confidence_threshold'] + 0.05
                )
            elif pos_ratio < 0.3:
                self._user_preferences['confidence_threshold'] = max(
                    0.5, self._user_preferences['confidence_threshold'] - 0.05
                )

    def _apply_adaptations(self, inputs: Dict, context) -> Dict[str, Any]:
        """Apply learned adaptations to system behavior."""
        context.update_progress(20, "Applying learned adaptations")

        adaptation_target = inputs.get('adaptation_target', self.config.get('adaptation_target', 'all'))
        adaptations_applied = []

        # Use AdaptationEngine if available
        if self.adaptation_engine:
            try:
                # Get adaptation suggestions
                suggestions = asyncio.run(self.adaptation_engine.suggest_adaptations())
                
                # Apply high-confidence adaptations
                for suggestion in suggestions:
                    if suggestion.confidence > 0.7:
                        asyncio.run(self.adaptation_engine.apply_adaptation(suggestion))
                        adaptations_applied.append({
                            'type': 'adaptation_engine',
                            'component': suggestion.target_component,
                            'action': suggestion.action,
                            'reason': suggestion.reason,
                            'confidence': suggestion.confidence
                        })

            except Exception as e:
                self.logger.warning(f"AdaptationEngine adaptation failed: {e}")

        # Apply fallback adaptations
        fallback_adaptations = self._apply_fallback_adaptations(adaptation_target)
        adaptations_applied.extend(fallback_adaptations)

        self._adaptations.extend(adaptations_applied)

        context.update_progress(80, f"Applied {len(adaptations_applied)} adaptations")

        return {
            'learning_applied': len(adaptations_applied) > 0,
            'operation': 'adapt',
            'adaptations': adaptations_applied,
            'adaptation_target': adaptation_target,
            'confidence_adjustments': self._get_confidence_adjustments_summary(),
            'metrics': self._get_learning_metrics(),
            'fallback_used': self.adaptation_engine is None
        }

    def _apply_fallback_adaptations(self, target: str) -> List[Dict[str, Any]]:
        """Apply adaptations using fallback learning mechanism."""
        adaptations = []

        # Analyze entity confidence patterns
        if target in ['extraction', 'all']:
            low_confidence_entities = [
                eid for eid, data in self._entity_confidence.items()
                if data['confidence'] < 0.5
            ]
            if low_confidence_entities:
                adaptations.append({
                    'type': 'fallback',
                    'component': 'extraction',
                    'action': 'exclude_low_confidence_entities',
                    'affected_entities': low_confidence_entities[:10],  # Limit list
                    'reason': f'Found {len(low_confidence_entities)} entities with confidence < 0.5'
                })

        # Analyze triple confidence patterns
        if target in ['query', 'all']:
            high_confidence_triples = [
                tid for tid, data in self._triple_confidence.items()
                if data['confidence'] > 0.9
            ]
            if high_confidence_triples:
                adaptations.append({
                    'type': 'fallback',
                    'component': 'query',
                    'action': 'prioritize_high_confidence_triples',
                    'affected_triples': len(high_confidence_triples),
                    'reason': f'Found {len(high_confidence_triples)} high-confidence triples to prioritize'
                })

        # Adjust confidence threshold based on feedback
        if target in ['reasoning', 'all']:
            if self._learning_metrics['total_feedback_received'] > 20:
                pos_ratio = (
                    self._learning_metrics['positive_feedback_count'] /
                    self._learning_metrics['total_feedback_received']
                )
                old_threshold = self._user_preferences['confidence_threshold']
                
                if pos_ratio > 0.7:
                    new_threshold = min(0.9, old_threshold + 0.05)
                elif pos_ratio < 0.4:
                    new_threshold = max(0.5, old_threshold - 0.05)
                else:
                    new_threshold = old_threshold

                if new_threshold != old_threshold:
                    self._user_preferences['confidence_threshold'] = new_threshold
                    adaptations.append({
                        'type': 'fallback',
                        'component': 'reasoning',
                        'action': 'adjust_confidence_threshold',
                        'old_threshold': old_threshold,
                        'new_threshold': new_threshold,
                        'reason': f'Adjusted based on {pos_ratio:.2%} positive feedback ratio'
                    })

        return adaptations

    def _trigger_improvements(self, inputs: Dict, context) -> Dict[str, Any]:
        """Trigger active learning improvements."""
        context.update_progress(20, "Triggering learning improvements")

        adaptation_target = inputs.get('adaptation_target', self.config.get('adaptation_target', 'all'))
        improvements = []

        # Use AdaptationEngine if available
        if self.adaptation_engine:
            try:
                summary = asyncio.run(self.adaptation_engine.get_learning_summary())
                
                # Identify underperforming components
                for name, profile in summary.get('component_performance', {}).items():
                    if profile.get('average_success_rate', 1.0) < 0.7:
                        improvements.append({
                            'type': 'adaptation_engine',
                            'component': name,
                            'action': 'improve_component',
                            'current_success_rate': profile.get('average_success_rate'),
                            'recommendation': 'Review and optimize component implementation'
                        })

            except Exception as e:
                self.logger.warning(f"AdaptationEngine improvement analysis failed: {e}")

        # Fallback improvements
        # Analyze feedback patterns to suggest improvements
        if self._learning_metrics['total_feedback_received'] > 0:
            pos_ratio = (
                self._learning_metrics['positive_feedback_count'] /
                self._learning_metrics['total_feedback_received']
            )
            
            if pos_ratio < 0.5:
                improvements.append({
                    'type': 'fallback',
                    'component': adaptation_target,
                    'action': 'review_extraction_quality',
                    'current_positive_ratio': pos_ratio,
                    'recommendation': 'High negative feedback rate suggests extraction quality issues'
                })

        # Suggest entity profile updates based on corrections
        correction_count = self._learning_metrics['corrections_applied']
        if correction_count > 5:
            improvements.append({
                'type': 'fallback',
                'component': 'entity_profiles',
                'action': 'review_corrected_entities',
                'correction_count': correction_count,
                'recommendation': f'{correction_count} corrections suggest need for entity profile updates'
            })

        context.update_progress(80, f"Identified {len(improvements)} improvement opportunities")

        return {
            'learning_applied': len(improvements) > 0,
            'operation': 'improve',
            'adaptations': improvements,
            'improvement_count': len(improvements),
            'confidence_adjustments': self._get_confidence_adjustments_summary(),
            'metrics': self._get_learning_metrics(),
            'fallback_used': self.adaptation_engine is None
        }

    def _analyze_learning(self, inputs: Dict, context) -> Dict[str, Any]:
        """Analyze learning history and generate metrics."""
        context.update_progress(20, "Analyzing learning history")

        # Get AdaptationEngine summary if available
        adaptation_summary = None
        if self.adaptation_engine:
            try:
                adaptation_summary = asyncio.run(self.adaptation_engine.get_learning_summary())
            except Exception as e:
                self.logger.warning(f"Could not get AdaptationEngine summary: {e}")

        # Build analysis results
        analysis = {
            'learning_applied': True,
            'operation': 'analyze_learning',
            'metrics': self._get_learning_metrics(),
            'feedback_statistics': self._get_feedback_statistics(),
            'entity_confidence_summary': self._get_entity_confidence_summary(),
            'triple_confidence_summary': self._get_triple_confidence_summary(),
            'user_preferences': self._get_user_preferences_summary(),
            'adaptations_history': self._adaptations[-20:] if self._adaptations else [],  # Last 20
            'adaptation_engine_summary': adaptation_summary,
            'fallback_used': self.adaptation_engine is None
        }

        context.update_progress(80, "Learning analysis complete")

        return analysis

    def _reset_learning(self, inputs: Dict, context) -> Dict[str, Any]:
        """Reset learning state."""
        context.update_progress(20, "Resetting learning state")

        reset_scope = inputs.get('scope', 'all')  # 'all', 'feedback', 'confidence', 'preferences'

        reset_actions = []

        # Reset AdaptationEngine if available
        if self.adaptation_engine and reset_scope in ['all', 'feedback']:
            try:
                # Note: AdaptationEngine doesn't have a reset method in current implementation
                # This is a placeholder for future enhancement
                reset_actions.append({'component': 'adaptation_engine', 'action': 'reset_requested'})
            except Exception as e:
                self.logger.warning(f"Could not reset AdaptationEngine: {e}")

        # Reset fallback learning
        if reset_scope in ['all', 'feedback']:
            old_count = len(self._feedback_history)
            self._feedback_history.clear()
            reset_actions.append({'component': 'feedback_history', 'action': 'cleared', 'count': old_count})

        if reset_scope in ['all', 'confidence']:
            entity_count = len(self._entity_confidence)
            triple_count = len(self._triple_confidence)
            self._entity_confidence.clear()
            self._triple_confidence.clear()
            reset_actions.append({'component': 'entity_confidence', 'action': 'cleared', 'count': entity_count})
            reset_actions.append({'component': 'triple_confidence', 'action': 'cleared', 'count': triple_count})

        if reset_scope in ['all', 'preferences']:
            self._user_preferences = {
                'preferred_sources': [],
                'confidence_threshold': 0.7,
                'excluded_entities': set(),
                'preferred_relationships': []
            }
            reset_actions.append({'component': 'user_preferences', 'action': 'reset_to_defaults'})

        if reset_scope == 'all':
            # Reset metrics
            self._learning_metrics = {
                'total_feedback_received': 0,
                'positive_feedback_count': 0,
                'negative_feedback_count': 0,
                'corrections_applied': 0,
                'confidence_adjustments': 0,
                'entity_updates': 0,
                'last_learning_at': None
            }
            self._adaptations.clear()
            reset_actions.append({'component': 'learning_metrics', 'action': 'reset'})

        context.update_progress(80, f"Learning reset complete: {reset_scope}")

        return {
            'learning_applied': True,
            'operation': 'reset',
            'reset_scope': reset_scope,
            'reset_actions': reset_actions,
            'adaptations': [],
            'confidence_adjustments': {},
            'metrics': self._get_learning_metrics(),
            'fallback_used': self.adaptation_engine is None
        }

    def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics."""
        metrics = self._learning_metrics.copy()
        metrics['entity_tracked'] = len(self._entity_confidence)
        metrics['triples_tracked'] = len(self._triple_confidence)
        metrics['adaptations_total'] = len(self._adaptations)
        metrics['feedback_history_size'] = len(self._feedback_history)
        return metrics

    def _get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about feedback history."""
        if not self._feedback_history:
            return {'total': 0}

        feedback_counts = defaultdict(int)
        source_counts = defaultdict(int)
        target_counts = defaultdict(int)

        for fb in self._feedback_history:
            feedback_counts[fb['feedback_type']] += 1
            source_counts[fb['source']] += 1
            if fb['entity_id']:
                target_counts['entity'] += 1
            if fb['triple_id']:
                target_counts['triple'] += 1

        # Recent feedback (last 10)
        recent = list(self._feedback_history)[-10:]

        return {
            'total': len(self._feedback_history),
            'by_type': dict(feedback_counts),
            'by_source': dict(source_counts),
            'by_target': dict(target_counts),
            'recent_feedback': recent
        }

    def _get_entity_confidence_summary(self) -> Dict[str, Any]:
        """Get summary of entity confidence data."""
        if not self._entity_confidence:
            return {'count': 0}

        confidences = [d['confidence'] for d in self._entity_confidence.values()]
        high_conf = sum(1 for c in confidences if c >= 0.9)
        medium_conf = sum(1 for c in confidences if 0.7 <= c < 0.9)
        low_conf = sum(1 for c in confidences if c < 0.7)

        return {
            'count': len(self._entity_confidence),
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf,
            'top_entities': sorted(
                [(eid, d['confidence']) for eid, d in self._entity_confidence.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def _get_triple_confidence_summary(self) -> Dict[str, Any]:
        """Get summary of triple confidence data."""
        if not self._triple_confidence:
            return {'count': 0}

        confidences = [d['confidence'] for d in self._triple_confidence.values()]
        high_conf = sum(1 for c in confidences if c >= 0.9)
        medium_conf = sum(1 for c in confidences if 0.7 <= c < 0.9)
        low_conf = sum(1 for c in confidences if c < 0.7)

        return {
            'count': len(self._triple_confidence),
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf
        }

    def _get_user_preferences_summary(self) -> Dict[str, Any]:
        """Get summary of user preferences."""
        prefs = self._user_preferences.copy()
        # Convert set to list for JSON serialization
        if 'excluded_entities' in prefs:
            prefs['excluded_entities'] = list(prefs['excluded_entities'])
        if 'corrections' in prefs:
            prefs['corrections'] = len(prefs['corrections'])
        return prefs

    def _get_confidence_adjustments_summary(self) -> Dict[str, Any]:
        """Get summary of confidence adjustments."""
        return {
            'entity_adjustments': len(self._entity_confidence),
            'triple_adjustments': len(self._triple_confidence),
            'total_adjustments': self._learning_metrics['confidence_adjustments']
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns:
            JSON schema dictionary for UI configuration
        """
        return {
            "type": "object",
            "title": "Knowledge Learning Configuration",
            "description": "Configure knowledge learning and adaptation parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of learning operation to perform",
                    "enum": ["feedback", "adapt", "improve", "analyze_learning", "reset"],
                    "enumNames": [
                        "Feedback - Record user feedback on results",
                        "Adapt - Apply learned adaptations",
                        "Improve - Trigger active learning improvements",
                        "Analyze Learning - Analyze learning history and metrics",
                        "Reset - Reset learning state"
                    ],
                    "default": "feedback"
                },
                "feedback_type": {
                    "type": "string",
                    "title": "Feedback Type",
                    "description": "Type of user feedback (for feedback operation)",
                    "enum": ["positive", "negative", "correction", "confirmation"],
                    "enumNames": [
                        "Positive - Result was good/correct",
                        "Negative - Result was bad/incorrect",
                        "Correction - Provide corrected data",
                        "Confirmation - Confirm result is correct"
                    ],
                    "default": "positive"
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "ID of the entity being rated/feedback on",
                    "default": ""
                },
                "triple_id": {
                    "type": "string",
                    "title": "Triple ID",
                    "description": "ID of the triple being rated/feedback on",
                    "default": ""
                },
                "correction": {
                    "type": "object",
                    "title": "Correction Data",
                    "description": "Corrected data if feedback type is 'correction'",
                    "additionalProperties": True,
                    "default": {}
                },
                "learning_rate": {
                    "type": "number",
                    "title": "Learning Rate",
                    "description": "Rate at which to adjust confidence scores (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1
                },
                "adaptation_target": {
                    "type": "string",
                    "title": "Adaptation Target",
                    "description": "Which system component to adapt",
                    "enum": ["extraction", "query", "reasoning", "all"],
                    "enumNames": [
                        "Extraction - Adapt knowledge extraction",
                        "Query - Adapt query processing",
                        "Reasoning - Adapt reasoning/inference",
                        "All - Adapt all components"
                    ],
                    "default": "all"
                },
                "feedback_history_limit": {
                    "type": "integer",
                    "title": "Feedback History Limit",
                    "description": "Maximum number of feedback records to retain",
                    "minimum": 100,
                    "maximum": 10000,
                    "default": 1000
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["feedback"]}
                            },
                            "required": ["feedback_type"],
                            "description": "Provide feedback on extraction or query results"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["adapt"]}
                            },
                            "description": "Apply learned adaptations to the system"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["improve"]}
                            },
                            "description": "Trigger active learning improvements"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["analyze_learning"]}
                            },
                            "description": "Analyze learning history and generate metrics"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["reset"]}
                            },
                            "description": "Reset learning state"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (always returns True as fallback is available)
        """
        try:
            # Basic health check - ensure fallback learning is initialized
            return (
                hasattr(self, '_feedback_history') and
                hasattr(self, '_entity_confidence') and
                hasattr(self, '_triple_confidence')
            )
        except Exception:
            return False

    def get_learning_state(self) -> Dict[str, Any]:
        """
        Get the current learning state for persistence.

        Returns:
            Dictionary containing all learning state data
        """
        return {
            'feedback_history': list(self._feedback_history),
            'entity_confidence': self._entity_confidence,
            'triple_confidence': self._triple_confidence,
            'user_preferences': {
                **self._user_preferences,
                'excluded_entities': list(self._user_preferences.get('excluded_entities', set()))
            },
            'learning_metrics': self._learning_metrics,
            'adaptations': self._adaptations,
            'adaptation_engine_available': self.adaptation_engine is not None,
            'exported_at': datetime.now(timezone.utc).isoformat()
        }

    def restore_learning_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore learning state from persisted data.

        Args:
            state: Dictionary containing learning state data

        Returns:
            True if restore was successful
        """
        try:
            if 'feedback_history' in state:
                self._feedback_history = deque(
                    state['feedback_history'],
                    maxlen=self.config.get('feedback_history_limit', 1000)
                )
            if 'entity_confidence' in state:
                self._entity_confidence = state['entity_confidence']
            if 'triple_confidence' in state:
                self._triple_confidence = state['triple_confidence']
            if 'user_preferences' in state:
                self._user_preferences = state['user_preferences']
                # Convert list back to set
                if 'excluded_entities' in self._user_preferences:
                    self._user_preferences['excluded_entities'] = set(
                        self._user_preferences['excluded_entities']
                    )
            if 'learning_metrics' in state:
                self._learning_metrics = state['learning_metrics']
            if 'adaptations' in state:
                self._adaptations = state['adaptations']

            self.logger.info("Learning state restored successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore learning state: {e}")
            return False
