"""
Recommendation Engine Node for BubbleLabs Integration

Provides intelligent recommendation capabilities:
- Recommend related entities based on knowledge graph relationships
- Suggest next actions based on context and workflow state
- Recommend knowledge to explore based on interests
- Personalized recommendations using user history
- Collaborative filtering support for recommendations
- Explainable recommendations with reasoning

This node integrates with the Unified Knowledge Graph and NeuralKG
to provide ML-powered recommendations with fallback to rule-based methods.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone
from collections import defaultdict
import math
import random

from .base_node import BubbleLabsNode, NodeExecutionError


class RecommendationEngineNode(BubbleLabsNode):
    """
    Smart recommendation engine for entities, actions, and knowledge exploration.

    Supports five operations:
    - recommend_entities: Find related/similar entities in the knowledge graph
    - suggest_actions: Recommend next actions based on context
    - explore_knowledge: Suggest knowledge areas to explore
    - personalized: Generate personalized recommendations based on user history
    - collaborative: Use collaborative filtering for recommendations

    All recommendations include confidence scores and explanations.
    """

    # Node metadata
    DISPLAY_NAME = "Recommendation Engine"
    DESCRIPTION = "Smart recommendations for entities, actions, and knowledge exploration"
    ICON = "recommendation"
    CATEGORY = "intelligence"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports for knowledge engine integration
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for RecommendationEngineNode"
        )

        self.NeuralKGIntegration = self.safe_import(
            'knowledge_engine.integrations.neuralkg_integration.NeuralKGIntegration',
            fallback_value=None,
            error_msg="NeuralKGIntegration not available for RecommendationEngineNode"
        )

        # Initialize instances if available
        self.kg_hub_instance = None
        self.neuralkg_instance = None
        self._embeddings_cache = {}
        self._user_history_cache = {}

        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub_instance = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for RecommendationEngineNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub_instance = None

        if self.NeuralKGIntegration:
            try:
                self.neuralkg_instance = self.NeuralKGIntegration()
                self.logger.info("NeuralKGIntegration initialized for RecommendationEngineNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize NeuralKGIntegration: {e}")
                self.neuralkg_instance = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields vary by operation:
        - recommend_entities: Requires 'entity_id' in inputs or config
        - suggest_actions: Requires 'context' with current workflow state
        - explore_knowledge: Requires 'entity_id' or 'context'
        - personalized: Requires 'user_id' for personalization
        - collaborative: Requires 'entity_id' or 'user_id'
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation')
        if operation is None:
            operation = self.config.get('operation')

        if operation is None:
            errors.append("Missing required field: operation (must be 'recommend_entities', 'suggest_actions', 'explore_knowledge', 'personalized', or 'collaborative')")
            return errors

        valid_operations = ['recommend_entities', 'suggest_actions', 'explore_knowledge', 'personalized', 'collaborative']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Validate operation-specific requirements
        if operation == 'recommend_entities':
            entity_id = inputs.get('entity_id') or self.config.get('entity_id')
            if not entity_id:
                errors.append("Recommend entities operation requires 'entity_id' (in inputs or config)")

        elif operation == 'suggest_actions':
            context = inputs.get('context') or self.config.get('context')
            if not context:
                self.logger.warning("No context provided for suggest_actions, will use minimal defaults")

        elif operation == 'explore_knowledge':
            entity_id = inputs.get('entity_id') or self.config.get('entity_id')
            context = inputs.get('context') or self.config.get('context')
            if not entity_id and not context:
                errors.append("Explore knowledge operation requires 'entity_id' or 'context' (in inputs or config)")

        elif operation == 'personalized':
            user_id = inputs.get('user_id') or self.config.get('user_id')
            if not user_id:
                errors.append("Personalized operation requires 'user_id' (in inputs or config)")

        elif operation == 'collaborative':
            entity_id = inputs.get('entity_id') or self.config.get('entity_id')
            user_id = inputs.get('user_id') or self.config.get('user_id')
            if not entity_id and not user_id:
                errors.append("Collaborative operation requires 'entity_id' or 'user_id' (in inputs or config)")

        # Validate numeric parameters if provided
        if 'max_recommendations' in inputs:
            try:
                max_rec = int(inputs['max_recommendations'])
                if max_rec < 1:
                    errors.append("max_recommendations must be at least 1")
                if max_rec > 100:
                    errors.append("max_recommendations must not exceed 100")
            except (ValueError, TypeError):
                errors.append("max_recommendations must be an integer")

        if 'min_confidence' in inputs:
            try:
                confidence = float(inputs['min_confidence'])
                if not (0.0 <= confidence <= 1.0):
                    errors.append("min_confidence must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("min_confidence must be a number")

        if 'diversity_factor' in inputs:
            try:
                diversity = float(inputs['diversity_factor'])
                if not (0.0 <= diversity <= 1.0):
                    errors.append("diversity_factor must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("diversity_factor must be a number")

        # Validate recommendation_type if provided
        if 'recommendation_type' in inputs:
            valid_types = ['similar', 'related', 'complementary', 'trending']
            if inputs['recommendation_type'] not in valid_types:
                errors.append(f"Invalid recommendation_type: {inputs['recommendation_type']}. Must be one of: {', '.join(valid_types)}")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the recommendation operation.

        Args:
            inputs: Operation specification including operation type and parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing recommendation results:
                - recommendations: List of recommended items
                - explanations: Explanations for each recommendation
                - confidence_scores: Confidence scores for each recommendation
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If operation execution fails
        """
        # Get operation parameters
        operation = inputs.get('operation', self.config.get('operation'))
        entity_id = inputs.get('entity_id') or self.config.get('entity_id')
        user_id = inputs.get('user_id') or self.config.get('user_id')
        recommendation_type = inputs.get('recommendation_type', self.config.get('recommendation_type', 'similar'))
        ctx = inputs.get('context') or self.config.get('context') or {}
        max_recommendations = int(inputs.get('max_recommendations', self.config.get('max_recommendations', 10)))
        min_confidence = float(inputs.get('min_confidence', self.config.get('min_confidence', 0.6)))
        include_explanation = bool(inputs.get('include_explanation', self.config.get('include_explanation', True)))
        diversity_factor = float(inputs.get('diversity_factor', self.config.get('diversity_factor', 0.3)))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Executing recommendation operation: {operation}")

        try:
            # Get knowledge graph data
            kg_data = self._get_knowledge_graph_data(inputs, context)

            context.update_progress(30, "Processing data")

            # Execute the appropriate operation
            if operation == 'recommend_entities':
                result = self._recommend_entities(
                    entity_id, kg_data, recommendation_type, max_recommendations,
                    min_confidence, include_explanation, diversity_factor, context
                )
            elif operation == 'suggest_actions':
                result = self._suggest_actions(
                    ctx, kg_data, max_recommendations, min_confidence,
                    include_explanation, context
                )
            elif operation == 'explore_knowledge':
                result = self._explore_knowledge(
                    entity_id, ctx, kg_data, max_recommendations,
                    min_confidence, include_explanation, context
                )
            elif operation == 'personalized':
                result = self._personalized_recommendations(
                    user_id, ctx, kg_data, recommendation_type, max_recommendations,
                    min_confidence, include_explanation, diversity_factor, context
                )
            elif operation == 'collaborative':
                result = self._collaborative_filtering(
                    user_id, entity_id, kg_data, max_recommendations,
                    min_confidence, include_explanation, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['recommend_entities', 'suggest_actions', 'explore_knowledge', 'personalized', 'collaborative']}
                )

            context.update_progress(90, "Processing results")

            # Add execution metadata
            result['metadata'] = {
                'operation': operation,
                'executed_at': datetime.now(timezone.utc).isoformat(),
                'execution_id': self.execution_id,
                'parameters': {
                    'entity_id': entity_id,
                    'user_id': user_id,
                    'recommendation_type': recommendation_type,
                    'max_recommendations': max_recommendations,
                    'min_confidence': min_confidence,
                    'include_explanation': include_explanation,
                    'diversity_factor': diversity_factor
                },
                'kg_hub_available': self.kg_hub_instance is not None,
                'neuralkg_available': self.neuralkg_instance is not None and (
                    hasattr(self.neuralkg_instance, 'is_available') and 
                    self.neuralkg_instance.is_available() if hasattr(self.neuralkg_instance, 'is_available') else False
                ),
                'recommendation_count': len(result.get('recommendations', []))
            }

            context.update_progress(100, f"Operation complete: {len(result.get('recommendations', []))} recommendations")
            self.logger.info(f"Recommendation operation completed: {len(result.get('recommendations', []))} recommendations")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Recommendation failed: {str(e)}",
                details={
                    'operation': operation,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_knowledge_graph_data(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Get knowledge graph data from inputs or context.

        Priority:
        1. kg_data from inputs
        2. From UnifiedKGIntegrationHub
        3. Sample/test data
        4. Empty structure
        """
        # Check if knowledge graph data was provided in inputs
        if 'kg_data' in inputs:
            return inputs['kg_data']

        # Try to get from UnifiedKGIntegrationHub
        if self.kg_hub_instance:
            try:
                if hasattr(self.kg_hub_instance, 'get_all_entities'):
                    entities = self.kg_hub_instance.get_all_entities()
                    triples = self.kg_hub_instance.get_all_triples() if hasattr(self.kg_hub_instance, 'get_all_triples') else []
                    return {
                        'entities': entities,
                        'triples': triples
                    }
            except Exception as e:
                self.logger.warning(f"Could not get data from UnifiedKGIntegrationHub: {e}")

        # Return empty structure
        return {
            'entities': inputs.get('entities', []),
            'triples': inputs.get('triples', [])
        }

    def _recommend_entities(
        self,
        entity_id: str,
        kg_data: Dict[str, Any],
        recommendation_type: str,
        max_recommendations: int,
        min_confidence: float,
        include_explanation: bool,
        diversity_factor: float,
        context
    ) -> Dict[str, Any]:
        """Recommend entities based on the specified type."""
        context.update_progress(40, f"Finding {recommendation_type} entities for: {entity_id}")

        triples = kg_data.get('triples', [])

        if recommendation_type == 'similar':
            recommendations = self._find_similar_entities(entity_id, kg_data, max_recommendations, min_confidence)
        elif recommendation_type == 'related':
            recommendations = self._find_related_entities(entity_id, triples, max_recommendations, min_confidence)
        elif recommendation_type == 'complementary':
            recommendations = self._find_complementary_entities(entity_id, triples, max_recommendations, min_confidence)
        elif recommendation_type == 'trending':
            recommendations = self._find_trending_entities(kg_data, max_recommendations, min_confidence)
        else:
            # Default: combine related and similar
            related = self._find_related_entities(entity_id, triples, max_recommendations // 2, min_confidence)
            similar = self._find_similar_entities(entity_id, kg_data, max_recommendations // 2, min_confidence)
            recommendations = self._merge_recommendations(related, similar, max_recommendations, diversity_factor)

        # Generate explanations if requested
        explanations = []
        if include_explanation:
            explanations = self._generate_explanations(recommendations, entity_id, recommendation_type)

        # Apply diversity if requested
        if diversity_factor > 0 and len(recommendations) > 1:
            recommendations = self._apply_diversity(recommendations, diversity_factor)

        return {
            'recommendations': recommendations[:max_recommendations],
            'explanations': explanations[:max_recommendations],
            'confidence_scores': [r.get('confidence', 0.0) for r in recommendations[:max_recommendations]],
            'source_entity': entity_id,
            'recommendation_type': recommendation_type,
            'method': 'graph_based' if not self.neuralkg_instance else 'hybrid'
        }

    def _suggest_actions(
        self,
        ctx: Dict,
        kg_data: Dict[str, Any],
        max_recommendations: int,
        min_confidence: float,
        include_explanation: bool,
        context
    ) -> Dict[str, Any]:
        """Suggest next actions based on context and workflow state."""
        context.update_progress(40, "Analyzing context for action suggestions")

        current_state = ctx.get('current_state', 'initial')
        previous_actions = ctx.get('previous_actions', [])
        available_actions = ctx.get('available_actions', [])
        goals = ctx.get('goals', [])

        recommendations = []

        # If available actions provided, rank them
        if available_actions:
            recommendations = self._rank_available_actions(
                available_actions, current_state, previous_actions, goals, kg_data, min_confidence
            )
        else:
            # Generate action suggestions based on state
            recommendations = self._generate_action_suggestions(
                current_state, previous_actions, goals, kg_data, min_confidence
            )

        # Generate explanations if requested
        explanations = []
        if include_explanation:
            explanations = self._generate_action_explanations(recommendations, current_state, previous_actions)

        return {
            'recommendations': recommendations[:max_recommendations],
            'explanations': explanations[:max_recommendations],
            'confidence_scores': [r.get('confidence', 0.0) for r in recommendations[:max_recommendations]],
            'current_state': current_state,
            'method': 'context_based'
        }

    def _explore_knowledge(
        self,
        entity_id: Optional[str],
        ctx: Dict,
        kg_data: Dict[str, Any],
        max_recommendations: int,
        min_confidence: float,
        include_explanation: bool,
        context
    ) -> Dict[str, Any]:
        """Recommend knowledge areas to explore."""
        context.update_progress(40, "Identifying knowledge exploration opportunities")

        interests = ctx.get('interests', [])
        explored_topics = ctx.get('explored_topics', [])
        knowledge_gaps = ctx.get('knowledge_gaps', [])

        recommendations = []

        # If knowledge gaps are specified, prioritize them
        if knowledge_gaps:
            for gap in knowledge_gaps:
                recommendations.append({
                    'entity': gap,
                    'type': 'knowledge_gap',
                    'confidence': 0.9,
                    'reason': 'Identified knowledge gap'
                })

        # Find related knowledge areas
        if entity_id:
            related = self._find_related_entities(entity_id, kg_data.get('triples', []), max_recommendations // 2, min_confidence)
            for rec in related:
                rec['type'] = 'related_knowledge'
                if rec.get('entity') not in [r.get('entity') for r in recommendations]:
                    recommendations.append(rec)

        # Suggest based on interests
        if interests and self.kg_hub_instance:
            try:
                for interest in interests:
                    similar = self._find_similar_entities(interest, kg_data, 3, min_confidence)
                    for rec in similar:
                        rec['type'] = 'interest_based'
                        if rec.get('entity') not in [r.get('entity') for r in recommendations]:
                            recommendations.append(rec)
            except Exception as e:
                self.logger.warning(f"Could not get interest-based recommendations: {e}")

        # Filter out already explored topics
        recommendations = [
            r for r in recommendations 
            if r.get('entity') not in explored_topics
        ]

        # Generate explanations if requested
        explanations = []
        if include_explanation:
            explanations = self._generate_knowledge_explanations(recommendations, entity_id, interests)

        return {
            'recommendations': recommendations[:max_recommendations],
            'explanations': explanations[:max_recommendations],
            'confidence_scores': [r.get('confidence', 0.0) for r in recommendations[:max_recommendations]],
            'starting_entity': entity_id,
            'method': 'exploration_based'
        }

    def _personalized_recommendations(
        self,
        user_id: str,
        ctx: Dict,
        kg_data: Dict[str, Any],
        recommendation_type: str,
        max_recommendations: int,
        min_confidence: float,
        include_explanation: bool,
        diversity_factor: float,
        context
    ) -> Dict[str, Any]:
        """Generate personalized recommendations based on user history."""
        context.update_progress(40, f"Generating personalized recommendations for user: {user_id}")

        # Get user history
        user_history = self._get_user_history(user_id, ctx)
        viewed_entities = user_history.get('viewed_entities', [])
        preferred_types = user_history.get('preferred_types', [])
        interaction_patterns = user_history.get('interaction_patterns', [])

        recommendations = []
        entity_scores = defaultdict(float)

        # Score entities based on user history
        for entity in viewed_entities:
            # Find similar entities
            similar = self._find_similar_entities(entity, kg_data, 5, min_confidence * 0.8)
            for sim in similar:
                entity_scores[sim['entity']] += sim.get('confidence', 0.5) * 0.7

            # Find related entities
            related = self._find_related_entities(entity, kg_data.get('triples', []), 5, min_confidence * 0.8)
            for rel in related:
                entity_scores[rel['entity']] += rel.get('confidence', 0.5) * 0.5

        # Boost scores for preferred types
        if preferred_types:
            for entity, score in entity_scores.items():
                entity_type = self._get_entity_type(entity, kg_data)
                if entity_type in preferred_types:
                    entity_scores[entity] *= 1.3

        # Convert to recommendations
        for entity, score in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= min_confidence and entity not in viewed_entities:
                recommendations.append({
                    'entity': entity,
                    'confidence': round(min(score, 1.0), 4),
                    'type': recommendation_type,
                    'personalized': True
                })

        # Apply diversity
        if diversity_factor > 0 and len(recommendations) > 1:
            recommendations = self._apply_diversity(recommendations, diversity_factor)

        # Generate explanations if requested
        explanations = []
        if include_explanation:
            explanations = self._generate_personalized_explanations(
                recommendations, user_id, viewed_entities, preferred_types
            )

        return {
            'recommendations': recommendations[:max_recommendations],
            'explanations': explanations[:max_recommendations],
            'confidence_scores': [r.get('confidence', 0.0) for r in recommendations[:max_recommendations]],
            'user_id': user_id,
            'based_on_history': len(viewed_entities) > 0,
            'method': 'personalized'
        }

    def _collaborative_filtering(
        self,
        user_id: Optional[str],
        entity_id: Optional[str],
        kg_data: Dict[str, Any],
        max_recommendations: int,
        min_confidence: float,
        include_explanation: bool,
        context
    ) -> Dict[str, Any]:
        """Use collaborative filtering for recommendations."""
        context.update_progress(40, "Applying collaborative filtering")

        # Get user interactions from context or simulate
        user_interactions = self._get_collaborative_data(user_id, entity_id, kg_data)

        recommendations = []

        if user_id and user_id in user_interactions:
            # Find similar users
            similar_users = self._find_similar_users(user_id, user_interactions)

            # Get entities liked by similar users but not by current user
            current_user_entities = set(user_interactions[user_id])

            for similar_user, similarity in similar_users:
                for entity in user_interactions.get(similar_user, []):
                    if entity not in current_user_entities:
                        score = similarity * 0.8 + 0.2  # Base score with similarity weight
                        recommendations.append({
                            'entity': entity,
                            'confidence': round(min(score, 1.0), 4),
                            'type': 'collaborative',
                            'source_users': [similar_user]
                        })
        elif entity_id:
            # Item-based collaborative filtering
            similar_items = self._find_similar_items(entity_id, user_interactions)
            for item, score in similar_items:
                if score >= min_confidence:
                    recommendations.append({
                        'entity': item,
                        'confidence': round(score, 4),
                        'type': 'collaborative_item_based',
                        'based_on': entity_id
                    })

        # Remove duplicates and sort
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            entity = rec.get('entity')
            if entity and entity not in seen:
                seen.add(entity)
                unique_recommendations.append(rec)

        unique_recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # Generate explanations if requested
        explanations = []
        if include_explanation:
            explanations = self._generate_collaborative_explanations(
                unique_recommendations[:max_recommendations], user_id
            )

        return {
            'recommendations': unique_recommendations[:max_recommendations],
            'explanations': explanations,
            'confidence_scores': [r.get('confidence', 0.0) for r in unique_recommendations[:max_recommendations]],
            'user_id': user_id,
            'source_entity': entity_id,
            'method': 'collaborative_filtering'
        }

    # Helper methods

    def _find_similar_entities(self, entity_id: str, kg_data: Dict, top_k: int, threshold: float) -> List[Dict]:
        """Find semantically similar entities using NeuralKG or fallback."""
        # Try NeuralKG if available
        if self.neuralkg_instance and hasattr(self.neuralkg_instance, 'find_similar_entities'):
            try:
                result = self.neuralkg_instance.find_similar_entities(
                    entity=entity_id,
                    top_k=top_k * 2  # Get more to filter
                )
                if result.get('status') == 'success':
                    similar = []
                    for item in result.get('similar_entities', []):
                        confidence = item.get('similarity', 0)
                        if confidence >= threshold:
                            similar.append({
                                'entity': item.get('entity'),
                                'confidence': round(confidence, 4),
                                'type': 'similar',
                                'method': 'neuralkg'
                            })
                    return similar[:top_k]
            except Exception as e:
                self.logger.warning(f"NeuralKG similarity failed: {e}")

        # Fallback: use graph-based similarity
        return self._graph_based_similarity(entity_id, kg_data, top_k, threshold)

    def _graph_based_similarity(self, entity_id: str, kg_data: Dict, top_k: int, threshold: float) -> List[Dict]:
        """Calculate entity similarity based on graph structure."""
        triples = kg_data.get('triples', [])

        # Build entity relationship profiles
        entity_relations = defaultdict(lambda: defaultdict(int))
        for h, r, t in triples:
            entity_relations[h][r] += 1
            entity_relations[t][f"inverse_{r}"] += 1

        if entity_id not in entity_relations:
            return []

        source_profile = entity_relations[entity_id]
        similarities = []

        for entity, profile in entity_relations.items():
            if entity != entity_id:
                # Calculate Jaccard similarity of relationship sets
                common = len(set(source_profile.keys()) & set(profile.keys()))
                union = len(set(source_profile.keys()) | set(profile.keys()))
                similarity = common / union if union > 0 else 0

                if similarity >= threshold:
                    similarities.append({
                        'entity': entity,
                        'confidence': round(similarity, 4),
                        'type': 'similar',
                        'method': 'graph_structure'
                    })

        similarities.sort(key=lambda x: x['confidence'], reverse=True)
        return similarities[:top_k]

    def _find_related_entities(self, entity_id: str, triples: List, top_k: int, threshold: float) -> List[Dict]:
        """Find directly related entities via triples."""
        related = defaultdict(lambda: {'confidence': 0, 'relations': []})

        for h, r, t in triples:
            if h == entity_id:
                related[t]['confidence'] += 0.5
                related[t]['relations'].append((r, 'outgoing'))
            elif t == entity_id:
                related[h]['confidence'] += 0.5
                related[h]['relations'].append((r, 'incoming'))

        recommendations = []
        for entity, data in related.items():
            confidence = min(data['confidence'], 1.0)
            if confidence >= threshold:
                recommendations.append({
                    'entity': entity,
                    'confidence': round(confidence, 4),
                    'type': 'related',
                    'relations': data['relations'],
                    'method': 'direct_connection'
                })

        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_k]

    def _find_complementary_entities(self, entity_id: str, triples: List, top_k: int, threshold: float) -> List[Dict]:
        """Find complementary entities that complete or enhance the source entity."""
        # Find entities that are commonly connected to the same entities but not directly connected
        source_related = set()
        for h, r, t in triples:
            if h == entity_id:
                source_related.add(t)
            elif t == entity_id:
                source_related.add(h)

        complementary_scores = defaultdict(float)

        for related_entity in source_related:
            for h, r, t in triples:
                if h == related_entity and t != entity_id and t not in source_related:
                    complementary_scores[t] += 0.3
                elif t == related_entity and h != entity_id and h not in source_related:
                    complementary_scores[h] += 0.3

        recommendations = []
        for entity, score in sorted(complementary_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                recommendations.append({
                    'entity': entity,
                    'confidence': round(min(score, 1.0), 4),
                    'type': 'complementary',
                    'method': 'indirect_connection'
                })

        return recommendations[:top_k]

    def _find_trending_entities(self, kg_data: Dict, top_k: int, threshold: float) -> List[Dict]:
        """Find trending/popular entities based on connection count."""
        triples = kg_data.get('triples', [])
        entity_connections = defaultdict(int)

        for h, r, t in triples:
            entity_connections[h] += 1
            entity_connections[t] += 1

        # Calculate average and find entities above threshold
        if not entity_connections:
            return []

        avg_connections = sum(entity_connections.values()) / len(entity_connections)
        max_connections = max(entity_connections.values())

        recommendations = []
        for entity, count in sorted(entity_connections.items(), key=lambda x: x[1], reverse=True):
            # Normalize score
            score = count / max_connections if max_connections > 0 else 0
            if score >= threshold:
                recommendations.append({
                    'entity': entity,
                    'confidence': round(score, 4),
                    'type': 'trending',
                    'connection_count': count,
                    'method': 'popularity_based'
                })

        return recommendations[:top_k]

    def _rank_available_actions(
        self,
        available_actions: List[Dict],
        current_state: str,
        previous_actions: List[str],
        goals: List[str],
        kg_data: Dict,
        min_confidence: float
    ) -> List[Dict]:
        """Rank available actions based on context."""
        recommendations = []

        for action in available_actions:
            score = 0.5  # Base score

            # Boost if action matches goals
            action_name = action.get('name', action.get('action', ''))
            for goal in goals:
                if goal.lower() in action_name.lower():
                    score += 0.3

            # Penalize recently taken actions
            if action_name in previous_actions[-5:]:
                score -= 0.2

            # Boost based on state relevance
            if current_state in action.get('valid_states', [current_state]):
                score += 0.2

            if score >= min_confidence:
                recommendations.append({
                    'action': action_name,
                    'confidence': round(min(score, 1.0), 4),
                    'parameters': action.get('parameters', {}),
                    'type': 'action'
                })

        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations

    def _generate_action_suggestions(
        self,
        current_state: str,
        previous_actions: List[str],
        goals: List[str],
        kg_data: Dict,
        min_confidence: float
    ) -> List[Dict]:
        """Generate action suggestions based on state machine logic."""
        suggestions = []

        # State-based action templates
        state_actions = {
            'initial': ['explore', 'search', 'analyze'],
            'exploring': ['drill_down', 'compare', 'summarize'],
            'analyzing': ['validate', 'extend', 'report'],
            'processing': ['continue', 'pause', 'review'],
            'complete': ['export', 'share', 'archive']
        }

        actions = state_actions.get(current_state, ['explore', 'review'])

        for action in actions:
            score = 0.7
            if action not in previous_actions[-3:]:
                score += 0.2

            if score >= min_confidence:
                suggestions.append({
                    'action': action,
                    'confidence': round(score, 4),
                    'type': 'suggested_action',
                    'valid_in_state': current_state
                })

        return suggestions

    def _get_user_history(self, user_id: str, ctx: Dict) -> Dict:
        """Get user history for personalization."""
        # First check context
        history = ctx.get('user_history', {})
        if history:
            return history

        # Check cache
        if user_id in self._user_history_cache:
            return self._user_history_cache[user_id]

        # Try to get from knowledge hub
        if self.kg_hub_instance and hasattr(self.kg_hub_instance, 'get_user_history'):
            try:
                history = self.kg_hub_instance.get_user_history(user_id)
                self._user_history_cache[user_id] = history
                return history
            except Exception as e:
                self.logger.warning(f"Could not get user history from KG hub: {e}")

        # Return empty history
        return {
            'viewed_entities': [],
            'preferred_types': [],
            'interaction_patterns': []
        }

    def _get_collaborative_data(self, user_id: Optional[str], entity_id: Optional[str], kg_data: Dict) -> Dict:
        """Get collaborative filtering data."""
        # In a real system, this would come from a database
        # Here we simulate based on triples
        triples = kg_data.get('triples', [])

        # Build user-entity interaction matrix (simulated)
        interactions = defaultdict(list)

        # Use entity relationships to simulate user preferences
        entities = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)

        # Generate synthetic users based on entity clusters
        entity_list = list(entities)
        for i in range(min(10, len(entity_list))):
            user_name = f"user_{i}"
            # Each "user" likes entities with similar characteristics
            seed = entity_list[i % len(entity_list)]
            interactions[user_name].append(seed)

            # Add related entities
            for h, r, t in triples:
                if h == seed and t not in interactions[user_name]:
                    interactions[user_name].append(t)
                elif t == seed and h not in interactions[user_name]:
                    interactions[user_name].append(h)

        if user_id and user_id not in interactions:
            interactions[user_id] = [entity_id] if entity_id else []

        return dict(interactions)

    def _find_similar_users(self, user_id: str, user_interactions: Dict) -> List[Tuple[str, float]]:
        """Find users with similar interaction patterns."""
        if user_id not in user_interactions:
            return []

        target_items = set(user_interactions[user_id])
        if not target_items:
            return []

        similarities = []
        for other_user, items in user_interactions.items():
            if other_user != user_id:
                other_items = set(items)
                # Jaccard similarity
                intersection = len(target_items & other_items)
                union = len(target_items | other_items)
                similarity = intersection / union if union > 0 else 0

                if similarity > 0:
                    similarities.append((other_user, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]  # Top 5 similar users

    def _find_similar_items(self, entity_id: str, user_interactions: Dict) -> List[Tuple[str, float]]:
        """Find items similar to the given entity based on co-occurrence."""
        # Find users who interacted with this entity
        entity_users = set()
        for user, items in user_interactions.items():
            if entity_id in items:
                entity_users.add(user)

        if not entity_users:
            return []

        # Score other items by how often they appear with the same users
        item_scores = defaultdict(float)
        for user in entity_users:
            for item in user_interactions.get(user, []):
                if item != entity_id:
                    item_scores[item] += 1.0 / len(entity_users)

        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

    def _merge_recommendations(self, list1: List[Dict], list2: List[Dict], max_count: int, diversity: float) -> List[Dict]:
        """Merge two recommendation lists with diversity."""
        merged = []
        seen = set()

        # Alternate between lists for diversity
        i, j = 0, 0
        while len(merged) < max_count and (i < len(list1) or j < len(list2)):
            if i < len(list1):
                item = list1[i]
                if item.get('entity') not in seen:
                    merged.append(item)
                    seen.add(item.get('entity'))
                i += 1

            if j < len(list2) and len(merged) < max_count:
                item = list2[j]
                if item.get('entity') not in seen:
                    merged.append(item)
                    seen.add(item.get('entity'))
                j += 1

        return merged

    def _apply_diversity(self, recommendations: List[Dict], diversity_factor: float) -> List[Dict]:
        """Apply diversity to recommendations by penalizing similar items."""
        if len(recommendations) <= 1 or diversity_factor <= 0:
            return recommendations

        diverse_recs = [recommendations[0]]  # Keep top recommendation

        for rec in recommendations[1:]:
            # Check similarity with already selected recommendations
            max_sim = 0
            for selected in diverse_recs:
                sim = self._calculate_item_similarity(rec, selected)
                max_sim = max(max_sim, sim)

            # Adjust score based on diversity
            adjusted_score = rec.get('confidence', 0) * (1 - diversity_factor * max_sim)
            rec['confidence'] = round(adjusted_score, 4)
            rec['diversity_penalty'] = round(max_sim, 4)

            diverse_recs.append(rec)

        # Re-sort by adjusted confidence
        diverse_recs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return diverse_recs

    def _calculate_item_similarity(self, item1: Dict, item2: Dict) -> float:
        """Calculate similarity between two recommendation items."""
        # Simple similarity based on shared attributes
        entity1 = item1.get('entity', '')
        entity2 = item2.get('entity', '')

        # Text similarity
        words1 = set(entity1.lower().split())
        words2 = set(entity2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _get_entity_type(self, entity: str, kg_data: Dict) -> Optional[str]:
        """Get the type of an entity from triples."""
        for h, r, t in kg_data.get('triples', []):
            if h == entity and r.lower() in ['is_a', 'type', 'isa', 'category']:
                return t
        return None

    # Explanation generation methods

    def _generate_explanations(self, recommendations: List[Dict], source_entity: str, rec_type: str) -> List[str]:
        """Generate explanations for entity recommendations."""
        explanations = []

        for rec in recommendations:
            entity = rec.get('entity', '')
            confidence = rec.get('confidence', 0)
            method = rec.get('method', '')

            if rec_type == 'similar':
                exp = f"'{entity}' is similar to '{source_entity}' based on {method.replace('_', ' ')} (confidence: {confidence:.0%})"
            elif rec_type == 'related':
                relations = rec.get('relations', [])
                if relations:
                    rel_str = ', '.join([r[0] for r in relations[:2]])
                    exp = f"'{entity}' is directly related to '{source_entity}' via: {rel_str} (confidence: {confidence:.0%})"
                else:
                    exp = f"'{entity}' has direct connections to '{source_entity}' (confidence: {confidence:.0%})"
            elif rec_type == 'complementary':
                exp = f"'{entity}' complements '{source_entity}' and may provide additional insights (confidence: {confidence:.0%})"
            elif rec_type == 'trending':
                count = rec.get('connection_count', 0)
                exp = f"'{entity}' is a trending entity with {count} connections (confidence: {confidence:.0%})"
            else:
                exp = f"'{entity}' recommended based on {method.replace('_', ' ')} (confidence: {confidence:.0%})"

            explanations.append(exp)

        return explanations

    def _generate_action_explanations(self, recommendations: List[Dict], current_state: str, previous_actions: List[str]) -> List[str]:
        """Generate explanations for action suggestions."""
        explanations = []

        for rec in recommendations:
            action = rec.get('action', '')
            confidence = rec.get('confidence', 0)

            if action in previous_actions:
                exp = f"'{action}' is suggested again based on current '{current_state}' state (confidence: {confidence:.0%})"
            else:
                exp = f"'{action}' is a suitable next step from '{current_state}' state (confidence: {confidence:.0%})"

            explanations.append(exp)

        return explanations

    def _generate_knowledge_explanations(self, recommendations: List[Dict], source_entity: Optional[str], interests: List[str]) -> List[str]:
        """Generate explanations for knowledge exploration recommendations."""
        explanations = []

        for rec in recommendations:
            entity = rec.get('entity', '')
            rec_type = rec.get('type', '')
            confidence = rec.get('confidence', 0)

            if rec_type == 'knowledge_gap':
                exp = f"Explore '{entity}' to fill a knowledge gap (confidence: {confidence:.0%})"
            elif rec_type == 'related_knowledge':
                exp = f"'{entity}' is related to '{source_entity}' and worth exploring (confidence: {confidence:.0%})"
            elif rec_type == 'interest_based':
                exp = f"'{entity}' aligns with your interests in {', '.join(interests[:2])} (confidence: {confidence:.0%})"
            else:
                exp = f"Consider exploring '{entity}' (confidence: {confidence:.0%})"

            explanations.append(exp)

        return explanations

    def _generate_personalized_explanations(self, recommendations: List[Dict], user_id: str, viewed: List[str], preferences: List[str]) -> List[str]:
        """Generate explanations for personalized recommendations."""
        explanations = []

        for rec in recommendations:
            entity = rec.get('entity', '')
            confidence = rec.get('confidence', 0)

            base_exp = f"Recommended for you based on"

            if viewed:
                base_exp += f" your previous interest in {len(viewed)} entities"
            if preferences:
                base_exp += f" and preference for {', '.join(preferences[:2])}"

            base_exp += f" (confidence: {confidence:.0%})"

            explanations.append(base_exp)

        return explanations

    def _generate_collaborative_explanations(self, recommendations: List[Dict], user_id: Optional[str]) -> List[str]:
        """Generate explanations for collaborative filtering recommendations."""
        explanations = []

        for rec in recommendations:
            entity = rec.get('entity', '')
            confidence = rec.get('confidence', 0)
            rec_type = rec.get('type', '')

            if rec_type == 'collaborative':
                exp = f"Users similar to you have shown interest in '{entity}' (confidence: {confidence:.0%})"
            elif rec_type == 'collaborative_item_based':
                based_on = rec.get('based_on', '')
                exp = f"Users interested in '{based_on}' also liked '{entity}' (confidence: {confidence:.0%})"
            else:
                exp = f"Recommended based on community patterns (confidence: {confidence:.0%})"

            explanations.append(exp)

        return explanations

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration including:
        - operation: Type of recommendation operation
        - entity_id: Starting entity for recommendations
        - user_id: User identifier for personalized recommendations
        - recommendation_type: Type of entity recommendation
        - context: Current workflow context
        - max_recommendations: Number of recommendations to return
        - min_confidence: Minimum confidence threshold
        - include_explanation: Whether to include explanations
        - diversity_factor: Factor for recommendation diversity
        """
        return {
            "type": "object",
            "title": "Recommendation Engine Configuration",
            "description": "Configure recommendation engine parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of recommendation operation to execute",
                    "enum": ["recommend_entities", "suggest_actions", "explore_knowledge", "personalized", "collaborative"],
                    "enumNames": [
                        "Recommend Entities - Find related/similar entities",
                        "Suggest Actions - Recommend next actions based on context",
                        "Explore Knowledge - Suggest knowledge areas to explore",
                        "Personalized - Generate personalized recommendations",
                        "Collaborative - Use collaborative filtering"
                    ],
                    "default": "recommend_entities"
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "Starting entity for recommendations",
                    "default": ""
                },
                "user_id": {
                    "type": "string",
                    "title": "User ID",
                    "description": "User identifier for personalized recommendations",
                    "default": ""
                },
                "recommendation_type": {
                    "type": "string",
                    "title": "Recommendation Type",
                    "description": "Type of entity recommendation for recommend_entities operation",
                    "enum": ["similar", "related", "complementary", "trending"],
                    "enumNames": [
                        "Similar - Semantically similar entities",
                        "Related - Directly connected entities",
                        "Complementary - Entities that complete/enhance",
                        "Trending - Popular/well-connected entities"
                    ],
                    "default": "similar"
                },
                "context": {
                    "type": "object",
                    "title": "Context",
                    "description": "Current workflow context/state",
                    "additionalProperties": True,
                    "default": {}
                },
                "max_recommendations": {
                    "type": "integer",
                    "title": "Max Recommendations",
                    "description": "Maximum number of recommendations to return",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "description": "Minimum confidence threshold (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.6
                },
                "include_explanation": {
                    "type": "boolean",
                    "title": "Include Explanations",
                    "description": "Whether to include explanations for recommendations",
                    "default": True
                },
                "diversity_factor": {
                    "type": "number",
                    "title": "Diversity Factor",
                    "description": "Factor for ensuring diverse recommendations (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["recommend_entities"]}
                            },
                            "required": ["entity_id"],
                            "description": "Recommend entities similar or related to the specified entity"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["suggest_actions"]}
                            },
                            "description": "Suggest next actions based on workflow context"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["explore_knowledge"]}
                            },
                            "description": "Recommend knowledge areas to explore based on interests and gaps"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["personalized"]}
                            },
                            "required": ["user_id"],
                            "description": "Generate personalized recommendations based on user history"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["collaborative"]}
                            },
                            "description": "Use collaborative filtering to find recommendations from similar users"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node can provide recommendations (with or without ML),
            False only if critical dependencies are missing
        """
        try:
            # Node is healthy if it can at least provide fallback recommendations
            # The knowledge graph integrations are optional enhancements
            return True
        except Exception:
            return False
