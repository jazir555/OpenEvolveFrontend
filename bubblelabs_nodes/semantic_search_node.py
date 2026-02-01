"""
Semantic Search Node for BubbleLabs Integration

Provides semantic search capabilities using NeuralKG embeddings:
- Generate embeddings for entities
- Find semantically similar entities
- Search by natural language query
- Cluster entities by similarity
- Recommend related entities
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math
import re
from collections import defaultdict

from .base_node import BubbleLabsNode, NodeExecutionError


class SemanticSearchNode(BubbleLabsNode):
    """
    Semantic search node using NeuralKG embeddings for knowledge discovery.

    Supports five operations:
    - generate_embeddings: Create vector embeddings for entities
    - find_similar: Find semantically similar entities
    - search: Search by natural language query
    - cluster: Cluster entities by similarity
    - recommend: Recommend related entities

    Falls back to text-based similarity when NeuralKG is unavailable.
    """

    # Node metadata
    DISPLAY_NAME = "Semantic Search"
    DESCRIPTION = "Find similar entities and search knowledge using neural embeddings"
    ICON = "semantic-search"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports for NeuralKG integration
        NeuralKGIntegration = self.safe_import(
            'knowledge_engine.integrations.neuralkg_integration.NeuralKGIntegration',
            fallback_value=None,
            error_msg="NeuralKGIntegration not available for SemanticSearchNode"
        )

        # Also try alternative import paths
        if NeuralKGIntegration is None:
            NeuralKGIntegration = self.safe_import(
                'knowledge_engine.integrations.neuralkg_integration.NeuralKGIntegration',
                fallback_value=None,
                error_msg="NeuralKGIntegration not found in alternate path"
            )

        self.NeuralKGIntegration = NeuralKGIntegration

        # Safe import for UnifiedKGIntegrationHub
        UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for SemanticSearchNode"
        )
        self.UnifiedKGIntegrationHub = UnifiedKGIntegrationHub

        # Initialize NeuralKG instance if available
        self.neuralkg_instance = None
        self.unified_hub_instance = None
        self._embeddings_cache = {}

        if NeuralKGIntegration:
            try:
                self.neuralkg_instance = NeuralKGIntegration()
                self.logger.info("NeuralKG integration initialized for SemanticSearchNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize NeuralKG: {e}")
                self.neuralkg_instance = None

        if UnifiedKGIntegrationHub:
            try:
                self.unified_hub_instance = UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for SemanticSearchNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.unified_hub_instance = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields vary by operation:
        - generate_embeddings: Requires 'entities' or triples in context
        - find_similar: Requires 'entity' or 'query' in inputs
        - search: Requires 'query' in inputs
        - cluster: Requires 'entities' or embeddings in context
        - recommend: Requires 'entity' in inputs
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation')
        if operation is None:
            operation = self.config.get('operation')

        if operation is None:
            errors.append("Missing required field: operation (must be 'generate_embeddings', 'find_similar', 'search', 'cluster', or 'recommend')")
            return errors

        valid_operations = ['generate_embeddings', 'find_similar', 'search', 'cluster', 'recommend']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Validate operation-specific requirements
        if operation == 'find_similar':
            entity = inputs.get('entity') or inputs.get('query') or self.config.get('entity') or self.config.get('query')
            if not entity:
                errors.append("Find similar operation requires 'entity' or 'query' (in inputs or config)")

        elif operation == 'search':
            query = inputs.get('query') or self.config.get('query')
            if not query:
                errors.append("Search operation requires 'query' (in inputs or config)")

        elif operation == 'recommend':
            entity = inputs.get('entity') or self.config.get('entity')
            if not entity:
                errors.append("Recommend operation requires 'entity' (in inputs or config)")

        # Validate numeric parameters if provided
        if 'top_k' in inputs:
            try:
                top_k = int(inputs['top_k'])
                if top_k < 1:
                    errors.append("top_k must be at least 1")
            except (ValueError, TypeError):
                errors.append("top_k must be an integer")

        if 'similarity_threshold' in inputs:
            try:
                threshold = float(inputs['similarity_threshold'])
                if not (0.0 <= threshold <= 1.0):
                    errors.append("similarity_threshold must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("similarity_threshold must be a number")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the semantic search operation.

        Args:
            inputs: Operation specification including operation type and parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing operation results:
                - operation: The type of operation executed
                - results: Operation-specific results
                - embeddings: Entity embeddings (if applicable)
                - clusters: Cluster assignments (if applicable)
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If operation execution fails
        """
        # Get operation parameters
        operation = inputs.get('operation', self.config.get('operation'))
        query = inputs.get('query') or self.config.get('query', '')
        entity = inputs.get('entity') or inputs.get('query') or self.config.get('entity') or self.config.get('query', '')
        top_k = inputs.get('top_k', self.config.get('top_k', 10))
        similarity_threshold = inputs.get('similarity_threshold', self.config.get('similarity_threshold', 0.7))
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        embedding_model = inputs.get('embedding_model', self.config.get('embedding_model', 'transE'))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Executing semantic search operation: {operation}")

        try:
            # Get knowledge graph data
            kg_data = self._get_knowledge_graph_data(inputs, context)

            context.update_progress(30, "Processing data")

            # Execute the appropriate operation
            if operation == 'generate_embeddings':
                result = self._generate_embeddings(kg_data, embedding_model, context)
            elif operation == 'find_similar':
                result = self._find_similar(entity, kg_data, top_k, similarity_threshold, context)
            elif operation == 'search':
                result = self._search(query, kg_data, top_k, similarity_threshold, entity_types, context)
            elif operation == 'cluster':
                result = self._cluster_entities(kg_data, top_k, similarity_threshold, context)
            elif operation == 'recommend':
                result = self._recommend_entities(entity, kg_data, top_k, similarity_threshold, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['generate_embeddings', 'find_similar', 'search', 'cluster', 'recommend']}
                )

            context.update_progress(90, "Processing results")

            # Add execution metadata
            result['metadata'] = {
                'operation': operation,
                'executed_at': datetime.now().isoformat(),
                'execution_id': self.execution_id,
                'parameters': {
                    'query': query,
                    'entity': entity,
                    'top_k': top_k,
                    'similarity_threshold': similarity_threshold,
                    'entity_types': entity_types,
                    'embedding_model': embedding_model
                },
                'neuralkg_available': self.neuralkg_instance is not None and (
                    self.neuralkg_instance.is_available() if hasattr(self.neuralkg_instance, 'is_available') else False
                )
            }

            context.update_progress(100, f"Operation complete: {operation}")
            self.logger.info(f"Semantic search operation completed: {operation}")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Semantic search failed: {str(e)}",
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
        if self.unified_hub_instance:
            try:
                # Try to get entities and relationships from the hub
                if hasattr(self.unified_hub_instance, 'get_all_entities'):
                    entities = self.unified_hub_instance.get_all_entities()
                    triples = self.unified_hub_instance.get_all_triples() if hasattr(self.unified_hub_instance, 'get_all_triples') else []
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

    def _generate_embeddings(
        self,
        kg_data: Dict[str, Any],
        model: str,
        context
    ) -> Dict[str, Any]:
        """Generate embeddings for entities using NeuralKG or fallback."""
        triples = kg_data.get('triples', [])
        entities = kg_data.get('entities', [])

        # Convert entities to triples if needed
        if not triples and entities:
            triples = self._entities_to_triples(entities)

        if not triples:
            # Generate sample triples for demonstration
            triples = self._generate_sample_triples()

        context.update_progress(40, "Generating embeddings")

        # Try to use NeuralKG if available
        if self.neuralkg_instance and hasattr(self.neuralkg_instance, 'is_available'):
            if self.neuralkg_instance.is_available():
                try:
                    result = self.neuralkg_instance.generate_embeddings(
                        triples=triples,
                        model=model.lower()
                    )

                    if result.get('status') == 'success':
                        embeddings = result.get('embeddings', {})
                        self._embeddings_cache = embeddings

                        return {
                            'operation': 'generate_embeddings',
                            'results': {
                                'entity_count': len(embeddings.get('entities', {})),
                                'relation_count': len(embeddings.get('relations', {}))
                            },
                            'embeddings': embeddings,
                            'clusters': [],
                            'method': 'neuralkg'
                        }
                except Exception as e:
                    self.logger.warning(f"NeuralKG embedding generation failed: {e}")

        # Fallback to simple embeddings
        context.update_progress(50, "Using fallback embedding method")
        embeddings = self._generate_fallback_embeddings(triples)
        self._embeddings_cache = embeddings

        return {
            'operation': 'generate_embeddings',
            'results': {
                'entity_count': len(embeddings.get('entities', {})),
                'relation_count': len(embeddings.get('relations', {}))
            },
            'embeddings': embeddings,
            'clusters': [],
            'method': 'fallback'
        }

    def _find_similar(
        self,
        entity: str,
        kg_data: Dict[str, Any],
        top_k: int,
        threshold: float,
        context
    ) -> Dict[str, Any]:
        """Find semantically similar entities."""
        # First ensure we have embeddings
        if not self._embeddings_cache:
            self._generate_embeddings(kg_data, 'transE', context)

        context.update_progress(50, f"Finding entities similar to: {entity}")

        embeddings = self._embeddings_cache.get('entities', {})

        # Try NeuralKG if available
        if self.neuralkg_instance and hasattr(self.neuralkg_instance, 'find_similar_entities'):
            try:
                result = self.neuralkg_instance.find_similar_entities(
                    entity=entity,
                    embeddings=self._embeddings_cache,
                    top_k=top_k
                )

                if result.get('status') == 'success':
                    similar = [
                        s for s in result.get('similar_entities', [])
                        if s.get('similarity', 0) >= threshold
                    ]
                    return {
                        'operation': 'find_similar',
                        'results': similar,
                        'embeddings': [],
                        'clusters': [],
                        'query_entity': entity,
                        'method': 'neuralkg'
                    }
            except Exception as e:
                self.logger.warning(f"NeuralKG similarity search failed: {e}")

        # Fallback to cosine similarity
        similar = self._calculate_similarity_fallback(entity, embeddings, top_k, threshold)

        return {
            'operation': 'find_similar',
            'results': similar,
            'embeddings': [],
            'clusters': [],
            'query_entity': entity,
            'method': 'fallback'
        }

    def _search(
        self,
        query: str,
        kg_data: Dict[str, Any],
        top_k: int,
        threshold: float,
        entity_types: List[str],
        context
    ) -> Dict[str, Any]:
        """Search entities by natural language query."""
        context.update_progress(50, f"Searching for: {query}")

        entities = kg_data.get('entities', [])
        triples = kg_data.get('triples', [])

        # If no embeddings, generate them
        if not self._embeddings_cache:
            self._generate_embeddings(kg_data, 'transE', context)

        embeddings = self._embeddings_cache.get('entities', {})

        # Search using text similarity
        results = self._text_based_search(query, entities, triples, embeddings, top_k, threshold)

        # Filter by entity types if specified
        if entity_types:
            results = [
                r for r in results
                if any(t in r.get('entity', '') for t in entity_types)
            ]

        return {
            'operation': 'search',
            'results': results[:top_k],
            'embeddings': [],
            'clusters': [],
            'query': query,
            'method': 'text_based'
        }

    def _cluster_entities(
        self,
        kg_data: Dict[str, Any],
        num_clusters: int,
        threshold: float,
        context
    ) -> Dict[str, Any]:
        """Cluster entities by similarity."""
        context.update_progress(50, "Clustering entities")

        # Generate embeddings if needed
        if not self._embeddings_cache:
            self._generate_embeddings(kg_data, 'transE', context)

        embeddings = self._embeddings_cache.get('entities', {})
        entities = list(embeddings.keys())

        if len(entities) < 2:
            return {
                'operation': 'cluster',
                'results': [],
                'embeddings': [],
                'clusters': [],
                'method': 'none'
            }

        # Simple clustering using similarity threshold
        clusters = self._cluster_by_similarity(entities, embeddings, num_clusters, threshold)

        return {
            'operation': 'cluster',
            'results': [{'cluster_id': i, 'entities': c} for i, c in enumerate(clusters)],
            'embeddings': [],
            'clusters': clusters,
            'method': 'similarity_based'
        }

    def _recommend_entities(
        self,
        entity: str,
        kg_data: Dict[str, Any],
        top_k: int,
        threshold: float,
        context
    ) -> Dict[str, Any]:
        """Recommend related entities."""
        context.update_progress(50, f"Generating recommendations for: {entity}")

        triples = kg_data.get('triples', [])

        # Find directly related entities
        related = self._find_related_from_triples(entity, triples)

        # Also find similar entities
        similar_result = self._find_similar(entity, kg_data, top_k * 2, threshold, context)
        similar = similar_result.get('results', [])

        # Combine and rank recommendations
        recommendations = self._rank_recommendations(entity, related, similar, top_k)

        return {
            'operation': 'recommend',
            'results': recommendations,
            'embeddings': [],
            'clusters': [],
            'source_entity': entity,
            'method': 'hybrid'
        }

    # Helper methods

    def _entities_to_triples(self, entities: List[Dict]) -> List[Tuple[str, str, str]]:
        """Convert entity list to triples."""
        triples = []
        for entity in entities:
            name = entity.get('name') or entity.get('id') or str(entity)
            entity_type = entity.get('type', 'entity')
            triples.append((name, 'is_a', entity_type))

            # Add relationships if present
            for rel, target in entity.get('relationships', {}).items():
                triples.append((name, rel, target))

        return triples

    def _generate_sample_triples(self) -> List[Tuple[str, str, str]]:
        """Generate sample triples for demonstration."""
        return [
            ('OpenAI', 'is_a', 'AI Research Lab'),
            ('DeepMind', 'is_a', 'AI Research Lab'),
            ('Anthropic', 'is_a', 'AI Research Lab'),
            ('Google', 'is_a', 'Tech Company'),
            ('Microsoft', 'is_a', 'Tech Company'),
            ('Amazon', 'is_a', 'Tech Company'),
            ('GPT-4', 'developed_by', 'OpenAI'),
            ('Gemini', 'developed_by', 'Google'),
            ('Claude', 'developed_by', 'Anthropic'),
            ('AlphaGo', 'developed_by', 'DeepMind'),
            ('Machine Learning', 'is_a', 'Field'),
            ('Deep Learning', 'is_a', 'Field'),
            ('NLP', 'is_a', 'Field'),
            ('Computer Vision', 'is_a', 'Field'),
            ('GPT-4', 'uses', 'Deep Learning'),
            ('AlphaGo', 'uses', 'Machine Learning'),
            ('Claude', 'uses', 'NLP'),
            ('OpenAI', 'focuses_on', 'NLP'),
            ('DeepMind', 'focuses_on', 'Machine Learning'),
            ('Anthropic', 'focuses_on', 'AI Safety')
        ]

    def _generate_fallback_embeddings(self, triples: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Generate simple fallback embeddings based on co-occurrence."""
        import random
        random.seed(42)

        # Build entity and relation sets
        entities = set()
        relations = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)

        # Generate simple embeddings based on entity characteristics
        entity_embeddings = {}
        for entity in entities:
            # Create a simple feature vector based on string characteristics
            # and connection patterns
            features = self._extract_entity_features(entity, triples)
            entity_embeddings[entity] = features

        # Normalize embeddings
        for entity in entity_embeddings:
            emb = entity_embeddings[entity]
            norm = math.sqrt(sum(x * x for x in emb))
            if norm > 0:
                entity_embeddings[entity] = [x / norm for x in emb]

        # Generate relation embeddings
        relation_embeddings = {}
        for relation in relations:
            # Simple embedding based on relation characteristics
            features = self._extract_relation_features(relation, triples)
            relation_embeddings[relation] = features

        return {
            'entities': entity_embeddings,
            'relations': relation_embeddings
        }

    def _extract_entity_features(self, entity: str, triples: List[Tuple[str, str, str]]) -> List[float]:
        """Extract feature vector for an entity."""
        # Simple 20-dimensional feature vector
        features = [0.0] * 20

        # Character-based features (first 5 dims)
        features[0] = len(entity) / 100.0
        features[1] = sum(1 for c in entity if c.isupper()) / max(len(entity), 1)
        features[2] = ord(entity[0]) / 255.0 if entity else 0
        features[3] = ord(entity[-1]) / 255.0 if entity else 0
        features[4] = entity.count(' ') / max(len(entity), 1)

        # Connection-based features (next 10 dims)
        outgoing = [t for h, r, t in triples if h == entity]
        incoming = [h for h, r, t in triples if t == entity]
        relations = [r for h, r, t in triples if h == entity or t == entity]

        features[5] = len(outgoing) / 10.0
        features[6] = len(incoming) / 10.0
        features[7] = len(set(outgoing)) / max(len(outgoing), 1)
        features[8] = len(set(incoming)) / max(len(incoming), 1)
        features[9] = len(relations) / 10.0

        # Relation type diversity
        unique_relations = set(relations)
        features[10] = len(unique_relations) / 10.0

        # Hash-based features for uniqueness (remaining dims)
        entity_hash = hash(entity) % 10000 / 10000.0
        for i in range(9):
            features[11 + i] = ((entity_hash * (i + 1)) % 1.0) * 2 - 1

        return features

    def _extract_relation_features(self, relation: str, triples: List[Tuple[str, str, str]]) -> List[float]:
        """Extract feature vector for a relation."""
        features = [0.0] * 20

        # Character-based features
        features[0] = len(relation) / 50.0
        features[1] = sum(1 for c in relation if c == '_') / max(len(relation), 1)
        features[2] = ord(relation[0]) / 255.0 if relation else 0

        # Usage-based features
        relation_triples = [(h, t) for h, r, t in triples if r == relation]
        features[3] = len(relation_triples) / 10.0
        features[4] = len(set(h for h, t in relation_triples)) / max(len(relation_triples), 1)
        features[5] = len(set(t for h, t in relation_triples)) / max(len(relation_triples), 1)

        # Hash-based features
        relation_hash = hash(relation) % 10000 / 10000.0
        for i in range(14):
            features[6 + i] = ((relation_hash * (i + 1)) % 1.0) * 2 - 1

        return features

    def _calculate_similarity_fallback(
        self,
        entity: str,
        embeddings: Dict[str, List[float]],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Calculate cosine similarity using fallback embeddings."""
        if entity not in embeddings:
            return []

        query_emb = embeddings[entity]
        similarities = []

        for other_entity, other_emb in embeddings.items():
            if other_entity != entity:
                similarity = self._cosine_similarity(query_emb, other_emb)
                if similarity >= threshold:
                    similarities.append({
                        'entity': other_entity,
                        'similarity': round(similarity, 4)
                    })

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _text_based_search(
        self,
        query: str,
        entities: List[Any],
        triples: List[Tuple[str, str, str]],
        embeddings: Dict[str, List[float]],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Search entities using text-based similarity."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []

        # Search in entity names
        for entity_name in embeddings.keys():
            entity_lower = entity_name.lower()
            entity_words = set(entity_lower.split())

            # Calculate text similarity
            word_overlap = len(query_words & entity_words)
            jaccard = word_overlap / max(len(query_words | entity_words), 1)

            # Check for substring match
            substring_score = 1.0 if query_lower in entity_lower else 0.0

            # Combined score
            score = jaccard * 0.5 + substring_score * 0.5

            if score > 0:
                results.append({
                    'entity': entity_name,
                    'score': round(score, 4),
                    'match_type': 'text'
                })

        # Search in triples
        for h, r, t in triples:
            triple_text = f"{h} {r} {t}".lower()
            if query_lower in triple_text:
                score = 0.5  # Partial match in triple
                if h not in [r['entity'] for r in results]:
                    results.append({
                        'entity': h,
                        'score': round(score, 4),
                        'match_type': 'triple'
                    })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def _cluster_by_similarity(
        self,
        entities: List[str],
        embeddings: Dict[str, List[float]],
        num_clusters: int,
        threshold: float
    ) -> List[List[str]]:
        """Simple clustering based on similarity threshold."""
        if not entities:
            return []

        clusters = []
        assigned = set()

        for entity in entities:
            if entity in assigned:
                continue

            # Create new cluster
            cluster = [entity]
            assigned.add(entity)

            # Find similar entities
            for other in entities:
                if other not in assigned and other in embeddings:
                    similarity = self._cosine_similarity(
                        embeddings[entity],
                        embeddings[other]
                    )
                    if similarity >= threshold:
                        cluster.append(other)
                        assigned.add(other)

            clusters.append(cluster)

            if len(clusters) >= num_clusters:
                break

        # Add remaining entities to closest cluster
        for entity in entities:
            if entity not in assigned:
                # Find closest cluster
                best_cluster = None
                best_similarity = -1

                for cluster in clusters:
                    if cluster:
                        centroid = self._calculate_centroid(cluster, embeddings)
                        if entity in embeddings:
                            sim = self._cosine_similarity(embeddings[entity], centroid)
                            if sim > best_similarity:
                                best_similarity = sim
                                best_cluster = cluster

                if best_cluster:
                    best_cluster.append(entity)
                elif clusters:
                    clusters[0].append(entity)

        return clusters

    def _calculate_centroid(self, cluster: List[str], embeddings: Dict[str, List[float]]) -> List[float]:
        """Calculate centroid of a cluster."""
        if not cluster:
            return [0.0] * 20

        vectors = [embeddings[e] for e in cluster if e in embeddings]
        if not vectors:
            return [0.0] * 20

        dim = len(vectors[0])
        centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
        return centroid

    def _find_related_from_triples(self, entity: str, triples: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Find entities directly related via triples."""
        related = []

        for h, r, t in triples:
            if h == entity:
                related.append({
                    'entity': t,
                    'relation': r,
                    'direction': 'outgoing'
                })
            elif t == entity:
                related.append({
                    'entity': h,
                    'relation': r,
                    'direction': 'incoming'
                })

        return related

    def _rank_recommendations(
        self,
        source_entity: str,
        related: List[Dict],
        similar: List[Dict],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank and combine recommendations."""
        scores = defaultdict(float)
        info = {}

        # Score related entities (direct connection)
        for r in related:
            entity = r['entity']
            scores[entity] += 2.0  # Higher weight for direct relations
            info[entity] = {
                'relation': r['relation'],
                'direction': r['direction'],
                'reason': f"Direct {r['direction']} relation: {r['relation']}"
            }

        # Score similar entities
        for s in similar:
            entity = s['entity']
            scores[entity] += s.get('similarity', 0.5)
            if entity not in info:
                info[entity] = {
                    'reason': f"Semantic similarity: {s.get('similarity', 0):.2f}"
                }

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        recommendations = []
        for entity, score in ranked[:top_k]:
            if entity != source_entity:
                rec = {
                    'entity': entity,
                    'score': round(score, 4),
                    **info.get(entity, {})
                }
                recommendations.append(rec)

        return recommendations

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration including:
        - operation: Type of semantic search operation
        - query: Natural language query or entity name
        - top_k: Number of results to return
        - similarity_threshold: Minimum similarity threshold
        - entity_types: Filter by entity types
        - embedding_model: Model to use for embeddings
        """
        return {
            "type": "object",
            "title": "Semantic Search Configuration",
            "description": "Configure semantic search parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of semantic search operation to execute",
                    "enum": ["generate_embeddings", "find_similar", "search", "cluster", "recommend"],
                    "enumNames": [
                        "Generate Embeddings",
                        "Find Similar Entities",
                        "Search by Query",
                        "Cluster Entities",
                        "Recommend Related"
                    ],
                    "default": "search"
                },
                "query": {
                    "type": "string",
                    "title": "Query",
                    "description": "Natural language query or entity name to search for",
                    "default": ""
                },
                "entity": {
                    "type": "string",
                    "title": "Entity",
                    "description": "Entity name for find_similar or recommend operations",
                    "default": ""
                },
                "top_k": {
                    "type": "integer",
                    "title": "Top K Results",
                    "description": "Number of results to return",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Similarity Threshold",
                    "description": "Minimum similarity threshold (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Filter results by entity types (optional)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "embedding_model": {
                    "type": "string",
                    "title": "Embedding Model",
                    "description": "Model to use for generating embeddings",
                    "enum": ["transE", "rotatE", "complEx", "distMult", "rgcn", "compgcn"],
                    "enumNames": [
                        "TransE (Translation-based)",
                        "RotatE (Rotation in complex space)",
                        "ComplEx (Complex embeddings)",
                        "DistMult (DistMult model)",
                        "RGCN (Graph Neural Network)",
                        "CompGCN (Composition GNN)"
                    ],
                    "default": "transE"
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["generate_embeddings"]}
                            },
                            "description": "Generate embeddings for entities in the knowledge graph"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["find_similar"]}
                            },
                            "required": ["entity"],
                            "description": "Find entities similar to the specified entity"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["search"]}
                            },
                            "required": ["query"],
                            "description": "Search entities by natural language query"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["cluster"]}
                            },
                            "description": "Cluster entities by semantic similarity"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["recommend"]}
                            },
                            "required": ["entity"],
                            "description": "Recommend related entities based on connections and similarity"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if the node can perform semantic search operations,
            False otherwise. Note: The node can still function with
            fallback methods even if NeuralKG is not available.
        """
        try:
            # Basic health check - ensure we can at least use fallback methods
            # The node is considered healthy if fallback embeddings can be generated
            test_triples = [("test", "is_a", "entity")]
            test_emb = self._generate_fallback_embeddings(test_triples)

            # Check if embeddings were generated successfully
            if test_emb and 'entities' in test_emb:
                return True

            return False
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
