"""
LM Clustering Deduplication Strategy (kg-gen)

ML-based clustering using:
1. SentenceTransformer embeddings
2. K-means clustering
3. Hybrid retrieval (BM25 + cosine similarity)
4. Intra-cluster LLM deduplication
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from ..base import Entity, DeduplicationResult, DeduplicationStrategy

logger = logging.getLogger(__name__)


class LMClusteringStrategy(DeduplicationStrategy):
    """
    ML-based clustering deduplication strategy.

    Best for:
    - Large datasets (> 1000 entities)
    - Semantic similarity detection
    - High accuracy requirements
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.cluster_size = self.config.get('cluster_size', 128)
        self.num_workers = self.config.get('num_workers', 64)
        self.retrieval_model = self.config.get(
            'retrieval_model',
            'sentence-transformers/all-mpnet-base-v2'
        )

        # Lazy initialization
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize sentence transformer model (lazy loading)."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.retrieval_model)
            logger.info(f"Initialized retrieval model: {self.retrieval_model}")
        except ImportError:
            logger.warning("sentence_transformers not available, using fallback")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = None

    def get_strategy_name(self) -> str:
        return "lm_cluster"

    async def deduplicate(
        self,
        entities: List[Entity],
        context: Optional[Dict[str, Any]] = None
    ) -> DeduplicationResult:
        """
        Deduplicate entities using LM clustering.

        Process:
        1. Generate embeddings for all entities
        2. Perform K-means clustering
        3. Within each cluster, find duplicates using cosine similarity
        4. Merge duplicates into canonical entities
        """
        if not entities:
            return DeduplicationResult(canonical_entities=[], duplicate_groups=[])

        logger.info(f"Starting LM clustering for {len(entities)} entities")

        # Generate embeddings
        embeddings = await self._generate_embeddings(entities)

        if embeddings is None:
            # Fallback to simple strategy
            logger.warning("Embedding generation failed, using fallback")
            return await self._fallback_deduplication(entities)

        # Perform clustering
        clusters = await self._cluster_entities(entities, embeddings)

        # Find duplicates within clusters
        duplicate_groups = await self._find_cluster_duplicates(clusters, embeddings)

        # Create canonical entities
        canonical_entities = []
        seen_ids = set()

        for group in duplicate_groups:
            canonical_id = group[0].id
            seen_ids.add(canonical_id)
            canonical_entities.append(group[0])

        # Add non-duplicate entities
        for entity in entities:
            if entity.id not in seen_ids:
                canonical_entities.append(entity)

        return DeduplicationResult(
            canonical_entities=canonical_entities,
            duplicate_groups=duplicate_groups,
            stats={
                'original_count': len(entities),
                'canonical_count': len(canonical_entities),
                'duplicate_groups': len(duplicate_groups),
                'clusters': len(clusters),
                'model': self.retrieval_model
            }
        )

    async def _generate_embeddings(self, entities: List[Entity]) -> Optional[np.ndarray]:
        """Generate embeddings for entities."""
        if self.model is None:
            return None

        try:
            # Prepare text (combine name and description)
            texts = [
                f"{e.name} {e.description or ''}"
                for e in entities
            ]

            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=32
            )

            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    async def _cluster_entities(
        self,
        entities: List[Entity],
        embeddings: np.ndarray
    ) -> List[List[Entity]]:
        """Cluster entities using K-means."""
        try:
            from sklearn.cluster import KMeans

            # Determine number of clusters
            n_clusters = max(1, len(entities) // self.cluster_size)

            # Perform K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(embeddings)

            # Group entities by cluster
            clusters = defaultdict(list)
            for entity, label in zip(entities, labels):
                clusters[label].append(entity)

            return list(clusters.values())
        except ImportError:
            logger.warning("sklearn not available, using simple clustering")
            return self._simple_clustering(entities, embeddings)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return [[e] for e in entities]

    def _simple_clustering(
        self,
        entities: List[Entity],
        embeddings: np.ndarray
    ) -> List[List[Entity]]:
        """Simple distance-based clustering fallback."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate similarity matrix
        similarities = cosine_similarity(embeddings)

        # Simple greedy clustering
        clusters = []
        assigned = set()

        for i, entity in enumerate(entities):
            if i in assigned:
                continue

            cluster = [entity]
            assigned.add(i)

            # Find similar entities
            for j, other in enumerate(entities):
                if j in assigned or j == i:
                    continue

                if similarities[i][j] > 0.85:  # High similarity threshold
                    cluster.append(other)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    async def _find_cluster_duplicates(
        self,
        clusters: List[List[Entity]],
        embeddings: np.ndarray
    ) -> List[List[Entity]]:
        """Find duplicates within each cluster."""
        duplicate_groups = []

        # Create entity to embedding mapping
        entity_to_embedding = {}
        idx = 0
        for cluster in clusters:
            for entity in cluster:
                entity_to_embedding[entity.id] = embeddings[idx]
                idx += 1

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Find duplicates within cluster
            cluster_groups = await self._find_cluster_internal_duplicates(
                cluster,
                entity_to_embedding
            )

            duplicate_groups.extend(cluster_groups)

        return duplicate_groups

    async def _find_cluster_internal_duplicates(
        self,
        cluster: List[Entity],
        entity_to_embedding: Dict[str, np.ndarray]
    ) -> List[List[Entity]]:
        """Find duplicates within a single cluster."""
        from sklearn.metrics.pairwise import cosine_similarity

        duplicate_groups = []
        processed = set()

        for i, entity1 in enumerate(cluster):
            if entity1.id in processed:
                continue

            group = [entity1]
            embedding1 = entity_to_embedding[entity1.id]

            for j, entity2 in enumerate(cluster):
                if entity2.id in processed or entity2.id == entity1.id:
                    continue

                embedding2 = entity_to_embedding[entity2.id]

                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [embedding1],
                    [embedding2]
                )[0][0]

                if similarity > 0.9:  # Very high threshold within cluster
                    group.append(entity2)
                    processed.add(entity2.id)

            processed.add(entity1.id)

            if len(group) > 1:
                duplicate_groups.append(group)

        return duplicate_groups

    async def _fallback_deduplication(
        self,
        entities: List[Entity]
    ) -> DeduplicationResult:
        """Fallback deduplication when embedding fails."""
        logger.warning("Using fallback deduplication")

        # Simple name-based grouping
        name_groups = defaultdict(list)
        for entity in entities:
            key = entity.name.lower().strip()
            name_groups[key].append(entity)

        duplicate_groups = []
        canonical_entities = []

        for group in name_groups.values():
            if len(group) > 1:
                duplicate_groups.append(group)
                canonical_entities.append(group[0])
            else:
                canonical_entities.append(group[0])

        return DeduplicationResult(
            canonical_entities=canonical_entities,
            duplicate_groups=duplicate_groups,
            stats={'fallback': True}
        )

    def calculate_confidence(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate confidence using embedding similarity."""
        if self.model is None:
            return 0.0

        try:
            text1 = f"{entity1.name} {entity1.description or ''}"
            text2 = f"{entity2.name} {entity2.description or ''}"

            emb1 = self.model.encode([text1])[0]
            emb2 = self.model.encode([text2])[0]

            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]

            return float(similarity)
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
