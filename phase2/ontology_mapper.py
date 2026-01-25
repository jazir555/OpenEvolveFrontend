"""
Ontology Mapper (Ψ₂)

Semantic mapping between problem domains using NLP and knowledge graphs.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import os
import json
import logging
import sqlite3
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MappingResult:
    """
    Result of ontology mapping between two domains
    """
    concept_mapping: Dict[str, str] = field(default_factory=dict)
    relation_mapping: Dict[str, str] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"MappingResult(concepts={len(self.concept_mapping)}, "
                f"relations={len(self.relation_mapping)}, "
                f"avg_confidence={np.mean(list(self.confidence.values())) if self.confidence else 0:.3f})")


class OntologyMapper:
    """
    Semantic ontology mapper for cross-domain knowledge transfer.

    Combines multiple similarity signals:
    - Lexical similarity (string matching)
    - Semantic similarity (embeddings)
    - Graph structural similarity (embeddings)
    - Knowledge graph validation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ontology mapper

        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = {
            # Lexical matching
            'lexical_threshold': 0.3,
            'similarity_method': 'jaro-winkler',

            # Semantic matching
            'semantic_model': 'all-MiniLM-L6-v2',
            'semantic_threshold': 0.5,
            'embedding_dim': 384,

            # Graph embedding
            'graph_embedding_dim': 64,
            'walk_length': 40,
            'num_walks': 20,
            'p': 1.0,
            'q': 1.0,
            'graph_threshold': 0.5,

            # Knowledge graph validation
            'kg_enabled': True,
            'kg_cache_size': 10000,
            'kg_timeout': 5,

            # Confidence aggregation
            'w_lexical': 0.15,
            'w_semantic': 0.40,
            'w_graph': 0.30,
            'w_kg': 0.15,
            'final_threshold': 0.5,

            # Performance
            'use_cache': True,
            'cache_dir': 'rese/phase2/ontology_cache',
        }

        # Update with user config
        if config:
            self.config.update(config)

        # Initialize components
        self.lexical_matcher = None
        self.semantic_matcher = None
        self.graph_embedder = None
        self.kg_validator = None

        # Cache
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.kg_cache_db: Optional[sqlite3.Connection] = None

        # Initialize
        self._initialize_components()
        self._initialize_cache()

        logger.info("Ontology Mapper initialized")

    def _initialize_components(self):
        """Initialize sub-components"""
        try:
            from .ontology_components.lexical_matcher import LexicalMatcher
            self.lexical_matcher = LexicalMatcher(
                threshold=self.config['lexical_threshold'],
                method=self.config['similarity_method']
            )
            logger.info("Lexical matcher initialized")
        except ImportError:
            logger.warning("Lexical matcher not available, using fallback")

        try:
            from .ontology_components.semantic_matcher import SemanticMatcher
            self.semantic_matcher = SemanticMatcher(
                model_name=self.config['semantic_model'],
                threshold=self.config['semantic_threshold']
            )
            logger.info("Semantic matcher initialized")
        except ImportError:
            logger.warning("Semantic matcher not available, using fallback")

        try:
            from .ontology_components.graph_embedder import GraphEmbedder
            self.graph_embedder = GraphEmbedder(
                dimensions=self.config['graph_embedding_dim'],
                walk_length=self.config['walk_length'],
                num_walks=self.config['num_walks'],
                p=self.config['p'],
                q=self.config['q']
            )
            logger.info("Graph embedder initialized")
        except ImportError:
            logger.warning("Graph embedder not available, using fallback")

        if self.config['kg_enabled']:
            try:
                from .ontology_components.kg_validator import KGValidator
                self.kg_validator = KGValidator(
                    cache_size=self.config['kg_cache_size'],
                    timeout=self.config['kg_timeout']
                )
                logger.info("KG validator initialized")
            except ImportError:
                logger.warning("KG validator not available")

    def _initialize_cache(self):
        """Initialize caching system"""
        if not self.config['use_cache']:
            return

        cache_dir = Path(self.config['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize KG cache database
        kg_cache_path = cache_dir / "kg_cache.db"
        self.kg_cache_db = sqlite3.connect(str(kg_cache_path), check_same_thread=False)
        self._create_kg_cache_tables()

        logger.info(f"Cache initialized at {cache_dir}")

    def _create_kg_cache_tables(self):
        """Create KG cache tables"""
        cursor = self.kg_cache_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_cache (
                concept1 TEXT,
                concept2 TEXT,
                relationship_type TEXT,
                score REAL,
                timestamp TEXT,
                PRIMARY KEY (concept1, concept2, relationship_type)
            )
        """)
        self.kg_cache_db.commit()

    def map_ontologies(
        self,
        source_domain: Any,
        target_domain: Any,
        use_stages: Optional[List[str]] = None
    ) -> MappingResult:
        """
        Map ontologies between source and target domains

        Args:
            source_domain: Source domain object
            target_domain: Target domain object
            use_stages: List of stages to use (default: all)

        Returns:
            MappingResult with concept and relation mappings
        """
        logger.info(f"Mapping {source_domain.name} → {target_domain.name}")

        # Default: use all stages
        if use_stages is None:
            use_stages = ['lexical', 'semantic', 'graph', 'kg', 'aggregate']

        # Stage 1: Preprocessing
        source_graph, source_concepts = self._preprocess_domain(source_domain)
        target_graph, target_concepts = self._preprocess_domain(target_domain)

        # Stage 2: Candidate generation (lexical)
        candidate_pairs = []
        if 'lexical' in use_stages:
            candidate_pairs = self._generate_candidates(
                source_concepts,
                target_concepts
            )

        # Stage 3: Semantic similarity
        semantic_scores = {}
        if 'semantic' in use_stages:
            semantic_scores = self._compute_semantic_similarity(
                candidate_pairs,
                source_concepts,
                target_concepts
            )

        # Stage 4: Graph embedding similarity
        graph_scores = {}
        if 'graph' in use_stages:
            graph_scores = self._compute_graph_similarity(
                source_graph,
                target_graph,
                source_concepts,
                target_concepts
            )

        # Stage 5: Knowledge graph validation
        kg_scores = {}
        if 'kg' in use_stages and self.kg_validator:
            kg_scores = self._validate_with_kg(candidate_pairs)

        # Stage 6: Confidence aggregation
        final_mapping = MappingResult()

        if 'aggregate' in use_stages:
            final_mapping = self._aggregate_confidence(
                candidate_pairs,
                {
                    'lexical': {},  # Already used in candidate generation
                    'semantic': semantic_scores,
                    'graph': graph_scores,
                    'kg': kg_scores
                },
                threshold=self.config['final_threshold']
            )

        # Add metadata
        final_mapping.metadata = {
            'algorithm': 'OntologyMapper',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'w_lexical': self.config['w_lexical'],
                'w_semantic': self.config['w_semantic'],
                'w_graph': self.config['w_graph'],
                'w_kg': self.config['w_kg'],
                'threshold': self.config['final_threshold']
            },
            'stages_used': use_stages,
            'source_domain': source_domain.name,
            'target_domain': target_domain.name
        }

        logger.info(f"Mapping complete: {len(final_mapping.concept_mapping)} concepts, "
                   f"{len(final_mapping.relation_mapping)} relations")

        return final_mapping

    def _preprocess_domain(self, domain: Any) -> Tuple[nx.Graph, List[str]]:
        """
        Preprocess domain into graph and concept list

        Args:
            domain: Domain object

        Returns:
            Tuple of (graph, concept_list)
        """
        # Extract graph from FDG if available
        if hasattr(domain, 'fdg') and domain.fdg is not None:
            graph = domain.fdg.to_networkx()
        else:
            # Create simple graph from constraints
            graph = nx.DiGraph()
            if hasattr(domain, 'formal_constraints'):
                for constraint in domain.formal_constraints:
                    if hasattr(constraint, 'variables'):
                        for var in constraint.variables:
                            graph.add_node(var)

        # Extract concepts (node names)
        concepts = list(graph.nodes())

        # Normalize concept names
        concepts = [self._normalize_concept(c) for c in concepts]

        return graph, concepts

    def _normalize_concept(self, concept: str) -> str:
        """
        Normalize concept name

        Args:
            concept: Raw concept name

        Returns:
            Normalized concept name
        """
        # Lowercase
        concept = concept.lower()

        # Remove special characters
        concept = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in concept)

        # Remove extra spaces
        concept = ' '.join(concept.split())

        return concept

    def _generate_candidates(
        self,
        source_concepts: List[str],
        target_concepts: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Generate candidate concept pairs using lexical similarity

        Args:
            source_concepts: Source domain concepts
            target_concepts: Target domain concepts

        Returns:
            List of (source, target, score) tuples
        """
        candidates = []

        for source in source_concepts:
            for target in target_concepts:
                # Compute lexical similarity
                if self.lexical_matcher:
                    score = self.lexical_matcher.similarity(source, target)
                else:
                    # Fallback: simple Jaro-Winkler
                    score = self._jaro_winkler_similarity(source, target)

                if score >= self.config['lexical_threshold']:
                    candidates.append((source, target, score))

        logger.info(f"Generated {len(candidates)} candidate pairs (lexical)")
        return candidates

    def _compute_semantic_similarity(
        self,
        candidate_pairs: List[Tuple[str, str, float]],
        source_concepts: List[str],
        target_concepts: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute semantic similarity for candidate pairs

        Args:
            candidate_pairs: Candidate pairs from lexical stage
            source_concepts: All source concepts
            target_concepts: All target concepts

        Returns:
            Dictionary mapping (source, target) → similarity score
        """
        if not self.semantic_matcher:
            return {}

        scores = {}

        # Process in batches for efficiency
        batch_size = 32

        for i in range(0, len(candidate_pairs), batch_size):
            batch = candidate_pairs[i:i + batch_size]

            for source, target, _ in batch:
                # Check cache first
                cache_key = f"{source}|||{target}"
                if cache_key in self.embedding_cache:
                    scores[(source, target)] = self.embedding_cache[cache_key]
                    continue

                # Compute semantic similarity
                score = self.semantic_matcher.similarity(source, target)
                scores[(source, target)] = score

                # Cache result
                self.embedding_cache[cache_key] = score

        logger.info(f"Computed semantic similarity for {len(scores)} pairs")
        return scores

    def _compute_graph_similarity(
        self,
        source_graph: nx.Graph,
        target_graph: nx.Graph,
        source_concepts: List[str],
        target_concepts: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute graph structural similarity

        Args:
            source_graph: Source domain graph
            target_graph: Target domain graph
            source_concepts: Source concepts
            target_concepts: Target concepts

        Returns:
            Dictionary mapping (source, target) → similarity score
        """
        if not self.graph_embedder:
            return {}

        # Generate graph embeddings
        source_embeddings = self.graph_embedder.fit_transform(source_graph)
        target_embeddings = self.graph_embedder.fit_transform(target_graph)

        scores = {}

        # Compute cosine similarity between all pairs
        for source in source_concepts:
            if source not in source_embeddings:
                continue

            for target in target_concepts:
                if target not in target_embeddings:
                    continue

                # Cosine similarity
                sim = self._cosine_similarity(
                    source_embeddings[source],
                    target_embeddings[target]
                )

                if sim >= self.config['graph_threshold']:
                    scores[(source, target)] = sim

        logger.info(f"Computed graph similarity for {len(scores)} pairs")
        return scores

    def _validate_with_kg(
        self,
        candidate_pairs: List[Tuple[str, str, float]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Validate candidate pairs using knowledge graphs

        Args:
            candidate_pairs: Candidate pairs

        Returns:
            Dictionary mapping (source, target) → KG confidence score
        """
        if not self.kg_validator:
            return {}

        scores = {}

        for source, target, _ in candidate_pairs:
            # Check cache
            cursor = self.kg_cache_db.cursor()
            cursor.execute(
                "SELECT score FROM kg_cache WHERE concept1=? AND concept2=?",
                (source, target)
            )
            row = cursor.fetchone()

            if row:
                scores[(source, target)] = row[0]
                continue

            # Query KG
            kg_score = self.kg_validator.validate_relation(source, target)

            if kg_score is not None:
                scores[(source, target)] = kg_score

                # Cache result
                cursor.execute(
                    "INSERT OR REPLACE INTO kg_cache VALUES (?, ?, ?, ?, ?)",
                    (source, target, 'related', kg_score, datetime.now().isoformat())
                )
                self.kg_cache_db.commit()

        logger.info(f"Validated {len(scores)} pairs with KG")
        return scores

    def _aggregate_confidence(
        self,
        candidate_pairs: List[Tuple[str, str, float]],
        scores: Dict[str, Dict[Tuple[str, str], float]],
        threshold: float = 0.5
    ) -> MappingResult:
        """
        Aggregate confidence scores from all sources

        Args:
            candidate_pairs: Candidate pairs with lexical scores
            scores: Dictionary of scores from different sources
            threshold: Final threshold for including mapping

        Returns:
            MappingResult with final mappings
        """
        result = MappingResult()

        for source, target, lexical_score in candidate_pairs:
            # Collect all available scores
            semantic_score = scores.get('semantic', {}).get((source, target), 0.0)
            graph_score = scores.get('graph', {}).get((source, target), 0.0)
            kg_score = scores.get('kg', {}).get((source, target), 0.0)

            # Weighted combination
            final_score = (
                self.config['w_lexical'] * lexical_score +
                self.config['w_semantic'] * semantic_score +
                self.config['w_graph'] * graph_score +
                self.config['w_kg'] * kg_score
            )

            # Apply threshold
            if final_score >= threshold:
                # Check for conflicts (one-to-one mapping)
                if source in result.concept_mapping:
                    # Keep the one with higher score
                    existing_score = result.confidence[source]
                    if final_score > existing_score:
                        result.concept_mapping[source] = target
                        result.confidence[source] = final_score
                else:
                    result.concept_mapping[source] = target
                    result.confidence[source] = final_score

        logger.info(f"Aggregated to {len(result.concept_mapping)} mappings (threshold={threshold})")
        return result

    # Utility methods

    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """
        Compute Jaro-Winkler similarity

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score [0, 1]
        """
        # Implementation of Jaro-Winkler similarity
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Match distance
        match_distance = max(len(s1), len(s2)) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        # Find matches
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)

        matches = 0
        transpositions = 0

        for i in range(len(s1)):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len(s2))

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        # Jaro similarity
        jaro = (
            (matches / len(s1) +
             matches / len(s2) +
             (matches - transpositions / 2) / matches) / 3
        )

        # Winkler modification
        prefix = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        prefix = min(prefix, 4)
        winkler = jaro + prefix * 0.1 * (1 - jaro)

        return winkler

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity [-1, 1]
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)

    def save_mapping(self, mapping: MappingResult, filepath: str):
        """
        Save mapping result to file

        Args:
            mapping: Mapping result to save
            filepath: Output file path
        """
        # Convert numpy floats to Python floats for JSON serialization
        serializable_data = {
            'concept_mapping': mapping.concept_mapping,
            'relation_mapping': mapping.relation_mapping,
            'confidence': {
                f"{k}": float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in mapping.confidence.items()
            },
            'metadata': mapping.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        logger.info(f"Saved mapping to {filepath}")

    def load_mapping(self, filepath: str) -> MappingResult:
        """
        Load mapping result from file

        Args:
            filepath: Input file path

        Returns:
            MappingResult
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        result = MappingResult(
            concept_mapping=data['concept_mapping'],
            relation_mapping=data.get('relation_mapping', {}),
            confidence={
                tuple(k.split('|||')) if '|||' in k else k: v
                for k, v in data['confidence'].items()
            },
            metadata=data.get('metadata', {})
        )

        logger.info(f"Loaded mapping from {filepath}")
        return result


# Convenience functions

def create_mapper(config: Optional[Dict[str, Any]] = None) -> OntologyMapper:
    """
    Create ontology mapper instance

    Args:
        config: Optional configuration

    Returns:
        OntologyMapper instance
    """
    return OntologyMapper(config)


def map_domains(
    source_domain: Any,
    target_domain: Any,
    config: Optional[Dict[str, Any]] = None
) -> MappingResult:
    """
    Map ontologies between two domains (convenience function)

    Args:
        source_domain: Source domain
        target_domain: Target domain
        config: Optional configuration

    Returns:
        MappingResult
    """
    mapper = OntologyMapper(config)
    return mapper.map_ontologies(source_domain, target_domain)


if __name__ == "__main__":
    # Demo
    print("Ontology Mapper (Ψ₂)")
    print("=" * 50)

    # Create mapper
    mapper = create_mapper()

    print("\n✅ Ontology Mapper initialized")
    print("\nReady for domain mapping!")
    print("\nExample usage:")
    print("  from phase2.ontology_mapper import map_domains")
    print("  result = map_domains(source_domain, target_domain)")
