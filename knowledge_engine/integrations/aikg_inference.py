"""
AI-Knowledge-Graph Relationship Inference Integration

This module integrates ai-knowledge-graph's relationship inference capabilities,
providing multiple strategies for inferring new relationships.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


class InferenceResult:
    """Result of relationship inference process."""

    def __init__(
        self,
        original_triples: List,
        inferred_triples: List,
        confidence_scores: Dict[Tuple[str, str, str], float],
        inference_sources: Dict[Tuple[str, str, str], str]
    ):
        self.original_triples = original_triples
        self.inferred_triples = inferred_triples
        self.confidence_scores = confidence_scores
        self.inference_sources = inference_sources  # triple -> source method

    @property
    def all_triples(self) -> List:
        """Get all triples (original + inferred)."""
        return self.original_triples + self.inferred_triples

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            'original_triples': len(self.original_triples),
            'inferred_triples': len(self.inferred_triples),
            'total_triples': len(self.all_triples),
            'avg_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores)
            if self.confidence_scores else 0.0,
            'inference_methods': list(set(self.inference_sources.values()))
        }


class AIKGRelationshipInference:
    """
    Infers new relationships using multiple strategies.

    Strategies:
    1. Transitive inference (A→B, B→C → A→C)
    2. LLM-based inter-community inference
    3. Within-community inference
    4. Lexical similarity inference
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the relationship inference engine.

        Args:
            config: Configuration dictionary with options:
                - apply_transitive: Whether to apply transitive inference
                - use_llm_for_inference: Whether to use LLM for inference
                - similarity_threshold: Threshold for lexical similarity (0-1)
                - max_inference_depth: Maximum depth for transitive inference
                - llm_client: Optional LLM client for advanced inference
        """
        self.apply_transitive = config.get('apply_transitive', True)
        self.use_llm_for_inference = config.get('use_llm_for_inference', False)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_inference_depth = config.get('max_inference_depth', 3)
        self.llm_client = config.get('llm_client')

        logger.info(
            f"AIKGRelationshipInference initialized: "
            f"transitive={self.apply_transitive}, "
            f"llm={self.use_llm_for_inference}"
        )

    async def infer_relationships(
        self,
        triples: List,
        entities: List
    ) -> InferenceResult:
        """
        Infer new relationships using all strategies.

        Args:
            triples: List of existing triples
            entities: List of entities

        Returns:
            InferenceResult with original and inferred triples
        """
        logger.info(f"Starting relationship inference for {len(triples)} triples")

        inferred_triples = []
        confidence_scores = {}
        inference_sources = {}

        # Strategy 1: Transitive inference
        if self.apply_transitive:
            transitive_triples = await self.transitive_inference(triples)
            for triple in transitive_triples:
                inferred_triples.append(triple)
                confidence_scores[triple.to_tuple()] = 0.8
                inference_sources[triple.to_tuple()] = 'transitive'
            logger.info(f"Transitive inference produced {len(transitive_triples)} triples")

        # Strategy 2: Within-community inference
        # Build graph for community detection
        graph = self._build_graph(triples)
        communities = self._detect_communities(graph)

        for community in communities:
            if len(community) > 1:
                community_triples = await self.within_community_inference(
                    community, triples
                )
                for triple in community_triples:
                    inferred_triples.append(triple)
                    confidence_scores[triple.to_tuple()] = 0.6
                    inference_sources[triple.to_tuple()] = 'within_community'
        logger.info(f"Within-community inference produced {len(inferred_triples)} triples")

        # Strategy 3: Lexical similarity inference
        lexical_triples = await self.lexical_similarity_inference(entities, triples)
        for triple in lexical_triples:
            inferred_triples.append(triple)
            confidence_scores[triple.to_tuple()] = 0.5
            inference_sources[triple.to_tuple()] = 'lexical_similarity'
        logger.info(f"Lexical similarity inference produced {len(lexical_triples)} triples")

        # Strategy 4: LLM-based inter-community inference (optional)
        if self.use_llm_for_inference and self.llm_client and communities:
            llm_triples = await self.llm_inter_community_inference(graph, communities)
            for triple in llm_triples:
                inferred_triples.append(triple)
                confidence_scores[triple.to_tuple()] = 0.7
                inference_sources[triple.to_tuple()] = 'llm_inter_community'
            logger.info(f"LLM inter-community inference produced {len(llm_triples)} triples")

        # Deduplicate inferred triples
        inferred_triples = await self.deduplicate_inferences(triples, inferred_triples)

        logger.info(f"Inference complete: {len(inferred_triples)} new relationships")

        return InferenceResult(
            original_triples=triples,
            inferred_triples=inferred_triples,
            confidence_scores=confidence_scores,
            inference_sources=inference_sources
        )

    async def transitive_inference(
        self,
        triples: List
    ) -> List:
        """
        Perform transitive inference.

        Logic:
        If A→B and B→C exist, infer A→C

        Example:
        - (Python, used_for, WebDev)
        - (WebDev, used_for, Django)
        → Infer: (Python, used_for, Django)

        Args:
            triples: List of existing triples

        Returns:
            List of inferred triples
        """
        inferred = []
        graph = self._build_graph(triples)

        # Find transitive paths up to max depth
        for depth in range(2, self.max_inference_depth + 1):
            # Find all paths of length 'depth'
            paths = self._find_paths_of_length(graph, depth)

            # Infer relationships from paths
            for path in paths:
                subject = path[0]
                obj = path[-1]

                # Use most common predicate along path
                predicates = []
                for i in range(len(path) - 1):
                    # Find predicate between path[i] and path[i+1]
                    for triple in triples:
                        if triple.subject == path[i] and triple.object == path[i+1]:
                            predicates.append(triple.predicate)
                            break

                if predicates:
                    # Use most common predicate
                    predicate = max(set(predicates), key=predicates.count)

                    # Create inferred triple
                    inferred_triple = type(triple)(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.8 / depth,  # Confidence decreases with depth
                        source='inferred'
                    )

                    # Check if not already in original or inferred
                    if not self._triple_exists(inferred_triple, triples + inferred):
                        inferred.append(inferred_triple)

        return inferred

    def _build_graph(self, triples: List) -> nx.DiGraph:
        """Build NetworkX graph from triples."""
        graph = nx.DiGraph()

        for triple in triples:
            graph.add_node(triple.subject)
            graph.add_node(triple.object)
            graph.add_edge(
                triple.subject,
                triple.object,
                predicate=triple.predicate,
                confidence=triple.confidence
            )

        return graph

    def _find_paths_of_length(
        self,
        graph: nx.DiGraph,
        length: int
    ) -> List[List[str]]:
        """Find all simple paths of specified length."""
        paths = []

        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        for path in nx.all_simple_paths(graph, source, target, cutoff=length):
                            if len(path) == length + 1:  # length+1 nodes = length edges
                                paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

        return paths

    def _triple_exists(self, triple: Any, triples: List) -> bool:
        """Check if triple already exists in list."""
        return any(
            t.subject == triple.subject and
            t.predicate == triple.predicate and
            t.object == triple.object
            for t in triples
        )

    def _detect_communities(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        Detect communities using connected components.

        Args:
            graph: NetworkX graph

        Returns:
            List of communities (each community is a list of node names)
        """
        # Convert to undirected for community detection
        undirected = graph.to_undirected()

        # Find connected components
        communities = []
        for component in nx.connected_components(undirected):
            communities.append(list(component))

        return communities

    async def llm_inter_community_inference(
        self,
        graph: nx.DiGraph,
        communities: List[List[str]]
    ) -> List:
        """
        Use LLM to infer relationships between disconnected communities.

        Process:
        1. Identify disconnected graph components
        2. For each component pair, ask LLM about potential relationships
        3. Validate inferred relationships
        4. Return high-confidence inferences

        Args:
            graph: NetworkX graph
            communities: List of communities

        Returns:
            List of inferred triples
        """
        if not self.llm_client:
            return []

        inferred = []

        # Compare each pair of communities
        for i, comm1 in enumerate(communities):
            for comm2 in communities[i+1:]:
                # Build LLM prompt
                prompt = self._build_inter_community_prompt(comm1, comm2)

                try:
                    # Call LLM
                    response = await self.llm_client(prompt)

                    # Parse response for relationships
                    relationships = self._parse_llm_relationships(response, comm1, comm2)

                    for rel in relationships:
                        inferred.append(rel)

                except Exception as e:
                    logger.error(f"LLM inter-community inference failed: {e}")
                    continue

        return inferred

    def _build_inter_community_prompt(
        self,
        comm1: List[str],
        comm2: List[str]
    ) -> str:
        """Build prompt for inter-community relationship inference."""
        return f"""Given two sets of related concepts:

Set A: {', '.join(comm1[:10])}{'...' if len(comm1) > 10 else ''}

Set B: {', '.join(comm2[:10])}{'...' if len(comm2) > 10 else ''}

Identify potential relationships that might exist between concepts in Set A and concepts in Set B.

Format your response as:
Subject1 | predicate | Object1
Subject2 | predicate | Object2

Only include high-confidence relationships. Use predicates like: related_to, similar_to, used_for, part_of, depends_on."""

    def _parse_llm_relationships(
        self,
        response: str,
        comm1: List[str],
        comm2: List[str]
    ) -> List:
        """Parse LLM response for relationships."""
        relationships = []
        lines = response.strip().split('\n')

        for line in lines:
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3:
                    subject, predicate, obj = parts

                    # Validate that subject and obj are in the communities
                    comm1_set = set(c.lower() for c in comm1)
                    comm2_set = set(c.lower() for c in comm2)

                    if (subject.lower() in comm1_set and obj.lower() in comm2_set) or \
                       (subject.lower() in comm2_set and obj.lower() in comm1_set):

                        from aikg_standardization import Triple
                        triple = Triple(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            confidence=0.7,
                            source='inferred'
                        )
                        relationships.append(triple)

        return relationships

    async def within_community_inference(
        self,
        community: List[str],
        triples: List
    ) -> List:
        """
        Infer missing relationships within a community.

        Uses:
        - Graph connectivity patterns
        - Common relationship types
        - Entity co-occurrence

        Args:
            community: List of entity names in the community
            triples: List of existing triples

        Returns:
            List of inferred triples
        """
        inferred = []

        # Build subgraph for this community
        community_triples = [
            t for t in triples
            if t.subject in community or t.object in community
        ]

        if len(community_triples) < 2:
            return inferred

        # Find common predicates in community
        predicate_counts = defaultdict(int)
        for triple in community_triples:
            predicate_counts[triple.predicate] += 1

        if not predicate_counts:
            return inferred

        # Most common predicate
        common_predicate = max(predicate_counts, key=predicate_counts.get)

        # Find disconnected entity pairs in community
        connected = set()
        for triple in community_triples:
            connected.add((triple.subject, triple.object))
            connected.add((triple.object, triple.subject))

        # Infer relationships between disconnected entities
        from aikg_standardization import Triple
        for i, entity1 in enumerate(community):
            for entity2 in community[i+1:]:
                if (entity1, entity2) not in connected:
                    # Infer relationship with common predicate
                    inferred.append(Triple(
                        subject=entity1,
                        predicate=common_predicate,
                        object=entity2,
                        confidence=0.6,
                        source='inferred'
                    ))

        return inferred[:10]  # Limit to avoid over-inference

    async def lexical_similarity_inference(
        self,
        entities: List,
        triples: List
    ) -> List:
        """
        Infer relationships based on lexical similarity.

        Process:
        1. Calculate word overlap between entity names
        2. If overlap > threshold, infer relationship
        3. Example: "machine learning" and "learning algorithm" → similar

        Args:
            entities: List of entities
            triples: List of existing triples

        Returns:
            List of inferred triples
        """
        inferred = []

        # Build entity name index
        entity_names = [e.name for e in entities]

        # Compare all pairs
        for i, name1 in enumerate(entity_names):
            for name2 in entity_names[i+1:]:
                # Calculate word overlap similarity
                similarity = self._calculate_lexical_similarity(name1, name2)

                if similarity >= self.similarity_threshold:
                    # Check if relationship doesn't exist
                    from aikg_standardization import Triple
                    if not self._triple_exists(
                        Triple(name1, 'similar_to', name2),
                        triples + inferred
                    ):
                        inferred.append(Triple(
                            subject=name1,
                            predicate='similar_to',
                            object=name2,
                            confidence=similarity,
                            source='inferred'
                        ))

        return inferred

    def _calculate_lexical_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate lexical similarity between two entity names.

        Uses word overlap ratio.
        """
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def deduplicate_inferences(
        self,
        original_triples: List,
        inferred_triples: List
    ) -> List:
        """
        Remove duplicate inferred triples.

        Priority:
        1. Original triples (keep)
        2. Inferred triples (only if not duplicate)
        3. Prefer high-confidence inferences

        Args:
            original_triples: List of original triples
            inferred_triples: List of inferred triples

        Returns:
            Deduplicated list of inferred triples
        """
        # Build set of original triples
        original_set = set()
        for triple in original_triples:
            original_set.add((triple.subject, triple.predicate, triple.object))

        # Filter inferred triples
        deduped = []
        seen = set()

        # Sort by confidence (highest first)
        sorted_inferred = sorted(
            inferred_triples,
            key=lambda t: t.confidence,
            reverse=True
        )

        for triple in sorted_inferred:
            key = (triple.subject, triple.predicate, triple.object)

            # Skip if in original
            if key in original_set:
                continue

            # Skip if already seen
            if key in seen:
                continue

            seen.add(key)
            deduped.append(triple)

        return deduped
