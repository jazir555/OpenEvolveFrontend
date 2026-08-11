"""
Semantic Analyzer for Enhanced Domain Context Analysis

This module provides semantic analysis capabilities for problem decomposition,
including concept extraction, relationship analysis, and semantic clustering.

Key Features:
- LLM-based concept extraction with NLP fallback
- Semantic relationship analysis between concepts
- Graph-based clustering for decomposition guidance
- Rich domain context generation
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# MIGRATION: Fixed imports for semantic_analyzer.py
try:
    from sovereign_data_models import ProblemDefinition
except ImportError as e:
    logging.warning(f"Failed to import from sovereign_data_models: {e}")

try:
    from crewai_state_management import SolutionAttempt
except ImportError:
    SolutionAttempt = None  # type: ignore

def _generate_id(prefix: str = ""):
    import uuid
    return f"{prefix}_{str(uuid.uuid4())[:8]}" if prefix else str(uuid.uuid4())[:8]

try:
    from sovereign_data_models import generate_id
except ImportError:
    generate_id = _generate_id

DomainContext = None  # type: ignore - STUB

EnhancedDomainContext = None  # type: ignore - STUB


logger = logging.getLogger(__name__)


# Import OpenEvolve client
try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
except ImportError:
    logger.warning("OpenEvolveClient not found. LLM features will use fallback.")
    OpenEvolveClient = None
    OPENEVOLVE_AVAILABLE = False


class SemanticAnalyzer:
    """
    Analyzes semantic relationships in problem descriptions.

    Provides:
    - Key concept extraction from problem text
    - Relationship analysis between concepts
    - Semantic clustering for decomposition boundaries
    - Enhanced domain context generation
    """

    def __init__(self, llm_client: Optional['OpenEvolveClient'] = None):
        """
        Initialize the semantic analyzer.

        Args:
            llm_client: Optional LLM client for deep analysis
        """
        self.llm_client = llm_client
        self._init_client()

        # NLP stopwords for fallback extraction
        self._stopwords = self._get_default_stopwords()

    def _init_client(self):
        """Initialize LLM client if not provided."""
        if not self.llm_client and OPENEVOLVE_AVAILABLE:
            try:
                self.llm_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for semantic analysis")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to instantiate OpenEvolve client: {e}")
                self.llm_client = None

    # ==========================================================================
    # MAIN ANALYSIS METHODS
    # ==========================================================================

    def extract_key_concepts(
        self,
        problem: ProblemDefinition,
        domain: str,
        max_concepts: int = 15
    ) -> List[str]:
        """
        Extract key concepts from problem description.

        Args:
            problem: The problem definition
            domain: Domain context
            max_concepts: Maximum number of concepts to extract

        Returns:
            List of important concepts/entities
        """
        logger.info(f"Extracting key concepts for problem {problem.id}")

        # Try LLM-based extraction first
        if self.llm_client:
            try:
                concepts = self._extract_concepts_with_llm(
                    problem.description,
                    domain,
                    max_concepts
                )
                if concepts:
                    logger.info(f"Extracted {len(concepts)} concepts using LLM")
                    return concepts
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"LLM concept extraction failed: {e}, using NLP fallback")

        # Fallback to NLP-based extraction
        concepts = self._extract_concepts_with_nlp(problem.description, max_concepts)
        logger.info(f"Extracted {len(concepts)} concepts using NLP fallback")
        return concepts

    def analyze_concept_relationships(
        self,
        concepts: List[str],
        problem_text: str
    ) -> Dict[str, List[str]]:
        """
        Analyze relationships between concepts.

        Args:
            concepts: List of concepts to analyze
            problem_text: Original problem text for context

        Returns:
            Mapping: concept -> [related concepts]
            Relationships include: depends_on, similar_to, part_of, conflicts_with
        """
        logger.info(f"Analyzing relationships for {len(concepts)} concepts")

        if not concepts:
            return {}

        # Try LLM-based analysis first
        if self.llm_client:
            try:
                relationships = self._analyze_relationships_with_llm(
                    concepts,
                    problem_text
                )
                if relationships:
                    logger.info(f"Identified relationships using LLM")
                    return relationships
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"LLM relationship analysis failed: {e}, using heuristic fallback")

        # Fallback to heuristic analysis
        relationships = self._analyze_relationships_heuristic(concepts, problem_text)
        logger.info(f"Identified relationships using heuristics")
        return relationships

    def identify_semantic_clusters(
        self,
        concepts: List[str],
        relationships: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        Identify clusters of semantically related concepts.

        Uses graph analysis to find connected components.

        Args:
            concepts: List of all concepts
            relationships: Concept relationships

        Returns:
            List of clusters, each containing related concepts
        """
        logger.info(f"Identifying semantic clusters for {len(concepts)} concepts")

        if not concepts:
            return []

        # Use graph-based clustering
        clusters = self._identify_clusters_with_graph_analysis(
            concepts,
            relationships
        )

        logger.info(f"Identified {len(clusters)} semantic clusters")
        return clusters

    def build_enhanced_domain_context(
        self,
        problem: ProblemDefinition,
        base_context: Optional[DomainContext] = None
    ) -> EnhancedDomainContext:
        """
        Build complete enhanced domain context.

        Combines semantic analysis with domain knowledge.

        Args:
            problem: The problem definition
            base_context: Optional base domain context

        Returns:
            EnhancedDomainContext with rich semantic information
        """
        logger.info(f"Building enhanced domain context for problem {problem.id}")

        # Extract base information
        domain = base_context.domain if base_context else problem.domain_context.domain
        subdomain = base_context.subdomain if base_context else problem.domain_context.subdomain

        # Extract concepts
        concepts = self.extract_key_concepts(problem, domain)

        # Analyze relationships
        relationships = self.analyze_concept_relationships(
            concepts,
            problem.description
        )

        # Identify clusters
        clusters = self.identify_semantic_clusters(concepts, relationships)

        # Extract terminology
        terminology = self._extract_terminology(problem, concepts, domain)

        # Analyze domain characteristics
        domain_complexity = self._assess_domain_complexity(problem, concepts)
        abstraction_level = self._assess_abstraction_level(problem, concepts)
        decomposition_approach = self._suggest_decomposition_approach(
            problem,
            concepts,
            clusters
        )

        # Identify patterns and best practices
        domain_patterns = self._identify_domain_patterns(problem, domain)
        best_practices = self._identify_best_practices(problem, domain)

        # Create enhanced context
        enhanced_context = EnhancedDomainContext(
            domain=domain,
            subdomain=subdomain,
            related_domains=base_context.related_domains if base_context else [],
            domain_knowledge=base_context.domain_knowledge if base_context else {},
            key_concepts=concepts,
            concept_relationships=relationships,
            semantic_clusters=clusters,
            terminology=terminology,
            domain_complexity=domain_complexity,
            abstraction_level=abstraction_level,
            typical_decomposition_approach=decomposition_approach,
            similar_problems=[],  # Would be populated from knowledge base
            domain_patterns=domain_patterns,
            best_practices=best_practices,
            context_sources=["semantic_analysis"],
            confidence_score=self._calculate_confidence_score(
                concepts,
                relationships,
                clusters
            ),
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "num_concepts": len(concepts),
                "num_clusters": len(clusters),
                "analyzer_version": "1.0"
            }
        )

        logger.info(f"Built enhanced context with {len(concepts)} concepts in {len(clusters)} clusters")
        return enhanced_context

    # ==========================================================================
    # LLM-BASED EXTRACTION METHODS
    # ==========================================================================

    def _extract_concepts_with_llm(
        self,
        problem_text: str,
        domain: str,
        max_concepts: int
    ) -> List[str]:
        """Use LLM to extract key concepts."""
        prompt = f"""You are an expert in {domain} domain analysis.

Extract the {max_concepts} most important concepts, entities, or technologies from this problem description.

PROBLEM:
{problem_text}

GUIDELINES:
- Focus on technical terms, domain-specific concepts, and key technologies
- Include both explicit and implicit concepts
- Prioritize actionable and measurable concepts
- Avoid generic terms unless they're critical to the domain

Return ONLY a JSON list of concept strings. Format:
["concept1", "concept2", "concept3", ...]"""

        result = self.llm_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=1000
        )

        if result.success and result.best_code:
            try:
                # Parse JSON response
                concepts = json.loads(result.best_code.strip())
                if isinstance(concepts, list):
                    return [str(c).strip() for c in concepts[:max_concepts]]
            except json.JSONDecodeError:
                # Try to extract list from text
                concepts = self._extract_list_from_text(result.best_code)
                if concepts:
                    return concepts[:max_concepts]

        return []

    def _analyze_relationships_with_llm(
        self,
        concepts: List[str],
        problem_text: str
    ) -> Dict[str, List[str]]:
        """Use LLM to identify semantic relationships."""
        prompt = f"""You are an expert in semantic relationship analysis.

Analyze relationships between these concepts in the context of the problem:

CONCEPTS:
{', '.join(concepts)}

PROBLEM CONTEXT:
{problem_text}

For each concept, identify its relationships to other concepts:
- depends_on: This concept requires the other to function or exist
- similar_to: This concept is similar or related to the other
- part_of: This concept is a component or sub-concept of the other
- conflicts_with: This concept conflicts or competes with the other

Return ONLY a JSON object. Format:
{{
  "concept1": ["concept2", "concept3"],
  "concept2": ["concept4"],
  ...
}}

Note: Only include direct, meaningful relationships. If concept A relates to concept B,
you don't need to list B relating to A (relationships are bidirectional by default)."""

        result = self.llm_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=2000
        )

        if result.success and result.best_code:
            try:
                relationships = json.loads(result.best_code.strip())
                if isinstance(relationships, dict):
                    # Validate that all referenced concepts exist
                    valid_relationships = {}
                    for concept, relations in relationships.items():
                        if concept in concepts:
                            valid_relations = [r for r in relations if r in concepts]
                            if valid_relations:
                                valid_relationships[concept] = valid_relations
                    return valid_relationships
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM relationship response")

        return {}

    # ==========================================================================
    # NLP-BASED FALLBACK METHODS
    # ==========================================================================

    def _extract_concepts_with_nlp(
        self,
        text: str,
        max_concepts: int
    ) -> List[str]:
        """
        Use NLP techniques to extract concepts.

        Implements:
        - Named entity extraction
        - Technical term extraction
        - Frequent noun/phrase extraction
        """
        concepts = []

        # 1. Extract capitalized words (likely named entities)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        concepts.extend(capitalized)

        # 2. Extract technical terms (words with numbers, hyphens, etc.)
        technical = re.findall(r'\b\w+(?:-\w+)+\b|\b\w+\d+\b', text)
        concepts.extend(technical)

        # 3. Extract noun phrases (simple heuristic: 2-3 word sequences)
        words = text.split()
        for i in range(len(words) - 1):
            # 2-word phrases
            if words[i][0].isupper() and words[i+1][0].isupper():
                phrase = f"{words[i]} {words[i+1]}"
                concepts.append(phrase)

        # 4. Count frequency and get top terms
        concept_counts = Counter(concepts)

        # Filter out stopwords and common words
        filtered_concepts = [
            concept for concept, count in concept_counts.most_common(max_concepts * 2)
            if concept.lower() not in self._stopwords and len(concept) > 2
        ]

        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for concept in filtered_concepts:
            normalized = concept.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_concepts.append(concept)

        return unique_concepts[:max_concepts]

    def _analyze_relationships_heuristic(
        self,
        concepts: List[str],
        problem_text: str
    ) -> Dict[str, List[str]]:
        """
        Use heuristics to identify concept relationships.

        Implements:
        - Co-occurrence analysis
        - Shared keyword matching
        - Proximity in text
        """
        relationships = defaultdict(set)

        # Convert concepts to lowercase for matching
        concept_lower = {c.lower(): c for c in concepts}
        words = problem_text.lower().split()

        # 1. Co-occurrence analysis
        # If concepts appear near each other (within 5 words), they're related
        concept_positions = {}
        for idx, word in enumerate(words):
            if word in concept_lower:
                concept = concept_lower[word]
                if concept not in concept_positions:
                    concept_positions[concept] = []
                concept_positions[concept].append(idx)

        # Find concepts that appear near each other
        for concept1, positions1 in concept_positions.items():
            for concept2, positions2 in concept_positions.items():
                if concept1 >= concept2:
                    continue

                # Check if any positions are within 5 words
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) <= 5:
                            relationships[concept1].add(concept2)
                            relationships[concept2].add(concept1)
                            break

        # 2. Shared keyword matching
        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 >= concept2:
                    continue

                # Check if concepts share keywords
                words1 = set(concept1.lower().split())
                words2 = set(concept2.lower().split())

                if words1 & words2:  # Intersection
                    relationships[concept1].add(concept2)
                    relationships[concept2].add(concept1)

        return {k: list(v) for k, v in relationships.items()}

    # ==========================================================================
    # CLUSTERING METHODS
    # ==========================================================================

    def _identify_clusters_with_graph_analysis(
        self,
        concepts: List[str],
        relationships: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        Use graph analysis to find semantic clusters.

        Algorithm:
        1. Build graph where nodes = concepts
        2. Add edges for relationships
        3. Find connected components
        4. Each component = semantic cluster
        """
        # Build adjacency list
        graph = defaultdict(set)
        for concept, relations in relationships.items():
            graph[concept].update(relations)
            for related in relations:
                graph[related].add(concept)

        # Find connected components using BFS
        visited = set()
        clusters = []

        for concept in concepts:
            if concept not in visited:
                # Start BFS to find this component
                cluster = []
                queue = [concept]
                visited.add(concept)

                while queue:
                    current = queue.pop(0)
                    cluster.append(current)

                    # Add unvisited neighbors
                    for neighbor in graph.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if cluster:
                    clusters.append(cluster)

        return clusters

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def _extract_terminology(
        self,
        problem: ProblemDefinition,
        concepts: List[str],
        domain: str
    ) -> Dict[str, str]:
        """Extract domain-specific terminology with definitions."""
        # For now, return simple definitions
        # In production, this could use a domain knowledge base
        terminology = {}

        # Extract terms from concepts that are likely domain-specific
        for concept in concepts:
            if len(concept.split()) > 1:  # Multi-word terms
                # Create a simple definition based on context
                terminology[concept] = f"Domain-specific concept related to {domain}"

        return terminology

    def _assess_domain_complexity(
        self,
        problem: ProblemDefinition,
        concepts: List[str]
    ) -> float:
        """Assess complexity of the domain (0-1)."""
        # Base complexity from problem complexity score
        base_complexity = problem.complexity_score.overall_complexity / 10.0

        # Adjust based on number of concepts
        # More concepts = potentially more complex
        concept_factor = min(len(concepts) / 20.0, 0.3)

        return min(base_complexity + concept_factor, 1.0)

    def _assess_abstraction_level(
        self,
        problem: ProblemDefinition,
        concepts: List[str]
    ) -> str:
        """Assess abstraction level: low, medium, or high."""
        # Heuristic based on concept types
        abstract_terms = ["architecture", "design", "framework", "paradigm"]
        concrete_terms = ["implementation", "code", "database", "api", "function"]

        abstract_count = sum(1 for c in concepts if any(t in c.lower() for t in abstract_terms))
        concrete_count = sum(1 for c in concepts if any(t in c.lower() for t in concrete_terms))

        if abstract_count > concrete_count:
            return "high"
        elif concrete_count > abstract_count:
            return "low"
        else:
            return "medium"

    def _suggest_decomposition_approach(
        self,
        problem: ProblemDefinition,
        concepts: List[str],
        clusters: List[List[str]]
    ) -> str:
        """Suggest the best decomposition approach for this problem."""
        # If we have clear clusters, semantic decomposition works well
        if len(clusters) >= 3:
            return "semantic"

        # If problem has clear dependencies, use dependency-based
        if "depends_on" in problem.description.lower() or "prerequisite" in problem.description.lower():
            return "dependency"

        # Default to hybrid
        return "hybrid"

    def _identify_domain_patterns(
        self,
        problem: ProblemDefinition,
        domain: str
    ) -> List[str]:
        """Identify common patterns in this domain."""
        # In production, this would query a knowledge base
        patterns = []

        # Simple heuristic patterns
        if domain.lower() in ["software", "engineering", "technology"]:
            patterns = ["layered architecture", "separation of concerns", "modularity"]

        return patterns

    def _identify_best_practices(
        self,
        problem: ProblemDefinition,
        domain: str
    ) -> List[str]:
        """Identify domain-specific best practices."""
        # In production, this would query a knowledge base
        practices = []

        # Simple heuristic practices
        if domain.lower() in ["software", "engineering"]:
            practices = ["test-driven development", "code review", "documentation"]

        return practices

    def _calculate_confidence_score(
        self,
        concepts: List[str],
        relationships: Dict[str, List[str]],
        clusters: List[List[str]]
    ) -> float:
        """Calculate confidence in the semantic analysis (0-1)."""
        # Start with base confidence
        confidence = 0.5

        # More concepts = higher confidence (up to a point)
        concept_factor = min(len(concepts) / 10.0, 0.2)
        confidence += concept_factor

        # More relationships = higher confidence
        total_relations = sum(len(rels) for rels in relationships.values())
        relation_factor = min(total_relations / 20.0, 0.2)
        confidence += relation_factor

        # Good clustering (not too many singletons) = higher confidence
        if clusters:
            non_singletons = sum(1 for c in clusters if len(c) > 1)
            cluster_factor = non_singletons / len(clusters)
            confidence += cluster_factor * 0.1

        return min(confidence, 1.0)

    def _get_default_stopwords(self) -> Set[str]:
        """Get default stopwords for NLP extraction."""
        return {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "that", "this", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "what", "which", "who", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "into", "over", "after", "before", "between"
        }

    def _extract_list_from_text(self, text: str) -> List[str]:
        """Extract a list from text that might contain JSON-like content."""
        # Try to find JSON-like list in text
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
                if isinstance(items, list):
                    return items
            except json.JSONDecodeError:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {__name__}", exc_info=True)
                raise  # Re-raise the exception

        # Try to extract quoted strings
        matches = re.findall(r'"([^"]+)"', text)
        if matches:
            return matches

        return []
