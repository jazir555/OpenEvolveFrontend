"""
RAGBits-Enhanced Agent Tools

Advanced tools for task agents that leverage RAGBits semantic search
and knowledge retrieval capabilities.

All tools are designed to fail gracefully and return sensible defaults
when RAGBits is unavailable or errors occur.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from knowledge_engine.ragbits_retriever import RAGBitsEnhancedRetriever, get_ragbits_retriever
    from knowledge_engine.ragbits_safety import (
        safe_execute,
        validate_query,
        validate_top_k,
        validate_filters,
        generate_fallback_result,
        get_safety_manager
    )
except ImportError:
    RAGBitsEnhancedRetriever = None
    get_ragbits_retriever = None
    # Define fallback safety functions
    def safe_execute(fallback_value=None, log_errors=True, reraise=False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_errors:
                        logging.getLogger(__name__).error(f"Error in {func.__name__}: {e}")
                    return fallback_value if not reraise else None
            return wrapper
        return decorator

    def validate_query(query): return query is not None and isinstance(query, str) and len(query.strip()) > 0
    def validate_top_k(top_k): return max(1, min(100, int(top_k) if isinstance(top_k, int) else 5))
    def validate_filters(filters): return filters if isinstance(filters, dict) else {}
    def generate_fallback_result(query, result_type): return {"content": f"Fallback: {query}", "score": 0.5}
    def get_safety_manager(): return None

from ragbits_integration.agents.base_agent import AgentTool

logger = logging.getLogger(__name__)


class RAGBitsKnowledgeSearchTool(AgentTool):
    """
    Enhanced knowledge search tool powered by RAGBits.

    Provides agents with advanced semantic search capabilities:
    - Similar solution search
    - Decomposition pattern search
    - Critique pattern search
    - Verification benchmark search
    - Context-aware search

    Usage:
        tool = RAGBitsKnowledgeSearchTool()
        results = await tool.execute(
            search_type="similar_solutions",
            query="microservices authentication",
            top_k=5,
            filters={"stage": "stage_3"}
        )
    """

    def __init__(self, retriever: Optional[RAGBitsEnhancedRetriever] = None):
        """
        Initialize the RAGBits knowledge search tool.

        Args:
            retriever: RAGBitsEnhancedRetriever instance (uses singleton if not provided)
        """
        super().__init__(
            name="ragbits_knowledge_search",
            description="Advanced semantic search for solutions, patterns, and historical knowledge"
        )

        if retriever:
            self.retriever = retriever
        elif get_ragbits_retriever:
            self.retriever = get_ragbits_retriever()
        else:
            logger.warning("⚠️ RAGBits retriever not available, tool will have limited functionality")
            self.retriever = None

    async def execute(
        self,
        search_type: str,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute knowledge search using RAGBits.

        Args:
            search_type: Type of search
                - "similar_solutions": Find similar solutions from history
                - "decomposition_patterns": Find decomposition strategies
                - "critique_patterns": Find critique and feedback patterns
                - "verification_benchmarks": Find test cases and benchmarks
                - "contextual": Context-aware search
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (stage, team, document_type, etc.)
            **kwargs: Additional search parameters
                - context: Workflow context (for contextual search)
                - min_success_rate: Minimum success rate (for solutions)
                - min_depth/max_depth: Decomposition depth range
                - critique_type: Type of critique
                - min_coverage: Minimum test coverage

        Returns:
            List of search results with metadata and scores

        Example:
            >>> results = await tool.execute(
            ...     search_type="similar_solutions",
            ...     query="REST API authentication",
            ...     top_k=3,
            ...     filters={"stage": "stage_3"},
            ...     min_success_rate=0.8
            ... )
        """
        logger.info(f"🔍 RAGBits {search_type} search: {query[:100]}...")

        if not self.retriever:
            logger.warning("⚠️ RAGBits retriever not available, returning empty results")
            return []

        try:
            if search_type == "similar_solutions":
                return await self.retriever.search_similar_solutions(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    min_success_rate=kwargs.get("min_success_rate", 0.0)
                )

            elif search_type == "decomposition_patterns":
                return await self.retriever.search_decomposition_patterns(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    min_depth=kwargs.get("min_depth", 2),
                    max_depth=kwargs.get("max_depth", 10)
                )

            elif search_type == "critique_patterns":
                return await self.retriever.search_critique_patterns(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    critique_type=kwargs.get("critique_type")
                )

            elif search_type == "verification_benchmarks":
                return await self.retriever.search_verification_benchmarks(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    min_coverage=kwargs.get("min_coverage", 0.7)
                )

            elif search_type == "contextual":
                context = kwargs.get("context", {})
                return await self.retriever.search_contextual_knowledge(
                    query=query,
                    context=context,
                    top_k=top_k
                )

            else:
                logger.warning(f"❓ Unknown search type: {search_type}")
                return []

        except Exception as e:
            logger.error(f"❌ RAGBits search failed: {e}")
            return []


class RAGBitsContextGathererTool(AgentTool):
    """
    Tool for gathering comprehensive context using RAGBits.

    Automatically searches for relevant context across multiple dimensions:
    - Similar solutions
    - Decomposition patterns
    - Relevant critiques
    - Verification benchmarks

    Usage:
        tool = RAGBitsContextGathererTool()
        context = await tool.execute(
            query="user authentication system",
            sub_problem_id="sub_1",
            stage="stage_3"
        )
    """

    def __init__(self, retriever: Optional[RAGBitsEnhancedRetriever] = None):
        """Initialize the context gatherer tool"""
        super().__init__(
            name="ragbits_context_gatherer",
            description="Gather comprehensive context from multiple knowledge sources"
        )

        if retriever:
            self.retriever = retriever
        elif get_ragbits_retriever:
            self.retriever = get_ragbits_retriever()
        else:
            self.retriever = None

    async def execute(
        self,
        query: str,
        sub_problem_id: Optional[str] = None,
        stage: Optional[str] = None,
        team: Optional[str] = None,
        max_results_per_category: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Gather comprehensive context for a task.

        Args:
            query: Query describing the context needed
            sub_problem_id: Sub-problem identifier
            stage: Workflow stage
            team: Team identifier
            max_results_per_category: Max results per search category
            **kwargs: Additional parameters

        Returns:
            Comprehensive context dict with multiple knowledge categories

        Example:
            >>> context = await tool.execute(
            ...     query="API rate limiting",
            ...     sub_problem_id="sub_2",
            ...     stage="stage_3",
            ...     team="blue"
            ... )
            >>> print(context["similar_solutions"])  # List of similar solutions
            >>> print(context["decomposition_patterns"])  # Decomposition strategies
        """
        logger.info(f"📚 Gathering context for: {query[:100]}...")

        if not self.retriever:
            return {
                "query": query,
                "similar_solutions": [],
                "decomposition_patterns": [],
                "critique_patterns": [],
                "verification_benchmarks": [],
                "warnings": ["RAGBits retriever not available"]
            }

        # Build filters
        filters = {}
        if sub_problem_id:
            filters["sub_problem_id"] = sub_problem_id
        if stage:
            filters["stage"] = stage
        if team:
            filters["team"] = team

        try:
            # Parallel searches across categories
            similar_solutions = await self.retriever.search_similar_solutions(
                query=query,
                top_k=max_results_per_category,
                filters=filters
            )

            decomposition_patterns = await self.retriever.search_decomposition_patterns(
                query=query,
                top_k=max_results_per_category,
                filters=filters
            )

            critique_patterns = await self.retriever.search_critique_patterns(
                query=query,
                top_k=max_results_per_category,
                filters=filters
            )

            verification_benchmarks = await self.retriever.search_verification_benchmarks(
                query=query,
                top_k=max_results_per_category,
                filters=filters
            )

            context = {
                "query": query,
                "filters": filters,
                "similar_solutions": similar_solutions,
                "decomposition_patterns": decomposition_patterns,
                "critique_patterns": critique_patterns,
                "verification_benchmarks": verification_benchmarks,
                "gathered_at": datetime.utcnow().isoformat()
            }

            logger.info(f"✅ Gathered context: {len(similar_solutions)} solutions, "
                       f"{len(decomposition_patterns)} patterns, "
                       f"{len(critique_patterns)} critiques, "
                       f"{len(verification_benchmarks)} benchmarks")

            return context

        except Exception as e:
            logger.error(f"❌ Context gathering failed: {e}")
            return {
                "query": query,
                "filters": filters,
                "similar_solutions": [],
                "decomposition_patterns": [],
                "critique_patterns": [],
                "verification_benchmarks": [],
                "errors": [str(e)]
            }


class RAGBitsArtifactIndexerTool(AgentTool):
    """
    Tool for indexing workflow artifacts into RAGBits.

    Automatically ingests agent outputs into the knowledge base for future retrieval.

    Usage:
        tool = RAGBitsArtifactIndexerTool()
        artifact_id = await tool.execute(
            content="Solution for authentication...",
            metadata={"stage": "stage_3", "team": "blue"},
            artifact_type="solution"
        )
    """

    def __init__(self, retriever: Optional[RAGBitsEnhancedRetriever] = None):
        """Initialize the artifact indexer tool"""
        super().__init__(
            name="ragbits_artifact_indexer",
            description="Index workflow artifacts into knowledge base"
        )

        if retriever:
            self.retriever = retriever
        elif get_ragbits_retriever:
            self.retriever = get_ragbits_retriever()
        else:
            self.retriever = None

    async def execute(
        self,
        content: str,
        metadata: Dict[str, Any],
        artifact_type: str = "solution",
        **kwargs
    ) -> str:
        """
        Index an artifact into the knowledge base.

        Args:
            content: Artifact content
            metadata: Artifact metadata (stage, team, sub_problem_id, etc.)
            artifact_type: Type of artifact
                - "solution": Generated solution
                - "critique": Red team critique
                - "verification": Gold team verification
                - "decomposition": Problem decomposition
                - "general": General artifact
            **kwargs: Additional parameters

        Returns:
            Artifact ID

        Example:
            >>> artifact_id = await tool.execute(
            ...     content="Implement JWT authentication...",
            ...     metadata={
            ...         "stage": "stage_3",
            ...         "team": "blue",
            ...         "sub_problem_id": "sub_1"
            ...     },
            ...     artifact_type="solution"
            ... )
        """
        logger.info(f"📥 Indexing {artifact_type} artifact...")

        if not self.retriever:
            logger.warning("⚠️ RAGBits retriever not available")
            return ""

        try:
            # Add timestamp and type to metadata
            enhanced_metadata = {
                **metadata,
                "artifact_type": artifact_type,
                "indexed_at": datetime.utcnow().isoformat()
            }

            artifact_id = await self.retriever.ingest_artifact(
                content=content,
                metadata=enhanced_metadata,
                artifact_type=artifact_type
            )

            if artifact_id:
                logger.info(f"✅ Indexed artifact: {artifact_id}")
            else:
                logger.warning("⚠️ Artifact indexing returned empty ID")

            return artifact_id

        except Exception as e:
            logger.error(f"❌ Artifact indexing failed: {e}")
            return ""


class RAGBitsPatternAnalyzerTool(AgentTool):
    """
    Tool for analyzing patterns in historical data using RAGBits.

    Identifies:
    - Common solution patterns
    - Recurring issues in critiques
    - Successful decomposition strategies
    - Effective verification approaches

    Usage:
        tool = RAGBitsPatternAnalyzerTool()
        patterns = await tool.execute(
            analysis_type="solutions",
            query="authentication patterns",
            filters={"stage": "stage_3"}
        )
    """

    def __init__(self, retriever: Optional[RAGBitsEnhancedRetriever] = None):
        """Initialize the pattern analyzer tool"""
        super().__init__(
            name="ragbits_pattern_analyzer",
            description="Analyze patterns in historical data"
        )

        if retriever:
            self.retriever = retriever
        elif get_ragbits_retriever:
            self.retriever = get_ragbits_retriever()
        else:
            self.retriever = None

    async def execute(
        self,
        analysis_type: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze patterns in historical data.

        Args:
            analysis_type: Type of analysis
                - "solutions": Analyze solution patterns
                - "decompositions": Analyze decomposition strategies
                - "critiques": Analyze critique patterns
                - "verification": Analyze verification approaches
            query: Analysis query
            filters: Metadata filters
            **kwargs: Additional parameters

        Returns:
            Analysis results with patterns and insights

        Example:
            >>> patterns = await tool.execute(
            ...     analysis_type="solutions",
            ...     query="successful authentication patterns",
            ...     filters={"stage": "stage_3"}
            ... )
        """
        logger.info(f"📊 Analyzing {analysis_type} patterns: {query[:100]}...")

        if not self.retriever:
            return {
                "analysis_type": analysis_type,
                "query": query,
                "patterns": [],
                "insights": [],
                "warnings": ["RAGBits retriever not available"]
            }

        try:
            # Search for relevant historical data
            top_k = kwargs.get("top_k", 10)

            if analysis_type == "solutions":
                results = await self.retriever.search_similar_solutions(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            elif analysis_type == "decompositions":
                results = await self.retriever.search_decomposition_patterns(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            elif analysis_type == "critiques":
                results = await self.retriever.search_critique_patterns(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            elif analysis_type == "verification":
                results = await self.retriever.search_verification_benchmarks(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            else:
                results = []

            # Extract patterns from results
            patterns = self._extract_patterns(results, analysis_type)

            # Generate insights
            insights = self._generate_insights(patterns, results)

            return {
                "analysis_type": analysis_type,
                "query": query,
                "filters": filters,
                "total_results": len(results),
                "patterns": patterns,
                "insights": insights,
                "analyzed_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
            return {
                "analysis_type": analysis_type,
                "query": query,
                "patterns": [],
                "insights": [],
                "errors": [str(e)]
            }

    def _extract_patterns(
        self,
        results: List[Dict[str, Any]],
        analysis_type: str
    ) -> List[Dict[str, Any]]:
        """Extract common patterns from results"""
        # This would use more sophisticated pattern extraction
        # For now, return top results as patterns
        return [
            {
                "pattern": r.get("content", "")[:200],
                "frequency": r.get("score", 0.0),
                "source": r.get("metadata", {}).get("source", "unknown")
            }
            for r in results[:5]
        ]

    def _generate_insights(
        self,
        patterns: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from patterns and results"""
        insights = []

        if len(results) > 0:
            insights.append(f"Found {len(results)} relevant historical items")

        if len(patterns) > 0:
            insights.append(f"Identified {len(patterns)} common patterns")

        avg_score = sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0
        insights.append(f"Average relevance score: {avg_score:.2f}")

        return insights
