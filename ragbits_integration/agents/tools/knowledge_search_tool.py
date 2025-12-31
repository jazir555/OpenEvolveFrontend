"""
Knowledge Search Tool

Allows agents to search for similar solutions, patterns, and historical data.
"""

from typing import List, Dict, Any, Optional
import logging

from ragbits_integration.agents.base_agent import AgentTool

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(AgentTool):
    """
    Tool for searching knowledge base using semantic search.

    Provides agents with ability to:
    - Find similar solutions from history
    - Retrieve decomposition patterns
    - Get critique patterns
    - Access verification benchmarks

    Usage:
        tool = KnowledgeSearchTool(knowledge_retriever)
        results = await tool.execute(
            search_type="similar_solutions",
            query="user authentication system",
            top_k=5
        )
    """

    def __init__(self, knowledge_retriever):
        """
        Initialize the knowledge search tool.

        Args:
            knowledge_retriever: RagbitsKnowledgeRetriever instance
        """
        super().__init__(
            name="knowledge_search",
            description="Search for similar solutions, patterns, and historical knowledge"
        )
        self.retriever = knowledge_retriever

    async def execute(
        self,
        search_type: str,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute knowledge search.

        Args:
            search_type: Type of search ("similar_solutions", "decomposition_patterns", "critique_patterns", "verification_benchmarks")
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results with metadata

        Example:
            >>> results = await tool.execute(
            ...     search_type="similar_solutions",
            ...     query="REST API authentication",
            ...     top_k=3,
            ...     min_success_rate=0.8
            ... )
        """
        logger.info(f"Executing {search_type} search: {query[:100]}...")

        try:
            if search_type == "similar_solutions":
                return await self._search_similar_solutions(query, top_k, **kwargs)
            elif search_type == "decomposition_patterns":
                return await self._search_decomposition_patterns(query, top_k, **kwargs)
            elif search_type == "critique_patterns":
                return await self._search_critique_patterns(query, top_k, **kwargs)
            elif search_type == "verification_benchmarks":
                return await self._search_verification_benchmarks(query, top_k, **kwargs)
            else:
                # General semantic search
                return await self._semantic_search(query, top_k, **kwargs)

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []

    async def _search_similar_solutions(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar solutions"""
        min_success_rate = kwargs.get("min_success_rate", 0.0)
        team = kwargs.get("team")

        results = await self.retriever.retrieve_similar_solutions(
            problem_description=query,
            top_k=top_k,
            min_success_rate=min_success_rate,
            team=team
        )

        # Format results for agent consumption
        formatted = []
        for result in results:
            formatted.append({
                "content": result["content"],
                "success_rate": result["success_rate"],
                "team_used": result["team_used"],
                "similarity": result["similarity"],
                "domain": result.get("domain"),
                "summary": result["content"][:200] + "..."
            })

        return formatted

    async def _search_decomposition_patterns(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for decomposition patterns"""
        complexity = kwargs.get("complexity")
        domain = kwargs.get("domain")

        # Extract problem type from query
        problem_type = query.lower().split()[0] if query else "general"

        results = await self.retriever.retrieve_relevant_decompositions(
            problem_type=problem_type,
            complexity=complexity or 7.0,
            top_k=top_k,
            domain=domain
        )

        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "content": result["content"],
                "sub_problem_count": result["sub_problem_count"],
                "effectiveness": result["effectiveness"],
                "strategy": result.get("strategy"),
                "summary": result["content"][:200] + "..."
            })

        return formatted

    async def _search_critique_patterns(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for critique patterns"""
        severity = kwargs.get("severity")

        results = await self.retriever.retrieve_critique_patterns(
            solution_type=query,
            top_k=top_k,
            severity=severity
        )

        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "issue_type": result["issue_type"],
                "severity": result["severity"],
                "frequency": result["frequency"],
                "pattern": result["pattern"][:200] + "..."
            })

        return formatted

    async def _search_verification_benchmarks(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for verification benchmarks"""
        domain = kwargs.get("domain", query)

        results = await self.retriever.retrieve_verification_benchmarks(
            solution_domain=domain,
            top_k=top_k
        )

        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "criteria": result["criteria"],
                "threshold": result["threshold"],
                "description": result["content"][:200] + "..."
            })

        return formatted

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform general semantic search"""
        artifact_type = kwargs.get("artifact_type")
        stage = kwargs.get("stage")

        results = await self.retriever.semantic_search(
            query=query,
            artifact_type=artifact_type,
            stage=stage,
            top_k=top_k
        )

        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "content": result["content"][:500],
                "similarity": result["similarity"],
                "metadata": result["metadata"]
            })

        return formatted
