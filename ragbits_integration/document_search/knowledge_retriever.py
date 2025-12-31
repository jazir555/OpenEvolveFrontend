"""
Knowledge Retriever

Semantic search and retrieval over historical solutions, patterns,
and workflow knowledge using RAGBits DocumentSearch.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RagbitsKnowledgeRetriever:
    """
    Retrieve relevant knowledge using RAGBits Document Search.

    Provides semantic search capabilities over:
    - Historical solutions and their success rates
    - Decomposition patterns
    - Critique patterns and common issues
    - Team performance metrics
    - Verification benchmarks

    Usage:
        retriever = RagbitsKnowledgeRetriever(document_search)

        # Find similar solutions
        similar = await retriever.retrieve_similar_solutions(
            problem_description="Implement user authentication system",
            top_k=5
        )

        # Find decomposition patterns
        patterns = await retriever.retrieve_relevant_decompositions(
            problem_type="software_architecture",
            complexity=8.5
        )
    """

    def __init__(self, document_search):
        """
        Initialize the knowledge retriever.

        Args:
            document_search: RAGBits DocumentSearch instance
        """
        self.document_search = document_search
        logger.info("RagbitsKnowledgeRetriever initialized")

    async def retrieve_similar_solutions(
        self,
        problem_description: str,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
        min_success_rate: float = 0.0,
        team: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar solutions for a given problem.

        Args:
            problem_description: The problem to find similar solutions for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            min_success_rate: Minimum success rate for solutions
            team: Optional team filter

        Returns:
            List of similar solutions with metadata

        Example:
            >>> solutions = await retriever.retrieve_similar_solutions(
            ...     problem_description="Create RESTful API for user management",
            ...     top_k=3,
            ...     min_success_rate=0.8
            ... )
            >>> for sol in solutions:
            ...     print(f"Solution: {sol['content'][:100]}...")
            ...     print(f"Success Rate: {sol['success_rate']}")
        """
        logger.info(f"Searching for similar solutions to: {problem_description[:100]}...")

        # Build filters
        filters = {
            "type": "solution_draft",
            "status": "final"  # Only get completed solutions
        }

        if min_success_rate > 0:
            filters["success_rate"] = {"$gte": min_success_rate}

        if team:
            filters["team"] = team

        try:
            # Search for similar solutions
            chunks = await self.document_search.search(
                query=problem_description,
                filters=filters,
                top_k=top_k * 2  # Get more to filter by threshold
            )

            # Filter by similarity threshold and extract metadata
            similar_solutions = []
            for chunk in chunks:
                similarity = chunk.metadata.get("similarity", 0)

                if similarity >= similarity_threshold:
                    similar_solutions.append({
                        "content": chunk.text_representation,
                        "metadata": chunk.metadata,
                        "similarity": similarity,
                        "success_rate": chunk.metadata.get("success_rate", 0),
                        "team_used": chunk.metadata.get("team"),
                        "solution_id": chunk.metadata.get("artifact_id"),
                        "domain": chunk.metadata.get("domain"),
                        "complexity": chunk.metadata.get("complexity")
                    })

                if len(similar_solutions) >= top_k:
                    break

            logger.info(f"Found {len(similar_solutions)} similar solutions (threshold: {similarity_threshold})")
            return similar_solutions

        except Exception as e:
            logger.error(f"Failed to retrieve similar solutions: {e}")
            return []

    async def retrieve_relevant_decompositions(
        self,
        problem_type: str,
        complexity: float,
        top_k: int = 3,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant decomposition plans for similar problems.

        Args:
            problem_type: Type of problem (e.g., "software_architecture")
            complexity: Problem complexity score (0-10)
            top_k: Number of results to return
            domain: Optional domain filter

        Returns:
            List of decomposition plans with metadata

        Example:
            >>> patterns = await retriever.retrieve_relevant_decompositions(
            ...     problem_type="distributed_systems",
            ...     complexity=8.0
            ... )
            >>> for pattern in patterns:
            ...     print(f"Sub-problems: {pattern['sub_problem_count']}")
            ...     print(f"Effectiveness: {pattern['effectiveness']}")
        """
        logger.info(f"Searching for decomposition patterns: {problem_type}, complexity: {complexity}")

        # Build search query
        query = f"problem type: {problem_type}, complexity: {complexity}"

        # Build filters
        filters = {
            "type": "decomposition_plan"
        }

        if domain:
            filters["domain"] = domain

        # Filter by complexity range (±2 points)
        complexity_range = {
            "$gte": max(0, complexity - 2),
            "$lte": min(10, complexity + 2)
        }

        try:
            chunks = await self.document_search.search(
                query=query,
                filters=filters,
                top_k=top_k
            )

            decompositions = []
            for chunk in chunks:
                chunk_complexity = chunk.metadata.get("complexity", 0)
                if complexity_range["$gte"] <= chunk_complexity <= complexity_range["$lte"]:
                    decompositions.append({
                        "content": chunk.text_representation,
                        "metadata": chunk.metadata,
                        "problem_type": chunk.metadata.get("problem_type"),
                        "sub_problem_count": chunk.metadata.get("sub_problem_count", 0),
                        "effectiveness": chunk.metadata.get("effectiveness_score", 0),
                        "strategy": chunk.metadata.get("strategy"),
                        "plan_id": chunk.metadata.get("artifact_id"),
                        "complexity": chunk_complexity
                    })

            logger.info(f"Found {len(decompositions)} relevant decomposition patterns")
            return decompositions

        except Exception as e:
            logger.error(f"Failed to retrieve decomposition patterns: {e}")
            return []

    async def retrieve_critique_patterns(
        self,
        solution_type: str,
        top_k: int = 5,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve common critique patterns for similar solutions.

        Useful for Red Team to identify common issues and patterns.

        Args:
            solution_type: Type of solution being critiqued
            top_k: Number of results to return
            severity: Optional severity filter (low, medium, high)

        Returns:
            List of critique patterns with metadata

        Example:
            >>> patterns = await retriever.retrieve_critique_patterns(
            ...     solution_type="authentication_system"
            ... )
            >>> for pattern in patterns:
            ...     print(f"Issue: {pattern['issue_type']}")
            ...     print(f"Severity: {pattern['severity']}")
            ...     print(f"Frequency: {pattern['frequency']}")
        """
        logger.info(f"Searching for critique patterns: {solution_type}")

        query = f"critique patterns for {solution_type}"

        filters = {"type": "critique"}
        if severity:
            filters["severity"] = severity

        try:
            chunks = await self.document_search.search(
                query=query,
                filters=filters,
                top_k=top_k
            )

            patterns = []
            for chunk in chunks:
                patterns.append({
                    "content": chunk.text_representation,
                    "metadata": chunk.metadata,
                    "issue_type": chunk.metadata.get("issue_type"),
                    "severity": chunk.metadata.get("severity", "medium"),
                    "frequency": chunk.metadata.get("frequency", 1),
                    "pattern_id": chunk.metadata.get("artifact_id")
                })

            logger.info(f"Found {len(patterns)} critique patterns")
            return patterns

        except Exception as e:
            logger.error(f"Failed to retrieve critique patterns: {e}")
            return []

    async def retrieve_verification_benchmarks(
        self,
        solution_domain: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve verification benchmarks for a domain.

        Useful for Gold Team to verify solutions against established standards.

        Args:
            solution_domain: Domain of the solution
            top_k: Number of results to return

        Returns:
            List of verification benchmarks

        Example:
            >>> benchmarks = await retriever.retrieve_verification_benchmarks(
            ...     solution_domain="security"
            ... )
            >>> for benchmark in benchmarks:
            ...     print(f"Criteria: {benchmark['criteria']}")
            ...     print(f"Threshold: {benchmark['threshold']}")
        """
        logger.info(f"Searching for verification benchmarks: {solution_domain}")

        query = f"verification benchmarks for {solution_domain}"

        filters = {
            "type": "verification",
            "domain": solution_domain
        }

        try:
            chunks = await self.document_search.search(
                query=query,
                filters=filters,
                top_k=top_k
            )

            benchmarks = []
            for chunk in chunks:
                benchmarks.append({
                    "content": chunk.text_representation,
                    "metadata": chunk.metadata,
                    "criteria": chunk.metadata.get("criteria", []),
                    "threshold": chunk.metadata.get("threshold", 0.8),
                    "benchmark_id": chunk.metadata.get("artifact_id")
                })

            logger.info(f"Found {len(benchmarks)} verification benchmarks")
            return benchmarks

        except Exception as e:
            logger.error(f"Failed to retrieve verification benchmarks: {e}")
            return []

    async def retrieve_team_performance(
        self,
        team: str,
        task_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve team performance metrics and patterns.

        Args:
            team: Team identifier (blue, red, gold)
            task_type: Optional task type filter
            top_k: Number of results to return

        Returns:
            List of performance metrics

        Example:
            >>> metrics = await retriever.retrieve_team_performance(
            ...     team="blue",
            ...     task_type="solution_generation"
            ... )
            >>> for metric in metrics:
            ...     print(f"Average Quality: {metric['avg_quality']}")
            ...     print(f"Success Rate: {metric['success_rate']}")
        """
        logger.info(f"Retrieving performance metrics for team: {team}")

        query = f"team {team} performance metrics"

        filters = {"team": team}
        if task_type:
            filters["task_type"] = task_type

        try:
            chunks = await self.document_search.search(
                query=query,
                filters=filters,
                top_k=top_k
            )

            metrics_list = []
            for chunk in chunks:
                metrics_list.append({
                    "content": chunk.text_representation,
                    "metadata": chunk.metadata,
                    "avg_quality": chunk.metadata.get("avg_quality", 0),
                    "success_rate": chunk.metadata.get("success_rate", 0),
                    "total_tasks": chunk.metadata.get("total_tasks", 0),
                    "timestamp": chunk.metadata.get("timestamp")
                })

            logger.info(f"Found {len(metrics_list)} performance records for team {team}")
            return metrics_list

        except Exception as e:
            logger.error(f"Failed to retrieve team performance: {e}")
            return []

    async def semantic_search(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        stage: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a flexible semantic search over all artifacts.

        Args:
            query: Search query
            artifact_type: Optional artifact type filter
            stage: Optional stage filter
            top_k: Number of results to return
            filters: Additional custom filters

        Returns:
            List of matching artifacts

        Example:
            >>> results = await retriever.semantic_search(
            ...     query="scalable architecture patterns",
            ...     artifact_type="solution_draft",
            ...     top_k=10
            ... )
        """
        logger.info(f"Semantic search: {query[:100]}...")

        # Build base filters
        search_filters = {}
        if artifact_type:
            search_filters["type"] = artifact_type
        if stage:
            search_filters["stage"] = stage
        if filters:
            search_filters.update(filters)

        try:
            chunks = await self.document_search.search(
                query=query,
                filters=search_filters if search_filters else None,
                top_k=top_k
            )

            results = []
            for chunk in chunks:
                results.append({
                    "content": chunk.text_representation,
                    "metadata": chunk.metadata,
                    "similarity": chunk.metadata.get("similarity", 0)
                })

            logger.info(f"Semantic search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []

    async def get_knowledge_summary(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of available knowledge in the system.

        Args:
            domain: Optional domain filter

        Returns:
            Summary statistics and overview
        """
        logger.info("Generating knowledge summary")

        summary = {
            "total_artifacts": 0,
            "by_type": {},
            "by_status": {},
            "by_stage": {},
            "by_domain": {}
        }

        # This would typically involve aggregation queries
        # For now, return a basic structure
        try:
            # Get counts by type
            for artifact_type in ["solution_draft", "critique", "verification", "decomposition_plan"]:
                filters = {"type": artifact_type}
                if domain:
                    filters["domain"] = domain

                results = await self.document_search.search(
                    query=f"count of {artifact_type}",
                    filters=filters,
                    top_k=1
                )

                summary["by_type"][artifact_type] = len(results)
                summary["total_artifacts"] += len(results)

            logger.info(f"Knowledge summary: {summary['total_artifacts']} total artifacts")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate knowledge summary: {e}")
            return summary
