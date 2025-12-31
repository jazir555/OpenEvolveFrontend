"""
Context Gatherer

Specialized context retrieval for different workflow stages.
Provides high-level APIs for agents to gather relevant context.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ContextGatherer:
    """
    High-level context gathering API for workflow agents.

    Simplifies the process of gathering relevant context for specific
    workflow stages and agent roles.

    Usage:
        gatherer = ContextGatherer(storage_manager)

        # Blue Team gathering context for solution generation
        context = await gatherer.gather_for_blue_team(
            sub_problem_id="sub_1",
            problem_description="Implement user authentication"
        )

        # Red Team gathering context for critique
        context = await gatherer.gather_for_red_team(
            sub_problem_id="sub_1"
        )
    """

    def __init__(self, storage_manager):
        """
        Initialize the context gatherer.

        Args:
            storage_manager: IntermediaryStorageManager instance
        """
        self.storage = storage_manager
        logger.info("ContextGatherer initialized")

    async def gather_for_blue_team(
        self,
        sub_problem_id: str,
        problem_description: str,
        include_similar_solutions: bool = True
    ) -> Dict[str, Any]:
        """
        Gather context for Blue Team (solution generation).

        Retrieves:
        - Content analysis from Stage 0
        - Relevant decomposition plan
        - Similar solutions from history (optional)

        Args:
            sub_problem_id: Sub-problem to solve
            problem_description: Problem description for semantic search
            include_similar_solutions: Whether to retrieve similar solutions

        Returns:
            Context dict with all relevant information
        """
        logger.info(f"Gathering Blue Team context for sub_problem {sub_problem_id}")

        context = await self.storage.retrieve_context_for_stage(
            stage="stage_3_blue_team_solution",
            sub_problem_id=sub_problem_id,
            query=problem_description if include_similar_solutions else None
        )

        # Add specialized Blue Team metadata
        context["agent_role"] = "blue_team"
        context["task"] = "generate_solution"
        context["sub_problem_id"] = sub_problem_id

        # Extract similar solutions if available
        if include_similar_solutions and context.get("similar_historical"):
            context["similar_solutions"] = self._extract_solution_info(
                context["similar_historical"]
            )

        return context

    async def gather_for_red_team(
        self,
        sub_problem_id: str,
        solution_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gather context for Red Team (critique).

        Retrieves:
        - Blue Team's solution (mandatory)
        - Similar critique patterns from history
        - Historical issues with similar solutions

        Args:
            sub_problem_id: Sub-problem being critiqued
            solution_text: Optional solution text for semantic search

        Returns:
            Context dict with solution and critique patterns
        """
        logger.info(f"Gathering Red Team context for sub_problem {sub_problem_id}")

        context = await self.storage.retrieve_context_for_stage(
            stage="stage_3_red_team_critique",
            sub_problem_id=sub_problem_id,
            query=solution_text
        )

        # Add specialized Red Team metadata
        context["agent_role"] = "red_team"
        context["task"] = "critique_solution"
        context["sub_problem_id"] = sub_problem_id

        # Extract Blue Team solution
        if context["artifacts"].get("blue_solution"):
            context["solution_to_critique"] = context["artifacts"]["blue_solution"][0]

        # Extract similar critiques
        if context.get("similar_historical"):
            context["critique_patterns"] = self._extract_critique_patterns(
                context["similar_historical"]
            )

        return context

    async def gather_for_gold_team(
        self,
        sub_problem_id: str
    ) -> Dict[str, Any]:
        """
        Gather context for Gold Team (verification).

        Retrieves:
        - Blue Team's solution
        - Red Team's critique
        - Historical verification benchmarks

        Args:
            sub_problem_id: Sub-problem being verified

        Returns:
            Context dict with solution, critique, and benchmarks
        """
        logger.info(f"Gathering Gold Team context for sub_problem {sub_problem_id}")

        context = await self.storage.retrieve_context_for_stage(
            stage="stage_3_gold_team_verification",
            sub_problem_id=sub_problem_id
        )

        # Add specialized Gold Team metadata
        context["agent_role"] = "gold_team"
        context["task"] = "verify_solution"
        context["sub_problem_id"] = sub_problem_id

        # Extract solution and critique
        if context["artifacts"].get("solution"):
            context["solution"] = context["artifacts"]["solution"][0]

        if context["artifacts"].get("critique"):
            context["critique"] = context["artifacts"]["critique"][0]

        return context

    async def gather_for_decomposition(
        self,
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Gather context for problem decomposition (Stage 1).

        Retrieves:
        - Content analysis from Stage 0
        - Similar decomposition patterns from history

        Args:
            problem_description: Problem being decomposed

        Returns:
            Context dict with analysis and patterns
        """
        logger.info("Gathering context for decomposition")

        context = await self.storage.retrieve_context_for_stage(
            stage="stage_1_decomposition"
        )

        context["agent_role"] = "decomposer"
        context["task"] = "decompose_problem"
        context["problem_description"] = problem_description

        # Retrieve similar decomposition patterns
        similar_patterns = await self.storage._search_artifacts(
            query=problem_description,
            filters={"type": "decomposition_plan"},
            top_k=5
        )

        context["similar_patterns"] = similar_patterns

        return context

    async def gather_for_reassembly(
        self,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gather context for solution reassembly (Stage 4).

        Retrieves:
        - All verified sub-problem solutions
        - Successful assembly patterns from history

        Args:
            workflow_id: Optional workflow identifier

        Returns:
            Context dict with verified solutions and patterns
        """
        logger.info("Gathering context for reassembly")

        context = await self.storage.retrieve_context_for_stage(
            stage="stage_4_reassembly"
        )

        context["agent_role"] = "assembler"
        context["task"] = "assemble_solution"
        context["workflow_id"] = workflow_id

        # Get verified solutions
        verified_solutions = await self.storage.get_artifacts_by_stage(
            stage="stage_3",
            status="verified"
        )

        # Group by sub-problem
        context["verified_solutions_by_subproblem"] = self._group_by_subproblem(
            verified_solutions
        )

        return context

    async def gather_for_final_verification(
        self,
        assembled_solution: str
    ) -> Dict[str, Any]:
        """
        Gather context for final verification (Stage 5).

        Retrieves:
        - Assembled solution
        - Historical benchmarks
        - Similar final solutions

        Args:
            assembled_solution: The assembled solution to verify

        Returns:
            Context dict with solution and benchmarks
        """
        logger.info("Gathering context for final verification")

        context = await self.storage.retrieve_context_for_stage(
            stage="stage_5_final_verification"
        )

        context["agent_role"] = "final_verifier"
        context["task"] = "final_verification"

        # Get similar final solutions for comparison
        similar_final = await self.storage._search_artifacts(
            query=assembled_solution,
            filters={"type": "final_solution"},
            top_k=5
        )

        context["similar_solutions"] = similar_final

        return context

    def _extract_solution_info(
        self,
        artifacts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relevant information from solution artifacts"""
        solutions = []
        for artifact in artifacts:
            metadata = artifact.get("metadata", {})
            solutions.append({
                "content": artifact.get("content", ""),
                "success_rate": metadata.get("success_rate", 0),
                "team_used": metadata.get("team"),
                "artifact_id": metadata.get("artifact_id")
            })
        return solutions

    def _extract_critique_patterns(
        self,
        artifacts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract critique patterns from artifacts"""
        patterns = []
        for artifact in artifacts:
            metadata = artifact.get("metadata", {})
            patterns.append({
                "pattern": artifact.get("content", ""),
                "issue_type": metadata.get("issue_type"),
                "severity": metadata.get("severity"),
                "frequency": metadata.get("frequency", 1)
            })
        return patterns

    def _group_by_subproblem(
        self,
        artifacts: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group artifacts by sub_problem_id"""
        grouped = {}
        for artifact in artifacts:
            metadata = artifact.get("metadata", {})
            sub_problem_id = metadata.get("sub_problem_id", "unknown")
            if sub_problem_id not in grouped:
                grouped[sub_problem_id] = []
            grouped[sub_problem_id].append(artifact)
        return grouped

    async def get_full_artifact_chain(
        self,
        artifact_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get the complete chain of related artifacts.

        Useful for understanding the complete history of a solution
        through critique, verification, and refinement.

        Args:
            artifact_id: Starting artifact ID

        Returns:
            Ordered list of artifacts in the chain
        """
        return await self.storage.get_artifact_chain(artifact_id)

    async def get_subproblem_summary(
        self,
        sub_problem_id: str
    ) -> Dict[str, Any]:
        """
        Get a complete summary of all artifacts for a sub-problem.

        Useful for understanding the complete workflow history for
        a specific sub-problem.

        Args:
            sub_problem_id: Sub-problem identifier

        Returns:
            Summary dict with all artifacts grouped by type
        """
        logger.info(f"Getting summary for sub_problem {sub_problem_id}")

        artifacts = await self.storage.get_artifacts_by_sub_problem(sub_problem_id)

        summary = {
            "sub_problem_id": sub_problem_id,
            "total_artifacts": len(artifacts),
            "by_type": {},
            "by_status": {},
            "by_team": {},
            "timeline": []
        }

        # Group by different dimensions
        for artifact in artifacts:
            metadata = artifact.get("metadata", {})

            # By type
            artifact_type = metadata.get("type", "unknown")
            if artifact_type not in summary["by_type"]:
                summary["by_type"][artifact_type] = []
            summary["by_type"][artifact_type].append(artifact)

            # By status
            status = metadata.get("status", "unknown")
            if status not in summary["by_status"]:
                summary["by_status"][status] = []
            summary["by_status"][status].append(artifact)

            # By team
            team = metadata.get("team", "unknown")
            if team not in summary["by_team"]:
                summary["by_team"][team] = []
            summary["by_team"][team].append(artifact)

            # Timeline
            timestamp = metadata.get("timestamp", 0)
            summary["timeline"].append({
                "artifact_id": metadata.get("artifact_id"),
                "type": artifact_type,
                "team": team,
                "status": status,
                "timestamp": timestamp
            })

        # Sort timeline by timestamp
        summary["timeline"].sort(key=lambda x: x["timestamp"])

        return summary
