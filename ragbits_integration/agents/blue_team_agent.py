"""
Blue Team Agent

Specialized agent for generating high-quality solutions to sub-problems.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ragbits_integration.agents.base_agent import BaseWorkflowAgent, AgentTool
from ragbits_integration.agents.tools.knowledge_search_tool import KnowledgeSearchTool

# **ACTUAL INTEGRATION**: Alerting and knowledge for Blue Team Agent
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


class BlueTeamAgent(BaseWorkflowAgent):
    """
    Blue Team agent for solution generation.

    Responsibilities:
    - Analyze sub-problems thoroughly
    - Generate comprehensive, implementable solutions
    - Leverage similar solutions from history
    - Ensure solutions address all requirements
    - Collaborate with Red Team based on feedback

    Usage:
        agent = BlueTeamAgent(
            crewai_client=crewai,
            storage_manager=storage,
            knowledge_retriever=retriever
        )

        result = await agent.generate_solution(
            sub_problem={
                "title": "User Authentication",
                "description": "Implement secure user authentication",
                "requirements": ["JWT", "OAuth support", "bcrypt password hashing"]
            },
            context={...},
            use_rag=True
        )
    """

    def __init__(
        self,
        crewai_client=None,
        storage_manager=None,
        knowledge_retriever=None,
        tools: Optional[List[AgentTool]] = None
    ):
        # Initialize with Blue Team role
        model_config = {
            "model_id": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2500
        }

        super().__init__(
            role=self.ROLE_BLUE_TEAM,
            crewai_client=crewai_client,
            model_config=model_config,
            storage_manager=storage_manager,
            knowledge_retriever=knowledge_retriever,
            tools=tools or []
        )

        # Add default tools
        if self.knowledge_retriever:
            self.add_tool(KnowledgeSearchTool(self.knowledge_retriever))

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Blue Team task.

        Args:
            task: Task description
            context: Task context
            **kwargs: Additional arguments

        Returns:
            Result dict with solution and metadata
        """
        logger.info(f"Blue Team executing task: {task[:100]}...")

        # Route to specific method based on task
        if "generate_solution" in task.lower() or "solution" in task.lower():
            return await self.generate_solution(
                sub_problem=context.get("sub_problem", {}),
                context=context,
                **kwargs
            )
        elif "refine" in task.lower():
            return await self.refine_solution(
                current_solution=context.get("current_solution"),
                critique=context.get("critique"),
                **kwargs
            )
        else:
            # General task execution
            return await self._execute_general_task(task, context, **kwargs)

    async def generate_solution(
        self,
        sub_problem: Dict[str, Any],
        context: Dict[str, Any],
        use_rag: bool = True,
        use_knowledge_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a solution for a sub-problem.

        Args:
            sub_problem: Sub-problem definition with title, description, requirements
            context: Additional context (parent problem, constraints, etc.)
            use_rag: Whether to use RAG for retrieving similar solutions
            use_knowledge_tools: Whether to use knowledge search tools

        Returns:
            Generated solution with metadata

        Example:
            >>> result = await agent.generate_solution(
            ...     sub_problem={
            ...         "title": "User Authentication",
            ...         "description": "Implement secure user authentication",
            ...         "requirements": ["JWT", "OAuth", "bcrypt"]
            ...     },
            ...     context={},
            ...     use_rag=True
            ... )
            >>> solution = result["solution"]
        """
        logger.info(f"Generating solution for: {sub_problem.get('title', 'Unknown')}")

        # Prepare context
        enriched_context = await self._prepare_solution_context(
            sub_problem, context, use_rag, use_knowledge_tools
        )

        # Build prompt
        prompt = self._build_solution_prompt(sub_problem, enriched_context)

        # Call LLM
        response = await self._call_llm(prompt)

        # Store the solution
        artifact_id = None
        if self.storage:
            artifact_id = await self.storage.store_artifact(
                artifact_type="solution_draft",
                content=response,
                metadata={
                    "stage": "stage_3",
                    "team": "blue",
                    "sub_problem_id": sub_problem.get("id", "unknown"),
                    "title": sub_problem.get("title"),
                    "status": "draft"
                }
            )

        # Parse and return result
        result = {
            "solution": response,
            "sub_problem_id": sub_problem.get("id"),
            "sub_problem_title": sub_problem.get("title"),
            "artifact_id": artifact_id,
            "rag_enabled": use_rag,
            "similar_solutions_used": len(enriched_context.get("similar_solutions", [])),
            "agent_metadata": self.get_metadata()
        }

        logger.info(f"Solution generated: {len(response)} chars, artifact_id: {artifact_id}")

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful solution generation
        self._extract_blue_team_knowledge("generate_solution", sub_problem.get("id"), sub_problem.get("title"), result)
        self._track_blue_team_performance("generate_solution", True, len(response))

        return result

    async def refine_solution(
        self,
        current_solution: str,
        critique: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Refine a solution based on critique feedback.

        Args:
            current_solution: Current solution to refine
            critique: Critique feedback from Red Team
            **kwargs: Additional arguments

        Returns:
            Refined solution
        """
        logger.info("Refining solution based on critique")

        # Build refinement prompt
        prompt = self._build_refinement_prompt(current_solution, critique)

        # Call LLM
        refined_solution = await self._call_llm(prompt)

        # Store refined solution
        artifact_id = None
        if self.storage:
            artifact_id = await self.storage.store_artifact(
                artifact_type="solution_draft",
                content=refined_solution,
                metadata={
                    "stage": "stage_3",
                    "team": "blue",
                    "refinement": True,
                    "iteration": kwargs.get("iteration", 2)
                },
                links_to=critique.get("artifact_id") if critique.get("artifact_id") else None
            )

        result = {
            "solution": refined_solution,
            "original_solution": current_solution,
            "critique_addressed": critique.get("issues", []),
            "artifact_id": artifact_id,
            "agent_metadata": self.get_metadata()
        }

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful refinement
        self._extract_blue_team_knowledge("refine_solution", None, "solution_refinement", {
            "solution": refined_solution,
            "original_solution": current_solution,
            "critique_addressed": critique.get("issues", []),
            "artifact_id": artifact_id,
            "agent_metadata": self.get_metadata()
        })
        self._track_blue_team_performance("refine_solution", True, len(refined_solution))

        return result

    async def _prepare_solution_context(
        self,
        sub_problem: Dict[str, Any],
        context: Dict[str, Any],
        use_rag: bool,
        use_knowledge_tools: bool
    ) -> Dict[str, Any]:
        """Prepare enriched context for solution generation"""
        enriched_context = context.copy()

        # Use knowledge search tools to find similar solutions
        if use_rag and use_knowledge_tools and "knowledge_search" in self.tools:
            try:
                similar_solutions = await self.use_tool(
                    "knowledge_search",
                    search_type="similar_solutions",
                    query=sub_problem.get("description", ""),
                    top_k=3,
                    min_success_rate=0.7
                )

                enriched_context["similar_solutions"] = similar_solutions
                logger.info(f"Found {len(similar_solutions)} similar solutions")

            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")
                enriched_context["similar_solutions"] = []

        # Add decomposition info if available
        if "decomposition_plan" in context:
            enriched_context["decomposition_info"] = context["decomposition_plan"]

        return enriched_context

    def _build_solution_prompt(
        self,
        sub_problem: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for solution generation"""
        parts = []

        # Add sub-problem details
        parts.append("# Sub-Problem to Solve")
        parts.append(f"Title: {sub_problem.get('title', 'Unknown')}")
        parts.append(f"Description: {sub_problem.get('description', 'No description')}")
        parts.append(f"Requirements:")
        for req in sub_problem.get("requirements", []):
            parts.append(f"  - {req}")

        # Add similar solutions if available
        if context.get("similar_solutions"):
            parts.append("\n# Similar Solutions from History")
            for i, sol in enumerate(context["similar_solutions"][:3], 1):
                parts.append(f"\n## Similar Solution {i}")
                parts.append(f"Success Rate: {sol.get('success_rate', 'N/A')}")
                parts.append(f"Summary: {sol.get('summary', sol.get('content', ''))[:300]}...")

        # Add constraints
        if context.get("constraints"):
            parts.append("\n# Constraints")
            for constraint in context["constraints"]:
                parts.append(f"  - {constraint}")

        # Add instructions
        parts.append("""
# Solution Requirements

Provide a comprehensive solution that:
1. Addresses all listed requirements
2. Is technically sound and implementable
3. Follows best practices for the domain
4. Considers edge cases
5. Is clear and well-structured

# Output Format

Provide your solution in the following structure:

## Overview
[Brief overview of the solution approach]

## Implementation Details
[Detailed implementation steps]

## Key Components
[Main components and their responsibilities]

## Considerations
[Edge cases, security, performance, etc.]

## Testing Recommendations
[How to test this solution]
""")

        return "\n".join(parts)

    def _build_refinement_prompt(
        self,
        current_solution: str,
        critique: Dict[str, Any]
    ) -> str:
        """Build prompt for solution refinement"""
        parts = [
            "# Solution Refinement",
            "\n## Original Solution:",
            current_solution,
            "\n## Critique Feedback:",
            f"Issues Identified: {critique.get('issues', [])}",
            f"\nKey Concerns: {critique.get('concerns', [])}",
            f"\nRecommendations: {critique.get('recommendations', [])}",
            "\n# Instructions",
            "Please refine the solution to address the critique feedback.",
            "Maintain what works well and improve what doesn't.",
            "\nProvide the refined solution in the same structure as the original."
        ]

        return "\n".join(parts)

    async def _execute_general_task(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a general Blue Team task"""
        prompt = self._build_prompt(task, context, include_tools=True)

        response = await self._call_llm(prompt)

        return {
            "response": response,
            "task": task,
            "agent_metadata": self.get_metadata()
        }

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Blue Team Agent
    # =========================================================================

    def _trigger_blue_team_alerts(
        self,
        operation: str,
        success: bool,
        sub_problem_title: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for blue team solution generation failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures
            if not success:
                alert_manager.create_alert(
                    title=f"Blue Team Agent Alert: {operation}",
                    description=f"Blue team operation '{operation}' failed" +
                                 (f" for '{sub_problem_title}'" if sub_problem_title else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="blue_team_agent",
                    component="solution_generation",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Blue Team alert: {e}")

    def _extract_blue_team_knowledge(
        self,
        operation: str,
        sub_problem_id: Optional[str],
        sub_problem_title: Optional[str],
        solution_data: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract blue team knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"blue_team_{operation}_{sub_problem_id or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="blue_team_solution",
                source_component="blue_team_agent",
                title=f"Blue Team Solution: {sub_problem_title or 'Unknown'} ({operation})",
                content={
                    "operation": operation,
                    "sub_problem_id": sub_problem_id,
                    "sub_problem_title": sub_problem_title,
                    "solution_length": len(solution_data.get("solution", "")),
                    "rag_enabled": solution_data.get("rag_enabled", False),
                    "similar_solutions_used": solution_data.get("similar_solutions_used", 0),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "artifact_id": solution_data.get("artifact_id"),
                    "agent_metadata": solution_data.get("agent_metadata", {})
                },
                tags=["blue_team", "solution_generation", operation]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Blue Team knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Blue Team knowledge: {e}")
            return False

    def _track_blue_team_performance(
        self,
        operation: str,
        success: bool,
        solution_length: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track blue team performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            # Quality based on success and solution length (longer may be better)
            quality = 1.0 if success else 0.0
            if success and solution_length > 0:
                # Normalize length factor (assume ~1000 chars is good)
                length_factor = min(solution_length / 1000.0, 1.5)
                quality = min(quality * length_factor, 1.0)

            performance_data = StrategyPerformanceData(
                strategy_name=f"blue_team_agent_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation, "solution_length": solution_length}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Blue Team performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Blue Team performance: {e}")
