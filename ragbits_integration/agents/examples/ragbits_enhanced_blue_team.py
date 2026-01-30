"""
ragbits_enhanced_blue_team.py - CrewAI Integration

This file has been migrated from crewai (AGPL) to CrewAI (MIT).

Migration Date: 2026-01-21
Migration Status: Complete

All crewai references have been replaced with CrewAI equivalents.
The functionality remains the same, but now uses local CrewAI execution
instead of remote crewai API calls.

For questions, see: CREWAI_MIGRATION_MASTER_TASKLIST.md
"""

"""
RAGBits-Enhanced Blue Team Agent

Blue team agent with advanced knowledge retrieval and context gathering capabilities.
"""

import logging
from typing import Dict, Any, List, Optional

from ragbits_integration.agents.base_agent import BaseWorkflowAgent
from ragbits_integration.agents.tools.ragbits_enhanced_tools import (
    RAGBitsKnowledgeSearchTool,
    RAGBitsContextGathererTool,
    RAGBitsArtifactIndexerTool,
    RAGBitsPatternAnalyzerTool
)

logger = logging.getLogger(__name__)


class RAGBitsEnhancedBlueTeamAgent(BaseWorkflowAgent):
    """
    Blue team agent enhanced with RAGBits knowledge capabilities.

    Features:
    - Semantic search for similar solutions
    - Context gathering from multiple knowledge sources
    - Automatic artifact indexing
    - Pattern analysis for informed decision-making

    Usage:
        agent = RAGBitsEnhancedBlueTeamAgent(
            crewai_client=crewai,
            storage_manager=storage
        )

        result = await agent.execute(
            task="generate_solution",
            context={
                "sub_problem": sub_problem,
                "stage": "stage_3"
            }
        )
    """

    def __init__(
        self,
        crewai_client=None,
        model_config: Optional[Dict[str, Any]] = None,
        storage_manager=None,
        knowledge_retriever=None,
        enable_ragbits: bool = True
    ):
        """
        Initialize the RAGBits-enhanced blue team agent.

        Args:
            crewai_client: crewai client for LLM access
            model_config: Model configuration
            storage_manager: Storage manager for artifacts
            knowledge_retriever: Knowledge retriever (creates default if not provided)
            enable_ragbits: Enable RAGBits tools
        """
        # Initialize base agent
        super().__init__(
            role=self.ROLE_BLUE_TEAM,
            crewai_client=crewai_client,
            model_config=model_config,
            storage_manager=storage_manager,
            knowledge_retriever=knowledge_retriever
        )

        self.enable_ragbits = enable_ragbits

        # Initialize RAGBits tools if enabled
        if enable_ragbits:
            self.ragbits_search = RAGBitsKnowledgeSearchTool()
            self.ragbits_context = RAGBitsContextGathererTool()
            self.ragbits_indexer = RAGBitsArtifactIndexerTool()
            self.ragbits_analyzer = RAGBitsPatternAnalyzerTool()

            # Register tools
            self.tools["ragbits_search"] = self.ragbits_search
            self.tools["ragbits_context"] = self.ragbits_context
            self.tools["ragbits_indexer"] = self.ragbits_indexer
            self.tools["ragbits_analyzer"] = self.ragbits_analyzer

            logger.info("✅ RAGBits tools initialized for blue team agent")
        else:
            logger.info("ℹ️ RAGBits tools disabled")

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute blue team task with RAGBits enhancement.

        Args:
            task: Task to execute ("generate_solution", "analyze_patterns")
            context: Task context (sub_problem, stage, etc.)
            **kwargs: Additional parameters

        Returns:
            Result dict with solution and metadata
        """
        logger.info(f"🔵 Blue team executing: {task}")

        try:
            if task == "generate_solution":
                return await self._generate_solution(context, **kwargs)
            elif task == "analyze_patterns":
                return await self._analyze_patterns(context, **kwargs)
            else:
                return await self._default_task(task, context, **kwargs)

        except Exception as e:
            logger.error(f"❌ Blue team task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "role": self.ROLE_BLUE_TEAM
            }

    async def _generate_solution(
        self,
        context: Dict[str, Any],
        use_knowledge: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate solution with knowledge from RAGBits.

        Args:
            context: Task context
            use_knowledge: Whether to use RAGBits knowledge retrieval
            **kwargs: Additional parameters

        Returns:
            Generated solution with metadata
        """
        sub_problem = context.get("sub_problem", {})
        stage = context.get("stage", "unknown")
        sub_problem_id = context.get("sub_problem_id", "unknown")

        logger.info(f"🔵 Generating solution for: {sub_problem.get('description', '')[:100]}...")

        # Gather context if RAGBits is enabled
        knowledge_context = {}
        if use_knowledge and self.enable_ragbits:
            knowledge_context = await self.ragbits_context.execute(
                query=sub_problem.get("description", ""),
                sub_problem_id=sub_problem_id,
                stage=stage,
                team=self.ROLE_BLUE_TEAM,
                max_results_per_category=3
            )
            logger.info(f"📚 Gathered {len(knowledge_context.get('similar_solutions', []))} similar solutions")

        # Build prompt with context
        prompt = self._build_solution_prompt(sub_problem, knowledge_context)

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        solution = self._parse_solution(response)

        # Index the solution artifact
        if self.enable_ragbits and self.storage:
            await self.ragbits_indexer.execute(
                content=solution.get("content", response),
                metadata={
                    "stage": stage,
                    "team": self.ROLE_BLUE_TEAM,
                    "sub_problem_id": sub_problem_id,
                    "problem_description": sub_problem.get("description", "")
                },
                artifact_type="solution"
            )

        return {
            "success": True,
            "role": self.ROLE_BLUE_TEAM,
            "solution": solution,
            "knowledge_context": knowledge_context,
            "stage": stage,
            "sub_problem_id": sub_problem_id
        }

    async def _analyze_patterns(
        self,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze patterns in historical solutions.

        Args:
            context: Task context
            **kwargs: Additional parameters

        Returns:
            Pattern analysis results
        """
        query = context.get("query", "")
        filters = context.get("filters", {})

        logger.info(f"🔵 Analyzing patterns: {query[:100]}...")

        if not self.enable_ragbits:
            return {
                "success": False,
                "error": "RAGBits tools not enabled"
            }

        # Analyze solution patterns
        analysis = await self.ragbits_analyzer.execute(
            analysis_type="solutions",
            query=query,
            filters=filters
        )

        return {
            "success": True,
            "role": self.ROLE_BLUE_TEAM,
            "analysis": analysis
        }

    async def _default_task(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Default task handler"""
        logger.info(f"🔵 Executing default task: {task}")

        prompt = f"""
You are the blue team agent. Execute the following task:

Task: {task}

Context:
{context.get('description', '')}

Provide your response.
"""

        response = await self._call_llm(prompt)

        return {
            "success": True,
            "role": self.ROLE_BLUE_TEAM,
            "task": task,
            "response": response
        }

    def _build_solution_prompt(
        self,
        sub_problem: Dict[str, Any],
        knowledge_context: Dict[str, Any]
    ) -> str:
        """Build prompt for solution generation with knowledge context"""

        prompt = f"""
You are the blue team agent. Generate a solution for the following sub-problem:

**Sub-Problem:**
{sub_problem.get('description', '')}

**Constraints:**
{sub_problem.get('constraints', '')}

**Acceptance Criteria:**
{sub_problem.get('acceptance_criteria', '')}

"""

        # Add knowledge context if available
        if knowledge_context and knowledge_context.get("similar_solutions"):
            prompt += "\n**Similar Solutions from History:**\n"
            for i, sol in enumerate(knowledge_context["similar_solutions"][:3], 1):
                prompt += f"\n{i}. {sol.get('content', '')[:300]}...\n"

        if knowledge_context and knowledge_context.get("decomposition_patterns"):
            prompt += "\n**Relevant Decomposition Patterns:**\n"
            for i, pattern in enumerate(knowledge_context["decomposition_patterns"][:2], 1):
                prompt += f"\n{i}. {pattern.get('content', '')[:200]}...\n"

        prompt += """

**Instructions:**
1. Analyze the sub-problem carefully
2. Consider patterns from similar historical solutions
3. Generate a comprehensive solution
4. Ensure solution addresses all constraints and acceptance criteria
5. Provide clear implementation steps

**Solution:**
"""

        return prompt

    def _parse_solution(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into solution structure"""
        # For now, return as-is
        # Could add more sophisticated parsing
        return {
            "content": response,
            "approach": "LLM-generated",
            "timestamp": str(asyncio.get_event_loop().time())
        }


async def demo_ragbits_blue_team():
    """
    Demo function showing RAGBits-enhanced blue team agent usage.
    """
    import asyncio

    logger.info("🚀 Starting RAGBits-enhanced blue team agent demo")

    # Create agent (without crewai for demo)
    agent = RAGBitsEnhancedBlueTeamAgent(
        crewai_client=None,
        enable_ragbits=True
    )

    # Example 1: Generate solution with knowledge
    logger.info("\n" + "="*80)
    logger.info("Example 1: Generate solution with RAGBits knowledge")
    logger.info("="*80 + "\n")

    sub_problem = {
        "description": "Implement JWT-based authentication for REST API",
        "constraints": "Use industry-standard libraries, support token refresh",
        "acceptance_criteria": "Authentication works, tokens can be refreshed"
    }

    context = {
        "sub_problem": sub_problem,
        "stage": "stage_3",
        "sub_problem_id": "sub_1"
    }

    result = await agent.execute(
        task="generate_solution",
        context=context,
        use_knowledge=True
    )

    logger.info(f"\n✅ Solution generated: {result.get('success', False)}")
    logger.info(f"📊 Similar solutions found: {len(result.get('knowledge_context', {}).get('similar_solutions', []))}")

    # Example 2: Analyze patterns
    logger.info("\n" + "="*80)
    logger.info("Example 2: Analyze historical patterns")
    logger.info("="*80 + "\n")

    analysis_context = {
        "query": "authentication patterns in microservices",
        "filters": {"stage": "stage_3"}
    }

    analysis = await agent.execute(
        task="analyze_patterns",
        context=analysis_context
    )

    logger.info(f"\n✅ Patterns analyzed: {analysis.get('success', False)}")
    logger.info(f"📊 Patterns found: {len(analysis.get('analysis', {}).get('patterns', []))}")

    logger.info("\n🎉 Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_ragbits_blue_team())
