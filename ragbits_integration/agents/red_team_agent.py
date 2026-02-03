"""
Red Team Agent

Specialized agent for critiquing solutions and identifying issues.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ragbits_integration.agents.base_agent import BaseWorkflowAgent, AgentTool
from ragbits_integration.agents.tools.knowledge_search_tool import KnowledgeSearchTool
from ragbits_integration.agents.tools.pattern_analysis_tool import PatternAnalysisTool

# **ACTUAL INTEGRATION**: Alerting and knowledge for Red Team Agent
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


class RedTeamAgent(BaseWorkflowAgent):
    """
    Red Team agent for solution critique.

    Responsibilities:
    - Thoroughly critique solutions
    - Identify potential issues and edge cases
    - Suggest improvements
    - Check against common failure patterns
    - Provide constructive feedback

    Usage:
        agent = RedTeamAgent(
            crewai_client=crewai,
            storage_manager=storage,
            knowledge_retriever=retriever
        )

        result = await agent.critique_solution(
            solution="Solution content here...",
            sub_problem={...},
            context={...}
        )
    """

    def __init__(
        self,
        crewai_client=None,
        storage_manager=None,
        knowledge_retriever=None,
        tools: Optional[List[AgentTool]] = None
    ):
        # Initialize with Red Team role
        model_config = {
            "model_id": "claude-sonnet",
            "temperature": 0.5,
            "max_tokens": 2000
        }

        super().__init__(
            role=self.ROLE_RED_TEAM,
            crewai_client=crewai_client,
            model_config=model_config,
            storage_manager=storage_manager,
            knowledge_retriever=knowledge_retriever,
            tools=tools or []
        )

        # Add default tools
        if self.knowledge_retriever:
            self.add_tool(KnowledgeSearchTool(self.knowledge_retriever))
            self.add_tool(PatternAnalysisTool(self.knowledge_retriever))

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Red Team task.

        Args:
            task: Task description
            context: Task context
            **kwargs: Additional arguments

        Returns:
            Result dict with critique and metadata
        """
        logger.info(f"Red Team executing task: {task[:100]}...")

        # Route to specific method
        if "critique" in task.lower():
            return await self.critique_solution(
                solution=context.get("solution"),
                sub_problem=context.get("sub_problem", {}),
                context=context,
                **kwargs
            )
        else:
            return await self._execute_general_task(task, context, **kwargs)

    async def critique_solution(
        self,
        solution: str,
        sub_problem: Dict[str, Any],
        context: Dict[str, Any],
        use_patterns: bool = True,
        severity_threshold: str = "medium"
    ) -> Dict[str, Any]:
        """
        Critique a solution thoroughly.

        Args:
            solution: Solution to critique
            sub_problem: Original sub-problem definition
            context: Additional context
            use_patterns: Whether to use historical critique patterns
            severity_threshold: Minimum severity level to report

        Returns:
            Critique with identified issues and recommendations

        Example:
            >>> result = await agent.critique_solution(
            ...     solution="Implement JWT auth...",
            ...     sub_problem={"title": "Authentication", "requirements": [...]},
            ...     context={}
            ... )
            >>> issues = result["issues"]
        """
        logger.info(f"Critiquing solution for: {sub_problem.get('title', 'Unknown')}")

        # Prepare context with historical patterns
        enriched_context = await self._prepare_critique_context(
            solution, sub_problem, context, use_patterns
        )

        # Build critique prompt
        prompt = self._build_critique_prompt(
            solution, sub_problem, enriched_context
        )

        # Call LLM
        critique = await self._call_llm(prompt)

        # Store the critique
        artifact_id = None
        if self.storage:
            artifact_id = await self.storage.store_artifact(
                artifact_type="critique",
                content=critique,
                metadata={
                    "stage": "stage_3",
                    "team": "red",
                    "sub_problem_id": sub_problem.get("id", "unknown"),
                    "solution_title": sub_problem.get("title"),
                    "severity_threshold": severity_threshold
                },
                links_to=context.get("solution_artifact_id")
            )

        # Parse critique for structured output
        parsed_critique = self._parse_critique(critique)

        result = {
            "critique": critique,
            "parsed": parsed_critique,
            "sub_problem_id": sub_problem.get("id"),
            "sub_problem_title": sub_problem.get("title"),
            "artifact_id": artifact_id,
            "total_issues": len(parsed_critique.get("issues", [])),
            "patterns_used": len(enriched_context.get("critique_patterns", [])),
            "agent_metadata": self.get_metadata()
        }

        logger.info(f"Critique complete: {result['total_issues']} issues identified")

        # **ACTUAL INTEGRATION**: Extract knowledge, track performance, and trigger alerts for high issue counts
        self._extract_red_team_knowledge("critique_solution", sub_problem.get("id"), sub_problem.get("title"), result)
        self._track_red_team_performance("critique_solution", True, result['total_issues'], parsed_critique.get("overall_score", 0))

        # Trigger alert if too many issues found
        if result['total_issues'] > 10:
            self._trigger_red_team_alerts(
                "critique_solution",
                True,
                sub_problem.get("title"),
                result['total_issues'],
                None,
                {"artifact_id": artifact_id}
            )

        return result

    async def _prepare_critique_context(
        self,
        solution: str,
        sub_problem: Dict[str, Any],
        context: Dict[str, Any],
        use_patterns: bool
    ) -> Dict[str, Any]:
        """Prepare enriched context for critique"""
        enriched_context = context.copy()

        # Get historical critique patterns
        if use_patterns and "pattern_analysis" in self.tools:
            try:
                domain = sub_problem.get("domain") or sub_problem.get("title", "").split()[0]
                patterns = await self.use_tool(
                    "pattern_analysis",
                    analysis_type="common_issues",
                    domain=domain
                )

                enriched_context["critique_patterns"] = patterns.get("common_issues", [])
                logger.info(f"Found {len(enriched_context['critique_patterns'])} common issue patterns")

            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")
                enriched_context["critique_patterns"] = []

        # Get similar solutions for comparison
        if "knowledge_search" in self.tools:
            try:
                similar = await self.use_tool(
                    "knowledge_search",
                    search_type="similar_solutions",
                    query=sub_problem.get("description", ""),
                    top_k=3
                )

                enriched_context["similar_solutions"] = similar
            except Exception as e:
                logger.warning(f"Similar solutions search failed: {e}")

        return enriched_context

    def _build_critique_prompt(
        self,
        solution: str,
        sub_problem: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for solution critique"""
        parts = [
            "# Solution Critique",
            "\n## Sub-Problem:",
            f"Title: {sub_problem.get('title', 'Unknown')}",
            f"Description: {sub_problem.get('description', 'No description')}",
            f"\nRequirements:",
        ]

        for req in sub_problem.get("requirements", []):
            parts.append(f"  - {req}")

        parts.append([
            "\n## Solution to Critique:",
            solution,
            "\n## Critique Guidelines:",
            "1. Check if all requirements are addressed",
            "2. Identify potential issues and edge cases",
            "3. Assess technical correctness",
            "4. Evaluate completeness and clarity",
            "5. Consider security, performance, and scalability",
            "6. Provide specific, actionable feedback"
        ])

        # Add historical patterns if available
        if context.get("critique_patterns"):
            parts.append("\n## Common Issues to Check:")
            for i, pattern in enumerate(context["critique_patterns"][:5], 1):
                parts.append(f"{i}. {pattern.get('issue_type', 'General issue')}")

        parts.append("""
# Output Format

Provide your critique in the following structure:

## Overall Assessment
[Overall quality score 1-10 and brief summary]

## Requirements Coverage
[Which requirements are met and which are missing]

## Issues Identified
For each issue:
- Severity (critical/high/medium/low)
- Description
- Impact
- Recommendation

## Strengths
[What the solution does well]

## Recommendations
[Specific, actionable improvements]

## Verdict
PASS - Ready for verification
NEEDS_IMPROVEMENT - Requires refinement before verification
FAIL - Fundamentally flawed, needs complete rework
""")

        return "\n".join(str(p) for p in parts)

    def _parse_critique(self, critique: str) -> Dict[str, Any]:
        """Parse critique into structured format"""
        parsed = {
            "overall_score": 0,
            "verdict": "NEEDS_IMPROVEMENT",
            "requirements_coverage": {},
            "issues": [],
            "strengths": [],
            "recommendations": []
        }

        import re

        # Extract overall score
        score_match = re.search(r'overall.*?(\d+(?:\.\d+)?)', critique, re.IGNORECASE)
        if score_match:
            parsed["overall_score"] = float(score_match.group(1))

        # Extract verdict
        verdict_match = re.search(
            r'(PASS|NEEDS_IMPROVEMENT|FAIL)',
            critique,
            re.IGNORECASE
        )
        if verdict_match:
            parsed["verdict"] = verdict_match.group(1).upper()

        # Extract issues
        issues_section = re.search(
            r'Issues Identified.*?(?=Strengths|$)',
            critique,
            re.DOTALL | re.IGNORECASE
        )
        if issues_section:
            issues = re.findall(
                r'(?:critical|high|medium|low)[\s:]*[-.]?\s*(.+?)(?=(?:critical|high|medium|low)|Strengths|$)',
                issues_section.group(0),
                re.IGNORECASE
            )
            parsed["issues"] = [issue.strip() for issue in issues if issue.strip()]

        # Extract strengths
        strengths_section = re.search(
            r'Strengths.*?(?=Recommendations|$)',
            critique,
            re.DOTALL | re.IGNORECASE
        )
        if strengths_section:
            strengths = re.findall(r'^-\s*(.+)$', strengths_section.group(0), re.MULTILINE)
            parsed["strengths"] = strengths

        # Extract recommendations
        recommendations_section = re.search(
            r'Recommendations.*?(?=Verdict|$)',
            critique,
            re.DOTALL | re.IGNORECASE
        )
        if recommendations_section:
            recommendations = re.findall(r'^-\s*(.+)$', recommendations_section.group(0), re.MULTILINE)
            parsed["recommendations"] = recommendations

        return parsed

    async def _execute_general_task(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a general Red Team task"""
        prompt = self._build_prompt(task, context, include_tools=True)
        response = await self._call_llm(prompt)

        return {
            "response": response,
            "task": task,
            "agent_metadata": self.get_metadata()
        }

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Red Team Agent
    # =========================================================================

    def _trigger_red_team_alerts(
        self,
        operation: str,
        success: bool,
        sub_problem_title: Optional[str] = None,
        num_issues: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for red team critique failures or high issue counts."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures or high issue counts
            if not success or num_issues > 10:
                severity = AlertSeverity.HIGH if not success else AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Red Team Agent Alert: {operation}",
                    description=f"Red team operation '{operation}' " +
                                 ("failed" if not success else f"found {num_issues} issues") +
                                 (f" for '{sub_problem_title}'" if sub_problem_title else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="red_team_agent",
                    component="solution_critique",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Red Team alert: {e}")

    def _extract_red_team_knowledge(
        self,
        operation: str,
        sub_problem_id: Optional[str],
        sub_problem_title: Optional[str],
        critique_data: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract red team knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            parsed = critique_data.get("parsed", {})

            artifact = KnowledgeArtifact(
                artifact_id=f"red_team_{operation}_{sub_problem_id or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="red_team_critique",
                source_component="red_team_agent",
                title=f"Red Team Critique: {sub_problem_title or 'Unknown'} ({operation})",
                content={
                    "operation": operation,
                    "sub_problem_id": sub_problem_id,
                    "sub_problem_title": sub_problem_title,
                    "total_issues": critique_data.get("total_issues", 0),
                    "overall_score": parsed.get("overall_score", 0),
                    "verdict": parsed.get("verdict", "UNKNOWN"),
                    "patterns_used": critique_data.get("patterns_used", 0),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "artifact_id": critique_data.get("artifact_id"),
                    "issues": parsed.get("issues", [])[:10],  # Store top 10 issues
                    "agent_metadata": critique_data.get("agent_metadata", {})
                },
                tags=["red_team", "critique", operation]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Red Team knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Red Team knowledge: {e}")
            return False

    def _track_red_team_performance(
        self,
        operation: str,
        success: bool,
        num_issues: int = 0,
        overall_score: float = 0.0
    ):
        """**ACTUAL INTEGRATION**: Track red team performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            # Quality based on success and thoroughness (more issues found = more thorough)
            quality = 0.5 if success else 0.0
            if success:
                # Higher quality if many issues found (thorough critique)
                issue_factor = min(num_issues / 10.0, 1.5)
                score_factor = overall_score / 10.0
                quality = min((issue_factor + score_factor) / 2.0, 1.0)

            performance_data = StrategyPerformanceData(
                strategy_name=f"red_team_agent_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation, "num_issues": num_issues, "overall_score": overall_score}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Red Team performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Red Team performance: {e}")
