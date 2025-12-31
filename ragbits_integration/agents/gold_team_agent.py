"""
Gold Team Agent

Specialized agent for verifying solutions against requirements.
"""

from typing import List, Dict, Any, Optional
import logging

from ragbits_integration.agents.base_agent import BaseWorkflowAgent, AgentTool
from ragbits_integration.agents.tools.knowledge_search_tool import KnowledgeSearchTool
from ragbits_integration.agents.tools.solution_eval_tool import SolutionEvaluationTool

logger = logging.getLogger(__name__)


class GoldTeamAgent(BaseWorkflowAgent):
    """
    Gold Team agent for solution verification.

    Responsibilities:
    - Verify solutions meet requirements
    - Check for correctness and completeness
    - Validate against benchmarks
    - Assess overall quality
    - Provide verification results

    Usage:
        agent = GoldTeamAgent(
            hephaestus_client=hephaestus,
            storage_manager=storage,
            knowledge_retriever=retriever
        )

        result = await agent.verify_solution(
            solution="Solution content...",
            critique="Critique content...",
            sub_problem={...}
        )
    """

    def __init__(
        self,
        hephaestus_client=None,
        storage_manager=None,
        knowledge_retriever=None,
        tools: Optional[List[AgentTool]] = None
    ):
        # Initialize with Gold Team role
        model_config = {
            "model_id": "gpt-4-turbo",
            "temperature": 0.3,
            "max_tokens": 2000
        }

        super().__init__(
            role=self.ROLE_GOLD_TEAM,
            hephaestus_client=hephaestus_client,
            model_config=model_config,
            storage_manager=storage_manager,
            knowledge_retriever=knowledge_retriever,
            tools=tools or []
        )

        # Add default tools
        if self.knowledge_retriever:
            self.add_tool(KnowledgeSearchTool(self.knowledge_retriever))

        if hephaestus_client:
            self.add_tool(SolutionEvaluationTool(hephaestus_client, storage_manager))

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Gold Team task.

        Args:
            task: Task description
            context: Task context
            **kwargs: Additional arguments

        Returns:
            Result dict with verification and metadata
        """
        logger.info(f"Gold Team executing task: {task[:100]}...")

        # Route to specific method
        if "verify" in task.lower():
            return await self.verify_solution(
                solution=context.get("solution"),
                critique=context.get("critique"),
                sub_problem=context.get("sub_problem", {}),
                context=context,
                **kwargs
            )
        else:
            return await self._execute_general_task(task, context, **kwargs)

    async def verify_solution(
        self,
        solution: str,
        critique: Optional[Dict[str, Any]],
        sub_problem: Dict[str, Any],
        context: Dict[str, Any],
        use_benchmarks: bool = True,
        evaluation_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Verify a solution against requirements and benchmarks.

        Args:
            solution: Solution to verify
            critique: Optional Red Team critique
            sub_problem: Original sub-problem definition
            context: Additional context
            use_benchmarks: Whether to compare against benchmarks
            evaluation_criteria: Specific criteria to evaluate

        Returns:
            Verification result with pass/fail and detailed assessment

        Example:
            >>> result = await agent.verify_solution(
            ...     solution="Implement JWT auth...",
            ...     critique={"issues": ["missing rate limiting"]},
            ...     sub_problem={"title": "Auth", "requirements": ["JWT", "OAuth"]},
            ...     context={}
            ... )
            >>> passes = result["passes"]
        """
        logger.info(f"Verifying solution for: {sub_problem.get('title', 'Unknown')}")

        # Determine evaluation criteria
        if not evaluation_criteria:
            evaluation_criteria = self._get_default_criteria(sub_problem)

        # Perform evaluation
        if "solution_eval" in self.tools:
            evaluation = await self.use_tool(
                "solution_eval",
                solution=solution,
                criteria=evaluation_criteria,
                context={
                    "requirements": sub_problem.get("requirements", []),
                    "constraints": context.get("constraints", [])
                }
            )
        else:
            evaluation = {"overall_score": 0, "feedback": "No evaluator available"}

        # Get benchmarks if available
        benchmarks = []
        if use_benchmarks and "knowledge_search" in self.tools:
            try:
                domain = sub_problem.get("domain", "general")
                benchmark_results = await self.use_tool(
                    "knowledge_search",
                    search_type="verification_benchmarks",
                    query=domain,
                    top_k=3
                )
                benchmarks = benchmark_results
            except Exception as e:
                logger.warning(f"Benchmark retrieval failed: {e}")

        # Build verification prompt
        prompt = self._build_verification_prompt(
            solution, critique, sub_problem, evaluation, benchmarks
        )

        # Call LLM for final verification
        verification = await self._call_llm(prompt)

        # Parse verification result
        parsed_verification = self._parse_verification(verification)

        # Determine if solution passes
        passes = self._determine_pass_fail(
            parsed_verification,
            evaluation.get("overall_score", 0),
            critique
        )

        # Store verification report
        artifact_id = None
        if self.storage:
            artifact_id = await self.storage.store_artifact(
                artifact_type="verification",
                content=verification,
                metadata={
                    "stage": "stage_3",
                    "team": "gold",
                    "sub_problem_id": sub_problem.get("id", "unknown"),
                    "passes": passes,
                    "overall_score": parsed_verification.get("overall_score", 0)
                },
                links_to=[
                    context.get("solution_artifact_id"),
                    context.get("critique_artifact_id")
                ]
            )

        result = {
            "verification": verification,
            "parsed": parsed_verification,
            "passes": passes,
            "sub_problem_id": sub_problem.get("id"),
            "sub_problem_title": sub_problem.get("title"),
            "artifact_id": artifact_id,
            "overall_score": parsed_verification.get("overall_score", 0),
            "evaluation": evaluation,
            "benchmarks_used": len(benchmarks),
            "agent_metadata": self.get_metadata()
        }

        logger.info(f"Verification complete: passes={passes}, score={result['overall_score']}")
        return result

    def _get_default_criteria(self, sub_problem: Dict[str, Any]) -> List[str]:
        """Get default evaluation criteria based on sub-problem"""
        criteria = ["completeness", "correctness", "efficiency", "clarity"]

        # Add domain-specific criteria
        domain = sub_problem.get("domain", "").lower()
        if "security" in domain or "auth" in sub_problem.get("title", "").lower():
            criteria.append("security")
        if "performance" in domain or "scalability" in domain:
            criteria.append("performance")
        if "ui" in domain or "user" in domain:
            criteria.append("user_experience")

        return criteria

    def _build_verification_prompt(
        self,
        solution: str,
        critique: Optional[Dict[str, Any]],
        sub_problem: Dict[str, Any],
        evaluation: Dict[str, Any],
        benchmarks: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for solution verification"""
        parts = [
            "# Solution Verification",
            "\n## Sub-Problem:",
            f"Title: {sub_problem.get('title', 'Unknown')}",
            f"Description: {sub_problem.get('description', 'No description')}",
            f"\nRequirements:",
        ]

        for req in sub_problem.get("requirements", []):
            parts.append(f"  - {req}")

        parts.append([
            "\n## Solution to Verify:",
            solution
        ])

        # Add critique if available
        if critique:
            parts.append([
                "\n## Red Team Critique:",
                f"Issues: {critique.get('issues', [])}",
                f"Concerns: {critique.get('concerns', [])}"
            ])

        # Add evaluation
        parts.append([
            f"\n## Automated Evaluation:",
            f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10",
            f"Criteria Scores: {evaluation.get('criteria_scores', {})}"
        ])

        # Add benchmarks if available
        if benchmarks:
            parts.append("\n## Benchmark Comparisons:")
            for i, benchmark in enumerate(benchmarks[:3], 1):
                parts.append(f"\nBenchmark {i}:")
                parts.append(f"Criteria: {benchmark.get('criteria', [])}")
                parts.append(f"Threshold: {benchmark.get('threshold', 'N/A')}")

        parts.append("""
# Verification Instructions

1. Verify all requirements are addressed
2. Check if Red Team concerns are mitigated
3. Evaluate against automated assessment
4. Compare with benchmarks if available
5. Assess overall quality and readiness

# Output Format

Provide your verification in the following structure:

## Overall Assessment
[Overall score 1-10 and summary]

## Requirements Verification
[For each requirement: MET / NOT_MET with explanation]

## Issue Resolution
[How Red Team issues were addressed or not]

## Benchmark Comparison
[How solution compares to benchmarks]

## Quality Assessment
- Completeness: [score 1-10]
- Correctness: [score 1-10]
- Efficiency: [score 1-10]
- Clarity: [score 1-10]

## Final Verdict
VERIFIED_PASSED - Solution meets all requirements
VERIFIED_CONDITIONAL - Solution passes with minor reservations
VERIFIED_FAILED - Solution does not meet requirements

## Recommendations
[Any final recommendations or conditions]
""")

        return "\n".join(str(p) for p in parts)

    def _parse_verification(self, verification: str) -> Dict[str, Any]:
        """Parse verification into structured format"""
        parsed = {
            "overall_score": 0,
            "verdict": "VERIFIED_FAILED",
            "requirements_verification": {},
            "quality_scores": {},
            "recommendations": []
        }

        import re

        # Extract overall score
        score_match = re.search(r'overall.*?(\d+(?:\.\d+)?)', verification, re.IGNORECASE)
        if score_match:
            parsed["overall_score"] = float(score_match.group(1))

        # Extract verdict
        verdict_match = re.search(
            r'(VERIFIED_PASSED|VERIFIED_CONDITIONAL|VERIFIED_FAILED)',
            verification,
            re.IGNORECASE
        )
        if verdict_match:
            parsed["verdict"] = verdict_match.group(1)

        # Extract quality scores
        quality_metrics = ["completeness", "correctness", "efficiency", "clarity"]
        for metric in quality_metrics:
            metric_match = re.search(
                rf'{metric}.*?:.*?(\d+(?:\.\d+)?)',
                verification,
                re.IGNORECASE
            )
            if metric_match:
                parsed["quality_scores"][metric] = float(metric_match.group(1))

        # Extract recommendations
        rec_section = re.search(
            r'Recommendations.*?$',
            verification,
            re.DOTALL | re.IGNORECASE
        )
        if rec_section:
            recommendations = re.findall(r'^-\s*(.+)$', rec_section.group(0), re.MULTILINE)
            parsed["recommendations"] = recommendations

        return parsed

    def _determine_pass_fail(
        self,
        verification: Dict[str, Any],
        evaluation_score: float,
        critique: Optional[Dict[str, Any]]
    ) -> bool:
        """Determine if solution passes verification"""
        # Check verdict
        verdict = verification.get("verdict", "VERIFIED_FAILED")
        if "PASSED" in verdict:
            return True
        if "FAILED" in verdict:
            return False

        # Check overall score (threshold: 7.0)
        overall_score = verification.get("overall_score", evaluation_score)
        if overall_score >= 7.0:
            return True

        # Check if critical issues remain
        if critique:
            critical_issues = [
                issue for issue in critique.get("issues", [])
                if "critical" in issue.lower() or "high" in issue.lower()
            ]
            if len(critical_issues) > 0:
                return False

        return False

    async def _execute_general_task(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a general Gold Team task"""
        prompt = self._build_prompt(task, context, include_tools=True)
        response = await self._call_llm(prompt)

        return {
            "response": response,
            "task": task,
            "agent_metadata": self.get_metadata()
        }
