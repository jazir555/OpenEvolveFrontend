"""
Solution Evaluation Tool

Allows agents to evaluate solutions against various criteria.
"""

from typing import List, Dict, Any, Optional
import logging

from ragbits_integration.agents.base_agent import AgentTool

logger = logging.getLogger(__name__)


class SolutionEvaluationTool(AgentTool):
    """
    Tool for evaluating solution quality.

    Provides agents with ability to:
    - Evaluate solution completeness
    - Check for common issues
    - Compare with benchmarks
    - Calculate quality scores

    Usage:
        tool = SolutionEvaluationTool(llm_client)
        evaluation = await tool.execute(
            solution="Solution content here...",
            criteria=["completeness", "correctness", "efficiency"]
        )
    """

    def __init__(self, llm_client, storage_manager=None):
        """
        Initialize the solution evaluation tool.

        Args:
            llm_client: LLM client for evaluation (can be Hephaestus)
            storage_manager: Optional storage manager for retrieving benchmarks
        """
        super().__init__(
            name="solution_eval",
            description="Evaluate solution quality against criteria and benchmarks"
        )
        self.llm = llm_client
        self.storage = storage_manager

    async def execute(
        self,
        solution: str,
        criteria: List[str],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute solution evaluation.

        Args:
            solution: Solution text to evaluate
            criteria: List of criteria to evaluate against
            context: Optional context (requirements, constraints, etc.)
            **kwargs: Additional parameters

        Returns:
            Evaluation dict with scores and feedback

        Example:
            >>> evaluation = await tool.execute(
            ...     solution="Implement JWT authentication...",
            ...     criteria=["completeness", "security", "efficiency"],
            ...     context={"requirements": ["must support OAuth", "must use bcrypt"]}
            ... )
        """
        logger.info(f"Evaluating solution against {len(criteria)} criteria")

        try:
            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(solution, criteria, context)

            # Call LLM for evaluation
            response = await self._call_llm(prompt)

            # Parse evaluation
            evaluation = self._parse_evaluation(response, criteria)

            # Add metadata
            evaluation["solution_length"] = len(solution)
            evaluation["criteria_evaluated"] = criteria
            evaluation["timestamp"] = self._get_timestamp()

            logger.info(f"Evaluation complete: overall_score={evaluation.get('overall_score', 0)}")
            return evaluation

        except Exception as e:
            logger.error(f"Solution evaluation failed: {e}")
            return {
                "error": str(e),
                "overall_score": 0,
                "criteria_scores": {},
                "feedback": "Evaluation failed"
            }

    def _build_evaluation_prompt(
        self,
        solution: str,
        criteria: List[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build evaluation prompt"""
        parts = [
            "You are an expert solution evaluator. Evaluate the following solution against the given criteria.",
            "\n# Solution to Evaluate:",
            solution,
            "\n# Evaluation Criteria:",
        ]

        # Add criteria
        for i, criterion in enumerate(criteria, 1):
            parts.append(f"{i}. {criterion}")

        # Add context if available
        if context:
            parts.append("\n# Context:")
            if "requirements" in context:
                parts.append("\nRequirements:")
                for req in context["requirements"]:
                    parts.append(f"- {req}")
            if "constraints" in context:
                parts.append("\nConstraints:")
                for constraint in context["constraints"]:
                    parts.append(f"- {constraint}")

        parts.append("""
# Evaluation Instructions:

For each criterion:
1. Provide a score from 0-10
2. Explain your reasoning
3. Identify specific issues or strengths

Provide your response in the following format:

## Overall Assessment
[Overall score 0-10 and brief summary]

## Criteria Scores
### [Criterion 1]
Score: [0-10]
Reasoning: [Your analysis]

### [Criterion 2]
Score: [0-10]
Reasoning: [Your analysis]

## Key Strengths
- [Strength 1]
- [Strength 2]

## Key Issues
- [Issue 1]
- [Issue 2]

## Recommendations
- [Recommendation 1]
- [Recommendation 2]
""")

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for evaluation"""
        if hasattr(self.llm, 'generate'):
            # Hephaestus or similar client
            response = await self.llm.generate(prompt=prompt)
            return response.get("text", response) if isinstance(response, dict) else str(response)
        elif hasattr(self.llm, '__call__'):
            # Direct callable
            response = await self.llm(prompt)
            return str(response)
        else:
            # Fallback
            return "Mock evaluation response"

    def _parse_evaluation(
        self,
        response: str,
        criteria: List[str]
    ) -> Dict[str, Any]:
        """Parse evaluation response"""
        evaluation = {
            "overall_score": 0,
            "criteria_scores": {},
            "reasoning": {},
            "strengths": [],
            "issues": [],
            "recommendations": [],
            "feedback": response
        }

        try:
            # Try to extract overall score
            import re
            overall_match = re.search(r'Overall.*?(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if overall_match:
                evaluation["overall_score"] = float(overall_match.group(1))

            # Extract criteria scores
            for criterion in criteria:
                criterion_match = re.search(
                    rf'{criterion}.*?Score.*?(\d+(?:\.\d+)?)',
                    response,
                    re.IGNORECASE | re.DOTALL
                )
                if criterion_match:
                    evaluation["criteria_scores"][criterion] = float(criterion_match.group(1))

            # Extract strengths
            strengths_match = re.search(
                r'Key Strengths.*?(?=Key Issues|$)',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if strengths_match:
                strengths = re.findall(r'^-\s*(.+)$', strengths_match.group(0), re.MULTILINE)
                evaluation["strengths"] = strengths

            # Extract issues
            issues_match = re.search(
                r'Key Issues.*?(?=Recommendations|$)',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if issues_match:
                issues = re.findall(r'^-\s*(.+)$', issues_match.group(0), re.MULTILINE)
                evaluation["issues"] = issues

            # Extract recommendations
            recommendations_match = re.search(
                r'Recommendations.*?$',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if recommendations_match:
                recommendations = re.findall(
                    r'^-\s*(.+)$',
                    recommendations_match.group(0),
                    re.MULTILINE
                )
                evaluation["recommendations"] = recommendations

        except Exception as e:
            logger.warning(f"Failed to parse evaluation response: {e}")

        return evaluation

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
