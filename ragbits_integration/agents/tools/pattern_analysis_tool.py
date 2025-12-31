"""
Pattern Analysis Tool

Allows agents to analyze patterns in solutions and critiques.
"""

from typing import List, Dict, Any, Optional
import logging
import re

from ragbits_integration.agents.base_agent import AgentTool

logger = logging.getLogger(__name__)


class PatternAnalysisTool(AgentTool):
    """
    Tool for analyzing patterns in solutions and critiques.

    Provides agents with ability to:
    - Identify common patterns in successful solutions
    - Detect anti-patterns in failed solutions
    - Extract best practices
    - Find recurring issues

    Usage:
        tool = PatternAnalysisTool(knowledge_retriever)
        patterns = await tool.execute(
            analysis_type="solution_patterns",
            domain="authentication"
        )
    """

    def __init__(self, knowledge_retriever):
        """
        Initialize the pattern analysis tool.

        Args:
            knowledge_retriever: RagbitsKnowledgeRetriever instance
        """
        super().__init__(
            name="pattern_analysis",
            description="Analyze patterns in solutions, critiques, and workflows"
        )
        self.retriever = knowledge_retriever

    async def execute(
        self,
        analysis_type: str,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute pattern analysis.

        Args:
            analysis_type: Type of analysis ("solution_patterns", "anti_patterns", "common_issues", "best_practices")
            domain: Optional domain to focus on
            **kwargs: Additional parameters

        Returns:
            Analysis results with identified patterns

        Example:
            >>> patterns = await tool.execute(
            ...     analysis_type="solution_patterns",
            ...     domain="security",
            ...     min_success_rate=0.8
            ... )
        """
        logger.info(f"Executing {analysis_type} analysis for domain: {domain or 'general'}")

        try:
            if analysis_type == "solution_patterns":
                return await self._analyze_solution_patterns(domain, **kwargs)
            elif analysis_type == "anti_patterns":
                return await self._analyze_anti_patterns(domain, **kwargs)
            elif analysis_type == "common_issues":
                return await self._analyze_common_issues(domain, **kwargs)
            elif analysis_type == "best_practices":
                return await self._analyze_best_practices(domain, **kwargs)
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_solution_patterns(
        self,
        domain: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze common patterns in successful solutions"""
        min_success_rate = kwargs.get("min_success_rate", 0.75)
        top_k = kwargs.get("top_k", 10)

        # Search for successful solutions in the domain
        query = f"{domain} solutions" if domain else "successful solutions"

        solutions = await self.retriever.retrieve_similar_solutions(
            problem_description=query,
            top_k=top_k,
            min_success_rate=min_success_rate
        )

        # Extract patterns
        patterns = {
            "domain": domain or "general",
            "total_solutions_analyzed": len(solutions),
            "patterns_found": [],
            "common_elements": [],
            "recommended_approaches": []
        }

        # Analyze solutions for patterns
        for solution in solutions:
            content = solution["content"]

            # Extract common elements
            elements = self._extract_elements(content)
            patterns["common_elements"].extend(elements)

            # Extract approach
            approach = self._extract_approach(content)
            if approach:
                patterns["recommended_approaches"].append({
                    "approach": approach,
                    "success_rate": solution["success_rate"],
                    "frequency": 1
                })

        # Aggregate patterns
        patterns["common_elements"] = self._aggregate_patterns(patterns["common_elements"])
        patterns["recommended_approaches"] = self._aggregate_approaches(patterns["recommended_approaches"])

        return patterns

    async def _analyze_anti_patterns(
        self,
        domain: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze anti-patterns in failed solutions"""
        max_success_rate = kwargs.get("max_success_rate", 0.5)
        top_k = kwargs.get("top_k", 10)

        # Search for less successful solutions
        query = f"{domain} solutions" if domain else "solutions"

        # Note: We'd need to modify retrieve_similar_solutions to support max_success_rate
        # For now, we'll search and filter
        solutions = await self.retriever.retrieve_similar_solutions(
            problem_description=query,
            top_k=top_k * 2  # Get more to filter
        )

        # Filter for low success rate
        failed_solutions = [
            s for s in solutions
            if s["success_rate"] <= max_success_rate
        ][:top_k]

        anti_patterns = {
            "domain": domain or "general",
            "total_analyzed": len(failed_solutions),
            "anti_patterns": []
        }

        # Extract anti-patterns
        for solution in failed_solutions:
            content = solution["content"]
            issues = self._extract_issues(content)

            for issue in issues:
                anti_patterns["anti_patterns"].append({
                    "issue": issue,
                    "frequency": 1,
                    "context": content[:200] + "..."
                })

        # Aggregate anti-patterns
        anti_patterns["anti_patterns"] = self._aggregate_patterns(anti_patterns["anti_patterns"])

        return anti_patterns

    async def _analyze_common_issues(
        self,
        domain: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze common issues from critiques"""
        severity = kwargs.get("severity")
        top_k = kwargs.get("top_k", 20)

        # Get critique patterns
        query = domain or "general"
        critiques = await self.retriever.retrieve_critique_patterns(
            solution_type=query,
            top_k=top_k,
            severity=severity
        )

        issues = {
            "domain": domain or "general",
            "total_critiques_analyzed": len(critiques),
            "common_issues": []
        }

        # Extract and aggregate issues
        issue_counts = {}
        for critique in critiques:
            issue_type = critique.get("issue_type", "general")
            if issue_type not in issue_counts:
                issue_counts[issue_type] = {
                    "count": 0,
                    "severity": critique.get("severity", "medium"),
                    "descriptions": []
                }

            issue_counts[issue_type]["count"] += critique.get("frequency", 1)
            issue_counts[issue_type]["descriptions"].append(critique.get("pattern", ""))

        # Format results
        for issue_type, data in issue_counts.items():
            issues["common_issues"].append({
                "issue_type": issue_type,
                "frequency": data["count"],
                "severity": data["severity"],
                "examples": data["descriptions"][:3]  # Top 3 examples
            })

        # Sort by frequency
        issues["common_issues"].sort(key=lambda x: x["frequency"], reverse=True)

        return issues

    async def _analyze_best_practices(
        self,
        domain: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze best practices from successful solutions"""
        min_success_rate = kwargs.get("min_success_rate", 0.85)
        top_k = kwargs.get("top_k", 10)

        # Get top solutions
        query = f"{domain} solutions" if domain else "best solutions"

        solutions = await self.retriever.retrieve_similar_solutions(
            problem_description=query,
            top_k=top_k,
            min_success_rate=min_success_rate
        )

        best_practices = {
            "domain": domain or "general",
            "total_solutions_analyzed": len(solutions),
            "best_practices": []
        }

        # Extract best practices
        practice_counts = {}
        for solution in solutions:
            content = solution["content"]
            practices = self._extract_best_practices(content)

            for practice in practices:
                if practice not in practice_counts:
                    practice_counts[practice] = {
                        "count": 0,
                        "success_rate_sum": 0,
                        "sources": []
                    }

                practice_counts[practice]["count"] += 1
                practice_counts[practice]["success_rate_sum"] += solution["success_rate"]
                practice_counts[practice]["sources"].append({
                    "solution_id": solution.get("solution_id"),
                    "team": solution.get("team_used")
                })

        # Format results
        for practice, data in practice_counts.items():
            best_practices["best_practices"].append({
                "practice": practice,
                "frequency": data["count"],
                "avg_success_rate": data["success_rate_sum"] / data["count"],
                "source_count": len(data["sources"])
            })

        # Sort by frequency
        best_practices["best_practices"].sort(key=lambda x: x["frequency"], reverse=True)

        return best_practices

    def _extract_elements(self, text: str) -> List[str]:
        """Extract key elements from solution text"""
        # Look for bullet points, numbered lists, etc.
        elements = []

        # Bullet points
        bullets = re.findall(r'^[-*]\s+(.+)$', text, re.MULTILINE)
        elements.extend(bullets)

        # Numbered lists
        numbered = re.findall(r'^\d+\.\s+(.+)$', text, re.MULTILINE)
        elements.extend(numbered)

        return elements[:10]  # Limit to top 10

    def _extract_approach(self, text: str) -> Optional[str]:
        """Extract the main approach from solution"""
        # Look for approach/strategy keywords
        approach_match = re.search(
            r'(?:approach|strategy|method)[\s:]+([^\n]+(?:\n(?!approach|strategy|method)[^\n]+)*)',
            text,
            re.IGNORECASE
        )
        if approach_match:
            return approach_match.group(1).strip()
        return None

    def _extract_issues(self, text: str) -> List[str]:
        """Extract issues from solution/critique text"""
        issues = []

        # Look for issue/problem keywords
        issue_patterns = [
            r'(?:issue|problem|error|bug)[\s:]+([^\n]+)',
            r'(?:lacks|missing|fail)[\s:]+([^\n]+)',
            r'(?:not|doesn\'t|won\'t)[\s]+([^\n]+)'
        ]

        for pattern in issue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            issues.extend(matches)

        return issues[:10]

    def _extract_best_practices(self, text: str) -> List[str]:
        """Extract best practices from solution text"""
        practices = []

        # Look for best practice keywords
        practice_patterns = [
            r'(?:best practice|recommended|should)[\s:]+([^\n]+)',
            r'(?:ensure|implement|use)[\s]+([^\n]+?(?:for|to|with)[^\n]+)'
        ]

        for pattern in practice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            practices.extend(matches)

        return practices[:10]

    def _aggregate_patterns(self, patterns: List[str]) -> List[Dict[str, Any]]:
        """Aggregate similar patterns"""
        pattern_counts = {}
        for pattern in patterns:
            # Normalize pattern
            normalized = pattern.lower().strip()
            if normalized not in pattern_counts:
                pattern_counts[normalized] = 0
            pattern_counts[normalized] += 1

        # Sort by frequency
        sorted_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"pattern": pattern, "frequency": count}
            for pattern, count in sorted_patterns[:10]
        ]

    def _aggregate_approaches(self, approaches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate similar approaches"""
        approach_groups = {}

        for approach in approaches:
            approach_text = approach["approach"].lower() if approach.get("approach") else "unknown"

            if approach_text not in approach_groups:
                approach_groups[approach_text] = {
                    "approach": approach_text,
                    "count": 0,
                    "success_rate_sum": 0
                }

            approach_groups[approach_text]["count"] += 1
            approach_groups[approach_text]["success_rate_sum"] += approach.get("success_rate", 0)

        # Sort by count
        sorted_approaches = sorted(
            approach_groups.values(),
            key=lambda x: x["count"],
            reverse=True
        )

        # Calculate average success rate
        for approach in sorted_approaches:
            approach["avg_success_rate"] = (
                approach["success_rate_sum"] / approach["count"]
                if approach["count"] > 0 else 0
            )
            del approach["success_rate_sum"]

        return sorted_approaches[:10]
