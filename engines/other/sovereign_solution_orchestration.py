"""
Sovereign-Grade Problem Decomposition System - Solution Orchestration
Tracks solution attempts, validates, and integrates sub-solutions.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from sovereign_data_models import (
    DecompositionPlan, SubProblem, SolutionAttempt, ValidationResult,
    Feedback, generate_id
)
from sovereign_reliability import with_retry, with_error_handling, ValidationError, ErrorSeverity

# **LEAN INTEGRATION**: Formal verification for orchestration
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSolution:
    """Represents an integrated solution from multiple sub-solutions."""
    id: str
    plan_id: str
    sub_solutions: List[SolutionAttempt]
    integration_method: str
    final_content: str
    confidence_score: float
    validation_results: List[ValidationResult]
    conflicts_resolved: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class Conflict:
    """Represents a conflict between sub-solutions."""
    id: str
    solution_ids: List[str]
    conflict_type: str
    description: str
    severity: str  # low, medium, high, critical
    suggested_resolution: Optional[str] = None


class SolutionOrchestrator:
    """Orchestrates solution attempts and integration with LLM-powered conflict detection."""
    
    def __init__(self, openevolve_client=None):
        self.solution_attempts: Dict[str, List[SolutionAttempt]] = {}
        self.integrated_solutions: Dict[str, IntegratedSolution] = {}
        self.openevolve_client = openevolve_client
        self.logger = logging.getLogger(__name__)
        
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for solution orchestration")
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda sub_problem_id, approach, solution_content, team_id, confidence_score: SolutionAttempt(
        id=generate_id("solution"), sub_problem_id=sub_problem_id, approach=approach, solution_content=solution_content, team_id=team_id, confidence_score=confidence_score, status="error", metadata={"error": "Failed to track solution attempt"}
    ))
    def track_solution_attempt(
        self,
        sub_problem_id: str,
        approach: str,
        solution_content: str,
        team_id: str,
        confidence_score: float = 0.8
    ) -> SolutionAttempt:
        """
        Records solution attempt with metadata.
        
        Args:
            sub_problem_id: ID of the sub-problem being solved
            approach: Description of the approach taken
            solution_content: The actual solution content
            team_id: ID of the team that created the solution
            confidence_score: Confidence in the solution (0-1)
            
        Returns:
            SolutionAttempt object
        """
        self.logger.info(f"Tracking solution attempt for sub-problem {sub_problem_id}")
        
        attempt = SolutionAttempt(
            id=generate_id("solution"),
            sub_problem_id=sub_problem_id,
            approach=approach,
            solution_content=solution_content,
            team_id=team_id,
            confidence_score=confidence_score,
            validation_results=[],
            feedback=[],
            status="pending",
            created_at=datetime.now(),
            metadata={
                'content_length': len(solution_content),
                'approach_type': self._classify_approach(approach)
            }
        )
        
        # Store attempt
        if sub_problem_id not in self.solution_attempts:
            self.solution_attempts[sub_problem_id] = []
        self.solution_attempts[sub_problem_id].append(attempt)
        
        return attempt
    
    @with_error_handling(fallback=lambda *args, **kwargs: ValidationResult(validator="fallback", passed=False, score=0.0, feedback="Validation failed due to an unexpected error.", improvements=[], timestamp=datetime.now()), severity=ErrorSeverity.HIGH)
    @with_retry(max_attempts=2, retry_on=(RuntimeError,))
    def validate_solution(
        self,
        attempt: SolutionAttempt,
        sub_problem: SubProblem
    ) -> ValidationResult:
        """
        Validates solution against success criteria using LLM analysis.
        
        Args:
            attempt: The solution attempt to validate
            sub_problem: The sub-problem with success criteria
            
        Returns:
            ValidationResult with pass/fail and feedback
        """
        self.logger.info(f"Validating solution {attempt.id} with LLM.")
        
        if not sub_problem.success_criteria:
            # No criteria defined, assume valid
            result = ValidationResult(
                validator="solution_orchestrator",
                passed=True,
                score=0.8,
                feedback="No success criteria defined, solution accepted",
                improvements=[],
                timestamp=datetime.now()
            )
        else:
            if not self.openevolve_client:
                raise RuntimeError("OpenEvolve client not available for solution validation.")

            try:
                result = self._validate_solution_with_llm(attempt, sub_problem)
            except Exception as e:
                self.logger.error(f"LLM-based solution validation failed: {e}")
                raise ValidationError(f"Failed to validate solution using LLM: {e}") from e

        # Store validation result
        attempt.validation_results.append(result)
        
        # Update attempt status
        if result.passed:
            attempt.status = "validated"
        else:
            attempt.status = "rejected"
        
        return result

    def _validate_solution_with_llm(
        self,
        attempt: SolutionAttempt,
        sub_problem: SubProblem
    ) -> ValidationResult:
        """Use LLM to validate a solution against success criteria."""
        
        criteria_summary = "\n".join([
            f"- {sc.description} (Metric: {sc.metric}, Threshold: {sc.threshold})"
            for sc in sub_problem.success_criteria
        ])

        prompt = f"""You are an expert solution validator. Assess if the given solution meets the success criteria for the sub-problem.

SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}

SUCCESS CRITERIA:
{criteria_summary}

SOLUTION ATTEMPT:
Approach: {attempt.approach}
Content:
{attempt.solution_content}

VALIDATION TASK:
For each success criterion, determine if the solution meets it.
Provide a score from 0 to 100 for each criterion, where 100 is a perfect match.
Then provide an overall assessment.

Provide your assessment in this EXACT format:
Criterion 1 Score: <score>
Criterion 2 Score: <score>
...
Overall Score: <average score>
Feedback: <2-3 sentence summary of how well the solution meets the criteria>
Improvements: <improvement1> | <improvement2>

Be critical and justify your scores.
"""

        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=600
        )

        if not result.success or not result.best_code:
            raise RuntimeError("LLM evolution failed to produce a result for solution validation.")

        return self._parse_validation_response(result.best_code)

    def _parse_validation_response(self, response: str) -> ValidationResult:
        """Parse LLM validation response."""
        lines = response.strip().split('\n')
        scores = []
        overall_score = 0.0
        feedback = ""
        improvements = []

        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if "criterion" in key and "score" in key:
                try:
                    scores.append(float(value))
                except ValueError:
                    self.logger.debug("Failed to parse criterion score '%s' in validation response.", value)
            elif key == "overall score":
                try:
                    overall_score = float(value) / 100.0
                except ValueError:
                    self.logger.debug("Failed to parse overall score '%s' in validation response.", value)
            elif key == "feedback":
                feedback = value
            elif key == "improvements":
                improvements = [i.strip() for i in value.split('|') if i.strip()]

        if not scores and overall_score == 0.0:
             raise ValueError("LLM response for validation did not contain scores.")

        if scores and overall_score == 0.0:
            overall_score = (sum(scores) / len(scores)) / 100.0
        
        passed = overall_score >= 0.7

        return ValidationResult(
            validator="llm_solution_validator",
            passed=passed,
            score=overall_score,
            feedback=feedback or f"Overall validation score: {overall_score:.2f}",
            improvements=improvements,
            timestamp=datetime.now()
        )
    
    @with_error_handling(fallback=lambda *args, **kwargs: None, severity=ErrorSeverity.CRITICAL)
    @with_retry(max_attempts=2, retry_on=(RuntimeError,))
    def integrate_solutions(
        self,
        plan: DecompositionPlan,
        attempts: Optional[List[SolutionAttempt]] = None
    ) -> IntegratedSolution:
        """
        Combines sub-solutions into final solution using intelligent merging.
        
        Args:
            plan: The decomposition plan
            attempts: Optional list of solution attempts (uses stored if None)
            
        Returns:
            IntegratedSolution with combined content
        """
        self.logger.info(f"Intelligently integrating solutions for plan {plan.id}")
        
        # Get solution attempts for each sub-problem
        if attempts is None:
            attempts = []
            for sp in plan.sub_problems:
                sp_attempts = self.solution_attempts.get(sp.id, [])
                # Use the best validated attempt
                validated = [a for a in sp_attempts if a.status == "validated"]
                if validated:
                    best = max(validated, key=lambda a: a.confidence_score)
                    attempts.append(best)
        
        if not attempts:
            raise ValueError("No solution attempts available for integration")
        
        # Check for conflicts
        conflicts = self.detect_conflicts(attempts)
        
        # Integrate solutions intelligently
        integration_method = "intelligent_merge"
        combined_content = self.merge_solutions_intelligently(attempts, conflicts)
        
        # Calculate overall confidence
        overall_confidence = self.calculate_confidence(attempts)
        
        # Create integrated solution
        integrated = IntegratedSolution(
            id=generate_id("integrated"),
            plan_id=plan.id,
            sub_solutions=attempts,
            integration_method=integration_method,
            final_content=combined_content,
            confidence_score=overall_confidence,
            validation_results=[],
            conflicts_resolved=[c.id for c in conflicts],
            created_at=datetime.now(),
            metadata={
                'sub_solution_count': len(attempts),
                'total_length': len(combined_content),
                'conflicts_found': len(conflicts)
            }
        )
        
        # Store integrated solution
        self.integrated_solutions[plan.id] = integrated
        
        return integrated
    
    @with_error_handling(fallback=lambda *args, **kwargs: [], severity=ErrorSeverity.HIGH)
    @with_retry(max_attempts=2, retry_on=(RuntimeError,))
    def detect_conflicts(self, solutions: List[SolutionAttempt]) -> List[Conflict]:
        """
        Identifies conflicting sub-solutions using LLM analysis.
        
        Args:
            solutions: List of solution attempts
            
        Returns:
            List of detected conflicts
            
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        self.logger.info(f"Detecting conflicts among {len(solutions)} solutions with LLM.")
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for conflict detection.")

        if len(solutions) < 2:
            return []

        try:
            llm_conflicts = self._detect_conflicts_with_llm(solutions)
            self.logger.info(f"LLM detected {len(llm_conflicts)} conflicts")
            return llm_conflicts
        except Exception as e:
            self.logger.error(f"LLM-based conflict detection failed: {e}")
            raise RuntimeError(f"Failed to detect conflicts using LLM: {e}") from e
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda solutions: 0.0)
    def calculate_confidence(self, solutions: List[SolutionAttempt]) -> float:
        """
        Calculates overall solution confidence.
        
        Args:
            solutions: List of solution attempts
            
        Returns:
            Overall confidence score (0-1)
        """
        if not solutions:
            return 0.0
        
        # Base confidence is average of individual confidences
        avg_confidence = sum(s.confidence_score for s in solutions) / len(solutions)
        
        # Adjust based on validation results
        validated = sum(1 for s in solutions if s.status == "validated")
        validation_ratio = validated / len(solutions)
        
        # Adjust based on conflicts
        # (This would be called after conflict detection in practice)
        
        # Weighted combination
        overall = (avg_confidence * 0.6 + validation_ratio * 0.4)
        
        return min(1.0, overall)
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda plan_id: {})
    def get_solution_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Gets status of solution attempts for a plan.
        
        Args:
            plan_id: ID of the decomposition plan
            
        Returns:
            Dictionary with solution status information
        """
        integrated = self.integrated_solutions.get(plan_id)
        
        if integrated:
            return {
                'plan_id': plan_id,
                'status': 'integrated',
                'confidence': integrated.confidence_score,
                'sub_solution_count': len(integrated.sub_solutions),
                'conflicts_resolved': len(integrated.conflicts_resolved),
                'created_at': integrated.created_at.isoformat()
            }
        else:
            # Count attempts
            total_attempts = sum(len(attempts) for attempts in self.solution_attempts.values())
            validated = sum(1 for attempts in self.solution_attempts.values() 
                          for a in attempts if a.status == "validated")
            
            return {
                'plan_id': plan_id,
                'status': 'in_progress',
                'total_attempts': total_attempts,
                'validated_attempts': validated,
                'integration_ready': False
            }
    
    # Helper methods
    
    def _classify_approach(self, approach: str) -> str:
        """Classify the approach type."""
        approach_lower = approach.lower()
        
        if any(word in approach_lower for word in ['algorithm', 'compute', 'calculate']):
            return 'algorithmic'
        elif any(word in approach_lower for word in ['design', 'architecture', 'structure']):
            return 'architectural'
        elif any(word in approach_lower for word in ['implement', 'code', 'develop']):
            return 'implementation'
        elif any(word in approach_lower for word in ['analyze', 'study', 'research']):
            return 'analytical'
        else:
            return 'general'
    

    

    
    def clear_attempts(self, plan_id: Optional[str] = None):
        """Clear solution attempts (for testing or reset)."""
        if plan_id:
            # Clear attempts for specific plan
            to_remove = [sp_id for sp_id, attempts in self.solution_attempts.items()
                        if any(a.metadata.get('plan_id') == plan_id for a in attempts)]
            for sp_id in to_remove:
                del self.solution_attempts[sp_id]
            
            if plan_id in self.integrated_solutions:
                del self.integrated_solutions[plan_id]
        else:
            # Clear all
            self.solution_attempts.clear()
            self.integrated_solutions.clear()

    def _detect_conflicts_with_llm(self, solutions: List[SolutionAttempt]) -> List[Conflict]:
        """Use LLM to detect sophisticated conflicts between solutions."""
        # Build solutions summary
        sol_summary = "\n\n".join([
            f"SOLUTION {i+1} (ID: {s.id[:8]}):\nApproach: {s.approach}\nConfidence: {s.confidence_score:.2f}\nContent: {s.solution_content[:200]}..."
            for i, s in enumerate(solutions[:5])  # Limit to 5 for tokens
        ])
        
        prompt = f"""You are an expert at detecting conflicts and inconsistencies between solution attempts. Analyze these solutions for conflicts.

SOLUTIONS TO ANALYZE:
{sol_summary}

CONFLICT DETECTION:
Identify conflicts in these categories:

1. LOGICAL CONFLICTS: Contradictory logic or approaches
2. ASSUMPTION CONFLICTS: Incompatible assumptions
3. INTERFACE CONFLICTS: Incompatible interfaces or data formats
4. DEPENDENCY CONFLICTS: Conflicting dependencies or requirements
5. QUALITY CONFLICTS: Significant quality differences

For each conflict found, provide:
- Type: (logical/assumption/interface/dependency/quality)
- Description: Clear description of the conflict
- Severity: (critical/high/medium/low)
- Resolution: Suggested way to resolve

Format EXACTLY as:
---
CONFLICT 1
Type: <type>
Description: <description>
Severity: <severity>
Resolution: <resolution>
---

List all conflicts found (or "NO CONFLICTS" if none):"""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=800
        )
        
        if result.success and result.best_code:
            if "NO CONFLICTS" in result.best_code.upper():
                return []
            return self._parse_conflict_response(result.best_code, solutions)
        
        return []
    
    def _parse_conflict_response(self, response: str, solutions: List[SolutionAttempt]) -> List[Conflict]:
        """Parse LLM conflict detection response."""
        conflicts = []
        sections = response.split('---')
        
        for section in sections:
            section = section.strip()
            if not section or 'CONFLICT' not in section:
                continue
            
            try:
                conflict_type = self._extract_conflict_field(section, 'Type:')
                description = self._extract_conflict_field(section, 'Description:')
                severity = self._extract_conflict_field(section, 'Severity:').lower()
                resolution = self._extract_conflict_field(section, 'Resolution:')
                
                if not description:
                    continue
                
                conflict = Conflict(
                    id=generate_id("conflict"),
                    solution_ids=[s.id for s in solutions],
                    conflict_type=conflict_type or "unknown",
                    description=description,
                    severity=severity if severity in ['critical', 'high', 'medium', 'low'] else 'medium',
                    suggested_resolution=resolution
                )
                conflicts.append(conflict)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse conflict section: {e}")
                continue
        
        return conflicts
    
    def _extract_conflict_field(self, text: str, field_name: str) -> str:
        """Extract field value from conflict text."""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(field_name):
                return line.split(':', 1)[1].strip()
        return ""
    
    def merge_solutions_intelligently(self, solutions: List[SolutionAttempt], conflicts: List[Conflict]) -> str:
        """
        Intelligently merge solutions using LLM, resolving conflicts.
        
        Args:
            solutions: Solutions to merge
            conflicts: Detected conflicts
            
        Returns:
            Merged solution content
            
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for intelligent merging.")
        
        try:
            # Build context
            sol_summary = "\n\n".join([
                f"SOLUTION {i+1}:\n{s.solution_content[:300]}..."
                for i, s in enumerate(solutions[:5])
            ])
            
            conflict_summary = "\n".join([
                f"- {c.description} (Severity: {c.severity})"
                for c in conflicts[:5]
            ])
            
            prompt = f"""You are an expert at merging multiple solution attempts into a cohesive final solution. Merge these solutions while resolving conflicts.

SOLUTIONS TO MERGE:
{sol_summary}

CONFLICTS TO RESOLVE:
{conflict_summary if conflicts else "No conflicts detected"}

MERGING TASK:
Create a unified solution that:
1. Incorporates the best aspects of each solution
2. Resolves all conflicts intelligently
3. Maintains consistency and coherence
4. Preserves critical functionality from all solutions

Provide the merged solution:"""
            
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="code",
                max_iterations=1,
                temperature=0.4,
                max_tokens=1500
            )
            
            if result.success and result.best_code:
                self.logger.info("LLM successfully merged solutions")
                return result.best_code
            else:
                raise RuntimeError("LLM evolution failed to produce a result for intelligent merging.")
        
        except Exception as e:
            self.logger.error(f"LLM merging failed: {e}")
            raise RuntimeError(f"Failed to merge solutions intelligently using LLM: {e}") from e

    async def verify_solution_with_lean(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        **LEAN INTEGRATION**: Solution verification using Lean theorem prover.
        
        Verifies solution content with formal mathematical methods.
        
        Args:
            solution: Solution dictionary to verify
            
        Returns:
            Dict with verification results
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "reason": "Lean unavailable"}
        
        try:
            client = LeanAideClient()
            content = solution.get('final_content', solution.get('solution_content', str(solution)))
            
            # Autoformalize
            formalized = await client.translate_thm(content)
            
            if formalized.success and formalized.data:
                # Verify with Lean
                result = await client.elaborate(formalized.data.get('result', ''))
                
                return {
                    "verified": result.success,
                    "confidence": 1.0 if result.success else 0.0,
                    "proof": result.data.get('result') if result.data else None,
                    "solution_valid": result.success,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "verified": False,
                    "reason": "Autoformalization failed",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Lean verification error: {e}")
            return {"verified": False, "reason": str(e), "timestamp": datetime.now().isoformat()}


async def verify_with_lean(solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    **LEAN INTEGRATION**: Standalone solution verification using Lean theorem prover.
    
    Args:
        solution: Solution dictionary to verify
        
    Returns:
        Dict with verification results
    """
    if not LEAN_AVAILABLE:
        return {"verified": False, "reason": "Lean unavailable"}
    
    try:
        client = LeanAideClient()
        content = solution.get('final_content', solution.get('solution_content', str(solution)))
        
        # Autoformalize
        formalized = await client.translate_thm(content)
        
        if formalized.success and formalized.data:
            # Verify with Lean
            result = await client.elaborate(formalized.data.get('result', ''))
            
            return {
                "verified": result.success,
                "confidence": 1.0 if result.success else 0.0,
                "proof": result.data.get('result') if result.data else None,
                "solution_valid": result.success,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "verified": False,
                "reason": "Autoformalization failed",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {"verified": False, "reason": str(e), "timestamp": datetime.now().isoformat()}
