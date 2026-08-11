"""
Sovereign-Grade Problem Decomposition System - Refinement Coordinator
Implements iterative refinement with feedback loops and continuous improvement.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

from sovereign_data_models import (
    DecompositionPlan, SubProblem, Feedback, ValidationResult,
    QualityScores, SolutionAttempt, generate_id
)
from sovereign_gauntlets import GauntletSystem
from sovereign_quality_assessment import QualityAssessor
from sovereign_team_coordination import TeamCoordinator

logger = logging.getLogger(__name__)


@dataclass
class RefinementPlan:
    """Plan for refining a decomposition."""
    id: str
    plan_id: str
    issues: List[Dict[str, Any]]
    improvements: List[str]
    priority_order: List[str]  # Issue IDs in priority order
    estimated_effort: int  # hours
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RefinementCycle:
    """Tracks a single refinement cycle."""
    cycle_number: int
    plan_id: str
    feedback_received: List[Feedback]
    improvements_applied: List[str]
    quality_before: float
    quality_after: float
    gauntlet_results: Dict[str, ValidationResult]
    converged: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RefinementMetrics:
    """Metrics for refinement process."""
    total_cycles: int
    quality_improvement: float
    issues_resolved: int
    issues_remaining: int
    convergence_rate: float
    time_spent: float  # hours


class RefinementCoordinator:
    """Coordinates iterative refinement of decomposition plans with LLM intelligence."""
    
    def __init__(
        self,
        gauntlet_system: Optional[GauntletSystem] = None,
        quality_assessor: Optional[QualityAssessor] = None,
        team_coordinator: Optional[TeamCoordinator] = None,
        openevolve_client=None
    ):
        """
        Initialize refinement coordinator.
        
        Args:
            gauntlet_system: Optional GauntletSystem instance
            quality_assessor: Optional QualityAssessor instance
            team_coordinator: Optional TeamCoordinator instance
            openevolve_client: Optional OpenEvolve client for LLM
        """
        self.gauntlet_system = gauntlet_system or GauntletSystem()
        self.quality_assessor = quality_assessor or QualityAssessor()
        self.team_coordinator = team_coordinator or TeamCoordinator()
        self.openevolve_client = openevolve_client
        
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger = logging.getLogger(__name__)
                self.logger.warning("OpenEvolve client not available for refinement")
        
        # Track refinement history
        self.refinement_history: Dict[str, List[RefinementCycle]] = defaultdict(list)
        self.refinement_plans: Dict[str, RefinementPlan] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def process_feedback(
        self,
        plan: DecompositionPlan,
        feedback_list: List[Feedback]
    ) -> Dict[str, Any]:
        """
        Processes feedback from multiple sources.
        
        Aggregates feedback, prioritizes by severity, and generates
        actionable improvements.
        
        Args:
            plan: The decomposition plan
            feedback_list: List of Feedback objects
            
        Returns:
            Dictionary with processed feedback and priorities
        """
        self.logger.info(f"Processing {len(feedback_list)} feedback items for plan {plan.id}")
        
        # Aggregate feedback by category
        categorized = self._categorize_feedback(feedback_list)
        
        # Prioritize by severity
        prioritized = self._prioritize_feedback(feedback_list)
        
        # Generate actionable improvements
        improvements = self._generate_improvements(feedback_list, plan)
        
        # Identify critical issues
        critical_issues = [f for f in feedback_list if f.severity == 'critical']
        
        return {
            'total_feedback': len(feedback_list),
            'categorized': categorized,
            'prioritized': prioritized,
            'improvements': improvements,
            'critical_count': len(critical_issues),
            'critical_issues': critical_issues,
            'actionable': len(improvements) > 0
        }
    
    def generate_refinement_plan(
        self,
        plan: DecompositionPlan,
        feedback_list: List[Feedback],
        strategy: Optional[Dict[str, Any]] = None
    ) -> RefinementPlan:
        """
        Generates a refinement plan based on feedback and a smart strategy.
        
        Args:
            plan: The decomposition plan to refine
            feedback_list: Feedback to address
            strategy: Optional smart strategy to guide the refinement
            
        Returns:
            RefinementPlan with prioritized improvements
        """
        self.logger.info(f"Generating refinement plan for {plan.id} using strategy: {strategy.get('strategy_type') if strategy else 'default'}")
        
        # Process feedback
        processed = self.process_feedback(plan, feedback_list)
        
        # Use smart strategy actions if available
        if strategy and strategy.get('actions'):
            improvements = strategy['actions']
        else:
            improvements = processed['improvements']

        # Extract issues
        issues = []
        for feedback in feedback_list:
            issue = {
                'id': feedback.id,
                'source': feedback.source,
                'type': feedback.feedback_type,
                'severity': feedback.severity,
                'content': feedback.content,
                'actionable': feedback.actionable
            }
            issues.append(issue)
        
        # Prioritize issues based on strategy if available
        if strategy and strategy.get('priority'):
            priority_keyword = strategy['priority']
            priority_order = [
                issue['id'] for issue in issues if priority_keyword in issue['severity']
            ]
            # Add the rest
            priority_order.extend([
                issue['id'] for issue in issues if priority_keyword not in issue['severity']
            ])
        else:
            priority_order = [
                issue['id'] for issue in 
                sorted(issues, key=lambda x: self._severity_score(x['severity']), reverse=True)
            ]
        
        # Estimate effort
        estimated_effort = self._estimate_refinement_effort(issues)
        
        # Create refinement plan
        refinement_plan = RefinementPlan(
            id=generate_id("refinement_plan"),
            plan_id=plan.id,
            issues=issues,
            improvements=improvements,
            priority_order=priority_order,
            estimated_effort=estimated_effort
        )
        
        self.refinement_plans[refinement_plan.id] = refinement_plan
        
        self.logger.info(f"Created refinement plan with {len(issues)} issues, "
                        f"estimated effort: {estimated_effort}h")
        
        return refinement_plan
    
    def execute_refinement(
        self,
        plan: DecompositionPlan,
        refinement_plan: RefinementPlan
    ) -> Tuple[DecompositionPlan, RefinementMetrics]:
        """
        Executes refinement plan to improve decomposition.
        
        Args:
            plan: The decomposition plan to refine
            refinement_plan: The refinement plan to execute
            
        Returns:
            Tuple of (refined_plan, metrics)
        """
        self.logger.info(f"Executing refinement plan {refinement_plan.id}")
        
        start_time = datetime.now()
        
        # Get initial quality
        initial_quality = self.quality_assessor.generate_quality_report(plan)
        quality_before = initial_quality.metrics.overall_score
        
        # Apply improvements
        refined_plan = self._apply_improvements(plan, refinement_plan)
        
        # Re-run validation
        gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(refined_plan)
        
        # Get final quality
        final_quality = self.quality_assessor.generate_quality_report(refined_plan)
        quality_after = final_quality.metrics.overall_score
        
        # Calculate metrics
        end_time = datetime.now()
        time_spent = (end_time - start_time).total_seconds() / 3600  # hours
        
        issues_resolved = sum(1 for issue in refinement_plan.issues 
                             if self._is_issue_resolved(issue, gauntlet_results))
        
        metrics = RefinementMetrics(
            total_cycles=1,
            quality_improvement=quality_after - quality_before,
            issues_resolved=issues_resolved,
            issues_remaining=len(refinement_plan.issues) - issues_resolved,
            convergence_rate=quality_after / max(quality_before, 0.01),
            time_spent=time_spent
        )
        
        self.logger.info(f"Refinement complete: quality {quality_before:.2f} -> {quality_after:.2f}")
        
        return refined_plan, metrics
    
    def track_refinement_cycles(
        self,
        plan: DecompositionPlan,
        max_cycles: int = 5,
        convergence_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Tracks refinement cycles with convergence detection using smart strategies.
        
        Args:
            plan: The decomposition plan
            max_cycles: Maximum refinement iterations
            convergence_threshold: Quality improvement threshold for convergence
            
        Returns:
            Dictionary with cycle tracking information
        """
        self.logger.info(f"Starting smart refinement cycle tracking for {plan.id}")
        
        current_plan = plan
        cycle_number = 0
        converged = False
        previous_quality = 0.0
        
        while cycle_number < max_cycles and not converged:
            cycle_number += 1
            self.logger.info(f"Refinement cycle {cycle_number}/{max_cycles}")
            
            # Run gauntlets
            gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(current_plan)
            
            # Get quality score
            quality_report = self.quality_assessor.generate_quality_report(current_plan)
            current_quality = quality_report.metrics.overall_score
            
            # Check convergence
            if cycle_number > 1:
                improvement = current_quality - previous_quality
                if improvement < convergence_threshold:
                    converged = True
                    self.logger.info(f"Converged after {cycle_number} cycles")
            
            # Collect feedback
            feedback = self.gauntlet_system.process_gauntlet_feedback(gauntlet_results)
            
            # Create cycle record
            cycle = RefinementCycle(
                cycle_number=cycle_number,
                plan_id=plan.id,
                feedback_received=feedback,
                improvements_applied=[],
                quality_before=previous_quality,
                quality_after=current_quality,
                gauntlet_results={name: result for name, result in gauntlet_results.items()},
                converged=converged
            )
            
            self.refinement_history[plan.id].append(cycle)
            
            # If not converged and feedback exists, refine
            if not converged and feedback:
                # Generate a smart strategy
                smart_strategy = self.generate_smart_refinement_strategy(current_plan, feedback)
                self.logger.info(f"Applying smart refinement strategy: {smart_strategy.get('strategy_type')}")

                # Generate a refinement plan based on the smart strategy
                refinement_plan = self.generate_refinement_plan(current_plan, feedback, smart_strategy)
                
                # Execute the refinement
                current_plan, _ = self.execute_refinement(current_plan, refinement_plan)
                cycle.improvements_applied = refinement_plan.improvements
            
            previous_quality = current_quality
        
        return {
            'plan_id': plan.id,
            'total_cycles': cycle_number,
            'converged': converged,
            'final_quality': previous_quality,
            'cycles': self.refinement_history[plan.id],
            'max_cycles_reached': cycle_number >= max_cycles
        }
    
    def get_refinement_history(self, plan_id: str) -> List[RefinementCycle]:
        """Get refinement history for a plan."""
        return self.refinement_history.get(plan_id, [])
    
    def get_convergence_metrics(self, plan_id: str) -> Dict[str, Any]:
        """Calculate convergence metrics for a plan."""
        cycles = self.refinement_history.get(plan_id, [])
        
        if not cycles:
            return {
                'has_data': False,
                'total_cycles': 0
            }
        
        quality_progression = [c.quality_after for c in cycles]
        improvements = [c.quality_after - c.quality_before for c in cycles if c.quality_before > 0]
        
        return {
            'has_data': True,
            'total_cycles': len(cycles),
            'quality_progression': quality_progression,
            'total_improvement': quality_progression[-1] - quality_progression[0] if quality_progression else 0,
            'avg_improvement_per_cycle': sum(improvements) / len(improvements) if improvements else 0,
            'converged': cycles[-1].converged if cycles else False,
            'final_quality': quality_progression[-1] if quality_progression else 0
        }
    
    # Helper methods
    
    def _categorize_feedback(self, feedback_list: List[Feedback]) -> Dict[str, List[Feedback]]:
        """Categorize feedback by type."""
        categorized = defaultdict(list)
        for feedback in feedback_list:
            categorized[feedback.feedback_type].append(feedback)
        return dict(categorized)
    
    def _prioritize_feedback(self, feedback_list: List[Feedback]) -> List[Feedback]:
        """Prioritize feedback by severity."""
        severity_order = {'critical': 4, 'major': 3, 'minor': 2, 'info': 1}
        return sorted(
            feedback_list,
            key=lambda f: severity_order.get(f.severity, 0),
            reverse=True
        )
    
    def _generate_improvements(
        self,
        feedback_list: List[Feedback],
        plan: DecompositionPlan
    ) -> List[str]:
        """Generate actionable improvements from feedback."""
        improvements = []
        
        for feedback in feedback_list:
            if feedback.actionable:
                # Extract improvement suggestions from feedback
                if 'improvements' in feedback.metadata:
                    improvements.extend(feedback.metadata['improvements'])
                else:
                    # Generate generic improvement
                    improvements.append(f"Address {feedback.severity} issue: {feedback.content[:100]}")
        
        return improvements
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score."""
        scores = {'critical': 4, 'major': 3, 'minor': 2, 'info': 1}
        return scores.get(severity, 0)
    
    def _estimate_refinement_effort(self, issues: List[Dict[str, Any]]) -> int:
        """Estimate effort required for refinement in hours."""
        effort_by_severity = {'critical': 4, 'major': 2, 'minor': 1, 'info': 0.5}
        total_effort = sum(effort_by_severity.get(issue['severity'], 1) for issue in issues)
        return int(total_effort)

    def _compute_fatigue_score(self, text: str) -> float:
        """
        Compute a fatigue score based on repetition and lexical diversity.
        Higher values indicate potential stagnation.
        """
        tokens = [t for t in (text or "").split() if t]
        if not tokens:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)
        repetition_rate = 1.0 - unique_ratio
        perplexity_proxy = min(1.0, 1.0 - unique_ratio)
        fatigue = min(1.0, 0.6 * repetition_rate + 0.4 * perplexity_proxy)
        return fatigue
    
    def _apply_improvements(
        self,
        plan: DecompositionPlan,
        refinement_plan: RefinementPlan
    ) -> DecompositionPlan:
        """Apply improvements to decomposition plan."""
        self.logger.info(f"Applying {len(refinement_plan.improvements)} improvements")
        
        # Create a new plan with improvements
        refined_sub_problems = []
        
        for sub_problem in plan.sub_problems:
            # Check if this sub-problem has specific issues to address
            relevant_issues = [
                issue for issue in refinement_plan.issues
                if self._is_issue_relevant_to_subproblem(issue, sub_problem)
            ]
            
            if relevant_issues:
                # If there are specific issues for this sub-problem, refine it
                try:
                    refined_sp = self._refine_sub_problem(sub_problem, relevant_issues)
                    refined_sub_problems.append(refined_sp)
                except Exception as e:
                    self.logger.error(f"Failed to refine sub-problem {sub_problem.id}: {e}")
                    # Keep original sub-problem if refinement fails
                    refined_sub_problems.append(sub_problem)
            else:
                # No specific issues, keep original
                refined_sub_problems.append(sub_problem)
        
        # Create new plan with refined sub-problems
        refined_plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id=plan.problem_id,
            strategy=plan.strategy,
            sub_problems=refined_sub_problems,
            dependency_graph=plan.dependency_graph,
            validation_checkpoints=plan.validation_checkpoints,
            quality_scores=plan.quality_scores,
            confidence_level=min(1.0, plan.confidence_level + 0.05),  # Slightly increase confidence
            created_by="refinement_coordinator",
            metadata={
                **(plan.metadata or {}),
                'refined_from': plan.id,
                'refinement_count': (plan.metadata or {}).get('refinement_count', 0) + 1,
                'last_refinement': datetime.now().isoformat()
            }
        )
        
        return refined_plan
    
    def _is_issue_relevant_to_subproblem(self, issue: Dict[str, Any], sub_problem: SubProblem) -> bool:
        """Check if an issue is relevant to a specific sub-problem."""
        # Simple relevance check based on content
        issue_content = issue.get('content', '').lower()
        sub_problem_info = (sub_problem.title + sub_problem.description).lower()
        
        # Check if issue content mentions anything in the sub-problem
        issue_keywords = issue_content.split()
        for keyword in issue_keywords[:5]:  # Check first 5 words
            if len(keyword) > 3 and keyword in sub_problem_info:
                return True
        
        return False
    
    def _refine_sub_problem(self, sub_problem: SubProblem, issues: List[Dict[str, Any]]) -> SubProblem:
        """Refine a specific sub-problem based on issues using LLM."""
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for sub-problem refinement.")

        try:
            return self._refine_sub_problem_with_llm(sub_problem, issues)
        except Exception as e:
            self.logger.error(f"LLM sub-problem refinement failed: {e}")
            # In this case, we can return the original sub-problem to avoid halting the entire process.
            # This is a case where a fallback is acceptable.
            return sub_problem
    
    def _refine_sub_problem_with_llm(self, sub_problem: SubProblem, issues: List[Dict[str, Any]]) -> SubProblem:
        """Use LLM to refine a sub-problem based on identified issues."""
        issues_text = "\n".join([
            f"- [{issue['severity']}] {issue['content']}"
            for issue in issues
        ])
        
        prompt = f"""You are an expert at refining problem decomposition sub-problems. Improve this sub-problem based on the identified issues.

CURRENT SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}
Type: {sub_problem.type.value}
Priority: {sub_problem.priority}
Effort: {sub_problem.estimated_effort} hours

IDENTIFIED ISSUES:
{issues_text}

IMPROVEMENT REQUEST:
Based on the issues, provide an improved version of this sub-problem:

1. REVISED TITLE: (if needed)
2. REVISED DESCRIPTION: (revised to address issues)
3. PRIORITY ADJUSTMENT: (1-10, adjusted based on importance after fixing issues)
4. EFFORT ADJUSTMENT: (hours, updated if scope changed)
5. IMPROVED SUCCESS CRITERIA: (if needed, specific to issue resolution)

Format your response EXACTLY as:
Title: <revised title>
Description: <revised description>
Priority: <1-10>
Effort: <hours>
SuccessCriteria: <improved criteria>

Be specific and actionable."""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=800
        )

        fatigue_score = self._compute_fatigue_score(result.best_code or "")
        if fatigue_score > 0.8:
            self.logger.warning("Agent fatigue detected (score %.2f). Forcing temperature reset.", fatigue_score)
            fallback_model = os.getenv("FATIGUE_FALLBACK_MODEL")
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="analysis",
                max_iterations=1,
                temperature=0.7,
                max_tokens=800,
                model_name=fallback_model
            )
        
        if not result.success or not result.best_code:
            raise RuntimeError("LLM evolution failed to produce a result for sub-problem refinement.")

        # Parse the response and update the sub-problem
        lines = result.best_code.strip().split('\n')
        
        title = sub_problem.title
        description = sub_problem.description
        priority = sub_problem.priority
        effort = sub_problem.estimated_effort
        success_criteria = sub_problem.success_criteria
        
        for line in lines:
            line = line.strip()
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("Description:"):
                description = line.split(":", 1)[1].strip()
            elif line.startswith("Priority:"):
                try:
                    priority = int(line.split(":", 1)[1].strip())
                    priority = max(1, min(10, priority))  # Clamp to 1-10
                except:
                    self.logger.debug(f"Failed to parse priority from line: {line}")
            elif line.startswith("Effort:"):
                try:
                    effort_raw = line.split(":", 1)[1].strip()
                    effort = int(effort_raw.split()[0])  # Extract number from "X hours"
                except:
                    self.logger.debug(f"Failed to parse effort from line: {line}")
            elif line.startswith("SuccessCriteria:"):
                criteria_text = line.split(":", 1)[1].strip()
                if criteria_text.lower() != 'none' and criteria_text:
                    success_criteria = [
                        SuccessCriterion(
                            id=generate_id("criterion"),
                            description=criteria_text,
                            metric="issue_resolution",
                            threshold=1.0,
                            validation_method="review"
                        )
                    ]
        
        # Create refined sub-problem
        refined_sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id=sub_problem.parent_id,
            title=title,
            description=description,
            type=sub_problem.type,
            complexity_score=sub_problem.complexity_score,
            dependencies=sub_problem.dependencies,
            success_criteria=success_criteria,
            validation_gauntlet=sub_problem.validation_gauntlet,
            priority=priority,
            estimated_effort=effort,
            metadata={
                **(sub_problem.metadata or {}),
                'refined_in_cycle': True,
                'llm_refined': True
            }
        )
        
        return refined_sp

    def auto_remediate_bugs(
        self,
        plan: DecompositionPlan,
        scan_root: str = "."
    ) -> Tuple[DecompositionPlan, Optional[RefinementMetrics]]:
        """
        Connect bug_scanner outputs to refinement loops for high severity issues.

        Args:
            plan: Decomposition plan to refine
            scan_root: Root directory to scan

        Returns:
            Tuple of (refined_plan, metrics) or (plan, None) if no action
        """
        try:
            from bug_scanner import scan_all_files
        except ImportError as e:
            self.logger.error(f"bug_scanner not available: {e}")
            return plan, None

        scan_results = scan_all_files(scan_root)
        high_severity = [
            bug for bug in scan_results
            if bug.get("severity") in {"CRITICAL", "HIGH"}
        ]
        if not high_severity:
            return plan, None

        feedback_list = []
        for bug in high_severity:
            feedback_list.append(
                Feedback(
                    id=generate_id("feedback"),
                    source="bug_scanner",
                    feedback_type="security",
                    content=f"{bug.get('description')} ({bug.get('file')}:{bug.get('line')})",
                    severity=bug.get("severity", "HIGH"),
                    actionable=True,
                    metadata=bug
                )
            )

        refinement_plan = self.generate_refinement_plan(
            plan,
            feedback_list,
            strategy={"strategy_type": "bug_scanner"}
        )
        refined_plan, metrics = self.execute_refinement(plan, refinement_plan)

        # Validate fix by re-scanning
        remaining = [
            bug for bug in scan_all_files(scan_root)
            if bug.get("severity") in {"CRITICAL", "HIGH"}
        ]
        if remaining:
            self.logger.warning("Bug scanner remediation incomplete; high severity issues remain.")
            return plan, metrics

        return refined_plan, metrics
    
    def _is_issue_resolved(
        self,
        issue: Dict[str, Any],
        gauntlet_results: Dict[str, ValidationResult]
    ) -> bool:
        """Check if an issue has been resolved."""
        # Check if gauntlets now pass
        for result in gauntlet_results.values():
            if result.passed:
                return True
        return False

    def generate_smart_refinement_strategy(self, plan: DecompositionPlan, feedback_list: List[Feedback]) -> Dict[str, Any]:
        """
        Use LLM to generate intelligent refinement strategy based on feedback.
        
        Args:
            plan: Decomposition plan to refine
            feedback_list: Feedback from validation
            
        Returns:
            Dictionary with refinement strategy and specific actions
        """
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for smart refinement strategy generation.")

        if not feedback_list:
            return {
                'strategy_type': 'none',
                'primary_focus': 'none',
                'actions': [],
                'priority': 'none',
                'expected_improvement': 'No feedback to process.',
                'method': 'llm'
            }

        try:
            # Build feedback summary
            feedback_summary = "\n".join([
                f"[{f.severity}] {f.source}: {f.content}"
                for f in feedback_list[:10]  # Limit to 10 for tokens
            ])
            
            # Build plan summary
            plan_summary = f"Strategy: {plan.strategy.value}, Sub-problems: {len(plan.sub_problems)}"
            
            prompt = f"""You are an expert at refining problem decompositions. Analyze this feedback and create a smart refinement strategy.

CURRENT DECOMPOSITION:
{plan_summary}

FEEDBACK RECEIVED:
{feedback_summary}

REFINEMENT STRATEGY:
Based on the feedback, provide a targeted refinement strategy:

1. STRATEGY TYPE: Choose one (restructure/enhance/split/merge/rebalance)
2. PRIMARY FOCUS: What aspect needs most attention
3. SPECIFIC ACTIONS: List 3-5 concrete actions to take
4. PRIORITY: Which issues to address first
5. EXPECTED IMPROVEMENT: What quality improvement to expect

Format EXACTLY as:
StrategyType: <type>
PrimaryFocus: <focus area>
Actions: <action1> | <action2> | <action3>
Priority: <high/medium/low priority issues>
ExpectedImprovement: <description>

Be specific and actionable."""
            
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="analysis",
                max_iterations=1,
                temperature=0.4,
                max_tokens=600
            )
            
            if result.success and result.best_code:
                return self._parse_refinement_strategy(result.best_code)
            else:
                raise RuntimeError("LLM evolution failed to produce a result for smart refinement strategy generation.")
        
        except Exception as e:
            self.logger.error(f"LLM refinement strategy generation failed: {e}")
            raise RuntimeError(f"Failed to generate smart refinement strategy using LLM: {e}") from e
    
    def _parse_refinement_strategy(self, response: str) -> Dict[str, Any]:
        """Parse LLM refinement strategy response."""
        lines = response.strip().split('\n')
        
        strategy = {
            'strategy_type': 'enhance',
            'primary_focus': 'quality',
            'actions': [],
            'priority': 'high',
            'expected_improvement': 'Improved quality',
            'method': 'llm'
        }
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'StrategyType':
                strategy['strategy_type'] = value.lower()
            elif key == 'PrimaryFocus':
                strategy['primary_focus'] = value
            elif key == 'Actions':
                strategy['actions'] = [a.strip() for a in value.split('|') if a.strip()]
            elif key == 'Priority':
                strategy['priority'] = value.lower()
            elif key == 'ExpectedImprovement':
                strategy['expected_improvement'] = value
        
        return strategy
    
