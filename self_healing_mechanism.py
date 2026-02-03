"""
Self-Healing Mechanism

Automatically detects and fixes issues in decompositions and solutions.

Features:
- Detects common decomposition problems
- Attempts automatic fixes with multiple strategies
- Falls back to manual review if can't fix
- Learns from successful healings
- Tracks quality improvement
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from collections import defaultdict, deque

from sovereign_data_models import (
    DecompositionPlan, SubProblem, ProblemDefinition,
    DependencyGraph, HealthIssue, HealingResult,
    ComplexityScore, EnhancedQualityScores, SuccessCriterion,
    generate_id
)

logger = logging.getLogger(__name__)



class SelfHealingMechanism:
    """
    Automatically detects and fixes issues in decompositions and solutions.

    Features:
    - Detects common problems (circular deps, complexity imbalance, etc.)
    - Attempts automatic fixes
    - Falls back to manual review if can't fix
    - Learns from successful healings
    """

    def __init__(self, llm_client=None):
        """
        Initialize with optional LLM for intelligent healing.

        Args:
            llm_client: Optional LLM client for intelligent healing operations
        """
        self.llm_client = llm_client
        self.healing_history: List[HealingResult] = []
        self.healing_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "avg_improvement": 0.0
        })

    def diagnose_problem(
        self,
        decomposition_plan: DecompositionPlan,
        quality_assessment: Optional[EnhancedQualityScores] = None
    ) -> List[HealthIssue]:
        """
        Diagnose problems in decomposition.

        Detects:
        1. Circular dependencies
        2. Unbalanced complexity
        3. Missing dependencies
        4. Inconsistent quality
        5. Invalid references
        6. Empty or vague sub-problems

        Args:
            decomposition_plan: Plan to diagnose
            quality_assessment: Optional quality assessment results

        Returns:
            List of identified health issues
        """
        issues = []

        # 1. Check for circular dependencies
        circular_issues = self._detect_circular_dependencies(decomposition_plan)
        issues.extend(circular_issues)

        # 2. Check for complexity imbalance
        complexity_issues = self._detect_complexity_imbalance(decomposition_plan)
        issues.extend(complexity_issues)

        # 3. Check for missing dependencies
        missing_dep_issues = self._detect_missing_dependencies(decomposition_plan)
        issues.extend(missing_dep_issues)

        # 4. Check for inconsistent quality
        if quality_assessment:
            quality_issues = self._detect_inconsistent_quality(
                decomposition_plan, quality_assessment
            )
            issues.extend(quality_issues)

        # 5. Check for invalid references
        ref_issues = self._detect_invalid_references(decomposition_plan)
        issues.extend(ref_issues)

        # 6. Check for vague sub-problems
        vague_issues = self._detect_vague_sub_problems(decomposition_plan)
        issues.extend(vague_issues)

        # 7. Check for orphan sub-problems
        orphan_issues = self._detect_orphan_sub_problems(decomposition_plan)
        issues.extend(orphan_issues)

        logger.info(f"Diagnosed {len(issues)} health issues in decomposition plan")
        return issues

    def attempt_healing(
        self,
        decomposition_plan: DecompositionPlan,
        health_issues: List[HealthIssue]
    ) -> HealingResult:
        """
        Attempt to automatically fix identified issues.

        Healing strategies:
        1. Remove circular dependencies
        2. Rebalance complexity (split/merge sub-problems)
        3. Add missing dependencies
        4. Fix invalid references
        5. Enhance vague sub-problems
        6. Regenerate low-quality sub-problems

        Args:
            decomposition_plan: Plan to heal
            health_issues: Issues to fix

        Returns:
            HealingResult with details of what was fixed
        """
        start_time = time.time()
        healing_id = f"healing_{uuid.uuid4().hex[:12]}"

        # Get initial quality
        quality_before = self._calculate_plan_quality(decomposition_plan)

        # Track changes
        sub_problems_added = 0
        sub_problems_removed = 0
        sub_problems_modified = 0
        dependencies_added: List[Tuple[str, str]] = []
        dependencies_removed: List[Tuple[str, str]] = []

        issues_healed = []
        issues_failed = []

        healing_methods_used = []

        # Group issues by type for efficient healing
        issues_by_type = self._group_issues_by_type(health_issues)

        # Heal each type of issue
        for issue_type, issues in issues_by_type.items():
            logger.info(f"Attempting to heal {len(issues)} issues of type: {issue_type}")

            # Update statistics
            self.healing_statistics[issue_type]["attempts"] += len(issues)

            if issue_type == "circular_dependency":
                healed, failed, changes = self.heal_circular_dependencies(
                    decomposition_plan, issues
                )
                issues_healed.extend(healed)
                issues_failed.extend(failed)
                dependencies_added.extend(changes.get("deps_added", []))
                dependencies_removed.extend(changes.get("deps_removed", []))
                sub_problems_modified += changes.get("modified", 0)
                healing_methods_used.append("circular_dependency_removal")

            elif issue_type == "complexity_imbalance":
                healed, failed, changes = self.heal_complexity_imbalance(
                    decomposition_plan
                )
                issues_healed.extend(healed)
                issues_failed.extend(failed)
                sub_problems_added += changes.get("added", 0)
                sub_problems_removed += changes.get("removed", 0)
                sub_problems_modified += changes.get("modified", 0)
                healing_methods_used.append("complexity_rebalancing")

            elif issue_type == "missing_dependencies":
                healed, failed, changes = self.heal_missing_dependencies(
                    decomposition_plan
                )
                issues_healed.extend(healed)
                issues_failed.extend(failed)
                dependencies_added.extend(changes.get("deps_added", []))
                sub_problems_modified += changes.get("modified", 0)
                healing_methods_used.append("dependency_inference")

            elif issue_type == "invalid_references":
                healed, failed, changes = self.heal_invalid_references(
                    decomposition_plan, issues
                )
                issues_healed.extend(healed)
                issues_failed.extend(failed)
                sub_problems_modified += changes.get("modified", 0)
                healing_methods_used.append("reference_correction")

            elif issue_type == "vague_sub_problems":
                healed, failed, changes = self.heal_vague_sub_problems(
                    decomposition_plan, issues
                )
                issues_healed.extend(healed)
                issues_failed.extend(failed)
                sub_problems_modified += changes.get("modified", 0)
                healing_methods_used.append("sub_problem_enhancement")

            elif issue_type == "orphan_sub_problems":
                healed, failed, changes = self.heal_orphan_sub_problems(
                    decomposition_plan, issues
                )
                issues_healed.extend(healed)
                issues_failed.extend(failed)
                dependencies_added.extend(changes.get("deps_added", []))
                sub_problems_modified += changes.get("modified", 0)
                healing_methods_used.append("orphan_integration")

            # Update success statistics
            for issue in healed:
                self.healing_statistics[issue_type]["successes"] += 1

        # Calculate final quality
        quality_after = self._calculate_plan_quality(decomposition_plan)
        quality_improvement = quality_after - quality_before

        # Calculate success rate
        total_issues = len(health_issues)
        healing_success_rate = len(issues_healed) / total_issues if total_issues > 0 else 0.0

        # Create healing result
        healing_duration = time.time() - start_time

        result = HealingResult(
            healing_id=healing_id,
            original_issues=health_issues,
            issues_healed=issues_healed,
            issues_failed=issues_failed,
            healing_success_rate=healing_success_rate,
            sub_problems_added=sub_problems_added,
            sub_problems_removed=sub_problems_removed,
            sub_problems_modified=sub_problems_modified,
            dependencies_added=dependencies_added,
            dependencies_removed=dependencies_removed,
            quality_before=quality_before,
            quality_after=quality_after,
            quality_improvement=quality_improvement,
            healing_duration=healing_duration,
            healing_methods_used=healing_methods_used,
            metadata={
                "total_issues": total_issues,
                "issues_processed": len(issues_healed) + len(issues_failed)
            }
        )

        # Record healing
        self.healing_history.append(result)
        self._update_healing_statistics(result)

        logger.info(
            f"Healing complete: {len(issues_healed)}/{total_issues} issues healed, "
            f"quality improved by {quality_improvement:.3f}"
        )

        return result

    def heal_circular_dependencies(
        self,
        plan: DecompositionPlan,
        cycles: List[HealthIssue]
    ) -> Tuple[List[HealthIssue], List[HealthIssue], Dict[str, Any]]:
        """
        Heal circular dependencies.

        Strategies:
        - Remove one edge from each cycle
        - Merge sub-problems in cycle
        - Re-decompose affected sub-problems

        Args:
            plan: Plan to heal
            cycles: Circular dependency issues

        Returns:
            Tuple of (healed_issues, failed_issues, changes)
        """
        healed = []
        failed = []
        changes = {"deps_removed": [], "modified": 0}

        for cycle_issue in cycles:
            if not cycle_issue.affected_sub_problems:
                failed.append(cycle_issue)
                continue

            cycle_nodes = cycle_issue.affected_sub_problems

            # Strategy: Remove edge with lowest impact
            # Find the edge in the cycle whose removal minimally affects plan
            edge_to_remove = self._find_best_edge_to_remove(plan, cycle_nodes)

            if edge_to_remove:
                # Remove the dependency
                for sub_problem in plan.sub_problems:
                    if edge_to_remove[1] in sub_problem.dependencies:
                        sub_problem.dependencies.remove(edge_to_remove[1])
                        changes["deps_removed"].append(edge_to_remove)
                        changes["modified"] += 1
                        break

                healed.append(cycle_issue)
                logger.info(f"Removed circular dependency edge: {edge_to_remove}")
            else:
                failed.append(cycle_issue)
                logger.warning(f"Could not find suitable edge to remove from cycle")

        return healed, failed, changes

    def heal_complexity_imbalance(
        self,
        plan: DecompositionPlan
    ) -> Tuple[List[HealthIssue], List[HealthIssue], Dict[str, Any]]:
        """
        Heal complexity imbalance.

        Strategies:
        - Split overly complex sub-problems
        - Merge overly simple sub-problems
        - Target: all sub-problems in 0.3-0.7 complexity range

        Args:
            plan: Plan to heal

        Returns:
            Tuple of (healed_issues, failed_issues, changes)
        """
        healed = []
        failed = []
        changes = {"added": 0, "removed": 0, "modified": 0}

        if not plan.sub_problems:
            return healed, failed, changes

        # Calculate complexity scores
        complexities = [
            (sp, sp.complexity_score.overall_complexity / 10.0)  # Normalize to 0-1
            for sp in plan.sub_problems
        ]

        avg_complexity = sum(c[1] for c in complexities) / len(complexities)

        # Find sub-problems that are too complex or too simple
        too_complex = [(sp, score) for sp, score in complexities if score > 0.7]
        too_simple = [(sp, score) for sp, score in complexities if score < 0.3]

        # Try to split complex ones
        for sp, score in too_complex:
            if self._split_sub_problem(plan, sp):
                changes["added"] += 1
                changes["modified"] += 1
                healed.append(HealthIssue(
                    issue_id=generate_id("complexity"),
                    issue_type="complexity_imbalance",
                    severity="medium",
                    description=f"Split overly complex sub-problem: {sp.title}",
                    affected_sub_problems=[sp.id],
                    healable=True,
                    healing_strategy="split",
                    healing_confidence=0.8,
                    quality_impact=0.2,
                    urgency="soon"
                ))

        # Try to merge simple ones
        for i in range(0, len(too_simple) - 1, 2):
            if i + 1 < len(too_simple):
                sp1, _ = too_simple[i]
                sp2, _ = too_simple[i + 1]

                if self._merge_sub_problems(plan, sp1, sp2):
                    changes["removed"] += 1
                    changes["modified"] += 1
                    healed.append(HealthIssue(
                        issue_id=generate_id("complexity"),
                        issue_type="complexity_imbalance",
                        severity="medium",
                        description=f"Merged simple sub-problems: {sp1.title} and {sp2.title}",
                        affected_sub_problems=[sp1.id, sp2.id],
                        healable=True,
                        healing_strategy="merge",
                        healing_confidence=0.7,
                        quality_impact=0.15,
                        urgency="soon"
                    ))

        return healed, failed, changes

    def heal_missing_dependencies(
        self,
        plan: DecompositionPlan
    ) -> Tuple[List[HealthIssue], List[HealthIssue], Dict[str, Any]]:
        """
        Heal missing dependencies.

        Analyzes sub-problem descriptions and adds
        implicit dependencies that were missed.

        Args:
            plan: Plan to heal

        Returns:
            Tuple of (healed_issues, failed_issues, changes)
        """
        healed = []
        failed = []
        changes = {"deps_added": [], "modified": 0}

        if not plan.sub_problems:
            return healed, failed, changes

        # Build keyword -> sub_problem mapping
        keyword_map = self._build_keyword_map(plan.sub_problems)

        # For each sub-problem, check for missing dependencies
        for sp in plan.sub_problems:
            # Find potential dependencies based on keywords
            potential_deps = self._find_potential_dependencies(
                sp, plan.sub_problems, keyword_map
            )

            # Add missing dependencies
            for dep_id in potential_deps:
                if dep_id not in sp.dependencies and dep_id != sp.id:
                    sp.dependencies.append(dep_id)
                    changes["deps_added"].append((sp.id, dep_id))
                    changes["modified"] += 1

                    healed.append(HealthIssue(
                        issue_id=generate_id("missing_dep"),
                        issue_type="missing_dependencies",
                        severity="low",
                        description=f"Added missing dependency from {sp.id} to {dep_id}",
                        affected_sub_problems=[sp.id],
                        healable=True,
                        healing_strategy="add_dependency",
                        healing_confidence=0.6,
                        quality_impact=0.1,
                        urgency="eventually"
                    ))

        return healed, failed, changes

    def heal_invalid_references(
        self,
        plan: DecompositionPlan,
        issues: List[HealthIssue]
    ) -> Tuple[List[HealthIssue], List[HealthIssue], Dict[str, Any]]:
        """
        Heal invalid references.

        Removes or updates references to non-existent sub-problems.

        Args:
            plan: Plan to heal
            issues: Invalid reference issues

        Returns:
            Tuple of (healed_issues, failed_issues, changes)
        """
        healed = []
        failed = []
        changes = {"modified": 0}

        valid_ids = {sp.id for sp in plan.sub_problems}

        for issue in issues:
            if not issue.affected_sub_problems:
                failed.append(issue)
                continue

            for sp_id in issue.affected_sub_problems:
                sp = next((s for s in plan.sub_problems if s.id == sp_id), None)
                if not sp:
                    failed.append(issue)
                    continue

                # Remove invalid dependencies
                original_deps = sp.dependencies.copy()
                sp.dependencies = [d for d in sp.dependencies if d in valid_ids]

                if len(sp.dependencies) < len(original_deps):
                    changes["modified"] += 1
                    healed.append(issue)
                else:
                    failed.append(issue)

        return healed, failed, changes

    def heal_vague_sub_problems(
        self,
        plan: DecompositionPlan,
        issues: List[HealthIssue]
    ) -> Tuple[List[HealthIssue], List[HealthIssue], Dict[str, Any]]:
        """
        Heal vague sub-problems.

        Enhances descriptions with more detail and clarity.

        Args:
            plan: Plan to heal
            issues: Vague sub-problem issues

        Returns:
            Tuple of (healed_issues, failed_issues, changes)
        """
        healed = []
        failed = []
        changes = {"modified": 0}

        for issue in issues:
            if not issue.affected_sub_problems:
                failed.append(issue)
                continue

            for sp_id in issue.affected_sub_problems:
                sp = next((s for s in plan.sub_problems if s.id == sp_id), None)
                if not sp:
                    failed.append(issue)
                    continue

                # Enhance description
                enhanced = self._enhance_sub_problem_description(sp)

                if enhanced:
                    changes["modified"] += 1
                    healed.append(issue)
                else:
                    failed.append(issue)

        return healed, failed, changes

    def heal_orphan_sub_problems(
        self,
        plan: DecompositionPlan,
        issues: List[HealthIssue]
    ) -> Tuple[List[HealthIssue], List[HealthIssue], Dict[str, Any]]:
        """
        Heal orphan sub-problems.

        Adds dependencies to connect orphans to the plan.

        Args:
            plan: Plan to heal
            issues: Orphan sub-problem issues

        Returns:
            Tuple of (healed_issues, failed_issues, changes)
        """
        healed = []
        failed = []
        changes = {"deps_added": [], "modified": 0}

        for issue in issues:
            if not issue.affected_sub_problems:
                failed.append(issue)
                continue

            for orphan_id in issue.affected_sub_problems:
                orphan = next((s for s in plan.sub_problems if s.id == orphan_id), None)
                if not orphan:
                    failed.append(issue)
                    continue

                # Find a suitable parent/dependency
                parent = self._find_suitable_parent(orphan, plan.sub_problems)

                if parent:
                    orphan.dependencies.append(parent.id)
                    changes["deps_added"].append((orphan.id, parent.id))
                    changes["modified"] += 1
                    healed.append(issue)
                else:
                    failed.append(issue)

        return healed, failed, changes

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _detect_circular_dependencies(self, plan: DecompositionPlan) -> List[HealthIssue]:
        """Detect circular dependencies in the plan."""
        issues = []

        # Build dependency graph
        graph = {sp.id: sp.dependencies for sp in plan.sub_problems}

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def detect_cycles(node, path, cycles):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    detect_cycles(neighbor, path, cycles)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        cycles = []
        for node_id in graph:
            if node_id not in visited:
                detect_cycles(node_id, [], cycles)

        # Create health issues for each cycle
        for cycle in cycles:
            issues.append(HealthIssue(
                issue_id=generate_id("cycle"),
                issue_type="circular_dependency",
                severity="critical",
                description=f"Circular dependency detected: {' -> '.join(cycle)}",
                affected_sub_problems=cycle,
                healable=True,
                healing_strategy="remove_edge",
                healing_confidence=0.9,
                quality_impact=0.5,
                urgency="immediate"
            ))

        return issues

    def _detect_complexity_imbalance(self, plan: DecompositionPlan) -> List[HealthIssue]:
        """Detect complexity imbalance in sub-problems."""
        issues = []

        if not plan.sub_problems:
            return issues

        complexities = [sp.complexity_score.overall_complexity for sp in plan.sub_problems]
        avg_complexity = sum(complexities) / len(complexities)

        # Check for sub-problems too far from average
        for sp in plan.sub_problems:
            diff = abs(sp.complexity_score.overall_complexity - avg_complexity)
            if diff > avg_complexity * 0.5:  # More than 50% deviation
                issues.append(HealthIssue(
                    issue_id=generate_id("complexity"),
                    issue_type="complexity_imbalance",
                    severity="medium",
                    description=f"Sub-problem complexity ({sp.complexity_score.overall_complexity}) "
                                f"deviates significantly from average ({avg_complexity:.1f})",
                    affected_sub_problems=[sp.id],
                    healable=True,
                    healing_strategy="split" if sp.complexity_score.overall_complexity > avg_complexity else "merge",
                    healing_confidence=0.7,
                    quality_impact=0.2,
                    urgency="soon"
                ))

        return issues

    def _detect_missing_dependencies(self, plan: DecompositionPlan) -> List[HealthIssue]:
        """Detect potentially missing dependencies."""
        issues = []

        # Build keyword map
        keyword_map = self._build_keyword_map(plan.sub_problems)

        # Check for missing keyword-based dependencies
        for sp in plan.sub_problems:
            potential_deps = self._find_potential_dependencies(sp, plan.sub_problems, keyword_map)

            for dep_id in potential_deps:
                if dep_id not in sp.dependencies and dep_id != sp.id:
                    issues.append(HealthIssue(
                        issue_id=generate_id("missing_dep"),
                        issue_type="missing_dependencies",
                        severity="low",
                        description=f"Potential missing dependency from {sp.title} to sub-problem {dep_id}",
                        affected_sub_problems=[sp.id],
                        healable=True,
                        healing_strategy="add_dependency",
                        healing_confidence=0.6,
                        quality_impact=0.1,
                        urgency="eventually"
                    ))

        return issues

    def _detect_inconsistent_quality(
        self,
        plan: DecompositionPlan,
        quality: EnhancedQualityScores
    ) -> List[HealthIssue]:
        """Detect inconsistent quality across sub-problems."""
        issues = []

        # Check for very low quality dimensions
        low_dimensions = quality.get_lowest_dimensions(threshold=0.5)

        for dimension in low_dimensions:
            issues.append(HealthIssue(
                issue_id=generate_id("quality"),
                issue_type="inconsistent_quality",
                severity="high",
                description=f"Low quality score in {dimension}: "
                            f"{quality.get_dimension_summary()[dimension]:.2f}",
                affected_sub_problems=[sp.id for sp in plan.sub_problems],
                healable=False,
                healing_strategy="manual_review",
                healing_confidence=0.3,
                quality_impact=0.4,
                urgency="soon"
            ))

        return issues

    def _detect_invalid_references(self, plan: DecompositionPlan) -> List[HealthIssue]:
        """Detect references to non-existent sub-problems."""
        issues = []
        valid_ids = {sp.id for sp in plan.sub_problems}

        for sp in plan.sub_problems:
            invalid_deps = [d for d in sp.dependencies if d not in valid_ids]
            if invalid_deps:
                issues.append(HealthIssue(
                    issue_id=generate_id("invalid_ref"),
                    issue_type="invalid_references",
                    severity="high",
                    description=f"Invalid dependencies: {invalid_deps}",
                    affected_sub_problems=[sp.id],
                    healable=True,
                    healing_strategy="remove_invalid",
                    healing_confidence=0.95,
                    quality_impact=0.3,
                    urgency="immediate"
                ))

        return issues

    def _detect_vague_sub_problems(self, plan: DecompositionPlan) -> List[HealthIssue]:
        """Detect sub-problems with vague descriptions."""
        issues = []

        for sp in plan.sub_problems:
            # Check description length
            if len(sp.description) < 50:
                issues.append(HealthIssue(
                    issue_id=generate_id("vague"),
                    issue_type="vague_sub_problems",
                    severity="medium",
                    description=f"Sub-problem description too short ({len(sp.description)} chars)",
                    affected_sub_problems=[sp.id],
                    healable=True,
                    healing_strategy="enhance_description",
                    healing_confidence=0.7,
                    quality_impact=0.2,
                    urgency="soon"
                ))

        return issues

    def _detect_orphan_sub_problems(self, plan: DecompositionPlan) -> List[HealthIssue]:
        """Detect sub-problems with no dependencies pointing to them."""
        issues = []

        if not plan.sub_problems:
            return issues

        # Count incoming dependencies
        incoming_count = defaultdict(int)
        for sp in plan.sub_problems:
            for dep in sp.dependencies:
                incoming_count[dep] += 1

        # Find sub-problems with no incoming dependencies (excluding first in execution order)
        for sp in plan.sub_problems:
            if incoming_count[sp.id] == 0 and len(plan.sub_problems) > 1:
                # Check if it's not the first in the plan
                if plan.sub_problems.index(sp) > 0:
                    issues.append(HealthIssue(
                        issue_id=generate_id("orphan"),
                        issue_type="orphan_sub_problems",
                        severity="low",
                        description=f"Sub-problem has no incoming dependencies",
                        affected_sub_problems=[sp.id],
                        healable=True,
                        healing_strategy="add_parent_dependency",
                        healing_confidence=0.6,
                        quality_impact=0.15,
                        urgency="eventually"
                    ))

        return issues

    def _find_best_edge_to_remove(
        self,
        plan: DecompositionPlan,
        cycle_nodes: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Find the best edge to remove from a cycle."""
        # Find edges in the cycle
        edges = []
        for sp in plan.sub_problems:
            if sp.id in cycle_nodes:
                for dep in sp.dependencies:
                    if dep in cycle_nodes:
                        edges.append((sp.id, dep))

        if not edges:
            return None

        # Prefer edge with lowest impact (simplest sub-problem)
        edges_with_complexity = []
        for from_id, to_id in edges:
            from_sp = next((s for s in plan.sub_problems if s.id == from_id), None)
            if from_sp:
                edges_with_complexity.append((
                    (from_id, to_id),
                    from_sp.complexity_score.overall_complexity
                ))

        if edges_with_complexity:
            # Return edge with lowest complexity
            edges_with_complexity.sort(key=lambda x: x[1])
            return edges_with_complexity[0][0]

        return edges[0] if edges else None

    def _split_sub_problem(self, plan: DecompositionPlan, sp: SubProblem) -> bool:
        """Split a complex sub-problem into two."""
        # Create new sub-problem with half the complexity
        new_sp = SubProblem(
            id=generate_id("sub"),
            parent_id=sp.parent_id,
            title=f"{sp.title} (Part 2)",
            description=f"Additional aspects of: {sp.description}",
            type=sp.type,
            complexity_score=ComplexityScore(
                explanation="Split from parent sub-problem",
                cognitive_complexity=sp.complexity_score.cognitive_complexity / 2,
                computational_complexity=sp.complexity_score.computational_complexity / 2,
                domain_complexity=sp.complexity_score.domain_complexity,
                integration_complexity=sp.complexity_score.integration_complexity / 2,
                overall_complexity=sp.complexity_score.overall_complexity / 2
            ),
            dependencies=[sp.id],
            success_criteria=sp.success_criteria.copy(),
            estimated_effort=sp.estimated_effort // 2,
            priority=sp.priority,
            status=sp.status
        )

        # Adjust original complexity
        sp.complexity_score = ComplexityScore(
            explanation=f"Split (original: {sp.complexity_score.explanation})",
            cognitive_complexity=sp.complexity_score.cognitive_complexity / 2,
            computational_complexity=sp.complexity_score.computational_complexity / 2,
            domain_complexity=sp.complexity_score.domain_complexity,
            integration_complexity=sp.complexity_score.integration_complexity / 2,
            overall_complexity=sp.complexity_score.overall_complexity / 2
        )

        # Add to plan
        plan.sub_problems.append(new_sp)
        return True

    def _merge_sub_problems(
        self,
        plan: DecompositionPlan,
        sp1: SubProblem,
        sp2: SubProblem
    ) -> bool:
        """Merge two simple sub-problems."""
        # Update sp1 to include content from sp2
        sp1.title = f"{sp1.title} & {sp2.title}"
        sp1.description = f"{sp1.description}\n\nAlso covers: {sp2.description}"
        sp1.estimated_effort += sp2.estimated_effort

        # Update complexity
        sp1.complexity_score = ComplexityScore(
            explanation="Merged from two sub-problems",
            cognitive_complexity=(sp1.complexity_score.cognitive_complexity +
                                 sp2.complexity_score.cognitive_complexity),
            computational_complexity=(sp1.complexity_score.computational_complexity +
                                     sp2.complexity_score.computational_complexity),
            domain_complexity=max(sp1.complexity_score.domain_complexity,
                                 sp2.complexity_score.domain_complexity),
            integration_complexity=(sp1.complexity_score.integration_complexity +
                                   sp2.complexity_score.integration_complexity),
            overall_complexity=min(10.0, (sp1.complexity_score.overall_complexity +
                                        sp2.complexity_score.overall_complexity))
        )

        # Remove sp2 from plan
        plan.sub_problems = [sp for sp in plan.sub_problems if sp.id != sp2.id]

        # Update dependencies that pointed to sp2
        for sp in plan.sub_problems:
            if sp2.id in sp.dependencies:
                sp.dependencies.remove(sp2.id)
                if sp1.id not in sp.dependencies:
                    sp.dependencies.append(sp1.id)

        return True

    def _build_keyword_map(self, sub_problems: List[SubProblem]) -> Dict[str, str]:
        """Build keyword to sub-problem ID mapping."""
        keyword_map = {}

        for sp in sub_problems:
            # Extract keywords from title and description
            words = (sp.title + " " + sp.description).lower().split()
            for word in words:
                if len(word) > 4:  # Only significant words
                    keyword_map[word] = sp.id

        return keyword_map

    def _find_potential_dependencies(
        self,
        sp: SubProblem,
        all_sub_problems: List[SubProblem],
        keyword_map: Dict[str, str]
    ) -> List[str]:
        """Find potential dependencies based on keyword overlap."""
        text = (sp.title + " " + sp.description).lower()
        words = set(text.split())

        potential_deps = set()
        for word in words:
            if word in keyword_map and keyword_map[word] != sp.id:
                potential_deps.add(keyword_map[word])

        return list(potential_deps)

    def _enhance_sub_problem_description(self, sp: SubProblem) -> bool:
        """Enhance a vague sub-problem description."""
        if len(sp.description) < 50:
            # Add more detail
            sp.description += f"\n\nThis sub-problem requires: "
            sp.description += f"\n- Clear analysis and approach\n- "
            sp.description += f"Detailed implementation plan\n- "
            sp.description += f"Validation and testing strategy"
            return True
        return False

    def _find_suitable_parent(
        self,
        orphan: SubProblem,
        all_sub_problems: List[SubProblem]
    ) -> Optional[SubProblem]:
        """Find a suitable parent for an orphan sub-problem."""
        # Prefer sub-problems with similar keywords
        orphan_text = (orphan.title + " " + orphan.description).lower()

        best_match = None
        best_score = 0

        for sp in all_sub_problems:
            if sp.id == orphan.id:
                continue

            sp_text = (sp.title + " " + sp.description).lower()

            # Calculate word overlap
            orphan_words = set(orphan_text.split())
            sp_words = set(sp_text.split())
            overlap = len(orphan_words & sp_words) / len(orphan_words | sp_words) if orphan_words | sp_words else 0

            if overlap > best_score and overlap > 0.1:
                best_score = overlap
                best_match = sp

        return best_match

    def _group_issues_by_type(self, issues: List[HealthIssue]) -> Dict[str, List[HealthIssue]]:
        """Group issues by type for batch processing."""
        grouped = defaultdict(list)
        for issue in issues:
            grouped[issue.issue_type].append(issue)
        return grouped

    def _calculate_plan_quality(self, plan: DecompositionPlan) -> float:
        """Calculate overall plan quality score."""
        if plan.enhanced_quality_scores:
            return plan.enhanced_quality_scores.overall_score
        elif plan.quality_scores:
            return plan.quality_scores.overall_score
        else:
            # Calculate basic quality
            if not plan.sub_problems:
                return 0.5

            avg_complexity = sum(sp.complexity_score.overall_complexity
                               for sp in plan.sub_problems) / len(plan.sub_problems)
            # Normalize to 0-1
            return min(1.0, avg_complexity / 10.0)

    def _update_healing_statistics(self, result: HealingResult):
        """Update healing statistics with result."""
        for method in result.healing_methods_used:
            if method not in self.healing_statistics:
                self.healing_statistics[method] = {
                    "attempts": 0,
                    "successes": 0,
                    "avg_improvement": 0.0
                }

            stats = self.healing_statistics[method]
            stats["attempts"] += 1
            if result.quality_improvement > 0:
                stats["successes"] += 1

            # Update average improvement
            current_avg = stats["avg_improvement"]
            n = stats["attempts"]
            stats["avg_improvement"] = (current_avg * (n - 1) + result.quality_improvement) / n

    def get_healing_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get healing statistics."""
        return dict(self.healing_statistics)

    def get_healing_summary(self) -> Dict[str, Any]:
        """Get summary of healing operations."""
        if not self.healing_history:
            return {
                "total_healings": 0,
                "avg_quality_improvement": 0.0,
                "success_rate": 0.0
            }

        total_healings = len(self.healing_history)
        avg_improvement = sum(h.quality_improvement for h in self.healing_history) / total_healings
        avg_success_rate = sum(h.healing_success_rate for h in self.healing_history) / total_healings

        return {
            "total_healings": total_healings,
            "avg_quality_improvement": avg_improvement,
            "success_rate": avg_success_rate,
            "most_common_issues": self._get_most_common_issues(),
            "most_successful_methods": self._get_most_successful_methods()
        }

    def _get_most_common_issues(self) -> List[Tuple[str, int]]:
        """Get most common issue types."""
        issue_counts = defaultdict(int)
        for result in self.healing_history:
            for issue in result.original_issues:
                issue_counts[issue.issue_type] += 1

        return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    def _get_most_successful_methods(self) -> List[Tuple[str, float]]:
        """Get most successful healing methods."""
        method_success = {}
        for method, stats in self.healing_statistics.items():
            if stats["attempts"] > 0:
                method_success[method] = stats["successes"] / stats["attempts"]

        return sorted(method_success.items(), key=lambda x: x[1], reverse=True)[:5]
