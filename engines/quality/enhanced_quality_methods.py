"""
Enhanced quality assessment methods to be appended to decomposition_engine.py
"""
from __future__ import annotations


METHODS_CODE = '''

    def _assess_quality_enhanced(self,
                                problem: ProblemDefinition,
                                sub_problems: List[SubProblem]) -> 'EnhancedQualityScores':
        """
        Enhanced multi-dimensional quality assessment.

        Assesses 5 dimensions:
        - Completeness: All aspects addressed
        - Consistency: No contradictions, stakeholder alignment
        - Feasibility: Realistic with resources
        - Dependencies: Valid, no cycles
        - Balance: Complexity/evenness distributed

        Returns EnhancedQualityScores with detailed breakdowns and recommendations.
        """
        from sovereign_data_models import EnhancedQualityScores
        from datetime import datetime

        if not sub_problems:
            return EnhancedQualityScores(
                overall_score=0.0,
                meets_thresholds=False,
                completeness_score=0.0,
                consistency_score=0.0,
                feasibility_score=0.0,
                dependency_score=0.0,
                balance_score=0.0,
                completeness_details={"error": "No sub-problems generated"},
                consistency_details={"error": "No sub-problems generated"},
                feasibility_details={"error": "No sub-problems generated"},
                dependency_details={"error": "No sub-problems generated"},
                balance_details={"error": "No sub-problems generated"},
                improvement_recommendations=["Generate sub-problems first"],
                critical_issues=["No sub-problems to assess"],
                validation_checkpoints=[],
                timestamp=datetime.now()
            )

        # Assess each dimension
        completeness_score, completeness_details = self._assess_completeness(problem, sub_problems)
        consistency_score, consistency_details = self._assess_consistency(problem, sub_problems)
        feasibility_score, feasibility_details = self._assess_feasibility(problem, sub_problems)
        dependency_score, dependency_details = self._assess_dependency_validity(problem, sub_problems)
        balance_score, balance_details = self._assess_balance(problem, sub_problems)

        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'feasibility': 0.25,
            'dependency': 0.15,
            'balance': 0.15
        }

        overall_score = (
            completeness_score * weights['completeness'] +
            consistency_score * weights['consistency'] +
            feasibility_score * weights['feasibility'] +
            dependency_score * weights['dependency'] +
            balance_score * weights['balance']
        )

        # Check if meets thresholds
        threshold = 0.7  # Configurable
        meets_thresholds = overall_score >= threshold

        # Collect recommendations and issues
        improvement_recommendations = []
        critical_issues = []

        for name, score_val, details in [
            ("Completeness", completeness_score, completeness_details),
            ("Consistency", consistency_score, consistency_details),
            ("Feasibility", feasibility_score, feasibility_details),
            ("Dependency", dependency_score, dependency_details),
            ("Balance", balance_score, balance_details)
        ]:
            if score_val < threshold:
                improvement_recommendations.extend(details.get('recommendations', []))
                if score_val < 0.5:
                    critical_issues.append(f"{name} score critically low ({score_val:.2f})")

        # Validation checkpoints
        validation_checkpoints = [
            f"Completeness assessment: {completeness_score:.2f}",
            f"Consistency assessment: {consistency_score:.2f}",
            f"Feasibility assessment: {feasibility_score:.2f}",
            f"Dependency validation: {dependency_score:.2f}",
            f"Balance assessment: {balance_score:.2f}",
            f"Overall quality: {overall_score:.2f}"
        ]

        return EnhancedQualityScores(
            overall_score=round(overall_score, 3),
            meets_thresholds=meets_thresholds,
            completeness_score=round(completeness_score, 3),
            consistency_score=round(consistency_score, 3),
            feasibility_score=round(feasibility_score, 3),
            dependency_score=round(dependency_score, 3),
            balance_score=round(balance_score, 3),
            completeness_details=completeness_details,
            consistency_details=consistency_details,
            feasibility_details=feasibility_details,
            dependency_details=dependency_details,
            balance_details=balance_details,
            improvement_recommendations=improvement_recommendations[:10],  # Top 10
            critical_issues=critical_issues[:5],  # Top 5
            validation_checkpoints=validation_checkpoints,
            timestamp=datetime.now()
        )

    def _assess_completeness(self,
                            problem: ProblemDefinition,
                            sub_problems: List[SubProblem]) -> Tuple[float, Dict[str, Any]]:
        """
        Assess if decomposition covers all problem aspects.

        Checks:
        - All success criteria addressed
        - All stakeholder needs covered
        - No missing critical components
        - Alignment with problem scope

        Returns: (score, details_dict)
        """
        details = {
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }

        score = 1.0
        sub_problem_count = len(sub_problems)

        # Check 1: Success criteria coverage
        if problem.success_criteria:
            criteria_coverage = set()
            for sp in sub_problems:
                # Extract keywords from success criteria
                for sc in problem.success_criteria:
                    criteria_keywords = set(sc.description.lower().split())
                    sp_keywords = set(sp.description.lower().split())
                    if criteria_keywords & sp_keywords:  # Intersection
                        criteria_coverage.add(sc.description)

            coverage_ratio = len(criteria_coverage) / max(1, len(problem.success_criteria))
            details['checks_performed'].append(f"Success criteria coverage: {coverage_ratio:.2%}")
            if coverage_ratio < 0.8:
                score -= 0.15 * (1.0 - coverage_ratio)
                details['issues_found'].append(f"Only {coverage_ratio:.0%} of success criteria covered")
                details['recommendations'].append("Ensure all success criteria are explicitly addressed in sub-problems")

        # Check 2: Problem complexity alignment
        complexity = problem.complexity_score.overall_complexity if problem.complexity_score else 5.0

        if complexity <= 3.0:
            expected_min, expected_max = 2, 4
        elif complexity <= 6.0:
            expected_min, expected_max = 3, 7
        else:
            expected_min, expected_max = 5, 12

        if sub_problem_count < expected_min:
            penalty = (expected_min - sub_problem_count) / expected_min * 0.3
            score -= penalty
            details['issues_found'].append(f"Too few sub-problems ({sub_problem_count} < {expected_min})")
            details['recommendations'].append(f"Decompose into at least {expected_min} sub-problems for this complexity")
        elif sub_problem_count > expected_max:
            penalty = (sub_problem_count - expected_max) / sub_problem_count * 0.2
            score -= penalty
            details['issues_found'].append(f"Too many sub-problems ({sub_problem_count} > {expected_max})")
            details['recommendations'].append(f"Consolidate to {expected_max} or fewer sub-problems")

        details['checks_performed'].append(f"Sub-problem count: {sub_problem_count} (expected: {expected_min}-{expected_max})")

        # Check 3: Stakeholder needs coverage
        if problem.stakeholders:
            stakeholder_coverage = set()
            for sp in sub_problems:
                for stakeholder in problem.stakeholders:
                    if stakeholder.lower() in sp.description.lower():
                        stakeholder_coverage.add(stakeholder)

            coverage_ratio = len(stakeholder_coverage) / max(1, len(problem.stakeholders))
            details['checks_performed'].append(f"Stakeholder coverage: {coverage_ratio:.2%}")
            if coverage_ratio < 0.7:
                score -= 0.1 * (1.0 - coverage_ratio)
                details['issues_found'].append(f"Only {coverage_ratio:.0%} of stakeholders addressed")
                details['recommendations'].append("Ensure all stakeholders' needs are considered")

        # Check 4: Empty or minimal descriptions
        empty_count = sum(1 for sp in sub_problems if not sp.description or len(sp.description.strip()) < 20)
        if empty_count > 0:
            penalty = (empty_count / sub_problem_count) * 0.3
            score -= penalty
            details['issues_found'].append(f"{empty_count} sub-problems have minimal descriptions")
            details['recommendations'].append("Provide detailed descriptions for all sub-problems")

        # Check 5: Domain coverage
        if problem.domain_context:
            domain_mentioned = sum(
                1 for sp in sub_problems
                if problem.domain_context.domain.lower() in sp.description.lower()
            )
            domain_coverage = domain_mentioned / sub_problem_count
            details['checks_performed'].append(f"Domain coverage: {domain_coverage:.2%}")
            if domain_coverage < 0.5:
                score -= 0.05
                details['recommendations'].append("Ensure domain context is reflected in sub-problems")

        score = max(0.0, min(1.0, score))
        details['final_score'] = round(score, 3)

        return score, details

    def _assess_consistency(self,
                           problem: ProblemDefinition,
                           sub_problems: List[SubProblem]) -> Tuple[float, Dict[str, Any]]:
        """
        Assess internal consistency of decomposition.

        Checks:
        - No contradictions between sub-problems
        - Aligned with stakeholder needs
        - Coherent terminology
        - Consistent approach

        Returns: (score, details_dict)
        """
        details = {
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }

        score = 1.0

        # Check 1: Terminology consistency
        all_terms = set()
        for sp in sub_problems:
            words = re.findall(r'\\b[A-Z][a-z]+\\b', sp.description)  # Capitalized terms
            all_terms.update(words)

        # Check for conflicting approaches
        approaches = []
        for sp in sub_problems:
            if 'implement' in sp.description.lower():
                approaches.append('implementation')
            if 'analyze' in sp.description.lower():
                approaches.append('analysis')
            if 'research' in sp.description.lower():
                approaches.append('research')
            if 'design' in sp.description.lower():
                approaches.append('design')

        if len(set(approaches)) > 4:
            score -= 0.1
            details['issues_found'].append("Mixed approaches may indicate inconsistency")
            details['recommendations'].append("Ensure sub-problems follow a coherent methodology")

        details['checks_performed'].append(f"Approaches identified: {set(approaches)}")

        # Check 2: Contradiction detection (simple heuristic)
        # Look for antonym pairs
        antonym_pairs = [
            ('increase', 'decrease'),
            ('add', 'remove'),
            ('create', 'delete'),
            ('enable', 'disable'),
            ('optimize', 'simplify')
        ]

        for antonym1, antonym2 in antonym_pairs:
            has_antonym1 = any(antonym1 in sp.description.lower() for sp in sub_problems)
            has_antonym2 = any(antonym2 in sp.description.lower() for sp in sub_problems)
            if has_antonym1 and has_antonym2:
                # This might be OK if in different sub-problems, but flag it
                details['checks_performed'].append(f"Found antonyms: {antonym1}/{antonym2}")

        # Check 3: Stakeholder alignment
        if problem.stakeholders:
            misaligned = []
            for sp in sub_problems:
                mentions_stakeholder = any(
                    stakeholder.lower() in sp.description.lower()
                    for stakeholder in problem.stakeholders
                )
                if not mentions_stakeholder and sp.priority >= 8:
                    misaligned.append(sp.id)

            if misaligned:
                penalty = len(misaligned) / len(sub_problems) * 0.2
                score -= penalty
                details['issues_found'].append(f"{len(misaligned)} high-priority sub-problems don't mention stakeholders")
                details['recommendations'].append("Align high-priority sub-problems with stakeholder needs")

        # Check 4: Constraint consistency
        for constraint in problem.constraints[:3]:  # Check first 3 constraints
            constraint_addressed = any(
                constraint.description.lower() in sp.description.lower() or
                any(sp_constraint.description.lower() in constraint.description.lower()
                    for sp_constraint in sp.constraints)
                for sp in sub_problems
            )
            if not constraint_addressed:
                score -= 0.05
                details['issues_found'].append(f"Constraint not addressed: {constraint.description[:50]}...")

        # Check 5: Priority distribution consistency
        priorities = [sp.priority for sp in sub_problems]
        if priorities:
            avg_priority = sum(priorities) / len(priorities)
            if avg_priority > 8:
                details['recommendations'].append("Most sub-problems marked high priority - consider re-prioritizing")
            elif avg_priority < 3:
                details['recommendations'].append("Most sub-problems marked low priority - review priorities")

        details['checks_performed'].append(f"Average priority: {avg_priority:.1f}")

        # Check 6: Type distribution
        type_counts = {}
        for sp in sub_problems:
            sp_type = sp.type.value if hasattr(sp.type, 'value') else str(sp.type)
            type_counts[sp_type] = type_counts.get(sp_type, 0) + 1

        if len(type_counts) == 1:
            details['recommendations'].append("All sub-problems are same type - consider variety")
        elif len(type_counts) > 5:
            details['recommendations'].append("Too many different types - consider consolidation")

        details['checks_performed'].append(f"Type distribution: {type_counts}")

        score = max(0.0, min(1.0, score))
        details['final_score'] = round(score, 3)

        return score, details

    def _assess_feasibility(self,
                           problem: ProblemDefinition,
                           sub_problems: List[SubProblem]) -> Tuple[float, Dict[str, Any]]:
        """
        Assess if decomposition is feasible with resources.

        Checks:
        - Time estimates realistic
        - Technical feasibility
        - Resource availability
        - Budget constraints

        Returns: (score, details_dict)
        """
        details = {
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }

        score = 1.0

        # Check 1: Complexity vs resources
        complexity_scores = [
            sp.complexity_score.overall_complexity if sp.complexity_score else 5
            for sp in sub_problems
        ]
        avg_complexity = sum(complexity_scores) / max(1, len(complexity_scores))
        max_complexity = max(complexity_scores) if complexity_scores else 5

        details['checks_performed'].append(f"Average complexity: {avg_complexity:.1f}/10")
        details['checks_performed'].append(f"Max complexity: {max_complexity:.1f}/10")

        # Check against resource constraints
        max_allowed = problem.resources_available.get("max_subproblem_complexity") or \\
                     problem.resources_available.get("max_complexity", 10)

        if max_complexity > max_allowed:
            penalty = (max_complexity - max_allowed) / max_allowed * 0.3
            score -= penalty
            details['issues_found'].append(f"Sub-problem complexity ({max_complexity:.1f}) exceeds limit ({max_allowed})")
            details['recommendations'].append(f"Break down sub-problems with complexity > {max_allowed}")

        # Check 2: Effort estimates
        efforts = [sp.effort_hours for sp in sub_problems if sp.effort_hours]
        if efforts:
            total_effort = sum(efforts)
            avg_effort = total_effort / len(efforts)

            details['checks_performed'].append(f"Total effort: {total_effort:.0f} hours")
            details['checks_performed'].append(f"Average effort: {avg_effort:.0f} hours")

            # Check if any single sub-problem is too large
            max_effort = max(efforts)
            if max_effort > 80:  # More than 2 weeks
                score -= 0.15
                details['issues_found'].append(f"Sub-problem requires {max_effort:.0f} hours - too large")
                details['recommendations'].append("Break down large sub-problems (>80 hours)")

            # Check if total effort is reasonable
            max_total_effort = problem.resources_available.get("max_total_effort", 1000)
            if total_effort > max_total_effort:
                penalty = (total_effort - max_total_effort) / max_total_effort * 0.2
                score -= penalty
                details['issues_found'].append(f"Total effort ({total_effort:.0f}h) exceeds limit ({max_total_effort}h)")
                details['recommendations'].append("Reduce scope or increase resources")

        # Check 3: Resource availability
        required_resources = set()
        for sp in sub_problems:
            if sp.required_skills:
                required_resources.update(sp.required_skills)

        available_resources = set(problem.resources_available.get("skills", []))
        missing_resources = required_resources - available_resources

        if missing_resources:
            penalty = len(missing_resources) / max(1, len(required_resources)) * 0.2
            score -= penalty
            details['issues_found'].append(f"Missing required skills: {missing_resources}")
            details['recommendations'].append("Ensure required skills are available or plan for training")

        details['checks_performed'].append(f"Required skills: {required_resources}")

        # Check 4: Technical feasibility indicators
        low_feasibility = 0
        for sp in sub_problems:
            desc_lower = sp.description.lower()

            # Red flags for feasibility
            if any(term in desc_lower for term in ['revolutionary', 'paradigm shift', 'breakthrough']):
                low_feasibility += 1

            # Green flags for feasibility
            if any(term in desc_lower for term in ['prototype', 'mvp', 'poc', 'pilot']):
                # This is good - incremental approach
                pass

        if low_feasibility > 0:
            penalty = (low_feasibility / len(sub_problems)) * 0.1
            score -= penalty
            details['issues_found'].append(f"{low_feasibility} sub-problems may be technically ambitious")
            details['recommendations'].append("Consider incremental/iterative approaches for ambitious goals")

        # Check 5: Constraint realism
        for constraint in problem.constraints:
            if hasattr(constraint, 'severity') and constraint.severity and hasattr(constraint.severity, 'value'):
                if constraint.severity.value == 'critical':
                    # Critical constraints should have realistic implementation
                    details['checks_performed'].append(f"Critical constraint: {constraint.description[:50]}...")

        score = max(0.0, min(1.0, score))
        details['final_score'] = round(score, 3)

        return score, details

    def _assess_dependency_validity(self,
                                   problem: ProblemDefinition,
                                   sub_problems: List[SubProblem]) -> Tuple[float, Dict[str, Any]]:
        """
        Assess if dependencies are valid.

        Checks:
        - No circular dependencies
        - Dependency references exist
        - Critical path reasonable
        - Execution order logical

        Returns: (score, details_dict)
        """
        details = {
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }

        score = 1.0

        if not sub_problems:
            return 0.0, details

        # Check 1: Build dependency graph
        sub_problem_ids = {sp.id for sp in sub_problems}
        total_dependencies = sum(len(sp.dependencies) for sp in sub_problems)

        details['checks_performed'].append(f"Total dependencies: {total_dependencies}")
        details['checks_performed'].append(f"Sub-problems: {len(sub_problems)}")

        # Check 2: Missing dependencies
        missing_deps = []
        for sp in sub_problems:
            for dep in sp.dependencies:
                if dep not in sub_problem_ids:
                    missing_deps.append((sp.id, dep))

        if missing_deps:
            penalty = len(missing_deps) / max(1, total_dependencies) * 0.4
            score -= penalty
            details['issues_found'].append(f"{len(missing_deps)} missing dependency references")
            details['recommendations'].append("Fix all missing dependency references")

        # Check 3: Self-dependencies
        self_deps = [sp.id for sp in sub_problems if sp.id in sp.dependencies]
        if self_deps:
            penalty = len(self_deps) / len(sub_problems) * 0.3
            score -= penalty
            details['issues_found'].append(f"{len(self_deps)} sub-problems depend on themselves")
            details['recommendations'].append("Remove self-dependencies")

        # Check 4: Circular dependencies
        execution_order = self._topological_sort(sub_problems)
        has_cycles = len(execution_order) < len(sub_problems)

        if has_cycles:
            score -= 0.4
            details['issues_found'].append("Circular dependencies detected")
            details['recommendations'].append("Break circular dependencies to enable execution")

        # Check 5: Dependency density (should be balanced)
        if len(sub_problems) > 1:
            max_possible_deps = len(sub_problems) * (len(sub_problems) - 1)
            dependency_density = total_dependencies / max_possible_deps if max_possible_deps > 0 else 0

            details['checks_performed'].append(f"Dependency density: {dependency_density:.2%}")

            if dependency_density > 0.7:
                score -= 0.1
                details['issues_found'].append("Very high dependency density - may indicate over-coupling")
                details['recommendations'].append("Consider reducing dependencies for better parallelization")
            elif dependency_density < 0.1 and len(sub_problems) > 3:
                details['recommendations'].append("Very low dependency density - consider if sub-problems are coordinated")

        # Check 6: Critical path analysis
        if total_dependencies > 0 and not has_cycles:
            # Identify longest path (simplified)
            levels = {}
            for sp_id in execution_order:
                # Count how many dependencies this sub-problem has (directly + indirectly)
                sp = next(sp for sp in sub_problems if sp.id == sp_id)
                level = 0
                to_check = list(sp.dependencies)
                checked = set()

                while to_check:
                    dep_id = to_check.pop(0)
                    if dep_id in checked:
                        continue
                    checked.add(dep_id)

                    level += 1
                    dep_sp = next((sp for sp in sub_problems if sp.id == dep_id), None)
                    if dep_sp:
                        to_check.extend(dep_sp.dependencies)

                levels[sp_id] = level

            max_level = max(levels.values()) if levels else 0
            details['checks_performed'].append(f"Max dependency depth: {max_level}")

            if max_level > len(sub_problems) / 2:
                details['recommendations'].append("Deep dependency chain - may slow down execution")

        # Check 7: Orphan sub-problems (no dependencies and no dependents)
        if len(sub_problems) > 1:
            isolated = []
            for sp in sub_problems:
                # No dependencies
                no_deps = len(sp.dependencies) == 0
                # Nothing depends on this
                no_dependents = all(sp.id not in other_sp.dependencies for other_sp in sub_problems)

                if no_deps and no_dependents and len(sub_problems) > 1:
                    isolated.append(sp.id)

            if isolated:
                details['checks_performed'].append(f"Isolated sub-problems: {len(isolated)}")
                if len(isolated) > len(sub_problems) / 2:
                    details['recommendations'].append("Many isolated sub-problems - consider if they form a coherent solution")

        score = max(0.0, min(1.0, score))
        details['final_score'] = round(score, 3)

        return score, details

    def _assess_balance(self,
                       problem: ProblemDefinition,
                       sub_problems: List[SubProblem]) -> Tuple[float, Dict[str, Any]]:
        """
        Assess if complexity/effect is evenly distributed.

        Checks:
        - No sub-problem >30% complexity
        - Effort distributed
        - Risk distributed
        - Value distributed

        Returns: (score, details_dict)
        """
        details = {
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }

        score = 1.0

        if not sub_problems:
            return 0.0, details

        # Check 1: Complexity distribution
        complexity_scores = [
            sp.complexity_score.overall_complexity if sp.complexity_score else 5
            for sp in sub_problems
        ]
        total_complexity = sum(complexity_scores)

        if total_complexity > 0:
            complexity_ratios = [cs / total_complexity for cs in complexity_scores]
            max_complexity_ratio = max(complexity_ratios)

            details['checks_performed'].append(f"Max complexity ratio: {max_complexity_ratio:.2%}")

            # Check if any sub-problem > 30% of total complexity
            if max_complexity_ratio > 0.30:
                penalty = (max_complexity_ratio - 0.30) / 0.30 * 0.3
                score -= penalty
                details['issues_found'].append(f"Sub-problem with {max_complexity_ratio:.1%} of complexity (limit: 30%)")
                details['recommendations'].append("Redistribute complexity more evenly across sub-problems")

        # Check 2: Effort distribution
        efforts = [sp.effort_hours for sp in sub_problems if sp.effort_hours]
        if efforts:
            total_effort = sum(efforts)
            effort_ratios = [e / total_effort for e in efforts]
            max_effort_ratio = max(effort_ratios)

            details['checks_performed'].append(f"Max effort ratio: {max_effort_ratio:.2%}")

            if max_effort_ratio > 0.40:
                penalty = (max_effort_ratio - 0.40) / 0.40 * 0.2
                score -= penalty
                details['issues_found'].append(f"Sub-problem with {max_effort_ratio:.1%} of effort")
                details['recommendations'].append("Balance effort more evenly across sub-problems")

            # Check coefficient of variation (standard deviation / mean)
            if len(efforts) > 1:
                mean_effort = sum(efforts) / len(efforts)
                variance = sum((e - mean_effort) ** 2 for e in efforts) / len(efforts)
                std_dev = variance ** 0.5
                cv = std_dev / mean_effort if mean_effort > 0 else 0

                details['checks_performed'].append(f"Effort variation (CV): {cv:.2f}")

                if cv > 1.0:  # High variation
                    score -= 0.1
                    details['issues_found'].append(f"High effort variation (CV={cv:.2f})")
                    details['recommendations'].append("Standardize sub-problem sizes for better predictability")

        # Check 3: Priority distribution
        priorities = [sp.priority for sp in sub_problems]
        if priorities:
            high_priority_count = sum(1 for p in priorities if p >= 8)
            low_priority_count = sum(1 for p in priorities if p <= 3)

            details['checks_performed'].append(f"High priority: {high_priority_count}/{len(priorities)}")
            details['checks_performed'].append(f"Low priority: {low_priority_count}/{len(priorities)}")

            if high_priority_count == len(priorities):
                details['recommendations'].append("All sub-problems marked high priority - differentiation needed")
            elif low_priority_count == len(priorities):
                details['recommendations'].append("All sub-problems marked low priority - review prioritization")

        # Check 4: Type balance
        type_counts = {}
        for sp in sub_problems:
            sp_type = sp.type.value if hasattr(sp.type, 'value') else str(sp.type)
            type_counts[sp_type] = type_counts.get(sp_type, 0) + 1

        if len(type_counts) > 0:
            max_type_count = max(type_counts.values())
            max_type_ratio = max_type_count / len(sub_problems)

            details['checks_performed'].append(f"Type distribution: {type_counts}")

            if max_type_ratio > 0.7 and len(sub_problems) > 2:
                details['recommendations'].append(f"Dominated by one type ({max_type_ratio:.0%} same type)")

        # Check 5: Risk distribution (based on constraints)
        constraint_counts = [len(sp.constraints) for sp in sub_problems]
        if constraint_counts:
            avg_constraints = sum(constraint_counts) / len(constraint_counts)
            max_constraints = max(constraint_counts)

            details['checks_performed'].append(f"Average constraints: {avg_constraints:.1f}")
            details['checks_performed'].append(f"Max constraints: {max_constraints}")

            if max_constraints > avg_constraints * 3:
                details['issues_found'].append(f"Sub-problem with {max_constraints} constraints (high complexity)")
                details['recommendations'].append("Consider simplifying over-constrained sub-problems")

        # Check 6: Description length balance
        desc_lengths = [len(sp.description) for sp in sub_problems]
        if desc_lengths:
            avg_length = sum(desc_lengths) / len(desc_lengths)
            max_length = max(desc_lengths)

            if max_length > avg_length * 3:
                details['recommendations'].append("High variation in description detail - standardize detail level")

        score = max(0.0, min(1.0, score))
        details['final_score'] = round(score, 3)

        return score, details
'''

if __name__ == "__main__":
    # Append to decomposition_engine.py
    with open("C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\decomposition_engine.py", "a") as f:
        f.write(METHODS_CODE)
    print("Enhanced quality assessment methods appended to decomposition_engine.py")
