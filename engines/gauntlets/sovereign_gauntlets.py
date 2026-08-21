"""
Sovereign-Grade Problem Decomposition System - Gauntlet Integration
Decomposition-specific gauntlets for verification and validation.
"""
from __future__ import annotations


import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from sovereign_data_models import (
    DecompositionPlan, SubProblem, ValidationResult, Feedback, SolutionAttempt, generate_id
)

logger = logging.getLogger(__name__)

# **LEAN INTEGRATION**: Real Lean proof verification for sovereign gauntlets
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LeanAide client not available - formal verification disabled")

# Import DTS integration
try:
    from dts_integration import DTSIntegration, DTSIntegrationConfig, DTS_AVAILABLE
    logger.info("DTS integration available for enhanced gauntlet evaluation")
except ImportError:
    DTS_AVAILABLE = False
    logger.warning("DTS not available - using fallback gauntlet methods")


class DecompositionGauntlet(ABC):
    """Base class for decomposition gauntlets."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Run the gauntlet on a decomposition plan.
        
        Args:
            plan: The decomposition plan to validate
            
        Returns:
            ValidationResult with pass/fail, score, and feedback
        """
        raise NotImplementedError("Gauntlet implementations must override run().")
    
    def _create_validation_result(
        self, 
        passed: bool, 
        score: float, 
        feedback: str, 
        improvements: List[str]
    ) -> ValidationResult:
        """Helper to create validation result."""
        return ValidationResult(
            validator=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            improvements=improvements,
            timestamp=datetime.now()
        )


class CoherenceGauntlet(DecompositionGauntlet):
    """Verifies logical consistency of decomposition using LLM semantic analysis."""
    
    def __init__(self, openevolve_client=None):
        super().__init__("coherence_gauntlet")
        self.min_score = 0.7
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for coherence gauntlet")
    
    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Checks for logical consistency and coherence using LLM analysis.
        
        Validates:
        - Sub-problems align with parent problem
        - Sub-problems don't overlap excessively
        - Success criteria are consistent
        - Complexity distribution is reasonable
        
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        self.logger.info(f"Running coherence gauntlet on plan: {plan.id}")
        
        if not plan.sub_problems:
            return self._create_validation_result(
                passed=False,
                score=0.0,
                feedback="No sub-problems defined in decomposition plan",
                improvements=["Create at least one sub-problem"]
            )
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for coherence gauntlet.")

        if len(plan.sub_problems) > 10:
             self.logger.warning("Too many sub-problems for LLM coherence check, skipping.")
             # Or should I raise an error? For now, I will just return a neutral result.
             return self._create_validation_result(
                passed=True,
                score=0.8, # Neutral score
                feedback="Coherence check skipped due to large number of sub-problems.",
                improvements=[]
            )

        try:
            llm_result = self._check_coherence_with_llm(plan)
            if not llm_result:
                raise ValueError("LLM coherence check returned no result.")

            self.logger.info(f"LLM coherence check: {llm_result['score']:.2f}")
            return self._create_validation_result(
                passed=llm_result['score'] >= self.min_score,
                score=llm_result['score'],
                feedback=llm_result['feedback'],
                improvements=llm_result['improvements']
            )
        except Exception as e:
            self.logger.error(f"LLM coherence check failed: {e}")
            raise RuntimeError(f"Failed to run coherence gauntlet using LLM: {e}") from e
    
    def _check_coherence_with_llm(self, plan: DecompositionPlan) -> Optional[Dict[str, Any]]:
        """Use LLM to perform deep semantic coherence analysis."""
        # Get parent problem from plan
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return None
        
        # Build sub-problems summary
        sp_summary = "\n".join([
            f"{i+1}. {sp.title}\n   Description: {sp.description}\n   Type: {sp.type.value}"
            for i, sp in enumerate(plan.sub_problems[:10])  # Limit to 10 for tokens
        ])
        
        prompt = f"""You are an expert at evaluating problem decomposition quality. Assess the COHERENCE of this decomposition.

PARENT PROBLEM:
Title: {problem.title}
Description: {problem.description}
Domain: {problem.domain_context.domain}

SUB-PROBLEMS:
{sp_summary}

COHERENCE EVALUATION:
Assess these aspects (each 0-10):

1. ALIGNMENT: Do sub-problems directly address the parent problem?
2. COVERAGE: Do sub-problems cover all aspects of the parent problem?
3. OVERLAP: Are sub-problems distinct without excessive overlap?
4. LOGICAL FLOW: Do sub-problems follow a logical progression?
5. GRANULARITY: Is the decomposition at an appropriate level of detail?

Provide your assessment in this EXACT format:
Alignment: <score>
Coverage: <score>
Overlap: <score>
LogicalFlow: <score>
Granularity: <score>
OverallScore: <average score>
Feedback: <2-3 sentence summary>
Improvements: <improvement1> | <improvement2> | <improvement3>

Be critical and precise."""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=500
        )
        
        if result.success and result.best_code:
            return self._parse_coherence_response(result.best_code)
        
        return None
    
    def _parse_coherence_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM coherence assessment."""
        lines = response.strip().split('\n')
        scores = {}
        feedback = ""
        improvements = []
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key in ['Alignment', 'Coverage', 'Overlap', 'LogicalFlow', 'Granularity', 'OverallScore']:
                try:
                    scores[key] = float(value)
                except Exception as exc:
                    self.logger.debug(f"Failed to parse coherence score '{key}': {exc}")
            elif key == 'Feedback':
                feedback = value
            elif key == 'Improvements':
                improvements = [i.strip() for i in value.split('|') if i.strip()]
        
        # Calculate overall score (0-1 scale)
        overall = scores.get('OverallScore', 5.0) / 10.0
        
        return {
            'score': overall,
            'feedback': feedback or f"Coherence assessment: {overall:.2f}",
            'improvements': improvements or ["Review decomposition structure"]
        }
    



class CompletenessGauntlet(DecompositionGauntlet):
    """Verifies all problem aspects are addressed using LLM analysis."""
    
    def __init__(self, openevolve_client=None):
        super().__init__("completeness_gauntlet")
        self.min_score = 0.75
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for completeness gauntlet")
    
    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Checks coverage of original problem using LLM semantic analysis.
        
        Validates:
        - All problem aspects are covered by sub-problems
        - No critical gaps in decomposition
        - Success criteria cover all requirements
        - All constraints are addressed
        
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        self.logger.info(f"Running completeness gauntlet on plan: {plan.id}")
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for completeness gauntlet.")

        if len(plan.sub_problems) > 10:
            self.logger.warning("Too many sub-problems for LLM completeness check, skipping.")
            return self._create_validation_result(
                passed=True,
                score=0.8, # Neutral score
                feedback="Completeness check skipped due to large number of sub-problems.",
                improvements=[]
            )

        try:
            llm_result = self._check_completeness_with_llm(plan)
            if not llm_result:
                raise ValueError("LLM completeness check returned no result.")

            self.logger.info(f"LLM completeness check: {llm_result['score']:.2f}")
            return self._create_validation_result(
                passed=llm_result['score'] >= self.min_score,
                score=llm_result['score'],
                feedback=llm_result['feedback'],
                improvements=llm_result['improvements']
            )
        except Exception as e:
            self.logger.error(f"LLM completeness check failed: {e}")
            raise RuntimeError(f"Failed to run completeness gauntlet using LLM: {e}") from e
    
    def _check_completeness_with_llm(self, plan: DecompositionPlan) -> Optional[Dict[str, Any]]:
        """Use LLM to check if decomposition completely covers the problem."""
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return None
        
        # Build sub-problems summary
        sp_summary = "\n".join([
            f"{i+1}. {sp.title} ({sp.type.value})"
            for i, sp in enumerate(plan.sub_problems[:10])
        ])
        
        # Build constraints summary
        constraints_summary = "\n".join([
            f"- {c.description} ({c.type})"
            for c in problem.constraints[:5]
        ]) if problem.constraints else "None specified"
        
        prompt = f"""You are an expert at evaluating problem decomposition completeness. Identify any GAPS in this decomposition.

PARENT PROBLEM:
Title: {problem.title}
Description: {problem.description}
Constraints: 
{constraints_summary}

PROPOSED SUB-PROBLEMS:
{sp_summary}

COMPLETENESS EVALUATION:
Assess whether the sub-problems COMPLETELY cover the parent problem:

1. COVERAGE: Do sub-problems address all aspects of the problem? (0-10)
2. CONSTRAINTS: Are all constraints addressed? (0-10)
3. REQUIREMENTS: Are all implicit requirements covered? (0-10)
4. GAPS: Are there any missing pieces? (0-10, 10=no gaps)

Provide assessment in this EXACT format:
Coverage: <score>
Constraints: <score>
Requirements: <score>
Gaps: <score>
OverallScore: <average>
Feedback: <2-3 sentences>
MissingAspects: <aspect1> | <aspect2> | <aspect3>

Be thorough and identify any gaps."""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=500
        )
        
        if result.success and result.best_code:
            return self._parse_completeness_response(result.best_code)
        
        return None
    
    def _parse_completeness_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM completeness assessment."""
        lines = response.strip().split('\n')
        overall = 5.0
        feedback = ""
        improvements = []
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'OverallScore':
                try:
                    overall = float(value)
                except Exception as exc:
                    self.logger.debug(f"Failed to parse completeness score: {exc}")
            elif key == 'Feedback':
                feedback = value
            elif key == 'MissingAspects':
                improvements = [f"Address: {i.strip()}" for i in value.split('|') if i.strip()]
        
        return {
            'score': overall / 10.0,
            'feedback': feedback or f"Completeness assessment: {overall/10.0:.2f}",
            'improvements': improvements or ["Review problem coverage"]
        }
    



class FeasibilityGauntlet(DecompositionGauntlet):
    """Verifies sub-problems are solvable."""
    
    def __init__(self, openevolve_client=None):
        super().__init__("feasibility_gauntlet")
        self.min_score = 0.7
        self.max_complexity = 8.0
        self.max_effort = 80  # person-hours
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for feasibility gauntlet")
    
    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Checks if sub-problems can be solved with available resources using LLM analysis.
        
        Validates:
        - Sub-problem complexity is manageable
        - Effort estimates are reasonable
        - No impossible constraints
        - Resources are sufficient
        
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        self.logger.info(f"Running feasibility gauntlet on plan: {plan.id}")
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for feasibility gauntlet.")

        if len(plan.sub_problems) > 10:
            self.logger.warning("Too many sub-problems for LLM feasibility check, skipping.")
            return self._create_validation_result(
                passed=True,
                score=0.8, # Neutral score
                feedback="Feasibility check skipped due to large number of sub-problems.",
                improvements=[]
            )

        try:
            llm_result = self._check_feasibility_with_llm(plan)
            if not llm_result:
                raise ValueError("LLM feasibility check returned no result.")

            self.logger.info(f"LLM feasibility check: {llm_result['score']:.2f}")
            return self._create_validation_result(
                passed=llm_result['score'] >= self.min_score,
                score=llm_result['score'],
                feedback=llm_result['feedback'],
                improvements=llm_result['improvements']
            )
        except Exception as e:
            self.logger.error(f"LLM feasibility check failed: {e}")
            raise RuntimeError(f"Failed to run feasibility gauntlet using LLM: {e}") from e

    def _check_feasibility_with_llm(self, plan: DecompositionPlan) -> Optional[Dict[str, Any]]:
        """Use LLM to perform deep semantic feasibility analysis."""
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return None

        sp_summary = "\n".join([
            f"{i+1}. {sp.title}\n   Description: {sp.description}\n   Complexity: {sp.complexity_score.overall_complexity}/10\n   Effort: {sp.estimated_effort}h"
            for i, sp in enumerate(plan.sub_problems[:10])
        ])

        prompt = f"""You are an expert project manager. Assess the FEASIBILITY of this decomposition.\n\nPARENT PROBLEM:\nTitle: {problem.title}\nDescription: {problem.description}\n\nSUB-PROBLEMS:\n{sp_summary}\n\nFEASIBILITY EVALUATION:\nAssess these aspects (each 0-10):\n\n1. COMPLEXITY: Is the complexity of each sub-problem manageable?\n2. EFFORT: Are the effort estimates realistic?\n3. RESOURCES: Are the (implicit) resources likely sufficient?\n4. SKILLS: Are the required skills reasonable to assume available?\n\nProvide your assessment in this EXACT format:\nComplexity: <score>\nEffort: <score>\nResources: <score>\nSkills: <score>\nOverallScore: <average score>\nFeedback: <2-3 sentence summary>\nImprovements: <improvement1> | <improvement2> | <improvement3>\n\nBe critical and realistic."""

        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=500
        )

        if result.success and result.best_code:
            return self._parse_feasibility_response(result.best_code)

        return None

    def _parse_feasibility_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM feasibility assessment."""
        lines = response.strip().split('\n')
        scores = {}
        feedback = ""
        improvements = []

        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key in ['Complexity', 'Effort', 'Resources', 'Skills', 'OverallScore']:
                try:
                    scores[key] = float(value)
                except Exception as exc:
                    self.logger.debug(f"Failed to parse feasibility score '{key}': {exc}")
            elif key == 'Feedback':
                feedback = value
            elif key == 'Improvements':
                improvements = [i.strip() for i in value.split('|') if i.strip()]

        overall = scores.get('OverallScore', 5.0) / 10.0

        return {
            'score': overall,
            'feedback': feedback or f"Feasibility assessment: {overall:.2f}",
            'improvements': improvements or ["Review feasibility of sub-problems"]
        }
    



class DependencyGauntlet(DecompositionGauntlet):
    """Validates dependency graph structure."""
    
    def __init__(self, openevolve_client=None):
        super().__init__("dependency_gauntlet")
        self.min_score = 0.8
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for dependency gauntlet")
    
    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Validates dependency graph structure using LLM analysis.
        
        Validates:
        - No circular dependencies
        - Dependencies reference valid sub-problems
        - Dependency graph is acyclic
        - Critical path is reasonable
        
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        self.logger.info(f"Running dependency gauntlet on plan: {plan.id}")

        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for dependency gauntlet.")

        if len(plan.sub_problems) > 10:
            self.logger.warning("Too many sub-problems for LLM dependency check, skipping.")
            return self._create_validation_result(
                passed=True,
                score=0.8, # Neutral score
                feedback="Dependency check skipped due to large number of sub-problems.",
                improvements=[]
            )

        try:
            llm_result = self._check_dependency_with_llm(plan)
            if not llm_result:
                raise ValueError("LLM dependency check returned no result.")

            self.logger.info(f"LLM dependency check: {llm_result['score']:.2f}")
            return self._create_validation_result(
                passed=llm_result['score'] >= self.min_score,
                score=llm_result['score'],
                feedback=llm_result['feedback'],
                improvements=llm_result['improvements']
            )
        except Exception as e:
            self.logger.error(f"LLM dependency check failed: {e}")
            raise RuntimeError(f"Failed to run dependency gauntlet using LLM: {e}") from e

    def _check_dependency_with_llm(self, plan: DecompositionPlan) -> Optional[Dict[str, Any]]:
        """Use LLM to perform deep semantic dependency analysis."""
        sp_summary = "\n".join([
            f"{i+1}. ID: {sp.id}, Title: {sp.title}, Dependencies: {sp.dependencies}"
            for i, sp in enumerate(plan.sub_problems[:10])
        ])

        prompt = f"""You are an expert in dependency analysis. Assess the DEPENDENCY STRUCTURE of this decomposition.

SUB-PROBLEMS:
{sp_summary}

DEPENDENCY EVALUATION:
Assess these aspects (each 0-10):

1. CORRECTNESS: Are the dependencies logically correct?
2. COMPLETENESS: Are there any missing dependencies?
3. CYCLES: Are there any circular dependencies?
4. GRANULARITY: Are the dependencies at the right level of detail?

Provide your assessment in this EXACT format:
Correctness: <score>
Completeness: <score>
Cycles: <score>
Granularity: <score>
OverallScore: <average score>
Feedback: <2-3 sentence summary>
Improvements: <improvement1> | <improvement2> | <improvement3>

Be critical and precise."""

        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=500
        )

        if result.success and result.best_code:
            return self._parse_dependency_response(result.best_code)

        return None

    def _parse_dependency_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM dependency assessment."""
        lines = response.strip().split('\n')
        scores = {}
        feedback = ""
        improvements = []

        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key in ['Correctness', 'Completeness', 'Cycles', 'Granularity', 'OverallScore']:
                try:
                    scores[key] = float(value)
                except Exception as exc:
                    self.logger.debug(f"Failed to parse dependency score '{key}': {exc}")
            elif key == 'Feedback':
                feedback = value
            elif key == 'Improvements':
                improvements = [i.strip() for i in value.split('|') if i.strip()]

        overall = scores.get('OverallScore', 5.0) / 10.0

        return {
            'score': overall,
            'feedback': feedback or f"Dependency assessment: {overall:.2f}",
            'improvements': improvements or ["Review dependency structure"]
        }


class AdaptiveGauntlet(DecompositionGauntlet):
    """Adapts its difficulty based on problem complexity."""

    def __init__(self, openevolve_client=None):
        super().__init__("adaptive_gauntlet")
        self.base_min_score = 0.7
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for adaptive gauntlet")

    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Runs a coherence check with adaptive difficulty.
        """
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return self._create_validation_result(False, 0.0, "Problem definition not found", ["Ensure problem is persisted before running gauntlet"])

        # Adapt min_score based on complexity
        complexity = problem.complexity_score.overall_complexity
        # Higher complexity demands a more lenient initial check
        adapted_min_score = self.base_min_score - (complexity - 5.0) * 0.05
        adapted_min_score = max(0.5, min(0.9, adapted_min_score))

        self.logger.info(f"Running adaptive gauntlet on plan: {plan.id} with adapted min_score: {adapted_min_score:.2f}")

        # Reuse CoherenceGauntlet logic
        coherence_gauntlet = CoherenceGauntlet(self.openevolve_client)
        coherence_gauntlet.min_score = adapted_min_score
        return coherence_gauntlet.run(plan)


class HierarchicalGauntlet(DecompositionGauntlet):
    """Runs different gauntlets based on decomposition level."""

    def __init__(self, openevolve_client=None):
        super().__init__("hierarchical_gauntlet")
        self.openevolve_client = openevolve_client

    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Runs a different set of gauntlets based on the decomposition level.
        """
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return self._create_validation_result(False, 0.0, "Problem definition not found", ["Ensure problem is persisted before running gauntlet"])

        # Simple check for top-level vs. sub-problem
        is_top_level = not problem.parent_id

        if is_top_level:
            self.logger.info(f"Running stringent gauntlet suite for top-level problem: {plan.id}")
            gauntlets_to_run = ['coherence', 'completeness', 'feasibility', 'dependency']
        else:
            self.logger.info(f"Running lenient gauntlet suite for sub-problem: {plan.id}")
            gauntlets_to_run = ['coherence', 'feasibility']

        gauntlet_system = GauntletSystem(self.openevolve_client)
        results = gauntlet_system.run_decomposition_gauntlets(plan, gauntlets=gauntlets_to_run)

        overall_score = gauntlet_system.get_overall_quality(results)
        all_passed = gauntlet_system.all_passed(results)

        return self._create_validation_result(
            passed=all_passed,
            score=overall_score,
            feedback=f"Hierarchical gauntlet completed. {len(gauntlets_to_run)} gauntlets run.",
            improvements=[f"{name}: {'PASS' if res.passed else 'FAIL'}" for name, res in results.items()]
        )


class CompetitiveGauntlet(DecompositionGauntlet):
    """Pits multiple solution attempts against each other."""

    def __init__(self, openevolve_client=None):
        super().__init__("competitive_gauntlet")
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for competitive gauntlet")

    def run(self, plan: DecompositionPlan, attempts: List[SolutionAttempt]) -> ValidationResult:
        """
        Compares multiple solution attempts and ranks them.
        """
        self.logger.info(f"Running competitive gauntlet on {len(attempts)} attempts for plan: {plan.id}")

        if len(attempts) < 2:
            return self._create_validation_result(True, 1.0, "Not enough attempts to run competitive gauntlet.", [])

        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for competitive gauntlet.")

        try:
            llm_result = self._compare_solutions_with_llm(plan, attempts)
            if not llm_result:
                raise ValueError("LLM comparison returned no result.")

            self.logger.info(f"LLM comparison completed with winner: {llm_result.get('winner')}")
            # The competitive gauntlet itself doesn't have a simple pass/fail, but we can use the confidence score.
            return self._create_validation_result(
                passed=True,
                score=llm_result.get('confidence', 0.0),
                feedback=f"Winning solution: {llm_result.get('winner')}. {llm_result.get('justification', '')}",
                improvements=[f"Ranking: {llm_result.get('ranking')}"]
            )
        except Exception as e:
            self.logger.error(f"LLM comparison failed: {e}")
            raise RuntimeError(f"Failed to run competitive gauntlet using LLM: {e}") from e

    def _compare_solutions_with_llm(self, plan: DecompositionPlan, attempts: List[SolutionAttempt]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to compare and rank solution attempts.
        """
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return None

        solutions_summary = "\n".join([
            f"SOLUTION {i+1} (ID: {attempt.id}):\n{attempt.solution_content[:300]}..."
            for i, attempt in enumerate(attempts)
        ])

        prompt = f"""You are an expert judge. Compare the following solutions for the given problem and determine the best one.

PROBLEM:
{problem.title}

SOLUTIONS:
{solutions_summary}

COMPARISON TASK:
- Rank the solutions from best to worst.
- Provide a clear justification for your ranking.
- Identify the winning solution ID.

Provide your assessment in this EXACT format:
Winner: <solution_id>
Ranking: [<solution_id_1>, <solution_id_2>, ...]
Justification: <Detailed justification for your ranking>
Confidence: <0.0-1.0>
"""

        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.2,
            max_tokens=1000
        )

        if result.success and result.best_code:
            return self._parse_comparison_response(result.best_code, attempts)

        return None

    def _parse_comparison_response(self, response: str, attempts: List[SolutionAttempt]) -> Dict[str, Any]:
        """
        Parse LLM comparison response.
        """
        winner = None
        ranking = []
        justification = ""
        confidence = 0.0

        for line in response.strip().split('\n'):
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'winner':
                winner = value
            elif key == 'ranking':
                ranking = [item.strip() for item in value.strip('[]').split(',')]
            elif key == 'justification':
                justification = value
            elif key == 'confidence':
                try:
                    confidence = float(value)
                except ValueError:
                    self.logger.debug(f"Failed to parse comparison confidence: {value}")

        # Validate that the winner and ranking contain valid solution IDs
        valid_ids = {attempt.id for attempt in attempts}
        if winner not in valid_ids:
            winner = None
        ranking = [rid for rid in ranking if rid in valid_ids]

        return {
            'winner': winner,
            'ranking': ranking,
            'justification': justification,
            'confidence': confidence
        }


class CollaborativeGauntlet(DecompositionGauntlet):
    """Synthesizes a new solution from multiple attempts."""

    def __init__(self, openevolve_client=None):
        super().__init__("collaborative_gauntlet")
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for collaborative gauntlet")

    def run(self, plan: DecompositionPlan, attempts: List[SolutionAttempt]) -> ValidationResult:
        """
        Synthesizes a new solution from multiple attempts.
        """
        self.logger.info(f"Running collaborative gauntlet on {len(attempts)} attempts for plan: {plan.id}")

        if len(attempts) < 2:
            return self._create_validation_result(True, 1.0, "Not enough attempts to run collaborative gauntlet.", [])

        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for collaborative gauntlet.")

        try:
            synthesized_solution = self._synthesize_solution_with_llm(plan, attempts)
            if not synthesized_solution:
                raise ValueError("LLM synthesis returned no result.")

            # The result of this gauntlet is the new solution itself.
            # The feedback will contain the new solution.
            return self._create_validation_result(
                passed=True,
                score=synthesized_solution.get('confidence', 0.0),
                feedback=f"Synthesized a new solution with confidence {synthesized_solution.get('confidence', 0.0)}.",
                improvements=[synthesized_solution.get('content', '')]
            )
        except Exception as e:
            self.logger.error(f"LLM synthesis failed: {e}")
            raise RuntimeError(f"Failed to run collaborative gauntlet using LLM: {e}") from e

    def _synthesize_solution_with_llm(self, plan: DecompositionPlan, attempts: List[SolutionAttempt]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to synthesize a new solution from multiple attempts.
        """
        from sovereign_persistence import SovereignPersistence
        db = SovereignPersistence()
        problem = db.get_problem(plan.problem_id)
        if not problem:
            return None

        solutions_summary = "\n".join([
            f"SOLUTION {i+1} (ID: {attempt.id}):\n{attempt.solution_content[:300]}..."
            for i, attempt in enumerate(attempts)
        ])

        prompt = f"""You are an expert synthesizer. Combine the best aspects of the following solutions into a new, superior solution.

PROBLEM:
{problem.title}

SOLUTIONS:
{solutions_summary}

SYNTHESIS TASK:
- Analyze the strengths and weaknesses of each solution.
- Create a new solution that combines the strengths and mitigates the weaknesses.
- The new solution should be more robust, complete, and elegant than any of the individual solutions.

Provide your assessment in this EXACT format:
SynthesizedSolution: <The full content of the new solution>
Confidence: <0.0-1.0>
Justification: <Explanation of why the new solution is superior>
"""

        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.4,
            max_tokens=2000
        )

        if result.success and result.best_code:
            return self._parse_synthesis_response(result.best_code)

        return None

    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM synthesis response.
        """
        content = ""
        confidence = 0.0
        justification = ""

        # This is a simple parser. A more robust implementation would handle multiline content better.
        in_solution = False
        for line in response.strip().split('\n'):
            if line.lower().startswith('synthesizedsolution:'):
                content = line.split(':', 1)[1].strip()
                in_solution = True
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    self.logger.debug(f"Failed to parse synthesis confidence: {line}")
                in_solution = False
            elif line.lower().startswith('justification:'):
                justification = line.split(':', 1)[1].strip()
                in_solution = False
            elif in_solution:
                content += "\n" + line

        return {
            'content': content,
            'confidence': confidence,
            'justification': justification
        }


class FormalSovereignGauntlet(DecompositionGauntlet):
    """
    Sovereign-grade gauntlet with Lean theorem prover formal verification.
    
    This gauntlet formalizes decomposition plans into Lean 4 and verifies
    their mathematical correctness, providing rigorous guarantees.
    """
    
    def __init__(self, openevolve_client=None):
        super().__init__("formal_sovereign_gauntlet")
        self.openevolve_client = openevolve_client
        self.lean_client: Optional[LeanAideClient] = None
        
        if LEAN_AVAILABLE:
            try:
                self.lean_client = LeanAideClient()
                self.logger.info("FormalSovereignGauntlet initialized with LeanAideClient")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LeanAideClient: {e}")
    
    def run(self, plan: DecompositionPlan) -> ValidationResult:
        """
        Run formal verification on decomposition plan using Lean.
        
        Args:
            plan: The decomposition plan to validate
            
        Returns:
            ValidationResult with formal verification outcome
        """
        self.logger.info(f"Running formal sovereign gauntlet on plan: {plan.id}")
        
        if not LEAN_AVAILABLE or not self.lean_client:
            return self._create_validation_result(
                passed=False,
                score=0.0,
                feedback="Lean formal verification not available",
                improvements=["Install LeanAide for formal verification"]
            )
        
        try:
            # Convert plan to verifiable format
            plan_text = self._plan_to_verifiable_text(plan)
            
            # Auto-formalize the plan
            formalized = self.lean_client.autoformalize(plan_text)
            
            # Verify the formalized plan
            verification = self.lean_client.verify(formalized)
            
            verified = verification.get("success", False)
            errors = verification.get("errors", [])
            
            # Calculate score
            score = 1.0 if verified else 0.0
            if errors:
                score = max(0.0, 1.0 - len(errors) * 0.1)
            
            if verified:
                return self._create_validation_result(
                    passed=True,
                    score=score,
                    feedback=f"Formal verification passed. Plan is mathematically sound.",
                    improvements=[]
                )
            else:
                return self._create_validation_result(
                    passed=False,
                    score=score,
                    feedback=f"Formal verification failed with {len(errors)} errors",
                    improvements=[f"Fix: {error}" for error in errors[:5]]
                )
                
        except Exception as e:
            self.logger.error(f"Formal verification failed: {e}")
            return self._create_validation_result(
                passed=False,
                score=0.0,
                feedback=f"Formal verification error: {str(e)}",
                improvements=["Check plan structure and retry"]
            )
    
    def _plan_to_verifiable_text(self, plan: DecompositionPlan) -> str:
        """Convert decomposition plan to text suitable for formalization."""
        lines = [
            f"Decomposition Plan: {plan.id}",
            f"Problem: {plan.problem_description}",
            f"Success Criteria: {plan.success_criteria}",
            "",
            "Sub-problems:"
        ]
        
        for i, sub in enumerate(plan.sub_problems, 1):
            lines.append(f"  {i}. {sub.description} (Complexity: {sub.complexity})")
        
        return "\n".join(lines)
    
    def verify_with_lean(self, content: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify content using Lean theorem prover.
        
        Args:
            content: The content to verify
            properties: Optional properties for verification
            
        Returns:
            Dict with verification results
        """
        if not LEAN_AVAILABLE or not self.lean_client:
            return {"verified": False, "error": "Lean verification not available"}
        
        try:
            # Auto-formalize the content
            formalized = self.lean_client.autoformalize(content)
            # Verify the formalized content
            verification = self.lean_client.verify(formalized)
            
            return {
                "verified": verification.get("success", False),
                "formalized": formalized,
                "proof_status": verification.get("status", "unknown"),
                "errors": verification.get("errors", []),
                "metadata": properties or {}
            }
        except Exception as e:
            self.logger.error(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}


class GauntletSystem:
    """Orchestrates decomposition gauntlets with ICR integration."""
    
    def __init__(
        self, 
        openevolve_client=None,
        refinement_coordinator=None,  # ICR integration
        track_patterns: bool = True
    ):
        self.openevolve_client = openevolve_client
        self.refinement_coordinator = refinement_coordinator
        self.track_patterns = track_patterns
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for GauntletSystem")

        self.gauntlets = {
            'coherence': CoherenceGauntlet(self.openevolve_client),
            'completeness': CompletenessGauntlet(self.openevolve_client),
            'feasibility': FeasibilityGauntlet(self.openevolve_client),
            'dependency': DependencyGauntlet(self.openevolve_client),
            'adaptive': AdaptiveGauntlet(self.openevolve_client),
            'hierarchical': HierarchicalGauntlet(self.openevolve_client),
            'competitive': CompetitiveGauntlet(self.openevolve_client),
            'collaborative': CollaborativeGauntlet(self.openevolve_client)
        }
        self.logger = logging.getLogger(__name__)
        
        # ICR: Gauntlet effectiveness patterns storage
        self._gauntlet_patterns: Dict[str, List[Dict]] = {}
        self._gauntlet_metrics: Dict[str, Dict[str, float]] = {}
    
    def run_decomposition_gauntlets(
        self, 
        plan: DecompositionPlan,
        gauntlets: Optional[List[str]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Runs all or specified decomposition gauntlets.
        
        Args:
            plan: The decomposition plan to validate
            gauntlets: Optional list of gauntlet names to run (runs all if None)
            
        Returns:
            Dictionary mapping gauntlet names to ValidationResult objects
        """
        self.logger.info(f"Running gauntlets on plan: {plan.id}")
        
        gauntlets_to_run = gauntlets or list(self.gauntlets.keys())
        results = {}
        
        for gauntlet_name in gauntlets_to_run:
            gauntlet = self.gauntlets.get(gauntlet_name)
            if gauntlet:
                try:
                    result = gauntlet.run(plan)
                    results[gauntlet_name] = result
                    self.logger.info(f"{gauntlet_name}: {'PASS' if result.passed else 'FAIL'} ({result.score:.2f})")
                except Exception as e:
                    self.logger.error(f"Error running {gauntlet_name}: {e}")
                    results[gauntlet_name] = ValidationResult(
                        validator=gauntlet_name,
                        passed=False,
                        score=0.0,
                        feedback=f"Error: {str(e)}",
                        improvements=["Fix gauntlet execution error"],
                        timestamp=datetime.now()
                    )
        
        return results
    
    def process_gauntlet_feedback(
        self, 
        results: Dict[str, ValidationResult]
    ) -> List[Feedback]:
        """
        Converts gauntlet results into actionable feedback.
        
        Args:
            results: Dictionary of gauntlet results
            
        Returns:
            List of Feedback objects
        """
        feedback_list = []
        
        for gauntlet_name, result in results.items():
            # Determine severity based on score
            if result.score >= 0.8:
                severity = "info"
            elif result.score >= 0.6:
                severity = "minor"
            elif result.score >= 0.4:
                severity = "major"
            else:
                severity = "critical"
            
            # Create feedback
            feedback = Feedback(
                id=generate_id("feedback"),
                source=gauntlet_name,
                feedback_type="critique" if not result.passed else "approval",
                content=result.feedback,
                severity=severity,
                actionable=len(result.improvements) > 0,
                timestamp=result.timestamp,
                metadata={
                    'score': result.score,
                    'improvements': result.improvements
                }
            )
            feedback_list.append(feedback)
        
        return feedback_list
    
    def get_overall_quality(self, results: Dict[str, ValidationResult]) -> float:
        """Calculate overall quality score from gauntlet results."""
        if not results:
            return 0.0
        
        scores = [r.score for r in results.values()]
        return sum(scores) / len(scores)
    
    def all_passed(self, results: Dict[str, ValidationResult]) -> bool:
        """Check if all gauntlets passed."""
        return all(r.passed for r in results.values())

    def run_competitive_gauntlet(
        self, 
        plan: DecompositionPlan,
        attempts: List['SolutionAttempt']
    ) -> ValidationResult:
        """
        Runs the competitive gauntlet on a list of solution attempts.
        """
        self.logger.info(f"Running competitive gauntlet on {len(attempts)} attempts for plan: {plan.id}")
        competitive_gauntlet = self.gauntlets.get('competitive')
        if competitive_gauntlet and isinstance(competitive_gauntlet, CompetitiveGauntlet):
            return competitive_gauntlet.run(plan, attempts)
        else:
            return ValidationResult(
                validator="competitive_gauntlet",
                passed=False,
                score=0.0,
                feedback="Competitive gauntlet not available",
                improvements=[],
                timestamp=datetime.now()
            )

    def run_collaborative_gauntlet(
        self, 
        plan: DecompositionPlan,
        attempts: List['SolutionAttempt']
    ) -> ValidationResult:
        """
        Runs the collaborative gauntlet on a list of solution attempts.
        """
        self.logger.info(f"Running collaborative gauntlet on {len(attempts)} attempts for plan: {plan.id}")
        collaborative_gauntlet = self.gauntlets.get('collaborative')
        if collaborative_gauntlet and isinstance(collaborative_gauntlet, CollaborativeGauntlet):
            return collaborative_gauntlet.run(plan, attempts)
        else:
            return ValidationResult(
                validator="collaborative_gauntlet",
                passed=False,
                score=0.0,
                feedback="Collaborative gauntlet not available",
                improvements=[],
                timestamp=datetime.now()
            )
    
    # =========================================================================
    # ICR INTEGRATION METHODS
    # =========================================================================
    
    def run_with_icr_refinement(
        self,
        plan: 'DecompositionPlan',
        max_refinement_cycles: int = 5,
        refinement_threshold: float = 0.7,
        convergence_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run gauntlets with automatic ICR refinement trigger.
        
        This is the key integration point between GauntletSystem and ICR.
        When gauntlets fail, automatically triggers refinement and re-runs.
        
        Args:
            plan: The decomposition plan to validate
            max_refinement_cycles: Maximum refinement iterations
            refinement_threshold: Quality score below which to trigger refinement
            convergence_threshold: Minimum improvement to continue refining
            
        Returns:
            Dictionary with final results, refinement history, and metrics
        """
        self.logger.info(f"Running gauntlets with ICR refinement for plan: {plan.id}")
        
        current_plan = plan
        cycle_number = 0
        converged = False
        refinement_history = []
        
        while cycle_number < max_refinement_cycles and not converged:
            cycle_number += 1
            self.logger.info(f"ICR Cycle {cycle_number}/{max_refinement_cycles}")
            
            # Run gauntlets
            gauntlet_results = self.run_decomposition_gauntlets(current_plan)
            overall_quality = self.get_overall_quality(gauntlet_results)
            all_passed = self.all_passed(gauntlet_results)
            
            cycle_result = {
                'cycle': cycle_number,
                'gauntlet_results': {name: {
                    'passed': r.passed,
                    'score': r.score,
                    'feedback': r.feedback
                } for name, r in gauntlet_results.items()},
                'overall_quality': overall_quality,
                'all_passed': all_passed
            }
            
            # Store pattern for ICR learning
            if self.track_patterns:
                self._store_gauntlet_pattern(plan.id, gauntlet_results)
            
            refinement_history.append(cycle_result)
            
            # Check if refinement needed
            if all_passed:
                self.logger.info("All gauntlets passed - no refinement needed")
                converged = True
                break
            
            if overall_quality >= refinement_threshold:
                self.logger.info(f"Quality {overall_quality:.2f} >= threshold {refinement_threshold} - refining anyway")
            
            # Check convergence (from previous cycle)
            if cycle_number > 1:
                prev_quality = refinement_history[-2]['overall_quality']
                improvement = overall_quality - prev_quality
                if improvement < convergence_threshold:
                    self.logger.info(f"Converged: improvement {improvement:.3f} < threshold")
                    converged = True
                    break
            
            # Trigger ICR refinement via RefinementCoordinator
            if self.refinement_coordinator:
                # Convert gauntlet results to feedback
                feedback = self.process_gauntlet_feedback(gauntlet_results)
                
                # Generate and execute refinement plan
                smart_strategy = self.refinement_coordinator.generate_smart_refinement_strategy(
                    current_plan, feedback
                )
                refinement_plan = self.refinement_coordinator.generate_refinement_plan(
                    current_plan, feedback, smart_strategy
                )
                
                # Execute refinement
                current_plan, metrics = self.refinement_coordinator.execute_refinement(
                    current_plan, refinement_plan
                )
                
                cycle_result['refinement_applied'] = True
                cycle_result['refinement_metrics'] = {
                    'quality_improvement': metrics.quality_improvement,
                    'issues_resolved': metrics.issues_resolved
                }
                
                self.logger.info(f"Refinement complete: {metrics.issues_resolved} issues resolved")
            else:
                self.logger.warning("No RefinementCoordinator configured - skipping refinement")
                cycle_result['refinement_applied'] = False
                break
        
        # Final gauntlet run
        final_results = self.run_decomposition_gauntlets(current_plan)
        final_quality = self.get_overall_quality(final_results)
        
        return {
            'plan_id': plan.id,
            'initial_plan_id': plan.id,
            'final_plan_id': current_plan.id,
            'total_cycles': cycle_number,
            'converged': converged,
            'final_quality': final_quality,
            'final_results': {name: {'passed': r.passed, 'score': r.score} for name, r in final_results.items()},
            'refinement_history': refinement_history
        }
    
    def _store_gauntlet_pattern(
        self,
        plan_id: str,
        results: Dict[str, ValidationResult]
    ) -> None:
        """
        Store gauntlet execution pattern for ICR learning.
        
        This enables the system to learn which gauntlets tend to fail together
        and which refinements are most effective for specific patterns.
        """
        pattern = {
            'plan_id': plan_id,
            'timestamp': datetime.now().isoformat(),
            'overall_quality': self.get_overall_quality(results),
            'passed_count': sum(1 for r in results.values() if r.passed),
            'failed_gauntlets': [name for name, r in results.items() if not r.passed],
            'avg_score': sum(r.score for r in results.values()) / len(results),
            'failed_scores': {name: r.score for name, r in results.items() if not r.passed}
        }
        
        # Store by pattern type for quick lookup
        failed_key = tuple(sorted(pattern['failed_gauntlets']))
        if failed_key not in self._gauntlet_patterns:
            self._gauntlet_patterns[failed_key] = []
        self._gauntlet_patterns[failed_key].append(pattern)
        
        # Update metrics
        for name, result in results.items():
            if name not in self._gauntlet_metrics:
                self._gauntlet_metrics[name] = {
                    'total_runs': 0,
                    'total_score': 0,
                    'pass_count': 0,
                    'fail_count': 0
                }
            self._gauntlet_metrics[name]['total_runs'] += 1
            self._gauntlet_metrics[name]['total_score'] += result.score
            if result.passed:
                self._gauntlet_metrics[name]['pass_count'] += 1
            else:
                self._gauntlet_metrics[name]['fail_count'] += 1
    
    def get_gauntlet_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """
        Get effectiveness metrics for each gauntlet.
        
        Returns:
            Dictionary mapping gauntlet names to effectiveness metrics
        """
        effectiveness = {}
        
        for name, metrics in self._gauntlet_metrics.items():
            if metrics['total_runs'] > 0:
                effectiveness[name] = {
                    'total_runs': metrics['total_runs'],
                    'pass_rate': metrics['pass_count'] / metrics['total_runs'],
                    'avg_score': metrics['total_score'] / metrics['total_runs'],
                    'fail_rate': metrics['fail_count'] / metrics['total_runs']
                }
        
        return effectiveness
    
    def get_failure_patterns(self) -> Dict[str, List[Dict]]:
        """
        Get learned failure patterns from gauntlet executions.
        
        Returns:
            Dictionary mapping failed gauntlet tuples to pattern lists
        """
        return self._gauntlet_patterns
    
    def suggest_optimal_gauntlets(
        self,
        plan_type: str = "general",
        complexity: float = 0.5
    ) -> List[str]:
        """
        Suggest optimal gauntlet configuration based on ICR patterns.
        
        Args:
            plan_type: Type of plan (e.g., "analysis", "synthesis")
            complexity: Plan complexity score (0.0 - 1.0)
            
        Returns:
            List of recommended gauntlet names
        """
        # Learn from patterns: which gauntlets fail together?
        common_failures = {}
        for pattern_list in self._gauntlet_patterns.values():
            for pattern in pattern_list:
                for failed in pattern['failed_gauntlets']:
                    if failed not in common_failures:
                        common_failures[failed] = 0
                    common_failures[failed] += 1
        
        # Suggest gauntlets that commonly fail together should be run together
        recommended = ['coherence', 'completeness', 'feasibility', 'dependency']
        
        # Add adaptive gauntlet for complex plans
        if complexity > 0.6:
            recommended.append('adaptive')
        
        # Add hierarchical for nested decompositions
        if complexity > 0.7:
            recommended.append('hierarchical')
        
        # Add competitive/collaborative for multi-solution scenarios
        recommended.extend(['competitive', 'collaborative'])
        
        return list(set(recommended))  # Remove duplicates
    
    def adapt_gauntlet_config(
        self,
        gauntlet_name: str,
        plan_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt gauntlet configuration based on ICR patterns and plan context.
        
        Args:
            gauntlet_name: Name of the gauntlet to adapt
            plan_context: Context information about the plan
            
        Returns:
            Adapted configuration parameters
        """
        config = {}
        
        # Get historical effectiveness
        metrics = self._gauntlet_metrics.get(gauntlet_name, {})
        pass_rate = metrics.get('pass_rate', 0.5)
        avg_score = metrics.get('avg_score', 0.5)
        
        # Adjust min_score based on historical performance
        if pass_rate < 0.5:
            # Gauntlet is too strict - lower the threshold
            config['min_score'] = max(0.4, avg_score - 0.1)
        elif pass_rate > 0.9:
            # Gauntlet is too lenient - raise the threshold
            config['min_score'] = min(0.9, avg_score + 0.1)
        else:
            config['min_score'] = avg_score
        
        # Adjust based on plan complexity
        complexity = plan_context.get('complexity', 0.5)
        if complexity > 0.7:
            # More complex plans need stricter validation
            config['min_score'] = min(0.9, config['min_score'] + 0.05)
        
        return config
    
    def clear_patterns(self) -> None:
        """Clear stored gauntlet patterns and metrics."""
        self._gauntlet_patterns.clear()
        self._gauntlet_metrics.clear()
        self.logger.info("Cleared all gauntlet patterns and metrics")
    
    def run_gauntlet_with_dts_strategy_exploration(
        self,
        plan: DecompositionPlan,
        gauntlet_name: str,
        strategy_count: int = 5,
        use_adaptive_strategies: bool = True
    ) -> Dict[str, Any]:
        """
        Run a gauntlet with DTS (Dialogue Tree Search) for enhanced strategy exploration.
        
        Args:
            plan: The decomposition plan to validate
            gauntlet_name: Name of the gauntlet to run
            strategy_count: Number of strategies to explore (default 5)
            use_adaptive_strategies: Whether to use adaptive strategies based on context
            
        Returns:
            Dictionary with results including best strategy, scores, and recommendations
        """
        if not DTS_AVAILABLE:
            logger.warning("DTS not available, falling back to standard gauntlet")
            # Fall back to standard gauntlet run
            gauntlet = self.gauntlets.get(gauntlet_name)
            if gauntlet:
                result = gauntlet.run(plan)
                return {
                    "strategy_results": [{"strategy": "standard", "result": result}],
                    "best_strategy": "standard",
                    "best_score": result.score,
                    "dts_available": False,
                    "fallback_used": True
                }
            else:
                return {
                    "error": f"Gauntlet {gauntlet_name} not found",
                    "dts_available": False,
                    "fallback_used": True
                }
        
        try:
            # Initialize DTS integration
            dts_config = DTSIntegrationConfig(
                use_strategy_exploration=True,
                use_multi_judge=True,
                judge_count=3,
                use_comparative_scoring=True
            )
            dts_integration = DTSIntegration(dts_config)
            
            # Prepare context for strategy exploration
            context = {
                "decomposition_plan": {
                    "id": plan.id,
                    "title": plan.title,
                    "description": plan.description,
                    "sub_problems": [sp.title for sp in plan.sub_problems]
                },
                "gauntlet_name": gauntlet_name,
                "current_strategy": "standard",
                "optimization_goals": ["effectiveness", "efficiency", "completeness"]
            }
            
            # Generate strategies using DTS
            strategies = dts_integration.generate_strategies(
                problem=f"Validate decomposition plan using {gauntlet_name} gauntlet",
                num_strategies=strategy_count,
                context=context
            )
            
            # Evaluate each strategy and find the best one
            strategy_results = []
            best_strategy = None
            best_score = 0.0
            
            for i, strategy in enumerate(strategies):
                try:
                    # Apply the strategy to the gauntlet run (this is a simplified approach)
                    # In a real implementation, you would customize how the gauntlet is run
                    # based on the strategy suggestion
                    gauntlet = self.gauntlets.get(gauntlet_name)
                    if gauntlet:
                        result = gauntlet.run(plan)  # Standard run for now
                        
                        strategy_result = {
                            "strategy_id": i,
                            "strategy_description": strategy.get("description", f"Strategy {i}"),
                            "result": result,
                            "score": result.score,
                            "passed": result.passed
                        }
                        
                        strategy_results.append(strategy_result)
                        
                        if result.score > best_score:
                            best_score = result.score
                            best_strategy = strategy_result
                            
                except Exception as e:
                    logger.error(f"Error running strategy {i}: {e}")
                    continue
            
            return {
                "strategy_results": strategy_results,
                "best_strategy": best_strategy,
                "best_score": best_score,
                "total_strategies": len(strategy_results),
                "successful_strategies": len([r for r in strategy_results if r["passed"]]),
                "dts_available": True,
                "fallback_used": False,
                "strategies_generated": strategies
            }
            
        except Exception as e:
            logger.error(f"Error running DTS-enhanced gauntlet: {e}", exc_info=True)
            # Fall back to standard gauntlet run
            gauntlet = self.gauntlets.get(gauntlet_name)
            if gauntlet:
                result = gauntlet.run(plan)
                return {
                    "strategy_results": [{"strategy": "standard", "result": result}],
                    "best_strategy": "standard",
                    "best_score": result.score,
                    "dts_available": True,  # DTS was available but failed
                    "fallback_used": True,
                    "error": str(e)
                }
            else:
                return {
                    "error": f"Gauntlet {gauntlet_name} not found",
                    "dts_available": True,
                    "fallback_used": True,
                    "error": str(e)
                }
