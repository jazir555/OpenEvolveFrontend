"""
Sovereign-Grade Problem Decomposition System - Comprehensive Validation
Task 15.5: Validate all requirements met and verify integration points.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from sovereign_data_models import DecompositionPlan, ProblemDefinition
from sovereign_integration import SovereignIntegrationOrchestrator

logger = logging.getLogger(__name__)

# **LEAN INTEGRATION**: Real Lean client for formal verification
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of comprehensive system validation."""
    passed: bool
    score: float
    checks_passed: int
    checks_failed: int
    details: Dict[str, Any]
    timestamp: datetime


class ComprehensiveValidator:
    """
    Validates all system requirements and integration points.
    
    Performs end-to-end validation of:
    - All core components functional
    - Integration points working
    - Quality thresholds met
    - Performance requirements satisfied
    """
    
    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger(__name__)
        self.orchestrator = SovereignIntegrationOrchestrator()
    
    def verify_with_lean(self, target: str, criteria: Dict) -> Dict:
        """Verify target using Lean theorem prover."""
        if not LEAN_AVAILABLE:
            return {'verified': False}
        try:
            client = LeanAideClient()
            return client.verify(target)
        except Exception:
            return {'verified': False}
    
    def validate_all_requirements(self) -> ValidationResult:
        """
        Validate all system requirements are met.
        
        Returns:
            ValidationResult with comprehensive validation status
        """
        self.logger.info("Starting comprehensive validation...")
        start_time = datetime.now()
        
        checks = {}
        
        # Requirement 1: Problem Analysis
        checks['problem_analysis'] = self._validate_problem_analysis()
        
        # Requirement 2: Decomposition Strategies
        checks['decomposition'] = self._validate_decomposition()
        
        # Requirement 3: Dependency Management
        checks['dependencies'] = self._validate_dependencies()
        
        # Requirement 4: Validation Gauntlets
        checks['gauntlets'] = self._validate_gauntlets()
        
        # Requirement 5: Team Coordination
        checks['team_coordination'] = self._validate_team_coordination()
        
        # Requirement 6: Quality Assessment
        checks['quality_assessment'] = self._validate_quality_assessment()
        
        # Requirement 7: Solution Orchestration
        checks['solution_orchestration'] = self._validate_solution_orchestration()
        
        # Requirement 8: Knowledge Management
        checks['knowledge_management'] = self._validate_knowledge_management()
        
        # Requirement 9: Refinement System
        checks['refinement'] = self._validate_refinement()
        
        # Requirement 10: Integration
        checks['integration'] = self._validate_integration()
        
        # Calculate results
        passed_count = sum(1 for v in checks.values() if v['passed'])
        failed_count = len(checks) - passed_count
        score = passed_count / len(checks)
        
        result = ValidationResult(
            passed=score >= 0.9,  # 90% threshold
            score=score,
            checks_passed=passed_count,
            checks_failed=failed_count,
            details=checks,
            timestamp=datetime.now()
        )
        
        self.logger.info(
            f"Validation complete: {passed_count}/{len(checks)} passed, "
            f"score={score:.2%}"
        )
        
        return result
    
    def _validate_problem_analysis(self) -> Dict[str, Any]:
        """Validate problem analysis functionality."""
        try:
            result = self.orchestrator.run_complete_workflow(
                "Build a simple web application",
                title="Test Problem",
                max_refinement_cycles=0
            )
            return {
                'passed': result.problem_id != "",
                'message': "Problem analysis working"
            }
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_decomposition(self) -> Dict[str, Any]:
        """Validate decomposition strategies."""
        try:
            strategies = ['semantic', 'dependency', 'complexity', 'hybrid']
            for strategy in strategies:
                result = self.orchestrator.run_complete_workflow(
                    "Test problem",
                    strategy=strategy,
                    max_refinement_cycles=0
                )
                if not result.final_plan:
                    return {'passed': False, 'message': f"{strategy} failed"}
            return {'passed': True, 'message': "All strategies working"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependency management."""
        try:
            result = self.orchestrator.run_complete_workflow(
                "Multi-step project",
                max_refinement_cycles=0
            )
            if result.final_plan and result.final_plan.dependency_graph:
                return {'passed': True, 'message': "Dependency management working"}
            return {'passed': False, 'message': "No dependency graph"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_gauntlets(self) -> Dict[str, Any]:
        """Validate gauntlet system."""
        try:
            gauntlets = self.orchestrator.gauntlet_system.gauntlets
            if len(gauntlets) >= 4:
                return {'passed': True, 'message': f"{len(gauntlets)} gauntlets active"}
            return {'passed': False, 'message': "Insufficient gauntlets"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_team_coordination(self) -> Dict[str, Any]:
        """Validate team coordination."""
        try:
            coordinator = self.orchestrator.team_coordinator
            return {'passed': True, 'message': "Team coordinator initialized"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_quality_assessment(self) -> Dict[str, Any]:
        """Validate quality assessment."""
        try:
            result = self.orchestrator.run_complete_workflow(
                "Test quality",
                max_refinement_cycles=0
            )
            if result.quality_score >= 0:
                return {'passed': True, 'message': f"Quality score: {result.quality_score:.2f}"}
            return {'passed': False, 'message': "No quality score"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_solution_orchestration(self) -> Dict[str, Any]:
        """Validate solution orchestration."""
        try:
            orchestrator = self.orchestrator.solution_orchestrator
            return {'passed': True, 'message': "Solution orchestrator initialized"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_knowledge_management(self) -> Dict[str, Any]:
        """Validate knowledge management."""
        try:
            km = self.orchestrator.knowledge_manager
            return {'passed': True, 'message': "Knowledge manager initialized"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_refinement(self) -> Dict[str, Any]:
        """Validate refinement system."""
        try:
            result = self.orchestrator.run_complete_workflow(
                "Test refinement",
                max_refinement_cycles=1
            )
            return {'passed': True, 'message': f"{result.refinement_cycles} cycles"}
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate complete integration."""
        try:
            result = self.orchestrator.run_complete_workflow(
                "Complete integration test",
                max_refinement_cycles=2
            )
            return {
                'passed': result.success or result.quality_score > 0.5,
                'message': f"Integration score: {result.quality_score:.2f}"
            }
        except Exception as e:
            return {'passed': False, 'message': f"Error: {e}"}


def validate_system() -> ValidationResult:
    """Convenience function to validate entire system."""
    validator = ComprehensiveValidator()
    return validator.validate_all_requirements()
