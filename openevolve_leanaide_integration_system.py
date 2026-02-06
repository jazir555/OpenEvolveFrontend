#!/usr/bin/env python3
"""
OpenEvolve-LeanAIDE Integration System

This module provides comprehensive integration between OpenEvolve's workflow engine
and LeanAIDE's autoformalization capabilities. It extends the workflow engine to
support automatic formalization of mathematical problems throughout the evolution lifecycle.

Key Features:
- Seamless integration with OpenEvolve workflow stages
- Automatic detection and formalization of mathematical problems
- Support for decomposition, evolution, and verification stages
- Confidence-based decision making
- Comprehensive monitoring and reporting
- Error handling and fallback mechanisms

Author: OpenEvolve
Created: 2026-01-01
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

# Import OpenEvolve components
try:
    from workflow_engine import WorkflowEngine
    from workflow_structures import (
        WorkflowState, SubProblem, SolutionAttempt, VerificationReport,
        DecompositionPlan, MathematicalDomain
    )
    from workflow_stage_functions import (
        decompose_problem, generate_solution_attempt, verify_solution_attempt
    )
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    logger.warning("OpenEvolve components not available - using fallback types")
    OPENEVOLVE_AVAILABLE = False

# Import LeanAIDE bridge
try:
    from openevolve_leanaide_bridge import (
        OpenEvolveLeanAideBridge, OpenEvolveLeanAideConfig,
        AutoformalizationStage, AutoformalizationResult
    )
    LEANAIDE_BRIDGE_AVAILABLE = True
except ImportError:
    logger.warning("LeanAIDE bridge not available")
    LEANAIDE_BRIDGE_AVAILABLE = False

# REAL Lean Integration
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

@dataclass
class EnhancedWorkflowState(WorkflowState):
    """Extended workflow state with LeanAIDE autoformalization support."""
    autoformalization_enabled: bool = True
    autoformalization_results: Dict[str, Any] = field(default_factory=dict)
    mathematical_domains_detected: List[str] = field(default_factory=list)
    formal_verification_results: Dict[str, Any] = field(default_factory=dict)
    
    def enable_autoformalization(self):
        """Enable autoformalization for this workflow."""
        self.autoformalization_enabled = True
        
    def disable_autoformalization(self):
        """Disable autoformalization for this workflow."""
        self.autoformalization_enabled = False
        
    def add_autoformalization_result(self, stage: str, result: AutoformalizationResult):
        """Add an autoformalization result to the workflow state."""
        self.autoformalization_results[stage] = result.to_dict()
        if result.mathematical_domain and result.mathematical_domain not in self.mathematical_domains_detected:
            self.mathematical_domains_detected.append(result.mathematical_domain)
            
    def add_formal_verification_result(self, stage: str, result: Dict[str, Any]):
        """Add a formal verification result to the workflow state."""
        self.formal_verification_results[stage] = result


class OpenEvolveLeanAideIntegrationSystem:
    """Main integration system connecting OpenEvolve workflows with LeanAIDE."""
    
    def __init__(self, workflow_engine: Optional[WorkflowEngine] = None):
        """Initialize the integration system."""
        self.workflow_engine = workflow_engine
        self.leanaide_bridge = None
        self.setup_bridge()
        
    def setup_bridge(self):
        """Setup the LeanAIDE bridge."""
        if not LEANAIDE_BRIDGE_AVAILABLE:
            logger.warning("LeanAIDE bridge not available")
            return
            
        # Create configuration for the bridge
        config = OpenEvolveLeanAideConfig(
            autoformalization_enabled=True,
            auto_detect_math_problems=True,
            integrate_with_decomposition=True,
            integrate_with_evolution=True,
            integrate_with_verification=True,
            enable_caching=True
        )
        
        self.leanaide_bridge = OpenEvolveLeanAideBridge(config)
        logger.info("LeanAIDE bridge initialized successfully")

    def is_autoformalization_enabled(self, workflow_state: WorkflowState) -> bool:
        """Check if autoformalization is enabled for a workflow."""
        if isinstance(workflow_state, EnhancedWorkflowState):
            return workflow_state.autoformalization_enabled
        return True  # Default to enabled

    async def enhanced_decompose_problem(
        self,
        workflow_state: WorkflowState,
        problem_statement: str
    ) -> DecompositionPlan:
        """Enhanced decomposition with LeanAIDE autoformalization."""
        # Perform standard decomposition
        decomposition_plan = await decompose_problem(workflow_state, problem_statement)
        
        # Check if autoformalization is enabled
        if not self.is_autoformalization_enabled(workflow_state):
            return decomposition_plan
            
        # Integrate LeanAIDE autoformalization
        if self.leanaide_bridge:
            enhanced_plan = await self.leanaide_bridge.integrate_with_decomposition(
                decomposition_plan, workflow_state.__dict__ if hasattr(workflow_state, '__dict__') else {}
            )
            return enhanced_plan
            
        return decomposition_plan

    async def enhanced_generate_solution_attempt(
        self,
        workflow_state: WorkflowState,
        subproblem: SubProblem
    ) -> SolutionAttempt:
        """Enhanced solution generation with LeanAIDE autoformalization."""
        # Perform standard solution generation
        solution_attempt = await generate_solution_attempt(workflow_state, subproblem)
        
        # Check if autoformalization is enabled
        if not self.is_autoformalization_enabled(workflow_state):
            return solution_attempt
            
        # Integrate LeanAIDE autoformalization
        if self.leanaide_bridge:
            enhanced_solution = await self.leanaide_bridge.integrate_with_evolution(
                solution_attempt, workflow_state.__dict__ if hasattr(workflow_state, '__dict__') else {}
            )
            return enhanced_solution
            
        return solution_attempt

    async def enhanced_verify_solution_attempt(
        self,
        workflow_state: WorkflowState,
        solution_attempt: SolutionAttempt,
        subproblem: SubProblem
    ) -> VerificationReport:
        """Enhanced verification with LeanAIDE formal verification."""
        # Perform standard verification
        verification_report = await verify_solution_attempt(workflow_state, solution_attempt, subproblem)
        
        # Check if autoformalization is enabled
        if not self.is_autoformalization_enabled(workflow_state):
            return verification_report
            
        # Integrate LeanAIDE verification
        if self.leanaide_bridge:
            enhanced_report = await self.leanaide_bridge.integrate_with_verification(
                verification_report, {
                    'original_problem': subproblem.description,
                    'solution_attempt': solution_attempt,
                    'workflow_state': workflow_state.__dict__ if hasattr(workflow_state, '__dict__') else {}
                }
            )
            return enhanced_report
            
        return verification_report

    async def autoformalize_workflow_stage(
        self,
        workflow_state: WorkflowState,
        stage: AutoformalizationStage,
        stage_data: Any
    ) -> WorkflowState:
        """Autoformalize a specific workflow stage."""
        if not self.is_autoformalization_enabled(workflow_state):
            return workflow_state
            
        if self.leanaide_bridge:
            workflow_dict = workflow_state.__dict__ if hasattr(workflow_state, '__dict__') else {}
            workflow_dict['stage_data'] = stage_data
            
            result = await self.leanaide_bridge.autoformalize_workflow_stage(
                workflow_dict, stage
            )
            
            # Update workflow state with autoformalization results
            if isinstance(workflow_state, EnhancedWorkflowState):
                for stage_name, stage_result in result.get('autoformalization_results', {}).items():
                    if isinstance(stage_result, AutoformalizationResult):
                        workflow_state.add_autoformalization_result(stage_name, stage_result)
            
        return workflow_state

    async def create_comprehensive_autoformalization_report(
        self,
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Create a comprehensive autoformalization report for the workflow."""
        if not self.leanaide_bridge:
            return {
                'error': 'LeanAIDE bridge not available',
                'timestamp': time.time()
            }
            
        workflow_dict = workflow_state.__dict__ if hasattr(workflow_state, '__dict__') else {}
        return self.leanaide_bridge.create_autoformalization_report(workflow_dict)

    def get_autoformalization_strategy_recommendation(
        self,
        problem_text: str,
        workflow_stage: str
    ) -> str:
        """Get strategy recommendation for autoformalization."""
        if not self.leanaide_bridge:
            return "adaptive"  # Default strategy
            
        strategy = self.leanaide_bridge.get_autoformalization_strategy_recommendation(
            problem_text, workflow_stage
        )
        return strategy.name

    async def autoformalize_and_verify_workflow(
        self,
        workflow_state: WorkflowState,
        problem_statement: str
    ) -> Dict[str, Any]:
        """Complete autoformalization and verification workflow."""
        if not self.is_autoformalization_enabled(workflow_state):
            return {
                'status': 'disabled',
                'message': 'Autoformalization is disabled for this workflow'
            }
            
        if not self.leanaide_bridge:
            return {
                'status': 'error',
                'message': 'LeanAIDE bridge not available'
            }
            
        # Step 1: Autoformalize the problem
        autoformalization_result = await self.leanaide_bridge.autoformalize_problem(problem_statement)
        
        if not autoformalization_result.success:
            return {
                'status': 'autoformalization_failed',
                'result': autoformalization_result.to_dict(),
                'errors': autoformalization_result.errors
            }
            
        # Step 2: Verify the formalized solution
        if autoformalization_result.lean_code:
            verification_result = await self.leanaide_bridge.verify_formalized_solution(
                problem_text=problem_statement,
                lean_code=autoformalization_result.lean_code
            )
            
            return {
                'status': 'completed',
                'autoformalization': autoformalization_result.to_dict(),
                'verification': {
                    'success': verification_result.success,
                    'confidence_score': verification_result.confidence_score,
                    'errors': verification_result.errors,
                    'warnings': verification_result.warnings
                }
            }
            
        return {
            'status': 'partial',
            'autoformalization': autoformalization_result.to_dict(),
            'message': 'No Lean code generated for verification'
        }

    def create_enhanced_workflow_state(
        self,
        original_problem: str,
        workflow_id: Optional[str] = None
    ) -> EnhancedWorkflowState:
        """Create an enhanced workflow state with LeanAIDE support."""
        return EnhancedWorkflowState(
            problem_statement=original_problem,
            workflow_id=workflow_id or f"workflow_{int(time.time())}",
            current_stage="initialization",
            subproblems=[],
            solution_attempts=[],
            verification_reports=[],
            autoformalization_enabled=True,
            autoformalization_results={},
            mathematical_domains_detected=[],
            formal_verification_results={}
        )

    async def run_enhanced_workflow(
        self,
        problem_statement: str,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a complete enhanced workflow with LeanAIDE integration."""
        # Create enhanced workflow state
        workflow_state = self.create_enhanced_workflow_state(problem_statement, workflow_id)
        
        # Initialize workflow engine if not provided
        if not self.workflow_engine:
            from workflow_engine import WorkflowEngine
            self.workflow_engine = WorkflowEngine()
            
        # Run the workflow with LeanAIDE integration
        try:
            # Stage 1: Decomposition with autoformalization
            decomposition_plan = await self.enhanced_decompose_problem(workflow_state, problem_statement)
            workflow_state.decomposition_plan = decomposition_plan
            
            # Stage 2: Process each subproblem
            for subproblem in decomposition_plan.subproblems:
                # Generate solution attempt with autoformalization
                solution_attempt = await self.enhanced_generate_solution_attempt(workflow_state, subproblem)
                workflow_state.solution_attempts.append(solution_attempt)
                
                # Verify solution with formal verification
                verification_report = await self.enhanced_verify_solution_attempt(
                    workflow_state, solution_attempt, subproblem
                )
                workflow_state.verification_reports.append(verification_report)
                
            # Stage 3: Create comprehensive report
            autoformalization_report = await self.create_comprehensive_autoformalization_report(workflow_state)
            
            return {
                'status': 'completed',
                'workflow_state': workflow_state.__dict__,
                'decomposition_plan': decomposition_plan.__dict__ if hasattr(decomposition_plan, '__dict__') else decomposition_plan,
                'autoformalization_report': autoformalization_report,
                'solution_attempts': [sa.__dict__ if hasattr(sa, '__dict__') else sa for sa in workflow_state.solution_attempts],
                'verification_reports': [vr.__dict__ if hasattr(vr, '__dict__') else vr for vr in workflow_state.verification_reports]
            }
            
        except (ConnectionError, TimeoutError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Enhanced workflow failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'workflow_state': workflow_state.__dict__ if hasattr(workflow_state, '__dict__') else {}
            }

    def integrate_with_existing_workflow_engine(self, workflow_engine: WorkflowEngine):
        """Integrate LeanAIDE capabilities with an existing workflow engine."""
        self.workflow_engine = workflow_engine
        
        # Monkey-patch the workflow engine methods to add LeanAIDE integration
        original_decompose = workflow_engine.decompose_problem
        original_generate = workflow_engine.generate_solution_attempt
        original_verify = workflow_engine.verify_solution_attempt
        
        async def enhanced_decompose(workflow_state, problem_statement):
            result = await original_decompose(workflow_state, problem_statement)
            if self.is_autoformalization_enabled(workflow_state):
                result = await self.enhanced_decompose_problem(workflow_state, problem_statement)
            return result
            
        async def enhanced_generate(workflow_state, subproblem):
            result = await original_generate(workflow_state, subproblem)
            if self.is_autoformalization_enabled(workflow_state):
                result = await self.enhanced_generate_solution_attempt(workflow_state, subproblem)
            return result
            
        async def enhanced_verify(workflow_state, solution_attempt, subproblem):
            result = await original_verify(workflow_state, solution_attempt, subproblem)
            if self.is_autoformalization_enabled(workflow_state):
                result = await self.enhanced_verify_solution_attempt(workflow_state, solution_attempt, subproblem)
            return result
            
        # Replace the original methods
        workflow_engine.decompose_problem = enhanced_decompose
        workflow_engine.generate_solution_attempt = enhanced_generate
        workflow_engine.verify_solution_attempt = enhanced_verify
        
        logger.info("Successfully integrated LeanAIDE capabilities with existing workflow engine")

    def create_workflow_monitoring_hooks(self) -> Dict[str, Callable]:
        """Create monitoring hooks for workflow events."""
        if not self.leanaide_bridge:
            return {}
            
        hooks = {
            'on_decomposition_complete': self._on_decomposition_complete,
            'on_solution_generated': self._on_solution_generated,
            'on_verification_complete': self._on_verification_complete,
            'on_workflow_complete': self._on_workflow_complete
        }
        
        return hooks
        
    async def _on_decomposition_complete(self, workflow_state: WorkflowState, decomposition_plan: DecompositionPlan):
        """Handle decomposition complete event."""
        if self.is_autoformalization_enabled(workflow_state):
            logger.info("Decomposition complete - running LeanAIDE autoformalization")
            enhanced_plan = await self.enhanced_decompose_problem(workflow_state, workflow_state.problem_statement)
            return enhanced_plan
        return decomposition_plan
        
    async def _on_solution_generated(self, workflow_state: WorkflowState, solution_attempt: SolutionAttempt):
        """Handle solution generated event."""
        if self.is_autoformalization_enabled(workflow_state):
            logger.info("Solution generated - running LeanAIDE autoformalization")
            # Find the corresponding subproblem
            subproblem = None
            for sp in workflow_state.decomposition_plan.subproblems:
                if sp.id == solution_attempt.subproblem_id:
                    subproblem = sp
                    break
            
            if subproblem:
                enhanced_solution = await self.enhanced_generate_solution_attempt(workflow_state, subproblem)
                return enhanced_solution
        return solution_attempt
        
    async def _on_verification_complete(self, workflow_state: WorkflowState, verification_report: VerificationReport):
        """Handle verification complete event."""
        if self.is_autoformalization_enabled(workflow_state):
            logger.info("Verification complete - running LeanAIDE formal verification")
            # Find the corresponding solution attempt
            solution_attempt = None
            for sa in workflow_state.solution_attempts:
                if sa.id == verification_report.solution_attempt_id:
                    solution_attempt = sa
                    break
            
            if solution_attempt:
                # Find the corresponding subproblem
                subproblem = None
                for sp in workflow_state.decomposition_plan.subproblems:
                    if sp.id == solution_attempt.subproblem_id:
                        subproblem = sp
                        break
                
                if subproblem:
                    enhanced_report = await self.enhanced_verify_solution_attempt(workflow_state, solution_attempt, subproblem)
                    return enhanced_report
        return verification_report
        
    async def _on_workflow_complete(self, workflow_state: WorkflowState):
        """Handle workflow complete event."""
        if self.is_autoformalization_enabled(workflow_state):
            logger.info("Workflow complete - generating comprehensive autoformalization report")
            report = await self.create_comprehensive_autoformalization_report(workflow_state)
            return report
        return {}

    def verify_with_lean(self, workflow_or_node) -> Dict[str, Any]:
        """
        REAL Lean verification for system-wide workflows.
        
        Args:
            workflow_or_node: Workflow component or node to verify
            
        Returns:
            Dictionary with verification results
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "error": "Lean not available"}
        
        try:
            client = LeanAideClient()
            formalized = client.autoformalize(str(workflow_or_node))
            return client.verify(formalized)
        except Exception as e:
            logger.warning(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}


# Global integration system instance
_openevolve_leanaide_integration = None


def get_openevolve_leanaide_integration(workflow_engine: Optional[WorkflowEngine] = None) -> OpenEvolveLeanAideIntegrationSystem:
    """Get the global OpenEvolve-LeanAIDE integration system instance."""
    global _openevolve_leanaide_integration
    if not _openevolve_leanaide_integration:
        _openevolve_leanaide_integration = OpenEvolveLeanAideIntegrationSystem(workflow_engine)
    return _openevolve_leanaide_integration


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize the integration system
        integration_system = get_openevolve_leanaide_integration()
        
        # Example problem
        problem = "Prove that the sum of the first n odd numbers equals n²"
        
        # Run enhanced workflow
        result = await integration_system.run_enhanced_workflow(problem)
        
        print(f"Enhanced Workflow Result:")
        print(f"Status: {result['status']}")
        print(f"Autoformalization Report: {result['autoformalization_report']}")
        
    asyncio.run(main())