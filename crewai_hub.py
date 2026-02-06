"""
CrewAI Integration Hub - Complete Integration Module

This module provides the central hub for all CrewAI integrations in the OpenEvolve system.
It consolidates all the various CrewAI components and provides a unified interface.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Import all CrewAI related components
from crewai_integration_complete import CrewAIIntegration, execute_crewai_workflow
from crewai_state_management import StateManager, WorkflowState, ExecutionMethod, WorkflowStatus
from crewai_zero_error_workflow import ZeroErrorWorkflow, create_workflow_definition
from ace_crewai_bridge import ACECrewAIWorkflowBridge
from crewai_client import CrewAIClient, CrewAIMonitor, create_crewai_client, create_crewai_monitor
from crewai_unified_flow import CrewAIUnifiedFlow, ExecutionMethod as FlowExecutionMethod
from bubblelabs_maker_integration import CrewAIDelegationManager

logger = logging.getLogger(__name__)


class CrewAIHub:
    """
    Central hub for all CrewAI integrations in OpenEvolve.
    
    This class provides a unified interface to all CrewAI components and ensures
    consistent behavior across the entire system.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        state_storage_dir: str = "./crewai_states",
        enable_learning: bool = True,
        enable_zero_error: bool = True,
        enable_persistence: bool = True
    ):
        """
        Initialize the CrewAI Hub with all integrated components.
        
        Args:
            model: LLM model to use for CrewAI agents
            state_storage_dir: Directory for state persistence
            enable_learning: Enable ACE learning integration
            enable_zero_error: Enable zero-error workflow orchestration
            enable_persistence: Enable state persistence
        """
        self.model = model
        self.state_storage_dir = state_storage_dir
        self.enable_learning = enable_learning
        self.enable_zero_error = enable_zero_error
        self.enable_persistence = enable_persistence
        
        # Initialize all components
        self.state_manager = StateManager(state_storage_dir)
        self.unified_flow = CrewAIUnifiedFlow(
            default_execution_method=FlowExecutionMethod.AUTO,
            enable_persistence=enable_persistence,
            state_storage_dir=state_storage_dir
        )
        self.client = CrewAIClient(
            state_storage_dir=state_storage_dir,
            enable_persistence=enable_persistence
        )
        self.monitor = CrewAIMonitor(client=self.client)
        self.integration = CrewAIIntegration(
            model=model,
            state_storage_dir=state_storage_dir,
            enable_learning=enable_learning,
            enable_zero_error=enable_zero_error
        )
        self.delegation_manager = CrewAIDelegationManager()
        
        # Initialize ACE bridge if available
        self.ace_bridge = None
        if enable_learning:
            try:
                self.ace_bridge = ACECrewAIWorkflowBridge(
                    model=model,
                    state_storage_dir=state_storage_dir,
                    enable_learning=enable_learning
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ACE bridge: {e}")
        
        logger.info("CrewAI Hub initialized with all components")
    
    async def execute_workflow(
        self,
        problem_statement: str,
        execution_method: Union[ExecutionMethod, FlowExecutionMethod] = ExecutionMethod.AUTO,
        agents_config: Optional[List[Dict[str, Any]]] = None,
        tasks_config: Optional[List[Dict[str, Any]]] = None,
        workflow_id: Optional[str] = None,
        enable_learning: bool = True,
        enable_zero_error: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a complete CrewAI workflow using the most appropriate method.
        
        Args:
            problem_statement: The problem to solve
            execution_method: Execution method to use
            agents_config: Configuration for agents (if using direct execution)
            tasks_config: Configuration for tasks (if using direct execution)
            workflow_id: Optional workflow ID
            enable_learning: Enable learning from execution
            enable_zero_error: Enable zero-error execution
            
        Returns:
            Dictionary with execution results
        """
        try:
            # If specific agents and tasks are provided, use direct integration
            if agents_config and tasks_config:
                result = await self.integration.create_and_execute_workflow(
                    problem_statement=problem_statement,
                    agents_config=agents_config,
                    tasks_config=tasks_config,
                    workflow_id=workflow_id,
                    execution_method=execution_method
                )
            else:
                # Otherwise, use the unified flow
                result = self.unified_flow.execute_full_workflow(
                    problem_statement=problem_statement,
                    execution_method=execution_method
                )
            
            # Apply learning if enabled and ACE bridge is available
            if enable_learning and self.ace_bridge:
                try:
                    learning_result = self.ace_bridge.execute_phase_6_final(
                        final_solution=str(result.get("final_solution", "")),
                        problem_statement=problem_statement,
                        enable_learning=True
                    )
                    result["learning_result"] = learning_result
                except Exception as e:
                    logger.warning(f"ACE learning failed: {e}")
            
            # Update workflow state
            if workflow_id and self.state_manager:
                state = self.state_manager.load_state(workflow_id)
                if state:
                    state.status = WorkflowStatus.COMPLETED if result.get("success", True) else WorkflowStatus.FAILED
                    self.state_manager.save_state(workflow_id, state)
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }
    
    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get the current state of a workflow."""
        return self.state_manager.load_state(workflow_id)
    
    def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[str]:
        """List all workflow IDs, optionally filtered by status."""
        return self.state_manager.list_workflows(status=status)
    
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a workflow."""
        state = self.get_workflow_state(workflow_id)
        if not state:
            return {"error": f"Workflow {workflow_id} not found"}
        
        # Get metrics from monitor
        monitor_metrics = self.monitor.track_workflow(workflow_id)
        
        # Combine with state information
        return {
            "workflow_id": workflow_id,
            "state_info": {
                "phase": state.phase,
                "status": state.status.value,
                "execution_method": state.execution_method.value,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "has_decomposition": state.decomposition_plan is not None,
                "has_solutions": len(state.sub_solutions) > 0,
                "has_critiques": len(state.critique_reports) > 0,
                "has_verifications": len(state.verification_results) > 0,
                "has_reassembly": state.reassembly_result is not None,
                "has_final_validation": state.final_validation is not None,
            },
            "monitor_metrics": monitor_metrics,
            "summary": self.state_manager.get_state_summary(workflow_id)
        }
    
    def delegate_task_to_crewai(
        self,
        task_name: str,
        task_description: str,
        workflow_epic_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Delegate a task to CrewAI workflow system."""
        try:
            # Create delegation using the delegation manager
            delegation = self.delegation_manager.delegate_tool_execution(
                tool_id=f"task_{int(datetime.now().timestamp())}",
                tool_name=task_name,
                input_data={"description": task_description},
                workflow_epic_id=workflow_epic_id
            )
            
            if delegation:
                return {
                    "success": True,
                    "delegation_id": delegation.delegation_id,
                    "task_id": delegation.task_id,
                    "status": delegation.status.value
                }
            else:
                return {"success": False, "error": "Delegation failed"}
                
        except Exception as e:
            logger.error(f"Task delegation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def sync_with_crewai(self) -> int:
        """Sync all delegations with CrewAI workflow system."""
        try:
            synced = self.delegation_manager.sync_from_crewai()
            return synced
        except Exception as e:
            logger.error(f"Sync with CrewAI failed: {e}")
            return 0
    
    def get_crewai_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all CrewAI components."""
        return {
            "hub": {
                "initialized": True,
                "model": self.model,
                "state_storage_dir": self.state_storage_dir,
                "features": {
                    "learning_enabled": self.enable_learning,
                    "zero_error_enabled": self.enable_zero_error,
                    "persistence_enabled": self.enable_persistence
                }
            },
            "components": {
                "state_manager": {
                    "available": self.state_manager is not None,
                    "storage_dir": self.state_storage_dir,
                    "workflows_count": len(self.list_workflows())
                },
                "unified_flow": self.unified_flow.get_status(),
                "client": self.client.get_status(),
                "monitor": self.monitor.get_metrics_summary(),
                "integration": {
                    "available": self.integration is not None,
                    "zero_error_enabled": self.enable_zero_error
                },
                "delegation_manager": {
                    "available": self.delegation_manager is not None,
                    "delegations_count": len(getattr(self.delegation_manager, 'delegations', {}))
                },
                "ace_bridge": {
                    "available": self.ace_bridge is not None,
                    "learning_enabled": self.enable_learning
                }
            }
        }
    
    def cleanup(self):
        """Clean up all resources."""
        if self.ace_bridge:
            self.ace_bridge.cleanup()
        if self.integration:
            self.integration.cleanup()


# Global instance for easy access
_crewai_hub = None


def get_crewai_hub() -> CrewAIHub:
    """Get the global CrewAI Hub instance."""
    global _crewai_hub
    if _crewai_hub is None:
        _crewai_hub = CrewAIHub()
    return _crewai_hub


async def execute_crewai_task(
    problem_statement: str,
    execution_method: Union[ExecutionMethod, FlowExecutionMethod] = ExecutionMethod.AUTO,
    agents_config: Optional[List[Dict[str, Any]]] = None,
    tasks_config: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute a CrewAI task through the hub.
    
    Args:
        problem_statement: The problem to solve
        execution_method: Execution method to use
        agents_config: Configuration for agents
        tasks_config: Configuration for tasks
        
    Returns:
        Dictionary with execution results
    """
    hub = get_crewai_hub()
    return await hub.execute_workflow(
        problem_statement=problem_statement,
        execution_method=execution_method,
        agents_config=agents_config,
        tasks_config=tasks_config
    )


def get_crewai_workflow_state(workflow_id: str) -> Optional[WorkflowState]:
    """Get the state of a specific workflow."""
    hub = get_crewai_hub()
    return hub.get_workflow_state(workflow_id)


def list_crewai_workflows(status: Optional[WorkflowStatus] = None) -> List[str]:
    """List all CrewAI workflows."""
    hub = get_crewai_hub()
    return hub.list_workflows(status)


def get_crewai_workflow_metrics(workflow_id: str) -> Dict[str, Any]:
    """Get metrics for a specific workflow."""
    hub = get_crewai_hub()
    return hub.get_workflow_metrics(workflow_id)


def delegate_to_crewai(
    task_name: str,
    task_description: str,
    workflow_epic_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Delegate a task to the CrewAI system."""
    hub = get_crewai_hub()
    return hub.delegate_task_to_crewai(
        task_name=task_name,
        task_description=task_description,
        workflow_epic_id=workflow_epic_id
    )


def sync_crewai_delegations() -> int:
    """Sync all delegations with the CrewAI system."""
    hub = get_crewai_hub()
    return hub.sync_with_crewai()


def get_crewai_status() -> Dict[str, Any]:
    """Get the status of all CrewAI components."""
    hub = get_crewai_hub()
    return hub.get_crewai_status()


# Example usage
async def example_usage():
    """Example of how to use the CrewAI Hub."""
    
    print("CrewAI Hub Example")
    print("=" * 30)
    
    # Get the hub instance
    hub = get_crewai_hub()
    
    # Example agents configuration
    agents_config = [
        {
            "role": "Research Analyst",
            "goal": "Analyze market trends for AI tools",
            "backstory": "An expert analyst with deep market knowledge.",
            "allow_delegation": False,
        },
        {
            "role": "Business Strategist", 
            "goal": "Develop business strategies based on research",
            "backstory": "A strategic thinker with business expertise.",
            "allow_delegation": False,
        }
    ]
    
    # Example tasks configuration
    tasks_config = [
        {
            "description": "Analyze the current market trends for AI tools",
            "expected_output": "A detailed report on market trends"
        },
        {
            "description": "Develop a business strategy based on the market analysis",
            "expected_output": "A comprehensive business strategy document"
        }
    ]
    
    # Execute a workflow
    result = await hub.execute_workflow(
        problem_statement="How should we position our new AI tool in the market?",
        agents_config=agents_config,
        tasks_config=tasks_config,
        execution_method=ExecutionMethod.ROMA_MDAP_MAKER
    )
    
    print(f"Workflow completed: {result.get('success', False)}")
    print(f"Status: {result.get('status', 'N/A')}")
    
    # Get hub status
    status = hub.get_crewai_status()
    print(f"Components available: {len(status['components'])}")
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())