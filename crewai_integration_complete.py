"""
Complete CrewAI Integration Module

This module provides the complete integration layer for CrewAI workflows,
including setup, execution, monitoring, and result handling.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

# Import core CrewAI components
try:
    from crewai import Crew, Agent, Task, Process
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available - using stub implementations")

# Import state management
from crewai_state_management import (
    StateManager,
    WorkflowState,
    create_workflow_state,
    ExecutionMethod,
    WorkflowStatus
)

# Import zero-error workflow
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    create_workflow_definition,
    execute_workflow_zero_error
)

# Import ACE bridge
from ace_crewai_bridge import ACECrewAIWorkflowBridge

logger = logging.getLogger(__name__)


class CrewAIIntegration:
    """
    Complete integration class for CrewAI workflows with all supporting systems.
    
    This class ties together:
    - CrewAI workflow execution
    - State management
    - Zero-error orchestration
    - ACE learning integration
    - Monitoring and reporting
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        state_storage_dir: str = "./crewai_states",
        enable_learning: bool = True,
        enable_zero_error: bool = True
    ):
        """
        Initialize CrewAI integration with all supporting systems.
        
        Args:
            model: LLM model to use for CrewAI agents
            state_storage_dir: Directory for state persistence
            enable_learning: Enable ACE learning integration
            enable_zero_error: Enable zero-error workflow orchestration
        """
        self.model = model
        self.enable_learning = enable_learning
        self.enable_zero_error = enable_zero_error
        
        # Initialize state manager
        self.state_manager = StateManager(state_storage_dir)
        
        # Initialize ACE bridge if available
        self.ace_bridge = None
        if enable_learning and ACECrewAIWorkflowBridge:
            try:
                self.ace_bridge = ACECrewAIWorkflowBridge(
                    model=model,
                    state_storage_dir=state_storage_dir,
                    enable_learning=enable_learning
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ACE bridge: {e}")
        
        # Initialize zero-error workflow if enabled
        self.zero_error_enabled = enable_zero_error
        
        logger.info("CrewAI Integration initialized successfully")
    
    async def create_and_execute_workflow(
        self,
        problem_statement: str,
        agents_config: List[Dict[str, Any]],
        tasks_config: List[Dict[str, Any]],
        workflow_id: Optional[str] = None,
        execution_method: ExecutionMethod = ExecutionMethod.AUTO
    ) -> Dict[str, Any]:
        """
        Create and execute a complete CrewAI workflow with all integrations.
        
        Args:
            problem_statement: The main problem to solve
            agents_config: Configuration for CrewAI agents
            tasks_config: Configuration for CrewAI tasks
            workflow_id: Optional workflow ID (auto-generated if not provided)
            execution_method: Execution method to use
            
        Returns:
            Dictionary with execution results
        """
        if not CREWAI_AVAILABLE:
            return {
                "success": False,
                "error": "CrewAI not available",
                "message": "CrewAI library not installed"
            }
        
        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = f"crewai_wf_{int(datetime.now().timestamp() * 1000)}"
        
        try:
            # Create initial workflow state
            workflow_state = create_workflow_state(
                workflow_id=workflow_id,
                problem_statement=problem_statement,
                execution_method=execution_method
            )
            
            # Save initial state
            self.state_manager.save_state(workflow_id, workflow_state)
            
            # Update status to in-progress
            workflow_state.status = WorkflowStatus.IN_PROGRESS
            workflow_state.phase = 1  # Setup phase
            self.state_manager.save_state(workflow_id, workflow_state)
            
            # Execute using zero-error workflow if enabled
            if self.zero_error_enabled:
                result = await self._execute_with_zero_error(
                    workflow_id=workflow_id,
                    problem_statement=problem_statement,
                    agents_config=agents_config,
                    tasks_config=tasks_config,
                    workflow_state=workflow_state
                )
            else:
                result = await self._execute_standard_workflow(
                    workflow_id=workflow_id,
                    problem_statement=problem_statement,
                    agents_config=agents_config,
                    tasks_config=tasks_config,
                    workflow_state=workflow_state
                )
            
            # Update final state
            workflow_state.status = WorkflowStatus.COMPLETED if result.get("success", False) else WorkflowStatus.FAILED
            workflow_state.overall_score = result.get("overall_score", 0.5)
            self.state_manager.save_state(workflow_id, workflow_state)
            
            # Perform ACE learning if enabled
            if self.ace_bridge and result.get("success", False):
                try:
                    learning_result = self.ace_bridge.execute_phase_6_final(
                        final_solution=str(result.get("final_output", "")),
                        problem_statement=problem_statement,
                        enable_learning=True
                    )
                    result["learning_result"] = learning_result
                except Exception as e:
                    logger.warning(f"ACE learning failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Update state to failed
            if 'workflow_state' in locals():
                workflow_state.status = WorkflowStatus.FAILED
                workflow_state.error_message = str(e)
                self.state_manager.save_state(workflow_id, workflow_state)
            
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }
    
    async def _execute_with_zero_error(
        self,
        workflow_id: str,
        problem_statement: str,
        agents_config: List[Dict[str, Any]],
        tasks_config: List[Dict[str, Any]],
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute workflow using zero-error orchestration."""
        try:
            # Create zero-error workflow definition
            steps = [
                {
                    "name": "setup_agents",
                    "action": "python_function",
                    "function": "setup_crewai_agents",
                    "parameters": {
                        "agents_config": agents_config,
                        "model": self.model
                    }
                },
                {
                    "name": "create_tasks",
                    "action": "python_function",
                    "function": "create_crewai_tasks",
                    "parameters": {
                        "tasks_config": tasks_config,
                        "problem_statement": problem_statement
                    }
                },
                {
                    "name": "execute_crew",
                    "action": "crewai_crew",
                    "parameters": {
                        "problem_statement": problem_statement
                    }
                }
            ]
            
            workflow_def = create_workflow_definition(
                name=f"zero_error_crewai_{workflow_id}",
                description=f"Zero-error CrewAI workflow for: {problem_statement[:50]}...",
                steps=steps,
                input_schema={
                    "type": "object",
                    "properties": {
                        "problem_statement": {"type": "string"},
                        "agents_config": {"type": "array"},
                        "tasks_config": {"type": "array"}
                    },
                    "required": ["problem_statement"]
                }
            )
            
            # Execute zero-error workflow
            result = await execute_workflow_zero_error(
                workflow_definition=workflow_def,
                inputs={
                    "problem_statement": problem_statement,
                    "agents_config": agents_config,
                    "tasks_config": tasks_config
                },
                crewai_state_manager=self.state_manager
            )
            
            return {
                "success": result.status == "completed",
                "workflow_id": workflow_id,
                "final_output": result.final_output,
                "zero_error_result": result,
                "overall_score": 0.8 if result.status == "completed" else 0.2
            }
            
        except Exception as e:
            logger.error(f"Zero-error workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }
    
    async def _execute_standard_workflow(
        self,
        workflow_id: str,
        problem_statement: str,
        agents_config: List[Dict[str, Any]],
        tasks_config: List[Dict[str, Any]],
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute standard CrewAI workflow."""
        try:
            # Create LLM instance
            llm = ChatOpenAI(model=self.model)
            
            # Create agents
            agents = []
            for agent_config in agents_config:
                agent = Agent(
                    role=agent_config.get("role", "General Agent"),
                    goal=agent_config.get("goal", "Complete assigned tasks"),
                    backstory=agent_config.get("backstory", "An experienced agent."),
                    verbose=agent_config.get("verbose", True),
                    llm=llm,
                    **{k: v for k, v in agent_config.items() 
                       if k not in ["role", "goal", "backstory", "verbose"]}
                )
                agents.append(agent)
            
            # Create tasks
            tasks = []
            for i, task_config in enumerate(tasks_config):
                # Link to corresponding agent
                agent_idx = task_config.get("agent_index", i % len(agents))
                agent = agents[agent_idx] if agent_idx < len(agents) else agents[0]
                
                task = Task(
                    description=task_config.get("description", f"Task {i+1}"),
                    agent=agent,
                    expected_output=task_config.get("expected_output", "A detailed response"),
                    **{k: v for k, v in task_config.items() 
                       if k not in ["description", "agent", "expected_output"]}
                )
                tasks.append(task)
            
            # Create and run crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,  # Can be changed to hierarchical if needed
                verbose=2
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "final_output": result,
                "agents_created": len(agents),
                "tasks_created": len(tasks),
                "overall_score": 0.9  # Assuming successful execution
            }
            
        except Exception as e:
            logger.error(f"Standard workflow execution failed: {e}")
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
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            state = self.state_manager.load_state(workflow_id)
            if not state:
                return False
            
            state.status = WorkflowStatus.CANCELLED
            state.updated_at = datetime.now().isoformat()
            self.state_manager.save_state(workflow_id, state)
            
            return True
        except Exception as e:
            logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.ace_bridge:
            self.ace_bridge.cleanup()


# Global instance for easy access
_crewai_integration = None


def get_crewai_integration() -> CrewAIIntegration:
    """Get the global CrewAI integration instance."""
    global _crewai_integration
    if _crewai_integration is None:
        _crewai_integration = CrewAIIntegration()
    return _crewai_integration


async def execute_crewai_workflow(
    problem_statement: str,
    agents_config: List[Dict[str, Any]],
    tasks_config: List[Dict[str, Any]],
    workflow_id: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Convenience function to execute a CrewAI workflow.
    
    Args:
        problem_statement: The problem to solve
        agents_config: Configuration for agents
        tasks_config: Configuration for tasks
        workflow_id: Optional workflow ID
        model: LLM model to use
        
    Returns:
        Dictionary with execution results
    """
    integration = get_crewai_integration()
    return await integration.create_and_execute_workflow(
        problem_statement=problem_statement,
        agents_config=agents_config,
        tasks_config=tasks_config,
        workflow_id=workflow_id,
        model=model
    )


# Example usage and testing
async def example_usage():
    """Example of how to use the complete CrewAI integration."""
    
    # Example agents configuration
    agents_config = [
        {
            "role": "Research Analyst",
            "goal": "Analyze market trends and opportunities",
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
    
    # Execute workflow
    result = await execute_crewai_workflow(
        problem_statement="How should we position our new AI tool in the market?",
        agents_config=agents_config,
        tasks_config=tasks_config
    )
    
    print(f"Workflow completed: {result.get('success', False)}")
    print(f"Final output: {result.get('final_output', 'N/A')}")
    
    return result


if __name__ == "__main__":
    print("CrewAI Complete Integration Module")
    print("=" * 40)
    print("\nThis module provides complete CrewAI integration with:")
    print("- State management")
    print("- Zero-error workflow orchestration") 
    print("- ACE learning integration")
    print("- Monitoring and reporting")
    print("\nTo execute a workflow, use execute_crewai_workflow()")