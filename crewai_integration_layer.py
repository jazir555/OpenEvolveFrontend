"""
CrewAI Integration Layer for BubbleLab MCP Server

This module provides advanced orchestration capabilities by integrating
CrewAI with BubbleLab through the MCP protocol. It handles complex
multi-agent workflows, task delegation, and result aggregation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from uuid import uuid4
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    CREWAI_AVAILABLE = True
    logger.info("✅ CrewAI components available for integration")
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("⚠️ CrewAI not available - using mock implementations")


class CrewAIService:
    """
    Service class that handles all CrewAI integration functionality.
    
    This service provides methods for creating agents, tasks, crews,
    and managing complex orchestration workflows between BubbleLab and CrewAI.
    """
    
    def __init__(self):
        self.active_crews = {}
        self.agent_templates = {}
        self.task_templates = {}
        self.results_cache = {}
        
        # Initialize default agent templates
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default agent and task templates."""
        # Default agent templates
        self.agent_templates = {
            "researcher": {
                "role": "Senior Research Analyst",
                "goal": "Uncover cutting-edge developments in AI and data science",
                "backstory": "You are a Senior Research Analyst at a leading tech think tank."
            },
            "writer": {
                "role": "Content Writer",
                "goal": "Craft compelling content about AI advancements",
                "backstory": "You are a renowned Content Writer, known for your insightful and engaging articles."
            },
            "reviewer": {
                "role": "Project Reviewer",
                "goal": "Review and validate the work of other agents",
                "backstory": "You are a meticulous reviewer, ensuring accuracy and quality in all deliverables."
            },
            "orchestrator": {
                "role": "Workflow Orchestrator",
                "goal": "Coordinate and manage complex multi-agent workflows",
                "backstory": "You are an expert orchestrator, skilled at managing complex workflows involving multiple agents."
            }
        }
        
        # Default task templates
        self.task_templates = {
            "research": {
                "description": "Conduct thorough research on {topic}",
                "expected_output": "A detailed report on {topic} with key findings and insights"
            },
            "writing": {
                "description": "Write a comprehensive article about {topic} based on research findings",
                "expected_output": "A well-structured, engaging article about {topic}"
            },
            "review": {
                "description": "Review the {deliverable} for accuracy, quality, and completeness",
                "expected_output": "A detailed review with suggestions for improvement"
            }
        }
    
    async def create_agent_from_template(
        self, 
        template_name: str, 
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an agent using a predefined template with optional custom configuration."""
        if not CREWAI_AVAILABLE:
            return await self._create_mock_agent(template_name, custom_config)
        
        if template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.agent_templates[template_name].copy()
        
        # Apply custom configuration
        if custom_config:
            template.update(custom_config)
        
        try:
            # Create the agent
            agent = Agent(
                role=template["role"],
                goal=template["goal"],
                backstory=template["backstory"],
                verbose=True,
                allow_delegation=True
            )
            
            agent_id = str(uuid4())
            
            return {
                "id": agent_id,
                "role": template["role"],
                "goal": template["goal"],
                "backstory": template["backstory"],
                "status": "created",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating agent from template {template_name}: {e}")
            raise
    
    async def _create_mock_agent(
        self, 
        template_name: str, 
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a mock agent when CrewAI is not available."""
        template = self.agent_templates.get(template_name, {
            "role": "Mock Agent",
            "goal": "Simulate agent behavior",
            "backstory": "A mock agent for testing purposes"
        })
        
        if custom_config:
            template.update(custom_config)
        
        return {
            "id": str(uuid4()),
            "role": template["role"],
            "goal": template["goal"],
            "backstory": template["backstory"],
            "status": "mock_created",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "CrewAI not available - using mock agent"
        }
    
    async def create_custom_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom agent with fully specified configuration."""
        if not CREWAI_AVAILABLE:
            return await self._create_mock_agent("custom", config)
        
        required_fields = ["role", "goal", "backstory"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        try:
            agent = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                verbose=config.get("verbose", True),
                allow_delegation=config.get("allow_delegation", True),
                max_iter=config.get("max_iter", 15)
            )
            
            agent_id = str(uuid4())
            
            return {
                "id": agent_id,
                "role": config["role"],
                "goal": config["goal"],
                "backstory": config["backstory"],
                "status": "created",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating custom agent: {e}")
            raise
    
    async def create_task_from_template(
        self, 
        template_name: str, 
        agent_id: str, 
        task_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a task using a predefined template."""
        if template_name not in self.task_templates:
            raise ValueError(f"Task template '{template_name}' not found")
        
        template = self.task_templates[template_name].copy()
        
        # Apply task parameters to template
        if task_params:
            for key, value in task_params.items():
                template["description"] = template["description"].replace(f"{{{key}}}", str(value))
                template["expected_output"] = template["expected_output"].replace(f"{{{key}}}", str(value))
        
        task_id = str(uuid4())
        
        return {
            "id": task_id,
            "agent_id": agent_id,
            "description": template["description"],
            "expected_output": template["expected_output"],
            "status": "created",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def create_custom_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom task with fully specified configuration."""
        required_fields = ["description", "expected_output", "agent_id"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        task_id = str(uuid4())
        
        return {
            "id": task_id,
            "agent_id": config["agent_id"],
            "description": config["description"],
            "expected_output": config["expected_output"],
            "status": "created",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def create_crew(
        self, 
        agents: List[Dict[str, Any]], 
        tasks: List[Dict[str, Any]], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a crew with specified agents and tasks."""
        if not CREWAI_AVAILABLE:
            return await self._create_mock_crew(agents, tasks, config)
        
        try:
            # Create actual CrewAI agents
            crewai_agents = []
            agent_lookup = {}  # Map agent IDs to CrewAI agent objects
            
            for agent_config in agents:
                crewai_agent = Agent(
                    role=agent_config["role"],
                    goal=agent_config["goal"],
                    backstory=agent_config["backstory"],
                    verbose=True,
                    allow_delegation=True
                )
                crewai_agents.append(crewai_agent)
                agent_lookup[agent_config["id"]] = crewai_agent
            
            # Create actual CrewAI tasks
            crewai_tasks = []
            for task_config in tasks:
                agent = agent_lookup.get(task_config["agent_id"])
                if not agent:
                    raise ValueError(f"Agent with ID {task_config['agent_id']} not found")
                
                crewai_task = Task(
                    description=task_config["description"],
                    expected_output=task_config["expected_output"],
                    agent=agent
                )
                crewai_tasks.append(crewai_task)
            
            # Create the crew
            crew = Crew(
                agents=crewai_agents,
                tasks=crewai_tasks,
                verbose=2
            )
            
            crew_id = str(uuid4())
            self.active_crews[crew_id] = {
                "crew": crew,
                "agents": agents,
                "tasks": tasks,
                "config": config or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            return {
                "id": crew_id,
                "status": "created",
                "agent_count": len(agents),
                "task_count": len(tasks),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating crew: {e}")
            raise
    
    async def _create_mock_crew(
        self, 
        agents: List[Dict[str, Any]], 
        tasks: List[Dict[str, Any]], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a mock crew when CrewAI is not available."""
        crew_id = str(uuid4())
        
        self.active_crews[crew_id] = {
            "agents": agents,
            "tasks": tasks,
            "config": config or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "id": crew_id,
            "status": "mock_created",
            "agent_count": len(agents),
            "task_count": len(tasks),
            "timestamp": datetime.utcnow().isoformat(),
            "message": "CrewAI not available - using mock crew"
        }
    
    async def execute_crew(
        self, 
        crew_id: str, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a previously created crew with given inputs."""
        if crew_id not in self.active_crews:
            raise ValueError(f"Crew with ID {crew_id} not found")
        
        crew_data = self.active_crews[crew_id]
        
        if not CREWAI_AVAILABLE:
            return await self._execute_mock_crew(crew_id, inputs)
        
        try:
            crew = crew_data["crew"]
            inputs = inputs or {}
            
            # Execute the crew
            result = crew.kickoff(inputs=inputs)
            
            # Cache the result
            result_id = str(uuid4())
            self.results_cache[result_id] = {
                "crew_id": crew_id,
                "inputs": inputs,
                "result": str(result),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return {
                "result_id": result_id,
                "result": str(result),
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error executing crew {crew_id}: {e}")
            raise
    
    async def _execute_mock_crew(
        self, 
        crew_id: str, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a mock crew when CrewAI is not available."""
        inputs = inputs or {}
        
        # Simulate crew execution
        result = f"Mock execution result for crew {crew_id} with inputs: {inputs}"
        
        # Cache the result
        result_id = str(uuid4())
        self.results_cache[result_id] = {
            "crew_id": crew_id,
            "inputs": inputs,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "result_id": result_id,
            "result": result,
            "status": "mock_completed",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "CrewAI not available - using mock execution"
        }
    
    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a result by its ID."""
        return self.results_cache.get(result_id)
    
    async def get_crew_status(self, crew_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a crew."""
        if crew_id not in self.active_crews:
            return None
        
        crew_data = self.active_crews[crew_id]
        return {
            "id": crew_id,
            "status": "active",
            "agent_count": len(crew_data["agents"]),
            "task_count": len(crew_data["tasks"]),
            "created_at": crew_data["created_at"],
            "crewai_available": CREWAI_AVAILABLE
        }
    
    async def create_delegation_workflow(
        self, 
        task_description: str, 
        required_outputs: List[str], 
        constraints: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a workflow for delegating tasks to CrewAI."""
        # Create orchestrator agent
        orchestrator_agent = await self.create_agent_from_template("orchestrator")
        
        # Create a comprehensive task for the orchestrator
        task_details = {
            "task_description": task_description,
            "required_outputs": ", ".join(required_outputs),
            "constraints": ", ".join(constraints or []),
            "context": str(context or {})
        }
        
        task_config = {
            "description": (
                f"Act as a workflow orchestrator to handle this complex task: {task_description}. "
                f"Your goal is to break this down into manageable subtasks, coordinate agents to complete them, "
                f"and synthesize the results. Required outputs: {task_details['required_outputs']}. "
                f"Constraints: {task_details['constraints']}. Context: {task_details['context']}."
            ),
            "expected_output": (
                f"A comprehensive result addressing all required outputs: {task_details['required_outputs']}, "
                f"following all constraints: {task_details['constraints']}. Include a summary of the approach taken."
            ),
            "agent_id": orchestrator_agent["id"]
        }
        
        # Create the task
        task = await self.create_custom_task(task_config)
        
        # Create a crew with the orchestrator agent and the task
        crew = await self.create_crew(
            agents=[orchestrator_agent],
            tasks=[task],
            config={"delegation_workflow": True}
        )
        
        return {
            "crew_id": crew["id"],
            "orchestrator_agent_id": orchestrator_agent["id"],
            "task_id": task["id"],
            "status": "delegation_workflow_created",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup_crew(self, crew_id: str) -> bool:
        """Clean up a crew and remove it from active crews."""
        if crew_id in self.active_crews:
            del self.active_crews[crew_id]
            return True
        return False
    
    async def get_available_templates(self) -> Dict[str, Any]:
        """Get available agent and task templates."""
        return {
            "agent_templates": list(self.agent_templates.keys()),
            "task_templates": list(self.task_templates.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
crewai_service = CrewAIService()


def get_crewai_service() -> CrewAIService:
    """Get the global CrewAI service instance."""
    return crewai_service