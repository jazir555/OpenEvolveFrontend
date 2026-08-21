"""
MCP Server for BubbleLab - CrewAI Orchestration Integration

This server implements the Model Context Protocol (MCP) to allow BubbleLab
to communicate with CrewAI for advanced orchestration capabilities.

The server provides:
- Tool discovery and execution for CrewAI agents
- Context management between BubbleLab and CrewAI
- Task delegation from BubbleLab workflows to CrewAI
- Result aggregation and return to BubbleLab
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import CrewAI components (will be handled gracefully if not available)
try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
    logger.info("[OK] CrewAI components loaded successfully")
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("[WARN] CrewAI not available - using mock implementation")

# Import our integration layer
from .crewai_integration_layer import get_crewai_service, CrewAIService


class MCPServer:
    """
    Model Context Protocol Server for BubbleLab - CrewAI integration.
    
    This server allows BubbleLab to delegate complex orchestration tasks to CrewAI
    through standardized MCP communication.
    """
    
    def __init__(self):
        self.app = FastAPI(title="BubbleLab CrewAI MCP Server")
        self.sessions = {}  # Store active sessions
        self.tools = {}     # Store available tools
        self.crews = {}     # Store active crews
        self.crewai_service = get_crewai_service()  # Get the CrewAI service
        self.setup_routes()

        # Register default tools
        self.register_default_tools()
    
    def setup_routes(self):
        """Set up the MCP server routes."""
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "crewai_available": CREWAI_AVAILABLE
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """Return list of available tools that can be called by BubbleLab."""
            return {
                "tools": [
                    {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                    for name, tool in self.tools.items()
                ]
            }
        
        @self.app.post("/tools/{tool_name}")
        async def call_tool(tool_name: str, request: Request):
            """Execute a specific tool with provided parameters."""
            body = await request.json()
            params = body.get("parameters", {})
            
            if tool_name not in self.tools:
                return {"error": f"Tool '{tool_name}' not found"}
            
            try:
                result = await self.execute_tool(tool_name, params)
                return {"result": result, "success": True}
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {"error": str(e), "success": False}
        
        @self.app.post("/create_crew")
        async def create_crew(request: Request):
            """Create a new CrewAI crew for orchestration."""
            body = await request.json()
            crew_id = str(uuid4())
            
            try:
                crew = await self.create_crew_from_spec(body)
                self.crews[crew_id] = crew
                return {"crew_id": crew_id, "status": "created"}
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error creating crew: {e}")
                return {"error": str(e), "success": False}
        
        @self.app.post("/execute_crew/{crew_id}")
        async def execute_crew(crew_id: str, request: Request):
            """Execute a specific crew with provided inputs."""
            if crew_id not in self.crews:
                return {"error": f"Crew '{crew_id}' not found"}
            
            body = await request.json()
            inputs = body.get("inputs", {})
            
            try:
                result = await self.execute_crew_by_id(crew_id, inputs)
                return {"result": result, "success": True}
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error executing crew {crew_id}: {e}")
                return {"error": str(e), "success": False}
    
    def register_default_tools(self):
        """Register default tools for BubbleLab integration."""
        # Tool for creating CrewAI agents
        self.tools["create_crewai_agent"] = {
            "description": "Create a CrewAI agent with specified role, goal, and tools",
            "parameters": {
                "role": {"type": "string", "description": "Role of the agent"},
                "goal": {"type": "string", "description": "Goal of the agent"},
                "backstory": {"type": "string", "description": "Backstory of the agent"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools available to the agent"}
            }
        }
        
        # Tool for creating CrewAI tasks
        self.tools["create_crewai_task"] = {
            "description": "Create a CrewAI task with specified description and expected output",
            "parameters": {
                "description": {"type": "string", "description": "Description of the task"},
                "expected_output": {"type": "string", "description": "Expected output of the task"},
                "agent_role": {"type": "string", "description": "Role of the agent to assign this task to"}
            }
        }
        
        # Tool for creating and executing CrewAI crews
        self.tools["execute_crewai_crew"] = {
            "description": "Create and execute a CrewAI crew with specified agents and tasks",
            "parameters": {
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "goal": {"type": "string"},
                            "backstory": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "description": "List of agents for the crew"
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "expected_output": {"type": "string"},
                            "agent_role": {"type": "string"}
                        }
                    },
                    "description": "List of tasks for the crew"
                },
                "inputs": {"type": "object", "description": "Inputs for the crew execution"}
            }
        }
        
        # Tool for delegating BubbleLab workflow to CrewAI
        self.tools["delegate_to_crewai"] = {
            "description": "Delegate a complex workflow task to CrewAI for orchestration",
            "parameters": {
                "task_description": {"type": "string", "description": "Description of the task to delegate"},
                "required_outputs": {"type": "array", "items": {"type": "string"}, "description": "Required outputs"},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints for the task"},
                "context": {"type": "object", "description": "Context to provide to the crew"}
            }
        }
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]):
        """Execute a specific tool with given parameters."""
        if tool_name == "create_crewai_agent":
            return await self._create_crewai_agent(parameters)
        elif tool_name == "create_crewai_task":
            return await self._create_crewai_task(parameters)
        elif tool_name == "execute_crewai_crew":
            return await self._execute_crewai_crew(parameters)
        elif tool_name == "delegate_to_crewai":
            return await self._delegate_to_crewai(parameters)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _create_crewai_agent(self, params: Dict[str, Any]):
        """Create a CrewAI agent."""
        try:
            # Use the integration layer to create the agent
            result = await self.crewai_service.create_agent_from_template(
                template_name=params.get("template", "researcher"),
                custom_config=params
            )
            return result
        except (ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to create agent: {str(e)}") from e
    
    async def _create_crewai_task(self, params: Dict[str, Any]):
        """Create a CrewAI task."""
        try:
            # Use the integration layer to create the task
            result = await self.crewai_service.create_task_from_template(
                template_name=params.get("template", "research"),
                agent_id=params.get("agent_id", "default"),
                task_params=params
            )
            return result
        except (ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to create task: {str(e)}") from e

    async def _execute_crewai_crew(self, params: Dict[str, Any]):
        """Execute a CrewAI crew."""
        try:
            # Use the integration layer to create and execute the crew
            agents = params.get("agents", [])
            tasks = params.get("tasks", [])
            inputs = params.get("inputs", {})

            # Create the crew
            crew_result = await self.crewai_service.create_crew(agents, tasks)
            crew_id = crew_result["id"]

            # Execute the crew
            execution_result = await self.crewai_service.execute_crew(crew_id, inputs)

            return {
                "result": execution_result["result"],
                "status": "executed",
                "crew_id": crew_id,
                "message": "Crew executed successfully"
            }
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            raise RuntimeError(f"Failed to execute crew: {str(e)}") from e

    async def _delegate_to_crewai(self, params: Dict[str, Any]):
        """Delegate a complex task to CrewAI for orchestration."""
        try:
            # Use the integration layer to create a delegation workflow
            task_description = params.get("task_description", "No task description provided")
            required_outputs = params.get("required_outputs", ["No specific outputs"])
            constraints = params.get("constraints", [])
            context = params.get("context", {})

            result = await self.crewai_service.create_delegation_workflow(
                task_description=task_description,
                required_outputs=required_outputs,
                constraints=constraints,
                context=context
            )

            # Execute the delegation workflow
            execution_result = await self.crewai_service.execute_crew(
                result["crew_id"],
                context
            )

            return {
                "result": execution_result["result"],
                "status": "delegated_and_executed",
                "crew_id": result["crew_id"],
                "message": "Task delegated to CrewAI and executed successfully"
            }
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            raise RuntimeError(f"Failed to delegate task: {str(e)}") from e
    
    async def create_crew_from_spec(self, spec: Dict[str, Any]):
        """Create a crew from specification."""
        try:
            # Use the integration layer to create the crew
            agents = spec.get("agents", [])
            tasks = spec.get("tasks", [])
            config = spec.get("config", {})

            result = await self.crewai_service.create_crew(agents, tasks, config)
            return result
        except (ValueError, TypeError, KeyError) as e:
            raise RuntimeError(f"Failed to create crew from spec: {str(e)}") from e

    async def execute_crew_by_id(self, crew_id: str, inputs: Dict[str, Any]):
        """Execute a crew by its ID."""
        try:
            # Use the integration layer to execute the crew
            result = await self.crewai_service.execute_crew(crew_id, inputs)
            return result
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            raise RuntimeError(f"Failed to execute crew {crew_id}: {str(e)}") from e
    
    def run(self, host: str = "0.0.0.0", port: int = 8003):
        """Run the MCP server."""
        logger.info(f"🚀 Starting BubbleLab CrewAI MCP Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global server instance
mcp_server = MCPServer()


def main():
    """Main entry point for the MCP server."""
    import os
    
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8003"))
    
    logger.info("BubbleLab CrewAI MCP Server starting...")
    logger.info(f"Server will run on {host}:{port}")
    logger.info(f"CrewAI available: {CREWAI_AVAILABLE}")
    
    mcp_server.run(host=host, port=port)


if __name__ == "__main__":
    main()