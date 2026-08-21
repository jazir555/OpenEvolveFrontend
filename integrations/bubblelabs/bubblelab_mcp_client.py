"""
BubbleLab MCP Client for CrewAI Orchestration

This module provides the client-side implementation for BubbleLab
to communicate with the CrewAI MCP server for advanced orchestration.
"""
from __future__ import annotations


import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from enum import Enum

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BubbleLabMCPClient:
    """
    Client for BubbleLab to communicate with the CrewAI MCP server.
    
    This client allows BubbleLab to delegate complex orchestration tasks
    to CrewAI through the MCP protocol.
    """
    
    def __init__(self, server_url: str = "http://localhost:8003"):
        self.server_url = server_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the MCP server is healthy."""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except (ConnectionError, TimeoutError, IOError) as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from the MCP server."""
        try:
            response = await self.client.get(f"{self.server_url}/tools")
            response.raise_for_status()
            return response.json()["tools"]
        except (ConnectionError, TimeoutError, IOError) as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        try:
            response = await self.client.post(
                f"{self.server_url}/tools/{tool_name}",
                json={"parameters": parameters}
            )
            response.raise_for_status()
            return response.json()
        except (ConnectionError, TimeoutError, IOError) as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            return {"error": str(e), "success": False}
    
    async def create_crewai_agent(
        self, 
        role: str, 
        goal: str, 
        backstory: str,
        tools: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a CrewAI agent through the MCP server."""
        params = {
            "role": role,
            "goal": goal,
            "backstory": backstory,
        }
        
        if tools:
            params["tools"] = tools
        if template:
            params["template"] = template
        
        return await self.call_tool("create_crewai_agent", params)
    
    async def create_crewai_task(
        self,
        description: str,
        expected_output: str,
        agent_role: str,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a CrewAI task through the MCP server."""
        params = {
            "description": description,
            "expected_output": expected_output,
            "agent_role": agent_role,
        }
        
        if template:
            params["template"] = template
        
        return await self.call_tool("create_crewai_task", params)
    
    async def execute_crewai_crew(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a CrewAI crew through the MCP server."""
        params = {
            "agents": agents,
            "tasks": tasks,
            "inputs": inputs or {}
        }
        
        return await self.call_tool("execute_crewai_crew", params)
    
    async def delegate_to_crewai(
        self,
        task_description: str,
        required_outputs: List[str],
        constraints: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Delegate a complex task to CrewAI for orchestration."""
        params = {
            "task_description": task_description,
            "required_outputs": required_outputs,
            "constraints": constraints or [],
            "context": context or {}
        }
        
        return await self.call_tool("delegate_to_crewai", params)
    
    async def create_crew_via_api(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a crew directly via the API (not using a tool)."""
        try:
            response = await self.client.post(
                f"{self.server_url}/create_crew",
                json={
                    "agents": agents,
                    "tasks": tasks,
                    "config": config or {}
                }
            )
            response.raise_for_status()
            return response.json()
        except (ConnectionError, TimeoutError, IOError) as e:
            logger.error(f"Failed to create crew via API: {e}")
            return {"error": str(e), "success": False}
    
    async def execute_crew_by_id(
        self,
        crew_id: str,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a crew by its ID."""
        try:
            response = await self.client.post(
                f"{self.server_url}/execute_crew/{crew_id}",
                json={"inputs": inputs or {}}
            )
            response.raise_for_status()
            return response.json()
        except (ConnectionError, TimeoutError, IOError) as e:
            logger.error(f"Failed to execute crew {crew_id}: {e}")
            return {"error": str(e), "success": False}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class BubbleLabCrewAIBridge:
    """
    Bridge class that provides high-level orchestration capabilities
    for BubbleLab to integrate with CrewAI.
    """
    
    def __init__(self, mcp_client: BubbleLabMCPClient):
        self.client = mcp_client
    
    async def create_research_workflow(
        self,
        topic: str,
        research_depth: int = 3,
        additional_constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a research-focused workflow using CrewAI."""
        # Create agents
        researcher_agent = await self.client.create_crewai_agent(
            role="Senior Research Analyst",
            goal=f"Uncover cutting-edge developments in {topic}",
            backstory="You are a Senior Research Analyst at a leading tech think tank."
        )
        
        writer_agent = await self.client.create_crewai_agent(
            role="Content Writer",
            goal=f"Craft compelling content about {topic}",
            backstory="You are a renowned Content Writer, known for your insightful and engaging articles."
        )
        
        reviewer_agent = await self.client.create_crewai_agent(
            role="Project Reviewer",
            goal="Review and validate the work of other agents",
            backstory="You are a meticulous reviewer, ensuring accuracy and quality in all deliverables."
        )
        
        # Create tasks
        research_task = await self.client.create_crewai_task(
            description=f"Conduct thorough research on {topic} with depth level {research_depth}",
            expected_output=f"A detailed report on {topic} with key findings and insights",
            agent_role="Senior Research Analyst"
        )
        
        writing_task = await self.client.create_crewai_task(
            description=f"Write a comprehensive article about {topic} based on research findings",
            expected_output=f"A well-structured, engaging article about {topic}",
            agent_role="Content Writer"
        )
        
        review_task = await self.client.create_crewai_task(
            description=f"Review the research report and article for accuracy and quality",
            expected_output="A detailed review with suggestions for improvement",
            agent_role="Project Reviewer"
        )
        
        # Execute the crew
        crew_result = await self.client.execute_crewai_crew(
            agents=[researcher_agent, writer_agent, reviewer_agent],
            tasks=[research_task, writing_task, review_task],
            inputs={"topic": topic, "depth": research_depth}
        )
        
        return crew_result
    
    async def create_custom_orchestration(
        self,
        task_description: str,
        required_outputs: List[str],
        agent_configs: List[Dict[str, Any]],
        task_configs: List[Dict[str, Any]],
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a custom orchestration with specified agents and tasks."""
        # Create agents
        created_agents = []
        for agent_config in agent_configs:
            agent = await self.client.create_crewai_agent(**agent_config)
            created_agents.append(agent)
        
        # Create tasks
        created_tasks = []
        for task_config in task_configs:
            task = await self.client.create_crewai_task(**task_config)
            created_tasks.append(task)
        
        # Execute the crew
        crew_result = await self.client.execute_crewai_crew(
            agents=created_agents,
            tasks=created_tasks,
            inputs={
                "task_description": task_description,
                "required_outputs": required_outputs
            }
        )
        
        return crew_result
    
    async def delegate_complex_task(
        self,
        task_description: str,
        required_outputs: List[str],
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delegate a complex task to CrewAI for autonomous orchestration."""
        result = await self.client.delegate_to_crewai(
            task_description=task_description,
            required_outputs=required_outputs,
            context=context,
            constraints=constraints
        )
        
        return result
    
    async def run_multi_stage_workflow(
        self,
        stages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run a multi-stage workflow where each stage feeds into the next."""
        results = []
        context = {}
        
        for i, stage in enumerate(stages):
            logger.info(f"Executing stage {i+1}/{len(stages)}: {stage.get('name', 'Unnamed Stage')}")
            
            # Add previous results to context
            if results:
                context[f"previous_stage_{i}"] = results[-1]
            
            # Execute the stage
            stage_result = await self.delegate_complex_task(
                task_description=stage["task_description"],
                required_outputs=stage["required_outputs"],
                context={**context, **stage.get("context", {})}
            )
            
            results.append(stage_result)
            context["current_result"] = stage_result
        
        return {
            "results": results,
            "final_result": results[-1] if results else None,
            "stage_count": len(stages),
            "status": "completed"
        }


# Convenience function to create the bridge
async def create_bubblelab_crewai_bridge(server_url: str = "http://localhost:8003") -> BubbleLabCrewAIBridge:
    """Create a BubbleLab CrewAI bridge with the specified server URL."""
    client = BubbleLabMCPClient(server_url)
    health = await client.health_check()
    
    if health.get("status") != "healthy":
        raise ConnectionError(f"MCP server at {server_url} is not healthy: {health}")
    
    bridge = BubbleLabCrewAIBridge(client)
    return bridge


# Example usage
async def example_usage():
    """Example of how to use the BubbleLab CrewAI integration."""
    try:
        # Create the bridge
        bridge = await create_bubblelab_crewai_bridge()
        
        # Example 1: Create a research workflow
        print("Creating research workflow...")
        research_result = await bridge.create_research_workflow(
            topic="Artificial Intelligence in Healthcare",
            research_depth=2
        )
        print(f"Research result: {research_result}")
        
        # Example 2: Delegate a complex task
        print("\nDelegating complex task...")
        delegation_result = await bridge.delegate_complex_task(
            task_description="Analyze market trends for renewable energy sector",
            required_outputs=[
                "Market growth projections",
                "Key player analysis", 
                "Investment opportunities"
            ],
            constraints=["Use only publicly available data", "Focus on US market"]
        )
        print(f"Delegation result: {delegation_result}")
        
        # Example 3: Multi-stage workflow
        print("\nRunning multi-stage workflow...")
        stages = [
            {
                "name": "Data Collection",
                "task_description": "Gather relevant data about customer satisfaction metrics",
                "required_outputs": ["Key metrics", "Data sources"]
            },
            {
                "name": "Analysis",
                "task_description": "Analyze the collected data to identify trends",
                "required_outputs": ["Trends identified", "Statistical significance"]
            },
            {
                "name": "Recommendations",
                "task_description": "Provide actionable recommendations based on the analysis",
                "required_outputs": ["Top 5 recommendations", "Implementation timeline"]
            }
        ]
        
        workflow_result = await bridge.run_multi_stage_workflow(stages)
        print(f"Workflow result: {workflow_result}")
        
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"Error in example usage: {e}")
    finally:
        # Close the client
        if 'bridge' in locals():
            await bridge.client.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())