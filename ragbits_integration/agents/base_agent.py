"""
Base Workflow Agent

Provides the foundation for RAGBits-integrated agents with crewai LLM management.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgentTool:
    """
    Base class for agent tools.

    Tools are capabilities that agents can use during execution,
    such as searching knowledge, evaluating solutions, etc.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, **kwargs) -> Any:
        """Execute the tool. Must be implemented by subclasses."""
        raise NotImplementedError


class BaseWorkflowAgent(ABC):
    """
    Base class for workflow agents integrated with RAGBits and crewai.

    Provides:
    - LLM management via crewai
    - Tool management and execution
    - Prompt construction
    - Response parsing
    - A2A communication support

    Usage:
        class BlueTeamAgent(BaseWorkflowAgent):
            async def generate_solution(self, sub_problem, context):
                prompt = self._build_prompt(sub_problem, context)
                response = await self._call_llm(prompt)
                return self._parse_response(response)
    """

    # Agent roles
    ROLE_BLUE_TEAM = "blue_team"
    ROLE_RED_TEAM = "red_team"
    ROLE_GOLD_TEAM = "gold_team"
    ROLE_DECOMPOSER = "decomposer"
    ROLE_ASSEMBLER = "assembler"
    ROLE_VERIFIER = "verifier"

    def __init__(
        self,
        role: str,
        crewai_client=None,
        model_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[AgentTool]] = None,
        storage_manager=None,
        knowledge_retriever=None
    ):
        """
        Initialize the base workflow agent.

        Args:
            role: Agent role (blue_team, red_team, gold_team, etc.)
            crewai_client: crewai client for LLM access
            model_config: Model configuration (model_id, temperature, etc.)
            tools: List of tools available to the agent
            storage_manager: IntermediaryStorageManager for artifact storage
            knowledge_retriever: KnowledgeRetriever for semantic search
        """
        self.role = role
        self.crewai = crewai_client
        self.storage = storage_manager
        self.knowledge_retriever = knowledge_retriever

        # Model configuration
        self.model_config = model_config or self._get_default_model_config()

        # Tools
        self.tools = {tool.name: tool for tool in (tools or [])}

        # Agent metadata
        self.agent_id = f"{role}_{int(datetime.utcnow().timestamp())}"
        self.conversation_history = []
        self.tool_calls_made = []

        logger.info(f"Initialized {role} agent with model {self.model_config.get('model_id')}")

    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration based on agent role"""
        defaults = {
            self.ROLE_BLUE_TEAM: {
                "model_id": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            self.ROLE_RED_TEAM: {
                "model_id": "claude-sonnet",
                "temperature": 0.5,
                "max_tokens": 1500
            },
            self.ROLE_GOLD_TEAM: {
                "model_id": "gpt-4-turbo",
                "temperature": 0.3,
                "max_tokens": 1500
            }
        }
        return defaults.get(self.role, {
            "model_id": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 1000
        })

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the agent's primary task.

        Must be implemented by subclasses.

        Args:
            task: The task to execute
            context: Additional context for the task
            **kwargs: Additional arguments

        Returns:
            Result dict with response and metadata
        """
        raise NotImplementedError

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call LLM via crewai.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional LLM parameters

        Returns:
            LLM response text
        """
        if not self.crewai:
            # Fallback for testing without crewai
            logger.warning("No crewai client - returning mock response")
            return f"Mock response for: {prompt[:100]}..."

        try:
            # Merge model config with kwargs
            params = {**self.model_config, **kwargs}

            # Call crewai
            if system_prompt:
                params["system_message"] = system_prompt

            response = await self.crewai.generate(
                prompt=prompt,
                **params
            )

            # Extract text from response
            if isinstance(response, dict):
                return response.get("text", response.get("content", str(response)))
            return str(response)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _build_prompt(
        self,
        task: str,
        context: Dict[str, Any],
        include_tools: bool = True
    ) -> str:
        """
        Build a prompt from task and context.

        Args:
            task: The task description
            context: Context information
            include_tools: Whether to include available tools

        Returns:
            Formatted prompt string
        """
        parts = []

        # Add role-specific system prompt
        parts.append(self._get_system_prompt())

        # Add task
        parts.append(f"\n# Task")
        parts.append(task)

        # Add context
        if context:
            parts.append(f"\n# Context")
            for key, value in context.items():
                if isinstance(value, list):
                    parts.append(f"\n{key}:")
                    for item in value:
                        parts.append(f"  - {str(item)[:200]}...")
                elif isinstance(value, dict):
                    parts.append(f"\n{key}: {str(value)[:200]}...")
                else:
                    parts.append(f"\n{key}: {str(value)[:200]}...")

        # Add available tools
        if include_tools and self.tools:
            parts.append(f"\n# Available Tools")
            for tool_name, tool in self.tools.items():
                parts.append(f"- {tool_name}: {tool.description}")

        # Add role-specific instructions
        parts.append(self._get_task_instructions())

        return "\n".join(parts)

    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt"""
        prompts = {
            self.ROLE_BLUE_TEAM: """You are a Blue Team agent specialized in generating high-quality solutions.

Your role is to:
- Analyze sub-problems thoroughly
- Generate comprehensive, implementable solutions
- Leverage similar solutions from history when available
- Ensure solutions address all requirements
- Provide clear implementation details

Always provide structured, detailed solutions.""",

            self.ROLE_RED_TEAM: """You are a Red Team agent specialized in critical analysis.

Your role is to:
- Thoroughly critique solutions
- Identify potential issues and edge cases
- Suggest improvements
- Check against common failure patterns
- Provide constructive feedback

Be thorough but constructive in your critiques.""",

            self.ROLE_GOLD_TEAM: """You are a Gold Team agent specialized in verification.

Your role is to:
- Verify solutions meet requirements
- Check for correctness and completeness
- Validate against benchmarks
- Assess overall quality
- Provide verification results

Be objective and thorough in verification."""
        }
        return prompts.get(self.role, "You are a helpful AI assistant.")

    def _get_task_instructions(self) -> str:
        """Get role-specific task instructions"""
        return "\nPlease provide your response in a clear, structured format."

    def _parse_response(
        self,
        response: str,
        parse_as: str = "text"
    ) -> Any:
        """
        Parse LLM response.

        Args:
            response: Raw response text
            parse_as: How to parse ("text", "json", "list")

        Returns:
            Parsed response
        """
        if parse_as == "json":
            import json
            try:
                # Try to extract JSON from response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response")

        elif parse_as == "list":
            import re
            # Try to extract list items
            items = re.findall(r'^[-*]\s+(.+)$', response, re.MULTILINE)
            return items

        return response

    async def use_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to use
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        tool = self.tools[tool_name]
        result = await tool.execute(**kwargs)

        # Track tool call
        self.tool_calls_made.append({
            "tool": tool_name,
            "args": kwargs,
            "result": str(result)[:200],  # Truncate for logging
            "timestamp": datetime.utcnow().timestamp()
        })

        logger.info(f"Agent {self.role} used tool {tool_name}")
        return result

    def add_tool(self, tool: AgentTool):
        """Add a tool to the agent"""
        self.tools[tool.name] = tool
        logger.info(f"Added tool {tool.name} to agent {self.role}")

    def get_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()

    def get_tool_call_history(self) -> List[Dict[str, Any]]:
        """Get history of tool calls"""
        return self.tool_calls_made.copy()

    def clear_history(self):
        """Clear conversation and tool call history"""
        self.conversation_history.clear()
        self.tool_calls_made.clear()
        logger.info(f"Cleared history for agent {self.role}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model_id": self.model_config.get("model_id"),
            "tools_available": list(self.tools.keys()),
            "conversation_turns": len(self.conversation_history),
            "tool_calls_made": len(self.tool_calls_made)
        }
