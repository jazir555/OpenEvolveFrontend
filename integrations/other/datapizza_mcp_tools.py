"""
DataPizza MCP Tools for CREWAI Agents

This module provides Model Context Protocol (MCP) tools that CREWAI agents
can use to leverage DataPizza's multi-agent framework for problem solving.

Architecture:
    CREWAI Agent -> MCP Tool -> DataPizza Agent -> Tools -> Result

Key Features:
    - Multi-agent coordination (Blue/Red/Gold teams)
    - Tool use (FileSystem, Web Search, SQL, Web Fetch)
    - OpenTelemetry tracing
    - Planning capabilities
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Try to import DataPizza components
try:
    from datapizza.agents import Agent
    from datapizza.clients import Client
    from datapizza.tools import Tool
    DATAPIZZA_AVAILABLE = True
    logger.info("DataPizza core imported successfully")
except ImportError as e:
    logger.warning(f"DataPizza core not available: {e}")
    DATAPIZZA_AVAILABLE = False
    Agent = None
    Client = None
    Tool = None

# Try to import DataPizza clients
try:
    from datapizza.clients.openai import OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza OpenAI client not available")
    OPENAI_CLIENT_AVAILABLE = False
    OpenAIClient = None

try:
    from datapizza.clients.anthropic import AnthropicClient
    ANTHROPIC_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza Anthropic client not available")
    ANTHROPIC_CLIENT_AVAILABLE = False
    AnthropicClient = None

try:
    from datapizza.clients.google import GoogleClient
    GOOGLE_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza Google client not available")
    GOOGLE_CLIENT_AVAILABLE = False
    GoogleClient = None

# Try to import DataPizza tools
try:
    from datapizza.tools.filesystem import FileSystem
    FILESYSTEM_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza FileSystem tool not available")
    FILESYSTEM_AVAILABLE = False
    FileSystem = None

try:
    from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza DuckDuckGo tool not available")
    DUCKDUCKGO_AVAILABLE = False
    DuckDuckGoSearchTool = None

try:
    from datapizza.tools.SQLDatabase import SQLDatabaseTool
    SQL_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza SQL tool not available")
    SQL_AVAILABLE = False
    SQLDatabaseTool = None

try:
    from datapizza.tools.web_fetch import WebFetchTool
    WEB_FETCH_AVAILABLE = True
except ImportError:
    logger.warning("DataPizza WebFetch tool not available")
    WEB_FETCH_AVAILABLE = False
    WebFetchTool = None


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered DataPizza MCP tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered DataPizza MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Optional[Client]:
    """
    Create a DataPizza client for the specified provider.

    Args:
        provider: Provider name ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name to use
        **kwargs: Additional client parameters

    Returns:
        Client instance or None if unavailable
    """
    if not DATAPIZZA_AVAILABLE:
        return None

    provider = provider.lower()

    if provider == "openai" and OPENAI_CLIENT_AVAILABLE:
        return OpenAIClient(
            api_key=api_key or kwargs.get("openai_api_key"),
            model=model or kwargs.get("openai_model", "gpt-4o-mini"),
        )
    elif provider == "anthropic" and ANTHROPIC_CLIENT_AVAILABLE:
        return AnthropicClient(
            api_key=api_key or kwargs.get("anthropic_api_key"),
            model=model or kwargs.get("anthropic_model", "claude-3-5-sonnet-20241022"),
        )
    elif provider == "google" and GOOGLE_CLIENT_AVAILABLE:
        return GoogleClient(
            api_key=api_key or kwargs.get("google_api_key"),
            model=model or kwargs.get("google_model", "gemini-pro"),
        )
    else:
        logger.warning(f"Provider {provider} not available or client not imported")
        return None


def _create_tools(
    tools: Optional[List[str]] = None,
    paths_to_include: Optional[List[str]] = None,
    paths_to_exclude: Optional[List[str]] = None,
) -> List[Tool]:
    """
    Create DataPizza tools based on requested tool names.

    Args:
        tools: List of tool names ("filesystem", "duckduckgo", "sql", "web_fetch")
        paths_to_include: Include patterns for FileSystem
        paths_to_exclude: Exclude patterns for FileSystem

    Returns:
        List of Tool instances
    """
    if not DATAPIZZA_AVAILABLE:
        return []

    tool_list = []
    tools = tools or []

    if "filesystem" in tools and FILESYSTEM_AVAILABLE:
        tool_list.append(FileSystem(
            paths_to_include=paths_to_include,
            paths_to_exclude=paths_to_exclude,
        ))

    if "duckduckgo" in tools and DUCKDUCKGO_AVAILABLE:
        tool_list.append(DuckDuckGoSearchTool())

    if "sql" in tools and SQL_AVAILABLE:
        tool_list.append(SQLDatabaseTool())

    if "web_fetch" in tools and WEB_FETCH_AVAILABLE:
        tool_list.append(WebFetchTool())

    return tool_list


# =============================================================================
# DATAPIZZA MCP TOOLS
# =============================================================================

@mcp_tool("create_datapizza_agent")
def create_datapizza_agent(
    agent_name: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[str]] = None,
    planning_interval: int = 0,
    max_steps: Optional[int] = None,
    paths_to_include: Optional[List[str]] = None,
    paths_to_exclude: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a DataPizza agent with specified configuration.

    This creates an agent instance that can be used for subsequent executions.

    Args:
        agent_name: Name of the agent
        provider: AI provider ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name to use
        system_prompt: System prompt for the agent
        tools: List of tools to enable ("filesystem", "duckduckgo", "sql", "web_fetch")
        planning_interval: Steps between planning cycles (0 = no planning)
        max_steps: Maximum execution steps (None = unlimited)
        paths_to_include: Include patterns for FileSystem tool
        paths_to_exclude: Exclude patterns for FileSystem tool
        **kwargs: Additional parameters

    Returns:
        Dict with agent configuration (for serialization)
    """
    logger.info(f"Creating DataPizza agent: {agent_name} (provider={provider})")

    if not DATAPIZZA_AVAILABLE:
        return {
            "error": "DataPizza not available",
            "agent_name": agent_name,
        }

    try:
        # Create client
        client = _create_client(
            provider=provider,
            api_key=api_key,
            model=model,
            **kwargs
        )

        if not client:
            return {
                "error": f"Failed to create client for provider: {provider}",
                "agent_name": agent_name,
                "available_providers": {
                    "openai": OPENAI_CLIENT_AVAILABLE,
                    "anthropic": ANTHROPIC_CLIENT_AVAILABLE,
                    "google": GOOGLE_CLIENT_AVAILABLE,
                }
            }

        # Create tools
        agent_tools = _create_tools(
            tools=tools,
            paths_to_include=paths_to_include,
            paths_to_exclude=paths_to_exclude,
        )

        # Create agent
        agent = Agent(
            name=agent_name,
            client=client,
            system_prompt=system_prompt or "You are a helpful assistant.",
            tools=agent_tools,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        return {
            "agent_name": agent_name,
            "provider": provider,
            "model": model,
            "system_prompt": system_prompt,
            "tools": tools or [],
            "tool_count": len(agent_tools),
            "planning_interval": planning_interval,
            "max_steps": max_steps,
            "status": "created",
        }

    except Exception as e:
        logger.error(f"Failed to create DataPizza agent: {e}")
        return {
            "error": str(e),
            "agent_name": agent_name,
        }


@mcp_tool("run_datapizza_agent")
def run_datapizza_agent(
    agent_name: str,
    prompt: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[str]] = None,
    planning_interval: int = 0,
    max_steps: Optional[int] = None,
    paths_to_include: Optional[List[str]] = None,
    paths_to_exclude: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a task using a DataPizza agent.

    This creates an agent and runs it with the given prompt in one call.

    Args:
        agent_name: Name of the agent
        prompt: User prompt/task to execute
        provider: AI provider ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name to use
        system_prompt: System prompt for the agent
        tools: List of tools to enable
        planning_interval: Steps between planning cycles
        max_steps: Maximum execution steps
        paths_to_include: Include patterns for FileSystem
        paths_to_exclude: Exclude patterns for FileSystem
        **kwargs: Additional parameters

    Returns:
        Dict with execution results
    """
    logger.info(f"Running DataPizza agent {agent_name} with prompt: {prompt[:100]}...")

    if not DATAPIZZA_AVAILABLE:
        return {
            "error": "DataPizza not available",
            "agent_name": agent_name,
            "prompt": prompt,
        }

    try:
        # Create client
        client = _create_client(
            provider=provider,
            api_key=api_key,
            model=model,
            **kwargs
        )

        if not client:
            return {
                "error": f"Failed to create client for provider: {provider}",
                "agent_name": agent_name,
                "prompt": prompt,
            }

        # Create tools
        agent_tools = _create_tools(
            tools=tools,
            paths_to_include=paths_to_include,
            paths_to_exclude=paths_to_exclude,
        )

        # Create and run agent
        agent = Agent(
            name=agent_name,
            client=client,
            system_prompt=system_prompt or "You are a helpful assistant.",
            tools=agent_tools,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        result = agent.run(prompt)

        # Extract result data
        response_data = {
            "agent_name": agent_name,
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "response_text": result.text if hasattr(result, 'text') else str(result),
            "status": "completed",
        }

        # Add step information if available
        if hasattr(result, 'index'):
            response_data["steps"] = result.index

        # Add tool usage if available
        if hasattr(result, 'tools_used'):
            response_data["tools_used"] = [t.function for t in result.tools_used]

        # Add token usage if available
        if hasattr(result, 'usage') and result.usage:
            response_data["token_usage"] = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }

        return response_data

    except Exception as e:
        logger.error(f"Failed to run DataPizza agent {agent_name}: {e}")
        return {
            "error": str(e),
            "agent_name": agent_name,
            "prompt": prompt,
        }


@mcp_tool("solve_with_datapizza_agent")
def solve_with_datapizza_agent(
    sub_problem_id: str,
    sub_problem_description: str,
    agent_role: str = "solver",
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    planning_interval: int = 3,
    max_steps: int = 20,
    working_directory: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve a sub-problem using a DataPizza agent.

    This is the main integration point for Decomposition Workflow Stage 3A.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of the problem to solve
        agent_role: Role of the agent ("solver", "critiquer", "verifier")
        provider: AI provider ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name to use
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        tools: List of tools to enable
        planning_interval: Steps between planning cycles
        max_steps: Maximum execution steps
        working_directory: Working directory for file operations
        **kwargs: Additional parameters

    Returns:
        Dict with solution attempt
    """
    logger.info(f"Solving {sub_problem_id} with DataPizza agent (role={agent_role})")

    if not DATAPIZZA_AVAILABLE:
        return {
            "error": "DataPizza not available",
            "sub_problem_id": sub_problem_id,
            "execution_method_used": "datapizza",
        }

    try:
        # Determine system prompt based on role
        system_prompts = {
            "solver": "You are an expert solution architect. Your task is to analyze the problem and provide a complete, implementable solution.",
            "critiquer": "You are a critical reviewer. Your task is to find flaws, weaknesses, and potential issues in the provided solution.",
            "verifier": "You are a quality assurance specialist. Your task is to verify that the solution meets all requirements and is production-ready.",
        }
        system_prompt = system_prompts.get(agent_role, system_prompts["solver"])

        # Build enhanced prompt
        prompt_parts = [f"Sub-Problem ID: {sub_problem_id}", sub_problem_description]

        if context:
            prompt_parts.append(f"\nContext: {context}")

        if constraints:
            prompt_parts.append("\nConstraints:")
            for c in constraints:
                prompt_parts.append(f"  - {c}")

        if requirements:
            prompt_parts.append("\nRequirements:")
            for r in requirements:
                prompt_parts.append(f"  - {r}")

        prompt = "\n".join(prompt_parts)

        # Configure filesystem paths if working directory specified
        paths_to_include = None
        if working_directory:
            paths_to_include = [f"{working_directory}/**"]

        # Create and run agent
        result = run_datapizza_agent(
            agent_name=f"{agent_role}_{sub_problem_id}",
            prompt=prompt,
            provider=provider,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            tools=tools or ["filesystem"],
            planning_interval=planning_interval,
            max_steps=max_steps,
            paths_to_include=paths_to_include,
            paths_to_exclude=["*.pyc", "__pycache__/", ".git/", "node_modules/"],
            **kwargs
        )

        if "error" in result:
            return {
                "error": result["error"],
                "sub_problem_id": sub_problem_id,
                "execution_method_used": "datapizza",
            }

        return {
            "sub_problem_id": sub_problem_id,
            "solution": result["response_text"],
            "agent_role": agent_role,
            "generated_by": f"DataPizza ({provider})",
            "status": "completed",
            "execution_method_used": "datapizza",
            "steps_taken": result.get("steps", 0),
            "tools_used": result.get("tools_used", []),
            "token_usage": result.get("token_usage"),
        }

    except Exception as e:
        logger.error(f"Failed to solve {sub_problem_id} with DataPizza: {e}")
        return {
            "error": str(e),
            "sub_problem_id": sub_problem_id,
            "execution_method_used": "datapizza",
        }


@mcp_tool("create_multi_agent_system")
def create_multi_agent_system(
    team_name: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    working_directory: Optional[str] = None,
    enable_filesystem: bool = True,
    enable_web_search: bool = True,
    planning_interval: int = 3,
    max_steps: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a multi-agent system with Blue/Red/Gold team structure.

    This creates three coordinated agents:
    - Blue Agent: Solves problems
    - Red Agent: Critiques solutions
    - Gold Agent: Verifies quality

    Args:
        team_name: Base name for the team
        provider: AI provider for all agents
        api_key: API key for the provider
        model: Model name to use
        working_directory: Working directory for file operations
        enable_filesystem: Enable filesystem tools
        enable_web_search: Enable web search tools
        planning_interval: Planning interval for agents
        max_steps: Maximum steps per agent
        **kwargs: Additional parameters

    Returns:
        Dict with multi-agent system configuration
    """
    logger.info(f"Creating multi-agent system: {team_name}")

    if not DATAPIZZA_AVAILABLE:
        return {
            "error": "DataPizza not available",
            "team_name": team_name,
        }

    try:
        # Create client (shared by all agents)
        client = _create_client(
            provider=provider,
            api_key=api_key,
            model=model,
            **kwargs
        )

        if not client:
            return {
                "error": f"Failed to create client for provider: {provider}",
                "team_name": team_name,
            }

        # Configure tools
        tools = []
        paths_to_include = [f"{working_directory}/**"] if working_directory else None
        paths_to_exclude = ["*.pyc", "__pycache__/", ".git/", "node_modules/"]

        if enable_filesystem and FILESYSTEM_AVAILABLE:
            tools.append(FileSystem(
                paths_to_include=paths_to_include,
                paths_to_exclude=paths_to_exclude,
            ))

        if enable_web_search and DUCKDUCKGO_AVAILABLE:
            tools.append(DuckDuckGoSearchTool())

        # Create Blue Agent (Solver)
        blue_agent = Agent(
            name=f"{team_name}_blue",
            client=client,
            system_prompt="You are an expert solution architect. Your task is to analyze problems and provide complete, implementable solutions.",
            tools=tools,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        # Create Red Agent (Critiquer)
        red_agent = Agent(
            name=f"{team_name}_red",
            client=client,
            system_prompt="You are a critical reviewer. Your task is to find flaws, weaknesses, and potential issues in solutions. Be thorough and constructive.",
            tools=tools,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        # Create Gold Agent (Verifier)
        gold_agent = Agent(
            name=f"{team_name}_gold",
            client=client,
            system_prompt="You are a quality assurance specialist. Your task is to verify that solutions meet all requirements and are production-ready.",
            tools=tools,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        # Set up agent coordination
        blue_agent.can_call([red_agent, gold_agent])

        return {
            "team_name": team_name,
            "provider": provider,
            "model": model,
            "agents": {
                "blue": f"{team_name}_blue",
                "red": f"{team_name}_red",
                "gold": f"{team_name}_gold",
            },
            "tools_enabled": [t.__class__.__name__ for t in tools],
            "planning_interval": planning_interval,
            "max_steps": max_steps,
            "coordination": "blue -> red, gold",
            "status": "created",
        }

    except Exception as e:
        logger.error(f"Failed to create multi-agent system {team_name}: {e}")
        return {
            "error": str(e),
            "team_name": team_name,
        }


@mcp_tool("run_multi_agent_task")
def run_multi_agent_task(
    team_name: str,
    task: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    working_directory: Optional[str] = None,
    enable_filesystem: bool = True,
    enable_web_search: bool = True,
    planning_interval: int = 3,
    max_steps: int = 20,
    workflow: str = "blue_red_gold",
    **kwargs
) -> Dict[str, Any]:
    """
    Run a task using a multi-agent system.

    This creates agents on-the-fly and executes them in the specified workflow.

    Args:
        team_name: Base name for the team
        task: Task description to execute
        provider: AI provider for all agents
        api_key: API key for the provider
        model: Model name to use
        working_directory: Working directory for file operations
        enable_filesystem: Enable filesystem tools
        enable_web_search: Enable web search tools
        planning_interval: Planning interval for agents
        max_steps: Maximum steps per agent
        workflow: Workflow pattern ("blue_red_gold", "blue_only", "parallel")
        **kwargs: Additional parameters

    Returns:
        Dict with multi-agent execution results
    """
    logger.info(f"Running multi-agent task with team {team_name} (workflow={workflow})")

    if not DATAPIZZA_AVAILABLE:
        return {
            "error": "DataPizza not available",
            "team_name": team_name,
            "task": task,
        }

    try:
        # Create multi-agent system
        system_config = create_multi_agent_system(
            team_name=team_name,
            provider=provider,
            api_key=api_key,
            model=model,
            working_directory=working_directory,
            enable_filesystem=enable_filesystem,
            enable_web_search=enable_web_search,
            planning_interval=planning_interval,
            max_steps=max_steps,
            **kwargs
        )

        if "error" in system_config:
            return system_config

        # Re-create agents (in real implementation, would cache these)
        client = _create_client(provider=provider, api_key=api_key, model=model, **kwargs)
        if not client:
            return {"error": f"Failed to create client for provider: {provider}"}

        tools = []
        if working_directory:
            paths_to_include = [f"{working_directory}/**"]
        else:
            paths_to_include = None
        paths_to_exclude = ["*.pyc", "__pycache__/", ".git/", "node_modules/"]

        if enable_filesystem and FILESYSTEM_AVAILABLE:
            tools.append(FileSystem(paths_to_include=paths_to_include, paths_to_exclude=paths_to_exclude))
        if enable_web_search and DUCKDUCKGO_AVAILABLE:
            tools.append(DuckDuckGoSearchTool())

        blue_agent = Agent(
            name=f"{team_name}_blue",
            client=client,
            system_prompt="You are an expert solution architect. Your task is to analyze problems and provide complete, implementable solutions.",
            tools=tools,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        # Execute based on workflow
        results = {}

        if workflow == "blue_only":
            blue_result = blue_agent.run(task)
            results["blue"] = {
                "response": blue_result.text if hasattr(blue_result, 'text') else str(blue_result),
                "steps": blue_result.index if hasattr(blue_result, 'index') else 0,
            }

        elif workflow == "blue_red_gold":
            # Blue: Solve
            blue_result = blue_agent.run(task)
            results["blue"] = {
                "response": blue_result.text if hasattr(blue_result, 'text') else str(blue_result),
                "steps": blue_result.index if hasattr(blue_result, 'index') else 0,
            }

            # Red: Critique
            red_agent = Agent(
                name=f"{team_name}_red",
                client=client,
                system_prompt="You are a critical reviewer. Your task is to find flaws, weaknesses, and potential issues in the following solution. Be thorough and constructive.",
                tools=tools,
                planning_interval=planning_interval,
                max_steps=max_steps,
            )
            red_task = f"Review this solution:\n\n{results['blue']['response']}"
            red_result = red_agent.run(red_task)
            results["red"] = {
                "response": red_result.text if hasattr(red_result, 'text') else str(red_result),
                "steps": red_result.index if hasattr(red_result, 'index') else 0,
            }

            # Gold: Verify
            gold_agent = Agent(
                name=f"{team_name}_gold",
                client=client,
                system_prompt="You are a quality assurance specialist. Your task is to verify that the solution meets requirements. Given the original solution and the critique, provide a final assessment.",
                tools=tools,
                planning_interval=planning_interval,
                max_steps=max_steps,
            )
            gold_task = f"Original solution:\n{results['blue']['response']}\n\nCritique:\n{results['red']['response']}\n\nProvide your final verification."
            gold_result = gold_agent.run(gold_task)
            results["gold"] = {
                "response": gold_result.text if hasattr(gold_result, 'text') else str(gold_result),
                "steps": gold_result.index if hasattr(gold_result, 'index') else 0,
            }

        elif workflow == "parallel":
            # Run Blue, Red, Gold independently on same task
            agents = {}
            for role, system_prompt in [
                ("blue", "You are an expert solution architect. Provide a complete solution."),
                ("red", "You are a critical reviewer. Analyze this task for potential challenges."),
                ("gold", "You are a QA specialist. Identify requirements and success criteria."),
            ]:
                agent = Agent(
                    name=f"{team_name}_{role}",
                    client=client,
                    system_prompt=system_prompt,
                    tools=tools,
                    planning_interval=planning_interval,
                    max_steps=max_steps,
                )
                result = agent.run(task)
                results[role] = {
                    "response": result.text if hasattr(result, 'text') else str(result),
                    "steps": result.index if hasattr(result, 'index') else 0,
                }

        return {
            "team_name": team_name,
            "task": task,
            "workflow": workflow,
            "provider": provider,
            "model": model,
            "results": results,
            "total_steps": sum(r.get("steps", 0) for r in results.values()),
            "status": "completed",
            "execution_method_used": "datapizza",
        }

    except Exception as e:
        logger.error(f"Failed to run multi-agent task {team_name}: {e}")
        return {
            "error": str(e),
            "team_name": team_name,
            "task": task,
        }


@mcp_tool("get_datapizza_status")
def get_datapizza_status() -> Dict[str, Any]:
    """Get the status of the DataPizza integration"""
    return {
        "available": DATAPIZZA_AVAILABLE,
        "components": {
            "core": DATAPIZZA_AVAILABLE,
            "openai_client": OPENAI_CLIENT_AVAILABLE,
            "anthropic_client": ANTHROPIC_CLIENT_AVAILABLE,
            "google_client": GOOGLE_CLIENT_AVAILABLE,
            "filesystem_tool": FILESYSTEM_AVAILABLE,
            "duckduckgo_tool": DUCKDUCKGO_AVAILABLE,
            "sql_tool": SQL_AVAILABLE,
            "web_fetch_tool": WEB_FETCH_AVAILABLE,
        },
        "supported_providers": ["openai", "anthropic", "google"],
        "supported_tools": ["filesystem", "duckduckgo", "sql", "web_fetch"],
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all DataPizza MCP tools"""
    logger.info("Initializing DataPizza MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} DataPizza MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


# Auto-initialize on import
initialize_mcp_tools()
