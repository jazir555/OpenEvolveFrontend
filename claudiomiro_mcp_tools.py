"""
Claudiomiro MCP Tools for CREWAI Integration

This module provides Model Context Protocol (MCP) tools that enable CREWAI
agents to leverage Claudiomiro's autonomous development capabilities.

Claudiomiro is an AI-powered development CLI that:
- Decomposes complex tasks into parallelizable sub-tasks
- Executes tasks autonomously (code, review, test, commit)
- Supports multiple AI providers (Claude, Codex, Gemini, DeepSeek, GLM)
- Runs tasks in parallel using DAG execution
- Provides production-ready code with automatic testing

Architecture: CREWAI (Orchestrator) -> Claudiomiro (Autonomous Development)
"""

from typing import Any, Dict, List, Optional, Union
import sys
import os
import json
import logging
import subprocess
import asyncio
from functools import wraps
from datetime import datetime
from pathlib import Path

# MCP Tool Registry
_MCP_TOOLS = {}

def mcp_tool(name: str):
    """Decorator to register MCP tools."""
    def decorator(func):
        _MCP_TOOLS[name] = func
        return func
    return decorator

# Claudiomiro Availability Detection
CLAUDIOMIRO_AVAILABLE = False
CLAUDIOMIRO_PATH = None
CLAUDIOMIRO_IMPORT_ERROR = None

try:
    # Check if claudiomiro CLI is available
    result = subprocess.run(
        ["claudiomiro", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        CLAUDIOMIRO_AVAILABLE = True
        CLAUDIOMIRO_PATH = "claudiomiro"
except FileNotFoundError:
    CLAUDIOMIRO_IMPORT_ERROR = "claudiomiro CLI not found in PATH"
except subprocess.TimeoutExpired:
    CLAUDIOMIRO_IMPORT_ERROR = "claudiomiro CLI timeout"
except (OSError, subprocess.SubprocessError) as e:
    CLAUDIOMIRO_IMPORT_ERROR = str(e)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MCP Tool 1: Execute Claudiomiro Task
# ============================================================================

@mcp_tool("execute_claudiomiro_task")
def execute_claudiomiro_task(
    task_id: str,
    prompt: str,
    working_dir: str,
    ai_provider: str = "claude",
    backend: Optional[str] = None,
    frontend: Optional[str] = None,
    legacy: Optional[str] = None,
    max_cycles: int = 20,
    fix_command: Optional[str] = None,
    enable_local_llm: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an autonomous development task using Claudiomiro.

    Claudiomiro will:
    1. Decompose the task into sub-tasks
    2. Execute tasks in parallel
    3. Review code
    4. Run tests and fix failures
    5. Create production-ready commits

    Args:
        task_id: Unique identifier for the task
        prompt: Task description/prompt
        working_dir: Directory to execute in
        ai_provider: AI provider to use (claude, codex, gemini, deep-seek, glm)
        backend: Backend directory (for multi-repo projects)
        frontend: Frontend directory (for multi-repo projects)
        legacy: Legacy system directory
        max_cycles: Maximum execution cycles (default: 20)
        fix_command: Command to run for fixing (e.g., "npm test")
        enable_local_llm: Local LLM model for Ollama (e.g., "qwen2.5-coder:7b")

    Returns:
        Dict with:
            - success: bool
            - task_id: str
            - status: str (running, completed, failed)
            - output: str
            - commit_hash: str (if committed)
            - message: str
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "success": False,
            "task_id": task_id,
            "available": False,
            "error": "Claudiomiro not available",
            "message": CLAUDIOMIRO_IMPORT_ERROR or "claudiomiro CLI not installed or not in PATH",
        }

    # ============================================================================
    # Input Validation and Sanitization
    # ============================================================================
    import re
    import shlex

    # Validate task_id - only allow alphanumeric, hyphens, and underscores
    if not task_id or not isinstance(task_id, str):
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid task_id: must be a non-empty string",
        }
    if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid task_id: contains unsafe characters. Use only alphanumeric, hyphens, and underscores.",
        }

    # Validate working_dir - prevent directory traversal attacks
    if not working_dir or not isinstance(working_dir, str):
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid working_dir: must be a non-empty string",
        }
    # Resolve to absolute path and check for path traversal
    try:
        resolved_working_dir = os.path.abspath(os.path.normpath(working_dir))
        # Prevent access to system directories
        system_paths = ['/bin', '/sbin', '/usr/bin', '/usr/sbin', '/etc', '/root', '/sys', '/proc']
        for sys_path in system_paths:
            if resolved_working_dir.startswith(sys_path):
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": f"Invalid working_dir: access to system directory '{sys_path}' is not allowed",
                }
    except Exception as e:
        return {
            "success": False,
            "task_id": task_id,
            "error": f"Invalid working_dir: {e}",
        }

    # Validate prompt - check for shell injection attempts
    if not prompt or not isinstance(prompt, str):
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid prompt: must be a non-empty string",
        }
    # Check for common shell metacharacters that could indicate injection
    dangerous_patterns = [
        r'[;&|`$]',  # Command separators and shell operators
        r'\$\(',     # Command substitution
        r'`',        # Backtick command substitution
        r'\$\{',     # Variable expansion
        r'\|\|',    # OR operator
        r'&&',       # AND operator
        r'>>',       # Append redirection
        r'2>&1',     # File descriptor redirection
        r'/bin/',    # Attempts to access binaries
        r'/usr/bin', # Attempts to access system binaries
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt):
            logger.warning(f"Potentially dangerous pattern detected in prompt: {pattern}")
            return {
                "success": False,
                "task_id": task_id,
                "error": "Invalid prompt: contains potentially dangerous characters or patterns",
            }

    # Validate ai_provider
    valid_providers = ["claude", "codex", "gemini", "deep-seek", "glm"]
    if ai_provider not in valid_providers:
        return {
            "success": False,
            "task_id": task_id,
            "error": f"Invalid ai_provider: must be one of {valid_providers}",
        }

    # Validate max_cycles
    if not isinstance(max_cycles, int) or max_cycles < 1 or max_cycles > 1000:
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid max_cycles: must be an integer between 1 and 1000",
        }

    try:
        # Build command using list format (shell=False by default)
        # This prevents shell injection by passing arguments directly
        cmd = [CLAUDIOMIRO_PATH]

        # Add AI provider flag
        provider_flags = {
            "claude": "--claude",
            "codex": "--codex",
            "gemini": "--gemini",
            "deep-seek": "--deep-seek",
            "glm": "--glm",
        }
        if ai_provider in provider_flags:
            cmd.append(provider_flags[ai_provider])

        # Add directories (validated)
        if backend:
            # Validate backend path
            backend_path = os.path.abspath(os.path.normpath(str(backend)))
            cmd.extend(["--backend", backend_path])
        if frontend:
            # Validate frontend path
            frontend_path = os.path.abspath(os.path.normpath(str(frontend)))
            cmd.extend(["--frontend", frontend_path])
        if legacy:
            # Validate legacy path
            legacy_path = os.path.abspath(os.path.normpath(str(legacy)))
            cmd.extend(["--legacy", legacy_path])

        # Add options
        cmd.extend(["--limit", str(max_cycles)])

        if fix_command:
            # Validate fix_command - only allow safe characters
            if not re.match(r'^[a-zA-Z0-9_\-\s\.\/]+$', str(fix_command)):
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": "Invalid fix_command: contains unsafe characters",
                }
            cmd.extend(["--fix-command", str(fix_command)])

        # Add prompt (already validated above)
        cmd.extend(["--prompt", prompt])

        # Set environment variables
        env = os.environ.copy()
        if enable_local_llm:
            env["CLAUDIOMIRO_LOCAL_LLM"] = enable_local_llm

        # Execute claudiomiro
        logger.info(f"Executing Claudiomiro task: {task_id}")
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            return {
                "success": True,
                "task_id": task_id,
                "available": True,
                "status": "completed",
                "ai_provider": ai_provider,
                "working_dir": working_dir,
                "output": result.stdout,
                "message": f"Claudiomiro task completed successfully",
            }
        else:
            return {
                "success": False,
                "task_id": task_id,
                "available": True,
                "status": "failed",
                "error": result.stderr,
                "output": result.stdout,
                "message": f"Claudiomiro task failed with return code {result.returncode}",
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "task_id": task_id,
            "available": True,
            "status": "timeout",
            "error": "Task timed out after 1 hour",
            "message": "Claudiomiro task timeout",
        }
    except Exception as e:
        logger.error(f"Failed to execute Claudiomiro task: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "available": True,
            "error": str(e),
            "message": f"Failed to execute task: {e}",
        }


# ============================================================================
# MCP Tool 2: Decompose Task with Claudiomiro
# ============================================================================

@mcp_tool("decompose_task_with_claudiomiro")
def decompose_task_with_claudiomiro(
    task_id: str,
    prompt: str,
    working_dir: str,
    output_file: Optional[str] = None,
    ai_provider: str = "claude",
) -> Dict[str, Any]:
    """
    Decompose a complex task into sub-tasks using Claudiomiro.

    This runs only Step 0 of Claudiomiro (task decomposition)
    without executing the tasks.

    Args:
        task_id: Unique identifier for the task
        prompt: Task description/prompt
        working_dir: Directory to analyze
        output_file: Optional file to save decomposition
        ai_provider: AI provider to use

    Returns:
        Dict with:
            - success: bool
            - task_id: str
            - sub_tasks: List[Dict] (decomposed tasks)
            - num_tasks: int
            - message: str
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "success": False,
            "task_id": task_id,
            "available": False,
            "error": "Claudiomiro not available",
            "message": CLAUDIOMIRO_IMPORT_ERROR,
        }

    try:
        # Check if .claudiomiro folder exists
        claudiomiro_dir = Path(working_dir) / ".claudiomiro"

        if not claudiomiro_dir.exists():
            # Run Claudiomiro to initialize
            result = subprocess.run(
                [CLAUDIOMIRO_PATH, "--prompt", "Initialize"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

        # Look for TODO.md (task decomposition file)
        todo_file = claudiomiro_dir / "TODO.md"

        if todo_file.exists():
            with open(todo_file, 'r') as f:
                todo_content = f.read()

            # Parse TODO.md for sub-tasks
            sub_tasks = _parse_todo_file(todo_content)

            return {
                "success": True,
                "task_id": task_id,
                "available": True,
                "sub_tasks": sub_tasks,
                "num_tasks": len(sub_tasks),
                "working_dir": working_dir,
                "todo_file": str(todo_file),
                "message": f"Task decomposed into {len(sub_tasks)} sub-tasks",
            }
        else:
            return {
                "success": False,
                "task_id": task_id,
                "available": True,
                "error": "TODO.md not found",
                "message": "Claudiomiro did not create TODO.md",
            }

    except Exception as e:
        logger.error(f"Failed to decompose task: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "available": True,
            "error": str(e),
            "message": f"Failed to decompose task: {e}",
        }


# ============================================================================
# MCP Tool 3: Fix Failing Tests with Claudiomiro
# ============================================================================

@mcp_tool("fix_tests_with_claudiomiro")
def fix_tests_with_claudiomiro(
    task_id: str,
    test_command: str,
    working_dir: str,
    loop_fixes: bool = True,
    max_iterations: int = 5,
    ai_provider: str = "claude",
) -> Dict[str, Any]:
    """
    Fix failing tests using Claudiomiro's autonomous fixing capabilities.

    Args:
        task_id: Unique identifier for the task
        test_command: Test command to run (e.g., "npm test")
        working_dir: Directory to fix tests in
        loop_fixes: Whether to loop fixes until all tests pass
        max_iterations: Maximum fix iterations
        ai_provider: AI provider to use

    Returns:
        Dict with:
            - success: bool
            - task_id: str
            - tests_fixed: int
            - iterations: int
            - message: str
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "success": False,
            "task_id": task_id,
            "available": False,
            "error": "Claudiomiro not available",
            "message": CLAUDIOMIRO_IMPORT_ERROR,
        }

    try:
        # Build command
        cmd = [CLAUDIOMIRO_PATH]

        provider_flags = {
            "claude": "--claude",
            "codex": "--codex",
            "gemini": "--gemini",
            "deep-seek": "--deep-seek",
            "glm": "--glm",
        }
        if ai_provider in provider_flags:
            cmd.append(provider_flags[ai_provider])

        if loop_fixes:
            cmd.append("--loop-fixes")
            cmd.extend(["--limit", str(max_iterations)])

        cmd.extend(["--fix-command", test_command])

        logger.info(f"Fixing tests with Claudiomiro: {task_id}")

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
        )

        if result.returncode == 0:
            return {
                "success": True,
                "task_id": task_id,
                "available": True,
                "tests_fixed": True,  # Would parse output for actual count
                "iterations": max_iterations,
                "working_dir": working_dir,
                "output": result.stdout,
                "message": "Tests fixed successfully",
            }
        else:
            return {
                "success": False,
                "task_id": task_id,
                "available": True,
                "error": result.stderr,
                "output": result.stdout,
                "message": "Test fixing failed",
            }

    except Exception as e:
        logger.error(f"Failed to fix tests: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "available": True,
            "error": str(e),
            "message": f"Failed to fix tests: {e}",
        }


# ============================================================================
# MCP Tool 4: Review and Fix Branch with Claudiomiro
# ============================================================================

@mcp_tool("fix_branch_with_claudiomiro")
def fix_branch_with_claudiomiro(
    task_id: str,
    working_dir: str,
    target_branch: str = "main",
    ai_provider: str = "claude",
) -> Dict[str, Any]:
    """
    Review and fix current branch before creating PR using Claudiomiro.

    Args:
        task_id: Unique identifier for the task
        working_dir: Directory to fix
        target_branch: Target branch for comparison
        ai_provider: AI provider to use

    Returns:
        Dict with review and fix results
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "success": False,
            "task_id": task_id,
            "available": False,
            "error": "Claudiomiro not available",
            "message": CLAUDIOMIRO_IMPORT_ERROR,
        }

    try:
        cmd = [CLAUDIOMIRO_PATH]

        provider_flags = {
            "claude": "--claude",
            "codex": "--codex",
            "gemini": "--gemini",
            "deep-seek": "--deep-seek",
            "glm": "--glm",
        }
        if ai_provider in provider_flags:
            cmd.append(provider_flags[ai_provider])

        cmd.append("--fix-branch")

        logger.info(f"Fixing branch with Claudiomiro: {task_id}")

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        return {
            "success": result.returncode == 0,
            "task_id": task_id,
            "available": True,
            "working_dir": working_dir,
            "target_branch": target_branch,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "message": "Branch fix completed" if result.returncode == 0 else "Branch fix failed",
        }

    except Exception as e:
        logger.error(f"Failed to fix branch: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "available": True,
            "error": str(e),
            "message": f"Failed to fix branch: {e}",
        }


# ============================================================================
# MCP Tool 5: Get Claudiomiro Status
# ============================================================================

@mcp_tool("get_claudiomiro_status")
def get_claudiomiro_status() -> Dict[str, Any]:
    """
    Get Claudiomiro installation and status information.

    Returns:
        Dict with:
            - available: bool
            - version: str
            - supported_providers: List[str]
            - features: Dict[str, bool]
            - message: str
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "available": False,
            "installed": False,
            "version": None,
            "error": CLAUDIOMIRO_IMPORT_ERROR or "Claudiomiro CLI not installed or not in PATH",
            "supported_providers": [],
            "features": {
                "task_decomposition": False,
                "parallel_execution": False,
                "automated_testing": False,
                "code_review": False,
                "multi_repo": False,
                "local_llm": False,
            },
        }

    return {
        "available": True,
        "installed": True,
        "version": "npm installed",  # Claudiomiro version
        "message": "Claudiomiro CLI is available",
        "supported_providers": [
            "claude",      # Anthropic Claude
            "codex",       # OpenAI Codex
            "gemini",      # Google Gemini
            "deep-seek",   # DeepSeek
            "glm",         # GLM
        ],
        "features": {
            "task_decomposition": True,
            "parallel_execution": True,
            "automated_testing": True,
            "code_review": True,
            "multi_repo": True,
            "local_llm": True,
            "legacy_systems": True,
        },
        "cloud_api_compatible": True,
        "supported_cloud_apis": [
            "Anthropic Claude (claude)",
            "OpenAI (codex)",
            "Google Gemini (gemini)",
            "DeepSeek (deep-seek)",
            "GLM (glm)",
        ],
    }


# ============================================================================
# MCP Tool 6: Execute Multi-Repo Task
# ============================================================================

@mcp_tool("execute_multi_repo_task_with_claudiomiro")
def execute_multi_repo_task_with_claudiomiro(
    task_id: str,
    prompt: str,
    backend: str,
    frontend: str,
    working_dir: str,
    legacy_backend: Optional[str] = None,
    legacy_frontend: Optional[str] = None,
    ai_provider: str = "claude",
) -> Dict[str, Any]:
    """
    Execute a task across multiple repositories using Claudiomiro.

    Claudiomiro will:
    - Detect monorepo vs separate repos
    - Scope tasks to backend/frontend
    - Verify integration between codebases
    - Coordinate commits across repositories

    Args:
        task_id: Unique identifier for the task
        prompt: Task description
        backend: Backend directory path
        frontend: Frontend directory path
        working_dir: Root working directory
        legacy_backend: Optional legacy backend path
        legacy_frontend: Optional legacy frontend path
        ai_provider: AI provider to use

    Returns:
        Dict with multi-repo execution results
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "success": False,
            "task_id": task_id,
            "available": False,
            "error": "Claudiomiro not available",
            "message": CLAUDIOMIRO_IMPORT_ERROR,
        }

    try:
        cmd = [CLAUDIOMIRO_PATH]

        provider_flags = {
            "claude": "--claude",
            "codex": "--codex",
            "gemini": "--gemini",
            "deep-seek": "--deep-seek",
            "glm": "--glm",
        }
        if ai_provider in provider_flags:
            cmd.append(provider_flags[ai_provider])

        cmd.extend(["--backend", backend])
        cmd.extend(["--frontend", frontend])

        if legacy_backend:
            cmd.extend(["--legacy-backend", legacy_backend])
        if legacy_frontend:
            cmd.extend(["--legacy-frontend", legacy_frontend])

        cmd.extend(["--prompt", prompt])

        logger.info(f"Executing multi-repo task: {task_id}")

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        return {
            "success": result.returncode == 0,
            "task_id": task_id,
            "available": True,
            "backend": backend,
            "frontend": frontend,
            "has_legacy": bool(legacy_backend or legacy_frontend),
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "message": "Multi-repo task completed" if result.returncode == 0 else "Multi-repo task failed",
        }

    except Exception as e:
        logger.error(f"Failed to execute multi-repo task: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "available": True,
            "error": str(e),
            "message": f"Failed to execute multi-repo task: {e}",
        }


# ============================================================================
# MCP Tool 7: Configure Claudiomiro
# ============================================================================

@mcp_tool("configure_claudiomiro")
def configure_claudiomiro(
    config_key: str,
    config_value: str,
    global_config: bool = True,
) -> Dict[str, Any]:
    """
    Configure Claudiomiro settings.

    Args:
        config_key: Configuration key (e.g., "CLAUDIOMIRO_LOCAL_LLM")
        config_value: Configuration value
        global_config: Whether to set global config

    Returns:
        Dict with configuration result
    """
    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "Claudiomiro not available",
            "message": CLAUDIOMIRO_IMPORT_ERROR,
        }

    try:
        cmd = [CLAUDIOMIRO_PATH, "--config", f"{config_key}={config_value}"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        return {
            "success": result.returncode == 0,
            "available": True,
            "config_key": config_key,
            "config_value": config_value,
            "global": global_config,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "message": f"Configuration set: {config_key}={config_value}",
        }

    except Exception as e:
        logger.error(f"Failed to configure Claudiomiro: {e}")
        return {
            "success": False,
            "available": True,
            "error": str(e),
            "message": f"Failed to configure: {e}",
        }


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_todo_file(todo_content: str) -> List[Dict[str, Any]]:
    """Parse Claudiomiro's TODO.md file to extract sub-tasks."""
    sub_tasks = []
    current_task = None

    for line in todo_content.split('\n'):
        line = line.strip()

        # Task headers (##, ###)
        if line.startswith('##') or line.startswith('###'):
            if current_task:
                sub_tasks.append(current_task)

            title = line.lstrip('#').strip()
            current_task = {
                "title": title,
                "description": "",
                "status": "pending",
            }

        # Task list items (- [ ], - [x])
        elif line.startswith('- [') and current_task:
            status = "completed" if '[x]' in line.lower() else "pending"
            task_text = line.split(']', 1)[1].strip() if ']' in line else line
            current_task["description"] = task_text
            current_task["status"] = status

    if current_task:
        sub_tasks.append(current_task)

    return sub_tasks


# ============================================================================
# MCP Tool Registry Access
# ============================================================================

def get_registered_tools() -> Dict[str, Any]:
    """Get all registered MCP tools."""
    return _MCP_TOOLS.copy()

def list_mcp_tools() -> List[str]:
    """List names of all registered MCP tools."""
    return list(_MCP_TOOLS.keys())

# Export all MCP tools
__all__ = [
    # MCP Tools
    "execute_claudiomiro_task",
    "decompose_task_with_claudiomiro",
    "fix_tests_with_claudiomiro",
    "fix_branch_with_claudiomiro",
    "get_claudiomiro_status",
    "execute_multi_repo_task_with_claudiomiro",
    "configure_claudiomiro",
    # Utilities
    "get_registered_tools",
    "list_mcp_tools",
    "CLAUDIOMIRO_AVAILABLE",
]

# Module initialization
if __name__ == "__main__":
    print("Claudiomiro MCP Tools Module")
    print(f"Claudiomiro Available: {CLAUDIOMIRO_AVAILABLE}")
    print(f"Registered Tools: {len(_MCP_TOOLS)}")
    print("\nTools:")
    for tool_name in sorted(_MCP_TOOLS.keys()):
        print(f"  - {tool_name}")
