"""
RLM Core Integration

Integrates the MIT Recursive Language Model (RLM) as the execution engine
for OpenEvolve's decomposition and problem-solving workflows.

RLM replaces standard LLM calls with recursive, code-executing calls that can:
1. Spawn sub-LMs for sub-problems
2. Execute code in sandboxed environments
3. Maintain state across recursive calls
4. Handle tasks of arbitrary complexity through recursion
"""
from __future__ import annotations


import sys
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Add RLM to path
RLM_PATH = os.path.join(os.path.dirname(__file__), "core-projects", "rlm")
if RLM_PATH not in sys.path:
    sys.path.insert(0, RLM_PATH)

# Try to import RLM
try:
    from rlm import RLM
    from rlm.core.types import RLMChatCompletion, RLMIteration
    from rlm.environments import LocalREPL, DockerREPL
    from rlm.clients import OpenAIClient, AnthropicClient
    RLM_AVAILABLE = True
except ImportError as e:
    print(f"RLM import error: {e}")
    RLM_AVAILABLE = False
    RLM = None


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class RLMExecutionResult:
    """Result from RLM execution."""
    solution: str
    iterations: List[Any]  # List[RLMIteration] when RLM available
    success: bool
    execution_trace: Dict[str, Any]


@dataclass
class SubproblemSolution:
    """Solution to a sub-problem."""
    answer: str
    confidence: float
    sub_calls: List[str]


@dataclass
class ExecutionOutput:
    """Standard execution output."""
    stdout: str
    stderr: str
    return_code: int
    execution_trace: Dict[str, Any]


# ============================================================================
# RLM CONFIGURATION
# ============================================================================

@dataclass
class RLMIntegrationConfig:
    """
    Configuration for RLM integration.
    
    RLM is the recursive execution engine - it handles:
    - Code execution in REPL environments
    - Recursive sub-LM spawning
    - State persistence across calls
    """
    # Backend configuration
    backend: str = "openai"  # "openai", "anthropic", "azure", etc.
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    
    # Environment configuration
    environment: str = "local"  # "local", "docker", "modal", "prime"
    environment_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Recursion settings
    max_depth: int = 3  # How many levels of recursion
    max_iterations: int = 30  # Max iterations per call
    
    # Sub-LM configuration (for recursive calls)
    sub_backend: Optional[str] = None  # Different model for sub-calls
    sub_model_name: Optional[str] = None
    
    # Persistence
    persistent: bool = True  # Reuse environment across calls
    log_dir: Optional[str] = "./rlm_logs"


# ============================================================================
# RLM EXECUTION ENGINE
# ============================================================================

class RLMExecutionEngine:
    """
    RLM as the core execution engine for problem-solving.
    
    This engine:
    1. Takes a problem/task
    2. Spawns RLM with appropriate environment
    3. Lets RLM recursively solve through code execution + sub-LM calls
    4. Returns final solution
    
    The RLM can:
    - Execute Python code in REPL
    - Spawn child RLMs for sub-problems
    - Iterate until solution found
    """
    
    def __init__(self, config: RLMIntegrationConfig):
        self.config = config
        self.rlm: Optional[Any] = None  # Optional[RLM] when available
        self._init_rlm()
    
    def _init_rlm(self):
        """Initialize RLM instance."""
        if not RLM_AVAILABLE:
            raise RuntimeError("RLM not available")
        
        backend_kwargs = {"model_name": self.config.model_name}
        if self.config.api_key:
            backend_kwargs["api_key"] = self.config.api_key
        
        # Setup sub-backend for recursive calls
        other_backends = None
        other_backend_kwargs = None
        if self.config.sub_backend:
            other_backends = [self.config.sub_backend]
            sub_kwargs = {"model_name": self.config.sub_model_name or self.config.model_name}
            if self.config.api_key:
                sub_kwargs["api_key"] = self.config.api_key
            other_backend_kwargs = [sub_kwargs]
        
        self.rlm = RLM(
            backend=self.config.backend,
            backend_kwargs=backend_kwargs,
            environment=self.config.environment,
            environment_kwargs=self.config.environment_kwargs,
            max_depth=self.config.max_depth,
            max_iterations=self.config.max_iterations,
            other_backends=other_backends,
            other_backend_kwargs=other_backend_kwargs,
            persistent=self.config.persistent,
        )
    
    def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None
    ) -> RLMExecutionResult:
        """
        Execute task using RLM.
        
        The RLM will:
        1. Analyze the task
        2. Write code to make progress
        3. Execute code in REPL
        4. Spawn sub-LMs if needed (for sub-problems)
        5. Iterate until solution
        
        Args:
            task: The task description
            context: Additional context (files, data, etc.)
            tools: Available tools for the RLM to use
            
        Returns:
            RLMExecutionResult with solution and execution trace
        """
        # Build prompt with context
        prompt = self._build_prompt(task, context, tools)
        
        # Execute via RLM
        completion = self.rlm.completion(prompt)
        
        return RLMExecutionResult(
            solution=completion.response,
            iterations=completion.metadata.iterations if hasattr(completion, 'metadata') else [],
            success=True,
            execution_trace=self._extract_trace(completion)
        )
    
    def solve_subproblem(
        self,
        subproblem: str,
        parent_context: Dict[str, Any]
    ) -> SubproblemSolution:
        """
        Solve a sub-problem (called recursively).
        
        This is how RLM handles decomposition - it spawns sub-calls
        for sub-problems automatically.
        """
        # RLM handles recursion internally via depth parameter
        result = self.execute(
            task=subproblem,
            context=parent_context
        )
        return SubproblemSolution(
            answer=result.solution,
            confidence=0.9,  # RLM doesn't expose confidence directly
            sub_calls=[]  # Would extract from execution trace
        )
    
    def _build_prompt(
        self,
        task: str,
        context: Optional[Dict],
        tools: Optional[List[str]]
    ) -> str:
        """Build RLM prompt with context and tools."""
        prompt_parts = [f"Task: {task}"]
        
        if context:
            prompt_parts.append(f"\nContext: {json.dumps(context)}")
        
        if tools:
            prompt_parts.append(f"\nAvailable tools: {', '.join(tools)}")
        
        return "\n".join(prompt_parts)
    
    def _extract_trace(self, completion: Any) -> Dict[str, Any]:
        """Extract execution trace from RLM completion."""
        trace = {
            "response": completion.response if hasattr(completion, 'response') else str(completion),
            "iterations": []
        }
        
        if hasattr(completion, 'metadata') and completion.metadata:
            if hasattr(completion.metadata, 'iterations'):
                trace["iterations"] = completion.metadata.iterations
            if hasattr(completion.metadata, 'execution_time'):
                trace["execution_time"] = completion.metadata.execution_time
        
        return trace


# ============================================================================
# RLM-AS-EXECUTOR PATTERN
# ============================================================================

class RLMExecutor:
    """
    Executor pattern using RLM.
    
    This wraps RLM to provide a standard executor interface
    that can be used by ROMA, Decomposition, MDAP, etc.
    """
    
    def __init__(self, config: Optional[RLMIntegrationConfig] = None):
        self.config = config or RLMIntegrationConfig()
        self.engine = RLMExecutionEngine(self.config)
    
    def run(
        self,
        command: str,
        input_data: Optional[Any] = None,
        timeout: Optional[int] = None
    ) -> ExecutionOutput:
        """
        Standard executor interface.
        
        Executes command via RLM, which may:
        - Execute code directly
        - Spawn sub-LMs for complex steps
        - Iterate multiple times
        """
        result = self.engine.execute(
            task=command,
            context={"input": input_data} if input_data else None
        )
        
        return ExecutionOutput(
            stdout=result.solution,
            stderr="",
            return_code=0 if result.success else 1,
            execution_trace=result.execution_trace
        )
    
    def run_with_tools(
        self,
        command: str,
        tools: List[str],
        input_data: Optional[Any] = None
    ) -> ExecutionOutput:
        """
        Execute with specific tools available.
        
        Args:
            command: The command/task to execute
            tools: List of available tool names
            input_data: Optional input data
            
        Returns:
            ExecutionOutput with results
        """
        result = self.engine.execute(
            task=command,
            context={"input": input_data} if input_data else None,
            tools=tools
        )
        
        return ExecutionOutput(
            stdout=result.solution,
            stderr="",
            return_code=0 if result.success else 1,
            execution_trace=result.execution_trace
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_rlm_engine(
    model: str = "gpt-4o",
    environment: str = "local",
    max_depth: int = 3,
    api_key: Optional[str] = None
) -> RLMExecutionEngine:
    """
    Factory for creating RLM execution engine.
    
    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-opus")
        environment: Execution environment ("local", "docker", "modal", "prime")
        max_depth: Maximum recursion depth for sub-LM calls
        api_key: API key for the LLM backend
        
    Returns:
        Configured RLMExecutionEngine instance
    """
    config = RLMIntegrationConfig(
        model_name=model,
        environment=environment,
        max_depth=max_depth,
        api_key=api_key
    )
    return RLMExecutionEngine(config)


def create_rlm_executor(**kwargs) -> RLMExecutor:
    """
    Factory for creating RLM executor.
    
    Args:
        **kwargs: Configuration parameters for RLMIntegrationConfig
        
    Returns:
        Configured RLMExecutor instance
    """
    config = RLMIntegrationConfig(**kwargs)
    return RLMExecutor(config)


def create_rlm_with_fallback(
    primary_model: str = "gpt-4o",
    fallback_model: str = "gpt-4o-mini",
    environment: str = "local"
) -> RLMExecutionEngine:
    """
    Create RLM engine with fallback model for sub-calls.
    
    Uses primary model for main task, fallback for recursive sub-calls.
    This is cost-effective for complex multi-step problems.
    
    Args:
        primary_model: Model for main execution
        fallback_model: Model for recursive sub-calls
        environment: Execution environment
        
    Returns:
        Configured RLMExecutionEngine with sub-backend
    """
    config = RLMIntegrationConfig(
        model_name=primary_model,
        environment=environment,
        sub_backend="openai",
        sub_model_name=fallback_model
    )
    return RLMExecutionEngine(config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_rlm_available() -> bool:
    """Check if RLM is available for use."""
    return RLM_AVAILABLE


def get_rlm_version() -> Optional[str]:
    """Get RLM version if available."""
    if not RLM_AVAILABLE:
        return None
    try:
        import rlm
        return getattr(rlm, "__version__", "unknown")
    except Exception:
        return "unknown"


def check_rlm_environment(environment: str) -> bool:
    """
    Check if specified environment is available.
    
    Args:
        environment: Environment name ("local", "docker", etc.)
        
    Returns:
        True if environment is available
    """
    if not RLM_AVAILABLE:
        return False
    
    if environment == "local":
        try:
            from rlm.environments import LocalREPL
            return True
        except ImportError:
            return False
    elif environment == "docker":
        try:
            from rlm.environments import DockerREPL
            return True
        except ImportError:
            return False
    
    return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Simple test/demo
    print(f"RLM Available: {RLM_AVAILABLE}")
    print(f"RLM Version: {get_rlm_version()}")
    
    if RLM_AVAILABLE:
        print("\nRLM environments available:")
        for env in ["local", "docker"]:
            print(f"  - {env}: {check_rlm_environment(env)}")
        
        # Create engine
        try:
            engine = create_rlm_engine()
            print("\nRLM Engine created successfully!")
        except Exception as e:
            print(f"\nFailed to create RLM Engine: {e}")
