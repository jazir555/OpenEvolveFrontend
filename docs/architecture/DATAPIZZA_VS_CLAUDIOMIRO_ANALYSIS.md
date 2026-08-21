# DataPizza vs Claudiomiro Analysis

> **STATUS: unrelated subsystem (informational).** This document analyzes external tools DataPizza and Claudiomiro and is not part of the OpenEvolve/BubbleLab integration. No code in this repository implements DataPizza or Claudiomiro as core components. Retained for historical context only.
> **Last reconciled: 2026-08-20**

**Date**: 2025-12-29
**Status**: ANALYSIS COMPLETE
**Recommendation**: USE TOGETHER (Complementary)

---

## Executive Summary

**DataPizza** and **Claudiomiro** are fundamentally different tools that serve complementary purposes:

- **DataPizza** = Python-native AI Agent Framework (for building intelligent agents)
- **Claudiomiro** = Autonomous Development CLI (for doing coding work)

**Recommendation**: **USE BOTH TOGETHER** - Each fills a different gap in the CrewAI ecosystem.

---

## What is DataPizza?

**DataPizza** is a Python-based GenAI framework for building reliable AI agents with tools.

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DataPizza Agent                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Client    │  │   Tools      │  │      Memory         │  │
│  │ (OpenAI,    │  │ - Filesystem │  │  - Conversational   │  │
│  │  Anthropic, │  │ - Web Search │  │  - Context Aware    │  │
│  │  Google,    │  │ - SQL        │  │                     │  │
│  │  Mistral)   │  │ - Web Fetch  │  │                     │  │
│  └─────────────┘  └──────────────┘  └─────────────────────┘  │
│                                                                 │
│  Features:                                                     │
│  - Multi-agent coordination (can_call other agents)            │
│  - Planning with planning_interval                             │
│  - Tool use with @tool decorator                               │
│  - OpenTelemetry tracing                                       │
│  - Step-by-step execution with max_steps                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Agent Framework**
   - Multi-agent systems with `can_call([other_agents])`
   - Planning capabilities with planning intervals
   - Step-by-step execution
   - Tool use with decorators

2. **Tools Available**
   - **FileSystem**: read_file, write_file, replace_in_file, list_directory, create_directory, delete_file, move_item, copy_file
   - **DuckDuckGo**: Web search
   - **SQLDatabase**: Execute SQL queries
   - **WebFetch**: Fetch web content

3. **Observability**
   - OpenTelemetry tracing built-in
   - Client I/O tracing
   - Custom spans for fine-grained debugging

4. **Cloud Providers**
   - OpenAI, Google Gemini, Anthropic, Mistral, Azure OpenAI
   - Vendor-agnostic client interface

5. **RAG Capabilities**
   - Document ingestion (PDF, DOCX)
   - Vector stores (Qdrant, Milvus)
   - Embeddings (OpenAI, Google, Cohere, FastEmbed)
   - Rerankers (Cohere, Together AI)

---

## What is Claudiomiro?

**Claudiomiro** is a Node.js CLI tool for autonomous development automation.

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claudiomiro CLI                              │
│                                                                 │
│  Workflow:                                                      │
│  1. Decompose task into sub-tasks                               │
│  2. Execute sub-tasks in parallel (DAG-based)                   │
│  3. Run tests                                                  │
│  4. Review code                                                │
│  5. Fix failures (loop until pass)                             │
│  6. Commit changes                                              │
│                                                                 │
│  Cloud APIs:                                                    │
│  - Claude (Anthropic)                                           │
│  - Codex (OpenAI)                                               │
│  - Gemini (Google)                                             │
│  - DeepSeek                                                     │
│  - GLM                                                          │
│                                                                 │
│  Capabilities:                                                  │
│  - Execute arbitrary shell commands                            │
│  - Multi-repository support (backend/frontend)                  │
│  - Autonomous from prompt to commit                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Autonomous Development**
   - Task decomposition → Code → Test → Review → Commit
   - Fully autonomous execution
   - Loop until tests pass

2. **Multi-Repository**
   - Backend and frontend coordination
   - Cross-repository changes

3. **Shell Command Execution**
   - Run npm install, python tests, etc.
   - Full system access

4. **Cloud API Compatible**
   - All major providers supported
   - Native CLI flags for each

---

## Comparison Table

| Aspect | DataPizza | Claudiomiro |
|--------|-----------|-------------|
| **Primary Purpose** | AI Agent Framework | Autonomous Development CLI |
| **Language** | Python | Node.js (CLI) |
| **Integration** | Python library (`import`) | Subprocess invocation |
| **Control Level** | High (build your own agent) | Low (black-box execution) |
| **Observability** | OpenTelemetry tracing | stdout/stderr only |
| **Tool Use** | Filesystem, Web, SQL | Shell commands (anything) |
| **Multi-Agent** | ✅ Yes (agent coordination) | ❌ No (single execution) |
| **Planning** | ✅ Yes (planning_interval) | ✅ Yes (built-in DAG) |
| **Testing** | ❌ No (custom needed) | ✅ Yes (auto-fix loop) |
| **Git Integration** | ❌ No | ✅ Yes (auto-commit) |
| **Code Editing** | ✅ Yes (replace_in_file) | ✅ Yes (via shell/tools) |
| **Cloud APIs** | ✅ Yes (5+ providers) | ✅ Yes (5 providers) |
| **RAG Support** | ✅ Yes (full pipeline) | ❌ No |
| **Memory** | ✅ Yes (conversational) | ❌ No |
| **Custom Tools** | ✅ Yes (@tool decorator) | ❌ No (fixed workflow) |
| **CrewAI Fit** | Excellent (Phase 2-4 agents) | Excellent (Phase 2 execution) |

---

## Technical Differences

### Code Editing

**DataPizza (replace_in_file)**:
```python
@tool
def replace_in_file(self, file_path: str, old_string: str, new_string: str) -> str:
    """Replaces a string in a file, but only if it appears exactly once."""
    # Enforces uniqueness validation
    # Safe, controlled edits
```

**Claudiomiro**:
```bash
# Can use any tool: sed, awk, Python scripts, etc.
# Via shell command execution
# More flexible but less safe
```

### Multi-Agent Coordination

**DataPizza**:
```python
planner_agent = Agent(name="planner", client=client)
coder_agent = Agent(name="coder", client=client)
tester_agent = Agent(name="tester", client=client)

planner_agent.can_call([coder_agent, tester_agent])
# Agents can delegate to each other
```

**Claudiomiro**:
```bash
# Single monolithic execution
# No agent-to-agent communication
```

### Observability

**DataPizza**:
```python
with ContextTracing().trace("my_ai_operation"):
    response = agent.run("Tell me about Bitcoin")

# Output shows:
# ╭─ Trace Summary of my_ai_operation ──────────────────╮
# │ Total Spans: 3                                       │
# │ Duration: 2.45s                                      │
# │ Model: gpt-4o-mini                                   │
# │ Prompt Tokens: 31                                   │
# │ Completion Tokens: 27                               │
# ╰──────────────────────────────────────────────────────╯
```

**Claudiomiro**:
```bash
# Only stdout/stderr
# No structured tracing
```

---

## Use Case Analysis

### When to Use DataPizza

1. **Multi-Agent Problem Solving** (Decomposition Workflow Stages 3A-3C)
   - Blue Team agent for solving
   - Red Team agent for critique
   - Gold Team agent for verification
   - Agents can call each other

2. **RAG and Knowledge Retrieval**
   - Document ingestion
   - Vector search
   - Context-augmented generation

3. **Tool-Augmented Agents**
   - Web search (DuckDuckGo)
   - SQL queries
   - Filesystem operations
   - Web fetching

4. **Observability is Critical**
   - OpenTelemetry tracing
   - Debugging complex agent workflows
   - Production monitoring

5. **Custom Agent Behavior**
   - Need full control over agent logic
   - Custom tools and workflows
   - Stateful agents with memory

### When to Use Claudiomiro

1. **Autonomous Feature Implementation** (Decomposition Workflow Stage 3A - Implementation)
   - "Implement user authentication with JWT"
   - "Add REST API endpoints for user management"
   - Full autonomy from prompt to commit

2. **Multi-Repository Development**
   - Backend (Python/FastAPI) + Frontend (React/Vue)
   - Cross-repository changes
   - Coordinated commits

3. **Test-Driven Development**
   - Auto-run tests
   - Auto-fix failures
   - Loop until pass

4. **Git Workflow Automation**
   - Automatic commits
   - Branch management
   - PR preparation

5. **Simple "Get It Done" Tasks**
   - Don't need custom agent logic
   - Just want code written and tested
   - Black-box execution is fine

---

## Integration Strategy: USE BOTH TOGETHER

### Architecture: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CrewAI Orchestrator                             │
│  Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Phase 2: Solution Generation
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Decomposition Workflow (solve_sub_problem)                 │
│                                                                              │
│  Option 1: Traditional → OpenEvolve + LLM                                   │
│  Option 2: DataPizza → Agent with tools                                    │
│  Option 3: Claudiomiro → Autonomous CLI                                    │
│  Option 4: Auto → Smart selection based on task                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
┌───────────────────────────┐     ┌───────────────────────────────────┐
│   DataPizza Agent          │     │     Claudiomiro CLI               │
│  ┌─────────────────────┐   │     │  ┌─────────────────────────────┐ │
│  │ Blue Agent (solve)  │   │     │  │ Autonomous Development       │ │
│  │ Red Agent (critique)│   │     │  │ - Decompose                  │ │
│  │ Gold Agent (verify) │   │     │  │ - Code                       │ │
│  └─────────────────────┘   │     │  │ - Test                       │ │
│  Tools:                   │     │  │ - Fix                        │ │
│  - Filesystem (edit)      │     │  │ - Commit                     │ │
│  - Web Search             │     │  └─────────────────────────────┘ │
│  - SQL                    │     │                                   │
│  - Memory                 │     │  Multi-repo support               │
│  - Planning               │     │  Shell command execution          │
│  - Tracing                │     │  Loop until tests pass            │
└───────────────────────────┘     └───────────────────────────────────┘
```

### Recommended Integration

#### 1. Use DataPizza for Multi-Agent Decomposition (Stages 3A-3C)

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.filesystem import FileSystem
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

# Blue Team: Solution generation
blue_agent = Agent(
    name="blue_solver",
    client=client,
    system_prompt="You are a solution expert. Generate implementation solutions.",
    tools=[FileSystem(paths_to_include=["./project/**"])],
    planning_interval=3,
)

# Red Team: Critique
red_agent = Agent(
    name="red_critiquer",
    client=client,
    system_prompt="You are a critic. Find flaws in solutions.",
    tools=[DuckDuckGoSearchTool()],
)

# Gold Team: Verification
gold_agent = Agent(
    name="gold_verifier",
    client=client,
    system_prompt="You are a verifier. Ensure solutions meet requirements.",
)

blue_agent.can_call([red_agent, gold_agent])
```

#### 2. Use Claudiomiro for Autonomous Implementation

```python
from decomposition_mcp_tools import solve_sub_problem_with_team

result = solve_sub_problem_with_team(
    sub_problem_id="SP-001",
    sub_problem_description="Implement user authentication with JWT tokens",
    team_name="Blue-Team-Alpha",
    execution_method="claudiomiro",  # Use Claudiomiro
    claudiomiro_provider="claude",
    working_dir="./project",
)
```

#### 3. Enhanced Auto-Selection Logic

```python
def _determine_execution_method(sub_problem_description: str) -> str:
    """Intelligent selection between DataPizza and Claudiomiro"""

    # Use Claudiomiro for: implementation tasks, multi-repo, testing
    claudiomiro_keywords = [
        "implement", "code", "function", "class", "api", "endpoint",
        "test", "fix", "backend", "frontend", "repository", "commit"
    ]

    # Use DataPizza for: analysis, research, planning, critique
    datapizza_keywords = [
        "analyze", "research", "plan", "design", "critique", "verify",
        "search", "query", "database", "web", "investigate"
    ]

    if any(kw in sub_problem_description.lower() for kw in claudiomiro_keywords):
        return "claudiomiro"
    elif any(kw in sub_problem_description.lower() for kw in datapizza_keywords):
        return "datapizza"

    # Default based on complexity
    return "datapizza"  # Safer default with more control
```

---

## Implementation Plan

### Phase 1: DataPizza Integration (Week 1)

1. **Create DataPizza MCP Tools** (`datapizza_mcp_tools.py`)
   - `create_datapizza_agent()` - Create agent with client and tools
   - `run_datapizza_agent()` - Execute agent with prompt
   - `create_multi_agent_system()` - Create Blue/Red/Gold agents
   - `run_multi_agent_task()` - Execute with agent coordination
   - `get_datapizza_status()` - Check installation

2. **Create DataPizza CrewAI Bridge** (`datapizza_crewai_bridge.py`)
   - Map CrewAI phases to DataPizza agents
   - Phase 2: Blue agent solving
   - Phase 3: Red agent critique
   - Phase 4: Gold agent verification

3. **Enhance Decomposition Workflow**
   - Add "datapizza" execution method option
   - Update `_solve_with_datapizza()` helper function
   - Integrate with existing auto-selection logic

### Phase 2: Three-Way Integration (Week 2)

1. **Update `solve_sub_problem_with_team()`**
   ```python
   execution_method: "traditional" | "claudiomiro" | "datapizza" | "auto"
   ```

2. **Implement Smart Routing**
   - Traditional: OpenEvolve + LLM prompts
   - Claudiomiro: Autonomous implementation
   - DataPizza: Multi-agent problem solving
   - Auto: Choose based on task characteristics

3. **Add DataPizza-Specific Parameters**
   ```python
   use_datapizza: bool = False
   datapizza_provider: str = "openai"
   datapizza_tools: List[str] = ["filesystem", "web"]
   datapizza_planning_interval: int = 3
   datapizza_max_steps: int = 20
   ```

---

## Summary Comparison

| Aspect | DataPizza | Claudiomiro | Recommendation |
|--------|-----------|-------------|----------------|
| **Type** | Framework (library) | Tool (CLI) | Different purposes |
| **Language** | Python | Node.js | Python matches codebase |
| **Control** | High (customizable) | Low (black-box) | DataPizza for custom needs |
| **Multi-Agent** | ✅ Yes | ❌ No | DataPizza for coordination |
| **Autonomy** | Manual | ✅ Full | Claudiomiro for hands-off |
| **Observability** | ✅ OpenTelemetry | ❌ Basic | DataPizza for production |
| **RAG** | ✅ Yes | ❌ No | DataPizza for knowledge |
| **Git** | ❌ No | ✅ Yes | Claudiomiro for commits |
| **Testing** | ❌ No | ✅ Auto-fix | Claudiomiro for TDD |
| **Shell Access** | ❌ Limited | ✅ Full | Claudiomiro for system |

---

## Final Recommendation

### USE BOTH TOGETHER

**DataPizza** for:
- Multi-agent decomposition workflow (Blue/Red/Gold teams)
- Complex problem-solving requiring agent coordination
- RAG and knowledge retrieval
- Production workflows with observability requirements
- Custom tool use and agent behavior

**Claudiomiro** for:
- Autonomous feature implementation
- Multi-repository development
- Test-driven development with auto-fix
- Git workflow automation
- "Get it done" black-box execution

**Together** they provide:
1. **Flexibility**: Choose the right tool for each sub-problem
2. **Complementarity**: DataPizza's agents + Claudiomiro's autonomy
3. **Coverage**: Analysis, planning, implementation, testing, verification
4. **Production-Ready**: Observability (DataPizza) + Automation (Claudiomiro)

---

**Date**: 2025-12-29
**Status**: ANALYSIS COMPLETE
**Recommendation**: INTEGRATE BOTH - Use DataPizza for multi-agent workflows, Claudiomiro for autonomous implementation
