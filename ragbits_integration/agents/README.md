# Phase 2: Agent Coordination with A2A Protocol

## Overview

Phase 2 implements RAGBits-integrated agent coordination with Agent-to-Agent (A2A) protocol for the Decomposition Workflow. This enables seamless communication between Blue, Red, and Gold teams during workflow execution.

## Components

### 1. Base Workflow Agent (`base_agent.py`)

**Purpose**: Foundation for all workflow agents with CrewAI LLM integration

**Key Features**:
- LLM management via CrewAI (no direct LLM dependencies)
- Tool management and execution
- Prompt construction and response parsing
- Conversation history tracking
- Metadata collection

**Usage**:
```python
from ragbits_integration.agents.base_agent import BaseWorkflowAgent

agent = BaseWorkflowAgent(
    role="blue_team",
    crewai_client=crewai_client,
    model_config={"model_id": "gpt-4", "temperature": 0.7}
)

result = await agent.execute(
    task="Generate solution for authentication",
    context={"requirements": ["JWT", "OAuth"]}
)
```

### 2. Blue Team Agent (`blue_team_agent.py`)

**Purpose**: Generate high-quality solutions for sub-problems

**Key Features**:
- Leverages similar solutions from knowledge base
- Generates comprehensive, implementable solutions
- Refines solutions based on Red Team feedback
- Stores solutions for Red Team review

**Usage**:
```python
from ragbits_integration.agents import BlueTeamAgent

blue_agent = BlueTeamAgent(
    crewai_client=crewai,
    storage_manager=storage,
    knowledge_retriever=retriever
)

result = await blue_agent.generate_solution(
    sub_problem={
        "id": "sub_1",
        "title": "User Authentication",
        "description": "Implement secure auth",
        "requirements": ["JWT", "OAuth", "bcrypt"]
    },
    context={},
    use_rag=True
)

solution = result["solution"]
```

### 3. Red Team Agent (`red_team_agent.py`)

**Purpose**: Critique solutions and identify issues

**Key Features**:
- Thorough critique based on requirements
- Identifies potential issues and edge cases
- Leverages historical critique patterns
- Provides actionable feedback

**Usage**:
```python
from ragbits_integration.agents import RedTeamAgent

red_agent = RedTeamAgent(
    crewai_client=crewai,
    storage_manager=storage,
    knowledge_retriever=retriever
)

result = await red_agent.critique_solution(
    solution=solution_text,
    sub_problem=sub_problem,
    context={},
    use_patterns=True
)

issues = result["parsed"]["issues"]
recommendations = result["parsed"]["recommendations"]
```

### 4. Gold Team Agent (`gold_team_agent.py`)

**Purpose**: Verify solutions against requirements

**Key Features**:
- Verifies all requirements are met
- Checks if Red Team concerns are addressed
- Evaluates against benchmarks
- Provides pass/fail determination

**Usage**:
```python
from ragbits_integration.agents import GoldTeamAgent

gold_agent = GoldTeamAgent(
    crewai_client=crewai,
    storage_manager=storage,
    knowledge_retriever=retriever
)

result = await gold_agent.verify_solution(
    solution=solution_text,
    critique=critique_dict,
    sub_problem=sub_problem,
    context={}
)

passes = result["passes"]
score = result["overall_score"]
```

### 5. Agent Tools (`tools/`)

#### Knowledge Search Tool
Search for similar solutions and patterns from knowledge base.

```python
results = await agent.use_tool(
    "knowledge_search",
    search_type="similar_solutions",
    query="authentication system",
    top_k=5
)
```

#### Solution Evaluation Tool
Evaluate solution quality against criteria.

```python
evaluation = await agent.use_tool(
    "solution_eval",
    solution=solution_text,
    criteria=["completeness", "security", "efficiency"]
)
```

#### Pattern Analysis Tool
Analyze patterns in solutions and critiques.

```python
patterns = await agent.use_tool(
    "pattern_analysis",
    analysis_type="common_issues",
    domain="security"
)
```

### 6. A2A Protocol (`communication/a2a_protocol.py`)

**Purpose**: Agent-to-Agent communication protocol

**Key Features**:
- Message passing between agents
- Message routing and delivery
- Request/response tracking
- Broadcast messaging
- Message priority and deadlines

**Usage**:
```python
from ragbits_integration.agents.communication import (
    A2AProtocol,
    MessageType,
    MessageBuilder
)

# Initialize protocol
protocol = A2AProtocol()

# Send message
await protocol.send_message(
    sender="blue_team",
    recipient="red_team",
    message_type=MessageType.SOLUTION_SUBMITTED,
    content="Solution ready for review",
    sub_problem_id="sub_1",
    artifact_id="artifact_123"
)

# Receive messages
messages = await protocol.get_messages("red_team")

# Send reply
await protocol.send_reply(
    original_message=original_msg,
    reply_content="Review complete"
)

# Broadcast to multiple agents
await protocol.broadcast(
    sender="orchestrator",
    recipients=["blue_team", "red_team", "gold_team"],
    message_type=MessageType.STATUS_UPDATE,
    content="Status update"
)
```

**MessageBuilder** - Convenience methods for common messages:
```python
# Solution submitted
solution_msg = MessageBuilder.solution_submitted(
    sender="blue_team",
    recipient="red_team",
    solution="...",
    sub_problem_id="sub_1",
    artifact_id="artifact_123"
)

# Critique submitted
critique_msg = MessageBuilder.critique_submitted(
    sender="red_team",
    recipient="blue_team",
    critique="...",
    issues=["issue1", "issue2"],
    sub_problem_id="sub_1",
    artifact_id="artifact_123"
)

# Refinement request
refinement_msg = MessageBuilder.refinement_request(
    sender="red_team",
    recipient="blue_team",
    issues=["security", "performance"],
    sub_problem_id="sub_1",
    artifact_id="artifact_123"
)

# Verification result
verification_msg = MessageBuilder.verification_result(
    sender="gold_team",
    recipient="blue_team",
    passes=True,
    score=8.5,
    sub_problem_id="sub_1",
    artifact_id="artifact_123"
)
```

## Complete Workflow Example

```python
import asyncio
from ragbits_integration.agents import (
    BlueTeamAgent,
    RedTeamAgent,
    GoldTeamAgent
)
from ragbits_integration.agents.communication import A2AProtocol
from ragbits_integration import IntermediaryStorageManager

async def complete_workflow():
    # Setup
    protocol = A2AProtocol()
    storage = IntermediaryStorageManager(document_search)

    blue_agent = BlueTeamAgent(crewai, storage, retriever)
    red_agent = RedTeamAgent(crewai, storage, retriever)
    gold_agent = GoldTeamAgent(crewai, storage, retriever)

    sub_problem = {
        "id": "sub_1",
        "title": "User Authentication",
        "requirements": ["JWT", "OAuth", "bcrypt"]
    }

    # 1. Blue Team generates solution
    blue_result = await blue_agent.generate_solution(
        sub_problem=sub_problem,
        context={},
        use_rag=True
    )

    # 2. Notify Red Team
    await protocol.send_message(
        sender="blue_team",
        recipient="red_team",
        message_type=MessageType.SOLUTION_SUBMITTED,
        content=blue_result["solution"][:200],
        sub_problem_id=sub_problem["id"],
        artifact_id=blue_result["artifact_id"]
    )

    # 3. Red Team critiques
    red_result = await red_agent.critique_solution(
        solution=blue_result["solution"],
        sub_problem=sub_problem,
        context={"solution_artifact_id": blue_result["artifact_id"]}
    )

    # 4. Request refinement if needed
    if red_result["total_issues"] > 0:
        await protocol.send_message(
            sender="red_team",
            recipient="blue_team",
            message_type=MessageType.REFINEMENT_REQUEST,
            content="Please address identified issues",
            metadata={"issues": red_result["parsed"]["issues"]},
            sub_problem_id=sub_problem["id"],
            artifact_id=blue_result["artifact_id"]
        )

        # 5. Blue Team refines
        refined = await blue_agent.refine_solution(
            current_solution=blue_result["solution"],
            critique=red_result["parsed"]
        )

    # 6. Gold Team verifies
    gold_result = await gold_agent.verify_solution(
        solution=refined["solution"],
        critique=red_result["parsed"],
        sub_problem=sub_problem
    )

    # 7. Communicate result
    await protocol.send_message(
        sender="gold_team",
        recipient="blue_team",
        message_type=MessageType.VERIFICATION_SUBMITTED,
        content=f"Verification: {'PASSED' if gold_result['passes'] else 'FAILED'}",
        metadata={"score": gold_result["overall_score"]},
        sub_problem_id=sub_problem["id"],
        artifact_id=refined["artifact_id"]
    )

    return {
        "blue_result": blue_result,
        "red_result": red_result,
        "gold_result": gold_result,
        "verified": gold_result["passes"]
    }

# Run workflow
result = asyncio.run(complete_workflow())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Workflow Orchestrator                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐
        │   Blue   │      │    Red   │      │   Gold   │
        │   Team   │─────││   Team   │─────││   Team   │
        │  Agent   │A2A  ││  Agent   │A2A  ││  Agent   │
        └──────────┘     └──────────┘     └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌───────────┴───────────┐
                    │   Intermediary Storage │
                    │    (solutions, critiques)│
                    └──────────────────────┘
```

## Message Types

| Type | Description | Direction |
|------|-------------|-----------|
| `SOLUTION_SUBMITTED` | Blue Team completed solution | Blue → Red |
| `CRITIQUE_SUBMITTED` | Red Team completed critique | Red → Blue/Gold |
| `REFINEMENT_REQUEST` | Request solution refinement | Red → Blue |
| `VERIFICATION_SUBMITTED` | Gold Team completed verification | Gold → All |
| `STATUS_UPDATE` | General status updates | Any → All |

## Testing

Run Phase 2 tests:

```bash
# Run all Phase 2 tests
python ragbits_integration/agents/run_phase2_tests.py

# Or with pytest
python -m pytest ragbits_integration/agents/tests/test_agent_coordination.py
```

## Integration with Decomposition Workflow

Phase 2 agents integrate with the existing workflow at Stage 3 (Sub-Problem Solving Loop):

**Stage 3A: Blue Team Solution Generation**
- ContextGatherer retrieves similar solutions
- BlueTeamAgent generates solution
- Solution stored immediately in vector store
- Red Team notified via A2A

**Stage 3B: Red Team Critique**
- RedTeamAgent retrieves Blue Team's solution
- Leverages historical critique patterns
- Generates structured critique
- Requests refinement if needed

**Stage 3C: Gold Team Verification**
- GoldTeamAgent retrieves both solution and critique
- Evaluates against requirements and benchmarks
- Provides pass/fail determination
- Result stored and communicated

## Files Structure

```
ragbits_integration/agents/
├── __init__.py
├── base_agent.py                 # Base agent class
├── blue_team_agent.py            # Blue Team implementation
├── red_team_agent.py             # Red Team implementation
├── gold_team_agent.py            # Gold Team implementation
├── tools/
│   ├── __init__.py
│   ├── knowledge_search_tool.py # Semantic search tool
│   ├── solution_eval_tool.py    # Solution evaluation tool
│   └── pattern_analysis_tool.py  # Pattern analysis tool
├── communication/
│   ├── __init__.py
│   └── a2a_protocol.py            # A2A messaging protocol
├── tests/
│   ├── __init__.py
│   └── test_agent_coordination.py # Unit tests
└── run_phase2_tests.py           # Integration tests
```

## Next Steps

Phase 3: Evaluation Framework Integration
- Integrate RAGBits evaluation framework
- Enhance gauntlet validation
- Add evaluation metrics and dashboards

## Status

✅ **COMPLETE** - All Phase 2 components implemented and tested

- Base agent with CrewAI integration
- Blue, Red, and Gold team agents
- Agent tools (knowledge search, evaluation, patterns)
- A2A protocol with message routing
- Comprehensive unit and integration tests
