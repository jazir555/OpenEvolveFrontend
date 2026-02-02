---
color: blue
position:
  x: 535
  y: -834
isContextNode: false
agent_name: Codebase Explorer
---

# CrewAI Integration Layer

## Summary
The CrewAI Integration Layer provides a unified interface for orchestrating multi-agent "crews" using the CrewAI framework. It manages the standard 6-phase workflow (Setup, Solve, Critique, Verify, Reassemble, Validate) and includes a "Zero Error Workflow" system for high-reliability task execution with built-in retries and error correction.

## Core Flow
```mermaid
flowchart TD
    P1[Phase 1: Setup] --> P2[Phase 2: Solve]
    P2 --> P3[Phase 3: Critique]
    P3 --> P4[Phase 4: Verify]
    P4 --> P5[Phase 5: Reassemble]
    P5 --> P6[Phase 6: Final Validation]
    
    subgraph Zero Error System
        ZE[Validator] --> C[Correction Strategy]
        C --> R[Retry/Rollback]
    end
    
    P2 -.-> ZE
    ZE -.-> P2
```

## Notable Gotchas & Tech Debt
- **Legacy Bridges**: Multiple bridge implementations (`CrewAIUnifiedBridge`, `ACECrewAIWorkflowBridge`) exist for backward compatibility with the older Hephaestus system, creating some redundant code.
- **Distributed State**: Managing state across distributed agents can lead to consistency issues if the `StateManager` is not used rigorously.
- **Complexity of Phase 2**: The "Solve" phase often branches out into MDAP or MAKER sub-workflows, which can be difficult to trace in logs.

[[run_me.md]]
