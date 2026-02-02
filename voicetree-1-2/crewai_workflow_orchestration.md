---
color: blue
position:
  x: 199
  y: -553
isContextNode: false
agent_name: Codebase Explorer
---

# CrewAI Workflow Orchestration Engine (Unified Flow)

## Summary
The core engine that implements the 6-phase evolutionary lifecycle (Setup, Solve, Critique, Verify, Reassemble, Validate). It uses an event-driven "Flow" architecture to manage the transition between phases and coordinate agent activities.

## Core Flow
```mermaid
flowchart TD
    S[Start] --> P1[Phase 1: Setup]
    P1 --> P2[Phase 2: Solve]
    P2 --> P3[Phase 3: Critique]
    P3 --> P4[Phase 4: Verify]
    P4 --> P5[Phase 5: Reassemble]
    P5 --> P6[Phase 6: Validate]
    P6 --> E[End]
```

## Notable Gotchas & Tech Debt
- **State Consistency**: Ensuring that state is correctly preserved and migrated across different execution methods (e.g., Traditional vs. ROMA) is highly complex.
- **Event Debugging**: Tracing the flow of events across multiple asynchronous phases can be difficult with standard logging.

[[crewai_integration_layer.md]]
