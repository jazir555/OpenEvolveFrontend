---
color: blue
position:
  x: 792
  y: -1102
isContextNode: false
agent_name: Codebase Explorer
---

# ROMA Verifier

## Summary
The Quality Assurance module of the ROMA loop. it validates whether the candidate output (from an Executor or Aggregator) actually satisfies the original goal. if validation fails, it provides detailed feedback to trigger re-planning or re-execution.

## Core Flow
```mermaid
flowchart LR
    CO[Candidate Output] --> G[Original Goal]
    G --> V[Verification Logic]
    V --> S{Status?}
    S -- Pass --> F[Finalize]
    S -- Fail --> RF[Feedback & Retry]
```

## Notable Gotchas & Tech Debt
- **Shallow Verification**: The verifier may perform a superficial check that misses deeper logical or technical inaccuracies.
- **Verification-Execution Loops**: Faulty goals or inadequate toolkits can lead to infinite loops between execution and failed verification.

[[roma_reasoning_on_multi_agent.md]]
