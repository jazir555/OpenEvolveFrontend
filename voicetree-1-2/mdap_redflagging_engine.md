---
color: blue
position:
  x: 1614
  y: -2441
isContextNode: false
agent_name: Codebase Explorer
---

# MDAP Red-Flagging Engine

## Summary
The primary safety and quality filter for the MDAP framework. it uses a library of "red flag rules" to scan agent outputs for common failures (e.g., "sorry" tokens, logic loops, or schema violations) before they are passed to the voting engine.

## Core Flow
```mermaid
flowchart LR
    AO[Agent Output] --> RR[Red-Flag Rules]
    RR --> CM[Criteria Matching]
    CM --> FS[Filter/Pass Status]
    FS -- Flagged --> R[Reject/Retry]
```

## Notable Gotchas & Tech Debt
- **False Negatives**: Subtle logic errors can easily bypass simple pattern-based red-flagging rules.
- **Performance Burden**: Running deep semantic checks on every agent sample can significantly increase total execution time.

[[mdap_multi_dimensional_agentic_processing.md]]
