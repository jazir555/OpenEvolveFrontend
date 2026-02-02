---
color: blue
position:
  x: 2246
  y: -1448
isContextNode: false
agent_name: Codebase Explorer
---

# ACE External Integrations Bridge

## Summary
Connectors that allow ACE to manage context and learning for other agent frameworks, such as `browser-use`, `LangChain`, and `Claude Code`. This enables ACE's context optimization logic to be used in various external tools.

## Core Flow
```mermaid
flowchart LR
    EXT[External Framework] --> W[ACE Wrapper]
    W --> CE[Context Enrichment]
    CE --> EX[Execution]
```

## Notable Gotchas & Tech Debt
- **Feature Flags**: Many integrations only load if specific libraries are present, leading to "hidden" capabilities or runtime errors if dependencies are missing.
- **Abstraction Leaks**: Some external frameworks have unique context requirements that are difficult to fit into the generic ACE model.

[[ace_agentic_context_engine.md]]
