---
color: blue
position:
  x: 1300
  y: -1439
isContextNode: false
agent_name: Codebase Explorer
---

# ROMA Toolkit & Tool Management

## Summary
The capability provider for the ROMA framework. it manages the bridge between agents and external systems (e.g., Google Search, local file system, code interpreters) via the Model Context Protocol (MCP) and local toolkits.

## Core Flow
```mermaid
flowchart LR
    A[Agent] --> T[Toolkit Manager]
    T --> MCP[MCP Clients]
    T --> L[Local Tools]
    MCP & L --> E[Environment]
```

## Notable Gotchas & Tech Debt
- **Registry Fragmentation**: Tools are registered in multiple places across the codebase, making it difficult to maintain a single source of truth for agent capabilities.
- **MCP Versioning**: Changes in the MCP protocol or server implementations can break existing tool definitions.

[[roma_reasoning_on_multi_agent.md]]
