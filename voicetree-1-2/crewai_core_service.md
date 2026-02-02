---
color: blue
position:
  x: 366
  y: -189
isContextNode: false
agent_name: Codebase Explorer
---

# CrewAI Core Integration Service

## Summary
The low-level foundation of the module that manages agent and task templates. It handles the instantiation of CrewAI objects and provides mock fallbacks for environments where the actual CrewAI SDK is not installed.

## Core Flow
```mermaid
flowchart LR
    C[Config] --> R[Template Registry]
    R --> F[CrewAI Factory]
    F --> O[CrewAI Objects/Agents]
```

## Notable Gotchas & Tech Debt
- **Mock Fidelity**: Mock implementations are very simple and do not accurately simulate the behavior of real agents, leading to false confidence in test environments.
- **Template Bloat**: The number of hardcoded templates is growing, making the `CrewAIService` class increasingly cluttered.

[[crewai_integration_layer.md]]
