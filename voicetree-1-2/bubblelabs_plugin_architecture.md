---
color: blue
position:
  x: 612
  y: -2922
isContextNode: false
agent_name: Codebase Explorer
---

# BubbleLabs Plugin Architecture

## Summary
A modular extension system that allows isolated modules (e.g., LeanAide, Mitosis) to register custom visual nodes and specialized backend logic. It defines a formal lifecycle for plugins from discovery to shutdown.

## Core Flow
```mermaid
flowchart TD
    D[Discovery] --> V[Metadata Validation]
    V --> L[Lifecycle: Load]
    L --> I[Lifecycle: Init]
    I --> S[Lifecycle: Start]
    S --> EB[Event Bus Sub]
```

## Notable Gotchas & Tech Debt
- **Shutdown Order**: Plugins must be shut down in reverse-dependency order to avoid dangling resources or memory leaks.
- **Plugin Isolation**: Ensuring that one faulty plugin doesn't crash the entire BubbleLabs runtime is an ongoing challenge.

[[bubblelabs.md]]
