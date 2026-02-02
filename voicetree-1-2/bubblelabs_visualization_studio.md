---
color: blue
position:
  x: 979
  y: -2715
isContextNode: false
agent_name: Codebase Explorer
---

# BubbleLabs Visualization Layer (Flow Studio)

## Summary
The React-based frontend application that uses `ReactFlow` to provide a drag-and-drop workflow designer. It allows users to visually compose agent tasks and monitor their progress in real-time.

## Core Flow
```mermaid
flowchart LR
    E[User Node Edit] --> S[Zustand Store]
    S --> D[API Dispatch]
    D --> R[Runtime Execution]
```

## Notable Gotchas & Tech Debt
- **Real-time Sync**: Synchronizing the complex frontend graph state with asynchronous backend execution requires robust handling of WebSockets and polling fallbacks.
- **Node Performance**: Large graphs with many custom visual components can lead to UI latency on lower-end hardware.

[[bubblelabs.md]]
