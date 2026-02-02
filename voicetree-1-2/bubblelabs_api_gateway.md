---
color: blue
position:
  x: 716
  y: -2708
isContextNode: false
agent_name: Codebase Explorer
---

# BubbleLabs API Gateway & Connectivity

## Summary
The backend proxy layer (using Bun and Hono) that routes traffic and normalizes data between the JavaScript-based frontend and Python-based backend services. It manages authentication and request forwarding.

## Core Flow
```mermaid
flowchart LR
    R[Frontend Request] --> P[Proxy]
    P --> A[Auth Header Injection]
    A --> OE[OpenEvolve API]
    OE --> N[Response Normalization]
```

## Notable Gotchas & Tech Debt
- **Error Normalization**: Standardizing errors across multiple programming languages (JS/TS and Python) is complex and requires a unified JSON error format.
- **Payload Limits**: Large workflow definitions can exceed default proxy payload limits, requiring manual configuration adjustments.

[[bubblelabs.md]]
