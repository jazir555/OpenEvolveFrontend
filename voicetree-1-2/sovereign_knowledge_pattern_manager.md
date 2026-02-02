---
color: blue
position:
  x: 630
  y: -2075
isContextNode: false
agent_name: Codebase Explorer
---

# Sovereign Knowledge & Pattern Manager

## Summary
The "long-term memory" of the system. it extracts reusable patterns from high-quality completed workflows and stores them in a database. These patterns are then injected into future runs to improve decomposition and refinement quality.

## Core Flow
```mermaid
flowchart LR
    SP[Success Plan] --> PE[Pattern Extraction]
    PE --> DB[(Sovereign DB)]
    DB --> NPC[New Problem Context]
    NPC --> DE[Decomposition]
```

## Notable Gotchas & Tech Debt
- **Knowledge Pollution**: Patterns from low-quality runs must be strictly filtered (quality < 0.7) to prevent "polluting" the database with bad strategies.
- **Pattern Overfitting**: Excessive reliance on historical patterns can sometimes prevent the engine from finding novel, more effective solutions for unique problems.

[[sovereign_system.md]]
