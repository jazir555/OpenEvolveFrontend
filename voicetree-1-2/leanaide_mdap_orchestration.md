---
color: blue
position:
  x: -154
  y: -1613
isContextNode: false
agent_name: Codebase Explorer
---

# LeanAide MDAP Orchestration Layer

## Summary
The central hub for coordinating multiple proof generation strategies. It enables parallel execution of different agents (MCTS, Evolution, etc.) and uses a voting-based consensus mechanism to select the most promising proof candidate.

## Core Flow
```mermaid
flowchart TD
    T[Theorem] --> AS[Agent Selection]
    AS --> PE[Parallel Execution: MCTS, Evolution, etc.]
    PE --> CV[Consensus Voting]
    CV --> WS[Winner Selection]
```

## Notable Gotchas & Tech Debt
- **Canonicalization**: Robustly aggregating votes requires 'canonicalizing' Lean code so that syntactically different but semantically identical proofs are counted correctly.
- **Vote Weights**: Determining the correct weight for different agent types in the consensus process is an ongoing tuning challenge.

[[leanaide_formal_verification.md]]
