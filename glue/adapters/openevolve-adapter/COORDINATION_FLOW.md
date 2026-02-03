# OpenEvolve Coordination Flow Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenEvolve Federation                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    OpenEvolve Main Adapter                            │   │
│  │                   (Orchestration Hub)                                 │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐     │   │
│  │  │              Anti-Corruption Layer (ACL)                    │     │   │
│  │  │  • Canonical Schema Validation                             │     │   │
│  │  │  • Data Transformation (Source ↔ Canonical)                 │     │   │
│  │  │  • Contract Enforcement                                     │     │   │
│  │  └─────────────────────────────────────────────────────────────┘     │   │
│  │                               ↓                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐     │   │
│  │  │                  Circuit Breakers                            │     │   │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │     │   │
│  │  │  │  Z3  │ │LeanA │ │RAGBit│ │Vector│ │Graphi│ │Karate│      │     │   │
│  │  │  │      │ │ ide  │ │  s   │ │  DB  │ │  ti  │ │ Club│      │     │   │
│  │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │     │   │
│  │  └─────────────────────────────────────────────────────────────┘     │   │
│  │                               ↓                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐     │   │
│  │  │             Integration Coordinator                         │     │   │
│  │  │  • Problem Type → Adapter Mapping                           │     │   │
│  │  │  • Capability-based Selection                               │     │   │
│  │  │  • Execution Planning (Parallel/Sequential)                 │     │   │
│  │  │  • Health Monitoring & Fallback                             │     │   │
│  │  └─────────────────────────────────────────────────────────────┘     │   │
│  │                               ↓                                      │   │
│  │  ┌────────────────────────┐  ┌──────────────────────────────────┐    │   │
│  │  │  Workflow Orchestrator  │  │    Knowledge Aggregator          │    │   │
│  │  │  • Stage Management     │  │  • Unified Query Interface       │    │   │
│  │  │  • Progress Tracking    │  │  • Multi-source Fusion           │    │   │
│  │  │  • Error Recovery       │  │  • Semantic Search               │    │   │
│  │  │  • Checkpointing        │  │  • Knowledge Graph Construction  │    │   │
│  │  └────────────────────────┘  └──────────────────────────────────┘    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Event Bus                                     │   │
│  │              (Pub/Sub for Async Communication)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Integrated Adapters                                  │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │ Z3 Adapter   │  │LeanAide      │  │RAGBits       │                    │
│  │              │  │Adapter       │  │Adapter       │                    │
│  │ SMT Solver   │  │              │  │              │                    │
│  └──────────────┘  │Proof Assistant│  │RAG System    │                    │
│                    └──────────────┘  └──────────────┘                    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │Vector DB     │  │Graphiti      │  │KarateClub    │                    │
│  │Adapter       │  │Adapter       │  │Adapter       │                    │
│  │              │  │              │  │              │                    │
│  │Vector Store  │  │Graph DB      │  │Graph ML      │                    │
│  └──────────────┘  └──────────────┘  └──────────────┘                    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐                                       │
│  │Knowledge     │  │Other         │                                       │
│  │Engine        │  │Adapters...   │                                       │
│  └──────────────┘  └──────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Request Flow: Coordination Sequence

```
1. CLIENT REQUEST
   ↓
   POST /openevolve/workflows
   {
     "workflow_id": "math-proof-001",
     "problem_statement": "Prove theorem X",
     "sub_problems": [...]
   }
   ↓
2. ANTI-CORRUPTION LAYER (ACL)
   ↓
   ✓ Validate request against canonical schema
   ✓ Transform to internal format
   ↓
3. INTEGRATION COORDINATOR
   ↓
   Analyze Request:
   - Problem Type: "formal_verification"
   - Domain: "mathematics"
   - Capabilities: ["smt_solving", "tactic_execution"]
   ↓
   Select Adapters:
   → Z3 (smt_solving capability)
   → LeanAide (tactic_execution capability)
   ↓
   Plan Execution:
   - Parallel: YES (different adapter types)
   - Estimated Duration: 3000ms
   - Fallback Order: [Z3, LeanAide]
   ↓
4. CIRCUIT BREAKER CHECK
   ↓
   Check Z3 Circuit Breaker:
   → State: CLOSED ✓
   Check LeanAide Circuit Breaker:
   → State: CLOSED ✓
   ↓
5. PARALLEL EXECUTION (with Retry)
   ↓
   ┌─────────────────────┬──────────────────────┐
   │                     │                      │
   ↓                     ↓                      ↓
Z3 Adapter         LeanAide Adapter      (Other Adapters...)
   │                     │
   │  ←───────┐         │  ←───────┐
   │          │         │          │
   ↓          │         ↓          │
Request 1   │      Request 1      │
   │          │         │          │
   ↓          │         ↓          │
Success! ✓   │      Failure! ✗    │
   │          │         │          │
   └──────────┘         ↓
                   Retry 2 (1000ms delay)
                        ↓
                   Retry 3 (2000ms delay)
                        ↓
                   Success! ✓
   ↓                     ↓
   └─────────────────────┴──────────────────────┐
                                                 ↓
6. AGGREGATE RESULTS
   ↓
   Z3 Result: {
     "status": "success",
     "data": { "proof": "..." }
   }
   LeanAide Result: {
     "status": "success",
     "data": { "tactics": [...] }
   }
   ↓
7. TRANSFORM TO CANONICAL (ACL)
   ↓
   Canonical Format: {
     "workflow_id": "math-proof-001",
     "status": "completed",
     "results": [
       {
         "source": "z3",
         "content": {...}
       },
       {
         "source": "leanaide",
         "content": {...}
       }
     ]
   }
   ↓
8. RESPONSE TO CLIENT
   ↓
   200 OK
   {
     "workflow_id": "math-proof-001",
     "status": "completed",
     "progress": 1.0,
     "results": [...]
   }
```

## Workflow Execution Stages

```
STAGE 1: Content Analysis
┌─────────────────────────────────────────────┐
│ Input: Problem Statement                     │
│ Adapters: [RAGBits, Vector DB]              │
│ Execution: Parallel                          │
│ Output: Analyzed Context                    │
└─────────────────────────────────────────────┘
                    ↓
STAGE 2: Decomposition Planning
┌─────────────────────────────────────────────┐
│ Input: Analyzed Context                     │
│ Adapters: [Graphiti, Knowledge Engine]      │
│ Execution: Sequential                       │
│ Output: Sub-problems Plan                   │
└─────────────────────────────────────────────┘
                    ↓
STAGE 3-N: Sub-problem Solving (Parallel)
┌─────────────────────────────────────────────┐
│ Sub-Problem 1 → [Z3]                        │
│ Sub-Problem 2 → [LeanAide]                  │
│ Sub-Problem 3 → [Z3 + LeanAide]             │
│ Execution: Parallel (different adapters)    │
│ Output: Individual Solutions                │
└─────────────────────────────────────────────┘
                    ↓
STAGE N+1: Solution Assembly
┌─────────────────────────────────────────────┐
│ Input: All Sub-problem Solutions            │
│ Adapters: [Knowledge Engine]                │
│ Execution: Sequential                       │
│ Output: Assembled Final Solution            │
└─────────────────────────────────────────────┘
                    ↓
STAGE N+2: Final Verification
┌─────────────────────────────────────────────┐
│ Input: Final Solution                       │
│ Adapters: [Z3 (verification mode)]          │
│ Execution: Sequential                       │
│ Output: Verification Report                 │
└─────────────────────────────────────────────┘
                    ↓
STAGE N+3: Knowledge Extraction
┌─────────────────────────────────────────────┐
│ Input: Entire Workflow State                │
│ Adapters: [Knowledge Aggregator]            │
│ Execution: Sequential                       │
│ Output: Knowledge Artifacts                 │
└─────────────────────────────────────────────┘
                    ↓
              WORKFLOW COMPLETE
```

## Knowledge Aggregation Flow

```
CLIENT QUERY
↓
POST /openevolve/knowledge/query
{
  "query": "Pythagorean theorem proofs",
  "domain": "mathematics",
  "max_results": 20
}
↓
KNOWLEDGE AGGREGATOR
↓
Check Cache
↓
Cache Miss → Query All Sources
↓
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│   Z3    │LeanAide │RAGBits  │Vector DB│Graphiti │
│  (0.8)  │  (0.7)  │  (0.9)  │  (0.85) │  (0.75) │
└─────────┴─────────┴─────────┴─────────┴─────────┘
    ↓         ↓         ↓         ↓         ↓
    └─────────┴─────────┴─────────┴─────────┘
                     ↓
              FUSE RESULTS
              • Sort by relevance
              • Apply threshold (>0.7)
              • Limit to 20 results
                     ↓
              CACHE RESULTS
              (5 minute TTL)
                     ↓
            RETURN TO CLIENT
```

## Circuit Breaker State Transitions

```
INITIAL STATE: CLOSED (Normal Operation)
    ↓
[Request succeeds]
    ↓
Reset failure count
    ↓
Stay CLOSED ✓

[Request fails]
    ↓
Increment failure count
    ↓
Failure count >= threshold?
    ↓
YES → OPEN (Reject requests)
    ↓
Wait for timeout (60s)
    ↓
HALF_OPEN (Allow test request)
    ↓
[Test request succeeds]
    ↓
CLOSED ✓

[Test request fails]
    ↓
OPEN again
```

## Error Handling Flow

```
REQUEST MADE
    ↓
ERROR OCCURS
    ↓
Classify Error Type
    ↓
┌────────────────┬────────────────┬────────────────┐
│                │                │                │
Transient      Logic          System
Failure        Failure        Failure
(Timeout)      (Bad Data)     (Service Down)
    ↓              ↓              ↓
Retry with      Dead Letter    Circuit
Backoff         Queue (DLQ)    Breaker OPEN
    │              │              │
    └───Max        └───Log        └───Stop
    Retries          Error          Trying
    ↓                                    │
Success ✓                              │
    ↓                                    │
Return                                   │
Result                                   │
                                         │
                                    Wait for
                                    Health Check
                                         │
                                    ↓ Pass
                                    HALF_OPEN
                                         │
                                    Test Request
                                         │
                                    ↓ Success
                                    CLOSED
```

## Logging Flow (JSON Lines)

```
┌─────────────────────────────────────────────────────────────┐
│ Request Received                                            │
│ ↓                                                            │
│ Log: {                                                      │
│   "timestamp": "2025-02-03T12:34:56.789Z",                  │
│   "level": "info",                                          │
│   "message": "Creating workflow",                           │
│   "service": "openevolve-adapter",                          │
│   "correlation_id": "abc-123-def",                          │
│   "workflow_id": "math-proof-001"                           │
│ }                                                           │
│ ↓                                                            │
│ ACL Validation                                              │
│ ↓                                                            │
│ Log: {                                                      │
│   "timestamp": "2025-02-03T12:34:57.123Z",                  │
│   "level": "debug",                                         │
│   "message": "Schema validation passed",                    │
│   "service": "openevolve-adapter",                          │
│   "correlation_id": "abc-123-def",                          │
│   "validation_errors": 0                                    │
│ }                                                           │
│ ↓                                                            │
│ Adapter Coordination                                        │
│ ↓                                                            │
│ Log: {                                                      │
│   "timestamp": "2025-02-03T12:34:58.456Z",                  │
│   "level": "info",                                          │
│   "message": "Selected adapters for coordination",          │
│   "service": "integration-coordinator",                     │
│   "correlation_id": "abc-123-def",                          │
│   "selected_adapters": ["z3", "leanaide"],                  │
│   "execution_mode": "parallel"                              │
│ }                                                           │
│ ↓                                                            │
│ ... (more logs for each stage) ...                         │
│ ↓                                                            │
│ Response Sent                                               │
│ ↓                                                            │
│ Log: {                                                      │
│   "timestamp": "2025-02-03T12:35:10.789Z",                  │
│   "level": "info",                                          │
│   "message": "Workflow completed successfully",             │
│   "service": "openevolve-adapter",                          │
│   "correlation_id": "abc-123-def",                          │
│   "workflow_id": "math-proof-001",                          │
│   "duration_ms": 14000                                      │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow: Canonical Schema Transformation

```
Z3 Adapter Format (Source)
{
  "teamName": "solver-blue",
  "teamRole": "solver",
  "models": [
    {
      "modelId": "z3-solver",
      "config": {...}
    }
  ]
}
        ↓
    ACL TRANSFORM
        ↓
Canonical Format
{
  "name": "solver-blue",
  "role": "Blue",  // Canonical role
  "members": [
    {
      "model_id": "z3-solver",
      "api_key": "",
      "api_base": "...",
      "temperature": 0.7,
      "max_tokens": 4096
    }
  ],
  "description": null
}
        ↓
    VALIDATE SCHEMA
        ↓
    USE IN ORCHESTRATION
        ↓
    TRANSFORM BACK (if needed)
        ↓
Z3 Adapter Format (Target)
{
  "teamName": "solver-blue",
  "teamRole": "Blue",
  "models": [...]
}
```

## Summary

The OpenEvolve coordination flow ensures:

1. **Decoupling**: ACL prevents schema leakage
2. **Resilience**: Circuit breakers prevent cascading failures
3. **Intelligence**: Smart adapter selection based on capabilities
4. **Observability**: Structured logs with correlation IDs
5. **Flexibility**: Parallel/sequential execution as needed
6. **Reliability**: Retries and fallbacks for transient failures
7. **Performance**: Caching and parallel execution where possible
