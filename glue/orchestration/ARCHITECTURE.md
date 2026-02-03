# Orchestration Layer Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OpenEvolve Mega-Structure                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Application Layer                            │   │
│  │                                                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Web API    │  │   CLI Tool   │  │  Dashboard   │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Main Adapter Layer                             │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │            OpenEvolve Main Adapter                              │  │   │
│  │  │                                                                  │  │   │
│  │  │  - Request routing                                              │  │   │
│  │  │  - Response aggregation                                          │  │   │
│  │  │  - Error handling                                                │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     ORCHESTRATION LAYER ◄─────────────────────────   │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                     Event Bus                                   │  │   │
│  │  │                                                                  │  │   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │  │   │
│  │  │  │  Publish   │  │ Subscribe  │  │   Replay   │               │  │   │
│  │  │  └────────────┘  └────────────┘  └────────────┘               │  │   │
│  │  │                                                                  │  │   │
│  │  │  Events:                                                         │  │   │
│  │  │  • KnowledgeExtracted  • ProofVerified  • GraphUpdated          │  │   │
│  │  │  • VectorIndexed     • RAGRetrieved  • Workflow*               │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                    │                                 │   │
│  │                                    ▼                                 │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                   Workflow Engine                              │  │   │
│  │  │                                                                  │  │   │
│  │  │  Predefined Workflows:                                          │  │   │
│  │  │  • Z3 → LeanAide Cross-Validation                               │  │   │
│  │  │  • RAGBits → Vector DB → Knowledge Graph                        │  │   │
│  │  │  • Document → Embedding → Index                                 │  │   │
│  │  │                                                                  │  │   │
│  │  │  Features:                                                       │  │   │
│  │  │  • Sequential execution  • Parallel execution  • State mgmt    │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                    │                                 │   │
│  │  ┌────────────────┬─────────────────┴────────────────┬──────────────┐  │   │
│  │  │                │                                  │              │  │   │
│  │  ▼                ▼                                  ▼              │  │   │
│  │  ┌──────────────┐ ┌──────────────┐       ┌──────────────────────┐  │  │   │
│  │  │  Correlation │ │    DLQ       │       │  Circuit Breakers    │  │  │   │
│  │  │   Tracker    │ │              │       │                      │  │  │   │
│  │  │              │ │ • Retry      │       │ • Failure detection  │  │  │   │
│  │  │ • UUID v4    │ │ • Backoff    │       │ • Auto recovery      │  │  │   │
│  │  │ • Tracing    │ │ • Monitoring │       │ • Timeout handling   │  │  │   │
│  │  │ • Lineage    │ │              │       │                      │  │  │   │
│  │  └──────────────┘ └──────────────┘       └──────────────────────┘  │  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Adapter Layer (Glue)                                 │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Z3 Adapter  │  │LeanAide      │  │ RAGBits      │  │ Vector DB    │   │
│  │              │  │  Adapter     │  │  Adapter     │  │  Adapter     │   │
│  │ • SMT solver │  │ • Proofs     │  │ • Chunks     │  │ • Chroma     │   │
│  │ • Theorems   │  │ • Formal ver │  │ • Extraction │  │ • Pinecone   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Graphiti    │  │ KarateClub   │  │ BubbleLab    │  │   ...        │   │
│  │  Adapter     │  │  Adapter     │  │  Adapter     │  │              │   │
│  │ • Knowledge  │  │ • Graph ML   │  │ • Algebra    │  │              │   │
│  │   Graph      │  │ • Clustering │  │ • Visualization│   │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Core Projects (Immutable)                           │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │     Z3       │  │   Lean4      │  │  RAGBits     │  │  ChromaDB    │   │
│  │  Core        │  │   Core       │  │   Core       │  │   Core       │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │   Graphiti   │  │ KarateClub   │  │ BubbleLab    │                      │
│  │   Core       │  │   Core       │  │   Core       │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Event Flow Diagram

```
┌─────────────┐
│   Client    │
│   Request   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Main Adapter                              │
│                                                               │
│  1. Generate Correlation ID (UUID v4)                        │
│  2. Create Distributed Trace                                 │
│  3. Route to appropriate workflow                            │
└──────┬────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Workflow Engine                              │
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Step 1     │───▶│  Step 2     │───▶│  Step 3     │     │
│  │  RAGBits    │    │  Vector DB  │    │  Graphiti   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │            │
│         └───────────────────┴───────────────────┘            │
│                             │                                 │
│                             ▼                                 │
│                    ┌───────────────┐                         │
│                    │ Event:        │                         │
│                    │ Workflow*     │                         │
│                    └───────────────┘                         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Event Bus                                │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Event: KnowledgeExtracted              │   │
│  └────────────┬────────────────────────────────────────┘   │
│               │                                              │
│               ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Event: VectorIndexed                   │   │
│  └────────────┬────────────────────────────────────────┘   │
│               │                                              │
│               ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Event: GraphUpdated                    │   │
│  └────────────┬────────────────────────────────────────┘   │
└────────────────┼─────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Adapters Subscribe                        │
│                                                               │
│  • Vector DB subscribes to KnowledgeExtracted               │
│  • Graphiti subscribes to VectorIndexed                     │
│  • Analytics subscribes to all events                       │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
┌──────────────┐
│ Event Handler│
└──────┬───────┘
       │
       ▼
  Try to Execute
       │
       ├──Success────────▶ Log Success ──▶ Publish Next Event
       │
       └──Failure────────▶ Analyze Error
                             │
                             ├── Transient? ──▶ Retry with Backoff
                             │                   │
                             │                   └──Still Fail? ──▶ DLQ
                             │
                             └── Logic Error? ──▶ Dead Letter Queue
                                                    │
                                                    ▼
                                         ┌──────────────────┐
                                         │   Dead Letter    │
                                         │     Queue        │
                                         │                  │
                                         │ • Event stored   │
                                         │ • Retry scheduled│
                                         │ • Alert sent     │
                                         └──────────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────┐
                                         │ Retry Process    │
                                         │ (periodic)       │
                                         └──────────────────┘
                                                    │
                                                    ├──Success─▶ Complete
                                                    │
                                                    └──Failed──▶ Manual Review
```

## Cross-Validation Workflow (Z3 ↔ LeanAide)

```
┌─────────────────────────────────────────────────────────────────┐
│              Z3 → LeanAide Cross-Validation Workflow            │
└─────────────────────────────────────────────────────────────────┘

Input: Formal Proof
  │
  ▼
┌─────────────────┐
│  Step 1: Z3     │
│  Verification   │
└────────┬────────┘
         │
         ▼
    Result: Valid/Invalid
         │
         ▼
┌─────────────────┐
│  Step 2: Lean   │
│  Verification   │
└────────┬────────┘
         │
         ▼
    Result: Valid/Invalid
         │
         ▼
┌─────────────────┐
│  Step 3: Cross  │
│  Validation     │
└────────┬────────┘
         │
         ▼
    ┌──────────────┐
    │ Both Agree?  │
    └──┬────────┬──┘
       │        │
      Yes       No
       │        │
       ▼        ▼
   ┌──────┐  ┌─────────────┐
   │ VALID│  │ Investigate │
   └──────┘  │ Mismatch    │
             └─────────────┘

Output: {
  cross_validated: true/false,
  z3_result: { verified: true, system: 'z3' },
  lean_result: { verified: true, system: 'lean-aide' }
}
```

## RAG Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│              RAGBits → Vector DB → Knowledge Graph              │
└─────────────────────────────────────────────────────────────────┘

Input: Document
  │
  ▼
┌─────────────────┐
│  RAGBits:       │
│  Extract Chunks │──▶ Event: KnowledgeExtracted
└────────┬────────┘
         │
         ▼
    Chunks: [
      { chunk_id, content, metadata },
      ...
    ]
         │
         ▼
┌─────────────────┐
│  Vector DB:     │
│  Create Embeds  │
└────────┬────────┘
         │
         ▼
    Embeddings: [
      { chunk_id, vector: [1536 floats] },
      ...
    ]
         │
         ▼
┌─────────────────┐
│  Vector DB:     │
│  Index Embeds   │──▶ Event: VectorIndexed
└────────┬────────┘
         │
         ▼
    Indexed: {
      index_id: 'idx-123',
      count: 10
    }
         │
         ▼
┌─────────────────┐
│  Graphiti:      │
│  Update Graph   │──▶ Event: GraphUpdated
└────────┬────────┘
         │
         ▼
    Graph Updated: {
      graph_id: 'graph-456',
      nodes_added: 10,
      edges_added: 15
    }

Output: Complete RAG pipeline processed
```

## Component Interaction Matrix

| Component        | EventBus | WorkflowEngine | DLQ | CorrelationTracker |
|------------------|----------|----------------|-----|-------------------|
| **EventBus**     | -        | ✓              | ✓   | ✓                 |
| **WorkflowEngine**| ✓       | -              | ✓   | ✓                 |
| **DLQ**          | ✓        | -              | -   | -                 |
| **CorrelationTracker**| ✓   | ✓              | -   | -                 |
| **Adapters**     | ✓        | -              | ✓   | ✓                 |

## Data Flow Summary

```
Request → Main Adapter → Workflow Engine → Adapters → Core Projects
                                   │
                                   ▼
                            Event Bus
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
          Correlation          DLQ (if error)    Subscribers
          Tracker
```

## Technology Stack

- **Event Bus**: EventEmitter (memory), Redis (distributed), RabbitMQ/Kafka (enterprise)
- **Workflow**: Custom DSL with state management
- **Retry**: Exponential backoff with jitter
- **Circuit Breaker**: Custom implementation
- **Tracing**: Distributed span tracking
- **Logging**: JSON Lines (structured)

## Configuration

All configuration via environment variables (Law of Configuration Explicitness):

```bash
EVENT_BUS_TYPE=memory|redis|rabbitmq|kafka
EVENT_BUS_URL=connection_url
EVENT_PERSISTENCE_ENABLED=true
CIRCUIT_BREAKER_ENABLED=true
DLQ_ENABLED=true
```
