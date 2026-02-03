# Phase 1 Architecture Overview

System architecture for OpenEvolve Knowledge Engine Phase 1 implementation.

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Integration Points](#integration-points)
7. [Scalability](#scalability)

## System Overview

The Knowledge Engine Phase 1 provides a comprehensive system for knowledge management with temporal reasoning, graph-based extraction, and intelligent retrieval.

### Design Principles

1. **Modularity**: Each component is independently deployable
2. **Extensibility**: Easy to add new integrations and processors
3. **Performance**: Optimized for large-scale knowledge operations
4. **Reliability**: Fault-tolerant with graceful degradation

## Core Components

```mermaid
graph TB
    A[Applications] --> B[Knowledge Engine API]
    B --> C[Temporal Knowledge Engine]
    B --> D[Extraction Pipeline]
    B --> E[Visualization Service]

    C --> F[Graphiti Bridge]
    C --> G[Local Storage]

    D --> H[KG-Gen Pipeline]
    D --> I[OneKE Integration]
    D --> J[AI-KG Integration]

    F --> K[Neo4j]
    F --> L[Vector Store]

    H --> K
    I --> K
    J --> K
```

### Component Details

#### 1. Temporal Knowledge Engine

**Purpose**: Core engine for temporal knowledge operations

**Features**:
- Temporal artifact storage
- Point-in-time queries
- Contradiction detection
- Timeline reconstruction

**Technology**: Python, asyncio, Neo4j

#### 2. Extraction Pipeline

**Purpose**: Multi-stage knowledge graph extraction

**Features**:
- 3-stage pipeline (Entity → Relation → Dedup)
- Parallel chunk processing
- Advanced deduplication
- Neo4j auto-upload

**Technology**: DSPy, Neo4j, scikit-learn

#### 3. Graphiti Bridge

**Purpose**: Integration with Graphiti temporal knowledge graph

**Features**:
- Hybrid search (BM25 + Vector + Graph)
- Temporal filtering
- Episode management
- Custom type mapping

**Technology**: Graphiti-core, Neo4j

#### 4. Visualization Service

**Purpose**: Interactive knowledge graph visualization

**Features**:
- Force-directed graphs
- Timeline views
- Entity exploration
- Relationship filtering

**Technology**: D3.js, React

## Architecture Diagrams

### High-Level Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        A[Documents]
        B[API Calls]
        C[Web Interface]
    end

    subgraph "Processing Layer"
        D[Extraction Pipeline]
        E[Temporal Engine]
        F[Query Processor]
    end

    subgraph "Storage Layer"
        G[Neo4j]
        H[Vector Store]
        I[File Storage]
    end

    subgraph "Output Layer"
        J[REST API]
        K[Visualization]
        L[Export]
    end

    A --> D
    B --> E
    C --> F
    D --> G
    D --> H
    E --> G
    E --> H
    F --> G
    G --> J
    H --> K
    I --> L
```

### Temporal Knowledge System

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Engine
    participant Graphiti
    participant Neo4j

    Client->>API: Add Knowledge
    API->>Engine: add_knowledge_temporal()
    Engine->>Engine: Create KnowledgeArtifact
    Engine->>Graphiti: Add Episode
    Graphiti->>Neo4j: Store with Timestamps
    Neo4j-->>Graphiti: Success
    Graphiti-->>Engine: Artifact Created
    Engine-->>API: Success
    API-->>Client: Artifact ID

    Client->>API: Query at Time
    API->>Engine: query_at_time()
    Engine->>Graphiti: Temporal Search
    Graphiti->>Neo4j: Time-filtered Query
    Neo4j-->>Graphiti: Valid Artifacts
    Graphiti-->>Engine: Results
    Engine-->>API: KnowledgeArtifacts
    API-->>Client: Results
```

### Extraction Pipeline

```mermaid
graph TB
    A[Input Document] --> B[Document Chunker]
    B --> C{Parallel Processing}

    C --> D[Chunk 1]
    C --> E[Chunk 2]
    C --> F[Chunk N]

    D --> G[Stage 1: Entity Extraction]
    E --> G
    F --> G

    G --> H[Stage 2: Relation Extraction]
    H --> I[Stage 3: Deduplication]

    I --> J[SEMHASH]
    I --> K[LM Clustering]

    J --> L[Merge Results]
    K --> L

    L --> M[Knowledge Graph]
    M --> N[Neo4j Upload]
```

## Data Flow

### Knowledge Addition Flow

```mermaid
flowchart TD
    Start([User Input]) --> Validate[Validate Input]
    Validate --> CreateArtifact[Create KnowledgeArtifact]
    CreateArtifact --> StoreLocal[Store Locally]
    StoreLocal --> CheckGraphiti{Graphiti Available?}
    CheckGraphiti -->|Yes| AddEpisode[Add Episode to Graphiti]
    CheckGraphiti -->|No| Return[Return Artifact]
    AddEpisode --> UpdateIndex[Update Search Index]
    UpdateIndex --> Return
    Return --> End([Complete])
```

### Query Processing Flow

```mermaid
flowchart TD
    Start([Query Request]) --> Parse[Parse Query]
    Parse --> CheckTemporal{Temporal Query?}
    CheckTemporal -->|Yes| ApplyTemporal[Apply Temporal Filters]
    CheckTemporal -->|No| StandardSearch[Standard Search]
    ApplyTemporal --> HybridSearch{Use Hybrid?}
    StandardSearch --> HybridSearch
    HybridSearch -->|Yes| ExecuteHybrid[Execute Hybrid Search]
    HybridSearch -->|No| ExecuteBasic[Execute Basic Search]
    ExecuteHybrid --> Rerank[Rerank Results]
    ExecuteBasic --> Convert[Convert to Artifacts]
    Rerank --> Convert
    Convert --> Filter[Filter by Time]
    Filter --> Rank[Rank Results]
    Rank --> Return([Return Results])
```

## Technology Stack

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.9+ |
| Async Framework | asyncio | Built-in |
| Graph Database | Neo4j | 5.x |
| Vector Store | Qdrant | 1.7+ |
| ORM | SQLAlchemy | 2.x |
| API Framework | FastAPI | 0.100+ |

### AI/ML

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | OpenAI GPT-4 | Entity/Relation extraction |
| Embeddings | sentence-transformers | Semantic similarity |
| DSPy | DSPy | Extraction pipeline |
| Graphiti | graphiti-core | Temporal knowledge |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React | UI components |
| Visualization | D3.js | Graph rendering |
| UI Library | Material-UI | Component library |
| State Management | Redux | Application state |

## Integration Points

### External Systems

```mermaid
graph LR
    A[Knowledge Engine] --> B[Graphiti]
    A --> C[KG-Gen]
    A --> D[OneKE]
    A --> E[AI-KG]

    F[Neo4j] <--> B
    F <--> C
    F <--> D
    F <--> E

    G[Vector Store] <--> A
```

### APIs

#### REST API

```
POST   /api/v1/knowledge          # Add knowledge
GET    /api/v1/knowledge/{id}     # Get knowledge
PUT    /api/v1/knowledge/{id}     # Update knowledge
DELETE /api/v1/knowledge/{id}     # Delete knowledge
GET    /api/v1/search             # Search knowledge
POST   /api/v1/query/temporal     # Temporal query
GET    /api/v1/timeline/{entity}  # Entity timeline
```

#### Python API

```python
from knowledge_engine import TemporalKnowledgeEngine

# Initialize
engine = TemporalKnowledgeEngine()

# Add knowledge
artifact = await engine.add_knowledge_temporal(...)

# Query
results = await engine.query_at_time(...)
```

## Scalability

### Horizontal Scaling

```mermaid
graph TB
    A[Load Balancer] --> B[Instance 1]
    A --> C[Instance 2]
    A --> D[Instance N]

    B --> E[Shared Neo4j Cluster]
    C --> E
    D --> E

    B --> F[Shared Vector Store]
    C --> F
    D --> F
```

### Performance Optimization

1. **Connection Pooling**: Reuse database connections
2. **Batch Processing**: Process multiple artifacts together
3. **Caching**: Cache frequently accessed knowledge
4. **Async Operations**: Non-blocking I/O
5. **Parallel Processing**: Multi-core utilization

### Throughput

| Operation | Throughput | Latency (p50) |
|-----------|-----------|---------------|
| Add Knowledge | 1000/sec | 50ms |
| Query (Current) | 500/sec | 100ms |
| Query (Temporal) | 200/sec | 200ms |
| Extract (Small Doc) | 50/sec | 2s |
| Extract (Large Doc) | 5/sec | 30s |

## Security

### Authentication

- JWT-based authentication
- API key authentication
- OAuth 2.0 integration

### Authorization

- Role-based access control (RBAC)
- Group-based knowledge isolation
- Fine-grained permissions

### Data Protection

- Encryption at rest (Neo4j encryption)
- Encryption in transit (TLS)
- PII redaction
- Audit logging

## Monitoring

### Metrics

- Knowledge artifact count
- Query latency (p50, p95, p99)
- Extraction success rate
- System resource usage

### Logging

Structured JSON logging with context:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "event": "knowledge_added",
  "artifact_id": "artifact_001",
  "user": "user@example.com",
  "duration_ms": 45
}
```

## Deployment

### Container Architecture

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY knowledge_engine/ ./knowledge_engine/

CMD ["python", "-m", "knowledge_engine.api"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  knowledge-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:5.11
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
```

## Next Steps

- [Temporal System Design](temporal_system_design.md)
- [Extraction Pipeline Architecture](extraction_pipeline_architecture.md)
- [Data Flow Diagrams](data_flow.md)
