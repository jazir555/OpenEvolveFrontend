# Architecture Diagrams

**Version:** 1.0.0
**Last Updated:** 2025-01-03

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Execution Methods](#execution-methods)
5. [Integration Patterns](#integration-patterns)
6. [Deployment](#deployment)

---

## System Overview

### High-Level Architecture

```mermaid
graph TB
    User[User/Hephaestus Agent] --> MCP[MCP Tools Layer]

    MCP --> Decomp[Decomposition Engine]
    MCP --> Teams[Team Manager]
    MCP --> Gauntlets[Gauntlet Manager]

    Decomp --> Semantic[Semantic Strategy]
    Decomp --> Hierarchical[Hierarchical Strategy]
    Decomp --> Flow[Flow-Based Strategy]

    Semantic --> LLM[OpenEvolve LLM]
    Hierarchical --> Template[Template Engine]
    Flow --> Template

    LLM --> Cache[Cache Layer]

    style User fill:#e1f5ff
    style MCP fill:#fff4e1
    style Decomp fill:#f0e1ff
    style LLM fill:#ffe1f0
```

---

### Component Interaction

```mermaid
sequenceDiagram
    participant H as Hephaestus
    participant M as MCP Tools
    participant D as Decomp Engine
    participant L as OpenEvolve LLM
    participant T as Team Manager
    participant G as Gauntlet Manager

    H->>M: analyze_problem()
    M->>D: analyze_problem()
    D->>L: Request analysis
    L-->>D: Analysis result
    D-->>M: Analysis
    M-->>H: Analysis

    H->>M: decompose_problem()
    M->>D: decompose(semantic)
    D->>L: Request decomposition
    L-->>D: Sub-problems
    D-->>M: Sub-problems
    M-->>H: Sub-problems

    H->>M: solve_sub_problem()
    M->>T: assign_team()
    T-->>M: Team assigned
    M->>D: solve()
    D->>L: Generate solution
    L-->>D: Solution
    D-->>M: Solution
    M-->>H: Solution

    H->>M: critique_solution()
    M->>G: run_gauntlet(Red Team)
    G-->>M: Critique
    M-->>H: Critique

    H->>M: verify_solution()
    M->>G: run_gauntlet(Gold Team)
    G-->>M: Verification
    M-->>H: Verification
```

---

## Component Architecture

### Decomposition Engine

```mermaid
graph LR
    A[DecompositionEngine] --> B[SemanticDecomposition]
    A --> C[HierarchicalDecomposition]
    A --> D[FlowBasedDecomposition]

    B --> E[LLM Analysis]
    B --> F[Semantic Clustering]

    C --> G[Top-Level Components]
    C --> H[Recursive Breakdown]

    D --> I[Stage Identification]
    D --> J[Dependency Mapping]

    style A fill:#e1f5ff
    style B fill:#f0e1ff
    style C fill:#f0e1ff
    style D fill:#f0e1ff
```

---

### MCP Tools Architecture

```mermaid
graph TB
    subgraph "MCP Tools Layer"
        A[analyze_problem_for_decomposition]
        B[decompose_problem_into_sub_problems]
        C[create_decomposition_plan]
        D[solve_sub_problem_with_team]
        E[critique_solution_with_gauntlet]
        F[verify_solution_with_gauntlet]
        G[list_available_teams]
        H[list_available_gauntlets]
        I[get_decomposition_status]
    end

    subgraph "Execution Methods"
        D --> T[Traditional LLM]
        D --> CL[Claudiomiro]
        D --> DP[DataPizza]
        D --> R[ROMA]
        D --> HY[Hybrid]
        D --> RM[ROMA-MDAP-MAKER]
        D --> AU[Auto]
    end

    style A fill:#fff4e1
    style B fill:#fff4e1
    style D fill:#ffe1f0
```

---

## Data Flow

### Decomposition Workflow

```mermaid
flowchart TD
    Start([Start]) --> Input[Input Problem Statement]

    Input --> Analyze[Analyze Problem]
    Analyze --> Analysis{Analysis Complete?}
    Analysis -->|No| Reanalyze[Re-analyze]
    Reanalyze --> Analyze
    Analysis -->|Yes| Select[Select Strategy]

    Select --> Strat{Strategy?}
    Strat -->|Semantic| Sem[Semantic Decomp]
    Strat -->|Hierarchical| Hier[Hierarchical Decomp]
    Strat -->|Flow| Flow[Flow-Based Decomp]

    Sem --> Decomp[Decomposition]
    Hier --> Decomp
    Flow --> Decomp

    Decomp --> Validate{Quality Check}
    Validate -->|Fail| Adjust[Adjust Parameters]
    Adjust --> Select
    Validate -->|Pass| Output[Output Sub-Problems]

    Output --> Dependencies[Resolve Dependencies]
    Dependencies --> Order[Calculate Execution Order]
    Order --> End([End])

    style Start fill:#90EE90
    style End fill:#90EE90
    style Input fill:#e1f5ff
    style Output fill:#ffe1f0
```

---

### Solution Integration Flow

```mermaid
flowchart TD
    SubProblems[Sub-Problems] --> Exec{Execution Method}

    Exec -->|Traditional| Trad[Traditional LLM]
    Exec -->|Claudiomiro| Cl[Claudiomiro CLI]
    Exec -->|DataPizza| DP[DataPizza Agents]
    Exec -->|ROMA| R[ROMA Recursive]
    Exec -->|Hybrid| Hy[ROMA + Decomposition]
    Exec -->|ROMA-MDAP-MAKER| RM[Zero-Error Voting]

    Trad --> Sol[Solutions Generated]
    Cl --> Sol
    DP --> Sol
    R --> Sol
    Hy --> Sol
    RM --> Sol

    Sol --> Critique[Red Team Critique]
    Critique --> Issues{Issues Found?}

    Issues -->|Yes| Revise[Revise Solution]
    Revise --> Critique

    Issues -->|No| Verify[Gold Team Verification]
    Verify --> Approved{Approved?}

    Approved -->|No| Reject[Reject/Revise]
    Reject --> Exec

    Approved -->|Yes| Final[Final Solutions]
    Final --> Int[Integrate Solutions]
    Int --> Complete[Complete Solution]

    style SubProblems fill:#e1f5ff
    style Complete fill:#90EE90
    style Sol fill:#fff4e1
    style Final fill:#ffe1f0
```

---

## Execution Methods

### Method Comparison

```mermaid
graph TB
    subgraph "Traditional Method"
        T1[LLM Prompt] --> T2[Solution Generation]
        T2 --> T3[Optional Evolution]
    end

    subgraph "Claudiomiro Method"
        C1[Problem Statement] --> C2[Claudiomiro CLI]
        C2 --> C3[Autonomous Development]
        C3 --> C4[File Generation]
    end

    subgraph "DataPizza Method"
        D1[Problem Statement] --> D2[Multi-Agent System]
        D2 --> D3[Tool Use]
        D3 --> D4[Collaborative Solution]
    end

    subgraph "ROMA Method"
        R1[Problem Statement] --> R2[Recursive Decomposition]
        R2 --> R3[Hierarchical Execution]
        R3 --> R4[Aggregated Solution]
    end

    subgraph "ROMA-MDAP-MAKER Method"
        M1[Problem Statement] --> M2[ROMA Decomposition]
        M2 --> M3[Atomic Tasks]
        M3 --> M4[MAKER Voting]
        M4 --> M5[Zero-Error Result]
    end

    style T3 fill:#e1f5ff
    style C4 fill:#f0e1ff
    style D4 fill:#fff4e1
    style R4 fill:#ffe1f0
    style M5 fill:#e1ffe1
```

---

### ROMA-MDAP-MAKER Zero-Error Flow

```mermaid
flowchart TD
    Input[Task] --> ROMA[ROMA Recursive Decomposition]
    ROMA --> Atomic[Atomic Tasks Generated]

    Atomic --> Vote[MAKER Voting Round]
    Vote --> Sample1[Sample 1]
    Vote --> Sample2[Sample 2]
    Vote --> Sample3[Sample N]

    Sample1 --> Check1{Red-Flag Check}
    Sample2 --> Check2{Red-Flag Check}
    Sample3 --> Check3{Red-Flag Check}

    Check1 -->|Clean| Results1[Result 1]
    Check1 -->|Flagged| Reject1[Reject]

    Check2 -->|Clean| Results2[Result 2]
    Check2 -->|Flagged| Reject2[Reject]

    Check3 -->|Clean| Results3[Result N]
    Check3 -->|Flagged| Reject3[Reject]

    Results1 --> Ahead{First-to-Ahead-by-K?}
    Results2 --> Ahead
    Results3 --> Ahead

    Ahead -->|Yes| Winner[Winner Selected]
    Ahead -->|No| Vote2[Next Voting Round]

    Vote2 --> Adaptive{Adaptive K?}
    Adaptive -->|Yes| IncreaseK[Increase K]
    Adaptive -->|No| MoreSamples[More Samples]
    IncreaseK --> Vote2

    MoreSamples --> Vote

    Winner --> Confidence[Confidence Score]
    Confidence --> ErrorRate[Error Rate <br/>99.9%+]

    ErrorRate --> Success[Zero-Error Solution]
    Reject1 --> Retry[Retry with New Samples]
    Reject2 --> Retry
    Reject3 --> Retry

    Retry --> Vote

    style Input fill:#e1f5ff
    style Success fill:#90EE90
    style Winner fill:#fff4e1
```

---

## Integration Patterns

### Hephaestus Integration

```mermaid
graph TB
    subgraph "Hephaestus Orchestrator"
        H1[Phase 1: Decomposition]
        H2[Phase 2: Solution Generation]
        H3[Phase 3: Quality Assurance]
        H4[Phase 4: Integration]
    end

    subgraph "Decomposition Workflow"
        D1[Stage 0: Analysis]
        D2[Stage 1: Decomposition]
        D3[Stage 2: Planning]
        D4[Stage 3: Solving]
        D5[Stage 4: Critique]
        D6[Stage 5: Verification]
        D7[Stage 6: Integration]
    end

    H1 --> D1
    D1 --> D2
    D2 --> D3

    H2 --> D4
    D3 --> D4

    H3 --> D5
    D4 --> D5
    D5 --> D6

    H4 --> D7
    D6 --> D7

    style H1 fill:#e1f5ff
    style H2 fill:#fff4e1
    style H3 fill:#ffe1f0
    style H4 fill:#e1ffe1
```

---

### OpenEvolve Integration

```mermaid
graph LR
    subgraph "Decomposition Engine"
        DE[Decomposition Engine]
    end

    subgraph "OpenEvolve Client"
        OE[OpenEvolve Client]
        Cache[Cache Layer]
        API[LLM API]
    end

    subgraph "Evolutionary Engine"
        EE[Evolutionary Engine]
        Pop[Population]
        Fit[Fitness Function]
        Sel[Selection]
        Mut[Mutation]
    end

    DE --> OE
    OE --> Cache
    OE --> API

    DE --> EE
    EE --> Pop
    Pop --> Fit
    Fit --> Sel
    Sel --> Mut
    Mut --> Pop

    Cache --> OE

    style DE fill:#e1f5ff
    style OE fill:#fff4e1
    style EE fill:#f0e1ff
```

---

## Deployment

### Production Deployment

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Load Balancer]
    end

    subgraph "Application Servers"
        AS1[App Server 1]
        AS2[App Server 2]
        AS3[App Server N]
    end

    subgraph "Decomposition Service"
        DS1[Decomp Instance 1]
        DS2[Decomp Instance 2]
        DS3[Decomp Instance N]
    end

    subgraph "Cache Layer"
        Redis[(Redis Cluster)]
    end

    subgraph "Message Queue"
        MQ[RabbitMQ/Kafka]
    end

    subgraph "Databases"
        PG[(PostgreSQL)]
        Mongo[(MongoDB)]
    end

    LB --> AS1
    LB --> AS2
    LB --> AS3

    AS1 --> DS1
    AS2 --> DS2
    AS3 --> DS3

    DS1 --> Redis
    DS2 --> Redis
    DS3 --> Redis

    DS1 --> MQ
    DS2 --> MQ
    DS3 --> MQ

    DS1 --> PG
    DS2 --> PG
    DS3 --> PG

    DS1 --> Mongo
    DS2 --> Mongo
    DS3 --> Mongo

    style LB fill:#e1f5ff
    style Redis fill:#f0e1ff
    style MQ fill:#fff4e1
    style PG fill:#ffe1f0
    style Mongo fill:#e1ffe1
```

---

### Development Environment

```mermaid
graph LR
    Dev[Developer] --> Git[Git Repository]
    Git --> Local[Local Development]
    Local --> Docker[Docker Compose]

    Docker --> App[App Container]
    Docker --> DB[Database Container]
    Docker --> Cache[Cache Container]

    App --> Test[Automated Tests]
    Test --> CI[CI/CD Pipeline]

    CI --> Staging[Staging Environment]
    Staging --> Prod[Production Environment]

    style Dev fill:#e1f5ff
    style Local fill:#fff4e1
    style Prod fill:#90EE90
```

---

## Performance Optimization

### Caching Strategy

```mermaid
graph TB
    Request[Request] --> CacheCheck{Cache Hit?}

    CacheCheck -->|Yes| Return[Return Cached]
    CacheCheck -->|No| Process[Process Request]

    Process --> LLM[LLM Call]
    LLM --> Result[Result]

    Result --> Store[Store in Cache]
    Store --> Return

    CacheCheck -->|Stale| Invalidate[Invalidate Cache]
    Invalidate --> Process

    style Request fill:#e1f5ff
    style Return fill:#90EE90
    style LLM fill:#ffe1f0
```

---

### Parallel Execution

```mermaid
graph TB
    SubProblems[Sub-Problems] --> Filter{Dependencies?}

    Filter -->|No Deps| Parallel[Parallel Execution]
    Filter -->|Has Deps| Serial[Serial Execution]

    Parallel --> Pool[Thread Pool]
    Pool --> W1[Worker 1]
    Pool --> W2[Worker 2]
    Pool --> W3[Worker N]

    W1 --> Results[Collect Results]
    W2 --> Results
    W3 --> Results

    Serial --> Queue[Execution Queue]
    Queue --> Process[Process Sequentially]
    Process --> Results

    Results --> Complete[Complete]

    style Parallel fill:#e1f5ff
    style Serial fill:#fff4e1
    style Complete fill:#90EE90
```

---

## Monitoring

### Metrics Flow

```mermaid
graph LR
    App[Application] --> Metrics[Metrics Collection]

    Metrics --> Prometheus[Prometheus]
    Metrics --> Grafana[Grafana]

    App --> Logs[Log Files]
    Logs --> ELK[ELK Stack]

    Prometheus --> Alert[AlertManager]
    Alert --> Pager[PagerDuty/Slack]

    Grafana --> Dashboard[Dashboard]
    Dashboard --> User[User]

    style App fill:#e1f5ff
    style Prometheus fill:#fff4e1
    style Grafana fill:#f0e1ff
    style User fill:#90EE90
```

---

## Security Architecture

```mermaid
graph TB
    Client[Client] --> API[API Gateway]

    API --> Auth[Authentication]
    Auth --> Valid{Valid Token?}

    Valid -->|No| Reject[Reject]
    Valid -->|Yes| Rate[Rate Limiting]

    Rate --> App[Application]

    App --> Encrypt[Data Encryption]
    Encrypt --> DB[(Database)]

    App --> Audit[Audit Logging]
    Audit --> SIEM[SIEM System]

    style Client fill:#e1f5ff
    style Reject fill:#ff9999
    style App fill:#fff4e1
    style DB fill:#90EE90
```

---

## Summary

### Key Architectural Principles

1. **Modularity**: Clear separation of concerns
2. **Scalability**: Horizontal scaling support
3. **Reliability**: Redundancy and failover
4. **Performance**: Caching and optimization
5. **Security**: Encryption and authentication
6. **Monitoring**: Comprehensive observability

### Technology Stack

- **Language**: Python 3.8+
- **LLM**: OpenAI GPT-4 / Anthropic Claude
- **Caching**: Redis
- **Database**: PostgreSQL / MongoDB
- **Message Queue**: RabbitMQ / Kafka
- **Monitoring**: Prometheus / Grafana
- **Deployment**: Docker / Kubernetes

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-03
