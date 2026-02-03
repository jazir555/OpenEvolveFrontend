# BubbleLab Architecture

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Principles](#architecture-principles)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Security Architecture](#security-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Integration Patterns](#integration-patterns)
- [Scalability Considerations](#scalability-considerations)
- [Technology Stack](#technology-stack)

---

## System Overview

BubbleLab is a modular, type-safe workflow automation platform that compiles visual workflows into production-ready TypeScript code. The architecture follows a monorepo pattern with clear separation between core packages, applications, and integration layers.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        BS[Bubble Studio UI]
    end

    subgraph "API Layer"
        API[BubbleLab API]
        Auth[Authentication Service]
        Execution[Workflow Execution Engine]
    end

    subgraph "Core Layer"
        BC[Bubble Core]
        BR[Bubble Runtime]
        BSchemas[Shared Schemas]
    end

    subgraph "Integration Layer"
        Bubbles[Bubble Definitions]
        Tools[Tool Bubbles]
        Services[Service Bubbles]
    end

    subgraph "External Services"
        AI[AI Providers]
        DB[(Database)]
        Storage[(Object Storage)]
    end

    BS -->|HTTP/WebSocket| API
    API --> Auth
    API --> Execution
    Execution --> BC
    BC --> BR
    BC --> BSchemas
    BC --> Bubbles
    Bubbles --> Tools
    Bubbles --> Services
    Tools --> AI
    Services --> DB
    Services --> Storage

    style BS fill:#e1f5ff
    style API fill:#fff4e1
    style BC fill:#f0e1ff
    style Bubbles fill:#e1ffe1
```

---

## Architecture Principles

### 1. **Type Safety First**
- All workflows compile to TypeScript
- Strong typing throughout the stack
- Compile-time error detection
- IDE-friendly development experience

### 2. **Separation of Concerns**
- Clear boundaries between UI, API, and core logic
- Independent packages with well-defined interfaces
- Minimal coupling between components

### 3. **Observability**
- Built-in tracing and logging
- Execution metrics and performance monitoring
- Token usage and cost tracking
- Detailed execution history

### 4. **Extensibility**
- Plugin-based bubble system
- Easy to add new integrations
- Custom tools and services
- Template system for reusability

### 5. **Developer Experience**
- Visual builder with code export
- Hot-reload development
- Clear error messages
- Comprehensive documentation

---

## Component Architecture

### Monorepo Structure

```mermaid
graph LR
    subgraph "BubbleLab Monorepo"
        Root[Root]

        subgraph "Applications"
            Studio[Bubble Studio<br/>React + Vite]
            API[BubbleLab API<br/>Bun + Hono]
        end

        subgraph "Core Packages"
            Core[Bubble Core<br/>Workflow Engine]
            Runtime[Bubble Runtime<br/>Execution Environment]
            Schemas[Shared Schemas<br/>Type Definitions]
            Scope[Scope Manager<br/>TypeScript Analysis]
            Create[Create App<br/>Project Generator]
        end

        subgraph "Deployment"
            Deploy[Deployment Configs]
            Docker[Docker Files]
            Compose[Docker Compose]
        end
    end

    Root --> Studio
    Root --> API
    Root --> Core
    Root --> Runtime
    Root --> Schemas
    Root --> Scope
    Root --> Create
    Root --> Deploy
    Root --> Docker
    Root --> Compose

    Studio --> Core
    Studio --> Schemas
    API --> Core
    API --> Runtime
    API --> Schemas
    Runtime --> Core
    Create --> Runtime
```

### Core Components

#### 1. Bubble Studio (Frontend)
**Location:** `apps/bubble-studio`

**Responsibilities:**
- Visual workflow builder interface
- Real-time collaboration features
- Workflow validation and testing
- Code generation and export
- Execution history visualization

**Technology Stack:**
- React 18+ with TypeScript
- Vite for build tooling
- TanStack Router for routing
- TanStack Query for state management
- Radix UI for components
- TailwindCSS for styling

#### 2. BubbleLab API (Backend)
**Location:** `apps/bubblelab-api`

**Responsibilities:**
- RESTful API for workflow CRUD operations
- Workflow execution orchestration
- User authentication and authorization
- Credential management
- Webhook handling
- Template management

**Technology Stack:**
- Bun runtime
- Hono web framework
- Drizzle ORM for database access
- PostgreSQL/SQLite for persistence
- Zod for validation
- OpenTelemetry for tracing

#### 3. Bubble Core (Workflow Engine)
**Location:** `packages/bubble-core`

**Responsibilities:**
- Bubble definition and validation
- Workflow compilation
- Type generation
- Dependency injection
- Execution context management

**Key Features:**
- Abstract base classes for all bubble types
- Strong TypeScript typing
- Parameter validation
- Result transformation
- Error handling

#### 4. Bubble Runtime (Execution Environment)
**Location:** `packages/bubble-runtime`

**Responsibilities:**
- Workflow execution engine
- Bubble instantiation
- Data flow management
- Error propagation
- Execution tracing
- Metrics collection

**Key Features:**
- Isolated execution contexts
- Automatic dependency resolution
- Circular dependency detection
- Timeout management
- Memory limits

---

## Data Flow

### Workflow Creation Flow

```mermaid
sequenceDiagram
    participant User
    participant Studio
    participant API
    participant DB

    User->>Studio: Create new workflow
    Studio->>Studio: Initialize BubbleFlow instance
    User->>Studio: Add bubbles (drag & drop)
    Studio->>Studio: Validate connections
    User->>Studio: Configure bubble parameters
    Studio->>Studio: Generate TypeScript code
    User->>Studio: Save workflow
    Studio->>API: POST /api/bubble-flows
    API->>DB: Persist workflow definition
    DB-->>API: Confirmation
    API-->>Studio: Workflow ID
    Studio-->>User: Success message
```

### Workflow Execution Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Runtime
    participant Core
    participant Bubble
    participant External

    Client->>API: POST /api/execute {workflowId, payload}
    API->>Runtime: Load workflow
    Runtime->>Core: Compile workflow
    Core-->>Runtime: Executable code

    loop For each bubble
        Runtime->>Bubble: Instantiate with params
        Bubble->>Bubble: Validate inputs
        Bubble->>External: Execute action
        External-->>Bubble: Result
        Bubble->>Bubble: Transform output
        Bubble-->>Runtime: BubbleResult
        Runtime->>Runtime: Log execution
    end

    Runtime-->>API: ExecutionResult
    API->>API: Store execution history
    API-->>Client: Final result
```

### AI-Powered Flow Generation

```mermaid
sequenceDiagram
    participant User
    participant PearlAI
    participant Generator
    participant Compiler
    participant Studio

    User->>PearlAI: "Scrape Reddit and summarize"
    PearlAI->>Generator: Generate workflow structure
    Generator->>Generator: Select appropriate bubbles
    Generator-->>PearlAI: Workflow plan
    PearlAI->>Compiler: Generate TypeScript code
    Compiler-->>PearlAI: BubbleFlow class
    PearlAI-->>Studio: Render visual workflow
    Studio-->>User: Display generated workflow

    Note over User,Studio: User can now edit,<br/>test, or export the workflow
```

---

## Security Architecture

### Authentication & Authorization

```mermaid
graph TB
    subgraph "Authentication Flow"
        Client[Client App]
        Clerk[Clerk Auth]
        API[BubbleLab API]

        Client -->|1. Login| Clerk
        Clerk -->|2. JWT Token| Client
        Client -->|3. API Request + Token| API
        API -->|4. Verify Token| Clerk
        Clerk -->|5. User Info| API
        API -->|6. Context| API
    end

    style Client fill:#e1f5ff
    style Clerk fill:#ffe1e1
    style API fill:#fff4e1
```

### Credential Management

```mermaid
graph LR
    User[User] -->|Encrypt| Enc[Encrypted Credentials]
    Enc -->|Store| DB[(Database)]
    Enc -->|Runtime Decrypt| Runtime[Workflow Execution]

    Key[Encryption Key] -->|App Start| Runtime
    Key -->|Store| DB

    style User fill:#e1f5ff
    style Enc fill:#ffe1e1
    style Runtime fill:#e1ffe1
    style DB fill:#f0e1ff
```

**Security Features:**

1. **Encryption at Rest**
   - AES-256 encryption for stored credentials
   - Environment-specific encryption keys
   - Key rotation support

2. **Secure Execution**
   - Isolated execution contexts
   - Timeout enforcement
   - Memory limits
   - Sandbox mode for tool bubbles

3. **API Security**
   - CORS configuration
   - Rate limiting
   - Request validation
   - SQL injection prevention (parameterized queries)

4. **Webhook Security**
   - HMAC signature verification
   - Replay attack prevention
   - IP whitelisting support

---

## Deployment Architecture

### Development Environment

```mermaid
graph TB
    subgraph "Local Development"
        Dev[Developer Machine]

        subgraph "Services"
            Studio[Bubble Studio<br/>:3000]
            API[BubbleLab API<br/>:3001]
            DB[(SQLite Dev DB)]
        end

        Dev --> Studio
        Dev --> API
        API --> DB
        Studio --> API
    end
```

### Production Deployment

```mermaid
graph TB
    subgraph "Production Infrastructure"
        subgraph "Load Balancer"
            LB[Load Balancer / CDN]
        end

        subgraph "Application Layer"
            Studio1[Bubble Studio 1]
            Studio2[Bubble Studio 2]
            API1[BubbleLab API 1]
            API2[BubbleLab API 2]
        end

        subgraph "Database Layer"
            PG[(PostgreSQL<br/>Primary)]
            PG_Replica[(PostgreSQL<br/>Replica)]
        end

        subgraph "Caching Layer"
            Redis[(Redis Cache)]
        end

        subgraph "Monitoring"
            Jaeger[Jaeger Tracing]
            Prometheus[Prometheus Metrics]
            Loki[Log Aggregation]
        end

        LB --> Studio1
        LB --> Studio2
        Studio1 --> API1
        Studio2 --> API2
        API1 --> PG
        API2 --> PG
        API1 --> Redis
        API2 --> Redis
        API1 --> Jaeger
        API2 --> Jaeger
        API1 --> Prometheus
        API2 --> Prometheus
        API1 --> Loki
        API2 --> Loki
    end
```

### Container Architecture

```mermaid
graph TB
    subgraph "Docker Compose / Kubernetes"
        subgraph "Frontend"
            StudioContainer[Bubble Studio Container]
        end

        subgraph "Backend"
            APIContainer[BubbleLab API Container]
            WorkerContainer[Background Worker]
        end

        subgraph "Infrastructure"
            DBContainer[PostgreSQL Container]
            RedisContainer[Redis Container]
            Traefik[Traefik Reverse Proxy]
        end

        subgraph "Observability"
            JaegerContainer[Jaeger Container]
            PrometheusContainer[Prometheus Container]
            GrafanaContainer[Grafana Container]
        end

        Traefik --> StudioContainer
        Traefik --> APIContainer
        APIContainer --> DBContainer
        APIContainer --> RedisContainer
        APIContainer --> JaegerContainer
        WorkerContainer --> DBContainer
        WorkerContainer --> RedisContainer
        WorkerContainer --> JaegerContainer
    end
```

---

## Integration Patterns

### Bubble System Architecture

```mermaid
classDiagram
    class Bubble {
        <<abstract>>
        +id: string
        +name: string
        +params: TParams
        +action(): Promise~BubbleResult~
    }

    class ServiceBubble {
        <<abstract>>
        +authenticate()
        +executeAction()
    }

    class ToolBubble {
        <<abstract>>
        +validateInput()
        +transformOutput()
    }

    class AIAgentBubble {
        +model: AIModel
        +message: string
        +action(): Promise~AIResponse~
    }

    class HTTPBubble {
        +url: string
        +method: HttpMethod
        +headers: Record
        +action(): Promise~HttpResponse~
    }

    class BubbleFlow {
        +bubbles: Bubble[]
        +execute(payload): Promise~FlowResult~
    }

    Bubble <|-- ServiceBubble
    Bubble <|-- ToolBubble
    ToolBubble <|-- AIAgentBubble
    ServiceBubble <|-- HTTPBubble
    BubbleFlow o-- Bubble
```

### Integration Points

```mermaid
graph TB
    subgraph "BubbleLab Core"
        BF[BubbleFlow]
    end

    subgraph "Service Bubbles"
        HTTP[HTTP]
        PostgreSQL[PostgreSQL]
        Slack[Slack]
        GoogleSheets[Google Sheets]
        Gmail[Gmail]
    end

    subgraph "Tool Bubbles"
        AIAgent[AI Agent]
        Code[Code Editor]
        Chart[Chart.js]
        Maps[Google Maps]
    end

    subgraph "External APIs"
        OpenAI[OpenAI API]
        Anthropic[Anthropic API]
        Google[Google API]
    end

    BF --> HTTP
    BF --> PostgreSQL
    BF --> Slack
    BF --> GoogleSheets
    BF --> Gmail
    BF --> AIAgent
    BF --> Code
    BF --> Chart
    BF --> Maps

    AIAgent --> OpenAI
    AIAgent --> Anthropic
    AIAgent --> Google
    GoogleSheets --> Google
    Gmail --> Google
    Maps --> Google

    style BF fill:#e1f5ff
    style HTTP fill:#ffe1e1
    style AIAgent fill:#e1ffe1
```

---

## Scalability Considerations

### Horizontal Scaling

**API Layer:**
- Stateless design enables horizontal scaling
- Load balancer distributes requests
- Session state stored in Redis
- Database connection pooling

**Execution Layer:**
- Worker queues for async execution
- Background job processing
- Priority queues for different execution tiers
- Dead letter queues for failed jobs

### Vertical Scaling

**Database:**
- Read replicas for query scaling
- Connection pooling limits
- Query optimization
- Indexing strategy

**Caching:**
- Redis for session data
- API response caching
- Workflow template caching
- Bubble metadata caching

### Performance Optimization

```mermaid
graph LR
    subgraph "Optimization Strategies"
        Cache[Caching]
        Pool[Connection Pooling]
        Async[Async Processing]
        Lazy[Lazy Loading]
    end

    subgraph "Benefits"
        Perf[Improved Performance]
        Scale[Better Scalability]
        Cost[Reduced Costs]
        User[Better UX]
    end

    Cache --> Perf
    Cache --> Scale
    Pool --> Perf
    Pool --> Cost
    Async --> Scale
    Async --> User
    Lazy --> Perf
    Lazy --> User
```

---

## Technology Stack

### Frontend

| Technology | Purpose | Version |
|------------|---------|---------|
| React | UI Framework | 18+ |
| TypeScript | Type Safety | 5+ |
| Vite | Build Tool | Latest |
| TanStack Router | Routing | Latest |
| TanStack Query | State Management | Latest |
| Radix UI | Component Library | Latest |
| TailwindCSS | Styling | 3+ |
| React Flow | Visual Builder | Latest |

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| Bun | Runtime | Latest |
| Hono | Web Framework | Latest |
| Drizzle ORM | Database ORM | Latest |
| PostgreSQL | Database | 14+ |
| SQLite | Dev Database | 3+ |
| Zod | Validation | Latest |
| OpenTelemetry | Tracing | Latest |
| JWT | Authentication | Latest |

### DevOps

| Technology | Purpose | Version |
|------------|---------|---------|
| Docker | Containerization | Latest |
| Docker Compose | Local Dev | Latest |
| Kubernetes | Orchestration | 1.25+ |
| Traefik | Reverse Proxy | Latest |
| Prometheus | Metrics | Latest |
| Jaeger | Tracing | Latest |
| Grafana | Visualization | Latest |

### Development Tools

| Technology | Purpose | Version |
|------------|---------|---------|
| pnpm | Package Manager | 8+ |
| Turbo | Build System | Latest |
| ESLint | Linting | Latest |
| Prettier | Formatting | Latest |
| Husky | Git Hooks | Latest |
| TypeScript | Type Checking | 5+ |

---

## Monitoring & Observability

### Metrics Collection

```mermaid
graph TB
    subgraph "Application Metrics"
        API[API Metrics]
        Runtime[Runtime Metrics]
        DB[Database Metrics]
    end

    subgraph "OpenTelemetry Stack"
        Collector[OTel Collector]
        Trace[Traces]
        Metric[Metrics]
        Log[Logs]
    end

    subgraph "Visualization"
        Prometheus[Prometheus]
        Grafana[Grafana]
        Jaeger[Jaeger]
    end

    API --> Collector
    Runtime --> Collector
    DB --> Collector

    Collector --> Trace
    Collector --> Metric
    Collector --> Log

    Trace --> Jaeger
    Metric --> Prometheus
    Log --> Grafana

    Prometheus --> Grafana

    style Collector fill:#e1f5ff
    style Jaeger fill:#ffe1e1
    style Prometheus fill:#e1ffe1
    style Grafana fill:#f0e1ff
```

### Key Metrics

**System Metrics:**
- CPU usage
- Memory usage
- Disk I/O
- Network I/O

**Application Metrics:**
- Request rate
- Response time
- Error rate
- Active executions

**Business Metrics:**
- Workflow executions
- Bubble usage statistics
- Token consumption
- Cost tracking

---

## Future Architecture Considerations

### Planned Enhancements

1. **Multi-Region Deployment**
   - Geographic distribution
   - Data locality compliance
   - Reduced latency

2. **Advanced Caching**
   - Edge caching
   - CDN integration
   - Intelligent cache invalidation

3. **Event-Driven Architecture**
   - Message queue integration
   - Event sourcing
   - CQRS pattern

4. **Microservices Transition**
   - Service decomposition
   - Independent scaling
   - Technology diversity

5. **AI/ML Enhancements**
   - Workflow optimization
   - Anomaly detection
   - Predictive scaling

---

## Documentation References

- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Deployment instructions
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Development setup
- [docs/runbooks/](./docs/runbooks/) - Operational procedures
- [docs/integrations/](./docs/integrations/) - Integration guides
- [docs/security/](./docs/security/) - Security documentation

---

## Support & Community

- **Documentation:** https://docs.bubblelab.ai/
- **Discord:** https://discord.gg/PkJvcU2myV
- **GitHub Issues:** https://github.com/bubblelabai/BubbleLab/issues
- **Email:** support@bubblelab.ai

---

*Last Updated: January 2026*
