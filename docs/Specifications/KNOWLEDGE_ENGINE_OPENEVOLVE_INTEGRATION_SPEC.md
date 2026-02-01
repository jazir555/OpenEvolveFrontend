# OpenEvolve-Knowledge Engine Integration Specification

## Document Information
- **Version**: 1.0
- **Date**: February 1, 2026
- **Status**: Final
- **Authors**: OpenEvolve Team

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Integration Points](#integration-points)
4. [API Contracts](#api-contracts)
5. [Data Flow Patterns](#data-flow-patterns)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Monitoring & Observability](#monitoring--observability)
10. [Security](#security)
11. [Testing](#testing)

## Overview

### Purpose
This document specifies the integration between OpenEvolve and the Knowledge Engine. The integration enables OpenEvolve to leverage the Knowledge Engine's capabilities for knowledge extraction, processing, and reasoning during the evolutionary process, while maintaining a clean separation of concerns.

### Goals
- Enable OpenEvolve to access and utilize knowledge from the Knowledge Engine
- Provide bidirectional communication between systems
- Maintain loose coupling between OpenEvolve and Knowledge Engine
- Ensure scalable and resilient integration
- Support real-time knowledge updates and project context injection
- Resolve duplicate knowledge engine in openevolve package

### Non-Goals
- Tight coupling between the systems
- Sharing of internal data structures directly
- Complex interdependency management

## Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   OpenEvolve    │    │  Integration Layer   │    │  Knowledge      │
│                 │◄──►│  (OpenEvolve-KG)     │◄──►│  Engine         │
│  Evolution      │    │                      │    │                 │
│  Process        │    │  - Context Injection │    │  - Knowledge    │
│                 │    │  - Artifact Exchange │    │    Extraction   │
│  - Controllers  │    │  - API Gateway      │    │  - Graph        │
│  - Evaluators   │    │  - Event Bridge     │    │    Processing   │
│  - Database     │    │                      │    │  - Analytics    │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
```

### Component Mapping
| OpenEvolve Component | Integration Component | Knowledge Engine Component |
|---------------------|----------------------|---------------------------|
| Controller | OpenEvolveIntegration | KnowledgeEngineIntegration |
| Evaluator | ContextInjector | KnowledgeProcessor |
| Database | ArtifactExchange | KnowledgeStorage |
| API | APIBridge | QueryEngine |

## Integration Points

### 1. Project Context Injection
**Purpose**: Inject project-specific context into OpenEvolve's evolutionary process

**Mechanism**:
- OpenEvolveIntegration manages project lifecycles
- ContextInjector enriches queries with project context
- Bidirectional synchronization of project state

**API Endpoint**: `/api/projects/{project_id}/inject-context`

**Data Flow**:
1. OpenEvolve initiates project registration
2. Integration layer creates project context
3. Context is injected into evolutionary prompts
4. Results are tagged with project context

**Implementation Details**:
- Project contexts are stored in the Knowledge Engine's project registry
- Context injection happens at the prompt generation stage
- Projects maintain their own knowledge graphs for isolation

### 2. Knowledge Artifact Exchange
**Purpose**: Share knowledge artifacts between systems

**Mechanism**:
- OpenEvolve generates artifacts during evolution
- Knowledge Engine processes artifacts for insights
- Processed knowledge is fed back to OpenEvolve

**API Endpoint**: `/api/artifacts/exchange`

**Data Flow**:
1. OpenEvolve produces evolution artifacts
2. Artifacts are sent to Knowledge Engine
3. Knowledge Engine extracts insights
4. Insights are returned to OpenEvolve

**Implementation Details**:
- Artifacts include code changes, performance metrics, and error logs
- Knowledge Engine applies extraction techniques (DeepKE, KG-Gen, etc.)
- Processed artifacts are indexed for future retrieval

### 3. Real-Time Updates
**Purpose**: Synchronize state changes in real-time

**Mechanism**:
- Event-driven architecture
- WebSocket connections for real-time updates
- Circuit breaker patterns for resilience

**API Endpoint**: `/ws/updates`

**Data Flow**:
1. Events are published by either system
2. Integration layer processes events
3. Relevant updates are propagated
4. Systems react to updates asynchronously

**Implementation Details**:
- Events include project state changes, evolution milestones, and errors
- Updates are queued to handle network interruptions
- Circuit breakers prevent cascading failures

### 4. Multi-Project Support
**Purpose**: Manage multiple concurrent projects across systems

**Mechanism**:
- Project isolation
- Context switching
- Resource allocation

**API Endpoint**: `/api/projects/manage`

**Data Flow**:
1. Projects are registered with unique IDs
2. Resources are allocated per project
3. Context is maintained separately
4. Results are isolated by project

**Implementation Details**:
- Each project gets its own knowledge graph
- Resource quotas prevent one project from affecting others
- Cross-project knowledge sharing is configurable

## API Contracts

### Project Management API
```yaml
openapi: 3.0.0
info:
  title: OpenEvolve-Knowledge Engine Integration API
  version: 1.0.0
paths:
  /api/projects:
    post:
      summary: Register a new project
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ProjectContext'
      responses:
        '201':
          description: Project registered successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProjectRegistrationResponse'

  /api/projects/{project_id}/context:
    put:
      summary: Update project context
      parameters:
        - name: project_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ContextUpdate'
      responses:
        '200':
          description: Context updated successfully

  /api/projects/{project_id}/inject-context:
    post:
      summary: Inject project context into query
      parameters:
        - name: project_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
      responses:
        '200':
          description: Context injected successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  enriched_query:
                    type: string
```

### Artifact Exchange API
```yaml
paths:
  /api/artifacts/exchange:
    post:
      summary: Exchange knowledge artifacts
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                source_system:
                  type: string
                  enum: [openevolve, knowledge_engine]
                artifact_type:
                  type: string
                  enum: [code_change, performance_metric, error_log, evaluation_result]
                artifact_data:
                  type: object
                project_id:
                  type: string
                metadata:
                  type: object
      responses:
        '200':
          description: Artifact processed successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  processed_artifact:
                    type: object
                  insights:
                    type: array
                    items:
                      type: string
                  status:
                    type: string
                    enum: [success, partial, error]
                  processing_time_ms:
                    type: number
```

### Schema Definitions
```yaml
components:
  schemas:
    ProjectContext:
      type: object
      properties:
        project_id:
          type: string
          description: Unique identifier for the project
        name:
          type: string
          description: Human-readable name for the project
        description:
          type: string
          description: Brief description of the project
        stage:
          type: string
          enum: [initialized, planning, in_progress, review, completed, archived]
          description: Current lifecycle stage of the project
        metadata:
          type: object
          description: Additional project-specific metadata
        team_members:
          type: array
          items:
            type: string
          description: List of team member identifiers
        workflows:
          type: array
          items:
            type: string
          description: Active workflows associated with the project
        knowledge_graph_id:
          type: string
          description: Identifier for the project's knowledge graph
        created_at:
          type: string
          format: date-time
          description: Timestamp when the project was created
        updated_at:
          type: string
          format: date-time
          description: Timestamp when the project was last updated

    ContextUpdate:
      type: object
      properties:
        project_id:
          type: string
          description: Project identifier
        update_type:
          type: string
          description: Type of update being performed
        data:
          type: object
          description: Update payload
        timestamp:
          type: string
          format: date-time
          description: When the update occurred
        source:
          type: string
          description: System that originated the update

    ProjectRegistrationResponse:
      type: object
      properties:
        project_id:
          type: string
          description: Assigned project identifier
        status:
          type: string
          description: Registration status
        knowledge_graph_id:
          type: string
          description: Created knowledge graph identifier
        created_at:
          type: string
          format: date-time
          description: Registration timestamp
        endpoints:
          type: object
          properties:
            query_endpoint:
              type: string
              description: Endpoint for querying project knowledge
            update_endpoint:
              type: string
              description: Endpoint for updating project context
```

## Data Flow Patterns

### 1. Synchronous Request-Response
**Use Case**: Project registration, context injection, immediate queries
**Pattern**: OpenEvolve → Integration → Knowledge Engine → Response
**Characteristics**:
- Blocking calls
- Guaranteed delivery
- Error propagation
- Timeout handling

**Implementation**:
- Use HTTP/1.1 or HTTP/2 for synchronous communication
- Implement circuit breakers to prevent cascading failures
- Set appropriate timeouts (default 30 seconds)
- Retry with exponential backoff on transient errors

### 2. Asynchronous Event Streaming
**Use Case**: Real-time updates, artifact processing, notifications
**Pattern**: Event Publisher → Message Broker → Event Consumers
**Characteristics**:
- Non-blocking
- At-least-once delivery
- Event ordering preserved
- Backpressure handling

**Implementation**:
- Use message queues (e.g., Redis, RabbitMQ, Kafka) for event streaming
- Implement dead letter queues for failed events
- Use event sourcing for state reconstruction
- Implement consumer groups for scalability

### 3. Batch Processing
**Use Case**: Bulk artifact exchange, periodic synchronization, analytics
**Pattern**: Batch Job → Integration Layer → Knowledge Engine
**Characteristics**:
- Scheduled execution
- Bulk operations
- Error tolerance
- Progress tracking

**Implementation**:
- Use job schedulers (e.g., Celery, Airflow) for batch jobs
- Implement checkpointing for restart capability
- Track progress and provide status updates
- Handle partial failures gracefully

### 4. Publish-Subscribe
**Use Case**: Notification of state changes, broadcast updates
**Pattern**: Publisher → Topic → Multiple Subscribers
**Characteristics**:
- Fan-out pattern
- Loose coupling
- Scalable
- Message persistence

**Implementation**:
- Use pub-sub systems (e.g., Redis Pub/Sub, Apache Pulsar)
- Implement durable subscriptions for reliability
- Support topic-based filtering
- Handle subscriber failures gracefully

## Implementation Guidelines

### 1. Resolving Duplicate Knowledge Engine
The duplicate knowledge engine in the openevolve package must be resolved by:

1. **Redirecting imports**: Modify the openevolve package to import from the main knowledge engine
2. **Maintaining compatibility**: Ensure existing code continues to work
3. **Updating documentation**: Reflect the new architecture in docs
4. **Testing**: Verify all functionality remains intact

### 2. Integration Layer Implementation
The integration layer should be implemented as follows:

1. **Separate module**: Create a dedicated integration module
2. **Loose coupling**: Use interfaces/abstractions to decouple systems
3. **Error handling**: Implement comprehensive error handling
4. **Monitoring**: Include metrics and logging
5. **Configuration**: Allow runtime configuration of integration parameters

### 3. API Implementation
APIs should follow these principles:

1. **RESTful design**: Follow REST conventions where appropriate
2. **Consistent error handling**: Use consistent error response formats
3. **Rate limiting**: Implement rate limiting to prevent abuse
4. **Authentication**: Secure all endpoints appropriately
5. **Documentation**: Provide comprehensive API documentation

### 4. Data Consistency
Ensure data consistency through:

1. **Eventual consistency**: Accept eventual consistency where strong consistency isn't required
2. **Compensating transactions**: Implement compensating actions for failed operations
3. **Idempotency**: Design operations to be idempotent where possible
4. **Transaction boundaries**: Clearly define transaction boundaries

## Configuration

### Environment Variables
```bash
# Integration Configuration
OPENEVOLVE_KG_API_ENDPOINT=http://localhost:8000
OPENEVOLVE_KG_API_KEY=your-api-key
OPENEVOLVE_KG_ENABLE_REALTIME_UPDATES=true
OPENEVOLVE_KG_AUTO_SYNC=true
OPENEVOLVE_KG_SYNC_INTERVAL_SECONDS=300

# Circuit Breaker Configuration
OPENEVOLVE_KG_CIRCUIT_BREAKER_ENABLED=true
OPENEVOLVE_KG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
OPENEVOLVE_KG_CIRCUIT_BREAKER_RESET_TIMEOUT_MS=30000

# Timeout Configuration
OPENEVOLVE_KG_REQUEST_TIMEOUT=30
OPENEVOLVE_KG_CONNECTION_POOL_SIZE=10

# Batch Processing Configuration
OPENEVOLVE_KG_BATCH_SIZE=100
OPENEVOLVE_KG_BATCH_INTERVAL_SECONDS=60
```

### Integration Configuration File
```yaml
openevolve_kg_integration:
  api_endpoint: "http://localhost:8000"
  api_key: "${OPENEVOLVE_KG_API_KEY}"
  enable_realtime_updates: true
  auto_sync: true
  sync_interval_seconds: 300
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout_ms: 30000
    success_threshold: 2
  retry_policy:
    max_attempts: 3
    delay_ms: 1000
    backoff_multiplier: 2
  connection_pool:
    max_connections: 10
    max_keepalive_connections: 5
    keepalive_expiry: 300
  timeouts:
    connect_timeout: 10
    request_timeout: 30
  batch_processing:
    batch_size: 100
    interval_seconds: 60
    max_retries: 3
  event_streaming:
    buffer_size: 1000
    flush_interval_ms: 1000
    max_buffer_age_seconds: 300
```

## Deployment

### Container Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  openevolve:
    image: openevolve/core:latest
    environment:
      - OPENEVOLVE_KG_API_ENDPOINT=http://knowledge-engine:8000
      - OPENEVOLVE_KG_API_KEY=${OPENEVOLVE_KG_API_KEY}
    depends_on:
      - knowledge-engine
    ports:
      - "8080:8080"
    volumes:
      - ./openevolve-config:/app/config
    restart: unless-stopped

  knowledge-engine:
    image: openevolve/knowledge-engine:latest
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/kg_db
      - REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"
    volumes:
      - kg-data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  kg-data:
  redis-data:
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openevolve-integration
spec:
  replicas: 2
  selector:
    matchLabels:
      app: openevolve-integration
  template:
    metadata:
      labels:
        app: openevolve-integration
    spec:
      containers:
      - name: integration
        image: openevolve/integration-layer:latest
        env:
        - name: OPENEVOLVE_ENDPOINT
          value: "http://openevolve-service:8080"
        - name: KNOWLEDGE_ENGINE_ENDPOINT
          value: "http://knowledge-engine-service:8000"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        ports:
        - containerPort: 8081
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: integration-service
spec:
  selector:
    app: openevolve-integration
  ports:
    - protocol: TCP
      port: 8081
      targetPort: 8081
  type: ClusterIP
```

## Monitoring & Observability

### Metrics Collection
- **Request rates and latencies**: Track API call volume and response times
- **Error rates and types**: Monitor error frequency and categorization
- **Integration success/failure rates**: Measure integration reliability
- **Resource utilization**: Monitor CPU, memory, and disk usage
- **Queue depths**: Track message queue lengths for async operations
- **Processing throughput**: Measure artifacts processed per unit time

### Logging Standards
```json
{
  "timestamp": "2026-02-01T12:00:00Z",
  "level": "INFO",
  "service": "openevolve-integration",
  "correlation_id": "abc123-def456-ghi789",
  "trace_id": "trace-987-fed654-cba321",
  "span_id": "span-555",
  "event": "project_context_injected",
  "project_id": "proj-456",
  "duration_ms": 150,
  "source": "openevolve",
  "destination": "knowledge_engine",
  "user_id": "user-123",
  "session_id": "session-xyz"
}
```

### Health Checks
- **Connectivity**: Verify connectivity to both systems
- **Circuit breaker status**: Monitor circuit breaker states
- **Queue health**: Check message queue status
- **Resource availability**: Monitor system resources
- **Dependency health**: Verify dependent services are available

### Distributed Tracing
- Implement OpenTelemetry for distributed tracing
- Trace requests across system boundaries
- Correlate logs with traces
- Monitor service dependencies

## Security

### Authentication
- **API key-based authentication**: Use rotating API keys for service-to-service communication
- **Mutual TLS**: Implement mTLS for internal communications
- **OAuth 2.0**: Use OAuth 2.0 for user-facing endpoints
- **JWT tokens**: Use JWT for stateless authentication

### Authorization
- **Role-based access control**: Implement RBAC for fine-grained permissions
- **Project-level permissions**: Control access at the project level
- **Resource isolation**: Ensure proper resource isolation
- **Principle of least privilege**: Grant minimal necessary permissions

### Data Protection
- **Encryption in transit**: Use TLS 1.3 for all communications
- **Encryption at rest**: Encrypt sensitive data at rest
- **Data anonymization**: Anonymize data used for analytics
- **PII protection**: Implement PII detection and protection

### Audit Trail
- **Activity logging**: Log all integration activities
- **Access logs**: Maintain comprehensive access logs
- **Change tracking**: Track configuration changes
- **Compliance reporting**: Generate compliance reports

## Testing

### Unit Tests
- **Individual component testing**: Test each component in isolation
- **Mock external dependencies**: Use mocks for external services
- **Edge case validation**: Test boundary conditions
- **Error condition testing**: Verify error handling

### Integration Tests
- **End-to-end workflow testing**: Test complete workflows
- **Cross-system functionality**: Validate inter-system communication
- **Error scenario testing**: Test failure conditions
- **Performance validation**: Verify performance under load

### Performance Tests
- **Load testing**: Test expected traffic volumes
- **Stress testing**: Test peak load conditions
- **Latency measurements**: Measure response times
- **Resource utilization**: Monitor resource usage under load

### Chaos Engineering
- **Network partition simulation**: Test network failure scenarios
- **Service failure injection**: Test service failure handling
- **Resource exhaustion**: Test under resource constraints
- **Data inconsistency**: Test handling of inconsistent data

## Appendix

### Glossary
- **OpenEvolve**: The evolutionary coding agent system that performs automated code optimization and algorithm discovery
- **Knowledge Engine**: The knowledge extraction and processing system that provides semantic understanding and reasoning capabilities
- **Integration Layer**: The middleware facilitating communication between OpenEvolve and Knowledge Engine
- **Project Context**: Information about an active project including goals, constraints, history, and domain-specific requirements
- **Knowledge Artifact**: Processed information extracted from code, documentation, or other sources during the evolutionary process
- **Single Source of Truth (SSOT)**: The principle that duplicate knowledge engines should be consolidated to maintain consistency

### References
- OpenEvolve Architecture Documentation
- Knowledge Engine API Documentation
- Integration Best Practices Guide
- Microservices Communication Patterns
- Event-Driven Architecture Principles

### Migration Plan
1. **Assessment**: Identify all references to the duplicate knowledge engine
2. **Redirect**: Update imports to use the main knowledge engine
3. **Validate**: Test all functionality remains intact
4. **Deploy**: Roll out changes with rollback capability
5. **Cleanup**: Remove duplicate knowledge engine directory