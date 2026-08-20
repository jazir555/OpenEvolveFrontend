# OpenEvolve BubbleLabs Integration - System Architecture

## Overview

The OpenEvolve BubbleLabs Integration creates a comprehensive workflow management system that allows users to visualize, control, and execute OpenEvolve's sophisticated evolutionary computing workflows through BubbleLabs' intuitive visual interface. This integration bridges the gap between OpenEvolve's advanced AI-driven problem-solving capabilities and BubbleLabs' workflow automation platform.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   BubbleLabs UI Layer                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │    Dashboard    │  │   Flow Studio   │  │  Bubble Nodes   │  │  Parameter      │    │
│  │   Management    │  │   (React)      │  │   (ReactFlow)   │  │  Management     │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │ API Requests/Responses
                    ┌───────────────▼─────────────────────────────────────────────────────┤
                    │                     BubbleLabs Backend                              │
                    │                      (Bun + Hono)                                   │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
                    │  │ OpenEvolve API  │  │ Bubble Runtime  │  │ Authentication  │    │
                    │  │    Proxy        │  │    Engine       │  │    Service      │    │
                    │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
                    └───────────────┬─────────────────────────────────────────────────────┤
                                    │ HTTP Requests
                    ┌───────────────▼─────────────────────────────────────────────────────┤
                    │                    OpenEvolve Backend                               │
                    │                    (Python/FastAPI)                                 │
                    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
                    │  │ Evolution      │  │ Team Manager    │  │ Workflow        │    │
                    │  │   Engine       │  │    Service      │  │    Orchestration│    │
                    │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
                    └─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. BubbleLabs UI Layer

#### Flow Studio (React Application)
- **Technology**: React 19, TypeScript, Vite
- **Purpose**: Visual workflow builder and management interface
- **Key Features**:
  - Drag-and-drop workflow design
  - Real-time visualization of OpenEvolve workflows
  - Parameter configuration panels
  - Execution monitoring and results display
  - Version control for workflows

#### ReactFlow Visualization
- **Technology**: @xyflow/react
- **Purpose**: Graph-based visualization of workflow structures
- **Key Features**:
  - Node-based workflow representation
  - Connection visualization between workflow steps
  - Real-time execution status updates
  - Interactive node management

#### Parameter Management System
- **Technology**: React, Zustand (state management)
- **Purpose**: Configuration of OpenEvolve-specific parameters
- **Key Features**:
  - Dynamic parameter forms based on bubble schemas
  - Validation and error handling
  - Credential management integration
  - Preset and template support

### 2. BubbleLabs Backend Services

#### OpenEvolve API Proxy
- **Technology**: Bun, Hono
- **Location**: `apps/bubblelab-api/src/routes/openevolve.ts`
- **Purpose**: Optional mediation layer between BubbleLabs and the OpenEvolve backend
- **Key Features**:
  - Passive reverse proxy to `OPENEVOLVE_API_URL` (default `http://localhost:8000`)
  - Forwards documented routes (`GET /api/v1/health`, `POST /api/v1/evolve`,
    `GET /api/v1/runs/:id`, `POST /api/v1/workflows/orchestrate`) and a `/api/*`
    catch-all verbatim to the upstream
  - Returns the upstream status + body unchanged; fails per-request with a 502
    (not at startup) if the backend is unreachable
  - NOTE: The UI currently talks to the FastAPI service **directly** via
    `OPENEVOLVE_API_BASE_URL`; this proxy is available for mediation but is not
    required by the current client path contract.

#### Bubble Runtime Engine
- **Technology**: TypeScript, Custom runtime
- **Purpose**: Executes Bubble workflows and manages execution state
- **Key Features**:
  - Bubble execution orchestration
  - State management during execution
  - Real-time status updates
  - Error recovery and logging

#### Authentication Service
- **Technology**: Clerk, JWT
- **Purpose**: User authentication and authorization
- **Key Features**:
  - User session management
  - API key handling
  - Role-based access control
  - Secure credential storage

### 3. OpenEvolve Backend Services

> **Backend reality (path contract):** The primary OpenEvolve backend is
> `services/openevolve-api` (FastAPI). Its routers are mounted **already prefixed**
> (`/api/workflows`, `/api/teams`, `/api/gauntlets`, `/api/executions`,
> `/api/monitoring`, `/api/analytics`, ...). There is **no** `rewrite_api_prefix`
> middleware — clients send the `/api/...` paths as-is. The control-plane routes
> (`/health`, `/bubblelabs/...`) are served unprefixed.
>
> A **separate** library server (`core-projects/openevolve/openevolve/server_stdlib.py`)
> also exists and exposes `/api/v1/...` routes that wrap the real engine. The
> BubbleLab Hono proxy (`apps/bubblelab-api/src/routes/openevolve.ts`) can forward
> to either, but the UI today calls the FastAPI service directly. There is no
> non-existent proxy or `rewrite_api_prefix` middleware in the request path.

#### Evolution Engine
- **Technology**: Python, Custom evolutionary algorithms
- **Purpose**: Core evolutionary computing functionality
- **Key Features**:
  - Population-based optimization
  - Multi-objective optimization
  - Quality-diversity algorithms (MAP-Elites)
  - Adversarial evolution capabilities

#### Team Manager Service
- **Technology**: Python, REST API
- **Purpose**: Manages AI teams and their configurations
- **Key Features**:
  - Team definition and configuration
  - Member assignment and management
  - Performance tracking
  - Capability-based team assignment

#### Workflow Orchestration
- **Technology**: Python, Custom orchestrator
- **Purpose**: Coordinates complex multi-step workflows
- **Key Features**:
  - Sequential workflow execution
  - Parallel execution capabilities
  - State persistence and recovery
  - Dependency management

## Integration Points

### 1. Bubble Definition Integration
- **Location**: `packages/bubble-core/src/bubbles/openevolve/`
- **Purpose**: Defines OpenEvolve-specific bubble types
- **Implementation**:
  - OpenEvolveContentAnalyzerBubble
  - OpenEvolveDecomposerBubble
  - OpenEvolveSolverBubble
  - OpenEvolveVerifierBubble
  - OpenEvolveFullWorkflowBubble

### 2. API Gateway Integration
- **Location**: `apps/bubblelab-api/src/routes/openevolve.ts`
- **Purpose**: Optional Hono proxy that forwards OpenEvolve-specific requests to the
  FastAPI backend (`OPENEVOLVE_API_URL`, default `http://localhost:8000`)
- **Implementation**:
  - Passive proxy for `/api/*` (and `/api/v1/*`) routes — no path rewriting
  - Returns upstream status + body unchanged; 502 on unreachable backend
  - The UI may bypass this and call the FastAPI service directly (see backend
    reality note in §3)

### 3. UI Component Integration
- **Location**: `apps/bubble-studio/src/components/`
- **Purpose**: Visual representation of OpenEvolve workflows
- **Implementation**:
  - Custom node components for OpenEvolve bubbles
  - Parameter configuration UI
  - Execution status visualization

## Data Flow Architecture

### 1. Workflow Design Flow
```
User Interface → Bubble Schema Definition → Parameter Validation → Workflow Storage
```

### 2. Execution Flow
```
Workflow Execution Request → Bubble Runtime → OpenEvolve API Proxy → OpenEvolve Backend → Results
```

### 3. Status Update Flow
```
OpenEvolve Backend → Bubble Runtime → Real-time Updates → UI Components → User Interface
```

## Security Architecture

### 1. Authentication Flow
```
User Authentication → Session Token → API Key Forwarding → OpenEvolve Authentication
```

### 2. Data Protection
- API keys stored securely in BubbleLabs credential management
- End-to-end encryption for sensitive data
- Role-based access control for workflows
- Audit logging for security events

### 3. Network Security
- HTTPS for all API communications
- Request validation and sanitization
- Rate limiting for API protection
- Input validation for security

## Performance Considerations

### 1. Caching Strategy
- Bubble execution results caching
- API response caching
- Parameter schema caching
- UI component caching

### 2. Resource Management
- Connection pooling for database access
- Efficient memory usage during execution
- Asynchronous processing for long-running tasks
- Resource cleanup after execution

### 3. Scalability Features
- Horizontal scaling for API services
- Load balancing for high availability
- Database optimization for performance
- Caching layers for reduced load

## Monitoring and Observability

### 1. Application Metrics
- Bubble execution performance
- OpenEvolve API response times
- UI rendering performance
- Database query performance

### 2. Business Metrics
- Workflow execution success rates
- User engagement with workflows
- Feature usage analytics
- Error rates and types

### 3. Infrastructure Monitoring
- Server resource utilization
- Database performance metrics
- API endpoint health checks
- Real-time alerting for issues

## Deployment Architecture

### 1. Containerized Deployment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  BubbleLabs     │    │  OpenEvolve     │    │    Database     │
│   Frontend      │    │    Backend      │    │    Service      │
│   (React)       │    │   (FastAPI)     │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  BubbleLabs     │
                    │    Backend      │
                    │    (Bun)        │
                    └─────────────────┘
```

### 2. Environment Configuration
- Development: Local services with mock authentication
- Staging: Cloned production environment
- Production: Full-scale, highly available deployment

### 3. CI/CD Pipeline
- Automated testing for all components
- Container image building and deployment
- Environment-specific configuration
- Rollback capabilities for failed deployments

## Integration Patterns

### 1. API Proxy Pattern
- BubbleLabs API acts as a proxy to OpenEvolve services
- Handles authentication forwarding
- Provides centralized logging and monitoring
- Implements retry logic and circuit breakers

### 2. Event-Driven Architecture
- Real-time updates using WebSocket connections
- Event sourcing for execution state tracking
- Asynchronous processing for long-running tasks
- Reactive UI updates based on state changes

### 3. Adapter Pattern
- OpenEvolve bubbles adapt to BubbleLabs interface
- Parameter schema translation layer
- Response format standardization
- Error handling normalization

## Error Handling Strategy

### 1. Circuit Breaker Pattern
- Prevents cascading failures
- Implements fallback mechanisms
- Monitors service health
- Automatically recovers from failures

### 2. Retry Logic
- Exponential backoff for API calls
- Configurable retry attempts
- Context-aware retry decisions
- Graceful degradation when needed

### 3. Fallback Mechanisms
- Default parameter values
- Offline operation capabilities
- Graceful degradation of features
- User-friendly error messages

## Future Scalability Considerations

### 1. Horizontal Scaling
- Microservice architecture for independent scaling
- Database sharding for large-scale data
- CDN for static assets and caching
- Load balancer for traffic distribution

### 2. Performance Optimization
- Database query optimization
- Caching layer implementation
- Asynchronous processing for background tasks
- Resource pooling for efficient utilization

### 3. Feature Expansion
- Support for additional OpenEvolve capabilities
- Integration with other AI platforms
- Advanced analytics and insights
- Multi-tenant architecture support

This architecture provides a robust, scalable foundation for the OpenEvolve BubbleLabs integration, enabling complete control and visualization of OpenEvolve workflows through the BubbleLabs interface while maintaining performance, security, and reliability.