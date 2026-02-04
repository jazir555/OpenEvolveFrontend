# OpenEvolve Federation - Comprehensive Documentation

**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Status**: Production Ready
**Maintainer**: OpenEvolve Federation Team

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Component Reference](#3-component-reference)
4. [Integration Guides](#4-integration-guides)
5. [API Reference](#5-api-reference)
6. [Configuration](#6-configuration)
7. [Development](#7-development)
8. [Deployment](#8-deployment)
9. [Monitoring & Observability](#9-monitoring--observability)
10. [Troubleshooting](#10-troubleshooting)
11. [Best Practices](#11-best-practices)
12. [Federation Constitution](#12-federation-constitution)
13. [Glossary](#13-glossary)
14. [Appendices](#14-appendices)

---

## 1. Introduction

### 1.1 Purpose

The OpenEvolve Federation is a **mega-structure** that integrates 30+ open-source systems into a cohesive, intelligent development platform. This documentation provides comprehensive guidance for:

- Developers integrating new systems
- Operators managing the federation
- Architects understanding system design
- Contributors extending functionality

### 1.2 Scope

This documentation covers:
- **All 12 integrated components** with full technical details
- **Complete API reference** for all adapters and services
- **Deployment procedures** for production environments
- **Monitoring and observability** best practices
- **Troubleshooting guides** for common issues

### 1.3 Audience

| Role | Sections | Prerequisites |
|------|----------|----------------|
| **Developers** | API Reference, Integration Guides | TypeScript, Node.js |
| **Operators** | Deployment, Monitoring, Troubleshooting | Docker, Linux |
| **Architects** | Architecture Overview, Best Practices | Systems Design |
| **Contributors** | Development, Federation Constitution | Git, Testing |

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OpenEvolve Federation                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Core Projects (Immutable)                      │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                │ │
│  │  │   ICR   │ │OpenEvolve│ │BubbleLab│ │ RAGBits │  ... (30+) │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    ↓ AIR GAP                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      Glue Layer (Integration)                    │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              Adapters (Sidecars)                            │ │ │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐             │ │ │
│  │  │  │   ICR  │ │  Evol  │ │  Sync  │ │  Verif │  ... (12) │ │ │
│  │  │  └────────┘ └────────┘ └────────┘ └────────┘             │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                   Orchestration                            │ │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │ │ │
│  │  │  │ Event Bus  │ │ Workflows  │ │  Circuit   │            │ │ │
│  │  │  └────────────┘ └────────────┘ └────────────┘            │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Shared Libraries                        │ │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │ │ │
│  │  │  │  Logger    │ │   Retry    │ │  Circuit    │            │ │ │
│  │  │  └────────────┘ └────────────┘ └────────────┘            │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Knowledge Systems                             │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                            │ │
│  │  │ RAGBits │ │ Graphiti│ │Vector DB│  ...                        │ │
│  │  └─────────┘ └─────────┘ └─────────┘                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture

```
User Request
    ↓
┌───────────────────────────────────────────────────────┐
│  Request Router (Analyzes & Routes to Systems)       │
└────────────┬──────────────────────────────────────────┘
             │
    ┌────────┴────────┬───────────────┬──────────────┐
    │                 │               │              │
    ▼                 ▼               ▼              ▼
┌─────────┐      ┌─────────┐     ┌─────────┐   ┌─────────┐
│  ICR    │      │RAGBits  │     │ Graphiti│   │Vector DB│
│ Adapter │      │ Adapter  │     │ Adapter │   │ Adapter │
└────┬────┘      └────┬────┘     └────┬────┘   └────┬────┘
     │                │                │              │
     └────────┬───────┴────────┬────────┴──────────┘
              │               │
              ▼               ▼
    ┌─────────────────────────────────┐
    │  Result Fusion & Aggregation    │
    │  - Score Normalization         │
    │  - Conflict Resolution          │
    │  - Ranking & Filtering          │
    └────────────┬────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Unified       │
         │  Response     │
         └───────────────┘
```

### 2.3 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19, Next.js 16, Vite 7 | UI frameworks |
| **Backend** | Node.js 20+, TypeScript 5.8 | Server runtime |
| **Integration** | Zod, Axios, UUID-fast | Data validation & HTTP |
| **Knowledge** | Qdrant, Neo4j, PostgreSQL | Knowledge storage |
| **Verification** | Z3, LeanAide | Formal methods |
| **Observability** | Prom-client, OpenTelemetry | Metrics & tracing |

---

## 3. Component Reference

### 3.1 ICR Adapter

**Location**: `glue/adapters/icr-adapter/`

**Purpose**: Integration with Iterative Contextual Refinements system

**Description**: The ICR adapter provides access to 7 powerful operational modes for iterative refinement and content generation.

#### Supported Modes

| Mode | Purpose | Use Case |
|------|---------|----------|
| **Refine** | Traditional iterative refinement with feature suggestion | Code optimization, content improvement |
| **React** | React application development with parallel workers | Frontend development |
| **Deepthink** | Strategic problem decomposition with hypothesis exploration | Complex problem-solving |
| **Adaptive Deepthink** | Full deepthink access for agents | Research and analysis |
| **Agentic** | Tool-based content manipulation with LangChain | Automation workflows |
| **Contextual** | Multi-agent collaboration with persistent memory | Long-running sessions |
| **Generative UI** | Interactive UI generation with user feedback | Prototyping |

#### API Reference

```typescript
interface ICRAdapter {
  // Refine Mode
  createRefinementRequest(
    prompt: string,
    options: RefineOptions,
    correlationId?: string
  ): Promise<RefineModeResponse>

  // React Mode
  createReactRequest(
    prompt: string,
    options: ReactOptions,
    correlationId?: string
  ): Promise<ReactModeResponse>

  // Deepthink Mode
  createDeepthinkRequest(
    prompt: string,
    options: DeepthinkOptions,
    correlationId?: string
  ): Promise<DeepthinkModeResponse>

  // Adaptive Deepthink Mode
  createAdaptiveDeepthinkRequest(
    prompt: string,
    options: AdaptiveDeepthinkOptions,
    correlationId?: string
  ): Promise<AdaptiveDeepthinkModeResponse>

  // Agentic Mode
  createAgenticRequest(
    prompt: string,
    options: AgenticOptions,
    correlationId?: string
  ): Promise<AgenticModeResponse>

  // Contextual Mode with Memory
  createContextualRequestWithMemory(
    prompt: string,
    options: ContextualOptions,
    correlationId?: string
  ): Promise<ContextualModeResponse>

  // Generative UI Mode
  createGenerativeUIRequest(
    prompt: string,
    options: GenerativeUIOptions,
    correlationId?: string
  ): Promise<GenerativeUIModeResponse>

  // Health Check
  healthCheck(correlationId?: string): Promise<HealthCheckResult>
}
```

#### Usage Example

```typescript
import { icrAdapter } from '@openevolve/icr-adapter';

// Contextual refinement with memory
const result = await icrAdapter.createContextualRequestWithMemory(
  "Refine this authentication system for better security",
  {
    context_window: 10,
    enable_learning: true,
    temperature: 0.7,
    max_iterations: 5
  },
  "ctx-123"
);

console.log('Refined Content:', result.result.content);
console.log('Iterations:', result.result.iterations);
console.log('Historical Patterns:', result.enriched_context.historical_knowledge);
```

#### Configuration

Required environment variables:
```bash
OPENEVOLVE_ICR_API_URL=http://localhost:8080  # REQUIRED
TIMEOUT_MS=5000                             # REQUIRED
```

Optional:
```bash
CIRCUIT_BREAKER_THRESHOLD=5              # Default: 5
CIRCUIT_BREAKER_TIMEOUT_MS=60000         # Default: 60000
RETRY_MAX_ATTEMPTS=3                    # Default: 3
RETRY_BASE_DELAY_MS=1000                 # Default: 1000
```

---

### 3.2 Evolved Code Capture System

**Location**: `glue/lib/evolved-code-capture/`

**Purpose**: Capture and store OpenEvolve's evolved code for semantic search and reuse

**Description**: This system captures the best solutions from OpenEvolve evolutionary runs and stores them in both Vector DB (for semantic search) and Graphiti (for lineage tracking).

#### Architecture

```
OpenEvolve Evolution Complete
        ↓
Extract Best Solution
        ↓
[Generate Embedding] + [Create Graph Episode]
        ↓                    ↓
   Vector DB          Graphiti
   (Semantic Search)  (Lineage Tracking)
        ↓                    ↓
        Unified Knowledge Storage
```

#### API Reference

```typescript
class EvolvedCodeCapturer {
  // Capture evolution result
  async captureEvolution(
    problem: Problem,
    solution: EvolvedCode,
    metrics: EvolutionMetrics,
    correlationId?: string
  ): Promise<CaptureResult>

  // Search similar problems
  async searchSimilarProblems(
    problem: Problem,
    maxResults: number,
    correlationId?: string
  ): Promise<SimilarSolution[]>

  // Get evolution lineage
  async getEvolutionLineage(
    codeId: string,
    correlationId?: string
  ): Promise<EvolutionLineage>

  // Get system metrics
  async getMetrics(correlationId?: string): Promise<CaptureMetrics>
}
```

#### Canonical Schemas

```typescript
interface Problem {
  id: string
  description: string
  type: ProblemType
  constraints: string[]
  test_cases?: TestCase[]
}

interface EvolvedCode {
  id: string
  code: string
  language: string
  fitness: number
  generation: number
  timestamp_utc: string
}

interface EvolutionMetrics {
  iterations: number
  population_size: number
  duration_ms: number
  improvement: number
  final_fitness: number
}
```

#### Usage Example

```typescript
import { EvolvedCodeCapturer } from '@openevolve/evolved-code-capture';

const capturer = new EvolvedCodeCapturer({
  vector_storage: {
    vectordb_adapter_url: 'http://vectordb-adapter:8000',
    collection_name: 'evolved_code',
    embedding_api_key: process.env.OPENAI_API_KEY,
  },
  graph_storage: {
    graphiti_adapter_url: 'http://graphiti-adapter:8000',
  },
});

// Capture evolution
await capturer.captureEvolution(
  {
    id: 'prob-1',
    description: 'Optimize sorting algorithm for large datasets',
    type: 'optimization',
    constraints: ['O(n log n)', 'stable']
  },
  {
    id: 'code-1',
    code: 'function mergeSort(arr) { ... }',
    language: 'javascript',
    fitness: 0.95,
    generation: 50
  },
  {
    iterations: 50,
    population_size: 100,
    duration_ms: 5000,
    improvement: 0.40,
    final_fitness: 0.95
  }
);

// Search similar problems
const similar = await capturer.searchSimilarProblems(
  {
    description: 'Optimize search algorithm',
    type: 'optimization'
  },
  10
);

console.log('Similar Solutions:', similar);
```

---

### 3.3 Unified Knowledge Query Interface

**Location**: `glue/lib/unified-knowledge-query/`

**Purpose**: Query multiple knowledge systems simultaneously with intelligent result fusion

**Description**: Provides a single interface to query RAGBits (documents), Graphiti (knowledge graphs), and Vector DB (semantic search) with automatic result merging and conflict resolution.

#### Architecture

```
User Query
    ↓
Parse & Validate
    ↓
┌─────────────────────────────────────┐
│     Query Router                    │
│  - Analyze query type               │
│  - Select optimal systems            │
│  - Plan execution strategy           │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┬───────────────┐
    │                 │               │
    ▼                 ▼               ▼
┌─────────┐      ┌─────────┐     ┌─────────┐
│ RAGBits │      │ Graphiti│     │Vector DB│
└────┬────┘      └────┬────┘     └────┬────┘
     │                │                │
     └────────┬───────┴────────┬────────┘
              │               │
              ▼               ▼
    ┌─────────────────────────────────┐
    │      Result Fusion              │
    │  - Score normalization          │
    │  - Conflict detection          │
    │  - Ranking & filtering         │
    └────────────┬────────────────────┘
                 │
                 ▼
         Unified Result
```

#### API Reference

```typescript
class UnifiedKnowledgeQueryEngine {
  constructor(config: UnifiedKnowledgeQueryConfig)

  // Main query method
  async query(
    query: string,
    options: QueryOptions
  ): Promise<UnifiedQueryResult>

  // Health check all systems
  async healthCheck(): Promise<SystemHealth[]>

  // Get engine metrics
  async getMetrics(): Promise<EngineMetrics>
}

interface QueryOptions {
  domains?: KnowledgeDomain[]         // Which systems to query
  maxResults?: number                  // Maximum results
  minConfidence?: number               // Minimum confidence threshold
  temporalFilter?: TemporalFilter      // Time range for Graphiti
  knowledgeTypes?: KnowledgeType[]     // Type of knowledge
}
```

#### Query Types

| Type | Description | Use Case |
|------|-------------|----------|
| **semantic** | Vector similarity search | Find similar documents/code |
| **temporal** | Time-based graph queries | Historical knowledge |
| **graph** | Knowledge graph traversal | Entity relationships |
| **hybrid** | Combine all sources | Comprehensive search |
| **fallback** | Graceful degradation | System resilience |

#### Usage Example

```typescript
import { UnifiedKnowledgeQueryEngine } from '@openevolve/unified-knowledge-query';

const engine = new UnifiedKnowledgeQueryEngine({
  ragbitsUrl: process.env.RAGBITS_URL,
  graphitiUrl: process.env.GRAPHITI_URL,
  vectordbUrl: process.env.VECTORDB_URL,
  timeout: 10000,
  retry: { maxAttempts: 3, baseDelay: 1000 }
});

// Hybrid query across all systems
const results = await engine.query(
  "Latest developments in formal verification",
  {
    domains: ['ragbits', 'graphiti', 'vectordb'],
    maxResults: 50,
    minConfidence: 0.7,
    temporalFilter: {
      startDate: '2025-01-01T00:00:00Z',
      endDate: '2026-02-03T23:59:59Z'
    }
  }
);

console.log(`Found ${results.results.length} results`);
console.log(`Confidence: ${results.confidence}`);
console.log(`Sources: ${Array.from(results.sources).join(', ')}`);
console.log(`Execution time: ${results.executionTimeMs}ms`);

// Access individual results
results.results.forEach((result, index) => {
  console.log(`[${index + 1}] ${result.title}`);
  console.log(`   Source: ${result.source}`);
  console.log(`   Score: ${result.score}`);
  console.log(`   Content: ${result.content}`);
});
```

---

### 3.4 Unified Verification Orchestrator

**Location**: `glue/orchestration/unified-verification/`

**Purpose**: Cross-validation between Z3 SMT solver and LeanAide theorem prover

**Description**: Provides intelligent strategy selection, parallel/sequential execution, cross-validation, and confidence aggregation for formal verification tasks.

#### Architecture

```
Verification Request
        ↓
┌───────────────────────────────────┐
│   Strategy Selector               │
│  - Analyze problem type            │
│  - Select verification strategy     │
│  - Estimate success rates           │
└────────────┬──────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐      ┌─────────┐
│   Z3    │      │LeanAide │
│ Parallel│      │Parallel  │
└────┬────┘      └────┬────┘
     │                │
     └────────┬───────┘
              │
              ▼
┌───────────────────────────────────┐
│     Cross Validator               │
│  - Compare results                │
│  - Detect disagreements            │
│  - Calculate agreement             │
└────────────┬──────────────────────┘
             │
             ▼
┌───────────────────────────────────┐
│   Confidence Aggregator           │
│  - Normalize scores                │
│  - Calculate weights               │
│  - Combine evidence                │
└────────────┬──────────────────────┘
             │
             ▼
      Verification Result
```

#### API Reference

```typescript
class UnifiedVerificationOrchestrator {
  constructor(
    z3Url: string,
    leanaideUrl: string,
    options?: OrchestratorOptions
  )

  // Single system verification
  async verify(
    problem: Problem,
    constraints: Constraints,
    options: VerificationOptions
  ): Promise<VerificationResult>

  // Cross-validation with both systems
  async verifyWithCrossValidation(
    problem: Problem,
    options: VerificationOptions
  ): Promise<CrossValidationResult>

  // Batch verification
  async batchVerify(
    problems: Problem[],
    options: VerificationOptions
  ): Promise<VerificationResult[]>
}
```

#### Verification Strategies

| Strategy | Description | Use When |
|----------|-------------|----------|
| **z3_only** | Z3 SMT solver only | SMT constraints, model checking |
| **leanaide_only** | LeanAide theorem prover only | Mathematical proofs |
| **parallel** | Both systems simultaneously | Maximum confidence, time-critical |
| **sequential** | Z3 first, then LeanAide with early termination | Resource optimization |
| **hybrid** | Adaptive approach based on problem | Unknown problem types |

#### Usage Example

```typescript
import { UnifiedVerificationOrchestrator } from '@openevolve/unified-verification';

const orchestrator = new UnifiedVerificationOrchestrator(
  process.env.Z3_API_URL!,
  process.env.LEANAIDE_API_URL!,
  {
    timeout: 30000,
    retry: { maxAttempts: 3, baseDelay: 2000 }
  }
);

// Cross-validate theorem
const result = await orchestrator.verifyWithCrossValidation(
  {
    id: 'thm-1',
    type: 'THEOREM_PROVING',
    statement: 'theorem add_zero (n : Nat) : n + 0 = n',
    proof: 'by simp',
    constraints: ['n ∈ Nat']
  },
  {
    confidenceRequired: 0.95,
    strategy: 'parallel',
    storeResults: true
  }
);

console.log('Verified:', result.verified);
console.log('Agreement:', result.agreement);
console.log('Confidence:', result.confidence);
console.log('Evidence:', result.evidence);

// Access system results
result.systemResults.forEach(sys => {
  console.log(`[${sys.system}]`);
  console.log(`  Output: ${sys.output}`);
  console.log(`  Confidence: ${sys.confidence}`);
});
```

---

## 4. Integration Guides

### 4.1 Integrating a New Core Project

This guide demonstrates how to integrate a new core project into the OpenEvolve Federation.

#### Step 1: Discovery Phase

Run probe scripts to verify the project's API:

```bash
# Create probe directory
mkdir -p glue/adapters/newproject-adapter/probes

# Create check_api.sh probe
cat > glue/adapters/newproject-adapter/probes/check_api.sh << 'EOF'
#!/bin/bash
set -e

echo "=== NewProject API Probe ==="

# Required env vars
: ${NEWPROJECT_API_URL:?"Error: NEWPROJECT_API_URL not set"}

# Health check
echo "Testing health endpoint..."
response=$(curl -s -o /dev/null -w "%{http_code}" \
  "${NEWPROJECT_API_URL}/health" \
  --max-time ${TIMEOUT_MS:-5000})

if [ "$response" -eq 200 ]; then
  echo "✓ Health check passed"
else
  echo "✗ Health check failed: HTTP $response"
  exit 1
fi

# Test main API endpoint
echo "Testing main API..."
response=$(curl -s -o /dev/null -w "%{http_code}" \
  "${NEWPROJECT_API_URL}/api/v1/query" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' \
  --max-time ${TIMEOUT_MS:-5000})

if [ "$response" -eq 200 ] || [ "$response" -eq 201 ]; then
  echo "✓ API endpoint accessible"
else
  echo "✗ API endpoint failed: HTTP $response"
  exit 1
fi

echo "=== All Probes Passed ==="
EOF

chmod +x glue/adapters/newproject-adapter/probes/check_api.sh
```

#### Step 2: Define Canonical Schema

```typescript
// glue/adapters/newproject-adapter/src/canonical.ts
import { z } from 'zod';

// Canonical request schema
export const NewProjectRequestSchema = z.object({
  query: z.string(),
  parameters: z.record(z.any()).optional(),
  options: z.object({
    temperature: z.number().min(0).max(2).optional(),
    max_tokens: z.number().positive().optional(),
  }).optional(),
  metadata: z.object({
    correlation_id: z.string().uuid().optional(),
    timestamp_utc: z.string().datetime().optional(),
  }).optional(),
});

export type NewProjectRequest = z.infer<typeof NewProjectRequestSchema>;

// Canonical response schema
export const NewProjectResponseSchema = z.object({
  result: z.string(),
  metadata: z.object({
    model: z.string(),
    tokens_used: z.number().optional(),
    timestamp_utc: z.string().datetime(),
  }),
});

export type NewProjectResponse = z.infer<typeof NewProjectResponseSchema>;
```

#### Step 3: Implement Adapter

```typescript
// glue/adapters/newproject-adapter/src/adapter.ts
import {
  NewProjectRequest,
  NewProjectResponse,
} from './canonical';
import { CircuitBreaker } from '@openevolve/lib';
import { StructuredLogger } from '@openevolve/lib';
import axios from 'axios';

export class NewProjectAdapter {
  private client: axios.AxiosInstance;
  private circuitBreaker: CircuitBreaker;
  private logger: StructuredLogger;
  private apiUrl: string;
  private timeout: number;

  constructor(config: NewProjectAdapterConfig) {
    this.apiUrl = config.api_url;
    this.timeout = config.timeout_ms;

    // Validate required config
    if (!this.apiUrl) {
      throw new Error('NEWPROJECT_API_URL is required');
    }

    // Initialize HTTP client
    this.client = axios.create({
      baseURL: this.apiUrl,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      failureThreshold: config.circuit_breaker?.failure_threshold ?? 5,
      successThreshold: config.circuit_breaker?.success_threshold ?? 2,
      timeout: config.circuit_breaker?.timeout_ms ?? 60000,
    });

    // Initialize logger
    this.logger = new StructuredLogger({
      service: 'newproject-adapter',
    });
  }

  async executeQuery(
    request: NewProjectRequest,
    correlationId?: string
  ): Promise<NewProjectResponse> {
    const cid = correlationId ?? this.generateCorrelationId();

    this.logger.info({
      msg: 'Executing NewProject query',
      correlation_id: cid,
      query: request.query,
    });

    // Check circuit breaker
    if (this.circuitBreaker.state === 'open') {
      throw new Error('Circuit breaker is open');
    }

    try {
      // Execute query with retry
      const response = await this.retryRequest(async () => {
        const { data } = await this.client.post('/api/v1/query', {
          query: request.query,
          parameters: request.parameters,
          options: request.options,
        });

        // Validate response
        return NewProjectResponseSchema.parse(data);
      });

      // Success: record in circuit breaker
      this.circuitBreaker.recordSuccess();

      this.logger.info({
        msg: 'NewProject query successful',
        correlation_id: cid,
        result_length: response.result.length,
      });

      return response;

    } catch (error) {
      // Failure: record in circuit breaker
      this.circuitBreaker.recordFailure();

      this.logger.error({
        msg: 'NewProject query failed',
        correlation_id: cid,
        error: error.message,
      });

      throw error;
    }
  }

  async healthCheck(correlationId?: string): Promise<HealthCheckResult> {
    try {
      const { status } = await this.client.get('/health');

      return {
        status: status === 200 ? 'healthy' : 'degraded',
        timestamp_utc: new Date().toISOString(),
        correlation_id: correlationId,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        timestamp_utc: new Date().toISOString(),
        correlation_id: correlationId,
        error: error.message,
      };
    }
  }

  private async retryRequest<T>(
    fn: () => Promise<T>
  ): Promise<T> {
    const maxRetries = 3;
    const baseDelay = 1000;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxRetries - 1) throw error;

        const isRetryable = error.status >= 500 || error.code === 'ECONNREFUSED';
        if (!isRetryable) throw error;

        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 500;
        await this.sleep(delay);
      }
    }

    throw new Error('Max retries exceeded');
  }

  private generateCorrelationId(): string {
    return `corr-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

#### Step 4: Create Contract Tests

```typescript
// glue/adapters/newproject-adapter/tests/contract.test.ts
import { describe, it, expect, beforeAll } from '@jest/globals';
import { NewProjectAdapter } from '../src/adapter';
import { NewProjectRequestSchema } from '../src/canonical';

describe('NewProject Adapter Contract Tests', () => {
  let adapter: NewProjectAdapter;

  beforeAll(() => {
    adapter = new NewProjectAdapter({
      api_url: process.env.NEWPROJECT_API_URL!,
      timeout_ms: 5000,
    });
  });

  describe('Health Check', () => {
    it('should return healthy status', async () => {
      const health = await adapter.healthCheck();

      expect(health.status).toBe('healthy');
      expect(health.timestamp_utc).toBeDefined();
    });
  });

  describe('Query Execution', () => {
    it('should execute valid query', async () => {
      const request = NewProjectRequestSchema.parse({
        query: 'test query',
        metadata: {
          correlation_id: 'test-123',
          timestamp_utc: new Date().toISOString(),
        },
      });

      const response = await adapter.executeQuery(request, 'test-123');

      expect(response.result).toBeDefined();
      expect(response.metadata).toBeDefined();
      expect(response.metadata.timestamp_utc).toBeDefined();
    });

    it('should reject invalid query', async () => {
      await expect(async () => {
        await adapter.executeQuery({ query: 123 } as any);
      }).rejects.toThrow();
    });
  });

  describe('Circuit Breaker', () => {
    it('should open circuit after threshold failures', async () => {
      // Trigger failures
      for (let i = 0; i < 10; i++) {
        try {
          await adapter.executeQuery({ query: 'test' });
        } catch (error) {
          // Expected failures
        }
      }

      const state = adapter.getCircuitBreakerState();
      expect(state.state).toBe('open');
    });
  });
});
```

#### Step 5: Create Package Configuration

```json
{
  "name": "@openevolve/newproject-adapter",
  "version": "1.0.0",
  "description": "NewProject adapter for OpenEvolve Federation",
  "main": "src/index.ts",
  "types": "src/index.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "test:contract": "jest tests/contract.test.ts",
    "probe": "bash probes/check_api.sh"
  },
  "dependencies": {
    "axios": "^1.6.5",
    "zod": "^3.22.4"
  },
  "devDependencies": {
    "@types/node": "^20.10.0",
    "typescript": "^5.3.0",
    "jest": "^29.7.0",
    "@jest/globals": "^29.7.0",
    "ts-jest": "^29.1.0"
  },
  "peerDependencies": {
    "@openevolve/lib": "^1.0.0"
  }
}
```

### 4.2 Creating Workflow Compositions

This guide shows how to create workflow compositions in BubbleLab.

#### Basic Workflow Composition

```typescript
// core-projects/BubbleLab/apps/bubble-studio/src/bubbles/workflow-compositions.ts

import { EvolutionTriggerBubble } from './EvolutionTriggerBubble';
import { EvolutionValidationBubble } from './EvolutionValidationBubble';
import { EvolutionApplicationBubble } from './EvolutionApplicationBubble';
import { KnowledgeRetrievalWorkflow } from './knowledge-retrieval.workflow';
import { KnowledgeAugmentedWorkflow } from './knowledge-augmented-workflow';

// Evolution pipeline with knowledge
export const EvolutionKnowledgePipeline = {
  name: 'evolution-knowledge-pipeline',
  description: 'Evolve solution with knowledge augmentation',

  steps: [
    {
      name: 'knowledge-retrieval',
      bubble: KnowledgeRetrievalWorkflow,
      config: {
        query: 'optimization best practices',
        sources: ['ragbits', 'graphiti', 'vectordb'],
        maxResults: 10
      }
    },
    {
      name: 'evolution-trigger',
      bubble: EvolutionTriggerBubble,
      config: {
        iterations: 100,
        populationSize: 50,
        useKnowledge: true
      }
    },
    {
      name: 'evolution-validation',
      bubble: EvolutionValidationBubble,
      config: {
        validation: 'full',
        enable_z3: true,
        enable_leanaide: true,
        run_tests: true
      }
    },
    {
      name: 'evolution-application',
      bubble: EvolutionApplicationBubble,
      config: {
        targetSystem: 'file',
        targetPath: './evolved-solution.ts'
      }
    },
    {
      name: 'knowledge-capture',
      bubble: KnowledgeCaptureWorkflow,
      config: {
        storeLearnings: true
      }
    }
  ]
};
```

#### Workflow with Conditional Logic

```typescript
export const AdaptiveEvolutionWorkflow = {
  name: 'adaptive-evolution',
  description: 'Evolution with conditional logic based on results',

  steps: [
    {
      name: 'initial-analysis',
      bubble: EvolutionTriggerBubble,
      config: {
        iterations: 50,
        quickTest: true
      }
    },
    {
      name: 'check-quality',
      type: 'conditional',
      condition: 'previous.fitness > 0.8',
      true: [
        {
          name: 'full-validation',
          bubble: EvolutionValidationBubble,
          config: { validation: 'full' }
        },
        {
          name: 'apply',
          bubble: EvolutionApplicationBubble
        }
      ],
      false: [
        {
          name: 'continue-evolution',
          bubble: EvolutionTriggerBubble,
          config: {
            iterations: 100,
            useKnowledge: true
          }
        }
      ]
    }
  ]
};
```

### 4.3 Implementing Memory-Enabled Sessions

This guide shows how to use ICR's persistent memory.

#### Session with Memory

```typescript
import { ICRAdapter } from '@openevolve/icr-adapter';

const icr = new ICRAdapter({
  api_url: process.env.ICR_API_URL!,
  timeout_ms: 10000,
  memoryAgentConfig: {
    graphiti: {
      graphiti_api_url: process.env.GRAPHITI_API_URL!,
      neo4j_uri: process.env.NEO4J_URI!,
      neo4j_user: process.env.NEO4J_USER!,
      neo4j_password: process.env.NEO4J_PASSWORD!,
    },
    enable_historical_retrieval: true,
    enable_pattern_learning: true,
    default_context_window: 10
  }
});

// Start a memory-enabled session
const sessionId = `session-${Date.now()}`;

// First request: Retrieve historical knowledge
const request1 = await icr.createContextualRequestWithMemory(
  "Refine this sorting algorithm",
  {
    sessionId,
    context_window: 5,
    enable_learning: true
  }
);

console.log('Historical patterns found:', request1.enriched_context.historical_knowledge.length);
console.log('Suggested approaches:', request1.enriched_context.suggested_approaches);

// Second request: Builds on previous learning
const request2 = await icr.createContextualRequestWithMemory(
  "Optimize the same algorithm for memory usage",
  {
    sessionId,
    context_window: 10,
    enable_learning: true
  }
);

// Session learns across multiple requests
const session = await icr.getSessionMemory(sessionId);
console.log('Session patterns:', session.patterns);
console.log('Improvement over session:', session.improvement_trajectory);
```

---

## 5. API Reference

### 5.1 Adapter Interfaces

All adapters implement the common interface:

```typescript
interface BaseAdapter {
  // Health check
  healthCheck(correlationId?: string): Promise<HealthCheckResult>

  // Circuit breaker state
  getCircuitBreakerState(): CircuitBreakerState
  resetCircuitBreaker(): void

  // Metrics
  getMetrics(): AdapterMetrics
}

interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp_utc: string
  correlation_id?: string
  error?: string
}

interface CircuitBreakerState {
  state: 'closed' | 'open' | 'half_open'
  failureCount: number
  successCount: number
  lastFailureTime?: string
  lastSuccessTime?: string
}

interface AdapterMetrics {
  total_requests: number
  successful_requests: number
  failed_requests: number
  success_rate: number
  average_request_duration_ms: number
  circuit_breaker_state: string
}
```

### 5.2 Common Canonical Schemas

All adapters use Zod for runtime validation:

```typescript
// Base metadata
export const MetadataSchema = z.object({
  correlation_id: z.string().uuid().optional(),
  timestamp_utc: z.string().datetime(),
  source_service: z.string(),
  target_service: z.string(),
});

// Error response
export const ErrorResponseSchema = z.object({
  error: z.string(),
  message: z.string().optional(),
  code: z.string().optional(),
  timestamp_utc: z.string().datetime(),
  correlation_id: z.string().uuid().optional(),
});
```

### 5.3 Query Options Interface

```typescript
interface QueryOptions {
  // System selection
  domains?: Array<'ragbits' | 'graphiti' | 'vectordb' | 'all'>

  // Result limits
  maxResults?: number
  minConfidence?: number

  // Filtering
  knowledgeTypes?: Array<'document' | 'entity' | 'proof' | 'code'>

  // Temporal filtering
  temporalFilter?: {
    startDate: string
    endDate: string
    pointInTime?: string
  }

  // Execution options
  timeout?: number
  retry?: boolean
  parallel?: boolean
}
```

---

## 6. Configuration

### 6.1 Environment Variables Reference

#### ICR Adapter

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENEVOLVE_ICR_API_URL` | Yes | - | ICR API endpoint |
| `TIMEOUT_MS` | Yes | - | Request timeout in ms |
| `CIRCUIT_BREAKER_THRESHOLD` | No | 5 | Failures before trip |
| `CIRCUIT_BREAKER_TIMEOUT_MS` | No | 60000 | Open state duration |
| `RETRY_MAX_ATTEMPTS` | No | 3 | Max retry attempts |
| `RETRY_BASE_DELAY_MS` | No | 1000 | Base retry delay |

#### Evolved Code Capture

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VECTORDB_ADAPTER_URL` | Yes | - | Vector DB adapter URL |
| `GRAPHITI_ADAPTER_URL` | Yes | - | Graphiti adapter URL |
| `EVOLVED_CODE_COLLECTION` | No | evolved_code | Collection name |
| `OPENAI_API_KEY` | No* | - | Embedding API key |
| `EMBEDDING_DIMENSION` | No | 1536 | Embedding dimension |

*Required for vector embeddings; hash-based fallback available.

#### Unified Knowledge Query

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RAGBITS_API_URL` | Yes | - | RAGBits endpoint |
| `GRAPHITI_API_URL` | Yes | - | Graphiti endpoint |
| `VECTORDB_URL` | Yes | - | Vector DB endpoint |
| `QUERY_TIMEOUT_MS` | No | 10000 | Query timeout |
| `MIN_CONFIDENCE` | No | 0.6 | Minimum confidence |

#### Unified Verification

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `Z3_API_URL` | Yes | - | Z3 SMT solver endpoint |
| `LEANAIDE_API_URL` | Yes | - | LeanAide endpoint |
| `VERIFICATION_TIMEOUT_MS` | No | 30000 | Verification timeout |
| `ENABLE_PARALLEL` | No | true | Enable parallel execution |

### 6.2 Configuration File Example

```bash
# .env.example
# Copy this file to .env and fill in your values

# ICR Configuration
OPENEVOLVE_ICR_API_URL=http://localhost:8080
TIMEOUT_MS=5000

# OpenEvolve Configuration
OPENEVOLVE_API_URL=http://localhost:8002

# Knowledge Systems
RAGBITS_API_URL=http://localhost:8082
GRAPHITI_API_URL=http://localhost:8084
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USER=neo4j
GRAPHITI_NEO4J_PASSWORD=password

VECTORDB_URL=http://localhost:8083

# Vector Database
VECTORDB_TYPE=qdrant
VECTORDB_COLLECTION=knowledge

# Formal Verification
Z3_API_URL=http://localhost:8080
LEANAIDE_API_URL=http://localhost:8081

# BubbleLab
BUBBLELAB_API_URL=http://localhost:3000

# Optional: OpenAI API for embeddings
OPENAI_API_KEY=sk-your-key-here

# Circuit Breaker Defaults
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_MS=60000

# Retry Defaults
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY_MS=1000
RETRY_MAX_DELAY_MS=10000
```

---

## 7. Development

### 7.1 Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/frontend.git
cd frontend

# Install dependencies
npm install

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Run tests
npm test

# Build all packages
npm run build

# Run development server
npm run dev
```

### 7.2 Project Structure

```
frontend/
├── core-projects/          # 30+ immutable projects
│   ├── Iterative-Contextual-Refinements/
│   ├── OpenEvolve/
│   ├── BubbleLab/
│   └── ...
├── glue/                    # Integration layer
│   ├── adapters/           # Adapters (sidecars)
│   │   ├── icr-adapter/
│   │   ├── openevolve-adapter/
│   │   ├── bubblelab-adapter/
│   │   └── ...
│   ├── orchestration/       # Event bus, workflows
│   ├── schemas/             # Canonical data models
│   └── lib/                 # Shared utilities
├── infra/                   # Infrastructure
│   ├── docker/
│   └── k8s/
└── tests/                   # Integration tests
```

### 7.3 Testing Strategy

#### Probe Testing

Probes verify APIs before use (Law of Runtime Truth):

```bash
# Run all probes
find glue -name "check_*.sh" -exec bash {} \;

# Run specific adapter probes
bash glue/adapters/icr-adapter/probes/check_api.sh
```

#### Contract Testing

Contract tests validate API contracts at runtime:

```bash
# Run all contract tests
npm run test:contract

# Run specific adapter contract tests
cd glue/adapters/icr-adapter
npm run test:contract
```

#### Integration Testing

```bash
# Run integration tests
npm run test:integration
```

### 7.4 Code Quality

All code must pass:

```bash
# Linting
npm run lint

# Type checking
npm run typecheck

# Formatting
npm run format
```

---

## 8. Deployment

### 8.1 Docker Deployment

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ICR Adapter
  icr-adapter:
    build: ./glue/adapters/icr-adapter
    environment:
      - OPENEVOLVE_ICR_API_URL=${ICR_API_URL}
      - TIMEOUT_MS=${TIMEOUT_MS}
    ports:
      - "8080:8080"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Evolved Code Capture
  evolved-code-capture:
    build: ./glue/lib/evolved-code-capture
    environment:
      - VECTORDB_ADAPTER_URL=${VECTORDB_URL}
      - GRAPHITI_ADAPTER_URL=${GRAPHITI_API_URL}
    depends_on:
      - vectordb-adapter
      - graphiti-adapter
    restart: unless-stopped

  # Unified Knowledge Query
  unified-knowledge-query:
    build: ./glue/lib/unified-knowledge-query
    environment:
      - RAGBITS_API_URL=${RAGBITS_URL}
      - GRAPHITI_API_URL=${GRAPHITI_URL}
      - VECTORDB_URL=${VECTORDB_URL}
    restart: unless-stopped
```

### 8.2 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: icr-adapter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: icr-adapter
  template:
    metadata:
      labels:
        app: icr-adapter
    spec:
      containers:
      - name: icr-adapter
        image: openevolve/icr-adapter:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENEVOLVE_ICR_API_URL
          valueFrom:
            configMapKeyRef:
              name: icr-config
              key: api-url
        - name: TIMEOUT_MS
          valueFrom:
            configMapKeyRef:
              name: icr-config
              key: timeout
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: icr-config
data:
  api-url: "http://icr-core:8080"
  timeout: "5000"
```

### 8.3 Deployment Checklist

- [ ] All environment variables configured
- [ ] All probes passing
- [ ] All contract tests passing
- [ ] Docker images built
- [ ] Kubernetes resources created
- [ ] Health checks configured
- [ ] Monitoring setup
- [ ] Log aggregation configured
- [ ] Rollback plan tested

---

## 9. Monitoring & Observability

### 9.1 Metrics

All components expose Prometheus-compatible metrics:

```typescript
// Common metrics
interface ComponentMetrics {
  // Request metrics
  total_requests: Counter
  successful_requests: Counter
  failed_requests: Counter
  request_duration_ms: Histogram

  // Circuit breaker metrics
  circuit_breaker_state: Gauge
  circuit_breaker_failures: Counter
  circuit_breaker_successes: Counter

  // System-specific metrics
  system_specific_metrics: object
}
```

#### Metrics Collection

```typescript
import { Prometheus } from '@openevolve/lib';

const prometheus = new Prometheus({
  port: 9090,
});

// Expose metrics
prometheus.expose('icr_adapter_requests_total');
prometheus.expose('icr_adapter_request_duration_ms');
```

### 9.2 Logging

All components use structured JSON Lines logging:

```json
{
  "level": "info",
  "msg": "Request completed",
  "timestamp_utc": "2026-02-03T12:34:56.789Z",
  "service": "icr-adapter",
  "correlation_id": "abc-123-def",
  "source_service": "icr-adapter",
  "target_service": "icr-core",
  "duration_ms": 1234,
  "result": "success"
}
```

### 9.3 Distributed Tracing

OpenTelemetry integration for end-to-end tracing:

```typescript
import { trace } from '@opentelemetry/api';

const tracer = trace.getTracer('icr-adapter');

async function executeQuery(query: string) {
  const span = tracer.startSpan('executeQuery');

  try {
    const result = await icrClient.query(query);
    span.setStatus({ code: SpanStatusCode.OK });
    return result;
  } catch (error) {
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message
    });
    throw error;
  } finally {
    span.end();
  }
}
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Circuit Breaker Open

**Symptoms**: Requests failing with "Circuit breaker is open" error

**Diagnosis**:
```bash
# Check circuit breaker state
curl http://icr-adapter:8080/metrics | jq '.circuit_breaker_state'

# Check recent error logs
kubectl logs -l app=icr-adapter --tail=100 | grep ERROR
```

**Solutions**:
1. Wait for automatic reset (60 seconds default)
2. Manual reset:
   ```typescript
   await adapter.resetCircuitBreaker();
   ```
3. Fix upstream service and retry

#### Timeout Errors

**Symptoms**: Requests timing out after TIMEOUT_MS

**Diagnosis**:
```bash
# Check service health
curl http://target-service:8080/health

# Check network latency
ping -c 5 target-service
```

**Solutions**:
1. Increase TIMEOUT_MS
2. Optimize the operation
3. Check for resource constraints

#### Schema Validation Failed

**Symptoms**: "Schema validation failed" error

**Diagnosis**:
```bash
# Check canonical schema
cat glue/adapters/icr-adapter/src/canonical.ts | grep "Schema"

# Validate request manually
```

**Solutions**:
1. Check required fields
2. Verify UTC timestamp format
3. Check data types match schema

### 10.2 Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=debug
```

Enable verbose output:

```bash
export VERBOSE=true
```

### 10.3 Health Check Commands

```bash
# ICR Adapter
curl http://icr-adapter:8080/health

# Unified Knowledge Query
curl http://unified-knowledge-query:8080/health

# All adapters
for url in \
  http://icr-adapter:8080 \
  http://unified-knowledge-query:8080 \
  http://unified-verification:8080
do
  echo "Checking $url..."
  curl -s "$url/health" | jq '.status'
done
```

---

## 11. Best Practices

### 11.1 Using Adapters

**DO**:
- ✅ Always provide correlation IDs
- ✅ Check health status before critical operations
- ✅ Implement circuit breaker handling
- ✅ Use structured logging
- ✅ Monitor metrics

**DON'T**:
- ❌ Ignore circuit breaker state
- ❌ Skip validation
- ❌ Hardcode timeouts
- ❌ Use console.log
- ❌ Assume services are available

### 11.2 Error Handling

```typescript
// Good: Proper error handling
try {
  const result = await adapter.executeQuery(request, correlationId);
  return result;
} catch (error) {
  if (error instanceof CircuitBreakerOpenError) {
    logger.warn({
      msg: 'Circuit breaker open, using fallback',
      correlation_id,
      error: error.message,
    });
    return await fallbackOperation(request, correlationId);
  }

  logger.error({
    msg: 'Operation failed',
    correlation_id,
    error: error.message,
  });
  throw error;
}

// Bad: Ignoring errors
try {
  return await adapter.executeQuery(request);
} catch (error) {
  // Ignored!
  return null;
}
```

### 11.3 Performance Optimization

**Use Parallel Execution**:
```typescript
// Good: Parallel queries
const [docs, entities, vectors] = await Promise.all([
  ragbits.search(query),
  graphiti.getEntities(query),
  vectordb.search(query)
]);

// Bad: Sequential
const docs = await ragbits.search(query);
const entities = await graphiti.getEntities(query);
const vectors = await vectordb.search(query);
```

**Use Caching**:
```typescript
const cache = new LRUCache({
  max: 1000,
  ttl: 300000 // 5 minutes
});

async function getCachedResult(key: string) {
  let result = cache.get(key);
  if (result) return result;

  result = await expensiveOperation(key);
  cache.set(key, result);
  return result;
}
```

**Use Connection Pooling**:
```typescript
// Configure HTTP client with connection pool
const client = axios.create({
  baseURL: apiUrl,
  httpAgent: new http.Agent({
    keepAlive: true,
    maxSockets: 100,
    maxFreeSockets: 10,
  }),
});
```

---

## 12. Federation Constitution

### 12.1 The 6 Immutable Laws

#### Law 1: Air Gap (Source Code Isolation)

**The Rule**: No imports from `core-projects/`

**Implementation**:
- All adapters communicate via HTTP APIs
- No direct dependencies on core project code
- Complete isolation between glue layer and core

**Compliance Check**:
```bash
# Verify no imports from core-projects
grep -r "from.*core-projects" glue/ || echo "✓ Compliant"
```

#### Law 2: Runtime Truth (Anti-Hallucination)

**The Rule**: Probes verify APIs before implementation

**Implementation**:
- Probe scripts for each adapter
- Contract tests validate on startup
- API behavior verified, not documentation

**Compliance Check**:
```bash
# Run probes before using
bash glue/adapters/icr-adapter/probes/check_api.sh
```

#### Law 3: Untouchable DB (Read-Only State)

**The Rule**: SELECT privileges only

**Implementation**:
- No direct database writes
- Writes only through APIs
- Idempotent operations only

**Compliance Check**:
```bash
# Verify read-only access
grep -r "INSERT\|UPDATE\|DELETE" glue/ || echo "✓ Compliant"
```

#### Law 4: Idempotency (The Replayability Pact)

**The Rule**: All operations safe to run 100 times

**Implementation**:
- Check before create (avoid duplicates)
- UPSERT operations
- Deduplication by ID

**Compliance Check**:
```typescript
// Good: Idempotent create
async function createResource(id: string, data: any) {
  const existing = await db.get(id);
  if (existing) return existing;  // Already exists

  return await db.create(id, data);  // Create new
}

// Bad: Not idempotent
async function createResource(id: string, data: any) {
  return await db.create(id, data);  // Fails on retry
}
```

#### Law 5: Configuration Explicitness

**The Rule**: No magic defaults; crash if config missing

**Implementation**:
```typescript
// Good: Explicit validation
const apiUrl = process.env.API_URL;
if (!apiUrl) {
  throw new Error('API_URL is required');
}

// Bad: Magic default
const apiUrl = process.env.API_URL || 'http://localhost:8080';
```

#### Law 6: UTC

**The Rule**: All timestamps in UTC ISO-8601 format

**Implementation**:
```typescript
// Good: UTC timestamp
const timestamp = new Date().toISOString();

// Bad: Local timestamp
const timestamp = new Date().toLocaleString();
```

### 12.2 Constitution Compliance Checklist

Each adapter/component includes:

- [ ] No imports from `core-projects/`
- [ ] Probe scripts for API verification
- [ ] Contract tests for schema validation
- [ ] Circuit breaker implementation
- [ ] Retry logic with exponential backoff
- [ ] Structured JSON logging
- [ ] UTC timestamps only
- [ ] Required environment variables validated
- [ ] Idempotent operations

---

## 13. Glossary

| Term | Definition |
|------|------------|
| **Adapter** | Sidecar component that integrates a core project with the federation |
| **Air Gap** | Complete isolation between glue layer and core projects |
| **Circuit Breaker** | Fault tolerance pattern that prevents cascading failures |
| **Canonical Schema** | Standardized data format for cross-system communication |
| **Contract Test** | Runtime test that validates API structure compliance |
| **Probe** | Shell script that verifies API functionality before use |
| **Correlation ID** | Unique identifier that tracks a request across systems |
| **UTC** | Coordinated Universal Time (no timezone offset) |
| **Idempotency** | Property of an operation that produces same result on multiple executions |

---

## 14. Appendices

### Appendix A: File Structure

```
frontend/
├── core-projects/                          # 30+ immutable projects
│   ├── Iterative-Contextual-Refinements/    # ICR system
│   ├── OpenEvolve/                         # Evolutionary coding
│   ├── BubbleLab/                          # Workflow engine
│   ├── ragbits/                            # RAG system
│   └── ...
├── glue/                                   # Integration layer
│   ├── adapters/
│   │   ├── icr-adapter/                   # ✅ NEW
│   │   ├── openevolve-adapter/
│   │   ├── bubblelab-adapter/
│   │   ├── ragbits-adapter/
│   │   ├── ragbits-graphiti-sync/          # ✅ NEW
│   │   ├── vectordb-adapter/
│   │   ├── graphiti-adapter/
│   │   ├── z3-adapter/
│   │   ├── leanaide-adapter/
│   │   └── karateclub-adapter/
│   ├── orchestration/
│   │   ├── unified-verification/          # ✅ NEW
│   │   └── knowledge-flow-orchestrator/  # ✅ NEW
│   ├── schemas/
│   └── lib/
│       ├── evolved-code-capture/         # ✅ NEW
│       ├── unified-knowledge-query/       # ✅ NEW
│       ├── proof-knowledge-base/           # ✅ NEW
│       ├── metrics/
│       └── tracing/
├── infra/
│   ├── docker/
│   └── k8s/
├── tests/
│   ├── integration/
│   └── contract/
└── docs/
    ├── INTEGRATION_GAP_ANALYSIS_REPORT.md
    ├── INTEGRATION_IMPLEMENTATION_COMPLETE.md
    ├── QUICK_START_GUIDE.md
    └── COMPREHENSIVE_DOCUMENTATION.md    # This file
```

### Appendix B: Environment Variables

**Complete environment variable reference for all components:**

```bash
# ================================
# Core Services
# ================================

# ICR (Iterative Contextual Refinements)
OPENEVOLVE_ICR_API_URL=http://localhost:8080

# OpenEvolve
OPENEVOLVE_API_URL=http://localhost:8002

# BubbleLab
BUBBLELAB_API_URL=http://localhost:3000

# Knowledge Systems
RAGBITS_API_URL=http://localhost:8082
GRAPHITI_API_URL=http://localhost:8084
VECTORDB_URL=http://localhost:8083

# Formal Verification
Z3_API_URL=http://localhost:8080
LEANAIDE_API_URL=http://localhost:8081

# ================================
# Common Configuration
# ================================

# Timeouts
TIMEOUT_MS=5000

# Circuit Breakers
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_MS=60000

# Retry Logic
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY_MS=1000
RETRY_MAX_DELAY_MS=10000

# ================================
# Optional Services
# ================================

# Vector DB Backend
VECTORDB_TYPE=qdrant
VECTORDB_COLLECTION=knowledge

# Graphiti Backend
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USER=neo4j
GRAPHITI_NEO4J_PASSWORD=password

# OpenAI (for embeddings)
OPENAI_API_KEY=sk-your-key-here

# ================================
# Adapter-Specific Configuration
# ================================

# ICR Memory
ICR_MEMORY_ENABLED=true
ICR_CONTEXT_WINDOW=10

# Verification
ENABLE_PARALLEL_VERIFICATION=true
VERIFICATION_TIMEOUT_MS=30000

# Knowledge Query
UNIFIED_QUERY_TIMEOUT_MS=10000
MIN_CONFIDENCE=0.6

# Code Capture
ENABLE_CODE_CAPTURE=true
CODE_CAPTURE_COLLECTION=evolved_code

# Proof Storage
PROOF_STORAGE_BACKEND=graph
PROOF_EMBEDDING_MODEL=text-embedding-3-small
```

### Appendix C: API Endpoint Reference

#### ICR Adapter Endpoints

```
POST /api/v1/refine
POST /api/v1/react
POST /api/v1/deepthink
POST /api/v1/adaptive-deepthink
POST /api/v1/agentic
POST /api/v1/contextual
POST /api/v1/generative-ui
GET  /health
GET  /metrics
```

#### Unified Knowledge Query Endpoints

```
POST /api/v1/query
POST /api/v1/semantic
POST /api/v1/temporal
POST /api/v1/graph
GET  /health
GET  /metrics
```

#### Unified Verification Endpoints

```
POST /api/v1/verify
POST /api/v1/verify-cross-validation
POST /api/v1/batch-verify
GET  /health
GET  /metrics
```

### Appendix D: Migration Guide

#### Migrating from Direct API Calls

**Before** (direct API calls):
```typescript
// Direct call to ICR - NOT RECOMMENDED
const response = await fetch('http://icr-core:8080/refine', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'Refine this' })
});
```

**After** (using adapter):
```typescript
// Using ICR adapter - RECOMMENDED
import { icrAdapter } from '@openevolve/icr-adapter';

const response = await icrAdapter.createRefinementRequest(
  'Refine this',
  { temperature: 0.7 },
  'corr-123'
);
```

**Benefits**:
- ✅ Circuit breaker protection
- ✅ Automatic retry
- ✅ Structured logging
- ✅ Correlation ID tracking
- ✅ Type safety
- ✅ Schema validation

### Appendix E: Performance Tuning

#### Circuit Breaker Tuning

```typescript
// For reliable services
circuitBreaker: {
  failureThreshold: 10,        // Allow more failures
  successThreshold: 2,         // Quick recovery
  timeout: 30000               // 30s open state
}

// For unreliable services
circuitBreaker: {
  failureThreshold: 3,         // Trip quickly
  successThreshold: 5,         // Slow recovery
  timeout: 120000              // 2m open state
}
```

#### Retry Tuning

```typescript
// For latency-sensitive operations
retry: {
  maxAttempts: 2,               // Fewer retries
  baseDelay: 500,               // Faster retry
  maxDelay: 2000                // Low max delay
}

// For critical operations
retry: {
  maxAttempts: 5,               // More retries
  baseDelay: 2000,              // Slower start
  maxDelay: 30000               // Allow long delays
}
```

#### Query Optimization

```typescript
// For faster queries
queryOptions: {
  maxResults: 10,               // Fewer results
  minConfidence: 0.8,           // Higher threshold
  parallel: true,                // Execute in parallel
  cacheEnabled: true             // Use cache
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-03 | Initial release - Complete implementation of all 12 components |

---

## Support

For questions, issues, or contributions:

- **Documentation**: See component-specific README files
- **Issues**: Create GitHub issues with component tag
- **Discussions**: Use GitHub Discussions for questions
- **Architecture Review**: See ADR.md files for design decisions

---

**End of Comprehensive Documentation**

*This documentation covers the complete OpenEvolve Federation integration implementation. For specific component details, refer to individual README files and ADR documents.*
EOF