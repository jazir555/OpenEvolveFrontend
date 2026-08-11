# Hybrid PES System Development Guide

Comprehensive guide for developers working on the OpenEvolve LoongFlow PES hybrid system.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Code Structure](#code-structure)
3. [Adding New Adapters](#adding-new-adapters)
4. [Adding New Workflows](#adding-new-workflows)
5. [Testing Guidelines](#testing-guidelines)
6. [Deployment Process](#deployment-process)
7. [Coding Standards](#coding-standards)
8. [Git Workflow](#git-workflow)
9. [Documentation Standards](#documentation-standards)

## Development Environment Setup

### Prerequisites

- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Python**: 3.9+ (for some adapters)
- **Redis**: 6.0+ (for event bus)
- **Git**: Latest version

### 1. Clone Repository

```bash
git clone https://github.com/your-org/hybrid-pes-system.git
cd hybrid-pes-system
```

### 2. Install Dependencies

```bash
# Install root dependencies
npm install

# Install adapter dependencies
cd glue/adapters/loongflow-adapter
npm install

cd ../openevolve-adapter
npm install

cd ../..
```

### 3. Setup Environment Variables

```bash
# Copy environment templates
cp infra/.env.loongflow.example infra/.env.loongflow
cp infra/.env.example infra/.env

# Edit environment files
nano infra/.env.loongflow
```

**Required Variables**:
```bash
# LoongFlow
LOONGFLOW_API_URL=http://loongflow-core:8050
LOONGFLOW_LLM_API_KEY=sk-your-key-here

# OpenEvolve
OPENEVOLVE_API_URL=http://openevolve-core:8000

# Event Bus
EVENT_BUS_URL=redis://event-bus:6379

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Timezone (Law of UTC)
TZ=UTC
```

### 4. Start Development Services

```bash
cd infra

# Start LoongFlow core
docker-compose -f docker-compose.loongflow-core.yml --env-file .env.loongflow up -d

# Start all adapters
docker-compose -f docker-compose-all-adapters.yml up -d

# Start Redis (event bus)
docker-compose up -d redis
```

### 5. Verify Setup

```bash
# Check LoongFlow core
curl http://localhost:8050/health

# Check LoongFlow adapter
curl http://localhost:8040/health

# Check OpenEvolve adapter
curl http://localhost:8000/health

# Check Redis
redis-cli -h localhost -p 6379 ping
```

### 6. Run Tests

```bash
# LoongFlow adapter tests
cd glue/adapters/loongflow-adapter
npm test

# OpenEvolve adapter tests
cd ../openevolve-adapter
npm test

# All tests
cd ../../..
npm run test:all
```

## Code Structure

### Directory Layout

```
hybrid-pes-system/
├── glue/                          # Glue layer (isolated from core)
│   ├── adapters/                  # Adapter implementations
│   │   ├── loongflow-adapter/     # LoongFlow integration
│   │   │   ├── src/               # Source code
│   │   │   │   ├── adapter.ts     # Main adapter class
│   │   │   │   ├── index.ts       # Exports
│   │   │   │   └── types.ts       # TypeScript types
│   │   │   ├── tests/             # Contract tests
│   │   │   │   ├── contract.test.ts
│   │   │   │   └── fixtures/      # Test data
│   │   │   ├── probes/            # API verification scripts
│   │   │   ├── dist/              # Compiled output
│   │   │   ├── package.json
│   │   │   ├── tsconfig.json
│   │   │   └── README.md
│   │   └── openevolve-adapter/    # OpenEvolve integration
│   │       └── (similar structure)
│   ├── orchestration/             # Orchestration layer
│   │   ├── event-bus.ts           # Event bus implementation
│   │   ├── dead-letter-queue.ts   # DLQ implementation
│   │   ├── circuit-breaker.ts     # Circuit breaker
│   │   ├── correlation-tracker.ts # Request tracing
│   │   └── examples/              # Example workflows
│   │       ├── knowledge-processing-workflow.ts
│   │       └── hybrid-pes-evolution-workflow.ts
│   ├── schemas/                   # Canonical schemas
│   │   ├── pes-canonical.ts       # Core PES schemas
│   │   ├── loongflow-canonical.ts # LoongFlow schemas
│   │   ├── openevolve-canonical.ts # OpenEvolve schemas
│   │   ├── hybrid-pes-evolution-canonical.ts # Hybrid schemas
│   │   └── index.ts               # Schema exports
│   ├── lib/                       # Shared utilities
│   │   ├── circuit-breaker.ts     # Circuit breaker implementation
│   │   ├── retry.ts               # Retry logic
│   │   ├── logger.ts              # Structured logger
│   │   └── validator.ts           # Environment validator
│   └── tests/                     # Integration tests
│       └── test_rese_complete_pipeline.py
├── core-projects/                 # Immutable core projects
│   ├── LoongFlow/                 # READ ONLY
│   ├── OpenEvolve/                # READ ONLY
│   └── ...
├── infra/                         # Infrastructure
│   ├── docker-compose.yml         # Main compose file
│   ├── docker-compose.loongflow-core.yml
│   ├── docker-compose-all-adapters.yml
│   ├── k8s-loongflow-core.yaml    # Kubernetes manifests
│   ├── k8s-loongflow-deployment.yaml
│   └── scripts/                   # Deployment scripts
│       ├── deploy-loongflow.sh
│       ├── validate-loongflow-deployment.sh
│       └── health-check.sh
├── docs/                          # Documentation
│   ├── architecture/
│   ├── guides/
│   └── api/
└── HYBRID_PES_README.md           # Main README
```

### File Naming Conventions

- **TypeScript files**: `kebab-case.ts` (e.g., `event-bus.ts`)
- **Test files**: `*.test.ts` or `*.spec.ts`
- **Schema files**: `*-canonical.ts` (e.g., `loongflow-canonical.ts`)
- **Config files**: `*.config.js` or `*.config.ts`
- **Documentation**: `UPPERCASE.md` (e.g., `README.md`)

### Import Conventions

```typescript
// Absolute imports for project files
import { LoongFlowAdapter } from '@/adapters/loongflow-adapter';
import { PlanSchema } from '@/schemas/pes-canonical';

// Relative imports for local files
import { logger } from './lib/logger';
import { EventTypes } from './event-types';
```

## Adding New Adapters

### Step 1: Create Adapter Directory

```bash
mkdir -p glue/adapters/new-adapter/src
mkdir -p glue/adapters/new-adapter/tests
mkdir -p glue/adapters/new-adapter/probes
```

### Step 2: Initialize Package

```bash
cd glue/adapters/new-adapter
npm init -y
npm install axios uuid zod
npm install --save-dev typescript @types/node @types/uuid jest ts-jest
```

### Step 3: Create TypeScript Config

**File**: `tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "composite": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

### Step 4: Implement Adapter

**File**: `src/adapter.ts`

```typescript
import axios, { AxiosInstance } from 'axios';
import { CircuitBreaker } from '@/lib/circuit-breaker';
import { logger } from '@/lib/logger';

export interface NewAdapterConfig {
  apiURL: string;
  timeout?: number;
  maxRetries?: number;
}

export class NewAdapter {
  private client: AxiosInstance;
  private circuitBreaker: CircuitBreaker;

  constructor(config: NewAdapterConfig) {
    this.client = axios.create({
      baseURL: config.apiURL,
      timeout: config.timeout || 30000,
    });

    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout: 60000,
      halfOpenAttempts: 3,
    });
  }

  async executeOperation(params: any): Promise<any> {
    return this.circuitBreaker.execute(async () => {
      logger.info({ msg: 'Executing operation', params });

      const response = await this.client.post('/api/operation', params);

      logger.info({
        msg: 'Operation completed',
        duration: response.config.headers['request-duration']
      });

      return response.data;
    });
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }
}
```

### Step 5: Create Canonical Schema

**File**: `glue/schemas/new-adapter-canonical.ts`

```typescript
import { z } from 'zod';

export const NewAdapterOperationRequestSchema = z.object({
  param1: z.string().min(1),
  param2: z.number().min(0),
  options: z.object({
    enableFeature: z.boolean().optional(),
  }).optional(),
});

export const NewAdapterOperationResponseSchema = z.object({
  operationId: z.string().uuid(),
  status: z.enum(['pending', 'running', 'completed', 'failed']),
  result: z.any().optional(),
  error: z.string().optional(),
});

export type NewAdapterOperationRequest = z.infer<typeof NewAdapterOperationRequestSchema>;
export type NewAdapterOperationResponse = z.infer<typeof NewAdapterOperationResponseSchema>;
```

### Step 6: Write Contract Tests

**File**: `tests/contract.test.ts`

```typescript
import { NewAdapter } from '../src/adapter';
import { NewAdapterOperationRequestSchema } from '@/schemas/new-adapter-canonical';

describe('NewAdapter Contract Tests', () => {
  let adapter: NewAdapter;

  beforeAll(() => {
    adapter = new NewAdapter({
      apiURL: process.env.NEW_ADAPTER_API_URL || 'http://localhost:8080',
    });
  });

  describe('Environment', () => {
    it('should have API URL configured', () => {
      expect(process.env.NEW_ADAPTER_API_URL).toBeDefined();
    });

    it('should respond to health check', async () => {
      const health = await adapter.healthCheck();
      expect(health.status).toBe('healthy');
    });
  });

  describe('Operations', () => {
    it('should execute operation successfully', async () => {
      const request = {
        param1: 'test',
        param2: 100,
        options: { enableFeature: true }
      };

      const validated = NewAdapterOperationRequestSchema.parse(request);
      const result = await adapter.executeOperation(validated);

      expect(result.operationId).toBeDefined();
      expect(result.status).toBe('completed');
    });
  });
});
```

### Step 7: Create Probe Scripts

**File**: `probes/check_api.sh`

```bash
#!/bin/bash
# Probe script to verify API is accessible

API_URL="${NEW_ADAPTER_API_URL:-http://localhost:8080}"

echo "Checking New Adapter API at ${API_URL}/health..."

response=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health")

if [ "$response" -eq 200 ]; then
    echo "✓ API is accessible"
    exit 0
else
    echo "✗ API is not accessible (HTTP ${response})"
    exit 1
fi
```

Make executable:
```bash
chmod +x probes/check_api.sh
```

### Step 8: Create Dockerfile

**File**: `Dockerfile`

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8080/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Start service
CMD ["node", "dist/index.js"]
```

### Step 9: Update Documentation

**File**: `README.md` in adapter directory

```markdown
# New Adapter

Integration with New Service.

## Quick Start

\`\`\`bash
npm install
npm run build
npm test
\`\`\`

## Configuration

Environment variables:
- \`NEW_ADAPTER_API_URL\`: API URL (required)

## API

See \`API.md\` for complete API documentation.
```

## Adding New Workflows

### Step 1: Create Workflow File

**File**: `glue/orchestration/examples/new-workflow.ts`

```typescript
import { LoongFlowAdapter } from '@/adapters/loongflow-adapter';
import { OpenEvolveAdapter } from '@/adapters/openevolve-adapter';
import { EventBus } from '@/orchestration/event-bus';
import { logger } from '@/lib/logger';

export interface NewWorkflowConfig {
  loongflowURL: string;
  openevolveURL: string;
  eventBusURL: string;
}

export class NewWorkflow {
  private loongflow: LoongFlowAdapter;
  private openevolve: OpenEvolveAdapter;
  private eventBus: EventBus;

  constructor(config: NewWorkflowConfig) {
    this.loongflow = new LoongFlowAdapter({ apiURL: config.loongflowURL });
    this.openevolve = new OpenEvolveAdapter({ baseURL: config.openevolveURL });
    this.eventBus = new EventBus({ url: config.eventBusURL });
  }

  async execute(input: any): Promise<any> {
    const correlationId = logger.generateCorrelationId();

    logger.info({
      msg: 'Starting new workflow',
      correlation_id: correlationId,
      input,
    });

    try {
      // Phase 1: PES Planning
      await this.eventBus.publish('workflow.phase1.started', { correlationId });
      const plan = await this.loongflow.executePESWorkflow({
        query: input.query,
        maxIterations: 3,
      });
      await this.eventBus.publish('workflow.phase1.completed', { correlationId, plan });

      // Phase 2: Evolutionary Optimization
      await this.eventBus.publish('workflow.phase2.started', { correlationId });
      const evolved = await this.openevolve.evolveSystem({
        initialPrompt: plan.summary.summary,
        generations: 5,
      });
      await this.eventBus.publish('workflow.phase2.completed', { correlationId, evolved });

      // Phase 3: Aggregation
      const result = {
        plan: plan.summary,
        evolved: evolved.bestPrompt,
        correlationId,
      };

      await this.eventBus.publish('workflow.completed', { correlationId, result });

      logger.info({
        msg: 'Workflow completed',
        correlation_id: correlationId,
      });

      return result;
    } catch (error) {
      logger.error({
        msg: 'Workflow failed',
        correlation_id: correlationId,
        error: error.message,
      });

      await this.eventBus.publish('workflow.failed', { correlationId, error });
      throw error;
    }
  }
}
```

### Step 2: Create Workflow Tests

**File**: `glue/orchestration/examples/new-workflow.test.ts`

```typescript
import { NewWorkflow } from './new-workflow';

describe('NewWorkflow', () => {
  let workflow: NewWorkflow;

  beforeAll(() => {
    workflow = new NewWorkflow({
      loongflowURL: process.env.LOONGFLOW_API_URL || '',
      openevolveURL: process.env.OPENEVOLVE_API_URL || '',
      eventBusURL: process.env.EVENT_BUS_URL || '',
    });
  });

  it('should execute workflow end-to-end', async () => {
    const result = await workflow.execute({
      query: 'Test query',
    });

    expect(result.plan).toBeDefined();
    expect(result.evolved).toBeDefined();
    expect(result.correlationId).toBeDefined();
  }, 60000);
});
```

### Step 3: Register Workflow

**File**: `glue/orchestration/index.ts`

```typescript
export * from './event-bus';
export * from './dead-letter-queue';
export * from './circuit-breaker';
export * from './correlation-tracker';
export * from './examples/new-workflow'; // Add this
```

## Testing Guidelines

### Unit Tests

- **Location**: `src/**/*.test.ts`
- **Framework**: Jest
- **Coverage Goal**: 80%+

**Example**:
```typescript
import { CircuitBreaker } from './circuit-breaker';

describe('CircuitBreaker', () => {
  it('should open after threshold failures', async () => {
    const cb = new CircuitBreaker({ threshold: 3, timeout: 60000 });

    for (let i = 0; i < 3; i++) {
      try {
        await cb.execute(async () => { throw new Error('Fail'); });
      } catch (e) {
        // Expected
      }
    }

    expect(cb.getState()).toBe('open');
  });
});
```

### Contract Tests

- **Location**: `tests/contract.test.ts`
- **Purpose**: Validate API contracts
- **Run**: Before every deployment

**Example**:
```typescript
describe('API Contract Tests', () => {
  it('should return valid workflow response', async () => {
    const response = await adapter.executePESWorkflow({ query: 'test' });
    expect(() => {
      LoongFlowWorkflowResponseSchema.parse(response);
    }).not.toThrow();
  });
});
```

### Integration Tests

- **Location**: `glue/tests/`
- **Framework**: Jest or pytest
- **Purpose**: Test cross-system workflows

**Example**:
```typescript
describe('Hybrid Workflow Integration', () => {
  it('should execute PES + Evolution workflow', async () => {
    const workflow = new HybridPESEvolutionWorkflow({ /* config */ });
    const result = await workflow.execute({ query: 'test' });

    expect(result.plan).toBeDefined();
    expect(result.evolved).toBeDefined();
  }, 120000);
});
```

### Test Commands

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- new-workflow.test.ts

# Run in watch mode
npm test -- --watch

# Run contract tests only
npm run test:contract
```

## Deployment Process

### Development Deployment

```bash
# Build adapters
cd glue/adapters/loongflow-adapter
npm run build

# Deploy to local Docker
cd infra
docker-compose -f docker-compose-all-adapters.yml up -d --build

# Verify deployment
./scripts/health-check.sh
```

### Production Deployment

#### 1. Build Production Images

```bash
# Tag with version
docker build -t loongflow-adapter:v1.0.0 .
docker tag loongflow-adapter:v1.0.0 registry.example.com/loongflow-adapter:v1.0.0

# Push to registry
docker push registry.example.com/loongflow-adapter:v1.0.0
```

#### 2. Update Kubernetes Manifests

```yaml
# k8s-loongflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loongflow-adapter
spec:
  template:
    spec:
      containers:
      - name: loongflow-adapter
        image: registry.example.com/loongflow-adapter:v1.0.0 # Update this
```

#### 3. Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s-loongflow-deployment.yaml

# Verify rollout
kubectl rollout status deployment/loongflow-adapter

# Check pods
kubectl get pods -l app=loongflow-adapter
```

#### 4. Run Smoke Tests

```bash
./scripts/smoke-test.sh
```

### Rollback Procedure

```bash
# Kubernetes
kubectl rollout undo deployment/loongflow-adapter

# Docker Compose
docker-compose down
git revert HEAD
docker-compose up -d --build
```

## Coding Standards

### TypeScript Style

- **Indentation**: 2 spaces
- **Quotes**: Single quotes for strings, double quotes for JSON
- **Semicolons**: Required
- **Line length**: Max 100 characters
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `camelCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Types/interfaces: `PascalCase`

### Example

```typescript
// Good
interface WorkflowConfig {
  maxIterations: number;
  enableCheckpointing: boolean;
}

const DEFAULT_TIMEOUT = 30000;

class WorkflowExecutor {
  async execute(config: WorkflowConfig): Promise<Result> {
    // Implementation
  }
}

// Bad
interface workflow_config {  // Should be PascalCase
  max_iterations: number;    // Should be camelCase
  enable_checkpoint: boolean // Missing type annotation
}
```

### Error Handling

```typescript
// Always handle errors
try {
  const result = await adapter.executeWorkflow(request);
  return result;
} catch (error) {
  logger.error({
    msg: 'Workflow execution failed',
    error: error.message,
    stack: error.stack,
    correlation_id: ctx.correlationId,
  });
  throw new WorkflowExecutionError('Failed to execute workflow', { cause: error });
}
```

### Logging

```typescript
// Use structured logging
logger.info({
  msg: 'Workflow started',
  workflow_id: workflowId,
  correlation_id: correlationId,
  timestamp: new Date().toISOString(),
});

// Not
console.log('Workflow started'); // Bad
```

### Async/Await

```typescript
// Use async/await, not Promises directly
async function fetchData(): Promise<Data> {
  const response = await fetch(url);
  const data = await response.json();
  return data;
}

// Not
function fetchData(): Promise<Data> {
  return fetch(url)
    .then(response => response.json())
    .then(data => data); // Bad
}
```

## Git Workflow

### Branch Strategy

- `main`: Production code
- `develop`: Development code
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Emergency production fixes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build process or auxiliary tool changes

**Examples**:
```
feat(loongflow-adapter): add workflow cancellation support

Implement workflow cancellation endpoint that allows
stopping running workflows gracefully.

Closes #123
```

```
fix(event-bus): handle Redis connection errors

Add retry logic for Redis connection failures and
circuit breaker to prevent cascading failures.

Fixes #456
```

### Pull Request Process

1. Create feature branch:
```bash
git checkout -b feature/new-workflow
```

2. Make changes and commit:
```bash
git add .
git commit -m "feat(workflows): add new workflow"
```

3. Push and create PR:
```bash
git push origin feature/new-workflow
# Create PR on GitHub
```

4. PR checklist:
- [ ] Tests pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Follows coding standards

## Documentation Standards

### Code Documentation

```typescript
/**
 * Executes a PES workflow with the given parameters.
 *
 * @param request - The workflow request parameters
 * @param request.query - The problem statement or query
 * @param request.maxIterations - Maximum number of PES iterations (default: 5)
 * @returns Promise resolving to the workflow execution result
 * @throws {WorkflowExecutionError} If workflow execution fails
 * @throws {ValidationError} If request validation fails
 *
 * @example
 * ```typescript
 * const result = await adapter.executePESWorkflow({
 *   query: 'Solve the TSP problem',
 *   maxIterations: 5
 * });
 * ```
 */
async executePESWorkflow(request: PESWorkflowRequest): Promise<PESWorkflowResponse> {
  // Implementation
}
```

### README Structure

Each adapter should have a README with:

1. Overview
2. Quick Start
3. Configuration
4. API Documentation
5. Testing
6. Deployment
7. Troubleshooting
8. Contributing

---

**Last Updated**: 2024-02-22
**For questions**: Contact the development team or create an issue
