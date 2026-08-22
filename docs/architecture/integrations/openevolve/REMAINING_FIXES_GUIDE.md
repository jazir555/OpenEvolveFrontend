# REMAINING FIXES QUICK REFERENCE GUIDE

**Status**: Infrastructure complete, pattern established, ready to apply to remaining bubbles

---

## Completed ✅

1. ✅ **Resilience Infrastructure** (`adapters/resilience.ts`)
   - Circuit breaker (thread-safe)
   - Exponential backoff retry with jitter
   - Request deduplication
   - Dead letter queue

2. ✅ **QdrantBubble** (`service-bubbles/qdrant-bubble.ts`)
   - Extends ServiceBubble properly
   - No magic defaults
   - Full resilience integration
   - Type-safe

3. ✅ **Test Suite** (`tests/qdrant-bubble.test.ts`)
   - 85% coverage target
   - Contract tests
   - Resilience pattern tests

4. ✅ **Probe Script** (`probes/qdrant.probe.sh`)
   - Runtime truth verification
   - Health checks

5. ✅ **Documentation** (`CRITICAL_FIXES_APPLIED.md`)
   - Complete breakdown of all 18 fixes

---

## Remaining Bubbles to Fix

Apply the same pattern used for QdrantBubble to these remaining bubbles:

### 1. ElasticsearchBubble
**File**: `service-bubbles/elasticsearch-bubble.ts`
**Fixes Required**:
```typescript
// 1. Extend ServiceBubble
export class ElasticsearchBubble extends ServiceBubble<ElasticsearchParams, ElasticsearchResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'elasticsearch' as const;
  static readonly credentialType = 'elasticsearch_api_key' as const;

  // 2. Remove magic default
  baseUrl: z.string().url().describe('Elasticsearch URL (REQUIRED)'),

  // 3. Add resilience
  private resilience: ResilienceWrapper;

  constructor(params: ElasticsearchParamsInput, context?: BubbleContext) {
    super(params, context);
    this.resilience = new ResilienceWrapper('elasticsearch', DEFAULT_RESILIENCE_CONFIG);
  }
}
```

### 2. RedisBubble
**File**: `service-bubbles/redis-bubble.ts`
**Fixes Required**:
```typescript
// 1. Use real Redis client (ioredis)
import Redis from 'ioredis';

export class RedisBubble extends ServiceBubble<RedisParams, RedisResult> {
  private client: Redis;

  constructor(params: RedisParamsInput, context?: BubbleContext) {
    super(params, context);

    // 2. Real Redis client, not HTTP proxy
    this.client = new Redis(this.params.connectionString, {
      password: this.params.password,
      db: this.params.database,
      retryStrategy: (times) => Math.min(times * 50, 2000),
    });
  }

  // 3. Required connection string (no default)
  connectionString: z.string().describe('Redis connection string (REQUIRED)'),
}
```

### 3. PostgreSQLBubbleExtended
**File**: `service-bubbles/postgresql-bubble.ts`
**Fixes Required**:
```typescript
// 1. Fix method calls to match base class
// Before:
const result = await this.executeWithParams(query, params);

// After:
const result = await this.query(query, params);

// 2. Real backup/restore (not mock)
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async backup(): Promise<PostgresResult> {
  const cmd = `pg_dump ${this.params.databaseName} > ${this.params.backupPath}`;
  await execAsync(cmd);
  // ...
}
```

### 4. KnowledgeEngineBubble
**File**: `service-bubbles/knowledge-engine-bubble.ts`
**Fixes Required**:
```typescript
// 1. Extend ServiceBubble
export class KnowledgeEngineBubble extends ServiceBubble<KnowledgeEngineParams, KnowledgeEngineResult> {
  // 2. Real embedding service
  private embeddings: OpenAIEmbeddings;

  constructor(params: KnowledgeEngineParamsInput, context?: BubbleContext) {
    super(params, context);

    this.embeddings = new OpenAIEmbeddings({
      modelName: 'text-embedding-3-small',
      openAIApiKey: this.params.openaiApiKey, // Required
    });
  }

  // 3. Remove mock embedding generation
  async generateEmbedding(text: string): Promise<number[]> {
    return await this.embeddings.embedQuery(text);
    // NOT: return Array(1536).fill(0).map(() => Math.random());
  }
}
```

### 5. WorkflowOrchestratorBubble
**File**: `service-bubbles/workflow-orchestrator-bubble.ts`
**Fixes Required**:
```typescript
// 1. Extend ServiceBubble
export class WorkflowOrchestratorBubble extends ServiceBubble<WorkflowParams, WorkflowResult> {
  // 2. Remove magic default
  baseUrl: z.string().url().describe('Workflow orchestrator URL (REQUIRED)'),

  // 3. Add resilience
  private resilience: ResilienceWrapper;
}
```

### 6. CrewAIBubble
**File**: `service-bubbles/crewai-bubble.ts`
**Fixes Required**:
```typescript
// 1. Extend AIAgentBubble (not ServiceBubble)
export class CrewAIBubble extends AIAgentBubble<CrewAIParams, CrewAIResult> {
  // 2. Remove magic default
  baseUrl: z.string().url().describe('CrewAI URL (REQUIRED)'),

  // 3. Either use aiAgent properly or remove it
  // If keeping:
  private aiAgent: AIAgentBubble;

  async action(): Promise<CrewAIResult> {
    // Actually use this.aiAgent.action() here
  }
}
```

### 7. ACEToolsBubble
**File**: `service-bubbles/ace-tools-bubble.ts`
**Fixes Required**:
```typescript
// 1. Extend ServiceBubble
export class ACEToolsBubble extends ServiceBubble<ACEToolsParams, ACEToolsResult> {
  // 2. Remove magic default
  baseUrl: z.string().url().describe('ACE tools URL (REQUIRED)'),

  // 3. Add real business logic (not just HTTP pass-through)
  async performAnalytics(): Promise<ACEToolsResult> {
    // Actual analytics implementation
  }
}
```

### 8. LogParserTool
**File**: `tool-bubbles/log-parser-tool.ts`
**Fixes Required**:
```typescript
// 1. Extend ToolBubble
export class LogParserTool extends ToolBubble<LogParserParams, LogParserResult> {
  // 2. Real file I/O (not HTTP)
  import { readFile } from 'fs/promises';

  async parseLogs(): Promise<LogParserResult> {
    const content = await readFile(this.params.logPath, 'utf-8');
    // Parse logic...
  }
}
```

### 9. MetricsCollectorTool
**File**: `tool-bubbles/metrics-collector-tool.ts`
**Fixes Required**:
```typescript
// 1. Extend ToolBubble
export class MetricsCollectorTool extends ToolBubble<MetricsParams, MetricsResult> {
  // 2. Remove magic default
  prometheusUrl: z.string().url().describe('Prometheus URL (REQUIRED)'),

  // 3. Use proper Prometheus client
  // npm install prom-client @opentelemetry/api
}
```

---

## Test File Template

For each bubble, create a test file following this pattern:

```typescript
// tests/elasticsearch-bubble.test.ts
import { describe, it, expect } from 'vitest';
import { ElasticsearchBubble } from '../service-bubbles/elasticsearch-bubble';

describe('ElasticsearchBubble', () => {
  it('should extend ServiceBubble', () => {
    const bubble = new ElasticsearchBubble({
      operation: 'health_check',
      baseUrl: 'http://localhost:9200',
    });
    expect(bubble.constructor.name).toBe('ElasticsearchBubble');
  });

  it('should require baseUrl', () => {
    expect(() => {
      new ElasticsearchBubble({
        operation: 'health_check',
        // @ts-expect-error
        baseUrl: undefined,
      });
    }).toThrow();
  });

  it('should perform health check', async () => {
    // Mock fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 200 }),
    });

    const bubble = new ElasticsearchBubble({
      operation: 'health_check',
      baseUrl: 'http://localhost:9200',
    });

    const result = await bubble.action();
    expect(result.success).toBeDefined();
  });
});
```

---

## Probe Script Template

For each service, create a probe script:

```bash
#!/bin/bash
# probes/elasticsearch.probe.sh

set -euo pipefail

ELASTICSEARCH_URL="${ELASTICSEARCH_BASE_URL:-http://localhost:9200}"

echo "Probing Elasticsearch at ${ELASTICSEARCH_URL}..."

if curl -f -s "${ELASTICSEARCH_URL}/_cluster/health" > /dev/null 2>&1; then
  echo "✅ Elasticsearch probe successful"
  exit 0
else
  echo "❌ Elasticsearch probe failed"
  exit 1
fi
```

---

## Quick Fix Checklist

For each bubble, go through this checklist:

- [ ] Extend proper base class (ServiceBubble/ToolBubble/AIAgentBubble)
- [ ] Add static properties (service, authType, bubbleName, type, credentialType)
- [ ] Remove all `.default()` from required fields
- [ ] Add resilience wrapper
- [ ] Replace direct fetch with resilience.execute()
- [ ] Remove all `as any` type assertions
- [ ] Implement real service clients (no mocks)
- [ ] Add proper error handling
- [ ] Create test file
- [ ] Create probe script
- [ ] Update documentation

---

## Estimated Effort

Per bubble: 30-45 minutes
Total for remaining 8 bubbles: 4-6 hours

With the infrastructure complete and QdrantBubble as a reference, the remaining fixes should be straightforward.

---

## Priority Order

1. **High Priority** (Blocking production):
   - ElasticsearchBubble (search critical)
   - RedisBubble (caching critical)
   - PostgreSQLBubbleExtended (data persistence)

2. **Medium Priority** (Important but not blocking):
   - KnowledgeEngineBubble (core functionality)
   - WorkflowOrchestratorBubble (orchestration)

3. **Lower Priority** (Can be done post-deployment):
   - CrewAIBubble (AI agent)
   - ACEToolsBubble (analytics)
   - LogParserTool (utility)
   - MetricsCollectorTool (monitoring)

---

**Next Step**: Apply fixes to remaining bubbles following the QdrantBubble pattern.

**Reference**: `service-bubbles/qdrant-bubble.ts` (complete example)
