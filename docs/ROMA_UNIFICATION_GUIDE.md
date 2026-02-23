# Guide: Unifying OpenEvolve and BubbleLab ROMA Implementations

## Problem

Two separate, disconnected ROMA implementations exist:

1. **OpenEvolve-Plugin ROMA** (`OpenEvolve-Plugin/`)
   - Visual programming node
   - Synchronous execution
   - 5 reasoning modes, 6 agent roles
   - Calls `/api/openevolve/roma/solve`

2. **ROMA BubbleLab Plugin** (`glue/adapters/roma/roma-bubblelab-plugin/`)
   - Comprehensive adapter (3-layer architecture)
   - Asynchronous execution
   - Full API coverage (15+ endpoints)
   - Retry, caching, MCP, toolkits, profiles
   - Calls ROMA backend at `/api/v1/*`

## Solution: Unified Integration

### Step 1: Update OpenEvolve-Plugin to Use BubbleLab Service

**File:** `OpenEvolve-Plugin/src/nodes/ROMANode.ts`

```typescript
import { RomaService } from '../../../glue/adapters/roma/roma-bubblelab-plugin/src/services/RomaService';
import { RomaClient } from '../../../glue/adapters/roma/roma-bubblelab-plugin/src/services/RomaClient';

// BEFORE: Direct API call
async executeNode(inputs) {
  const response = await fetch('/api/openevolve/roma/solve', {
    method: 'POST',
    body: JSON.stringify(inputs),
  });
  return response.json();
}

// AFTER: Use BubbleLab service layer
async executeNode(inputs) {
  // Initialize service
  const client = new RomaClient({
    serverUrl: process.env.ROMA_SERVER_URL || 'http://localhost:8000',
    apiKey: this.config.apiKey,
    timeout: this.config.timeout || 30000,
  });

  const service = new RomaService(client);

  // Use BubbleLab's executeTaskWithRetry (includes retry logic!)
  const result = await service.executeTaskWithRetry(
    inputs.task,
    {
      maxDepth: inputs.maxDepth || 3,
      executionMethod: this.mapReasoningModeToExecutionMethod(inputs.reasoningMode),
    },
    3,  // retries
    1000  // base delay
  );

  return {
    solution: result.result,
    confidence: result.statistics?.confidence || 0,
    reasoning_trace: result.result?.reasoning,
    agent_votes: result.metadata?.agentVotes,
    quality_metrics: service.analyzeExecutionPerformance(result),
    execution_id: result.executionId,
  };
}

private mapReasoningModeToExecutionMethod(mode: string): string {
  const mapping = {
    'collaborative': 'roma',
    'adversarial': 'roma_mdap_maker',
    'debate': 'roma',
    'consensus': 'parallel',
    'hierarchical': 'roma',
  };
  return mapping[mode] || 'auto';
}
```

### Step 2: Update OpenEvolve Schema

**File:** `OpenEvolve-Plugin/src/schemas/roma.ts`

```typescript
// Add advanced configuration options from BubbleLab plugin
export interface ROMANodeConfig {
  // Existing fields
  taskDescription: string;
  reasoningMode: RomaReasoningMode;
  numAgents: number;
  agentRoles: RomaAgentRole[];
  reasoningRounds: number;
  confidenceThreshold: number;
  includeReasoningTrace: boolean;
  enableAgentVoting: boolean;

  // NEW: Advanced fields from BubbleLab plugin
  maxDepth?: number;  // From BubbleLab
  executionMethod?: RomaExecutionMethod;  // From BubbleLab
  enableRetry?: boolean;  // Use BubbleLab's retry logic
  enableCaching?: boolean;  // Use BubbleLab's cache
  cacheTTL?: number;  // Cache TTL in ms
  enableMcp?: boolean;  // Enable MCP server integration
  mcpServers?: string[];  // MCP servers to use
  toolkits?: string[];  // Toolkits to enable
}
```

### Step 3: Create Shared Types Module

**File:** `glue/adapters/roma/shared-types.ts`

```typescript
/**
 * Shared ROMA types for both OpenEvolve-Plugin and BubbleLab plugin
 */

// Reasoning modes map to execution methods
export const REASONING_MODE_TO_EXECUTION_METHOD: Record<string, string> = {
  'collaborative': 'roma',
  'adversarial': 'roma_mdap_maker',
  'debate': 'parallel',
  'consensus': 'majority',
  'hierarchical': 'chain_of_thought',
};

// Agent roles map to ROMA modules
export const AGENT_ROLE_TO_ROMA_MODULE: Record<string, string> = {
  'analyst': 'atomizer',
  'critic': 'verifier',
  'synthesizer': 'aggregator',
  'validator': 'verifier',
  'explorer': 'executor',
  'integrator': 'planner',
};

/**
 * Transform OpenEvolve node inputs to BubbleLab plugin format
 */
export function transformOpenEvolveToBubbleLab(inputs: any): {
  goal: string;
  maxDepth?: number;
  executionMethod?: string;
  options: any;
} {
  return {
    goal: inputs.taskDescription,
    maxDepth: inputs.maxDepth || 3,
    executionMethod: REASONING_MODE_TO_EXECUTION_METHOD[inputs.reasoningMode],
    options: {
      numAgents: inputs.numAgents || 3,
      agentRoles: inputs.agentRoles || [],
      reasoningRounds: inputs.reasoningRounds || 3,
      confidenceThreshold: inputs.confidenceThreshold || 0.7,
      enableMcp: inputs.enableMcp || false,
      mcpServers: inputs.mcpServers || [],
      toolkits: inputs.toolkits || [],
      enableRetry: inputs.enableRetry !== false,
      enableCaching: inputs.enableCaching !== false,
      cacheTTL: inputs.cacheTTL || 3600000,
    },
  };
}

/**
 * Transform BubbleLab plugin output to OpenEvolve format
 */
export function transformBubbleLabToOpenEvolve(result: any, service: any): any {
  return {
    solution: result.result?.summary || result.result,
    confidence: result.statistics?.confidence || 0,
    reasoning_trace: result.result?.reasoning,
    agent_votes: result.metadata?.agentVotes,
    quality_metrics: service.analyzeExecutionPerformance(result),
    execution_id: result.executionId,
    status: result.status,
    timestamp: result.timestamp,
  };
}
```

### Step 4: Update OpenEvolve Plugin Registry

**File:** `OpenEvolve-Plugin/src/plugin.ts`

```typescript
// Import BubbleLab plugin types and utilities
import type {
  RomaExecutionMethod,
  RomaModuleType,
} from '../../../glue/adapters/roma/roma-bubblelab-plugin/src/types/plugin-types';

// Extend plugin capabilities
export const OpenEvolvePluginConfig = {
  capabilities: {
    roma: true,
    roma_advanced: true,  // NEW: Advanced ROMA features
    roma_mdap_maker: true,  // NEW: MDAP/MAKER support
    roma_mcp: true,  // NEW: MCP integration
  },
  dependencies: {
    // Explicitly declare BubbleLab plugin dependency
    'roma-bubblelab-plugin': '^1.0.0',
  },
};
```

### Step 5: Create Unified ROMA Service

**File:** `glue/adapters/roma/unified-roma-service.ts`

```typescript
/**
 * Unified ROMA Service
 *
 * Provides a single entry point for both OpenEvolve-Plugin
 * and BubbleLab plugin to access ROMA functionality.
 */

import { RomaService } from '../roma/roma-bubblelab-plugin/src/services/RomaService';
import { RomaClient } from '../roma/roma-bubblelab-plugin/src/services/RomaClient';
import type {
  RomaExecutionRequest,
  RomaExecutionResponse,
} from '../../schemas/roma-canonical';

export class UnifiedRomaService {
  private service: RomaService;

  constructor(config: any = {}) {
    const client = new RomaClient({
      serverUrl: config.serverUrl || process.env.ROMA_SERVER_URL || 'http://localhost:8000',
      apiKey: config.apiKey || process.env.ROMA_API_KEY,
      timeout: config.timeout || 30000,
    });

    this.service = new RomaService(client);
  }

  /**
   * Execute ROMA task (for BubbleLab plugin)
   */
  async executeTask(
    request: RomaExecutionRequest,
    context: any
  ): Promise<RomaExecutionResponse> {
    const result = await this.service.executeTaskWithCache(
      request.goal,
      {
        maxDepth: request.max_depth,
        executionMethod: request.execution_method,
        correlationId: request.correlation_id,
        ...request.metadata,
      },
      3,  // retries
      1000  // base delay
    );

    return {
      execution_id: result.executionId,
      status: result.status,
      initial_goal: result.goal,
      result: result.result,
      statistics: result.statistics,
      timestamp: result.timestamp,
      error: result.error,
    };
  }

  /**
   * Execute with OpenEvolve node inputs (for OpenEvolve-Plugin)
   */
  async executeForOpenEvolve(nodeInputs: any): Promise<any> {
    const transformed = transformOpenEvolveToBubbleLab(nodeInputs);

    const result = await this.service.executeTaskWithRetry(
      transformed.goal,
      {
        maxDepth: transformed.maxDepth,
        executionMethod: transformed.executionMethod,
        ...transformed.options,
      },
      3,
      1000
    );

    return transformBubbleLabToOpenEvolve(result, this.service);
  }

  /**
   * Get execution details
   */
  async getExecution(executionId: string): Promise<RomaExecutionResponse> {
    return await this.service.getExecution(executionId);
  }

  /**
   * Cancel execution
   */
  async cancelExecution(executionId: string): Promise<void> {
    await this.service.cancelExecution(executionId);
  }

  /**
   * Get execution plan with subtasks
   */
  async getExecutionPlan(executionId: string): Promise<any> {
    return await this.service.getExecutionPlan(executionId);
  }

  /**
   * Analyze execution performance
   */
  analyzePerformance(result: any): any {
    return this.service.analyzeExecutionPerformance(result);
  }

  /**
   * Get cache statistics
   */
  getCacheStatistics(): any {
    return this.service.getCacheStatistics();
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.service.clearCache();
  }
}

// Singleton instance
let unifiedRomaService: UnifiedRomaService | null = null;

export function getUnifiedRomaService(): UnifiedRomaService {
  if (!unifiedRomaService) {
    unifiedRomaService = new UnifiedRomaService();
  }
  return unifiedRomaService;
}

export function resetUnifiedRomaService(): void {
  unifiedRomaService = null;
}
```

### Step 6: Update Package Dependencies

**File:** `OpenEvolve-Plugin/package.json`

```json
{
  "dependencies": {
    "roma-bubblelab-plugin": "file:../../glue/adapters/roma/roma-bubblelab-plugin"
  }
}
```

### Step 7: Integration Tests

**File:** `OpenEvolve-Plugin/src/tests/integration/roma-unified.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { getUnifiedRomaService } from '../../../../glue/adapters/roma/unified-roma-service';

describe('Unified ROMA Integration', () => {
  it('should execute task from OpenEvolve inputs', async () => {
    const service = getUnifiedRomaService();

    const result = await service.executeForOpenEvolve({
      taskDescription: 'What is 2+2?',
      reasoningMode: 'collaborative',
      numAgents: 3,
      reasoningRounds: 2,
      confidenceThreshold: 0.7,
    });

    expect(result).toBeDefined();
    expect(result.execution_id).toBeDefined();
    expect(result.solution).toBeDefined();
  });

  it('should execute task from BubbleLab inputs', async () => {
    const service = getUnifiedRomaService();

    const result = await service.executeTask({
      goal: 'What is 2+2?',
      maxDepth: 1,
      executionMethod: 'roma',
    }, {
      correlationId: 'test-123',
      timestamp: new Date().toISOString(),
      sourceService: 'test',
    });

    expect(result.execution_id).toBeDefined();
    expect(result.status).toBeDefined();
  });
});
```

## Benefits of Unification

1. **Code Reuse:** OpenEvolve-Plugin leverages BubbleLab's robust service layer
2. **Consistent Behavior:** Both implementations use same retry/caching/validation
3. **Advanced Features:** OpenEvolve gets MDAP/MAKER, MCP, toolkits
4. **Single Source of Truth:** Unified service manages ROMA backend communication
5. **Easier Maintenance:** Bug fixes in one place benefit both implementations
6. **Better Testing:** Can test unified service once instead of twice

## Migration Checklist

- [ ] Create `glue/adapters/roma/shared-types.ts`
- [ ] Create `glue/adapters/roma/unified-roma-service.ts`
- [ ] Update `OpenEvolve-Plugin/src/nodes/ROMANode.ts` to use unified service
- [ ] Update `OpenEvolve-Plugin/src/schemas/roma.ts` with advanced fields
- [ ] Update `OpenEvolve-Plugin/package.json` with BubbleLab dependency
- [ ] Create integration tests
- [ ] Update documentation
- [ ] Test both OpenEvolve and BubbleLab flows

**Estimated Time:** 4-6 hours

---

**Created:** 2026-02-22
**Part of:** ROMA Integration Completion Task #16
