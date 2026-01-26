# BubbleLab Test Templates

**Purpose:** Ready-to-use test templates for all bubble types
**Framework:** Vitest
**Language:** TypeScript

---

## Table of Contents

1. [Service Bubble Template](#service-bubble-template)
2. [Tool Bubble Template](#tool-bubble-template)
3. [Workflow Bubble Template](#workflow-bubble-template)
4. [Test Helper Templates](#test-helper-templates)
5. [Mock Templates](#mock-templates)

---

## Service Bubble Template

### File Structure

```
service-bubble/
├── {bubble-name}.ts                    # Implementation
├── {bubble-name}.test.ts               # Unit tests
├── {bubble-name}.integration.test.ts   # Integration tests
└── __tests__/
    ├── mocks.ts                        # Bubble-specific mocks
    └── fixtures.ts                     # Bubble-specific fixtures
```

### Unit Test Template

```typescript
/**
 * {Bubble Name} Unit Tests
 * File: service-bubble/{bubble-name}.test.ts
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { {BubbleName} } from './{bubble-name}.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('{BubbleName}', () => {
  let mockFetch: any;

  beforeEach(() => {
    // Mock fetch API for HTTP requests
    mockFetch = vi.fn();
    global.fetch = mockFetch;

    // Clear any mock state
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Construction', () => {
    it('should create instance with valid parameters', () => {
      const bubble = new {BubbleName}({
        // Required params
        operation: 'operationName',
        // ... other params
      });

      expect(bubble).toBeDefined();
      expect(bubble.params.operation).toBe('operationName');
    });

    it('should validate required parameters', () => {
      expect(() => {
        new {BubbleName}({
          // Missing required params
        } as any);
      }).toThrow();
    });

    it('should set default values for optional parameters', () => {
      const bubble = new {BubbleName}({
        operation: 'operationName',
      });

      expect(bubble.params.optionalParam).toBeDefined();
    });
  });

  describe('Authentication', () => {
    it('should use provided credentials', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        credentials: {
          [CredentialType.{CREDENTIAL_TYPE}]: JSON.stringify({
            token: 'test-token',
          }),
        },
      });

      await bubble.act();

      // Verify credentials were used
      expect(mockFetch).toHaveBeenCalled();
      const authHeader = mockFetch.mock.calls[0][1].headers['Authorization'];
      expect(authHeader).toContain('test-token');
    });

    it('should handle missing credentials gracefully', async () => {
      const bubble = new {BubbleName}({
        operation: 'operationName',
        credentials: undefined,
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('credentials');
    });

    it('should test credential validity', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ valid: true }),
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        credentials: {
          [CredentialType.{CREDENTIAL_TYPE}]: 'valid-credential',
        },
      });

      const isValid = await bubble.testCredential();

      expect(isValid).toBe(true);
    });
  });

  describe('Operation Execution', () => {
    it('should execute operation successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ data: 'test' }),
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        // ... params
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should handle operation errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({ error: 'Internal Server Error' }),
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const bubble = new {BubbleName}({
        operation: 'operationName',
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });
  });

  describe('Input Validation', () => {
    it('should validate URL format', () => {
      expect(() => {
        new {BubbleName}({
          operation: 'operationName',
          url: 'invalid-url',
        } as any);
      }).toThrow();
    });

    it('should validate parameter ranges', () => {
      expect(() => {
        new {BubbleName}({
          operation: 'operationName',
          timeout: -1, // Invalid
        } as any);
      }).toThrow();
    });

    it('should validate enum values', () => {
      expect(() => {
        new {BubbleName}({
          operation: 'operationName',
          method: 'INVALID_METHOD' as any,
        });
      }).toThrow();
    });
  });

  describe('Error Handling', () => {
    it('should retry on retryable errors', async () => {
      let attemptCount = 0;
      mockFetch.mockImplementation(() => {
        attemptCount++;
        if (attemptCount < 3) {
          return Promise.resolve({
            ok: false,
            status: 503,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        });
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        retryEnabled: true,
        maxRetries: 3,
      });

      const result = await bubble.act();

      expect(attemptCount).toBe(3);
      expect(result.success).toBe(true);
    });

    it('should not retry on non-retryable errors', async () => {
      let attemptCount = 0;
      mockFetch.mockImplementation(() => {
        attemptCount++;
        return Promise.resolve({
          ok: false,
          status: 404,
        });
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        retryEnabled: true,
        maxRetries: 3,
      });

      await bubble.act();

      expect(attemptCount).toBe(1); // No retries
    });

    it('should timeout after specified duration', async () => {
      mockFetch.mockImplementation(() =>
        new Promise((resolve) => setTimeout(resolve, 10000))
      );

      const bubble = new {BubbleName}({
        operation: 'operationName',
        timeout: 100,
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });
  });

  describe('Response Handling', () => {
    it('should parse JSON responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'test', count: 42 }),
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
      });

      const result = await bubble.act();

      expect(result.data).toEqual({ data: 'test', count: 42 });
    });

    it('should parse text responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        text: async () => 'plain text',
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        responseType: 'text',
      });

      const result = await bubble.act();

      expect(result.data).toBe('plain text');
    });

    it('should handle malformed responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
      });

      const result = await bubble.act();

      expect(result.success).toBe(true); // Should still succeed
      expect(result.data).toBeDefined();
    });
  });

  describe('Metrics', () => {
    it('should track response time', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
      });

      const result = await bubble.act();

      expect(result.metrics.responseTime).toBeGreaterThan(0);
    });

    it('should track retry count', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
      });

      const bubble = new {BubbleName}({
        operation: 'operationName',
        retryEnabled: true,
        maxRetries: 3,
      });

      const result = await bubble.act();

      expect(result.metrics.retryCount).toBe(3);
      expect(result.metrics.totalAttempts).toBe(4);
    });
  });
});
```

### Integration Test Template

```typescript
/**
 * {Bubble Name} Integration Tests
 * File: service-bubble/{bubble-name}.integration.test.ts
 */

import { describe, it, expect } from 'vitest';
import { {BubbleName} } from './{bubble-name}.js';

describe('{BubbleName} - Integration Tests', () => {
  // Skip tests if credentials not available
  const testCredentials = process.env.TEST_{CREDENTIAL_NAME}_CRED;

  if (!testCredentials) {
    console.warn(`Skipping integration tests: TEST_{CREDENTIAL_NAME}_CRED not set`);
  }

  describe('Real API Integration', () => {
    it.skipIf(!testCredentials)('should connect to real service', async () => {
      const bubble = new {BubbleName}({
        operation: 'testOperation',
        credentials: {
          [CredentialType.{CREDENTIAL_TYPE}]: testCredentials!,
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
    });

    it.skipIf(!testCredentials)('should handle real data', async () => {
      const bubble = new {BubbleName}({
        operation: 'operationName',
        // Real parameters
        credentials: {
          [CredentialType.{CREDENTIAL_TYPE}]: testCredentials!,
        },
      });

      const result = await bubble.act();

      expect(result.data).toBeDefined();
      expect(result.success).toBe(true);
    });
  });

  describe('Error Scenarios', () => {
    it.skipIf(!testCredentials)('should handle invalid credentials', async () => {
      const bubble = new {BubbleName}({
        operation: 'operationName',
        credentials: {
          [CredentialType.{CREDENTIAL_TYPE}]: 'invalid-credential',
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it.skipIf(!testCredentials)('should handle rate limiting', async () => {
      // Make multiple rapid requests to trigger rate limit
      const requests = Array(10).fill(null).map(() =>
        new {BubbleName}({
          operation: 'operationName',
          credentials: {
            [CredentialType.{CREDENTIAL_TYPE}]: testCredentials!,
          },
        }).act()
      );

      const results = await Promise.all(requests);

      // At least one should hit rate limit
      const rateLimited = results.some((r) => r.error?.includes('rate'));

      if (rateLimited) {
        console.log('Rate limiting detected (expected behavior)');
      }
    });
  });

  describe('Performance', () => {
    it.skipIf(!testCredentials)('should complete within reasonable time', async () => {
      const startTime = Date.now();

      const bubble = new {BubbleName}({
        operation: 'operationName',
        credentials: {
          [CredentialType.{CREDENTIAL_TYPE}]: testCredentials!,
        },
      });

      await bubble.act();

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(5000); // 5 seconds max
    });
  });
});
```

---

## Tool Bubble Template

### Unit Test Template

```typescript
/**
 * {Tool Name} Unit Tests
 * File: tool-bubble/{tool-name}.test.ts
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { {ToolName} } from './{tool-name}.js';

describe('{ToolName}', () => {
  let testData: any;

  beforeEach(() => {
    // Set up test data
    testData = {
      // Test data setup
    };
  });

  describe('Construction', () => {
    it('should create instance with valid parameters', () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        // Required params
      });

      expect(tool).toBeDefined();
      expect(tool.params.operation).toBe('operationName');
    });

    it('should validate required parameters', () => {
      expect(() => {
        new {ToolName}({
          // Missing required params
        } as any);
      }).toThrow();
    });

    it('should apply default values', () => {
      const tool = new {ToolName}({
        operation: 'operationName',
      });

      expect(tool.params.optionalParam).toBeDefined();
    });
  });

  describe('Operation Execution', () => {
    it('should execute operation successfully', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: testData,
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should handle missing input data', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: undefined,
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('input');
    });

    it('should handle invalid data format', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: 'invalid-data',
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
    });
  });

  describe('Data Processing', () => {
    it('should process data correctly', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: testData,
        processingOptions: {
          // Options
        },
      });

      const result = await tool.act();

      expect(result.data).toBeDefined();
      expect(result.processedCount).toBeGreaterThan(0);
    });

    it('should handle empty data', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: [],
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
      expect(result.data).toEqual([]);
    });

    it('should handle large datasets efficiently', async () => {
      const largeData = Array(10000).fill({ value: 1 });

      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: largeData,
      });

      const startTime = Date.now();
      const result = await tool.act();
      const duration = Date.now() - startTime;

      expect(result.success).toBe(true);
      expect(duration).toBeLessThan(5000); // Should complete in < 5s
    });
  });

  describe('Validation', () => {
    it('should validate input schema', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: testData,
        validationSchema: {
          field: 'string',
        },
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
      expect(result.validationErrors).toBeUndefined();
    });

    it('should detect validation errors', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: {
          field: 123, // Should be string
        },
        validationSchema: {
          field: 'string',
        },
      });

      const result = await tool.act();

      expect(result.validationErrors).toBeDefined();
      expect(result.validationErrors?.length).toBeGreaterThan(0);
    });

    it('should report validation statistics', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: [
          { field: 'valid' },
          { field: 123 }, // Invalid
          { field: 'also valid' },
        ],
        validationSchema: {
          field: 'string',
        },
      });

      const result = await tool.act();

      expect(result.statistics).toBeDefined();
      expect(result.statistics?.validRows).toBe(2);
      expect(result.statistics?.invalidRows).toBe(1);
    });
  });

  describe('Transformation', () => {
    it('should apply transformations correctly', async () => {
      const tool = new {ToolName}({
        operation: 'transform',
        inputData: [{ value: 10 }],
        transformations: [
          {
            field: 'value',
            operation: 'multiply',
            operand: 2,
          },
        ],
      });

      const result = await tool.act();

      expect(result.data[0].value).toBe(20);
    });

    it('should handle multiple transformations', async () => {
      const tool = new {ToolName}({
        operation: 'transform',
        inputData: [{ value: 10 }],
        transformations: [
          { field: 'value', operation: 'multiply', operand: 2 },
          { field: 'value', operation: 'add', operand: 5 },
        ],
      });

      const result = await tool.act();

      expect(result.data[0].value).toBe(25); // (10 * 2) + 5
    });

    it('should skip transformations for non-existent fields', async () => {
      const tool = new {ToolName}({
        operation: 'transform',
        inputData: [{ otherField: 'value' }],
        transformations: [
          { field: 'nonExistent', operation: 'upper' },
        ],
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
    });
  });

  describe('Filtering', () => {
    it('should filter data correctly', async () => {
      const tool = new {ToolName}({
        operation: 'filter',
        inputData: [
          { value: 10 },
          { value: 20 },
          { value: 30 },
        ],
        filters: [
          {
            field: 'value',
            operator: 'gt',
            value: 15,
          },
        ],
      });

      const result = await tool.act();

      expect(result.data).toHaveLength(2);
      expect(result.data.every((item: any) => item.value > 15)).toBe(true);
    });

    it('should handle multiple filters', async () => {
      const tool = new {ToolName}({
        operation: 'filter',
        inputData: [
          { value: 10, category: 'A' },
          { value: 20, category: 'B' },
          { value: 30, category: 'A' },
        ],
        filters: [
          { field: 'value', operator: 'gt', value: 15 },
          { field: 'category', operator: 'equals', value: 'A' },
        ],
      });

      const result = await tool.act();

      expect(result.data).toHaveLength(1);
      expect(result.data[0].value).toBe(30);
    });
  });

  describe('Aggregation', () => {
    it('should aggregate data correctly', async () => {
      const tool = new {ToolName}({
        operation: 'aggregate',
        inputData: [
          { category: 'A', value: 10 },
          { category: 'A', value: 20 },
          { category: 'B', value: 30 },
        ],
        groupBy: ['category'],
        aggregations: [
          { field: 'value', operation: 'sum' },
        ],
      });

      const result = await tool.act();

      expect(result.data).toHaveLength(2);
      expect(result.data[0]).toMatchObject({
        category: 'A',
        value_sum: 30,
      });
    });

    it('should handle multiple aggregations', async () => {
      const tool = new {ToolName}({
        operation: 'aggregate',
        inputData: [
          { category: 'A', value: 10 },
          { category: 'A', value: 20 },
        ],
        groupBy: ['category'],
        aggregations: [
          { field: 'value', operation: 'sum', alias: 'total' },
          { field: 'value', operation: 'avg', alias: 'average' },
          { field: 'value', operation: 'count', alias: 'count' },
        ],
      });

      const result = await tool.act();

      expect(result.data[0].total).toBe(30);
      expect(result.data[0].average).toBe(15);
      expect(result.data[0].count).toBe(2);
    });
  });

  describe('Error Handling', () => {
    it('should handle processing errors gracefully', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: ['valid', 'invalid', 'also valid'],
      });

      const result = await tool.act();

      expect(result.success).toBe(true);
      expect(result.errors).toBeDefined();
      expect(result.processedCount).toBeGreaterThan(0);
    });

    it('should provide meaningful error messages', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: null,
      });

      const result = await tool.act();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error.length).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    it('should track processing time', async () => {
      const tool = new {ToolName}({
        operation: 'operationName',
        inputData: testData,
      });

      const result = await tool.act();

      expect(result.statistics?.processingTime).toBeGreaterThan(0);
    });

    it('should handle concurrent operations', async () => {
      const tools = Array(10).fill(null).map(() =>
        new {ToolName}({
          operation: 'operationName',
          inputData: testData,
        })
      );

      const startTime = Date.now();
      await Promise.all(tools.map((t) => t.act()));
      const duration = Date.now() - startTime;

      // Should complete faster than sequential execution
      expect(duration).toBeLessThan(1000);
    });
  });
});
```

---

## Workflow Bubble Template

### Unit Test Template

```typescript
/**
 * {Workflow Name} Unit Tests
 * File: workflow-bubble/{workflow-name}.test.ts
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { {WorkflowName} } from './{workflow-name}.js';

describe('{WorkflowName}', () => {
  let mockDependencies: any;

  beforeEach(() => {
    // Mock all external dependencies
    mockDependencies = {
      dataSource: vi.fn(),
      transformer: vi.fn(),
      dataSink: vi.fn(),
    };

    vi.clearAllMocks();
  });

  describe('Workflow Construction', () => {
    it('should create workflow with valid config', () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ test: 'data' }],
        },
        transformations: [],
        destination: {
          type: 'memory',
        },
      });

      expect(workflow).toBeDefined();
      expect(workflow.config.source).toBeDefined();
      expect(workflow.config.destination).toBeDefined();
    });

    it('should validate required configuration', () => {
      expect(() => {
        new {WorkflowName}({
          // Missing required config
        } as any);
      }).toThrow();
    });

    it('should apply default configuration', () => {
      const workflow = new {WorkflowName}({
        source: { type: 'memory', data: [] },
        destination: { type: 'memory' },
      });

      expect(workflow.config.options).toBeDefined();
    });
  });

  describe('Workflow Execution', () => {
    it('should execute workflow successfully', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 1 }],
        },
        transformations: [],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.success).toBe(true);
      expect(result.stats).toBeDefined();
    });

    it('should extract data from source', async () => {
      const testData = [{ id: 1 }, { id: 2 }];
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: testData,
        },
        transformations: [],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.extracted).toEqual(testData);
      expect(result.stats.inputRows).toBe(2);
    });

    it('should apply transformations', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 10 }],
        },
        transformations: [
          {
            type: 'map',
            expression: 'item.value * 2',
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.transformed[0].value).toBe(20);
    });

    it('should load data to destination', async () => {
      const mockSink = vi.fn().mockResolvedValue({ success: true });
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 1 }],
        },
        transformations: [],
        destination: {
          type: 'custom',
          client: mockSink,
        },
      });

      const result = await workflow.execute();

      expect(result.loaded).toBe(true);
      expect(mockSink).toHaveBeenCalled();
    });
  });

  describe('Step Execution', () => {
    it('should execute steps in order', async () => {
      const executionOrder: string[] = [];

      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [],
        },
        transformations: [
          {
            type: 'custom',
            fn: async () => {
              executionOrder.push('step1');
            },
          },
          {
            type: 'custom',
            fn: async () => {
              executionOrder.push('step2');
            },
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      await workflow.execute();

      expect(executionOrder).toEqual(['step1', 'step2']);
    });

    it('should handle step failures', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [],
        },
        transformations: [
          {
            type: 'custom',
            fn: async () => {
              throw new Error('Step failed');
            },
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Step failed');
    });

    it('should skip optional steps on failure', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [],
        },
        transformations: [
          {
            type: 'custom',
            fn: async () => {
              throw new Error('Failed');
            },
            optional: true,
          },
          {
            type: 'custom',
            fn: async () => {
              return { success: true };
            },
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.success).toBe(true);
      expect(result.stats.skippedSteps).toContain(0);
    });
  });

  describe('Data Flow', () => {
    it('should pass data between steps', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 1 }],
        },
        transformations: [
          {
            type: 'map',
            expression: 'item.value + 1',
          },
          {
            type: 'map',
            expression: 'item.value * 2',
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      // (1 + 1) * 2 = 4
      expect(result.transformed[0].value).toBe(4);
    });

    it('should filter data', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [
            { value: 1 },
            { value: 2 },
            { value: 3 },
          ],
        },
        transformations: [
          {
            type: 'filter',
            condition: 'item.value > 1',
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.transformed).toHaveLength(2);
      expect(result.transformed.every((item: any) => item.value > 1)).toBe(true);
    });

    it('should aggregate data', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [
            { category: 'A', value: 10 },
            { category: 'A', value: 20 },
            { category: 'B', value: 30 },
          ],
        },
        transformations: [
          {
            type: 'aggregate',
            groupBy: ['category'],
            aggregations: [
              { field: 'value', operation: 'sum' },
            ],
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.transformed).toHaveLength(2);
      expect(result.transformed[0]).toMatchObject({
        category: 'A',
        value_sum: 30,
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle source errors', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'http',
          url: 'https://nonexistent.example.com',
        },
        transformations: [],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should handle transformation errors', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 'not a number' }],
        },
        transformations: [
          {
            type: 'map',
            expression: 'item.value * 2', // Will fail
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.success).toBe(false);
    });

    it('should handle destination errors', async () => {
      const mockSink = vi.fn().mockRejectedValue(new Error('Destination error'));

      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 1 }],
        },
        transformations: [],
        destination: {
          type: 'custom',
          client: mockSink,
        },
      });

      const result = await workflow.execute();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Destination error');
    });
  });

  describe('Statistics', () => {
    it('should track execution statistics', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: Array(100).fill({ value: 1 }),
        },
        transformations: [],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.stats).toBeDefined();
      expect(result.stats.inputRows).toBe(100);
      expect(result.stats.outputRows).toBe(100);
      expect(result.stats.duration).toBeGreaterThan(0);
    });

    it('should track step timings', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 1 }],
        },
        transformations: [
          {
            type: 'custom',
            fn: async () => {
              await new Promise((resolve) => setTimeout(resolve, 100));
              return { value: 1 };
            },
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(result.stats.stepTimings).toBeDefined();
      expect(result.stats.stepTimings[0]).toBeGreaterThan(0);
    });
  });

  describe('Workflow State', () => {
    it('should maintain workflow state', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: [{ value: 1 }],
        },
        transformations: [],
        destination: {
          type: 'memory',
        },
      });

      const result = await workflow.execute();

      expect(workflow.getStatus()).toBeDefined();
      expect(workflow.getStatus()).toBe('completed');
    });

    it('should support workflow cancellation', async () => {
      const workflow = new {WorkflowName}({
        source: {
          type: 'memory',
          data: Array(10000).fill({ value: 1 }),
        },
        transformations: [
          {
            type: 'custom',
            fn: async () => {
              await new Promise((resolve) => setTimeout(resolve, 1000));
              return { value: 1 };
            },
          },
        ],
        destination: {
          type: 'memory',
        },
      });

      // Start execution
      const execution = workflow.execute();

      // Cancel after 100ms
      setTimeout(() => workflow.cancel(), 100);

      const result = await execution;

      expect(result.success).toBe(false);
      expect(result.error).toContain('cancelled');
    });
  });
});
```

---

## Test Helper Templates

### Mock Factory Template

```typescript
/**
 * Mock Factory Template
 * File: __tests__/helpers/mock-factory.ts
 */

import { vi } from 'vitest';

export class MockFactory {
  /**
   * Create mock HTTP response
   */
  static mockResponse(overrides: Partial<Response> = {}): Response {
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers(),
      json: async () => ({}),
      text: async () => '',
      blob: async () => new Blob(),
      arrayBuffer: async () => new ArrayBuffer(0),
      ...overrides,
    } as Response;
  }

  /**
   * Create mock error response
   */
  static mockErrorResponse(status: number, error: string): Response {
    return {
      ok: false,
      status,
      statusText: error,
      headers: new Headers(),
      json: async () => ({ error }),
    } as Response;
  }

  /**
   * Create mock credentials
   */
  static mockCredentials(type: string, data: any): Record<string, string> {
    return {
      [type]: typeof data === 'string' ? data : JSON.stringify(data),
    };
  }

  /**
   * Create mock database connection
   */
  static mockDatabase() {
    return {
      query: vi.fn().mockResolvedValue({ rows: [] }),
      connect: vi.fn().mockResolvedValue(undefined),
      disconnect: vi.fn().mockResolvedValue(undefined),
      transaction: vi.fn().mockResolvedValue(undefined),
    };
  }

  /**
   * Create mock file system
   */
  static mockFileSystem() {
    return {
      readFile: vi.fn().mockResolvedValue('file content'),
      writeFile: vi.fn().mockResolvedValue(undefined),
      exists: vi.fn().mockResolvedValue(true),
      delete: vi.fn().mockResolvedValue(undefined),
    };
  }
}
```

### Test Data Template

```typescript
/**
 * Test Data Template
 * File: __tests__/helpers/test-data.ts
 */

export const TestData = {
  // HTTP test data
  http: {
    validUrls: [
      'https://api.example.com/endpoint',
      'http://localhost:8080/api/v1/resource',
    ],
    invalidUrls: [
      'not-a-url',
      'http://',
      'javascript:alert(1)',
    ],
  },

  // CSV test data
  csv: {
    simple: 'name,age\nJohn,30\nJane,25',
    withQuotes: 'name,description\n"John Doe","Person, with comma"',
    malformed: 'name,age\nJohn,30\nJane,',
  },

  // Email test data
  emails: {
    valid: ['test@example.com', 'user.name@example.co.uk'],
    invalid: ['not-an-email', '@example.com', 'user@'],
  },

  // Generate test data
  generateArray: (count: number, factory: (i: number) => any) => {
    return Array(count).fill(null).map((_, i) => factory(i));
  },

  generateCSV: (rows: number, columns: string[]) => {
    const header = columns.join(',');
    const data = TestData.generateArray(rows, (i) =>
      columns.map((col) => `${col}_${i}`).join(',')
    ).join('\n');
    return `${header}\n${data}`;
  },
};
```

### Custom Assertions Template

```typescript
/**
 * Custom Assertions Template
 * File: __tests__/helpers/assertion-helpers.ts
 */

import { expect } from 'vitest';

export class CustomAssertions {
  /**
   * Assert successful bubble result
   */
  static assertSuccess(result: any) {
    expect(result).toBeDefined();
    expect(result.success).toBe(true);
    expect(result.error).toBeUndefined();
  }

  /**
   * Assert failed bubble result
   */
  static assertFailure(result: any, expectedError?: string) {
    expect(result).toBeDefined();
    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();

    if (expectedError) {
      expect(result.error).toContain(expectedError);
    }
  }

  /**
   * Assert metrics are present
   */
  static assertMetrics(result: any, requiredKeys: string[] = []) {
    expect(result.metrics).toBeDefined();

    for (const key of requiredKeys) {
      expect(result.metrics[key]).toBeDefined();
    }
  }

  /**
   * Assert validation passed
   */
  static assertValidation(result: any) {
    expect(result.success).toBe(true);
    expect(result.validationErrors).toBeUndefined();
  }

  /**
   * Assert validation failed
   */
  static assertValidationFailed(result: any, errorCount?: number) {
    expect(result.validationErrors).toBeDefined();
    expect(result.validationErrors.length).toBeGreaterThan(0);

    if (errorCount !== undefined) {
      expect(result.validationErrors.length).toBe(errorCount);
    }
  }

  /**
   * Assert data transformation
   */
  static assertTransformed(
    result: any,
    input: any[],
    transformer: (item: any) => any
  ) {
    const expected = input.map(transformer);
    expect(result.data).toEqual(expected);
  }

  /**
   * Assert data filtering
   */
  static assertFiltered(result: any, predicate: (item: any) => boolean) {
    expect(result.data.every(predicate)).toBe(true);
  }

  /**
   * Assert performance
   */
  static assertPerformance(result: any, maxDuration: number) {
    expect(result.stats?.duration).toBeLessThan(maxDuration);
  }

  /**
   * Assert statistics
   */
  static assertStatistics(result: any, expectedStats: Record<string, number>) {
    expect(result.stats).toBeDefined();

    for (const [key, value] of Object.entries(expectedStats)) {
      expect(result.stats[key]).toBe(value);
    }
  }
}
```

---

## Usage Examples

### Creating a New Test File

```bash
# 1. Navigate to bubble directory
cd bubble-core/src/bubbles/service-bubble

# 2. Create test file using template
cp {template-name}.test.ts my-bubble.test.ts

# 3. Replace placeholders
# - {BubbleName} → Actual bubble class name
# - operationName → Actual operation names
# - testData → Actual test data

# 4. Run tests
pnpm test my-bubble
```

### Running Tests

```bash
# Run all tests
pnpm test

# Run specific test file
pnpm test http-bubble

# Run with coverage
pnpm test:coverage

# Watch mode
pnpm test:watch
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: pnpm/action-setup@v2
      - uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install dependencies
        run: pnpm install

      - name: Run tests
        run: pnpm test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Best Practices

1. **Test Structure:** Use AAA pattern (Arrange, Act, Assert)
2. **Test Isolation:** Each test should be independent
3. **Mocking:** Mock external dependencies (APIs, databases, files)
4. **Data:** Use realistic test data fixtures
5. **Errors:** Test both success and error scenarios
6. **Performance:** Include performance tests for critical paths
7. **Documentation:** Add clear comments explaining complex test logic
8. **Coverage:** Aim for 80%+ coverage, 100% for critical paths

---

**Last Updated:** 2025-01-18
**Framework:** Vitest
**Templates:** Service, Tool, Workflow bubbles + Helpers
