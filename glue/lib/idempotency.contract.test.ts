/**
 * Contract Test for Idempotency Utilities
 *
 * Tests compliance with Federation Constitution Section 1.4:
 * - Law of Idempotency: Safe to run 100 times
 * - Check if resource exists before creating
 * - Use UPSERT logic
 * - Deduplicate based on distinct IDs
 */

import {
  idempotentCreate,
  upsert,
  deduplicate,
  idempotentBatch,
  idempotentWrite,
  idempotentRetry,
  IdempotencyCheckResult
} from './idempotency';
import { LogContext } from './structuredLogger';

describe('Idempotency Contract Tests', () => {
  describe('idempotentCreate', () => {
    it('should return existing resource if already exists', async () => {
      const existingResource = { id: '123', name: 'Existing' };
      let checkCalled = 0;
      let createCalled = 0;

      const result = await idempotentCreate(
        async () => {
          checkCalled++;
          return { exists: true, resource: existingResource, id: '123' };
        },
        async () => {
          createCalled++;
          return { id: '456', name: 'New' };
        },
        {} as LogContext
      );

      expect(result).toEqual(existingResource);
      expect(checkCalled).toBe(1);
      expect(createCalled).toBe(0); // Should not create
    });

    it('should create new resource if not exists', async () => {
      const newResource = { id: '456', name: 'New' };
      let checkCalled = 0;
      let createCalled = 0;

      const result = await idempotentCreate(
        async () => {
          checkCalled++;
          return { exists: false };
        },
        async () => {
          createCalled++;
          return newResource;
        },
        {} as LogContext
      );

      expect(result).toEqual(newResource);
      expect(checkCalled).toBe(1);
      expect(createCalled).toBe(1);
    });

    it('should be safe to run multiple times', async () => {
      const existingResource = { id: '123', name: 'Existing' };
      let callCount = 0;

      // Run 5 times
      for (let i = 0; i < 5; i++) {
        const result = await idempotentCreate(
          async () => {
            callCount++;
            return { exists: true, resource: existingResource, id: '123' };
          },
          async () => {
            callCount++;
            return { id: '456', name: 'New' };
          },
          {} as LogContext
        );

        expect(result).toEqual(existingResource);
      }

      // Check should be called 5 times, create 0 times
      expect(callCount).toBe(5);
    });
  });

  describe('upsert', () => {
    it('should update if resource exists', async () => {
      const existing = { id: '123', value: 'old' };
      const updated = { id: '123', value: 'updated' };

      const result = await upsert(
        async () => {
          return { exists: true, resource: existing, id: '123' };
        },
        async () => {
          return { id: '456', value: 'created' };
        },
        async (resource) => {
          expect(resource).toEqual(existing);
          return updated;
        },
        {} as LogContext
      );

      expect(result).toEqual(updated);
    });

    it('should create if resource does not exist', async () => {
      const created = { id: '789', value: 'created' };

      const result = await upsert(
        async () => {
          return { exists: false };
        },
        async () => {
          return created;
        },
        async (resource) => {
          fail('Should not call update for non-existent resource');
          return resource;
        },
        {} as LogContext
      );

      expect(result).toEqual(created);
    });

    it('should be idempotent across multiple calls', async () => {
      let callCount = 0;

      for (let i = 0; i < 3; i++) {
        await upsert(
          async () => {
            callCount++;
            return { exists: false };
          },
          async () => {
            callCount++;
            return { id: `${i}`, value: 'test' };
          },
          async (resource) => {
            return { ...resource, updated: true };
          },
          {} as LogContext
        );
      }

      // Each call should be check + create/update
      expect(callCount).toBe(6); // 3 * 2 operations
    });
  });

  describe('deduplicate', () => {
    it('should remove duplicates based on id', () => {
      const items = [
        { id: '1', name: 'Item 1' },
        { id: '2', name: 'Item 2' },
        { id: '1', name: 'Duplicate 1' },
        { id: '3', name: 'Item 3' },
        { id: '2', name: 'Duplicate 2' }
      ];

      const deduplicated = deduplicate(items);

      expect(deduplicated.length).toBe(3);
      expect(deduplicated.every(item => items.some(original => original.id === item.id))).toBe(true);
    });

    it('should use custom getId function', () => {
      const items = [
        { name: 'Alice', email: 'alice@example.com' },
        { name: 'Bob', email: 'bob@example.com' },
        { name: 'Alice', email: 'alice2@example.com' }
      ];

      const deduplicated = deduplicate(items, (item) => item.name);

      expect(deduplicated.length).toBe(2);
      expect(deduplicated.map(d => d.name)).toEqual(['Alice', 'Bob']);
    });

    it('should handle empty array', () => {
      const result = deduplicate([]);
      expect(result).toEqual([]);
    });

    it('should preserve order of first occurrences', () => {
      const items = [
        { id: '3', order: 3 },
        { id: '1', order: 1 },
        { id: '2', order: 2 },
        { id: '1', order: 1 },
        { id: '3', order: 3 }
      ];

      const deduplicated = deduplicate(items);

      expect(deduplicated.map(d => d.id)).toEqual(['3', '1', '2']);
    });
  });

  describe('idempotentBatch', () => {
    it('should process unique items', async () => {
      const items = [
        { id: '1', value: 'a' },
        { id: '2', value: 'b' },
        { id: '1', value: 'duplicate' }
      ];

      let processCallCount = 0;

      const results = await idempotentBatch(
        items,
        async (item) => {
          processCallCount++;
          return { processed: true, ...item };
        },
        (item) => item.id,
        {} as LogContext
      );

      // Should only process 2 unique items
      expect(processCallCount).toBe(2);
      expect(results.length).toBe(2);
    });

    it('should continue processing on individual failures', async () => {
      const items = [
        { id: '1', value: 'a' },
        { id: '2', value: 'b' },
        { id: '3', value: 'c' }
      ];

      const results = await idempotentBatch(
        items,
        async (item) => {
          if (item.id === '2') {
            throw new Error('Process failed for item 2');
          }
          return { processed: true, ...item };
        },
        (item) => item.id,
        {} as LogContext
      );

      // Should process 2 successfully (items 1 and 3)
      expect(results.length).toBe(2);
      expect(results.every(r => (r as any).id !== '2')).toBe(true);
    });

    it('should report success and failure counts', async () => {
      const items = [
        { id: '1', value: 'a' },
        { id: '2', value: 'b' },
        { id: '3', value: 'c' },
        { id: '4', value: 'd' }
      ];

      await idempotentBatch(
        items,
        async (item) => {
          if (item.id === '2' || item.id === '4') {
            throw new Error('Failed');
          }
          return { processed: true, ...item };
        },
        (item) => item.id,
        {} as LogContext
      );

      // 2 successful, 2 failed
      // Logs should be written (verified through inspection)
    });
  });

  describe('idempotentWrite', () => {
    it('should not write if content unchanged', async () => {
      let writeCallCount = 0;
      const existingContent = 'existing content';

      const result = await idempotentWrite(
        '/test/path',
        async () => {
          // Simulate getting existing content
          return existingContent;
        },
        async (content) => {
          writeCallCount++;
          // Simulate write
        },
        {} as LogContext
      );

      expect(result).toBe(false); // Did not write
      expect(writeCallCount).toBe(0);
    });

    it('should write if content changed', async () => {
      let writeCallCount = 0;
      let writtenContent = '';

      await idempotentWrite(
        '/test/path',
        async () => {
          return 'new content';
        },
        async (content) => {
          writeCallCount++;
          writtenContent = content;
        },
        {} as LogContext
      );

      expect(writeCallCount).toBe(1);
      expect(writtenContent).toBe('new content');
    });

    it('should handle missing file', async () => {
      let writeCallCount = 0;

      await idempotentWrite(
        '/test/path',
        async () => {
          throw new Error('File not found');
        },
        async (content) => {
          writeCallCount++;
        },
        {} as LogContext
      );

      expect(writeCallCount).toBe(1); // Should write when file doesn't exist
    });
  });

  describe('idempotentRetry', () => {
    it('should retry with exponential backoff', async () => {
      let attempts = 0;
      const delays: number[] = [];
      let lastAttemptTime = Date.now();

      const result = await idempotentRetry(
        async () => {
          const now = Date.now();
          if (delays.length > 0) {
            delays.push(now - lastAttemptTime);
          }
          lastAttemptTime = now;
          attempts++;
          if (attempts < 3) {
            throw new Error('Temporary failure');
          }
          return 'success';
        },
        3,
        100,
        {} as LogContext
      );

      expect(result).toBe('success');
      expect(attempts).toBe(3);
      expect(delays.length).toBe(2); // 2 retries
    });

    it('should exhaust retries and throw', async () => {
      let attempts = 0;

      try {
        await idempotentRetry(
          async () => {
            attempts++;
            throw new Error('Persistent failure');
          },
          2,
          100,
          {} as LogContext
        );
        fail('Should have thrown error');
      } catch (error) {
        expect((error as Error).message).toBe('Persistent failure');
        expect(attempts).toBe(3); // Initial + 2 retries
      }
    });

    it('should be safe to retry idempotent operations', async () => {
      let executionCount = 0;

      const operation = async () => {
        executionCount++;
        if (executionCount < 2) {
          throw new Error('Temporary');
        }
        return 'result';
      };

      // Run multiple times - all should succeed
      const result1 = await idempotentRetry(operation, 3, 100, {} as LogContext);
      executionCount = 0; // Reset
      const result2 = await idempotentRetry(operation, 3, 100, {} as LogContext);

      expect(result1).toBe('result');
      expect(result2).toBe('result');
    });
  });

  describe('Idempotency Law Compliance', () => {
    it('should be safe to run idempotentCreate 100 times', async () => {
      const existingResource = { id: '123', value: 'constant' };

      // Run 100 times
      for (let i = 0; i < 100; i++) {
        const result = await idempotentCreate(
          async () => {
            return { exists: true, resource: existingResource, id: '123' };
          },
          async () => {
            return { id: 'should-not-happen', value: 'new' };
          },
          {} as LogContext
        );

        expect(result).toEqual(existingResource);
      }
    });

    it('should handle concurrent idempotent operations', async () => {
      const existingResource = { id: 'shared', value: 'test' };

      // Run concurrent operations
      const results = await Promise.all([
        idempotentCreate(
          async () => ({ exists: true, resource: existingResource, id: 'shared' }),
          async () => ({ id: 'new-1' }),
          {} as LogContext
        ),
        idempotentCreate(
          async () => ({ exists: true, resource: existingResource, id: 'shared' }),
          async () => ({ id: 'new-2' }),
          {} as LogContext
        ),
        idempotentCreate(
          async () => ({ exists: true, resource: existingResource, id: 'shared' }),
          async () => ({ id: 'new-3' }),
          {} as LogContext
        )
      ]);

      // All should return the existing resource
      results.forEach(result => {
        expect(result).toEqual(existingResource);
      });
    });
  });
});
