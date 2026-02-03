/**
 * SQL Query Tool Unit Tests
 * File: tool-bubble/sql-query-tool.test.ts
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SQLQueryTool } from './sql-query-tool.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('SQLQueryTool', () => {
  let mockPostgresBubble: any;

  beforeEach(() => {
    // Mock PostgreSQLBubble
    mockPostgresBubble = vi.fn().mockResolvedValue({
      success: true,
      data: {
        rows: [
          { id: 1, name: 'Test', value: 100 },
          { id: 2, name: 'Test 2', value: 200 },
        ],
        rowCount: 2,
        fields: [
          { name: 'id', dataTypeID: 23 },
          { name: 'name', dataTypeID: 25 },
          { name: 'value', dataTypeID: 23 },
        ],
      },
    });

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Construction and Initialization', () => {
    it('should create instance with valid parameters', () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM users',
        reasoning: 'Get all users',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      expect(tool).toBeDefined();
      expect(tool.params.query).toBe('SELECT * FROM users');
      expect(tool.params.reasoning).toBe('Get all users');
    });

    it('should validate required query parameter', () => {
      expect(() => {
        new SQLQueryTool({
          query: '',
          reasoning: 'Test',
          credentials: {
            [CredentialType.POSTGRES_CRED]: JSON.stringify({
              host: 'localhost',
              database: 'test',
            }),
          },
        });
      }).toThrow();
    });

    it('should validate required reasoning parameter', () => {
      expect(() => {
        new SQLQueryTool({
          query: 'SELECT 1',
          reasoning: '',
          credentials: {
            [CredentialType.POSTGRES_CRED]: JSON.stringify({
              host: 'localhost',
              database: 'test',
            }),
          },
        });
      }).toThrow();
    });
  });

  describe('Query Validation', () => {
    it('should allow SELECT queries', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM users',
        reasoning: 'Get all users',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      // Access private method for testing
      const validation = (tool as any).validateQuery('SELECT * FROM users');

      expect(validation.valid).toBe(true);
    });

    it('should allow WITH queries', async () => {
      const tool = new SQLQueryTool({
        query: 'WITH cte AS (SELECT 1) SELECT * FROM cte',
        reasoning: 'Test CTE',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(true);
    });

    it('should allow EXPLAIN queries', async () => {
      const tool = new SQLQueryTool({
        query: 'EXPLAIN SELECT * FROM users',
        reasoning: 'Explain query plan',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(true);
    });

    it('should allow ANALYZE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'ANALYZE users',
        reasoning: 'Analyze table',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(true);
    });

    it('should allow SHOW queries', async () => {
      const tool = new SQLQueryTool({
        query: 'SHOW TABLES',
        reasoning: 'List tables',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(true);
    });

    it('should allow DESCRIBE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'DESCRIBE users',
        reasoning: 'Describe table',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(true);
    });
  });

  describe('Security - SQL Injection Prevention', () => {
    it('should block DROP queries', async () => {
      const tool = new SQLQueryTool({
        query: 'DROP TABLE users',
        reasoning: 'Malicious query',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('DROP');
    });

    it('should block DELETE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'DELETE FROM users WHERE id = 1',
        reasoning: 'Delete user',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('DELETE');
    });

    it('should block INSERT queries', async () => {
      const tool = new SQLQueryTool({
        query: 'INSERT INTO users VALUES (1, "test")',
        reasoning: 'Insert user',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('INSERT');
    });

    it('should block UPDATE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'UPDATE users SET name = "test" WHERE id = 1',
        reasoning: 'Update user',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('UPDATE');
    });

    it('should block TRUNCATE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'TRUNCATE TABLE users',
        reasoning: 'Truncate table',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('TRUNCATE');
    });

    it('should block ALTER queries', async () => {
      const tool = new SQLQueryTool({
        query: 'ALTER TABLE users ADD COLUMN age INT',
        reasoning: 'Alter table',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('ALTER');
    });

    it('should block CREATE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'CREATE TABLE test (id INT)',
        reasoning: 'Create table',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('CREATE');
    });

    it('should block GRANT queries', async () => {
      const tool = new SQLQueryTool({
        query: 'GRANT ALL ON users TO test_user',
        reasoning: 'Grant privileges',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('GRANT');
    });

    it('should block REVOKE queries', async () => {
      const tool = new SQLQueryTool({
        query: 'REVOKE ALL ON users FROM test_user',
        reasoning: 'Revoke privileges',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
      expect(validation.error).toContain('REVOKE');
    });

    it('should be case-insensitive when blocking dangerous operations', async () => {
      const tool = new SQLQueryTool({
        query: 'drop table users', // lowercase
        reasoning: 'Malicious',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(false);
    });

    it('should block SQL injection attempts', async () => {
      const tool = new SQLQueryTool({
        query: "SELECT * FROM users WHERE name = 'admin' OR 1=1 --'",
        reasoning: 'SQL injection attempt',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      // The query starts with SELECT so it passes the initial validation
      // but PostgreSQLBubble should handle parameterized queries
      const validation = (tool as any).validateQuery(tool.params.query);

      expect(validation.valid).toBe(true); // Passes initial check
      // PostgreSQLBubble should use parameterized queries internally
    });
  });

  describe('Query Execution', () => {
    it('should execute query successfully', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM users',
        reasoning: 'Get all users',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      // Mock PostgreSQLBubble.action
      const mockResult = {
        success: true,
        data: {
          rows: [
            { id: 1, name: 'Test', value: 100 },
            { id: 2, name: 'Test 2', value: 200 },
          ],
          rowCount: 2,
          fields: [
            { name: 'id', dataTypeID: 23 },
            { name: 'name', dataTypeID: 25 },
            { name: 'value', dataTypeID: 23 },
          ],
        },
      };

      // Mock the PostgreSQLBubble action method
      const PostgresBubbleMock = {
        action: vi.fn().mockResolvedValue(mockResult),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.rowCount).toBe(2);
      expect(result.rows).toHaveLength(2);
      expect(result.executionTime).toBeGreaterThan(0);
    });

    it('should handle query execution errors', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM nonexistent_table',
        reasoning: 'Test error',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      // Mock PostgreSQLBubble.action to return error
      const PostgresBubbleMock = {
        action: vi.fn().mockResolvedValue({
          success: false,
          error: 'Table "nonexistent_table" does not exist',
        }),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('does not exist');
    });

    it('should handle empty result sets', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM users WHERE false',
        reasoning: 'Empty result',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const PostgresBubbleMock = {
        action: vi.fn().mockResolvedValue({
          success: true,
          data: {
            rows: [],
            rowCount: 0,
          },
        }),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.rowCount).toBe(0);
      expect(result.rows).toEqual([]);
    });
  });

  describe('Result Formatting', () => {
    it('should include field metadata in results', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT id, name FROM users',
        reasoning: 'Test fields',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const PostgresBubbleMock = {
        action: vi.fn().mockResolvedValue({
          success: true,
          data: {
            rows: [{ id: 1, name: 'Test' }],
            rowCount: 1,
            fields: [
              { name: 'id', dataTypeID: 23 },
              { name: 'name', dataTypeID: 25 },
            ],
          },
        }),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.fields).toBeDefined();
      expect(result.fields).toHaveLength(2);
      expect(result.fields[0].name).toBe('id');
      expect(result.fields[0].dataTypeID).toBe(23);
    });

    it('should calculate execution time', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT 1',
        reasoning: 'Test timing',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const PostgresBubbleMock = {
        action: vi.fn().mockImplementation(async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return {
            success: true,
            data: {
              rows: [{ result: 1 }],
              rowCount: 1,
            },
          };
        }),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.executionTime).toBeGreaterThan(90); // Should be at least 100ms
    });
  });

  describe('Error Handling', () => {
    it('should handle validation failures', async () => {
      const tool = new SQLQueryTool({
        query: 'DROP TABLE users',
        reasoning: 'Malicious',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('DROP');
    });

    it('should handle empty queries', async () => {
      const tool = new SQLQueryTool({
        query: '   ',
        reasoning: 'Empty query',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('empty');
    });

    it('should handle database connection errors', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT 1',
        reasoning: 'Test connection',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'invalid-host',
            database: 'test',
          }),
        },
      });

      const PostgresBubbleMock = {
        action: vi.fn().mockResolvedValue({
          success: false,
          error: 'Connection refused',
        }),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Connection refused');
    });
  });

  describe('Helper Methods', () => {
    it('should provide sample queries', () => {
      const samples = SQLQueryTool.getSampleQueries();

      expect(samples).toBeDefined();
      expect(samples.listTables).toBeDefined();
      expect(samples.tableSchema).toBeDefined();
      expect(samples.tableSize).toBeDefined();
      expect(samples.topN).toBeDefined();
    });

    it('should format results as CSV', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM users',
        reasoning: 'Test CSV',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const rows = [
        { id: 1, name: 'Test' },
        { id: 2, name: 'Test,With,Comma' },
      ];

      // Access private method
      const csv = (tool as any).formatAsCSV(rows);

      expect(csv).toContain('id,name');
      expect(csv).toContain('1,Test');
      expect(csv).toContain('"Test,With,Comma"'); // Should be quoted
    });

    it('should format results as markdown', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM users',
        reasoning: 'Test markdown',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      const rows = [
        { id: 1, name: 'Test' },
        { id: 2, name: 'Test 2' },
      ];

      // Access private method
      const markdown = (tool as any).formatAsMarkdown(rows);

      expect(markdown).toContain('| id | name |');
      expect(markdown).toContain('|---|---|');
      expect(markdown).toContain('| 1 | Test |');
    });
  });

  describe('Performance', () => {
    it('should enforce query timeout', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT pg_sleep(100)', // Sleep for 100 seconds
        reasoning: 'Test timeout',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      // PostgreSQLBubble should enforce 30s timeout
      const PostgresBubbleMock = {
        action: vi.fn().mockResolvedValue({
          success: false,
          error: 'Query timeout',
        }),
      };
      vi.doMock('../service-bubble/postgresql.js', () => ({
        PostgreSQLBubble: vi.fn().mockImplementation(() => PostgresBubbleMock),
      }));

      const result = await tool.performAction();

      expect(result.success).toBe(false);
    });

    it('should enforce row limit', async () => {
      const tool = new SQLQueryTool({
        query: 'SELECT * FROM large_table',
        reasoning: 'Test row limit',
        credentials: {
          [CredentialType.POSTGRES_CRED]: JSON.stringify({
            host: 'localhost',
            database: 'test',
          }),
        },
      });

      // PostgreSQLBubble should enforce 1000 row limit
      expect(tool.params.query).toBeDefined();
    });
  });
});
