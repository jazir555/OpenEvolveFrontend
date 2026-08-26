/**
 * SQL Query Tool - unit tests
 * Rewritten to exercise the real SQLQueryTool API (performAction).
 * The external PostgreSQLBubble is mocked.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';

const { pgState } = vi.hoisted(() => ({ pgState: { fail: false } }));

vi.mock('../service-bubble/postgresql.ts', () => {
  class MockPostgreSQLBubble {
    async action() {
      if (pgState.fail) {
        return { success: false, error: 'Table does not exist' };
      }
      return {
        success: true,
        data: {
          rows: [{ id: 1, name: 'Test' }],
          rowCount: 1,
          fields: [
            { name: 'id', dataTypeID: 23 },
            { name: 'name', dataTypeID: 25 },
          ],
        },
      };
    }
  }
  return { PostgreSQLBubble: MockPostgreSQLBubble };
});

import { SQLQueryTool } from './sql-query-tool';

describe('SQLQueryTool', () => {
  afterEach(() => {
    pgState.fail = false;
  });

  it('constructs with query and reasoning', () => {
    const tool = new SQLQueryTool({ query: 'SELECT 1', reasoning: 'test' });
    expect(tool).toBeDefined();
    expect(tool.params.query).toBe('SELECT 1');
  });

  it('executes a query successfully', async () => {
    const tool = new SQLQueryTool({
      query: 'SELECT * FROM users',
      reasoning: 'get users',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.rowCount).toBe(1);
    expect(result.rows).toHaveLength(1);
    expect(result.fields).toHaveLength(2);
    expect(result.executionTime).toBeGreaterThanOrEqual(0);
  });

  it('propagates query errors', async () => {
    pgState.fail = true;
    const tool = new SQLQueryTool({ query: 'SELECT 1', reasoning: 'x' });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Table does not exist');
  });
});
