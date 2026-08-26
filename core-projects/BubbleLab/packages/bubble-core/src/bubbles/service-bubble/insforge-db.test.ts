/**
 * Test Suite for InsForgeDbBubble
 *
 * The previous contents of this file were a generated placeholder that tested a
 * non-existent API (`instance.authenticate()`, `instance.execute()`, an `env`
 * context) and did not even parse. It has been rewritten against the real
 * bubble:
 *  - params: { query, allowedOperations, parameters, timeout, maxRows, credentials }
 *  - SQL allow-listing + safety guards enforced in the constructor
 *  - `action()` returns a BubbleResult wrapper `{ success, data, error }`
 *  - schema failures are captured at construction (NOT thrown) and surfaced as a
 *    controlled error from `action()`
 *  - all network access goes through the global `fetch`, mocked here
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { InsForgeDbBubble } from './insforge-db.js';
import { CredentialType } from '@bubblelab/shared-schemas';

const mockCredentials = {
  [CredentialType.INSFORGE_BASE_URL]: 'https://insforge.test',
  [CredentialType.INSFORGE_API_KEY]: 'ins_test_key',
};

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

function errorResponse(status: number, text: string) {
  return {
    ok: false,
    status,
    statusText: 'Error',
    json: async () => ({ error: text }),
    text: async () => text,
  } as unknown as Response;
}

describe('InsForgeDbBubble', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ========================================
  // METADATA & DEFAULTS
  // ========================================
  describe('Bubble metadata', () => {
    it('exposes the expected static metadata', () => {
      expect(InsForgeDbBubble.bubbleName).toBe('insforge-db');
      expect(InsForgeDbBubble.type).toBe('service');
      expect(InsForgeDbBubble.service).toBe('insforge');
      expect(InsForgeDbBubble.authType).toBe('apikey');
      expect(InsForgeDbBubble.alias).toBe('insforge');
      expect(InsForgeDbBubble.schema).toBeDefined();
      expect(InsForgeDbBubble.resultSchema).toBeDefined();
    });

    it('defaults to a read-only SELECT when constructed with no params', () => {
      const bubble = new InsForgeDbBubble();
      expect(bubble.currentParams.query).toBe('SELECT 1');
      expect(bubble.currentParams.allowedOperations).toEqual(['SELECT']);
    });

    it('applies schema defaults', () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT * FROM users',
        credentials: mockCredentials,
      });

      expect(bubble.currentParams.allowedOperations).toEqual([
        'SELECT',
        'WITH',
      ]);
      expect(bubble.currentParams.parameters).toEqual([]);
      expect(bubble.currentParams.timeout).toBe(30000);
      expect(bubble.currentParams.maxRows).toBe(1000);
    });

    it('excludes credentials from currentParams', () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: mockCredentials,
      });
      expect(
        (bubble.currentParams as Record<string, unknown>).credentials
      ).toBeUndefined();
    });
  });

  // ========================================
  // SCHEMA VALIDATION
  // ========================================
  // The Bubble base constructor records schema failures instead of throwing.
  describe('Schema Validation', () => {
    it('returns a controlled error for an empty query', async () => {
      const bubble = new InsForgeDbBubble({
        query: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Query is required');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('returns a controlled error for a non-positive timeout', async () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        timeout: 0,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('returns a controlled error for an unknown SQL operation name', async () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        // @ts-expect-error 'MERGE' is not in the SqlOperations enum
        allowedOperations: ['MERGE'],
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('allowedOperations');
    });
  });

  // ========================================
  // SQL SAFETY GUARDS (constructor-enforced)
  // ========================================
  // Unlike schema validation, these guards DO throw from the constructor:
  // insforge-db.ts calls validateSqlOperation() after super().
  describe('SQL Safety Guards', () => {
    it('rejects an operation outside the allow-list', () => {
      expect(
        () =>
          new InsForgeDbBubble({
            query: 'DELETE FROM users WHERE id = 1',
            allowedOperations: ['SELECT'],
            credentials: mockCredentials,
          })
      ).toThrow("SQL operation 'DELETE' is not allowed");
    });

    it('rejects DELETE without a WHERE clause', () => {
      expect(
        () =>
          new InsForgeDbBubble({
            query: 'DELETE FROM users',
            allowedOperations: ['DELETE'],
            credentials: mockCredentials,
          })
      ).toThrow('DELETE queries must include a WHERE clause');
    });

    it('rejects UPDATE without a WHERE clause', () => {
      expect(
        () =>
          new InsForgeDbBubble({
            query: "UPDATE users SET name = 'x'",
            allowedOperations: ['UPDATE'],
            credentials: mockCredentials,
          })
      ).toThrow('UPDATE queries must include a WHERE clause');
    });

    it('allows DELETE with a WHERE clause', () => {
      expect(
        () =>
          new InsForgeDbBubble({
            query: 'DELETE FROM users WHERE id = $1',
            allowedOperations: ['DELETE'],
            parameters: [1],
            credentials: mockCredentials,
          })
      ).not.toThrow();
    });

    it.each(['DROP', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE'])(
      'blocks the dangerous keyword %s',
      (keyword) => {
        expect(
          () =>
            new InsForgeDbBubble({
              query: `SELECT * FROM users; ${keyword} TABLE users`,
              allowedOperations: ['SELECT'],
              credentials: mockCredentials,
            })
        ).toThrow('potentially dangerous operations');
      }
    );

    it('allows WITH (CTE) queries by default', () => {
      expect(
        () =>
          new InsForgeDbBubble({
            query: 'WITH t AS (SELECT 1) SELECT * FROM t',
            credentials: mockCredentials,
          })
      ).not.toThrow();
    });

    it('is case-insensitive about the leading keyword', () => {
      expect(
        () =>
          new InsForgeDbBubble({
            query: 'select * from users',
            credentials: mockCredentials,
          })
      ).not.toThrow();
    });

    it('does not run the SQL guard when schema validation already failed', () => {
      // `query` is invalid, so this.params holds raw input. The guard must be
      // skipped rather than crashing on undefined.
      expect(
        () =>
          new InsForgeDbBubble({
            query: '',
            credentials: mockCredentials,
          })
      ).not.toThrow();
    });
  });

  // ========================================
  // QUERY EXECUTION
  // ========================================
  describe('Query execution', () => {
    it('returns rows for a successful array response', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse([
          { id: 1, name: 'Ada' },
          { id: 2, name: 'Grace' },
        ])
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT * FROM users',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.rows).toHaveLength(2);
      expect(result.data?.rowCount).toBe(2);
      expect(result.data?.command).toBe('SELECT');
      expect(result.data?.error).toBe('');
      expect(JSON.parse(result.data!.cleanedJSONString)).toHaveLength(2);
      expect(typeof result.data?.executionTime).toBe('number');
    });

    it('accepts a { rows: [...] } response shape', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ rows: [{ id: 1 }] })
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.rows).toEqual([{ id: 1 }]);
    });

    it('returns an empty result set without error', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse([]));

      const bubble = new InsForgeDbBubble({
        query: 'SELECT * FROM users WHERE 1 = 0',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.rows).toEqual([]);
      expect(result.data?.rowCount).toBe(0);
      expect(result.data?.cleanedJSONString).toBe('[]');
    });

    it('truncates rows to maxRows but reports the full count', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse(Array.from({ length: 50 }, (_, i) => ({ id: i })))
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT * FROM users',
        maxRows: 10,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.rows).toHaveLength(10);
      expect(result.data?.rowCount).toBe(50);
    });

    it('posts the query and parameters to the raw SQL endpoint', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse([]));

      const bubble = new InsForgeDbBubble({
        query: 'SELECT * FROM users WHERE active = $1',
        parameters: [true],
        credentials: mockCredentials,
      });

      await bubble.action();

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toBe(
        'https://insforge.test/api/database/advance/rawsql/unrestricted'
      );
      expect(init.method).toBe('POST');
      expect((init.headers as Record<string, string>).Authorization).toBe(
        'Bearer ins_test_key'
      );
      expect(JSON.parse(init.body as string)).toEqual({
        query: 'SELECT * FROM users WHERE active = $1',
        params: [true],
      });
    });

    it('strips a trailing slash from the base URL', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse([]));

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: {
          [CredentialType.INSFORGE_BASE_URL]: 'https://insforge.test/',
          [CredentialType.INSFORGE_API_KEY]: 'ins_test_key',
        },
      });

      await bubble.action();

      expect(vi.mocked(global.fetch).mock.calls[0][0]).toBe(
        'https://insforge.test/api/database/advance/rawsql/unrestricted'
      );
    });
  });

  // ========================================
  // ERROR HANDLING
  // ========================================
  describe('Error handling', () => {
    it('surfaces an HTTP failure as a controlled error result', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(500, 'relation "users" does not exist')
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT * FROM users',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.data?.error).toContain('InsForge query failed: 500');
      expect(result.data?.rows).toEqual([]);
      expect(result.data?.rowCount).toBeNull();
      expect(result.data?.command).toBe('SELECT');
    });

    it('surfaces a network failure as a controlled error result', async () => {
      vi.mocked(global.fetch).mockRejectedValueOnce(
        new Error('Network error: ECONNREFUSED')
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.data?.error).toContain('Network error');
    });

    // NOTE: getCredentials() runs OUTSIDE performAction()'s try/catch, so
    // credential problems propagate and the base class wraps them in a
    // BubbleExecutionError rather than returning { success: false }.
    it('reports a missing base URL without hitting the network', async () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: {
          [CredentialType.INSFORGE_API_KEY]: 'ins_test_key',
        },
      });

      await expect(bubble.action()).rejects.toThrow(
        'InsForge base URL not provided'
      );
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('reports a missing API key without hitting the network', async () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: {
          [CredentialType.INSFORGE_BASE_URL]: 'https://insforge.test',
        },
      });

      await expect(bubble.action()).rejects.toThrow(
        'InsForge API key not provided'
      );
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('reports entirely missing credentials', async () => {
      const bubble = new InsForgeDbBubble({ query: 'SELECT 1' });

      await expect(bubble.action()).rejects.toThrow(
        'No InsForge credentials provided'
      );
    });

    it('does not leak the API key in error messages', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'unauthorized')
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.data?.error).not.toContain('ins_test_key');
    });
  });

  // ========================================
  // CREDENTIALS
  // ========================================
  describe('testCredential', () => {
    it('validates with base URL + API key via a probe query', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse([{ test: 1 }])
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).resolves.toBe(true);

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(JSON.parse(init.body as string).query).toBe('SELECT 1 as test');
    });

    it('rejects when the probe query fails', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'bad key')
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).rejects.toThrow(
        'InsForge credential validation failed: 401'
      );
    });

    it('falls back to a health check when only the base URL is provided', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse({ ok: true }));

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: {
          [CredentialType.INSFORGE_BASE_URL]: 'https://insforge.test',
        },
      });

      await expect(bubble.testCredential()).resolves.toBe(true);
      expect(vi.mocked(global.fetch).mock.calls[0][0]).toBe(
        'https://insforge.test/api/health'
      );
    });

    it('rejects when the health check fails', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(503, 'down')
      );

      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: {
          [CredentialType.INSFORGE_BASE_URL]: 'https://insforge.test',
        },
      });

      await expect(bubble.testCredential()).rejects.toThrow(
        'InsForge health check failed: 503'
      );
    });

    it('assumes valid when only an API key is provided (no URL to check)', async () => {
      const bubble = new InsForgeDbBubble({
        query: 'SELECT 1',
        credentials: {
          [CredentialType.INSFORGE_API_KEY]: 'ins_test_key',
        },
      });

      await expect(bubble.testCredential()).resolves.toBe(true);
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('rejects when no credentials are supplied at all', async () => {
      const bubble = new InsForgeDbBubble({ query: 'SELECT 1' });

      await expect(bubble.testCredential()).rejects.toThrow(
        'No InsForge credentials provided'
      );
    });
  });

  // ========================================
  // CONCURRENCY
  // ========================================
  describe('Concurrency', () => {
    it('handles concurrent queries independently', async () => {
      vi.mocked(global.fetch).mockImplementation(async () =>
        jsonResponse([{ id: 1 }])
      );

      const bubbles = Array.from(
        { length: 5 },
        (_, i) =>
          new InsForgeDbBubble({
            query: `SELECT ${i + 1}`,
            credentials: mockCredentials,
          })
      );

      const results = await Promise.all(bubbles.map((b) => b.action()));

      expect(results).toHaveLength(5);
      expect(results.every((r) => r.success)).toBe(true);
      expect(global.fetch).toHaveBeenCalledTimes(5);
    });
  });
});
