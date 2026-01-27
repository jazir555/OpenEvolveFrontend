/**
 * Test Utilities
 * Reusable helper functions for testing
 */
import { BubbleFactory } from '../bubble-factory.js';
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * Create a test credential set
 */
export declare const createTestCredentials: (type: CredentialType, value: string) => {
    [x: string]: string;
};
/**
 * Create database test credentials
 */
export declare const createDatabaseCredentials: () => {
    DATABASE_CRED: string;
};
/**
 * Create API key test credentials
 */
export declare const createApiCredentials: () => {
    CUSTOM_AUTH_KEY: string;
};
/**
 * Create OAuth test credentials
 */
export declare const createOAuthCredentials: () => {
    OAUTH_TOKEN: string;
};
/**
 * Wait for a specified amount of time
 */
export declare const wait: (ms: number) => Promise<unknown>;
/**
 * Create a mock fetch response
 */
export declare const createMockResponse: (data: unknown, status?: number, ok?: boolean) => {
    ok: boolean;
    status: number;
    statusText: string;
    text: () => Promise<string>;
    json: () => Promise<unknown>;
    headers: Map<string, string>;
};
/**
 * Create a mock error response
 */
export declare const createMockErrorResponse: (error: string, status?: number) => {
    ok: boolean;
    status: number;
    statusText: string;
    text: () => Promise<string>;
    json: () => Promise<{
        error: string;
    }>;
    headers: Map<string, string>;
};
/**
 * Mock PostgreSQL Pool
 */
export declare class MockPostgresPool {
    readonly connectionString: string;
    private connected;
    constructor(connectionString: string);
    query(text: string, params?: unknown[]): Promise<{
        rows: {
            id: number;
            name: string;
            created_at: Date;
        }[];
        rowCount: number;
        command: string;
        fields: {
            name: string;
            dataTypeID: number;
        }[];
    }>;
    end(): Promise<void>;
    connect(): {
        release: () => void;
    };
}
/**
 * Mock Redis client
 */
export declare class MockRedisClient {
    private data;
    get(key: string): Promise<string | undefined>;
    set(key: string, value: string): Promise<string>;
    del(key: string): Promise<0 | 1>;
    exists(key: string): Promise<0 | 1>;
    expire(key: string, seconds: number): Promise<number>;
    quit(): Promise<void>;
    flushall(): Promise<string>;
}
/**
 * Mock HTTP fetch
 */
export declare const mockFetch: (responses: unknown[]) => import("vitest").Mock<() => Promise<unknown>>;
/**
 * Test data generators
 */
export declare const generateTestData: {
    user: () => {
        id: number;
        name: string;
        email: string;
        createdAt: string;
    };
    users: (count?: number) => {
        id: number;
        name: string;
        email: string;
        createdAt: string;
    }[];
    queryResult: (rows: Record<string, unknown>[]) => {
        rows: Record<string, unknown>[];
        rowCount: number;
        command: string;
        fields: {
            name: string;
            dataTypeID: number;
        }[];
    };
};
/**
 * Assert error is thrown
 */
export declare const expectError: (fn: () => Promise<unknown> | unknown) => Promise<boolean>;
/**
 * Create a test context
 */
export declare const createTestContext: () => {
    correlationId: string;
    timestamp: string;
};
/**
 * Mock service bubbles factory
 */
export declare const createMockFactory: () => Promise<BubbleFactory>;
/**
 * Security test payloads
 */
export declare const securityPayloads: {
    sqlInjection: string[];
    xss: string[];
    pathTraversal: string[];
    commandInjection: string[];
    ssrf: string[];
};
/**
 * Performance test helpers
 */
export declare const measurePerformance: (fn: () => Promise<unknown> | unknown, maxDuration: number) => Promise<number>;
//# sourceMappingURL=test-utils.d.ts.map