/**
 * Mock Services
 * Centralized mock implementations for external dependencies
 */
import { vi } from 'vitest';
/**
 * Mock PostgreSQL Client
 */
export const mockPostgresClient = {
    Pool: vi.fn(() => ({
        connect: vi.fn(() => ({
            query: vi.fn(),
            release: vi.fn(),
        })),
        query: vi.fn(),
        end: vi.fn(),
    })),
};
/**
 * Mock Redis Client
 */
export const mockRedisClient = {
    createClient: vi.fn(() => ({
        connect: vi.fn(),
        get: vi.fn(),
        set: vi.fn(),
        del: vi.fn(),
        exists: vi.fn(),
        expire: vi.fn(),
        quit: vi.fn(),
        flushall: vi.fn(),
        on: vi.fn(),
    })),
};
/**
 * Mock Qdrant Client
 */
export const mockQdrantClient = {
    QdrantClient: vi.fn(() => ({
        getCollections: vi.fn(),
        getCollection: vi.fn(),
        createCollection: vi.fn(),
        deleteCollection: vi.fn(),
        upsert: vi.fn(),
        search: vi.fn(),
        scroll: vi.fn(),
        delete: vi.fn(),
    })),
};
/**
 * Mock Elasticsearch Client
 */
export const mockElasticsearchClient = {
    Client: vi.fn(() => ({
        indices: {
            create: vi.fn(),
            delete: vi.fn(),
            exists: vi.fn(),
            get: vi.fn(),
        },
        search: vi.fn(),
        index: vi.fn(),
        bulk: vi.fn(),
        get: vi.fn(),
        delete: vi.fn(),
    })),
};
/**
 * Mock AWS SDK
 */
export const mockAwsSdk = {
    S3Client: vi.fn(() => ({
        send: vi.fn(),
    })),
    PutObjectCommand: vi.fn(),
    GetObjectCommand: vi.fn(),
    DeleteObjectCommand: vi.fn(),
};
/**
 * Mock Google APIs
 */
export const mockGoogleApis = {
    drive: {
        files: {
            create: vi.fn(),
            get: vi.fn(),
            list: vi.fn(),
            update: vi.fn(),
            delete: vi.fn(),
        },
    },
    sheets: {
        spreadsheets: {
            values: {
                get: vi.fn(),
                update: vi.fn(),
                append: vi.fn(),
                batchGet: vi.fn(),
                batchUpdate: vi.fn(),
            },
        },
    },
};
/**
 * Mock Notion API
 */
export const mockNotionClient = {
    databases: {
        query: vi.fn(),
        retrieve: vi.fn(),
    },
    pages: {
        create: vi.fn(),
        retrieve: vi.fn(),
        update: vi.fn(),
    },
    blocks: {
        children: {
            list: vi.fn(),
            append: vi.fn(),
        },
    },
};
/**
 * Mock Stripe API
 */
export const mockStripeClient = {
    charges: {
        create: vi.fn(),
        retrieve: vi.fn(),
        list: vi.fn(),
    },
    customers: {
        create: vi.fn(),
        retrieve: vi.fn(),
        update: vi.fn(),
    },
    paymentIntents: {
        create: vi.fn(),
        confirm: vi.fn(),
        cancel: vi.fn(),
    },
};
/**
 * Mock Twilio API
 */
export const mockTwilioClient = {
    messages: {
        create: vi.fn(),
        list: vi.fn(),
        retrieve: vi.fn(),
    },
    calls: {
        create: vi.fn(),
        list: vi.fn(),
    },
};
/**
 * Mock SendGrid API
 */
export const mockSendGridClient = {
    send: vi.fn(),
    setApiKey: vi.fn(),
};
/**
 * Mock Apify Client
 */
export const mockApifyClient = {
    actor: vi.fn(() => ({
        call: vi.fn(),
        start: vi.fn(),
        info: vi.fn(),
    })),
};
/**
 * Mock Webhook delivery
 */
export const mockWebhookDelivery = vi.fn();
/**
 * Setup all mocks
 */
export const setupMocks = () => {
    vi.mock('pg', () => mockPostgresClient);
    vi.mock('redis', () => ({ createClient: mockRedisClient.createClient }));
    vi.mock('@elastic/elasticsearch', () => mockElasticsearchClient);
    vi.mock('@aws-sdk/client-s3', () => ({
        S3Client: mockAwsSdk.S3Client,
        PutObjectCommand: mockAwsSdk.PutObjectCommand,
        GetObjectCommand: mockAwsSdk.GetObjectCommand,
        DeleteObjectCommand: mockAwsSdk.DeleteObjectCommand,
    }));
};
/**
 * Clear all mocks
 */
export const clearMocks = () => {
    vi.clearAllMocks();
};
//# sourceMappingURL=index.js.map