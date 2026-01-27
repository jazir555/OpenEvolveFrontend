/**
 * Mock Services
 * Centralized mock implementations for external dependencies
 */
/**
 * Mock PostgreSQL Client
 */
export declare const mockPostgresClient: {
    Pool: import("vitest").Mock<() => {
        connect: import("vitest").Mock<() => {
            query: import("vitest").Mock<(...args: any[]) => any>;
            release: import("vitest").Mock<(...args: any[]) => any>;
        }>;
        query: import("vitest").Mock<(...args: any[]) => any>;
        end: import("vitest").Mock<(...args: any[]) => any>;
    }>;
};
/**
 * Mock Redis Client
 */
export declare const mockRedisClient: {
    createClient: import("vitest").Mock<() => {
        connect: import("vitest").Mock<(...args: any[]) => any>;
        get: import("vitest").Mock<(...args: any[]) => any>;
        set: import("vitest").Mock<(...args: any[]) => any>;
        del: import("vitest").Mock<(...args: any[]) => any>;
        exists: import("vitest").Mock<(...args: any[]) => any>;
        expire: import("vitest").Mock<(...args: any[]) => any>;
        quit: import("vitest").Mock<(...args: any[]) => any>;
        flushall: import("vitest").Mock<(...args: any[]) => any>;
        on: import("vitest").Mock<(...args: any[]) => any>;
    }>;
};
/**
 * Mock Qdrant Client
 */
export declare const mockQdrantClient: {
    QdrantClient: import("vitest").Mock<() => {
        getCollections: import("vitest").Mock<(...args: any[]) => any>;
        getCollection: import("vitest").Mock<(...args: any[]) => any>;
        createCollection: import("vitest").Mock<(...args: any[]) => any>;
        deleteCollection: import("vitest").Mock<(...args: any[]) => any>;
        upsert: import("vitest").Mock<(...args: any[]) => any>;
        search: import("vitest").Mock<(...args: any[]) => any>;
        scroll: import("vitest").Mock<(...args: any[]) => any>;
        delete: import("vitest").Mock<(...args: any[]) => any>;
    }>;
};
/**
 * Mock Elasticsearch Client
 */
export declare const mockElasticsearchClient: {
    Client: import("vitest").Mock<() => {
        indices: {
            create: import("vitest").Mock<(...args: any[]) => any>;
            delete: import("vitest").Mock<(...args: any[]) => any>;
            exists: import("vitest").Mock<(...args: any[]) => any>;
            get: import("vitest").Mock<(...args: any[]) => any>;
        };
        search: import("vitest").Mock<(...args: any[]) => any>;
        index: import("vitest").Mock<(...args: any[]) => any>;
        bulk: import("vitest").Mock<(...args: any[]) => any>;
        get: import("vitest").Mock<(...args: any[]) => any>;
        delete: import("vitest").Mock<(...args: any[]) => any>;
    }>;
};
/**
 * Mock AWS SDK
 */
export declare const mockAwsSdk: {
    S3Client: import("vitest").Mock<() => {
        send: import("vitest").Mock<(...args: any[]) => any>;
    }>;
    PutObjectCommand: import("vitest").Mock<(...args: any[]) => any>;
    GetObjectCommand: import("vitest").Mock<(...args: any[]) => any>;
    DeleteObjectCommand: import("vitest").Mock<(...args: any[]) => any>;
};
/**
 * Mock Google APIs
 */
export declare const mockGoogleApis: {
    drive: {
        files: {
            create: import("vitest").Mock<(...args: any[]) => any>;
            get: import("vitest").Mock<(...args: any[]) => any>;
            list: import("vitest").Mock<(...args: any[]) => any>;
            update: import("vitest").Mock<(...args: any[]) => any>;
            delete: import("vitest").Mock<(...args: any[]) => any>;
        };
    };
    sheets: {
        spreadsheets: {
            values: {
                get: import("vitest").Mock<(...args: any[]) => any>;
                update: import("vitest").Mock<(...args: any[]) => any>;
                append: import("vitest").Mock<(...args: any[]) => any>;
                batchGet: import("vitest").Mock<(...args: any[]) => any>;
                batchUpdate: import("vitest").Mock<(...args: any[]) => any>;
            };
        };
    };
};
/**
 * Mock Notion API
 */
export declare const mockNotionClient: {
    databases: {
        query: import("vitest").Mock<(...args: any[]) => any>;
        retrieve: import("vitest").Mock<(...args: any[]) => any>;
    };
    pages: {
        create: import("vitest").Mock<(...args: any[]) => any>;
        retrieve: import("vitest").Mock<(...args: any[]) => any>;
        update: import("vitest").Mock<(...args: any[]) => any>;
    };
    blocks: {
        children: {
            list: import("vitest").Mock<(...args: any[]) => any>;
            append: import("vitest").Mock<(...args: any[]) => any>;
        };
    };
};
/**
 * Mock Stripe API
 */
export declare const mockStripeClient: {
    charges: {
        create: import("vitest").Mock<(...args: any[]) => any>;
        retrieve: import("vitest").Mock<(...args: any[]) => any>;
        list: import("vitest").Mock<(...args: any[]) => any>;
    };
    customers: {
        create: import("vitest").Mock<(...args: any[]) => any>;
        retrieve: import("vitest").Mock<(...args: any[]) => any>;
        update: import("vitest").Mock<(...args: any[]) => any>;
    };
    paymentIntents: {
        create: import("vitest").Mock<(...args: any[]) => any>;
        confirm: import("vitest").Mock<(...args: any[]) => any>;
        cancel: import("vitest").Mock<(...args: any[]) => any>;
    };
};
/**
 * Mock Twilio API
 */
export declare const mockTwilioClient: {
    messages: {
        create: import("vitest").Mock<(...args: any[]) => any>;
        list: import("vitest").Mock<(...args: any[]) => any>;
        retrieve: import("vitest").Mock<(...args: any[]) => any>;
    };
    calls: {
        create: import("vitest").Mock<(...args: any[]) => any>;
        list: import("vitest").Mock<(...args: any[]) => any>;
    };
};
/**
 * Mock SendGrid API
 */
export declare const mockSendGridClient: {
    send: import("vitest").Mock<(...args: any[]) => any>;
    setApiKey: import("vitest").Mock<(...args: any[]) => any>;
};
/**
 * Mock Apify Client
 */
export declare const mockApifyClient: {
    actor: import("vitest").Mock<() => {
        call: import("vitest").Mock<(...args: any[]) => any>;
        start: import("vitest").Mock<(...args: any[]) => any>;
        info: import("vitest").Mock<(...args: any[]) => any>;
    }>;
};
/**
 * Mock Webhook delivery
 */
export declare const mockWebhookDelivery: import("vitest").Mock<(...args: any[]) => any>;
/**
 * Setup all mocks
 */
export declare const setupMocks: () => void;
/**
 * Clear all mocks
 */
export declare const clearMocks: () => void;
//# sourceMappingURL=index.d.ts.map