"use strict";
/**
 * Graphiti Adapter Contract Tests
 *
 * Comprehensive contract tests for the Graphiti temporal knowledge graph adapter.
 * These tests validate that the adapter correctly implements the canonical schema
 * and maintains the contract with Graphiti core.
 *
 * Environment Variables:
 *   NEO4J_URI         - Neo4j connection URI (required)
 *   NEO4J_USER        - Neo4j username (required)
 *   NEO4J_PASSWORD    - Neo4j password (required)
 *   OPENAI_API_KEY    - OpenAI API key for LLM features (optional)
 *   TIMEOUT_MS        - Request timeout in milliseconds (default: 30000)
 *   SKIP_INTEGRATION_TESTS - Skip integration tests if true
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const adapter_1 = require("../src/adapter");
const graphiti_canonical_1 = require("../../schemas/graphiti-canonical");
// Configuration from environment
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USER = process.env.NEO4J_USER || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || '';
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || '30000', 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const SKIP_INTEGRATION_TESTS = process.env.SKIP_INTEGRATION_TESTS === 'true';
// Test utilities
const generateTestId = () => {
    return `test-${Date.now()}-${Math.random().toString(36).substring(7)}`;
};
const sleep = (ms) => {
    return new Promise(resolve => setTimeout(resolve, ms));
};
// Adapter instance
let adapter;
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Initialization', () => {
    (0, globals_1.test)('should initialize successfully with valid configuration', async () => {
        const config = {
            graphiti_api_url: 'http://localhost:8000',
            neo4j_uri: NEO4J_URI,
            neo4j_user: NEO4J_USER,
            neo4j_password: NEO4J_PASSWORD,
            openai_api_key: OPENAI_API_KEY,
            timeout_ms: TIMEOUT_MS,
        };
        adapter = new adapter_1.GraphitiAdapter(config);
        await adapter.initialize();
        const health = await adapter.healthCheck();
        (0, globals_1.expect)(health.initialized).toBe(true);
        (0, globals_1.expect)(health.graphiti_connected).toBe(true);
        (0, globals_1.expect)(health.healthy).toBe(true);
    });
    (0, globals_1.test)('should report correct statistics', async () => {
        const stats = await adapter.getStatistics();
        (0, globals_1.expect)(stats).toHaveProperty('entities_count');
        (0, globals_1.expect)(stats).toHaveProperty('relationships_count');
        (0, globals_1.expect)(stats).toHaveProperty('episodes_count');
        (0, globals_1.expect)(stats).toHaveProperty('initialized', true);
        (0, globals_1.expect)(stats).toHaveProperty('connection_status', 'connected');
        (0, globals_1.expect)(stats.entities_count).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(stats.relationships_count).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(stats.episodes_count).toBeGreaterThanOrEqual(0);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Episode Operations', () => {
    const testEpisodeId = generateTestId();
    (0, globals_1.test)('should add an episode to the knowledge graph', async () => {
        const operation = {
            name: `test-episode-${testEpisodeId}`,
            content: 'John Doe works at OpenEvolve as a software engineer. He started in January 2024.',
            source_description: 'Test episode for contract validation',
            episode_type: 'text',
            valid_at: new Date().toISOString(),
            update_communities: false,
        };
        // Validate operation against canonical schema
        const validation = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.AddEpisodeOperationSchema, operation);
        (0, globals_1.expect)(validation.success).toBe(true);
        if (!validation.success) {
            console.error('Validation errors:', validation.errors);
        }
        const result = await adapter.addEpisode(operation);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.episode_id).toBeDefined();
        (0, globals_1.expect)(result.entities_extracted).toBeGreaterThan(0);
        (0, globals_1.expect)(result.relationships_extracted).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(result.processing_time_ms).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(result.correlation_id).toBeDefined();
    });
    (0, globals_1.test)('should add episodes in bulk', async () => {
        const operations = [
            {
                name: `bulk-episode-1-${testEpisodeId}`,
                content: 'Alice is a project manager at OpenEvolve.',
                episode_type: 'text',
                valid_at: new Date().toISOString(),
            },
            {
                name: `bulk-episode-2-${testEpisodeId}`,
                content: 'Bob is a data scientist at OpenEvolve.',
                episode_type: 'text',
                valid_at: new Date().toISOString(),
            },
        ];
        const results = await adapter.addEpisodesBulk(operations);
        (0, globals_1.expect)(results).toHaveLength(2);
        (0, globals_1.expect)(results[0].success).toBe(true);
        (0, globals_1.expect)(results[1].success).toBe(true);
        (0, globals_1.expect)(results[0].entities_extracted).toBeGreaterThan(0);
        (0, globals_1.expect)(results[1].entities_extracted).toBeGreaterThan(0);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Triplet Operations', () => {
    (0, globals_1.test)('should add a triplet (subject -> predicate -> object)', async () => {
        const operation = {
            subject: {
                name: `TestEntity_${generateTestId()}`,
                labels: ['Person', 'Test'],
                summary: 'A test person entity',
            },
            predicate: {
                relation_type: 'WORKS_AT',
                fact: 'works at',
            },
            object: {
                name: 'OpenEvolve',
                labels: ['Organization'],
                summary: 'OpenEvolve company',
            },
            valid_at: new Date().toISOString(),
        };
        // Validate operation against canonical schema
        const validation = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.AddTripletOperationSchema, operation);
        (0, globals_1.expect)(validation.success).toBe(true);
        if (!validation.success) {
            console.error('Validation errors:', validation.errors);
        }
        const result = await adapter.addTriplet(operation);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.subject_uuid).toBeDefined();
        (0, globals_1.expect)(result.object_uuid).toBeDefined();
        (0, globals_1.expect)(result.edge_uuid).toBeDefined();
        (0, globals_1.expect)(result.processing_time_ms).toBeGreaterThanOrEqual(0);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Search Operations', () => {
    (0, globals_1.beforeAll)(async () => {
        // Ensure we have some data to search
        const episode = {
            name: `search-test-episode-${generateTestId()}`,
            content: 'Sarah Connor is a resistance leader. She fights against Skynet.',
            episode_type: 'text',
            valid_at: new Date().toISOString(),
        };
        await adapter.addEpisode(episode);
        await sleep(1000); // Give graph time to index
    });
    (0, globals_1.test)('should search the knowledge graph', async () => {
        const query = {
            query: 'Sarah Connor',
            temporal_filter: 'current',
            max_results: 10,
        };
        // Validate query against canonical schema
        const validation = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.CanonicalSearchQuerySchema, query);
        (0, globals_1.expect)(validation.success).toBe(true);
        const result = await adapter.search(query);
        (0, globals_1.expect)(result.edges).toBeDefined();
        (0, globals_1.expect)(result.nodes).toBeDefined();
        (0, globals_1.expect)(result.total_count).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(result.query_time_ms).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(Array.isArray(result.edges)).toBe(true);
        (0, globals_1.expect)(Array.isArray(result.nodes)).toBe(true);
    });
    (0, globals_1.test)('should search with temporal filter', async () => {
        const query = {
            query: 'resistance leader',
            temporal_filter: 'time_range',
            start_time: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
            end_time: new Date().toISOString(),
            max_results: 5,
        };
        const result = await adapter.search(query);
        (0, globals_1.expect)(result.edges).toBeDefined();
        (0, globals_1.expect)(result.nodes).toBeDefined();
        (0, globals_1.expect)(result.total_count).toBeGreaterThanOrEqual(0);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Entity Operations', () => {
    (0, globals_1.test)('should retrieve an entity by UUID', async () => {
        // First, create an entity via episode
        const episode = {
            name: `entity-test-${generateTestId()}`,
            content: 'TestEntity123 is a fictional character created for testing.',
            episode_type: 'text',
            valid_at: new Date().toISOString(),
        };
        const episodeResult = await adapter.addEpisode(episode);
        (0, globals_1.expect)(episodeResult.success).toBe(true);
        // Search for the entity
        const searchResult = await adapter.search({
            query: 'TestEntity123',
            max_results: 1,
        });
        if (searchResult.nodes.length > 0) {
            const entityUuid = searchResult.nodes[0].id;
            const entity = await adapter.getEntity(entityUuid);
            (0, globals_1.expect)(entity).toBeDefined();
            (0, globals_1.expect)(entity).toHaveProperty('id', entityUuid);
            (0, globals_1.expect)(entity).toHaveProperty('name');
            (0, globals_1.expect)(entity).toHaveProperty('labels');
            (0, globals_1.expect)(entity).toHaveProperty('created_at');
            // Validate against canonical schema
            const validation = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.CanonicalEntitySchema, entity);
            (0, globals_1.expect)(validation.success).toBe(true);
        }
    });
    (0, globals_1.test)('should return null for non-existent entity', async () => {
        const fakeUuid = '00000000-0000-0000-0000-000000000000';
        const entity = await adapter.getEntity(fakeUuid);
        (0, globals_1.expect)(entity).toBeNull();
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Temporal Operations', () => {
    (0, globals_1.test)('should query at point in time', async () => {
        const timestamp = new Date().toISOString();
        const result = await adapter.queryAtPointInTime('character', timestamp, 5);
        (0, globals_1.expect)(result.edges).toBeDefined();
        (0, globals_1.expect)(result.nodes).toBeDefined();
        (0, globals_1.expect)(Array.isArray(result.edges)).toBe(true);
        (0, globals_1.expect)(Array.isArray(result.nodes)).toBe(true);
    });
    (0, globals_1.test)('should get entity timeline', async () => {
        const startTime = new Date(Date.now() - 86400000).toISOString();
        const endTime = new Date().toISOString();
        const timeline = await adapter.getEntityTimeline('Sarah', startTime, endTime);
        (0, globals_1.expect)(Array.isArray(timeline)).toBe(true);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Error Handling', () => {
    (0, globals_1.test)('should handle invalid episode operation gracefully', async () => {
        const invalidOperation = {
            name: '', // Invalid: empty name
            content: 'Test content',
            episode_type: 'text',
            valid_at: 'invalid-date', // Invalid: bad date format
        };
        const result = await adapter.addEpisode(invalidOperation);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.error).toBeDefined();
    });
    (0, globals_1.test)('should handle invalid search query', async () => {
        const invalidQuery = {
            query: '', // Invalid: empty query
            temporal_filter: 'invalid_filter', // Invalid: bad filter
            max_results: -1, // Invalid: negative max_results
        };
        await (0, globals_1.expect)(adapter.search(invalidQuery)).rejects.toThrow();
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - UTC Compliance', () => {
    (0, globals_1.test)('all timestamps should be in UTC ISO-8601 format', async () => {
        const episode = {
            name: `utc-test-${generateTestId()}`,
            content: 'Test UTC timestamp compliance',
            episode_type: 'text',
            valid_at: new Date().toISOString(),
        };
        const result = await adapter.addEpisode(episode);
        (0, globals_1.expect)(result.success).toBe(true);
        // Verify the episode was added with UTC timestamp
        const searchResult = await adapter.search({
            query: 'UTC timestamp compliance',
            max_results: 1,
        });
        if (searchResult.edges.length > 0) {
            const edge = searchResult.edges[0];
            (0, globals_1.expect)(edge.created_at).toBeDefined();
            // Validate ISO-8601 UTC format (ends with Z)
            const timestamp = new Date(edge.created_at);
            (0, globals_1.expect)(timestamp.toISOString()).toBe(edge.created_at);
        }
    });
});
(0, globals_1.describe)('Graphiti Adapter - Canonical Schema Validation', () => {
    (0, globals_1.test)('CanonicalEntity schema should validate correctly', () => {
        const entity = {
            id: '550e8400-e29b-41d4-a716-446655440000',
            name: 'Test Entity',
            labels: ['Entity', 'Test'],
            summary: 'A test entity',
            created_at: '2024-01-15T10:30:00.000Z',
            attributes: {},
        };
        const result = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.CanonicalEntitySchema, entity);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(entity);
    });
    (0, globals_1.test)('AddEpisodeOperation schema should validate correctly', () => {
        const operation = {
            name: 'Test Episode',
            content: 'Test episode content',
            episode_type: 'text',
            valid_at: '2024-01-15T10:30:00.000Z',
        };
        const result = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.AddEpisodeOperationSchema, operation);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('AddTripletOperation schema should validate correctly', () => {
        const operation = {
            subject: {
                name: 'Subject',
                labels: ['Entity'],
            },
            predicate: {
                relation_type: 'RELATES_TO',
                fact: 'relates to',
            },
            object: {
                name: 'Object',
                labels: ['Entity'],
            },
            valid_at: '2024-01-15T10:30:00.000Z',
        };
        const result = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.AddTripletOperationSchema, operation);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('CanonicalSearchQuery schema should validate correctly', () => {
        const query = {
            query: 'test search',
            temporal_filter: 'current',
            max_results: 10,
        };
        const result = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.CanonicalSearchQuerySchema, query);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('should reject invalid entity data', () => {
        const invalidEntity = {
            id: 'not-a-uuid', // Invalid UUID format
            name: 'Test',
            labels: ['Test'],
            created_at: '2024-01-15T10:30:00.000Z',
        };
        const result = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.CanonicalEntitySchema, invalidEntity);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors).toBeDefined();
        (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('should reject invalid episode operation', () => {
        const invalidOperation = {
            name: '', // Empty name not allowed
            content: 'Test',
            episode_type: 'invalid_type', // Invalid episode type
            valid_at: 'not-a-date', // Invalid date format
        };
        const result = (0, graphiti_canonical_1.validateCanonical)(graphiti_canonical_1.AddEpisodeOperationSchema, invalidOperation);
        (0, globals_1.expect)(result.success).toBe(false);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Health Check', () => {
    (0, globals_1.test)('should pass health check', async () => {
        const health = await adapter.healthCheck();
        (0, globals_1.expect)(health).toHaveProperty('healthy');
        (0, globals_1.expect)(health).toHaveProperty('circuit_state');
        (0, globals_1.expect)(health).toHaveProperty('initialized');
        (0, globals_1.expect)(health).toHaveProperty('graphiti_connected');
        (0, globals_1.expect)(health.initialized).toBe(true);
        (0, globals_1.expect)(health.graphiti_connected).toBe(true);
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Cleanup', () => {
    (0, globals_1.test)('should close adapter gracefully', async () => {
        await adapter.close();
        // Verify health check shows disconnected
        const health = await adapter.healthCheck();
        (0, globals_1.expect)(health.initialized).toBe(false);
    });
});
//# sourceMappingURL=contract.test.js.map