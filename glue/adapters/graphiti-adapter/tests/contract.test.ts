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

import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';
import { GraphitiAdapter } from '../src/adapter';
import { GraphitiClient } from '../src/graph-client';
import {
  CanonicalEntity,
  CanonicalEntityEdge,
  AddEpisodeOperation,
  AddTripletOperation,
  CanonicalSearchQuery,
  EpisodeType,
  validateCanonical,
  CanonicalEntitySchema,
  CanonicalEpisodeSchema,
  AddEpisodeOperationSchema,
  AddTripletOperationSchema,
  CanonicalSearchQuerySchema,
} from '../../schemas/graphiti-canonical';

// Configuration from environment
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USER = process.env.NEO4J_USER || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || '';
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || '30000', 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const SKIP_INTEGRATION_TESTS = process.env.SKIP_INTEGRATION_TESTS === 'true';

// Test utilities
const generateTestId = (): string => {
  return `test-${Date.now()}-${Math.random().toString(36).substring(7)}`;
};

const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

// Adapter instance
let adapter: GraphitiAdapter;

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Initialization', () => {
  test('should initialize successfully with valid configuration', async () => {
    const config = {
      graphiti_api_url: 'http://localhost:8000',
      neo4j_uri: NEO4J_URI,
      neo4j_user: NEO4J_USER,
      neo4j_password: NEO4J_PASSWORD,
      openai_api_key: OPENAI_API_KEY,
      timeout_ms: TIMEOUT_MS,
    };

    adapter = new GraphitiAdapter(config);

    await adapter.initialize();
    const health = await adapter.healthCheck();

    expect(health.initialized).toBe(true);
    expect(health.graphiti_connected).toBe(true);
    expect(health.healthy).toBe(true);
  });

  test('should report correct statistics', async () => {
    const stats = await adapter.getStatistics();

    expect(stats).toHaveProperty('entities_count');
    expect(stats).toHaveProperty('relationships_count');
    expect(stats).toHaveProperty('episodes_count');
    expect(stats).toHaveProperty('initialized', true);
    expect(stats).toHaveProperty('connection_status', 'connected');
    expect(stats.entities_count).toBeGreaterThanOrEqual(0);
    expect(stats.relationships_count).toBeGreaterThanOrEqual(0);
    expect(stats.episodes_count).toBeGreaterThanOrEqual(0);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Episode Operations', () => {
  const testEpisodeId = generateTestId();

  test('should add an episode to the knowledge graph', async () => {
    const operation: AddEpisodeOperation = {
      name: `test-episode-${testEpisodeId}`,
      content: 'John Doe works at OpenEvolve as a software engineer. He started in January 2024.',
      source_description: 'Test episode for contract validation',
      episode_type: 'text',
      valid_at: new Date().toISOString(),
      update_communities: false,
    };

    // Validate operation against canonical schema
    const validation = validateCanonical(AddEpisodeOperationSchema, operation);
    expect(validation.success).toBe(true);
    if (!validation.success) {
      console.error('Validation errors:', validation.errors);
    }

    const result = await adapter.addEpisode(operation);

    expect(result.success).toBe(true);
    expect(result.episode_id).toBeDefined();
    expect(result.entities_extracted).toBeGreaterThan(0);
    expect(result.relationships_extracted).toBeGreaterThanOrEqual(0);
    expect(result.processing_time_ms).toBeGreaterThanOrEqual(0);
    expect(result.correlation_id).toBeDefined();
  });

  test('should add episodes in bulk', async () => {
    const operations: AddEpisodeOperation[] = [
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

    expect(results).toHaveLength(2);
    expect(results[0].success).toBe(true);
    expect(results[1].success).toBe(true);
    expect(results[0].entities_extracted).toBeGreaterThan(0);
    expect(results[1].entities_extracted).toBeGreaterThan(0);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Triplet Operations', () => {
  test('should add a triplet (subject -> predicate -> object)', async () => {
    const operation: AddTripletOperation = {
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
    const validation = validateCanonical(AddTripletOperationSchema, operation);
    expect(validation.success).toBe(true);
    if (!validation.success) {
      console.error('Validation errors:', validation.errors);
    }

    const result = await adapter.addTriplet(operation);

    expect(result.success).toBe(true);
    expect(result.subject_uuid).toBeDefined();
    expect(result.object_uuid).toBeDefined();
    expect(result.edge_uuid).toBeDefined();
    expect(result.processing_time_ms).toBeGreaterThanOrEqual(0);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Search Operations', () => {
  beforeAll(async () => {
    // Ensure we have some data to search
    const episode: AddEpisodeOperation = {
      name: `search-test-episode-${generateTestId()}`,
      content: 'Sarah Connor is a resistance leader. She fights against Skynet.',
      episode_type: 'text',
      valid_at: new Date().toISOString(),
    };
    await adapter.addEpisode(episode);
    await sleep(1000); // Give graph time to index
  });

  test('should search the knowledge graph', async () => {
    const query: CanonicalSearchQuery = {
      query: 'Sarah Connor',
      temporal_filter: 'current',
      max_results: 10,
    };

    // Validate query against canonical schema
    const validation = validateCanonical(CanonicalSearchQuerySchema, query);
    expect(validation.success).toBe(true);

    const result = await adapter.search(query);

    expect(result.edges).toBeDefined();
    expect(result.nodes).toBeDefined();
    expect(result.total_count).toBeGreaterThanOrEqual(0);
    expect(result.query_time_ms).toBeGreaterThanOrEqual(0);
    expect(Array.isArray(result.edges)).toBe(true);
    expect(Array.isArray(result.nodes)).toBe(true);
  });

  test('should search with temporal filter', async () => {
    const query: CanonicalSearchQuery = {
      query: 'resistance leader',
      temporal_filter: 'time_range',
      start_time: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
      end_time: new Date().toISOString(),
      max_results: 5,
    };

    const result = await adapter.search(query);

    expect(result.edges).toBeDefined();
    expect(result.nodes).toBeDefined();
    expect(result.total_count).toBeGreaterThanOrEqual(0);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Entity Operations', () => {
  test('should retrieve an entity by UUID', async () => {
    // First, create an entity via episode
    const episode: AddEpisodeOperation = {
      name: `entity-test-${generateTestId()}`,
      content: 'TestEntity123 is a fictional character created for testing.',
      episode_type: 'text',
      valid_at: new Date().toISOString(),
    };

    const episodeResult = await adapter.addEpisode(episode);
    expect(episodeResult.success).toBe(true);

    // Search for the entity
    const searchResult = await adapter.search({
      query: 'TestEntity123',
      max_results: 1,
    });

    if (searchResult.nodes.length > 0) {
      const entityUuid = searchResult.nodes[0].id;
      const entity = await adapter.getEntity(entityUuid);

      expect(entity).toBeDefined();
      expect(entity).toHaveProperty('id', entityUuid);
      expect(entity).toHaveProperty('name');
      expect(entity).toHaveProperty('labels');
      expect(entity).toHaveProperty('created_at');

      // Validate against canonical schema
      const validation = validateCanonical(CanonicalEntitySchema, entity);
      expect(validation.success).toBe(true);
    }
  });

  test('should return null for non-existent entity', async () => {
    const fakeUuid = '00000000-0000-0000-0000-000000000000';
    const entity = await adapter.getEntity(fakeUuid);

    expect(entity).toBeNull();
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Temporal Operations', () => {
  test('should query at point in time', async () => {
    const timestamp = new Date().toISOString();

    const result = await adapter.queryAtPointInTime(
      'character',
      timestamp,
      5
    );

    expect(result.edges).toBeDefined();
    expect(result.nodes).toBeDefined();
    expect(Array.isArray(result.edges)).toBe(true);
    expect(Array.isArray(result.nodes)).toBe(true);
  });

  test('should get entity timeline', async () => {
    const startTime = new Date(Date.now() - 86400000).toISOString();
    const endTime = new Date().toISOString();

    const timeline = await adapter.getEntityTimeline(
      'Sarah',
      startTime,
      endTime
    );

    expect(Array.isArray(timeline)).toBe(true);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Error Handling', () => {
  test('should handle invalid episode operation gracefully', async () => {
    const invalidOperation = {
      name: '', // Invalid: empty name
      content: 'Test content',
      episode_type: 'text',
      valid_at: 'invalid-date', // Invalid: bad date format
    } as any;

    const result = await adapter.addEpisode(invalidOperation);

    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });

  test('should handle invalid search query', async () => {
    const invalidQuery = {
      query: '', // Invalid: empty query
      temporal_filter: 'invalid_filter', // Invalid: bad filter
      max_results: -1, // Invalid: negative max_results
    } as any;

    await expect(adapter.search(invalidQuery)).rejects.toThrow();
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - UTC Compliance', () => {
  test('all timestamps should be in UTC ISO-8601 format', async () => {
    const episode: AddEpisodeOperation = {
      name: `utc-test-${generateTestId()}`,
      content: 'Test UTC timestamp compliance',
      episode_type: 'text',
      valid_at: new Date().toISOString(),
    };

    const result = await adapter.addEpisode(episode);

    expect(result.success).toBe(true);

    // Verify the episode was added with UTC timestamp
    const searchResult = await adapter.search({
      query: 'UTC timestamp compliance',
      max_results: 1,
    });

    if (searchResult.edges.length > 0) {
      const edge = searchResult.edges[0];
      expect(edge.created_at).toBeDefined();

      // Validate ISO-8601 UTC format (ends with Z)
      const timestamp = new Date(edge.created_at);
      expect(timestamp.toISOString()).toBe(edge.created_at);
    }
  });
});

describe('Graphiti Adapter - Canonical Schema Validation', () => {
  test('CanonicalEntity schema should validate correctly', () => {
    const entity: CanonicalEntity = {
      id: '550e8400-e29b-41d4-a716-446655440000',
      name: 'Test Entity',
      labels: ['Entity', 'Test'],
      summary: 'A test entity',
      created_at: '2024-01-15T10:30:00.000Z',
      attributes: {},
    };

    const result = validateCanonical(CanonicalEntitySchema, entity);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(entity);
  });

  test('AddEpisodeOperation schema should validate correctly', () => {
    const operation: AddEpisodeOperation = {
      name: 'Test Episode',
      content: 'Test episode content',
      episode_type: 'text',
      valid_at: '2024-01-15T10:30:00.000Z',
    };

    const result = validateCanonical(AddEpisodeOperationSchema, operation);
    expect(result.success).toBe(true);
  });

  test('AddTripletOperation schema should validate correctly', () => {
    const operation: AddTripletOperation = {
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

    const result = validateCanonical(AddTripletOperationSchema, operation);
    expect(result.success).toBe(true);
  });

  test('CanonicalSearchQuery schema should validate correctly', () => {
    const query: CanonicalSearchQuery = {
      query: 'test search',
      temporal_filter: 'current',
      max_results: 10,
    };

    const result = validateCanonical(CanonicalSearchQuerySchema, query);
    expect(result.success).toBe(true);
  });

  test('should reject invalid entity data', () => {
    const invalidEntity = {
      id: 'not-a-uuid', // Invalid UUID format
      name: 'Test',
      labels: ['Test'],
      created_at: '2024-01-15T10:30:00.000Z',
    };

    const result = validateCanonical(CanonicalEntitySchema, invalidEntity);
    expect(result.success).toBe(false);
    expect(result.errors).toBeDefined();
    expect(result.errors!.length).toBeGreaterThan(0);
  });

  test('should reject invalid episode operation', () => {
    const invalidOperation = {
      name: '', // Empty name not allowed
      content: 'Test',
      episode_type: 'invalid_type', // Invalid episode type
      valid_at: 'not-a-date', // Invalid date format
    };

    const result = validateCanonical(AddEpisodeOperationSchema, invalidOperation);
    expect(result.success).toBe(false);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Health Check', () => {
  test('should pass health check', async () => {
    const health = await adapter.healthCheck();

    expect(health).toHaveProperty('healthy');
    expect(health).toHaveProperty('circuit_state');
    expect(health).toHaveProperty('initialized');
    expect(health).toHaveProperty('graphiti_connected');
    expect(health.initialized).toBe(true);
    expect(health.graphiti_connected).toBe(true);
  });
});

describe.skipIf(SKIP_INTEGRATION_TESTS)('Graphiti Adapter - Cleanup', () => {
  test('should close adapter gracefully', async () => {
    await adapter.close();

    // Verify health check shows disconnected
    const health = await adapter.healthCheck();
    expect(health.initialized).toBe(false);
  });
});
