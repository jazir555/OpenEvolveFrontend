/**
 * BubbleLab Load Testing Suite
 *
 * Tool: k6 (https://k6.io/)
 * Target: 1000 requests/minute (16.67 requests/second)
 *
 * Prerequisites:
 * 1. Install k6: https://k6.io/docs/getting-started/installation/
 * 2. Set environment variables:
 *    - API_BASE_URL
 *    - API_KEY (if authentication required)
 *
 * Run tests:
 *   k6 run tests/load/k6-load-test.js
 *
 * Run with specific options:
 *   k6 run --vus 10 --duration 5m tests/load/k6-load-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const requestCount = new Counter('requests');

// Test configuration
export const options = {
  scenarios: {
    // Normal load test: 100 req/s for 5 minutes
    normal_load: {
      executor: 'constant-arrival-rate',
      rate: 100, // 100 requests per second
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 50,
      maxVUs: 100,
      exec: 'normalLoadTest',
    },

    // Peak load test: 500 req/s for 2 minutes
    peak_load: {
      executor: 'constant-arrival-rate',
      rate: 500, // 500 requests per second
      timeUnit: '1s',
      duration: '2m',
      preAllocatedVUs: 100,
      maxVUs: 200,
      startTime: '5m',
      exec: 'peakLoadTest',
    },

    // Stress test: 1000 req/s for 1 minute
    stress_test: {
      executor: 'constant-arrival-rate',
      rate: 1000, // 1000 requests per second (target)
      timeUnit: '1s',
      duration: '1m',
      preAllocatedVUs: 200,
      maxVUs: 400,
      startTime: '7m',
      exec: 'stressTest',
    },

    // Soak test: 50 req/s for 30 minutes
    soak_test: {
      executor: 'constant-arrival-rate',
      rate: 50, // 50 requests per second
      timeUnit: '1s',
      duration: '30m',
      preAllocatedVUs: 50,
      maxVUs: 100,
      startTime: '8m',
      exec: 'soakTest',
    },
  },

  thresholds: {
    // Assert that 95% of requests finish within 500ms
    'http_req_duration': ['p(95)<500'],

    // Assert that error rate is below 1%
    'errors': ['rate<0.01'],

    // Assert that 99% of requests are successful
    'checks': ['rate>0.99'],
  },
};

// Base URL from environment
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:3000';
const API_KEY = __ENV.API_KEY || '';

// Headers
const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_KEY}`,
};

/**
 * Service Bubbles Load Tests
 */

// Qdrant Bubble Tests
export function qdrantTests() {
  const collectionName = `test-collection-${Math.random().toString(36).substr(2, 9)}`;

  // Create collection
  const createPayload = JSON.stringify({
    vectors: {
      size: 1536,
      distance: 'Cosine',
    },
  });

  const createRes = http.post(
    `${BASE_URL}/api/bubbles/qdrant/collections`,
    createPayload,
    { headers }
  );

  check(createRes, {
    'Qdrant create collection status 201': (r) => r.status === 201,
    'Qdrant create collection response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(createRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Insert points
  const insertPayload = JSON.stringify({
    points: [
      {
        id: 1,
        vector: Array(1536).fill(0.1),
        payload: { test: 'data' },
      },
    ],
  });

  const insertRes = http.put(
    `${BASE_URL}/api/bubbles/qdrant/collections/${collectionName}/points`,
    insertPayload,
    { headers }
  );

  check(insertRes, {
    'Qdrant insert points status 200': (r) => r.status === 200,
    'Qdrant insert points response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(insertRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Search points
  const searchPayload = JSON.stringify({
    vector: Array(1536).fill(0.1),
    limit: 10,
  });

  const searchRes = http.post(
    `${BASE_URL}/api/bubbles/qdrant/collections/${collectionName}/points/search`,
    searchPayload,
    { headers }
  );

  check(searchRes, {
    'Qdrant search status 200': (r) => r.status === 200,
    'Qdrant search response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(searchRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Delete collection
  const deleteRes = http.del(
    `${BASE_URL}/api/bubbles/qdrant/collections/${collectionName}`,
    null,
    { headers }
  );

  check(deleteRes, {
    'Qdrant delete collection status 200': (r) => r.status === 200,
    'Qdrant delete collection response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(deleteRes.timings.duration);
  requestCount.add(1);
}

// Elasticsearch Bubble Tests
export function elasticsearchTests() {
  const indexName = `test-index-${Math.random().toString(36).substr(2, 9)}`;

  // Create index
  const createRes = http.put(
    `${BASE_URL}/api/bubbles/elasticsearch/indices/${indexName}`,
    JSON.stringify({
      mappings: {
        properties: {
          title: { type: 'text' },
          content: { type: 'text' },
        },
      },
    }),
    { headers }
  );

  check(createRes, {
    'Elasticsearch create index status 200': (r) => r.status === 200,
    'Elasticsearch create index response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(createRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Index document
  const indexDocRes = http.post(
    `${BASE_URL}/api/bubbles/elasticsearch/indices/${indexName}/docs`,
    JSON.stringify({
      title: 'Test Document',
      content: 'This is a test document for load testing',
    }),
    { headers }
  );

  check(indexDocRes, {
    'Elasticsearch index doc status 201': (r) => r.status === 201,
    'Elasticsearch index doc response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(indexDocRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Search
  const searchRes = http.post(
    `${BASE_URL}/api/bubbles/elasticsearch/indices/${indexName}/search`,
    JSON.stringify({
      query: {
        match: { content: 'test' },
      },
    }),
    { headers }
  );

  check(searchRes, {
    'Elasticsearch search status 200': (r) => r.status === 200,
    'Elasticsearch search response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(searchRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Delete index
  const deleteRes = http.del(
    `${BASE_URL}/api/bubbles/elasticsearch/indices/${indexName}`,
    null,
    { headers }
  );

  check(deleteRes, {
    'Elasticsearch delete index status 200': (r) => r.status === 200,
    'Elasticsearch delete index response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(deleteRes.timings.duration);
  requestCount.add(1);
}

// Redis Bubble Tests
export function redisTests() {
  const key = `test-key-${Math.random().toString(36).substr(2, 9)}`;

  // Set value
  const setRes = http.post(
    `${BASE_URL}/api/bubbles/redis/set`,
    JSON.stringify({ key, value: 'test-value', ttl: 60 }),
    { headers }
  );

  check(setRes, {
    'Redis set status 200': (r) => r.status === 200,
    'Redis set response time < 100ms': (r) => r.timings.duration < 100,
  }) || errorRate.add(1);

  responseTime.add(setRes.timings.duration);
  requestCount.add(1);

  sleep(0.5);

  // Get value
  const getRes = http.get(
    `${BASE_URL}/api/bubbles/redis/get?key=${encodeURIComponent(key)}`,
    { headers }
  );

  check(getRes, {
    'Redis get status 200': (r) => r.status === 200,
    'Redis get response time < 100ms': (r) => r.timings.duration < 100,
  }) || errorRate.add(1);

  responseTime.add(getRes.timings.duration);
  requestCount.add(1);

  sleep(0.5);

  // Delete value
  const delRes = http.del(
    `${BASE_URL}/api/bubbles/redis/delete?key=${encodeURIComponent(key)}`,
    null,
    { headers }
  );

  check(delRes, {
    'Redis delete status 200': (r) => r.status === 200,
    'Redis delete response time < 100ms': (r) => r.timings.duration < 100,
  }) || errorRate.add(1);

  responseTime.add(delRes.timings.duration);
  requestCount.add(1);
}

// PostgreSQL Bubble Tests
export function postgresqlTests() {
  const tableName = `test_table_${Math.random().toString(36).substr(2, 9)}`;

  // Create table
  const createTableRes = http.post(
    `${BASE_URL}/api/bubbles/postgresql/execute`,
    JSON.stringify({
      query: `CREATE TABLE ${tableName} (id SERIAL PRIMARY KEY, name VARCHAR(255), data JSONB)`,
    }),
    { headers }
  );

  check(createTableRes, {
    'PostgreSQL create table status 200': (r) => r.status === 200,
    'PostgreSQL create table response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(createTableRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Insert row
  const insertRes = http.post(
    `${BASE_URL}/api/bubbles/postgresql/execute`,
    JSON.stringify({
      query: `INSERT INTO ${tableName} (name, data) VALUES ($1, $2)`,
      params: ['test', { key: 'value' }],
    }),
    { headers }
  );

  check(insertRes, {
    'PostgreSQL insert status 200': (r) => r.status === 200,
    'PostgreSQL insert response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(insertRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Query
  const queryRes = http.post(
    `${BASE_URL}/api/bubbles/postgresql/query`,
    JSON.stringify({
      query: `SELECT * FROM ${tableName}`,
    }),
    { headers }
  );

  check(queryRes, {
    'PostgreSQL query status 200': (r) => r.status === 200,
    'PostgreSQL query response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(queryRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Drop table
  const dropTableRes = http.post(
    `${BASE_URL}/api/bubbles/postgresql/execute`,
    JSON.stringify({
      query: `DROP TABLE ${tableName}`,
    }),
    { headers }
  );

  check(dropTableRes, {
    'PostgreSQL drop table status 200': (r) => r.status === 200,
    'PostgreSQL drop table response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(dropTableRes.timings.duration);
  requestCount.add(1);
}

/**
 * Workflow Load Tests
 */

export function workflowTests() {
  const workflowId = Math.random().toString(36).substr(2, 9);

  // Create workflow
  const createRes = http.post(
    `${BASE_URL}/api/workflows`,
    JSON.stringify({
      id: workflowId,
      name: `Test Workflow ${workflowId}`,
      bubbles: [
        {
          id: 'test-bubble',
          type: 'service',
          service: 'qdrant',
        },
      ],
    }),
    { headers }
  );

  check(createRes, {
    'Workflow create status 201': (r) => r.status === 201,
    'Workflow create response time < 1000ms': (r) => r.timings.duration < 1000,
  }) || errorRate.add(1);

  responseTime.add(createRes.timings.duration);
  requestCount.add(1);

  sleep(2);

  // Execute workflow
  const executeRes = http.post(
    `${BASE_URL}/api/workflows/${workflowId}/execute`,
    JSON.stringify({
      input: { test: 'data' },
    }),
    { headers }
  );

  check(executeRes, {
    'Workflow execute status 200': (r) => r.status === 200,
    'Workflow execute response time < 5000ms': (r) => r.timings.duration < 5000,
  }) || errorRate.add(1);

  responseTime.add(executeRes.timings.duration);
  requestCount.add(1);

  sleep(5);

  // Get workflow status
  const statusRes = http.get(
    `${BASE_URL}/api/workflows/${workflowId}/status`,
    { headers }
  );

  check(statusRes, {
    'Workflow status 200': (r) => r.status === 200,
    'Workflow status response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(statusRes.timings.duration);
  requestCount.add(1);

  sleep(1);

  // Delete workflow
  const deleteRes = http.del(
    `${BASE_URL}/api/workflows/${workflowId}`,
    null,
    { headers }
  );

  check(deleteRes, {
    'Workflow delete status 200': (r) => r.status === 200,
    'Workflow delete response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  responseTime.add(deleteRes.timings.duration);
  requestCount.add(1);
}

/**
 * Connection Pooling Tests
 */

export function connectionPoolingTests() {
  // Rapid consecutive requests to test connection pooling
  const promises = [];

  for (let i = 0; i < 10; i++) {
    promises.push(
      http.get(`${BASE_URL}/api/bubbles/redis/health`, { headers })
    );
  }

  const responses = Promise.all(promises);

  responses.forEach((res) => {
    check(res, {
      'Connection pool request status 200': (r) => r.status === 200,
      'Connection pool response time < 100ms': (r) => r.timings.duration < 100,
    }) || errorRate.add(1);

    responseTime.add(res.timings.duration);
    requestCount.add(1);
  });

  sleep(1);
}

/**
 * Scenario Executors
 */

export function normalLoadTest() {
  // Test all service bubbles under normal load
  qdrantTests();
  elasticsearchTests();
  redisTests();
  postgresqlTests();
  workflowTests();
  connectionPoolingTests();

  sleep(Math.random() * 3); // Random think time 0-3 seconds
}

export function peakLoadTest() {
  // Focus on critical operations during peak load
  redisTests();
  qdrantTests();
  workflowTests();

  sleep(Math.random() * 2); // Random think time 0-2 seconds
}

export function stressTest() {
  // Maximum load on most critical operations
  redisTests();
  connectionPoolingTests();

  sleep(Math.random() * 1); // Random think time 0-1 seconds
}

export function soakTest() {
  // Normal operations over extended period
  redisTests();
  qdrantTests();
  elasticsearchTests();
  postgresqlTests();

  sleep(Math.random() * 5); // Random think time 0-5 seconds
}

/**
 * Setup and Teardown
 */

export function setup() {
  // Setup tasks before test starts
  console.log(`Starting load test against ${BASE_URL}`);
  console.log(`Test start time: ${new Date().toISOString()}`);

  return {
    startTime: new Date().toISOString(),
  };
}

export function teardown(data) {
  // Cleanup tasks after test ends
  console.log(`Test end time: ${new Date().toISOString()}`);
  console.log(`Total requests: ${requestCount}`);
  console.log(`Error rate: ${errorRate}`);
  console.log(`Average response time: ${responseTime}`);
}
