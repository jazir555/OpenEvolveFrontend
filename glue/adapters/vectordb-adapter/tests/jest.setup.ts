/**
 * Jest Setup Configuration
 *
 * Global test setup for Vector DB adapter contract tests.
 */

// Set test environment variables
process.env.VECTORDB_TYPE = 'qdrant';
process.env.VECTORDB_URL = 'http://localhost:6333';
process.env.TIMEOUT_MS = '5000';

// Increase timeout for integration tests
jest.setTimeout(30000);
