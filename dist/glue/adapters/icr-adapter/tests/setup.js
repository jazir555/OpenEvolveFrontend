"use strict";
/**
 * Test Setup
 *
 * Configure test environment and mocks
 */
// Set required environment variables for tests
process.env.OPENEVOLVE_ICR_API_URL = 'http://localhost:8080';
process.env.TIMEOUT_MS = '5000';
process.env.MAX_RETRIES = '3';
process.env.DEBUG = 'true';
// Mock console methods to reduce noise in tests
global.console = {
    ...console,
    log: jest.fn(),
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
};
//# sourceMappingURL=setup.js.map