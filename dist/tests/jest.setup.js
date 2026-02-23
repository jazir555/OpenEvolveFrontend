"use strict";
/**
 * Jest Setup File
 *
 * Global setup for Jest tests.
 * This file is run before each test file.
 */
Object.defineProperty(exports, "__esModule", { value: true });
// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error'; // Reduce noise during tests
process.env.SKIP_SLOW_TESTS = 'true';
process.env.ENABLE_KNOWLEDGE_TESTS = 'false';
// Mock console methods to reduce noise (optional)
const originalError = console.error;
console.error = (...args) => {
    // Still show actual errors, but filter out expected warnings
    const msg = args[0];
    if (typeof msg === 'string') {
        // Suppress certain expected warnings
        if (msg.includes('ExperimentalWarning') || msg.includes('Warning:')) {
            return;
        }
    }
    originalError.call(console, ...args);
};
// Increase timeout for integration tests
jest.setTimeout(60000);
// Global test utilities
global.testUtils = {
    // Create a test problem
    createTestProblem: (overrides = {}) => ({
        id: crypto.randomUUID(),
        type: 'optimization',
        description: 'Test optimization problem',
        context: {},
        constraints: [],
        success_criteria: [],
        created_at: new Date().toISOString(),
        ...overrides,
    }),
    // Wait for async operations
    wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
    // Retry helper
    retry: async (fn, maxRetries = 3, delay = 100) => {
        for (let i = 0; i < maxRetries; i++) {
            try {
                return await fn();
            }
            catch (error) {
                if (i === maxRetries - 1)
                    throw error;
                await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
            }
        }
        throw new Error('Retry failed');
    },
};
// Teardown
afterEach(() => {
    // Clean up any global state between tests
    jest.clearAllMocks();
});
//# sourceMappingURL=jest.setup.js.map