/**
 * Jest Setup File
 *
 * Global setup for Jest tests.
 * This file is run before each test file.
 */

export {};

// Extend global interface for test utilities
declare global {
  namespace NodeJS {
    interface Global {
      testUtils: {
        createTestProblem: (overrides?: any) => any;
        wait: (ms: number) => Promise<void>;
        retry: <T>(fn: () => Promise<T>, maxRetries?: number, delay?: number) => Promise<T>;
      };
    }
  }
}

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error'; // Reduce noise during tests
process.env.SKIP_SLOW_TESTS = 'true';
process.env.ENABLE_KNOWLEDGE_TESTS = 'false';

// Mock console methods to reduce noise (optional)
const originalError = console.error;
console.error = (...args: any[]) => {
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
(global as any).testUtils = {
  // Create a test problem
  createTestProblem: (overrides: any = {}) => ({
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
  wait: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)),

  // Retry helper
  retry: async <T>(
    fn: () => Promise<T>,
    maxRetries = 3,
    delay = 100
  ): Promise<T> => {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
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
