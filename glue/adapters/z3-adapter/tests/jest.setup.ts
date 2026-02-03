/**
 * Jest Setup File for Z3 Adapter Contract Tests
 *
 * This file runs before all test suites.
 * Configure global test utilities and mocks here.
 */

import { jest } from '@jest/globals';

// Global test timeout for async operations
jest.setTimeout(10000);

// Mock console methods to reduce noise in test output
global.console = {
  ...console,
  // Uncomment to silence console.log during tests
  // log: jest.fn(),
  // debug: jest.fn(),
  // info: jest.fn(),
};

// Add custom matchers if needed
// expect.extend({
//   toBeValidISO8601(received: string) {
//     const pass = !isNaN(Date.parse(received));
//     return {
//       pass,
//       message: () => `expected ${received} to be valid ISO 8601 timestamp`,
//     };
//   },
// });

// Global test utilities
global.testUtils = {
  generateCorrelationId: () => `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,

  generateUTCDate: () => new Date().toISOString(),

  sleep: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)),
};

console.log('✅ Z3 Adapter Contract Tests Setup Complete');
