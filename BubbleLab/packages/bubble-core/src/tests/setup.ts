/**
 * Test Setup File
 * Global test configuration and mocks
 */

import { vi } from 'vitest';

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  error: vi.fn(),
  warn: vi.fn(),
  log: vi.fn(),
  info: vi.fn(),
  debug: vi.fn(),
};

// Mock environment variables
process.env.NODE_ENV = 'test';

// Set default timeout for tests
vi.setConfig({ testTimeout: 60000 });

// Global test utilities
export {};
