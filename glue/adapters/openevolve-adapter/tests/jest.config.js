/**
 * Jest Configuration for OpenEvolve Adapter Contract Tests
 *
 * This configuration is specifically designed for testing the OpenEvolve
 * orchestration adapter's API contracts and integration coordination.
 */

module.exports = {
  // Test environment
  testEnvironment: 'node',

  // Root directories
  roots: ['<rootDir>/tests'],

  // Test file patterns
  testMatch: [
    '**/__tests__/**/*.test.ts',
    '**/?(*.)+(spec|test).ts',
  ],

  // TypeScript transformation
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        tsconfig: '<rootDir>/tsconfig.json',
      },
    ],
  },

  // Module name mappings for clean imports
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
  },

  // Coverage collection
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.interface.ts',
    '!src/index.ts',
  ],

  // Coverage reporters
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html', 'json'],

  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },

  // Timeout for tests (important for API calls)
  testTimeout: 30000,

  // Setup files
  setupFilesAfterEnv: [],
  globalSetup: undefined,
  globalTeardown: undefined,

  // Verbose output
  verbose: true,

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Module directories
  moduleDirectories: ['node_modules', '<rootDir>/src'],

  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
  ],

  // Coverage path ignore patterns
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/tests/',
  ],

  // Max workers (1 for sequential execution, useful for integration tests)
  maxWorkers: 1,

  // Globals
  globals: {
    'ts-jest': {
      isolatedModules: true,
    },
  },

  // Display configuration
  displayName: {
    name: 'OpenEvolve Adapter',
    color: 'cyan',
  },

  // Cache configuration
  cache: true,
  cacheDirectory: '<rootDir>/.jest-cache',

  // Detect open handles
  detectOpenHandles: true,
  forceExit: true,
};
