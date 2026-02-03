/**
 * Jest Configuration for Graphiti Adapter Tests
 *
 * Following CLAUDE.md principles:
 * - Configuration explicitness via environment variables
 * - Runtime truth: verify actual adapter behavior
 */

module.exports = {
  // Test environment
  testEnvironment: 'node',

  // Root directory for tests
  rootDir: '..',

  // Test file patterns
  testMatch: [
    '**/tests/**/*.test.ts',
  ],

  // TypeScript configuration
  preset: 'ts-jest',
  extensionsToTreatAsEsm: ['.ts'],
  globals: {
    'ts-jest': {
      useESM: true,
      tsconfig: {
        esModuleInterop: true,
        allowSyntheticDefaultImports: true,
      },
    },
  },

  // Module name mapper for path aliases
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },

  // Coverage configuration
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 60,
      functions: 60,
      lines: 60,
      statements: 60,
    },
  },

  // Setup files
  setupFilesAfterEnv: [],

  // Timeout for integration tests (milliseconds)
  testTimeout: 30000,

  // Verbose output
  verbose: true,

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Transform configuration
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: true,
      },
    ],
  },
};
