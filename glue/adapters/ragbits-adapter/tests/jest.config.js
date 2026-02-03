/**
 * Jest Configuration for RAGBits Adapter Tests
 *
 * Following Federation Constitution test principles:
 * - FAIL FAST: Contract violations halt execution
 * - MOCK ONLY: No running instance required
 * - IDEMPOTENT: Safe to run 100 times
 */

export default {
  // Test environment
  testEnvironment: 'node',

  // Root directory for tests
  roots: ['<rootDir>'],

  // Test file patterns
  testMatch: [
    '**/tests/**/*.test.ts',
    '**/tests/**/*.spec.ts',
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

  // Coverage configuration
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },

  // Setup files
  setupFilesAfterEnv: [],

  // Module name mapper for absolute imports
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
  },

  // Transform configuration
  transform: {
    '^.+\\.ts$': ['ts-jest', {
      tsconfig: {
        jsx: 'react',
        esModuleInterop: true,
        allowSyntheticDefaultImports: true,
      },
    }],
  },

  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
  ],

  // Module directories
  moduleDirectories: [
    'node_modules',
    'src',
  ],

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Verbose output
  verbose: true,

  // Max workers (1 for simplicity)
  maxWorkers: 1,

  // Test timeout (30 seconds)
  testTimeout: 30000,
};
