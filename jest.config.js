/**
 * Jest Configuration
 *
 * Configuration for Jest test runner.
 * Supports both TypeScript and JavaScript tests.
 */

module.exports = {
  // Use ts-jest preset for TypeScript
  preset: 'ts-jest',

  // Test environment
  testEnvironment: 'node',

  // Roots of the project
  roots: ['<rootDir>/tests', '<rootDir>/glue'],

  // Test match patterns
  testMatch: [
    '**/__tests__/**/*.test.ts',
    '**/__tests__/**/*.test.tsx',
    '**/tests/**/*.test.ts',
    '**/tests/**/*.test.tsx',
  ],

  // Module file extensions
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],

  // Module name mapper for path aliases
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '^@glue/(.*)$': '<rootDir>/glue/$1',
  },

  // Coverage collection
  collectCoverageFrom: [
    'glue/**/*.{ts,tsx}',
    '!glue/**/*.d.ts',
    '!glue/**/*.test.ts',
    '!glue/**/*.spec.ts',
    '!glue/**/node_modules/**',
  ],

  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 60,
      functions: 60,
      lines: 60,
      statements: 60,
    },
  },

  // Coverage reporters
  coverageReporters: ['text', 'lcov', 'html'],

  // Setup files
  setupFilesAfterEnv: ['<rootDir>/tests/jest.setup.ts'],

  // Transform configuration
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        esModuleInterop: true,
        allowSyntheticDefaultImports: true,
      },
    }],
  },

  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/core-projects/',
    '/openevolve_test_env/',
  ],

  // Module directories
  moduleDirectories: ['node_modules', 'glue'],

  // Clear mocks between tests
  clearMocks: true,

  // Reset modules between tests
  resetModules: false,

  // Restore mocks after each test
  restoreMocks: false,

  // Verbose output
  verbose: true,

  // Max workers (parallel execution)
  maxWorkers: '50%',

  // Test timeout (default 5 seconds)
  testTimeout: 60000,

  // Global setup/teardown
  globalSetup: undefined,
  globalTeardown: undefined,

  // Detect open handles
  detectOpenHandles: true,

  // Force exit after tests
  forceExit: true,
};
