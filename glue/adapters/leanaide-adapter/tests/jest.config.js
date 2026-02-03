/**
 * Jest Configuration for LeanAide Contract Tests
 *
 * Following the Federation Constitution's testing standards:
 * - TypeScript support
 * - Strict mode enabled
 * - Coverage thresholds enforced
 * - Fail-fast on errors
 */

module.exports = {
  // Test environment
  testEnvironment: 'node',

  // Root directory for tests
  rootDir: '.',

  // Test file patterns
  testMatch: [
    '**/tests/**/*.test.ts',
    '**/__tests__/**/*.test.ts',
  ],

  // TypeScript configuration
  preset: 'ts-jest/presets/default-esm',
  extensionsToTreatAsEsm: ['.ts'],

  // Module paths
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/../../../$1',
    '^@tests/(.*)$': '<rootDir>/$1',
  },

  // Transform configuration
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: true,
        tsconfig: {
          esModuleInterop: true,
          allowSyntheticDefaultImports: true,
        },
      },
    ],
  },

  // Coverage configuration
  collectCoverageFrom: [
    '../src/**/*.ts',
    '!../src/**/*.d.ts',
    '!../src/**/index.ts',
  ],

  // Coverage thresholds (enforced)
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },

  // Coverage reporters
  coverageReporters: [
    'json',
    'lcov',
    'text',
    'text-summary',
    'html',
  ],

  // Output configuration
  verbose: true,
  bail: false, // Don't stop on first failure (run all tests)

  // Timeout configuration (Law of Configuration Explicitness)
  testTimeout: parseInt(process.env.JEST_TIMEOUT_MS || '30000', 10),

  // Setup files
  setupFilesAfterEnv: [],

  // Global configuration
  globals: {
    'ts-jest': {
      isolatedModules: true,
    },
  },

  // Error handling
  errorOnDeprecated: true,
  maxWorkers: '50%', // Use half of available CPUs

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Test result formatting
  reporters: [
    'default',
    [
      'jest-junit',
      {
        outputDirectory: './test-results',
        outputName: 'junit.xml',
        classNameTemplate: '{classname}',
        titleTemplate: '{title}',
        ancestorSeparator: ' › ',
        usePathForSuiteName: true,
      },
    ],
  ],
};
