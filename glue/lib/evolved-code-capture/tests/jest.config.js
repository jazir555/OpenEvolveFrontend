/**
 * Jest Configuration for Evolved Code Capture Tests
 *
 * Following CLAUDE.md Federation Constitution:
 * - Contract tests run on container startup
 * - If contract is violated, adapter refuses to start
 */

module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/../src', '<rootDir>'],
  testMatch: [
    '**/__tests__/**/*.test.ts',
    '**/?(*.)+(spec|test).ts',
  ],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    '../src/**/*.ts',
    '!../src/**/*.d.ts',
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  testTimeout: 30000,
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/../src/$1',
    '^@tests/(.*)$': '<rootDir>/$1',
    '^@lib/(.*)$': '<rootDir>/../../$1',
  },
  globals: {
    'ts-jest': {
      isolatedModules: true,
    },
  },
};
