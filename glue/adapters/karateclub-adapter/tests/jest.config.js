/**
 * Jest Configuration for KarateClub Adapter Tests
 *
 * Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
 * - Phase 2: The Contract (Defense)
 */

module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>'],
  testMatch: ['**/*.test.ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    '../src/**/*.ts',
    '!../src/**/*.d.ts',
    '!**/node_modules/**',
  ],
  coverageDirectory: './coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  verbose: true,
  testTimeout: 60000, // 60 seconds for ML operations
  moduleNameMapper: {
    '^@schemas/(.*)$': '<rootDir>/../../schemas/$1',
  },
};
