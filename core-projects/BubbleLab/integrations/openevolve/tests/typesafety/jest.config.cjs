/**
 * Jest Configuration for Type Safety Tests
 */

module.exports = {
  displayName: 'typesafety',
  testEnvironment: 'node',
  testMatch: ['**/*.test.ts'],
  roots: ['<rootDir>'],
  collectCoverageFrom: [
    '**/*.{ts,tsx}',
    '!**/*.test.ts',
    '!**/*.config.ts',
    '!**/*.config.cjs',
    '!node_modules/**',
    '!dist/**',
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  moduleNameMapper: {
    '^zod$': '<rootDir>/../../../../node_modules/zod',
    '^@/(.*)$': '<rootDir>/$1',
  },
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: {
        esModuleInterop: true,
        allowSyntheticDefaultImports: true,
        module: 'commonjs',
      },
    }],
  },
};
