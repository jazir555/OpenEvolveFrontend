module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.ts'],
  moduleNameMapper: {
    '^uuid$': 'uuid'
  },
  transformIgnorePatterns: [
    'node_modules/(?!(uuid)/)'
  ],
};
