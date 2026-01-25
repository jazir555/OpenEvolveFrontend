import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['**/*.test.ts'],
    globals: true,
    testTimeout: 60000,
    hookTimeout: 120000,
    teardownTimeout: 120000,
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: false,
        isolate: true,
      },
    },
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/**',
        'dist/**',
        'coverage/**',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/types/**',
        '**/templates/**',
        'vitest.config.ts',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
      perFile: false,
    },
    reporters: ['default', 'json', 'html'],
    outputFile: {
      json: './test-results/reliability-test-results.json',
      html: './test-results/reliability-test-report.html',
    },
    retry: 2,
    silent: false,
    verbose: true,
    watch: false,
  },
  resolve: {
    alias: {
      '^@bubblelab/shared-schemas$': new URL(
        '../BubbleLab/packages/bubble-shared-schemas/src/index.ts',
        import.meta.url
      ).pathname,
    },
  },
});
