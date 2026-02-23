"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const config_1 = require("vitest/config");
exports.default = (0, config_1.defineConfig)({
    test: {
        globals: true,
        environment: 'node',
        setupFiles: ['src/tests/setup.ts'],
        coverage: {
            provider: 'v8',
            reporter: ['text', 'json', 'html'],
            exclude: [
                'node_modules/',
                'dist/',
                '**/*.test.ts',
                '**/*.test.tsx',
            ],
        },
        testTimeout: 30000, // 30 second timeout for API calls
        hookTimeout: 30000,
        teardownTimeout: 10000,
    },
});
//# sourceMappingURL=vitest.config.js.map