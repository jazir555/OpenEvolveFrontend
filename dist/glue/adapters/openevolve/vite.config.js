"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vite_1 = require("vite");
const plugin_react_1 = __importDefault(require("@vitejs/plugin-react"));
const path_1 = require("path");
const vite_plugin_dts_1 = __importDefault(require("vite-plugin-dts"));
// https://vitejs.dev/config/
exports.default = (0, vite_1.defineConfig)({
    plugins: [
        (0, plugin_react_1.default)(),
        (0, vite_plugin_dts_1.default)({
            include: ['src/**/*'],
            exclude: ['**/*.test.ts', '**/*.test.tsx'],
        }),
    ],
    build: {
        lib: {
            entry: (0, path_1.resolve)(__dirname, 'src/index.ts'),
            name: 'OpenEvolveBubblelabPlugin',
            fileName: (format) => `index.${format}.js`,
        },
        rollupOptions: {
            external: ['react', 'react-dom', 'react-toastify', 'uuid', 'axios', 'zustand'],
            output: {
                globals: {
                    react: 'React',
                    'react-dom': 'ReactDOM',
                    'react-toastify': 'ReactToastify',
                    uuid: 'uuid',
                    axios: 'axios',
                    zustand: 'zustand',
                },
            },
        },
        sourcemap: true,
        emptyOutDir: true,
    },
    resolve: {
        alias: {
            '@': (0, path_1.resolve)(__dirname, 'src'),
        },
    },
    server: {
        port: 3001,
        open: false,
    },
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: './src/test/setup.ts',
        coverage: {
            reporter: ['text', 'json', 'html'],
        },
    },
});
//# sourceMappingURL=vite.config.js.map