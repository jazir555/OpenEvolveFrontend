"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vite_1 = require("vite");
const plugin_react_1 = __importDefault(require("@vitejs/plugin-react"));
const path_1 = require("path");
// https://vitejs.dev/config/
exports.default = (0, vite_1.defineConfig)({
    plugins: [(0, plugin_react_1.default)()],
    resolve: {
        alias: {
            '@': (0, path_1.resolve)(__dirname, 'src')
        }
    },
    build: {
        lib: {
            entry: (0, path_1.resolve)(__dirname, 'src/index.ts'),
            name: 'RomaBubblelabPlugin',
            fileName: (format) => `roma-bubblelab-plugin.${format}.js`
        },
        rollupOptions: {
            external: ['react', 'react-dom', 'zustand', 'axios', 'react-toastify', 'lucide-react'],
            output: {
                globals: {
                    react: 'React',
                    'react-dom': 'ReactDOM',
                    zustand: 'zustand',
                    axios: 'axios',
                    'react-toastify': 'ReactToastify',
                    'lucide-react': 'LucideReact'
                }
            }
        },
        sourcemap: true,
        emptyOutDir: true
    },
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: './src/setupTests.ts',
        coverage: {
            reporter: ['text', 'json', 'html']
        }
    },
    esbuild: {
        jsxFactory: 'React.createElement',
        jsxFragment: 'React.Fragment'
    }
});
//# sourceMappingURL=vite.config.js.map