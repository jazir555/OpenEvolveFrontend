"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vite_1 = require("vite");
const plugin_react_1 = __importDefault(require("@vitejs/plugin-react"));
const path_1 = require("path");
exports.default = (0, vite_1.defineConfig)({
    plugins: [(0, plugin_react_1.default)()],
    build: {
        lib: {
            entry: (0, path_1.resolve)(__dirname, 'src/index.ts'),
            name: 'RAGBitsPlugin',
            formats: ['es', 'cjs'],
            fileName: (format) => `index.${format === 'es' ? 'mjs' : 'js'}`
        },
        rollupOptions: {
            external: ['react', 'react-dom', 'lucide-react', 'react-toastify', 'zustand', 'react-hook-form'],
            output: {
                globals: {
                    react: 'React',
                    'react-dom': 'ReactDOM',
                    'lucide-react': 'LucideReact',
                    'react-toastify': 'ReactToastify',
                    zustand: 'Zustand',
                    'react-hook-form': 'ReactHookForm'
                }
            }
        }
    }
});
//# sourceMappingURL=vite.config.js.map