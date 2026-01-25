import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
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
