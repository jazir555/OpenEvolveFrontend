import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
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