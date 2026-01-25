import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import dts from 'vite-plugin-dts';
import { resolve } from 'path';

export default defineConfig({
  plugins: [
    react(),
    dts({
      include: ['src/**/*'],
      outDir: 'dist',
      insertTypesEntry: true,
      rollupTypes: false,
      compilerOptions: {
        skipLibCheck: true,
      },
    }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'OpenEvolvePlugin',
      formats: ['es', 'umd'],
      fileName: (format) => `openevolve-plugin.${format}.js`,
    },
    rollupOptions: {
      external: ['react', 'react-dom', 'zustand', '@tanstack/react-query'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          zustand: 'zustand',
          '@tanstack/react-query': 'ReactQuery',
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  test: {
    environment: 'jsdom',
  },
});
