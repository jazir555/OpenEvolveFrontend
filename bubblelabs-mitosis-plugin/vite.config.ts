import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import dts from 'vite-plugin-dts'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    react(),
    dts({
      insertTypesEntry: true,
    }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'BubbleLabMitosisPlugin',
      fileName: (format) => `index.${format}.js`,
      formats: ['es', 'cjs', 'umd'],
    },
    rollupOptions: {
      external: ['react', 'react-dom', 'zustand'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          zustand: 'zustand',
        },
        assetFileNames: (assetInfo) => {
          if (assetInfo.name === 'style.css') return 'mitosis-animations.css';
          return assetInfo.name;
        }
      },
    },
  },
})