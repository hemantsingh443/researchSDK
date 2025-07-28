import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    hmr: {
      host: 'localhost',
      port: 3000,
    },
    watch: {
      usePolling: true,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  define: {
    'process.env': {}
  }
});
