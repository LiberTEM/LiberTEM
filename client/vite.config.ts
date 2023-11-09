import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

const API_PORT = process.env.API_PORT ? process.env.API_PORT : 9000;
const API_HOST = process.env.API_HOST ? process.env.API_HOST : 'localhost';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '',  // use relative paths when referencing assets
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './tests/setup.ts',
  },
  server: {
    proxy: {
      '/api/events/': {
        target: `ws://${API_HOST}:${API_PORT}`,
        ws: true,
      },
      '/api': {
        target: `http://${API_HOST}:${API_PORT}`,
      }
    }
  }
})
