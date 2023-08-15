import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/events/': {
        target: 'ws://localhost:9000',
        ws: true,
      },
      '/api': {
        target: 'http://localhost:9000',
      }
    }
  }
})
