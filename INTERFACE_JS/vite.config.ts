import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const URL = 'http://138.231.149.81:8002'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/datasets': {
        target: URL,
        changeOrigin: true,
      },
      '/train_full': {
        target: URL,
        changeOrigin: true,
      },
      '/stop_training': {
        target: URL,
        changeOrigin: true,
      },
      '/model': {
        target: URL,
        changeOrigin: true,
      },
      '/predict': {
        target: URL,
        changeOrigin: true,
      },
    }
  }
})
