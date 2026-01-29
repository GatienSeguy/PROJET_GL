import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// const URL = 'http://92.168.1.190:8000'
const URL = 'http://192.168.1.94:8000'
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
