import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/datasets': {
        target: 'http://192.168.1.190:8000',
        changeOrigin: true,
      },
      '/train_full': {
        target: 'http://192.168.1.190:8000',
        changeOrigin: true,
      },
      '/stop_training': {
        target: 'http://192.168.1.190:8000',
        changeOrigin: true,
      },
      '/model': {
        target: 'http://192.168.1.190:8000',
        changeOrigin: true,
      },
      '/predict': {
        target: 'http://192.168.1.190:8000',
        changeOrigin: true,
      },
    }
  }
})
