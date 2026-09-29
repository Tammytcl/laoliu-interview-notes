import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser',
  use: { baseURL: 'http://127.0.0.1:4175', viewport: { width: 1440, height: 1000 } },
  webServer: [
    { command: 'npm run build && PORT=4175 npm run preview', url: 'http://127.0.0.1:4175', reuseExistingServer: false },
    { command: 'PORT=4176 BASE_PATH=/interview-notes npm run preview', url: 'http://127.0.0.1:4176/interview-notes/', reuseExistingServer: false }
  ]
});
