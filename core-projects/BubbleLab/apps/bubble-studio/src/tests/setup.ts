// Mock environment variables for testing
const mockEnv = {
  MODE: 'test',
  VITE_API_URL: '',
  VITE_API_ENDPOINT: '',
  VITE_EVOLUTION_API_URL: '',
  VITE_GATEWAY_URL: '',
  VITE_CLERK_PUBLISHABLE_KEY: '',
  VITE_SHOW_LEGACY_PARAMS: 'false',
  VITE_DISABLE_AUTH: 'false',
  VITE_POSTHOG_API_KEY: '',
  VITE_POSTHOG_HOST: 'https://us.i.posthog.com',
  VITE_ANALYTICS_ENABLED: 'false',
};

// Setup import.meta.env mock
Object.defineProperty(import.meta, 'env', {
  value: mockEnv,
  writable: true,
});

// Mock console methods to reduce noise
const originalWarn = console.warn;
const originalError = console.error;

beforeEach(() => {
  console.warn = vi.fn(originalWarn);
  console.error = vi.fn(originalError);
});

afterEach(() => {
  console.warn = originalWarn;
  console.error = originalError;
  // Reset import.meta.env to defaults
  Object.assign(import.meta.env, mockEnv);
});
