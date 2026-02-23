/**
 * ROMA BubbleLab Plugin - Test Setup
 *
 * Configures Vitest/Jest for ROMA plugin contract testing.
 */

import { vi } from 'vitest';

// Mock environment variables
process.env.ROMA_SERVER_URL = 'http://localhost:8000';
process.env.ROMA_API_KEY = 'test-api-key';
process.env.ROMA_TIMEOUT = '5000';

// Mock console methods to reduce test noise
global.console = {
  ...console,
  error: vi.fn(),
  warn: vi.fn(),
  info: vi.fn(),
  debug: vi.fn(),
};

// Mock performance API for tests
global.performance = {
  ...global.performance,
  now: vi.fn(() => Date.now()),
} as any;

// Setup axios mock defaults
vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    patch: vi.fn(),
    create: vi.fn(() => ({
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
      patch: vi.fn(),
      interceptors: {
        request: {
          use: vi.fn()
        },
        response: {
          use: vi.fn()
        },
      },
    })),
  },
}));

// Mock React and related UI libraries for hook tests
vi.mock('react', () => ({
  ...vi.importActual('react'),
  useState: vi.fn(),
  useEffect: vi.fn(),
  useMemo: vi.fn(),
  useCallback: vi.fn(),
  useRef: vi.fn(),
}));

vi.mock('react-toastify', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
  ToastContainer: vi.fn(),
}));

// Mock setTimeout/clearTimeout for retry tests
global.setTimeout = vi.fn((fn, delay) => {
  return (fn as any)() as any;
}) as any;
global.clearTimeout = vi.fn();

// Mock Date for consistent testing
const mockDate = new Date('2026-02-22T00:00:00Z');
global.Date = vi.fn(() => mockDate) as any;
global.Date.now = vi.fn(() => mockDate.getTime());

export {};