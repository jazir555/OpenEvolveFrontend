/**
 * Test Utilities
 * Helper functions for testing
 */

import { ReactElement } from 'react';

// Type declarations for vitest globals
declare const vi: {
  fn: (impl?: () => any) => any;
  spyOn: () => any;
  clearAllMocks: () => void;
  resetAllMocks: () => void;
  restoreAllMocks: () => void;
};

// Type declarations for test globals
declare const beforeAll: (fn: () => void) => void;
declare const afterAll: (fn: () => void) => void;
declare const beforeEach: (fn: () => void) => void;
declare const afterEach: (fn: () => void) => void;
declare const describe: (name: string, fn: () => void) => void;
declare const it: (name: string, fn: () => void) => void;
declare const test: (name: string, fn: () => void) => void;
declare const expect: any;

import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

/**
 * Custom render function with providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  function AllTheProviders({ children }: { children: React.ReactNode }) {
    return <BrowserRouter>{children}</BrowserRouter>;
  }

  return render(ui, { wrapper: AllTheProviders, ...options });
}

/**
 * Wait for async operations
 */
export async function wait(ms: number = 0): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait for component to update
 */
export async function waitForChange<T>(
  getter: () => T,
  options?: { timeout?: number }
): Promise<T> {
  const { timeout = 1000 } = options || {};
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    try {
      const result = getter();
      if (result) return result;
    } catch (e) {
      // Continue waiting
    }
    await wait(10);
  }

  throw new Error(`Timeout waiting for condition after ${timeout}ms`);
}

/**
 * Mock API response
 */
export function createMockResponse<T>(data: T, delay = 0): Promise<Response> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        ok: true,
        json: async () => data,
      } as Response);
    }, delay);
  });
}

/**
 * Mock API error
 */
export function createMockError(
  message: string,
  status = 500,
  delay = 0
): Promise<Response> {
  return new Promise((_, reject) => {
    setTimeout(() => {
      reject({
        status,
        message,
      });
    }, delay);
  });
}

/**
 * Create mock event
 */
export function createMockEvent(type: string, data?: unknown): Event {
  if (data) {
    return new CustomEvent(type, { detail: data });
  }
  return new Event(type);
}

/**
 * Mock localStorage
 */
export function createMockLocalStorage() {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => Object.keys(store)[index] || null,
  };
}

/**
 * Suppress console errors in tests
 */
export function suppressConsoleError() {
  const originalError = console.error;
  beforeAll(() => {
    console.error = vi.fn();
  });
  afterAll(() => {
    console.error = originalError;
  });
}

/**
 * Mock window.matchMedia
 */
export function mockMatchMedia(matches: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

/**
 * Mock IntersectionObserver
 */
export function mockIntersectionObserver() {
  class MockIntersectionObserver implements IntersectionObserver {
    readonly root: Element | null = null;
    readonly rootMargin: string = '';
    readonly thresholds: ReadonlyArray<number>;

    constructor(
      _callback: IntersectionObserverCallback,
      options?: IntersectionObserverInit
    ) {
      this.root = (options?.root as Element | null) || null;
      this.rootMargin = options?.rootMargin || '';
      this.thresholds = Array.isArray(options?.threshold)
        ? options.threshold
        : [options?.threshold || 0];
    }

    disconnect() {}
    observe() {}
    takeRecords(): IntersectionObserverEntry[] {
      return [];
    }
    unobserve() {}
  }

  Object.defineProperty(window, 'IntersectionObserver', {
    writable: true,
    configurable: true,
    value: MockIntersectionObserver,
  });
}

/**
 * Create mock component
 */
export function createMockComponent(name: string) {
  return ({ children }: { children?: React.ReactNode }) => (
    <div data-mock={name}>{children}</div>
  );
}

/**
 * Mock React Query
 */
export const mockReactQuery = {
  useQuery: vi.fn(),
  useMutation: vi.fn(),
  useQueryClient: vi.fn(() => ({
    invalidateQueries: vi.fn(),
    setQueryData: vi.fn(),
    getQueryData: vi.fn(),
  })),
};
