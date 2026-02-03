/**
 * Vitest Test Setup
 * Configure testing library and global test environment
 */

import { cleanup } from '@testing-library/react';

// Cleanup after each test
// @ts-ignore
afterEach(() => {
  cleanup();
});

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  // @ts-ignore
  value: vi.fn().mockImplementation((query: any) => ({
    matches: false,
    media: query,
    onchange: null,
    // @ts-ignore
    addListener: vi.fn(),
    // @ts-ignore
    removeListener: vi.fn(),
    // @ts-ignore
    addEventListener: vi.fn(),
    // @ts-ignore
    removeEventListener: vi.fn(),
    // @ts-ignore
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;
