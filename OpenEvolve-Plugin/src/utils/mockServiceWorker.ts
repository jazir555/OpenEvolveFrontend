/**
 * Service Worker Implementation
 *
 * Provides offline support, caching strategies, and API mocking for development.
 *
 * @module utils/mockServiceWorker
 * @version 2.0.0
 *
 * Features:
 * - Cache-first strategy for static assets
 * - Network-first strategy for API calls
 * - Mock data fallback when backend is unavailable
 * - Service worker lifecycle management
 * - Background sync capabilities
 * - Request interception and response mocking
 */

import { errorLogger } from './errorLogging';

// ==========================================================================
// Type Definitions
// ==========================================================================

/**
 * Service worker message types for communication
 */
export type ServiceWorkerMessageType =
  | 'sync_data'
  | 'process_background_task'
  | 'cache_refresh'
  | 'notification_request'
  | 'error'
  | 'success'
  | 'progress'
  | 'skip_waiting'
  | 'claim_clients';

/**
 * Service worker message payload
 */
export interface ServiceWorkerMessage {
  type: ServiceWorkerMessageType;
  payload: any;
  id?: string;
}

/**
 * Service worker response
 */
export interface ServiceWorkerResponse {
  id: string;
  type: 'success' | 'error' | 'progress';
  data?: any;
  error?: string;
}

/**
 * Cache strategy types
 */
export type CacheStrategy = 'cache-first' | 'network-first' | 'network-only' | 'cache-only' | 'stale-while-revalidate';

/**
 * Cache configuration
 */
export interface CacheConfig {
  name: string;
  maxEntries: number;
  maxAgeSeconds: number;
  strategy: CacheStrategy;
  patterns: RegExp[];
}

/**
 * Mock API response configuration
 */
export interface MockResponseConfig {
  pattern: RegExp | string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  response: any;
  status?: number;
  delay?: number;
  headers?: Record<string, string>;
}

// ==========================================================================
// Configuration
// ==========================================================================

/**
 * Default cache configurations
 */
const DEFAULT_CACHES: Record<string, CacheConfig> = {
  static: {
    name: 'openevolve-static-v1',
    maxEntries: 100,
    maxAgeSeconds: 7 * 24 * 60 * 60, // 7 days
    strategy: 'cache-first',
    patterns: [
      /\.(?:js|css|png|jpg|jpeg|svg|gif|webp|woff|woff2|ttf|eot)$/,
      /\/assets\//,
    ],
  },
  api: {
    name: 'openevolve-api-v1',
    maxEntries: 50,
    maxAgeSeconds: 5 * 60, // 5 minutes
    strategy: 'network-first',
    patterns: [/\/api\//],
  },
};

/**
 * Default mock responses for development
 */
const DEFAULT_MOCK_RESPONSES: MockResponseConfig[] = [
  {
    pattern: '/api/health',
    method: 'GET',
    response: { status: 'healthy', timestamp: new Date().toISOString() },
    status: 200,
    delay: 100,
  },
  {
    pattern: '/api/evolution',
    method: 'POST',
    response: {
      id: 'mock-evolution-id',
      status: 'running',
      generation: 0,
      fitness: 0,
    },
    status: 200,
    delay: 500,
  },
  {
    pattern: '/api/decomposition',
    method: 'POST',
    response: {
      id: 'mock-decomp-id',
      subProblems: [
        { id: 'sp1', description: 'Analyze problem structure', complexity: 0.3 },
        { id: 'sp2', description: 'Identify constraints', complexity: 0.5 },
        { id: 'sp3', description: 'Generate solutions', complexity: 0.7 },
      ],
    },
    status: 200,
    delay: 300,
  },
  {
    pattern: '/api/adversarial',
    method: 'POST',
    response: {
      id: 'mock-adversarial-id',
      attackResults: [
        { example: 'ex1', success: true, confidence: 0.85 },
        { example: 'ex2', success: false, confidence: 0.45 },
      ],
    },
    status: 200,
    delay: 400,
  },
  {
    pattern: '/api/knowledge/query',
    method: 'POST',
    response: {
      results: [
        { id: 'k1', content: 'Mock knowledge result 1', relevance: 0.9 },
        { id: 'k2', content: 'Mock knowledge result 2', relevance: 0.7 },
      ],
      total: 2,
    },
    status: 200,
    delay: 200,
  },
];

// ==========================================================================
// Service Worker Manager
// ==========================================================================

/**
 * Service Worker Manager
 *
 * Manages service worker registration, lifecycle, and communication.
 * Provides a high-level API for service worker operations.
 */
export class ServiceWorkerManager {
  private registration: ServiceWorkerRegistration | null = null;
  private listeners: Map<string, (message: ServiceWorkerResponse) => void> = new Map();
  private taskIdCounter: number = 0;
  private mockResponses: MockResponseConfig[] = [];
  private cacheConfigs: Record<string, CacheConfig>;
  private isMockEnabled: boolean = false;

  constructor(
    cacheConfigs: Record<string, CacheConfig> = DEFAULT_CACHES,
    mockResponses: MockResponseConfig[] = []
  ) {
    this.cacheConfigs = cacheConfigs;
    this.mockResponses = [...DEFAULT_MOCK_RESPONSES, ...mockResponses];
    this.isMockEnabled = process.env.NODE_ENV === 'development';

    if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
      this.initialize();
    }
  }

  /**
   * Initialize service worker
   */
  private async initialize(): Promise<void> {
    try {
      if ('serviceWorker' in navigator) {
        this.registration = await navigator.serviceWorker.register(
          new URL('./serviceWorker.ts', import.meta.url),
          { type: 'module' }
        );

        console.log('[ServiceWorker] Registered successfully');

        // Listen for updates
        this.registration.addEventListener('updatefound', () => {
          const newWorker = this.registration?.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && !navigator.serviceWorker.controller) {
                console.log('[ServiceWorker] Initial install complete');
              }
            });
          }
        });

        // Set up message listener
        navigator.serviceWorker.addEventListener('message', (event) => {
          this.handleMessage(event.data);
        });

        // Request immediate claim if waiting
        if (this.registration.waiting) {
          this.sendMessage({
            type: 'skip_waiting',
            payload: {},
          });
        }
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        {
          component: 'ServiceWorkerManager',
          function: 'initialize',
          operation: 'REGISTER_SERVICE_WORKER',
        }
      );
    }
  }

  /**
   * Handle messages from service worker
   */
  private handleMessage(message: ServiceWorkerResponse): void {
    const listener = this.listeners.get(message.id);
    if (listener) {
      try {
        listener(message);
      } catch (error) {
        console.error(`Error in service worker listener ${message.id}:`, error);
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          {
            component: 'ServiceWorkerManager',
            function: 'handleMessage',
            operation: 'NOTIFY_LISTENER',
            additionalData: { listenerId: message.id },
          }
        );
      }
    }

    // Also notify general listeners
    const generalListener = this.listeners.get('general');
    if (generalListener) {
      try {
        generalListener(message);
      } catch (error) {
        console.error('Error in general service worker listener:', error);
      }
    }
  }

  /**
   * Send a message to the service worker
   */
  public async sendMessage(message: ServiceWorkerMessage): Promise<ServiceWorkerResponse> {
    const taskId = message.id || `task_${this.taskIdCounter++}`;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.removeListener(taskId);
        reject(new Error('Service worker message timeout'));
      }, 30000); // 30 second timeout

      this.addListener(taskId, (response: ServiceWorkerResponse) => {
        clearTimeout(timeout);
        this.removeListener(taskId);

        if (response.type === 'error') {
          reject(new Error(response.error));
        } else {
          resolve(response);
        }
      });

      if (this.registration?.active) {
        this.registration.active.postMessage({ ...message, id: taskId });
      } else {
        // Service worker not active, simulate response
        setTimeout(() => {
          this.handleMessage({
            id: taskId,
            type: 'error',
            error: 'Service worker not active',
          });
        }, 100);
      }
    });
  }

  /**
   * Add a listener for service worker responses
   */
  public addListener(id: string, callback: (message: ServiceWorkerResponse) => void): void {
    this.listeners.set(id, callback);
  }

  /**
   * Remove a listener
   */
  public removeListener(id: string): void {
    this.listeners.delete(id);
  }

  /**
   * Clear all caches
   */
  public async clearAllCaches(): Promise<void> {
    if (!this.registration) return;

    try {
      const cacheNames = await caches.keys();
      await Promise.all(
        cacheNames.map((cacheName) => caches.delete(cacheName))
      );

      console.log('[ServiceWorker] All caches cleared');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        {
          component: 'ServiceWorkerManager',
          function: 'clearAllCaches',
          operation: 'DELETE_CACHES',
        }
      );
    }
  }

  /**
   * Refresh specific cache
   */
  public async refreshCache(cacheName: string): Promise<void> {
    return this.sendMessage({
      type: 'cache_refresh',
      payload: { cacheName },
    });
  }

  /**
   * Enable or disable mock responses
   */
  public setMockEnabled(enabled: boolean): void {
    this.isMockEnabled = enabled;

    // Update service worker
    if (this.registration?.active) {
      this.registration.active.postMessage({
        type: 'update_config',
        payload: { mockEnabled: enabled },
      });
    }
  }

  /**
   * Add a custom mock response
   */
  public addMockResponse(mock: MockResponseConfig): void {
    this.mockResponses.push(mock);

    // Update service worker
    if (this.registration?.active) {
      this.registration.active.postMessage({
        type: 'add_mock',
        payload: mock,
      });
    }
  }

  /**
   * Skip waiting and activate new service worker
   */
  public async skipWaiting(): Promise<void> {
    if (this.registration?.waiting) {
      this.registration.waiting.postMessage({ type: 'skip_waiting' });
    }
  }

  /**
   * Get current service worker state
   */
  public getState(): {
    active: boolean;
    waiting: boolean;
    installing: boolean;
  } {
    return {
      active: !!this.registration?.active,
      waiting: !!this.registration?.waiting,
      installing: !!this.registration?.installing,
    };
  }

  /**
   * Unregister service worker
   */
  public async unregister(): Promise<boolean> {
    if (this.registration) {
      const unregistered = await this.registration.unregister();
      if (unregistered) {
        console.log('[ServiceWorker] Unregistered successfully');
        this.registration = null;
        this.listeners.clear();
      }
      return unregistered;
    }
    return false;
  }
}

// ==========================================================================
// Mock Service Worker (Fallback for non-SW environments)
// ==========================================================================

/**
 * Mock Service Worker Class
 *
 * Provides service worker-like functionality for environments where
 * service workers are not available (e.g., non-secure contexts, some browsers).
 */
export class MockServiceWorker {
  private listeners: Map<string, (message: ServiceWorkerResponse) => void> = new Map();
  private taskIdCounter: number = 0;
  private mockResponses: MockResponseConfig[] = DEFAULT_MOCK_RESPONSES;
  private cache: Map<string, { response: any; timestamp: number }> = new Map();

  constructor() {
    console.log('[MockServiceWorker] Initialized for non-SW environment');
  }

  /**
   * Post a message to the service worker
   */
  async postMessage(message: ServiceWorkerMessage): Promise<ServiceWorkerResponse> {
    const taskId = message.id || `task_${this.taskIdCounter++}`;
    const startTime = Date.now();

    try {
      console.log(`[MockServiceWorker] Processing message: ${message.type}`, message.payload);

      // Simulate different operations based on message type
      let result: any;

      switch (message.type) {
        case 'sync_data':
          result = await this.handleSyncData(message.payload);
          break;
        case 'process_background_task':
          result = await this.handleBackgroundTask(message.payload);
          break;
        case 'cache_refresh':
          result = await this.handleCacheRefresh(message.payload);
          break;
        case 'notification_request':
          result = await this.handleNotificationRequest(message.payload);
          break;
        default:
          throw new Error(`Unknown message type: ${message.type}`);
      }

      const successResponse: ServiceWorkerResponse = {
        id: taskId,
        type: 'success',
        data: result,
      };

      this.notifyListeners(taskId, successResponse);
      return successResponse;
    } catch (error) {
      const errorResponse: ServiceWorkerResponse = {
        id: taskId,
        type: 'error',
        error: error instanceof Error ? error.message : String(error),
      };

      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        {
          component: 'MockServiceWorker',
          function: 'postMessage',
          operation: `HANDLE_${message.type.toUpperCase()}`,
          additionalData: {
            messageType: message.type,
            taskId,
            duration: Date.now() - startTime,
          },
        }
      );

      this.notifyListeners(taskId, errorResponse);
      return errorResponse;
    }
  }

  /**
   * Add a listener for service worker responses
   */
  addListener(id: string, callback: (message: ServiceWorkerResponse) => void): void {
    this.listeners.set(id, callback);
  }

  /**
   * Remove a listener
   */
  removeListener(id: string): void {
    this.listeners.delete(id);
  }

  /**
   * Handle sync data operation
   */
  private async handleSyncData(payload: any): Promise<any> {
    await this.delay(1000);

    // Simulate occasional failure
    if (Math.random() < 0.1) {
      throw new Error('Simulated sync failure');
    }

    return {
      syncedAt: new Date().toISOString(),
      recordsProcessed: payload.records?.length || 0,
      status: 'completed',
    };
  }

  /**
   * Handle background task operation
   */
  private async handleBackgroundTask(payload: any): Promise<any> {
    const totalSteps = payload.steps || 5;
    let currentStep = 0;

    while (currentStep < totalSteps) {
      await this.delay(500);
      currentStep++;

      const progressResponse: ServiceWorkerResponse = {
        id: `progress_${Date.now()}`,
        type: 'progress',
        data: {
          current: currentStep,
          total: totalSteps,
          percentage: Math.round((currentStep / totalSteps) * 100),
        },
      };

      this.notifyListeners('progress', progressResponse);
    }

    return {
      completedAt: new Date().toISOString(),
      stepsCompleted: totalSteps,
      status: 'completed',
    };
  }

  /**
   * Handle cache refresh operation
   */
  private async handleCacheRefresh(payload: any): Promise<any> {
    await this.delay(800);

    // Simulate occasional failure
    if (Math.random() < 0.05) {
      throw new Error('Cache refresh failed due to storage quota exceeded');
    }

    return {
      refreshedAt: new Date().toISOString(),
      cacheKeysUpdated: payload.keys || [],
      status: 'completed',
    };
  }

  /**
   * Handle notification request operation
   */
  private async handleNotificationRequest(payload: any): Promise<any> {
    await this.delay(300);

    if (!payload.permission || payload.permission !== 'granted') {
      throw new Error('Notification permission not granted');
    }

    return {
      notifiedAt: new Date().toISOString(),
      title: payload.title,
      body: payload.body,
      status: 'delivered',
    };
  }

  /**
   * Notify all listeners of a message
   */
  private notifyListeners(id: string, message: ServiceWorkerResponse): void {
    const listener = this.listeners.get(id);
    if (listener) {
      try {
        listener(message);
      } catch (error) {
        console.error(`Error in service worker listener ${id}:`, error);
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          {
            component: 'MockServiceWorker',
            function: 'notifyListeners',
            operation: 'NOTIFY_LISTENER',
            additionalData: { listenerId: id },
          }
        );
      }
    }

    const generalListener = this.listeners.get('general');
    if (generalListener) {
      try {
        generalListener(message);
      } catch (error) {
        console.error('Error in general service worker listener:', error);
      }
    }
  }

  /**
   * Close the service worker
   */
  close(): void {
    console.log('[MockServiceWorker] Service worker closing');
    this.listeners.clear();
    this.cache.clear();
  }

  /**
   * Helper method to create delays
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ==========================================================================
// Singleton Instances
// ==========================================================================

let serviceWorkerManager: ServiceWorkerManager | null = null;
let mockServiceWorker: MockServiceWorker | null = null;

/**
 * Get or create the service worker manager
 */
export function getServiceWorkerManager(): ServiceWorkerManager | MockServiceWorker {
  if (!serviceWorkerManager && !mockServiceWorker) {
    if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
      serviceWorkerManager = new ServiceWorkerManager();
    } else {
      mockServiceWorker = new MockServiceWorker();
    }
  }

  return serviceWorkerManager || mockServiceWorker!;
}

/**
 * Legacy singleton instance (for backwards compatibility)
 */
const legacyMockServiceWorker = new MockServiceWorker();

export default legacyMockServiceWorker;

// ==========================================================================
// Usage Examples
// ==========================================================================

/*
// Basic usage with the manager
const swManager = getServiceWorkerManager();

// Add a listener for responses
swManager.addListener('general', (response) => {
  if (response.type === 'success') {
    console.log('Task completed:', response.data);
  } else if (response.type === 'error') {
    console.error('Task failed:', response.error);
  }
});

// Send a sync message
swManager.sendMessage({
  type: 'sync_data',
  payload: { records: [1, 2, 3] }
});

// Cache operations
await swManager.refreshCache('api');

// Mock control
swManager.setMockEnabled(true);
swManager.addMockResponse({
  pattern: '/api/custom',
  method: 'GET',
  response: { custom: 'data' },
  status: 200
});

// Service worker lifecycle
const state = swManager.getState();
if (state.waiting) {
  await swManager.skipWaiting();
}

// Legacy usage (backwards compatible)
import mockServiceWorker from '@openevolve/plugin/utils/mockServiceWorker';

mockServiceWorker.addListener('general', handleResponse);
mockServiceWorker.postMessage({
  type: 'process_background_task',
  payload: { steps: 10 }
});
*/
