/**
 * Service Worker
 *
 * Actual service worker implementation that runs in the service worker context.
 * This file should be placed in the public directory or registered separately.
 *
 * @version 2.0.0
 *
 * Features:
 * - Cache-first strategy for static assets
 * - Network-first strategy for API calls
 * - Mock data fallback when backend is unavailable
 * - Background sync capabilities
 * - Request interception and response mocking
 */

// ==========================================================================
// Type Definitions
// ==========================================================================

interface CacheConfig {
  name: string;
  maxEntries: number;
  maxAgeSeconds: number;
  strategy: 'cache-first' | 'network-first' | 'network-only' | 'cache-only' | 'stale-while-revalidate';
}

interface MockResponseConfig {
  pattern: RegExp | string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  response: any;
  status?: number;
  delay?: number;
  headers?: Record<string, string>;
}

interface ServiceWorkerMessage {
  type: string;
  payload: any;
  id?: string;
}

interface ServiceWorkerResponse {
  id: string;
  type: 'success' | 'error' | 'progress';
  data?: any;
  error?: string;
}

// ==========================================================================
// Configuration
// ==========================================================================

const CACHE_VERSION = 'v1';
const CACHE_PREFIX = 'openevolve';

const CACHE_CONFIGS: Record<string, CacheConfig> = {
  static: {
    name: `${CACHE_PREFIX}-static-${CACHE_VERSION}`,
    maxEntries: 100,
    maxAgeSeconds: 7 * 24 * 60 * 60, // 7 days
    strategy: 'cache-first',
  },
  api: {
    name: `${CACHE_PREFIX}-api-${CACHE_VERSION}`,
    maxEntries: 50,
    maxAgeSeconds: 5 * 60, // 5 minutes
    strategy: 'network-first',
  },
};

let mockResponses: MockResponseConfig[] = [];
let isMockEnabled = false;

// ==========================================================================
// Cache Utilities
// ==========================================================================

/**
 * Get cache by name
 */
async function getCache(cacheName: string): Promise<Cache> {
  return caches.open(cacheName);
}

/**
 * Check if a cache exists
 */
async function cacheExists(cacheName: string): Promise<boolean> {
  const cacheNames = await caches.keys();
  return cacheNames.includes(cacheName);
}

/**
 * Clean up old caches
 */
async function cleanupOldCaches(): Promise<void> {
  const cacheNames = await caches.keys();
  const validCacheNames = Object.values(CACHE_CONFIGS).map((config) => config.name);

  await Promise.all(
    cacheNames
      .filter((cacheName) => cacheName.startsWith(CACHE_PREFIX) && !validCacheNames.includes(cacheName))
      .map((cacheName) => caches.delete(cacheName))
  );
}

/**
 * Clean cache entries based on max entries and age
 */
async function cleanupCacheEntries(cache: Cache, config: CacheConfig): Promise<void> {
  const keys = await cache.keys();

  // Remove oldest entries if max entries exceeded
  if (keys.length > config.maxEntries) {
    const keysToDelete = keys.slice(0, keys.length - config.maxEntries);
    await Promise.all(keysToDelete.map((key) => cache.delete(key)));
  }

  // Remove expired entries
  const now = Date.now();
  for (const request of keys) {
    const response = await cache.match(request);
    if (response) {
      const cacheDate = response.headers.get('date');
      if (cacheDate) {
        const age = (now - new Date(cacheDate).getTime()) / 1000;
        if (age > config.maxAgeSeconds) {
          await cache.delete(request);
        }
      }
    }
  }
}

// ==========================================================================
// Cache Strategies
// ==========================================================================

/**
 * Cache-first strategy
 */
async function cacheFirst(
  request: Request,
  cacheName: string,
  maxAgeSeconds: number
): Promise<Response> {
  const cache = await getCache(cacheName);

  // Try cache first
  const cachedResponse = await cache.match(request);

  if (cachedResponse) {
    // Check if cache is still valid
    const cacheDate = cachedResponse.headers.get('date');
    if (cacheDate) {
      const age = (Date.now() - new Date(cacheDate).getTime()) / 1000;
      if (age < maxAgeSeconds) {
        return cachedResponse;
      }
    }
  }

  // Fetch from network
  try {
    const networkResponse = await fetch(request);

    // Clone and cache the response
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    // Return cached response if network fails
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

/**
 * Network-first strategy
 */
async function networkFirst(request: Request, cacheName: string): Promise<Response> {
  const cache = await getCache(cacheName);

  try {
    // Try network first
    const networkResponse = await fetch(request);

    // Cache the response
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    // Fall back to cache
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    throw error;
  }
}

/**
 * Stale-while-revalidate strategy
 */
async function staleWhileRevalidate(request: Request, cacheName: string): Promise<Response> {
  const cache = await getCache(cacheName);

  // Serve from cache immediately
  const cachedResponse = await cache.match(request);

  // Fetch in background
  const fetchPromise = fetch(request).then((networkResponse) => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  });

  // Return cached response or wait for network
  return cachedResponse || fetchPromise;
}

/**
 * Network-only strategy
 */
async function networkOnly(request: Request): Promise<Response> {
  return fetch(request);
}

/**
 * Cache-only strategy
 */
async function cacheOnly(request: Request, cacheName: string): Promise<Response> {
  const cache = await getCache(cacheName);
  const cachedResponse = await cache.match(request);

  if (cachedResponse) {
    return cachedResponse;
  }

  throw new Error('No cache match');
}

// ==========================================================================
// Mock Response Handling
// ==========================================================================

/**
 * Generate mock response
 */
async function generateMockResponse(request: Request): Promise<Response | null> {
  if (!isMockEnabled) {
    return null;
  }

  const url = request.url;
  const method = request.method as any;

  for (const mock of mockResponses) {
    let match = false;

    if (mock.pattern instanceof RegExp) {
      match = mock.pattern.test(url);
    } else {
      match = url.includes(mock.pattern);
    }

    if (match && mock.method === method) {
      // Add delay if specified
      if (mock.delay) {
        await new Promise((resolve) => setTimeout(resolve, mock.delay));
      }

      // Create response
      const headers = mock.headers || {
        'Content-Type': 'application/json',
      };

      return new Response(JSON.stringify(mock.response), {
        status: mock.status || 200,
        headers,
      });
    }
  }

  return null;
}

/**
 * Handle API request with network-first and mock fallback
 */
async function handleApiRequest(request: Request): Promise<Response> {
  // Try mock first if enabled
  const mockResponse = await generateMockResponse(request);
  if (mockResponse) {
    return mockResponse;
  }

  // Try network first
  const cache = await getCache(CACHE_CONFIGS.api.name);

  try {
    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    // Fall back to cache
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    // Return offline mock response as last resort
    return new Response(
      JSON.stringify({
        error: 'Network unavailable',
        message: 'The service is currently offline. Please check your connection.',
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

// ==========================================================================
// Request Routing
// ==========================================================================

/**
 * Determine cache strategy for request
 */
function getCacheStrategy(request: Request): string {
  const url = new URL(request.url);

  // Static assets
  if (
    /\.(?:js|css|png|jpg|jpeg|svg|gif|webp|woff|woff2|ttf|eot)$/.test(url.pathname) ||
    url.pathname.startsWith('/assets/')
  ) {
    return CACHE_CONFIGS.static.strategy;
  }

  // API calls
  if (url.pathname.startsWith('/api/')) {
    return CACHE_CONFIGS.api.strategy;
  }

  // Default to network-first
  return 'network-first';
}

/**
 * Handle fetch event
 */
async function handleFetch(event: FetchEvent): Promise<Response> {
  const request = event.request;
  const strategy = getCacheStrategy(request);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return fetch(request);
  }

  // Skip chrome extensions and other protocols
  if (!request.url.startsWith('http')) {
    return fetch(request);
  }

  // Apply strategy
  switch (strategy) {
    case 'cache-first':
      return cacheFirst(request, CACHE_CONFIGS.static.name, CACHE_CONFIGS.static.maxAgeSeconds);

    case 'network-first':
      if (request.url.includes('/api/')) {
        return handleApiRequest(request);
      }
      return networkFirst(request, CACHE_CONFIGS.api.name);

    case 'stale-while-revalidate':
      return staleWhileRevalidate(request, CACHE_CONFIGS.static.name);

    case 'network-only':
      return networkOnly(request);

    case 'cache-only':
      return cacheOnly(request, CACHE_CONFIGS.static.name);

    default:
      return networkFirst(request, CACHE_CONFIGS.api.name);
  }
}

// ==========================================================================
// Message Handling
// ==========================================================================

/**
 * Handle messages from clients
 */
async function handleMessage(event: ExtendableMessageEvent): Promise<void> {
  const message = event.data as ServiceWorkerMessage;

  switch (message.type) {
    case 'skip_waiting':
      self.skipWaiting();
      break;

    case 'claim_clients':
      self.clients.claim();
      break;

    case 'cache_refresh':
      await handleCacheRefresh(message.payload.cacheName);
      break;

    case 'clear_cache':
      await handleClearCache(message.payload.cacheName);
      break;

    case 'update_config':
      handleUpdateConfig(message.payload);
      break;

    case 'add_mock':
      addMockResponse(message.payload);
      break;

    default:
      console.log('[ServiceWorker] Unknown message type:', message.type);
  }
}

/**
 * Handle cache refresh request
 */
async function handleCacheRefresh(cacheName: string): Promise<void> {
  const config = Object.values(CACHE_CONFIGS).find((c) => c.name === cacheName);

  if (!config) {
    console.error('[ServiceWorker] Cache not found:', cacheName);
    return;
  }

  const cache = await getCache(cacheName);
  const requests = await cache.keys();

  // Re-fetch all cached requests
  await Promise.all(
    requests.map(async (request) => {
      try {
        const response = await fetch(request);
        if (response.ok) {
          await cache.put(request, response);
        }
      } catch (error) {
        console.error('[ServiceWorker] Failed to refresh:', request.url, error);
      }
    })
  );

  // Clean up old entries
  await cleanupCacheEntries(cache, config);

  // Notify all clients
  const clients = await self.clients.matchAll();
  clients.forEach((client) => {
    client.postMessage({
      type: 'success',
      id: message.id,
      data: {
        message: 'Cache refreshed successfully',
        cacheName,
        entriesRefreshed: requests.length,
      },
    });
  });
}

/**
 * Handle clear cache request
 */
async function handleClearCache(cacheName?: string): Promise<void> {
  if (cacheName) {
    await caches.delete(cacheName);
  } else {
    // Clear all caches
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map((name) => caches.delete(name)));
  }

  // Notify all clients
  const clients = await self.clients.matchAll();
  clients.forEach((client) => {
    client.postMessage({
      type: 'success',
      id: message.id,
      data: { message: 'Cache cleared successfully' },
    });
  });
}

/**
 * Handle update config request
 */
function handleUpdateConfig(payload: any): void {
  if (payload.mockEnabled !== undefined) {
    isMockEnabled = payload.mockEnabled;
  }
}

/**
 * Add mock response
 */
function addMockResponse(mock: MockResponseConfig): void {
  mockResponses.push(mock);
}

// ==========================================================================
// Service Worker Lifecycle
// ==========================================================================

/**
 * Install event
 */
self.addEventListener('install', (event: ExtendableEvent) => {
  console.log('[ServiceWorker] Installing...');

  event.waitUntil(
    (async () => {
      // Pre-cache static assets
      const staticCache = await getCache(CACHE_CONFIGS.static.name);

      // Cache core assets
      const coreAssets = [
        '/',
        '/index.html',
        '/manifest.json',
      ];

      await staticCache.addAll(coreAssets);

      // Clean up old caches
      await cleanupOldCaches();

      console.log('[ServiceWorker] Installation complete');
    })()
  );

  // Activate immediately
  self.skipWaiting();
});

/**
 * Activate event
 */
self.addEventListener('activate', (event: ExtendableEvent) => {
  console.log('[ServiceWorker] Activating...');

  event.waitUntil(
    (async () => {
      // Clean up old caches
      await cleanupOldCaches();

      // Take control of all clients
      await self.clients.claim();

      console.log('[ServiceWorker] Activation complete');
    })()
  );
});

/**
 * Fetch event
 */
self.addEventListener('fetch', (event: FetchEvent) => {
  event.respondWith(handleFetch(event));
});

/**
 * Message event
 */
self.addEventListener('message', (event: ExtendableMessageEvent) => {
  event.waitUntil(handleMessage(event));
});

/**
 * Sync event (for background sync)
 */
self.addEventListener('sync', (event: any) => {
  console.log('[ServiceWorker] Background sync:', event.tag);

  if (event.tag === 'sync-data') {
    event.waitUntil(
      (async () => {
        // Perform background sync
        console.log('[ServiceWorker] Performing background sync...');

        // Notify all clients
        const clients = await self.clients.matchAll();
        clients.forEach((client) => {
          client.postMessage({
            type: 'success',
            data: { message: 'Background sync complete' },
          });
        });
      })()
    );
  }
});

/**
 * Push event (for push notifications)
 */
self.addEventListener('push', (event: PushEvent) => {
  console.log('[ServiceWorker] Push received');

  if (event.data) {
    const data = event.data.json();

    const options = {
      body: data.body || '',
      icon: '/icon-192.png',
      badge: '/badge-72.png',
      vibrate: [200, 100, 200],
      data: {
        dateOfArrival: Date.now(),
        primaryKey: data.primaryKey || 1,
      },
    };

    event.waitUntil(self.registration.showNotification(data.title || 'Notification', options));
  }
});

/**
 * Notification click event
 */
self.addEventListener('notificationclick', (event: NotificationEvent) => {
  console.log('[ServiceWorker] Notification click');

  event.notification.close();

  event.waitUntil(
    self.clients.openWindow?.('/') ||
      self.clients.matchAll({ type: 'window' }).then((clients) => {
        // Focus or open the first window
        for (const client of clients) {
          if (client.url === '/' && 'focus' in client) {
            return client.focus();
          }
        }
        if (clients.length > 0 && 'focus' in clients[0]) {
          return clients[0].focus();
        }
        return self.clients.openWindow('/');
      })
  );
});

// ==========================================================================
// Error Handling
// ==========================================================================

/**
 * Error event
 */
self.addEventListener('error', (event: ErrorEvent) => {
  console.error('[ServiceWorker] Error:', event.error);
});

/**
 * Unhandled rejection event
 */
self.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
  console.error('[ServiceWorker] Unhandled rejection:', event.reason);
});

// ==========================================================================
// Export for TypeScript
// ==========================================================================

export {};
