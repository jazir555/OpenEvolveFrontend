# Service Worker Implementation

## Overview

This service worker implementation provides comprehensive offline support, intelligent caching strategies, and API mocking capabilities for the OpenEvolve Plugin.

## Features

### 1. Offline Support
- **Cache-first strategy** for static assets (JS, CSS, images, fonts)
- **Network-first strategy** for API calls with automatic fallback to cache
- **Background sync** capabilities for deferred data synchronization
- **Offline mock responses** when backend is unavailable

### 2. Caching Strategies

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `cache-first` | Static assets | Serves from cache, updates in background |
| `network-first` | API calls | Tries network first, falls back to cache |
| `stale-while-revalidate` | Frequently accessed data | Serves cache immediately, updates async |
| `network-only` | Real-time data | Always fetches from network |
| `cache-only` | Critical static files | Only serves from cache |

### 3. Mock API Responses

Pre-configured mock responses for development:

- `/api/health` - Health check endpoint
- `/api/evolution` - Evolution API mock
- `/api/decomposition` - Decomposition API mock
- `/api/adversarial` - Adversarial testing API mock
- `/api/knowledge/query` - Knowledge query API mock

## Installation

### 1. Service Worker File Placement

The actual service worker (`serviceWorker.ts`) needs to be in a location accessible by the browser. Since service workers run in a separate context, you have two options:

#### Option A: Build and Copy to Public Directory

Add to your `vite.config.ts`:

```typescript
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    // Build service worker separately
    rollupOptions: {
      input: {
        'service-worker': resolve(__dirname, 'src/utils/serviceWorker.ts'),
      },
      output: {
        entryFileNames: '[name].js',
        dir: 'public',
      },
    },
  },
});
```

#### Option B: Use Vite Plugin

Install the plugin:
```bash
npm install --save-dev vite-plugin-pwa
```

Configure in `vite.config.ts`:
```typescript
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    VitePWA({
      strategies: 'networkFirst',
      srcDir: 'src/utils',
      filename: 'serviceWorker.js',
      includeAssets: ['*.png', '*.svg', '*.jpg'],
      manifest: {
        name: 'OpenEvolve Plugin',
        short_name: 'OpenEvolve',
        description: 'AI evolution and optimization platform',
        theme_color: '#ffffff',
        icons: [
          {
            src: 'icon-192.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: 'icon-512.png',
            sizes: '512x512',
            type: 'image/png',
          },
        ],
      },
    }),
  ],
});
```

### 2. Registration

The service worker is automatically registered when you import the module:

```typescript
import { getServiceWorkerManager } from '@openevolve/plugin/utils/mockServiceWorker';

// Service worker auto-registers on initialization
const swManager = getServiceWorkerManager();
```

## Usage

### Basic Usage

```typescript
import { getServiceWorkerManager } from '@openevolve/plugin/utils/mockServiceWorker';

const swManager = getServiceWorkerManager();

// Add a listener for responses
swManager.addListener('general', (response) => {
  if (response.type === 'success') {
    console.log('Task completed:', response.data);
  } else if (response.type === 'error') {
    console.error('Task failed:', response.error);
  }
});

// Send a message
await swManager.sendMessage({
  type: 'sync_data',
  payload: { records: [1, 2, 3] }
});
```

### Cache Management

```typescript
// Clear all caches
await swManager.clearAllCaches();

// Refresh specific cache
await swManager.refreshCache('api');

// Get cache state
const state = swManager.getState();
console.log('Active:', state.active);
console.log('Waiting:', state.waiting);
```

### Mock Responses

```typescript
// Enable mock responses
swManager.setMockEnabled(true);

// Add custom mock response
swManager.addMockResponse({
  pattern: '/api/custom-endpoint',
  method: 'GET',
  response: {
    data: 'Custom mock data',
    timestamp: new Date().toISOString()
  },
  status: 200,
  delay: 500, // Simulate network delay
});

// Disable mock responses
swManager.setMockEnabled(false);
```

### Service Worker Lifecycle

```typescript
// Check service worker state
const state = swManager.getState();

// Skip waiting and activate new service worker
if (state.waiting) {
  await swManager.skipWaiting();
}

// Unregister service worker
const unregistered = await swManager.unregister();
console.log('Unregistered:', unregistered);
```

### Legacy API (Backwards Compatible)

```typescript
import mockServiceWorker from '@openevolve/plugin/utils/mockServiceWorker';

const handleResponse = (response) => {
  if (response.type === 'success') {
    console.log('Task completed:', response.data);
  } else if (response.type === 'error') {
    console.error('Task failed:', response.error);
  } else if (response.type === 'progress') {
    console.log('Progress:', response.data.percentage + '%');
  }
};

mockServiceWorker.addListener('general', handleResponse);

// Send a message
mockServiceWorker.postMessage({
  type: 'process_background_task',
  payload: { steps: 10 }
});
```

## Configuration

### Cache Configuration

Modify cache behavior by passing custom config:

```typescript
import { ServiceWorkerManager } from '@openevolve/plugin/utils/mockServiceWorker';

const swManager = new ServiceWorkerManager({
  static: {
    name: 'myapp-static-v1',
    maxEntries: 200,
    maxAgeSeconds: 14 * 24 * 60 * 60, // 14 days
    strategy: 'cache-first',
    patterns: [
      /\.(?:js|css|png|jpg|jpeg|svg|gif|webp)$/,
      /\/assets\//,
    ],
  },
  api: {
    name: 'myapp-api-v1',
    maxEntries: 100,
    maxAgeSeconds: 10 * 60, // 10 minutes
    strategy: 'network-first',
    patterns: [/\/api\//],
  },
});
```

### Custom Mock Responses

```typescript
const swManager = new ServiceWorkerManager(
  DEFAULT_CACHES,
  [
    {
      pattern: '/api/users',
      method: 'GET',
      response: {
        users: [
          { id: 1, name: 'User 1' },
          { id: 2, name: 'User 2' },
        ],
      },
      status: 200,
      delay: 300,
    },
  ]
);
```

## API Reference

### ServiceWorkerManager

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `sendMessage(message)` | `ServiceWorkerMessage` | `Promise<ServiceWorkerResponse>` | Send message to service worker |
| `addListener(id, callback)` | `id: string`, `callback: Function` | `void` | Add response listener |
| `removeListener(id)` | `id: string` | `void` | Remove response listener |
| `clearAllCaches()` | - | `Promise<void>` | Clear all caches |
| `refreshCache(cacheName)` | `cacheName: string` | `Promise<void>` | Refresh specific cache |
| `setMockEnabled(enabled)` | `enabled: boolean` | `void` | Enable/disable mocks |
| `addMockResponse(mock)` | `MockResponseConfig` | `void` | Add custom mock |
| `skipWaiting()` | - | `Promise<void>` | Activate waiting SW |
| `getState()` | - | `{ active, waiting, installing }` | Get SW state |
| `unregister()` | - | `Promise<boolean>` | Unregister SW |

### MockServiceWorker

Same API as `ServiceWorkerManager` but runs in-memory for non-service-worker environments.

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `postMessage(message)` | `ServiceWorkerMessage` | `Promise<ServiceWorkerResponse>` | Send message |
| `addListener(id, callback)` | `id: string`, `callback: Function` | `void` | Add listener |
| `removeListener(id)` | `id: string` | `void` | Remove listener |
| `close()` | - | `void` | Close service worker |

### Helper Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `getServiceWorkerManager()` | - | `ServiceWorkerManager \| MockServiceWorker` | Get singleton instance |

## TypeScript Types

```typescript
// Service Worker Message Types
type ServiceWorkerMessageType =
  | 'sync_data'
  | 'process_background_task'
  | 'cache_refresh'
  | 'notification_request'
  | 'skip_waiting'
  | 'claim_clients';

// Cache Strategy Types
type CacheStrategy =
  | 'cache-first'
  | 'network-first'
  | 'network-only'
  | 'cache-only'
  | 'stale-while-revalidate';

// Message Interface
interface ServiceWorkerMessage {
  type: ServiceWorkerMessageType;
  payload: any;
  id?: string;
}

// Response Interface
interface ServiceWorkerResponse {
  id: string;
  type: 'success' | 'error' | 'progress';
  data?: any;
  error?: string;
}

// Cache Configuration
interface CacheConfig {
  name: string;
  maxEntries: number;
  maxAgeSeconds: number;
  strategy: CacheStrategy;
  patterns: RegExp[];
}

// Mock Response Configuration
interface MockResponseConfig {
  pattern: RegExp | string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  response: any;
  status?: number;
  delay?: number;
  headers?: Record<string, string>;
}
```

## Development

### Testing Offline Functionality

1. Open DevTools
2. Go to Application/Service Workers
3. Check "Offline" checkbox
4. Refresh the page

The application should still load static assets and serve cached API responses.

### Debugging

```typescript
// Enable debug logging
localStorage.setItem('openevolve_debug', 'true');

// View service worker logs
// 1. Open DevTools
// 2. Go to Application/Service Workers
// 3. Click on the service worker link
// 4. Check console for logs
```

### Cache Inspection

```typescript
// List all caches
caches.keys().then(names => console.log(names));

// Inspect specific cache
caches.open('openevolve-static-v1').then(cache => {
  cache.keys().then(requests => {
    requests.forEach(request => console.log(request.url));
  });
});
```

## Troubleshooting

### Service Worker Not Activating

1. Ensure HTTPS or localhost
2. Check service worker scope
3. Clear all caches and unregister:
   ```typescript
   await swManager.clearAllCaches();
   await swManager.unregister();
   ```

### Cache Not Updating

1. Increment cache version in `serviceWorker.ts`
2. Call `skipWaiting()` to activate new service worker
3. Hard refresh (Ctrl+Shift+R)

### Mock Responses Not Working

1. Ensure mock is enabled:
   ```typescript
   swManager.setMockEnabled(true);
   ```
2. Check mock pattern matches URL
3. Verify HTTP method matches

### Build Errors

If you get build errors for `serviceWorker.ts`, ensure:

1. TypeScript is configured for service worker environment
2. `dom` and `webworker" libs are enabled in `tsconfig.json`
3. Service worker file is built separately from main app

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (iOS 11.3+)
- Opera: Full support

## Security Considerations

1. **HTTPS Required**: Service workers only work on secure contexts (HTTPS or localhost)
2. **Scope Limitations**: Service worker can only control pages within its scope
3. **Cache Poisoning**: Validate API responses before caching
4. **Sensitive Data**: Never cache sensitive user data in service worker

## Performance Tips

1. **Pre-cache Critical Assets**: Cache core assets during install
2. **Cache Size Limits**: Keep caches under quota (typically ~50MB)
3. **Cleanup Old Caches**: Always clean up old cache versions
4. **Network-First for API**: Use network-first for frequently updated data

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
