/**
 * useBubbleLabIntegration Hook
 *
 * React hook to initialize the BubbleLab integration when the app starts.
 * This should be used in the root component of the application.
 */

import { useEffect, useState } from 'react';
import { initializeBubbleLabIntegration, getBubbleLabIntegration } from './plugin-integration';
import type { BubbleLabIntegration } from './plugin-integration';

export interface BubbleLabIntegrationConfig {
  ragbits?: {
    serverUrl: string;
    apiKey?: string;
    enabled?: boolean;
  };
  datapizza?: {
    serverUrl: string;
    apiKey?: string;
    enabled?: boolean;
  };
  autoStart?: boolean;
  healthCheckInterval?: number;
}

export interface UseBubbleLabIntegrationResult {
  integration: BubbleLabIntegration | null;
  isInitialized: boolean;
  isStarting: boolean;
  error: Error | null;
  status: {
    initialized: boolean;
    started: boolean;
    pluginCount: number;
    healthyPlugins: number;
  } | null;
}

/**
 * Hook to initialize BubbleLab integration
 *
 * @param config Integration configuration
 * @returns Integration status and instance
 *
 * @example
 * ```tsx
 * function App() {
 *   const { integration, isInitialized, error } = useBubbleLabIntegration({
 *     ragbits: { serverUrl: 'http://localhost:3000/ragbits' },
 *     datapizza: { serverUrl: 'http://localhost:3000/datapizza' }
 *   });
 *
 *   if (error) {
 *     return <div>Error: {error.message}</div>;
 *   }
 *
 *   if (!isInitialized) {
 *     return <div>Loading...</div>;
 *   }
 *
 *   return <MyApp />;
 * }
 * ```
 */
export function useBubbleLabIntegration(
  config: BubbleLabIntegrationConfig = {}
): UseBubbleLabIntegrationResult {
  const [integration, setIntegration] = useState<BubbleLabIntegration | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [status, setStatus] = useState<UseBubbleLabIntegrationResult['status']>(null);

  useEffect(() => {
    let mounted = true;

    async function initialize() {
      setIsStarting(true);
      setError(null);

      try {
        // Get existing instance if available
        const existing = getBubbleLabIntegration();
        if (existing) {
          if (mounted) {
            setIntegration(existing);
            setIsInitialized(true);
            setStatus(existing.getStatus());
          }
          setIsStarting(false);
          return;
        }

        // Initialize new instance
        const instance = await initializeBubbleLabIntegration(config);

        if (mounted) {
          setIntegration(instance);
          setIsInitialized(true);
          setStatus(instance.getStatus());
        }
      } catch (err) {
        const errorObj = err instanceof Error ? err : new Error(String(err));
        if (mounted) {
          setError(errorObj);
        }
      } finally {
        if (mounted) {
          setIsStarting(false);
        }
      }
    }

    initialize();

    return () => {
      mounted = false;
    };
  }, [JSON.stringify(config)]);

  // Update status periodically
  useEffect(() => {
    if (!integration) return;

    const interval = setInterval(() => {
      const currentStatus = integration.getStatus();
      setStatus(currentStatus);
    }, 5000);

    return () => clearInterval(interval);
  }, [integration]);

  return {
    integration,
    isInitialized,
    isStarting,
    error,
    status
  };
}

/**
 * Hook to get the integration instance (assumes already initialized)
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const integration = useBubbleLabIntegrationInstance();
 *   const registry = integration?.getRegistry();
 *   // ...
 * }
 * ```
 */
export function useBubbleLabIntegrationInstance(): BubbleLabIntegration | null {
  return getBubbleLabIntegration();
}

/**
 * Hook to get plugin registry
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const registry = usePluginRegistry();
 *   const plugins = registry?.getAllPlugins();
 *   // ...
 * }
 * ```
 */
export function usePluginRegistry() {
  const integration = useBubbleLabIntegrationInstance();
  return integration?.getRegistry() || null;
}

/**
 * Hook to get workflow orchestrator
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const orchestrator = useWorkflowOrchestrator();
 *   // ...
 * }
 * ```
 */
export function useWorkflowOrchestrator() {
  const integration = useBubbleLabIntegrationInstance();
  return integration?.getOrchestrator() || null;
}
