/**
 * useBubbleLabIntegration Hook
 *
 * React hook to initialize the BubbleLab integration when the app starts.
 * This should be used in the root component of the application.
 */
import { type BubbleLabIntegration, type BubbleLabIntegrationConfig } from '../lib/plugin-integration';
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
export declare function useBubbleLabIntegration(config?: BubbleLabIntegrationConfig): UseBubbleLabIntegrationResult;
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
export declare function useBubbleLabIntegrationInstance(): BubbleLabIntegration | null;
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
export declare function usePluginRegistry(): import("..").PluginRegistry | null;
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
export declare function useWorkflowOrchestrator(): import("..").WorkflowOrchestrator | null;
//# sourceMappingURL=useBubbleLabIntegration.d.ts.map