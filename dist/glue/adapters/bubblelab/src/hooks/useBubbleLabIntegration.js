"use strict";
/**
 * useBubbleLabIntegration Hook
 *
 * React hook to initialize the BubbleLab integration when the app starts.
 * This should be used in the root component of the application.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.useBubbleLabIntegration = useBubbleLabIntegration;
exports.useBubbleLabIntegrationInstance = useBubbleLabIntegrationInstance;
exports.usePluginRegistry = usePluginRegistry;
exports.useWorkflowOrchestrator = useWorkflowOrchestrator;
const react_1 = require("react");
const plugin_integration_1 = require("../lib/plugin-integration");
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
function useBubbleLabIntegration(config = {}) {
    const [integration, setIntegration] = (0, react_1.useState)(null);
    const [isInitialized, setIsInitialized] = (0, react_1.useState)(false);
    const [isStarting, setIsStarting] = (0, react_1.useState)(false);
    const [error, setError] = (0, react_1.useState)(null);
    const [status, setStatus] = (0, react_1.useState)(null);
    (0, react_1.useEffect)(() => {
        let mounted = true;
        async function initialize() {
            setIsStarting(true);
            setError(null);
            try {
                // Get existing instance if available
                const existing = (0, plugin_integration_1.getBubbleLabIntegration)();
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
                const instance = await (0, plugin_integration_1.initializeBubbleLabIntegration)(config);
                if (mounted) {
                    setIntegration(instance);
                    setIsInitialized(true);
                    setStatus(instance.getStatus());
                }
            }
            catch (err) {
                const errorObj = err instanceof Error ? err : new Error(String(err));
                if (mounted) {
                    setError(errorObj);
                }
            }
            finally {
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
    (0, react_1.useEffect)(() => {
        if (!integration)
            return;
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
function useBubbleLabIntegrationInstance() {
    return (0, plugin_integration_1.getBubbleLabIntegration)();
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
function usePluginRegistry() {
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
function useWorkflowOrchestrator() {
    const integration = useBubbleLabIntegrationInstance();
    return integration?.getOrchestrator() || null;
}
//# sourceMappingURL=useBubbleLabIntegration.js.map