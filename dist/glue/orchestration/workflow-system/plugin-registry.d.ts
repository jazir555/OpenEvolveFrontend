/**
 * BubbleLab Plugin Registry
 *
 * Centralized plugin management system for OpenEvolve.
 * Manages plugin lifecycle, discovery, and orchestration.
 *
 * Architecture Principles:
 * - Law of Air Gap: Plugins are isolated modules
 * - Law of Runtime Truth: Verify plugin capabilities at runtime
 * - Law of Configuration Explicitness: All configuration via environment/props
 * - Circuit Breaker Protection: Prevent cascading failures
 */
import { CircuitBreaker } from '../../../glue/lib/circuit-breaker';
export interface PluginMetadata {
    name: string;
    version: string;
    description: string;
    author: string;
    website?: string;
    enabled: boolean;
}
export interface PluginCapabilities {
    search?: boolean;
    processing?: boolean;
    indexing?: boolean;
    verification?: boolean;
    analysis?: boolean;
}
export interface PluginContext {
    config: Record<string, unknown>;
    state: Record<string, unknown>;
}
export interface PluginInterface {
    metadata: PluginMetadata;
    capabilities: PluginCapabilities;
    initialize(config?: Record<string, unknown>): Promise<void>;
    updateConfig(config: Record<string, unknown>): Promise<void>;
    resetConfig(): Promise<void>;
    healthCheck(): Promise<boolean>;
    getContext(): PluginContext;
    getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error';
    destroy(): Promise<void>;
}
export interface PluginRegistryConfig {
    autoInitialize?: boolean;
    healthCheckInterval?: number;
    maxRetries?: number;
    timeout?: number;
}
interface RegistryEntry {
    plugin: PluginInterface;
    circuitBreaker: CircuitBreaker;
    lastHealthCheck?: Date;
    healthStatus: 'unknown' | 'healthy' | 'unhealthy';
}
declare class PluginRegistry {
    private plugins;
    private config;
    private healthCheckTimer?;
    private correlationContext;
    constructor(config?: PluginRegistryConfig);
    /**
     * Register a plugin
     */
    registerPlugin(plugin: PluginInterface): Promise<void>;
    /**
     * Unregister a plugin
     */
    unregisterPlugin(pluginName: string): Promise<void>;
    /**
     * Initialize a specific plugin
     */
    initializePlugin(pluginName: string): Promise<void>;
    /**
     * Get a plugin by name
     */
    getPlugin(pluginName: string): PluginInterface | undefined;
    /**
     * Get all registered plugins
     */
    getAllPlugins(): PluginInterface[];
    /**
     * Get plugins by capability
     */
    getPluginsByCapability(capability: keyof PluginCapabilities): PluginInterface[];
    /**
     * Execute a function within a plugin's circuit breaker
     */
    executePlugin<T>(pluginName: string, fn: () => Promise<T>): Promise<T>;
    /**
     * Health check for all plugins
     */
    healthCheckAll(): Promise<Record<string, boolean>>;
    /**
     * Health check for specific plugin
     */
    healthCheck(pluginName: string): Promise<boolean>;
    /**
     * Start periodic health checks
     */
    startHealthChecks(): void;
    /**
     * Stop periodic health checks
     */
    stopHealthChecks(): void;
    /**
     * Get registry statistics
     */
    getStatistics(): {
        totalPlugins: number;
        enabledPlugins: number;
        healthyPlugins: number;
        unhealthyPlugins: number;
        pluginsByStatus: Record<string, number>;
    };
    /**
     * Get detailed plugin status
     */
    getPluginStatus(pluginName: string): {
        metadata: PluginMetadata;
        capabilities: PluginCapabilities;
        status: string;
        healthStatus: string;
        lastHealthCheck?: Date;
    } | null;
    /**
     * Destroy all plugins
     */
    destroy(): Promise<void>;
}
/**
 * Get or create the global plugin registry
 */
export declare function getPluginRegistry(config?: PluginRegistryConfig): PluginRegistry;
/**
 * Reset the global plugin registry (for testing)
 */
export declare function resetPluginRegistry(): void;
export { PluginRegistry };
export type { PluginRegistryConfig, RegistryEntry };
//# sourceMappingURL=plugin-registry.d.ts.map