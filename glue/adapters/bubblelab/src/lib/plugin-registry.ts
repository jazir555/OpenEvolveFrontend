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

import { apiLogger, LogContext } from '../../../../lib/structuredLogger';
import { CircuitBreaker } from '../../../../lib/circuit-breaker';

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

  // Lifecycle methods
  initialize(config?: Record<string, unknown>): Promise<void>;
  updateConfig(config: Record<string, unknown>): Promise<void>;
  resetConfig(): Promise<void>;
  healthCheck(): Promise<boolean>;

  // State management
  getContext(): PluginContext;
  getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error';

  // Cleanup
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

class PluginRegistry {
  private plugins = new Map<string, RegistryEntry>();
  private config: PluginRegistryConfig;
  private healthCheckTimer?: ReturnType<typeof setInterval>;
  private correlationContext: LogContext;

  constructor(config: PluginRegistryConfig = {}) {
    this.config = {
      autoInitialize: true,
      healthCheckInterval: 30000, // 30 seconds
      maxRetries: 3,
      timeout: 10000,
      ...config
    };

    this.correlationContext = {
      correlation_id: `plugin-registry-${Date.now()}`,
      source_service: 'plugin-registry',
      target_service: 'plugins'
    };

    apiLogger.info('Plugin Registry initialized', {
      ...this.correlationContext,
      config: this.config
    });
  }

  /**
   * Register a plugin
   */
  async registerPlugin(plugin: PluginInterface): Promise<void> {
    const pluginName = plugin.metadata.name;

    if (this.plugins.has(pluginName)) {
      throw new Error(`Plugin ${pluginName} is already registered`);
    }

    apiLogger.info('Registering plugin', {
      ...this.correlationContext,
      plugin: pluginName,
      version: plugin.metadata.version
    });

    // Create circuit breaker for this plugin
    const circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (old, newState) => {
        apiLogger.warn('Plugin circuit breaker state changed', {
          ...this.correlationContext,
          plugin: pluginName,
          old_state: old,
          new_state: newState
        });
      }
    });

    // Add to registry
    this.plugins.set(pluginName, {
      plugin,
      circuitBreaker,
      healthStatus: 'unknown'
    });

    // Auto-initialize if configured
    if (this.config.autoInitialize && plugin.metadata.enabled) {
      try {
        await this.initializePlugin(pluginName);
      } catch (error) {
        apiLogger.error('Failed to auto-initialize plugin', error as Error, {
          ...this.correlationContext,
          plugin: pluginName
        });
      }
    }
  }

  /**
   * Unregister a plugin
   */
  async unregisterPlugin(pluginName: string): Promise<void> {
    const entry = this.plugins.get(pluginName);
    if (!entry) {
      throw new Error(`Plugin ${pluginName} is not registered`);
    }

    apiLogger.info('Unregistering plugin', {
      ...this.correlationContext,
      plugin: pluginName
    });

    // Destroy plugin
    await entry.plugin.destroy();

    // Remove from registry
    this.plugins.delete(pluginName);
  }

  /**
   * Initialize a specific plugin
   */
  async initializePlugin(pluginName: string): Promise<void> {
    const entry = this.plugins.get(pluginName);
    if (!entry) {
      throw new Error(`Plugin ${pluginName} is not registered`);
    }

    return entry.circuitBreaker.execute(async () => {
      apiLogger.info('Initializing plugin', {
        ...this.correlationContext,
        plugin: pluginName
      });

      await entry.plugin.initialize();
      entry.healthStatus = 'healthy';
      entry.lastHealthCheck = new Date();

      apiLogger.info('Plugin initialized successfully', {
        ...this.correlationContext,
        plugin: pluginName
      });
    });
  }

  /**
   * Get a plugin by name
   */
  getPlugin(pluginName: string): PluginInterface | undefined {
    const entry = this.plugins.get(pluginName);
    return entry?.plugin;
  }

  /**
   * Get all registered plugins
   */
  getAllPlugins(): PluginInterface[] {
    return Array.from(this.plugins.values()).map(entry => entry.plugin);
  }

  /**
   * Get plugins by capability
   */
  getPluginsByCapability(capability: keyof PluginCapabilities): PluginInterface[] {
    return Array.from(this.plugins.values())
      .filter(entry => entry.plugin.capabilities[capability])
      .map(entry => entry.plugin);
  }

  /**
   * Execute a function within a plugin's circuit breaker
   */
  async executePlugin<T>(
    pluginName: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const entry = this.plugins.get(pluginName);
    if (!entry) {
      throw new Error(`Plugin ${pluginName} is not registered`);
    }

    return entry.circuitBreaker.execute(async () => {
      const startTime = Date.now();

      try {
        const result = await fn();

        const duration = Date.now() - startTime;
        apiLogger.info('Plugin execution succeeded', {
          ...this.correlationContext,
          plugin: pluginName,
          duration_ms: duration
        });

        return result;
      } catch (error) {
        const duration = Date.now() - startTime;
        apiLogger.error('Plugin execution failed', error as Error, {
          ...this.correlationContext,
          plugin: pluginName,
          duration_ms: duration
        });
        throw error;
      }
    });
  }

  /**
   * Health check for all plugins
   */
  async healthCheckAll(): Promise<Record<string, boolean>> {
    const results: Record<string, boolean> = {};

    for (const [name, entry] of this.plugins.entries()) {
      try {
        const healthy = await entry.plugin.healthCheck();
        entry.healthStatus = healthy ? 'healthy' : 'unhealthy';
        entry.lastHealthCheck = new Date();
        results[name] = healthy;
      } catch (error) {
        entry.healthStatus = 'unhealthy';
        results[name] = false;
      }
    }

    return results;
  }

  /**
   * Health check for specific plugin
   */
  async healthCheck(pluginName: string): Promise<boolean> {
    const entry = this.plugins.get(pluginName);
    if (!entry) {
      throw new Error(`Plugin ${pluginName} is not registered`);
    }

    try {
      const healthy = await entry.plugin.healthCheck();
      entry.healthStatus = healthy ? 'healthy' : 'unhealthy';
      entry.lastHealthCheck = new Date();
      return healthy;
    } catch (error) {
      entry.healthStatus = 'unhealthy';
      return false;
    }
  }

  /**
   * Start periodic health checks
   */
  startHealthChecks(): void {
    if (this.healthCheckTimer) {
      return; // Already running
    }

    this.healthCheckTimer = setInterval(async () => {
      await this.healthCheckAll();
    }, this.config.healthCheckInterval);

    apiLogger.info('Started periodic health checks', {
      ...this.correlationContext,
      interval_ms: this.config.healthCheckInterval
    });
  }

  /**
   * Stop periodic health checks
   */
  stopHealthChecks(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = undefined;

      apiLogger.info('Stopped periodic health checks', this.correlationContext);
    }
  }

  /**
   * Get registry statistics
   */
  getStatistics(): {
    totalPlugins: number;
    enabledPlugins: number;
    healthyPlugins: number;
    unhealthyPlugins: number;
    pluginsByStatus: Record<string, number>;
    } {
    const stats = {
      totalPlugins: this.plugins.size,
      enabledPlugins: 0,
      healthyPlugins: 0,
      unhealthyPlugins: 0,
      pluginsByStatus: {} as Record<string, number>
    };

    for (const entry of this.plugins.values()) {
      if (entry.plugin.metadata.enabled) {
        stats.enabledPlugins++;
      }

      if (entry.healthStatus === 'healthy') {
        stats.healthyPlugins++;
      } else if (entry.healthStatus === 'unhealthy') {
        stats.unhealthyPlugins++;
      }

      const status = entry.plugin.getStatus();
      stats.pluginsByStatus[status] = (stats.pluginsByStatus[status] || 0) + 1;
    }

    return stats;
  }

  /**
   * Get detailed plugin status
   */
  getPluginStatus(pluginName: string): {
    metadata: PluginMetadata;
    capabilities: PluginCapabilities;
    status: string;
    healthStatus: string;
    lastHealthCheck?: Date;
  } | null {
    const entry = this.plugins.get(pluginName);
    if (!entry) {
      return null;
    }

    return {
      metadata: entry.plugin.metadata,
      capabilities: entry.plugin.capabilities,
      status: entry.plugin.getStatus(),
      healthStatus: entry.healthStatus,
      lastHealthCheck: entry.lastHealthCheck
    };
  }

  /**
   * Destroy all plugins
   */
  async destroy(): Promise<void> {
    this.stopHealthChecks();

    const destroyPromises = Array.from(this.plugins.values()).map(
      async (entry) => {
        try {
          await entry.plugin.destroy();
        } catch (error) {
          apiLogger.error('Failed to destroy plugin', error as Error, {
            ...this.correlationContext,
            plugin: entry.plugin.metadata.name
          });
        }
      }
    );

    await Promise.all(destroyPromises);
    this.plugins.clear();

    apiLogger.info('Plugin Registry destroyed', this.correlationContext);
  }
}

// Global singleton instance
let globalRegistry: PluginRegistry | null = null;

/**
 * Get or create the global plugin registry
 */
export function getPluginRegistry(config?: PluginRegistryConfig): PluginRegistry {
  if (!globalRegistry) {
    globalRegistry = new PluginRegistry(config);
  }
  return globalRegistry;
}

/**
 * Reset the global plugin registry (for testing)
 */
export function resetPluginRegistry(): void {
  if (globalRegistry) {
    globalRegistry.destroy().catch(() => {
      // Ignore errors during cleanup
    });
    globalRegistry = null;
  }
}

export { PluginRegistry };
export type { RegistryEntry };
