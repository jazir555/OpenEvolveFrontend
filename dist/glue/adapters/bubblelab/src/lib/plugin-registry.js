"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.PluginRegistry = void 0;
exports.getPluginRegistry = getPluginRegistry;
exports.resetPluginRegistry = resetPluginRegistry;
const structuredLogger_1 = require("../../../../lib/structuredLogger");
const circuit_breaker_1 = require("../../../../lib/circuit-breaker");
class PluginRegistry {
    constructor(config = {}) {
        this.plugins = new Map();
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
        structuredLogger_1.apiLogger.info('Plugin Registry initialized', {
            ...this.correlationContext,
            config: this.config
        });
    }
    /**
     * Register a plugin
     */
    async registerPlugin(plugin) {
        const pluginName = plugin.metadata.name;
        if (this.plugins.has(pluginName)) {
            throw new Error(`Plugin ${pluginName} is already registered`);
        }
        structuredLogger_1.apiLogger.info('Registering plugin', {
            ...this.correlationContext,
            plugin: pluginName,
            version: plugin.metadata.version
        });
        // Create circuit breaker for this plugin
        const circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: 5,
            timeout_ms: 60000,
            onStateChange: (old, newState) => {
                structuredLogger_1.apiLogger.warn('Plugin circuit breaker state changed', {
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
            }
            catch (error) {
                structuredLogger_1.apiLogger.error('Failed to auto-initialize plugin', error, {
                    ...this.correlationContext,
                    plugin: pluginName
                });
            }
        }
    }
    /**
     * Unregister a plugin
     */
    async unregisterPlugin(pluginName) {
        const entry = this.plugins.get(pluginName);
        if (!entry) {
            throw new Error(`Plugin ${pluginName} is not registered`);
        }
        structuredLogger_1.apiLogger.info('Unregistering plugin', {
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
    async initializePlugin(pluginName) {
        const entry = this.plugins.get(pluginName);
        if (!entry) {
            throw new Error(`Plugin ${pluginName} is not registered`);
        }
        return entry.circuitBreaker.execute(async () => {
            structuredLogger_1.apiLogger.info('Initializing plugin', {
                ...this.correlationContext,
                plugin: pluginName
            });
            await entry.plugin.initialize();
            entry.healthStatus = 'healthy';
            entry.lastHealthCheck = new Date();
            structuredLogger_1.apiLogger.info('Plugin initialized successfully', {
                ...this.correlationContext,
                plugin: pluginName
            });
        });
    }
    /**
     * Get a plugin by name
     */
    getPlugin(pluginName) {
        const entry = this.plugins.get(pluginName);
        return entry?.plugin;
    }
    /**
     * Get all registered plugins
     */
    getAllPlugins() {
        return Array.from(this.plugins.values()).map(entry => entry.plugin);
    }
    /**
     * Get plugins by capability
     */
    getPluginsByCapability(capability) {
        return Array.from(this.plugins.values())
            .filter(entry => entry.plugin.capabilities[capability])
            .map(entry => entry.plugin);
    }
    /**
     * Execute a function within a plugin's circuit breaker
     */
    async executePlugin(pluginName, fn) {
        const entry = this.plugins.get(pluginName);
        if (!entry) {
            throw new Error(`Plugin ${pluginName} is not registered`);
        }
        return entry.circuitBreaker.execute(async () => {
            const startTime = Date.now();
            try {
                const result = await fn();
                const duration = Date.now() - startTime;
                structuredLogger_1.apiLogger.info('Plugin execution succeeded', {
                    ...this.correlationContext,
                    plugin: pluginName,
                    duration_ms: duration
                });
                return result;
            }
            catch (error) {
                const duration = Date.now() - startTime;
                structuredLogger_1.apiLogger.error('Plugin execution failed', error, {
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
    async healthCheckAll() {
        const results = {};
        for (const [name, entry] of this.plugins.entries()) {
            try {
                const healthy = await entry.plugin.healthCheck();
                entry.healthStatus = healthy ? 'healthy' : 'unhealthy';
                entry.lastHealthCheck = new Date();
                results[name] = healthy;
            }
            catch (error) {
                entry.healthStatus = 'unhealthy';
                results[name] = false;
            }
        }
        return results;
    }
    /**
     * Health check for specific plugin
     */
    async healthCheck(pluginName) {
        const entry = this.plugins.get(pluginName);
        if (!entry) {
            throw new Error(`Plugin ${pluginName} is not registered`);
        }
        try {
            const healthy = await entry.plugin.healthCheck();
            entry.healthStatus = healthy ? 'healthy' : 'unhealthy';
            entry.lastHealthCheck = new Date();
            return healthy;
        }
        catch (error) {
            entry.healthStatus = 'unhealthy';
            return false;
        }
    }
    /**
     * Start periodic health checks
     */
    startHealthChecks() {
        if (this.healthCheckTimer) {
            return; // Already running
        }
        this.healthCheckTimer = setInterval(async () => {
            await this.healthCheckAll();
        }, this.config.healthCheckInterval);
        structuredLogger_1.apiLogger.info('Started periodic health checks', {
            ...this.correlationContext,
            interval_ms: this.config.healthCheckInterval
        });
    }
    /**
     * Stop periodic health checks
     */
    stopHealthChecks() {
        if (this.healthCheckTimer) {
            clearInterval(this.healthCheckTimer);
            this.healthCheckTimer = undefined;
            structuredLogger_1.apiLogger.info('Stopped periodic health checks', this.correlationContext);
        }
    }
    /**
     * Get registry statistics
     */
    getStatistics() {
        const stats = {
            totalPlugins: this.plugins.size,
            enabledPlugins: 0,
            healthyPlugins: 0,
            unhealthyPlugins: 0,
            pluginsByStatus: {}
        };
        for (const entry of this.plugins.values()) {
            if (entry.plugin.metadata.enabled) {
                stats.enabledPlugins++;
            }
            if (entry.healthStatus === 'healthy') {
                stats.healthyPlugins++;
            }
            else if (entry.healthStatus === 'unhealthy') {
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
    getPluginStatus(pluginName) {
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
    async destroy() {
        this.stopHealthChecks();
        const destroyPromises = Array.from(this.plugins.values()).map(async (entry) => {
            try {
                await entry.plugin.destroy();
            }
            catch (error) {
                structuredLogger_1.apiLogger.error('Failed to destroy plugin', error, {
                    ...this.correlationContext,
                    plugin: entry.plugin.metadata.name
                });
            }
        });
        await Promise.all(destroyPromises);
        this.plugins.clear();
        structuredLogger_1.apiLogger.info('Plugin Registry destroyed', this.correlationContext);
    }
}
exports.PluginRegistry = PluginRegistry;
// Global singleton instance
let globalRegistry = null;
/**
 * Get or create the global plugin registry
 */
function getPluginRegistry(config) {
    if (!globalRegistry) {
        globalRegistry = new PluginRegistry(config);
    }
    return globalRegistry;
}
/**
 * Reset the global plugin registry (for testing)
 */
function resetPluginRegistry() {
    if (globalRegistry) {
        globalRegistry.destroy().catch(() => {
            // Ignore errors during cleanup
        });
        globalRegistry = null;
    }
}
//# sourceMappingURL=plugin-registry.js.map