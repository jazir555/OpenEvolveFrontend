"use strict";
/**
 * OpenEvolve-BubbleLab Integration
 *
 * Main integration module that initializes and connects all components:
 * - Plugin Registry
 * - Workflow Orchestrator
 * - Event Bus Integration
 * - RAGBits Plugin
 * - Datapizza Plugin
 *
 * Usage:
 * ```typescript
 * import { initializeBubbleLabIntegration } from '@/lib/plugin-integration';
 *
 * // Initialize with configuration
 * await initializeBubbleLabIntegration({
 *   ragbits: { serverUrl: 'http://localhost:3000/ragbits' },
 *   datapizza: { serverUrl: 'http://localhost:3000/datapizza' },
 *   autoStart: true
 * });
 * ```
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.BubbleLabIntegration = void 0;
exports.initializeBubbleLabIntegration = initializeBubbleLabIntegration;
exports.getBubbleLabIntegration = getBubbleLabIntegration;
exports.resetBubbleLabIntegration = resetBubbleLabIntegration;
const structuredLogger_1 = require("../../../../lib/structuredLogger");
const plugin_registry_1 = require("./plugin-registry");
const workflow_orchestrator_1 = require("./workflow-orchestrator");
const plugin_events_1 = require("./plugin-events");
const openevolveApi_1 = require("./openevolveApi");
const plugin_adapters_1 = require("./plugin-adapters");
const resolveRuntimeEnv = (...keys) => {
    if (typeof process !== 'undefined' && process.env) {
        for (const key of keys) {
            const value = process.env[key];
            if (value) {
                return value;
            }
        }
    }
    const runtime = globalThis;
    for (const key of keys) {
        const value = runtime[key];
        if (typeof value === 'string' && value.length > 0) {
            return value;
        }
    }
    return undefined;
};
class BubbleLabIntegration {
    constructor(config = {}) {
        this.state = null;
        this.config = {
            autoStart: true,
            healthCheckInterval: 30000,
            ...config
        };
        this.correlationContext = {
            correlation_id: `bubblelab-integration-${Date.now()}`,
            source_service: 'bubblelab-integration',
            target_service: 'all'
        };
        structuredLogger_1.apiLogger.info('BubbleLab Integration created', {
            ...this.correlationContext,
            config: this.config
        });
    }
    /**
     * Initialize the integration
     */
    async initialize() {
        if (this.state?.isInitialized) {
            structuredLogger_1.apiLogger.warn('Integration already initialized', this.correlationContext);
            return;
        }
        structuredLogger_1.apiLogger.info('Initializing BubbleLab Integration', this.correlationContext);
        try {
            // Initialize core components
            const registry = (0, plugin_registry_1.getPluginRegistry)({
                autoInitialize: false,
                healthCheckInterval: this.config.healthCheckInterval
            });
            const orchestrator = (0, workflow_orchestrator_1.getWorkflowOrchestrator)(registry);
            const eventIntegration = (0, plugin_events_1.getPluginEventIntegration)();
            // Register plugins
            await this.registerPlugins(registry);
            // Store state
            this.state = {
                registry,
                orchestrator,
                eventIntegration,
                isInitialized: true,
                isStarted: false
            };
            structuredLogger_1.apiLogger.info('BubbleLab Integration initialized successfully', this.correlationContext);
            // Auto-start if configured
            if (this.config.autoStart) {
                await this.start();
            }
        }
        catch (error) {
            structuredLogger_1.apiLogger.error('Failed to initialize BubbleLab Integration', error, this.correlationContext);
            throw error;
        }
    }
    /**
     * Register all plugins
     */
    async registerPlugins(registry) {
        structuredLogger_1.apiLogger.info('Registering plugins', this.correlationContext);
        // Register OpenEvolve API adapter (always available)
        if (this.config.openevolve?.enabled !== false) {
            try {
                const openevolveAdapter = new plugin_adapters_1.OpenEvolveApiAdapter(openevolveApi_1.openevolveApi, {
                    apiKey: this.resolveOpenEvolveApiKey(),
                    baseUrl: this.resolveOpenEvolveBaseUrl(),
                });
                await registry.registerPlugin(openevolveAdapter);
                structuredLogger_1.apiLogger.info('OpenEvolve API adapter registered', this.correlationContext);
            }
            catch (error) {
                structuredLogger_1.apiLogger.error('Failed to register OpenEvolve adapter', error, this.correlationContext);
            }
        }
        // Register RAGBits plugin
        if (this.config.ragbits?.enabled !== false) {
            try {
                const createRAGBitsPlugin = await this.resolvePluginFactory([
                    '@bubblelabs-ragbits-plugin',
                    'bubblelabs-ragbits-plugin',
                ]);
                const ragbitsPlugin = createRAGBitsPlugin(this.config.ragbits);
                const ragbitsAdapter = new plugin_adapters_1.RAGBitsPluginAdapter(ragbitsPlugin, {
                    ...this.config.ragbits,
                    enabled: true
                });
                await registry.registerPlugin(ragbitsAdapter);
                structuredLogger_1.apiLogger.info('RAGBits plugin registered', {
                    ...this.correlationContext,
                    server_url: this.config.ragbits?.serverUrl
                });
            }
            catch (error) {
                structuredLogger_1.apiLogger.warn('RAGBits plugin not available', {
                    ...this.correlationContext,
                    error: error instanceof Error ? error.message : String(error)
                });
            }
        }
        // Register Datapizza plugin
        if (this.config.datapizza?.enabled !== false) {
            try {
                const createDatapizzaPlugin = await this.resolvePluginFactory([
                    '@datapizza-bubblelab-plugin',
                    'datapizza-bubblelab-plugin',
                ]);
                const datapizzaPlugin = createDatapizzaPlugin(this.config.datapizza);
                const datapizzaAdapter = new plugin_adapters_1.DatapizzaPluginAdapter(datapizzaPlugin, {
                    ...this.config.datapizza,
                    enabled: true
                });
                await registry.registerPlugin(datapizzaAdapter);
                structuredLogger_1.apiLogger.info('Datapizza plugin registered', {
                    ...this.correlationContext,
                    server_url: this.config.datapizza?.serverUrl
                });
            }
            catch (error) {
                structuredLogger_1.apiLogger.warn('Datapizza plugin not available', {
                    ...this.correlationContext,
                    error: error instanceof Error ? error.message : String(error)
                });
            }
        }
        structuredLogger_1.apiLogger.info('All plugins registered', this.correlationContext);
    }
    resolveOpenEvolveApiKey() {
        return this.config.openevolve?.apiKey || resolveRuntimeEnv('OPENEVOLVE_API_KEY');
    }
    resolveOpenEvolveBaseUrl() {
        return this.config.openevolve?.baseUrl
            || resolveRuntimeEnv('OPENEVOLVE_API_BASE', 'OPENEVOLVE_API_BASE_URL', 'OPENEVOLVE_API_URL', 'NEXT_PUBLIC_OPENEVOLVE_API_BASE', 'NEXT_PUBLIC_OPENEVOLVE_API_URL');
    }
    async resolvePluginFactory(moduleNames) {
        const loadErrors = [];
        for (const moduleName of moduleNames) {
            try {
                const loaded = await Promise.resolve(`${moduleName}`).then(s => __importStar(require(s)));
                const createPlugin = loaded.createPlugin;
                if (typeof createPlugin === 'function') {
                    return createPlugin;
                }
                loadErrors.push(`${moduleName}: missing createPlugin export`);
            }
            catch (error) {
                loadErrors.push(`${moduleName}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
        throw new Error(`No compatible plugin module found (${loadErrors.join(' | ')})`);
    }
    /**
     * Start the integration
     */
    async start() {
        if (!this.state?.isInitialized) {
            throw new Error('Integration not initialized. Call initialize() first.');
        }
        if (this.state.isStarted) {
            structuredLogger_1.apiLogger.warn('Integration already started', this.correlationContext);
            return;
        }
        structuredLogger_1.apiLogger.info('Starting BubbleLab Integration', this.correlationContext);
        try {
            // Start health checks
            this.state.registry.startHealthChecks();
            // Initialize all plugins
            const plugins = this.state.registry.getAllPlugins();
            for (const plugin of plugins) {
                if (plugin.metadata.enabled) {
                    try {
                        await this.state.registry.initializePlugin(plugin.metadata.name);
                    }
                    catch (error) {
                        structuredLogger_1.apiLogger.error(`Failed to initialize plugin ${plugin.metadata.name}`, error, this.correlationContext);
                    }
                }
            }
            this.state.isStarted = true;
            structuredLogger_1.apiLogger.info('BubbleLab Integration started successfully', {
                ...this.correlationContext,
                plugins_count: plugins.length
            });
        }
        catch (error) {
            structuredLogger_1.apiLogger.error('Failed to start BubbleLab Integration', error, this.correlationContext);
            throw error;
        }
    }
    /**
     * Stop the integration
     */
    async stop() {
        if (!this.state?.isStarted) {
            return;
        }
        structuredLogger_1.apiLogger.info('Stopping BubbleLab Integration', this.correlationContext);
        try {
            // Stop health checks
            this.state.registry.stopHealthChecks();
            this.state.isStarted = false;
            structuredLogger_1.apiLogger.info('BubbleLab Integration stopped', this.correlationContext);
        }
        catch (error) {
            structuredLogger_1.apiLogger.error('Error stopping BubbleLab Integration', error, this.correlationContext);
        }
    }
    /**
     * Destroy the integration
     */
    async destroy() {
        if (!this.state) {
            return;
        }
        structuredLogger_1.apiLogger.info('Destroying BubbleLab Integration', this.correlationContext);
        try {
            await this.stop();
            // Destroy registry
            await this.state.registry.destroy();
            // Destroy event integration
            this.state.eventIntegration.destroy();
            this.state = null;
            structuredLogger_1.apiLogger.info('BubbleLab Integration destroyed', this.correlationContext);
        }
        catch (error) {
            structuredLogger_1.apiLogger.error('Error destroying BubbleLab Integration', error, this.correlationContext);
        }
    }
    /**
     * Get registry
     */
    getRegistry() {
        return this.state?.registry || null;
    }
    /**
     * Get orchestrator
     */
    getOrchestrator() {
        return this.state?.orchestrator || null;
    }
    /**
     * Get integration status
     */
    getStatus() {
        if (!this.state) {
            return {
                initialized: false,
                started: false,
                pluginCount: 0,
                healthyPlugins: 0
            };
        }
        const stats = this.state.registry.getStatistics();
        return {
            initialized: this.state.isInitialized,
            started: this.state.isStarted,
            pluginCount: stats.totalPlugins,
            healthyPlugins: stats.healthyPlugins
        };
    }
}
exports.BubbleLabIntegration = BubbleLabIntegration;
// Global singleton instance
let globalIntegration = null;
/**
 * Initialize the BubbleLab Integration
 */
async function initializeBubbleLabIntegration(config) {
    if (!globalIntegration) {
        globalIntegration = new BubbleLabIntegration(config);
        await globalIntegration.initialize();
    }
    return globalIntegration;
}
/**
 * Get the global BubbleLab Integration instance
 */
function getBubbleLabIntegration() {
    return globalIntegration;
}
/**
 * Reset the global BubbleLab Integration (for testing)
 */
async function resetBubbleLabIntegration() {
    if (globalIntegration) {
        await globalIntegration.destroy();
        globalIntegration = null;
    }
}
//# sourceMappingURL=plugin-integration.js.map