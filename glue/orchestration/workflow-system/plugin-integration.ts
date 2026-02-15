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

import { apiLogger, LogContext } from '../../lib/structuredLogger';
import { getPluginRegistry, type PluginRegistry } from './plugin-registry';
import { getWorkflowOrchestrator, type WorkflowOrchestrator } from './workflow-orchestrator';
import { getPluginEventIntegration } from './plugin-events';
import { openevolveApi } from '../../adapters/bubblelab/src/lib/openevolveApi';
import {
  RAGBitsPluginAdapter,
  DatapizzaPluginAdapter,
  OpenEvolveApiAdapter
} from './plugin-adapters';

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

interface IntegrationState {
  registry: PluginRegistry;
  orchestrator: WorkflowOrchestrator;
  eventIntegration: any;
  isInitialized: boolean;
  isStarted: boolean;
}

class BubbleLabIntegration {
  private state: IntegrationState | null = null;
  private config: BubbleLabIntegrationConfig;
  private correlationContext: LogContext;

  constructor(config: BubbleLabIntegrationConfig = {}) {
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

    apiLogger.info('BubbleLab Integration created', {
      ...this.correlationContext,
      config: this.config
    });
  }

  /**
   * Initialize the integration
   */
  async initialize(): Promise<void> {
    if (this.state?.isInitialized) {
      apiLogger.warn('Integration already initialized', this.correlationContext);
      return;
    }

    apiLogger.info('Initializing BubbleLab Integration', this.correlationContext);

    try {
      // Initialize core components
      const registry = getPluginRegistry({
        autoInitialize: false,
        healthCheckInterval: this.config.healthCheckInterval
      });

      const orchestrator = getWorkflowOrchestrator(registry);
      const eventIntegration = getPluginEventIntegration();

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

      apiLogger.info('BubbleLab Integration initialized successfully', this.correlationContext);

      // Auto-start if configured
      if (this.config.autoStart) {
        await this.start();
      }
    } catch (error) {
      apiLogger.error('Failed to initialize BubbleLab Integration', error as Error, this.correlationContext);
      throw error;
    }
  }

  /**
   * Register all plugins
   */
  private async registerPlugins(registry: PluginRegistry): Promise<void> {
    apiLogger.info('Registering plugins', this.correlationContext);

    // Register OpenEvolve API adapter (always available)
    try {
      const openevolveAdapter = new OpenEvolveApiAdapter(openevolveApi, {
        apiKey: process.env.OPENEVOLVE_API_KEY || '',
        baseUrl: process.env.OPENEVOLVE_API_BASE
      });
      await registry.registerPlugin(openevolveAdapter);
      apiLogger.info('OpenEvolve API adapter registered', this.correlationContext);
    } catch (error) {
      apiLogger.error('Failed to register OpenEvolve adapter', error as Error, this.correlationContext);
    }

    // Register RAGBits plugin
    if (this.config.ragbits?.enabled !== false) {
      try {
        // Dynamically import RAGBits plugin
        const { createPlugin: createRAGBitsPlugin } = await import('@bubblelabs-ragbits-plugin');
        const ragbitsPlugin = createRAGBitsPlugin(this.config.ragbits);
        const ragbitsAdapter = new RAGBitsPluginAdapter(ragbitsPlugin, {
          ...this.config.ragbits,
          enabled: true
        });
        await registry.registerPlugin(ragbitsAdapter);

        apiLogger.info('RAGBits plugin registered', {
          ...this.correlationContext,
          server_url: this.config.ragbits?.serverUrl
        });
      } catch (error) {
        apiLogger.warn('RAGBits plugin not available', error as Error, this.correlationContext);
      }
    }

    // Register Datapizza plugin
    if (this.config.datapizza?.enabled !== false) {
      try {
        // Dynamically import Datapizza plugin
        const { createPlugin: createDatapizzaPlugin } = await import('@datapizza-bubblelab-plugin');
        const datapizzaPlugin = createDatapizzaPlugin(this.config.datapizza);
        const datapizzaAdapter = new DatapizzaPluginAdapter(datapizzaPlugin, {
          ...this.config.datapizza,
          enabled: true
        });
        await registry.registerPlugin(datapizzaAdapter);

        apiLogger.info('Datapizza plugin registered', {
          ...this.correlationContext,
          server_url: this.config.datapizza?.serverUrl
        });
      } catch (error) {
        apiLogger.warn('Datapizza plugin not available', error as Error, this.correlationContext);
      }
    }

    apiLogger.info('All plugins registered', this.correlationContext);
  }

  /**
   * Start the integration
   */
  async start(): Promise<void> {
    if (!this.state?.isInitialized) {
      throw new Error('Integration not initialized. Call initialize() first.');
    }

    if (this.state.isStarted) {
      apiLogger.warn('Integration already started', this.correlationContext);
      return;
    }

    apiLogger.info('Starting BubbleLab Integration', this.correlationContext);

    try {
      // Start health checks
      this.state.registry.startHealthChecks();

      // Initialize all plugins
      const plugins = this.state.registry.getAllPlugins();
      for (const plugin of plugins) {
        if (plugin.metadata.enabled) {
          try {
            await this.state.registry.initializePlugin(plugin.metadata.name);
          } catch (error) {
            apiLogger.error(`Failed to initialize plugin ${plugin.metadata.name}`, error as Error, this.correlationContext);
          }
        }
      }

      this.state.isStarted = true;

      apiLogger.info('BubbleLab Integration started successfully', {
        ...this.correlationContext,
        plugins_count: plugins.length
      });
    } catch (error) {
      apiLogger.error('Failed to start BubbleLab Integration', error as Error, this.correlationContext);
      throw error;
    }
  }

  /**
   * Stop the integration
   */
  async stop(): Promise<void> {
    if (!this.state?.isStarted) {
      return;
    }

    apiLogger.info('Stopping BubbleLab Integration', this.correlationContext);

    try {
      // Stop health checks
      this.state.registry.stopHealthChecks();

      this.state.isStarted = false;

      apiLogger.info('BubbleLab Integration stopped', this.correlationContext);
    } catch (error) {
      apiLogger.error('Error stopping BubbleLab Integration', error as Error, this.correlationContext);
    }
  }

  /**
   * Destroy the integration
   */
  async destroy(): Promise<void> {
    if (!this.state) {
      return;
    }

    apiLogger.info('Destroying BubbleLab Integration', this.correlationContext);

    try {
      await this.stop();

      // Destroy registry
      await this.state.registry.destroy();

      // Destroy event integration
      this.state.eventIntegration.destroy();

      this.state = null;

      apiLogger.info('BubbleLab Integration destroyed', this.correlationContext);
    } catch (error) {
      apiLogger.error('Error destroying BubbleLab Integration', error as Error, this.correlationContext);
    }
  }

  /**
   * Get registry
   */
  getRegistry(): PluginRegistry | null {
    return this.state?.registry || null;
  }

  /**
   * Get orchestrator
   */
  getOrchestrator(): WorkflowOrchestrator | null {
    return this.state?.orchestrator || null;
  }

  /**
   * Get integration status
   */
  getStatus(): {
    initialized: boolean;
    started: boolean;
    pluginCount: number;
    healthyPlugins: number;
  } {
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

// Global singleton instance
let globalIntegration: BubbleLabIntegration | null = null;

/**
 * Initialize the BubbleLab Integration
 */
export async function initializeBubbleLabIntegration(
  config?: BubbleLabIntegrationConfig
): Promise<BubbleLabIntegration> {
  if (!globalIntegration) {
    globalIntegration = new BubbleLabIntegration(config);
    await globalIntegration.initialize();
  }
  return globalIntegration;
}

/**
 * Get the global BubbleLab Integration instance
 */
export function getBubbleLabIntegration(): BubbleLabIntegration | null {
  return globalIntegration;
}

/**
 * Reset the global BubbleLab Integration (for testing)
 */
export async function resetBubbleLabIntegration(): Promise<void> {
  if (globalIntegration) {
    await globalIntegration.destroy();
    globalIntegration = null;
  }
}

export { BubbleLabIntegration };
export type { BubbleLabIntegrationConfig, IntegrationState };
