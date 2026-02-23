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
import { type PluginRegistry } from './plugin-registry';
import { type WorkflowOrchestrator } from './workflow-orchestrator';
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
    openevolve?: {
        apiKey?: string;
        baseUrl?: string;
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
declare class BubbleLabIntegration {
    private state;
    private config;
    private correlationContext;
    constructor(config?: BubbleLabIntegrationConfig);
    /**
     * Initialize the integration
     */
    initialize(): Promise<void>;
    /**
     * Register all plugins
     */
    private registerPlugins;
    private resolveOpenEvolveApiKey;
    private resolveOpenEvolveBaseUrl;
    private resolvePluginFactory;
    /**
     * Start the integration
     */
    start(): Promise<void>;
    /**
     * Stop the integration
     */
    stop(): Promise<void>;
    /**
     * Destroy the integration
     */
    destroy(): Promise<void>;
    /**
     * Get registry
     */
    getRegistry(): PluginRegistry | null;
    /**
     * Get orchestrator
     */
    getOrchestrator(): WorkflowOrchestrator | null;
    /**
     * Get integration status
     */
    getStatus(): {
        initialized: boolean;
        started: boolean;
        pluginCount: number;
        healthyPlugins: number;
    };
}
/**
 * Initialize the BubbleLab Integration
 */
export declare function initializeBubbleLabIntegration(config?: BubbleLabIntegrationConfig): Promise<BubbleLabIntegration>;
/**
 * Get the global BubbleLab Integration instance
 */
export declare function getBubbleLabIntegration(): BubbleLabIntegration | null;
/**
 * Reset the global BubbleLab Integration (for testing)
 */
export declare function resetBubbleLabIntegration(): Promise<void>;
export { BubbleLabIntegration };
export type { IntegrationState };
//# sourceMappingURL=plugin-integration.d.ts.map