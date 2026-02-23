/**
 * Plugin Adapters
 *
 * Adapters to wrap existing plugins (RAGBits, Datapizza) to implement PluginInterface.
 * This allows the existing plugins to work with the plugin registry and workflow orchestrator.
 */
import type { PluginInterface, PluginMetadata, PluginCapabilities, PluginContext } from './plugin-registry';
type RAGBitsPlugin = any;
type RAGBitsPluginConfig = any;
type DatapizzaPlugin = any;
type DatapizzaPluginConfig = any;
/**
 * Adapter for RAGBits plugin
 */
export declare class RAGBitsPluginAdapter implements PluginInterface {
    private plugin;
    private config;
    constructor(plugin: RAGBitsPlugin, config: RAGBitsPluginConfig);
    get metadata(): PluginMetadata;
    get capabilities(): PluginCapabilities;
    initialize(config?: Record<string, unknown>): Promise<void>;
    updateConfig(config: Record<string, unknown>): Promise<void>;
    resetConfig(): Promise<void>;
    healthCheck(): Promise<boolean>;
    getContext(): PluginContext;
    getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error';
    destroy(): Promise<void>;
    search(request: any): Promise<any>;
    ingest(request: any): Promise<any>;
    batchIngest(requests: any[]): Promise<any[]>;
    getIndexStats(): Promise<any>;
    clearCache(): Promise<void>;
}
/**
 * Adapter for Datapizza plugin
 */
export declare class DatapizzaPluginAdapter implements PluginInterface {
    private plugin;
    private config;
    constructor(plugin: DatapizzaPlugin, config: DatapizzaPluginConfig);
    get metadata(): PluginMetadata;
    get capabilities(): PluginCapabilities;
    initialize(config?: Record<string, unknown>): Promise<void>;
    updateConfig(config: Record<string, unknown>): Promise<void>;
    resetConfig(): Promise<void>;
    healthCheck(): Promise<boolean>;
    getContext(): PluginContext;
    getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error';
    destroy(): Promise<void>;
    runPipeline(dataSource: string, pipelineType?: string): Promise<any>;
    processData(data: any, processingType?: string): Promise<any>;
    queryData(query: string, dataSource?: string): Promise<any>;
    getPipelineRecommendation(dataSource: string, context?: string): Promise<string>;
    detectDataDomain(data: any): Promise<string | null>;
    isProcessableData(data: any): Promise<boolean>;
    clearCache(): Promise<void>;
    getStatistics(): any;
    getOperationHistory(): any;
    clearOperationHistory(): void;
}
/**
 * OpenEvolve API Adapter
 *
 * Wraps the OpenEvolve API as a plugin for use in workflows
 */
export declare class OpenEvolveApiAdapter implements PluginInterface {
    private api;
    private config;
    constructor(api: any, config: {
        apiKey: string;
        baseUrl?: string;
    });
    get metadata(): PluginMetadata;
    get capabilities(): PluginCapabilities;
    initialize(): Promise<void>;
    updateConfig(config: Record<string, unknown>): Promise<void>;
    resetConfig(): Promise<void>;
    healthCheck(): Promise<boolean>;
    getContext(): PluginContext;
    getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error';
    destroy(): Promise<void>;
    bubblelabsZ3Prove(payload: any): Promise<any>;
    bubblelabsLeanAideProve(payload: any): Promise<any>;
    bubblelabsRomaAnalyze(payload: any): Promise<any>;
    bubblelabsKnowledgeStore(payload: any): Promise<any>;
    bubblelabsKnowledgeExtract(payload: any): Promise<any>;
    bubblelabsAnalyticsTrack(payload: any): Promise<any>;
    bubblelabsAnalyticsDashboard(): Promise<any>;
    getGauntlet(gauntletName: string): Promise<any>;
    createGauntlet(gauntlet: any): Promise<any>;
    updateGauntlet(gauntletName: string, gauntlet: any): Promise<any>;
    executeGauntlet(gauntletName: string, payload: any): Promise<any>;
    getGauntletExecutionStatus(executionId: string): Promise<any>;
    executeDecomposition(workflowId: string, payload: any): Promise<any>;
    getDecompositionExecutionStatus(executionId: string): Promise<any>;
    createWorkflow(payload: any): Promise<any>;
    getWorkflowPlan(workflowId: string): Promise<any>;
    getWorkflowResults(workflowId: string): Promise<any>;
    startEvolutionRun(payload: any): Promise<any>;
    getEvolutionRun(runId: string): Promise<any>;
    executeWorkflowTemplate(templateId: string, payload: any): Promise<any>;
    getWorkflowTemplateExecutionStatus(executionId: string): Promise<any>;
    getExecutionStatus(executionType: 'gauntlet' | 'decomposition' | 'workflow-template', executionId: string): Promise<any>;
}
export type { RAGBitsPluginAdapter, DatapizzaPluginAdapter, OpenEvolveApiAdapter };
//# sourceMappingURL=plugin-adapters.d.ts.map