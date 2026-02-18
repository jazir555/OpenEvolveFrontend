/**
 * Plugin Adapters
 *
 * Adapters to wrap existing plugins (RAGBits, Datapizza) to implement PluginInterface.
 * This allows the existing plugins to work with the plugin registry and workflow orchestrator.
 */

import type { PluginInterface, PluginMetadata, PluginCapabilities, PluginContext } from './plugin-registry';

// RAGBits plugin type imports
type RAGBitsPlugin = any; // Will be imported from @bubblelabs-ragbits-plugin
type RAGBitsPluginConfig = any;

// Datapizza plugin type imports
type DatapizzaPlugin = any; // Will be imported from @datapizza-bubblelab-plugin
type DatapizzaPluginConfig = any;

/**
 * Adapter for RAGBits plugin
 */
export class RAGBitsPluginAdapter implements PluginInterface {
  private plugin: RAGBitsPlugin;
  private config: RAGBitsPluginConfig;

  constructor(plugin: RAGBitsPlugin, config: RAGBitsPluginConfig) {
    this.plugin = plugin;
    this.config = config;
  }

  get metadata(): PluginMetadata {
    return {
      name: 'ragbits',
      version: this.plugin.metadata?.version || '1.0.0',
      description: this.plugin.metadata?.description || 'RAGBits Knowledge Search',
      author: this.plugin.metadata?.author || 'OpenEvolve',
      website: this.plugin.metadata?.website,
      enabled: this.config.enabled ?? true
    };
  }

  get capabilities(): PluginCapabilities {
    return {
      search: true,
      indexing: this.config.autoIndexArtifacts ?? true,
      processing: false,
      verification: false
    };
  }

  async initialize(config?: Record<string, unknown>): Promise<void> {
    if (config) {
      await this.plugin.initialize(config as Partial<RAGBitsPluginConfig>);
    } else {
      await this.plugin.initialize();
    }
  }

  async updateConfig(config: Record<string, unknown>): Promise<void> {
    await this.plugin.updateConfig(config as Partial<RAGBitsPluginConfig>);
    this.config = { ...this.config, ...config };
  }

  async resetConfig(): Promise<void> {
    await this.plugin.resetConfig();
  }

  async healthCheck(): Promise<boolean> {
    // RAGBits plugin doesn't have explicit healthCheck, so we test search
    try {
      const result = await this.plugin.search({
        query: '__health_check__',
        topK: 1,
        searchType: 'semantic' as any
      });
      return result.success !== false;
    } catch {
      return false;
    }
  }

  getContext(): PluginContext {
    return {
      config: this.config,
      state: this.plugin.getContext?.()?.state || {}
    };
  }

  getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error' {
    const status = this.plugin.getStatus?.();
    // Map RAGBits status to PluginInterface status
    if (status === 'ready') return 'ready';
    if (status === 'busy') return 'busy';
    if (status === 'error') return 'error';
    if (status === 'initializing') return 'initializing';
    return 'idle';
  }

  async destroy(): Promise<void> {
    // RAGBits plugin doesn't have explicit destroy
    // We just mark as idle
    this.plugin.getStatus = () => 'idle';
  }

  // Delegate to RAGBits plugin methods
  async search(request: any): Promise<any> {
    return await this.plugin.search(request);
  }

  async ingest(request: any): Promise<any> {
    return await this.plugin.ingest(request);
  }

  async batchIngest(requests: any[]): Promise<any[]> {
    return await this.plugin.batchIngest(requests);
  }

  async getIndexStats(): Promise<any> {
    return await this.plugin.getIndexStats();
  }

  async clearCache(): Promise<void> {
    return await this.plugin.clearCache();
  }
}

/**
 * Adapter for Datapizza plugin
 */
export class DatapizzaPluginAdapter implements PluginInterface {
  private plugin: DatapizzaPlugin;
  private config: DatapizzaPluginConfig;

  constructor(plugin: DatapizzaPlugin, config: DatapizzaPluginConfig) {
    this.plugin = plugin;
    this.config = config;
  }

  get metadata(): PluginMetadata {
    return {
      name: 'datapizza',
      version: this.plugin.metadata?.version || '1.0.0',
      description: this.plugin.metadata?.description || 'Datapizza Data Processing',
      author: this.plugin.metadata?.author || 'OpenEvolve',
      website: this.plugin.metadata?.website,
      enabled: this.config.enabled ?? true
    };
  }

  get capabilities(): PluginCapabilities {
    return {
      search: false,
      processing: true,
      indexing: false,
      verification: false
    };
  }

  async initialize(config?: Record<string, unknown>): Promise<void> {
    if (config) {
      await this.plugin.initialize(config as Partial<DatapizzaPluginConfig>);
    } else {
      await this.plugin.initialize();
    }
  }

  async updateConfig(config: Record<string, unknown>): Promise<void> {
    await this.plugin.updateConfig(config as Partial<DatapizzaPluginConfig>);
    this.config = { ...this.config, ...config };
  }

  async resetConfig(): Promise<void> {
    await this.plugin.resetConfig();
  }

  async healthCheck(): Promise<boolean> {
    // Datapizza plugin doesn't have explicit healthCheck
    // We test a simple processing operation
    try {
      const result = await this.plugin.isProcessableData({});
      return typeof result === 'boolean';
    } catch {
      return false;
    }
  }

  getContext(): PluginContext {
    return {
      config: this.config,
      state: this.plugin.getContext?.()?.state || {}
    };
  }

  getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error' {
    const status = this.plugin.getStatus?.();
    // Map Datapizza status to PluginInterface status
    if (status === 'ready') return 'ready';
    if (status === 'busy') return 'busy';
    if (status === 'error') return 'error';
    if (status === 'initializing') return 'initializing';
    return 'idle';
  }

  async destroy(): Promise<void> {
    // Datapizza plugin doesn't have explicit destroy
    // We just mark as idle
    this.plugin.getStatus = () => 'idle';
  }

  // Delegate to Datapizza plugin methods
  async runPipeline(dataSource: string, pipelineType?: string): Promise<any> {
    return await this.plugin.runPipeline(dataSource, pipelineType);
  }

  async processData(data: any, processingType?: string): Promise<any> {
    return await this.plugin.processData(data, processingType);
  }

  async queryData(query: string, dataSource?: string): Promise<any> {
    return await this.plugin.queryData(query, dataSource);
  }

  async getPipelineRecommendation(dataSource: string, context?: string): Promise<string> {
    return await this.plugin.getPipelineRecommendation(dataSource, context);
  }

  async detectDataDomain(data: any): Promise<string | null> {
    return await this.plugin.detectDataDomain(data);
  }

  async isProcessableData(data: any): Promise<boolean> {
    return await this.plugin.isProcessableData(data);
  }

  async clearCache(): Promise<void> {
    return await this.plugin.clearCache();
  }

  getStatistics(): any {
    return this.plugin.getStatistics();
  }

  getOperationHistory(): any {
    return this.plugin.getOperationHistory();
  }

  clearOperationHistory(): void {
    this.plugin.clearOperationHistory();
  }
}

/**
 * OpenEvolve API Adapter
 *
 * Wraps the OpenEvolve API as a plugin for use in workflows
 */
export class OpenEvolveApiAdapter implements PluginInterface {
  private api: any; // openevolveApi
  private config: { apiKey?: string; baseUrl?: string };

  constructor(api: any, config: { apiKey?: string; baseUrl?: string }) {
    this.api = api;
    this.config = config;
  }

  private getApiConfig(): { apiKey?: string; baseUrl?: string } {
    const apiConfig: { apiKey?: string; baseUrl?: string } = {};
    if (this.config.apiKey) {
      apiConfig.apiKey = this.config.apiKey;
    }
    if (this.config.baseUrl) {
      apiConfig.baseUrl = this.config.baseUrl;
    }
    return apiConfig;
  }

  get metadata(): PluginMetadata {
    return {
      name: 'openevolve',
      version: '1.0.0',
      description: 'OpenEvolve Core API',
      author: 'OpenEvolve',
      enabled: true
    };
  }

  get capabilities(): PluginCapabilities {
    return {
      search: true,
      processing: false,
      indexing: true,
      verification: true,
      analysis: true
    };
  }

  async initialize(): Promise<void> {
    // OpenEvolve API is always initialized
  }

  async updateConfig(config: Record<string, unknown>): Promise<void> {
    this.config = { ...this.config, ...config };
  }

  async resetConfig(): Promise<void> {
    // Reset to default
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.api.getHealth(this.getApiConfig());
      return true;
    } catch {
      return false;
    }
  }

  getContext(): PluginContext {
    return {
      config: this.config,
      state: {}
    };
  }

  getStatus(): 'idle' | 'initializing' | 'ready' | 'busy' | 'error' {
    return 'ready';
  }

  async destroy(): Promise<void> {
    // No cleanup needed
  }

  // BubbleLabs integration methods
  async bubblelabsZ3Prove(payload: any): Promise<any> {
    return await this.api.bubblelabsZ3Prove(payload, this.getApiConfig());
  }

  async bubblelabsZ3Solve(payload: any): Promise<any> {
    return await this.api.bubblelabsZ3Solve(payload, this.getApiConfig());
  }

  async bubblelabsLeanAideProve(payload: any): Promise<any> {
    return await this.api.bubblelabsLeanAideProve(payload, this.getApiConfig());
  }

  async bubblelabsRomaAnalyze(payload: any): Promise<any> {
    return await this.api.bubblelabsRomaAnalyze(payload, this.getApiConfig());
  }

  async bubblelabsKnowledgeStore(payload: any): Promise<any> {
    return await this.api.bubblelabsKnowledgeStore(payload, this.getApiConfig());
  }

  async bubblelabsKnowledgeExtract(payload: any): Promise<any> {
    return await this.api.bubblelabsKnowledgeExtract(payload, this.getApiConfig());
  }

  async bubblelabsAnalyticsTrack(payload: any): Promise<any> {
    return await this.api.bubblelabsAnalyticsTrack(payload, this.getApiConfig());
  }

  async bubblelabsAnalyticsDashboard(): Promise<any> {
    return await this.api.bubblelabsAnalyticsDashboard(this.getApiConfig());
  }
}
