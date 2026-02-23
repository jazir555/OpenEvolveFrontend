"use strict";
/**
 * Plugin Adapters
 *
 * Adapters to wrap existing plugins (RAGBits, Datapizza) to implement PluginInterface.
 * This allows the existing plugins to work with the plugin registry and workflow orchestrator.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.OpenEvolveApiAdapter = exports.DatapizzaPluginAdapter = exports.RAGBitsPluginAdapter = void 0;
/**
 * Adapter for RAGBits plugin
 */
class RAGBitsPluginAdapter {
    constructor(plugin, config) {
        this.plugin = plugin;
        this.config = config;
    }
    get metadata() {
        return {
            name: 'ragbits',
            version: this.plugin.metadata?.version || '1.0.0',
            description: this.plugin.metadata?.description || 'RAGBits Knowledge Search',
            author: this.plugin.metadata?.author || 'OpenEvolve',
            website: this.plugin.metadata?.website,
            enabled: this.config.enabled ?? true
        };
    }
    get capabilities() {
        return {
            search: true,
            indexing: this.config.autoIndexArtifacts ?? true,
            processing: false,
            verification: false
        };
    }
    async initialize(config) {
        if (config) {
            await this.plugin.initialize(config);
        }
        else {
            await this.plugin.initialize();
        }
    }
    async updateConfig(config) {
        await this.plugin.updateConfig(config);
        this.config = { ...this.config, ...config };
    }
    async resetConfig() {
        await this.plugin.resetConfig();
    }
    async healthCheck() {
        // RAGBits plugin doesn't have explicit healthCheck, so we test search
        try {
            const result = await this.plugin.search({
                query: '__health_check__',
                topK: 1,
                searchType: 'semantic'
            });
            return result.success !== false;
        }
        catch {
            return false;
        }
    }
    getContext() {
        return {
            config: this.config,
            state: this.plugin.getContext?.()?.state || {}
        };
    }
    getStatus() {
        const status = this.plugin.getStatus?.();
        // Map RAGBits status to PluginInterface status
        if (status === 'ready')
            return 'ready';
        if (status === 'busy')
            return 'busy';
        if (status === 'error')
            return 'error';
        if (status === 'initializing')
            return 'initializing';
        return 'idle';
    }
    async destroy() {
        // RAGBits plugin doesn't have explicit destroy
        // We just mark as idle
        this.plugin.getStatus = () => 'idle';
    }
    // Delegate to RAGBits plugin methods
    async search(request) {
        return await this.plugin.search(request);
    }
    async ingest(request) {
        return await this.plugin.ingest(request);
    }
    async batchIngest(requests) {
        return await this.plugin.batchIngest(requests);
    }
    async getIndexStats() {
        return await this.plugin.getIndexStats();
    }
    async clearCache() {
        return await this.plugin.clearCache();
    }
}
exports.RAGBitsPluginAdapter = RAGBitsPluginAdapter;
/**
 * Adapter for Datapizza plugin
 */
class DatapizzaPluginAdapter {
    constructor(plugin, config) {
        this.plugin = plugin;
        this.config = config;
    }
    get metadata() {
        return {
            name: 'datapizza',
            version: this.plugin.metadata?.version || '1.0.0',
            description: this.plugin.metadata?.description || 'Datapizza Data Processing',
            author: this.plugin.metadata?.author || 'OpenEvolve',
            website: this.plugin.metadata?.website,
            enabled: this.config.enabled ?? true
        };
    }
    get capabilities() {
        return {
            search: false,
            processing: true,
            indexing: false,
            verification: false
        };
    }
    async initialize(config) {
        if (config) {
            await this.plugin.initialize(config);
        }
        else {
            await this.plugin.initialize();
        }
    }
    async updateConfig(config) {
        await this.plugin.updateConfig(config);
        this.config = { ...this.config, ...config };
    }
    async resetConfig() {
        await this.plugin.resetConfig();
    }
    async healthCheck() {
        // Datapizza plugin doesn't have explicit healthCheck
        // We test a simple processing operation
        try {
            const result = await this.plugin.isProcessableData({});
            return typeof result === 'boolean';
        }
        catch {
            return false;
        }
    }
    getContext() {
        return {
            config: this.config,
            state: this.plugin.getContext?.()?.state || {}
        };
    }
    getStatus() {
        const status = this.plugin.getStatus?.();
        // Map Datapizza status to PluginInterface status
        if (status === 'ready')
            return 'ready';
        if (status === 'busy')
            return 'busy';
        if (status === 'error')
            return 'error';
        if (status === 'initializing')
            return 'initializing';
        return 'idle';
    }
    async destroy() {
        // Datapizza plugin doesn't have explicit destroy
        // We just mark as idle
        this.plugin.getStatus = () => 'idle';
    }
    // Delegate to Datapizza plugin methods
    async runPipeline(dataSource, pipelineType) {
        return await this.plugin.runPipeline(dataSource, pipelineType);
    }
    async processData(data, processingType) {
        return await this.plugin.processData(data, processingType);
    }
    async queryData(query, dataSource) {
        return await this.plugin.queryData(query, dataSource);
    }
    async getPipelineRecommendation(dataSource, context) {
        return await this.plugin.getPipelineRecommendation(dataSource, context);
    }
    async detectDataDomain(data) {
        return await this.plugin.detectDataDomain(data);
    }
    async isProcessableData(data) {
        return await this.plugin.isProcessableData(data);
    }
    async clearCache() {
        return await this.plugin.clearCache();
    }
    getStatistics() {
        return this.plugin.getStatistics();
    }
    getOperationHistory() {
        return this.plugin.getOperationHistory();
    }
    clearOperationHistory() {
        this.plugin.clearOperationHistory();
    }
}
exports.DatapizzaPluginAdapter = DatapizzaPluginAdapter;
/**
 * OpenEvolve API Adapter
 *
 * Wraps the OpenEvolve API as a plugin for use in workflows
 */
class OpenEvolveApiAdapter {
    constructor(api, config) {
        this.api = api;
        this.config = config;
    }
    getApiConfig() {
        const apiConfig = {};
        if (this.config.apiKey) {
            apiConfig.apiKey = this.config.apiKey;
        }
        if (this.config.baseUrl) {
            apiConfig.baseUrl = this.config.baseUrl;
        }
        return apiConfig;
    }
    get metadata() {
        return {
            name: 'openevolve',
            version: '1.0.0',
            description: 'OpenEvolve Core API',
            author: 'OpenEvolve',
            enabled: true
        };
    }
    get capabilities() {
        return {
            search: true,
            processing: false,
            indexing: true,
            verification: true,
            analysis: true
        };
    }
    async initialize() {
        // OpenEvolve API is always initialized
    }
    async updateConfig(config) {
        this.config = { ...this.config, ...config };
    }
    async resetConfig() {
        // Reset to default
    }
    async healthCheck() {
        try {
            await this.api.getHealth(this.getApiConfig());
            return true;
        }
        catch {
            return false;
        }
    }
    getContext() {
        return {
            config: this.config,
            state: {}
        };
    }
    getStatus() {
        return 'ready';
    }
    async destroy() {
        // No cleanup needed
    }
    // BubbleLabs integration methods
    async bubblelabsZ3Prove(payload) {
        return await this.api.bubblelabsZ3Prove(payload, this.getApiConfig());
    }
    async bubblelabsZ3Solve(payload) {
        return await this.api.bubblelabsZ3Solve(payload, this.getApiConfig());
    }
    async bubblelabsLeanAideProve(payload) {
        return await this.api.bubblelabsLeanAideProve(payload, this.getApiConfig());
    }
    async bubblelabsRomaAnalyze(payload) {
        return await this.api.bubblelabsRomaAnalyze(payload, this.getApiConfig());
    }
    async bubblelabsKnowledgeStore(payload) {
        return await this.api.bubblelabsKnowledgeStore(payload, this.getApiConfig());
    }
    async bubblelabsKnowledgeExtract(payload) {
        return await this.api.bubblelabsKnowledgeExtract(payload, this.getApiConfig());
    }
    async bubblelabsAnalyticsTrack(payload) {
        return await this.api.bubblelabsAnalyticsTrack(payload, this.getApiConfig());
    }
    async bubblelabsAnalyticsDashboard() {
        return await this.api.bubblelabsAnalyticsDashboard(this.getApiConfig());
    }
}
exports.OpenEvolveApiAdapter = OpenEvolveApiAdapter;
//# sourceMappingURL=plugin-adapters.js.map