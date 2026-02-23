"use strict";
// RAGBits Plugin Factory
// Creates a fully configured RAGBits plugin instance
Object.defineProperty(exports, "__esModule", { value: true });
exports.createRAGBitsPlugin = createRAGBitsPlugin;
exports.getRAGBitsPlugin = getRAGBitsPlugin;
exports.useRAGBitsPlugin = useRAGBitsPlugin;
const react_1 = require("react");
const react_toastify_1 = require("react-toastify");
const plugin_types_1 = require("../types/plugin-types");
const ragbitsClient_1 = require("../lib/ragbitsClient");
const ragbitsService_1 = require("../services/ragbitsService");
const RAGBitsConfigPanel_1 = require("../components/RAGBitsConfigPanel");
const RAGBitsIngestPanel_1 = require("../components/RAGBitsIngestPanel");
const RAGBitsSearchPanel_1 = require("../components/RAGBitsSearchPanel");
const RAGBitsSearchResults_1 = require("../components/RAGBitsSearchResults");
const RAGBitsStatusIndicator_1 = require("../components/RAGBitsStatusIndicator");
const useRAGBitsConfig_1 = require("../hooks/useRAGBitsConfig");
const useRAGBitsIngest_1 = require("../hooks/useRAGBitsIngest");
const useRAGBitsSearch_1 = require("../hooks/useRAGBitsSearch");
const useRAGBitsState_1 = require("../hooks/useRAGBitsState");
// Global plugin state management
let globalPluginState = {
    ...plugin_types_1.DEFAULT_RAGBITS_CONFIG,
    status: 'idle',
    operationHistory: [],
    statistics: {
        totalSearches: 0,
        successfulSearches: 0,
        failedSearches: 0,
        totalDocumentsIndexed: 0,
        averageSearchTime: 0,
        averageRelevanceScore: 0
    }
};
let globalPluginInstance = null;
/**
 * Create a new RAGBits plugin instance
 * @param initialConfig Optional initial configuration
 * @returns RAGBitsPlugin instance
 */
function createRAGBitsPlugin(initialConfig) {
    if (globalPluginInstance) {
        // Return existing instance if available
        return globalPluginInstance;
    }
    // Merge initial config with defaults
    const config = {
        ...plugin_types_1.DEFAULT_RAGBITS_CONFIG,
        ...initialConfig
    };
    // Initialize plugin state
    const state = {
        ...config,
        status: 'initializing',
        operationHistory: [],
        statistics: {
            totalSearches: 0,
            successfulSearches: 0,
            failedSearches: 0,
            totalDocumentsIndexed: 0,
            averageSearchTime: 0,
            averageRelevanceScore: 0
        }
    };
    // Update global state
    globalPluginState = state;
    // Create client and service instances
    const client = new ragbitsClient_1.RagbitsClient({
        serverUrl: config.serverUrl,
        apiKey: config.apiKey,
        timeout: config.timeout
    });
    const service = new ragbitsService_1.RagbitsService(client, config.enableCaching ? config.cacheTTLSeconds : 0);
    // Create plugin methods
    const pluginMethods = {
        metadata: {
            name: 'RAGBits Knowledge Search',
            version: '1.0.0',
            description: 'BubbleLabs plugin for semantic document search and knowledge retrieval',
            author: 'OpenEvolve',
            website: 'https://openevolve.com'
        },
        // Plugin methods
        async initialize(configUpdate) {
            try {
                state.status = 'initializing';
                if (configUpdate) {
                    Object.assign(config, configUpdate);
                    Object.assign(state, config);
                }
                // Initialize client with updated config
                client.configure({
                    serverUrl: config.serverUrl,
                    apiKey: config.apiKey,
                    timeout: config.timeout
                });
                // Test connection
                const connected = await client.testConnection();
                if (!connected) {
                    throw new Error('Failed to connect to RAGBits server');
                }
                state.status = 'ready';
                react_toastify_1.toast.success('RAGBits plugin initialized successfully');
            }
            catch (error) {
                state.status = 'error';
                react_toastify_1.toast.error(`Failed to initialize RAGBits plugin: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async updateConfig(configUpdate) {
            try {
                Object.assign(config, configUpdate);
                Object.assign(state, config);
                // Update client configuration
                client.configure({
                    serverUrl: config.serverUrl,
                    apiKey: config.apiKey,
                    timeout: config.timeout
                });
                react_toastify_1.toast.success('RAGBits configuration updated successfully');
            }
            catch (error) {
                react_toastify_1.toast.error(`Failed to update configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async resetConfig() {
            Object.assign(config, plugin_types_1.DEFAULT_RAGBITS_CONFIG);
            Object.assign(state, plugin_types_1.DEFAULT_RAGBITS_CONFIG);
            client.configure({
                serverUrl: config.serverUrl,
                apiKey: config.apiKey,
                timeout: config.timeout
            });
            react_toastify_1.toast.success('RAGBits configuration reset to defaults');
        },
        async search(request) {
            if (!config.enabled) {
                throw new Error('RAGBits plugin is disabled');
            }
            if (state.status !== 'ready') {
                throw new Error(`Plugin not ready. Current status: ${state.status}`);
            }
            try {
                state.status = 'busy';
                state.currentOperation = {
                    type: 'search',
                    startedAt: new Date(),
                    message: `Searching: ${request.query.substring(0, 50)}...`
                };
                const startTime = Date.now();
                const result = await service.search(request);
                const executionTime = Date.now() - startTime;
                // Update statistics
                state.statistics.totalSearches++;
                if (result.success) {
                    state.statistics.successfulSearches++;
                    state.statistics.averageSearchTime = ((state.statistics.averageSearchTime * (state.statistics.totalSearches - 1)) +
                        executionTime) / state.statistics.totalSearches;
                    // Calculate average relevance score
                    if (result.results.length > 0) {
                        const avgScore = result.results.reduce((sum, r) => sum + r.relevanceScore, 0) / result.results.length;
                        state.statistics.averageRelevanceScore = avgScore;
                    }
                }
                else {
                    state.statistics.failedSearches++;
                }
                state.statistics.lastOperationTime = new Date();
                // Add to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'search',
                    timestamp: new Date(),
                    success: result.success,
                    message: `Search ${result.success ? 'succeeded' : 'failed'}: ${request.query.substring(0, 50)}...`,
                    details: {
                        query: request.query,
                        resultsCount: result.results.length,
                        executionTime
                    }
                });
                // Keep history size manageable
                if (state.operationHistory.length > 100) {
                    state.operationHistory = state.operationHistory.slice(0, 100);
                }
                state.status = 'ready';
                state.currentOperation = undefined;
                return {
                    ...result,
                    executionTime: executionTime / 1000,
                    timestamp: new Date()
                };
            }
            catch (error) {
                state.status = 'error';
                state.currentOperation = undefined;
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                // Add to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'search',
                    timestamp: new Date(),
                    success: false,
                    message: `Search failed: ${errorMessage}`,
                    details: { query: request.query, error: errorMessage }
                });
                throw new Error(`Search failed: ${errorMessage}`);
            }
        },
        async ingest(request) {
            if (!config.enabled) {
                throw new Error('RAGBits plugin is disabled');
            }
            if (state.status !== 'ready') {
                throw new Error(`Plugin not ready. Current status: ${state.status}`);
            }
            try {
                state.status = 'busy';
                state.currentOperation = {
                    type: 'ingest',
                    startedAt: new Date(),
                    message: 'Ingesting document...'
                };
                const result = await service.ingest(request);
                // Update statistics
                state.statistics.totalDocumentsIndexed++;
                // Add to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'ingest',
                    timestamp: new Date(),
                    success: result.success,
                    message: `Document ingested: ${request.content.substring(0, 50)}...`,
                    details: {
                        documentId: result.documentId,
                        documentType: request.metadata.documentType
                    }
                });
                state.status = 'ready';
                state.currentOperation = undefined;
                return result;
            }
            catch (error) {
                state.status = 'error';
                state.currentOperation = undefined;
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                throw new Error(`Ingest failed: ${errorMessage}`);
            }
        },
        async batchIngest(requests) {
            const results = [];
            for (const request of requests) {
                const result = await this.ingest(request);
                results.push(result);
            }
            return results;
        },
        async getIndexStats() {
            if (state.status !== 'ready') {
                throw new Error(`Plugin not ready. Current status: ${state.status}`);
            }
            try {
                return await service.getIndexStats();
            }
            catch (error) {
                throw new Error(`Failed to get index stats: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
        },
        async clearCache() {
            try {
                await service.clearCache();
                react_toastify_1.toast.success('RAGBits cache cleared successfully');
            }
            catch (error) {
                react_toastify_1.toast.error(`Failed to clear cache: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        getStatistics() {
            return state.statistics;
        },
        getOperationHistory() {
            return state.operationHistory;
        },
        clearOperationHistory() {
            state.operationHistory = [];
            react_toastify_1.toast.success('Operation history cleared');
        },
        getStatus() {
            return state.status;
        },
        getContext() {
            return {
                config,
                state,
                searchTypes: plugin_types_1.RAGBITS_SEARCH_TYPES,
                documentTypes: plugin_types_1.RAGBITS_DOCUMENT_TYPES,
                capabilities: {
                    semanticSearch: true,
                    hybridSearch: config.enableHybridSearch,
                    keywordSearch: true,
                    reranking: config.enableReranking,
                    caching: config.enableCaching,
                    indexing: config.autoIndexArtifacts,
                    monitoring: true,
                    reporting: true
                }
            };
        },
        // React components (will be imported dynamically)
        components: {
            ConfigPanel: RAGBitsConfigPanel_1.RAGBitsConfigPanel,
            SearchPanel: RAGBitsSearchPanel_1.RAGBitsSearchPanel,
            IngestPanel: RAGBitsIngestPanel_1.RAGBitsIngestPanel,
            StatusIndicator: RAGBitsStatusIndicator_1.RAGBitsStatusIndicator,
            SearchResults: RAGBitsSearchResults_1.RAGBitsSearchResults
        },
        // React hooks
        hooks: {
            useRAGBitsConfig: useRAGBitsConfig_1.useRAGBitsConfig,
            useRAGBitsState: useRAGBitsState_1.useRAGBitsState,
            useRAGBitsSearch: useRAGBitsSearch_1.useRAGBitsSearch,
            useRAGBitsIngest: useRAGBitsIngest_1.useRAGBitsIngest
        }
    };
    // Set global instance
    globalPluginInstance = pluginMethods;
    return pluginMethods;
}
/**
 * Get the global plugin instance
 * @returns RAGBitsPlugin instance
 */
function getRAGBitsPlugin() {
    if (!globalPluginInstance) {
        globalPluginInstance = createRAGBitsPlugin();
    }
    return globalPluginInstance;
}
/**
 * React hook to use the RAGBits plugin
 * @returns RAGBitsPlugin instance
 */
function useRAGBitsPlugin() {
    const [plugin] = (0, react_1.useState)(() => getRAGBitsPlugin());
    (0, react_1.useEffect)(() => {
        // Initialize plugin if not already initialized
        if (plugin.getStatus() === 'idle') {
            plugin.initialize();
        }
    }, [plugin]);
    return plugin;
}
//# sourceMappingURL=createRAGBitsPlugin.js.map