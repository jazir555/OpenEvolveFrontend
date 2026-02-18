// Datapizza Plugin Factory
// Creates a fully configured Datapizza plugin instance
import { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { DEFAULT_DATAPIZZA_CONFIG, DATAPIZZA_PIPELINE_TYPES, DATAPIZZA_DATA_DOMAINS } from '../types/plugin-types';
import { DatapizzaClient } from '../services/DatapizzaClient';
import { DatapizzaService } from '../services/DatapizzaService';
// Global plugin state management
let globalPluginState = { ...DEFAULT_DATAPIZZA_CONFIG, ...{
        status: 'idle',
        operationHistory: [],
        statistics: {
            totalOperations: 0,
            successfulOperations: 0,
            failedOperations: 0,
            averageProcessingTime: 0
        }
    } };
let globalPluginInstance = null;
/**
 * Create a new Datapizza plugin instance
 * @param initialConfig Optional initial configuration
 * @returns DatapizzaPlugin instance
 */
export function createDatapizzaPlugin(initialConfig) {
    if (globalPluginInstance) {
        // Return existing instance if available
        return globalPluginInstance;
    }
    // Merge initial config with defaults
    const config = {
        ...DEFAULT_DATAPIZZA_CONFIG,
        ...initialConfig
    };
    // Initialize plugin state
    const state = {
        ...config,
        status: 'initializing',
        operationHistory: [],
        statistics: {
            totalOperations: 0,
            successfulOperations: 0,
            failedOperations: 0,
            averageProcessingTime: 0
        }
    };
    // Update global state
    globalPluginState = state;
    // Create client and service instances
    const client = new DatapizzaClient({
        baseUrl: config.serverUrl,
        apiKey: config.apiKey,
        timeout: config.timeout ?? DEFAULT_DATAPIZZA_CONFIG.timeout
    });
    const service = new DatapizzaService(client);
    // Create plugin methods
    const pluginMethods = {
        metadata: {
            name: 'Datapizza Data Processing',
            version: '1.0.0',
            description: 'BubbleLabs plugin for data pipeline processing and querying',
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
                    baseUrl: config.serverUrl,
                    apiKey: config.apiKey,
                    timeout: config.timeout ?? DEFAULT_DATAPIZZA_CONFIG.timeout
                });
                // Test connection
                await client.testConnection();
                state.status = 'ready';
                toast.success('Datapizza plugin initialized successfully');
            }
            catch (error) {
                state.status = 'error';
                toast.error(`Failed to initialize Datapizza plugin: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async updateConfig(configUpdate) {
            try {
                Object.assign(config, configUpdate);
                Object.assign(state, config);
                // Update client configuration
                client.configure({
                    baseUrl: config.serverUrl,
                    apiKey: config.apiKey,
                    timeout: config.timeout ?? DEFAULT_DATAPIZZA_CONFIG.timeout
                });
                toast.success('Datapizza configuration updated successfully');
            }
            catch (error) {
                toast.error(`Failed to update configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async resetConfig() {
            Object.assign(config, DEFAULT_DATAPIZZA_CONFIG);
            Object.assign(state, DEFAULT_DATAPIZZA_CONFIG);
            client.configure({
                baseUrl: config.serverUrl,
                apiKey: config.apiKey,
                timeout: config.timeout ?? DEFAULT_DATAPIZZA_CONFIG.timeout
            });
            toast.success('Datapizza configuration reset to defaults');
        },
        async runPipeline(dataSource, pipelineType) {
            if (!config.enabled) {
                throw new Error('Datapizza plugin is disabled');
            }
            if (state.status !== 'ready') {
                throw new Error(`Plugin not ready. Current status: ${state.status}`);
            }
            try {
                state.status = 'busy';
                state.currentOperation = {
                    type: 'pipeline',
                    startedAt: new Date(),
                    message: `Running pipeline for data source: ${dataSource.substring(0, 50)}...`
                };
                const startTime = Date.now();
                const result = await service.runPipeline(dataSource, pipelineType || config.defaultPipelineType);
                const executionTime = Date.now() - startTime;
                // Update statistics
                state.statistics.totalOperations++;
                if (result.success) {
                    state.statistics.successfulOperations++;
                    state.statistics.averageProcessingTime = ((state.statistics.averageProcessingTime * (state.statistics.totalOperations - 1)) +
                        executionTime) / state.statistics.totalOperations;
                }
                else {
                    state.statistics.failedOperations++;
                }
                state.statistics.lastOperationTime = new Date();
                // Add to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'pipeline',
                    timestamp: new Date(),
                    success: result.success,
                    message: `Pipeline ${result.success ? 'succeeded' : 'failed'}: ${dataSource.substring(0, 50)}...`,
                    details: {
                        dataSource,
                        pipelineType: pipelineType || config.defaultPipelineType,
                        confidenceScore: result.confidenceScore,
                        errors: result.errors
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
                    executionTime
                };
            }
            catch (error) {
                state.status = 'error';
                state.currentOperation = undefined;
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                toast.error(`Pipeline failed: ${errorMessage}`);
                // Add error to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'pipeline',
                    timestamp: new Date(),
                    success: false,
                    message: `Pipeline failed: ${dataSource.substring(0, 50)}...`,
                    details: {
                        dataSource,
                        pipelineType: pipelineType || config.defaultPipelineType,
                        error: errorMessage
                    }
                });
                throw error;
            }
        },
        async processData(data, processingType) {
            if (!config.enabled) {
                throw new Error('Datapizza plugin is disabled');
            }
            if (state.status !== 'ready') {
                throw new Error(`Plugin not ready. Current status: ${state.status}`);
            }
            try {
                state.status = 'busy';
                state.currentOperation = {
                    type: 'processing',
                    startedAt: new Date(),
                    message: `Processing data...`
                };
                const startTime = Date.now();
                const result = await service.processData(data, processingType);
                const executionTime = Date.now() - startTime;
                // Update statistics
                state.statistics.totalOperations++;
                if (result.success) {
                    state.statistics.successfulOperations++;
                    state.statistics.averageProcessingTime = ((state.statistics.averageProcessingTime * (state.statistics.totalOperations - 1)) +
                        executionTime) / state.statistics.totalOperations;
                }
                else {
                    state.statistics.failedOperations++;
                }
                state.statistics.lastOperationTime = new Date();
                // Add to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'processing',
                    timestamp: new Date(),
                    success: result.success,
                    message: `Data processing ${result.success ? 'succeeded' : 'failed'}`,
                    details: {
                        dataType: typeof data,
                        processingType,
                        confidenceScore: result.confidenceScore,
                        errors: result.errors
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
                    executionTime
                };
            }
            catch (error) {
                state.status = 'error';
                state.currentOperation = undefined;
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                toast.error(`Data processing failed: ${errorMessage}`);
                // Add error to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'processing',
                    timestamp: new Date(),
                    success: false,
                    message: `Data processing failed`,
                    details: {
                        dataType: typeof data,
                        processingType,
                        error: errorMessage
                    }
                });
                throw error;
            }
        },
        async queryData(query, dataSource) {
            if (!config.enabled) {
                throw new Error('Datapizza plugin is disabled');
            }
            if (state.status !== 'ready') {
                throw new Error(`Plugin not ready. Current status: ${state.status}`);
            }
            try {
                state.status = 'busy';
                state.currentOperation = {
                    type: 'query',
                    startedAt: new Date(),
                    message: `Executing query: ${query.substring(0, 50)}...`
                };
                const startTime = Date.now();
                const result = await service.queryData(query, dataSource);
                const executionTime = Date.now() - startTime;
                // Update statistics
                state.statistics.totalOperations++;
                if (result.success) {
                    state.statistics.successfulOperations++;
                    state.statistics.averageProcessingTime = ((state.statistics.averageProcessingTime * (state.statistics.totalOperations - 1)) +
                        executionTime) / state.statistics.totalOperations;
                }
                else {
                    state.statistics.failedOperations++;
                }
                state.statistics.lastOperationTime = new Date();
                // Add to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'query',
                    timestamp: new Date(),
                    success: result.success,
                    message: `Query ${result.success ? 'succeeded' : 'failed'}: ${query.substring(0, 50)}...`,
                    details: {
                        query,
                        dataSource,
                        confidenceScore: result.confidenceScore,
                        resultCount: result.results.length,
                        errors: result.errors
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
                    processingTime: executionTime
                };
            }
            catch (error) {
                state.status = 'error';
                state.currentOperation = undefined;
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                toast.error(`Query failed: ${errorMessage}`);
                // Add error to operation history
                state.operationHistory.unshift({
                    id: Date.now().toString(),
                    type: 'query',
                    timestamp: new Date(),
                    success: false,
                    message: `Query failed: ${query.substring(0, 50)}...`,
                    details: {
                        query,
                        dataSource,
                        error: errorMessage
                    }
                });
                throw error;
            }
        },
        async getPipelineRecommendation(dataSource, context) {
            if (!config.enabled) {
                throw new Error('Datapizza plugin is disabled');
            }
            try {
                return await service.getPipelineRecommendation(dataSource, context);
            }
            catch (error) {
                toast.error(`Failed to get pipeline recommendation: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async detectDataDomain(data) {
            if (!config.enabled) {
                throw new Error('Datapizza plugin is disabled');
            }
            try {
                return await service.detectDataDomain(data);
            }
            catch (error) {
                toast.error(`Failed to detect data domain: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async isProcessableData(data) {
            if (!config.enabled) {
                throw new Error('Datapizza plugin is disabled');
            }
            try {
                return await service.isProcessableData(data);
            }
            catch (error) {
                toast.error(`Failed to check data processability: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        async clearCache() {
            try {
                await service.clearCache();
                toast.success('Datapizza cache cleared successfully');
            }
            catch (error) {
                toast.error(`Failed to clear cache: ${error instanceof Error ? error.message : 'Unknown error'}`);
                throw error;
            }
        },
        getStatistics() {
            return { ...state.statistics };
        },
        getOperationHistory() {
            return [...state.operationHistory];
        },
        clearOperationHistory() {
            state.operationHistory = [];
            toast.success('Operation history cleared');
        },
        getStatus() {
            return state.status;
        },
        getContext() {
            return {
                config: { ...config },
                state: { ...state },
                availablePipelineTypes: [...DATAPIZZA_PIPELINE_TYPES],
                dataDomains: [...DATAPIZZA_DATA_DOMAINS],
                capabilities: {
                    pipelineProcessing: true,
                    dataQuerying: true,
                    caching: true,
                    monitoring: true,
                    reporting: true,
                    externalIntegration: true
                }
            };
        }
    };
    // Set global instance
    globalPluginInstance = pluginMethods;
    return pluginMethods;
}
// React hook for using the plugin
export function useDatapizzaPlugin() {
    const [plugin] = useState(() => createDatapizzaPlugin());
    useEffect(() => {
        // Initialize plugin on mount
        plugin.initialize();
        return () => {
            // Cleanup if needed
        };
    }, []);
    return plugin;
}
//# sourceMappingURL=createDatapizzaPlugin.js.map