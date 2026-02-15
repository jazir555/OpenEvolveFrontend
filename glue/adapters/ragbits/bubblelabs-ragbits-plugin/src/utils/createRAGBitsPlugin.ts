// RAGBits Plugin Factory
// Creates a fully configured RAGBits plugin instance

import { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import {
  RAGBitsPluginConfig,
  RAGBitsPluginState,
  RAGBitsPlugin,
  RAGBitsSearchRequest,
  RAGBitsSearchResponse,
  RAGBitsIngestRequest,
  RAGBitsIngestResponse,
  DEFAULT_RAGBITS_CONFIG,
  RAGBITS_SEARCH_TYPES,
  RAGBITS_DOCUMENT_TYPES
} from '../types/plugin-types';
import { RagbitsClient } from '../lib/ragbitsClient';
import { RagbitsService } from '../services/ragbitsService';

// Global plugin state management
let globalPluginState: RAGBitsPluginState = {
  ...DEFAULT_RAGBITS_CONFIG,
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

let globalPluginInstance: RAGBitsPlugin | null = null;

/**
 * Create a new RAGBits plugin instance
 * @param initialConfig Optional initial configuration
 * @returns RAGBitsPlugin instance
 */
export function createRAGBitsPlugin(initialConfig?: Partial<RAGBitsPluginConfig>): RAGBitsPlugin {
  if (globalPluginInstance) {
    // Return existing instance if available
    return globalPluginInstance;
  }

  // Merge initial config with defaults
  const config: RAGBitsPluginConfig = {
    ...DEFAULT_RAGBITS_CONFIG,
    ...initialConfig
  };

  // Initialize plugin state
  const state: RAGBitsPluginState = {
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
  const client = new RagbitsClient({
    serverUrl: config.serverUrl,
    apiKey: config.apiKey,
    timeout: config.timeout
  });

  const service = new RagbitsService(
    client,
    config.enableCaching ? config.cacheTTLSeconds : 0
  );

  // Create plugin methods
  const pluginMethods: RAGBitsPlugin = {
    metadata: {
      name: 'RAGBits Knowledge Search',
      version: '1.0.0',
      description: 'BubbleLabs plugin for semantic document search and knowledge retrieval',
      author: 'OpenEvolve',
      website: 'https://openevolve.com'
    },

    // Plugin methods
    async initialize(configUpdate?: Partial<RAGBitsPluginConfig>) {
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
        toast.success('RAGBits plugin initialized successfully');

      } catch (error) {
        state.status = 'error';
        toast.error(`Failed to initialize RAGBits plugin: ${error instanceof Error ? error.message : 'Unknown error'}`);
        throw error;
      }
    },

    async updateConfig(configUpdate: Partial<RAGBitsPluginConfig>) {
      try {
        Object.assign(config, configUpdate);
        Object.assign(state, config);

        // Update client configuration
        client.configure({
          serverUrl: config.serverUrl,
          apiKey: config.apiKey,
          timeout: config.timeout
        });

        toast.success('RAGBits configuration updated successfully');

      } catch (error) {
        toast.error(`Failed to update configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
        throw error;
      }
    },

    async resetConfig() {
      Object.assign(config, DEFAULT_RAGBITS_CONFIG);
      Object.assign(state, DEFAULT_RAGBITS_CONFIG);

      client.configure({
        serverUrl: config.serverUrl,
        apiKey: config.apiKey,
        timeout: config.timeout
      });

      toast.success('RAGBits configuration reset to defaults');
    },

    async search(request: RAGBitsSearchRequest): Promise<RAGBitsSearchResponse> {
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
          state.statistics.averageSearchTime = (
            (state.statistics.averageSearchTime * (state.statistics.totalSearches - 1)) +
            executionTime
          ) / state.statistics.totalSearches;

          // Calculate average relevance score
          if (result.results.length > 0) {
            const avgScore = result.results.reduce((sum, r) => sum + r.relevanceScore, 0) / result.results.length;
            state.statistics.averageRelevanceScore = avgScore;
          }
        } else {
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

      } catch (error) {
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

    async ingest(request: RAGBitsIngestRequest): Promise<RAGBitsIngestResponse> {
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

      } catch (error) {
        state.status = 'error';
        state.currentOperation = undefined;

        const errorMessage = error instanceof Error ? error.message : 'Unknown error';

        throw new Error(`Ingest failed: ${errorMessage}`);
      }
    },

    async batchIngest(requests: RAGBitsIngestRequest[]): Promise<RAGBitsIngestResponse[]> {
      const results: RAGBitsIngestResponse[] = [];

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
      } catch (error) {
        throw new Error(`Failed to get index stats: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    },

    async clearCache() {
      try {
        await service.clearCache();
        toast.success('RAGBits cache cleared successfully');
      } catch (error) {
        toast.error(`Failed to clear cache: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
      toast.success('Operation history cleared');
    },

    getStatus() {
      return state.status;
    },

    getContext() {
      return {
        config,
        state,
        searchTypes: RAGBITS_SEARCH_TYPES,
        documentTypes: RAGBITS_DOCUMENT_TYPES,
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
      ConfigPanel: () => import('../components/RAGBitsConfigPanel').then(m => m.RAGBitsConfigPanel),
      SearchPanel: () => import('../components/RAGBitsSearchPanel').then(m => m.RAGBitsSearchPanel),
      IngestPanel: () => import('../components/RAGBitsIngestPanel').then(m => m.RAGBitsIngestPanel),
      StatusIndicator: () => import('../components/RAGBitsStatusIndicator').then(m => m.RAGBitsStatusIndicator),
      SearchResults: () => import('../components/RAGBitsSearchResults').then(m => m.RAGBitsSearchResults)
    },

    // React hooks (will be imported dynamically)
    hooks: {
      useRAGBitsConfig: () => import('../hooks/useRAGBitsConfig').then(m => m.useRAGBitsConfig),
      useRAGBitsState: () => import('../hooks/useRAGBitsState').then(m => m.useRAGBitsState),
      useRAGBitsSearch: () => import('../hooks/useRAGBitsSearch').then(m => m.useRAGBitsSearch),
      useRAGBitsIngest: () => import('../hooks/useRAGBitsIngest').then(m => m.useRAGBitsIngest)
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
export function getRAGBitsPlugin(): RAGBitsPlugin {
  if (!globalPluginInstance) {
    globalPluginInstance = createRAGBitsPlugin();
  }
  return globalPluginInstance;
}

/**
 * React hook to use the RAGBits plugin
 * @returns RAGBitsPlugin instance
 */
export function useRAGBitsPlugin(): RAGBitsPlugin {
  const [plugin] = useState<RAGBitsPlugin>(() => getRAGBitsPlugin());

  useEffect(() => {
    // Initialize plugin if not already initialized
    if (plugin.getStatus() === 'idle') {
      plugin.initialize();
    }
  }, [plugin]);

  return plugin;
}
