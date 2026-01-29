// LeanAIDE Plugin Factory
// Creates a fully configured LeanAIDE plugin instance

import { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { LeanAIDEPluginConfig, LeanAIDEPluginState, LeanAIDEPlugin, 
         LeanAIDEAutoformalizationResult, LeanAIDEVerificationResult, 
         DEFAULT_LEANAIDE_CONFIG, LEANAIDE_STRATEGIES, MATHEMATICAL_DOMAINS } from '../types/plugin-types';
import { LeanAIDEClient } from '../services/LeanAIDEClient';
import { LeanAIDEService } from '../services/LeanAIDEService';

// Global plugin state management
let globalPluginState: LeanAIDEPluginState = { ...DEFAULT_LEANAIDE_CONFIG, ...{
  status: 'idle',
  operationHistory: [],
  statistics: {
    totalOperations: 0,
    successfulOperations: 0,
    failedOperations: 0,
    averageConfidence: 0
  }
} };

let globalPluginInstance: LeanAIDEPlugin | null = null;

/**
 * Create a new LeanAIDE plugin instance
 * @param initialConfig Optional initial configuration
 * @returns LeanAIDEPlugin instance
 */
export function createLeanAIDEPlugin(initialConfig?: Partial<LeanAIDEPluginConfig>): LeanAIDEPlugin {
  if (globalPluginInstance) {
    // Return existing instance if available
    return globalPluginInstance;
  }

  // Merge initial config with defaults
  const config: LeanAIDEPluginConfig = {
    ...DEFAULT_LEANAIDE_CONFIG,
    ...initialConfig
  };

  // Initialize plugin state
  const state: LeanAIDEPluginState = {
    ...config,
    status: 'initializing',
    operationHistory: [],
    statistics: {
      totalOperations: 0,
      successfulOperations: 0,
      failedOperations: 0,
      averageConfidence: 0
    }
  };

  // Update global state
  globalPluginState = state;

  // Create client and service instances
  const client = new LeanAIDEClient({
    baseUrl: config.serverUrl,
    apiKey: config.apiKey,
    timeout: config.timeout
  });

  const service = new LeanAIDEService(client);

  // Create plugin methods
  const pluginMethods: LeanAIDEPlugin = {
    metadata: {
      name: 'LeanAIDE Autoformalization',
      version: '1.0.0',
      description: 'BubbleLabs plugin for mathematical formalization and verification',
      author: 'OpenEvolve',
      website: 'https://openevolve.com'
    },

    // Plugin methods
    async initialize(configUpdate?: Partial<LeanAIDEPluginConfig>) {
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
          timeout: config.timeout
        });

        // Test connection
        await client.testConnection();

        state.status = 'ready';
        toast.success('LeanAIDE plugin initialized successfully');
        
      } catch (error) {
        state.status = 'error';
        toast.error(`Failed to initialize LeanAIDE plugin: ${error instanceof Error ? error.message : 'Unknown error'}`);
        throw error;
      }
    },

    async updateConfig(configUpdate: Partial<LeanAIDEPluginConfig>) {
      try {
        Object.assign(config, configUpdate);
        Object.assign(state, config);

        // Update client configuration
        client.configure({
          baseUrl: config.serverUrl,
          apiKey: config.apiKey,
          timeout: config.timeout
        });

        toast.success('LeanAIDE configuration updated successfully');
        
      } catch (error) {
        toast.error(`Failed to update configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
        throw error;
      }
    },

    async resetConfig() {
      Object.assign(config, DEFAULT_LEANAIDE_CONFIG);
      Object.assign(state, DEFAULT_LEANAIDE_CONFIG);
      
      client.configure({
        baseUrl: config.serverUrl,
        apiKey: config.apiKey,
        timeout: config.timeout
      });

      toast.success('LeanAIDE configuration reset to defaults');
    },

    async autoformalize(problem: string, strategy?: string): Promise<LeanAIDEAutoformalizationResult> {
      if (!config.enabled) {
        throw new Error('LeanAIDE plugin is disabled');
      }

      if (state.status !== 'ready') {
        throw new Error(`Plugin not ready. Current status: ${state.status}`);
      }

      try {
        state.status = 'busy';
        state.currentOperation = {
          type: 'autoformalization',
          startedAt: new Date(),
          message: `Autoformalizing problem: ${problem.substring(0, 50)}...`
        };

        const startTime = Date.now();
        
        const result = await service.autoformalize(problem, strategy || config.defaultStrategy);
        
        const executionTime = Date.now() - startTime;
        
        // Update statistics
        state.statistics.totalOperations++;
        if (result.success) {
          state.statistics.successfulOperations++;
          state.statistics.averageConfidence = (
            (state.statistics.averageConfidence * (state.statistics.totalOperations - 1)) + 
            result.confidenceScore
          ) / state.statistics.totalOperations;
        } else {
          state.statistics.failedOperations++;
        }
        state.statistics.lastOperationTime = new Date();

        // Add to operation history
        state.operationHistory.unshift({
          id: Date.now().toString(),
          type: 'autoformalization',
          timestamp: new Date(),
          success: result.success,
          message: `Autoformalization ${result.success ? 'succeeded' : 'failed'}: ${problem.substring(0, 50)}...`,
          details: {
            problem,
            strategy: strategy || config.defaultStrategy,
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
          executionTime: executionTime / 1000, // Convert to seconds
          timestamp: new Date()
        };

      } catch (error) {
        state.status = 'error';
        state.currentOperation = undefined;

        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        
        // Add to operation history
        state.operationHistory.unshift({
          id: Date.now().toString(),
          type: 'autoformalization',
          timestamp: new Date(),
          success: false,
          message: `Autoformalization failed: ${errorMessage}`,
          details: { problem, strategy, error: errorMessage }
        });

        throw new Error(`Autoformalization failed: ${errorMessage}`);
      }
    },

    async verify(problem: string, leanCode: string): Promise<LeanAIDEVerificationResult> {
      if (!config.enabled) {
        throw new Error('LeanAIDE plugin is disabled');
      }

      if (state.status !== 'ready') {
        throw new Error(`Plugin not ready. Current status: ${state.status}`);
      }

      try {
        state.status = 'busy';
        state.currentOperation = {
          type: 'verification',
          startedAt: new Date(),
          message: `Verifying solution for: ${problem.substring(0, 50)}...`
        };

        const startTime = Date.now();
        
        const result = await service.verify(problem, leanCode);
        
        const executionTime = Date.now() - startTime;
        
        // Update statistics
        state.statistics.totalOperations++;
        if (result.success) {
          state.statistics.successfulOperations++;
          state.statistics.averageConfidence = (
            (state.statistics.averageConfidence * (state.statistics.totalOperations - 1)) + 
            result.confidenceScore
          ) / state.statistics.totalOperations;
        } else {
          state.statistics.failedOperations++;
        }
        state.statistics.lastOperationTime = new Date();

        // Add to operation history
        state.operationHistory.unshift({
          id: Date.now().toString(),
          type: 'verification',
          timestamp: new Date(),
          success: result.success,
          message: `Verification ${result.success ? 'succeeded' : 'failed'}: ${problem.substring(0, 50)}...`,
          details: {
            problem,
            leanCode,
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
          executionTime: executionTime / 1000, // Convert to seconds
          timestamp: new Date()
        };

      } catch (error) {
        state.status = 'error';
        state.currentOperation = undefined;

        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        
        // Add to operation history
        state.operationHistory.unshift({
          id: Date.now().toString(),
          type: 'verification',
          timestamp: new Date(),
          success: false,
          message: `Verification failed: ${errorMessage}`,
          details: { problem, leanCode, error: errorMessage }
        });

        throw new Error(`Verification failed: ${errorMessage}`);
      }
    },

    async getStrategyRecommendation(problem: string, context: string = 'general'): Promise<string> {
      try {
        // Simple recommendation based on problem content
        const problemLower = problem.toLowerCase();
        
        if (problemLower.includes('induction') || problemLower.includes('recursion')) {
          return 'HYBRID';
        } else if (problemLower.includes('prove') && problemLower.includes('for all')) {
          return 'MAKER';
        } else if (problemLower.includes('show that') || problemLower.includes('demonstrate')) {
          return 'MDAP';
        } else if (problemLower.includes('simple') || problemLower.includes('basic')) {
          return 'DIRECT';
        } else {
          return config.defaultStrategy || 'ADAPTIVE';
        }

      } catch (error) {
        console.error('Strategy recommendation error:', error);
        return config.defaultStrategy || 'ADAPTIVE';
      }
    },

    async detectMathematicalDomain(problem: string): Promise<string | null> {
      try {
        return service.detectMathematicalDomain(problem);
      } catch (error) {
        console.error('Domain detection error:', error);
        return null;
      }
    },

    async isMathematicalProblem(problem: string): Promise<boolean> {
      try {
        return service.isMathematicalProblem(problem);
      } catch (error) {
        console.error('Mathematical detection error:', error);
        return false;
      }
    },

    async clearCache() {
      try {
        await service.clearCache();
        toast.success('LeanAIDE cache cleared successfully');
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
        availableStrategies: LEANAIDE_STRATEGIES,
        mathematicalDomains: MATHEMATICAL_DOMAINS,
        capabilities: {
          autoformalization: true,
          verification: true,
          caching: true,
          monitoring: true,
          reporting: true
        }
      };
    },

    // React components (will be imported dynamically)
    components: {
      ConfigPanel: () => import('../components/LeanAIDEConfigPanel').then(m => m.LeanAIDEConfigPanel),
      AutoformalizationPanel: () => import('../components/LeanAIDEAutoformalizationPanel').then(m => m.LeanAIDEAutoformalizationPanel),
      VerificationPanel: () => import('../components/LeanAIDEVerificationPanel').then(m => m.LeanAIDEVerificationPanel),
      StrategySelector: () => import('../components/LeanAIDEStrategySelector').then(m => m.LeanAIDEStrategySelector),
      StatusIndicator: () => import('../components/LeanAIDEStatusIndicator').then(m => m.LeanAIDEStatusIndicator)
    },

    // React hooks (will be imported dynamically)
    hooks: {
      useLeanAIDEConfig: () => import('../hooks/useLeanAIDEConfig').then(m => m.useLeanAIDEConfig),
      useLeanAIDEState: () => import('../hooks/useLeanAIDEState').then(m => m.useLeanAIDEState),
      useLeanAIDEAutoformalization: () => import('../hooks/useLeanAIDEAutoformalization').then(m => m.useLeanAIDEAutoformalization),
      useLeanAIDEVerification: () => import('../hooks/useLeanAIDEVerification').then(m => m.useLeanAIDEVerification)
    }
  };

  // Set global instance
  globalPluginInstance = pluginMethods;
  
  return pluginMethods;
}

/**
 * Get the global plugin instance
 * @returns LeanAIDEPlugin instance
 */
export function getLeanAIDEPlugin(): LeanAIDEPlugin {
  if (!globalPluginInstance) {
    globalPluginInstance = createLeanAIDEPlugin();
  }
  return globalPluginInstance;
}

/**
 * React hook to use the LeanAIDE plugin
 * @returns LeanAIDEPlugin instance
 */
export function useLeanAIDEPlugin(): LeanAIDEPlugin {
  const [plugin] = useState<LeanAIDEPlugin>(() => getLeanAIDEPlugin());
  
  useEffect(() => {
    // Initialize plugin if not already initialized
    if (plugin.getStatus() === 'idle') {
      plugin.initialize();
    }
  }, [plugin]);
  
  return plugin;
}