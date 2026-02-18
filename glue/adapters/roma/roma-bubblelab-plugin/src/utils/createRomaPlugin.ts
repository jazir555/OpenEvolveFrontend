/**
 * ROMA BubbleLabs Plugin Factory
 * 
 * This file implements the ROMA plugin factory with singleton pattern and global state management.
 * It follows the same pattern as other BubbleLabs plugins (LeanAIDE, ClaudieMiro, Datapizza).
 */

import { createStore } from 'zustand/vanilla';
import { RomaClient } from '../services/RomaClient';
import { RomaService } from '../services/RomaService';
import {
  RomaPlugin,
  RomaPluginConfig,
  RomaPluginState,
  RomaExecutionResult,
  RomaExecutionOptions,
  RomaMcpServerConfig,
  RomaToolkitConfig,
  RomaMdapMakerConfig,
  RomaExecutionStatistics,
  RomaPluginMetadata,
  RomaPluginError,
  DEFAULT_ROMA_CONFIG,
  ROMA_PLUGIN_CONSTANTS
} from '../types/plugin-types';

// Global plugin instance for singleton pattern
let globalPluginInstance: RomaPlugin | null = null;

/**
 * ROMA Plugin Store State
 */
interface RomaPluginStoreState extends RomaPluginState {
  setStatus: (status: RomaPluginState['status']) => void;
  addExecution: (execution: RomaExecutionResult) => void;
  updateExecution: (executionId: string, updates: Partial<RomaExecutionResult>) => void;
  clearHistory: () => void;
  updateStatistics: (updates: Partial<RomaExecutionStatistics>) => void;
  resetStatistics: () => void;
  updateConfig: (configUpdate: Partial<RomaPluginConfig>) => void;
  setCurrentExecution: (execution?: RomaExecutionResult) => void;
  setInitializationError: (error?: string) => void;
}

/**
 * Create ROMA Plugin Store
 */
const createRomaPluginStore = (initialConfig: RomaPluginConfig) => {
  return createStore<RomaPluginStoreState>((set) => ({
    ...initialConfig,
    status: 'initializing',
    executionHistory: [],
    statistics: {
      totalExecutions: 0,
      successfulExecutions: 0,
      failedExecutions: 0,
      averageExecutionTime: 0,
      totalExecutionTime: 0
    },
    isInitialized: false,
    
    // State update methods
    setStatus: (status) => set({ status }),
    addExecution: (execution) => set((state) => ({
      executionHistory: [execution, ...state.executionHistory.slice(0, 99)]
    })),
    updateExecution: (executionId, updates) => set((state) => ({
      executionHistory: state.executionHistory.map((exec) =>
        exec.executionId === executionId ? { ...exec, ...updates } : exec
      )
    })),
    clearHistory: () => set({ executionHistory: [] }),
    updateStatistics: (updates) => set((state) => ({
      statistics: { ...state.statistics, ...updates }
    })),
    resetStatistics: () => set({
      statistics: {
        totalExecutions: 0,
        successfulExecutions: 0,
        failedExecutions: 0,
        averageExecutionTime: 0,
        totalExecutionTime: 0
      }
    }),
    updateConfig: (configUpdate) => set((state) => ({
      ...state,
      ...configUpdate
    })),
    setCurrentExecution: (execution) => set({ currentExecution: execution }),
    setInitializationError: (error) => set({ initializationError: error })
  }));
};

/**
 * ROMA Plugin Metadata
 */
const ROMA_PLUGIN_METADATA: RomaPluginMetadata = {
  name: 'ROMA Recursive Open Meta-Agents',
  version: '1.0.0',
  description: 'ROMA (Recursive Open Meta-Agents) integration plugin for BubbleLabs',
  author: 'OpenEvolve Team',
  license: 'MIT',
  repository: 'https://github.com/sentient-agi/roma',
  documentation: 'https://github.com/sentient-agi/roma/blob/main/README.md'
};

/**
 * Create ROMA Plugin
 * 
 * This function creates a ROMA plugin instance with singleton pattern and global state management.
 * It initializes the client, service, and all plugin methods.
 */
export function createRomaPlugin(initialConfig?: Partial<RomaPluginConfig>): RomaPlugin {
  // Singleton instance management
  if (globalPluginInstance) {
    return globalPluginInstance;
  }

  // Configuration merging with defaults
  const config: RomaPluginConfig = { 
    ...DEFAULT_ROMA_CONFIG, 
    ...initialConfig 
  };

  // State initialization with statistics and operation history
  const store = createRomaPluginStore(config);
  const getState = () => store.getState();

  // Client and service instantiation
  const client = new RomaClient({
    baseUrl: config.serverUrl || ROMA_PLUGIN_CONSTANTS.DEFAULT_SERVER_URL,
    apiKey: config.apiKey,
    timeout: config.timeout
  });

  const service = new RomaService(client);

  /**
   * Generate unique execution ID
   */
  const generateExecutionId = (): string => {
    return `roma-exec-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`;
  };

  /**
   * Determine if MDAP/MAKER should be used for a goal (auto-selection)
   */
  const shouldUseMdapMakerForGoal = (goal: string, mdapMakerConfig?: RomaMdapMakerConfig): boolean => {
    if (!mdapMakerConfig?.enabled || !mdapMakerConfig.autoSelect) {
      return false;
    }

    // Check if goal contains any auto-selection keywords
    const keywords = mdapMakerConfig.autoSelectionKeywords || [];
    const goalLower = goal.toLowerCase();
    
    return keywords.some(keyword => 
      goalLower.includes(keyword.toLowerCase())
    );
  };

  /**
   * Update execution statistics
   */
  const updateExecutionStatistics = (execution: RomaExecutionResult) => {
    const executionTime = execution.statistics?.executionTime || 0;
    const currentState = getState();
    
    store.getState().updateStatistics({
      totalExecutions: currentState.statistics.totalExecutions + 1,
      totalExecutionTime: currentState.statistics.totalExecutionTime + executionTime,
      averageExecutionTime: (
        (currentState.statistics.totalExecutionTime + executionTime) / 
        (currentState.statistics.totalExecutions + 1)
      )
    });

    if (execution.status === 'completed') {
      store.getState().updateStatistics({
        successfulExecutions: currentState.statistics.successfulExecutions + 1
      });
    } else if (execution.status === 'failed') {
      store.getState().updateStatistics({
        failedExecutions: currentState.statistics.failedExecutions + 1
      });
    }
  };

  /**
   * Create execution result
   */
  const createExecutionResult = (
    executionId: string,
    goal: string,
    status: RomaExecutionResult['status'],
    result?: any,
    error?: string
  ): RomaExecutionResult => {
    return {
      executionId,
      goal,
      status,
      result,
      error,
      statistics: {
        executionTime: 0,
        subtasksCreated: 0,
        subtasksCompleted: 0,
        toolsUsed: [],
        modulesUsed: []
      },
      timestamp: Date.now()
    };
  };

  // Complete plugin methods implementation
  const pluginMethods: RomaPlugin = {
    metadata: ROMA_PLUGIN_METADATA,

    /**
     * Initialize the ROMA plugin
     */
    initialize: async (configUpdate?: Partial<RomaPluginConfig>) => {
      try {
        store.getState().setStatus('initializing');
        store.getState().setInitializationError(undefined);

        // Merge configuration updates
        if (configUpdate) {
          store.getState().updateConfig(configUpdate);
        }

        // Initialize client and service
        await client.initialize();
        await service.initialize();

        // Update state
        store.getState().setStatus('idle');
        store.getState().updateConfig({ isInitialized: true });

        console.log('ROMA plugin initialized successfully');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error';
        store.getState().setInitializationError(errorMessage);
        store.getState().setStatus('failed');
        console.error('ROMA plugin initialization failed:', error);
        throw new RomaPluginError(errorMessage, 'INITIALIZATION_FAILED', error);
      }
    },

    /**
     * Update plugin configuration
     */
    updateConfig: async (configUpdate: Partial<RomaPluginConfig>) => {
      try {
        store.getState().updateConfig(configUpdate);
        
        // Update client configuration if needed
        if (configUpdate.serverUrl || configUpdate.apiKey || configUpdate.timeout) {
          client.updateConfig({
            baseUrl: configUpdate.serverUrl || client.config.baseUrl,
            apiKey: configUpdate.apiKey || client.config.apiKey,
            timeout: configUpdate.timeout || client.config.timeout
          });
        }

        console.log('ROMA plugin configuration updated');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown configuration error';
        console.error('ROMA plugin configuration update failed:', error);
        throw new RomaPluginError(errorMessage, 'CONFIGURATION_FAILED', error);
      }
    },

    /**
     * Execute a task using ROMA
     */
    executeTask: async (goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult> => {
      try {
        const currentState = getState();

        if (!currentState.isInitialized) {
          throw new RomaPluginError('Plugin not initialized', 'PLUGIN_NOT_INITIALIZED');
        }

        if (currentState.status === 'executing') {
          throw new RomaPluginError('Execution already in progress', 'EXECUTION_IN_PROGRESS');
        }

        const executionId = generateExecutionId();
        const startTime = Date.now();

        // Create initial execution result
        const initialExecution = createExecutionResult(executionId, goal, 'executing');
        store.getState().setStatus('executing');
        store.getState().setCurrentExecution(initialExecution);
        store.getState().addExecution(initialExecution);

        console.log(`Starting ROMA execution: ${executionId} - ${goal}`);

        // Determine execution method
        const executionMethod = options?.executionMethod || currentState.defaultExecutionMethod || 'auto';
        
        // Check if MDAP/MAKER should be used (auto-selection or explicit)
        const shouldUseMdapMaker = executionMethod === 'roma_mdap_maker' || 
          (executionMethod === 'auto' && currentState.mdapMaker?.autoSelect &&
           shouldUseMdapMakerForGoal(goal, currentState.mdapMaker));

        // Execute task with service (includes retry and cache logic)
        const result = await service.executeTaskWithRetry(goal, {
          maxDepth: options?.maxDepth || currentState.maxDepth,
          timeout: options?.timeout || currentState.timeout,
          profile: options?.profile || currentState.defaultProfile,
          useCache: options?.useCache !== false, // Default to true
          debug: options?.debug || currentState.debugMode,
          executionMethod: shouldUseMdapMaker ? 'roma_mdap_maker' : executionMethod,
          mdapMakerConfig: shouldUseMdapMaker ? {
            ...currentState.mdapMaker,
            ...options?.mdapMakerConfig
          } : undefined
        });

        const executionTime = Date.now() - startTime;

        // Update execution with results
        const finalExecution: RomaExecutionResult = {
          ...result,
          executionId,
          goal,
          status: result.status || 'completed',
          statistics: {
            ...result.statistics,
            executionTime
          },
          timestamp: Date.now()
        };

        store.getState().updateExecution(executionId, finalExecution);
        store.getState().setCurrentExecution(finalExecution);
        store.getState().setStatus('completed');

        // Update statistics
        updateExecutionStatistics(finalExecution);

        console.log(`ROMA execution completed: ${executionId} - Status: ${finalExecution.status}`);

        return finalExecution;
      } catch (error) {
        const executionId = generateExecutionId();
        const errorMessage = error instanceof Error ? error.message : 'Unknown execution error';
        
        const failedExecution = createExecutionResult(
          executionId, 
          goal, 
          'failed',
          undefined,
          errorMessage
        );

        store.getState().addExecution(failedExecution);
        store.getState().setCurrentExecution(failedExecution);
        store.getState().setStatus('failed');

        // Update statistics
        updateExecutionStatistics(failedExecution);

        console.error(`ROMA execution failed: ${executionId} - ${errorMessage}`);

        if (error instanceof RomaPluginError) {
          throw error;
        } else {
          throw new RomaPluginError(errorMessage, 'EXECUTION_FAILED', error);
        }
      }
    },

    /**
     * Get current plugin state
     */
    getState: (): RomaPluginState => {
      return store.getState();
    },

    /**
     * Get execution history
     */
    getExecutionHistory: (limit?: number): RomaExecutionResult[] => {
      const currentState = getState();
      return limit ? currentState.executionHistory.slice(0, limit) : currentState.executionHistory;
    },

    /**
     * Get execution by ID
     */
    getExecution: (executionId: string): RomaExecutionResult | undefined => {
      return getState().executionHistory.find(exec => exec.executionId === executionId);
    },

    /**
     * Cancel current execution
     */
    cancelExecution: async (): Promise<void> => {
      try {
        const currentState = getState();
        if (currentState.status !== 'executing' || !currentState.currentExecution) {
          throw new RomaPluginError('No active execution to cancel', 'NO_ACTIVE_EXECUTION');
        }

        await client.cancelExecution(currentState.currentExecution.executionId);
        
        store.getState().updateExecution(currentState.currentExecution.executionId, {
          status: 'cancelled'
        });
        
        store.getState().setStatus('cancelled');
        store.getState().setCurrentExecution(undefined);

        console.log('ROMA execution cancelled');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown cancellation error';
        console.error('ROMA execution cancellation failed:', error);
        throw new RomaPluginError(errorMessage, 'CANCELLATION_FAILED', error);
      }
    },

    /**
     * Clear execution history
     */
    clearHistory: async (): Promise<void> => {
      store.getState().clearHistory();
      store.getState().resetStatistics();
      console.log('ROMA execution history cleared');
    },

    /**
     * Reset plugin state
     */
    reset: async (): Promise<void> => {
      store.getState().clearHistory();
      store.getState().resetStatistics();
      store.getState().setStatus('idle');
      store.getState().setCurrentExecution(undefined);
      store.getState().setInitializationError(undefined);
      console.log('ROMA plugin state reset');
    },

    /**
     * Get available MCP servers
     */
    getAvailableMcps: (): RomaMcpServerConfig[] => {
      return getState().mcpServers || [];
    },

    /**
     * Add MCP server configuration
     */
    addMcpServer: async (mcpConfig: RomaMcpServerConfig): Promise<void> => {
      try {
        const existingServers = getState().mcpServers || [];
        const existingServerIndex = existingServers.findIndex(s => s.server_name === mcpConfig.server_name);

        if (existingServerIndex >= 0) {
          // Update existing server
          const updatedServers = [...existingServers];
          updatedServers[existingServerIndex] = { ...updatedServers[existingServerIndex], ...mcpConfig };
          store.getState().updateConfig({ mcpServers: updatedServers });
        } else {
          // Add new server
          store.getState().updateConfig({ 
            mcpServers: [...existingServers, mcpConfig] 
          });
        }

        console.log(`MCP server added/updated: ${mcpConfig.server_name}`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown MCP server error';
        console.error('Failed to add MCP server:', error);
        throw new RomaPluginError(errorMessage, 'MCP_SERVER_ERROR', error);
      }
    },

    /**
     * Remove MCP server
     */
    removeMcpServer: async (serverName: string): Promise<void> => {
      try {
        const existingServers = getState().mcpServers || [];
        const updatedServers = existingServers.filter(s => s.server_name !== serverName);
        
        store.getState().updateConfig({ mcpServers: updatedServers });
        console.log(`MCP server removed: ${serverName}`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown MCP server error';
        console.error('Failed to remove MCP server:', error);
        throw new RomaPluginError(errorMessage, 'MCP_SERVER_ERROR', error);
      }
    },

    /**
     * Get available toolkits
     */
    getAvailableToolkits: (): RomaToolkitConfig[] => {
      // Get toolkits from all agents
      const allToolkits: RomaToolkitConfig[] = [];
      const currentState = getState();
      
      Object.values(currentState.agents || {}).forEach(agent => {
        if (agent?.toolkits) {
          allToolkits.push(...agent.toolkits);
        }
      });

      return allToolkits;
    },

    /**
     * Add toolkit configuration
     */
    addToolkit: async (toolkitConfig: RomaToolkitConfig): Promise<void> => {
      try {
        const currentState = getState();
        // Add toolkit to executor agent (primary agent for tool usage)
        const currentAgent = currentState.agents?.executor || {};
        const currentToolkits = currentAgent.toolkits || [];
        
        const existingToolkitIndex = currentToolkits.findIndex(t => t.class_name === toolkitConfig.class_name);

        if (existingToolkitIndex >= 0) {
          // Update existing toolkit
          const updatedToolkits = [...currentToolkits];
          updatedToolkits[existingToolkitIndex] = { ...updatedToolkits[existingToolkitIndex], ...toolkitConfig };
          store.getState().updateConfig({ 
            agents: {
              ...currentState.agents,
              executor: {
                ...currentAgent,
                toolkits: updatedToolkits
              }
            }
          });
        } else {
          // Add new toolkit
          store.getState().updateConfig({ 
            agents: {
              ...currentState.agents,
              executor: {
                ...currentAgent,
                toolkits: [...currentToolkits, toolkitConfig]
              }
            }
          });
        }

        console.log(`Toolkit added/updated: ${toolkitConfig.class_name}`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown toolkit error';
        console.error('Failed to add toolkit:', error);
        throw new RomaPluginError(errorMessage, 'TOOLKIT_ERROR', error);
      }
    },

    /**
     * Remove toolkit
     */
    removeToolkit: async (toolkitName: string): Promise<void> => {
      try {
        const currentState = getState();
        // Remove toolkit from executor agent
        const currentAgent = currentState.agents?.executor || {};
        const currentToolkits = currentAgent.toolkits || [];
        const updatedToolkits = currentToolkits.filter(t => t.class_name !== toolkitName);
        
        store.getState().updateConfig({ 
          agents: {
            ...currentState.agents,
            executor: {
              ...currentAgent,
              toolkits: updatedToolkits
            }
          }
        });

        console.log(`Toolkit removed: ${toolkitName}`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown toolkit error';
        console.error('Failed to remove toolkit:', error);
        throw new RomaPluginError(errorMessage, 'TOOLKIT_ERROR', error);
      }
    },

    /**
     * Get plugin statistics
     */
    getStatistics: (): RomaExecutionStatistics => {
      return getState().statistics;
    },

    /**
     * Export plugin state
     */
    exportState: (): RomaPluginState => {
      return store.getState();
    },

    /**
     * Import plugin state
     */
    importState: async (importedState: RomaPluginState): Promise<void> => {
      try {
        // Merge imported state with current state
        store.getState().updateConfig(importedState);
        
        if (importedState.executionHistory) {
          importedState.executionHistory.forEach(exec => {
            store.getState().addExecution(exec);
          });
        }

        if (importedState.statistics) {
          store.getState().updateStatistics(importedState.statistics);
        }

        console.log('ROMA plugin state imported');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown import error';
        console.error('ROMA plugin state import failed:', error);
        throw new RomaPluginError(errorMessage, 'STATE_IMPORT_FAILED', error);
      }
    },

    /**
     * Check if plugin is ready
     */
    isReady: (): boolean => {
      const currentState = getState();
      return currentState.isInitialized &&
        currentState.status !== 'failed' &&
        currentState.status !== 'initializing';
    },

    /**
     * Get plugin version
     */
    getVersion: (): string => {
      return ROMA_PLUGIN_METADATA.version;
    },

    /**
     * Get plugin metadata
     */
    getMetadata: (): RomaPluginMetadata => {
      return ROMA_PLUGIN_METADATA;
    }
  };

  // Set global instance for singleton pattern
  globalPluginInstance = pluginMethods;

  return pluginMethods;
}

/**
 * Reset global plugin instance (for testing)
 */
export function resetRomaPluginInstance(): void {
  globalPluginInstance = null;
}

/**
 * Get global plugin instance
 */
export function getRomaPluginInstance(): RomaPlugin | null {
  return globalPluginInstance;
}
