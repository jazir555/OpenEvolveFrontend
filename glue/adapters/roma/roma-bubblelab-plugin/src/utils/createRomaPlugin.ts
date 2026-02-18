import {
  DEFAULT_ROMA_CONFIG,
  ROMA_PLUGIN_CONSTANTS,
  type RomaExecutionOptions,
  type RomaExecutionResult,
  type RomaExecutionStatistics,
  type RomaMcpServerConfig,
  type RomaPlugin,
  type RomaPluginConfig,
  type RomaPluginMetadata,
  type RomaPluginState,
  type RomaToolkitConfig,
} from '../types/plugin-types';

const metadata: RomaPluginMetadata = {
  name: 'ROMA BubbleLab Plugin',
  version: '1.0.0',
  description: 'ROMA orchestration integration for BubbleLab',
  author: 'OpenEvolve',
  license: 'MIT',
};

function createInitialState(config?: Partial<RomaPluginConfig>): RomaPluginState {
  return {
    ...DEFAULT_ROMA_CONFIG,
    ...config,
    status: 'idle',
    executionHistory: [],
    statistics: {
      totalExecutions: 0,
      successfulExecutions: 0,
      failedExecutions: 0,
      averageExecutionTime: 0,
      totalExecutionTime: 0,
    },
    isInitialized: false,
  };
}

function cloneState(state: RomaPluginState): RomaPluginState {
  return {
    ...state,
    executionHistory: [...state.executionHistory],
    statistics: { ...state.statistics },
  };
}

export function createRomaPlugin(initialConfig?: Partial<RomaPluginConfig>): RomaPlugin {
  let state = createInitialState(initialConfig);

  const refreshStatistics = (execution: RomaExecutionResult) => {
    const stats = state.statistics;
    const executionTime = execution.statistics?.executionTime ?? 0;

    const totalExecutions = stats.totalExecutions + 1;
    const successfulExecutions = execution.status === 'completed'
      ? stats.successfulExecutions + 1
      : stats.successfulExecutions;
    const failedExecutions = execution.status === 'failed'
      ? stats.failedExecutions + 1
      : stats.failedExecutions;
    const totalExecutionTime = stats.totalExecutionTime + executionTime;

    state.statistics = {
      totalExecutions,
      successfulExecutions,
      failedExecutions,
      totalExecutionTime,
      averageExecutionTime: totalExecutions === 0 ? 0 : totalExecutionTime / totalExecutions,
      lastExecutionTime: executionTime,
      lastExecutionStatus: execution.status,
    };
  };

  return {
    metadata,

    async initialize(config?: Partial<RomaPluginConfig>) {
      state = createInitialState({ ...state, ...config });
      state.status = 'idle';
      state.isInitialized = true;
      state.initializationError = undefined;
    },

    async updateConfig(configUpdate: Partial<RomaPluginConfig>) {
      state = { ...state, ...configUpdate };
    },

    async executeTask(goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult> {
      const start = Date.now();
      state.status = 'executing';

      const execution: RomaExecutionResult = {
        executionId: `roma-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        goal,
        status: 'completed',
        result: {
          summary: `Simulated ROMA execution for goal: ${goal}`,
          profile: options?.profile ?? state.defaultProfile ?? 'general',
          method: options?.executionMethod ?? state.defaultExecutionMethod ?? 'auto',
        },
        statistics: {
          executionTime: Date.now() - start,
          subtasksCreated: 1,
          subtasksCompleted: 1,
          toolsUsed: [],
          modulesUsed: ['planner', 'executor'],
        },
        timestamp: Date.now(),
      };

      state.currentExecution = execution;
      state.executionHistory = [execution, ...state.executionHistory].slice(0, 100);
      refreshStatistics(execution);
      state.status = 'completed';

      return execution;
    },

    getState() {
      return cloneState(state);
    },

    getExecutionHistory(limit?: number) {
      return typeof limit === 'number' ? state.executionHistory.slice(0, limit) : [...state.executionHistory];
    },

    getExecution(executionId: string) {
      return state.executionHistory.find((item) => item.executionId === executionId);
    },

    async cancelExecution() {
      if (state.currentExecution) {
        state.currentExecution = { ...state.currentExecution, status: 'cancelled' };
      }
      state.status = 'cancelled';
    },

    async clearHistory() {
      state.executionHistory = [];
    },

    async reset() {
      state = createInitialState();
    },

    getAvailableMcps() {
      return [...(state.mcpServers ?? [])];
    },

    async addMcpServer(mcpConfig: RomaMcpServerConfig) {
      state.mcpServers = [...(state.mcpServers ?? []), mcpConfig];
    },

    async removeMcpServer(serverName: string) {
      state.mcpServers = (state.mcpServers ?? []).filter((server) => server.server_name !== serverName);
    },

    getAvailableToolkits() {
      return [...(state.agents?.executor?.toolkits ?? [])];
    },

    async addToolkit(toolkitConfig: RomaToolkitConfig) {
      const current = state.agents?.executor?.toolkits ?? [];
      state.agents = {
        ...state.agents,
        executor: {
          ...state.agents?.executor,
          toolkits: [...current, toolkitConfig],
        },
      };
    },

    async removeToolkit(toolkitName: string) {
      const current = state.agents?.executor?.toolkits ?? [];
      state.agents = {
        ...state.agents,
        executor: {
          ...state.agents?.executor,
          toolkits: current.filter((toolkit) => toolkit.class_name !== toolkitName),
        },
      };
    },

    getStatistics(): RomaExecutionStatistics {
      return { ...state.statistics };
    },

    exportState() {
      return cloneState(state);
    },

    async importState(importedState: RomaPluginState) {
      state = cloneState(importedState);
    },

    isReady() {
      return state.isInitialized && state.status !== 'failed';
    },

    getVersion() {
      return metadata.version;
    },

    getMetadata() {
      return metadata;
    },
  };
}

export const romaPlugin = createRomaPlugin({
  defaultExecutionMethod: ROMA_PLUGIN_CONSTANTS.DEFAULT_EXECUTION_METHOD,
});

export default createRomaPlugin;
