"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.romaPlugin = void 0;
exports.createRomaPlugin = createRomaPlugin;
const plugin_types_1 = require("../types/plugin-types");
const metadata = {
    name: 'ROMA BubbleLab Plugin',
    version: '1.0.0',
    description: 'ROMA orchestration integration for BubbleLab',
    author: 'OpenEvolve',
    license: 'MIT',
};
function createInitialState(config) {
    return {
        ...plugin_types_1.DEFAULT_ROMA_CONFIG,
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
function cloneState(state) {
    return {
        ...state,
        executionHistory: [...state.executionHistory],
        statistics: { ...state.statistics },
    };
}
function createRomaPlugin(initialConfig) {
    let state = createInitialState(initialConfig);
    const refreshStatistics = (execution) => {
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
        async initialize(config) {
            state = createInitialState({ ...state, ...config });
            state.status = 'idle';
            state.isInitialized = true;
            state.initializationError = undefined;
        },
        async updateConfig(configUpdate) {
            state = { ...state, ...configUpdate };
        },
        async executeTask(goal, options) {
            const start = Date.now();
            state.status = 'executing';
            const execution = {
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
        getExecutionHistory(limit) {
            return typeof limit === 'number' ? state.executionHistory.slice(0, limit) : [...state.executionHistory];
        },
        getExecution(executionId) {
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
        async addMcpServer(mcpConfig) {
            state.mcpServers = [...(state.mcpServers ?? []), mcpConfig];
        },
        async removeMcpServer(serverName) {
            state.mcpServers = (state.mcpServers ?? []).filter((server) => server.server_name !== serverName);
        },
        getAvailableToolkits() {
            return [...(state.agents?.executor?.toolkits ?? [])];
        },
        async addToolkit(toolkitConfig) {
            const current = state.agents?.executor?.toolkits ?? [];
            state.agents = {
                ...state.agents,
                executor: {
                    ...state.agents?.executor,
                    toolkits: [...current, toolkitConfig],
                },
            };
        },
        async removeToolkit(toolkitName) {
            const current = state.agents?.executor?.toolkits ?? [];
            state.agents = {
                ...state.agents,
                executor: {
                    ...state.agents?.executor,
                    toolkits: current.filter((toolkit) => toolkit.class_name !== toolkitName),
                },
            };
        },
        getStatistics() {
            return { ...state.statistics };
        },
        exportState() {
            return cloneState(state);
        },
        async importState(importedState) {
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
exports.romaPlugin = createRomaPlugin({
    defaultExecutionMethod: plugin_types_1.ROMA_PLUGIN_CONSTANTS.DEFAULT_EXECUTION_METHOD,
});
exports.default = createRomaPlugin;
//# sourceMappingURL=createRomaPlugin.js.map