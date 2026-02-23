"use strict";
/**
 * ROMA Client Service
 *
 * This service handles all HTTP communication with the ROMA backend API.
 * It provides methods for executing tasks, managing executions, and configuring MCP servers and toolkits.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RomaClient = void 0;
const axios_1 = __importDefault(require("axios"));
const plugin_types_1 = require("../types/plugin-types");
/**
 * ROMA Client Implementation
 */
class RomaClient {
    /**
     * Create ROMA Client
     * @param config Client configuration
     */
    constructor(config) {
        this.isInitialized = false;
        this.config = {
            baseUrl: config.baseUrl || import.meta.env.VITE_ROMA_SERVER_URL || 'http://localhost:8000',
            apiKey: config.apiKey,
            timeout: config.timeout || 30000,
            headers: config.headers || {}
        };
        // Create Axios instance
        this.axiosInstance = axios_1.default.create({
            baseURL: this.config.baseUrl,
            timeout: this.config.timeout,
            headers: {
                'Content-Type': 'application/json',
                ...this.config.headers
            }
        });
        // Add request interceptor for API key
        this.axiosInstance.interceptors.request.use((config) => {
            if (this.config.apiKey) {
                config.headers = config.headers || {};
                config.headers['Authorization'] = `Bearer ${this.config.apiKey}`;
            }
            return config;
        });
        // Add response interceptor for error handling
        this.axiosInstance.interceptors.response.use((response) => response, (error) => {
            return this.handleAxiosError(error);
        });
    }
    /**
     * Initialize the client
     */
    async initialize() {
        try {
            // Test connection to ROMA server
            const healthResponse = await this.getStatus();
            if (healthResponse.status === 'healthy') {
                this.isInitialized = true;
                console.log('ROMA client initialized successfully');
            }
            else {
                throw new plugin_types_1.RomaPluginError('ROMA server not healthy', 'SERVER_NOT_HEALTHY', healthResponse);
            }
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to initialize ROMA client';
            console.error('ROMA client initialization failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'CLIENT_INITIALIZATION_FAILED', error);
        }
    }
    /**
     * Update client configuration
     * @param configUpdate Configuration updates
     */
    updateConfig(configUpdate) {
        this.config = { ...this.config, ...configUpdate };
        // Update axios instance if base URL or timeout changed
        if (configUpdate.baseUrl || configUpdate.timeout) {
            this.axiosInstance.defaults.baseURL = this.config.baseUrl;
            this.axiosInstance.defaults.timeout = this.config.timeout;
        }
        // Update headers if provided
        if (configUpdate.headers) {
            this.axiosInstance.defaults.headers = {
                ...this.axiosInstance.defaults.headers,
                ...configUpdate.headers
            };
        }
    }
    /**
     * Execute a task using ROMA
     * @param goal The task goal
     * @param options Execution options
     */
    async executeTask(goal, options) {
        try {
            const requestData = {
                goal,
                max_depth: options?.maxDepth,
                timeout: options?.timeout,
                profile: options?.profile,
                use_cache: options?.useCache,
                debug: options?.debug
            };
            // Add execution method if specified
            if (options?.executionMethod) {
                requestData.execution_method = options.executionMethod;
            }
            // Add MDAP/MAKER configuration if provided
            if (options?.mdapMakerConfig) {
                requestData.mdap_maker_config = {
                    max_depth: options.mdapMakerConfig.maxDepth,
                    k_ahead: options.mdapMakerConfig.kAhead,
                    enable_red_flagging: options.mdapMakerConfig.enableRedFlagging,
                    enable_adaptive_k: options.mdapMakerConfig.enableAdaptiveK,
                    provider: options.mdapMakerConfig.provider,
                    model: options.mdapMakerConfig.model
                };
            }
            const response = await this.axiosInstance.post('/api/v1/executions', requestData);
            return this.mapExecutionResponse(response.data);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to execute task';
            console.error('ROMA task execution failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'TASK_EXECUTION_FAILED', error);
        }
    }
    /**
     * Get execution by ID
     * @param executionId Execution ID
     */
    async getExecution(executionId) {
        try {
            const response = await this.axiosInstance.get(`/api/v1/executions/${executionId}`);
            return this.mapExecutionResponse(response.data);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get execution';
            console.error('ROMA get execution failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_EXECUTION_FAILED', error);
        }
    }
    /**
     * Get execution history
     * @param limit Maximum number of results
     */
    async getExecutionHistory(limit) {
        try {
            const params = {};
            if (limit) {
                params.limit = limit;
            }
            const response = await this.axiosInstance.get('/api/v1/executions', { params });
            return response.data.map(exec => this.mapExecutionResponse(exec));
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get execution history';
            console.error('ROMA get execution history failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_EXECUTION_HISTORY_FAILED', error);
        }
    }
    /**
     * Cancel execution
     * @param executionId Execution ID
     */
    async cancelExecution(executionId) {
        try {
            await this.axiosInstance.post(`/api/v1/executions/${executionId}/cancel`);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to cancel execution';
            console.error('ROMA cancel execution failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'CANCEL_EXECUTION_FAILED', error);
        }
    }
    /**
     * Get server status
     */
    async getStatus() {
        try {
            const response = await this.axiosInstance.get('/health');
            return { status: response.data.status };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get server status';
            console.error('ROMA get status failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_STATUS_FAILED', error);
        }
    }
    /**
     * Get execution statistics
     */
    async getStatistics() {
        try {
            const response = await this.axiosInstance.get('/api/v1/statistics');
            return response.data;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get statistics';
            console.error('ROMA get statistics failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_STATISTICS_FAILED', error);
        }
    }
    /**
     * Get available MCP servers
     */
    async getAvailableMcps() {
        try {
            const response = await this.axiosInstance.get('/api/v1/mcp-servers');
            return response.data;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get MCP servers';
            console.error('ROMA get MCP servers failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_MCP_SERVERS_FAILED', error);
        }
    }
    /**
     * Add MCP server configuration
     * @param mcpConfig MCP server configuration
     */
    async addMcpServer(mcpConfig) {
        try {
            await this.axiosInstance.post('/api/v1/mcp-servers', mcpConfig);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to add MCP server';
            console.error('ROMA add MCP server failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'ADD_MCP_SERVER_FAILED', error);
        }
    }
    /**
     * Remove MCP server
     * @param serverName Server name
     */
    async removeMcpServer(serverName) {
        try {
            await this.axiosInstance.delete(`/api/v1/mcp-servers/${serverName}`);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to remove MCP server';
            console.error('ROMA remove MCP server failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'REMOVE_MCP_SERVER_FAILED', error);
        }
    }
    /**
     * Get available toolkits
     */
    async getAvailableToolkits() {
        try {
            const response = await this.axiosInstance.get('/api/v1/toolkits');
            return response.data;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get toolkits';
            console.error('ROMA get toolkits failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_TOOLKITS_FAILED', error);
        }
    }
    /**
     * Add toolkit configuration
     * @param toolkitConfig Toolkit configuration
     */
    async addToolkit(toolkitConfig) {
        try {
            await this.axiosInstance.post('/api/v1/toolkits', toolkitConfig);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to add toolkit';
            console.error('ROMA add toolkit failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'ADD_TOOLKIT_FAILED', error);
        }
    }
    /**
     * Remove toolkit
     * @param toolkitName Toolkit name
     */
    async removeToolkit(toolkitName) {
        try {
            await this.axiosInstance.delete(`/api/v1/toolkits/${toolkitName}`);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to remove toolkit';
            console.error('ROMA remove toolkit failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'REMOVE_TOOLKIT_FAILED', error);
        }
    }
    /**
     * Get available profiles
     */
    async getProfiles() {
        try {
            const response = await this.axiosInstance.get('/api/v1/profiles');
            return response.data;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get profiles';
            console.error('ROMA get profiles failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_PROFILES_FAILED', error);
        }
    }
    /**
     * Get profile configuration
     * @param profileName Profile name
     */
    async getProfileConfig(profileName) {
        try {
            const response = await this.axiosInstance.get(`/api/v1/profiles/${profileName}`);
            return response.data;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get profile config';
            console.error('ROMA get profile config failed:', error);
            throw new plugin_types_1.RomaPluginError(errorMessage, 'GET_PROFILE_CONFIG_FAILED', error);
        }
    }
    /**
     * Check if client is initialized
     */
    isClientInitialized() {
        return this.isInitialized;
    }
    /**
     * Handle Axios errors
     * @param error Axios error
     */
    handleAxiosError(error) {
        if (error.response) {
            // Server responded with a status code outside 2xx
            const status = error.response.status;
            const data = error.response.data;
            let errorMessage = 'Request failed';
            if (typeof data === 'string') {
                errorMessage = data;
            }
            else if (data && typeof data === 'object' && 'message' in data) {
                errorMessage = data.message;
            }
            else if (data && typeof data === 'object' && 'detail' in data) {
                errorMessage = data.detail;
            }
            console.error(`ROMA API Error ${status}: ${errorMessage}`);
            throw new plugin_types_1.RomaPluginError(errorMessage, `API_ERROR_${status}`, data);
        }
        else if (error.request) {
            // Request was made but no response received
            console.error('ROMA API No Response:', error.message);
            throw new plugin_types_1.RomaPluginError('No response from server', 'NO_RESPONSE', error.message);
        }
        else {
            // Something happened in setting up the request
            console.error('ROMA API Request Error:', error.message);
            throw new plugin_types_1.RomaPluginError(error.message, 'REQUEST_ERROR', error);
        }
    }
    /**
     * Map execution response to standard format
     * @param execution Execution data from API
     */
    mapExecutionResponse(execution) {
        return {
            executionId: execution.execution_id || execution.id || execution.executionId,
            goal: execution.goal,
            status: execution.status,
            result: execution.result,
            error: execution.error,
            statistics: {
                executionTime: execution.execution_time || execution.statistics?.executionTime || 0,
                subtasksCreated: execution.subtasks_created || execution.statistics?.subtasksCreated || 0,
                subtasksCompleted: execution.subtasks_completed || execution.statistics?.subtasksCompleted || 0,
                toolsUsed: execution.tools_used || execution.statistics?.toolsUsed || [],
                modulesUsed: execution.modules_used || execution.statistics?.modulesUsed || []
            },
            timestamp: execution.timestamp || Date.now()
        };
    }
    /**
     * Create request configuration with timeout
     * @param customConfig Custom configuration
     */
    createRequestConfig(customConfig) {
        return {
            timeout: customConfig?.timeout || this.config.timeout,
            ...customConfig
        };
    }
}
exports.RomaClient = RomaClient;
//# sourceMappingURL=RomaClient.js.map