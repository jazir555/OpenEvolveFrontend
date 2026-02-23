/**
 * ROMA Client Service
 *
 * This service handles all HTTP communication with the ROMA backend API.
 * It provides methods for executing tasks, managing executions, and configuring MCP servers and toolkits.
 */
import { RomaClient as RomaClientInterface, RomaClientConfig, RomaExecutionResult, RomaExecutionOptions, RomaMcpServerConfig, RomaToolkitConfig, RomaExecutionStatistics, RomaExecutionStatus, RomaPluginConfig } from '../types/plugin-types';
/**
 * ROMA Client Implementation
 */
export declare class RomaClient implements RomaClientInterface {
    config: RomaClientConfig;
    private axiosInstance;
    private isInitialized;
    /**
     * Create ROMA Client
     * @param config Client configuration
     */
    constructor(config: RomaClientConfig);
    /**
     * Initialize the client
     */
    initialize(): Promise<void>;
    /**
     * Update client configuration
     * @param configUpdate Configuration updates
     */
    updateConfig(configUpdate: Partial<RomaClientConfig>): void;
    /**
     * Execute a task using ROMA
     * @param goal The task goal
     * @param options Execution options
     */
    executeTask(goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult>;
    /**
     * Get execution by ID
     * @param executionId Execution ID
     */
    getExecution(executionId: string): Promise<RomaExecutionResult>;
    /**
     * Get execution history
     * @param limit Maximum number of results
     */
    getExecutionHistory(limit?: number): Promise<RomaExecutionResult[]>;
    /**
     * Cancel execution
     * @param executionId Execution ID
     */
    cancelExecution(executionId: string): Promise<void>;
    /**
     * Get server status
     */
    getStatus(): Promise<{
        status: RomaExecutionStatus;
    }>;
    /**
     * Get execution statistics
     */
    getStatistics(): Promise<RomaExecutionStatistics>;
    /**
     * Get available MCP servers
     */
    getAvailableMcps(): Promise<RomaMcpServerConfig[]>;
    /**
     * Add MCP server configuration
     * @param mcpConfig MCP server configuration
     */
    addMcpServer(mcpConfig: RomaMcpServerConfig): Promise<void>;
    /**
     * Remove MCP server
     * @param serverName Server name
     */
    removeMcpServer(serverName: string): Promise<void>;
    /**
     * Get available toolkits
     */
    getAvailableToolkits(): Promise<RomaToolkitConfig[]>;
    /**
     * Add toolkit configuration
     * @param toolkitConfig Toolkit configuration
     */
    addToolkit(toolkitConfig: RomaToolkitConfig): Promise<void>;
    /**
     * Remove toolkit
     * @param toolkitName Toolkit name
     */
    removeToolkit(toolkitName: string): Promise<void>;
    /**
     * Get available profiles
     */
    getProfiles(): Promise<string[]>;
    /**
     * Get profile configuration
     * @param profileName Profile name
     */
    getProfileConfig(profileName: string): Promise<Partial<RomaPluginConfig>>;
    /**
     * Check if client is initialized
     */
    isClientInitialized(): boolean;
    /**
     * Handle Axios errors
     * @param error Axios error
     */
    private handleAxiosError;
    /**
     * Map execution response to standard format
     * @param execution Execution data from API
     */
    private mapExecutionResponse;
    /**
     * Create request configuration with timeout
     * @param customConfig Custom configuration
     */
    private createRequestConfig;
}
//# sourceMappingURL=RomaClient.d.ts.map