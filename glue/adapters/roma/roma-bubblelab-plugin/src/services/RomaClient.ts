/**
 * ROMA Client Service
 *
 * This service handles all HTTP communication with the ROMA backend API.
 * It provides methods for executing tasks, managing executions, and configuring MCP servers and toolkits.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import {
  RomaClient as RomaClientInterface,
  RomaClientConfig,
  RomaExecutionResult,
  RomaExecutionOptions,
  RomaMcpServerConfig,
  RomaToolkitConfig,
  RomaExecutionStatistics,
  RomaExecutionStatus,
  RomaHealthStatus,
  RomaPluginConfig,
  RomaPluginError
} from '../types/plugin-types';

/**
 * ROMA Client Implementation
 */
export class RomaClient implements RomaClientInterface {
  public config: RomaClientConfig;
  private axiosInstance: AxiosInstance;
  private isInitialized: boolean = false;

  /**
   * Create ROMA Client
   * @param config Client configuration
   */
  constructor(config: RomaClientConfig) {
    this.config = {
      baseUrl: config.baseUrl || import.meta.env?.VITE_ROMA_SERVER_URL || 'http://localhost:8000',
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
      headers: config.headers || {}
    };

    // Create Axios instance
    this.axiosInstance = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.headers || {})
      }
    });

    // Add request interceptor for API key
    this.axiosInstance.interceptors.request.use((config) => {
      if (this.config.apiKey) {
        if (!config.headers) {
          config.headers = {} as any;
        }
        (config.headers as any)['Authorization'] = `Bearer ${this.config.apiKey}`;
      }
      return config;
    });

    // Add response interceptor for error handling
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        return this.handleAxiosError(error);
      }
    );
  }

  /**
   * Initialize the client
   */
  public async initialize(): Promise<void> {
    try {
      // Test connection to ROMA server
      const healthResponse = await this.getStatus();
      if (healthResponse.status === 'healthy') {
        this.isInitialized = true;
        console.log('ROMA client initialized successfully');
      } else {
        throw new RomaPluginError('ROMA server not healthy', 'SERVER_NOT_HEALTHY', healthResponse);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to initialize ROMA client';
      console.error('ROMA client initialization failed:', error);
      throw new RomaPluginError(errorMessage, 'CLIENT_INITIALIZATION_FAILED', error);
    }
  }

  /**
   * Update client configuration
   * @param configUpdate Configuration updates
   */
  public updateConfig(configUpdate: Partial<RomaClientConfig>): void {
    this.config = { ...this.config, ...configUpdate };

    // Update axios instance if base URL or timeout changed
    if (configUpdate.baseUrl || configUpdate.timeout) {
      this.axiosInstance.defaults.baseURL = this.config.baseUrl;
      this.axiosInstance.defaults.timeout = this.config.timeout;
    }

    // Update headers if provided
    if (configUpdate.headers) {
      this.axiosInstance.defaults.headers = {
        ...(this.axiosInstance.defaults.headers as any),
        ...configUpdate.headers
      } as any;
    }
  }

  /**
   * Execute a task using ROMA
   * @param goal The task goal
   * @param options Execution options
   */
  public async executeTask(
    goal: string,
    options?: RomaExecutionOptions
  ): Promise<RomaExecutionResult> {
    try {
      const requestData: Record<string, any> = {
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

      const response = await this.axiosInstance.post<RomaExecutionResult>(
        '/api/v1/executions',
        requestData
      );

      return this.mapExecutionResponse(response.data);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to execute task';
      console.error('ROMA task execution failed:', error);
      throw new RomaPluginError(errorMessage, 'TASK_EXECUTION_FAILED', error);
    }
  }

  /**
   * Get execution by ID
   * @param executionId Execution ID
   */
  public async getExecution(executionId: string): Promise<RomaExecutionResult> {
    try {
      const response = await this.axiosInstance.get<RomaExecutionResult>(
        `/api/v1/executions/${executionId}`
      );
      return this.mapExecutionResponse(response.data);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get execution';
      console.error('ROMA get execution failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_EXECUTION_FAILED', error);
    }
  }

  /**
   * Get execution history
   * @param limit Maximum number of results
   */
  public async getExecutionHistory(limit?: number): Promise<RomaExecutionResult[]> {
    try {
      const params: Record<string, any> = {};
      if (limit) {
        params.limit = limit;
      }

      const response = await this.axiosInstance.get<RomaExecutionResult[]>(
        '/api/v1/executions',
        { params }
      );

      return response.data.map(exec => this.mapExecutionResponse(exec));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get execution history';
      console.error('ROMA get execution history failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_EXECUTION_HISTORY_FAILED', error);
    }
  }

  /**
   * Cancel execution
   * @param executionId Execution ID
   */
  public async cancelExecution(executionId: string): Promise<void> {
    try {
      await this.axiosInstance.post(
        `/api/v1/executions/${executionId}/cancel`
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to cancel execution';
      console.error('ROMA cancel execution failed:', error);
      throw new RomaPluginError(errorMessage, 'CANCEL_EXECUTION_FAILED', error);
    }
  }

  /**
   * Get server status
   */
  public async getStatus(): Promise<{ status: RomaHealthStatus }> {
    try {
      const response = await this.axiosInstance.get('/health');
      return { status: response.data.status as RomaHealthStatus };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get server status';
      console.error('ROMA get status failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_STATUS_FAILED', error);
    }
  }

  /**
   * Get execution statistics
   */
  public async getStatistics(): Promise<RomaExecutionStatistics> {
    try {
      const response = await this.axiosInstance.get<RomaExecutionStatistics>(
        '/api/v1/statistics'
      );
      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get statistics';
      console.error('ROMA get statistics failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_STATISTICS_FAILED', error);
    }
  }

  /**
   * Get available MCP servers
   */
  public async getAvailableMcps(): Promise<RomaMcpServerConfig[]> {
    try {
      const response = await this.axiosInstance.get<RomaMcpServerConfig[]>(
        '/api/v1/mcp-servers'
      );
      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get MCP servers';
      console.error('ROMA get MCP servers failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_MCP_SERVERS_FAILED', error);
    }
  }

  /**
   * Add MCP server configuration
   * @param mcpConfig MCP server configuration
   */
  public async addMcpServer(mcpConfig: RomaMcpServerConfig): Promise<void> {
    try {
      await this.axiosInstance.post('/api/v1/mcp-servers', mcpConfig);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add MCP server';
      console.error('ROMA add MCP server failed:', error);
      throw new RomaPluginError(errorMessage, 'ADD_MCP_SERVER_FAILED', error);
    }
  }

  /**
   * Remove MCP server
   * @param serverName Server name
   */
  public async removeMcpServer(serverName: string): Promise<void> {
    try {
      await this.axiosInstance.delete(`/api/v1/mcp-servers/${serverName}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to remove MCP server';
      console.error('ROMA remove MCP server failed:', error);
      throw new RomaPluginError(errorMessage, 'REMOVE_MCP_SERVER_FAILED', error);
    }
  }

  /**
   * Get available toolkits
   */
  public async getAvailableToolkits(): Promise<RomaToolkitConfig[]> {
    try {
      const response = await this.axiosInstance.get<RomaToolkitConfig[]>(
        '/api/v1/toolkits'
      );
      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get toolkits';
      console.error('ROMA get toolkits failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_TOOLKITS_FAILED', error);
    }
  }

  /**
   * Add toolkit configuration
   * @param toolkitConfig Toolkit configuration
   */
  public async addToolkit(toolkitConfig: RomaToolkitConfig): Promise<void> {
    try {
      await this.axiosInstance.post('/api/v1/toolkits', toolkitConfig);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add toolkit';
      console.error('ROMA add toolkit failed:', error);
      throw new RomaPluginError(errorMessage, 'ADD_TOOLKIT_FAILED', error);
    }
  }

  /**
   * Remove toolkit
   * @param toolkitName Toolkit name
   */
  public async removeToolkit(toolkitName: string): Promise<void> {
    try {
      await this.axiosInstance.delete(`/api/v1/toolkits/${toolkitName}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to remove toolkit';
      console.error('ROMA remove toolkit failed:', error);
      throw new RomaPluginError(errorMessage, 'REMOVE_TOOLKIT_FAILED', error);
    }
  }

  /**
   * Get available profiles
   */
  public async getProfiles(): Promise<string[]> {
    try {
      const response = await this.axiosInstance.get<string[]>(
        '/api/v1/profiles'
      );
      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get profiles';
      console.error('ROMA get profiles failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_PROFILES_FAILED', error);
    }
  }

  /**
   * Get profile configuration
   * @param profileName Profile name
   */
  public async getProfileConfig(profileName: string): Promise<Partial<RomaPluginConfig>> {
    try {
      const response = await this.axiosInstance.get<Partial<RomaPluginConfig>>(
        `/api/v1/profiles/${profileName}`
      );
      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get profile config';
      console.error('ROMA get profile config failed:', error);
      throw new RomaPluginError(errorMessage, 'GET_PROFILE_CONFIG_FAILED', error);
    }
  }

  /**
   * Check if client is initialized
   */
  public isClientInitialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Handle Axios errors
   * @param error Axios error
   */
  private handleAxiosError(error: AxiosError): Promise<any> {
    if (error.response) {
      // Server responded with a status code outside 2xx
      const { status } = error.response;
      const { data } = error.response;

      let errorMessage = 'Request failed';
      if (typeof data === 'string') {
        errorMessage = data;
      } else if (data && typeof data === 'object' && 'message' in data) {
        errorMessage = (data as { message: string }).message;
      } else if (data && typeof data === 'object' && 'detail' in data) {
        errorMessage = (data as { detail: string }).detail;
      }

      console.error(`ROMA API Error ${status}: ${errorMessage}`);
      throw new RomaPluginError(errorMessage, `API_ERROR_${status}`, data);
    } else if (error.request) {
      // Request was made but no response received
      console.error('ROMA API No Response:', error.message);
      throw new RomaPluginError('No response from server', 'NO_RESPONSE', error.message);
    } else {
      // Something happened in setting up the request
      console.error('ROMA API Request Error:', error.message);
      throw new RomaPluginError(error.message, 'REQUEST_ERROR', error);
    }
  }

  /**
   * Map execution response to standard format
   * @param execution Execution data from API
   */
  private mapExecutionResponse(execution: any): RomaExecutionResult {
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
  private createRequestConfig(customConfig?: AxiosRequestConfig): AxiosRequestConfig {
    return {
      timeout: customConfig?.timeout || this.config.timeout,
      ...customConfig
    };
  }
}
