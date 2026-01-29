/**
 * OpenEvolve API Client
 * Handles all HTTP requests to the backend API
 */

import {
  Workflow,
  Team,
  Gauntlet,
  CreateWorkflowRequest,
  UpdateWorkflowRequest,
  CreateTeamRequest,
  UpdateTeamRequest,
  CreateGauntletRequest,
  UpdateGauntletRequest,
  LLMConfig,
  ExecutionResult,
  WorkflowListResponse,
  TeamListResponse,
  GauntletListResponse,
  ApiResponse,
  ListQueryParams,
} from '../types/api';
import { OPENEVOLVE_API_BASE_URL } from '../env';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL =
  OPENEVOLVE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  'http://localhost:8001';
const API_TIMEOUT = 30000; // 30 seconds

// ============================================================================
// API Client Class
// ============================================================================

class ApiClient {
  private baseURL: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string, timeout: number = API_TIMEOUT) {
    this.baseURL = baseURL;
    this.timeout = timeout;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string) {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Clear authentication token
   */
  clearAuthToken() {
    delete this.defaultHeaders['Authorization'];
  }

  /**
   * Make an HTTP request
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...config,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          error: response.statusText,
        }));
        throw new Error(error.error || error.detail || 'API request failed');
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        throw error;
      }

      throw new Error('Unknown error occurred');
    }
  }

  /**
   * GET request
   */
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<T> {
    const queryString = params
      ? `?${new URLSearchParams(params).toString()}`
      : '';
    return this.request<T>(`${endpoint}${queryString}`, {
      method: 'GET',
    });
  }

  /**
   * POST request
   */
  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * PUT request
   */
  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /**
   * PATCH request
   */
  async patch<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE request
   */
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    });
  }

  // ========================================================================
  // Health Check
  // ========================================================================

  async healthCheck() {
    return this.get('/health');
  }

  // ========================================================================
  // Workflows
  // ========================================================================

  async getWorkflows(params?: ListQueryParams) {
    return this.get<WorkflowListResponse>('/api/workflows', params);
  }

  async getWorkflow(workflowId: string) {
    return this.get<Workflow>(`/api/workflows/${workflowId}`);
  }

  async createWorkflow(data: CreateWorkflowRequest) {
    return this.post<Workflow>('/api/workflows', data);
  }

  async updateWorkflow(workflowId: string, data: UpdateWorkflowRequest) {
    return this.put<Workflow>(`/api/workflows/${workflowId}`, data);
  }

  async deleteWorkflow(workflowId: string) {
    return this.delete<{ message: string }>(`/api/workflows/${workflowId}`);
  }

  async startWorkflow(workflowId: string) {
    return this.post<Workflow>(`/api/workflows/${workflowId}/start`);
  }

  async pauseWorkflow(workflowId: string) {
    return this.post<Workflow>(`/api/workflows/${workflowId}/pause`);
  }

  async resumeWorkflow(workflowId: string) {
    return this.post<Workflow>(`/api/workflows/${workflowId}/resume`);
  }

  async stopWorkflow(workflowId: string) {
    return this.post<Workflow>(`/api/workflows/${workflowId}/stop`);
  }

  async getWorkflowResults(workflowId: string) {
    return this.get<ExecutionResult>(`/api/workflows/${workflowId}/results`);
  }

  // ========================================================================
  // Teams
  // ========================================================================

  async getTeams(params?: ListQueryParams) {
    return this.get<TeamListResponse>('/api/teams', params);
  }

  async getTeam(teamId: string) {
    return this.get<Team>(`/api/teams/${teamId}`);
  }

  async createTeam(data: CreateTeamRequest) {
    return this.post<Team>('/api/teams', data);
  }

  async updateTeam(teamId: string, data: UpdateTeamRequest) {
    return this.put<Team>(`/api/teams/${teamId}`, data);
  }

  async deleteTeam(teamId: string) {
    return this.delete<{ message: string }>(`/api/teams/${teamId}`);
  }

  // ========================================================================
  // Gauntlets
  // ========================================================================

  async getGauntlets(params?: ListQueryParams) {
    return this.get<GauntletListResponse>('/api/gauntlets', params);
  }

  async getGauntlet(gauntletId: string) {
    return this.get<Gauntlet>(`/api/gauntlets/${gauntletId}`);
  }

  async createGauntlet(data: CreateGauntletRequest) {
    return this.post<Gauntlet>('/api/gauntlets', data);
  }

  async updateGauntlet(gauntletId: string, data: UpdateGauntletRequest) {
    return this.put<Gauntlet>(`/api/gauntlets/${gauntletId}`, data);
  }

  async deleteGauntlet(gauntletId: string) {
    return this.delete<{ message: string }>(`/api/gauntlets/${gauntletId}`);
  }

  // ========================================================================
  // Settings
  // ========================================================================

  async getLLMConfig() {
    return this.get<LLMConfig>('/api/settings/llm');
  }

  async updateLLMConfig(data: Partial<LLMConfig>) {
    return this.put<LLMConfig>('/api/settings/llm', data);
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const apiClient = new ApiClient(API_BASE_URL);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Set auth token from Clerk
 */
export const setAuthToken = (token: string) => {
  apiClient.setAuthToken(token);
};

/**
 * Clear auth token
 */
export const clearAuthToken = () => {
  apiClient.clearAuthToken();
};

export default apiClient;
