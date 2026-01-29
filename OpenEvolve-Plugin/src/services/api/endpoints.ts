import { apiClient } from './client';
import { errorLogger } from '@/utils';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import type {
  User,
  WorkflowExecution,
  AnalyticsData,
  PerformanceAnalytics,
  KnowledgeArtifact,
  AdversarialTest,
  LeanCodeOutput,
  VerificationResult,
} from '@/stores/index';

/**
 * Authentication Endpoints
 */
export const authApi = {
  /**
   * Login user
   */
  login: async (email: string, password: string) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<{ access_token: string; refresh_token: string; user?: User }>(
        '/auth/login',
        { email, password }
      );
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'authApi',
        function: 'login',
        operation: 'USER_LOGIN',
        additionalData: { email }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Login API error'),
        'error',
        { component: 'authApi', function: 'login' }
      );
      throw result.error || new Error('Login API error');
    }

    return result.data!;
  },

  /**
   * Register new user
   */
  register: async (email: string, password: string, username: string, full_name?: string) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<User>('/auth/register', {
        email,
        password,
        username,
        full_name,
      });
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'authApi',
        function: 'register',
        operation: 'USER_REGISTRATION',
        additionalData: { email, username }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Register API error'),
        'error',
        { component: 'authApi', function: 'register' }
      );
      throw result.error || new Error('Register API error');
    }

    return result.data!;
  },

  /**
   * Refresh access token
   */
  refreshToken: async (refreshToken: string) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<{ access_token: string }>('/auth/refresh', {
        refresh_token: refreshToken,
      });
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 1000,
      showUserNotification: false, // Don't show notification for automatic token refresh
      logError: true,
      context: {
        component: 'authApi',
        function: 'refreshToken',
        operation: 'TOKEN_REFRESH',
        additionalData: { refreshToken: refreshToken ? '***' : 'null' }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Refresh token API error'),
        'error',
        { component: 'authApi', function: 'refreshToken' }
      );
      throw result.error || new Error('Refresh token API error');
    }

    return result.data!;
  },

  /**
   * Logout user
   */
  logout: async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post('/auth/logout', {});
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'authApi',
        function: 'logout',
        operation: 'USER_LOGOUT',
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Logout API error'),
        'error',
        { component: 'authApi', function: 'logout' }
      );
      throw result.error || new Error('Logout API error');
    }

    return result.data!;
  },
};

/**
 * User Management Endpoints
 */
export const userApi = {
  /**
   * Get current user profile
   */
  getProfile: async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.get<User>('/users/me');
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'userApi',
        function: 'getProfile',
        operation: 'GET_USER_PROFILE',
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Get profile API error'),
        'error',
        { component: 'userApi', function: 'getProfile' }
      );
      throw result.error || new Error('Get profile API error');
    }

    return result.data!;
  },

  /**
   * Update current user profile
   */
  updateProfile: async (updates: Partial<User>) => {
    try {
      return await apiClient.put<User>('/users/me', updates);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Update profile API error'),
        'error',
        { component: 'userApi', function: 'updateProfile' }
      );
      throw error;
    }
  },
};

/**
 * Evolution Engine Endpoints
 */
export const evolutionApi = {
  /**
   * Start evolution
   */
  start: async (data: {
    content: string;
    mode: 'standard' | 'quality_diversity' | 'island_model';
    parameters: {
      max_iterations: number;
      population_size: number;
      temperature: number;
      top_p: number;
    };
    models: Array<{
      provider: string;
      model: string;
      api_key: string;
    }>;
  }) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<{
        evolution_id: string;
        status: string;
        created_at: string;
        websocket_url: string;
      }>('/evolution/start', data);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'evolutionApi',
        function: 'start',
        operation: 'START_EVOLUTION',
        additionalData: { mode: data.mode, parameters: data.parameters }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Start evolution API error'),
        'error',
        { component: 'evolutionApi', function: 'start' }
      );
      throw result.error || new Error('Start evolution API error');
    }

    return result.data!;
  },

  /**
   * Get evolution status
   */
  getStatus: async (evolutionId: string) => {
    try {
      return await apiClient.get<WorkflowExecution>(`/evolution/${evolutionId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get evolution status API error'),
        'error',
        { component: 'evolutionApi', function: 'getStatus', additionalData: { evolutionId } }
      );
      throw error;
    }
  },

  /**
   * Pause evolution
   */
  pause: async (evolutionId: string) => {
    try {
      return await apiClient.post<{ evolution_id: string; status: string; paused_at: string }>(
        `/evolution/${evolutionId}/pause`,
        {}
      );
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Pause evolution API error'),
        'error',
        { component: 'evolutionApi', function: 'pause', additionalData: { evolutionId } }
      );
      throw error;
    }
  },

  /**
   * Resume evolution
   */
  resume: async (evolutionId: string) => {
    try {
      return await apiClient.post<{ evolution_id: string; status: string; resumed_at: string }>(
        `/evolution/${evolutionId}/resume`,
        {}
      );
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Resume evolution API error'),
        'error',
        { component: 'evolutionApi', function: 'resume', additionalData: { evolutionId } }
      );
      throw error;
    }
  },

  /**
   * Stop evolution
   */
  stop: async (evolutionId: string) => {
    try {
      return await apiClient.post<{
        evolution_id: string;
        status: string;
        stopped_at: string;
        final_results: any;
      }>(`/evolution/${evolutionId}/stop`, {});
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Stop evolution API error'),
        'error',
        { component: 'evolutionApi', function: 'stop', additionalData: { evolutionId } }
      );
      throw error;
    }
  },

  /**
   * Delete evolution
   */
  delete: async (evolutionId: string) => {
    try {
      return await apiClient.delete(`/evolution/${evolutionId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Delete evolution API error'),
        'error',
        { component: 'evolutionApi', function: 'delete', additionalData: { evolutionId } }
      );
      throw error;
    }
  },

  /**
   * List evolutions
   */
  list: async (params?: {
    status?: string;
    limit?: number;
    offset?: number;
    sort?: string;
    order?: 'asc' | 'desc';
  }) => {
    try {
      return await apiClient.get<{
        evolutions: WorkflowExecution[];
        total: number;
        limit: number;
        offset: number;
      }>('/evolution', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('List evolutions API error'),
        'error',
        { component: 'evolutionApi', function: 'list', additionalData: { params } }
      );
      throw error;
    }
  },
};

/**
 * Adversarial Testing Endpoints
 */
export const adversarialApi = {
  /**
   * Start adversarial test
   */
  start: async (data: {
    content: string;
    attack_modes: string[];
    parameters: {
      num_rounds: number;
      red_team_models: Array<{ provider: string; model: string }>;
      blue_team_models: Array<{ provider: string; model: string }>;
    };
  }) => {
    try {
      return await apiClient.post<{
        test_id: string;
        status: string;
        created_at: string;
        websocket_url: string;
      }>('/adversarial/start', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Start adversarial test API error'),
        'error',
        { component: 'adversarialApi', function: 'start' }
      );
      throw error;
    }
  },

  /**
   * Get adversarial test status
   */
  getStatus: async (testId: string) => {
    try {
      return await apiClient.get<AdversarialTest>(`/adversarial/${testId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get adversarial test status API error'),
        'error',
        { component: 'adversarialApi', function: 'getStatus', additionalData: { testId } }
      );
      throw error;
    }
  },

  /**
   * Approve or reject patch
   */
  approvePatch: async (
    testId: string,
    data: { round: number; approved: boolean; feedback?: string }
  ) => {
    try {
      return await apiClient.post<{ test_id: string; round: number; patch_approved: boolean }>(
        `/adversarial/${testId}/approve-patch`,
        data
      );
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Approve patch API error'),
        'error',
        { component: 'adversarialApi', function: 'approvePatch', additionalData: { testId } }
      );
      throw error;
    }
  },

  /**
   * Stop adversarial test
   */
  stop: async (testId: string) => {
    try {
      return await apiClient.post<{ test_id: string; status: string; stopped_at: string }>(
        `/adversarial/${testId}/stop`,
        {}
      );
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Stop adversarial test API error'),
        'error',
        { component: 'adversarialApi', function: 'stop', additionalData: { testId } }
      );
      throw error;
    }
  },

  /**
   * List adversarial tests
   */
  list: async (params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }) => {
    try {
      return await apiClient.get<{ tests: AdversarialTest[]; total: number }>('/adversarial', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('List adversarial tests API error'),
        'error',
        { component: 'adversarialApi', function: 'list', additionalData: { params } }
      );
      throw error;
    }
  },
};

/**
 * Analytics Endpoints
 */
export const analyticsApi = {
  /**
   * Get metrics
   */
  getMetrics: async (params: {
    start_date: string;
    end_date: string;
    granularity: 'hour' | 'day' | 'week' | 'month';
  }) => {
    try {
      return await apiClient.get<AnalyticsData>('/analytics/metrics', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get metrics API error'),
        'error',
        { component: 'analyticsApi', function: 'getMetrics', additionalData: { params } }
      );
      throw error;
    }
  },

  /**
   * Get performance analytics
   */
  getPerformance: async () => {
    try {
      return await apiClient.get<PerformanceAnalytics>('/analytics/performance');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get performance analytics API error'),
        'error',
        { component: 'analyticsApi', function: 'getPerformance' }
      );
      throw error;
    }
  },
};

/**
 * Monitoring Endpoints
 */
export const monitoringApi = {
  /**
   * Get system health
   */
  getHealth: async () => {
    try {
      return await apiClient.get<{
        status: string;
        services: Record<string, string>;
        resource_usage: {
          cpu_percent: number;
          memory_percent: number;
          disk_percent: number;
        };
        active_operations: {
          evolutions_running: number;
          adversarial_tests_running: number;
        };
      }>('/monitoring/health');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get system health API error'),
        'error',
        { component: 'monitoringApi', function: 'getHealth' }
      );
      throw error;
    }
  },

  /**
   * Get application logs
   */
  getLogs: async (params?: {
    level?: 'INFO' | 'WARNING' | 'ERROR';
    limit?: number;
    offset?: number;
  }) => {
    try {
      return await apiClient.get<{
        logs: Array<{
          timestamp: string;
          level: string;
          message: string;
          context?: any;
        }>;
        total: number;
      }>('/monitoring/logs', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get application logs API error'),
        'error',
        { component: 'monitoringApi', function: 'getLogs', additionalData: { params } }
      );
      throw error;
    }
  },
};

/**
 * Content Management Endpoints
 */
export const contentApi = {
  /**
   * Create content
   */
  create: async (data: {
    title: string;
    content: string;
    language?: string;
    tags?: string[];
  }) => {
    try {
      return await apiClient.post<KnowledgeArtifact>('/content', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Create content API error'),
        'error',
        { component: 'contentApi', function: 'create' }
      );
      throw error;
    }
  },

  /**
   * Get content by ID
   */
  getById: async (contentId: string) => {
    try {
      return await apiClient.get<KnowledgeArtifact>(`/content/${contentId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get content by ID API error'),
        'error',
        { component: 'contentApi', function: 'getById', additionalData: { contentId } }
      );
      throw error;
    }
  },

  /**
   * Update content
   */
  update: async (contentId: string, data: Partial<KnowledgeArtifact>) => {
    try {
      return await apiClient.put<KnowledgeArtifact>(`/content/${contentId}`, data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Update content API error'),
        'error',
        { component: 'contentApi', function: 'update', additionalData: { contentId } }
      );
      throw error;
    }
  },

  /**
   * Delete content
   */
  delete: async (contentId: string) => {
    try {
      return await apiClient.delete(`/content/${contentId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Delete content API error'),
        'error',
        { component: 'contentApi', function: 'delete', additionalData: { contentId } }
      );
      throw error;
    }
  },

  /**
   * List content
   */
  list: async (params?: {
    tag?: string;
    language?: string;
    limit?: number;
    offset?: number;
  }) => {
    try {
      return await apiClient.get<{ content: KnowledgeArtifact[]; total: number }>('/content', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('List content API error'),
        'error',
        { component: 'contentApi', function: 'list', additionalData: { params } }
      );
      throw error;
    }
  },
};

/**
 * Version Control Endpoints
 */
export const versionApi = {
  /**
   * Get version history
   */
  getHistory: async (contentId: string) => {
    try {
      return await apiClient.get<
        Array<{
          version: number;
          created_at: string;
          created_by: string;
          comment: string;
        }>
      >(`/content/${contentId}/versions`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get version history API error'),
        'error',
        { component: 'versionApi', function: 'getHistory', additionalData: { contentId } }
      );
      throw error;
    }
  },

  /**
   * Revert to version
   */
  revert: async (contentId: string, version: number) => {
    try {
      return await apiClient.post<{
        content_id: string;
        reverted_to_version: number;
        new_version: number;
        reverted_at: string;
      }>(`/content/${contentId}/versions/${version}/revert`, {});
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Revert to version API error'),
        'error',
        { component: 'versionApi', function: 'revert', additionalData: { contentId, version } }
      );
      throw error;
    }
  },

  /**
   * Get diff between versions
   */
  getDiff: async (contentId: string, version1: number, version2: number) => {
    try {
      return await apiClient.get<{
        version1: number;
        version2: number;
        diff: string;
      }>(`/content/${contentId}/versions/${version1}/diff/${version2}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get diff API error'),
        'error',
        { component: 'versionApi', function: 'getDiff', additionalData: { contentId, version1, version2 } }
      );
      throw error;
    }
  },

  /**
   * Create branch
   */
  createBranch: async (contentId: string, data: { branch_name: string; from_version: number }) => {
    try {
      return await apiClient.post<{
        branch_id: string;
        branch_name: string;
        created_at: string;
      }>(`/content/${contentId}/branches`, data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Create branch API error'),
        'error',
        { component: 'versionApi', function: 'createBranch', additionalData: { contentId } }
      );
      throw error;
    }
  },

  /**
   * List branches
   */
  listBranches: async (contentId: string) => {
    try {
      return await apiClient.get<
        Array<{
          branch_id: string;
          branch_name: string;
          version: number;
          created_at: string;
        }>
      >(`/content/${contentId}/branches`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('List branches API error'),
        'error',
        { component: 'versionApi', function: 'listBranches', additionalData: { contentId } }
      );
      throw error;
    }
  },
};

/**
 * Collaboration Endpoints
 */
export const collaborationApi = {
  /**
   * Create collaboration room
   */
  createRoom: async (data: { content_id: string; room_name?: string }) => {
    try {
      return await apiClient.post<{
        room_id: string;
        room_name: string;
        websocket_url: string;
        created_at: string;
      }>('/collaboration/rooms', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Create collaboration room API error'),
        'error',
        { component: 'collaborationApi', function: 'createRoom' }
      );
      throw error;
    }
  },

  /**
   * Get active users in room
   */
  getRoomUsers: async (roomId: string) => {
    try {
      return await apiClient.get<
        Array<{
          user_id: string;
          username: string;
          joined_at: string;
          cursor_position?: { line: number; column: number };
        }>
      >(`/collaboration/rooms/${roomId}/users`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get room users API error'),
        'error',
        { component: 'collaborationApi', function: 'getRoomUsers', additionalData: { roomId } }
      );
      throw error;
    }
  },
};

/**
 * Comments Endpoints
 */
export const commentsApi = {
  /**
   * Add comment to content
   */
  add: async (contentId: string, data: {
    comment: string;
    line_start?: number;
    line_end?: number;
    parent_comment_id?: string;
  }) => {
    try {
      return await apiClient.post<{
        comment_id: string;
        comment: string;
        created_at: string;
      }>(`/content/${contentId}/comments`, data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Add comment API error'),
        'error',
        { component: 'commentsApi', function: 'add', additionalData: { contentId } }
      );
      throw error;
    }
  },

  /**
   * Get comments for content
   */
  get: async (contentId: string) => {
    try {
      return await apiClient.get<
        Array<{
          comment_id: string;
          user_id: string;
          username: string;
          comment: string;
          line_start?: number;
          line_end?: number;
          created_at: string;
          replies: any[];
        }>
      >(`/content/${contentId}/comments`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get comments API error'),
        'error',
        { component: 'commentsApi', function: 'get', additionalData: { contentId } }
      );
      throw error;
    }
  },
};

/**
 * Configuration Endpoints
 */
export const configApi = {
  /**
   * Get available providers
   */
  getProviders: async () => {
    try {
      return await apiClient.get<
        Array<{
          provider: string;
          name: string;
          models: string[];
          requires_api_key: boolean;
        }>
      >('/config/providers');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get providers API error'),
        'error',
        { component: 'configApi', function: 'getProviders' }
      );
      throw error;
    }
  },

  /**
   * Save API key for provider
   */
  saveApiKey: async (provider: string, apiKey: string) => {
    try {
      return await apiClient.post<{
        provider: string;
        api_key_last_four: string;
        saved_at: string;
      }>(`/config/providers/${provider}/api-key`, { api_key: apiKey });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Save API key API error'),
        'error',
        { component: 'configApi', function: 'saveApiKey', additionalData: { provider } }
      );
      throw error;
    }
  },

  /**
   * Get user parameters
   */
  getParameters: async () => {
    try {
      return await apiClient.get<{
        generation: {
          temperature: number;
          top_p: number;
          max_tokens: number;
        };
        evolution: {
          max_iterations: number;
          population_size: number;
        };
      }>('/config/parameters');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get parameters API error'),
        'error',
        { component: 'configApi', function: 'getParameters' }
      );
      throw error;
    }
  },

  /**
   * Update user parameters
   */
  updateParameters: async (params: {
    generation?: {
      temperature?: number;
      top_p?: number;
      max_tokens?: number;
    };
    evolution?: {
      max_iterations?: number;
      population_size?: number;
    };
  }) => {
    try {
      return await apiClient.put('/config/parameters', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Update parameters API error'),
        'error',
        { component: 'configApi', function: 'updateParameters', additionalData: { params } }
      );
      throw error;
    }
  },
};

/**
 * Workflow Endpoints
 */
export const workflowApi = {
  /**
   * Start integrated workflow
   */
  start: async (data: {
    problem_statement: string;
    workflow_template?: string;
    parameters?: Record<string, any>;
  }) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<{
        workflow_id: string;
        status: string;
        current_stage: string;
        websocket_url: string;
      }>('/workflow/start', data);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'workflowApi',
        function: 'start',
        operation: 'START_WORKFLOW',
        additionalData: {
          hasTemplate: !!data.workflow_template,
          paramCount: Object.keys(data.parameters || {}).length
        }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Start workflow API error'),
        'error',
        { component: 'workflowApi', function: 'start' }
      );
      throw result.error || new Error('Start workflow API error');
    }

    return result.data!;
  },

  /**
   * Get workflow status
   */
  getStatus: async (workflowId: string) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.get<{
        workflow_id: string;
        status: string;
        current_stage: string;
        stages: Array<{
          stage: string;
          status: string;
          result?: any;
          progress?: number;
        }>;
      }>(`/workflow/${workflowId}`);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'workflowApi',
        function: 'getStatus',
        operation: 'GET_WORKFLOW_STATUS',
        additionalData: { workflowId }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Get workflow status API error'),
        'error',
        { component: 'workflowApi', function: 'getStatus', additionalData: { workflowId } }
      );
      throw result.error || new Error('Get workflow status API error');
    }

    return result.data!;
  },

  /**
   * Notify completion for knowledge extraction
   */
  notifyWorkflowComplete: async (data: { workflow_id: string; problem_statement: string; results: any }) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<any>('/api/openevolve/workflow/complete', data);
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'workflowApi',
        function: 'notifyWorkflowComplete',
        operation: 'NOTIFY_WORKFLOW_COMPLETE',
        additionalData: { workflowId: data.workflow_id, resultSize: JSON.stringify(data.results).length }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Notify workflow completion API error'),
        'error',
        { component: 'workflowApi', function: 'notifyWorkflowComplete', additionalData: { workflowId: data.workflow_id } }
      );
      throw result.error || new Error('Notify workflow completion API error');
    }

    return result.data!;
  },
};

/**
 * File Operations Endpoints
 */
export const filesApi = {
  /**
   * Upload file
   */
  upload: async (file: File, onProgress?: (progress: number) => void) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.uploadFile<{
        file_id: string;
        filename: string;
        size: number;
        mime_type: string;
        uploaded_at: string;
      }>('/files/upload', file, onProgress);
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'filesApi',
        function: 'upload',
        operation: 'FILE_UPLOAD',
        additionalData: { filename: file.name, size: file.size }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Upload file API error'),
        'error',
        { component: 'filesApi', function: 'upload' }
      );
      throw result.error || new Error('Upload file API error');
    }

    return result.data!;
  },

  /**
   * Download file
   */
  download: async (fileId: string, filename?: string) => {
    try {
      return await apiClient.downloadFile(`/files/${fileId}/download`, filename);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Download file API error'),
        'error',
        { component: 'filesApi', function: 'download', additionalData: { fileId } }
      );
      throw error;
    }
  },

  /**
   * Get file metadata
   */
  getMetadata: async (fileId: string) => {
    try {
      return await apiClient.get<{
        file_id: string;
        filename: string;
        size: number;
        mime_type: string;
        uploaded_at: string;
      }>(`/files/${fileId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get file metadata API error'),
        'error',
        { component: 'filesApi', function: 'getMetadata', additionalData: { fileId } }
      );
      throw error;
    }
  },
};

/**
 * LeanAide Endpoints
 */
export const leanaideApi = {
  /**
   * Generate Lean 4 proof
   */
  generateProof: async (data: {
    theorem: string;
    proof_attempt?: string;
    model: string;
    temperature: number;
  }) => {
    try {
      return await apiClient.post<LeanCodeOutput>('/leanaide/generate', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Generate proof API error'),
        'error',
        { component: 'leanaideApi', function: 'generateProof' }
      );
      throw error;
    }
  },

  /**
   * Verify Lean 4 proof
   */
  verifyProof: async (code: string) => {
    try {
      return await apiClient.post<VerificationResult>('/leanaide/verify', { code });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Verify proof API error'),
        'error',
        { component: 'leanaideApi', function: 'verifyProof' }
      );
      throw error;
    }
  },

  /**
   * Get supported models
   */
  getModels: async () => {
    try {
      return await apiClient.get<Array<{ provider: string; models: string[] }>>('/leanaide/models');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get models API error'),
        'error',
        { component: 'leanaideApi', function: 'getModels' }
      );
      throw error;
    }
  },

  /**
   * Run benchmark
   */
  runBenchmark: async (data: {
    dataset: any[];
    model: string;
    evaluator: string;
  }) => {
    try {
      return await apiClient.post<{ benchmark_id: string; status: string }>(
        '/leanaide/benchmark/start',
        data
      );
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Run benchmark API error'),
        'error',
        { component: 'leanaideApi', function: 'runBenchmark' }
      );
      throw error;
    }
  },

  /**
   * Get benchmark results
   */
  getBenchmarkResults: async (benchmarkId: string) => {
    try {
      return await apiClient.get<any[]>(`/leanaide/benchmark/${benchmarkId}/results`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get benchmark results API error'),
        'error',
        { component: 'leanaideApi', function: 'getBenchmarkResults', additionalData: { benchmarkId } }
      );
      throw error;
    }
  },
};

/**
 * Knowledge Engine Endpoints
 */
export const knowledgeApi = {
  /**
   * Extract knowledge from text
   */
  extract: async (data: { text: string; schema?: string[] }) => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      return await apiClient.post<{
        entities: any[];
        relations: any[];
        events?: any[];
        timestamp: string;
      }>('/api/openevolve/knowledge/extract', data);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'knowledgeApi',
        function: 'extract',
        operation: 'EXTRACT_KNOWLEDGE',
        additionalData: { textLength: data.text.length, schema: data.schema }
      }
    });

    if (!result.success) {
      errorLogger.logError(
        result.error instanceof Error ? result.error : new Error('Extract knowledge API error'),
        'error',
        { component: 'knowledgeApi', function: 'extract' }
      );
      throw result.error || new Error('Extract knowledge API error');
    }

    return result.data!;
  },

  /**
   * OneKE knowledge extraction (Schema-guided)
   */
  extractOneKE: async (data: { text: string; schema_name?: string }) => {
    try {
      return await apiClient.post<{
        schema_used: string;
        extracted_data: any;
        confidence: number;
        timestamp: string;
      }>('/api/openevolve/oneke/schema-extract', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('OneKE extract API error'),
        'error',
        { component: 'knowledgeApi', function: 'extractOneKE' }
      );
      throw error;
    }
  },

  /**
   * Index a project directory
   */
  indexProject: async (data: { project_path?: string; target_structure?: string; output_dir?: string }) => {
    try {
      return await apiClient.post<{ message: string; results: any }>('/api/openevolve/project/index', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Index project API error'),
        'error',
        { component: 'knowledgeApi', function: 'indexProject' }
      );
      throw error;
    }
  },

  /**
   * RAG Search
   */
  searchRag: async (data: { query: string }) => {
    try {
      return await apiClient.post<any[]>('/api/openevolve/rag/search', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('RAG search API error'),
        'error',
        { component: 'knowledgeApi', function: 'searchRag' }
      );
      throw error;
    }
  },

  /**
   * Graphiti Temporal Search
   */
  searchGraphiti: async (data: { query: string; num_results?: number }) => {
    try {
      return await apiClient.post<{
        nodes: any[];
        edges: any[];
        timestamp: string;
      }>('/api/openevolve/graphiti/search', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Graphiti search API error'),
        'error',
        { component: 'knowledgeApi', function: 'searchGraphiti' }
      );
      throw error;
    }
  },

  /**
   * Add knowledge to the graph
   */
  add: async (data: { entities?: any[]; relationships?: any[]; content?: string }) => {
    try {
      return await apiClient.post<{ message: string; items_processed: number }>('/api/openevolve/knowledge/add', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Add knowledge API error'),
        'error',
        { component: 'knowledgeApi', function: 'add' }
      );
      throw error;
    }
  },

  /**
   * List knowledge entities
   */
  list: async (params?: { limit?: number; offset?: number }) => {
    try {
      return await apiClient.get<{ entities: any[]; relationships: any[]; total_entities: number }>('/api/openevolve/knowledge/list', params);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('List knowledge entities API error'),
        'error',
        { component: 'knowledgeApi', function: 'list', additionalData: { params } }
      );
      throw error;
    }
  },

  /**
   * Get knowledge statistics
   */
  getStatistics: async () => {
    try {
      return await apiClient.get<{ entity_count: number; relationship_count: number }>('/api/openevolve/knowledge/statistics');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get knowledge statistics API error'),
        'error',
        { component: 'knowledgeApi', function: 'getStatistics' }
      );
      throw error;
    }
  },

  /**
   * Get specific entity
   */
  getEntity: async (entityId: string) => {
    try {
      return await apiClient.get<{ id: string; properties: any }>(`/api/openevolve/knowledge/entity/${entityId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get entity API error'),
        'error',
        { component: 'knowledgeApi', function: 'getEntity', additionalData: { entityId } }
      );
      throw error;
    }
  },

  /**
   * Ingest a document from path or URL
   */
  ingestDocument: async (data: { path_or_url: string }) => {
    try {
      return await apiClient.post<{ message: string; content_length: number }>('/api/openevolve/knowledge/ingest/document', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Ingest document API error'),
        'error',
        { component: 'knowledgeApi', function: 'ingestDocument' }
      );
      throw error;
    }
  },

  /**
   * Generate knowledge from context and query
   */
  generateKnowledge: async (data: { context: string; query: string }) => {
    try {
      return await apiClient.post<{ generated_knowledge: string }>('/api/openevolve/knowledge/generate', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Generate knowledge API error'),
        'error',
        { component: 'knowledgeApi', function: 'generateKnowledge' }
      );
      throw error;
    }
  },

  /**
   * Perform unified search (Facts + Skills + RAG)
   */
  unifiedSearch: async (query: string, top_k?: number) => {
    try {
      return await apiClient.get<any>('/api/openevolve/knowledge/unified-search', { params: { query, top_k } });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Unified search API error'),
        'error',
        { component: 'knowledgeApi', function: 'unifiedSearch', additionalData: { query } }
      );
      throw error;
    }
  },

  /**
   * Distill a reusable skill from an artifact
   */
  distillSkill: async (artifactId: string) => {
    try {
      return await apiClient.post<{ success: boolean; artifact_id: string }>(`/api/openevolve/knowledge/distill-skill/${artifactId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Distill skill API error'),
        'error',
        { component: 'knowledgeApi', function: 'distillSkill', additionalData: { artifactId } }
      );
      throw error;
    }
  },

  /**
   * Trigger autonomous self-healing of the knowledge base
   */
  selfHeal: async () => {
    try {
      return await apiClient.post<any>('/api/openevolve/knowledge/self-heal');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Self heal API error'),
        'error',
        { component: 'knowledgeApi', function: 'selfHeal' }
      );
      throw error;
    }
  },

  /**
   * Trigger recursive knowledge synthesis (Meta-Nodes)
   */
  synthesize: async () => {
    try {
      return await apiClient.post<any>('/api/openevolve/knowledge/synthesize');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Synthesize API error'),
        'error',
        { component: 'knowledgeApi', function: 'synthesize' }
      );
      throw error;
    }
  },

  /**
   * Perform deep multi-agent research
   */
  deepResearch: async (topic: string) => {
    try {
      return await apiClient.post<any>('/api/openevolve/knowledge/deep-research', { topic });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Deep research API error'),
        'error',
        { component: 'knowledgeApi', function: 'deepResearch', additionalData: { topic } }
      );
      throw error;
    }
  },

  /**
   * Formalize and verify a fact using LeanAide
   */
  verifyFact: async (text: string) => {
    try {
      return await apiClient.post<any>('/api/openevolve/knowledge/verify-fact', { text });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Verify fact API error'),
        'error',
        { component: 'knowledgeApi', function: 'verifyFact', additionalData: { text } }
      );
      throw error;
    }
  },

  /**
   * Search similar solutions using RAGBits
   */
  searchSolutions: async (data: { query: string; top_k?: number; filters?: any }) => {
    try {
      return await apiClient.post<any[]>('/api/openevolve/rag/solutions', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Search solutions API error'),
        'error',
        { component: 'knowledgeApi', function: 'searchSolutions' }
      );
      throw error;
    }
  },

  /**
   * Search decomposition patterns
   */
  searchPatterns: async (data: { query: string; pattern_type?: string }) => {
    try {
      return await apiClient.post<any[]>('/api/openevolve/rag/patterns', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Search patterns API error'),
        'error',
        { component: 'knowledgeApi', function: 'searchPatterns' }
      );
      throw error;
    }
  },

  /**
   * Fetch formal Lean 4 theorems
   */
  getLean4Theorems: async () => {
    try {
      return await apiClient.get<any[]>('/api/openevolve/mathematics/lean4');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get Lean4 theorems API error'),
        'error',
        { component: 'knowledgeApi', function: 'getLean4Theorems' }
      );
      throw error;
    }
  },

  /**
   * ACE (Agentic Context Engine) API
   */
  ace: {
    getStatus: async () => {
      try {
        return await apiClient.get<any>('/api/openevolve/ace/status');
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error('ACE get status API error'),
          'error',
          { component: 'knowledgeApi', function: 'ace.getStatus' }
        );
        throw error;
      }
    },
    getSkills: async (agentId?: string) => {
      try {
        return await apiClient.get<any>('/api/openevolve/ace/skills', { params: { agent_id: agentId } });
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error('ACE get skills API error'),
          'error',
          { component: 'knowledgeApi', function: 'ace.getSkills', additionalData: { agentId } }
        );
        throw error;
      }
    },
  },

  /**
   * Fetch red team attack patterns
   */
  getRedTeamAttacks: async () => {
    try {
      return await apiClient.get<any>('/api/openevolve/security/red-team');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get red team attacks API error'),
        'error',
        { component: 'knowledgeApi', function: 'getRedTeamAttacks' }
      );
      throw error;
    }
  },

  /**
   * Fetch blue team defense patterns
   */
  getBlueTeamDefenses: async () => {
    try {
      return await apiClient.get<any>('/api/openevolve/security/blue-team');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Get blue team defenses API error'),
        'error',
        { component: 'knowledgeApi', function: 'getBlueTeamDefenses' }
      );
      throw error;
    }
  },

  /**
   * Analyze knowledge graph with advanced algorithms (Karate Club)
   */
  analyze: async (data?: { graph_data?: any }) => {
    try {
      return await apiClient.post<any>('/api/openevolve/knowledge/analyze', data || {});
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Analyze knowledge graph API error'),
        'error',
        { component: 'knowledgeApi', function: 'analyze' }
      );
      throw error;
    }
  },

  /**
   * Update entity
   */
  updateEntity: async (entityId: string, data: { properties: any }) => {
    try {
      return await apiClient.put<{ message: string; id: string }>(`/api/openevolve/knowledge/entity/${entityId}`, data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Update entity API error'),
        'error',
        { component: 'knowledgeApi', function: 'updateEntity', additionalData: { entityId } }
      );
      throw error;
    }
  },

  /**
   * Delete entity
   */
  deleteEntity: async (entityId: string) => {
    try {
      return await apiClient.delete<{ message: string; id: string }>(`/api/openevolve/knowledge/entity/${entityId}`);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Delete entity API error'),
        'error',
        { component: 'knowledgeApi', function: 'deleteEntity', additionalData: { entityId } }
      );
      throw error;
    }
  },
};

/**
 * Invention Endpoints
 */
export const inventionApi = {
  /**
   * Create an invention plan
   */
  createPlan: async (data: {
    goal: string;
    domain: string;
    innovativeness: number;
    planning_stages: string[];
    constraints?: string;
    target_audience?: string;
    include_prior_art: boolean;
    include_feasibility: boolean;
    include_roadmap: boolean;
    detail_level: string;
  }) => {
    try {
      return await apiClient.post<any>('/api/openevolve/invention/plan', data);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error('Create invention plan API error'),
        'error',
        { component: 'inventionApi', function: 'createPlan' }
      );
      throw error;
    }
  },
};

/**
 * Export all APIs
 */
export const api = {
  auth: authApi,
  user: userApi,
  evolution: evolutionApi,
  adversarial: adversarialApi,
  analytics: analyticsApi,
  monitoring: monitoringApi,
  content: contentApi,
  version: versionApi,
  collaboration: collaborationApi,
  comments: commentsApi,
  config: configApi,
  workflow: workflowApi,
  files: filesApi,
  leanaide: leanaideApi,
  knowledge: knowledgeApi,
  invention: inventionApi,
};

// Export individual APIs for tree-shaking
export { authApi as auth };
export { userApi as user };
export { evolutionApi as evolution };
export { adversarialApi as adversarial };
export { analyticsApi as analytics };
export { monitoringApi as monitoring };
export { contentApi as content };
export { versionApi as version };
export { collaborationApi as collaboration };
export { commentsApi as comments };
export { configApi as config };
export { workflowApi as workflow };
export { filesApi as files };
export { leanaideApi as leanaide };
export { knowledgeApi as knowledge };
export { inventionApi as invention };
