/**
 * OpenEvolve API Client
 *
 * Connects BubbleLab frontend to the new OpenEvolve FastAPI service
 * Replaces the old Streamlit-based integration
 *
 * Architecture: Service-Oriented Architecture
 * - Eliminates parameter sync issues
 * - Clean REST API interface
 * - Full OpenAPI documentation at /docs
 *
 * @see BubbleLab/services/openevolve-api/README.md
 */

import { ApiClient, ApiClientConfig } from '@/lib/api';
import { OPENEVOLVE_API_BASE_URL } from '@/env';
import { logger } from '@/utils/logger';
import type {
  EvolutionParameters,
  AdversarialParameters,
  SovereignParameters,
} from '@/types/openevolve';

// Create ApiClient with retry and timeout configuration
const openevolveClientConfig: ApiClientConfig = {
  baseURL: OPENEVOLVE_API_BASE_URL,
  timeout: 60000,      // 60 seconds (evolution can take longer)
  enableRetry: true,   // Enable retry logic
  maxRetries: 3,       // Maximum 3 retries
  retryDelay: 2000,    // Base delay 2 seconds
};

const openevolveApiClient = new ApiClient(openevolveClientConfig);

// ==================== Types ====================

export type WorkflowType = 'evolution' | 'adversarial' | 'sovereign';
export type WorkflowStatus =
  | 'created'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'draft'
  | 'ready'
  | 'archived';
export type ExecutionStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface WorkflowCreate {
  name: string;
  description: string;
  workflow_type: WorkflowType;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
}

export interface WorkflowResponse {
  id: string;
  name: string;
  description: string;
  workflow_type: WorkflowType;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: Record<string, unknown>;
  parameters: Record<string, unknown>;
  status: WorkflowStatus;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  user_id?: string;
  tenant_id?: string;
}

export interface WorkflowListResponse {
  workflows: WorkflowResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface WorkflowInputs {
  problem_statement: string;
  context?: string;
}

export interface ExecutionResponse {
  execution_id: string;
  workflow_id: string;
  status: ExecutionStatus;
  progress: number;
  started_at?: string;
  completed_at?: string;
  result?: Record<string, unknown>;
  error?: string;
}

export interface ExecutionLogsResponse {
  logs: Array<Record<string, unknown>>;
  total: number;
  since?: string;
}

export interface TeamMember {
  id?: string;
  name: string;
  role: string;
  model: string;
  temperature: number;
  max_tokens: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  max_iterations?: number;
}

export interface TeamCreate {
  name: string;
  description: string;
  members: TeamMember[];
}

export interface TeamResponse {
  id: string;
  name: string;
  description: string;
  members: TeamMember[];
  created_at: string;
}

export interface GauntletRound {
  name: string;
  quorum_threshold: number;
  confidence_threshold: number;
  evaluation_type: string;
}

export interface GauntletCreate {
  name: string;
  description: string;
  rounds: GauntletRound[];
}

export interface GauntletResponse {
  id: string;
  name: string;
  description: string;
  rounds: GauntletRound[];
  created_at: string;
}

export interface BubbleLabsControlCatalogResponse {
  success: boolean;
  components: Record<string, string[]>;
  auto_discovery?: {
    enabled?: boolean;
    summary?: Record<string, unknown>;
    components?: Record<string, string[]>;
  };
}

export interface BubbleLabsControlDiscoveryResponse {
  success: boolean;
  discovered_components?: number;
  discovered_actions?: number;
  scanned_paths?: string[];
  indexed_components?: number;
  [key: string]: unknown;
}

export interface BubbleLabsControlExecuteResponse {
  success: boolean;
  component?: string;
  action?: string;
  result?: Record<string, unknown>;
  error?: string;
  [key: string]: unknown;
}

// ==================== API Client ====================

/**
 * OpenEvolve API Client
 *
 * Provides full access to OpenEvolve workflow execution capabilities
 */
export const openevolveApi = {
  // ==================== Health & Info ====================

  /**
   * Check API health
   */
  health: async (): Promise<{
    status: string;
    service: string;
    version: string;
    features: {
      evolution: boolean;
      adversarial: boolean;
      sovereign: boolean;
    };
  }> => {
    logger.debug({
      msg: 'Checking OpenEvolve API health',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get('/health');
  },

  // ==================== Workflows ====================

  /**
   * Create a new workflow
   */
  createWorkflow: async (workflow: WorkflowCreate): Promise<WorkflowResponse> => {
    logger.debug({
      msg: 'Creating workflow',
      component: 'openevolveApi',
      workflow_type: workflow.workflow_type,
      name: workflow.name,
    });

    return openevolveApiClient.post<WorkflowResponse>('/api/workflows', workflow);
  },

  /**
   * Get all workflows
   */
  listWorkflows: async (page = 1, pageSize = 10): Promise<WorkflowListResponse> => {
    logger.debug({
      msg: 'Listing workflows',
      component: 'openevolveApi',
      page,
      page_size: pageSize,
    });

    return openevolveApiClient.get<WorkflowListResponse>(
      `/api/workflows?page=${page}&page_size=${pageSize}`
    );
  },

  /**
   * Get a specific workflow
   */
  getWorkflow: async (workflowId: string): Promise<WorkflowResponse> => {
    logger.debug({
      msg: 'Getting workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.get<WorkflowResponse>(`/api/workflows/${workflowId}`);
  },

  /**
   * Update a workflow
   */
  updateWorkflow: async (
    workflowId: string,
    updates: Partial<Pick<WorkflowCreate, 'name' | 'description' | 'parameters'>>
  ): Promise<WorkflowResponse> => {
    logger.debug({
      msg: 'Updating workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.put<WorkflowResponse>(
      `/api/workflows/${workflowId}`,
      updates
    );
  },

  /**
   * Delete a workflow
   */
  deleteWorkflow: async (workflowId: string): Promise<{ message: string }> => {
    logger.debug({
      msg: 'Deleting workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.delete<{ message: string }>(`/api/workflows/${workflowId}`);
  },

  // ==================== Execution ====================

  /**
   * Execute a workflow
   */
  executeWorkflow: async (
    workflowId: string,
    inputs: WorkflowInputs
  ): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Executing workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
      problem_statement_length: inputs.problem_statement.length,
    });

    return openevolveApiClient.post<ExecutionResponse>(
      `/api/executions`,
      {
        workflow_id: workflowId,
        ...inputs,
      }
    );
  },

  /**
   * Get execution status
   */
  getExecutionStatus: async (executionId: string): Promise<ExecutionResponse> => {
    logger.debug({
      msg: 'Getting execution status',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return openevolveApiClient.get<ExecutionResponse>(`/api/executions/${executionId}`);
  },

  /**
   * Pause an execution
   */
  pauseExecution: async (executionId: string): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Pausing execution',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return openevolveApiClient.post<ExecutionResponse>(
      `/api/executions/${executionId}/pause`,
      {}
    );
  },

  /**
   * Resume a paused execution
   */
  resumeExecution: async (executionId: string): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Resuming execution',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return openevolveApiClient.post<ExecutionResponse>(
      `/api/executions/${executionId}/resume`,
      {}
    );
  },

  /**
   * Cancel an execution
   */
  cancelExecution: async (executionId: string): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Cancelling execution',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return openevolveApiClient.post<ExecutionResponse>(
      `/api/executions/${executionId}/cancel`,
      {}
    );
  },

  /**
   * Get execution logs
   */
  getExecutionLogs: async (
    executionId: string,
    since?: string
  ): Promise<ExecutionLogsResponse> => {
    logger.debug({
      msg: 'Getting execution logs',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    const url = since
      ? `/api/executions/${executionId}/logs?since=${since}`
      : `/api/executions/${executionId}/logs`;

    return openevolveApiClient.get<ExecutionLogsResponse>(url);
  },

  // ==================== Teams ====================

  /**
   * Create a team
   */
  createTeam: async (team: TeamCreate): Promise<TeamResponse> => {
    logger.debug({
      msg: 'Creating team',
      component: 'openevolveApi',
      name: team.name,
      member_count: team.members.length,
    });

    return openevolveApiClient.post<TeamResponse>('/api/teams', team);
  },

  /**
   * Get all teams
   */
  listTeams: async (): Promise<{ teams: TeamResponse[]; total: number }> => {
    logger.debug({
      msg: 'Listing teams',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ teams: TeamResponse[]; total: number }>('/api/teams');
  },

  /**
   * Get a specific team
   */
  getTeam: async (teamId: string): Promise<TeamResponse> => {
    logger.debug({
      msg: 'Getting team',
      component: 'openevolveApi',
      team_id: teamId,
    });

    return openevolveApiClient.get<TeamResponse>(`/api/teams/${teamId}`);
  },

  // ==================== Gauntlets ====================

  /**
   * Create a gauntlet
   */
  createGauntlet: async (gauntlet: GauntletCreate): Promise<GauntletResponse> => {
    logger.debug({
      msg: 'Creating gauntlet',
      component: 'openevolveApi',
      name: gauntlet.name,
      rounds_count: gauntlet.rounds.length,
    });

    return openevolveApiClient.post<GauntletResponse>('/api/gauntlets', gauntlet);
  },

  /**
   * Get all gauntlets
   */
  listGauntlets: async (): Promise<{ gauntlets: GauntletResponse[]; total: number }> => {
    logger.debug({
      msg: 'Listing gauntlets',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ gauntlets: GauntletResponse[]; total: number }>(
      '/api/gauntlets'
    );
  },

  /**
   * Get a specific gauntlet
   */
  getGauntlet: async (gauntletId: string): Promise<GauntletResponse> => {
    logger.debug({
      msg: 'Getting gauntlet',
      component: 'openevolveApi',
      gauntlet_id: gauntletId,
    });

    return openevolveApiClient.get<GauntletResponse>(`/api/gauntlets/${gauntletId}`);
  },

  // ==================== BubbleLabs Control Plane ====================

  /**
   * Get unified BubbleLabs/OpenEvolve control catalog.
   */
  getControlCatalog: async (): Promise<BubbleLabsControlCatalogResponse> => {
    logger.debug({
      msg: 'Fetching BubbleLabs control catalog',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<BubbleLabsControlCatalogResponse>(
      '/bubblelabs/control/catalog'
    );
  },

  /**
   * Refresh auto-discovered control components.
   */
  discoverControlComponents: async (
    force = false
  ): Promise<BubbleLabsControlDiscoveryResponse> => {
    logger.info({
      msg: 'Refreshing BubbleLabs control discovery',
      component: 'openevolveApi',
      force,
    });

    return openevolveApiClient.post<BubbleLabsControlDiscoveryResponse>(
      '/bubblelabs/control/discover',
      { force }
    );
  },

  /**
   * Execute a control action for a component.
   */
  executeControlAction: async (
    component: string,
    action: string,
    payload: Record<string, unknown> = {}
  ): Promise<BubbleLabsControlExecuteResponse> => {
    logger.info({
      msg: 'Executing BubbleLabs control action',
      component: 'openevolveApi',
      control_component: component,
      control_action: action,
    });

    return openevolveApiClient.post<BubbleLabsControlExecuteResponse>(
      '/bubblelabs/control/execute',
      {
        component,
        action,
        payload,
      }
    );
  },
};

// ==================== Convenience Functions ====================

/**
 * Quick evolution workflow execution
 *
 * Creates an evolution workflow and executes it in one call
 */
export const executeEvolution = async (
  problemStatement: string,
  parameters?: Partial<EvolutionParameters>,
  context?: string
): Promise<ExecutionResponse> => {
  logger.info({
    msg: 'Executing quick evolution',
    component: 'openevolveApi',
    problem_statement_length: problemStatement.length,
  });

  // Create workflow
  const workflow = await openevolveApi.createWorkflow({
    name: 'Quick Evolution',
    description: 'Auto-generated evolution workflow',
    workflow_type: 'evolution',
    parameters: parameters || {},
  });

  // Execute workflow
  return openevolveApi.executeWorkflow(workflow.id, {
    problem_statement: problemStatement,
    context,
  });
};

/**
 * Quick adversarial workflow execution
 */
export const executeAdversarial = async (
  problemStatement: string,
  parameters?: Partial<AdversarialParameters>,
  context?: string
): Promise<ExecutionResponse> => {
  logger.info({
    msg: 'Executing quick adversarial',
    component: 'openevolveApi',
    problem_statement_length: problemStatement.length,
  });

  const workflow = await openevolveApi.createWorkflow({
    name: 'Quick Adversarial',
    description: 'Auto-generated adversarial workflow',
    workflow_type: 'adversarial',
    parameters: parameters || {},
  });

  return openevolveApi.executeWorkflow(workflow.id, {
    problem_statement: problemStatement,
    context,
  });
};

/**
 * Quick sovereign workflow execution
 */
export const executeSovereign = async (
  problemStatement: string,
  parameters?: Partial<SovereignParameters>,
  context?: string
): Promise<ExecutionResponse> => {
  logger.info({
    msg: 'Executing quick sovereign',
    component: 'openevolveApi',
    problem_statement_length: problemStatement.length,
  });

  const workflow = await openevolveApi.createWorkflow({
    name: 'Quick Sovereign',
    description: 'Auto-generated sovereign workflow',
    workflow_type: 'sovereign',
    parameters: parameters || {},
  });

  return openevolveApi.executeWorkflow(workflow.id, {
    problem_statement: problemStatement,
    context,
  });
};
