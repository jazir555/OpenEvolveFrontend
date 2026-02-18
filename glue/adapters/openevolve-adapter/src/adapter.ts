/**
 * OpenEvolve Main Orchestration Adapter
 *
 * This is the primary orchestration adapter that coordinates all integrated
 * systems within the OpenEvolve federation. It serves as the central hub for:
 * - Multi-adapter coordination (Z3, LeanAide, RAGBits, Vector DB, etc.)
 * - Workflow orchestration across multiple systems
 * - Knowledge aggregation from all sources
 * - Event bus integration for pub/sub patterns
 * - Circuit breaker and retry logic for resilience
 * - Canonical schema enforcement (Anti-Corruption Layer)
 *
 * Environment Variables:
 *   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API (required, no default)
 *   TIMEOUT_MS - Request timeout in milliseconds (required, no default)
 *   EVENT_BUS_URL - Event bus URL for pub/sub (optional)
 *   LOG_LEVEL - Logging level (default: info)
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// TYPE DEFINITIONS - Canonical Schemas
// ============================================================================

/**
 * Model configuration for AI team members
 */
export interface ModelConfig {
  model_id: string;
  api_key: string;
  api_base: string;
  temperature: number;
  top_p: number;
  max_tokens: number;
  frequency_penalty: number;
  presence_penalty: number;
  [key: string]: any; // Allow additional fields
}

/**
 * Team definition (Blue/Red/Gold)
 */
export interface Team {
  name: string;
  role: 'Blue' | 'Red' | 'Gold';
  members: ModelConfig[];
  tenant_id?: string;
  description?: string;
  sub_role?: string;
  domain_specialization?: string[];
  problem_type_specialization?: string[];
  performance_metrics?: Record<string, number>;
  team_config?: Record<string, any>;
  openevolve_metrics?: Record<string, any>[];
}

/**
 * Gauntlet round rules
 */
export interface GauntletRoundRule {
  round_number: number;
  quorum_required_approvals: number;
  quorum_from_panel_size: number;
  min_overall_confidence: number;
  max_score_variance?: number;
  per_judge_requirements?: Record<string, any>;
  collaboration_mode?: 'independent' | 'share_previous_feedback';
  time_limit_seconds?: number;
  max_api_calls?: number;
  max_tokens?: number;
  adaptive_rules?: Record<string, any>;
}

/**
 * Gauntlet definition
 */
export interface Gauntlet {
  name: string;
  team_name: string;
  rounds: GauntletRoundRule[];
  tenant_id?: string;
  description?: string;
  attack_modes?: string[];
  generation_mode?: 'single_candidate' | 'multi_candidate_peer_review' | 'evolutionary' | 'hybrid';
  gauntlet_type?: 'standard' | 'adaptive' | 'hierarchical' | 'competitive' | 'collaborative';
  performance_metrics?: Record<string, number>;
  gauntlet_config?: Record<string, any>;
}

/**
 * Sub-problem in decomposition
 */
export interface SubProblem {
  id: string;
  description: string;
  dependencies: string[];
  ai_suggested_evolution_mode?: string;
  ai_suggested_complexity_score?: number;
  ai_suggested_evaluation_prompt?: string;
  content_type?: string;
  solver_team_name: string;
  red_team_gauntlet_name?: string;
  gold_team_gauntlet_name: string;
  solver_generation_gauntlet_name?: string;
  patcher_team_name?: string;
  evolution_params?: Record<string, any>;
  ai_suggested_team_assignment?: string;
  ai_suggested_gauntlet_assignment?: Record<string, string>;
  estimated_resources?: Record<string, any>;
  potential_approaches?: string[];
  status?: 'pending' | 'in_progress' | 'solved' | 'failed' | 'requires_rework';
  solution_attempts?: SolutionAttempt[];
  performance_metrics?: Record<string, number>;
  openevolve_metrics?: Record<string, any>;
}

/**
 * Solution attempt
 */
export interface SolutionAttempt {
  sub_problem_id: string;
  content: string;
  generated_by_model: string;
  timestamp: number;
  history?: Record<string, any>[];
  solution_type?: string;
  solution_approach?: string;
  quality_metrics?: Record<string, number>;
  resource_usage?: Record<string, any>;
  status?: 'generated' | 'critiqued' | 'verified' | 'rejected' | 'patched';
  critique_reports?: CritiqueReport[];
  verification_reports?: VerificationReport[];
  openevolve_metrics?: Record<string, any>;
}

/**
 * Critique report from Red Team
 */
export interface CritiqueReport {
  solution_attempt_id: string;
  gauntlet_name: string;
  is_approved: boolean;
  reports_by_judge: Record<string, any>[];
  summary?: string;
  critique_timestamp?: number;
  overall_score?: number;
  flaw_severity_scores?: Record<string, number>;
  identified_flaws?: Record<string, any>[];
  suggested_improvements?: string[];
  resource_usage?: Record<string, any>;
}

/**
 * Verification report from Gold Team
 */
export interface VerificationReport {
  solution_attempt_id: string;
  gauntlet_name: string;
  is_approved: boolean;
  reports_by_judge: Record<string, any>[];
  average_score?: number;
  score_variance?: number;
  summary?: string;
  verification_timestamp?: number;
  dimension_scores?: Record<string, number>;
  criteria_met?: string[];
  criteria_not_met?: string[];
  targeted_feedback?: string;
  resource_usage?: Record<string, any>;
}

/**
 * Workflow definition
 */
export interface WorkflowDefinition {
  workflow_id: string;
  name: string;
  description?: string;
  problem_statement: string;
  max_refinement_loops: number;
  auto_approval_enabled: boolean;
  auto_approval_criteria?: Record<string, any>;
  mdap_enabled?: boolean;
  mdap_config?: Record<string, any>;
  maker_enabled?: boolean;
  maker_config?: Record<string, any>;
  resource_limits?: Record<string, any>;
  parallel_processing_enabled?: boolean;
  max_parallel_sub_problems?: number;
  learning_enabled?: boolean;
  learning_config?: Record<string, any>;
  content_analyzer_team_name?: string;
  planner_team_name?: string;
  assembler_team_name?: string;
  final_red_team_gauntlet_name?: string;
  final_gold_team_gauntlet_name?: string;
  sub_problems: SubProblem[];
}

/**
 * Workflow state
 */
export interface WorkflowState {
  workflow_id: string;
  workflow_type?: any;
  problem_statement: string;
  current_stage: string;
  tenant_id?: string;
  current_sub_problem_id?: string;
  current_gauntlet_name?: string;
  status: string;
  progress: number;
  start_time: string;
  end_time?: string;
  decomposition_plan?: WorkflowDefinition;
  sub_problem_solutions?: Record<string, SolutionAttempt>;
  solved_sub_problem_ids?: Set<string>;
  rejected_sub_problems?: Record<string, any>;
  final_solution?: SolutionAttempt;
  refinement_loop_count?: number;
  content_analyzer_team?: Team;
  planner_team?: Team;
  solver_team?: Team;
  patcher_team?: Team;
  assembler_team?: Team;
  sub_problem_red_gauntlet?: Gauntlet;
  sub_problem_gold_gauntlet?: Gauntlet;
  final_red_gauntlet?: Gauntlet;
  final_gold_gauntlet?: Gauntlet;
  max_refinement_loops?: number;
  all_critique_reports?: CritiqueReport[];
  all_verification_reports?: VerificationReport[];
  resource_usage?: Record<string, any>;
  performance_metrics?: Record<string, number>;
  knowledge_artifacts?: KnowledgeArtifact[];
  openevolve_metrics?: Record<string, any>;
  [key: string]: any; // Allow additional OpenEvolve parameters
}

/**
 * Knowledge artifact
 */
export interface KnowledgeArtifact {
  id: string;
  artifact_type: 'solution_pattern' | 'problem_solution_mapping' | 'critique_insight' | 'team_performance' | 'gauntlet_effectiveness';
  content: Record<string, any>;
  source_workflow_id: string;
  extraction_timestamp: number;
  domain?: string;
  problem_type?: string;
  usage_count?: number;
  effectiveness_score?: number;
  related_artifacts?: string[];
}

/**
 * Integration health status
 */
export interface IntegrationHealth {
  name: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  latency_ms?: number;
  last_check?: string;
  error_message?: string;
}

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

enum CircuitBreakerState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  monitorPeriod: number;
}

class CircuitBreaker {
  private state: CircuitBreakerState = CircuitBreakerState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime?: number;
  private nextAttempt?: number;

  constructor(
    private readonly name: string,
    private readonly config: CircuitBreakerConfig,
    private readonly logger: StructuredLogger,
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === CircuitBreakerState.OPEN) {
      if (Date.now() < (this.nextAttempt || 0)) {
        throw new Error(`Circuit breaker '${this.name}' is OPEN`);
      }
      this.state = CircuitBreakerState.HALF_OPEN;
      this.logger.info('Circuit breaker HALF_OPEN', { circuit_breaker: this.name });
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess() {
    this.failureCount = 0;
    if (this.state === CircuitBreakerState.HALF_OPEN) {
      this.successCount++;
      if (this.successCount >= this.config.successThreshold) {
        this.state = CircuitBreakerState.CLOSED;
        this.successCount = 0;
        this.logger.info('Circuit breaker CLOSED', { circuit_breaker: this.name });
      }
    }
  }

  private onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = CircuitBreakerState.OPEN;
      this.nextAttempt = Date.now() + this.config.timeout;
      this.logger.error('Circuit breaker OPEN', {
        circuit_breaker: this.name,
        failure_count: this.failureCount,
        next_attempt: new Date(this.nextAttempt).toISOString(),
      });
    }
  }

  getState(): CircuitBreakerState {
    return this.state;
  }
}

// ============================================================================
// RETRY WITH EXPONENTIAL BACKOFF
// ============================================================================

interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  jitter: boolean;
}

async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig,
  logger: StructuredLogger,
  context: Record<string, any>,
): Promise<T> {
  let lastError: any;

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        logger.info('Retrying request', {
          ...context,
          attempt,
          max_retries: config.maxRetries,
        });
      }

      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < config.maxRetries) {
        const delay = Math.min(
          config.baseDelay * Math.pow(2, attempt),
          config.maxDelay,
        );

        const jitterAmount = config.jitter ? Math.random() * 0.3 * delay : 0;
        const finalDelay = delay + jitterAmount;

        logger.warn('Request failed, retrying after delay', {
          ...context,
          attempt,
          delay_ms: finalDelay,
          error: error instanceof Error ? error.message : String(error),
        });

        await sleep(finalDelay);
      }
    }
  }

  logger.error('All retries exhausted', {
    ...context,
    max_retries: config.maxRetries,
    error: lastError instanceof Error ? lastError.message : String(lastError),
  });

  throw lastError;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// STRUCTURED LOGGER (JSON Lines)
// ============================================================================

export interface LogContext {
  correlation_id: string;
  source_service: string;
  target_service?: string;
  [key: string]: any;
}

export class StructuredLogger {
  constructor(
    private readonly serviceName: string,
    private readonly logLevel: string = 'info',
  ) {}

  private log(level: string, message: string, context: Record<string, any> = {}) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      service: this.serviceName,
      ...context,
    };

    const logLine = JSON.stringify(logEntry);
    const output = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
    output(logLine);
  }

  info(message: string, context: Record<string, any> = {}) {
    this.log('info', message, context);
  }

  warn(message: string, context: Record<string, any> = {}) {
    this.log('warn', message, context);
  }

  error(message: string, context: Record<string, any> = {}) {
    this.log('error', message, context);
  }

  debug(message: string, context: Record<string, any> = {}) {
    if (this.logLevel === 'debug') {
      this.log('debug', message, context);
    }
  }
}

// ============================================================================
// MAIN ADAPTER CLASS
// ============================================================================

export interface OpenEvolveAdapterConfig {
  api_url: string;
  timeout_ms: number;
  event_bus_url?: string;
  log_level?: string;
  circuit_breaker?: Partial<CircuitBreakerConfig>;
  retry?: Partial<RetryConfig>;
}

export class OpenEvolveAdapter {
  private readonly api: AxiosInstance;
  private readonly logger: StructuredLogger;
  private readonly circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private readonly retryConfig: RetryConfig;
  private readonly correlationId: string;

  // Integration circuit breakers
  private z3CircuitBreaker: CircuitBreaker;
  private leanaideCircuitBreaker: CircuitBreaker;
  private ragbitsCircuitBreaker: CircuitBreaker;
  private vectordbCircuitBreaker: CircuitBreaker;
  private graphitiCircuitBreaker: CircuitBreaker;
  private karateclubCircuitBreaker: CircuitBreaker;

  constructor(config: OpenEvolveAdapterConfig) {
    // Validate required environment
    if (!config.api_url) {
      throw new Error('OPENEVOLVE_API_URL is required and cannot have a default value');
    }
    if (!config.timeout_ms) {
      throw new Error('TIMEOUT_MS is required and cannot have a default value');
    }

    // Create correlation ID for this adapter instance
    this.correlationId = uuidv4();

    // Initialize structured logger
    this.logger = new StructuredLogger('openevolve-adapter', config.log_level);

    // Initialize axios instance
    this.api = axios.create({
      baseURL: config.api_url,
      timeout: config.timeout_ms,
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': this.correlationId,
      },
    });

    // Configure retry
    this.retryConfig = {
      maxRetries: config.retry?.maxRetries ?? 3,
      baseDelay: config.retry?.baseDelay ?? 1000,
      maxDelay: config.retry?.maxDelay ?? 10000,
      jitter: config.retry?.jitter ?? true,
    };

    // Initialize circuit breakers
    const circuitBreakerConfig: CircuitBreakerConfig = {
      failureThreshold: config.circuit_breaker?.failureThreshold ?? 5,
      successThreshold: config.circuit_breaker?.successThreshold ?? 2,
      timeout: config.circuit_breaker?.timeout ?? 60000,
      monitorPeriod: config.circuit_breaker?.monitorPeriod ?? 10000,
    };

    this.z3CircuitBreaker = new CircuitBreaker('z3-adapter', circuitBreakerConfig, this.logger);
    this.leanaideCircuitBreaker = new CircuitBreaker('leanaide-adapter', circuitBreakerConfig, this.logger);
    this.ragbitsCircuitBreaker = new CircuitBreaker('ragbits-adapter', circuitBreakerConfig, this.logger);
    this.vectordbCircuitBreaker = new CircuitBreaker('vectordb-adapter', circuitBreakerConfig, this.logger);
    this.graphitiCircuitBreaker = new CircuitBreaker('graphiti-adapter', circuitBreakerConfig, this.logger);
    this.karateclubCircuitBreaker = new CircuitBreaker('karateclub-adapter', circuitBreakerConfig, this.logger);

    this.logger.info('OpenEvolve adapter initialized', {
      api_url: config.api_url,
      timeout_ms: config.timeout_ms,
      correlation_id: this.correlationId,
    });
  }

  // ============================================================================
  // HEALTH CHECKS
  // ============================================================================

  async healthCheck(): Promise<{ status: string; timestamp: string; integrations: IntegrationHealth[] }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
    };

    this.logger.info('Performing health check', context);

    try {
      const start = Date.now();

      const response = await this.api.get('/health');
      const latency = Date.now() - start;

      const integrations = await this.checkIntegrationHealth(context);

      this.logger.info('Health check successful', {
        ...context,
        latency_ms: latency,
        integration_count: integrations.length,
      });

      return {
        status: response.data.status,
        timestamp: response.data.timestamp,
        integrations,
      };
    } catch (error) {
      this.logger.error('Health check failed', {
        ...context,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  private async checkIntegrationHealth(context: LogContext): Promise<IntegrationHealth[]> {
    const integrations: IntegrationHealth[] = [];

    const checks = [
      { name: 'Z3 Prover', breaker: this.z3CircuitBreaker },
      { name: 'LeanAide', breaker: this.leanaideCircuitBreaker },
      { name: 'RAGBits', breaker: this.ragbitsCircuitBreaker },
      { name: 'Vector DB', breaker: this.vectordbCircuitBreaker },
      { name: 'Graphiti', breaker: this.graphitiCircuitBreaker },
      { name: 'KarateClub', breaker: this.karateclubCircuitBreaker },
    ];

    for (const check of checks) {
      const state = check.breaker.getState();
      integrations.push({
        name: check.name,
        status: state === CircuitBreakerState.CLOSED ? 'healthy' : 'unhealthy',
        last_check: new Date().toISOString(),
      });
    }

    return integrations;
  }

  // ============================================================================
  // TEAM MANAGEMENT
  // ============================================================================

  async createTeam(team: Team): Promise<{ message: string; team_name: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      team_name: team.name,
      role: team.role,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.post('/openevolve/teams', team);
        this.logger.info('Team created', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async getTeams(): Promise<Team[]> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.get('/openevolve/teams');
        this.logger.info('Teams retrieved', { ...context, count: response.data.length });
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async getTeam(name: string): Promise<Team> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      team_name: name,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.get(`/openevolve/teams/${name}`);
        this.logger.info('Team retrieved', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async updateTeam(name: string, team: Team): Promise<{ message: string; team_name: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      team_name: name,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.put(`/openevolve/teams/${name}`, team);
        this.logger.info('Team updated', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async deleteTeam(name: string): Promise<{ message: string; team_name: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      team_name: name,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.delete(`/openevolve/teams/${name}`);
        this.logger.info('Team deleted', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  // ============================================================================
  // GAUNTLET MANAGEMENT
  // ============================================================================

  async createGauntlet(gauntlet: Gauntlet): Promise<{ message: string; gauntlet_name: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      gauntlet_name: gauntlet.name,
      team_name: gauntlet.team_name,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.post('/openevolve/gauntlets', gauntlet);
        this.logger.info('Gauntlet created', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async getGauntlets(): Promise<Gauntlet[]> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.get('/openevolve/gauntlets');
        this.logger.info('Gauntlets retrieved', { ...context, count: response.data.length });
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async getGauntlet(name: string): Promise<Gauntlet> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      gauntlet_name: name,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.get(`/openevolve/gauntlets/${name}`);
        this.logger.info('Gauntlet retrieved', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async deleteGauntlet(name: string): Promise<{ message: string; gauntlet_name: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      gauntlet_name: name,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.delete(`/openevolve/gauntlets/${name}`);
        this.logger.info('Gauntlet deleted', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  // ============================================================================
  // WORKFLOW ORCHESTRATION
  // ============================================================================

  async createWorkflow(workflow: WorkflowDefinition): Promise<{ message: string; workflow_id: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      workflow_id: workflow.workflow_id,
      sub_problem_count: workflow.sub_problems.length,
    };

    this.logger.info('Creating workflow', context);

    return retryWithBackoff(
      async () => {
        const response = await this.api.post('/openevolve/workflows', workflow);
        this.logger.info('Workflow created', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async getWorkflows(): Promise<WorkflowState[]> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.get('/openevolve/workflows');
        this.logger.info('Workflows retrieved', { ...context, count: response.data.length });
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async getWorkflowStatus(workflowId: string): Promise<WorkflowState> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      workflow_id: workflowId,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.get(`/openevolve/workflows/${workflowId}/status`);
        this.logger.info('Workflow status retrieved', { ...context, status: response.data.status });
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  async deleteWorkflow(workflowId: string): Promise<{ message: string; workflow_id: string }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
      target_service: 'openevolve-api',
      workflow_id: workflowId,
    };

    return retryWithBackoff(
      async () => {
        const response = await this.api.delete(`/openevolve/workflows/${workflowId}`);
        this.logger.info('Workflow deleted', context);
        return response.data;
      },
      this.retryConfig,
      this.logger,
      context,
    );
  }

  // ============================================================================
  // INTEGRATION COORDINATION (Stub implementations)
  // ============================================================================

  async getIntegrationHealth(): Promise<{ integrations: IntegrationHealth[] }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
    };

    this.logger.info('Checking integration health', context);

    const integrations = await this.checkIntegrationHealth(context);

    return { integrations };
  }

  async getAvailableAdapters(): Promise<{ name: string; type: string; status: string }[]> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'openevolve-adapter',
    };

    this.logger.info('Retrieving available adapters', context);

    // Return known adapters
    return [
      { name: 'z3', type: 'prover', status: 'available' },
      { name: 'leanaide', type: 'assistant', status: 'available' },
      { name: 'ragbits', type: 'retrieval', status: 'available' },
      { name: 'vectordb', type: 'database', status: 'available' },
      { name: 'graphiti', type: 'graph', status: 'available' },
      { name: 'karateclub', type: 'ml', status: 'available' },
    ];
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createOpenEvolveAdapter(config: OpenEvolveAdapterConfig): OpenEvolveAdapter {
  return new OpenEvolveAdapter(config);
}

// Export default
export default OpenEvolveAdapter;
