/**
 * OpenEvolve Type Definitions
 *
 * Type definitions for OpenEvolve workflow parameters and responses
 * Matches Pydantic models in BubbleLab/services/openevolve-api/models/__init__.py
 */

// ==================== Evolution Parameters ====================

/**
 * Evolution workflow parameters
 */
export interface EvolutionParameters {
  max_iterations: number;        // Default: 100, Range: 1-200
  population_size: number;        // Default: 50, Range: 1-100
  temperature: number;            // Default: 0.7, Range: 0.0-2.0
  top_p: number;                  // Default: 1.0, Range: 0.0-1.0
  max_tokens: number;             // Default: 4096, Range: 1-100000
  frequency_penalty: number;      // Default: 0.0, Range: -2.0 to 2.0
  presence_penalty: number;       // Default: 0.0, Range: -2.0 to 2.0
  seed: number;                   // Default: 42, Range: -1 to 999999 (-1 for random)
}

// ==================== Adversarial Parameters ====================

/**
 * Adversarial testing workflow parameters
 */
export interface AdversarialParameters {
  test_cases: string[];           // Test cases for adversarial evaluation
  attack_types: string[];         // Types of attacks to test (default: ["fuzzing", "prompt_injection", "code_injection"])
  rounds: number;                 // Number of testing rounds, Default: 3, Range: 1-10
}

// ==================== Sovereign Parameters ====================

/**
 * Sovereign decomposition workflow parameters
 */
export interface SovereignParameters {
  decomposition_depth: number;    // Max depth of problem decomposition, Default: 3, Range: 1-10
  parallel_subproblems: number;   // Number of sub-problems to solve in parallel, Default: 5, Range: 1-20
  verification_strictness: 'lenient' | 'standard' | 'strict';  // Default: "standard"
}

// ==================== Workflow Types ====================

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

// ==================== Workflow Metadata ====================

export interface WorkflowMetadata {
  mdap_enabled?: boolean;
  maker_enabled?: boolean;
  maker_config?: Record<string, unknown>;
  adaptive_config?: Record<string, unknown>;
  evolution_params?: Record<string, unknown>;
  performance_params?: Record<string, unknown>;
}

// ==================== Common Interfaces ====================

/**
 * Base workflow interface
 */
export interface WorkflowBase {
  name: string;
  description: string;
  workflow_type: WorkflowType;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: WorkflowMetadata | null;
  parameters?: Record<string, unknown>;
}

/**
 * Workflow creation request
 */
export interface WorkflowCreate extends WorkflowBase {
  parameters?: Record<string, unknown>;
}

/**
 * Workflow update request
 */
export interface WorkflowUpdate {
  name?: string;
  description?: string;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: WorkflowMetadata | null;
  parameters?: Record<string, unknown>;
}

/**
 * Complete workflow response
 */
export interface WorkflowResponse extends WorkflowBase {
  id: string;
  parameters: Record<string, unknown>;
  status: WorkflowStatus;
  created_at: string;             // ISO 8601 datetime
  updated_at: string;             // ISO 8601 datetime
  started_at?: string | null;
  completed_at?: string | null;
  user_id?: string;
  tenant_id?: string;
}

/**
 * Workflow list response
 */
export interface WorkflowListResponse {
  workflows: WorkflowResponse[];
  total: number;
  page: number;
  page_size: number;
}

// ==================== Execution Types ====================

/**
 * Inputs for workflow execution
 */
export interface WorkflowInputs {
  problem_statement: string;
  context?: string;
}

/**
 * Execution response
 */
export interface ExecutionResponse {
  execution_id: string;
  workflow_id: string;
  status: ExecutionStatus;
  progress: number;               // Range: 0.0 to 1.0
  started_at?: string;
  completed_at?: string;
  result?: Record<string, unknown>;
  error?: string;
}

/**
 * Execution logs response
 */
export interface ExecutionLogsResponse {
  logs: Array<Record<string, unknown>>;
  total: number;
  since?: string;
}

// ==================== Team Types ====================

/**
 * Team member definition
 */
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

/**
 * Team creation request
 */
export interface TeamCreate {
  name: string;
  description: string;
  members: TeamMember[];
}

/**
 * Team response
 */
export interface TeamResponse {
  id: string;
  name: string;
  description: string;
  members: TeamMember[];
  created_at: string;
}

/**
 * Team list response
 */
export interface TeamListResponse {
  teams: TeamResponse[];
  total: number;
}

// ==================== Gauntlet Types ====================

/**
 * Gauntlet round definition
 */
export interface GauntletRound {
  name: string;
  quorum_threshold: number;       // Range: 0.0 to 1.0
  confidence_threshold: number;   // Range: 0.0 to 1.0
  evaluation_type: string;
}

/**
 * Gauntlet creation request
 */
export interface GauntletCreate {
  name: string;
  description: string;
  rounds: GauntletRound[];
}

/**
 * Gauntlet response
 */
export interface GauntletResponse {
  id: string;
  name: string;
  description: string;
  rounds: GauntletRound[];
  created_at: string;
}

/**
 * Gauntlet list response
 */
export interface GauntletListResponse {
  gauntlets: GauntletResponse[];
  total: number;
}

// ==================== Health & Status ====================

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  features: {
    evolution: boolean;
    adversarial: boolean;
    sovereign: boolean;
  };
}

// ==================== Default Values ====================

/**
 * Default evolution parameters
 */
export const DEFAULT_EVOLUTION_PARAMETERS: EvolutionParameters = {
  max_iterations: 100,
  population_size: 50,
  temperature: 0.7,
  top_p: 1.0,
  max_tokens: 4096,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  seed: 42,
};

/**
 * Default adversarial parameters
 */
export const DEFAULT_ADVERSARIAL_PARAMETERS: AdversarialParameters = {
  test_cases: [],
  attack_types: ['fuzzing', 'prompt_injection', 'code_injection'],
  rounds: 3,
};

/**
 * Default sovereign parameters
 */
export const DEFAULT_SOVEREIGN_PARAMETERS: SovereignParameters = {
  decomposition_depth: 3,
  parallel_subproblems: 5,
  verification_strictness: 'standard',
};
