/**
 * OpenEvolve API Type Definitions
 * TypeScript types matching Python Pydantic models
 */

// ============================================================================
// Base Types
// ============================================================================

/**
 * API response wrapper
 */
export interface ApiResponse<T = any> {
  data: T;
  success: boolean;
  message?: string;
  error?: string;
}

/**
 * API error response
 */
export interface ApiError {
  error: string;
  detail?: string;
  status_code: number;
  timestamp: string;
}

// ============================================================================
// Workflow Types
// ============================================================================

/**
 * Workflow execution status
 */
export enum WorkflowStatus {
  CREATED = "created",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled"
}

/**
 * Workflow definition
 */
export interface Workflow {
  id: string;
  name: string;
  description?: string;
  problem_statement: string;
  content_type: string;
  teams: string[];
  gauntlets: string[];
  status: WorkflowStatus;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  user_id: string;
  tenant_id: string;
  metadata?: WorkflowMetadata;
}

/**
 * Workflow metadata
 */
export interface WorkflowMetadata {
  mdap_enabled?: boolean;
  maker_enabled?: boolean;
  evolution_params?: EvolutionParameters;
  performance_params?: PerformanceParameters;
}

/**
 * Evolution parameters
 */
export interface EvolutionParameters {
  population_size: number;
  num_islands: number;
  migration_interval: number;
  migration_rate: number;
  archive_size: number;
  elite_ratio: number;
  exploration_ratio: number;
  exploitation_ratio: number;
}

/**
 * Performance parameters
 */
export interface PerformanceParameters {
  max_evaluations: number;
  parallel_evaluations: number;
  timeout_seconds: number;
  memory_limit_mb: number;
  cpu_limit: number;
}

/**
 * Create workflow request
 */
export interface CreateWorkflowRequest {
  name: string;
  description?: string;
  problem_statement: string;
  content_type: string;
  teams: string[];
  gauntlets: string[];
  metadata?: WorkflowMetadata;
}

/**
 * Update workflow request
 */
export interface UpdateWorkflowRequest {
  name?: string;
  description?: string;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: WorkflowMetadata;
}

// ============================================================================
// Execution Types
// ============================================================================

/**
 * Execution event types
 */
export enum ExecutionEventType {
  STARTED = "execution.started",
  PROGRESS = "execution.progress",
  SUBPROBLEM_CREATED = "subproblem.created",
  SUBPROBLEM_COMPLETED = "subproblem.completed",
  BUBBLE_ACTIVATED = "bubble.activated",
  BUBBLE_COMPLETED = "bubble.completed",
  SOLUTION_FOUND = "solution.found",
  ERROR = "execution.error",
  COMPLETED = "execution.completed",
  PAUSED = "execution.paused",
  RESUMED = "execution.resumed"
}

/**
 * Execution event
 */
export interface ExecutionEvent {
  id: string;
  event: ExecutionEventType;
  data: any;
  timestamp: string;
  workflow_id: string;
}

/**
 * Execution progress event
 */
export interface ExecutionProgressEvent extends ExecutionEvent {
  event: ExecutionEventType.PROGRESS;
  data: {
    progress: number;
    current_step: string;
    total_steps: number;
    message?: string;
  };
}

/**
 * Sub-problem event
 */
export interface SubProblemEvent extends ExecutionEvent {
  event: ExecutionEventType.SUBPROBLEM_CREATED | ExecutionEventType.SUBPROBLEM_COMPLETED;
  data: {
    subproblem_id: string;
    problem: string;
    status: string;
    solution?: string;
  };
}

/**
 * Bubble event
 */
export interface BubbleEvent extends ExecutionEvent {
  event: ExecutionEventType.BUBBLE_ACTIVATED | ExecutionEventType.BUBBLE_COMPLETED;
  data: {
    bubble_id: string;
    bubble_type: string;
    status: string;
    result?: any;
  };
}

/**
 * Execution result
 */
export interface ExecutionResult {
  workflow_id: string;
  status: WorkflowStatus;
  final_solution: string;
  sub_problems: SubProblemResult[];
  statistics: ExecutionStatistics;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
}

/**
 * Sub-problem result
 */
export interface SubProblemResult {
  subproblem_id: string;
  problem: string;
  solution: string;
  status: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
}

/**
 * Execution statistics
 */
export interface ExecutionStatistics {
  total_duration_seconds: number;
  total_tokens_used: number;
  total_api_calls: number;
  sub_problems_solved: number;
  success_rate: number;
  memory_used_mb: number;
  cpu_time_seconds: number;
}

// ============================================================================
// Team Types
// ============================================================================

/**
 * Team member configuration
 */
export interface TeamMember {
  id: string;
  name: string;
  model: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
  max_iterations: number;
  role: string;
}

/**
 * Team definition
 */
export interface Team {
  id: string;
  name: string;
  description?: string;
  members: TeamMember[];
  created_at: string;
  updated_at: string;
  user_id: string;
  tenant_id: string;
}

/**
 * Create team request
 */
export interface CreateTeamRequest {
  name: string;
  description?: string;
  members: Omit<TeamMember, "id">[];
}

/**
 * Update team request
 */
export interface UpdateTeamRequest {
  name?: string;
  description?: string;
  members?: Omit<TeamMember, "id">[];
}

// ============================================================================
// Gauntlet Types
// ============================================================================

/**
 * Round configuration
 */
export interface GauntletRound {
  id: string;
  name: string;
  quorum_threshold: number;
  confidence_threshold: number;
  evaluation_type: string;
  required_consensus: boolean;
  max_iterations: number;
}

/**
 * Gauntlet definition
 */
export interface Gauntlet {
  id: string;
  name: string;
  description?: string;
  rounds: GauntletRound[];
  created_at: string;
  updated_at: string;
  user_id: string;
  tenant_id: string;
}

/**
 * Create gauntlet request
 */
export interface CreateGauntletRequest {
  name: string;
  description?: string;
  rounds: Omit<GauntletRound, "id">[];
}

/**
 * Update gauntlet request
 */
export interface UpdateGauntletRequest {
  name?: string;
  description?: string;
  rounds?: Omit<GauntletRound, "id">[];
}

// ============================================================================
// Settings Types
// ============================================================================

/**
 * LLM provider
 */
export enum LLMProvider {
  OPENAI = "openai",
  ANTHROPIC = "anthropic",
  COHERE = "cohere",
  CUSTOM = "custom"
}

/**
 * LLM configuration
 */
export interface LLMConfig {
  provider: LLMProvider;
  api_key: string;
  base_url?: string;
  model_leanaide: string;
  model_text: string;
  model_img: string;
  temperature: number;
  top_p: number;
  max_tokens: number;
  frequency_penalty: number;
  presence_penalty: number;
}

/**
 * Update LLM config request
 */
export interface UpdateLLMConfigRequest {
  provider?: LLMProvider;
  api_key?: string;
  base_url?: string;
  model_leanaide?: string;
  model_text?: string;
  model_img?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
}

// ============================================================================
// Benchmark Types
// ============================================================================

/**
 * Benchmark status
 */
export enum BenchmarkStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed"
}

/**
 * Benchmark definition
 */
export interface Benchmark {
  id: string;
  name: string;
  description?: string;
  dataset_path: string;
  workflow_ids: string[];
  status: BenchmarkStatus;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  results?: BenchmarkResult[];
}

/**
 * Benchmark result
 */
export interface BenchmarkResult {
  workflow_id: string;
  workflow_name: string;
  duration_seconds: number;
  tokens_used: number;
  success_rate: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
}

/**
 * Create benchmark request
 */
export interface CreateBenchmarkRequest {
  name: string;
  description?: string;
  dataset_path: string;
  workflow_ids: string[];
}

// ============================================================================
// Analytics Types
// ============================================================================

/**
 * Analytics metrics
 */
export interface AnalyticsMetrics {
  total_executions: number;
  successful_executions: number;
  failed_executions: number;
  success_rate: number;
  average_duration_seconds: number;
  total_tokens_used: number;
  total_api_calls: number;
}

/**
 * Time series data point
 */
export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
}

/**
 * Execution timeline analytics
 */
export interface ExecutionTimeline {
  date: string;
  executions: number;
  successes: number;
  failures: number;
  average_duration: number;
}

// ============================================================================
// UI State Types
// ============================================================================

/**
 * Panel mode in the UI
 */
export enum PanelMode {
  BUBBLE_LIST = "bubble_list",
  MILKTEA = "milktea",
  PEARL = "pearl",
  CODE = "code",
  OUTPUT = "output",
  HISTORY = "history"
}

/**
 * UI state
 */
export interface UIState {
  selectedFlowId: string | null;
  panelMode: PanelMode;
  sidebarCollapsed: boolean;
  darkMode: boolean;
  fontSize: "small" | "medium" | "large";
  autoSave: boolean;
}

// ============================================================================
// File Types
// ============================================================================

/**
 * Uploaded file
 */
export interface UploadedFile {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  uploaded_at: string;
  url: string;
}

/**
 * File upload response
 */
export interface FileUploadResponse {
  file_id: string;
  filename: string;
  url: string;
}

// ============================================================================
// Notification Types
// ============================================================================

/**
 * Notification type
 */
export enum NotificationType {
  INFO = "info",
  SUCCESS = "success",
  WARNING = "warning",
  ERROR = "error"
}

/**
 * Notification
 */
export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action_url?: string;
}

// ============================================================================
// User Types
// ============================================================================

/**
 * User profile
 */
export interface UserProfile {
  id: string;
  email: string;
  name?: string;
  avatar_url?: string;
  tenant_id: string;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// API Client Types
// ============================================================================

/**
 * API client configuration
 */
export interface ApiClientConfig {
  baseURL: string;
  timeout?: number;
  headers?: Record<string, string>;
}

/**
 * Query parameters for listing
 */
export interface ListQueryParams {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: "asc" | "desc";
  search?: string;
  filters?: Record<string, any>;
}
