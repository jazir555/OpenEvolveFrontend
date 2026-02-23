/**
 * Adaptive MDAP Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Adaptive Multi-Domain
 * Adaptive Processing (MDAP) interactions. All adapters must normalize their
 * data to/from this format.
 */

import { z } from 'zod';

/**
 * Processing Domain Enum
 */
export const ProcessingDomain = z.enum([
  'text',
  'image',
  'audio',
  'video',
  'multimodal',
  'structured_data',
]);

export type ProcessingDomain = z.infer<typeof ProcessingDomain>;

/**
 * Adaptation Mode Enum
 */
export const AdaptationMode = z.enum([
  'static',
  'dynamic',
  'incremental',
  'continual',
]);

export type AdaptationMode = z.infer<typeof AdaptationMode>;

/**
 * MDAP Processing Request Schema
 */
export const AdaptiveMdapRequest = z.object({
  task_id: z.string()
    .min(1, "Task ID cannot be empty")
    .describe("Unique identifier for the processing task"),

  domain: ProcessingDomain.describe("Processing domain"),

  input_data: z.union([
    z.string(),
    z.record(z.any()),
    z.array(z.any()),
  ]).describe("Input data to process"),

  adaptation_config: z.object({
    mode: AdaptationMode.optional().describe("Adaptation learning mode"),
    learning_rate: z.number().positive().optional().describe("Learning rate for adaptation"),
    batch_size: z.number().int().positive().optional()
      .describe("Batch size for incremental updates"),
    threshold: z.number().min(0).max(1).optional()
      .describe("Confidence threshold for adaptation"),
  }).optional().describe("Adaptation configuration"),

  model_config: z.object({
    base_model: z.string().optional().describe("Base model identifier"),
    fine_tuned: z.boolean().optional().describe("Whether to use fine-tuned model"),
    parameters: z.record(z.any()).optional().describe("Additional model parameters"),
  }).optional().describe("Model configuration"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(3600000, "Timeout cannot exceed 1 hour")
    .describe("Processing timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  metadata: z.record(z.any()).optional()
    .describe("Optional metadata"),
});

export type AdaptiveMdapRequest = z.infer<typeof AdaptiveMdapRequest>;

/**
 * MDAP Processing Response Schema
 */
export const AdaptiveMdapResponse = z.object({
  task_id: z.string().describe("Task identifier"),

  status: z.enum([
    'pending',
    'processing',
    'completed',
    'failed',
    'timeout',
  ]).describe("Processing status"),

  result: z.union([
    z.string(),
    z.record(z.any()),
    z.array(z.any()),
  ]).optional().describe("Processing result"),

  adaptations: z.object({
    adaptations_made: z.number().optional().describe("Number of adaptations applied"),
    adaptation_history: z.array(z.object({
      timestamp: z.string().datetime(),
      change_type: z.string(),
      performance_delta: z.number().optional(),
    })).optional().describe("History of adaptations"),
    model_version: z.string().optional().describe("Current model version"),
  }).optional().describe("Adaptation information"),

  performance: z.object({
    accuracy: z.number().min(0).max(1).optional()
      .describe("Accuracy score"),
    latency_ms: z.number().optional().describe("Processing latency"),
    throughput: z.number().optional().describe("Throughput metrics"),
    resource_usage: z.record(z.number()).optional().describe("Resource utilization"),
  }).optional().describe("Performance metrics"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional().describe("Error information"),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime().describe("UTC timestamp (ISO-8601)"),
});

export type AdaptiveMdapResponse = z.infer<typeof AdaptiveMdapResponse>;

/**
 * Batch Processing Request Schema
 */
export const AdaptiveMdapBatchRequest = z.object({
  batch_id: z.string().describe("Batch identifier"),

  tasks: z.array(AdaptiveMdapRequest.omit({ timeout_ms: true, correlation_id: true }))
    .min(1, "Batch must contain at least one task")
    .describe("Tasks to process"),

  config: z.object({
    parallelism: z.number().int().min(1).max(100)
      .optional()
      .describe("Number of parallel tasks"),
    stop_on_error: z.boolean().optional()
      .describe("Whether to stop on first error"),
    timeout_ms: z.number().int().positive().max(7200000)
      .optional()
      .describe("Overall batch timeout (max 2 hours)"),
  }).optional().describe("Batch configuration"),

  timeout_ms: z.number().int().positive().max(3600000)
    .describe("Default timeout for individual tasks (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),
  metadata: z.record(z.any()).optional(),
});

export type AdaptiveMdapBatchRequest = z.infer<typeof AdaptiveMdapBatchRequest>;

/**
 * Batch Processing Response Schema
 */
export const AdaptiveMdapBatchResponse = z.object({
  batch_id: z.string().describe("Batch identifier"),

  status: z.enum([
    'pending',
    'processing',
    'completed',
    'partially_completed',
    'failed',
  ]).describe("Batch status"),

  results: z.array(AdaptiveMdapResponse).describe("Individual task results"),

  summary: z.object({
    total_tasks: z.number().describe("Total number of tasks"),
    completed: z.number().describe("Successfully completed tasks"),
    failed: z.number().describe("Failed tasks"),
    total_processing_time_ms: z.number().optional().describe("Total processing time"),
    average_latency_ms: z.number().optional().describe("Average latency per task"),
  }).describe("Batch summary"),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AdaptiveMdapBatchResponse = z.infer<typeof AdaptiveMdapBatchResponse>;

/**
 * Error Model
 */
export const AdaptiveMdapError = z.object({
  code: z.enum([
    'INVALID_INPUT',
    'DOMAIN_NOT_SUPPORTED',
    'MODEL_NOT_AVAILABLE',
    'ADAPTATION_FAILED',
    'PROCESSING_TIMEOUT',
    'RESOURCE_EXHAUSTED',
    'VALIDATION_ERROR',
    'UNKNOWN_ERROR',
  ]).describe("Error code"),

  message: z.string().describe("Human-readable error message"),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AdaptiveMdapError = z.infer<typeof AdaptiveMdapError>;

/**
 * Validation Functions
 */
export function validateAdaptiveMdapRequest(data: unknown): {
  success: boolean;
  data?: AdaptiveMdapRequest;
  errors?: string[];
} {
  const result = AdaptiveMdapRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateAdaptiveMdapResponse(data: unknown): {
  success: boolean;
  data?: AdaptiveMdapResponse;
  errors?: string[];
} {
  const result = AdaptiveMdapResponse.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Type Guards
 */
export function isAdaptiveMdapRequest(data: unknown): data is AdaptiveMdapRequest {
  return typeof data === 'object' && data !== null
    && 'task_id' in data && 'domain' in data && 'input_data' in data;
}

/**
 * Example usage
 */
export const AdaptiveMdapExamples = {
  validRequest: {
    task_id: "task_001",
    domain: "text" as const,
    input_data: "Sample text for processing",
    adaptation_config: {
      mode: "incremental" as const,
      learning_rate: 0.001,
      threshold: 0.85,
    },
    timeout_ms: 30000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  } as AdaptiveMdapRequest,

  validResponse: {
    task_id: "task_001",
    status: "completed" as const,
    result: { processed: true, output: "Processed result" },
    adaptations: {
      adaptations_made: 5,
      model_version: "v1.2.3",
    },
    performance: {
      accuracy: 0.92,
      latency_ms: 1250,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:30:45.000Z",
  } as AdaptiveMdapResponse,
};

// =============================================================================
// V2.0 Advanced Features - Canonical Schemas
// =============================================================================

/**
 * Workflow Type Enum
 */
export const WorkflowType = z.enum([
  'evolution',
  'sovereign',
  'iterative_refinement',
  'formal_verification',
  'agent_collaboration',
  'knowledge_retrieval',
  'multi_system',
]);

export type WorkflowType = z.infer<typeof WorkflowType>;

/**
 * Complexity Dimensions Schema
 */
export const ComplexityDimensions = z.object({
  text_length: z.number().min(0).max(1).optional(),
  dependencies: z.number().min(0).max(1).optional(),
  depth: z.number().min(0).max(1).optional(),
  domain_specific: z.number().min(0).max(1).optional(),
  uncertainty: z.number().min(0).max(1).optional(),
  abstraction: z.number().min(0).max(1).optional(),
});

export type ComplexityDimensions = z.infer<typeof ComplexityDimensions>;

/**
 * Complexity Score Schema
 */
export const ComplexityScore = z.object({
  overall_score: z.number().min(0).max(1).describe("Overall complexity (0-1)"),
  dimensions: ComplexityDimensions.optional().describe("Dimension breakdown"),
  confidence: z.number().min(0).max(1).optional()
    .describe("Confidence in score"),
  strategy: z.enum([
    'DIRECT',
    'SEQUENTIAL',
    'PARALLEL',
    'HIERARCHICAL',
    'ADAPTIVE',
  ]).optional().describe("Recommended strategy"),
});

export type ComplexityScore = z.infer<typeof ComplexityScore>;

/**
 * Sub-Problem Schema (for Decomposition)
 */
export const SubProblem = z.object({
  id: z.string().describe("Unique sub-problem identifier"),
  description: z.string().describe("Sub-problem description"),
  complexity: z.number().min(0).max(1).describe("Complexity score"),
  dependencies: z.array(z.string()).optional().describe("Dependencies on other sub-problems"),
  estimated_effort: z.string().optional().describe("Estimated effort (e.g., '2-4 hours')"),
  parent_id: z.string().optional().describe("Parent problem ID"),
});

export type SubProblem = z.infer<typeof SubProblem>;

/**
 * Problem Decomposition Result Schema (V2.0)
 */
export const ProblemDecompositionResult = z.object({
  workflow_id: z.string().describe("Workflow identifier"),
  decomposition_strategy: z.enum([
    'hierarchical',
    'functional',
    'domain_based',
    'hybrid',
  ]).describe("Strategy used for decomposition"),
  sub_problems: z.array(SubProblem).describe("Decomposed sub-problems"),
  recommended_parallelization: z.enum([
    'none',
    'partial',
    'full',
  ]).describe("Parallelization recommendation"),
  metadata: z.record(z.any()).optional(),
  timestamp: z.string().datetime().describe("UTC timestamp"),
});

export type ProblemDecompositionResult = z.infer<typeof ProblemDecompositionResult>;

/**
 * Team Member Schema
 */
export const TeamMember = z.object({
  name: z.string().describe("Team member name/identifier"),
  role: z.string().describe("Role in the team"),
  capabilities: z.array(z.string()).optional().describe("List of capabilities"),
  availability: z.boolean().optional().describe("Whether available"),
});

export type TeamMember = z.infer<typeof TeamMember>;

/**
 * Team Selection Result Schema (V2.0)
 */
export const TeamSelectionResult = z.object({
  workflow_id: z.string().describe("Workflow identifier"),
  stage: z.string().describe("Workflow stage"),
  workflow_type: WorkflowType.describe("Type of workflow"),
  complexity_score: z.number().min(0).max(1).describe("Complexity score"),
  recommended_teams: z.record(z.object({
    agents: z.array(z.string()).describe("Agent names/types"),
    reasoning: z.string().describe("Reasoning for selection"),
    confidence: z.number().min(0).max(1).optional()
      .describe("Confidence in selection"),
  })).describe("Recommended team composition"),
  estimated_cost: z.number().optional().describe("Estimated cost"),
  metadata: z.record(z.any()).optional(),
  timestamp: z.string().datetime(),
});

export type TeamSelectionResult = z.infer<typeof TeamSelectionResult>;

/**
 * Resource Optimization Result Schema (V2.0)
 */
export const ResourceOptimizationResult = z.object({
  workflow_id: z.string().describe("Workflow identifier"),
  stage: z.string().describe("Workflow stage"),
  complexity_score: z.number().min(0).max(1).describe("Complexity score"),
  cpu_allocation: z.number().describe("CPU cores allocated"),
  memory_allocation_mb: z.number().describe("Memory in MB"),
  timeout_ms: z.number().describe("Timeout in milliseconds"),
  estimated_duration_ms: z.number().optional().describe("Estimated duration"),
  estimated_cost_savings: z.number().min(0).max(1).optional()
    .describe("Cost savings (0-1)"),
  recommendations: z.array(z.string()).optional().describe("Optimization recommendations"),
  metadata: z.record(z.any()).optional(),
  timestamp: z.string().datetime(),
});

export type ResourceOptimizationResult = z.infer<typeof ResourceOptimizationResult>;

/**
 * Gauntlet Type Enum (V2.0)
 */
export const GauntletType = z.enum([
  'ADVERSARIAL',
  'FORMAL_VERIFICATION',
  'RED_TEAM',
  'CHAOS_ENGINEERING',
  'SECURITY_AUDIT',
  'PERFORMANCE_TEST',
  'CORRECTNESS_PROOF',
  'STRESS_TEST',
  'FUZZING',
  'COMPLIANCE_CHECK',
]);

export type GauntletType = z.infer<typeof GauntletType>;

/**
 * Gauntlet Severity Enum
 */
export const GauntletSeverity = z.enum([
  'BASIC',
  'STANDARD',
  'STRICT',
  'HARDCORE',
]);

export type GauntletSeverity = z.infer<typeof GauntletSeverity>;

/**
 * Gauntlet Config Schema (V2.0)
 */
export const GauntletConfig = z.object({
  gauntlet_type: GauntletType.describe("Type of gauntlet"),
  complexity_score: z.number().min(0).max(1).describe("Solution complexity"),
  severity: GauntletSeverity.optional().describe("Test severity"),
  custom_parameters: z.record(z.any()).optional().describe("Custom parameters"),
  timeout_ms: z.number().optional().describe("Gauntlet timeout"),
});

export type GauntletConfig = z.infer<typeof GauntletConfig>;

/**
 * Gauntlet Result Schema (V2.0)
 */
export const GauntletResult = z.object({
  gauntlet_type: GauntletType.describe("Type of gauntlet"),
  passed: z.boolean().describe("Whether gauntlet was passed"),
  score: z.number().min(0).max(1).describe("Score (0-1)"),
  reasoning: z.string().describe("Reasoning for result"),
  red_flags: z.array(z.string()).optional().describe("Issues found"),
  execution_time_ms: z.number().describe("Execution time"),
  timestamp: z.string().datetime(),
});

export type GauntletResult = z.infer<typeof GauntletResult>;

/**
 * Gauntlet Pipeline Schema (V2.0)
 */
export const GauntletPipeline = z.object({
  pipeline_id: z.string().describe("Pipeline identifier"),
  complexity_score: z.number().min(0).max(1).describe("Solution complexity"),
  base_gauntlet_type: GauntletType.describe("Base gauntlet type"),
  gauntlets: z.array(GauntletConfig).describe("Gauntlets in pipeline"),
  execution_mode: z.enum(['sequential', 'parallel', 'adaptive']).describe("Execution mode"),
  aggregation_strategy: z.enum(['all_must_pass', 'majority', 'weighted']).describe("How to aggregate results"),
  severity: GauntletSeverity.optional().describe("Overall severity"),
});

export type GauntletPipeline = z.infer<typeof GauntletPipeline>;

/**
 * Gauntlet Pipeline Result Schema (V2.0)
 */
export const GauntletPipelineResult = z.object({
  pipeline_id: z.string().describe("Pipeline identifier"),
  total_gauntlets: z.number().describe("Total gauntlets"),
  passed_gauntlets: z.number().describe("Gauntlets passed"),
  failed_gauntlets: z.number().describe("Gauntlets failed"),
  skipped_gauntlets: z.number().optional().describe("Gauntlets skipped"),
  overall_pass: z.boolean().describe("Whether pipeline passed overall"),
  aggregate_score: z.number().min(0).max(1).describe("Aggregate score"),
  gauntlet_results: z.array(GauntletResult).describe("Individual results"),
  execution_time_ms: z.number().describe("Total execution time"),
  timestamp: z.string().datetime(),
});

export type GauntletPipelineResult = z.infer<typeof GauntletPipelineResult>;

/**
 * ICR Pattern Type Enum (V2.0)
 */
export const ICRPatternType = z.enum([
  'WORKFLOW_EXECUTION',
  'REFINEMENT_LOOP',
  'RESOURCE_USAGE',
  'QUALITY_OUTCOME',
  'RETRY_PATTERN',
  'BOTTLENECK',
  'OPTIMIZATION',
  'SECURITY_POLICY',
  'GAUNTLET_OUTCOME',
]);

export type ICRPatternType = z.infer<typeof ICRPatternType>;

/**
 * ICR Pattern Schema (V2.0)
 */
export const ICRPattern = z.object({
  pattern_id: z.string().describe("Pattern identifier"),
  pattern_type: ICRPatternType.describe("Type of pattern"),
  context: z.record(z.any()).describe("Pattern context"),
  passed: z.boolean().describe("Whether pattern passed"),
  metrics: z.record(z.any()).optional().describe("Pattern metrics"),
  timestamp: z.string().datetime().describe("When pattern was recorded"),
  metadata: z.record(z.any()).optional(),
});

export type ICRPattern = z.infer<typeof ICRPattern>;

/**
 * ICR Prediction Schema (V2.0)
 */
export const ICRPrediction = z.object({
  pattern_type: ICRPatternType.describe("Type of pattern"),
  predicted_outcome: z.boolean().describe("Predicted outcome"),
  confidence: z.number().min(0).max(1).describe("Confidence in prediction"),
  recommended_action: z.string().describe("Recommended action"),
  pattern_count: z.number().optional().describe("Number of patterns used"),
  timestamp: z.string().datetime(),
});

export type ICRPrediction = z.infer<typeof ICRPrediction>;

/**
 * Pattern Cluster Schema (V2.0)
 */
export const PatternCluster = z.object({
  cluster_id: z.string().describe("Cluster identifier"),
  pattern_type: ICRPatternType.describe("Type of patterns in cluster"),
  patterns: z.array(z.lazy(() => ICRPattern)).describe("Patterns in cluster"),
  centroid: z.record(z.any()).describe("Cluster centroid"),
  similarity_score: z.number().min(0).max(1).describe("Cluster similarity"),
  timestamp: z.string().datetime(),
});

export type PatternCluster = z.infer<typeof PatternCluster>;

/**
 * ICR Pattern Insights Schema (V2.0)
 */
export const ICRPatternInsights = z.object({
  available: z.boolean().describe("Whether ICR is available"),
  pattern_types: z.record(z.object({
    count: z.number().describe("Number of patterns"),
    pass_rate: z.number().min(0).max(1).describe("Pass rate"),
    confidence: z.number().min(0).max(1).describe("Confidence"),
    recent_patterns: z.array(z.any()).optional().describe("Recent patterns"),
  })).optional().describe("Statistics by pattern type"),
  total_patterns: z.number().optional().describe("Total patterns across all types"),
  timestamp: z.string().datetime(),
});

export type ICRPatternInsights = z.infer<typeof ICRPatternInsights>;

/**
 * Chart Type Enum (V2.0 - UI)
 */
export const ChartType = z.enum([
  'RADAR',
  'BAR',
  'LINE',
  'PIE',
  'TIMELINE',
  'SCATTER',
  'HEATMAP',
]);

export type ChartType = z.infer<typeof ChartType>;

/**
 * UI Chart Data Schema (V2.0)
 */
export const UIChartData = z.object({
  chart_type: ChartType.describe("Type of chart"),
  title: z.string().describe("Chart title"),
  labels: z.array(z.string()).optional().describe("Chart labels"),
  datasets: z.array(z.object({
    label: z.string().optional(),
    data: z.array(z.union([z.number(), z.string(), z.boolean(), z.array(z.any())])),
    backgroundColor: z.array(z.string()).optional(),
    borderColor: z.array(z.string()).optional(),
  })).optional().describe("Chart datasets"),
  data: z.record(z.any()).optional().describe("Raw chart data"),
  options: z.record(z.any()).optional().describe("Chart options"),
  recommendations: z.array(z.string()).optional().describe("Recommendations based on chart"),
  timestamp: z.string().datetime(),
});

export type UIChartData = z.infer<typeof UIChartData>;

/**
 * Workflow Timeline Schema (V2.0 - UI)
 */
export const WorkflowTimeline = z.object({
  chart_type: ChartType.describe("Chart type"),
  workflow_id: z.string().describe("Workflow identifier"),
  stages: z.array(z.object({
    stage: z.string().describe("Stage name"),
    status: z.enum(['pending', 'in_progress', 'completed', 'failed', 'skipped']),
    duration_ms: z.number().optional().describe("Stage duration"),
    start_time: z.string().datetime().optional(),
    end_time: z.string().datetime().optional(),
  })).describe("Workflow stages"),
  total_duration_ms: z.number().describe("Total duration"),
  timestamp: z.string().datetime(),
});

export type WorkflowTimeline = z.infer<typeof WorkflowTimeline>;

/**
 * Adapter Health Status Schema (V2.0)
 */
export const AdapterHealthStatus = z.object({
  overall_status: z.enum(['healthy', 'degraded', 'unhealthy']).describe("Overall health"),
  components: z.record(z.object({
    status: z.enum(['healthy', 'degraded', 'unhealthy', 'disabled']),
    last_check: z.string().datetime().optional(),
    error: z.string().optional(),
  })).describe("Component health"),
  uptime_ms: z.number().optional().describe("Uptime in milliseconds"),
  alerts: z.array(z.object({
    severity: z.enum(['info', 'warning', 'error', 'critical']),
    message: z.string(),
    timestamp: z.string().datetime(),
  })).optional().describe("Active alerts"),
  timestamp: z.string().datetime(),
});

export type AdapterHealthStatus = z.infer<typeof AdapterHealthStatus>;

/**
 * Cache Statistics Schema (V2.0 - Performance)
 */
export const CacheStatistics = z.object({
  size: z.number().describe("Current cache size"),
  max_size: z.number().describe("Maximum cache size"),
  ttl: z.number().describe("Time-to-live in seconds"),
  total_hits: z.number().describe("Total cache hits"),
  total_misses: z.number().describe("Total cache misses"),
  hit_rate: z.number().min(0).max(1).describe("Cache hit rate"),
  evictions: z.number().optional().describe("Number of evictions"),
  timestamp: z.string().datetime(),
});

export type CacheStatistics = z.infer<typeof CacheStatistics>;

/**
 * Performance Metrics Schema (V2.0)
 */
export const PerformanceMetrics = z.object({
  operation: z.string().describe("Operation name"),
  count: z.number().describe("Number of operations"),
  avg_ms: z.number().describe("Average latency in ms"),
  min_ms: z.number().describe("Minimum latency in ms"),
  max_ms: z.number().describe("Maximum latency in ms"),
  p50_ms: z.number().describe("P50 latency in ms"),
  p95_ms: z.number().describe("P95 latency in ms"),
  p99_ms: z.number().describe("P99 latency in ms"),
  throughput_per_sec: z.number().optional().describe("Operations per second"),
  error_rate: z.number().min(0).max(1).optional()
    .describe("Error rate"),
  timestamp: z.string().datetime(),
});

export type PerformanceMetrics = z.infer<typeof PerformanceMetrics>;

/**
 * Async Operation Status Enum (V2.0)
 */
export const AsyncOperationStatus = z.enum([
  'pending',
  'scheduled',
  'running',
  'completed',
  'failed',
  'cancelled',
]);

export type AsyncOperationStatus = z.infer<typeof AsyncOperationStatus>;

/**
 * Async Operation Schema (V2.0)
 */
export const AsyncOperation = z.object({
  operation_id: z.string().describe("Operation identifier"),
  operation_type: z.string().describe("Type of operation"),
  status: AsyncOperationStatus.describe("Operation status"),
  input_data: z.record(z.any()).optional().describe("Input data"),
  result: z.any().optional().describe("Operation result"),
  error: z.string().optional().describe("Error message if failed"),
  created_at: z.string().datetime().describe("Creation timestamp"),
  started_at: z.string().datetime().optional().describe("Start timestamp"),
  completed_at: z.string().datetime().optional().describe("Completion timestamp"),
  duration_ms: z.number().optional().describe("Duration in milliseconds"),
  metadata: z.record(z.any()).optional(),
});

export type AsyncOperation = z.infer<typeof AsyncOperation>;

/**
 * Batch Operation Schema (V2.0)
 */
export const BatchOperation = z.object({
  batch_id: z.string().describe("Batch identifier"),
  operations: z.array(AsyncOperation).describe("Operations in batch"),
  total_operations: z.number().describe("Total number of operations"),
  completed_operations: z.number().describe("Completed operations"),
  failed_operations: z.number().describe("Failed operations"),
  max_concurrency: z.number().optional().describe("Max concurrent operations"),
  status: AsyncOperationStatus.describe("Batch status"),
  created_at: z.string().datetime(),
  completed_at: z.string().datetime().optional(),
  total_duration_ms: z.number().optional(),
});

export type BatchOperation = z.infer<typeof BatchOperation>;

/**
 * Additional System Type Enum (V2.0)
 */
export const AdditionalSystemType = z.enum([
  'crewai',
  'mcp_tools',
  'knowledge_engine',
  'leanaide',
  'z3_prover',
]);

export type AdditionalSystemType = z.infer<typeof AdditionalSystemType>;

/**
 * System Health Schema (V2.0)
 */
export const SystemHealth = z.object({
  system: AdditionalSystemType.describe("System identifier"),
  available: z.boolean().describe("Whether system is available"),
  status: z.enum(['healthy', 'degraded', 'unhealthy', 'disabled']).describe("System status"),
  last_check: z.string().datetime().optional().describe("Last health check"),
  reason: z.string().optional().describe("Reason if unavailable"),
  capabilities: z.array(z.string()).optional().describe("System capabilities"),
  metadata: z.record(z.any()).optional(),
  timestamp: z.string().datetime(),
});

export type SystemHealth = z.infer<typeof SystemHealth>;

/**
 * Unified System Health Schema (V2.0)
 */
export const UnifiedSystemHealth = z.object({
  overall_status: z.enum(['healthy', 'degraded', 'unhealthy']).describe("Overall status"),
  total_systems: z.number().describe("Total number of systems"),
  available_systems: z.number().describe("Number of available systems"),
  systems: z.record(SystemHealth).describe("Individual system health"),
  timestamp: z.string().datetime(),
});

export type UnifiedSystemHealth = z.infer<typeof UnifiedSystemHealth>;

/**
 * Workflow Step Schema (V2.0)
 */
export const WorkflowStep = z.object({
  step: z.number().describe("Step number"),
  system: AdditionalSystemType.optional().describe("System used"),
  action: z.string().describe("Action performed"),
  success: z.boolean().describe("Whether step succeeded"),
  result: z.any().optional().describe("Step result"),
  error: z.string().optional().describe("Error if failed"),
  duration_ms: z.number().optional().describe("Step duration"),
  timestamp: z.string().datetime(),
});

export type WorkflowStep = z.infer<typeof WorkflowStep>;

/**
 * Cross-System Workflow Result Schema (V2.0)
 */
export const CrossSystemWorkflowResult = z.object({
  workflow_type: z.string().describe("Type of workflow"),
  success: z.boolean().describe("Whether workflow succeeded"),
  steps: z.array(WorkflowStep).describe("Workflow steps"),
  result_count: z.number().optional().describe("Number of results"),
  result: z.any().optional().describe("Final result"),
  error: z.string().optional().describe("Error if failed"),
  total_duration_ms: z.number().optional().describe("Total duration"),
  metadata: z.record(z.any()).optional(),
  timestamp: z.string().datetime(),
});

export type CrossSystemWorkflowResult = z.infer<typeof CrossSystemWorkflowResult>;

/**
 * V2.0 Validation Functions
 */
export function validateProblemDecomposition(data: unknown): {
  success: boolean;
  data?: ProblemDecompositionResult;
  errors?: string[];
} {
  const result = ProblemDecompositionResult.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateTeamSelection(data: unknown): {
  success: boolean;
  data?: TeamSelectionResult;
  errors?: string[];
} {
  const result = TeamSelectionResult.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateGauntletPipelineResult(data: unknown): {
  success: boolean;
  data?: GauntletPipelineResult;
  errors?: string[];
} {
  const result = GauntletPipelineResult.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateICRPattern(data: unknown): {
  success: boolean;
  data?: ICRPattern;
  errors?: string[];
} {
  const result = ICRPattern.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateCrossSystemWorkflowResult(data: unknown): {
  success: boolean;
  data?: CrossSystemWorkflowResult;
  errors?: string[];
} {
  const result = CrossSystemWorkflowResult.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}
