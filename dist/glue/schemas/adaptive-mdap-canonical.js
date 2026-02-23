"use strict";
/**
 * Adaptive MDAP Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Adaptive Multi-Domain
 * Adaptive Processing (MDAP) interactions. All adapters must normalize their
 * data to/from this format.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CrossSystemWorkflowResult = exports.WorkflowStep = exports.UnifiedSystemHealth = exports.SystemHealth = exports.AdditionalSystemType = exports.BatchOperation = exports.AsyncOperation = exports.AsyncOperationStatus = exports.PerformanceMetrics = exports.CacheStatistics = exports.AdapterHealthStatus = exports.WorkflowTimeline = exports.UIChartData = exports.ChartType = exports.ICRPatternInsights = exports.PatternCluster = exports.ICRPrediction = exports.ICRPattern = exports.ICRPatternType = exports.GauntletPipelineResult = exports.GauntletPipeline = exports.GauntletResult = exports.GauntletConfig = exports.GauntletSeverity = exports.GauntletType = exports.ResourceOptimizationResult = exports.TeamSelectionResult = exports.TeamMember = exports.ProblemDecompositionResult = exports.SubProblem = exports.ComplexityScore = exports.ComplexityDimensions = exports.WorkflowType = exports.AdaptiveMdapExamples = exports.AdaptiveMdapError = exports.AdaptiveMdapBatchResponse = exports.AdaptiveMdapBatchRequest = exports.AdaptiveMdapResponse = exports.AdaptiveMdapRequest = exports.AdaptationMode = exports.ProcessingDomain = void 0;
exports.validateAdaptiveMdapRequest = validateAdaptiveMdapRequest;
exports.validateAdaptiveMdapResponse = validateAdaptiveMdapResponse;
exports.isAdaptiveMdapRequest = isAdaptiveMdapRequest;
exports.validateProblemDecomposition = validateProblemDecomposition;
exports.validateTeamSelection = validateTeamSelection;
exports.validateGauntletPipelineResult = validateGauntletPipelineResult;
exports.validateICRPattern = validateICRPattern;
exports.validateCrossSystemWorkflowResult = validateCrossSystemWorkflowResult;
const zod_1 = require("zod");
/**
 * Processing Domain Enum
 */
exports.ProcessingDomain = zod_1.z.enum([
    'text',
    'image',
    'audio',
    'video',
    'multimodal',
    'structured_data',
]);
/**
 * Adaptation Mode Enum
 */
exports.AdaptationMode = zod_1.z.enum([
    'static',
    'dynamic',
    'incremental',
    'continual',
]);
/**
 * MDAP Processing Request Schema
 */
exports.AdaptiveMdapRequest = zod_1.z.object({
    task_id: zod_1.z.string()
        .min(1, "Task ID cannot be empty")
        .describe("Unique identifier for the processing task"),
    domain: exports.ProcessingDomain.describe("Processing domain"),
    input_data: zod_1.z.union([
        zod_1.z.string(),
        zod_1.z.record(zod_1.z.any()),
        zod_1.z.array(zod_1.z.any()),
    ]).describe("Input data to process"),
    adaptation_config: zod_1.z.object({
        mode: exports.AdaptationMode.optional().describe("Adaptation learning mode"),
        learning_rate: zod_1.z.number().positive().optional().describe("Learning rate for adaptation"),
        batch_size: zod_1.z.number().int().positive().optional().describe("Batch size for incremental updates"),
        threshold: zod_1.z.number().min(0).max(1).optional().describe("Confidence threshold for adaptation"),
    }).optional().describe("Adaptation configuration"),
    model_config: zod_1.z.object({
        base_model: zod_1.z.string().optional().describe("Base model identifier"),
        fine_tuned: zod_1.z.boolean().optional().describe("Whether to use fine-tuned model"),
        parameters: zod_1.z.record(zod_1.z.any()).optional().describe("Additional model parameters"),
    }).optional().describe("Model configuration"),
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(3600000, "Timeout cannot exceed 1 hour")
        .describe("Processing timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata"),
});
/**
 * MDAP Processing Response Schema
 */
exports.AdaptiveMdapResponse = zod_1.z.object({
    task_id: zod_1.z.string().describe("Task identifier"),
    status: zod_1.z.enum([
        'pending',
        'processing',
        'completed',
        'failed',
        'timeout',
    ]).describe("Processing status"),
    result: zod_1.z.union([
        zod_1.z.string(),
        zod_1.z.record(zod_1.z.any()),
        zod_1.z.array(zod_1.z.any()),
    ]).optional().describe("Processing result"),
    adaptations: zod_1.z.object({
        adaptations_made: zod_1.z.number().optional().describe("Number of adaptations applied"),
        adaptation_history: zod_1.z.array(zod_1.z.object({
            timestamp: zod_1.z.string().datetime(),
            change_type: zod_1.z.string(),
            performance_delta: zod_1.z.number().optional(),
        })).optional().describe("History of adaptations"),
        model_version: zod_1.z.string().optional().describe("Current model version"),
    }).optional().describe("Adaptation information"),
    performance: zod_1.z.object({
        accuracy: zod_1.z.number().min(0).max(1).optional().describe("Accuracy score"),
        latency_ms: zod_1.z.number().optional().describe("Processing latency"),
        throughput: zod_1.z.number().optional().describe("Throughput metrics"),
        resource_usage: zod_1.z.record(zod_1.z.number()).optional().describe("Resource utilization"),
    }).optional().describe("Performance metrics"),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }).optional().describe("Error information"),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime().describe("UTC timestamp (ISO-8601)"),
});
/**
 * Batch Processing Request Schema
 */
exports.AdaptiveMdapBatchRequest = zod_1.z.object({
    batch_id: zod_1.z.string().describe("Batch identifier"),
    tasks: zod_1.z.array(exports.AdaptiveMdapRequest.omit({ timeout_ms: true, correlation_id: true }))
        .min(1, "Batch must contain at least one task")
        .describe("Tasks to process"),
    config: zod_1.z.object({
        parallelism: zod_1.z.number().int().min(1).max(100).optional()
            .describe("Number of parallel tasks"),
        stop_on_error: zod_1.z.boolean().optional()
            .describe("Whether to stop on first error"),
        timeout_ms: zod_1.z.number().int().positive().max(7200000).optional()
            .describe("Overall batch timeout (max 2 hours)"),
    }).optional().describe("Batch configuration"),
    timeout_ms: zod_1.z.number().int().positive().max(3600000)
        .describe("Default timeout for individual tasks (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Batch Processing Response Schema
 */
exports.AdaptiveMdapBatchResponse = zod_1.z.object({
    batch_id: zod_1.z.string().describe("Batch identifier"),
    status: zod_1.z.enum([
        'pending',
        'processing',
        'completed',
        'partially_completed',
        'failed',
    ]).describe("Batch status"),
    results: zod_1.z.array(exports.AdaptiveMdapResponse).describe("Individual task results"),
    summary: zod_1.z.object({
        total_tasks: zod_1.z.number().describe("Total number of tasks"),
        completed: zod_1.z.number().describe("Successfully completed tasks"),
        failed: zod_1.z.number().describe("Failed tasks"),
        total_processing_time_ms: zod_1.z.number().optional().describe("Total processing time"),
        average_latency_ms: zod_1.z.number().optional().describe("Average latency per task"),
    }).describe("Batch summary"),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Error Model
 */
exports.AdaptiveMdapError = zod_1.z.object({
    code: zod_1.z.enum([
        'INVALID_INPUT',
        'DOMAIN_NOT_SUPPORTED',
        'MODEL_NOT_AVAILABLE',
        'ADAPTATION_FAILED',
        'PROCESSING_TIMEOUT',
        'RESOURCE_EXHAUSTED',
        'VALIDATION_ERROR',
        'UNKNOWN_ERROR',
    ]).describe("Error code"),
    message: zod_1.z.string().describe("Human-readable error message"),
    details: zod_1.z.record(zod_1.z.any()).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Validation Functions
 */
function validateAdaptiveMdapRequest(data) {
    const result = exports.AdaptiveMdapRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateAdaptiveMdapResponse(data) {
    const result = exports.AdaptiveMdapResponse.safeParse(data);
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
function isAdaptiveMdapRequest(data) {
    return typeof data === 'object' && data !== null &&
        'task_id' in data && 'domain' in data && 'input_data' in data;
}
/**
 * Example usage
 */
exports.AdaptiveMdapExamples = {
    validRequest: {
        task_id: "task_001",
        domain: "text",
        input_data: "Sample text for processing",
        adaptation_config: {
            mode: "incremental",
            learning_rate: 0.001,
            threshold: 0.85,
        },
        timeout_ms: 30000,
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    },
    validResponse: {
        task_id: "task_001",
        status: "completed",
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
    },
};
// =============================================================================
// V2.0 Advanced Features - Canonical Schemas
// =============================================================================
/**
 * Workflow Type Enum
 */
exports.WorkflowType = zod_1.z.enum([
    'evolution',
    'sovereign',
    'iterative_refinement',
    'formal_verification',
    'agent_collaboration',
    'knowledge_retrieval',
    'multi_system',
]);
/**
 * Complexity Dimensions Schema
 */
exports.ComplexityDimensions = zod_1.z.object({
    text_length: zod_1.z.number().min(0).max(1).optional(),
    dependencies: zod_1.z.number().min(0).max(1).optional(),
    depth: zod_1.z.number().min(0).max(1).optional(),
    domain_specific: zod_1.z.number().min(0).max(1).optional(),
    uncertainty: zod_1.z.number().min(0).max(1).optional(),
    abstraction: zod_1.z.number().min(0).max(1).optional(),
});
/**
 * Complexity Score Schema
 */
exports.ComplexityScore = zod_1.z.object({
    overall_score: zod_1.z.number().min(0).max(1).describe("Overall complexity (0-1)"),
    dimensions: exports.ComplexityDimensions.optional().describe("Dimension breakdown"),
    confidence: zod_1.z.number().min(0).max(1).optional().describe("Confidence in score"),
    strategy: zod_1.z.enum([
        'DIRECT',
        'SEQUENTIAL',
        'PARALLEL',
        'HIERARCHICAL',
        'ADAPTIVE',
    ]).optional().describe("Recommended strategy"),
});
/**
 * Sub-Problem Schema (for Decomposition)
 */
exports.SubProblem = zod_1.z.object({
    id: zod_1.z.string().describe("Unique sub-problem identifier"),
    description: zod_1.z.string().describe("Sub-problem description"),
    complexity: zod_1.z.number().min(0).max(1).describe("Complexity score"),
    dependencies: zod_1.z.array(zod_1.z.string()).optional().describe("Dependencies on other sub-problems"),
    estimated_effort: zod_1.z.string().optional().describe("Estimated effort (e.g., '2-4 hours')"),
    parent_id: zod_1.z.string().optional().describe("Parent problem ID"),
});
/**
 * Problem Decomposition Result Schema (V2.0)
 */
exports.ProblemDecompositionResult = zod_1.z.object({
    workflow_id: zod_1.z.string().describe("Workflow identifier"),
    decomposition_strategy: zod_1.z.enum([
        'hierarchical',
        'functional',
        'domain_based',
        'hybrid',
    ]).describe("Strategy used for decomposition"),
    sub_problems: zod_1.z.array(exports.SubProblem).describe("Decomposed sub-problems"),
    recommended_parallelization: zod_1.z.enum([
        'none',
        'partial',
        'full',
    ]).describe("Parallelization recommendation"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
    timestamp: zod_1.z.string().datetime().describe("UTC timestamp"),
});
/**
 * Team Member Schema
 */
exports.TeamMember = zod_1.z.object({
    name: zod_1.z.string().describe("Team member name/identifier"),
    role: zod_1.z.string().describe("Role in the team"),
    capabilities: zod_1.z.array(zod_1.z.string()).optional().describe("List of capabilities"),
    availability: zod_1.z.boolean().optional().describe("Whether available"),
});
/**
 * Team Selection Result Schema (V2.0)
 */
exports.TeamSelectionResult = zod_1.z.object({
    workflow_id: zod_1.z.string().describe("Workflow identifier"),
    stage: zod_1.z.string().describe("Workflow stage"),
    workflow_type: exports.WorkflowType.describe("Type of workflow"),
    complexity_score: zod_1.z.number().min(0).max(1).describe("Complexity score"),
    recommended_teams: zod_1.z.record(zod_1.z.object({
        agents: zod_1.z.array(zod_1.z.string()).describe("Agent names/types"),
        reasoning: zod_1.z.string().describe("Reasoning for selection"),
        confidence: zod_1.z.number().min(0).max(1).optional().describe("Confidence in selection"),
    })).describe("Recommended team composition"),
    estimated_cost: zod_1.z.number().optional().describe("Estimated cost"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Resource Optimization Result Schema (V2.0)
 */
exports.ResourceOptimizationResult = zod_1.z.object({
    workflow_id: zod_1.z.string().describe("Workflow identifier"),
    stage: zod_1.z.string().describe("Workflow stage"),
    complexity_score: zod_1.z.number().min(0).max(1).describe("Complexity score"),
    cpu_allocation: zod_1.z.number().describe("CPU cores allocated"),
    memory_allocation_mb: zod_1.z.number().describe("Memory in MB"),
    timeout_ms: zod_1.z.number().describe("Timeout in milliseconds"),
    estimated_duration_ms: zod_1.z.number().optional().describe("Estimated duration"),
    estimated_cost_savings: zod_1.z.number().min(0).max(1).optional().describe("Cost savings (0-1)"),
    recommendations: zod_1.z.array(zod_1.z.string()).optional().describe("Optimization recommendations"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Gauntlet Type Enum (V2.0)
 */
exports.GauntletType = zod_1.z.enum([
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
/**
 * Gauntlet Severity Enum
 */
exports.GauntletSeverity = zod_1.z.enum([
    'BASIC',
    'STANDARD',
    'STRICT',
    'HARDCORE',
]);
/**
 * Gauntlet Config Schema (V2.0)
 */
exports.GauntletConfig = zod_1.z.object({
    gauntlet_type: exports.GauntletType.describe("Type of gauntlet"),
    complexity_score: zod_1.z.number().min(0).max(1).describe("Solution complexity"),
    severity: exports.GauntletSeverity.optional().describe("Test severity"),
    custom_parameters: zod_1.z.record(zod_1.z.any()).optional().describe("Custom parameters"),
    timeout_ms: zod_1.z.number().optional().describe("Gauntlet timeout"),
});
/**
 * Gauntlet Result Schema (V2.0)
 */
exports.GauntletResult = zod_1.z.object({
    gauntlet_type: exports.GauntletType.describe("Type of gauntlet"),
    passed: zod_1.z.boolean().describe("Whether gauntlet was passed"),
    score: zod_1.z.number().min(0).max(1).describe("Score (0-1)"),
    reasoning: zod_1.z.string().describe("Reasoning for result"),
    red_flags: zod_1.z.array(zod_1.z.string()).optional().describe("Issues found"),
    execution_time_ms: zod_1.z.number().describe("Execution time"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Gauntlet Pipeline Schema (V2.0)
 */
exports.GauntletPipeline = zod_1.z.object({
    pipeline_id: zod_1.z.string().describe("Pipeline identifier"),
    complexity_score: zod_1.z.number().min(0).max(1).describe("Solution complexity"),
    base_gauntlet_type: exports.GauntletType.describe("Base gauntlet type"),
    gauntlets: zod_1.z.array(exports.GauntletConfig).describe("Gauntlets in pipeline"),
    execution_mode: zod_1.z.enum(['sequential', 'parallel', 'adaptive']).describe("Execution mode"),
    aggregation_strategy: zod_1.z.enum(['all_must_pass', 'majority', 'weighted']).describe("How to aggregate results"),
    severity: exports.GauntletSeverity.optional().describe("Overall severity"),
});
/**
 * Gauntlet Pipeline Result Schema (V2.0)
 */
exports.GauntletPipelineResult = zod_1.z.object({
    pipeline_id: zod_1.z.string().describe("Pipeline identifier"),
    total_gauntlets: zod_1.z.number().describe("Total gauntlets"),
    passed_gauntlets: zod_1.z.number().describe("Gauntlets passed"),
    failed_gauntlets: zod_1.z.number().describe("Gauntlets failed"),
    skipped_gauntlets: zod_1.z.number().optional().describe("Gauntlets skipped"),
    overall_pass: zod_1.z.boolean().describe("Whether pipeline passed overall"),
    aggregate_score: zod_1.z.number().min(0).max(1).describe("Aggregate score"),
    gauntlet_results: zod_1.z.array(exports.GauntletResult).describe("Individual results"),
    execution_time_ms: zod_1.z.number().describe("Total execution time"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * ICR Pattern Type Enum (V2.0)
 */
exports.ICRPatternType = zod_1.z.enum([
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
/**
 * ICR Pattern Schema (V2.0)
 */
exports.ICRPattern = zod_1.z.object({
    pattern_id: zod_1.z.string().describe("Pattern identifier"),
    pattern_type: exports.ICRPatternType.describe("Type of pattern"),
    context: zod_1.z.record(zod_1.z.any()).describe("Pattern context"),
    passed: zod_1.z.boolean().describe("Whether pattern passed"),
    metrics: zod_1.z.record(zod_1.z.any()).optional().describe("Pattern metrics"),
    timestamp: zod_1.z.string().datetime().describe("When pattern was recorded"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * ICR Prediction Schema (V2.0)
 */
exports.ICRPrediction = zod_1.z.object({
    pattern_type: exports.ICRPatternType.describe("Type of pattern"),
    predicted_outcome: zod_1.z.boolean().describe("Predicted outcome"),
    confidence: zod_1.z.number().min(0).max(1).describe("Confidence in prediction"),
    recommended_action: zod_1.z.string().describe("Recommended action"),
    pattern_count: zod_1.z.number().optional().describe("Number of patterns used"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Pattern Cluster Schema (V2.0)
 */
exports.PatternCluster = zod_1.z.object({
    cluster_id: zod_1.z.string().describe("Cluster identifier"),
    pattern_type: exports.ICRPatternType.describe("Type of patterns in cluster"),
    patterns: zod_1.z.array(zod_1.z.lazy(() => exports.ICRPattern)).describe("Patterns in cluster"),
    centroid: zod_1.z.record(zod_1.z.any()).describe("Cluster centroid"),
    similarity_score: zod_1.z.number().min(0).max(1).describe("Cluster similarity"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * ICR Pattern Insights Schema (V2.0)
 */
exports.ICRPatternInsights = zod_1.z.object({
    available: zod_1.z.boolean().describe("Whether ICR is available"),
    pattern_types: zod_1.z.record(zod_1.z.object({
        count: zod_1.z.number().describe("Number of patterns"),
        pass_rate: zod_1.z.number().min(0).max(1).describe("Pass rate"),
        confidence: zod_1.z.number().min(0).max(1).describe("Confidence"),
        recent_patterns: zod_1.z.array(zod_1.z.any()).optional().describe("Recent patterns"),
    })).optional().describe("Statistics by pattern type"),
    total_patterns: zod_1.z.number().optional().describe("Total patterns across all types"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Chart Type Enum (V2.0 - UI)
 */
exports.ChartType = zod_1.z.enum([
    'RADAR',
    'BAR',
    'LINE',
    'PIE',
    'TIMELINE',
    'SCATTER',
    'HEATMAP',
]);
/**
 * UI Chart Data Schema (V2.0)
 */
exports.UIChartData = zod_1.z.object({
    chart_type: exports.ChartType.describe("Type of chart"),
    title: zod_1.z.string().describe("Chart title"),
    labels: zod_1.z.array(zod_1.z.string()).optional().describe("Chart labels"),
    datasets: zod_1.z.array(zod_1.z.object({
        label: zod_1.z.string().optional(),
        data: zod_1.z.array(zod_1.z.union([zod_1.z.number(), zod_1.z.string(), zod_1.z.boolean(), zod_1.z.array(zod_1.z.any())])),
        backgroundColor: zod_1.z.array(zod_1.z.string()).optional(),
        borderColor: zod_1.z.array(zod_1.z.string()).optional(),
    })).optional().describe("Chart datasets"),
    data: zod_1.z.record(zod_1.z.any()).optional().describe("Raw chart data"),
    options: zod_1.z.record(zod_1.z.any()).optional().describe("Chart options"),
    recommendations: zod_1.z.array(zod_1.z.string()).optional().describe("Recommendations based on chart"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Workflow Timeline Schema (V2.0 - UI)
 */
exports.WorkflowTimeline = zod_1.z.object({
    chart_type: exports.ChartType.describe("Chart type"),
    workflow_id: zod_1.z.string().describe("Workflow identifier"),
    stages: zod_1.z.array(zod_1.z.object({
        stage: zod_1.z.string().describe("Stage name"),
        status: zod_1.z.enum(['pending', 'in_progress', 'completed', 'failed', 'skipped']),
        duration_ms: zod_1.z.number().optional().describe("Stage duration"),
        start_time: zod_1.z.string().datetime().optional(),
        end_time: zod_1.z.string().datetime().optional(),
    })).describe("Workflow stages"),
    total_duration_ms: zod_1.z.number().describe("Total duration"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Adapter Health Status Schema (V2.0)
 */
exports.AdapterHealthStatus = zod_1.z.object({
    overall_status: zod_1.z.enum(['healthy', 'degraded', 'unhealthy']).describe("Overall health"),
    components: zod_1.z.record(zod_1.z.object({
        status: zod_1.z.enum(['healthy', 'degraded', 'unhealthy', 'disabled']),
        last_check: zod_1.z.string().datetime().optional(),
        error: zod_1.z.string().optional(),
    })).describe("Component health"),
    uptime_ms: zod_1.z.number().optional().describe("Uptime in milliseconds"),
    alerts: zod_1.z.array(zod_1.z.object({
        severity: zod_1.z.enum(['info', 'warning', 'error', 'critical']),
        message: zod_1.z.string(),
        timestamp: zod_1.z.string().datetime(),
    })).optional().describe("Active alerts"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Cache Statistics Schema (V2.0 - Performance)
 */
exports.CacheStatistics = zod_1.z.object({
    size: zod_1.z.number().describe("Current cache size"),
    max_size: zod_1.z.number().describe("Maximum cache size"),
    ttl: zod_1.z.number().describe("Time-to-live in seconds"),
    total_hits: zod_1.z.number().describe("Total cache hits"),
    total_misses: zod_1.z.number().describe("Total cache misses"),
    hit_rate: zod_1.z.number().min(0).max(1).describe("Cache hit rate"),
    evictions: zod_1.z.number().optional().describe("Number of evictions"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Performance Metrics Schema (V2.0)
 */
exports.PerformanceMetrics = zod_1.z.object({
    operation: zod_1.z.string().describe("Operation name"),
    count: zod_1.z.number().describe("Number of operations"),
    avg_ms: zod_1.z.number().describe("Average latency in ms"),
    min_ms: zod_1.z.number().describe("Minimum latency in ms"),
    max_ms: zod_1.z.number().describe("Maximum latency in ms"),
    p50_ms: zod_1.z.number().describe("P50 latency in ms"),
    p95_ms: zod_1.z.number().describe("P95 latency in ms"),
    p99_ms: zod_1.z.number().describe("P99 latency in ms"),
    throughput_per_sec: zod_1.z.number().optional().describe("Operations per second"),
    error_rate: zod_1.z.number().min(0).max(1).optional().describe("Error rate"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Async Operation Status Enum (V2.0)
 */
exports.AsyncOperationStatus = zod_1.z.enum([
    'pending',
    'scheduled',
    'running',
    'completed',
    'failed',
    'cancelled',
]);
/**
 * Async Operation Schema (V2.0)
 */
exports.AsyncOperation = zod_1.z.object({
    operation_id: zod_1.z.string().describe("Operation identifier"),
    operation_type: zod_1.z.string().describe("Type of operation"),
    status: exports.AsyncOperationStatus.describe("Operation status"),
    input_data: zod_1.z.record(zod_1.z.any()).optional().describe("Input data"),
    result: zod_1.z.any().optional().describe("Operation result"),
    error: zod_1.z.string().optional().describe("Error message if failed"),
    created_at: zod_1.z.string().datetime().describe("Creation timestamp"),
    started_at: zod_1.z.string().datetime().optional().describe("Start timestamp"),
    completed_at: zod_1.z.string().datetime().optional().describe("Completion timestamp"),
    duration_ms: zod_1.z.number().optional().describe("Duration in milliseconds"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Batch Operation Schema (V2.0)
 */
exports.BatchOperation = zod_1.z.object({
    batch_id: zod_1.z.string().describe("Batch identifier"),
    operations: zod_1.z.array(exports.AsyncOperation).describe("Operations in batch"),
    total_operations: zod_1.z.number().describe("Total number of operations"),
    completed_operations: zod_1.z.number().describe("Completed operations"),
    failed_operations: zod_1.z.number().describe("Failed operations"),
    max_concurrency: zod_1.z.number().optional().describe("Max concurrent operations"),
    status: exports.AsyncOperationStatus.describe("Batch status"),
    created_at: zod_1.z.string().datetime(),
    completed_at: zod_1.z.string().datetime().optional(),
    total_duration_ms: zod_1.z.number().optional(),
});
/**
 * Additional System Type Enum (V2.0)
 */
exports.AdditionalSystemType = zod_1.z.enum([
    'crewai',
    'mcp_tools',
    'knowledge_engine',
    'leanaide',
    'z3_prover',
]);
/**
 * System Health Schema (V2.0)
 */
exports.SystemHealth = zod_1.z.object({
    system: exports.AdditionalSystemType.describe("System identifier"),
    available: zod_1.z.boolean().describe("Whether system is available"),
    status: zod_1.z.enum(['healthy', 'degraded', 'unhealthy', 'disabled']).describe("System status"),
    last_check: zod_1.z.string().datetime().optional().describe("Last health check"),
    reason: zod_1.z.string().optional().describe("Reason if unavailable"),
    capabilities: zod_1.z.array(zod_1.z.string()).optional().describe("System capabilities"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Unified System Health Schema (V2.0)
 */
exports.UnifiedSystemHealth = zod_1.z.object({
    overall_status: zod_1.z.enum(['healthy', 'degraded', 'unhealthy']).describe("Overall status"),
    total_systems: zod_1.z.number().describe("Total number of systems"),
    available_systems: zod_1.z.number().describe("Number of available systems"),
    systems: zod_1.z.record(exports.SystemHealth).describe("Individual system health"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Workflow Step Schema (V2.0)
 */
exports.WorkflowStep = zod_1.z.object({
    step: zod_1.z.number().describe("Step number"),
    system: exports.AdditionalSystemType.optional().describe("System used"),
    action: zod_1.z.string().describe("Action performed"),
    success: zod_1.z.boolean().describe("Whether step succeeded"),
    result: zod_1.z.any().optional().describe("Step result"),
    error: zod_1.z.string().optional().describe("Error if failed"),
    duration_ms: zod_1.z.number().optional().describe("Step duration"),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Cross-System Workflow Result Schema (V2.0)
 */
exports.CrossSystemWorkflowResult = zod_1.z.object({
    workflow_type: zod_1.z.string().describe("Type of workflow"),
    success: zod_1.z.boolean().describe("Whether workflow succeeded"),
    steps: zod_1.z.array(exports.WorkflowStep).describe("Workflow steps"),
    result_count: zod_1.z.number().optional().describe("Number of results"),
    result: zod_1.z.any().optional().describe("Final result"),
    error: zod_1.z.string().optional().describe("Error if failed"),
    total_duration_ms: zod_1.z.number().optional().describe("Total duration"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * V2.0 Validation Functions
 */
function validateProblemDecomposition(data) {
    const result = exports.ProblemDecompositionResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateTeamSelection(data) {
    const result = exports.TeamSelectionResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateGauntletPipelineResult(data) {
    const result = exports.GauntletPipelineResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateICRPattern(data) {
    const result = exports.ICRPattern.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateCrossSystemWorkflowResult(data) {
    const result = exports.CrossSystemWorkflowResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
//# sourceMappingURL=adaptive-mdap-canonical.js.map