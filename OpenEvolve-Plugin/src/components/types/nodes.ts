// @ts-nocheck
/**
 * OpenEvolve Node Type System
 *
 * Comprehensive TypeScript type definitions for all OpenEvolve workflow nodes.
 * These types are used by the BubbleLab plugin and throughout the OpenEvolve system.
 *
 * @module OpenEvolveNodeTypes
 * @version 1.0.0
 */

// ============================================================================
// BASE NODE TYPES
// ============================================================================

/**
 * Universal node status across all OpenEvolve nodes
 */
export type NodeStatus =
  | 'idle'           // Node is ready to execute
  | 'running'        // Node is currently executing
  | 'completed'      // Node execution completed successfully
  | 'failed'         // Node execution failed
  | 'paused'         // Node execution is paused
  | 'cancelled';     // Node execution was cancelled

/**
 * Base interface for all OpenEvolve nodes
 * All specific node types extend this interface
 */
export interface OpenEvolveNode<TConfig extends NodeConfig = NodeConfig> {
  /**
   * Unique identifier for the node instance
   */
  id: string;

  /**
   * Type identifier for the node (e.g., 'decomposition', 'evolution')
   */
  type: string;

  /**
   * Human-readable name for the node
   */
  name: string;

  /**
   * Detailed description of what the node does
   */
  description: string;

  /**
   * Node configuration parameters
   */
  config: TConfig;

  /**
   * Current execution status
   */
  status: NodeStatus;

  /**
   * Execution metadata
   */
  metadata: NodeMetadata;

  /**
   * Execute the node with given inputs
   * @param inputs - Input data for node execution
   * @param context - Execution context with callbacks
   * @returns Promise resolving to execution result
   */
  execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;

  /**
   * Validate node configuration
   * @returns Validation result with any errors
   */
  validateConfig(): ValidationResult;

  /**
   * Reset node to initial state
   */
  reset(): void;
}

/**
 * Base configuration interface for all nodes
 */
export interface NodeConfig {
  /**
   * Whether the node is enabled
   */
  enabled: boolean;

  /**
   * Maximum execution time in seconds
   */
  timeout: number;

  /**
   * Maximum number of retry attempts
   */
  maxRetries: number;

  /**
   * Delay between retries in milliseconds
   */
  retryDelay: number;

  /**
   * Whether to continue workflow on failure
   */
  continueOnFailure: boolean;

  /**
   * Custom configuration parameters
   */
  [key: string]: unknown;
}

/**
 * Node execution metadata
 */
export interface NodeMetadata {
  /**
   * Timestamp when node was created
   */
  createdAt: Date;

  /**
   * Timestamp when node was last updated
   */
  updatedAt: Date;

  /**
   * Timestamp when node was last executed
   */
  lastExecutedAt?: Date;

  /**
   * Total number of times node has been executed
   */
  executionCount: number;

  /**
   * Number of successful executions
   */
  successCount: number;

  /**
   * Number of failed executions
   */
  failureCount: number;

  /**
   * Average execution time in milliseconds
   */
  averageExecutionTime: number;

  /**
   * Node version
   */
  version: string;

  /**
   * Tags for categorization
   */
  tags: string[];

  /**
   * Custom metadata properties
   */
  [key: string]: unknown;
}

/**
 * Execution context passed to all nodes
 * Provides callbacks for progress tracking, logging, etc.
 */
export interface ExecutionContext {
  /**
   * Update execution progress
   * @param percent - Progress percentage (0-100)
   * @param message - Progress message
   */
  updateProgress(percent: number, message: string): void;

  /**
   * Add an artifact to the execution result
   * @param type - Artifact type identifier
   * @param data - Artifact data
   */
  addArtifact(type: string, data: unknown): void;

  /**
   * Retrieve an artifact by type
   * @param type - Artifact type identifier
   * @returns Artifact data or null if not found
   */
  getArtifact(type: string): unknown | null;

  /**
   * Log a message at specified level
   * @param level - Log level (debug, info, warn, error)
   * @param message - Log message
   */
  log(level: LogLevel, message: string): void;

  /**
   * Check if execution should be cancelled
   * @returns True if execution should stop
   */
  isCancelled(): boolean;

  /**
   * Get a shared value from context
   * @param key - Value key
   * @returns Value or undefined
   */
  getValue(key: string): unknown | undefined;

  /**
   * Set a shared value in context
   * @param key - Value key
   * @param value - Value to store
   */
  setValue(key: string, value: unknown): void;

  /**
   * Get context-specific configuration
   * @returns Context configuration object
   */
  getConfig(): Record<string, unknown>;
}

/**
 * Log levels for execution context
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Result returned by node execution
 */
export interface NodeResult<TData = unknown> {
  /**
   * Whether execution was successful
   */
  success: boolean;

  /**
   * Primary output data
   */
  data: TData;

  /**
   * Artifacts generated during execution
   */
  artifacts?: Record<string, unknown>;

  /**
   * Execution metrics
   */
  metrics?: NodeMetrics;

  /**
   * Error details if execution failed
   */
  error?: NodeError;

  /**
   * Additional result metadata
   */
  metadata?: Record<string, unknown>;
}

/**
 * Metrics collected during node execution
 */
export interface NodeMetrics {
  /**
   * Execution time in milliseconds
   */
  executionTime: number;

  /**
   * Memory usage in bytes
   */
  memoryUsage: number;

  /**
   * Number of API calls made
   */
  apiCalls: number;

  /**
   * Number of tokens used (for LLM operations)
   */
  tokensUsed: number;

  /**
   * Number of cache hits
   */
  cacheHits: number;

  /**
   * Number of cache misses
   */
  cacheMisses: number;

  /**
   * Custom metrics
   */
  [key: string]: number | string | boolean;
}

/**
 * Error details from node execution
 */
export interface NodeError {
  /**
   * Error code or identifier
   */
  code: string;

  /**
   * Human-readable error message
   */
  message: string;

  /**
   * Detailed error description
   */
  details?: string;

  /**
   * Stack trace if available
   */
  stack?: string;

  /**
   * Error severity
   */
  severity: ErrorSeverity;

  /**
   * Whether error is recoverable
   */
  recoverable: boolean;

  /**
   * Suggested recovery actions
   */
  recoveryActions?: string[];

  /**
   * Original error object
   */
  originalError?: Error;
}

/**
 * Error severity levels
 */
export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

/**
 * Validation result for node configuration
 */
export interface ValidationResult {
  /**
   * Whether validation passed
   */
  valid: boolean;

  /**
   * Validation errors
   */
  errors: ValidationError[];

  /**
   * Validation warnings
   */
  warnings: ValidationWarning[];

  /**
   * Improvement suggestions
   */
  suggestions: string[];
}

/**
 * Validation error details
 */
export interface ValidationError {
  /**
   * Error code
   */
  code: string;

  /**
   * Error message
   */
  message: string;

  /**
   * Parameter path that failed validation
   */
  path: string;

  /**
   * Expected value/type
   */
  expected?: string;

  /**
   * Actual value received
   */
  actual?: string;
}

/**
 * Validation warning details
 */
export interface ValidationWarning {
  /**
   * Warning code
   */
  code: string;

  /**
   * Warning message
   */
  message: string;

  /**
   * Parameter path
   */
  path: string;

  /**
   * Suggested improvement
   */
  suggestion?: string;
}

// ============================================================================
// INPUT/OUTPUT TYPES
// ============================================================================

/**
 * Generic node inputs
 * All node-specific inputs extend this type
 */
export interface NodeInputs {
  /**
   * Primary input data
   */
  data: unknown;

  /**
   * Additional input parameters
   */
  parameters?: Record<string, unknown>;

  /**
   * Input artifacts from previous nodes
   */
  artifacts?: Record<string, unknown>;
}

/**
 * Generic node outputs
 * All node-specific outputs extend this type
 */
export interface NodeOutputs {
  /**
   * Primary output data
   */
  data: unknown;

  /**
   * Output artifacts
   */
  artifacts?: Record<string, unknown>;

  /**
   * Output metadata
   */
  metadata?: Record<string, unknown>;
}

// ============================================================================
// SPECIFIC NODE CONFIGURATION TYPES
// ============================================================================

/**
 * Configuration for DecompositionNode
 * Handles problem decomposition into sub-problems
 */
export interface DecompositionNodeConfig extends NodeConfig {
  /**
   * Decomposition strategy to use
   */
  strategy: DecompositionStrategy;

  /**
   * Maximum number of sub-problems to generate
   */
  maxSubProblems: number;
  /**
   * Recursion depth limit (0 = unlimited)
   */
  recursionDepthLimit: number;

  /**
   * Minimum sub-problem complexity score
   */
  minComplexity: number;

  /**
   * Maximum sub-problem complexity score
   */
  maxComplexity: number;

  /**
   * Whether to analyze dependencies between sub-problems
   */
  analyzeDependencies: boolean;

  /**
   * Whether to estimate time for each sub-problem
   */
  estimateTime: boolean;

  /**
   * Granularity level (coarse, medium, fine)
   */
  granularity: 'coarse' | 'medium' | 'fine';

  /**
   * Whether to validate decomposed problems
   */
  validateDecomposition: boolean;

  /**
   * Knowledge base sources to consult
   */
  knowledgeBaseSources: string[];

  /**
   * Domain-specific analysis enabled
   */
  domainSpecificAnalysis: boolean;
}

/**
 * Configuration for SubProblemNode
 * Handles individual sub-problem solving
 */
export interface SubProblemNodeConfig extends NodeConfig {
  /**
   * Sub-problem identifier
   */
  subProblemId: string;

  /**
   * Priority level (1-10)
   */
  priority: number;

  /**
   * Complexity score (0-1)
   */
  complexity: number;

  /**
   * Estimated time in seconds
   */
  estimatedTime: number;

  /**
   * Dependencies on other sub-problems
   */
  dependencies: string[];

  /**
   * Required skills/capabilities
   */
  requiredCapabilities: string[];

  /**
   * Maximum solution attempts
   */
  maxAttempts: number;

  /**
   * Quality threshold (0-1)
   */
  qualityThreshold: number;

  /**
   * Whether to use adversarial testing
   */
  useAdversarialTesting: boolean;

  /**
   * Whether to use evolutionary optimization
   */
  useEvolutionaryOptimization: boolean;
}

/**
 * Configuration for GauntletNode
 * Handles multi-stage validation and testing
 */
export interface GauntletNodeConfig extends NodeConfig {
  /**
   * Number of validation stages
   */
  stages: number;

  /**
   * Validation stages configuration
   */
  stageConfigs: ValidationStageConfig[];

  /**
   * Whether to use progressive validation
   */
  progressiveValidation: boolean;

  /**
   * Minimum score to pass each stage (0-1)
   */
  minStageScore: number;

  /**
   * Whether to stop on first failure
   */
  stopOnFailure: boolean;

  /**
   * Whether to generate detailed reports
   */
  detailedReports: boolean;

  /**
   * Validation strictness (lenient, medium, strict)
   */
  strictness: 'lenient' | 'medium' | 'strict';

  /**
   * Custom validation criteria
   */
  customCriteria: Record<string, unknown>;
}

/**
 * Configuration for SolutionNode
 * Handles solution generation and refinement
 */
export interface SolutionNodeConfig extends NodeConfig {
  /**
   * Solution generation method
   */
  method: SolutionMethod;

  /**
   * Maximum number of solution iterations
   */
  maxIterations: number;

  /**
   * Population size for evolutionary methods
   */
  populationSize: number;

  /**
   * Mutation rate (0-1)
   */
  mutationRate: number;

  /**
   * Crossover rate (0-1)
   */
  crossoverRate: number;

  /**
   * Elitism count
   */
  elitismCount: number;

  /**
   * Diversity maintenance enabled
   */
  diversityMaintenance: boolean;

  /**
   * Convergence threshold (0-1)
   */
  convergenceThreshold: number;

  /**
   * Whether to use adversarial improvement
   */
  useAdversarialImprovement: boolean;

  /**
   * Number of adversarial rounds
   */
  adversarialRounds: number;

  /**
   * Quality metrics to optimize
   */
  qualityMetrics: string[];
}

/**
 * Configuration for VerificationNode
 * Handles solution verification and validation
 */
export interface VerificationNodeConfig extends NodeConfig {
  /**
   * Verification criteria
   */
  criteria: VerificationCriteria[];

  /**
   * Verification method
   */
  method: VerificationMethod;

  /**
   * Strictness level (1-10)
   */
  strictness: number;

  /**
   * Whether to use automated testing
   */
  automatedTesting: boolean;

  /**
   * Whether to use manual review
   */
  manualReview: boolean;

  /**
   * Test coverage threshold (0-1)
   */
  testCoverageThreshold: number;

  /**
   * Performance benchmarks
   */
  performanceBenchmarks: PerformanceBenchmark[];

  /**
   * Security checks enabled
   */
  securityChecks: boolean;

  /**
   * Compliance standards to verify
   */
  complianceStandards: string[];
}

/**
 * Configuration for AssemblyNode
 * Handles solution assembly and integration
 */
export interface AssemblyNodeConfig extends NodeConfig {
  /**
   * Assembly strategy
   */
  strategy: AssemblyStrategy;

  /**
   * Integration method
   */
  integrationMethod: IntegrationMethod;

  /**
   * Conflict resolution strategy
   */
  conflictResolution: ConflictResolutionStrategy;

  /**
   * Whether to optimize assembly
   */
  optimizeAssembly: boolean;

  /**
   * Optimization objectives
   */
  optimizationObjectives: string[];

  /**
   * Whether to validate integrated solution
   */
  validateIntegration: boolean;

  /**
   * Integration testing enabled
   */
  integrationTesting: boolean;

  /**
   * Maximum assembly iterations
   */
  maxIterations: number;

  /**
   * Whether to generate assembly documentation
   */
  generateDocumentation: boolean;
}

/**
 * Configuration for OutputNode
 * Handles final output formatting and delivery
 */
export interface OutputNodeConfig extends NodeConfig {
  /**
   * Output format
   */
  format: OutputFormat;

  /**
   * Output template
   */
  template?: string;

  /**
   * Whether to include metadata
   */
  includeMetadata: boolean;

  /**
   * Whether to include metrics
   */
  includeMetrics: boolean;

  /**
   * Whether to include artifacts
   */
  includeArtifacts: boolean;

  /**
   * Output destination
   */
  destination: OutputDestination;

  /**
   * Output compression
   */
  compression?: 'none' | 'gzip' | 'zip';

  /**
   * Whether to sign output
   */
  signOutput: boolean;

  /**
   * Custom formatting options
   */
  formattingOptions: Record<string, unknown>;
}

/**
 * Configuration for KnowledgeExtractionNode
 * Handles knowledge extraction and integration
 */
export interface KnowledgeExtractionNodeConfig extends NodeConfig {
  /**
   * Extraction sources
   */
  sources: KnowledgeSource[];

  /**
   * Extraction method
   */
  method: ExtractionMethod;

  /**
   * Knowledge domain
   */
  domain: string;

  /**
   * Whether to validate extracted knowledge
   */
  validateKnowledge: boolean;

  /**
   * Confidence threshold (0-1)
   */
  confidenceThreshold: number;

  /**
   * Whether to update knowledge base
   */
  updateKnowledgeBase: boolean;

  /**
   * Maximum extraction depth
   */
  maxDepth: number;

  /**
   * Extraction filters
   */
  filters: ExtractionFilter[];

  /**
   * Whether to use knowledge graphs
   */
  useKnowledgeGraphs: boolean;

  /**
   * Graph traversal strategy
   */
  graphTraversalStrategy?: 'breadth' | 'depth' | 'best';
}

// ============================================================================
// DATA TYPES
// ============================================================================

/**
 * Sub-problem in decomposition
 */
export interface SubProblem {
  /**
   * Unique identifier
   */
  id: string;

  /**
   * Sub-problem title
   */
  title: string;

  /**
   * Detailed description
   */
  description: string;

  /**
   * Priority level (1-10)
   */
  priority: number;

  /**
   * Complexity score (0-1)
   */
  complexity: number;

  /**
   * Dependencies on other sub-problems
   */
  dependencies: string[];

  /**
   * Estimated time in seconds
   */
  estimated_time: number;

  /**
   * Current status
   */
  status: SubProblemStatus;

  /**
   * Required capabilities
   */
  required_capabilities: string[];

  /**
   * Success criteria
   */
  success_criteria: string[];

  /**
   * Additional metadata
   */
  metadata?: Record<string, unknown>;
}

/**
 * Sub-problem status
 */
export type SubProblemStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'blocked'
  | 'cancelled';

/**
 * Solution result from solution generation
 */
export interface SolutionResult<TSolution = unknown> {
  /**
   * The generated solution
   */
  solution: TSolution;

  /**
   * Confidence score (0-1)
   */
  confidence: number;

  /**
   * Quality score (0-1)
   */
  quality_score: number;

  /**
   * Generation method used
   */
  generation_method: string;

  /**
   * Number of iterations used
   */
  iterations_used: number;

  /**
   * Alternative solutions generated
   */
  alternative_solutions: TSolution[];

  /**
   * Quality metrics breakdown
   */
  quality_metrics: Record<string, number>;

  /**
   * Generation metadata
   */
  metadata: SolutionMetadata;
}

/**
 * Solution generation metadata
 */
export interface SolutionMetadata {
  /**
   * Generation timestamp
   */
  timestamp: Date;

  /**
   * Generation duration in milliseconds
   */
  duration: number;

  /**
   * Algorithm version
   */
  algorithm_version: string;

  /**
   * Parameters used
   */
  parameters: Record<string, unknown>;

  /**
   * Performance metrics
   */
  performance: Record<string, number>;
}

/**
 * Verification result
 */
export interface VerificationResult {
  /**
   * Whether verification passed
   */
  passed: boolean;

  /**
   * Overall score (0-1)
   */
  score: number;

  /**
   * Individual criteria results
   */
  criteria_results: CriteriaResult[];

  /**
   * Issues found
   */
  issues: VerificationIssue[];

  /**
   * Warnings
   */
  warnings: string[];

  /**
   * Recommendations
   */
  recommendations: string[];

  /**
   * Verification metadata
   */
  metadata: Record<string, unknown>;
}

/**
 * Individual criterion verification result
 */
export interface CriteriaResult {
  /**
   * Criterion name
   */
  criterion: string;

  /**
   * Whether criterion passed
   */
  passed: boolean;

  /**
   * Score (0-1)
   */
  score: number;

  /**
   * Details
   */
  details: string;

  /**
   * Evidence
   */
  evidence: string[];
}

/**
 * Verification issue
 */
export interface VerificationIssue {
  /**
   * Issue severity
   */
  severity: 'low' | 'medium' | 'high' | 'critical';

  /**
   * Issue type
   */
  type: string;

  /**
   * Issue description
   */
  description: string;

  /**
   * Location in solution
   */
  location?: string;

  /**
   * Suggested fix
   */
  suggested_fix?: string;
}

/**
 * Knowledge extraction result
 */
export interface KnowledgeExtractionResult {
  /**
   * Extracted knowledge entities
   */
  entities: KnowledgeEntity[];

  /**
   * Extracted relationships
   */
  relationships: KnowledgeRelationship[];

  /**
   * Confidence score (0-1)
   */
  confidence: number;

  /**
   * Extraction metadata
   */
  metadata: ExtractionMetadata;
}

/**
 * Knowledge entity
 */
export interface KnowledgeEntity {
  /**
   * Entity identifier
   */
  id: string;

  /**
   * Entity type
   */
  type: string;

  /**
   * Entity properties
   */
  properties: Record<string, unknown>;

  /**
   * Confidence score (0-1)
   */
  confidence: number;

  /**
   * Source reference
   */
  source: string;
}

/**
 * Knowledge relationship
 */
export interface KnowledgeRelationship {
  /**
   * Relationship identifier
   */
  id: string;

  /**
   * Source entity ID
   */
  source: string;

  /**
   * Target entity ID
   */
  target: string;

  /**
   * Relationship type
   */
  type: string;

  /**
   * Relationship properties
   */
  properties: Record<string, unknown>;

  /**
   * Confidence score (0-1)
   */
  confidence: number;
}

/**
 * Extraction metadata
 */
export interface ExtractionMetadata {
  /**
   * Extraction timestamp
   */
  timestamp: Date;

  /**
   * Extraction duration in milliseconds
   */
  duration: number;

  /**
   * Sources processed
   */
  sources_processed: string[];

  /**
   * Extraction method used
   */
  method: string;

  /**
   * Model version
   */
  model_version?: string;
}

// ============================================================================
// PARAMETER SCHEMA TYPES
// ============================================================================

/**
 * Parameter schema definition
 * Used for defining and validating node parameters
 */
export interface ParameterSchema {
  /**
   * Parameter type
   */
  type: ParameterType;

  /**
   * Parameter title
   */
  title: string;

  /**
   * Parameter description
   */
  description?: string;

  /**
   * For object types, property definitions
   */
  properties?: Record<string, ParameterDefinition>;

  /**
   * Required property names
   */
  required?: string[];

  /**
   * Enumerated values
   */
  enum?: (string | number | boolean)[];

  /**
   * Default value
   */
  default?: unknown;

  /**
   * Minimum value (for numbers)
   */
  minimum?: number;

  /**
   * Maximum value (for numbers)
   */
  maximum?: number;

  /**
   * Minimum length (for strings/arrays)
   */
  minLength?: number;

  /**
   * Maximum length (for strings/arrays)
   */
  maxLength?: number;

  /**
   * Pattern (for strings, regex)
   */
  pattern?: string;

  /**
   * Whether parameter is nullable
   */
  nullable?: boolean;

  /**
   * Custom validation rules
   */
  validation?: ValidationRule[];

  /**
   * UI hints
   */
  ui?: ParameterUIHints;
}

/**
 * Parameter type
 */
export type ParameterType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'object'
  | 'array'
  | 'enum'
  | 'any';

/**
 * Parameter definition
 */
export interface ParameterDefinition extends ParameterSchema {
  /**
   * Whether parameter is required
   */
  required: boolean;
}

/**
 * Validation rule
 */
export interface ValidationRule {
  /**
   * Rule name
   */
  name: string;

  /**
   * Rule condition (expression)
   */
  condition: string;

  /**
   * Error message
   */
  message: string;

  /**
   * Rule parameters
   */
  params?: Record<string, unknown>;
}

/**
 * Parameter UI hints
 */
export interface ParameterUIHints {
  /**
   * UI widget type
   */
  widget?: 'input' | 'textarea' | 'select' | 'checkbox' | 'slider' | 'color';

  /**
   * Placeholder text
   */
  placeholder?: string;

  /**
   * Help text
   */
  help?: string;

  /**
   * Field group
   */
  group?: string;

  /**
   * Display order
   */
  order?: number;

  /**
   * Whether to hide in UI
   */
  hidden?: boolean;

  /**
   * Custom widget properties
   */
  widgetProps?: Record<string, unknown>;
}

// ============================================================================
// ENUMERATED TYPES AND CONSTANTS
// ============================================================================

/**
 * Decomposition strategies
 */
export type DecompositionStrategy =
  | 'semantic'        // Natural language understanding based
  | 'hierarchical'    // Hierarchical breakdown
  | 'functional'      // Functional decomposition
  | 'modular'         // Module-based decomposition
  | 'temporal'        // Time-based decomposition
  | 'hybrid';         // Combination of strategies

/**
 * Solution generation methods
 */
export type SolutionMethod =
  | 'evolutionary'    // Evolutionary algorithms
  | 'analytical'      // Analytical methods
  | 'heuristic'       // Heuristic search
  | 'neural'          // Neural network based
  | 'hybrid';         // Hybrid approach

/**
 * Verification methods
 */
export type VerificationMethod =
  | 'automated'       // Fully automated testing
  | 'manual'          // Manual review
  | 'hybrid';         // Combination

/**
 * Assembly strategies
 */
export type AssemblyStrategy =
  | 'sequential'      // Sequential assembly
  | 'parallel'        // Parallel assembly
  | 'hierarchical'    // Hierarchical assembly
  | 'adaptive';       // Adaptive assembly

/**
 * Integration methods
 */
export type IntegrationMethod =
  | 'merge'           // Direct merge
  | 'compose'         // Composition
  | 'aggregate'       // Aggregation
  | 'pipeline';       // Pipeline integration

/**
 * Conflict resolution strategies
 */
export type ConflictResolutionStrategy =
  | 'priority'        // Priority-based
  | 'merge'           // Merge conflicts
  | 'voting'          // Voting mechanism
  | 'custom';         // Custom resolution

/**
 * Output formats
 */
export type OutputFormat =
  | 'json'            // JSON format
  | 'xml'             // XML format
  | 'yaml'            // YAML format
  | 'markdown'        // Markdown format
  | 'html'            // HTML format
  | 'text'            // Plain text
  | 'binary';         // Binary format

/**
 * Output destinations
 */
export type OutputDestination =
  | 'file'            // File system
  | 'api'             // API endpoint
  | 'database'        // Database
  | 'message_queue'   // Message queue
  | 'stream';         // Stream output

/**
 * Knowledge sources
 */
export type KnowledgeSourceType =
  | 'text'            // Text documents
  | 'database'        // Database
  | 'api'             // API
  | 'file'            // File system
  | 'graph'           // Knowledge graph
  | 'semantic'        // Semantic web
  | 'custom';         // Custom source

/**
 * Extraction methods
 */
export type ExtractionMethod =
  | 'pattern'         // Pattern-based
  | 'ml'              // Machine learning
  | 'llm'             // Large language model
  | 'hybrid';         // Hybrid approach

/**
 * Validation stage configuration
 */
export interface ValidationStageConfig {
  /**
   * Stage name
   */
  name: string;

  /**
   * Stage description
   */
  description?: string;

  /**
   * Validation criteria for this stage
   */
  criteria: VerificationCriteria[];

  /**
   * Stage timeout in seconds
   */
  timeout: number;

  /**
   * Minimum score to pass (0-1)
   */
  minScore: number;

  /**
   * Whether to run in parallel
   */
  parallel: boolean;
}

/**
 * Verification criteria
 */
export interface VerificationCriteria {
  /**
   * Criterion identifier
   */
  id: string;

  /**
   * Criterion name
   */
  name: string;

  /**
   * Criterion description
   */
  description?: string;

  /**
   * Criterion type
   */
  type: string;

  /**
   * Weight in overall score (0-1)
   */
  weight: number;

  /**
   * Threshold value
   */
  threshold?: number;

  /**
   * Validation function
   */
  validator?: string;

  /**
   * Criterion parameters
   */
  params?: Record<string, unknown>;
}

/**
 * Performance benchmark
 */
export interface PerformanceBenchmark {
  /**
   * Benchmark name
   */
  name: string;

  /**
   * Metric to measure
   */
  metric: string;

  /**
   * Target value
   */
  target: number;

  /**
   * Threshold value
   */
  threshold: number;

  /**
   * Unit of measurement
   */
  unit: string;

  /**
   * Comparison operator
   */
  operator: '>' | '<' | '=' | '>=' | '<=';
}

/**
 * Knowledge source
 */
export interface KnowledgeSource {
  /**
   * Source identifier
   */
  id: string;

  /**
   * Source type
   */
  type: KnowledgeSourceType;

  /**
   * Source location/URI
   */
  location: string;

  /**
   * Source configuration
   */
  config: Record<string, unknown>;

  /**
   * Source priority
   */
  priority: number;

  /**
   * Whether source is enabled
   */
  enabled: boolean;
}

/**
 * Extraction filter
 */
export interface ExtractionFilter {
  /**
   * Filter name
   */
  name: string;

  /**
   * Filter type
   */
  type: string;

  /**
   * Filter condition
   */
  condition: string;

  /**
   * Filter parameters
   */
  params: Record<string, unknown>;
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

/**
 * Extract node config type from node type
 */
export type ConfigOf<TNode extends OpenEvolveNode> = TNode extends OpenEvolveNode<infer TConfig> ? TConfig : never;

/**
 * Extract result data type from node result
 */
export type ResultDataOf<TResult extends NodeResult> = TResult extends NodeResult<infer TData> ? TData : never;

/**
 * Make specific properties optional
 */
export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Make specific properties required
 */
export type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

/**
 * Deep partial type
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends Array<infer U>
    ? Array<DeepPartial<U>>
    : T[P] extends ReadonlyArray<infer U>
    ? ReadonlyArray<DeepPartial<U>>
    : T[P] extends object
    ? DeepPartial<T[P]>
    : T[P];
};

/**
 * Node execution options
 */
export interface NodeExecutionOptions {
  /**
   * Whether to enable caching
   */
  enableCache?: boolean;

  /**
   * Whether to enable metrics collection
   */
  enableMetrics?: boolean;

  /**
   * Whether to enable detailed logging
   */
  enableDetailedLogging?: boolean;

  /**
   * Custom execution timeout
   */
  timeout?: number;

  /**
   * Maximum retry attempts
   */
  maxRetries?: number;

  /**
   * Execution metadata
   */
  metadata?: Record<string, unknown>;
}

/**
 * Node registry entry
 */
export interface NodeRegistryEntry {
  /**
   * Node type identifier
   */
  type: string;

  /**
   * Node class constructor
   */
  constructor: new (...args: unknown[]) => OpenEvolveNode;

  /**
   * Node configuration schema
   */
  configSchema: ParameterSchema;

  /**
   * Node category
   */
  category: string;

  /**
   * Node metadata
   */
  metadata: {
    name: string;
    description: string;
    version: string;
    author?: string;
    tags?: string[];
  };
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

/**
 * Type guard for OpenEvolveNode
 */
export function isOpenEvolveNode(value: unknown): value is OpenEvolveNode {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'type' in value &&
    'config' in value &&
    'status' in value &&
    'execute' in value &&
    typeof (value as OpenEvolveNode).execute === 'function'
  );
}

/**
 * Type guard for NodeResult
 */
export function isNodeResult(value: unknown): value is NodeResult {
  return (
    typeof value === 'object' &&
    value !== null &&
    'success' in value &&
    'data' in value
  );
}

/**
 * Type guard for NodeError
 */
export function isNodeError(value: unknown): value is NodeError {
  return (
    typeof value === 'object' &&
    value !== null &&
    'code' in value &&
    'message' in value &&
    'severity' in value
  );
}

/**
 * Type guard for ValidationResult
 */
export function isValidationResult(value: unknown): value is ValidationResult {
  return (
    typeof value === 'object' &&
    value !== null &&
    'valid' in value &&
    'errors' in value &&
    'warnings' in value
  );
}

// ============================================================================
// TYPE GUARDS AND EXPORTS ARE ALREADY DECLARED ABOVE
// ============================================================================
