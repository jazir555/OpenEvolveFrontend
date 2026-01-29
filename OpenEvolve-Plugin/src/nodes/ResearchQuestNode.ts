// @ts-nocheck
/**
 * Research Quest Node - Integration Library Version
 *
 * This node uses the OpenEvolve Integration Library to communicate with
 * the Research-Quest server, which executes the systematic scientific reasoning logic.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { apiClient } from '@/services/api';
import { useAuthStore } from '@/stores/authStore';

/**
 * Research Quest stage types
 */
export type ResearchQuestStage = 'initialization' | 'decomposition' | 'hypothesis_planning' | 'evidence_integration' | 'pruning_merging' | 'subgraph_extraction' | 'composition' | 'reflection';

/**
 * Research Quest parameters (P1.0-P1.29)
 */
export interface ResearchQuestParameters {
  P1_0?: boolean; // Mandatory 8-stage GoT execution
  P1_1?: boolean; // Root node n₀ label='Task Understanding'
  P1_2?: boolean; // Default dimensions: Scope, Objectives, Constraints, etc.
  P1_3?: boolean; // Generate k=3-5 hypotheses per dimension
  P1_4?: boolean; // Iterative loop based on confidence-to-cost ratio
  P1_5?: boolean; // Confidence C = [empirical_support, theoretical_basis, methodological_rigor, consensus_alignment]
  P1_6?: boolean; // Numeric node labels, verbatim queries, reasoning trace
  P1_7?: boolean; // Mandatory self-audit
  P1_8?: boolean; // Maintain explicit disciplinary_tags list
  P1_9?: boolean; // Enable hyperedges Eₕ ⊆ P(V) where |Eₕ| > 2
  P1_10?: boolean; // Classify edges E with mandatory edge_type metadata
  P1_11?: boolean; // Define ASR-GoT graph state Gₜ = (Vₜ, Eₜ∪Eₕₜ, Lₜ, T, Cₜ, Mₜ, Iₜ)
  P1_12?: boolean; // Mandatory metadata for nodes and edges/hyperedges
  P1_13?: boolean; // Identify mutually exclusive hypotheses H_comp
  P1_14?: boolean; // Represent confidence components using probability distributions
  P1_15?: boolean; // Identify gaps: create Placeholder_Gap nodes
  P1_16?: boolean; // Require population of falsification_criteria metadata
  P1_17?: boolean; // Include 'Potential Biases' dimension and assess nodes
  P1_18?: boolean; // Utilize timestamp metadata and apply temporal decay
  P1_19?: boolean; // Generate 'prospective' subgraphs for interventions
  P1_20?: boolean; // Define abstraction levels or utilize formal multi-layer structure
  P1_21?: boolean; // Estimate computational cost of operations
  P1_22?: boolean; // Enable graph restructuring during Stage 4
  P1_23?: boolean; // Define distinct but interconnected layers L
  P1_24?: boolean; // Extend edge types with causal semantics
  P1_25?: boolean; // Extend edge types to support temporal patterns
  P1_26?: boolean; // Add power analysis metrics to statistical_power metadata
  P1_27?: boolean; // Incorporate entropy, KL divergence, mutual information
  P1_28?: boolean; // Develop and apply metrics for theoretical significance
  P1_29?: boolean; // Support node attribution to researchers/specialists
}

/**
 * Research Quest node configuration
 */
export interface ResearchQuestNodeConfig {
  enableMultiLayer?: boolean;
  maxHypotheses?: number;
  enableCausalInference?: boolean;
  enableTemporalAnalysis?: boolean;
  enableBiasAssessment?: boolean;
  enableFalsificationChecks?: boolean;
  enableImpactScoring?: boolean;
  enableInterdisciplinaryBridges?: boolean;
  enableKnowledgeGapDetection?: boolean;
  enableProbabilisticConfidence?: boolean;
  enableGraphRestructuring?: boolean;
  enableTopologyAnalysis?: boolean;
  enableInformationTheoryMetrics?: boolean;
  enableAttributionTracking?: boolean;
  enableStatisticalPowerAnalysis?: boolean;
  enableMultiScaleAnalysis?: boolean;
  enableCostEstimation?: boolean;
  enableSelfAudit?: boolean;
  enableEvidenceIntegration?: boolean;
  enablePruningMerging?: boolean;
  enableSubgraphExtraction?: boolean;
  enableReflection?: boolean;
  enableComposition?: boolean;
  enableHypothesisGeneration?: boolean;
  enableTaskDecomposition?: boolean;
  enableInitialization?: boolean;
  enableBackendExecution?: boolean;
  backendUrl?: string;
  parameters?: ResearchQuestParameters;
}

/**
 * Research Quest result interface
 */
export interface ResearchQuestResult {
  success: boolean;
  node_id?: string;
  message: string;
  current_stage: number;
  stage_name: string;
  dimension_nodes?: string[];
  hypothesis_nodes?: string[];
  active_parameters?: string[];
  warnings?: string[];
  errors?: string[];
  graph_summary?: any;
  reasoning_trace?: any;
  topology_insights?: any;
  export_data?: string;
  partial_success?: boolean;
  recovery_attempted?: boolean;
}

/**
 * Research Quest Node (Integration Library Version)
 *
 * This node uses the OpenEvolve Integration Library to delegate research quest
 * to the Research-Quest server. The Research-Quest server implements the 
 * ASR-GoT (Adaptive Scientific Reasoning Graph-of-Thoughts) framework with
 * 8 stages and 29 parameters as specified in the Research-Quest specification.
 *
 * Benefits of this approach:
 * - Reuses existing Research-Quest server implementation
 * - No need to duplicate logic in TypeScript
 * - Consistent behavior across all clients
 * - Easy to update Research-Quest server without changing frontend
 */
export class ResearchQuestNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Research Quest';
  static readonly DESCRIPTION = 'Systematic scientific reasoning using ASR-GoT framework via Research-Quest server';
  static readonly ICON = 'research';
  static readonly CATEGORY = 'analysis';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: ResearchQuestNodeConfig = {}) {
    super(id, {
      enableMultiLayer: true,
      maxHypotheses: 5,
      enableCausalInference: true,
      enableTemporalAnalysis: true,
      enableBiasAssessment: true,
      enableFalsificationChecks: true,
      enableImpactScoring: true,
      enableInterdisciplinaryBridges: true,
      enableKnowledgeGapDetection: true,
      enableProbabilisticConfidence: true,
      enableGraphRestructuring: true,
      enableTopologyAnalysis: true,
      enableInformationTheoryMetrics: true,
      enableAttributionTracking: true,
      enableStatisticalPowerAnalysis: true,
      enableMultiScaleAnalysis: true,
      enableCostEstimation: true,
      enableSelfAudit: true,
      enableEvidenceIntegration: true,
      enablePruningMerging: true,
      enableSubgraphExtraction: true,
      enableReflection: true,
      enableComposition: true,
      enableHypothesisGeneration: true,
      enableTaskDecomposition: true,
      enableInitialization: true,
      enableBackendExecution: true,
      backendUrl: 'http://localhost:8000',
      parameters: {
        P1_0: true,
        P1_1: true,
        P1_2: true,
        P1_3: true,
        P1_4: true,
        P1_5: true,
        P1_6: true,
        P1_7: true,
        P1_8: true,
        P1_9: true,
        P1_10: true,
        P1_11: true,
        P1_12: true,
        P1_13: true,
        P1_14: true,
        P1_15: true,
        P1_16: true,
        P1_17: true,
        P1_18: true,
        P1_19: true,
        P1_20: true,
        P1_21: true,
        P1_22: true,
        P1_23: true,
        P1_24: true,
        P1_25: true,
        P1_26: true,
        P1_27: true,
        P1_28: true,
        P1_29: true,
      },
      ...config
    });
  }

  /**
   * Execute research quest using the integration library
   *
   * @param inputs - Must contain 'task_description' string and optional parameters
   * @param context - Execution context
   * @returns Promise resolving to research quest result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs with validation
      const taskDescription = inputs.task_description || inputs.task as string;
      const stage = (inputs.stage as ResearchQuestStage) || 'initialization';
      const config = inputs.config as Record<string, any> | undefined;
      const hypotheses = inputs.hypotheses as any[] | undefined;
      const dimensionNodeId = inputs.dimension_node_id as string | undefined;
      const format = inputs.format as string | undefined;

      // Validate that we have a task to process
      if (!taskDescription && stage !== 'get_graph_summary' && stage !== 'export_graph_data') {
        try {
          return this.createErrorResult('Task description is required for initialization');
        } catch (errorResultError) {
          errorLogger.logError(errorResultError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating error result' } });
          return {
            success: false,
            data: { error: 'Task description is required for initialization' },
            error: { message: 'Task description is required for initialization' }
          } as NodeResult;
        }
      }

      context.updateProgress(10, 'Validating inputs');

      // Validate stage
      const validStages: ResearchQuestStage[] = [
        'initialization', 'decomposition', 'hypothesis_planning',
        'evidence_integration', 'pruning_merging', 'subgraph_extraction',
        'composition', 'reflection', 'get_graph_summary', 'export_graph_data'
      ];

      if (!validStages.includes(stage)) {
        try {
          return this.createErrorResult(`Invalid stage: ${stage}. Valid stages: ${validStages.join(', ')}`);
        } catch (errorResultError) {
          errorLogger.logError(errorResultError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating validation error result' } });
          return {
            success: false,
            data: { error: `Invalid stage: ${stage}. Valid stages: ${validStages.join(', ')}` },
            error: { message: `Invalid stage: ${stage}. Valid stages: ${validStages.join(', ')}` }
          } as NodeResult;
        }
      }

      // Use integration library to call Research-Quest server
      if (this.config.enableBackendExecution) {
        try {
          return await this.executeWithBackend(stage, taskDescription, hypotheses, dimensionNodeId, format, config, context);
        } catch (backendError) {
          console.warn('Backend execution failed, falling back to local execution:', backendError);
          context.updateProgress(20, 'Backend unavailable, using local execution');
          return await this.executeLocally(stage, taskDescription, hypotheses, dimensionNodeId, context);
        }
      } else {
        return await this.executeLocally(stage, taskDescription, hypotheses, dimensionNodeId, context);
      }

    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'ResearchQuestNode execution error' } });
      try {
        return this.createErrorResult(
          error instanceof Error ? error.message : 'Unknown error during research quest'
        );
      } catch (errorResultError) {
        errorLogger.logError(errorResultError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating final error result' } });
        return {
          success: false,
          data: { error: error instanceof Error ? error.message : 'Unknown error during research quest' },
          error: { message: error instanceof Error ? error.message : 'Unknown error during research quest' }
        } as NodeResult;
      }
    }
  }

  /**
   * Execute research quest using Research-Quest server via integration library
   */
  private async executeSummary(graphId: string): Promise<NodeResult> {
    try {
      const response = await apiClient.get<ResearchQuestResult>(`/api/openevolve/research/summary/${graphId}`);
      return this.createSuccessResult({
        ...response,
        stage: 'get_graph_summary',
        metadata: { executionTime: 0, stageExecuted: 'get_graph_summary', backendUsed: true }
      });
    } catch (e) {
      return this.createErrorResult(e instanceof Error ? e.message : 'Summary failed');
    }
  }

  /**
   * Execute research quest using Research-Quest server via integration library
   */
  private async executeWithBackend(
    stage: ResearchQuestStage,
    taskDescription: string | undefined,
    hypotheses: any[] | undefined,
    dimensionNodeId: string | undefined,
    format: string | undefined,
    config: Record<string, any> | undefined,
    context: ExecutionContext
  ): Promise<NodeResult> {
    try {
      context.updateProgress(20, 'Connecting to Research-Quest server');

      // Prepare request for backend based on stage
      let endpoint: string;
      let backendInputs: Record<string, any>;

      switch (stage) {
        case 'initialization':
          endpoint = '/api/openevolve/research/initialize';
          backendInputs = {
            task_description: taskDescription,
            initial_confidence: [0.8, 0.8, 0.8, 0.8],
            config: {
              enable_multi_layer: this.config.enableMultiLayer,
              disciplinary_tags: config?.disciplinary_tags || ['general'],
              attribution: config?.attribution || []
            }
          };
          break;

        case 'decomposition':
          endpoint = '/api/openevolve/research/decompose';
          backendInputs = {
            graph_id: inputs.graph_id || context.graphId, // Expect graph_id from inputs or context
            dimensions: config?.dimensions || ['Scope', 'Objectives', 'Constraints', 'Data Needs', 'Use Cases', 'Potential Biases', 'Knowledge Gaps']
          };
          break;

        case 'hypothesis_planning':
          endpoint = '/api/openevolve/research/hypotheses';
          backendInputs = {
            graph_id: inputs.graph_id || context.graphId,
            dimension_node_id: dimensionNodeId || '2.1',
            hypotheses: hypotheses || [],
            config: {
              max_hypotheses: this.config.maxHypotheses
            }
          };
          break;

        case 'get_graph_summary':
          const gId = inputs.graph_id || context.graphId;
          endpoint = `/api/openevolve/research/summary/${gId}`;
          // GET request doesn't take body, but postToBackend handles bodies. 
          // We'll need to adjust postToBackend or use apiClient directly.
          return await this.executeSummary(gId);

        default:
          endpoint = '/api/openevolve/research/initialize';
          break;
      }

      context.updateProgress(30, `Executing ${stage} on Research-Quest server`);

      // Add timeout to prevent hanging requests
      const timeoutPromise = new Promise<ResearchQuestResult>((_, reject) => {
        setTimeout(() => reject(new Error('Request timeout after 30 seconds')), 30000);
      });

      const resultPromise = this.postToBackend(endpoint, backendInputs);
      const result: ResearchQuestResult = await Promise.race([resultPromise, timeoutPromise as Promise<ResearchQuestResult>]);

      context.updateProgress(100, `Research Quest ${stage} complete`);

      // Transform backend result to match expected output format with safety
      const transformedResult = {
        stage: stage,
        success: result?.success ?? false,
        message: result?.message ?? 'No message received from server',
        currentStage: result?.current_stage ?? 0,
        stageName: result?.stage_name ?? stage,
        nodeId: result?.node_id,
        dimensionNodes: result?.dimension_nodes,
        hypothesisNodes: result?.hypothesis_nodes,
        activeParameters: result?.active_parameters,
        warnings: result?.warnings,
        errors: result?.errors,
        partialSuccess: result?.partial_success,
        recoveryAttempted: result?.recovery_attempted,
        graphSummary: result?.graph_summary,
        reasoningTrace: result?.reasoning_trace,
        topologyInsights: result?.topology_insights,
        exportData: result?.export_data,
        metadata: {
          executionTime: Date.now() - startTime,
          stageExecuted: stage,
          parametersEnabled: this.config.parameters,
          backendUsed: true
        }
      };

      try {
        return this.createSuccessResult(transformedResult);
      } catch (creationError) {
        errorLogger.logError(creationError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating success result' } });
        return this.createErrorResult('Failed to create success result: ' + (creationError instanceof Error ? creationError.message : 'Unknown error'));
      }

    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend execution error' } });

      // Check if error is timeout-related
      if (error instanceof Error && error.message.includes('timeout')) {
        console.warn('Request timed out, falling back to local execution');
        return this.executeLocally(stage, taskDescription, hypotheses, dimensionNodeId, context);
      }

      // Check if error is network-related
      if (error instanceof TypeError && error.message.includes('fetch')) {
        console.warn('Network error, falling back to local execution');
        return this.executeLocally(stage, taskDescription, hypotheses, dimensionNodeId, context);
      }

      // For other errors, try to provide more context
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.warn(`Backend execution failed: ${errorMessage}, falling back to local execution`);

      return this.executeLocally(stage, taskDescription, hypotheses, dimensionNodeId, context);
    }
  }

  /**
   * Execute research quest locally (fallback/simplified version)
   * This is used when backend is unavailable or for testing
   */
  private async executeLocally(
    stage: ResearchQuestStage,
    taskDescription: string | undefined,
    hypotheses: any[] | undefined,
    dimensionNodeId: string | undefined,
    context: ExecutionContext
  ): Promise<NodeResult> {
    try {
      context.updateProgress(40, `Performing local ${stage} processing`);

      // Simple local research quest logic
      let result: any = {
        stage: stage,
        success: true,
        message: `Local ${stage} processing completed`,
        currentStage: 1,
        stageName: stage,
        nodeId: `local-${stage}-${Date.now()}`,
        metadata: {
          executionTime: 0,
          stageExecuted: stage,
          backendUsed: false,
          note: 'Executed locally (Research-Quest server unavailable)'
        }
      };

      switch (stage) {
        case 'initialization':
          result.nodeId = 'n0';
          result.message = 'Local graph initialized';
          break;
        case 'decomposition':
          result.dimensionNodes = ['2.1', '2.2', '2.3', '2.4', '2.5'];
          result.message = 'Local task decomposition completed';
          break;
        case 'hypothesis_planning':
          result.hypothesisNodes = ['3.1.1', '3.1.2', '3.1.3'];
          result.message = 'Local hypothesis generation completed';
          break;
        case 'get_graph_summary':
          try {
            result.graphSummary = {
              vertices_count: 10,
              edges_count: 15,
              current_stage: 3,
              stage_name: 'hypothesis_planning',
              active_parameters: Object.keys(this.config.parameters || {})
            };
          } catch (summaryError) {
            errorLogger.logError(summaryError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating local graph summary' } });
            result.graphSummary = {
              vertices_count: 0,
              edges_count: 0,
              current_stage: 0,
              stage_name: stage,
              active_parameters: []
            };
          }
          result.message = 'Local graph summary generated';
          break;
        case 'export_graph_data':
          try {
            result.exportData = JSON.stringify({
              stage: stage,
              timestamp: new Date().toISOString(),
              backendUsed: false,
              note: 'Local execution fallback'
            });
          } catch (jsonError) {
            errorLogger.logError(jsonError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error serializing local export data' } });
            result.exportData = '{"error": "Failed to serialize export data"}';
          }
          result.message = 'Local data export completed';
          break;
        default:
          result.message = `Local ${stage} processing completed`;
      }

      context.updateProgress(100, `Local ${stage} processing complete`);

      try {
        return this.createSuccessResult(result);
      } catch (creationError) {
        errorLogger.logError(creationError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating local success result' } });
        return this.createErrorResult('Failed to create local success result: ' + (creationError instanceof Error ? creationError.message : 'Unknown error'));
      }
    } catch (localError) {
      errorLogger.logError(localError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Local execution error' } });
      try {
        return this.createErrorResult(
          localError instanceof Error ? localError.message : 'Unknown error during local execution'
        );
      } catch (errorResultError) {
        errorLogger.logError(errorResultError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error creating local error result' } });
        return {
          success: false,
          data: { error: localError instanceof Error ? localError.message : 'Unknown error during local execution' },
          error: { message: localError instanceof Error ? localError.message : 'Unknown error during local execution' }
        } as NodeResult;
      }
    }
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    try {
      // Check for required task field for initialization
      const stage = inputs.stage as ResearchQuestStage || 'initialization';

      if (stage === 'initialization' || stage === 'decomposition' || stage === 'hypothesis_planning') {
        const task = inputs.task_description || inputs.task;

        if (!task) {
          errors.push({
            field: 'task_description',
            message: 'Task description is required for initialization and decomposition stages',
            severity: 'error'
          });
        }

        if (task && typeof task !== 'string') {
          errors.push({
            field: 'task_description',
            message: 'Task must be a string',
            severity: 'error'
          });
        }

        if (task && typeof task === 'string' && task.length < 10) {
          errors.push({
            field: 'task_description',
            message: 'Task description is too short (minimum 10 characters)',
            severity: 'warning'
          });
        }
      }

      // Validate stage if provided
      if (inputs.stage && typeof inputs.stage === 'string') {
        const validStages: ResearchQuestStage[] = [
          'initialization', 'decomposition', 'hypothesis_planning',
          'evidence_integration', 'pruning_merging', 'subgraph_extraction',
          'composition', 'reflection', 'get_graph_summary', 'export_graph_data'
        ];
        if (!validStages.includes(inputs.stage as ResearchQuestStage)) {
          errors.push({
            field: 'stage',
            message: `Stage must be one of: ${validStages.join(', ')}`,
            severity: 'error'
          });
        }
      }

      // Validate hypotheses if provided
      if (inputs.hypotheses && !Array.isArray(inputs.hypotheses)) {
        errors.push({
          field: 'hypotheses',
          message: 'Hypotheses must be an array',
          severity: 'error'
        });
      }

      // Validate dimension_node_id if provided
      if (inputs.dimension_node_id && typeof inputs.dimension_node_id !== 'string') {
        errors.push({
          field: 'dimension_node_id',
          message: 'Dimension node ID must be a string',
          severity: 'error'
        });
      }
    } catch (validationError) {
      // If validation itself fails, add a generic error
      errors.push({
        field: 'inputs',
        message: `Validation error: ${validationError instanceof Error ? validationError.message : 'Unknown validation error'}`,
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    try {
      return {
        type: 'object',
        properties: {
          enableMultiLayer: {
            type: 'boolean',
            description: 'Enable multi-layer network structure (P1.23)',
            default: true
          },
          maxHypotheses: {
            type: 'number',
            description: 'Maximum number of hypotheses per dimension (P1.3)',
            minimum: 3,
            maximum: 5,
            default: 5
          },
          enableCausalInference: {
            type: 'boolean',
            description: 'Enable causal inference capabilities (P1.24)',
            default: true
          },
          enableTemporalAnalysis: {
            type: 'boolean',
            description: 'Enable temporal pattern analysis (P1.25)',
            default: true
          },
          enableBiasAssessment: {
            type: 'boolean',
            description: 'Enable bias assessment and flags (P1.17)',
            default: true
          },
          enableFalsificationChecks: {
            type: 'boolean',
            description: 'Enable falsification criteria checking (P1.16)',
            default: true
          },
          enableImpactScoring: {
            type: 'boolean',
            description: 'Enable impact scoring for nodes (P1.28)',
            default: true
          },
          enableInterdisciplinaryBridges: {
            type: 'boolean',
            description: 'Enable interdisciplinary bridge nodes (P1.8)',
            default: true
          },
          enableKnowledgeGapDetection: {
            type: 'boolean',
            description: 'Enable knowledge gap detection (P1.15)',
            default: true
          },
          enableProbabilisticConfidence: {
            type: 'boolean',
            description: 'Enable probabilistic confidence distributions (P1.14)',
            default: true
          },
          enableGraphRestructuring: {
            type: 'boolean',
            description: 'Enable dynamic graph restructuring (P1.22)',
            default: true
          },
          enableTopologyAnalysis: {
            type: 'boolean',
            description: 'Enable topology metrics analysis (P1.22)',
            default: true
          },
          enableInformationTheoryMetrics: {
            type: 'boolean',
            description: 'Enable information theory metrics (P1.27)',
            default: true
          },
          enableAttributionTracking: {
            type: 'boolean',
            description: 'Enable researcher attribution tracking (P1.29)',
            default: true
          },
          enableStatisticalPowerAnalysis: {
            type: 'boolean',
            description: 'Enable statistical power analysis (P1.26)',
            default: true
          },
          enableMultiScaleAnalysis: {
            type: 'boolean',
            description: 'Enable multi-scale analysis (P1.20)',
            default: true
          },
          enableCostEstimation: {
            type: 'boolean',
            description: 'Enable computational cost estimation (P1.21)',
            default: true
          },
          enableSelfAudit: {
            type: 'boolean',
            description: 'Enable mandatory self-audit (P1.7)',
            default: true
          },
          enableEvidenceIntegration: {
            type: 'boolean',
            description: 'Enable evidence integration capabilities',
            default: true
          },
          enablePruningMerging: {
            type: 'boolean',
            description: 'Enable pruning and merging capabilities',
            default: true
          },
          enableSubgraphExtraction: {
            type: 'boolean',
            description: 'Enable subgraph extraction capabilities',
            default: true
          },
          enableReflection: {
            type: 'boolean',
            description: 'Enable reflection capabilities',
            default: true
          },
          enableComposition: {
            type: 'boolean',
            description: 'Enable composition capabilities',
            default: true
          },
          enableHypothesisGeneration: {
            type: 'boolean',
            description: 'Enable hypothesis generation capabilities',
            default: true
          },
          enableTaskDecomposition: {
            type: 'boolean',
            description: 'Enable task decomposition capabilities',
            default: true
          },
          enableInitialization: {
            type: 'boolean',
            description: 'Enable initialization capabilities',
            default: true
          },
          enableBackendExecution: {
            type: 'boolean',
            description: 'Use Research-Quest server via integration library',
            default: true
          },
          backendUrl: {
            type: 'string',
            description: 'URL of the Research-Quest server API',
            default: 'http://localhost:8000'
          },
          parameters: {
            type: 'object',
            description: 'Enable/disable specific Research-Quest parameters (P1.0-P1.29)',
            properties: {
              P1_0: { type: 'boolean', description: 'Mandatory 8-stage GoT execution', default: true },
              P1_1: { type: 'boolean', description: 'Root node n₀ label=\'Task Understanding\'', default: true },
              P1_2: { type: 'boolean', description: 'Default dimensions', default: true },
              P1_3: { type: 'boolean', description: 'Generate k=3-5 hypotheses per dimension', default: true },
              P1_4: { type: 'boolean', description: 'Iterative loop based on confidence-to-cost ratio', default: true },
              P1_5: { type: 'boolean', description: 'Confidence C = [empirical_support, theoretical_basis, methodological_rigor, consensus_alignment]', default: true },
              P1_6: { type: 'boolean', description: 'Numeric node labels, verbatim queries, reasoning trace', default: true },
              P1_7: { type: 'boolean', description: 'Mandatory self-audit', default: true },
              P1_8: { type: 'boolean', description: 'Maintain explicit disciplinary_tags list', default: true },
              P1_9: { type: 'boolean', description: 'Enable hyperedges', default: true },
              P1_10: { type: 'boolean', description: 'Classify edges E with mandatory edge_type metadata', default: true },
              P1_11: { type: 'boolean', description: 'Define ASR-GoT graph state', default: true },
              P1_12: { type: 'boolean', description: 'Mandatory metadata for nodes and edges', default: true },
              P1_13: { type: 'boolean', description: 'Identify mutually exclusive hypotheses', default: true },
              P1_14: { type: 'boolean', description: 'Represent confidence components using probability distributions', default: true },
              P1_15: { type: 'boolean', description: 'Identify gaps: create Placeholder_Gap nodes', default: true },
              P1_16: { type: 'boolean', description: 'Require population of falsification_criteria metadata', default: true },
              P1_17: { type: 'boolean', description: 'Include \'Potential Biases\' dimension and assess nodes', default: true },
              P1_18: { type: 'boolean', description: 'Utilize timestamp metadata and apply temporal decay', default: true },
              P1_19: { type: 'boolean', description: 'Generate \'prospective\' subgraphs for interventions', default: true },
              P1_20: { type: 'boolean', description: 'Define abstraction levels or utilize formal multi-layer structure', default: true },
              P1_21: { type: 'boolean', description: 'Estimate computational cost of operations', default: true },
              P1_22: { type: 'boolean', description: 'Enable graph restructuring during Stage 4', default: true },
              P1_23: { type: 'boolean', description: 'Define distinct but interconnected layers L', default: true },
              P1_24: { type: 'boolean', description: 'Extend edge types with causal semantics', default: true },
              P1_25: { type: 'boolean', description: 'Extend edge types to support temporal patterns', default: true },
              P1_26: { type: 'boolean', description: 'Add power analysis metrics to statistical_power metadata', default: true },
              P1_27: { type: 'boolean', description: 'Incorporate entropy, KL divergence, mutual information', default: true },
              P1_28: { type: 'boolean', description: 'Develop and apply metrics for theoretical significance', default: true },
              P1_29: { type: 'boolean', description: 'Support node attribution to researchers/specialists', default: true },
            }
          }
        },
        required: []
      };
    } catch (schemaError) {
      errorLogger.logError(schemaError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error generating parameter schema' } });
      // Return a minimal safe schema as fallback
      return {
        type: 'object',
        properties: {
          enableBackendExecution: {
            type: 'boolean',
            description: 'Use Research-Quest server via integration library',
            default: true
          },
          backendUrl: {
            type: 'string',
            description: 'URL of the Research-Quest server API',
            default: 'http://localhost:8000'
          }
        },
        required: []
      };
    }
  }

  /**
   * Cleanup when node is destroyed
   */
  private async postToBackend(endpoint: string, payload: Record<string, any>): Promise<ResearchQuestResult> {
    try {
      const backendUrl = (this.config.backendUrl as string | undefined) || '';

      if (!backendUrl) {
        return await apiClient.post<ResearchQuestResult>(endpoint, payload);
      }

      // Validate URL construction
      let url: string;
      try {
        url = new URL(endpoint, backendUrl).toString();
      } catch (urlError) {
        throw new Error(`Invalid URL constructed from backendUrl: ${backendUrl} and endpoint: ${endpoint}`);
      }

      let token;
      try {
        token = useAuthStore.getState().token;
      } catch (storeError) {
        console.warn('Auth store not available, proceeding without token:', storeError);
        token = null;
      }

      // Set a timeout for the fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        // Try to get error details from response
        let errorDetails = response.statusText;
        try {
          const errorPayload = await response.json();
          errorDetails = errorPayload?.error?.message || errorPayload?.message || response.statusText;
        } catch (parseError) {
          // If parsing fails, use the status text
          console.warn('Could not parse error response:', parseError);
        }

        throw new Error(`HTTP ${response.status}: ${errorDetails}`);
      }

      // Parse response with error handling
      const result = await response.json();
      return result as ResearchQuestResult;
    } catch (error) {
      // Handle different types of errors
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(`Network error: Unable to reach the Research-Quest server at ${this.config.backendUrl}. Please check your connection and server status.`);
      }

      if (error.name === 'AbortError') {
        throw new Error('Request timeout: The Research-Quest server took too long to respond. Falling back to local execution.');
      }

      if (error instanceof Error) {
        throw error; // Re-throw known errors
      }

      throw new Error('Unknown error occurred while communicating with the Research-Quest server');
    }
  }

  destroy(): void {
    try {
      // No-op for now; backend connections are stateless HTTP calls.
      // Clean up any local resources if needed
      console.log(`[ResearchQuestNode] Node ${this.id} destroyed`);
    } catch (error) {
      console.error(`[ResearchQuestNode] Error during destruction of node ${this.id}:`, error);
      // Still consider the node destroyed even if cleanup fails
    }
  }
}

export default ResearchQuestNode;