/**
 * Workflow Orchestrator
 *
 * Orchestrates multi-step workflows that span multiple adapters and systems.
 * This component handles:
 * - Complex multi-stage workflows
 * - Cross-adapter data flows
 * - Workflow state management
 * - Error handling and recovery
 * - Progress tracking and reporting
 *
 * Environment Variables:
 *   WORKFLOW_TIMEOUT_MS - Maximum workflow execution time
 *   MAX_CONCURRENT_WORKFLOWS - Maximum concurrent workflow executions
 *   WORKFLOW_CHECKPOINT_INTERVAL_MS - Checkpoint save interval
 */

import { OpenEvolveAdapter, WorkflowDefinition, WorkflowState, StructuredLogger, LogContext } from './adapter';
import { IntegrationCoordinator, CoordinationRequest, CoordinationPlan } from './integration-coordinator';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface WorkflowExecutionRequest {
  workflow: WorkflowDefinition;
  priority: 'low' | 'medium' | 'high';
  timeout_ms?: number;
  checkpoint_enabled?: boolean;
  on_progress?: (update: WorkflowProgressUpdate) => void;
  on_complete?: (result: WorkflowExecutionResult) => void;
  on_error?: (error: WorkflowError) => void;
}

export interface WorkflowProgressUpdate {
  workflow_id: string;
  current_stage: string;
  progress: number; // 0.0 to 1.0
  message: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface WorkflowExecutionResult {
  workflow_id: string;
  status: 'completed' | 'failed' | 'cancelled' | 'timeout';
  final_state: WorkflowState;
  duration_ms: number;
  stages_completed: string[];
  stages_failed: string[];
  knowledge_artifacts: any[];
  error?: WorkflowError;
}

export interface WorkflowError {
  workflow_id: string;
  stage: string;
  error_code: string;
  error_message: string;
  timestamp: string;
  recoverable: boolean;
  retry_count: number;
  context?: Record<string, any>;
}

export interface StageDefinition {
  name: string;
  type: 'analysis' | 'extraction' | 'validation' | 'transformation' | 'aggregation';
  adapter_required?: string;
  idempotent: boolean;
  retry_on_failure: boolean;
  timeout_ms: number;
  dependencies: string[]; // Names of stages that must complete first
}

// ============================================================================
// WORKFLOW ORCHESTRATOR CLASS
// ============================================================================

export class WorkflowOrchestrator {
  private readonly logger: StructuredLogger;
  private readonly correlationId: string;
  private readonly activeWorkflows: Map<string, WorkflowState> = new Map();
  private readonly workflowHistory: Map<string, WorkflowExecutionResult> = new Map();

  constructor(
    private readonly openEvolveAdapter: OpenEvolveAdapter,
    private readonly integrationCoordinator: IntegrationCoordinator,
    private readonly maxConcurrentWorkflows: number = 5,
  ) {
    this.logger = new StructuredLogger('workflow-orchestrator');
    this.correlationId = this.generateCorrelationId();

    this.logger.info('Workflow orchestrator initialized', {
      correlation_id: this.correlationId,
      max_concurrent_workflows: this.maxConcurrentWorkflows,
    });
  }

  // ==========================================================================
  // WORKFLOW EXECUTION
  // ==========================================================================

  async executeWorkflow(request: WorkflowExecutionRequest): Promise<WorkflowExecutionResult> {
    const workflowId = request.workflow.workflow_id;

    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'workflow-orchestrator',
      workflow_id: workflowId,
    };

    this.logger.info('Starting workflow execution', {
      ...context,
      priority: request.priority,
      sub_problem_count: request.workflow.sub_problems.length,
    });

    // Check workflow concurrency limit
    if (this.activeWorkflows.size >= this.maxConcurrentWorkflows) {
      const error: WorkflowError = {
        workflow_id: workflowId,
        stage: 'initialization',
        error_code: 'MAX_CONCURRENT_WORKFLOWS_EXCEEDED',
        error_message: 'Maximum concurrent workflow limit reached',
        timestamp: new Date().toISOString(),
        recoverable: true,
        retry_count: 0,
      };

      this.logger.error('Workflow execution rejected', { ...context, error });
      throw new Error(error.error_message);
    }

    const startTime = Date.now();

    try {
      // Create workflow state
      const workflowState = await this.initializeWorkflow(request.workflow);
      this.activeWorkflows.set(workflowId, workflowState);

      // Report initial progress
      this.reportProgress(request, {
        workflow_id: workflowId,
        current_stage: 'initialized',
        progress: 0.0,
        message: 'Workflow initialized',
        timestamp: new Date().toISOString(),
      });

      // Execute workflow stages
      const stagesCompleted: string[] = [];
      const stagesFailed: string[] = [];
      const knowledgeArtifacts: any[] = [];

      // Determine execution plan
      const stages = this.determineStages(request.workflow);

      // Execute stages in dependency order
      for (const stage of stages) {
        // Check dependencies
        if (!this.areDependenciesMet(stage, stagesCompleted)) {
          this.logger.warn('Stage dependencies not met, skipping', {
            ...context,
            stage: stage.name,
            dependencies: stage.dependencies,
          });
          continue;
        }

        // Execute stage
        try {
          this.logger.info('Executing stage', { ...context, stage: stage.name });

          const stageResult = await this.executeStage(stage, request.workflow, workflowState);

          stagesCompleted.push(stage.name);

          // Extract knowledge artifacts if any
          if (stageResult.knowledge_artifacts) {
            knowledgeArtifacts.push(...stageResult.knowledge_artifacts);
          }

          // Report progress
          const progress = stagesCompleted.length / stages.length;
          this.reportProgress(request, {
            workflow_id: workflowId,
            current_stage: stage.name,
            progress,
            message: `Stage "${stage.name}" completed`,
            timestamp: new Date().toISOString(),
          });

          // Save checkpoint if enabled
          if (request.checkpoint_enabled) {
            await this.saveCheckpoint(workflowState);
          }
        } catch (error) {
          stagesFailed.push(stage.name);

          const workflowError: WorkflowError = {
            workflow_id: workflowId,
            stage: stage.name,
            error_code: 'STAGE_EXECUTION_FAILED',
            error_message: error instanceof Error ? error.message : String(error),
            timestamp: new Date().toISOString(),
            recoverable: stage.retry_on_failure,
            retry_count: 0,
          };

          this.logger.error('Stage execution failed', {
            ...context,
            stage: stage.name,
            error: workflowError.error_message,
          });

          // Report error
          if (request.on_error) {
            request.on_error(workflowError);
          }

          // Stop workflow if stage is not recoverable
          if (!stage.retry_on_failure) {
            throw error;
          }
        }
      }

      // Finalize workflow
      const finalState = await this.finalizeWorkflow(workflowState);
      const duration = Date.now() - startTime;

      const result: WorkflowExecutionResult = {
        workflow_id: workflowId,
        status: 'completed',
        final_state: finalState,
        duration_ms: duration,
        stages_completed: stagesCompleted,
        stages_failed: stagesFailed,
        knowledge_artifacts: knowledgeArtifacts,
      };

      this.workflowHistory.set(workflowId, result);
      this.activeWorkflows.delete(workflowId);

      this.logger.info('Workflow execution completed', {
        ...context,
        duration_ms: duration,
        stages_completed: stagesCompleted.length,
        stages_failed: stagesFailed.length,
      });

      // Report completion
      if (request.on_complete) {
        request.on_complete(result);
      }

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      const state = this.activeWorkflows.get(workflowId);

      const workflowError: WorkflowError = {
        workflow_id: workflowId,
        stage: state?.current_stage || 'unknown',
        error_code: 'WORKFLOW_EXECUTION_FAILED',
        error_message: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
        recoverable: false,
        retry_count: 0,
      };

      const result: WorkflowExecutionResult = {
        workflow_id: workflowId,
        status: 'failed',
        final_state: state || await this.initializeWorkflow(request.workflow),
        duration_ms: duration,
        stages_completed: [],
        stages_failed: [],
        knowledge_artifacts: [],
        error: workflowError,
      };

      this.workflowHistory.set(workflowId, result);
      this.activeWorkflows.delete(workflowId);

      this.logger.error('Workflow execution failed', {
        ...context,
        duration_ms: duration,
        error: workflowError.error_message,
      });

      // Report error
      if (request.on_error) {
        request.on_error(workflowError);
      }

      throw error;
    }
  }

  // ==========================================================================
  // WORKFLOW STAGES
  // ==========================================================================

  private determineStages(workflow: WorkflowDefinition): StageDefinition[] {
    const stages: StageDefinition[] = [];

    // Stage 1: Content Analysis
    stages.push({
      name: 'content_analysis',
      type: 'analysis',
      idempotent: true,
      retry_on_failure: true,
      timeout_ms: 30000,
      dependencies: [],
    });

    // Stage 2: Decomposition Planning
    stages.push({
      name: 'decomposition_planning',
      type: 'analysis',
      idempotent: true,
      retry_on_failure: true,
      timeout_ms: 60000,
      dependencies: ['content_analysis'],
    });

    // Stage 3: Sub-problem Solving (parallel for each sub-problem)
    for (let i = 0; i < workflow.sub_problems.length; i++) {
      stages.push({
        name: `solve_sub_problem_${i}`,
        type: 'transformation',
        idempotent: false,
        retry_on_failure: true,
        timeout_ms: 120000,
        dependencies: ['decomposition_planning'],
      });
    }

    // Stage 4: Solution Assembly
    stages.push({
      name: 'solution_assembly',
      type: 'transformation',
      idempotent: true,
      retry_on_failure: true,
      timeout_ms: 60000,
      dependencies: workflow.sub_problems.map((_, i) => `solve_sub_problem_${i}`),
    });

    // Stage 5: Final Verification
    stages.push({
      name: 'final_verification',
      type: 'validation',
      idempotent: true,
      retry_on_failure: true,
      timeout_ms: 60000,
      dependencies: ['solution_assembly'],
    });

    // Stage 6: Knowledge Extraction
    stages.push({
      name: 'knowledge_extraction',
      type: 'extraction',
      idempotent: true,
      retry_on_failure: true,
      timeout_ms: 30000,
      dependencies: ['final_verification'],
    });

    return stages;
  }

  private async executeStage(
    stage: StageDefinition,
    workflow: WorkflowDefinition,
    state: WorkflowState,
  ): Promise<{ success: boolean; knowledge_artifacts?: any[] }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'workflow-orchestrator',
      workflow_id: workflow.workflow_id,
      stage: stage.name,
    };

    this.logger.info('Executing stage', context);

    // Update workflow state
    state.current_stage = stage.name;
    state.progress = this.calculateStageProgress(stage, workflow);

    // Stage-specific logic
    switch (stage.name) {
      case 'content_analysis':
        return await this.executeContentAnalysis(workflow, state);

      case 'decomposition_planning':
        return await this.executeDecompositionPlanning(workflow, state);

      case 'solution_assembly':
        return await this.executeSolutionAssembly(workflow, state);

      case 'final_verification':
        return await this.executeFinalVerification(workflow, state);

      case 'knowledge_extraction':
        return await this.executeKnowledgeExtraction(workflow, state);

      default:
        // Handle sub-problem solving stages
        if (stage.name.startsWith('solve_sub_problem_')) {
          const index = parseInt(stage.name.split('_').pop() || '0', 10);
          return await this.executeSubProblemSolving(workflow, state, index);
        }

        this.logger.warn('Unknown stage, treating as no-op', { ...context });
        return { success: true };
    }
  }

  private async executeContentAnalysis(workflow: WorkflowDefinition, state: WorkflowState): Promise<any> {
    // Create coordination request for content analysis
    const request: CoordinationRequest = {
      problem_type: 'code_analysis',
      domain: workflow.problem_statement,
      capabilities: ['semantic_search', 'graph_traversal'],
      priority: 'medium',
    };

    // Plan coordination
    const plan = await this.integrationCoordinator.planCoordination(request);

    // Execute coordination (this would integrate with actual content analysis logic)
    this.logger.info('Content analysis coordination planned', {
      workflow_id: workflow.workflow_id,
      adapter_count: plan.selected_adapters.length,
    });

    return { success: true };
  }

  private async executeDecompositionPlanning(workflow: WorkflowDefinition, state: WorkflowState): Promise<any> {
    this.logger.info('Executing decomposition planning', {
      workflow_id: workflow.workflow_id,
      sub_problem_count: workflow.sub_problems.length,
    });

    // Store decomposition plan in state
    state.decomposition_plan = workflow;

    return { success: true };
  }

  private async executeSubProblemSolving(workflow: WorkflowDefinition, state: WorkflowState, index: number): Promise<any> {
    const subProblem = workflow.sub_problems[index];

    this.logger.info('Solving sub-problem', {
      workflow_id: workflow.workflow_id,
      sub_problem_id: subProblem.id,
    });

    // Create coordination request for sub-problem
    const request: CoordinationRequest = {
      problem_type: 'formal_verification',
      domain: subProblem.description,
      capabilities: ['smt_solving', 'tactic_execution'],
      priority: 'medium',
    };

    // Plan and execute coordination
    const plan = await this.integrationCoordinator.planCoordination(request);
    // const results = await this.integrationCoordinator.executeCoordination(plan, request);

    return { success: true };
  }

  private async executeSolutionAssembly(workflow: WorkflowDefinition, state: WorkflowState): Promise<any> {
    this.logger.info('Assembling final solution', {
      workflow_id: workflow.workflow_id,
    });

    return { success: true };
  }

  private async executeFinalVerification(workflow: WorkflowDefinition, state: WorkflowState): Promise<any> {
    this.logger.info('Executing final verification', {
      workflow_id: workflow.workflow_id,
    });

    return { success: true };
  }

  private async executeKnowledgeExtraction(workflow: WorkflowDefinition, state: WorkflowState): Promise<any> {
    this.logger.info('Extracting knowledge artifacts', {
      workflow_id: workflow.workflow_id,
    });

    return { success: true, knowledge_artifacts: [] };
  }

  // ==========================================================================
  // WORKFLOW STATE MANAGEMENT
  // ==========================================================================

  private async initializeWorkflow(workflow: WorkflowDefinition): Promise<WorkflowState> {
    const state: WorkflowState = {
      workflow_id: workflow.workflow_id,
      problem_statement: workflow.problem_statement,
      current_stage: 'initializing',
      status: 'running',
      progress: 0.0,
      start_time: new Date().toISOString(),
      decomposition_plan: workflow,
      sub_problem_solutions: {},
      solved_sub_problem_ids: new Set<string>(),
      rejected_sub_problems: {},
      refinement_loop_count: 0,
      all_critique_reports: [],
      all_verification_reports: [],
      resource_usage: {},
      performance_metrics: {},
      knowledge_artifacts: [],
      openevolve_metrics: {},
    };

    return state;
  }

  private async finalizeWorkflow(state: WorkflowState): Promise<WorkflowState> {
    state.status = 'completed';
    state.end_time = new Date().toISOString();
    state.progress = 1.0;
    state.current_stage = 'completed';

    return state;
  }

  private calculateStageProgress(stage: StageDefinition, workflow: WorkflowDefinition): number {
    // Simple progress calculation based on stage type
    const stageWeights: Record<string, number> = {
      content_analysis: 0.1,
      decomposition_planning: 0.15,
      solution_assembly: 0.15,
      final_verification: 0.1,
      knowledge_extraction: 0.05,
    };

    // For sub-problem stages, distribute remaining weight
    const subProblemCount = workflow.sub_problems.length;
    const subProblemWeight = 0.45 / subProblemCount;

    if (stage.name.startsWith('solve_sub_problem_')) {
      return subProblemWeight;
    }

    return stageWeights[stage.name] || 0;
  }

  private areDependenciesMet(stage: StageDefinition, completedStages: string[]): boolean {
    return stage.dependencies.every(dep => completedStages.includes(dep));
  }

  private async saveCheckpoint(state: WorkflowState): Promise<void> {
    this.logger.info('Saving workflow checkpoint', {
      workflow_id: state.workflow_id,
      stage: state.current_stage,
      progress: state.progress,
    });

    // In a real implementation, this would persist to a database
    // For now, this is a no-op placeholder
  }

  // ==========================================================================
  // UTILITY METHODS
  // ==========================================================================

  private reportProgress(request: WorkflowExecutionRequest, update: WorkflowProgressUpdate): void {
    if (request.on_progress) {
      request.on_progress(update);
    }
  }

  private generateCorrelationId(): string {
    return `orchestrator-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  }

  getActiveWorkflows(): WorkflowState[] {
    return Array.from(this.activeWorkflows.values());
  }

  getWorkflowHistory(): WorkflowExecutionResult[] {
    return Array.from(this.workflowHistory.values());
  }

  getWorkflowState(workflowId: string): WorkflowState | undefined {
    return this.activeWorkflows.get(workflowId);
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createWorkflowOrchestrator(
  openEvolveAdapter: OpenEvolveAdapter,
  integrationCoordinator: IntegrationCoordinator,
  maxConcurrentWorkflows?: number,
): WorkflowOrchestrator {
  return new WorkflowOrchestrator(openEvolveAdapter, integrationCoordinator, maxConcurrentWorkflows);
}
