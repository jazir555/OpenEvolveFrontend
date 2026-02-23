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
import { OpenEvolveAdapter, WorkflowDefinition, WorkflowState } from './adapter';
import { IntegrationCoordinator } from './integration-coordinator';
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
    progress: number;
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
    dependencies: string[];
}
export declare class WorkflowOrchestrator {
    private readonly openEvolveAdapter;
    private readonly integrationCoordinator;
    private readonly maxConcurrentWorkflows;
    private readonly logger;
    private readonly correlationId;
    private readonly activeWorkflows;
    private readonly workflowHistory;
    constructor(openEvolveAdapter: OpenEvolveAdapter, integrationCoordinator: IntegrationCoordinator, maxConcurrentWorkflows?: number);
    executeWorkflow(request: WorkflowExecutionRequest): Promise<WorkflowExecutionResult>;
    private determineStages;
    private executeStage;
    private executeContentAnalysis;
    private executeDecompositionPlanning;
    private executeSubProblemSolving;
    private executeSolutionAssembly;
    private executeFinalVerification;
    private executeKnowledgeExtraction;
    private initializeWorkflow;
    private finalizeWorkflow;
    private calculateStageProgress;
    private areDependenciesMet;
    private saveCheckpoint;
    private reportProgress;
    private generateCorrelationId;
    getActiveWorkflows(): WorkflowState[];
    getWorkflowHistory(): WorkflowExecutionResult[];
    getWorkflowState(workflowId: string): WorkflowState | undefined;
}
export declare function createWorkflowOrchestrator(openEvolveAdapter: OpenEvolveAdapter, integrationCoordinator: IntegrationCoordinator, maxConcurrentWorkflows?: number): WorkflowOrchestrator;
//# sourceMappingURL=workflow-orchestrator.d.ts.map