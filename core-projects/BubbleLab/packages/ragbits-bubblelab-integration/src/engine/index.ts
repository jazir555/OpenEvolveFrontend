// Export workflow engine functionality

export * from './ragbits_workflow_engine';

// Export main WorkflowEngine class
import { RAGBitsWorkflowEngine } from './ragbits_workflow_engine';
export { RAGBitsWorkflowEngine };

// Export engine utilities
export function createWorkflowEngine(
  workflowConfig: any,
  options?: any,
  processor?: any
): RAGBitsWorkflowEngine {
  return new RAGBitsWorkflowEngine(workflowConfig, options, processor);
}