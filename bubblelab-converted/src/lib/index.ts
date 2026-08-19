/**
 * OpenEvolve Library Exports
 *
 * Central export point for all OpenEvolve libraries and integrations.
 */

// Plugin System
export {
  PluginRegistry,
  getPluginRegistry,
  resetPluginRegistry,
  type PluginInterface,
  type PluginMetadata,
  type PluginCapabilities,
  type PluginContext,
  type PluginRegistryConfig
} from './plugin-registry';

// Plugin Adapters
export {
  RAGBitsPluginAdapter,
  DatapizzaPluginAdapter,
  OpenEvolveApiAdapter
} from './plugin-adapters';

// Workflow System
export {
  WorkflowOrchestrator,
  getWorkflowOrchestrator,
  type WorkflowDefinition,
  type WorkflowStep,
  type WorkflowContext,
  type WorkflowExecutionResult
} from './workflow-orchestrator';

// Workflow Templates
export {
  RESEARCH_ASSISTANT_WORKFLOW,
  DATA_ANALYSIS_PIPELINE,
  PROOF_VERIFICATION_WORKFLOW,
  KNOWLEDGE_EXTRACTION_WORKFLOW,
  PROBLEM_SOLVING_WORKFLOW,
  WORKFLOW_TEMPLATES,
  getWorkflowTemplate,
  getAllWorkflowTemplates,
  getWorkflowTemplatesByCategory
} from './workflow-templates';

// Event Integration
export {
  PluginEventIntegration,
  getPluginEventIntegration,
  resetPluginEventIntegration,
  type PluginEvent,
  type PluginEventSubscriber
} from './plugin-events';

// Monitoring
export {
  WorkflowMonitor,
  getWorkflowMonitor,
  resetWorkflowMonitor,
  type WorkflowMetrics,
  type StepMetrics,
  type TelemetryConfig
} from './workflow-monitoring';

// Main Integration
export {
  BubbleLabIntegration,
  initializeBubbleLabIntegration,
  getBubbleLabIntegration,
  resetBubbleLabIntegration,
  type BubbleLabIntegrationConfig
} from './plugin-integration';

// API Client
export { openevolveApi } from './openevolveApi';
export type { ApiConfig } from './openevolveApi';

// Types
export * from './types';

// Re-export hooks for convenience
export {
  useBubbleLabIntegration,
  useBubbleLabIntegrationInstance,
  usePluginRegistry,
  useWorkflowOrchestrator
} from '../hooks/useBubbleLabIntegration';
