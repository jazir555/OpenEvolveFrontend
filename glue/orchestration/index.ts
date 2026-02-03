/**
 * Orchestration Layer Exports
 *
 * Event bus, workflow engine, dead letter queue, and correlation tracking
 */

export { EventBus, eventBus, EventBusType } from './event-bus';
export type { EventBusConfig, EventBusStats } from './event-bus';

export { WorkflowEngine, workflowEngine, PREDEFINED_WORKFLOWS } from './workflow-engine';
export type {
  WorkflowDefinition,
  WorkflowStep,
  WorkflowContext,
  WorkflowExecutionResult,
  WorkflowState
} from './workflow-engine';

export { DeadLetterQueue, deadLetterQueue } from './dead-letter-queue';
export type { DLQEntry, DLQStats, RetryPolicy } from './dead-letter-queue';

export { CorrelationTracker, correlationTracker, createCorrelationMiddleware } from './correlation-tracker';
export type { CorrelationContext, ServiceCall, DistributedTraceSpan } from './correlation-tracker';

export * from './event-types';
