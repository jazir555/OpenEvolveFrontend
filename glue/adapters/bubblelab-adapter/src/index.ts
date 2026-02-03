/**
 * BubbleLab Adapter Exports
 *
 * Main entry point for the BubbleLab adapter
 */

export {
  BubbleLabAdapter,
  createBubbleLabAdapter,
  type BubbleLabAdapterConfig,
  type AdapterMetrics,
} from './adapter';

export {
  BubbleLabClient,
  createBubbleLabClient,
  type BubbleLabClientConfig,
  type BubbleLabResponse,
  type BubbleFlowListResponse,
  type BubbleFlowCreateRequest,
  type BubbleFlowCreateResponse,
  type BubbleFlowExecuteRequest,
  type BubbleFlowExecuteResponse,
} from './bubble-client';

export {
  // Enums
  BubbleType,
  CredentialType,
  EventType,
  ExecutionStatus,

  // Schemas
  CanonicalBubbleSchema,
  CanonicalBubbleFlowSchema,
  CanonicalExecutionResultSchema,
  CanonicalBubbleLabEventSchema,
  CanonicalCredentialMappingSchema,

  // Types
  type CanonicalBubble,
  type CanonicalBubbleFlow,
  type CanonicalExecutionResult,
  type CanonicalBubbleLabEvent,
  type CanonicalCredentialMapping,

  // Mapping functions
  mapToCanonicalBubbleFlow,
  mapToCanonicalExecutionResult,
  mapFromCanonicalBubbleFlow,
  mapFromCanonicalCredentials,

  // Validation functions
  validateCanonicalBubbleFlow,
  validateCanonicalExecutionResult,
  validateCanonicalBubbleLabEvent,

  // Utility functions
  generateCorrelationId,
  toUTCISOString,
  fromUTCISOString,
} from './bubblelab-canonical';
