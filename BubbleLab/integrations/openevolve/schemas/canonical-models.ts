/**
 * Canonical Data Models for OpenEvolve Integration
 *
 * Defines standardized data structures and Zod schemas for
 * canonical data models used across all OpenEvolve services.
 * Provides transformation functions from service-specific to canonical.
 */

import { z } from 'zod';

// ============================================================================
// CORE CANONICAL MODELS
// ============================================================================

/**
 * Canonical User Model
 */
export const CanonicalUserSchema = z.object({
  id: z.string().describe('Unique user identifier'),
  username: z.string().describe('Username'),
  email: z.string().email().describe('User email'),
  role: z.enum(['admin', 'user', 'viewer', 'operator']).describe('User role'),
  permissions: z.array(z.string()).default([]).describe('User permissions'),
  metadata: z.record(z.unknown()).optional().describe('Additional metadata'),
  createdAt: z.string().datetime().describe('Creation timestamp (ISO 8601)'),
  updatedAt: z.string().datetime().describe('Last update timestamp (ISO 8601)'),
});

export type CanonicalUser = z.output<typeof CanonicalUserSchema>;

/**
 * Canonical Service Model
 */
export const CanonicalServiceSchema = z.object({
  id: z.string().describe('Service identifier'),
  name: z.string().describe('Service name'),
  type: z.enum([
    'knowledge_engine',
    'database',
    'workflow',
    'api',
    'monitoring',
    'messaging',
    'storage',
  ]).describe('Service type'),
  status: z.enum(['running', 'stopped', 'error', 'degraded', 'unknown']).describe('Service status'),
  health: z.enum(['healthy', 'unhealthy', 'degraded', 'unknown']).describe('Health status'),
  endpoint: z.string().url().describe('Service endpoint URL'),
  version: z.string().optional().describe('Service version'),
  metadata: z.record(z.unknown()).optional(),
  lastCheck: z.string().datetime().describe('Last health check timestamp'),
});

export type CanonicalService = z.output<typeof CanonicalServiceSchema>;

/**
 * Canonical Workflow Model
 */
export const CanonicalWorkflowSchema = z.object({
  id: z.string().describe('Workflow ID'),
  name: z.string().describe('Workflow name'),
  type: z.enum([
    'decomposition',
    'evolutionary',
    'mdap_maker',
    'adversarial',
    'integrated',
  ]).describe('Workflow type'),
  status: z.enum([
    'pending',
    'running',
    'paused',
    'completed',
    'failed',
    'cancelled',
  ]).describe('Workflow status'),
  definition: z.record(z.unknown()).describe('Workflow definition'),
  parameters: z.record(z.unknown()).optional().describe('Runtime parameters'),
  results: z.record(z.unknown()).optional().describe('Execution results'),
  metrics: z.object({
    executionTime: z.number().optional(),
    progress: z.number().min(0).max(1).optional(),
    resourceUsage: z.record(z.number()).optional(),
  }).optional().describe('Workflow metrics'),
  createdAt: z.string().datetime(),
  startedAt: z.string().datetime().optional(),
  completedAt: z.string().datetime().optional(),
  error: z.string().optional(),
});

export type CanonicalWorkflow = z.output<typeof CanonicalWorkflowSchema>;

/**
 * Canonical Knowledge Document Model
 */
export const CanonicalKnowledgeDocumentSchema = z.object({
  id: z.string().describe('Document ID'),
  content: z.string().describe('Document content'),
  embedding: z.array(z.number()).optional().describe('Vector embedding'),
  metadata: z.object({
    source: z.string().describe('Document source'),
    type: z.string().describe('Document type'),
    tags: z.array(z.string()).default([]).describe('Document tags'),
    author: z.string().optional(),
    timestamp: z.string().datetime(),
    language: z.string().default('en'),
    confidence: z.number().min(0).max(1).optional(),
  }).describe('Document metadata'),
  relationships: z.array(z.object({
    type: z.enum(['references', 'depends_on', 'related_to', 'derived_from']),
    targetId: z.string(),
    strength: z.number().min(0).max(1).optional(),
  })).optional().describe('Document relationships'),
});

export type CanonicalKnowledgeDocument = z.output<typeof CanonicalKnowledgeDocumentSchema>;

/**
 * Canonical Metric Model
 */
export const CanonicalMetricSchema = z.object({
  name: z.string().describe('Metric name'),
  type: z.enum(['counter', 'gauge', 'histogram', 'summary']).describe('Metric type'),
  value: z.number().describe('Metric value'),
  timestamp: z.string().datetime().describe('Metric timestamp'),
  labels: z.record(z.string()).optional().describe('Metric labels'),
  unit: z.string().optional().describe('Metric unit'),
  description: z.string().optional().describe('Metric description'),
});

export type CanonicalMetric = z.output<typeof CanonicalMetricSchema>;

/**
 * Canonical Log Entry Model
 */
export const CanonicalLogEntrySchema = z.object({
  timestamp: z.string().datetime().describe('Log timestamp'),
  level: z.enum(['debug', 'info', 'warn', 'error', 'fatal', 'trace']).describe('Log level'),
  message: z.string().describe('Log message'),
  service: z.string().describe('Service name'),
  component: z.string().optional().describe('Component name'),
  correlationId: z.string().optional().describe('Correlation ID'),
  metadata: z.record(z.unknown()).optional().describe('Additional metadata'),
  stackTrace: z.string().optional().describe('Stack trace for errors'),
  userId: z.string().optional().describe('User ID'),
  requestId: z.string().optional().describe('Request ID'),
});

export type CanonicalLogEntry = z.output<typeof CanonicalLogEntrySchema>;

/**
 * Canonical Event Model
 */
export const CanonicalEventSchema = z.object({
  id: z.string().describe('Event ID'),
  type: z.string().describe('Event type'),
  source: z.string().describe('Event source'),
  timestamp: z.string().datetime().describe('Event timestamp'),
  data: z.record(z.unknown()).describe('Event data'),
  correlationId: z.string().optional().describe('Correlation ID'),
  causationId: z.string().optional().describe('Causation ID'),
  metadata: z.record(z.unknown()).optional(),
});

export type CanonicalEvent = z.output<typeof CanonicalEventSchema>;

/**
 * Canonical Task Model
 */
export const CanonicalTaskSchema = z.object({
  id: z.string().describe('Task ID'),
  name: z.string().describe('Task name'),
  description: z.string().optional().describe('Task description'),
  status: z.enum([
    'pending',
    'assigned',
    'in_progress',
    'completed',
    'failed',
    'cancelled',
  ]).describe('Task status'),
  priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium').describe('Task priority'),
  assignee: z.string().optional().describe('Assigned user/team'),
  workflowId: z.string().optional().describe('Parent workflow ID'),
  dependencies: z.array(z.string()).default([]).describe('Task dependencies'),
  result: z.unknown().optional().describe('Task result'),
  error: z.string().optional().describe('Task error'),
  createdAt: z.string().datetime(),
  startedAt: z.string().datetime().optional(),
  completedAt: z.string().datetime().optional(),
  dueAt: z.string().datetime().optional(),
});

export type CanonicalTask = z.output<typeof CanonicalTaskSchema>;

/**
 * Canonical Error Model
 */
export const CanonicalErrorSchema = z.object({
  code: z.string().describe('Error code'),
  message: z.string().describe('Error message'),
  type: z.enum([
    'validation',
    'authentication',
    'authorization',
    'not_found',
    'conflict',
    'rate_limit',
    'internal',
    'external',
  ]).describe('Error type'),
  details: z.record(z.unknown()).optional().describe('Error details'),
  stackTrace: z.string().optional().describe('Stack trace'),
  timestamp: z.string().datetime().describe('Error timestamp'),
  service: z.string().describe('Service that raised the error'),
  correlationId: z.string().optional().describe('Correlation ID'),
  requestId: z.string().optional().describe('Request ID'),
});

export type CanonicalError = z.output<typeof CanonicalErrorSchema>;

// ============================================================================
// TRANSFORMATION FUNCTIONS
// ============================================================================

/**
 * Transform Qdrant point to canonical knowledge document
 */
export function qdrantPointToCanonical(point: any): CanonicalKnowledgeDocument {
  return {
    id: String(point.id),
    content: point.payload?.content || '',
    embedding: point.vector,
    metadata: {
      source: point.payload?.source || 'qdrant',
      type: point.payload?.type || 'unknown',
      tags: point.payload?.tags || [],
      timestamp: point.payload?.timestamp || new Date().toISOString(),
      language: point.payload?.language || 'en',
    },
    relationships: point.payload?.relationships,
  };
}

/**
 * Transform Elasticsearch document to canonical knowledge document
 */
export function elasticsearchDocToCanonical(doc: any): CanonicalKnowledgeDocument {
  const source = doc._source || {};
  return {
    id: doc._id,
    content: source.content || '',
    metadata: {
      source: source.source || 'elasticsearch',
      type: source.type || 'unknown',
      tags: source.tags || [],
      timestamp: source.timestamp || new Date().toISOString(),
      language: source.language || 'en',
    },
  };
}

/**
 * Transform OpenEvolve workflow to canonical workflow
 */
export function openEvolveWorkflowToCanonical(workflow: any): CanonicalWorkflow {
  return {
    id: workflow.id || workflow.workflow_id,
    name: workflow.name || workflow.workflow_name,
    type: workflow.type || 'integrated',
    status: workflow.status || 'pending',
    definition: workflow.definition || {},
    parameters: workflow.parameters,
    results: workflow.results,
    metrics: workflow.metrics,
    createdAt: workflow.created_at || new Date().toISOString(),
    startedAt: workflow.started_at,
    completedAt: workflow.completed_at,
    error: workflow.error,
  };
}

/**
 * Transform service health check to canonical service
 */
export function healthCheckToCanonical(serviceName: string, health: any): CanonicalService {
  return {
    id: serviceName.toLowerCase().replace(/\s+/g, '-'),
    name: serviceName,
    type: health.type || 'api',
    status: health.status === 'ok' ? 'running' : 'error',
    health: health.healthy ? 'healthy' : 'unhealthy',
    endpoint: health.endpoint || '',
    version: health.version,
    metadata: health.metadata,
    lastCheck: new Date().toISOString(),
  };
}

/**
 * Transform raw log to canonical log entry
 */
export function rawLogToCanonical(raw: any): CanonicalLogEntry {
  return {
    timestamp: raw.timestamp || raw.time || raw['@timestamp'] || new Date().toISOString(),
    level: raw.level || raw.severity || 'info',
    message: raw.message || raw.msg || '',
    service: raw.service || raw.app || 'unknown',
    component: raw.component,
    correlationId: raw.correlation_id || raw.correlationId,
    metadata: raw.metadata || raw.extra,
    stackTrace: raw.stack_trace || raw.stack,
    userId: raw.user_id || raw.userId,
    requestId: raw.request_id || raw.requestId,
  };
}

/**
 * Validate data against canonical schema
 */
export function validateCanonical<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: boolean; data?: T; error?: string } {
  try {
    const validated = schema.parse(data);
    return { success: true, data: validated };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return {
        success: false,
        error: error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', '),
      };
    }
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown validation error',
    };
  }
}

/**
 * Batch transform with validation
 */
export function batchTransform<T>(
  schema: z.ZodSchema<T>,
  items: unknown[],
  transformFn: (item: any) => T
): { success: boolean; data: T[]; errors: string[] } {
  const results: T[] = [];
  const errors: string[] = [];

  for (const item of items) {
    try {
      const transformed = transformFn(item);
      const validated = schema.parse(transformed);
      results.push(validated);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      errors.push(errorMsg);
    }
  }

  return {
    success: errors.length === 0,
    data: results,
    errors,
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  CanonicalUserSchema,
  CanonicalServiceSchema,
  CanonicalWorkflowSchema,
  CanonicalKnowledgeDocumentSchema,
  CanonicalMetricSchema,
  CanonicalLogEntrySchema,
  CanonicalEventSchema,
  CanonicalTaskSchema,
  CanonicalErrorSchema,
  qdrantPointToCanonical,
  elasticsearchDocToCanonical,
  openEvolveWorkflowToCanonical,
  healthCheckToCanonical,
  rawLogToCanonical,
  validateCanonical,
  batchTransform,
};
