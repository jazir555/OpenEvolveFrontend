/**
 * OpenEvolve Integration Adapters for BubbleLab
 *
 * Main entry point for all OpenEvolve service integrations.
 * Exports service bubbles, tool bubbles, schemas, and adapters.
 *
 * @module openevolve-integrations
 */

import { KnowledgeEngineBubble } from './service-bubbles/knowledge-engine-bubble';
import { WorkflowOrchestratorBubble } from './service-bubbles/workflow-orchestrator-bubble';
import { CrewAIBubble } from './service-bubbles/crewai-bubble';
import { LeanAideBubble } from './service-bubbles/leanaide-bubble';
import { Z3ProverBubble } from './service-bubbles/z3prover-bubble';

// ============================================================================
// SERVICE BUBBLES
// ============================================================================

export { QdrantBubble } from './service-bubbles/qdrant-bubble';
export { ElasticsearchBubble } from './service-bubbles/elasticsearch-bubble';
export { KnowledgeEngineBubble } from './service-bubbles/knowledge-engine-bubble';
export { WorkflowOrchestratorBubble } from './service-bubbles/workflow-orchestrator-bubble';
export { CrewAIBubble } from './service-bubbles/crewai-bubble';
export { LeanAideBubble } from './service-bubbles/leanaide-bubble';
export { Z3ProverBubble } from './service-bubbles/z3prover-bubble';
export { PostgreSQLBubbleExtended as PostgreSQLBubble } from './service-bubbles/postgresql-bubble';
export { RedisBubble } from './service-bubbles/redis-bubble';
export { ACEToolsBubble } from './service-bubbles/ace-tools-bubble';

// ============================================================================
// TOOL BUBBLES
// ============================================================================

export { LogParserTool } from './tool-bubbles/log-parser-tool';
export { MetricsCollectorTool } from './tool-bubbles/metrics-collector-tool';

// ============================================================================
// SCHEMAS & CANONICAL MODELS
// ============================================================================

export * from './schemas/canonical-models';

// ============================================================================
// ADAPTERS & ANTI-CORRUPTION LAYER
// ============================================================================

export { default as AntiCorruptionLayer } from './adapters/anti-corruption-layer';
export type {
  CanonicalRequest,
  CanonicalResponse,
  CanonicalData,
  CanonicalError,
} from './adapters/anti-corruption-layer';

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Create a complete OpenEvolve integration setup
 *
 * LAW OF RUNTIME TRUTH: Automatically validates service health on startup
 * unless explicitly skipped for testing purposes.
 *
 * @param config - Integration configuration options
 * @param skipValidation - If true, skip health validation (for testing only)
 * @throws {Error} If any service fails health check and skipValidation is false
 * @returns Configured integration instance with all service bubbles
 */
export async function createOpenEvolveIntegration(
  config?: {
    knowledgeBackend?: 'qdrant' | 'elasticsearch' | 'bedrock' | 'eks' | 'hybrid';
    enableCircuitBreaker?: boolean;
    enableMetrics?: boolean;
    enableValidation?: boolean;
  },
  skipValidation?: boolean
) {
  const crewai = new CrewAIBubble({
    operation: 'health_check',
  });

  const z3prover = new Z3ProverBubble({
    operation: 'health_check',
    baseUrl: process.env.Z3_API_URL || 'http://localhost:7655',
    timeout: parseInt(process.env.Z3_TIMEOUT || '60000', 10),
  });

  const integration = {
    knowledgeEngine: new KnowledgeEngineBubble({
      operation: 'health_check',
      backend: config?.knowledgeBackend || 'qdrant',
    }),

    workflowOrchestrator: new WorkflowOrchestratorBubble({
      operation: 'health_check',
      system: 'integrated',
    }),

    crewai,

    z3prover,

    acl: new AntiCorruptionLayer({
      enableValidation: config?.enableValidation !== false,
      enableMetrics: config?.enableMetrics !== false,
    }),
  };

  // LAW OF RUNTIME TRUTH: Validate integration at startup
  // Unless explicitly skipped (e.g., for unit tests)
  if (!skipValidation) {
    try {
      const validation = await validateIntegration(integration);

      if (!validation.valid) {
        const errorSummary = validation.errors.join('\n  - ');
        throw new Error(
          'OpenEvolve Integration health check failed.\n' +
          `Failed services:\n  - ${errorSummary}\n\n` +
          'To bypass this validation (not recommended), set skipValidation=true.\n' +
          'This is only safe for testing environments.'
        );
      }

      // Log successful validation in development
      if (import.meta.env?.DEV || process.env?.NODE_ENV === 'development') {
        console.log(
          '[OpenEvolve] Integration health check passed.\n' +
          `Services: ${Object.keys(validation.services).join(', ')}`
        );
      }
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(
          `OpenEvolve Integration validation failed: ${error.message}`
        );
      }
      throw error;
    }
  }

  return integration;
}

/**
 * Validate integration configuration
 */
export async function validateIntegration(
  integration: ReturnType<typeof createOpenEvolveIntegration>
): Promise<{
  valid: boolean;
  services: Record<string, boolean>;
  errors: string[];
}> {
  const results: Record<string, boolean> = {};
  const errors: string[] = [];

  try {
    const knowledgeHealth = await integration.knowledgeEngine.action();
    results.knowledgeEngine = knowledgeHealth.success;
    if (!knowledgeHealth.success) {
      errors.push(`Knowledge engine: ${knowledgeHealth.error || 'Unknown error'}`);
    }
  } catch (error) {
    results.knowledgeEngine = false;
    errors.push(`Knowledge engine: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  try {
    const workflowHealth = await integration.workflowOrchestrator.action();
    results.workflowOrchestrator = workflowHealth.success;
    if (!workflowHealth.success) {
      errors.push(`Workflow orchestrator: ${workflowHealth.error || 'Unknown error'}`);
    }
  } catch (error) {
    results.workflowOrchestrator = false;
    errors.push(`Workflow orchestrator: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  try {
    const crewaiHealth = await integration.crewai.action();
    results.crewai = crewaiHealth.success;
    if (!crewaiHealth.success) {
      errors.push(`CrewAI: ${crewaiHealth.error || 'Unknown error'}`);
    }
  } catch (error) {
    results.crewai = false;
    errors.push(`CrewAI: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  try {
    const z3proverHealth = await integration.z3prover.action();
    results.z3prover = z3proverHealth.success;
    if (!z3proverHealth.success) {
      errors.push(`Z3 Prover: ${z3proverHealth.error || 'Unknown error'}`);
    }
  } catch (error) {
    results.z3prover = false;
    errors.push(`Z3 Prover: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  return {
    valid: Object.values(results).every(v => v === true),
    services: results,
    errors,
  };
}

/**
 * Get integration health report
 */
export async function getHealthReport(
  integration: ReturnType<typeof createOpenEvolveIntegration>
): Promise<{
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: Record<string, {
    status: boolean;
    error?: string;
    timing?: number;
  }>;
}> {
  const validation = await validateIntegration(integration);

  const healthyCount = Object.values(validation.services).filter(v => v).length;
  const totalCount = Object.keys(validation.services).length;

  let status: 'healthy' | 'degraded' | 'unhealthy';
  if (healthyCount === totalCount) {
    status = 'healthy';
  } else if (healthyCount > 0) {
    status = 'degraded';
  } else {
    status = 'unhealthy';
  }

  return {
    status,
    timestamp: new Date().toISOString(),
    services: validation.services,
  };
}

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

export const OPENEVOLVE_VERSION = '1.0.0';
export const INTEGRATION_COUNT = 20; // Number of integration adapters created
export const SUPPORTED_PROTOCOLS = ['http', 'grpc', 'websocket'];
export const SUPPORTED_DATABASES = ['postgresql', 'redis', 'qdrant', 'elasticsearch'];
export const SUPPORTED_WORKFLOWS = ['decomposition', 'evolutionary', 'mdap_maker', 'adversarial'];

// ============================================================================
// DEFAULT EXPORTS
// ============================================================================

export default {
  // Service Bubbles
  QdrantBubble,
  ElasticsearchBubble,
  KnowledgeEngineBubble,
  WorkflowOrchestratorBubble,
  CrewAIBubble,
  LeanAideBubble,
  Z3ProverBubble,
  PostgreSQLBubble,
  RedisBubble,
  ACEToolsBubble,

  // Tool Bubbles
  LogParserTool,
  MetricsCollectorTool,

  // ACL
  AntiCorruptionLayer,

  // Utilities
  createOpenEvolveIntegration,
  validateIntegration,
  getHealthReport,

  // Constants
  OPENEVOLVE_VERSION,
  INTEGRATION_COUNT,
  SUPPORTED_PROTOCOLS,
  SUPPORTED_DATABASES,
  SUPPORTED_WORKFLOWS,
};
