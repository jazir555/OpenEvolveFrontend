"use strict";
/**
 * Hybrid Orchestration Workflows
 *
 * This module exports all hybrid workflows that orchestrate LoongFlow (PES)
 * and OpenEvolve (Evolution) in powerful hybrid patterns for advanced AI problem-solving.
 *
 * Workflow Types:
 * - PES_EVOLUTION: Sequential PES → Evolution → Summary
 * - KNOWLEDGE_EXTRACTION: Extract knowledge → formulate new problems
 * - ADAPTIVE_EXECUTION: Dynamically switch paradigms based on triggers
 * - MULTI_STAGE_REASONING: Complex multi-system reasoning with validation
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All config via ENV vars
 * - Law of Idempotency: All workflows safe to retry
 * - Observability: Structured logging with correlation IDs
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.WORKFLOW_METADATA = exports.WORKFLOWS = exports.createMultiStageReasoningWorkflow = exports.MultiStageReasoningWorkflow = exports.createAdaptiveExecutionWorkflow = exports.AdaptiveExecutionWorkflow = exports.createKnowledgeExtractionWorkflow = exports.KnowledgeExtractionWorkflow = exports.createPESEvolutionWorkflow = exports.PESEvolutionWorkflow = void 0;
exports.createWorkflow = createWorkflow;
exports.getAvailableWorkflowTypes = getAvailableWorkflowTypes;
exports.validateWorkflowConfig = validateWorkflowConfig;
exports.getWorkflowMetadata = getWorkflowMetadata;
exports.getAllWorkflowMetadata = getAllWorkflowMetadata;
// ============================================================================
// WORKFLOW EXPORTS
// ============================================================================
var pes_evolution_workflow_1 = require("./pes-evolution-workflow");
Object.defineProperty(exports, "PESEvolutionWorkflow", { enumerable: true, get: function () { return pes_evolution_workflow_1.PESEvolutionWorkflow; } });
Object.defineProperty(exports, "createPESEvolutionWorkflow", { enumerable: true, get: function () { return pes_evolution_workflow_1.createPESEvolutionWorkflow; } });
var knowledge_extraction_workflow_1 = require("./knowledge-extraction-workflow");
Object.defineProperty(exports, "KnowledgeExtractionWorkflow", { enumerable: true, get: function () { return knowledge_extraction_workflow_1.KnowledgeExtractionWorkflow; } });
Object.defineProperty(exports, "createKnowledgeExtractionWorkflow", { enumerable: true, get: function () { return knowledge_extraction_workflow_1.createKnowledgeExtractionWorkflow; } });
var adaptive_execution_workflow_1 = require("./adaptive-execution-workflow");
Object.defineProperty(exports, "AdaptiveExecutionWorkflow", { enumerable: true, get: function () { return adaptive_execution_workflow_1.AdaptiveExecutionWorkflow; } });
Object.defineProperty(exports, "createAdaptiveExecutionWorkflow", { enumerable: true, get: function () { return adaptive_execution_workflow_1.createAdaptiveExecutionWorkflow; } });
var multi_stage_reasoning_workflow_1 = require("./multi-stage-reasoning-workflow");
Object.defineProperty(exports, "MultiStageReasoningWorkflow", { enumerable: true, get: function () { return multi_stage_reasoning_workflow_1.MultiStageReasoningWorkflow; } });
Object.defineProperty(exports, "createMultiStageReasoningWorkflow", { enumerable: true, get: function () { return multi_stage_reasoning_workflow_1.createMultiStageReasoningWorkflow; } });
// ============================================================================
// WORKFLOW REGISTRY
// ============================================================================
/**
 * Available workflow types
 */
exports.WORKFLOWS = {
    PES_EVOLUTION: 'pes-evolution',
    KNOWLEDGE_EXTRACTION: 'knowledge-extraction',
    ADAPTIVE_EXECUTION: 'adaptive-execution',
    MULTI_STAGE_REASONING: 'multi-stage-reasoning',
};
/**
 * Workflow factory function
 *
 * Creates workflow instances based on type
 *
 * @param type - Workflow type
 * @param config - Workflow configuration
 * @returns Workflow instance
 */
function createWorkflow(type, config) {
    switch (type) {
        case exports.WORKFLOWS.PES_EVOLUTION:
            const { createPESEvolutionWorkflow } = require('./pes-evolution-workflow');
            return createPESEvolutionWorkflow(config);
        case exports.WORKFLOWS.KNOWLEDGE_EXTRACTION:
            const { createKnowledgeExtractionWorkflow } = require('./knowledge-extraction-workflow');
            return createKnowledgeExtractionWorkflow(config);
        case exports.WORKFLOWS.ADAPTIVE_EXECUTION:
            const { createAdaptiveExecutionWorkflow } = require('./adaptive-execution-workflow');
            return createAdaptiveExecutionWorkflow(config);
        case exports.WORKFLOWS.MULTI_STAGE_REASONING:
            const { createMultiStageReasoningWorkflow } = require('./multi-stage-reasoning-workflow');
            return createMultiStageReasoningWorkflow(config);
        default:
            throw new Error(`Unknown workflow type: ${type}`);
    }
}
/**
 * Get all available workflow types
 */
function getAvailableWorkflowTypes() {
    return Object.values(exports.WORKFLOWS);
}
/**
 * Validate workflow configuration
 */
function validateWorkflowConfig(type, config) {
    const errors = [];
    switch (type) {
        case exports.WORKFLOWS.PES_EVOLUTION:
            if (!config.loongflowAdapter) {
                errors.push('loongflowAdapter is required for PES_EVOLUTION workflow');
            }
            if (!config.openevolveAdapter) {
                errors.push('openevolveAdapter is required for PES_EVOLUTION workflow');
            }
            break;
        case exports.WORKFLOWS.KNOWLEDGE_EXTRACTION:
            if (!config.loongflowAdapter) {
                errors.push('loongflowAdapter is required for KNOWLEDGE_EXTRACTION workflow');
            }
            break;
        case exports.WORKFLOWS.ADAPTIVE_EXECUTION:
            if (!config.loongflowAdapter) {
                errors.push('loongflowAdapter is required for ADAPTIVE_EXECUTION workflow');
            }
            if (!config.openevolveAdapter) {
                errors.push('openevolveAdapter is required for ADAPTIVE_EXECUTION workflow');
            }
            break;
        case exports.WORKFLOWS.MULTI_STAGE_REASONING:
            if (!config.loongflowAdapter) {
                errors.push('loongflowAdapter is required for MULTI_STAGE_REASONING workflow');
            }
            if (!config.openevolveAdapter) {
                errors.push('openevolveAdapter is required for MULTI_STAGE_REASONING workflow');
            }
            break;
        default:
            errors.push(`Unknown workflow type: ${type}`);
    }
    return {
        valid: errors.length === 0,
        errors: errors.length > 0 ? errors : undefined,
    };
}
/**
 * Workflow metadata
 */
exports.WORKFLOW_METADATA = {
    [exports.WORKFLOWS.PES_EVOLUTION]: {
        name: 'PES-Evolution Hybrid Workflow',
        description: 'Sequential PES → Evolution → Summary for enhanced problem-solving',
        required_adapters: ['loongflow', 'openevolve'],
        optional_adapters: [],
        estimated_duration_ms: 300000, // 5 minutes
        timeout_ms: 600000, // 10 minutes
    },
    [exports.WORKFLOWS.KNOWLEDGE_EXTRACTION]: {
        name: 'Knowledge Extraction Workflow',
        description: 'Extract knowledge from solutions and formulate new problems',
        required_adapters: ['loongflow'],
        optional_adapters: ['graphiti', 'vectordb'],
        estimated_duration_ms: 60000, // 1 minute
        timeout_ms: 120000, // 2 minutes
    },
    [exports.WORKFLOWS.ADAPTIVE_EXECUTION]: {
        name: 'Adaptive Execution Workflow',
        description: 'Dynamically switch between PES and Evolution based on triggers',
        required_adapters: ['loongflow', 'openevolve'],
        optional_adapters: [],
        estimated_duration_ms: 600000, // 10 minutes
        timeout_ms: 900000, // 15 minutes
    },
    [exports.WORKFLOWS.MULTI_STAGE_REASONING]: {
        name: 'Multi-Stage Reasoning Workflow',
        description: 'Complex multi-system reasoning with Plan, Optimize, Validate, Refine, Summarize',
        required_adapters: ['loongflow', 'openevolve'],
        optional_adapters: ['z3', 'lean-aide'],
        estimated_duration_ms: 600000, // 10 minutes
        timeout_ms: 1200000, // 20 minutes
    },
};
/**
 * Get workflow metadata
 */
function getWorkflowMetadata(type) {
    return exports.WORKFLOW_METADATA[type];
}
/**
 * Get all workflow metadata
 */
function getAllWorkflowMetadata() {
    return exports.WORKFLOW_METADATA;
}
// ============================================================================
// EXAMPLE USAGE
// ============================================================================
/**
 * Example usage:
 *
 * ```typescript
 * import {
 *   createWorkflow,
 *   WORKFLOWS,
 *   createPESEvolutionWorkflow,
 *   validateWorkflowConfig,
 *   getWorkflowMetadata,
 * } from './workflows';
 *
 * import { LoongFlowAdapter } from '../../adapters/loongflow-adapter/src/adapter';
 * import { OpenEvolveAdapter } from '../../adapters/openevolve-adapter/src/adapter';
 *
 * // Create adapters
 * const loongflowAdapter = new LoongFlowAdapter({
 *   api_url: process.env.LOONGFLOW_API_URL!,
 *   timeout_ms: 30000,
 * });
 *
 * const openevolveAdapter = new OpenEvolveAdapter({
 *   api_url: process.env.OPENEVOLVE_API_URL!,
 *   timeout_ms: 30000,
 * });
 *
 * // Create PES-Evolution workflow using factory
 * const pesEvolutionWorkflow = createWorkflow(
 *   WORKFLOWS.PES_EVOLUTION,
 *   {
 *     loongflowAdapter,
 *     openevolveAdapter,
 *     checkpoints_enabled: true,
 *   }
 * );
 *
 * // Or create directly
 * const directWorkflow = createPESEvolutionWorkflow({
 *   loongflowAdapter,
 *   openevolveAdapter,
 * });
 *
 * // Validate configuration
 * const validation = validateWorkflowConfig(
 *   WORKFLOWS.PES_EVOLUTION,
 *   { loongflowAdapter, openevolveAdapter }
 * );
 *
 * if (!validation.valid) {
 *   console.error('Invalid config:', validation.errors);
 * }
 *
 * // Get workflow metadata
 * const metadata = getWorkflowMetadata(WORKFLOWS.PES_EVOLUTION);
 * console.log('Workflow:', metadata.name);
 * console.log('Description:', metadata.description);
 * console.log('Required adapters:', metadata.required_adapters);
 *
 * // Execute workflow
 * const result = await pesEvolutionWorkflow.execute(
 *   {
 *     problem: {
 *       id: 'problem-123',
 *       type: 'optimization',
 *       description: 'Optimize neural network hyperparameters',
 *       context: { dataset: 'MNIST' },
 *       constraints: ['max_layers <= 10'],
 *       success_criteria: ['accuracy > 0.95'],
 *       created_at: new Date().toISOString(),
 *     },
 *     pes_config: {
 *       max_iterations: 10,
 *       target_score: 0.9,
 *     },
 *     evolution_config: {
 *       generations: 10,
 *       population_size: 100,
 *     },
 *   }
 * );
 *
 * console.log('Workflow result:', result);
 * ```
 */
//# sourceMappingURL=index.js.map