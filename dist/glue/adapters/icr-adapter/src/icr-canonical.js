"use strict";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Canonical Schemas
 *
 * Canonical data models for all 7 ICR modes.
 * These schemas define the Anti-Corruption Layer (ACL) contract.
 * All data entering/leaving the ICR system MUST conform to these schemas.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Air Gap: No imports from core-projects
 * - Runtime Truth: Schemas reflect actual API behavior
 * - Configuration Explicitness: All fields required (no magic defaults)
 * - UTC: All timestamps in UTC ISO-8601 format
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ICRHealthCheckResponseSchema = exports.ICRHealthCheckRequestSchema = exports.ICRModeResponseSchema = exports.ICRModeRequestSchema = exports.GenerativeUIModeResponseSchema = exports.GenerativeUIModeRequestSchema = exports.ContextualModeResponseSchema = exports.ContextualModeRequestSchema = exports.AgenticModeResponseSchema = exports.AgenticModeRequestSchema = exports.AdaptiveDeepthinkResponseSchema = exports.AdaptiveDeepthinkRequestSchema = exports.DeepthinkModeResponseSchema = exports.DeepthinkModeRequestSchema = exports.ReactModeResponseSchema = exports.ReactModeRequestSchema = exports.RefineModeResponseSchema = exports.RefineModeRequestSchema = exports.ModeOptionsSchema = exports.ICRResultSchema = exports.ICRMetadataSchema = exports.ModeTypeSchema = void 0;
const zod_1 = require("zod");
// ============================================================================
// COMMON TYPES
// ============================================================================
/**
 * Mode type enum for all 7 ICR modes
 */
exports.ModeTypeSchema = zod_1.z.enum([
    'refine',
    'react',
    'deepthink',
    'adaptive_deepthink',
    'agentic',
    'contextual',
    'generative_ui'
]);
/**
 * Base metadata schema included in all requests/responses
 */
exports.ICRMetadataSchema = zod_1.z.object({
    correlation_id: zod_1.z.string().uuid(),
    timestamp_utc: zod_1.z.string().datetime(),
    source_service: zod_1.z.string().default('icr-adapter'),
    mode: exports.ModeTypeSchema,
    request_id: zod_1.z.string().optional()
});
/**
 * Base result schema for all mode responses
 */
exports.ICRResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    content: zod_1.z.string(),
    error: zod_1.z.string().optional(),
    execution_time_ms: zod_1.z.number().int().nonnegative(),
    iteration_count: zod_1.z.number().int().nonnegative().default(0),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
/**
 * Mode options schema
 */
exports.ModeOptionsSchema = zod_1.z.object({
    temperature: zod_1.z.number().min(0).max(2).optional(),
    top_p: zod_1.z.number().min(0).max(1).optional(),
    max_iterations: zod_1.z.number().int().positive().optional(),
    model_name: zod_1.z.string().optional(),
    provider: zod_1.z.enum(['google', 'openai', 'anthropic']).optional()
});
// ============================================================================
// REFINE MODE
// ============================================================================
/**
 * Refine Mode Request Schema
 * Mode: Traditional iterative refinements with automated feature suggestion
 */
exports.RefineModeRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('refine'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        evolution_mode: zod_1.z.enum(['novelty', 'quality', 'off']).optional(),
        refinement_stages: zod_1.z.number().int().positive().optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * Refine Mode Response Schema
 */
exports.RefineModeResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('refine'),
    request: exports.RefineModeRequestSchema,
    result: exports.ICRResultSchema.extend({
        iterations: zod_1.z.array(zod_1.z.object({
            iteration_number: zod_1.z.number().int(),
            content: zod_1.z.string(),
            suggested_features: zod_1.z.string().optional(),
            bug_fixes: zod_1.z.string().optional(),
            status: zod_1.z.enum(['pending', 'processing', 'completed', 'error', 'cancelled']),
            error: zod_1.z.string().optional()
        }))
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// REACT MODE
// ============================================================================
/**
 * React Mode Request Schema
 * Mode: React application development with orchestrator-coordination
 */
exports.ReactModeRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('react'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        worker_count: zod_1.z.number().int().positive().optional(),
        enable_preview: zod_1.z.boolean().optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * React Mode Response Schema
 */
exports.ReactModeResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('react'),
    request: exports.ReactModeRequestSchema,
    result: exports.ICRResultSchema.extend({
        orchestrator_plan: zod_1.z.string().optional(),
        workers: zod_1.z.array(zod_1.z.object({
            worker_id: zod_1.z.string(),
            title: zod_1.z.string(),
            system_instruction: zod_1.z.string().optional(),
            user_prompt: zod_1.z.string().optional(),
            generated_content: zod_1.z.string().optional(),
            status: zod_1.z.enum(['pending', 'processing', 'completed', 'error', 'cancelled']),
            error: zod_1.z.string().optional()
        })),
        preview_url: zod_1.z.string().optional()
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// DEEPTHINK MODE
// ============================================================================
/**
 * Deepthink Mode Request Schema
 * Mode: Complex problem-solving through strategic decomposition
 */
exports.DeepthinkModeRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('deepthink'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        strategy_count: zod_1.z.number().int().positive().optional(),
        sub_strategy_count: zod_1.z.number().int().positive().optional(),
        hypothesis_count: zod_1.z.number().int().positive().optional(),
        enable_iterative_corrections: zod_1.z.boolean().optional(),
        enable_red_team: zod_1.z.boolean().optional(),
        red_team_aggressiveness: zod_1.z.enum(['low', 'medium', 'high']).optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * Deepthink Mode Response Schema
 */
exports.DeepthinkModeResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('deepthink'),
    request: exports.DeepthinkModeRequestSchema,
    result: exports.ICRResultSchema.extend({
        strategies: zod_1.z.array(zod_1.z.object({
            strategy_id: zod_1.z.string(),
            strategy_text: zod_1.z.string(),
            sub_strategies: zod_1.z.array(zod_1.z.object({
                sub_strategy_id: zod_1.z.string(),
                sub_strategy_text: zod_1.z.string(),
                solution: zod_1.z.string().optional(),
                critique: zod_1.z.string().optional(),
                refined_solution: zod_1.z.string().optional(),
                status: zod_1.z.enum(['pending', 'processing', 'completed', 'error', 'cancelled'])
            }))
        })),
        hypotheses: zod_1.z.array(zod_1.z.object({
            hypothesis_id: zod_1.z.string(),
            hypothesis_text: zod_1.z.string(),
            test_result: zod_1.z.string().optional(),
            status: zod_1.z.enum(['pending', 'processing', 'completed', 'error', 'cancelled'])
        })).optional(),
        best_solution: zod_1.z.string().optional(),
        red_team_evaluations: zod_1.z.array(zod_1.z.object({
            strategy_id: zod_1.z.string(),
            evaluation: zod_1.z.string(),
            killed: zod_1.z.boolean()
        })).optional()
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// ADAPTIVE DEEPTHINK MODE
// ============================================================================
/**
 * Adaptive Deepthink Mode Request Schema
 * Mode: Full deepthink mode access to an agent
 */
exports.AdaptiveDeepthinkRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('adaptive_deepthink'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        conversation_id: zod_1.z.string().optional(),
        enable_streaming: zod_1.z.boolean().optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * Adaptive Deepthink Mode Response Schema
 */
exports.AdaptiveDeepthinkResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('adaptive_deepthink'),
    request: exports.AdaptiveDeepthinkRequestSchema,
    result: exports.ICRResultSchema.extend({
        conversation_id: zod_1.z.string().optional(),
        tool_calls: zod_1.z.array(zod_1.z.object({
            tool_name: zod_1.z.string(),
            parameters: zod_1.z.record(zod_1.z.any()),
            result: zod_1.z.any()
        })).optional(),
        reasoning_trace: zod_1.z.string().optional()
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// AGENTIC MODE
// ============================================================================
/**
 * Agentic Mode Request Schema
 * Mode: General-purpose iterative refinement with tool-based manipulation
 */
exports.AgenticModeRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('agentic'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        conversation_id: zod_1.z.string().optional(),
        enable_diff_tools: zod_1.z.boolean().optional(),
        enable_file_tools: zod_1.z.boolean().optional(),
        enable_web_search: zod_1.z.boolean().optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * Agentic Mode Response Schema
 */
exports.AgenticModeResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('agentic'),
    request: exports.AgenticModeRequestSchema,
    result: exports.ICRResultSchema.extend({
        conversation_id: zod_1.z.string().optional(),
        tool_calls: zod_1.z.array(zod_1.z.object({
            tool_name: zod_1.z.string(),
            parameters: zod_1.z.record(zod_1.z.any()),
            result: zod_1.z.any()
        })).optional(),
        diff_operations: zod_1.z.array(zod_1.z.object({
            type: zod_1.z.enum(['search_and_replace', 'delete', 'insert_before', 'insert_after']),
            params: zod_1.z.array(zod_1.z.string())
        })).optional()
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// CONTEXTUAL MODE
// ============================================================================
/**
 * Contextual Mode Request Schema
 * Mode: Iterative refinement through specialized agent collaboration
 */
exports.ContextualModeRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('contextual'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        conversation_id: zod_1.z.string().optional(),
        enable_memory_agent: zod_1.z.boolean().optional(),
        memory_compression_threshold: zod_1.z.number().int().positive().optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * Contextual Mode Response Schema
 */
exports.ContextualModeResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('contextual'),
    request: exports.ContextualModeRequestSchema,
    result: exports.ICRResultSchema.extend({
        conversation_id: zod_1.z.string().optional(),
        agent_interactions: zod_1.z.array(zod_1.z.object({
            agent_type: zod_1.z.enum(['main_generator', 'iterative_agent', 'memory_agent']),
            content: zod_1.z.string(),
            timestamp_utc: zod_1.z.string().datetime()
        })).optional(),
        memory_compression_events: zod_1.z.array(zod_1.z.object({
            timestamp_utc: zod_1.z.string().datetime(),
            compressed_message_count: zod_1.z.number().int()
        })).optional()
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// GENERATIVE UI MODE
// ============================================================================
/**
 * Generative UI Mode Request Schema
 * Mode: Interactive UI development with user interaction capture
 */
exports.GenerativeUIModeRequestSchema = zod_1.z.object({
    mode: zod_1.z.literal('generative_ui'),
    prompt: zod_1.z.string().min(1),
    options: exports.ModeOptionsSchema.extend({
        enable_interaction_capture: zod_1.z.boolean().optional(),
        quality_threshold: zod_1.z.number().min(0).max(1).optional(),
        max_iterations: zod_1.z.number().int().positive().optional()
    }).optional(),
    metadata: exports.ICRMetadataSchema
});
/**
 * Generative UI Mode Response Schema
 */
exports.GenerativeUIModeResponseSchema = zod_1.z.object({
    mode: zod_1.z.literal('generative_ui'),
    request: exports.GenerativeUIModeRequestSchema,
    result: exports.ICRResultSchema.extend({
        ui_structure: zod_1.z.string().optional(),
        html_content: zod_1.z.string().optional(),
        css_content: zod_1.z.string().optional(),
        js_content: zod_1.z.string().optional(),
        quality_score: zod_1.z.number().min(0).max(1).optional(),
        interactions_captured: zod_1.z.array(zod_1.z.object({
            interaction_type: zod_1.z.enum(['click', 'input', 'hover', 'submit']),
            element_id: zod_1.z.string(),
            timestamp_utc: zod_1.z.string().datetime(),
            value: zod_1.z.any().optional()
        })).optional()
    }),
    metadata: exports.ICRMetadataSchema.extend({
        completed_at_utc: zod_1.z.string().datetime()
    })
});
// ============================================================================
// UNION TYPES
// ============================================================================
/**
 * Union of all mode request types
 */
exports.ICRModeRequestSchema = zod_1.z.discriminatedUnion('mode', [
    exports.RefineModeRequestSchema,
    exports.ReactModeRequestSchema,
    exports.DeepthinkModeRequestSchema,
    exports.AdaptiveDeepthinkRequestSchema,
    exports.AgenticModeRequestSchema,
    exports.ContextualModeRequestSchema,
    exports.GenerativeUIModeRequestSchema
]);
/**
 * Union of all mode response types
 */
exports.ICRModeResponseSchema = zod_1.z.discriminatedUnion('mode', [
    exports.RefineModeResponseSchema,
    exports.ReactModeResponseSchema,
    exports.DeepthinkModeResponseSchema,
    exports.AdaptiveDeepthinkResponseSchema,
    exports.AgenticModeResponseSchema,
    exports.ContextualModeResponseSchema,
    exports.GenerativeUIModeResponseSchema
]);
// ============================================================================
// HEALTH CHECK
// ============================================================================
/**
 * Health check request schema
 */
exports.ICRHealthCheckRequestSchema = zod_1.z.object({
    correlation_id: zod_1.z.string().uuid().optional()
});
/**
 * Health check response schema
 */
exports.ICRHealthCheckResponseSchema = zod_1.z.object({
    status: zod_1.z.enum(['healthy', 'degraded', 'unhealthy']),
    version: zod_1.z.string(),
    available_modes: zod_1.z.array(exports.ModeTypeSchema),
    timestamp_utc: zod_1.z.string().datetime(),
    uptime_seconds: zod_1.z.number().nonnegative(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
//# sourceMappingURL=icr-canonical.js.map