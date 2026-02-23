"use strict";
/**
 * Z3 Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Z3 SMT solver interactions.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for Z3 data in the glue layer.
 * Do not pass raw Z3 API responses between services.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Z3Examples = exports.KnowledgeGraphResponse = exports.Relation = exports.Entity = exports.SolverResponse = exports.SolverRequest = exports.Z3ResultType = void 0;
exports.transformZ3ResponseToCanonical = transformZ3ResponseToCanonical;
exports.transformCanonicalToZ3Request = transformCanonicalToZ3Request;
exports.validateSolverRequest = validateSolverRequest;
exports.validateSolverResponse = validateSolverResponse;
exports.validateKnowledgeGraphResponse = validateKnowledgeGraphResponse;
const zod_1 = require("zod");
/**
 * Core Result Types from Z3
 */
exports.Z3ResultType = zod_1.z.enum([
    'sat', // Satisfiable - a solution exists
    'unsat', // Unsatisfiable - no solution exists
    'unknown', // Unknown - Z3 could not determine
]);
/**
 * Z3 Solver Request Schema
 *
 * Represents a request to solve a constraint problem using Z3.
 */
exports.SolverRequest = zod_1.z.object({
    // The problem description in SMT-LIB format or Z3-specific syntax
    problem: zod_1.z.string()
        .min(1, "Problem statement cannot be empty")
        .describe("The constraint problem to solve in SMT-LIB or Z3 format"),
    // Optional tactics to guide the solver (e.g., 'simplify', 'solve-eqs')
    tactics: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Optional array of Z3 tactics to apply"),
    // Timeout in milliseconds (MANDATORY - no infinite hangs)
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(300000, "Timeout cannot exceed 5 minutes")
        .describe("Solver timeout in milliseconds"),
    // Optional metadata for tracking and correlation
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata for observability and tracking"),
    // Optional correlation ID for distributed tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Z3 Solver Response Schema
 *
 * Represents the response from Z3 after solving a problem.
 */
exports.SolverResponse = zod_1.z.object({
    // The core result from Z3
    result: exports.Z3ResultType.describe("Solver result: sat, unsat, or unknown"),
    // Human-readable explanation of the result
    explanation: zod_1.z.string().optional()
        .describe("Human-readable explanation of the result"),
    // Model/counterexample if satisfiable (for 'sat' results)
    model: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Model or counterexample (present for 'sat' results)"),
    // Proof if unsatisfiable (for 'unsat' results)
    proof: zod_1.z.string().optional()
        .describe("Proof of unsatisfiability (present for 'unsat' results)"),
    // Reason if unknown
    reason: zod_1.z.string().optional()
        .describe("Reason for unknown result (e.g., timeout, resource limit)"),
    // Execution metadata
    metadata: zod_1.z.object({
        solver_version: zod_1.z.string().optional().describe("Z3 version used"),
        solve_time_ms: zod_1.z.number().optional().describe("Actual solve time in milliseconds"),
        tactics_applied: zod_1.z.array(zod_1.z.string()).optional().describe("Tactics that were applied"),
        memory_used_mb: zod_1.z.number().optional().describe("Memory usage in MB"),
    }).optional().describe("Execution metadata"),
    // Original correlation ID for tracing
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // Timestamp in UTC (Law of UTC)
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Knowledge Graph Entity Schema
 *
 * Represents an entity extracted from mathematical content.
 */
exports.Entity = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the entity"),
    type: zod_1.z.enum([
        'variable',
        'constant',
        'function',
        'predicate',
        'theorem',
        'axiom',
        'definition',
        'other',
    ]).describe("Type of the entity"),
    name: zod_1.z.string().describe("Name of the entity"),
    description: zod_1.z.string().optional().describe("Optional description"),
    properties: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional properties of the entity"),
    source_location: zod_1.z.object({
        file: zod_1.z.string().optional(),
        line_start: zod_1.z.number().optional(),
        line_end: zod_1.z.number().optional(),
    }).optional().describe("Source location of the entity"),
});
/**
 * Knowledge Graph Relation Schema
 *
 * Represents a relationship between entities in mathematical content.
 */
exports.Relation = zod_1.z.object({
    id: zod_1.z.string().uuid().describe("Unique identifier for the relation"),
    source: zod_1.z.string().uuid().describe("ID of the source entity"),
    target: zod_1.z.string().uuid().describe("ID of the target entity"),
    type: zod_1.z.enum([
        'depends_on',
        'implements',
        'refines',
        'contradicts',
        'implies',
        'equivalent_to',
        'instance_of',
        'uses',
        'defines',
        'proves',
        'other',
    ]).describe("Type of the relation"),
    properties: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional properties of the relation"),
});
/**
 * Knowledge Graph Response Schema
 *
 * Represents a knowledge graph extracted from mathematical content.
 */
exports.KnowledgeGraphResponse = zod_1.z.object({
    entities: zod_1.z.array(exports.Entity).describe("Array of entities in the graph"),
    relations: zod_1.z.array(exports.Relation).describe("Array of relations between entities"),
    metadata: zod_1.z.object({
        extraction_method: zod_1.z.string().optional().describe("Method used for extraction"),
        confidence_score: zod_1.z.number().min(0).max(1).optional()
            .describe("Confidence score for the extraction (0-1)"),
        processing_time_ms: zod_1.z.number().optional()
            .describe("Time taken to extract the graph"),
    }).optional().describe("Extraction metadata"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Transformation Functions
 *
 * Helper functions to convert between external formats and the canonical schema.
 */
/**
 * Transform raw Z3 API response to canonical SolverResponse
 */
function transformZ3ResponseToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        result: exports.Z3ResultType.parse(rawResponse.result.toLowerCase()),
        explanation: rawResponse.explanation,
        model: rawResponse.model,
        proof: rawResponse.proof,
        reason: rawResponse.reason,
        metadata: {
            solver_version: rawResponse.version,
            solve_time_ms: rawResponse.time,
            tactics_applied: rawResponse.tactics,
            memory_used_mb: rawResponse.memory,
        },
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * Transform canonical SolverRequest to Z3 API format
 */
function transformCanonicalToZ3Request(canonicalRequest) {
    return {
        // Convert to SMT-LIB format or Z3-specific format
        problem: canonicalRequest.problem,
        tactics: canonicalRequest.tactics || [],
        timeout: canonicalRequest.timeout_ms,
        // Metadata is passed through for tracing
        metadata: canonicalRequest.metadata,
    };
}
/**
 * Validate a SolverRequest against the schema
 */
function validateSolverRequest(data) {
    const result = exports.SolverRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate a SolverResponse against the schema
 */
function validateSolverResponse(data) {
    const result = exports.SolverResponse.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate a KnowledgeGraphResponse against the schema
 */
function validateKnowledgeGraphResponse(data) {
    const result = exports.KnowledgeGraphResponse.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Example usage and validation examples
 */
exports.Z3Examples = {
    validSolverRequest: {
        problem: "(declare-const x Int) (assert (> x 10)) (check-sat)",
        tactics: ["simplify", "solve-eqs"],
        timeout_ms: 5000,
        metadata: {
            source: "theorem_prover",
            priority: "high",
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    },
    validSolverResponse: {
        result: "sat",
        explanation: "The constraint is satisfiable",
        model: {
            x: 11,
        },
        metadata: {
            solver_version: "4.12.1",
            solve_time_ms: 45,
            tactics_applied: ["simplify"],
            memory_used_mb: 8,
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        timestamp: "2025-02-03T12:34:56.789Z",
    },
    validKnowledgeGraphResponse: {
        entities: [
            {
                id: "550e8400-e29b-41d4-a716-446655440001",
                type: "variable",
                name: "x",
                description: "Integer variable",
                properties: {
                    domain: "Int",
                },
            },
            {
                id: "550e8400-e29b-41d4-a716-446655440002",
                type: "constant",
                name: "10",
                description: "Integer constant",
                properties: {
                    value: 10,
                },
            },
        ],
        relations: [
            {
                id: "550e8400-e29b-41d4-a716-446655440003",
                source: "550e8400-e29b-41d4-a716-446655440001",
                target: "550e8400-e29b-41d4-a716-446655440002",
                type: "greater_than",
            },
        ],
        metadata: {
            extraction_method: "smt_parser",
            confidence_score: 0.95,
            processing_time_ms: 12,
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        timestamp: "2025-02-03T12:34:56.789Z",
    },
};
//# sourceMappingURL=z3-canonical.js.map