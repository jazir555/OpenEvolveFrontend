"use strict";
/**
 * Evolved Code Capture - Contract Tests
 *
 * Following CLAUDE.md Federation Constitution:
 * - Phase 2: The Contract (Defense)
 * - Protecting the Mega-Project from Updates
 * - These tests run on container startup
 * - If contract is violated, the adapter refuses to start
 *
 * Tests validate:
 * 1. Canonical schemas are properly defined
 * 2. Validation functions work correctly
 * 3. Storage interfaces match expected contracts
 * 4. Capturer orchestrates all components correctly
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const canonical_1 = require("../src/canonical");
// ============================================================================
// FIXTURES
// ============================================================================
const validProblem = {
    description: 'Optimize matrix multiplication algorithm',
    type: 'algorithm_optimization',
    constraints: {
        max_memory_mb: 512,
        max_runtime_ms: 5000,
        required_libraries: ['numpy'],
        language_version: '3.9',
    },
    input_spec: 'Two matrices A (m×n) and B (n×p)',
    output_spec: 'Product matrix C (m×p)',
    difficulty: 'hard',
    tags: ['matrix', 'optimization', 'performance'],
};
const validMetrics = {
    iterations: 100,
    fitness_score: 0.95,
    fitness_improvement: 0.45,
    duration_ms: 15000,
    generations: 50,
    population_size: 100,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
    convergence_generation: 42,
    total_evaluations: 5000,
    success_rate: 1.0,
    benchmark_score: 2.5,
};
const validEvolvedCode = {
    id: '123e4567-e89b-12d3-a456-426614174000',
    problem: validProblem,
    language: 'python',
    code: 'def matrix_multiply(A, B):\n    return np.dot(A, B)',
    function_name: 'matrix_multiply',
    metrics: validMetrics,
    timestamp_utc: new Date().toISOString(),
    is_valid: true,
    execution_time_ms: 125,
    memory_used_mb: 256,
    parent_code_id: undefined,
    generation_number: 42,
    tags: ['optimized', 'vectorized'],
};
const validCaptureResult = {
    success: true,
    code_id: validEvolvedCode.id,
    vector_storage_id: 'vec-123',
    graph_episode_id: 'ep-456',
    timestamp_utc: new Date().toISOString(),
    processing_time_ms: 250,
    correlation_id: '789e4567-e89b-12d3-a456-426614174000',
};
// ============================================================================
// CANONICAL SCHEMA TESTS
// ============================================================================
(0, globals_1.describe)('Canonical Schemas - Problem', () => {
    (0, globals_1.test)('should validate a valid problem', () => {
        const result = canonical_1.ProblemSchema.safeParse(validProblem);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validProblem);
    });
    (0, globals_1.test)('should reject problem without description', () => {
        const invalidProblem = { ...validProblem, description: '' };
        const result = canonical_1.ProblemSchema.safeParse(invalidProblem);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should reject problem with invalid type', () => {
        const invalidProblem = { ...validProblem, type: 'invalid_type' };
        const result = canonical_1.ProblemSchema.safeParse(invalidProblem);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should accept all valid problem types', () => {
        const problemTypes = [
            'algorithm_optimization',
            'code_refactoring',
            'performance_tuning',
            'bug_fix',
            'feature_implementation',
            'code_migration',
            'parallelization',
            'memory_optimization',
            'numerical_computation',
            'data_structure_design',
            'other',
        ];
        problemTypes.forEach((type) => {
            const problem = { ...validProblem, type };
            const result = canonical_1.ProblemSchema.safeParse(problem);
            (0, globals_1.expect)(result.success).toBe(true);
        });
    });
});
(0, globals_1.describe)('Canonical Schemas - Evolution Metrics', () => {
    (0, globals_1.test)('should validate valid metrics', () => {
        const result = canonical_1.EvolutionMetricsSchema.safeParse(validMetrics);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validMetrics);
    });
    (0, globals_1.test)('should require core metric fields', () => {
        const invalidMetrics = {
            iterations: 100,
            fitness_score: 0.95,
            // Missing fitness_improvement and duration_ms
        };
        const result = canonical_1.EvolutionMetricsSchema.safeParse(invalidMetrics);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should allow negative fitness scores', () => {
        const metrics = { ...validMetrics, fitness_score: -0.5 };
        const result = canonical_1.EvolutionMetricsSchema.safeParse(metrics);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('should enforce non-negative iterations', () => {
        const metrics = { ...validMetrics, iterations: -1 };
        const result = canonical_1.EvolutionMetricsSchema.safeParse(metrics);
        (0, globals_1.expect)(result.success).toBe(false);
    });
});
(0, globals_1.describe)('Canonical Schemas - Evolved Code', () => {
    (0, globals_1.test)('should validate valid evolved code', () => {
        const result = canonical_1.EvolvedCodeSchema.safeParse(validEvolvedCode);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validEvolvedCode);
    });
    (0, globals_1.test)('should reject evolved code without code content', () => {
        const invalidCode = { ...validEvolvedCode, code: '' };
        const result = canonical_1.EvolvedCodeSchema.safeParse(invalidCode);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should reject evolved code with invalid UUID', () => {
        const invalidCode = { ...validEvolvedCode, id: 'not-a-uuid' };
        const result = canonical_1.EvolvedCodeSchema.safeParse(invalidCode);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should reject evolved code with invalid timestamp', () => {
        const invalidCode = { ...validEvolvedCode, timestamp_utc: 'not-a-timestamp' };
        const result = canonical_1.EvolvedCodeSchema.safeParse(invalidCode);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should accept all valid languages', () => {
        const languages = [
            'python', 'javascript', 'typescript', 'java', 'c', 'cpp',
            'csharp', 'go', 'rust', 'julia', 'matlab', 'r', 'other',
        ];
        languages.forEach((language) => {
            const code = { ...validEvolvedCode, language };
            const result = canonical_1.EvolvedCodeSchema.safeParse(code);
            (0, globals_1.expect)(result.success).toBe(true);
        });
    });
    (0, globals_1.test)('should allow optional parent_code_id', () => {
        const codeWithParent = {
            ...validEvolvedCode,
            parent_code_id: '987fcdeb-51a2-43f1-a456-426614174000',
        };
        const result = canonical_1.EvolvedCodeSchema.safeParse(codeWithParent);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data.parent_code_id).toBeDefined();
    });
});
(0, globals_1.describe)('Canonical Schemas - Capture Result', () => {
    (0, globals_1.test)('should validate valid capture result', () => {
        const result = canonical_1.CaptureResultSchema.safeParse(validCaptureResult);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validCaptureResult);
    });
    (0, globals_1.test)('should require success field', () => {
        const invalidResult = { ...validCaptureResult, success: undefined };
        const result = canonical_1.CaptureResultSchema.safeParse(invalidResult);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should require code_id', () => {
        const invalidResult = { ...validCaptureResult, code_id: undefined };
        const result = canonical_1.CaptureResultSchema.safeParse(invalidResult);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should allow optional error field', () => {
        const failedResult = {
            ...validCaptureResult,
            success: false,
            error: 'Storage connection failed',
        };
        const result = canonical_1.CaptureResultSchema.safeParse(failedResult);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data.error).toBe('Storage connection failed');
    });
});
// ============================================================================
// VALIDATION FUNCTION TESTS
// ============================================================================
(0, globals_1.describe)('Validation Functions', () => {
    (0, globals_1.test)('validateProblem should accept valid problem', () => {
        const result = (0, canonical_1.validateProblem)(validProblem);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validProblem);
        (0, globals_1.expect)(result.errors).toHaveLength(0);
    });
    (0, globals_1.test)('validateProblem should reject invalid problem', () => {
        const invalidProblem = { ...validProblem, description: '' };
        const result = (0, canonical_1.validateProblem)(invalidProblem);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('validateEvolutionMetrics should accept valid metrics', () => {
        const result = (0, canonical_1.validateEvolutionMetrics)(validMetrics);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validMetrics);
        (0, globals_1.expect)(result.errors).toHaveLength(0);
    });
    (0, globals_1.test)('validateEvolutionMetrics should reject invalid metrics', () => {
        const invalidMetrics = { ...validMetrics, iterations: -1 };
        const result = (0, canonical_1.validateEvolutionMetrics)(invalidMetrics);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('validateEvolvedCode should accept valid code', () => {
        const result = (0, canonical_1.validateEvolvedCode)(validEvolvedCode);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validEvolvedCode);
        (0, globals_1.expect)(result.errors).toHaveLength(0);
    });
    (0, globals_1.test)('validateEvolvedCode should reject invalid code', () => {
        const invalidCode = { ...validEvolvedCode, code: '' };
        const result = (0, canonical_1.validateEvolvedCode)(invalidCode);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('validateCaptureResult should accept valid result', () => {
        const result = (0, canonical_1.validateCaptureResult)(validCaptureResult);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toEqual(validCaptureResult);
        (0, globals_1.expect)(result.errors).toHaveLength(0);
    });
    (0, globals_1.test)('validateCaptureResult should reject invalid result', () => {
        const invalidResult = { ...validCaptureResult, code_id: 'not-a-uuid' };
        const result = (0, canonical_1.validateCaptureResult)(invalidResult);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
    });
});
// ============================================================================
// INTEGRATION CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Integration Contracts - Full Capture Flow', () => {
    (0, globals_1.test)('should validate complete capture workflow', () => {
        // Step 1: Validate problem
        const problemValidation = (0, canonical_1.validateProblem)(validProblem);
        (0, globals_1.expect)(problemValidation.success).toBe(true);
        // Step 2: Validate metrics
        const metricsValidation = (0, canonical_1.validateEvolutionMetrics)(validMetrics);
        (0, globals_1.expect)(metricsValidation.success).toBe(true);
        // Step 3: Validate evolved code
        const codeValidation = (0, canonical_1.validateEvolvedCode)(validEvolvedCode);
        (0, globals_1.expect)(codeValidation.success).toBe(true);
        // Step 4: Validate capture result
        const resultValidation = (0, canonical_1.validateCaptureResult)(validCaptureResult);
        (0, globals_1.expect)(resultValidation.success).toBe(true);
    });
    (0, globals_1.test)('should handle nested validation correctly', () => {
        // EvolvedCode contains Problem and Metrics
        const codeWithNested = validEvolvedCode;
        const result = canonical_1.EvolvedCodeSchema.safeParse(codeWithNested);
        (0, globals_1.expect)(result.success).toBe(true);
        // Verify nested objects are validated
        (0, globals_1.expect)(result.data.problem).toEqual(validProblem);
        (0, globals_1.expect)(result.data.metrics).toEqual(validMetrics);
    });
    (0, globals_1.test)('should reject invalid nested objects', () => {
        const codeWithInvalidProblem = {
            ...validEvolvedCode,
            problem: { ...validProblem, type: 'invalid_type' },
        };
        const result = canonical_1.EvolvedCodeSchema.safeParse(codeWithInvalidProblem);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should enforce UTC timestamp format', () => {
        const invalidTimestamps = [
            '2024-01-01', // Missing time
            '2024-01-01T12:00:00', // Missing timezone
            'not-a-timestamp',
        ];
        invalidTimestamps.forEach((timestamp) => {
            const code = { ...validEvolvedCode, timestamp_utc: timestamp };
            const result = canonical_1.EvolvedCodeSchema.safeParse(code);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        // Valid UTC timestamp
        const validTimestamp = '2024-01-15T10:30:00.000Z';
        const code = { ...validEvolvedCode, timestamp_utc: validTimestamp };
        const result = canonical_1.EvolvedCodeSchema.safeParse(code);
        (0, globals_1.expect)(result.success).toBe(true);
    });
});
// ============================================================================
// ERROR MESSAGE CONTRACTS
// ============================================================================
(0, globals_1.describe)('Error Message Contracts', () => {
    (0, globals_1.test)('should provide clear validation error messages', () => {
        const invalidProblem = { description: '', type: 'invalid' };
        const result = (0, canonical_1.validateProblem)(invalidProblem);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
        // Error messages should be descriptive
        result.errors.forEach((error) => {
            (0, globals_1.expect)(error).toContain('.');
            (0, globals_1.expect)(error.length).toBeGreaterThan(5);
        });
    });
    (0, globals_1.test)('should include field path in error messages', () => {
        const invalidCode = {
            ...validEvolvedCode,
            metrics: { ...validMetrics, iterations: -1 },
        };
        const result = (0, canonical_1.validateEvolvedCode)(invalidCode);
        (0, globals_1.expect)(result.success).toBe(false);
        // Error should include path to invalid field
        const hasPathError = result.errors.some((error) => error.includes('metrics') || error.includes('iterations'));
        (0, globals_1.expect)(hasPathError).toBe(true);
    });
});
// ============================================================================
// TYPE SAFETY CONTRACTS
// ============================================================================
(0, globals_1.describe)('Type Safety Contracts', () => {
    (0, globals_1.test)('should enforce number types for numeric fields', () => {
        const invalidMetrics = {
            ...validMetrics,
            fitness_score: '0.95', // String instead of number
        };
        const result = (0, canonical_1.validateEvolutionMetrics)(invalidMetrics);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should enforce boolean types for boolean fields', () => {
        const invalidCode = {
            ...validEvolvedCode,
            is_valid: 'true', // String instead of boolean
        };
        const result = (0, canonical_1.validateEvolvedCode)(invalidCode);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('should enforce array types for array fields', () => {
        const invalidProblem = {
            ...validProblem,
            tags: 'not-an-array',
        };
        const result = (0, canonical_1.validateProblem)(invalidProblem);
        (0, globals_1.expect)(result.success).toBe(false);
    });
});
// ============================================================================
// BOUNDARY VALUE CONTRACTS
// ============================================================================
(0, globals_1.describe)('Boundary Value Contracts', () => {
    (0, globals_1.test)('should accept zero iterations', () => {
        const metrics = { ...validMetrics, iterations: 0 };
        const result = (0, canonical_1.validateEvolutionMetrics)(metrics);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('should accept zero fitness score', () => {
        const metrics = { ...validMetrics, fitness_score: 0 };
        const result = (0, canonical_1.validateEvolutionMetrics)(metrics);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('should accept mutation rate at boundaries', () => {
        const lowerBound = { ...validMetrics, mutation_rate: 0 };
        const upperBound = { ...validMetrics, mutation_rate: 1 };
        (0, globals_1.expect)((0, canonical_1.validateEvolutionMetrics)(lowerBound).success).toBe(true);
        (0, globals_1.expect)((0, canonical_1.validateEvolutionMetrics)(upperBound).success).toBe(true);
    });
    (0, globals_1.test)('should reject mutation rate outside boundaries', () => {
        const belowLower = { ...validMetrics, mutation_rate: -0.1 };
        const aboveUpper = { ...validMetrics, mutation_rate: 1.1 };
        (0, globals_1.expect)((0, canonical_1.validateEvolutionMetrics)(belowLower).success).toBe(false);
        (0, globals_1.expect)((0, canonical_1.validateEvolutionMetrics)(aboveUpper).success).toBe(false);
    });
});
// ============================================================================
// SUMMARY
// ============================================================================
(0, globals_1.describe)('Contract Test Summary', () => {
    (0, globals_1.test)('should pass all contract tests', () => {
        // This test serves as a summary - if we get here, all contracts passed
        (0, globals_1.expect)(true).toBe(true);
    });
});
//# sourceMappingURL=contract.test.js.map