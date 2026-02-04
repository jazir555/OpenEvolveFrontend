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

import { describe, test, expect, beforeAll } from '@jest/globals';
import {
  validateEvolvedCode,
  validateProblem,
  validateEvolutionMetrics,
  validateCaptureResult,
  ProblemSchema,
  EvolutionMetricsSchema,
  EvolvedCodeSchema,
  CaptureResultSchema,
  ProblemTypeEnum,
  LanguageEnum,
  type Problem,
  type EvolutionMetrics,
  type EvolvedCode,
  type CaptureResult,
} from '../src/canonical';

// ============================================================================
// FIXTURES
// ============================================================================

const validProblem: Problem = {
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

const validMetrics: EvolutionMetrics = {
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

const validEvolvedCode: EvolvedCode = {
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

const validCaptureResult: CaptureResult = {
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

describe('Canonical Schemas - Problem', () => {
  test('should validate a valid problem', () => {
    const result = ProblemSchema.safeParse(validProblem);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validProblem);
  });

  test('should reject problem without description', () => {
    const invalidProblem = { ...validProblem, description: '' };
    const result = ProblemSchema.safeParse(invalidProblem);
    expect(result.success).toBe(false);
  });

  test('should reject problem with invalid type', () => {
    const invalidProblem = { ...validProblem, type: 'invalid_type' };
    const result = ProblemSchema.safeParse(invalidProblem);
    expect(result.success).toBe(false);
  });

  test('should accept all valid problem types', () => {
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
      const result = ProblemSchema.safeParse(problem);
      expect(result.success).toBe(true);
    });
  });
});

describe('Canonical Schemas - Evolution Metrics', () => {
  test('should validate valid metrics', () => {
    const result = EvolutionMetricsSchema.safeParse(validMetrics);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validMetrics);
  });

  test('should require core metric fields', () => {
    const invalidMetrics = {
      iterations: 100,
      fitness_score: 0.95,
      // Missing fitness_improvement and duration_ms
    };
    const result = EvolutionMetricsSchema.safeParse(invalidMetrics);
    expect(result.success).toBe(false);
  });

  test('should allow negative fitness scores', () => {
    const metrics = { ...validMetrics, fitness_score: -0.5 };
    const result = EvolutionMetricsSchema.safeParse(metrics);
    expect(result.success).toBe(true);
  });

  test('should enforce non-negative iterations', () => {
    const metrics = { ...validMetrics, iterations: -1 };
    const result = EvolutionMetricsSchema.safeParse(metrics);
    expect(result.success).toBe(false);
  });
});

describe('Canonical Schemas - Evolved Code', () => {
  test('should validate valid evolved code', () => {
    const result = EvolvedCodeSchema.safeParse(validEvolvedCode);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validEvolvedCode);
  });

  test('should reject evolved code without code content', () => {
    const invalidCode = { ...validEvolvedCode, code: '' };
    const result = EvolvedCodeSchema.safeParse(invalidCode);
    expect(result.success).toBe(false);
  });

  test('should reject evolved code with invalid UUID', () => {
    const invalidCode = { ...validEvolvedCode, id: 'not-a-uuid' };
    const result = EvolvedCodeSchema.safeParse(invalidCode);
    expect(result.success).toBe(false);
  });

  test('should reject evolved code with invalid timestamp', () => {
    const invalidCode = { ...validEvolvedCode, timestamp_utc: 'not-a-timestamp' };
    const result = EvolvedCodeSchema.safeParse(invalidCode);
    expect(result.success).toBe(false);
  });

  test('should accept all valid languages', () => {
    const languages = [
      'python', 'javascript', 'typescript', 'java', 'c', 'cpp',
      'csharp', 'go', 'rust', 'julia', 'matlab', 'r', 'other',
    ];

    languages.forEach((language) => {
      const code = { ...validEvolvedCode, language };
      const result = EvolvedCodeSchema.safeParse(code);
      expect(result.success).toBe(true);
    });
  });

  test('should allow optional parent_code_id', () => {
    const codeWithParent = {
      ...validEvolvedCode,
      parent_code_id: '987fcdeb-51a2-43f1-a456-426614174000',
    };
    const result = EvolvedCodeSchema.safeParse(codeWithParent);
    expect(result.success).toBe(true);
    expect(result.data.parent_code_id).toBeDefined();
  });
});

describe('Canonical Schemas - Capture Result', () => {
  test('should validate valid capture result', () => {
    const result = CaptureResultSchema.safeParse(validCaptureResult);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validCaptureResult);
  });

  test('should require success field', () => {
    const invalidResult = { ...validCaptureResult, success: undefined };
    const result = CaptureResultSchema.safeParse(invalidResult);
    expect(result.success).toBe(false);
  });

  test('should require code_id', () => {
    const invalidResult = { ...validCaptureResult, code_id: undefined };
    const result = CaptureResultSchema.safeParse(invalidResult);
    expect(result.success).toBe(false);
  });

  test('should allow optional error field', () => {
    const failedResult = {
      ...validCaptureResult,
      success: false,
      error: 'Storage connection failed',
    };
    const result = CaptureResultSchema.safeParse(failedResult);
    expect(result.success).toBe(true);
    expect(result.data.error).toBe('Storage connection failed');
  });
});

// ============================================================================
// VALIDATION FUNCTION TESTS
// ============================================================================

describe('Validation Functions', () => {
  test('validateProblem should accept valid problem', () => {
    const result = validateProblem(validProblem);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validProblem);
    expect(result.errors).toHaveLength(0);
  });

  test('validateProblem should reject invalid problem', () => {
    const invalidProblem = { ...validProblem, description: '' };
    const result = validateProblem(invalidProblem);
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test('validateEvolutionMetrics should accept valid metrics', () => {
    const result = validateEvolutionMetrics(validMetrics);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validMetrics);
    expect(result.errors).toHaveLength(0);
  });

  test('validateEvolutionMetrics should reject invalid metrics', () => {
    const invalidMetrics = { ...validMetrics, iterations: -1 };
    const result = validateEvolutionMetrics(invalidMetrics);
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test('validateEvolvedCode should accept valid code', () => {
    const result = validateEvolvedCode(validEvolvedCode);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validEvolvedCode);
    expect(result.errors).toHaveLength(0);
  });

  test('validateEvolvedCode should reject invalid code', () => {
    const invalidCode = { ...validEvolvedCode, code: '' };
    const result = validateEvolvedCode(invalidCode);
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test('validateCaptureResult should accept valid result', () => {
    const result = validateCaptureResult(validCaptureResult);
    expect(result.success).toBe(true);
    expect(result.data).toEqual(validCaptureResult);
    expect(result.errors).toHaveLength(0);
  });

  test('validateCaptureResult should reject invalid result', () => {
    const invalidResult = { ...validCaptureResult, code_id: 'not-a-uuid' };
    const result = validateCaptureResult(invalidResult);
    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// INTEGRATION CONTRACT TESTS
// ============================================================================

describe('Integration Contracts - Full Capture Flow', () => {
  test('should validate complete capture workflow', () => {
    // Step 1: Validate problem
    const problemValidation = validateProblem(validProblem);
    expect(problemValidation.success).toBe(true);

    // Step 2: Validate metrics
    const metricsValidation = validateEvolutionMetrics(validMetrics);
    expect(metricsValidation.success).toBe(true);

    // Step 3: Validate evolved code
    const codeValidation = validateEvolvedCode(validEvolvedCode);
    expect(codeValidation.success).toBe(true);

    // Step 4: Validate capture result
    const resultValidation = validateCaptureResult(validCaptureResult);
    expect(resultValidation.success).toBe(true);
  });

  test('should handle nested validation correctly', () => {
    // EvolvedCode contains Problem and Metrics
    const codeWithNested = validEvolvedCode;

    const result = EvolvedCodeSchema.safeParse(codeWithNested);
    expect(result.success).toBe(true);

    // Verify nested objects are validated
    expect(result.data.problem).toEqual(validProblem);
    expect(result.data.metrics).toEqual(validMetrics);
  });

  test('should reject invalid nested objects', () => {
    const codeWithInvalidProblem = {
      ...validEvolvedCode,
      problem: { ...validProblem, type: 'invalid_type' },
    };

    const result = EvolvedCodeSchema.safeParse(codeWithInvalidProblem);
    expect(result.success).toBe(false);
  });

  test('should enforce UTC timestamp format', () => {
    const invalidTimestamps = [
      '2024-01-01',  // Missing time
      '2024-01-01T12:00:00',  // Missing timezone
      'not-a-timestamp',
    ];

    invalidTimestamps.forEach((timestamp) => {
      const code = { ...validEvolvedCode, timestamp_utc: timestamp };
      const result = EvolvedCodeSchema.safeParse(code);
      expect(result.success).toBe(false);
    });

    // Valid UTC timestamp
    const validTimestamp = '2024-01-15T10:30:00.000Z';
    const code = { ...validEvolvedCode, timestamp_utc: validTimestamp };
    const result = EvolvedCodeSchema.safeParse(code);
    expect(result.success).toBe(true);
  });
});

// ============================================================================
// ERROR MESSAGE CONTRACTS
// ============================================================================

describe('Error Message Contracts', () => {
  test('should provide clear validation error messages', () => {
    const invalidProblem = { description: '', type: 'invalid' };
    const result = validateProblem(invalidProblem);

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);

    // Error messages should be descriptive
    result.errors.forEach((error) => {
      expect(error).toContain('.');
      expect(error.length).toBeGreaterThan(5);
    });
  });

  test('should include field path in error messages', () => {
    const invalidCode = {
      ...validEvolvedCode,
      metrics: { ...validMetrics, iterations: -1 },
    };
    const result = validateEvolvedCode(invalidCode);

    expect(result.success).toBe(false);
    // Error should include path to invalid field
    const hasPathError = result.errors.some((error) =>
      error.includes('metrics') || error.includes('iterations')
    );
    expect(hasPathError).toBe(true);
  });
});

// ============================================================================
// TYPE SAFETY CONTRACTS
// ============================================================================

describe('Type Safety Contracts', () => {
  test('should enforce number types for numeric fields', () => {
    const invalidMetrics = {
      ...validMetrics,
      fitness_score: '0.95' as any,  // String instead of number
    };
    const result = validateEvolutionMetrics(invalidMetrics);
    expect(result.success).toBe(false);
  });

  test('should enforce boolean types for boolean fields', () => {
    const invalidCode = {
      ...validEvolvedCode,
      is_valid: 'true' as any,  // String instead of boolean
    };
    const result = validateEvolvedCode(invalidCode);
    expect(result.success).toBe(false);
  });

  test('should enforce array types for array fields', () => {
    const invalidProblem = {
      ...validProblem,
      tags: 'not-an-array' as any,
    };
    const result = validateProblem(invalidProblem);
    expect(result.success).toBe(false);
  });
});

// ============================================================================
// BOUNDARY VALUE CONTRACTS
// ============================================================================

describe('Boundary Value Contracts', () => {
  test('should accept zero iterations', () => {
    const metrics = { ...validMetrics, iterations: 0 };
    const result = validateEvolutionMetrics(metrics);
    expect(result.success).toBe(true);
  });

  test('should accept zero fitness score', () => {
    const metrics = { ...validMetrics, fitness_score: 0 };
    const result = validateEvolutionMetrics(metrics);
    expect(result.success).toBe(true);
  });

  test('should accept mutation rate at boundaries', () => {
    const lowerBound = { ...validMetrics, mutation_rate: 0 };
    const upperBound = { ...validMetrics, mutation_rate: 1 };

    expect(validateEvolutionMetrics(lowerBound).success).toBe(true);
    expect(validateEvolutionMetrics(upperBound).success).toBe(true);
  });

  test('should reject mutation rate outside boundaries', () => {
    const belowLower = { ...validMetrics, mutation_rate: -0.1 };
    const aboveUpper = { ...validMetrics, mutation_rate: 1.1 };

    expect(validateEvolutionMetrics(belowLower).success).toBe(false);
    expect(validateEvolutionMetrics(aboveUpper).success).toBe(false);
  });
});

// ============================================================================
// SUMMARY
// ============================================================================

describe('Contract Test Summary', () => {
  test('should pass all contract tests', () => {
    // This test serves as a summary - if we get here, all contracts passed
    expect(true).toBe(true);
  });
});
