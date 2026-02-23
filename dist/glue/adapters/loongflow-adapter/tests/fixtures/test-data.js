"use strict";
/**
 * LoongFlow Contract Test Fixtures
 *
 * This file provides test data for contract validation.
 * All fixtures represent realistic API responses from LoongFlow core.
 *
 * Purpose: Phase 2 - The Contract (Defense)
 * Law of Runtime Truth: These fixtures MUST match actual LoongFlow API responses
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.EMPTY_SAMPLE = exports.SAMPLED_SOLUTION = exports.INTERNAL_ERROR = exports.TIMEOUT_ERROR = exports.VALIDATION_ERROR = exports.NOT_FOUND_ERROR = exports.CHECKPOINTS_LIST = exports.CHECKPOINT_LOAD_RESPONSE = exports.CHECKPOINT_SAVE_RESPONSE = exports.CHECKPOINT_INFO = exports.TOP_K_BEST_SOLUTIONS = exports.SINGLE_ISLAND_BEST_SOLUTIONS = exports.BEST_SOLUTIONS_RESPONSE = exports.SINGLE_ISLAND_DATABASE_STATUS = exports.DATABASE_STATUS = exports.FAILED_EXECUTION_RESULT = exports.INTERRUPTED_EXECUTION_RESULT = exports.SUCCESSFUL_EXECUTION_RESULT = exports.FAILED_AGENT_STATE = exports.COMPLETED_AGENT_STATE = exports.RUNNING_AGENT_STATE = exports.PENDING_AGENT_STATE = exports.INVALID_SOLUTION_MISSING_FIELD = exports.INVALID_SOLUTION_ISLAND_ID = exports.INVALID_SOLUTION_ITERATION = exports.INVALID_SOLUTION_SCORE = exports.SOLUTION_WITHOUT_OPTIONAL_FIELDS = exports.SOLUTION_WITH_NULL_PARENT = exports.VALID_SOLUTION = exports.PROBLEM_SUBMISSION_RESPONSE = exports.INVALID_PROBLEM_REQUEST = exports.MINIMAL_PROBLEM_REQUEST = exports.VALID_PROBLEM_REQUEST = exports.UNHEALTHY_RESPONSE = exports.VALID_HEALTH_RESPONSE = void 0;
exports.generateTestId = generateTestId;
exports.generateTestUUID = generateTestUUID;
exports.getCurrentUTCTimestamp = getCurrentUTCTimestamp;
exports.isValidUUID = isValidUUID;
exports.isValidUTCTimestamp = isValidUTCTimestamp;
// =============================================================================
// HEALTH CHECK FIXTURES
// =============================================================================
exports.VALID_HEALTH_RESPONSE = {
    status: 'healthy',
    version: '1.0.0',
    timestamp: '2026-02-22T10:30:00.000Z',
};
exports.UNHEALTHY_RESPONSE = {
    status: 'unhealthy',
    version: '1.0.0',
    timestamp: '2026-02-22T10:30:00.000Z',
    error: 'Database connection failed',
};
// =============================================================================
// PROBLEM SUBMISSION FIXTURES
// =============================================================================
exports.VALID_PROBLEM_REQUEST = {
    task: 'Solve the traveling salesman problem for 20 cities',
    max_iterations: 100,
    target_score: 0.9,
    concurrency: 4,
    initial_code: 'def solve(cities): return 0',
    metadata: {
        domain: 'optimization',
        difficulty: 'hard',
    },
};
exports.MINIMAL_PROBLEM_REQUEST = {
    task: 'Simple optimization task',
};
exports.INVALID_PROBLEM_REQUEST = {
    // Missing required 'task' field
    max_iterations: 100,
    target_score: 0.9,
};
exports.PROBLEM_SUBMISSION_RESPONSE = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'pending',
    message: 'Problem submitted successfully',
    timestamp: '2026-02-22T10:30:00.000Z',
};
// =============================================================================
// SOLUTION DATA STRUCTURE FIXTURES
// =============================================================================
exports.VALID_SOLUTION = {
    solution: 'def optimized_tsp(cities):\n    # Use genetic algorithm\n    return best_route',
    solution_id: '123e4567-e89b-12d3-a456-426614174000',
    generate_plan: 'Use genetic algorithm with crossover and mutation operators',
    parent_id: '987fcdeb-51a2-43f1-a456-426614174000',
    island_id: 0,
    iteration: 10,
    timestamp: 1740223800.0,
    generation: 10,
    sample_cnt: 5,
    sample_weight: 0.5,
    score: 0.95,
    evaluation: 'Solution finds optimal route for 20 cities in polynomial time',
    summary: 'Improved fitness by 15% compared to baseline',
    metadata: {
        algorithm: 'genetic',
        population_size: 100,
        mutation_rate: 0.01,
    },
};
exports.SOLUTION_WITH_NULL_PARENT = {
    ...exports.VALID_SOLUTION,
    solution_id: '00000000-0000-0000-0000-000000000001',
    parent_id: '', // Empty string for initial population
    island_id: 1,
    iteration: 0,
};
exports.SOLUTION_WITHOUT_OPTIONAL_FIELDS = {
    solution: 'def solve(): return 42',
    solution_id: '123e4567-e89b-12d3-a456-426614174000',
    generate_plan: 'Simple solution',
    parent_id: '',
    island_id: 0,
    iteration: 1,
    timestamp: 1740223800.0,
    generation: 1,
    sample_cnt: 0,
    sample_weight: 0.0,
    score: 0.5,
    evaluation: 'Passable solution',
    summary: 'Basic approach',
    metadata: {},
};
exports.INVALID_SOLUTION_SCORE = {
    ...exports.VALID_SOLUTION,
    score: 1.5, // Invalid: must be 0-1
};
exports.INVALID_SOLUTION_ITERATION = {
    ...exports.VALID_SOLUTION,
    iteration: -1, // Invalid: must be non-negative
};
exports.INVALID_SOLUTION_ISLAND_ID = {
    ...exports.VALID_SOLUTION,
    island_id: -1, // Invalid: must be non-negative
};
exports.INVALID_SOLUTION_MISSING_FIELD = {
    solution: 'def solve(): return 42',
    solution_id: '123e4567-e89b-12d3-a456-426614174000',
    // Missing required: generate_plan, parent_id, island_id, iteration, timestamp,
    //                  generation, sample_cnt, sample_weight, score, evaluation, summary, metadata
};
// =============================================================================
// AGENT STATE FIXTURES
// =============================================================================
exports.PENDING_AGENT_STATE = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'idle',
    current_iteration: 0,
    max_iterations: 100,
    target_score: 0.9,
    best_score: 0.0,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: undefined,
    completion_count: 0,
    total_prompt_tokens: 0,
    total_completion_tokens: 0,
    total_cost: 0.0,
};
exports.RUNNING_AGENT_STATE = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'evolving',
    current_iteration: 45,
    max_iterations: 100,
    target_score: 0.9,
    best_score: 0.87,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: undefined,
    completion_count: 45,
    total_prompt_tokens: 150000,
    total_completion_tokens: 300000,
    total_cost: 2.50,
};
exports.COMPLETED_AGENT_STATE = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'completed',
    current_iteration: 100,
    max_iterations: 100,
    target_score: 0.9,
    best_score: 0.95,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: '2026-02-22T11:30:00.000Z',
    completion_count: 100,
    total_prompt_tokens: 300000,
    total_completion_tokens: 600000,
    total_cost: 5.00,
};
exports.FAILED_AGENT_STATE = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'failed',
    current_iteration: 23,
    max_iterations: 100,
    target_score: 0.9,
    best_score: 0.45,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: '2026-02-22T10:45:00.000Z',
    completion_count: 23,
    total_prompt_tokens: 50000,
    total_completion_tokens: 100000,
    total_cost: 0.80,
};
// =============================================================================
// EXECUTION RESULT FIXTURES
// =============================================================================
exports.SUCCESSFUL_EXECUTION_RESULT = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'completed',
    final_solution: exports.VALID_SOLUTION.solution,
    final_score: 0.95,
    best_solutions: [exports.VALID_SOLUTION],
    total_iterations: 100,
    total_tokens: 900000,
    total_cost: 5.00,
    was_interrupted: false,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: '2026-02-22T11:30:00.000Z',
};
exports.INTERRUPTED_EXECUTION_RESULT = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'cancelled',
    final_solution: undefined,
    final_score: 0.87,
    best_solutions: [exports.VALID_SOLUTION],
    total_iterations: 45,
    total_tokens: 450000,
    total_cost: 2.50,
    was_interrupted: true,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: '2026-02-22T10:45:00.000Z',
};
exports.FAILED_EXECUTION_RESULT = {
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    status: 'failed',
    final_solution: undefined,
    final_score: 0.0,
    best_solutions: [],
    total_iterations: 23,
    total_tokens: 150000,
    total_cost: 0.80,
    was_interrupted: false,
    start_time: '2026-02-22T10:30:00.000Z',
    end_time: '2026-02-22T10:45:00.000Z',
    error: 'LLM API rate limit exceeded',
};
// =============================================================================
// DATABASE STATUS FIXTURES
// =============================================================================
exports.DATABASE_STATUS = {
    global_status: {
        current_iteration: 50,
        best_score: 0.92,
        total_solutions: 500,
    },
    island_status: {
        0: {
            best_score: 0.95,
            total_solutions: 125,
        },
        1: {
            best_score: 0.89,
            total_solutions: 130,
        },
        2: {
            best_score: 0.91,
            total_solutions: 125,
        },
        3: {
            best_score: 0.88,
            total_solutions: 120,
        },
    },
};
exports.SINGLE_ISLAND_DATABASE_STATUS = {
    global_status: {
        current_iteration: 25,
        best_score: 0.87,
        total_solutions: 250,
    },
};
// =============================================================================
// BEST SOLUTIONS FIXTURES
// =============================================================================
exports.BEST_SOLUTIONS_RESPONSE = [
    {
        ...exports.VALID_SOLUTION,
        solution_id: '123e4567-e89b-12d3-a456-426614174000',
        score: 0.95,
        island_id: 0,
        iteration: 50,
    },
    {
        ...exports.VALID_SOLUTION,
        solution_id: '223e4567-e89b-12d3-a456-426614174001',
        score: 0.92,
        island_id: 1,
        iteration: 48,
    },
    {
        ...exports.VALID_SOLUTION,
        solution_id: '323e4567-e89b-12d3-a456-426614174002',
        score: 0.91,
        island_id: 2,
        iteration: 52,
    },
    {
        ...exports.VALID_SOLUTION,
        solution_id: '423e4567-e89b-12d3-a456-426614174003',
        score: 0.89,
        island_id: 3,
        iteration: 49,
    },
    {
        ...exports.VALID_SOLUTION,
        solution_id: '523e4567-e89b-12d3-a456-426614174004',
        score: 0.88,
        island_id: 0,
        iteration: 47,
    },
];
exports.SINGLE_ISLAND_BEST_SOLUTIONS = [
    {
        ...exports.VALID_SOLUTION,
        solution_id: '123e4567-e89b-12d3-a456-426614174000',
        score: 0.95,
        island_id: 0,
        iteration: 50,
    },
    {
        ...exports.VALID_SOLUTION,
        solution_id: '523e4567-e89b-12d3-a456-426614174004',
        score: 0.88,
        island_id: 0,
        iteration: 47,
    },
];
exports.TOP_K_BEST_SOLUTIONS = exports.BEST_SOLUTIONS_RESPONSE.slice(0, 3);
// =============================================================================
// CHECKPOINT FIXTURES
// =============================================================================
exports.CHECKPOINT_INFO = {
    checkpoint_path: '/data/checkpoints/loongflow/checkpoint_20260222_103000',
    tag: 'iteration-50',
    created_at: '2026-02-22T10:30:00.000Z',
    iteration: 50,
    completion_count: 50,
};
exports.CHECKPOINT_SAVE_RESPONSE = {
    message: 'Checkpoint saved successfully',
    checkpoint: exports.CHECKPOINT_INFO,
};
exports.CHECKPOINT_LOAD_RESPONSE = {
    message: 'Checkpoint loaded successfully',
    agent_id: '123e4567-e89b-12d3-a456-426614174000',
    restored_iteration: 50,
};
exports.CHECKPOINTS_LIST = [
    exports.CHECKPOINT_INFO,
    {
        checkpoint_path: '/data/checkpoints/loongflow/checkpoint_20260222_104500',
        tag: 'iteration-75',
        created_at: '2026-02-22T10:45:00.000Z',
        iteration: 75,
        completion_count: 75,
    },
    {
        checkpoint_path: '/data/checkpoints/loongflow/checkpoint_20260222_110000',
        tag: 'iteration-100',
        created_at: '2026-02-22T11:00:00.000Z',
        iteration: 100,
        completion_count: 100,
    },
];
// =============================================================================
// ERROR RESPONSE FIXTURES
// =============================================================================
exports.NOT_FOUND_ERROR = {
    detail: 'Agent not found',
    error_code: 'NOT_FOUND',
    timestamp: '2026-02-22T10:30:00.000Z',
};
exports.VALIDATION_ERROR = {
    detail: 'Invalid request: missing required field "task"',
    error_code: 'VALIDATION_ERROR',
    timestamp: '2026-02-22T10:30:00.000Z',
};
exports.TIMEOUT_ERROR = {
    detail: 'Request timeout exceeded',
    error_code: 'TIMEOUT',
    timestamp: '2026-02-22T10:30:00.000Z',
};
exports.INTERNAL_ERROR = {
    detail: 'Internal server error: LLM API connection failed',
    error_code: 'INTERNAL_ERROR',
    timestamp: '2026-02-22T10:30:00.000Z',
};
// =============================================================================
// SAMPLE SOLUTION FIXTURES
// =============================================================================
exports.SAMPLED_SOLUTION = {
    solution: exports.VALID_SOLUTION,
    boltzmann_probability: 0.23,
    temperature: 1.0,
};
exports.EMPTY_SAMPLE = {
    solution: {},
    message: 'No solutions available in database',
};
// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
/**
 * Generate a random test ID for uniqueness
 */
function generateTestId() {
    return `test-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}
/**
 * Create a valid UUID for testing
 */
function generateTestUUID() {
    return '123e4567-e89b-12d3-a456-426614174000';
}
/**
 * Create a current UTC timestamp for testing
 */
function getCurrentUTCTimestamp() {
    return new Date().toISOString();
}
/**
 * Validate if a string is a valid UUID
 */
function isValidUUID(uuid) {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    return uuidRegex.test(uuid);
}
/**
 * Validate if a string is a valid UTC ISO-8601 timestamp
 */
function isValidUTCTimestamp(timestamp) {
    try {
        const date = new Date(timestamp);
        return date.toISOString() === timestamp;
    }
    catch {
        return false;
    }
}
//# sourceMappingURL=test-data.js.map