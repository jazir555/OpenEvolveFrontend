/**
 * LoongFlow Contract Test Fixtures
 *
 * This file provides test data for contract validation.
 * All fixtures represent realistic API responses from LoongFlow core.
 *
 * Purpose: Phase 2 - The Contract (Defense)
 * Law of Runtime Truth: These fixtures MUST match actual LoongFlow API responses
 */
import { LoongFlowSolution, LoongFlowState } from '../../../../schemas/loongflow-canonical';
export declare const VALID_HEALTH_RESPONSE: {
    status: string;
    version: string;
    timestamp: string;
};
export declare const UNHEALTHY_RESPONSE: {
    status: string;
    version: string;
    timestamp: string;
    error: string;
};
export declare const VALID_PROBLEM_REQUEST: {
    task: string;
    max_iterations: number;
    target_score: number;
    concurrency: number;
    initial_code: string;
    metadata: {
        domain: string;
        difficulty: string;
    };
};
export declare const MINIMAL_PROBLEM_REQUEST: {
    task: string;
};
export declare const INVALID_PROBLEM_REQUEST: {
    max_iterations: number;
    target_score: number;
};
export declare const PROBLEM_SUBMISSION_RESPONSE: {
    agent_id: string;
    status: string;
    message: string;
    timestamp: string;
};
export declare const VALID_SOLUTION: LoongFlowSolution;
export declare const SOLUTION_WITH_NULL_PARENT: LoongFlowSolution;
export declare const SOLUTION_WITHOUT_OPTIONAL_FIELDS: LoongFlowSolution;
export declare const INVALID_SOLUTION_SCORE: {
    score: number;
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    evaluation: string;
    iteration: number;
    island_id: number;
    solution_id: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
};
export declare const INVALID_SOLUTION_ITERATION: {
    iteration: number;
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    score: number;
    evaluation: string;
    island_id: number;
    solution_id: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
};
export declare const INVALID_SOLUTION_ISLAND_ID: {
    island_id: number;
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    score: number;
    evaluation: string;
    iteration: number;
    solution_id: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
};
export declare const INVALID_SOLUTION_MISSING_FIELD: {
    solution: string;
    solution_id: string;
};
export declare const PENDING_AGENT_STATE: {
    agent_id: string;
    status: LoongFlowState;
    current_iteration: number;
    max_iterations: number;
    target_score: number;
    best_score: number;
    start_time: string;
    end_time: undefined;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
};
export declare const RUNNING_AGENT_STATE: {
    agent_id: string;
    status: LoongFlowState;
    current_iteration: number;
    max_iterations: number;
    target_score: number;
    best_score: number;
    start_time: string;
    end_time: undefined;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
};
export declare const COMPLETED_AGENT_STATE: {
    agent_id: string;
    status: LoongFlowState;
    current_iteration: number;
    max_iterations: number;
    target_score: number;
    best_score: number;
    start_time: string;
    end_time: string;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
};
export declare const FAILED_AGENT_STATE: {
    agent_id: string;
    status: LoongFlowState;
    current_iteration: number;
    max_iterations: number;
    target_score: number;
    best_score: number;
    start_time: string;
    end_time: string;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
};
export declare const SUCCESSFUL_EXECUTION_RESULT: {
    agent_id: string;
    status: string;
    final_solution: string;
    final_score: number;
    best_solutions: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }[];
    total_iterations: number;
    total_tokens: number;
    total_cost: number;
    was_interrupted: boolean;
    start_time: string;
    end_time: string;
};
export declare const INTERRUPTED_EXECUTION_RESULT: {
    agent_id: string;
    status: string;
    final_solution: undefined;
    final_score: number;
    best_solutions: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }[];
    total_iterations: number;
    total_tokens: number;
    total_cost: number;
    was_interrupted: boolean;
    start_time: string;
    end_time: string;
};
export declare const FAILED_EXECUTION_RESULT: {
    agent_id: string;
    status: string;
    final_solution: undefined;
    final_score: number;
    best_solutions: never[];
    total_iterations: number;
    total_tokens: number;
    total_cost: number;
    was_interrupted: boolean;
    start_time: string;
    end_time: string;
    error: string;
};
export declare const DATABASE_STATUS: {
    global_status: {
        current_iteration: number;
        best_score: number;
        total_solutions: number;
    };
    island_status: {
        0: {
            best_score: number;
            total_solutions: number;
        };
        1: {
            best_score: number;
            total_solutions: number;
        };
        2: {
            best_score: number;
            total_solutions: number;
        };
        3: {
            best_score: number;
            total_solutions: number;
        };
    };
};
export declare const SINGLE_ISLAND_DATABASE_STATUS: {
    global_status: {
        current_iteration: number;
        best_score: number;
        total_solutions: number;
    };
};
export declare const BEST_SOLUTIONS_RESPONSE: {
    solution_id: string;
    score: number;
    island_id: number;
    iteration: number;
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    evaluation: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
}[];
export declare const SINGLE_ISLAND_BEST_SOLUTIONS: {
    solution_id: string;
    score: number;
    island_id: number;
    iteration: number;
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    evaluation: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
}[];
export declare const TOP_K_BEST_SOLUTIONS: {
    solution_id: string;
    score: number;
    island_id: number;
    iteration: number;
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    evaluation: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
}[];
export declare const CHECKPOINT_INFO: {
    checkpoint_path: string;
    tag: string;
    created_at: string;
    iteration: number;
    completion_count: number;
};
export declare const CHECKPOINT_SAVE_RESPONSE: {
    message: string;
    checkpoint: {
        checkpoint_path: string;
        tag: string;
        created_at: string;
        iteration: number;
        completion_count: number;
    };
};
export declare const CHECKPOINT_LOAD_RESPONSE: {
    message: string;
    agent_id: string;
    restored_iteration: number;
};
export declare const CHECKPOINTS_LIST: {
    checkpoint_path: string;
    tag: string;
    created_at: string;
    iteration: number;
    completion_count: number;
}[];
export declare const NOT_FOUND_ERROR: {
    detail: string;
    error_code: string;
    timestamp: string;
};
export declare const VALIDATION_ERROR: {
    detail: string;
    error_code: string;
    timestamp: string;
};
export declare const TIMEOUT_ERROR: {
    detail: string;
    error_code: string;
    timestamp: string;
};
export declare const INTERNAL_ERROR: {
    detail: string;
    error_code: string;
    timestamp: string;
};
export declare const SAMPLED_SOLUTION: {
    solution: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    };
    boltzmann_probability: number;
    temperature: number;
};
export declare const EMPTY_SAMPLE: {
    solution: {};
    message: string;
};
/**
 * Generate a random test ID for uniqueness
 */
export declare function generateTestId(): string;
/**
 * Create a valid UUID for testing
 */
export declare function generateTestUUID(): string;
/**
 * Create a current UTC timestamp for testing
 */
export declare function getCurrentUTCTimestamp(): string;
/**
 * Validate if a string is a valid UUID
 */
export declare function isValidUUID(uuid: string): boolean;
/**
 * Validate if a string is a valid UTC ISO-8601 timestamp
 */
export declare function isValidUTCTimestamp(timestamp: string): boolean;
//# sourceMappingURL=test-data.d.ts.map