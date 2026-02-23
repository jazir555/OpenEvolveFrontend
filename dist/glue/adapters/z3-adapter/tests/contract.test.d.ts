/**
 * Z3 Adapter Contract Tests
 *
 * CRITICAL: These tests validate the contract between the Z3 Adapter and Z3 Core.
 * If these tests fail, the adapter MUST refuse to start to prevent data corruption.
 *
 * Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
 * - Phase 2: The Contract (Defense)
 * - Protecting the Mega-Project from Updates
 *
 * Test Principles:
 * 1. FAIL FAST - Contract violations immediately halt execution
 * 2. MOCK ONLY - Do not require running Z3 instance
 * 3. CANONICAL VALIDATION - Use canonical schemas for data structure validation
 * 4. IDPOTENT - Tests can be run 100 times safely
 */
declare const mockHealthResponse: {
    status: string;
    z3_available: boolean;
    version: string;
};
declare const mockHealthDegradedResponse: {
    status: string;
    z3_available: boolean;
    error: string;
};
declare const mockSolveResponse: {
    result: string;
    model: {
        x: number;
        y: number;
    };
    statistics: {
        'solver.time': number;
        'solver.decisions': number;
    };
    timing: number;
};
declare const mockSolveUnsatResponse: {
    result: string;
    statistics: {
        'solver.time': number;
    };
    timing: number;
};
declare const mockOptimizeResponse: {
    status: string;
    model: {
        x: number;
        y: number;
    };
    objective_values: {
        maximize_x: number;
        minimize_y: number;
    };
    timing: number;
};
declare const mockSimplifyResponse: {
    result: string;
    timing: number;
};
declare const mockTacticResponse: {
    status: string;
    goals: never[];
    model: {
        x: boolean;
    };
    timing: number;
};
declare const mockFixedpointResponse: {
    result: string;
    answer: string;
    timing: number;
};
export { mockHealthResponse, mockHealthDegradedResponse, mockSolveResponse, mockSolveUnsatResponse, mockOptimizeResponse, mockSimplifyResponse, mockTacticResponse, mockFixedpointResponse, };
//# sourceMappingURL=contract.test.d.ts.map