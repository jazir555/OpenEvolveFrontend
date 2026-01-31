/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MathSolver Module - Mathematical Theorem Proving Integration
 * 
 * Provides integration with Z3 SMT solver and Lean theorem prover
 * for automated mathematical reasoning within Iterative Studio.
 * 
 * API Version: 1.1.0 (matches backend)
 */

// ============================================================================
// Core Exports
// ============================================================================

export {
    MathSolverCore,
    MathSolverAPI,
    mathSolverAPI,
    formatProofForDisplay,
    detectDomain,
    recommendSolver
} from './MathSolverCore';

// ============================================================================
// Type Exports
// ============================================================================

export type {
    // Solver types
    SolverSystem,
    ProofStatus,
    ConsensusLevel,
    
    // Problem types
    MathProblem,
    MathSolverState,
    MathSolverMessage,
    KnowledgeEntry,
    SolveOptions,
    SolveResult,
    
    // Z3 API Types
    Z3SolveRequest,
    Z3SolveResponse,
    
    // Lean API Types
    ProveLeanRequest,
    ProveLeanResponse,
    
    // Unified API Types
    SolveUnifiedRequest,
    SolveUnifiedResponse,
    
    // Knowledge API Types
    LearnRequest,
    LearnResponse,
    SearchKnowledgeRequest,
    SearchKnowledgeResponse,
    StrategyRequest,
    StrategyResponse,
    KnowledgeStats,
    
    // Health API Types
    HealthResponse
} from './MathSolverCore';

// ============================================================================
// Prompt Exports
// ============================================================================

export {
    MATH_SOLVER_SYSTEM_PROMPT,
    Z3_FORMALIZATION_PROMPT,
    LEAN_FORMALIZATION_PROMPT,
    PROOF_EXPLANATION_PROMPT,
    PROOF_VERIFICATION_PROMPT,
    MATH_PROBLEM_ANALYSIS_PROMPT,
    CONSTRAINT_EXTRACTION_PROMPT,
    RESULT_INTERPRETATION_PROMPT,
    MATH_ITERATIVE_REFINEMENT_PROMPT,
    MATH_TOOL_DESCRIPTIONS,
    DEFAULT_MATH_SOLVER_CONFIG,
    ERROR_INTERPRETATION_PROMPTS
} from './MathSolverPrompts';

export type { MathSolverConfig } from './MathSolverPrompts';

// ============================================================================
// UI Component
// ============================================================================

export { MathSolverUI } from './MathSolverUI';

// ============================================================================
// Tool Exports
// ============================================================================

export { 
    executeMathToolCall, 
    MATH_TOOLS_PROMPT, 
    isMathTool,
    type MathToolCall,
    type ExtendedToolCall 
} from './MathTools';

// ============================================================================
// Agentic Integration
// ============================================================================

export {
    getExtendedSystemPrompt,
    executeExtendedToolCall,
    parseExtendedResponse,
    isMathTool as isExtendedMathTool,
    MathEnabledConversationManager,
    checkMathSolverIntegration
} from './AgenticIntegration';

export type { ExtendedToolCall as AgenticExtendedToolCall } from './AgenticIntegration';

// ============================================================================
// Mode Integration
// ============================================================================

export {
    initializeMathSolverMode,
    startMathSolverProcess,
    stopMathSolverProcess,
    getActiveMathSolverState,
    setActiveMathSolverState,
    isMathSolverRunning,
    getMathSolverSystemPrompt,
    rehydrateMathSolverUI,
    activeMathSolverCore
} from './MathSolverMode';

// ============================================================================
// Version Info
// ============================================================================

export const MATH_SOLVER_VERSION = '1.1.0';
export const MATH_SOLVER_NAME = 'MathSolver';
export const MATH_SOLVER_API_VERSION = '1.1.0';
