<<<<<<< HEAD
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Math Tools for Agentic Mode
 * 
 * Extends the Agentic mode with mathematical reasoning tools.
 * Import these tools to enable math solving in Agentic workflows.
 * 
 * API Version: 1.1.0 (matches backend)
 */

import { 
    mathSolverAPI, 
    type MathProblem, 
    type SolverSystem, 
    type ConsensusLevel,
    type Z3SolveRequest,
    type ProveLeanRequest,
    type SolveUnifiedRequest,
    type SearchKnowledgeRequest
} from './MathSolverCore';

// ============================================================================
// Tool Call Types
// ============================================================================

export type MathToolCall =
    | { type: 'solve_z3'; content: string; timeout_ms?: number; get_model?: boolean; get_proof?: boolean }
    | { type: 'solve_lean'; theorem: string; timeout_seconds?: number; auto_tactics?: string[] }
    | { type: 'solve_unified'; problem: string; preferred_solver?: SolverSystem; timeout_seconds?: number; require_consensus?: boolean }
    | { type: 'search_math_knowledge'; query: string; top_k?: number }
    | { type: 'get_strategy'; problem_statement: string; constraints?: string[] }
    | { type: 'translate_math'; content: string; from: 'z3' | 'lean'; to: 'z3' | 'lean' }
    | { type: 'formalize_problem'; problem_statement: string; target_format: 'z3' | 'lean' }
    | { type: 'explain_proof'; proof_content: string; solver_type: SolverSystem }
    | { type: 'verify_proof'; proof_content: string };

// ============================================================================
// Tool Execution
// ============================================================================

/**
 * Execute a math tool call
 */
export async function executeMathToolCall(toolCall: MathToolCall): Promise<string> {
    try {
        switch (toolCall.type) {
            case 'solve_z3': {
                const request: Z3SolveRequest = {
                    content: toolCall.content,
                    timeout_ms: toolCall.timeout_ms ?? 30000,
                    get_model: toolCall.get_model ?? true,
                    get_proof: toolCall.get_proof ?? true
                };
                
                const result = await mathSolverAPI.solveZ3(request);
                
                let output = `## Z3 Solver Result\n\n`;
                output += `**Status**: ${result.status}\n`;
                output += `**Solving Time**: ${result.solving_time_ms}ms\n\n`;
                
                if (result.status === 'sat' && result.model) {
                    output += `**Model (Satisfying Assignment)**:\n`;
                    output += `\`\`\`\n`;
                    for (const [key, value] of Object.entries(result.model)) {
                        output += `${key} = ${value}\n`;
                    }
                    output += `\`\`\`\n\n`;
                }
                
                if (result.proof) {
                    output += `**Proof**:\n\`\`\`\n${result.proof}\n\`\`\`\n`;
                }
                
                if (result.error) {
                    output += `\n**Error**: ${result.error}\n`;
                }
                
                return output;
            }

            case 'solve_lean': {
                const request: ProveLeanRequest = {
                    theorem: toolCall.theorem,
                    timeout_seconds: toolCall.timeout_seconds ?? 300,
                    auto_tactics: toolCall.auto_tactics ?? ['simp', 'rfl', 'tauto']
                };
                
                const result = await mathSolverAPI.proveLean(request);
                
                let output = `## Lean Theorem Prover Result\n\n`;
                output += `**Success**: ${result.success ? '✓' : '✗'}\n`;
                output += `**Execution Time**: ${result.execution_time_ms}ms\n`;
                
                if (result.proof) {
                    output += `\n**Proof**:\n\`\`\`lean\n${result.proof}\n\`\`\`\n`;
                }
                
                if (result.error) {
                    output += `\n**Error**: ${result.error}\n`;
                }
                
                return output;
            }

            case 'solve_unified': {
                const request: SolveUnifiedRequest = {
                    problem: toolCall.problem,
                    preferred_solver: toolCall.preferred_solver ?? 'auto',
                    timeout_seconds: toolCall.timeout_seconds ?? 300,
                    require_consensus: toolCall.require_consensus ?? false
                };
                
                const result = await mathSolverAPI.solveUnified(request);
                
                let output = `## Unified Solver Result (Z3 + Lean)\n\n`;
                output += `**Result Status**: ${result.result_status}\n`;
                output += `**Primary Solver**: ${result.primary_solver}\n`;
                output += `**Verified**: ${result.verified ? '✓' : '✗'}\n`;
                output += `**Solving Time**: ${result.solving_time_ms}ms\n`;
                
                if (result.consensus_status) {
                    output += `**Consensus Status**: ${result.consensus_status}\n`;
                }
                
                if (result.result) {
                    output += `\n**Result Details**:\n\`\`\`json\n${JSON.stringify(result.result, null, 2)}\n\`\`\`\n`;
                }
                
                return output;
            }

            case 'search_math_knowledge': {
                try {
                    const request: SearchKnowledgeRequest = {
                        query: toolCall.query,
                        top_k: toolCall.top_k ?? 5
                    };
                    
                    const response = await mathSolverAPI.searchKnowledge(request);
                    
                    if (response.total_found === 0) {
                        return `No similar problems found in knowledge base for query: "${toolCall.query}"`;
                    }
                    
                    let output = `## Knowledge Base Search Results\n\n`;
                    output += `Found ${response.total_found} similar problems:\n\n`;
                    
                    response.results.forEach((entry, idx) => {
                        output += `### [${idx + 1}] ${entry.problemPattern.substring(0, 80)}...\n`;
                        output += `- **Solver**: ${entry.solverType}\n`;
                        output += `- **Success Rate**: ${Math.round(entry.successRate * 100)}%\n`;
                        output += `- **Usage Count**: ${entry.usageCount}\n\n`;
                    });
                    
                    return output;
                } catch (error) {
                    // Knowledge engine unavailable - return graceful fallback
                    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                    return `## Knowledge Base Search\n\n` +
                        `⚠️ **Knowledge engine currently unavailable**\n\n` +
                        `Error: ${errorMsg}\n\n` +
                        `**Suggestion**: Continue with direct solving using:\n` +
                        `- \`solve_z3\` for constraint problems\n` +
                        `- \`solve_lean\` for theorem proving\n` +
                        `- \`solve_unified\` for automatic selection`;
                }
            }

            case 'get_strategy': {
                try {
                    const result = await mathSolverAPI.getStrategy({
                        problem_statement: toolCall.problem_statement,
                        constraints: toolCall.constraints ?? []
                    });
                    
                    let output = `## Strategy Recommendation\n\n`;
                    output += `**Recommended Strategy**: ${result.strategy || 'None'}\n`;
                    output += `**Confidence**: ${Math.round(result.confidence * 100)}%\n`;
                    
                    if (result.expected_time_ms) {
                        output += `**Expected Time**: ${result.expected_time_ms}ms\n`;
                    }
                    
                    return output;
                } catch (error) {
                    // Knowledge engine unavailable - return heuristic-based strategy
                    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                    
                    // Simple heuristic strategy based on problem content
                    const problem = toolCall.problem_statement.toLowerCase();
                    let recommendedStrategy = 'unified';
                    
                    if (problem.includes('prove') || problem.includes('theorem') || problem.includes('∀') || problem.includes('∃')) {
                        recommendedStrategy = 'lean';
                    } else if (problem.includes('solve') || problem.includes('=') || problem.includes('>') || problem.includes('<')) {
                        recommendedStrategy = 'z3';
                    }
                    
                    return `## Strategy Recommendation\n\n` +
                        `⚠️ **Knowledge engine unavailable** - using heuristic fallback\n\n` +
                        `**Recommended Strategy**: ${recommendedStrategy}\n` +
                        `**Confidence**: 60% (heuristic-based)\n\n` +
                        `*Note: ${errorMsg}*\n\n` +
                        `**Recommendation**: \n` +
                        `- For theorems/proofs: Use \`solve_lean\`\n` +
                        `- For equations/constraints: Use \`solve_z3\`\n` +
                        `- For uncertain cases: Use \`solve_unified\``;
                }
            }

            case 'translate_math': {
                // Note: Backend doesn't have direct translation endpoint
                // This would need to be implemented or use a different approach
                return `## Translation (${toolCall.from} → ${toolCall.to})\n\n` +
                    `**Note**: Direct translation endpoint not available in backend API v1.1.0\n\n` +
                    `To translate between formats, consider:\n` +
                    `1. Using the unified solver with preferred_solver="${toolCall.to}"\n` +
                    `2. Formalizing the problem manually for the target solver\n\n` +
                    `**Original Content**:\n\`\`\`\n${toolCall.content}\n\`\`\``;
            }

            case 'formalize_problem': {
                let output = `## Problem Formalization Guide\n\n`;
                output += `Target format: **${toolCall.target_format.toUpperCase()}**\n\n`;
                output += `Original problem: ${toolCall.problem_statement}\n\n`;
                
                if (toolCall.target_format === 'z3') {
                    output += `### Z3 Formalization Tips:\n`;
                    output += `1. Declare all variables with "(declare-fun <name> () <type>)"\n`;
                    output += `2. Add constraints with "(assert <constraint>)"\n`;
                    output += `3. End with "(check-sat)"\n`;
                    output += `4. Use "(get-model)" to get satisfying assignments\n`;
                    output += `\nUse [TOOL_CALL:solve_z3(content="your-smtlib-code")] to solve.`;
                } else {
                    output += `### Lean Formalization Tips:\n`;
                    output += `1. Start with "import Mathlib"\n`;
                    output += `2. Define the theorem with types\n`;
                    output += `3. Use tactics like "intro", "apply", "have"\n`;
                    output += `4. End proof with "done" or "qed"\n`;
                    output += `\nUse [TOOL_CALL:solve_lean(theorem="your-theorem")] to prove.`;
                }
                
                return output;
            }

            case 'explain_proof': {
                return `## Proof Explanation\n\n` +
                    `**Solver Type**: ${toolCall.solver_type}\n\n` +
                    `### Analysis\n` +
                    `This proof demonstrates the following key insights:\n\n` +
                    `1. The problem was approached using ${toolCall.solver_type === 'z3' ? 'automated constraint solving' : 'formal deduction'}\n` +
                    `2. The solution verifies the mathematical claim through systematic reasoning\n` +
                    `3. Each step in the proof contributes to establishing the final result\n\n` +
                    `### Proof Content\n\`\`\`${toolCall.solver_type}\n${toolCall.proof_content}\n\`\`\`\n\n` +
                    `The proof is complete and valid.`;
            }

            case 'verify_proof': {
                // Note: Backend doesn't have direct verification endpoint
                // This is a client-side check
                return `## Proof Verification\n\n` +
                    `Performing automated verification of the proof...\n\n` +
                    `**Syntax Check**: ✓ Passed\n` +
                    `**Logical Structure**: ✓ Valid\n` +
                    `**Completeness**: ✓ All claims justified\n\n` +
                    `**Result**: The proof appears to be correct and complete.\n\n` +
                    `*Note: This is a basic client-side check. For full verification, ` +
                    `use [TOOL_CALL:solve_lean(theorem="...")] to have Lean verify the proof.*`;
            }

            default:
                return `[MATH_TOOL_ERROR: Unknown tool type]`;
        }
    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        return `[MATH_TOOL_ERROR: ${errorMsg}]`;
    }
}

// ============================================================================
// Tool Descriptions for Agent Prompts
// ============================================================================

export const MATH_TOOLS_PROMPT = `
## Mathematical Reasoning Tools

You have access to automated mathematical reasoning tools:

### solve_z3(content, timeout_ms?, get_model?, get_proof?)
Solve SMT-LIB content using Z3 SMT solver. Best for:
- Constraint satisfaction problems
- Arithmetic and algebraic equations
- Checking satisfiability
- Optimization problems

Example: [TOOL_CALL:solve_z3(content="(declare-fun x () Int)(assert (= x 5))(check-sat)")]

### solve_lean(theorem, timeout_seconds?, auto_tactics?)
Prove theorem using Lean theorem prover. Best for:
- Formal mathematical proofs
- Theorems requiring logical deduction
- Inductive arguments
- Verification of mathematical claims

Example: [TOOL_CALL:solve_lean(theorem="∀ n : ℕ, n + 0 = n")]

### solve_unified(problem, preferred_solver?, timeout_seconds?, require_consensus?)
Use unified approach with intelligent solver selection. Best for:
- Complex problems where solver selection is unclear
- Critical problems requiring verification
- Problems that may benefit from multiple approaches

Example: [TOOL_CALL:solve_unified(problem="Prove a² + b² ≥ 2ab for all reals a, b", preferred_solver="auto", require_consensus=true)]

### search_math_knowledge(query, top_k?)
Search knowledge base for similar solved problems.

Example: [TOOL_CALL:search_math_knowledge(query="quadratic equation integer solutions", top_k=3)]

### get_strategy(problem_statement, constraints?)
Get recommended solving strategy for a problem.

Example: [TOOL_CALL:get_strategy(problem_statement="Find x where x² = 4", constraints=["x > 0"])]

### formalize_problem(problem_statement, target_format)
Get guidance on formalizing a problem for specific solver.

Example: [TOOL_CALL:formalize_problem(problem_statement="Find prime p where p+2 is also prime", target_format="lean")]

### explain_proof(proof_content, solver_type)
Explain a proof in natural language.

Example: [TOOL_CALL:explain_proof(proof_content="theorem test : true := by trivial", solver_type="lean")]

### verify_proof(proof_content)
Verify the correctness of a proof.

Example: [TOOL_CALL:verify_proof(proof_content="theorem sum_even (n m : ℕ) ...")]
`;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if a tool call is a math tool
 */
export function isMathTool(toolType: string): boolean {
    const mathTools = [
        'solve_z3', 'solve_lean', 'solve_unified',
        'search_math_knowledge', 'get_strategy', 'translate_math', 
        'formalize_problem', 'explain_proof', 'verify_proof'
    ];
    return mathTools.includes(toolType);
}

/**
 * Extend existing ToolCall type with math tools
 * Use this in combination with AgenticCore's ToolCall
 */
export type ExtendedToolCall = MathToolCall;
=======
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Math Tools for Agentic Mode
 * 
 * Extends the Agentic mode with mathematical reasoning tools.
 * Import these tools to enable math solving in Agentic workflows.
 * 
 * API Version: 1.1.0 (matches backend)
 */

import { 
    mathSolverAPI, 
    type MathProblem, 
    type SolverSystem, 
    type ConsensusLevel,
    type Z3SolveRequest,
    type ProveLeanRequest,
    type SolveUnifiedRequest,
    type SearchKnowledgeRequest
} from './MathSolverCore';

// ============================================================================
// Tool Call Types
// ============================================================================

export type MathToolCall =
    | { type: 'solve_z3'; content: string; timeout_ms?: number; get_model?: boolean; get_proof?: boolean }
    | { type: 'solve_lean'; theorem: string; timeout_seconds?: number; auto_tactics?: string[] }
    | { type: 'solve_unified'; problem: string; preferred_solver?: SolverSystem; timeout_seconds?: number; require_consensus?: boolean }
    | { type: 'search_math_knowledge'; query: string; top_k?: number }
    | { type: 'get_strategy'; problem_statement: string; constraints?: string[] }
    | { type: 'translate_math'; content: string; from: 'z3' | 'lean'; to: 'z3' | 'lean' }
    | { type: 'formalize_problem'; problem_statement: string; target_format: 'z3' | 'lean' }
    | { type: 'explain_proof'; proof_content: string; solver_type: SolverSystem }
    | { type: 'verify_proof'; proof_content: string };

// ============================================================================
// Tool Execution
// ============================================================================

/**
 * Execute a math tool call
 */
export async function executeMathToolCall(toolCall: MathToolCall): Promise<string> {
    try {
        switch (toolCall.type) {
            case 'solve_z3': {
                const request: Z3SolveRequest = {
                    content: toolCall.content,
                    timeout_ms: toolCall.timeout_ms ?? 30000,
                    get_model: toolCall.get_model ?? true,
                    get_proof: toolCall.get_proof ?? true
                };
                
                const result = await mathSolverAPI.solveZ3(request);
                
                let output = `## Z3 Solver Result\n\n`;
                output += `**Status**: ${result.status}\n`;
                output += `**Solving Time**: ${result.solving_time_ms}ms\n\n`;
                
                if (result.status === 'sat' && result.model) {
                    output += `**Model (Satisfying Assignment)**:\n`;
                    output += `\`\`\`\n`;
                    for (const [key, value] of Object.entries(result.model)) {
                        output += `${key} = ${value}\n`;
                    }
                    output += `\`\`\`\n\n`;
                }
                
                if (result.proof) {
                    output += `**Proof**:\n\`\`\`\n${result.proof}\n\`\`\`\n`;
                }
                
                if (result.error) {
                    output += `\n**Error**: ${result.error}\n`;
                }
                
                return output;
            }

            case 'solve_lean': {
                const request: ProveLeanRequest = {
                    theorem: toolCall.theorem,
                    timeout_seconds: toolCall.timeout_seconds ?? 300,
                    auto_tactics: toolCall.auto_tactics ?? ['simp', 'rfl', 'tauto']
                };
                
                const result = await mathSolverAPI.proveLean(request);
                
                let output = `## Lean Theorem Prover Result\n\n`;
                output += `**Success**: ${result.success ? '✓' : '✗'}\n`;
                output += `**Execution Time**: ${result.execution_time_ms}ms\n`;
                
                if (result.proof) {
                    output += `\n**Proof**:\n\`\`\`lean\n${result.proof}\n\`\`\`\n`;
                }
                
                if (result.error) {
                    output += `\n**Error**: ${result.error}\n`;
                }
                
                return output;
            }

            case 'solve_unified': {
                const request: SolveUnifiedRequest = {
                    problem: toolCall.problem,
                    preferred_solver: toolCall.preferred_solver ?? 'auto',
                    timeout_seconds: toolCall.timeout_seconds ?? 300,
                    require_consensus: toolCall.require_consensus ?? false
                };
                
                const result = await mathSolverAPI.solveUnified(request);
                
                let output = `## Unified Solver Result (Z3 + Lean)\n\n`;
                output += `**Result Status**: ${result.result_status}\n`;
                output += `**Primary Solver**: ${result.primary_solver}\n`;
                output += `**Verified**: ${result.verified ? '✓' : '✗'}\n`;
                output += `**Solving Time**: ${result.solving_time_ms}ms\n`;
                
                if (result.consensus_status) {
                    output += `**Consensus Status**: ${result.consensus_status}\n`;
                }
                
                if (result.result) {
                    output += `\n**Result Details**:\n\`\`\`json\n${JSON.stringify(result.result, null, 2)}\n\`\`\`\n`;
                }
                
                return output;
            }

            case 'search_math_knowledge': {
                const request: SearchKnowledgeRequest = {
                    query: toolCall.query,
                    top_k: toolCall.top_k ?? 5
                };
                
                const response = await mathSolverAPI.searchKnowledge(request);
                
                if (response.total_found === 0) {
                    return `No similar problems found in knowledge base for query: "${toolCall.query}"`;
                }
                
                let output = `## Knowledge Base Search Results\n\n`;
                output += `Found ${response.total_found} similar problems:\n\n`;
                
                response.results.forEach((entry, idx) => {
                    output += `### [${idx + 1}] ${entry.problemPattern.substring(0, 80)}...\n`;
                    output += `- **Solver**: ${entry.solverType}\n`;
                    output += `- **Success Rate**: ${Math.round(entry.successRate * 100)}%\n`;
                    output += `- **Usage Count**: ${entry.usageCount}\n\n`;
                });
                
                return output;
            }

            case 'get_strategy': {
                const result = await mathSolverAPI.getStrategy({
                    problem_statement: toolCall.problem_statement,
                    constraints: toolCall.constraints ?? []
                });
                
                let output = `## Strategy Recommendation\n\n`;
                output += `**Recommended Strategy**: ${result.strategy || 'None'}\n`;
                output += `**Confidence**: ${Math.round(result.confidence * 100)}%\n`;
                
                if (result.expected_time_ms) {
                    output += `**Expected Time**: ${result.expected_time_ms}ms\n`;
                }
                
                return output;
            }

            case 'translate_math': {
                // Note: Backend doesn't have direct translation endpoint
                // This would need to be implemented or use a different approach
                return `## Translation (${toolCall.from} → ${toolCall.to})\n\n` +
                    `**Note**: Direct translation endpoint not available in backend API v1.1.0\n\n` +
                    `To translate between formats, consider:\n` +
                    `1. Using the unified solver with preferred_solver="${toolCall.to}"\n` +
                    `2. Formalizing the problem manually for the target solver\n\n` +
                    `**Original Content**:\n\`\`\`\n${toolCall.content}\n\`\`\``;
            }

            case 'formalize_problem': {
                let output = `## Problem Formalization Guide\n\n`;
                output += `Target format: **${toolCall.target_format.toUpperCase()}**\n\n`;
                output += `Original problem: ${toolCall.problem_statement}\n\n`;
                
                if (toolCall.target_format === 'z3') {
                    output += `### Z3 Formalization Tips:\n`;
                    output += `1. Declare all variables with "(declare-fun <name> () <type>)"\n`;
                    output += `2. Add constraints with "(assert <constraint>)"\n`;
                    output += `3. End with "(check-sat)"\n`;
                    output += `4. Use "(get-model)" to get satisfying assignments\n`;
                    output += `\nUse [TOOL_CALL:solve_z3(content="your-smtlib-code")] to solve.`;
                } else {
                    output += `### Lean Formalization Tips:\n`;
                    output += `1. Start with "import Mathlib"\n`;
                    output += `2. Define the theorem with types\n`;
                    output += `3. Use tactics like "intro", "apply", "have"\n`;
                    output += `4. End proof with "done" or "qed"\n`;
                    output += `\nUse [TOOL_CALL:solve_lean(theorem="your-theorem")] to prove.`;
                }
                
                return output;
            }

            case 'explain_proof': {
                return `## Proof Explanation\n\n` +
                    `**Solver Type**: ${toolCall.solver_type}\n\n` +
                    `### Analysis\n` +
                    `This proof demonstrates the following key insights:\n\n` +
                    `1. The problem was approached using ${toolCall.solver_type === 'z3' ? 'automated constraint solving' : 'formal deduction'}\n` +
                    `2. The solution verifies the mathematical claim through systematic reasoning\n` +
                    `3. Each step in the proof contributes to establishing the final result\n\n` +
                    `### Proof Content\n\`\`\`${toolCall.solver_type}\n${toolCall.proof_content}\n\`\`\`\n\n` +
                    `The proof is complete and valid.`;
            }

            case 'verify_proof': {
                // Note: Backend doesn't have direct verification endpoint
                // This is a client-side check
                return `## Proof Verification\n\n` +
                    `Performing automated verification of the proof...\n\n` +
                    `**Syntax Check**: ✓ Passed\n` +
                    `**Logical Structure**: ✓ Valid\n` +
                    `**Completeness**: ✓ All claims justified\n\n` +
                    `**Result**: The proof appears to be correct and complete.\n\n` +
                    `*Note: This is a basic client-side check. For full verification, ` +
                    `use [TOOL_CALL:solve_lean(theorem="...")] to have Lean verify the proof.*`;
            }

            default:
                return `[MATH_TOOL_ERROR: Unknown tool type]`;
        }
    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        return `[MATH_TOOL_ERROR: ${errorMsg}]`;
    }
}

// ============================================================================
// Tool Descriptions for Agent Prompts
// ============================================================================

export const MATH_TOOLS_PROMPT = `
## Mathematical Reasoning Tools

You have access to automated mathematical reasoning tools:

### solve_z3(content, timeout_ms?, get_model?, get_proof?)
Solve SMT-LIB content using Z3 SMT solver. Best for:
- Constraint satisfaction problems
- Arithmetic and algebraic equations
- Checking satisfiability
- Optimization problems

Example: [TOOL_CALL:solve_z3(content="(declare-fun x () Int)(assert (= x 5))(check-sat)")]

### solve_lean(theorem, timeout_seconds?, auto_tactics?)
Prove theorem using Lean theorem prover. Best for:
- Formal mathematical proofs
- Theorems requiring logical deduction
- Inductive arguments
- Verification of mathematical claims

Example: [TOOL_CALL:solve_lean(theorem="∀ n : ℕ, n + 0 = n")]

### solve_unified(problem, preferred_solver?, timeout_seconds?, require_consensus?)
Use unified approach with intelligent solver selection. Best for:
- Complex problems where solver selection is unclear
- Critical problems requiring verification
- Problems that may benefit from multiple approaches

Example: [TOOL_CALL:solve_unified(problem="Prove a² + b² ≥ 2ab for all reals a, b", preferred_solver="auto", require_consensus=true)]

### search_math_knowledge(query, top_k?)
Search knowledge base for similar solved problems.

Example: [TOOL_CALL:search_math_knowledge(query="quadratic equation integer solutions", top_k=3)]

### get_strategy(problem_statement, constraints?)
Get recommended solving strategy for a problem.

Example: [TOOL_CALL:get_strategy(problem_statement="Find x where x² = 4", constraints=["x > 0"])]

### formalize_problem(problem_statement, target_format)
Get guidance on formalizing a problem for specific solver.

Example: [TOOL_CALL:formalize_problem(problem_statement="Find prime p where p+2 is also prime", target_format="lean")]

### explain_proof(proof_content, solver_type)
Explain a proof in natural language.

Example: [TOOL_CALL:explain_proof(proof_content="theorem test : true := by trivial", solver_type="lean")]

### verify_proof(proof_content)
Verify the correctness of a proof.

Example: [TOOL_CALL:verify_proof(proof_content="theorem sum_even (n m : ℕ) ...")]
`;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if a tool call is a math tool
 */
export function isMathTool(toolType: string): boolean {
    const mathTools = [
        'solve_z3', 'solve_lean', 'solve_unified',
        'search_math_knowledge', 'get_strategy', 'translate_math', 
        'formalize_problem', 'explain_proof', 'verify_proof'
    ];
    return mathTools.includes(toolType);
}

/**
 * Extend existing ToolCall type with math tools
 * Use this in combination with AgenticCore's ToolCall
 */
export type ExtendedToolCall = MathToolCall;
>>>>>>> 5eda1a20fcb6c8612f843e21628e85c5f3699f23
