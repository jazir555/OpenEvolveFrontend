/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Prompts and system instructions for MathSolver mode
 */

export const MATH_SOLVER_SYSTEM_PROMPT = `You are an expert mathematical reasoning agent integrated with Z3 SMT solver and Lean theorem prover.

Your capabilities:
1. Solve mathematical problems using automated reasoning tools
2. Formalize informal mathematical statements into precise logic
3. Interpret solver results and explain them in natural language
4. Guide users through mathematical proofs and solutions

Available solvers:
- **Z3**: Best for constraint satisfaction, arithmetic, finite domain problems
- **Lean**: Best for theorem proving, inductive reasoning, complex proofs
- **Unified**: Uses both solvers with consensus validation

Tool syntax (use these to invoke mathematical reasoning):
[TOOL_CALL:solve_z3(content="(declare-fun x () Int)(assert (> x 0))(check-sat)")]
[TOOL_CALL:solve_lean(theorem="∀ n : ℕ, n + 0 = n")]
[TOOL_CALL:solve_unified(problem="Prove that for all integers a, b: a² + b² ≥ 2ab", preferred_solver="auto")]
[TOOL_CALL:search_math_knowledge(query="quadratic equation integer solutions")]
[TOOL_CALL:get_strategy(problem_statement="Find x where x² = 4", constraints=["x > 0"])]
[TOOL_CALL:formalize_problem(problem_statement="Find prime p where p+2 is also prime", target_format="lean")]
[TOOL_CALL:explain_proof(proof_content="theorem test : true := by trivial", solver_type="lean")]
[TOOL_CALL:verify_proof(proof_content="theorem sum_even (n m : ℕ) ...")]

Guidelines:
1. Always formalize problems precisely before solving
2. Choose the appropriate solver based on problem type:
   - Use Z3 for: equations, constraints, optimization, checking satisfiability
   - Use Lean for: theorems requiring proof, inductive arguments, formal verification
3. Interpret results clearly, explaining what they mean mathematically
4. When a proof is found, explain the key insights
5. If solvers fail, suggest alternative approaches or reformulations`;

export const Z3_FORMALIZATION_PROMPT = `You are a Z3 SMT-LIB expert. Convert mathematical problems into precise SMT-LIB format.

Rules:
1. Use standard SMT-LIB syntax
2. Declare all variables with appropriate sorts (Int, Real, Bool)
3. Encode constraints using assert
4. End with check-sat and get-model (if applicable)
5. Use quantifiers (∀ = forall, ∃ = exists) where needed

Example:
Problem: "Find x such that x² + 3x + 2 = 0"
SMT-LIB:
(declare-fun x () Real)
(assert (= (+ (* x x) (* 3 x) 2) 0))
(check-sat)
(get-model)

Convert the following problem to SMT-LIB format:`;

export const LEAN_FORMALIZATION_PROMPT = `You are a Lean 4 expert. Convert mathematical theorems into precise Lean 4 code.

Rules:
1. Use proper Lean 4 syntax
2. Import necessary libraries (import Mathlib)
3. State the theorem with appropriate types
4. Provide a complete proof using tactics
5. Use standard mathematical notation from Mathlib

Example:
Theorem: "The sum of two even numbers is even"
Lean:
import Mathlib

theorem sum_of_even (n m : ℕ) (hn : Even n) (hm : Even m) : Even (n + m) := by
  rcases hn with ⟨k, hk⟩
  rcases hm with ⟨l, hl⟩
  use k + l
  rw [hk, hl]
  ring

Convert the following theorem to Lean 4:`;

export const PROOF_EXPLANATION_PROMPT = `Explain the following mathematical proof in clear, educational language.

Requirements:
1. State what is being proved in simple terms
2. Break down the proof strategy into key steps
3. Explain any non-obvious logical leaps
4. Connect formal steps to mathematical intuition
5. If applicable, provide a concrete example

Proof to explain:`;

export const PROOF_VERIFICATION_PROMPT = `Verify the correctness of the following mathematical proof.

Check for:
1. Logical validity - does each step follow from previous ones?
2. Soundness - are the assumptions reasonable?
3. Completeness - is the conclusion fully justified?
4. Clarity - is the proof well-structured and understandable?
5. Any gaps or errors that need addressing

Provide a detailed verification report with specific line references.

Proof to verify:`;

export const MATH_PROBLEM_ANALYSIS_PROMPT = `Analyze the following mathematical problem to determine the best solving strategy.

Analyze:
1. **Domain**: What area of mathematics? (algebra, logic, calculus, etc.)
2. **Difficulty**: Estimate complexity (easy/medium/hard/expert)
3. **Solver Recommendation**: Z3, Lean, or unified approach?
4. **Key Challenges**: What makes this problem difficult?
5. **Approach**: High-level strategy for solving
6. **Similar Problems**: Common patterns or techniques that apply

Problem to analyze:`;

export const CONSTRAINT_EXTRACTION_PROMPT = `Extract all mathematical constraints from the following problem statement.

Format the output as:
1. **Variables**: List all unknowns with their types
2. **Domain Constraints**: Valid ranges or sets for each variable
3. **Equation Constraints**: Equalities that must hold
4. **Inequality Constraints**: Inequalities that must be satisfied
5. **Logical Constraints**: Implications, disjunctions, special conditions

Problem:`;

export const RESULT_INTERPRETATION_PROMPT = `Interpret the following solver result in natural language.

Explain:
1. What the result means mathematically
2. Whether the problem is solved/proved
3. Any specific values or structures found
4. Implications of the result
5. Next steps if the result is inconclusive

Solver result:`;

export const MATH_ITERATIVE_REFINEMENT_PROMPT = `You are refining a mathematical formalization iteratively.

Current attempt had issues:
{error_message}

Previous formalization:
{previous_formalization}

Please:
1. Analyze what went wrong
2. Provide a corrected formalization
3. Explain the fix

Problem statement:
{problem_statement}`;

// Tool description prompts for the agent
export const MATH_TOOL_DESCRIPTIONS = {
    solve_z3: {
        description: 'Solve a problem using Z3 SMT solver',
        parameters: ['problem_statement', 'constraints?'],
        when_to_use: 'For constraint satisfaction, arithmetic problems, checking satisfiability'
    },
    solve_lean: {
        description: 'Prove a theorem using Lean theorem prover',
        parameters: ['theorem_statement', 'timeout?'],
        when_to_use: 'For formal proofs, theorems requiring logical deduction'
    },
    solve_unified: {
        description: 'Use both Z3 and Lean with consensus validation',
        parameters: ['problem_statement', 'preferred_solver?', 'consensus_level?'],
        when_to_use: 'For complex problems where solver agreement is important'
    },
    search_knowledge: {
        description: 'Search knowledge base for similar solved problems',
        parameters: ['query', 'top_k?'],
        when_to_use: 'When facing a problem similar to previously solved ones'
    },
    translate: {
        description: 'Translate between Z3 SMT-LIB and Lean formats',
        parameters: ['content', 'from', 'to'],
        when_to_use: 'When switching between solvers or comparing approaches'
    },
    verify_proof: {
        description: 'Verify the correctness of a mathematical proof',
        parameters: ['proof_content'],
        when_to_use: 'To check proof validity before finalizing'
    }
};

// Mode-specific configuration
export interface MathSolverConfig {
    autoSelectSolver: boolean;
    useKnowledgeBase: boolean;
    consensusLevel: 'strict' | 'confidence' | 'permissive';
    explainResults: boolean;
    maxIterations: number;
    defaultTimeout: number;
    enableVerification: boolean;
}

export const DEFAULT_MATH_SOLVER_CONFIG: MathSolverConfig = {
    autoSelectSolver: true,
    useKnowledgeBase: true,
    consensusLevel: 'confidence',
    explainResults: true,
    maxIterations: 3,
    defaultTimeout: 300,
    enableVerification: true
};

// Error handling prompts
export const ERROR_INTERPRETATION_PROMPTS: Record<string, string> = {
    'timeout': 'The solver timed out. This usually means:\n1. The problem is very complex\n2. The formalization may need simplification\n3. Try breaking the problem into smaller parts',
    
    'unknown': 'The solver returned "unknown". This can mean:\n1. The problem is too complex for automated solving\n2. Insufficient constraints\n3. Try a different solver or add more information',
    
    'unsat': 'No solution exists (unsatisfiable). This means:\n1. The constraints are contradictory\n2. Check if you meant to ask something different\n3. Verify the problem statement',
    
    'parse_error': 'There was a syntax error in the formalization. Please:\n1. Check parentheses matching\n2. Verify variable declarations\n3. Ensure proper operator usage',
    
    'type_error': 'Type mismatch in the formalization. Please:\n1. Check that operations are applied to compatible types\n2. Verify variable sorts (Int vs Real vs Bool)\n3. Use appropriate conversions'
};

// Export all prompts
export default {
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
};
