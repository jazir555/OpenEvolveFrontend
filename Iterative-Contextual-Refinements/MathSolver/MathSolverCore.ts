/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MathSolver Core - Mathematical Theorem Proving Integration
 * 
 * Connects Iterative Studio frontend with Z3-LeanAIDE Python backend
 * for automated mathematical reasoning, SMT solving, and theorem proving.
 * 
 * API Version: 1.1.0 (matches backend)
 */

// ============================================================================
// Types and Interfaces - Aligned with Backend API
// ============================================================================

export type SolverSystem = 'z3' | 'lean' | 'unified' | 'auto' | 'hybrid';
export type ProofStatus = 'proved' | 'disproved' | 'unknown' | 'timeout' | 'error';
export type ConsensusLevel = 'strict' | 'confidence' | 'permissive';

export interface MathProblem {
    id: string;
    statement: string;
    constraints?: string[];
    domain?: 'algebra' | 'arithmetic' | 'geometry' | 'calculus' | 'logic' | 'number_theory' | 'other';
    difficulty?: 'easy' | 'medium' | 'hard' | 'expert';
    expectedResult?: string;
}

// Z3 API Types (matching backend)
export interface Z3SolveRequest {
    content: string;           // SMT-LIB content
    timeout_ms?: number;       // Default: 30000
    get_model?: boolean;       // Default: true
    get_proof?: boolean;       // Default: true
}

export interface Z3SolveResponse {
    status: 'sat' | 'unsat' | 'unknown' | 'timeout' | 'error';
    model?: Record<string, any> | null;
    proof?: string | null;
    solving_time_ms: number;
    error?: string | null;
}

// Lean API Types (matching backend)
export interface ProveLeanRequest {
    theorem: string;           // Theorem statement
    timeout_seconds?: number;  // Default: 300
    auto_tactics?: string[];   // Default: ["simp", "rfl", "tauto"]
}

export interface ProveLeanResponse {
    success: boolean;
    proof?: string | null;
    error?: string | null;
    execution_time_ms: number;
}

// Unified API Types (matching backend)
export interface SolveUnifiedRequest {
    problem: string;           // Problem statement
    preferred_solver?: string; // "auto", "z3", "lean", "hybrid"
    timeout_seconds?: number;  // Default: 300
    require_consensus?: boolean; // Default: false
}

export interface SolveUnifiedResponse {
    result_status: string;
    primary_solver: string;
    result?: any;
    verified: boolean;
    consensus_status?: string | null;
    solving_time_ms: number;
}

// Knowledge API Types (matching backend)
export interface LearnRequest {
    problem_statement: string;
    constraints: string[];
    result: string;
    proof?: string | null;
    metadata?: Record<string, any> | null;
}

export interface LearnResponse {
    success: boolean;
    items_learned: number;
    features: Record<string, any>;
}

export interface SearchKnowledgeRequest {
    query: string;
    top_k?: number;           // Default: 5
    pattern_type?: string | null;
}

export interface SearchKnowledgeResponse {
    results: KnowledgeEntry[];
    total_found: number;
}

export interface StrategyRequest {
    problem_statement: string;
    constraints: string[];
}

export interface StrategyResponse {
    strategy?: string | null;
    confidence: number;
    expected_time_ms?: number | null;
}

export interface KnowledgeStats {
    total_patterns?: number;
    total_strategies?: number;
    learning_enabled?: boolean;
    [key: string]: any;
}

// Health check response
export interface HealthResponse {
    status: 'healthy' | 'unhealthy';
    components: {
        z3: boolean;
        leanaide: boolean;
        bridge: boolean;
        knowledge: boolean;
    };
    timestamp: string;
}

// Internal types
export interface KnowledgeEntry {
    id: string;
    problemPattern: string;
    solution: string;
    solverType: SolverSystem;
    successRate: number;
    usageCount: number;
    timestamp: number;
}

export interface MathSolverState {
    id: string;
    currentProblem: MathProblem | null;
    history: MathProblem[];
    z3Results: Map<string, Z3SolveResponse>;
    leanResults: Map<string, ProveLeanResponse>;
    unifiedResults: Map<string, SolveUnifiedResponse>;
    isProcessing: boolean;
    activeSolvers: SolverSystem[];
    messages: MathSolverMessage[];
    knowledgeBase: KnowledgeEntry[];
}

export interface MathSolverMessage {
    id: string;
    role: 'user' | 'agent' | 'system' | 'solver';
    content: string;
    timestamp: number;
    solverType?: SolverSystem;
    proofStatus?: ProofStatus;
    metadata?: Record<string, any>;
}

export interface SolveOptions {
    problem: MathProblem;
    preferredSolver?: SolverSystem;
    useKnowledgeBase?: boolean;
    consensusLevel?: ConsensusLevel;
    timeout?: number;
    requireConsensus?: boolean;
}

export interface SolveResult {
    success: boolean;
    problemId: string;
    z3Result?: Z3SolveResponse;
    leanResult?: ProveLeanResponse;
    unifiedResult?: SolveUnifiedResponse;
    knowledgeUsed?: KnowledgeEntry[];
    executionTimeMs: number;
    error?: string;
}

// ============================================================================
// API Client Configuration
// ============================================================================

const API_BASE_URL = (typeof process !== 'undefined' && process.env?.MATH_SOLVER_API_URL) 
    || 'http://localhost:8000';
const API_TIMEOUT = 300000; // 5 minutes for complex proofs

/**
 * API client for communicating with Z3-LeanAIDE Python backend
 * Aligned with backend API version 1.1.0
 */
export class MathSolverAPI {
    private baseUrl: string;
    private timeout: number;

    constructor(baseUrl: string = API_BASE_URL, timeout: number = API_TIMEOUT) {
        this.baseUrl = baseUrl;
        this.timeout = timeout;
    }

    /**
     * Solve using Z3 SMT solver
     * POST /solve/z3
     */
    async solveZ3(request: Z3SolveRequest): Promise<Z3SolveResponse> {
        const response = await this.post<Z3SolveResponse>('/solve/z3', {
            content: request.content,
            timeout_ms: request.timeout_ms ?? 30000,
            get_model: request.get_model ?? true,
            get_proof: request.get_proof ?? true
        });
        return response;
    }

    /**
     * Prove theorem using Lean
     * POST /solve/lean
     */
    async proveLean(request: ProveLeanRequest): Promise<ProveLeanResponse> {
        const response = await this.post<ProveLeanResponse>('/solve/lean', {
            theorem: request.theorem,
            timeout_seconds: request.timeout_seconds ?? 300,
            auto_tactics: request.auto_tactics ?? ['simp', 'rfl', 'tauto']
        });
        return response;
    }

    /**
     * Solve using unified approach
     * POST /solve/unified
     */
    async solveUnified(request: SolveUnifiedRequest): Promise<SolveUnifiedResponse> {
        const response = await this.post<SolveUnifiedResponse>('/solve/unified', {
            problem: request.problem,
            preferred_solver: request.preferred_solver ?? 'auto',
            timeout_seconds: request.timeout_seconds ?? 300,
            require_consensus: request.require_consensus ?? false
        });
        return response;
    }

    /**
     * Learn from a solution
     * POST /knowledge/learn
     */
    async learnFromSolution(request: LearnRequest): Promise<LearnResponse> {
        const response = await this.post<LearnResponse>('/knowledge/learn', {
            problem_statement: request.problem_statement,
            constraints: request.constraints,
            result: request.result,
            proof: request.proof ?? null,
            metadata: request.metadata ?? null
        });
        return response;
    }

    /**
     * Search knowledge base
     * POST /knowledge/search
     */
    async searchKnowledge(request: SearchKnowledgeRequest): Promise<SearchKnowledgeResponse> {
        const response = await this.post<SearchKnowledgeResponse>('/knowledge/search', {
            query: request.query,
            top_k: request.top_k ?? 5,
            pattern_type: request.pattern_type ?? null
        });
        return response;
    }

    /**
     * Get strategy recommendation
     * GET /knowledge/strategy
     */
    async getStrategy(request: StrategyRequest): Promise<StrategyResponse> {
        // GET request with query params
        const queryParams = new URLSearchParams({
            problem_statement: request.problem_statement,
            constraints: JSON.stringify(request.constraints)
        });
        const response = await this.get<StrategyResponse>(`/knowledge/strategy?${queryParams}`);
        return response;
    }

    /**
     * Get knowledge base statistics
     * GET /knowledge/stats
     */
    async getKnowledgeStats(): Promise<KnowledgeStats> {
        const response = await this.get<KnowledgeStats>('/knowledge/stats');
        return response;
    }

    /**
     * Get solver health status
     * GET /health
     */
    async getHealth(): Promise<HealthResponse> {
        const response = await this.get<HealthResponse>('/health');
        return response;
    }

    /**
     * Get API information
     * GET /
     */
    async getApiInfo(): Promise<{ name: string; version: string; endpoints: string[] }> {
        const response = await this.get<{ name: string; version: string; endpoints: string[] }>('/');
        return response;
    }

    // Private helper methods
    private async get<T>(path: string): Promise<T> {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(`${this.baseUrl}${path}`, {
                method: 'GET',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
            }
            return await response.json() as T;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error instanceof Error && error.name === 'AbortError') {
                throw new Error('Request timeout');
            }
            throw error;
        }
    }

    private async post<T>(path: string, body: any): Promise<T> {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(`${this.baseUrl}${path}`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(body),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
            }
            return await response.json() as T;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error instanceof Error && error.name === 'AbortError') {
                throw new Error('Request timeout - proof may be too complex');
            }
            throw error;
        }
    }
}

// Export singleton instance
export const mathSolverAPI = new MathSolverAPI();

// ============================================================================
// Core MathSolver Manager
// ============================================================================

export class MathSolverCore {
    private state: MathSolverState;
    private api: MathSolverAPI;
    private eventListeners: Map<string, ((data: any) => void)[]>;

    constructor() {
        this.state = this.createInitialState();
        this.api = mathSolverAPI;
        this.eventListeners = new Map();
    }

    private createInitialState(): MathSolverState {
        return {
            id: `mathsolver-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            currentProblem: null,
            history: [],
            z3Results: new Map(),
            leanResults: new Map(),
            unifiedResults: new Map(),
            isProcessing: false,
            activeSolvers: [],
            messages: [],
            knowledgeBase: []
        };
    }

    /**
     * Get current state
     */
    getState(): MathSolverState {
        return { ...this.state };
    }

    /**
     * Subscribe to state changes
     */
    on(event: string, callback: (data: any) => void): void {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event)!.push(callback);
    }

    /**
     * Emit event to listeners
     */
    private emit(event: string, data: any): void {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(cb => cb(data));
        }
    }

    /**
     * Create a new math problem
     */
    createProblem(
        statement: string, 
        options?: Partial<MathProblem>
    ): MathProblem {
        const problem: MathProblem = {
            id: `problem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            statement,
            constraints: options?.constraints || [],
            domain: options?.domain || detectDomain(statement),
            difficulty: options?.difficulty || 'medium',
            ...options
        };

        this.state.currentProblem = problem;
        this.state.history.push(problem);
        this.addMessage('system', `Created new problem: ${statement.substring(0, 100)}...`);
        
        this.emit('problemCreated', problem);
        return problem;
    }

    /**
     * Solve the current problem
     */
    async solve(options: SolveOptions): Promise<SolveResult> {
        const startTime = Date.now();
        const problem = options.problem;

        this.state.isProcessing = true;
        this.state.activeSolvers = [options.preferredSolver || 'auto'];
        this.addMessage('system', `Starting ${options.preferredSolver || 'auto'} solver...`);
        this.emit('solvingStarted', { problem, solver: options.preferredSolver });

        try {
            let result: SolveResult;

            if (options.useKnowledgeBase) {
                // Search for similar problems first
                const knowledge = await this.api.searchKnowledge({
                    query: problem.statement,
                    top_k: 3
                });
                this.state.knowledgeBase = [...this.state.knowledgeBase, ...knowledge.results];
                
                if (knowledge.results.length > 0 && knowledge.results[0].successRate > 0.8) {
                    this.addMessage('system', `Found similar problem in knowledge base (success rate: ${knowledge.results[0].successRate})`);
                }
            }

            switch (options.preferredSolver) {
                case 'z3':
                    result = await this.solveWithZ3(problem, options.timeout);
                    break;
                case 'lean':
                    result = await this.solveWithLean(problem, options.timeout);
                    break;
                case 'unified':
                case 'auto':
                case 'hybrid':
                default:
                    result = await this.solveWithUnified(problem, options);
                    break;
            }

            // Learn from successful solutions
            if (result.success && options.useKnowledgeBase) {
                this.learnFromSuccess(problem, result);
            }

            this.emit('solvingCompleted', result);
            return result;

        } catch (error) {
            const errorResponse: SolveResult = {
                success: false,
                problemId: problem.id,
                executionTimeMs: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error'
            };
            this.addMessage('system', `Error: ${errorResponse.error}`, 'error');
            this.emit('solvingError', errorResponse);
            return errorResponse;
        } finally {
            this.state.isProcessing = false;
            this.state.activeSolvers = [];
        }
    }

    private async solveWithZ3(problem: MathProblem, timeout?: number): Promise<SolveResult> {
        const startTime = Date.now();
        this.addMessage('solver', 'Solving with Z3 SMT solver...', undefined, 'z3');

        // Convert problem to SMT-LIB format
        const smtlibContent = this.problemToSMTLIB(problem);

        const request: Z3SolveRequest = {
            content: smtlibContent,
            timeout_ms: timeout ? timeout * 1000 : 30000,
            get_model: true,
            get_proof: true
        };

        const z3Result = await this.api.solveZ3(request);
        this.state.z3Results.set(problem.id, z3Result);

        const message = z3Result.status === 'sat' 
            ? `Z3 found satisfying assignment in ${z3Result.solving_time_ms}ms`
            : z3Result.status === 'unsat'
            ? `Z3 proved unsatisfiable in ${z3Result.solving_time_ms}ms`
            : `Z3 result: ${z3Result.status}`;
        
        this.addMessage('solver', message, z3Result.status === 'sat' ? 'proved' : 'unknown', 'z3');

        return {
            success: z3Result.status === 'sat' || z3Result.status === 'unsat',
            problemId: problem.id,
            z3Result,
            executionTimeMs: Date.now() - startTime
        };
    }

    private async solveWithLean(problem: MathProblem, timeout?: number): Promise<SolveResult> {
        const startTime = Date.now();
        this.addMessage('solver', 'Proving with Lean theorem prover...', undefined, 'lean');

        const request: ProveLeanRequest = {
            theorem: problem.statement,
            timeout_seconds: timeout ?? 300,
            auto_tactics: ['simp', 'rfl', 'tauto']
        };

        const leanResult = await this.api.proveLean(request);
        this.state.leanResults.set(problem.id, leanResult);

        const message = leanResult.success
            ? `Lean completed proof in ${leanResult.execution_time_ms}ms`
            : `Lean result: ${leanResult.error || 'failed'}`;

        if (leanResult.error) {
            this.addMessage('solver', `Error: ${leanResult.error}`, 'error', 'lean');
        }
        
        this.addMessage('solver', message, leanResult.success ? 'proved' : 'unknown', 'lean');

        return {
            success: leanResult.success,
            problemId: problem.id,
            leanResult,
            executionTimeMs: Date.now() - startTime
        };
    }

    private async solveWithUnified(problem: MathProblem, options: SolveOptions): Promise<SolveResult> {
        const startTime = Date.now();
        this.addMessage('solver', 'Running unified Z3+Lean solver...', undefined, 'unified');

        const request: SolveUnifiedRequest = {
            problem: problem.statement,
            preferred_solver: options.preferredSolver || 'auto',
            timeout_seconds: options.timeout ?? 300,
            require_consensus: options.requireConsensus ?? false
        };

        const unifiedResult = await this.api.solveUnified(request);
        this.state.unifiedResults.set(problem.id, unifiedResult);

        const consensusMsg = unifiedResult.verified
            ? `Consensus achieved (${unifiedResult.primary_solver})`
            : `Result: ${unifiedResult.result_status}`;
        
        this.addMessage('solver', consensusMsg, unifiedResult.verified ? 'proved' : 'unknown', 'unified');
        this.addMessage('system', `Recommended approach: ${unifiedResult.primary_solver}`);

        return {
            success: unifiedResult.verified || unifiedResult.result_status === 'success',
            problemId: problem.id,
            unifiedResult,
            executionTimeMs: Date.now() - startTime
        };
    }

    private async learnFromSuccess(problem: MathProblem, response: SolveResult): Promise<void> {
        try {
            const resultStr = response.z3Result?.status 
                || (response.leanResult?.success ? 'proved' : 'failed')
                || (response.unifiedResult?.result_status || 'unknown');

            await this.api.learnFromSolution({
                problem_statement: problem.statement,
                constraints: problem.constraints || [],
                result: resultStr,
                proof: response.z3Result?.proof || response.leanResult?.proof || null,
                metadata: {
                    solver: response.z3Result ? 'z3' : response.leanResult ? 'lean' : 'unified',
                    execution_time_ms: response.executionTimeMs
                }
            });
        } catch (error) {
            console.warn('Failed to learn from solution:', error);
        }
    }

    /**
     * Convert problem to SMT-LIB format (basic implementation)
     */
    private problemToSMTLIB(problem: MathProblem): string {
        // Basic SMT-LIB generation - can be enhanced
        let smtlib = '; Auto-generated SMT-LIB\n';
        smtlib += '(set-logic QF_LIA)\n\n';
        
        // Add problem statement as comment
        smtlib += `; Problem: ${problem.statement}\n`;
        
        // This is a placeholder - real implementation would parse the problem
        smtlib += '; Add variable declarations and constraints here\n';
        smtlib += '(check-sat)\n';
        smtlib += '(get-model)\n';
        
        return smtlib;
    }

    /**
     * Add message to conversation history
     */
    private addMessage(
        role: MathSolverMessage['role'], 
        content: string, 
        proofStatus?: ProofStatus,
        solverType?: SolverSystem
    ): void {
        const message: MathSolverMessage = {
            id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            role,
            content,
            timestamp: Date.now(),
            proofStatus,
            solverType
        };
        this.state.messages.push(message);
        this.emit('messageAdded', message);
    }

    /**
     * Get formatted proof for display
     */
    getFormattedProof(problemId: string, solver: SolverSystem): string | null {
        if (solver === 'z3') {
            const result = this.state.z3Results.get(problemId);
            return result?.proof || null;
        } else if (solver === 'lean') {
            const result = this.state.leanResults.get(problemId);
            return result?.proof || null;
        } else if (solver === 'unified') {
            const result = this.state.unifiedResults.get(problemId);
            return result?.result?.proof || null;
        }
        return null;
    }

    /**
     * Export current session state
     */
    exportState(): object {
        return {
            id: this.state.id,
            history: this.state.history,
            messages: this.state.messages,
            knowledgeBase: this.state.knowledgeBase,
            exportTimestamp: Date.now()
        };
    }

    /**
     * Import session state
     */
    importState(state: Partial<MathSolverState>): void {
        if (state.history) this.state.history = state.history;
        if (state.messages) this.state.messages = state.messages;
        if (state.knowledgeBase) this.state.knowledgeBase = state.knowledgeBase;
        this.emit('stateImported', this.state);
    }

    /**
     * Clear all state
     */
    reset(): void {
        this.state = this.createInitialState();
        this.emit('stateReset', null);
    }

    /**
     * Check if backend is available
     */
    async checkBackendHealth(): Promise<{ available: boolean; details?: HealthResponse }> {
        try {
            const health = await this.api.getHealth();
            return {
                available: health.status === 'healthy',
                details: health
            };
        } catch (error) {
            return {
                available: false,
                details: undefined
            };
        }
    }

    /**
     * Get knowledge base statistics
     */
    async getKnowledgeStats(): Promise<KnowledgeStats> {
        try {
            return await this.api.getKnowledgeStats();
        } catch (error) {
            console.warn('Failed to get knowledge stats:', error);
            return {};
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format proof for display with syntax highlighting markers
 */
export function formatProofForDisplay(proof: string, solver: SolverSystem): string {
    if (solver === 'z3') {
        // Format SMT-LIB proof
        return proof
            .replace(/;.*$/gm, '<comment>$&</comment>')
            .replace(/\b(declare-fun|assert|check-sat|get-model)\b/g, '<keyword>$1</keyword>')
            .replace(/\b(Bool|Int|Real|Array)\b/g, '<type>$1</type>');
    } else if (solver === 'lean') {
        // Format Lean proof
        return proof
            .replace(/--.*$/gm, '<comment>$&</comment>')
            .replace(/\b(theorem|lemma|proof|have|show|by|from|using)\b/g, '<keyword>$1</keyword>')
            .replace(/\b(∀|∃|→|↔|∧|∨|¬|≤|≥|≠)\b/g, '<operator>$&</operator>');
    }
    return proof;
}

/**
 * Detect mathematical domain from problem statement
 */
export function detectDomain(statement: string): MathProblem['domain'] {
    const lower = statement.toLowerCase();
    
    if (/\b(geometry|triangle|circle|angle|point|line|plane|polygon)\b/.test(lower)) {
        return 'geometry';
    }
    if (/\b(∫|derivative|integral|limit|continuity|differentiable|convergence)\b/.test(lower)) {
        return 'calculus';
    }
    if (/\b(∀|∃|→|↔|∧|∨|¬|predicate|proposition|implication)\b/.test(lower)) {
        return 'logic';
    }
    if (/\b(prime|divisor|modular|gcd|lcm|congruence|diophantine)\b/.test(lower)) {
        return 'number_theory';
    }
    if (/\b(equation|polynomial|matrix|vector|linear|algebraic)\b/.test(lower)) {
        return 'algebra';
    }
    if (/\b(arithmetic|number|sum|product|calculate|compute)\b/.test(lower)) {
        return 'arithmetic';
    }
    
    return 'other';
}

/**
 * Recommend solver based on problem characteristics
 */
export function recommendSolver(problem: MathProblem): SolverSystem {
    const domain = problem.domain || detectDomain(problem.statement);
    
    // Z3 excels at: arithmetic, constraints, finite domains
    // Lean excels at: proofs, theorems, inductive reasoning
    
    switch (domain) {
        case 'arithmetic':
        case 'algebra':
            return 'z3';
        case 'logic':
        case 'number_theory':
        case 'geometry':
            return 'lean';
        case 'calculus':
            return 'unified';
        default:
            return 'auto';
    }
}
