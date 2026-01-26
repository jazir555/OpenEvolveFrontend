import { BaseIntegrationAdapter } from './base';
import type { BackendClient } from '../api/backend';
import type { ParameterSchema, ExecutionOptions, RetryConfig, CircuitBreakerConfig } from '../api/types';
export interface LeanAideInputs {
    operation: 'translate' | 'prove' | 'verify' | 'mcts' | 'query';
    input: any;
    config?: any;
}
export interface LeanAideResult {
    type: string;
    result: any;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
export declare class LeanAideIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = LeanAideInputs, TResult = LeanAideResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    translateTheorem(theorem: string, options?: ExecutionOptions): Promise<any>;
    generateProof(theorem: string, strategy: string, options?: ExecutionOptions): Promise<any>;
    verifyProof(proof: string, options?: ExecutionOptions): Promise<any>;
    runMCTS(problem: string, config: any, options?: ExecutionOptions): Promise<any>;
    queryMath(question: string, options?: ExecutionOptions): Promise<any>;
}
export interface EvolutionInputs {
    operation: 'evolution' | 'adversarial' | 'coevolution';
    config: any;
    execConfig?: ExecutionOptions;
}
export interface EvolutionResult {
    executionId: string;
    bestSolution: any;
    bestFitness: number;
    fitnessHistory: number[];
    metadata: any;
}
export declare class EvolutionIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = EvolutionInputs, TResult = EvolutionResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    runEvolution(config: any, options?: ExecutionOptions): Promise<EvolutionResult>;
    runAdversarial(config: any, options?: ExecutionOptions): Promise<any>;
    runCoevolution(config: any, options?: ExecutionOptions): Promise<any>;
    getProgress(executionId: string, options?: ExecutionOptions): Promise<any>;
}
export interface KnowledgeInputs {
    operation: 'query' | 'extract' | 'search' | 'stats';
    input: any;
    config?: any;
}
export interface KnowledgeResult {
    nodes?: any[];
    edges?: any[];
    results?: any[];
    stats?: any;
    metadata: {
        graphId?: string;
        executionTime: number;
    };
}
export declare class KnowledgeIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = KnowledgeInputs, TResult = KnowledgeResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    queryGraph(query: any, options?: ExecutionOptions): Promise<KnowledgeResult>;
    extractKnowledge(document: string, options?: ExecutionOptions): Promise<KnowledgeResult>;
    searchKnowledge(query: string, options?: ExecutionOptions): Promise<KnowledgeResult>;
    getGraphStats(options?: ExecutionOptions): Promise<KnowledgeResult>;
}
export interface MakerInputs {
    operation: 'create' | 'execute' | 'validate' | 'list';
    input: any;
    config?: any;
}
export interface MakerResult {
    tool?: any;
    executionId?: string;
    status?: string;
    result?: any;
    tools?: any[];
    validation?: any;
    metadata: {
        executionTime?: number;
        timestamp: string;
    };
}
export declare class MakerIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = MakerInputs, TResult = MakerResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    createTool(config: any, options?: ExecutionOptions): Promise<MakerResult>;
    executeTool(toolId: string, input: any, options?: ExecutionOptions): Promise<MakerResult>;
    validateTool(toolId: string, options?: ExecutionOptions): Promise<MakerResult>;
}
export interface HephaestusInputs {
    operation: 'delegate' | 'status' | 'create' | 'list';
    input: any;
    config?: any;
}
export interface HephaestusResult {
    ticketId?: string;
    status?: string;
    assignedAgent?: string;
    tickets?: any[];
    result?: any;
    metadata: {
        executionTime?: number;
        timestamp: string;
    };
}
export declare class HephaestusIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = HephaestusInputs, TResult = HephaestusResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    delegateTask(task: any, options?: ExecutionOptions): Promise<HephaestusResult>;
    getTicketStatus(ticketId: string, options?: ExecutionOptions): Promise<HephaestusResult>;
    createTicket(ticket: any, options?: ExecutionOptions): Promise<HephaestusResult>;
}
export interface DecompositionInputs {
    operation: 'decompose' | 'subproblems' | 'dependencies';
    input: any;
    config?: any;
}
export interface DecompositionResult {
    planId?: string;
    subProblems?: any[];
    dependencyGraph?: any;
    executionOrder?: any[];
    metadata: {
        decompositionTime?: number;
        timestamp: string;
    };
}
export declare class DecompositionIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = DecompositionInputs, TResult = DecompositionResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    decompose(problem: string, strategy: string, options?: ExecutionOptions): Promise<DecompositionResult>;
    getSubProblems(planId: string, options?: ExecutionOptions): Promise<DecompositionResult>;
    getDependencyGraph(planId: string, options?: ExecutionOptions): Promise<DecompositionResult>;
}
export interface VerificationInputs {
    operation: 'verify' | 'checks' | 'validate';
    input: any;
    config?: any;
}
export interface VerificationResult {
    status: 'passed' | 'failed' | 'partial';
    score: number;
    checks: any[];
    requirementsCoverage?: any;
    metadata: {
        executionTime: number;
        timestamp: string;
    };
}
export declare class VerificationIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = VerificationInputs, TResult = VerificationResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    verifySolution(solution: any, requirements: string[], options?: ExecutionOptions): Promise<VerificationResult>;
    runChecks(solution: any, options?: ExecutionOptions): Promise<VerificationResult>;
}
export interface AssemblyInputs {
    operation: 'assemble' | 'integrate' | 'optimize';
    input: any;
    config?: any;
}
export interface AssemblyResult {
    status: 'success' | 'partial' | 'failed';
    assembledSolution?: any;
    integratedSystem?: any;
    optimizationResult?: any;
    statistics?: any;
    metadata: {
        executionTime: number;
        timestamp: string;
    };
}
export declare class AssemblyIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = AssemblyInputs, TResult = AssemblyResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    assembleSolutions(solutions: any[], options?: ExecutionOptions): Promise<AssemblyResult>;
    integrateSolution(assembledSolution: any, targetSystem: any, options?: ExecutionOptions): Promise<AssemblyResult>;
    optimizeSolution(solution: any, objectives: any[], options?: ExecutionOptions): Promise<AssemblyResult>;
}
export interface SolutionInputs {
    operation: 'generate' | 'optimize' | 'refine';
    input: any;
    config?: any;
}
export interface SolutionResult {
    solution: any;
    score?: number;
    metadata: any;
}
export declare class SolutionIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    execute<TInputs = SolutionInputs, TResult = SolutionResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
}
//# sourceMappingURL=all-integrations.d.ts.map