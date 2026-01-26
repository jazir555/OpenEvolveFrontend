export interface OpenEvolveConfig {
    baseUrl: string;
    apiKey?: string;
    timeout?: number;
    retries?: number;
    headers?: Record<string, string>;
    debug?: boolean;
}
export interface ProgressUpdate {
    progress: number;
    status: string;
    data?: any;
    timestamp: Date;
}
export interface ValidationResult {
    valid: boolean;
    errors: ValidationError[];
    warnings: ValidationError[];
}
export interface ValidationError {
    field: string;
    message: string;
    code?: string;
}
export interface ParameterSchema {
    type: 'object';
    properties: Record<string, ParameterProperty>;
    required?: string[];
}
export interface ParameterProperty {
    type: string;
    description?: string;
    default?: any;
    enum?: any[];
    minimum?: number;
    maximum?: number;
    pattern?: string;
}
export interface Integration<TInputs = any, TResult = any> {
    name: string;
    version: string;
    description?: string;
    execute(inputs: TInputs): Promise<TResult>;
    validate(inputs: TInputs): ValidationResult;
    getSchema(): ParameterSchema;
    executeStream?(inputs: TInputs, onUpdate: (update: ProgressUpdate) => void): Promise<TResult>;
}
export declare abstract class BaseIntegration<TInputs, TResult> implements Integration<TInputs, TResult> {
    abstract name: string;
    abstract version: string;
    description?: string;
    abstract execute(inputs: TInputs): Promise<TResult>;
    abstract getSchema(): ParameterSchema;
    validate(inputs: TInputs): ValidationResult;
    executeStream(inputs: TInputs, onUpdate: (update: ProgressUpdate) => void): Promise<TResult>;
}
export declare class OpenEvolveError extends Error {
    code: string;
    details?: any | undefined;
    constructor(code: string, message: string, details?: any | undefined);
}
export declare class NetworkError extends OpenEvolveError {
    constructor(message: string, details?: any);
}
export declare class ValidationError extends OpenEvolveError {
    constructor(message: string, details?: any);
}
export declare class ExecutionError extends OpenEvolveError {
    constructor(message: string, details?: any);
}
export interface DecompositionInputs {
    problem_statement: string;
    method?: 'hierarchical' | 'hybrid' | 'lean4';
    max_depth?: number;
    constraints?: Record<string, any>;
}
export interface DecompositionResult {
    sub_problems: SubProblem[];
    dependencies: Dependency[];
    metadata: DecompositionMetadata;
}
export interface SubProblem {
    id: string;
    description: string;
    complexity: number;
    estimated_time?: number;
    dependencies: string[];
    parameters?: Record<string, any>;
}
export interface Dependency {
    from: string;
    to: string;
    type: 'sequential' | 'parallel' | 'conditional';
}
export interface DecompositionMetadata {
    method: string;
    total_complexity: number;
    estimated_time: number;
    timestamp: Date;
}
export interface LeanAideInputs {
    mode: 'formal_verification' | 'mcts' | 'mdap';
    problem: string;
    tactics?: string[];
    iterations?: number;
    constraints?: Record<string, any>;
}
export interface LeanAideResult {
    proof?: FormalProof;
    plan?: MCTSPlan;
    optimization?: MDAPOptimization;
    metadata: LeanAideMetadata;
}
export interface FormalProof {
    lemma: string;
    tactics: string[];
    verified: boolean;
    proof_script: string;
}
export interface MCTSPlan {
    root_state: string;
    tree: MCTSNode;
    best_path: string[];
    expected_value: number;
}
export interface MCTSNode {
    state: string;
    visits: number;
    value: number;
    children: MCTSNode[];
}
export interface MDAPOptimization {
    parameters: Record<string, number>;
    objective_value: number;
    iterations: number;
    converged: boolean;
}
export interface LeanAideMetadata {
    mode: string;
    execution_time: number;
    iterations?: number;
    timestamp: Date;
}
export interface EvolutionInputs {
    mode?: 'evolutionary' | 'adversarial';
    initial_population: any[];
    fitness_function: string;
    generations: number;
    mutation_rate?: number;
    crossover_rate?: number;
    selection_method?: string;
}
export interface EvolutionResult {
    final_population: any[];
    best_solution: any;
    fitness_history: number[];
    generation_stats: GenerationStats[];
    metadata: EvolutionMetadata;
}
export interface GenerationStats {
    generation: number;
    best_fitness: number;
    average_fitness: number;
    diversity: number;
}
export interface EvolutionMetadata {
    mode: string;
    generations_completed: number;
    total_evaluations: number;
    execution_time: number;
    timestamp: Date;
}
export interface KnowledgeInputs {
    mode: 'extraction' | 'query' | 'update';
    source?: any;
    query?: string;
    graph_id?: string;
    extraction_type?: string;
    additions?: any[];
    deletions?: any[];
}
export interface KnowledgeResult {
    graph?: KnowledgeGraph;
    results?: QueryResult[];
    metadata: KnowledgeMetadata;
}
export interface KnowledgeGraph {
    id: string;
    nodes: GraphNode[];
    edges: GraphEdge[];
    metadata: Record<string, any>;
}
export interface GraphNode {
    id: string;
    type: string;
    properties: Record<string, any>;
}
export interface GraphEdge {
    id: string;
    source: string;
    target: string;
    type: string;
    properties: Record<string, any>;
}
export interface QueryResult {
    entity: string;
    relations: string[];
    confidence: number;
    evidence: any[];
}
export interface KnowledgeMetadata {
    mode: string;
    graph_id?: string;
    nodes_count?: number;
    edges_count?: number;
    execution_time: number;
    timestamp: Date;
}
export interface MakerInputs {
    mode: 'create_tool' | 'create_workflow' | 'execute';
    specification?: ToolSpecification;
    workflow?: WorkflowSpecification;
    tool_id?: string;
    inputs?: any;
}
export interface MakerResult {
    tool?: Tool;
    workflow?: Workflow;
    execution_result?: any;
    metadata: MakerMetadata;
}
export interface ToolSpecification {
    name: string;
    description?: string;
    inputs: ParameterDefinition[];
    outputs: ParameterDefinition[];
    logic?: string;
}
export interface WorkflowSpecification {
    name: string;
    steps: WorkflowStep[];
    dependencies?: WorkflowDependency[];
}
export interface WorkflowStep {
    id: string;
    tool: string;
    config: Record<string, any>;
}
export interface WorkflowDependency {
    from: string;
    to: string;
    condition?: string;
}
export interface Tool {
    id: string;
    name: string;
    specification: ToolSpecification;
    created_at: Date;
}
export interface Workflow {
    id: string;
    name: string;
    specification: WorkflowSpecification;
    created_at: Date;
}
export interface MakerMetadata {
    mode: string;
    tool_id?: string;
    workflow_id?: string;
    execution_time?: number;
    timestamp: Date;
}
export interface ParameterDefinition {
    name: string;
    type: string;
    required: boolean;
    description?: string;
    default?: any;
}
export interface HephaestusInputs {
    mode: 'delegate' | 'orchestrate' | 'monitor';
    task?: string;
    agent_type?: string;
    constraints?: Record<string, any>;
    workflow?: OrchestrationWorkflow;
    session_id?: string;
}
export interface HephaestusResult {
    delegation_result?: DelegationResult;
    orchestration_result?: OrchestrationResult;
    session_info?: SessionInfo;
    metadata: HephaestusMetadata;
}
export interface DelegationResult {
    task_id: string;
    agent: string;
    status: string;
    result?: any;
    execution_time: number;
}
export interface OrchestrationWorkflow {
    id: string;
    name: string;
    steps: OrchestrationStep[];
    dependencies: WorkflowDependency[];
}
export interface OrchestrationStep {
    id: string;
    task: string;
    agent_type: string;
    config: Record<string, any>;
}
export interface OrchestrationResult {
    workflow_id: string;
    status: string;
    results: Record<string, any>;
    execution_time: number;
}
export interface SessionInfo {
    session_id: string;
    status: string;
    active_tasks: string[];
    completed_tasks: string[];
    metrics: Record<string, number>;
}
export interface HephaestusMetadata {
    mode: string;
    session_id?: string;
    execution_time: number;
    timestamp: Date;
}
export interface Integrations {
    decomposition: Integration<DecompositionInputs, DecompositionResult>;
    leanaide: Integration<LeanAideInputs, LeanAideResult>;
    evolution: Integration<EvolutionInputs, EvolutionResult>;
    knowledge: Integration<KnowledgeInputs, KnowledgeResult>;
    maker: Integration<MakerInputs, MakerResult>;
    hephaestus: Integration<HephaestusInputs, HephaestusResult>;
}
export interface OpenEvolveClient {
    integrations: Integrations;
    config: OpenEvolveConfig;
    disconnect(): void;
}
//# sourceMappingURL=index.d.ts.map