/**
 * PyGraphistry Visualization Plugin Types
 */
export interface GraphNode {
    id: string;
    label?: string;
    type?: string;
    [key: string]: string | number | boolean | undefined;
}
export interface GraphEdge {
    source: string;
    target: string;
    type?: string;
    weight?: number;
    [key: string]: string | number | boolean | undefined;
}
export interface PyGraphistryConfig {
    apiKey?: string;
    username?: string;
    password?: string;
    server?: string;
    gpuAcceleration?: boolean;
}
export interface PyGraphistryVizOptions {
    nodes: GraphNode[];
    edges: GraphEdge[];
    layout?: 'force_directed' | 'circular' | 'hierarchical';
    clustering?: boolean;
    clusteringMethod?: 'dbscan' | 'kmeans';
}
export interface PyGraphistryPluginState {
    config: PyGraphistryConfig;
    lastVizUrl: string | null;
    isInitializing: boolean;
    error: string | null;
    features: {
        pygraphistryEnabled: boolean;
        causalDiscoveryEnabled: boolean;
        optimizationEnabled: boolean;
        uqEnabled: boolean;
        globalChemEnabled: boolean;
        curieEnabled: boolean;
        temporalGraphEnabled: boolean;
        onekeEnabled: boolean;
        leanAideEnabled: boolean;
        sopEnabled: boolean;
        adversarialEnabled: boolean;
        pamiEnabled: boolean;
        aceEnabled: boolean;
        romaEnabled: boolean;
        datapizzaEnabled: boolean;
        hephaestusEnabled: boolean;
        claudiomiroEnabled: boolean;
        steerEnabled: boolean;
        researchQuestEnabled: boolean;
        kgEnabled: boolean;
        sgdEnabled: boolean;
        globalAnalyticsEnabled: boolean;
        mapElitesEnabled: boolean;
        verificationEnabled: boolean;
        problemAnalysisEnabled: boolean;
        dependencyEnabled: boolean;
        artifactGraphEnabled: boolean;
        sceEnabled: boolean;
        staticAnalysisEnabled: boolean;
        lltlEnabled: boolean;
        collaborationEnabled: boolean;
        workflowMonitorEnabled: boolean;
        lineageEnabled: boolean;
        gauntletEnabled: boolean;
        patternMiningEnabled: boolean;
        adaptationEnabled: boolean;
        ditoEnabled: boolean;
        crewaiEnabled: boolean;
        ragEnabled: boolean;
        deepkeEnabled: boolean;
        lean4Enabled: boolean;
        makerEnabled: boolean;
        mdapEnabled: boolean;
        mctsEnabled: boolean;
        hybridMCTSEnabled: boolean;
        e2ePlannerEnabled: boolean;
        evaluatorTeamEnabled: boolean;
        redTeamEnabled: boolean;
        blueTeamEnabled: boolean;
        qaSuiteEnabled: boolean;
        reseEnabled: boolean;
        materialKGEnabled: boolean;
        gnomeEnabled: boolean;
        physicsNemoEnabled: boolean;
        autogptEnabled: boolean;
        autogenEnabled: boolean;
        metagptEnabled: boolean;
        llm4iasEnabled: boolean;
        claraverseEnabled: boolean;
        aiScientistEnabled: boolean;
        uncertainpyEnabled: boolean;
        riskAnalyzerEnabled: boolean;
        karateclubEnabled: boolean;
        neuralKGEnabled: boolean;
        pylabrobotEnabled: boolean;
        pinnsEnabled: boolean;
    };
}
export interface GenericResult {
    [key: string]: any;
    timestamp: string;
}
export interface Lean4Theorem {
    name: string;
    statement: string;
    proof_sketch: string;
    verified: boolean;
}
export interface DeepKEEntity {
    text: string;
    type: string;
    confidence: number;
}
export interface DeepKERelation {
    head: string;
    tail: string;
    relation: string;
    confidence: number;
}
export interface DeepKEEvent {
    trigger: string;
    type: string;
    arguments: string[];
}
export interface DeepKEResult {
    entities: DeepKEEntity[];
    relations: DeepKERelation[];
    events: DeepKEEvent[];
    timestamp: string;
}
export interface CrewAgent {
    role: string;
    goal?: string;
    status: string;
}
export interface CrewTask {
    description: string;
    status: string;
    agent: string;
}
export interface CrewAIResult {
    crew_name: string;
    agents: CrewAgent[];
    tasks: CrewTask[];
    process: string;
    progress: number;
    timestamp: string;
}
export interface RAGSearchResult {
    score: number;
    content: string;
    source: string;
}
export interface DITOContradiction {
    id: string;
    pair: string[];
    description: string;
    confidence: number;
}
export interface DITOResult {
    total_constraints: number;
    contradiction_count: number;
    contradictions: DITOContradiction[];
    stats: Record<string, number>;
    timestamp: string;
}
export interface AdaptationEvent {
    timestamp: string;
    gauntlet: string;
    change: string;
}
export interface AdaptationResult {
    total_adaptations: number;
    strictness_distribution: {
        more_strict: number;
        less_strict: number;
        similar: number;
    };
    recent_events: AdaptationEvent[];
    timestamp: string;
}
export interface MinedPatternCluster {
    cluster_id: number;
    size: number;
    avg_complexity: number;
    avg_success_rate: number;
    most_common_domain: string;
    description: string;
}
export interface GauntletSummary {
    gauntlet_id: string;
    gauntlet_type: string;
    avg_catch_rate: number;
    avg_false_positive_rate: number;
    effectiveness_score: number;
    total_runs: number;
}
export type GauntletEffectivenessResult = Record<string, GauntletSummary>;
export interface LineageImprovement {
    step: number;
    parent_id: string;
    child_id: string;
    improvement: Record<string, number>;
    generation?: number;
}
export interface LineageTrace {
    final_program_id: string;
    generation_depth: number;
    improvement_steps: LineageImprovement[];
}
export interface WorkflowMetricSet {
    best_fitness: number;
    avg_fitness: number;
    diversity: number;
    population_size: number;
}
export interface WorkflowResourceUsage {
    memory_mb: number;
    cpu_cores: number;
}
export interface WorkflowEvent {
    timestamp: string;
    status: string;
    message: string;
}
export interface WorkflowMonitorResult {
    workflow_id: string;
    status: string;
    progress: number;
    execution_time: number;
    current_stage: string;
    metrics: WorkflowMetricSet;
    resource_usage: WorkflowResourceUsage;
    events: WorkflowEvent[];
}
export interface CollabSession {
    session_id: string;
    name: string;
    participants: string[];
    status: 'active' | 'locked' | 'closed';
    last_edit: string;
    conflict_count: number;
}
export interface LossMapping {
    constraint_id: string;
    success: boolean;
    weight: number;
    fuzzy_type: string;
    error?: string;
}
export interface StaticAnalysisIssue {
    file: string;
    line: number;
    message: string;
    suggestion?: string;
}
export interface StaticAnalysisResult {
    summary: {
        total_issues: number;
        by_severity: Record<string, number>;
        by_category: Record<string, number>;
    };
    issues_by_severity: Record<string, StaticAnalysisIssue[]>;
    timestamp: string;
}
export interface SymbolicConstraint {
    id: string;
    type: 'hard' | 'soft' | 'preference';
    description: string;
    formalization: string;
    verified: boolean;
    source: string;
}
export interface ArtifactNode {
    id: string;
    label: string;
    type: string;
    domain?: string;
    confidence?: number;
}
export interface ArtifactEdge {
    source: string;
    target: string;
    label: string;
}
export interface ArtifactResult {
    nodes: ArtifactNode[];
    edges: ArtifactEdge[];
    timestamp: string;
}
export interface DependencyNode {
    id: string;
    label: string;
    status: string;
    complexity: number;
}
export interface DependencyEdge {
    source: string;
    target: string;
}
export interface DependencyResult {
    nodes: DependencyNode[];
    edges: DependencyEdge[];
    timestamp: string;
}
export interface ProviderMetrics {
    tokens: number;
    cost: number;
}
export interface GlobalAnalyticsResult {
    total_workflows: number;
    total_tokens: number;
    total_cost: number;
    avg_execution_time: number;
    provider_breakdown: Record<string, ProviderMetrics>;
    timestamp: string;
}
export interface ProblemAnalysisResult {
    title: string;
    domain: string;
    problem_type: string;
    complexity: {
        overall: number;
        cognitive: number;
        computational: number;
    };
    constraints: {
        description: string;
        type: string;
        severity: string;
    }[];
    success_criteria: {
        description: string;
        metric: string;
        threshold: number;
    }[];
    timestamp: string;
}
export interface VerificationTest {
    category: string;
    test: string;
    status: 'passed' | 'failed' | 'error';
}
export interface VerificationResult {
    total_tests: number;
    passed: number;
    failed: number;
    results: VerificationTest[];
    success_rate: number;
    timestamp: string;
}
export interface MAPElitesResult {
    generations: number[];
    best_scores: number[];
    average_scores: number[];
    diversity_scores: number[];
    map_elites_grid: number[][];
    feature_dimensions: string[];
    timestamp: string;
}
export interface SGDResult {
    active_workflows: number;
    completed_workflows: number;
    failed_workflows: number;
    active_tickets: number;
    completed_tickets: number;
    failed_tickets: number;
    total_gauntlet_runs: number;
    successful_gauntlet_runs: number;
    success_rate: number;
    timestamp: string;
}
export interface KGNode {
    id: string;
    label: string;
}
export interface KGEdge {
    source: string;
    target: string;
    label: string;
}
export interface KGResult {
    nodes: KGNode[];
    edges: KGEdge[];
    timestamp: string;
}
export interface ResearchStage {
    id: number;
    name: string;
    description: string;
    objectives: string[];
    outputs: string[];
    quality_checks: string[];
}
export interface SteerJudgeResult {
    judge: string;
    passed: boolean;
    reason: string;
    suggested_fixes?: {
        title: string;
        description: string;
    }[];
}
export interface SteerResult {
    all_passed: boolean;
    results: SteerJudgeResult[];
    ace_learning?: {
        status: string;
        learned_skills: string[];
    };
    timestamp: string;
}
export interface ClaudiomiroSubTask {
    title: string;
    description: string;
    status: 'pending' | 'completed' | 'failed';
}
export interface ClaudiomiroResult {
    task_id: string;
    sub_tasks: ClaudiomiroSubTask[];
    num_tasks: number;
    timestamp: string;
}
export interface TicketActivity {
    id: string;
    task: string;
    status: string;
}
export interface HephaestusResult {
    total_tickets: number;
    status_distribution: Record<string, number>;
    recent_activity: TicketActivity[];
    timestamp: string;
}
export interface DataPizzaAgentResult {
    response: string;
    steps: number;
}
export interface DataPizzaResult {
    team_name: string;
    task: string;
    workflow: string;
    results: {
        blue?: DataPizzaAgentResult;
        red?: DataPizzaAgentResult;
        gold?: DataPizzaAgentResult;
    };
    total_steps: number;
    status: string;
    timestamp: string;
}
export interface ROMAResult {
    task: string;
    synthesized_result: string;
    status: string;
    timestamp: string;
}
export interface TeamAnalytics {
    team_name: string;
    success_rate: number;
    avg_quality_score: number;
}
export interface GauntletAnalytics {
    gauntlet_name: string;
    detection_rate: number;
    precision: number;
}
export interface ACEResult {
    top_teams: TeamAnalytics[];
    top_gauntlets: GauntletAnalytics[];
    timestamp: string;
}
export interface PatternItem {
    items: string[];
    support: number;
}
export interface PAMIResult {
    patterns: PatternItem[];
    total_found: number;
    timestamp: string;
}
export interface AttackResult {
    strategy: string;
    success: boolean;
    severity: number;
    description: string;
}
export interface AdversarialResult {
    robustness_score: number;
    is_robust: boolean;
    total_attacks: number;
    attacks_blocked: number;
    attack_results: AttackResult[];
    timestamp: string;
}
export interface SOPStep {
    step_number: number;
    action: string;
    duration?: number;
    verification_method?: string;
    acceptance_criteria?: string;
    substeps?: string[];
}
export interface SOP {
    title: string;
    version: string;
    status: string;
    description: string;
    protocols: SOPStep[];
    equipment: any[];
    materials: any[];
    safety_protocols?: string[];
    quality_control?: string[];
    timestamp: string;
}
export interface LeanAideResult {
    theorem_lean: string;
    proof_status: string;
    confidence: number;
    timestamp: string;
}
export interface Entity {
    text: string;
    type: string;
    start?: number;
    end?: number;
}
export interface Relation {
    subject: string;
    predicate: string;
    object: string;
}
export interface ExtractionResult {
    entities: Entity[];
    relations: Relation[];
    triples: any[];
    confidence: number;
}
export interface TemporalNode {
    uuid: string;
    name: string;
    summary?: string;
    labels?: string[];
}
export interface TemporalEdge {
    uuid: string;
    fact: string;
    source_node: string;
    target_node: string;
    valid_at?: string;
}
export interface TemporalGraphResult {
    nodes: TemporalNode[];
    edges: TemporalEdge[];
}
export interface ExperimentStep {
    description: string;
    duration?: number;
    [key: string]: any;
}
export interface ExperimentProtocol {
    protocol_id: string;
    hypothesis: {
        statement: string;
        independent_variables: string[];
        dependent_variables: string[];
    };
    steps: ExperimentStep[];
    equipment: string[];
}
export interface ChemicalEntry {
    name: string;
    smiles: string;
    list?: string;
    molecular_formula?: string;
    molecular_weight?: number;
}
export interface UQResult {
    function: string;
    statistics: {
        mean: number;
        std: number;
        min: number;
        max: number;
        [key: string]: any;
    };
    sensitivity: {
        first_order: number[];
        method: string;
    } | null;
    output_samples: number[];
}
export interface CausalDiscoveryResult {
    nodes: string[];
    edges: [number, number, string][];
    adjacency_matrix: number[][];
    algorithm: string;
    timestamp: string;
}
export interface PyGraphistryPlugin {
    initialize(config: PyGraphistryConfig): Promise<void>;
    generateVisualization(options: PyGraphistryVizOptions): Promise<string | null>;
    getState(): PyGraphistryPluginState;
    updateFeatures(features: Partial<PyGraphistryPluginState['features']>): void;
    updateConfig(config: Partial<PyGraphistryConfig>): void;
}
