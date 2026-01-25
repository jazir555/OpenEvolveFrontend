import { OpenEvolveBaseNode } from './OpenEvolveBaseNode';
import { NodeConfig, NodeResult, ExecutionContext } from './BaseNode';
/**
 * Invention domains
 */
export type InventionDomain = 'technology' | 'hardware' | 'business' | 'process' | 'scientific' | 'creative';
/**
 * Planning stages
 */
export type PlanningStage = 'research' | 'ideation' | 'prototyping' | 'testing' | 'validation' | 'scaling' | 'commercialization';
/**
 * Detail levels
 */
export type DetailLevel = 'overview' | 'detailed' | 'comprehensive';
/**
 * Invention planning result
 */
export interface InventionResult {
    /** Generated invention plan */
    plan: InventionPlan;
    /** Prior art analysis if requested */
    priorArt?: PriorArtAnalysis;
    /** Feasibility analysis if requested */
    feasibility?: FeasibilityAnalysis;
    /** Implementation roadmap if requested */
    roadmap?: ImplementationRoadmap;
    /** Lean formalizations for math */
    leanProofs?: LeanProof[];
    /** Error analysis */
    errorAnalysis: ErrorAnalysis;
    /** Red team results */
    redTeamResults?: RedTeamResults;
    /** Blue team results */
    blueTeamResults?: BlueTeamResults;
    /** Success criteria */
    successCriteria: SuccessCriterion[];
    /** Execution time in seconds */
    executionTime: number;
    /** Quality assessment */
    qualityAssessment: {
        innovation: number;
        feasibility: number;
        clarity: number;
        completeness: number;
    };
}
/**
 * Invention plan
 */
export interface InventionPlan {
    /** Plan title */
    title: string;
    /** Executive summary */
    summary: string;
    /** Invention stages */
    stages: PlanStage[];
    /** Resource requirements */
    resources: Resource[];
    /** Timeline estimates */
    timeline: TimelineEstimate;
    /** Risk assessment */
    risks: Risk[];
}
/**
 * Plan stage
 */
export interface PlanStage {
    /** Stage name */
    name: PlanningStage;
    /** Description */
    description: string;
    /** Steps in this stage */
    steps: string[];
    /** Estimated duration */
    duration: string;
    /** Dependencies */
    dependencies: PlanningStage[];
}
/**
 * Resource
 */
export interface Resource {
    /** Resource type */
    type: 'human' | 'financial' | 'technical' | 'infrastructure';
    /** Resource description */
    description: string;
    /** Quantity */
    quantity: number;
    /** Unit */
    unit: string;
}
/**
 * Timeline estimate
 */
export interface TimelineEstimate {
    /** Total duration */
    totalDuration: string;
    /** Stage durations */
    stageDurations: Record<PlanningStage, string>;
}
/**
 * Risk
 */
export interface Risk {
    /** Risk description */
    description: string;
    /** Likelihood (high/medium/low) */
    likelihood: string;
    /** Impact (high/medium/low) */
    impact: string;
    /** Mitigation strategy */
    mitigation: string;
}
/**
 * Prior art analysis
 */
export interface PriorArtAnalysis {
    /** Existing solutions */
    existingSolutions: ExistingSolution[];
    /** Patent landscape */
    patents: PatentReference[];
    /** Gaps identified */
    gaps: string[];
    /** Novelty assessment */
    novelty: 'high' | 'medium' | 'low';
}
/**
 * Existing solution
 */
export interface ExistingSolution {
    /** Solution name */
    name: string;
    /** Description */
    description: string;
    /** Limitations */
    limitations: string[];
    /** Similarity score */
    similarity: number;
}
/**
 * Patent reference
 */
export interface PatentReference {
    /** Patent number */
    number: string;
    /** Title */
    title: string;
    /** Summary */
    summary: string;
    /** Relevance */
    relevance: string;
}
/**
 * Feasibility analysis
 */
export interface FeasibilityAnalysis {
    /** Technical feasibility */
    technical: FeasibilityAssessment;
    /** Financial feasibility */
    financial: FeasibilityAssessment;
    /** Market feasibility */
    market: FeasibilityAssessment;
    /** Overall feasibility */
    overall: 'high' | 'medium' | 'low';
}
/**
 * Feasibility assessment
 */
export interface FeasibilityAssessment {
    /** Score (0-1) */
    score: number;
    /** Key factors */
    factors: string[];
    /** Challenges */
    challenges: string[];
}
/**
 * Implementation roadmap
 */
export interface ImplementationRoadmap {
    /** Phases */
    phases: RoadmapPhase[];
    /** Milestones */
    milestones: Milestone[];
}
/**
 * Roadmap phase
 */
export interface RoadmapPhase {
    /** Phase name */
    name: string;
    /** Duration */
    duration: string;
    /** Objectives */
    objectives: string[];
    /** Deliverables */
    deliverables: string[];
}
/**
 * Milestone
 */
export interface Milestone {
    /** Milestone name */
    name: string;
    /** Target date */
    targetDate: string;
    /** Success criteria */
    criteria: string[];
}
/**
 * Lean proof
 */
export interface LeanProof {
    /** Proof name */
    name: string;
    /** Lean code */
    leanCode: string;
    /** Mathematical statement */
    statement: string;
    /** Verification status */
    verified: boolean;
}
/**
 * Error analysis
 */
export interface ErrorAnalysis {
    /** Identified error sources */
    errorSources: ErrorSource[];
    /** Mitigation strategies */
    mitigations: MitigationStrategy[];
}
/**
 * Error source
 */
export interface ErrorSource {
    /** Error description */
    description: string;
    /** Category */
    category: string;
    /** Probability */
    probability: 'high' | 'medium' | 'low';
    /** Severity */
    severity: 'high' | 'medium' | 'low';
}
/**
 * Mitigation strategy
 */
export interface MitigationStrategy {
    /** Error source */
    errorSource: string;
    /** Strategy */
    strategy: string;
    /** Effectiveness */
    effectiveness: number;
}
/**
 * Red team results
 */
export interface RedTeamResults {
    /** Attack vectors identified */
    attackVectors: string[];
    /** Vulnerabilities found */
    vulnerabilities: Vulnerability[];
    /** Test scenarios */
    scenarios: TestScenario[];
}
/**
 * Vulnerability
 */
export interface Vulnerability {
    /** Description */
    description: string;
    /** Severity */
    severity: 'critical' | 'high' | 'medium' | 'low';
    /** Exploitability */
    exploitability: 'high' | 'medium' | 'low';
}
/**
 * Test scenario
 */
export interface TestScenario {
    /** Scenario name */
    name: string;
    /** Description */
    description: string;
    /** Outcome */
    outcome: 'passed' | 'failed' | 'partial';
}
/**
 * Blue team results
 */
export interface BlueTeamResults {
    /** Defenses implemented */
    defenses: Defense[];
    /** Patches applied */
    patches: Patch[];
    /** Verification results */
    verifications: VerificationResult[];
}
/**
 * Defense
 */
export interface Defense {
    /** Description */
    description: string;
    /** Type */
    type: string;
    /** Effectiveness */
    effectiveness: number;
}
/**
 * Patch
 */
export interface Patch {
    /** Vulnerability addressed */
    vulnerability: string;
    /** Patch description */
    description: string;
    /** Status */
    status: 'applied' | 'pending' | 'failed';
}
/**
 * Success criterion
 */
export interface SuccessCriterion {
    /** Criterion name */
    name: string;
    /** Type */
    type: 'binary' | 'quantitative' | 'qualitative';
    /** Description */
    description: string;
    /** Pass condition */
    passCondition: string;
    /** Measurement method */
    measurementMethod?: string;
}
/**
 * Invention node configuration
 */
export interface InventionNodeConfig extends NodeConfig {
    /** Invention goal */
    goal: string;
    /** Primary domain */
    domain: InventionDomain;
    /** Innovativeness level (0-1) */
    innovativeness: number;
    /** Planning stages to include */
    planningStages: PlanningStage[];
    /** Constraints */
    constraints?: string;
    /** Target audience */
    targetAudience?: string;
    /** Include prior art analysis */
    includePriorArt: boolean;
    /** Include feasibility analysis */
    includeFeasibility: boolean;
    /** Include implementation roadmap */
    includeRoadmap: boolean;
    /** Detail level */
    detailLevel: DetailLevel;
}
/**
 * Invention Planner Node class
 */
export declare class InventionNode extends OpenEvolveBaseNode {
    /**
     * Node type identifier
     */
    static readonly NODE_TYPE = "Invention";
    /**
     * Node display name
     */
    static readonly DISPLAY_NAME = "End-to-End Invention Planner";
    /**
     * Node category
     */
    static readonly CATEGORY = "planning";
    /**
     * Node icon
     */
    static readonly ICON = "\uD83D\uDCA1";
    constructor(config: InventionNodeConfig);
    /**
     * Get parameter schema
     */
    getParameterSchema(): ({
        name: string;
        type: string;
        label: string;
        description: string;
        required: boolean;
        multiline: boolean;
        options?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        required: boolean;
        options: {
            value: string;
            label: string;
        }[];
        multiline?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        defaultValue: number;
        min: number;
        max: number;
        step: number;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        options: {
            value: string;
            label: string;
        }[];
        required?: undefined;
        multiline?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        multiline: boolean;
        required?: undefined;
        options?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description?: undefined;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        defaultValue: boolean;
        description?: undefined;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        options: {
            value: string;
            label: string;
        }[];
        description?: undefined;
        required?: undefined;
        multiline?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    })[];
    /**
     * Validate inputs
     */
    protected validate(inputs: any, context: ExecutionContext): Promise<void>;
    /**
     * Execute invention planning
     */
    protected execute(inputs: any, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Validate inputs
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get display name
     */
    getDisplayName(): string;
    /**
     * Get icon
     */
    getIcon(): string;
    /**
     * Get category
     */
    getCategory(): string;
    /**
     * Get version
     */
    getVersion(): string;
}
