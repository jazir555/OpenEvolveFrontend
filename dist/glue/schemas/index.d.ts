/**
 * Canonical Schemas Export - Anti-Corruption Layer
 *
 * Central export point for all canonical schemas in the glue layer.
 *
 * IMPORT RULE: All adapters must import schemas from this file, NOT directly.
 * This ensures a single source of truth for data models.
 *
 * @module glue/schemas
 */
export { SolverRequest, SolverResponse, KnowledgeGraphResponse, Entity, Relation, Z3ResultType, type Z3ResultType as Z3ResultTypeEnum, type SolverRequest as Z3SolverRequest, type SolverResponse as Z3SolverResponse, type KnowledgeGraphResponse as Z3KnowledgeGraphResponse, type Entity as Z3Entity, type Relation as Z3Relation, transformZ3ResponseToCanonical, transformCanonicalToZ3Request, validateSolverRequest, validateSolverResponse, validateKnowledgeGraphResponse, Z3Examples, } from './z3-canonical';
export { ProofVerificationRequest, ProofVerificationResponse, LeanCompilationRequest, LeanCompilationResponse, LeanMessage, LeanTactic, LeanSeverity, type LeanTactic as LeanTacticEnum, type LeanSeverity as LeanSeverityEnum, type ProofVerificationRequest as LeanAideProofVerificationRequest, type ProofVerificationResponse as LeanAideProofVerificationResponse, type LeanCompilationRequest as LeanAideCompilationRequest, type LeanCompilationResponse as LeanAideCompilationResponse, type LeanMessage as LeanCompilerMessage, transformLeanAideResponseToCanonical, transformCanonicalToLeanAideRequest, transformCompilationResponseToCanonical, validateProofVerificationRequest, validateProofVerificationResponse, validateLeanCompilationRequest, validateLeanCompilationResponse, LeanAideExamples, } from './leanaide-canonical';
export { RAGRequest, RAGResponse, DocumentChunk, DocumentIngestionRequest, DocumentIngestionResponse, RAGError, type RAGRequest as RAGBitsRequest, type RAGResponse as RAGBitsResponse, type DocumentChunk as RAGBitsDocumentChunk, type DocumentIngestionRequest as RAGBitsIngestionRequest, type DocumentIngestionResponse as RAGBitsIngestionResponse, transformRAGResponseToCanonical, transformCanonicalToRAGRequest, validateRAGRequest, validateRAGResponse, validateDocumentChunk, isRAGRequest, isRAGResponse, RAGExamples, } from './ragbits-canonical';
export { BubbleRequest, BubbleResponse, WorkflowRequest, WorkflowResponse, BubbleStatusRequest, BubbleStatusResponse, BubbleType, BubbleStatus, BubbleLabError, type BubbleType as BubbleTypeEnum, type BubbleStatus as BubbleStatusEnum, type BubbleRequest as BubbleLabRequest, type BubbleResponse as BubbleLabResponse, type WorkflowRequest as BubbleLabWorkflowRequest, type WorkflowResponse as BubbleLabWorkflowResponse, transformBubbleResponseToCanonical, transformCanonicalToBubbleRequest, transformWorkflowResponseToCanonical, transformCanonicalToWorkflowRequest, validateBubbleRequest, validateBubbleResponse, validateWorkflowRequest, validateWorkflowResponse, isBubbleRequest, isWorkflowRequest, BubbleLabExamples, } from './bubblelab-canonical';
export { AdaptiveMdapRequest, AdaptiveMdapResponse, AdaptiveMdapBatchRequest, AdaptiveMdapBatchResponse, AdaptiveMdapError, ProcessingDomain, AdaptationMode, type ProcessingDomain as ProcessingDomainEnum, type AdaptationMode as AdaptationModeEnum, validateAdaptiveMdapRequest, validateAdaptiveMdapResponse, isAdaptiveMdapRequest, AdaptiveMdapExamples, } from './adaptive-mdap-canonical';
export { WorkflowType, ComplexityDimensions, ComplexityScore, SubProblem, ProblemDecompositionResult, TeamMember, TeamSelectionResult, ResourceOptimizationResult, GauntletType, GauntletSeverity, GauntletConfig, GauntletResult, GauntletPipeline, GauntletPipelineResult, ICRPatternType, ICRPattern, ICRPrediction, PatternCluster, ICRPatternInsights, ChartType, UIChartData, WorkflowTimeline, AdapterHealthStatus, CacheStatistics, PerformanceMetrics, AsyncOperationStatus, AsyncOperation, BatchOperation, AdditionalSystemType, SystemHealth, UnifiedSystemHealth, WorkflowStep, CrossSystemWorkflowResult, type WorkflowType as WorkflowTypeEnum, type GauntletType as GauntletTypeEnum, type GauntletSeverity as GauntletSeverityEnum, type ICRPatternType as ICRPatternTypeEnum, type ChartType as ChartTypeEnum, type AsyncOperationStatus as AsyncOperationStatusEnum, type AdditionalSystemType as AdditionalSystemTypeEnum, validateProblemDecomposition, validateTeamSelection, validateGauntletPipelineResult, validateICRPattern, validateCrossSystemWorkflowResult, } from './adaptive-mdap-canonical';
export { VectorData, VectorMetadata, CollectionInfo, VectorUpsertRequest, VectorUpsertResponse, VectorSearchRequest, VectorSearchResponse, VectorSearchResult, VectorDeleteRequest, VectorDeleteResponse, CollectionCreateRequest, CollectionCreateResponse, VectorDBError, type VectorData as VectorDBVectorData, type CollectionInfo as VectorDBCollectionInfo, transformUpsertResponseToCanonical, transformCanonicalToUpsertRequest, transformSearchResponseToCanonical, transformCanonicalToSearchRequest, validateVectorUpsertRequest, validateVectorSearchRequest, validateVectorSearchResponse, validateCollectionInfo, isVectorSearchRequest, isVectorUpsertRequest, isCollectionInfo, VectorDBExamples, } from './vectordb-canonical';
export { CanonicalEntitySchema, CanonicalEntityEdgeSchema, CanonicalEpisodeSchema, CanonicalCommunitySchema, CanonicalSearchQuerySchema, CanonicalSearchResultSchema, AddEpisodeOperationSchema, AddEpisodeResultSchema, AddTripletOperationSchema, AddTripletResultSchema, GraphStatisticsSchema, EpisodeTypeEnum, TemporalFilterEnum, type CanonicalEntity, type CanonicalEntityEdge, type CanonicalEpisode, type CanonicalCommunity, type CanonicalSearchQuery, type CanonicalSearchResult, type AddEpisodeOperation, type AddEpisodeResult, type AddTripletOperation, type AddTripletResult, type GraphStatistics, type EpisodeType, type TemporalFilter, validateCanonical, } from './graphiti-canonical';
export { AlgorithmCategory, NodeEmbeddingAlgorithm, CommunityAlgorithm, GraphEmbeddingAlgorithm, GraphStructure, NodeEmbeddingRequest, NodeEmbeddingResponse, CommunityDetectionRequest, CommunityDetectionResponse, GraphEmbeddingRequest, GraphEmbeddingResponse, GraphAnalysisRequest, GraphAnalysisResponse, type AlgorithmCategory as KarateClubAlgorithmCategory, type NodeEmbeddingRequest, type NodeEmbeddingResponse, type CommunityDetectionRequest, type CommunityDetectionResponse, type GraphEmbeddingRequest, type GraphEmbeddingResponse, type GraphAnalysisRequest, type GraphAnalysisResponse, validateNodeEmbeddingRequest, validateCommunityDetectionRequest, validateGraphEmbeddingRequest, validateGraphAnalysisRequest, } from './karateclub-canonical';
export { RESEPhase, ConstraintCategory, LogicalFallacy, TacitAssumption, ContradictionDetection, FalsificationResult, EpistemicAuditResult, FunctionalDependencyGraph, CrossDomainPattern, InvertedConstraint, IsomorphicMapping, SearchTreeNode, Hypothesis, ValidationMetrics, MCTSSearchResult, ParadigmShift, SynthesizedKnowledge, ArchitectureAssembly, type RESEPhase as RESEPhaseEnum, type ConstraintCategory as ConstraintCategoryEnum, type LogicalFallacy as LogicalFallacyEnum, type TacitAssumption as RESETacitAssumption, type ContradictionDetection as RESEContradictionDetection, type FalsificationResult as RESEFalsificationResult, type EpistemicAuditResult as RESEEpistemicAuditResult, type FunctionalDependencyGraph as RESEFunctionalDependencyGraph, type CrossDomainPattern as RESECrossDomainPattern, type InvertedConstraint as RESEInvertedConstraint, type IsomorphicMapping as RESEIsomorphicMapping, type SearchTreeNode as RESESearchTreeNode, type Hypothesis as RESEHypothesis, type ValidationMetrics as RESEValidationMetrics, type MCTSSearchResult as RESEMCTSSearchResult, type ParadigmShift as RESEParadigmShift, type SynthesizedKnowledge as RESESynthesizedKnowledge, type ArchitectureAssembly as RESEArchitectureAssembly, transformEpistemicAuditToCanonical, transformIsomorphicMappingToCanonical, transformMCTSSearchToCanonical, transformArchitectureAssemblyToCanonical, validateEpistemicAuditResult, validateIsomorphicMapping, validateMCTSSearchResult, validateArchitectureAssembly, createUTCTimestamp as createRESEUTCTimestamp, createCorrelationId as createRESECorrelationId, RESEExamples, } from './rese-canonical';
export { Problem, ProblemType, ExecutionStep, ExecutionStepType, ExecutionPlan, ExecutionState, ExecutionResult, ExecutionMetrics, LogEntry, Artifact, Summary, PerformanceAssessment, type Problem as PESProblem, type ProblemType as PESProblemTypeEnum, type ExecutionStep as PESExecutionStep, type ExecutionStepType as PESExecutionStepTypeEnum, type ExecutionPlan as PESExecutionPlan, type ExecutionState as PESExecutionStateEnum, type ExecutionResult as PESExecutionResult, type ExecutionMetrics as PESExecutionMetrics, type LogEntry as PESLogEntry, type Artifact as PESArtifact, type Summary as PESSummary, type PerformanceAssessment as PESPerformanceAssessment, transformProblemToCanonical, transformCanonicalToProblem, transformExecutionResultToCanonical, transformCanonicalToSummary, validateProblem, validateExecutionPlan, validateExecutionResult, validateSummary, createPESUTCTimestamp, createPESCorrelationId, isProblem, isExecutionResult, isSummary, } from './pes-canonical';
export { LoongFlowSolution, LoongFlowState, LoongFlowWorkerType, LLMConfig, WorkerConfig, EvolutionConfig as LoongFlowEvolutionConfig, LoongFlowConfig, LoongFlowRequest, LoongFlowResponse, LoongFlowCheckpoint, type LoongFlowSolution as LoongFlowSolutionType, type LoongFlowState as LoongFlowStateEnum, type LoongFlowWorkerType as LoongFlowWorkerTypeEnum, type LLMConfig as LoongFlowLLMConfigType, type WorkerConfig as LoongFlowWorkerConfigType, type EvolutionConfig as LoongFlowEvolutionConfigType, type LoongFlowConfig as LoongFlowConfigType, type LoongFlowRequest as LoongFlowRequestType, type LoongFlowResponse as LoongFlowResponseType, type LoongFlowCheckpoint as LoongFlowCheckpointType, transformLoongFlowSolutionToCanonical, transformCanonicalToLoongFlowSolution, transformLoongFlowResponseToExecutionResult, transformCanonicalProblemToLoongFlowRequest, validateLoongFlowSolution, validateLoongFlowConfig, validateLoongFlowRequest, validateLoongFlowResponse, isLoongFlowSolution, isLoongFlowRequest, isLoongFlowResponse, } from './loongflow-canonical';
export { HybridTask, HybridTaskType, IntegrationStrategy, AdaptiveTriggerCondition, AdaptiveAction, KnowledgeSourceType, KnowledgeType, EvolutionConfig, EvolutionaryKnowledge, PopulationIndividual, EvolutionResult, IntegrationMetrics, HybridExecutionResult, AdaptiveTrigger, KnowledgeTransfer, type HybridTask as HybridTaskType, type HybridTaskType as HybridTaskTypeEnum, type IntegrationStrategy as IntegrationStrategyEnum, type AdaptiveTriggerCondition as AdaptiveTriggerConditionEnum, type AdaptiveAction as AdaptiveActionEnum, type KnowledgeSourceType as KnowledgeSourceTypeEnum, type KnowledgeType as KnowledgeTypeEnum, type EvolutionConfig as EvolutionConfigType, type EvolutionaryKnowledge as EvolutionaryKnowledgeType, type PopulationIndividual as PopulationIndividualType, type EvolutionResult as EvolutionResultType, type IntegrationMetrics as IntegrationMetricsType, type HybridExecutionResult as HybridExecutionResultType, type AdaptiveTrigger as AdaptiveTriggerType, type KnowledgeTransfer as KnowledgeTransferType, transformLoongFlowSolutionToKnowledge, transformHybridResultToSummary, validateHybridTask, validateEvolutionaryKnowledge, validateHybridExecutionResult, validateAdaptiveTrigger, isHybridTask, isHybridExecutionResult, isEvolutionaryKnowledge, isAdaptiveTrigger, } from './hybrid-pes-evolution-canonical';
/**
 * Schema Registry
 *
 * Central registry of all available canonical schemas.
 * Useful for introspection and documentation generation.
 */
export declare const SchemaRegistry: {
    readonly z3: {
        readonly name: "z3";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly SolverRequest: "SolverRequest";
            readonly SolverResponse: "SolverResponse";
            readonly KnowledgeGraphResponse: "KnowledgeGraphResponse";
            readonly Entity: "Entity";
            readonly Relation: "Relation";
        };
    };
    readonly leanaide: {
        readonly name: "leanaide";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly ProofVerificationRequest: "ProofVerificationRequest";
            readonly ProofVerificationResponse: "ProofVerificationResponse";
            readonly LeanCompilationRequest: "LeanCompilationRequest";
            readonly LeanCompilationResponse: "LeanCompilationResponse";
            readonly LeanMessage: "LeanMessage";
        };
    };
    readonly ragbits: {
        readonly name: "ragbits";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly RAGRequest: "RAGRequest";
            readonly RAGResponse: "RAGResponse";
            readonly DocumentChunk: "DocumentChunk";
            readonly DocumentIngestionRequest: "DocumentIngestionRequest";
            readonly DocumentIngestionResponse: "DocumentIngestionResponse";
            readonly RAGError: "RAGError";
        };
    };
    readonly bubblelab: {
        readonly name: "bubblelab";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly BubbleRequest: "BubbleRequest";
            readonly BubbleResponse: "BubbleResponse";
            readonly WorkflowRequest: "WorkflowRequest";
            readonly WorkflowResponse: "WorkflowResponse";
            readonly BubbleStatusRequest: "BubbleStatusRequest";
            readonly BubbleStatusResponse: "BubbleStatusResponse";
            readonly BubbleLabError: "BubbleLabError";
        };
    };
    readonly vectordb: {
        readonly name: "vectordb";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly VectorData: "VectorData";
            readonly VectorMetadata: "VectorMetadata";
            readonly CollectionInfo: "CollectionInfo";
            readonly VectorUpsertRequest: "VectorUpsertRequest";
            readonly VectorUpsertResponse: "VectorUpsertResponse";
            readonly VectorSearchRequest: "VectorSearchRequest";
            readonly VectorSearchResponse: "VectorSearchResponse";
            readonly VectorSearchResult: "VectorSearchResult";
            readonly VectorDeleteRequest: "VectorDeleteRequest";
            readonly VectorDeleteResponse: "VectorDeleteResponse";
            readonly CollectionCreateRequest: "CollectionCreateRequest";
            readonly CollectionCreateResponse: "CollectionCreateResponse";
            readonly VectorDBError: "VectorDBError";
        };
    };
    readonly graphiti: {
        readonly name: "graphiti";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly CanonicalEntity: "CanonicalEntity";
            readonly CanonicalEntityEdge: "CanonicalEntityEdge";
            readonly CanonicalEpisode: "CanonicalEpisode";
            readonly CanonicalCommunity: "CanonicalCommunity";
            readonly CanonicalSearchQuery: "CanonicalSearchQuery";
            readonly CanonicalSearchResult: "CanonicalSearchResult";
            readonly AddEpisodeOperation: "AddEpisodeOperation";
            readonly AddEpisodeResult: "AddEpisodeResult";
            readonly AddTripletOperation: "AddTripletOperation";
            readonly AddTripletResult: "AddTripletResult";
            readonly GraphStatistics: "GraphStatistics";
        };
    };
    readonly karateclub: {
        readonly name: "karateclub";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly GraphStructure: "GraphStructure";
            readonly NodeEmbeddingRequest: "NodeEmbeddingRequest";
            readonly NodeEmbeddingResponse: "NodeEmbeddingResponse";
            readonly CommunityDetectionRequest: "CommunityDetectionRequest";
            readonly CommunityDetectionResponse: "CommunityDetectionResponse";
            readonly GraphEmbeddingRequest: "GraphEmbeddingRequest";
            readonly GraphEmbeddingResponse: "GraphEmbeddingResponse";
            readonly GraphAnalysisRequest: "GraphAnalysisRequest";
            readonly GraphAnalysisResponse: "GraphAnalysisResponse";
        };
    };
    readonly rese: {
        readonly name: "rese";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly RESEPhase: "RESEPhase";
            readonly ConstraintCategory: "ConstraintCategory";
            readonly LogicalFallacy: "LogicalFallacy";
            readonly TacitAssumption: "TacitAssumption";
            readonly ContradictionDetection: "ContradictionDetection";
            readonly FalsificationResult: "FalsificationResult";
            readonly EpistemicAuditResult: "EpistemicAuditResult";
            readonly FunctionalDependencyGraph: "FunctionalDependencyGraph";
            readonly CrossDomainPattern: "CrossDomainPattern";
            readonly InvertedConstraint: "InvertedConstraint";
            readonly IsomorphicMapping: "IsomorphicMapping";
            readonly SearchTreeNode: "SearchTreeNode";
            readonly Hypothesis: "Hypothesis";
            readonly ValidationMetrics: "ValidationMetrics";
            readonly MCTSSearchResult: "MCTSSearchResult";
            readonly ParadigmShift: "ParadigmShift";
            readonly SynthesizedKnowledge: "SynthesizedKnowledge";
            readonly ArchitectureAssembly: "ArchitectureAssembly";
        };
    };
    readonly pes: {
        readonly name: "pes";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly Problem: "Problem";
            readonly ProblemType: "ProblemType";
            readonly ExecutionStep: "ExecutionStep";
            readonly ExecutionStepType: "ExecutionStepType";
            readonly ExecutionPlan: "ExecutionPlan";
            readonly ExecutionState: "ExecutionState";
            readonly ExecutionResult: "ExecutionResult";
            readonly ExecutionMetrics: "ExecutionMetrics";
            readonly LogEntry: "LogEntry";
            readonly Artifact: "Artifact";
            readonly Summary: "Summary";
            readonly PerformanceAssessment: "PerformanceAssessment";
        };
    };
    readonly loongflow: {
        readonly name: "loongflow";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly LoongFlowSolution: "LoongFlowSolution";
            readonly LoongFlowState: "LoongFlowState";
            readonly LoongFlowWorkerType: "LoongFlowWorkerType";
            readonly LLMConfig: "LLMConfig";
            readonly WorkerConfig: "WorkerConfig";
            readonly EvolutionConfig: "EvolutionConfig";
            readonly LoongFlowConfig: "LoongFlowConfig";
            readonly LoongFlowRequest: "LoongFlowRequest";
            readonly LoongFlowResponse: "LoongFlowResponse";
            readonly LoongFlowCheckpoint: "LoongFlowCheckpoint";
        };
    };
    readonly hybrid: {
        readonly name: "hybrid-pes-evolution";
        readonly version: "1.0.0";
        readonly schemas: {
            readonly HybridTask: "HybridTask";
            readonly HybridTaskType: "HybridTaskType";
            readonly IntegrationStrategy: "IntegrationStrategy";
            readonly AdaptiveTriggerCondition: "AdaptiveTriggerCondition";
            readonly AdaptiveAction: "AdaptiveAction";
            readonly KnowledgeSourceType: "KnowledgeSourceType";
            readonly KnowledgeType: "KnowledgeType";
            readonly EvolutionConfig: "EvolutionConfig";
            readonly EvolutionaryKnowledge: "EvolutionaryKnowledge";
            readonly PopulationIndividual: "PopulationIndividual";
            readonly EvolutionResult: "EvolutionResult";
            readonly IntegrationMetrics: "IntegrationMetrics";
            readonly HybridExecutionResult: "HybridExecutionResult";
            readonly AdaptiveTrigger: "AdaptiveTrigger";
            readonly KnowledgeTransfer: "KnowledgeTransfer";
        };
    };
};
/**
 * Validation Utilities
 *
 * Common validation helpers for all schemas.
 */
/**
 * Validate any data against a schema
 * This is a type-safe wrapper that works with any of our canonical schemas.
 */
export declare function validateSchema<T>(schema: {
    safeParse: (data: unknown) => {
        success: boolean;
        data?: T;
        error?: any;
    };
}, data: unknown): {
    success: boolean;
    data?: T;
    errors?: string[];
};
/**
 * Type Guards
 *
 * Runtime type checking utilities.
 */
/**
 * Check if data is a valid Z3 SolverRequest
 */
export declare function isZ3SolverRequest(data: unknown): data is import('./z3-canonical').SolverRequest;
/**
 * Check if data is a valid LeanAide ProofVerificationRequest
 */
export declare function isLeanAideProofVerificationRequest(data: unknown): data is import('./leanaide-canonical').ProofVerificationRequest;
/**
 * Check if data is a valid RAGBits RAGRequest
 */
export declare function isRAGBitsRequest(data: unknown): data is import('./ragbits-canonical').RAGRequest;
/**
 * Check if data is a valid BubbleLab BubbleRequest
 */
export declare function isBubbleLabRequest(data: unknown): data is import('./bubblelab-canonical').BubbleRequest;
/**
 * Check if data is a valid VectorDB VectorSearchRequest
 */
export declare function isVectorDBSearchRequest(data: unknown): data is import('./vectordb-canonical').VectorSearchRequest;
/**
 * Check if data is a valid Graphiti Episode
 */
export declare function isGraphitiEpisode(data: unknown): data is import('./graphiti-canonical').CanonicalEpisode;
/**
 * Check if data is a valid KarateClub NodeEmbeddingRequest
 */
export declare function isKarateClubNodeEmbeddingRequest(data: unknown): data is import('./karateclub-canonical').NodeEmbeddingRequest;
/**
 * Check if data is a valid RESE EpistemicAuditResult
 */
export declare function isRESEEpistemicAuditResult(data: unknown): data is import('./rese-canonical').EpistemicAuditResult;
/**
 * Check if data is a valid PES Problem
 */
export declare function isPESProblem(data: unknown): data is import('./pes-canonical').Problem;
/**
 * Check if data is a valid LoongFlow Solution
 */
export declare function isLoongFlowSolution(data: unknown): data is import('./loongflow-canonical').LoongFlowSolution;
/**
 * Check if data is a valid Hybrid Task
 */
export declare function isHybridTask(data: unknown): data is import('./hybrid-pes-evolution-canonical').HybridTask;
/**
 * Constants for Common Use Cases
 */
/**
 * Default timeout values (in milliseconds)
 * These are the recommended defaults based on experience.
 */
export declare const DEFAULT_TIMEOUTS: {
    readonly QUICK: 5000;
    readonly NORMAL: 15000;
    readonly LONG: 60000;
    readonly EXTENDED: 300000;
};
/**
 * Maximum sizes for various fields
 * Prevents memory issues and abuse.
 */
export declare const MAX_SIZES: {
    readonly PROBLEM_LENGTH: 100000;
    readonly PROOF_CODE_LENGTH: 500000;
    readonly IMPORTS_COUNT: 100;
    readonly TACTICS_COUNT: 1000;
    readonly ENTITIES_COUNT: 10000;
    readonly RELATIONS_COUNT: 50000;
    readonly MESSAGES_COUNT: 1000;
    readonly RAG_QUERY_LENGTH: 10000;
    readonly DOCUMENT_CHUNKS: 1000;
    readonly RETRIEVAL_COUNT: 100;
    readonly BUBBLE_NAME_LENGTH: 255;
    readonly WORKFLOW_STEPS: 100;
    readonly DEPENDENCY_CHAIN: 50;
    readonly VECTOR_DIMENSION: 10000;
    readonly VECTORS_PER_UPSERT: 1000;
    readonly SEARCH_TOP_K: 100;
    readonly EPISODE_CONTENT_LENGTH: 100000;
    readonly ENTITY_ATTRIBUTES: 100;
    readonly COMMUNITY_SIZE: 10000;
    readonly GRAPH_NODES: 1000000;
    readonly GRAPH_EDGES: 10000000;
    readonly EMBEDDING_DIMENSION: 1024;
    readonly TACIT_ASSUMPTIONS: 1000;
    readonly CONTRADICTIONS: 500;
    readonly HYPOTHESES: 10000;
    readonly SEARCH_TREE_NODES: 100000;
    readonly CROSS_DOMAIN_PATTERNS: 500;
    readonly PARADIGM_SHIFTS: 50;
    readonly SYNTHESIZED_KNOWLEDGE: 1000;
    readonly PROBLEM_DESCRIPTION_LENGTH: 100000;
    readonly EXECUTION_STEPS: 1000;
    readonly PLAN_DEPENDENCY_DEPTH: 50;
    readonly LOG_ENTRIES: 10000;
    readonly ARTIFACTS: 100;
    readonly INSIGHTS: 100;
    readonly RECOMMENDATIONS: 50;
    readonly LOONGFLOW_ISLANDS: 100;
    readonly LOONGFLOW_ITERATIONS: 10000;
    readonly LOONGFLOW_SAMPLE_SIZE: 10000;
    readonly LOONGFLOW_SOLUTIONS: 100000;
    readonly LOONGFLOW_CHECKPOINTS: 1000;
    readonly HYBRID_ADAPTIVE_TRIGGERS: 50;
    readonly EVOLUTIONARY_KNOWLEDGE: 10000;
    readonly EVOLUTION_GENERATIONS: 100000;
    readonly POPULATION_SIZE: 100000;
    readonly KNOWLEDGE_TRANSFERS: 1000;
};
/**
 * Error codes for common validation failures
 * Use these for consistent error messaging across adapters.
 */
export declare const VALIDATION_ERRORS: {
    readonly MISSING_FIELD: "MISSING_FIELD";
    readonly INVALID_TYPE: "INVALID_TYPE";
    readonly OUT_OF_RANGE: "OUT_OF_RANGE";
    readonly TOO_LONG: "TOO_LONG";
    readonly TOO_SHORT: "TOO_SHORT";
    readonly INVALID_FORMAT: "INVALID_FORMAT";
    readonly TIMEOUT_EXCEEDED: "TIMEOUT_EXCEEDED";
    readonly SIZE_LIMIT_EXCEEDED: "SIZE_LIMIT_EXCEEDED";
};
/**
 * Utility function to create a correlation ID
 * Uses UUID v4 format.
 */
export declare function createCorrelationId(): string;
/**
 * Utility function to create a UTC timestamp
 * Follows the Law of UTC.
 */
export declare function createUTCTimestamp(): string;
/**
 * Utility function to format validation errors
 * Converts Zod errors to a human-readable format.
 */
export declare function formatValidationErrors(errors: any[]): string;
/**
 * Re-export types for convenience
 * These can be used directly in adapter code.
 */
export type { SolverRequest as Z3SolverRequestType, SolverResponse as Z3SolverResponseType, KnowledgeGraphResponse as Z3KnowledgeGraphResponseType, Entity as Z3EntityType, Relation as Z3RelationType, ProofVerificationRequest as LeanAideProofVerificationRequestType, ProofVerificationResponse as LeanAideProofVerificationResponseType, LeanCompilationRequest as LeanAideCompilationRequestType, LeanCompilationResponse as LeanAideCompilationResponseType, LeanMessage as LeanMessageType, RAGRequest as RAGBitsRequestType, RAGResponse as RAGBitsResponseType, DocumentChunk as RAGBitsDocumentChunkType, DocumentIngestionRequest as RAGBitsIngestionRequestType, DocumentIngestionResponse as RAGBitsIngestionResponseType, RAGError as RAGBitsErrorType, BubbleRequest as BubbleLabRequestType, BubbleResponse as BubbleLabResponseType, WorkflowRequest as BubbleLabWorkflowRequestType, WorkflowResponse as BubbleLabWorkflowResponseType, BubbleStatusRequest as BubbleLabStatusRequestType, BubbleStatusResponse as BubbleLabStatusResponseType, BubbleLabError as BubbleLabErrorType, VectorData as VectorDBVectorDataType, CollectionInfo as VectorDBCollectionInfoType, VectorUpsertRequest as VectorDBUpsertRequestType, VectorUpsertResponse as VectorDBUpsertResponseType, VectorSearchRequest as VectorDBSearchRequestType, VectorSearchResponse as VectorDBSearchResponseType, VectorDeleteRequest as VectorDBDeleteRequestType, VectorDeleteResponse as VectorDBDeleteResponseType, CollectionCreateRequest as VectorDBCollectionCreateRequestType, CollectionCreateResponse as VectorDBCollectionCreateResponseType, VectorDBError as VectorDBErrorType, CanonicalEntity as GraphitiEntityType, CanonicalEntityEdge as GraphitiEntityEdgeType, CanonicalEpisode as GraphitiEpisodeType, CanonicalCommunity as GraphitiCommunityType, CanonicalSearchQuery as GraphitiSearchQueryType, CanonicalSearchResult as GraphitiSearchResultType, AddEpisodeOperation as GraphitiAddEpisodeOperationType, AddEpisodeResult as GraphitiAddEpisodeResultType, AddTripletOperation as GraphitiAddTripletOperationType, AddTripletResult as GraphitiAddTripletResultType, GraphStatistics as GraphitiStatisticsType, GraphStructure as KarateClubGraphStructureType, NodeEmbeddingRequest as KarateClubNodeEmbeddingRequestType, NodeEmbeddingResponse as KarateClubNodeEmbeddingResponseType, CommunityDetectionRequest as KarateClubCommunityDetectionRequestType, CommunityDetectionResponse as KarateClubCommunityDetectionResponseType, GraphEmbeddingRequest as KarateClubGraphEmbeddingRequestType, GraphEmbeddingResponse as KarateClubGraphEmbeddingResponseType, GraphAnalysisRequest as KarateClubGraphAnalysisRequestType, GraphAnalysisResponse as KarateClubGraphAnalysisResponseType, RESEPhase as RESEPhaseType, ConstraintCategory as RESEConstraintCategoryType, LogicalFallacy as RESELogicalFallacyType, TacitAssumption as RESETacitAssumptionType, ContradictionDetection as RESEContradictionDetectionType, FalsificationResult as RESEFalsificationResultType, EpistemicAuditResult as RESEEpistemicAuditResultType, FunctionalDependencyGraph as RESEFunctionalDependencyGraphType, CrossDomainPattern as RESECrossDomainPatternType, InvertedConstraint as RESEInvertedConstraintType, IsomorphicMapping as RESEIsomorphicMappingType, SearchTreeNode as RESESearchTreeNodeType, Hypothesis as RESEHypothesisType, ValidationMetrics as RESEValidationMetricsType, MCTSSearchResult as RESEMCTSSearchResultType, ParadigmShift as RESEParadigmShiftType, SynthesizedKnowledge as RESESynthesizedKnowledgeType, ArchitectureAssembly as RESEArchitectureAssemblyType, } from './index';
/**
 * Documentation Examples
 *
 * This section demonstrates common usage patterns.
 */
/**
 * Example 1: Validating a Z3 Request
 * @example
 * ```typescript
 * import { validateSolverRequest } from './glue/schemas';
 *
 * const requestData = {
 *   problem: "(declare-const x Int) (assert (> x 10))",
 *   timeout_ms: 5000,
 * };
 *
 * const validation = validateSolverRequest(requestData);
 * if (!validation.success) {
 *   console.error('Invalid request:', validation.errors);
 *   return;
 * }
 *
 * // Use validation.data safely
 * const request = validation.data;
 * ```
 */
/**
 * Example 2: Creating a RAGBits Request with Proper Fields
 * @example
 * ```typescript
 * import {
 *   RAGRequest,
 *   createCorrelationId,
 *   DEFAULT_TIMEOUTS
 * } from './glue/schemas';
 *
 * const request: RAGRequest = {
 *   query: "What are the key principles of machine learning?",
 *   retrieval_count: 5,
 *   timeout_ms: DEFAULT_TIMEOUTS.NORMAL,
 *   correlation_id: createCorrelationId(),
 * };
 * ```
 */
/**
 * Example 3: Creating a BubbleLab Workflow Request
 * @example
 * ```typescript
 * import {
 *   WorkflowRequest,
 *   createCorrelationId,
 *   createUTCTimestamp,
 *   DEFAULT_TIMEOUTS
 * } from './glue/schemas';
 *
 * const request: WorkflowRequest = {
 *   workflow_id: "workflow_xyz789",
 *   workspace_id: "workspace_abc123",
 *   parameters: {
 *     input_data: "/data/input.csv",
 *   },
 *   config: {
 *     timeout_ms: DEFAULT_TIMEOUTS.LONG,
 *     stop_on_error: false,
 *   },
 *   correlation_id: createCorrelationId(),
 * };
 * ```
 */
/**
 * Example 4: Creating a VectorDB Search Request
 * @example
 * ```typescript
 * import {
 *   VectorSearchRequest,
 *   createCorrelationId,
 *   DEFAULT_TIMEOUTS
 * } from './glue/schemas';
 *
 * const request: VectorSearchRequest = {
 *   collection_name: "documents",
 *   query_vector: [0.1, 0.2, 0.3, 0.4, 0.5],
 *   top_k: 10,
 *   include_metadata: true,
 *   timeout_ms: DEFAULT_TIMEOUTS.QUICK,
 *   correlation_id: createCorrelationId(),
 * };
 * ```
 */
/**
 * Example 5: Creating a Graphiti Episode
 * @example
 * ```typescript
 * import {
 *   AddEpisodeOperation,
 *   createUTCTimestamp
 * } from './glue/schemas';
 *
 * const episode: AddEpisodeOperation = {
 *   name: "User Login Event",
 *   content: "User john_doe logged into the system from IP 192.168.1.1",
 *   source_description: "Authentication service",
 *   episode_type: "event",
 *   valid_at: createUTCTimestamp(),
 *   update_communities: true,
 * };
 * ```
 */
/**
 * Example 6: Creating a KarateClub Node Embedding Request
 * @example
 * ```typescript
 * import {
 *   NodeEmbeddingRequest,
 *   createCorrelationId,
 *   DEFAULT_TIMEOUTS
 * } from './glue/schemas';
 *
 * const request: NodeEmbeddingRequest = {
 *   algorithm: "node2vec",
 *   graph: {
 *     nodes: [
 *       { id: "node1", features: [1.0, 2.0] },
 *       { id: "node2", features: [3.0, 4.0] },
 *     ],
 *     edges: [
 *       { source: "node1", target: "node2" },
 *     ],
 *     directed: false,
 *   },
 *   parameters: {
 *     dimensions: 128,
 *     walk_length: 80,
 *     walk_number: 10,
 *   },
 *   timeout_ms: DEFAULT_TIMEOUTS.EXTENDED,
 *   correlation_id: createCorrelationId(),
 * };
 * ```
 */
/**
 * Example 3: Transforming External Format to Canonical
 * @example
 * ```typescript
 * import { transformZ3ResponseToCanonical } from './glue/schemas';
 *
 * const rawZ3Response = {
 *   result: 'SAT',
 *   time: 45,
 *   version: '4.12.1',
 * };
 *
 * const canonical = transformZ3ResponseToCanonical(
 *   rawZ3Response,
 *   '550e8400-e29b-41d4-a716-446655440000'
 * );
 *
 * // canonical now conforms to SolverResponse schema
 * console.log(canonical.result); // 'sat'
 * ```
 */
//# sourceMappingURL=index.d.ts.map