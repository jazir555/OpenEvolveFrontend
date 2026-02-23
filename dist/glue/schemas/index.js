"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.BubbleLabError = exports.BubbleStatus = exports.BubbleType = exports.BubbleStatusResponse = exports.BubbleStatusRequest = exports.WorkflowResponse = exports.WorkflowRequest = exports.BubbleResponse = exports.BubbleRequest = exports.RAGExamples = exports.isRAGResponse = exports.isRAGRequest = exports.validateDocumentChunk = exports.validateRAGResponse = exports.validateRAGRequest = exports.transformCanonicalToRAGRequest = exports.transformRAGResponseToCanonical = exports.RAGError = exports.DocumentIngestionResponse = exports.DocumentIngestionRequest = exports.DocumentChunk = exports.RAGResponse = exports.RAGRequest = exports.LeanAideExamples = exports.validateLeanCompilationResponse = exports.validateLeanCompilationRequest = exports.validateProofVerificationResponse = exports.validateProofVerificationRequest = exports.transformCompilationResponseToCanonical = exports.transformCanonicalToLeanAideRequest = exports.transformLeanAideResponseToCanonical = exports.LeanSeverity = exports.LeanTactic = exports.LeanMessage = exports.LeanCompilationResponse = exports.LeanCompilationRequest = exports.ProofVerificationResponse = exports.ProofVerificationRequest = exports.Z3Examples = exports.validateKnowledgeGraphResponse = exports.validateSolverResponse = exports.validateSolverRequest = exports.transformCanonicalToZ3Request = exports.transformZ3ResponseToCanonical = exports.Z3ResultType = exports.Relation = exports.Entity = exports.KnowledgeGraphResponse = exports.SolverResponse = exports.SolverRequest = void 0;
exports.BatchOperation = exports.AsyncOperation = exports.AsyncOperationStatus = exports.PerformanceMetrics = exports.CacheStatistics = exports.AdapterHealthStatus = exports.WorkflowTimeline = exports.UIChartData = exports.ChartType = exports.ICRPatternInsights = exports.PatternCluster = exports.ICRPrediction = exports.ICRPattern = exports.ICRPatternType = exports.GauntletPipelineResult = exports.GauntletPipeline = exports.GauntletResult = exports.GauntletConfig = exports.GauntletSeverity = exports.GauntletType = exports.ResourceOptimizationResult = exports.TeamSelectionResult = exports.TeamMember = exports.ProblemDecompositionResult = exports.SubProblem = exports.ComplexityScore = exports.ComplexityDimensions = exports.WorkflowType = exports.AdaptiveMdapExamples = exports.isAdaptiveMdapRequest = exports.validateAdaptiveMdapResponse = exports.validateAdaptiveMdapRequest = exports.AdaptationMode = exports.ProcessingDomain = exports.AdaptiveMdapError = exports.AdaptiveMdapBatchResponse = exports.AdaptiveMdapBatchRequest = exports.AdaptiveMdapResponse = exports.AdaptiveMdapRequest = exports.BubbleLabExamples = exports.isWorkflowRequest = exports.isBubbleRequest = exports.validateWorkflowResponse = exports.validateWorkflowRequest = exports.validateBubbleResponse = exports.validateBubbleRequest = exports.transformCanonicalToWorkflowRequest = exports.transformWorkflowResponseToCanonical = exports.transformCanonicalToBubbleRequest = exports.transformBubbleResponseToCanonical = void 0;
exports.AlgorithmCategory = exports.validateCanonical = exports.TemporalFilterEnum = exports.EpisodeTypeEnum = exports.GraphStatisticsSchema = exports.AddTripletResultSchema = exports.AddTripletOperationSchema = exports.AddEpisodeResultSchema = exports.AddEpisodeOperationSchema = exports.CanonicalSearchResultSchema = exports.CanonicalSearchQuerySchema = exports.CanonicalCommunitySchema = exports.CanonicalEpisodeSchema = exports.CanonicalEntityEdgeSchema = exports.CanonicalEntitySchema = exports.VectorDBExamples = exports.isCollectionInfo = exports.isVectorUpsertRequest = exports.isVectorSearchRequest = exports.validateCollectionInfo = exports.validateVectorSearchResponse = exports.validateVectorSearchRequest = exports.validateVectorUpsertRequest = exports.transformCanonicalToSearchRequest = exports.transformSearchResponseToCanonical = exports.transformCanonicalToUpsertRequest = exports.transformUpsertResponseToCanonical = exports.VectorDBError = exports.CollectionCreateResponse = exports.CollectionCreateRequest = exports.VectorDeleteResponse = exports.VectorDeleteRequest = exports.VectorSearchResult = exports.VectorSearchResponse = exports.VectorSearchRequest = exports.VectorUpsertResponse = exports.VectorUpsertRequest = exports.CollectionInfo = exports.VectorMetadata = exports.VectorData = exports.validateCrossSystemWorkflowResult = exports.validateICRPattern = exports.validateGauntletPipelineResult = exports.validateTeamSelection = exports.validateProblemDecomposition = exports.CrossSystemWorkflowResult = exports.WorkflowStep = exports.UnifiedSystemHealth = exports.SystemHealth = exports.AdditionalSystemType = void 0;
exports.ExecutionPlan = exports.ExecutionStepType = exports.ExecutionStep = exports.ProblemType = exports.Problem = exports.RESEExamples = exports.createRESECorrelationId = exports.createRESEUTCTimestamp = exports.validateArchitectureAssembly = exports.validateMCTSSearchResult = exports.validateIsomorphicMapping = exports.validateEpistemicAuditResult = exports.transformArchitectureAssemblyToCanonical = exports.transformMCTSSearchToCanonical = exports.transformIsomorphicMappingToCanonical = exports.transformEpistemicAuditToCanonical = exports.ArchitectureAssembly = exports.SynthesizedKnowledge = exports.ParadigmShift = exports.MCTSSearchResult = exports.ValidationMetrics = exports.Hypothesis = exports.SearchTreeNode = exports.IsomorphicMapping = exports.InvertedConstraint = exports.CrossDomainPattern = exports.FunctionalDependencyGraph = exports.EpistemicAuditResult = exports.FalsificationResult = exports.ContradictionDetection = exports.TacitAssumption = exports.LogicalFallacy = exports.ConstraintCategory = exports.RESEPhase = exports.validateGraphAnalysisRequest = exports.validateGraphEmbeddingRequest = exports.validateCommunityDetectionRequest = exports.validateNodeEmbeddingRequest = exports.GraphAnalysisResponse = exports.GraphAnalysisRequest = exports.GraphEmbeddingResponse = exports.GraphEmbeddingRequest = exports.CommunityDetectionResponse = exports.CommunityDetectionRequest = exports.NodeEmbeddingResponse = exports.NodeEmbeddingRequest = exports.GraphStructure = exports.GraphEmbeddingAlgorithm = exports.CommunityAlgorithm = exports.NodeEmbeddingAlgorithm = void 0;
exports.PopulationIndividual = exports.EvolutionaryKnowledge = exports.EvolutionConfig = exports.KnowledgeType = exports.KnowledgeSourceType = exports.AdaptiveAction = exports.AdaptiveTriggerCondition = exports.IntegrationStrategy = exports.HybridTaskType = exports.HybridTask = exports.isLoongFlowResponse = exports.isLoongFlowRequest = exports.validateLoongFlowResponse = exports.validateLoongFlowRequest = exports.validateLoongFlowConfig = exports.validateLoongFlowSolution = exports.transformCanonicalProblemToLoongFlowRequest = exports.transformLoongFlowResponseToExecutionResult = exports.transformCanonicalToLoongFlowSolution = exports.transformLoongFlowSolutionToCanonical = exports.LoongFlowCheckpoint = exports.LoongFlowResponse = exports.LoongFlowRequest = exports.LoongFlowConfig = exports.LoongFlowEvolutionConfig = exports.WorkerConfig = exports.LLMConfig = exports.LoongFlowWorkerType = exports.LoongFlowState = exports.LoongFlowSolution = exports.isSummary = exports.isExecutionResult = exports.isProblem = exports.createPESCorrelationId = exports.createPESUTCTimestamp = exports.validateSummary = exports.validateExecutionResult = exports.validateExecutionPlan = exports.validateProblem = exports.transformCanonicalToSummary = exports.transformExecutionResultToCanonical = exports.transformCanonicalToProblem = exports.transformProblemToCanonical = exports.PerformanceAssessment = exports.Summary = exports.Artifact = exports.LogEntry = exports.ExecutionMetrics = exports.ExecutionResult = exports.ExecutionState = void 0;
exports.VALIDATION_ERRORS = exports.MAX_SIZES = exports.DEFAULT_TIMEOUTS = exports.SchemaRegistry = exports.isAdaptiveTrigger = exports.isEvolutionaryKnowledge = exports.isHybridExecutionResult = exports.validateAdaptiveTrigger = exports.validateHybridExecutionResult = exports.validateEvolutionaryKnowledge = exports.validateHybridTask = exports.transformHybridResultToSummary = exports.transformLoongFlowSolutionToKnowledge = exports.KnowledgeTransfer = exports.AdaptiveTrigger = exports.HybridExecutionResult = exports.IntegrationMetrics = exports.EvolutionResult = void 0;
exports.isLoongFlowSolution = isLoongFlowSolution;
exports.isHybridTask = isHybridTask;
exports.validateSchema = validateSchema;
exports.isZ3SolverRequest = isZ3SolverRequest;
exports.isLeanAideProofVerificationRequest = isLeanAideProofVerificationRequest;
exports.isRAGBitsRequest = isRAGBitsRequest;
exports.isBubbleLabRequest = isBubbleLabRequest;
exports.isVectorDBSearchRequest = isVectorDBSearchRequest;
exports.isGraphitiEpisode = isGraphitiEpisode;
exports.isKarateClubNodeEmbeddingRequest = isKarateClubNodeEmbeddingRequest;
exports.isRESEEpistemicAuditResult = isRESEEpistemicAuditResult;
exports.isPESProblem = isPESProblem;
exports.createCorrelationId = createCorrelationId;
exports.createUTCTimestamp = createUTCTimestamp;
exports.formatValidationErrors = formatValidationErrors;
// Export Z3 schemas
var z3_canonical_1 = require("./z3-canonical");
Object.defineProperty(exports, "SolverRequest", { enumerable: true, get: function () { return z3_canonical_1.SolverRequest; } });
Object.defineProperty(exports, "SolverResponse", { enumerable: true, get: function () { return z3_canonical_1.SolverResponse; } });
Object.defineProperty(exports, "KnowledgeGraphResponse", { enumerable: true, get: function () { return z3_canonical_1.KnowledgeGraphResponse; } });
Object.defineProperty(exports, "Entity", { enumerable: true, get: function () { return z3_canonical_1.Entity; } });
Object.defineProperty(exports, "Relation", { enumerable: true, get: function () { return z3_canonical_1.Relation; } });
Object.defineProperty(exports, "Z3ResultType", { enumerable: true, get: function () { return z3_canonical_1.Z3ResultType; } });
Object.defineProperty(exports, "transformZ3ResponseToCanonical", { enumerable: true, get: function () { return z3_canonical_1.transformZ3ResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToZ3Request", { enumerable: true, get: function () { return z3_canonical_1.transformCanonicalToZ3Request; } });
Object.defineProperty(exports, "validateSolverRequest", { enumerable: true, get: function () { return z3_canonical_1.validateSolverRequest; } });
Object.defineProperty(exports, "validateSolverResponse", { enumerable: true, get: function () { return z3_canonical_1.validateSolverResponse; } });
Object.defineProperty(exports, "validateKnowledgeGraphResponse", { enumerable: true, get: function () { return z3_canonical_1.validateKnowledgeGraphResponse; } });
Object.defineProperty(exports, "Z3Examples", { enumerable: true, get: function () { return z3_canonical_1.Z3Examples; } });
// Export LeanAide schemas
var leanaide_canonical_1 = require("./leanaide-canonical");
Object.defineProperty(exports, "ProofVerificationRequest", { enumerable: true, get: function () { return leanaide_canonical_1.ProofVerificationRequest; } });
Object.defineProperty(exports, "ProofVerificationResponse", { enumerable: true, get: function () { return leanaide_canonical_1.ProofVerificationResponse; } });
Object.defineProperty(exports, "LeanCompilationRequest", { enumerable: true, get: function () { return leanaide_canonical_1.LeanCompilationRequest; } });
Object.defineProperty(exports, "LeanCompilationResponse", { enumerable: true, get: function () { return leanaide_canonical_1.LeanCompilationResponse; } });
Object.defineProperty(exports, "LeanMessage", { enumerable: true, get: function () { return leanaide_canonical_1.LeanMessage; } });
Object.defineProperty(exports, "LeanTactic", { enumerable: true, get: function () { return leanaide_canonical_1.LeanTactic; } });
Object.defineProperty(exports, "LeanSeverity", { enumerable: true, get: function () { return leanaide_canonical_1.LeanSeverity; } });
Object.defineProperty(exports, "transformLeanAideResponseToCanonical", { enumerable: true, get: function () { return leanaide_canonical_1.transformLeanAideResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToLeanAideRequest", { enumerable: true, get: function () { return leanaide_canonical_1.transformCanonicalToLeanAideRequest; } });
Object.defineProperty(exports, "transformCompilationResponseToCanonical", { enumerable: true, get: function () { return leanaide_canonical_1.transformCompilationResponseToCanonical; } });
Object.defineProperty(exports, "validateProofVerificationRequest", { enumerable: true, get: function () { return leanaide_canonical_1.validateProofVerificationRequest; } });
Object.defineProperty(exports, "validateProofVerificationResponse", { enumerable: true, get: function () { return leanaide_canonical_1.validateProofVerificationResponse; } });
Object.defineProperty(exports, "validateLeanCompilationRequest", { enumerable: true, get: function () { return leanaide_canonical_1.validateLeanCompilationRequest; } });
Object.defineProperty(exports, "validateLeanCompilationResponse", { enumerable: true, get: function () { return leanaide_canonical_1.validateLeanCompilationResponse; } });
Object.defineProperty(exports, "LeanAideExamples", { enumerable: true, get: function () { return leanaide_canonical_1.LeanAideExamples; } });
// Export RAGbits schemas
var ragbits_canonical_1 = require("./ragbits-canonical");
Object.defineProperty(exports, "RAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.RAGRequest; } });
Object.defineProperty(exports, "RAGResponse", { enumerable: true, get: function () { return ragbits_canonical_1.RAGResponse; } });
Object.defineProperty(exports, "DocumentChunk", { enumerable: true, get: function () { return ragbits_canonical_1.DocumentChunk; } });
Object.defineProperty(exports, "DocumentIngestionRequest", { enumerable: true, get: function () { return ragbits_canonical_1.DocumentIngestionRequest; } });
Object.defineProperty(exports, "DocumentIngestionResponse", { enumerable: true, get: function () { return ragbits_canonical_1.DocumentIngestionResponse; } });
Object.defineProperty(exports, "RAGError", { enumerable: true, get: function () { return ragbits_canonical_1.RAGError; } });
Object.defineProperty(exports, "transformRAGResponseToCanonical", { enumerable: true, get: function () { return ragbits_canonical_1.transformRAGResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToRAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.transformCanonicalToRAGRequest; } });
Object.defineProperty(exports, "validateRAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.validateRAGRequest; } });
Object.defineProperty(exports, "validateRAGResponse", { enumerable: true, get: function () { return ragbits_canonical_1.validateRAGResponse; } });
Object.defineProperty(exports, "validateDocumentChunk", { enumerable: true, get: function () { return ragbits_canonical_1.validateDocumentChunk; } });
Object.defineProperty(exports, "isRAGRequest", { enumerable: true, get: function () { return ragbits_canonical_1.isRAGRequest; } });
Object.defineProperty(exports, "isRAGResponse", { enumerable: true, get: function () { return ragbits_canonical_1.isRAGResponse; } });
Object.defineProperty(exports, "RAGExamples", { enumerable: true, get: function () { return ragbits_canonical_1.RAGExamples; } });
// Export BubbleLab schemas
var bubblelab_canonical_1 = require("./bubblelab-canonical");
Object.defineProperty(exports, "BubbleRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleRequest; } });
Object.defineProperty(exports, "BubbleResponse", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleResponse; } });
Object.defineProperty(exports, "WorkflowRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.WorkflowRequest; } });
Object.defineProperty(exports, "WorkflowResponse", { enumerable: true, get: function () { return bubblelab_canonical_1.WorkflowResponse; } });
Object.defineProperty(exports, "BubbleStatusRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleStatusRequest; } });
Object.defineProperty(exports, "BubbleStatusResponse", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleStatusResponse; } });
Object.defineProperty(exports, "BubbleType", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleType; } });
Object.defineProperty(exports, "BubbleStatus", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleStatus; } });
Object.defineProperty(exports, "BubbleLabError", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleLabError; } });
Object.defineProperty(exports, "transformBubbleResponseToCanonical", { enumerable: true, get: function () { return bubblelab_canonical_1.transformBubbleResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToBubbleRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.transformCanonicalToBubbleRequest; } });
Object.defineProperty(exports, "transformWorkflowResponseToCanonical", { enumerable: true, get: function () { return bubblelab_canonical_1.transformWorkflowResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToWorkflowRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.transformCanonicalToWorkflowRequest; } });
Object.defineProperty(exports, "validateBubbleRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.validateBubbleRequest; } });
Object.defineProperty(exports, "validateBubbleResponse", { enumerable: true, get: function () { return bubblelab_canonical_1.validateBubbleResponse; } });
Object.defineProperty(exports, "validateWorkflowRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.validateWorkflowRequest; } });
Object.defineProperty(exports, "validateWorkflowResponse", { enumerable: true, get: function () { return bubblelab_canonical_1.validateWorkflowResponse; } });
Object.defineProperty(exports, "isBubbleRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.isBubbleRequest; } });
Object.defineProperty(exports, "isWorkflowRequest", { enumerable: true, get: function () { return bubblelab_canonical_1.isWorkflowRequest; } });
Object.defineProperty(exports, "BubbleLabExamples", { enumerable: true, get: function () { return bubblelab_canonical_1.BubbleLabExamples; } });
// Export Adaptive MDAP schemas (V2.0)
var adaptive_mdap_canonical_1 = require("./adaptive-mdap-canonical");
Object.defineProperty(exports, "AdaptiveMdapRequest", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptiveMdapRequest; } });
Object.defineProperty(exports, "AdaptiveMdapResponse", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptiveMdapResponse; } });
Object.defineProperty(exports, "AdaptiveMdapBatchRequest", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptiveMdapBatchRequest; } });
Object.defineProperty(exports, "AdaptiveMdapBatchResponse", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptiveMdapBatchResponse; } });
Object.defineProperty(exports, "AdaptiveMdapError", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptiveMdapError; } });
Object.defineProperty(exports, "ProcessingDomain", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.ProcessingDomain; } });
Object.defineProperty(exports, "AdaptationMode", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptationMode; } });
Object.defineProperty(exports, "validateAdaptiveMdapRequest", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.validateAdaptiveMdapRequest; } });
Object.defineProperty(exports, "validateAdaptiveMdapResponse", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.validateAdaptiveMdapResponse; } });
Object.defineProperty(exports, "isAdaptiveMdapRequest", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.isAdaptiveMdapRequest; } });
Object.defineProperty(exports, "AdaptiveMdapExamples", { enumerable: true, get: function () { return adaptive_mdap_canonical_1.AdaptiveMdapExamples; } });
// Export Adaptive MDAP V2.0 Advanced Features
var adaptive_mdap_canonical_2 = require("./adaptive-mdap-canonical");
Object.defineProperty(exports, "WorkflowType", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.WorkflowType; } });
Object.defineProperty(exports, "ComplexityDimensions", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ComplexityDimensions; } });
Object.defineProperty(exports, "ComplexityScore", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ComplexityScore; } });
Object.defineProperty(exports, "SubProblem", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.SubProblem; } });
Object.defineProperty(exports, "ProblemDecompositionResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ProblemDecompositionResult; } });
Object.defineProperty(exports, "TeamMember", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.TeamMember; } });
Object.defineProperty(exports, "TeamSelectionResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.TeamSelectionResult; } });
Object.defineProperty(exports, "ResourceOptimizationResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ResourceOptimizationResult; } });
Object.defineProperty(exports, "GauntletType", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.GauntletType; } });
Object.defineProperty(exports, "GauntletSeverity", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.GauntletSeverity; } });
Object.defineProperty(exports, "GauntletConfig", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.GauntletConfig; } });
Object.defineProperty(exports, "GauntletResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.GauntletResult; } });
Object.defineProperty(exports, "GauntletPipeline", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.GauntletPipeline; } });
Object.defineProperty(exports, "GauntletPipelineResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.GauntletPipelineResult; } });
Object.defineProperty(exports, "ICRPatternType", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ICRPatternType; } });
Object.defineProperty(exports, "ICRPattern", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ICRPattern; } });
Object.defineProperty(exports, "ICRPrediction", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ICRPrediction; } });
Object.defineProperty(exports, "PatternCluster", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.PatternCluster; } });
Object.defineProperty(exports, "ICRPatternInsights", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ICRPatternInsights; } });
Object.defineProperty(exports, "ChartType", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.ChartType; } });
Object.defineProperty(exports, "UIChartData", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.UIChartData; } });
Object.defineProperty(exports, "WorkflowTimeline", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.WorkflowTimeline; } });
Object.defineProperty(exports, "AdapterHealthStatus", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.AdapterHealthStatus; } });
Object.defineProperty(exports, "CacheStatistics", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.CacheStatistics; } });
Object.defineProperty(exports, "PerformanceMetrics", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.PerformanceMetrics; } });
Object.defineProperty(exports, "AsyncOperationStatus", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.AsyncOperationStatus; } });
Object.defineProperty(exports, "AsyncOperation", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.AsyncOperation; } });
Object.defineProperty(exports, "BatchOperation", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.BatchOperation; } });
Object.defineProperty(exports, "AdditionalSystemType", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.AdditionalSystemType; } });
Object.defineProperty(exports, "SystemHealth", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.SystemHealth; } });
Object.defineProperty(exports, "UnifiedSystemHealth", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.UnifiedSystemHealth; } });
Object.defineProperty(exports, "WorkflowStep", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.WorkflowStep; } });
Object.defineProperty(exports, "CrossSystemWorkflowResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.CrossSystemWorkflowResult; } });
Object.defineProperty(exports, "validateProblemDecomposition", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.validateProblemDecomposition; } });
Object.defineProperty(exports, "validateTeamSelection", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.validateTeamSelection; } });
Object.defineProperty(exports, "validateGauntletPipelineResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.validateGauntletPipelineResult; } });
Object.defineProperty(exports, "validateICRPattern", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.validateICRPattern; } });
Object.defineProperty(exports, "validateCrossSystemWorkflowResult", { enumerable: true, get: function () { return adaptive_mdap_canonical_2.validateCrossSystemWorkflowResult; } });
// Export VectorDB schemas
var vectordb_canonical_1 = require("./vectordb-canonical");
Object.defineProperty(exports, "VectorData", { enumerable: true, get: function () { return vectordb_canonical_1.VectorData; } });
Object.defineProperty(exports, "VectorMetadata", { enumerable: true, get: function () { return vectordb_canonical_1.VectorMetadata; } });
Object.defineProperty(exports, "CollectionInfo", { enumerable: true, get: function () { return vectordb_canonical_1.CollectionInfo; } });
Object.defineProperty(exports, "VectorUpsertRequest", { enumerable: true, get: function () { return vectordb_canonical_1.VectorUpsertRequest; } });
Object.defineProperty(exports, "VectorUpsertResponse", { enumerable: true, get: function () { return vectordb_canonical_1.VectorUpsertResponse; } });
Object.defineProperty(exports, "VectorSearchRequest", { enumerable: true, get: function () { return vectordb_canonical_1.VectorSearchRequest; } });
Object.defineProperty(exports, "VectorSearchResponse", { enumerable: true, get: function () { return vectordb_canonical_1.VectorSearchResponse; } });
Object.defineProperty(exports, "VectorSearchResult", { enumerable: true, get: function () { return vectordb_canonical_1.VectorSearchResult; } });
Object.defineProperty(exports, "VectorDeleteRequest", { enumerable: true, get: function () { return vectordb_canonical_1.VectorDeleteRequest; } });
Object.defineProperty(exports, "VectorDeleteResponse", { enumerable: true, get: function () { return vectordb_canonical_1.VectorDeleteResponse; } });
Object.defineProperty(exports, "CollectionCreateRequest", { enumerable: true, get: function () { return vectordb_canonical_1.CollectionCreateRequest; } });
Object.defineProperty(exports, "CollectionCreateResponse", { enumerable: true, get: function () { return vectordb_canonical_1.CollectionCreateResponse; } });
Object.defineProperty(exports, "VectorDBError", { enumerable: true, get: function () { return vectordb_canonical_1.VectorDBError; } });
Object.defineProperty(exports, "transformUpsertResponseToCanonical", { enumerable: true, get: function () { return vectordb_canonical_1.transformUpsertResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToUpsertRequest", { enumerable: true, get: function () { return vectordb_canonical_1.transformCanonicalToUpsertRequest; } });
Object.defineProperty(exports, "transformSearchResponseToCanonical", { enumerable: true, get: function () { return vectordb_canonical_1.transformSearchResponseToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToSearchRequest", { enumerable: true, get: function () { return vectordb_canonical_1.transformCanonicalToSearchRequest; } });
Object.defineProperty(exports, "validateVectorUpsertRequest", { enumerable: true, get: function () { return vectordb_canonical_1.validateVectorUpsertRequest; } });
Object.defineProperty(exports, "validateVectorSearchRequest", { enumerable: true, get: function () { return vectordb_canonical_1.validateVectorSearchRequest; } });
Object.defineProperty(exports, "validateVectorSearchResponse", { enumerable: true, get: function () { return vectordb_canonical_1.validateVectorSearchResponse; } });
Object.defineProperty(exports, "validateCollectionInfo", { enumerable: true, get: function () { return vectordb_canonical_1.validateCollectionInfo; } });
Object.defineProperty(exports, "isVectorSearchRequest", { enumerable: true, get: function () { return vectordb_canonical_1.isVectorSearchRequest; } });
Object.defineProperty(exports, "isVectorUpsertRequest", { enumerable: true, get: function () { return vectordb_canonical_1.isVectorUpsertRequest; } });
Object.defineProperty(exports, "isCollectionInfo", { enumerable: true, get: function () { return vectordb_canonical_1.isCollectionInfo; } });
Object.defineProperty(exports, "VectorDBExamples", { enumerable: true, get: function () { return vectordb_canonical_1.VectorDBExamples; } });
// Export Graphiti schemas
var graphiti_canonical_1 = require("./graphiti-canonical");
Object.defineProperty(exports, "CanonicalEntitySchema", { enumerable: true, get: function () { return graphiti_canonical_1.CanonicalEntitySchema; } });
Object.defineProperty(exports, "CanonicalEntityEdgeSchema", { enumerable: true, get: function () { return graphiti_canonical_1.CanonicalEntityEdgeSchema; } });
Object.defineProperty(exports, "CanonicalEpisodeSchema", { enumerable: true, get: function () { return graphiti_canonical_1.CanonicalEpisodeSchema; } });
Object.defineProperty(exports, "CanonicalCommunitySchema", { enumerable: true, get: function () { return graphiti_canonical_1.CanonicalCommunitySchema; } });
Object.defineProperty(exports, "CanonicalSearchQuerySchema", { enumerable: true, get: function () { return graphiti_canonical_1.CanonicalSearchQuerySchema; } });
Object.defineProperty(exports, "CanonicalSearchResultSchema", { enumerable: true, get: function () { return graphiti_canonical_1.CanonicalSearchResultSchema; } });
Object.defineProperty(exports, "AddEpisodeOperationSchema", { enumerable: true, get: function () { return graphiti_canonical_1.AddEpisodeOperationSchema; } });
Object.defineProperty(exports, "AddEpisodeResultSchema", { enumerable: true, get: function () { return graphiti_canonical_1.AddEpisodeResultSchema; } });
Object.defineProperty(exports, "AddTripletOperationSchema", { enumerable: true, get: function () { return graphiti_canonical_1.AddTripletOperationSchema; } });
Object.defineProperty(exports, "AddTripletResultSchema", { enumerable: true, get: function () { return graphiti_canonical_1.AddTripletResultSchema; } });
Object.defineProperty(exports, "GraphStatisticsSchema", { enumerable: true, get: function () { return graphiti_canonical_1.GraphStatisticsSchema; } });
Object.defineProperty(exports, "EpisodeTypeEnum", { enumerable: true, get: function () { return graphiti_canonical_1.EpisodeTypeEnum; } });
Object.defineProperty(exports, "TemporalFilterEnum", { enumerable: true, get: function () { return graphiti_canonical_1.TemporalFilterEnum; } });
Object.defineProperty(exports, "validateCanonical", { enumerable: true, get: function () { return graphiti_canonical_1.validateCanonical; } });
// Export KarateClub schemas
var karateclub_canonical_1 = require("./karateclub-canonical");
Object.defineProperty(exports, "AlgorithmCategory", { enumerable: true, get: function () { return karateclub_canonical_1.AlgorithmCategory; } });
Object.defineProperty(exports, "NodeEmbeddingAlgorithm", { enumerable: true, get: function () { return karateclub_canonical_1.NodeEmbeddingAlgorithm; } });
Object.defineProperty(exports, "CommunityAlgorithm", { enumerable: true, get: function () { return karateclub_canonical_1.CommunityAlgorithm; } });
Object.defineProperty(exports, "GraphEmbeddingAlgorithm", { enumerable: true, get: function () { return karateclub_canonical_1.GraphEmbeddingAlgorithm; } });
Object.defineProperty(exports, "GraphStructure", { enumerable: true, get: function () { return karateclub_canonical_1.GraphStructure; } });
Object.defineProperty(exports, "NodeEmbeddingRequest", { enumerable: true, get: function () { return karateclub_canonical_1.NodeEmbeddingRequest; } });
Object.defineProperty(exports, "NodeEmbeddingResponse", { enumerable: true, get: function () { return karateclub_canonical_1.NodeEmbeddingResponse; } });
Object.defineProperty(exports, "CommunityDetectionRequest", { enumerable: true, get: function () { return karateclub_canonical_1.CommunityDetectionRequest; } });
Object.defineProperty(exports, "CommunityDetectionResponse", { enumerable: true, get: function () { return karateclub_canonical_1.CommunityDetectionResponse; } });
Object.defineProperty(exports, "GraphEmbeddingRequest", { enumerable: true, get: function () { return karateclub_canonical_1.GraphEmbeddingRequest; } });
Object.defineProperty(exports, "GraphEmbeddingResponse", { enumerable: true, get: function () { return karateclub_canonical_1.GraphEmbeddingResponse; } });
Object.defineProperty(exports, "GraphAnalysisRequest", { enumerable: true, get: function () { return karateclub_canonical_1.GraphAnalysisRequest; } });
Object.defineProperty(exports, "GraphAnalysisResponse", { enumerable: true, get: function () { return karateclub_canonical_1.GraphAnalysisResponse; } });
Object.defineProperty(exports, "validateNodeEmbeddingRequest", { enumerable: true, get: function () { return karateclub_canonical_1.validateNodeEmbeddingRequest; } });
Object.defineProperty(exports, "validateCommunityDetectionRequest", { enumerable: true, get: function () { return karateclub_canonical_1.validateCommunityDetectionRequest; } });
Object.defineProperty(exports, "validateGraphEmbeddingRequest", { enumerable: true, get: function () { return karateclub_canonical_1.validateGraphEmbeddingRequest; } });
Object.defineProperty(exports, "validateGraphAnalysisRequest", { enumerable: true, get: function () { return karateclub_canonical_1.validateGraphAnalysisRequest; } });
// Export RESE schemas
var rese_canonical_1 = require("./rese-canonical");
Object.defineProperty(exports, "RESEPhase", { enumerable: true, get: function () { return rese_canonical_1.RESEPhase; } });
Object.defineProperty(exports, "ConstraintCategory", { enumerable: true, get: function () { return rese_canonical_1.ConstraintCategory; } });
Object.defineProperty(exports, "LogicalFallacy", { enumerable: true, get: function () { return rese_canonical_1.LogicalFallacy; } });
Object.defineProperty(exports, "TacitAssumption", { enumerable: true, get: function () { return rese_canonical_1.TacitAssumption; } });
Object.defineProperty(exports, "ContradictionDetection", { enumerable: true, get: function () { return rese_canonical_1.ContradictionDetection; } });
Object.defineProperty(exports, "FalsificationResult", { enumerable: true, get: function () { return rese_canonical_1.FalsificationResult; } });
Object.defineProperty(exports, "EpistemicAuditResult", { enumerable: true, get: function () { return rese_canonical_1.EpistemicAuditResult; } });
Object.defineProperty(exports, "FunctionalDependencyGraph", { enumerable: true, get: function () { return rese_canonical_1.FunctionalDependencyGraph; } });
Object.defineProperty(exports, "CrossDomainPattern", { enumerable: true, get: function () { return rese_canonical_1.CrossDomainPattern; } });
Object.defineProperty(exports, "InvertedConstraint", { enumerable: true, get: function () { return rese_canonical_1.InvertedConstraint; } });
Object.defineProperty(exports, "IsomorphicMapping", { enumerable: true, get: function () { return rese_canonical_1.IsomorphicMapping; } });
Object.defineProperty(exports, "SearchTreeNode", { enumerable: true, get: function () { return rese_canonical_1.SearchTreeNode; } });
Object.defineProperty(exports, "Hypothesis", { enumerable: true, get: function () { return rese_canonical_1.Hypothesis; } });
Object.defineProperty(exports, "ValidationMetrics", { enumerable: true, get: function () { return rese_canonical_1.ValidationMetrics; } });
Object.defineProperty(exports, "MCTSSearchResult", { enumerable: true, get: function () { return rese_canonical_1.MCTSSearchResult; } });
Object.defineProperty(exports, "ParadigmShift", { enumerable: true, get: function () { return rese_canonical_1.ParadigmShift; } });
Object.defineProperty(exports, "SynthesizedKnowledge", { enumerable: true, get: function () { return rese_canonical_1.SynthesizedKnowledge; } });
Object.defineProperty(exports, "ArchitectureAssembly", { enumerable: true, get: function () { return rese_canonical_1.ArchitectureAssembly; } });
Object.defineProperty(exports, "transformEpistemicAuditToCanonical", { enumerable: true, get: function () { return rese_canonical_1.transformEpistemicAuditToCanonical; } });
Object.defineProperty(exports, "transformIsomorphicMappingToCanonical", { enumerable: true, get: function () { return rese_canonical_1.transformIsomorphicMappingToCanonical; } });
Object.defineProperty(exports, "transformMCTSSearchToCanonical", { enumerable: true, get: function () { return rese_canonical_1.transformMCTSSearchToCanonical; } });
Object.defineProperty(exports, "transformArchitectureAssemblyToCanonical", { enumerable: true, get: function () { return rese_canonical_1.transformArchitectureAssemblyToCanonical; } });
Object.defineProperty(exports, "validateEpistemicAuditResult", { enumerable: true, get: function () { return rese_canonical_1.validateEpistemicAuditResult; } });
Object.defineProperty(exports, "validateIsomorphicMapping", { enumerable: true, get: function () { return rese_canonical_1.validateIsomorphicMapping; } });
Object.defineProperty(exports, "validateMCTSSearchResult", { enumerable: true, get: function () { return rese_canonical_1.validateMCTSSearchResult; } });
Object.defineProperty(exports, "validateArchitectureAssembly", { enumerable: true, get: function () { return rese_canonical_1.validateArchitectureAssembly; } });
Object.defineProperty(exports, "createRESEUTCTimestamp", { enumerable: true, get: function () { return rese_canonical_1.createUTCTimestamp; } });
Object.defineProperty(exports, "createRESECorrelationId", { enumerable: true, get: function () { return rese_canonical_1.createCorrelationId; } });
Object.defineProperty(exports, "RESEExamples", { enumerable: true, get: function () { return rese_canonical_1.RESEExamples; } });
// Export PES schemas (Plan-Execute-Summarize pattern)
var pes_canonical_1 = require("./pes-canonical");
Object.defineProperty(exports, "Problem", { enumerable: true, get: function () { return pes_canonical_1.Problem; } });
Object.defineProperty(exports, "ProblemType", { enumerable: true, get: function () { return pes_canonical_1.ProblemType; } });
Object.defineProperty(exports, "ExecutionStep", { enumerable: true, get: function () { return pes_canonical_1.ExecutionStep; } });
Object.defineProperty(exports, "ExecutionStepType", { enumerable: true, get: function () { return pes_canonical_1.ExecutionStepType; } });
Object.defineProperty(exports, "ExecutionPlan", { enumerable: true, get: function () { return pes_canonical_1.ExecutionPlan; } });
Object.defineProperty(exports, "ExecutionState", { enumerable: true, get: function () { return pes_canonical_1.ExecutionState; } });
Object.defineProperty(exports, "ExecutionResult", { enumerable: true, get: function () { return pes_canonical_1.ExecutionResult; } });
Object.defineProperty(exports, "ExecutionMetrics", { enumerable: true, get: function () { return pes_canonical_1.ExecutionMetrics; } });
Object.defineProperty(exports, "LogEntry", { enumerable: true, get: function () { return pes_canonical_1.LogEntry; } });
Object.defineProperty(exports, "Artifact", { enumerable: true, get: function () { return pes_canonical_1.Artifact; } });
Object.defineProperty(exports, "Summary", { enumerable: true, get: function () { return pes_canonical_1.Summary; } });
Object.defineProperty(exports, "PerformanceAssessment", { enumerable: true, get: function () { return pes_canonical_1.PerformanceAssessment; } });
Object.defineProperty(exports, "transformProblemToCanonical", { enumerable: true, get: function () { return pes_canonical_1.transformProblemToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToProblem", { enumerable: true, get: function () { return pes_canonical_1.transformCanonicalToProblem; } });
Object.defineProperty(exports, "transformExecutionResultToCanonical", { enumerable: true, get: function () { return pes_canonical_1.transformExecutionResultToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToSummary", { enumerable: true, get: function () { return pes_canonical_1.transformCanonicalToSummary; } });
Object.defineProperty(exports, "validateProblem", { enumerable: true, get: function () { return pes_canonical_1.validateProblem; } });
Object.defineProperty(exports, "validateExecutionPlan", { enumerable: true, get: function () { return pes_canonical_1.validateExecutionPlan; } });
Object.defineProperty(exports, "validateExecutionResult", { enumerable: true, get: function () { return pes_canonical_1.validateExecutionResult; } });
Object.defineProperty(exports, "validateSummary", { enumerable: true, get: function () { return pes_canonical_1.validateSummary; } });
Object.defineProperty(exports, "createPESUTCTimestamp", { enumerable: true, get: function () { return pes_canonical_1.createPESUTCTimestamp; } });
Object.defineProperty(exports, "createPESCorrelationId", { enumerable: true, get: function () { return pes_canonical_1.createPESCorrelationId; } });
Object.defineProperty(exports, "isProblem", { enumerable: true, get: function () { return pes_canonical_1.isProblem; } });
Object.defineProperty(exports, "isExecutionResult", { enumerable: true, get: function () { return pes_canonical_1.isExecutionResult; } });
Object.defineProperty(exports, "isSummary", { enumerable: true, get: function () { return pes_canonical_1.isSummary; } });
// Export LoongFlow schemas (PES + Evolutionary Optimization)
var loongflow_canonical_1 = require("./loongflow-canonical");
Object.defineProperty(exports, "LoongFlowSolution", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowSolution; } });
Object.defineProperty(exports, "LoongFlowState", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowState; } });
Object.defineProperty(exports, "LoongFlowWorkerType", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowWorkerType; } });
Object.defineProperty(exports, "LLMConfig", { enumerable: true, get: function () { return loongflow_canonical_1.LLMConfig; } });
Object.defineProperty(exports, "WorkerConfig", { enumerable: true, get: function () { return loongflow_canonical_1.WorkerConfig; } });
Object.defineProperty(exports, "LoongFlowEvolutionConfig", { enumerable: true, get: function () { return loongflow_canonical_1.EvolutionConfig; } });
Object.defineProperty(exports, "LoongFlowConfig", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowConfig; } });
Object.defineProperty(exports, "LoongFlowRequest", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowRequest; } });
Object.defineProperty(exports, "LoongFlowResponse", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowResponse; } });
Object.defineProperty(exports, "LoongFlowCheckpoint", { enumerable: true, get: function () { return loongflow_canonical_1.LoongFlowCheckpoint; } });
Object.defineProperty(exports, "transformLoongFlowSolutionToCanonical", { enumerable: true, get: function () { return loongflow_canonical_1.transformLoongFlowSolutionToCanonical; } });
Object.defineProperty(exports, "transformCanonicalToLoongFlowSolution", { enumerable: true, get: function () { return loongflow_canonical_1.transformCanonicalToLoongFlowSolution; } });
Object.defineProperty(exports, "transformLoongFlowResponseToExecutionResult", { enumerable: true, get: function () { return loongflow_canonical_1.transformLoongFlowResponseToExecutionResult; } });
Object.defineProperty(exports, "transformCanonicalProblemToLoongFlowRequest", { enumerable: true, get: function () { return loongflow_canonical_1.transformCanonicalProblemToLoongFlowRequest; } });
Object.defineProperty(exports, "validateLoongFlowSolution", { enumerable: true, get: function () { return loongflow_canonical_1.validateLoongFlowSolution; } });
Object.defineProperty(exports, "validateLoongFlowConfig", { enumerable: true, get: function () { return loongflow_canonical_1.validateLoongFlowConfig; } });
Object.defineProperty(exports, "validateLoongFlowRequest", { enumerable: true, get: function () { return loongflow_canonical_1.validateLoongFlowRequest; } });
Object.defineProperty(exports, "validateLoongFlowResponse", { enumerable: true, get: function () { return loongflow_canonical_1.validateLoongFlowResponse; } });
Object.defineProperty(exports, "isLoongFlowSolution", { enumerable: true, get: function () { return loongflow_canonical_1.isLoongFlowSolution; } });
Object.defineProperty(exports, "isLoongFlowRequest", { enumerable: true, get: function () { return loongflow_canonical_1.isLoongFlowRequest; } });
Object.defineProperty(exports, "isLoongFlowResponse", { enumerable: true, get: function () { return loongflow_canonical_1.isLoongFlowResponse; } });
// Export Hybrid PES-Evolution schemas
var hybrid_pes_evolution_canonical_1 = require("./hybrid-pes-evolution-canonical");
Object.defineProperty(exports, "HybridTask", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.HybridTask; } });
Object.defineProperty(exports, "HybridTaskType", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.HybridTaskType; } });
Object.defineProperty(exports, "IntegrationStrategy", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.IntegrationStrategy; } });
Object.defineProperty(exports, "AdaptiveTriggerCondition", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.AdaptiveTriggerCondition; } });
Object.defineProperty(exports, "AdaptiveAction", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.AdaptiveAction; } });
Object.defineProperty(exports, "KnowledgeSourceType", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.KnowledgeSourceType; } });
Object.defineProperty(exports, "KnowledgeType", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.KnowledgeType; } });
Object.defineProperty(exports, "EvolutionConfig", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.EvolutionConfig; } });
Object.defineProperty(exports, "EvolutionaryKnowledge", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.EvolutionaryKnowledge; } });
Object.defineProperty(exports, "PopulationIndividual", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.PopulationIndividual; } });
Object.defineProperty(exports, "EvolutionResult", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.EvolutionResult; } });
Object.defineProperty(exports, "IntegrationMetrics", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.IntegrationMetrics; } });
Object.defineProperty(exports, "HybridExecutionResult", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.HybridExecutionResult; } });
Object.defineProperty(exports, "AdaptiveTrigger", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.AdaptiveTrigger; } });
Object.defineProperty(exports, "KnowledgeTransfer", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.KnowledgeTransfer; } });
Object.defineProperty(exports, "transformLoongFlowSolutionToKnowledge", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.transformLoongFlowSolutionToKnowledge; } });
Object.defineProperty(exports, "transformHybridResultToSummary", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.transformHybridResultToSummary; } });
Object.defineProperty(exports, "validateHybridTask", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.validateHybridTask; } });
Object.defineProperty(exports, "validateEvolutionaryKnowledge", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.validateEvolutionaryKnowledge; } });
Object.defineProperty(exports, "validateHybridExecutionResult", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.validateHybridExecutionResult; } });
Object.defineProperty(exports, "validateAdaptiveTrigger", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.validateAdaptiveTrigger; } });
Object.defineProperty(exports, "isHybridTask", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.isHybridTask; } });
Object.defineProperty(exports, "isHybridExecutionResult", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.isHybridExecutionResult; } });
Object.defineProperty(exports, "isEvolutionaryKnowledge", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.isEvolutionaryKnowledge; } });
Object.defineProperty(exports, "isAdaptiveTrigger", { enumerable: true, get: function () { return hybrid_pes_evolution_canonical_1.isAdaptiveTrigger; } });
/**
 * Schema Registry
 *
 * Central registry of all available canonical schemas.
 * Useful for introspection and documentation generation.
 */
exports.SchemaRegistry = {
    z3: {
        name: 'z3',
        version: '1.0.0',
        schemas: {
            SolverRequest: 'SolverRequest',
            SolverResponse: 'SolverResponse',
            KnowledgeGraphResponse: 'KnowledgeGraphResponse',
            Entity: 'Entity',
            Relation: 'Relation',
        },
    },
    leanaide: {
        name: 'leanaide',
        version: '1.0.0',
        schemas: {
            ProofVerificationRequest: 'ProofVerificationRequest',
            ProofVerificationResponse: 'ProofVerificationResponse',
            LeanCompilationRequest: 'LeanCompilationRequest',
            LeanCompilationResponse: 'LeanCompilationResponse',
            LeanMessage: 'LeanMessage',
        },
    },
    ragbits: {
        name: 'ragbits',
        version: '1.0.0',
        schemas: {
            RAGRequest: 'RAGRequest',
            RAGResponse: 'RAGResponse',
            DocumentChunk: 'DocumentChunk',
            DocumentIngestionRequest: 'DocumentIngestionRequest',
            DocumentIngestionResponse: 'DocumentIngestionResponse',
            RAGError: 'RAGError',
        },
    },
    bubblelab: {
        name: 'bubblelab',
        version: '1.0.0',
        schemas: {
            BubbleRequest: 'BubbleRequest',
            BubbleResponse: 'BubbleResponse',
            WorkflowRequest: 'WorkflowRequest',
            WorkflowResponse: 'WorkflowResponse',
            BubbleStatusRequest: 'BubbleStatusRequest',
            BubbleStatusResponse: 'BubbleStatusResponse',
            BubbleLabError: 'BubbleLabError',
        },
    },
    vectordb: {
        name: 'vectordb',
        version: '1.0.0',
        schemas: {
            VectorData: 'VectorData',
            VectorMetadata: 'VectorMetadata',
            CollectionInfo: 'CollectionInfo',
            VectorUpsertRequest: 'VectorUpsertRequest',
            VectorUpsertResponse: 'VectorUpsertResponse',
            VectorSearchRequest: 'VectorSearchRequest',
            VectorSearchResponse: 'VectorSearchResponse',
            VectorSearchResult: 'VectorSearchResult',
            VectorDeleteRequest: 'VectorDeleteRequest',
            VectorDeleteResponse: 'VectorDeleteResponse',
            CollectionCreateRequest: 'CollectionCreateRequest',
            CollectionCreateResponse: 'CollectionCreateResponse',
            VectorDBError: 'VectorDBError',
        },
    },
    graphiti: {
        name: 'graphiti',
        version: '1.0.0',
        schemas: {
            CanonicalEntity: 'CanonicalEntity',
            CanonicalEntityEdge: 'CanonicalEntityEdge',
            CanonicalEpisode: 'CanonicalEpisode',
            CanonicalCommunity: 'CanonicalCommunity',
            CanonicalSearchQuery: 'CanonicalSearchQuery',
            CanonicalSearchResult: 'CanonicalSearchResult',
            AddEpisodeOperation: 'AddEpisodeOperation',
            AddEpisodeResult: 'AddEpisodeResult',
            AddTripletOperation: 'AddTripletOperation',
            AddTripletResult: 'AddTripletResult',
            GraphStatistics: 'GraphStatistics',
        },
    },
    karateclub: {
        name: 'karateclub',
        version: '1.0.0',
        schemas: {
            GraphStructure: 'GraphStructure',
            NodeEmbeddingRequest: 'NodeEmbeddingRequest',
            NodeEmbeddingResponse: 'NodeEmbeddingResponse',
            CommunityDetectionRequest: 'CommunityDetectionRequest',
            CommunityDetectionResponse: 'CommunityDetectionResponse',
            GraphEmbeddingRequest: 'GraphEmbeddingRequest',
            GraphEmbeddingResponse: 'GraphEmbeddingResponse',
            GraphAnalysisRequest: 'GraphAnalysisRequest',
            GraphAnalysisResponse: 'GraphAnalysisResponse',
        },
    },
    rese: {
        name: 'rese',
        version: '1.0.0',
        schemas: {
            RESEPhase: 'RESEPhase',
            ConstraintCategory: 'ConstraintCategory',
            LogicalFallacy: 'LogicalFallacy',
            TacitAssumption: 'TacitAssumption',
            ContradictionDetection: 'ContradictionDetection',
            FalsificationResult: 'FalsificationResult',
            EpistemicAuditResult: 'EpistemicAuditResult',
            FunctionalDependencyGraph: 'FunctionalDependencyGraph',
            CrossDomainPattern: 'CrossDomainPattern',
            InvertedConstraint: 'InvertedConstraint',
            IsomorphicMapping: 'IsomorphicMapping',
            SearchTreeNode: 'SearchTreeNode',
            Hypothesis: 'Hypothesis',
            ValidationMetrics: 'ValidationMetrics',
            MCTSSearchResult: 'MCTSSearchResult',
            ParadigmShift: 'ParadigmShift',
            SynthesizedKnowledge: 'SynthesizedKnowledge',
            ArchitectureAssembly: 'ArchitectureAssembly',
        },
    },
    pes: {
        name: 'pes',
        version: '1.0.0',
        schemas: {
            Problem: 'Problem',
            ProblemType: 'ProblemType',
            ExecutionStep: 'ExecutionStep',
            ExecutionStepType: 'ExecutionStepType',
            ExecutionPlan: 'ExecutionPlan',
            ExecutionState: 'ExecutionState',
            ExecutionResult: 'ExecutionResult',
            ExecutionMetrics: 'ExecutionMetrics',
            LogEntry: 'LogEntry',
            Artifact: 'Artifact',
            Summary: 'Summary',
            PerformanceAssessment: 'PerformanceAssessment',
        },
    },
    loongflow: {
        name: 'loongflow',
        version: '1.0.0',
        schemas: {
            LoongFlowSolution: 'LoongFlowSolution',
            LoongFlowState: 'LoongFlowState',
            LoongFlowWorkerType: 'LoongFlowWorkerType',
            LLMConfig: 'LLMConfig',
            WorkerConfig: 'WorkerConfig',
            EvolutionConfig: 'EvolutionConfig',
            LoongFlowConfig: 'LoongFlowConfig',
            LoongFlowRequest: 'LoongFlowRequest',
            LoongFlowResponse: 'LoongFlowResponse',
            LoongFlowCheckpoint: 'LoongFlowCheckpoint',
        },
    },
    hybrid: {
        name: 'hybrid-pes-evolution',
        version: '1.0.0',
        schemas: {
            HybridTask: 'HybridTask',
            HybridTaskType: 'HybridTaskType',
            IntegrationStrategy: 'IntegrationStrategy',
            AdaptiveTriggerCondition: 'AdaptiveTriggerCondition',
            AdaptiveAction: 'AdaptiveAction',
            KnowledgeSourceType: 'KnowledgeSourceType',
            KnowledgeType: 'KnowledgeType',
            EvolutionConfig: 'EvolutionConfig',
            EvolutionaryKnowledge: 'EvolutionaryKnowledge',
            PopulationIndividual: 'PopulationIndividual',
            EvolutionResult: 'EvolutionResult',
            IntegrationMetrics: 'IntegrationMetrics',
            HybridExecutionResult: 'HybridExecutionResult',
            AdaptiveTrigger: 'AdaptiveTrigger',
            KnowledgeTransfer: 'KnowledgeTransfer',
        },
    },
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
function validateSchema(schema, data) {
    const result = schema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Type Guards
 *
 * Runtime type checking utilities.
 */
/**
 * Check if data is a valid Z3 SolverRequest
 */
function isZ3SolverRequest(data) {
    // This is a runtime check - in practice you'd use the schema
    return typeof data === 'object' && data !== null && 'problem' in data && 'timeout_ms' in data;
}
/**
 * Check if data is a valid LeanAide ProofVerificationRequest
 */
function isLeanAideProofVerificationRequest(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'proof_code' in data &&
        'theorem' in data &&
        'timeout_ms' in data);
}
/**
 * Check if data is a valid RAGBits RAGRequest
 */
function isRAGBitsRequest(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'query' in data &&
        'retrieval_count' in data &&
        'timeout_ms' in data);
}
/**
 * Check if data is a valid BubbleLab BubbleRequest
 */
function isBubbleLabRequest(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'workspace_id' in data &&
        'bubble_type' in data &&
        'name' in data);
}
/**
 * Check if data is a valid VectorDB VectorSearchRequest
 */
function isVectorDBSearchRequest(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'collection_name' in data &&
        'query_vector' in data &&
        'top_k' in data);
}
/**
 * Check if data is a valid Graphiti Episode
 */
function isGraphitiEpisode(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'id' in data &&
        'name' in data &&
        'content' in data &&
        'valid_at' in data);
}
/**
 * Check if data is a valid KarateClub NodeEmbeddingRequest
 */
function isKarateClubNodeEmbeddingRequest(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'algorithm' in data &&
        'graph' in data &&
        'timeout_ms' in data);
}
/**
 * Check if data is a valid RESE EpistemicAuditResult
 */
function isRESEEpistemicAuditResult(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'phase' in data &&
        'audit_id' in data &&
        'problem_description' in data &&
        'timestamp' in data);
}
/**
 * Check if data is a valid PES Problem
 */
function isPESProblem(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'id' in data &&
        'type' in data &&
        'description' in data &&
        'created_at' in data);
}
/**
 * Check if data is a valid LoongFlow Solution
 */
function isLoongFlowSolution(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'solution' in data &&
        'solution_id' in data &&
        'score' in data &&
        'island_id' in data);
}
/**
 * Check if data is a valid Hybrid Task
 */
function isHybridTask(data) {
    return (typeof data === 'object' &&
        data !== null &&
        'id' in data &&
        'type' in data &&
        'problem' in data &&
        'integration_strategy' in data);
}
/**
 * Constants for Common Use Cases
 */
/**
 * Default timeout values (in milliseconds)
 * These are the recommended defaults based on experience.
 */
exports.DEFAULT_TIMEOUTS = {
    QUICK: 5000, // 5 seconds - for simple queries
    NORMAL: 15000, // 15 seconds - for average complexity
    LONG: 60000, // 1 minute - for complex proofs
    EXTENDED: 300000, // 5 minutes - maximum allowed timeout
};
/**
 * Maximum sizes for various fields
 * Prevents memory issues and abuse.
 */
exports.MAX_SIZES = {
    PROBLEM_LENGTH: 100000, // 100KB for problem statements
    PROOF_CODE_LENGTH: 500000, // 500KB for proof code
    IMPORTS_COUNT: 100, // Maximum number of imports
    TACTICS_COUNT: 1000, // Maximum number of tactics
    ENTITIES_COUNT: 10000, // Maximum entities in knowledge graph
    RELATIONS_COUNT: 50000, // Maximum relations in knowledge graph
    MESSAGES_COUNT: 1000, // Maximum compiler messages
    // RAGBits limits
    RAG_QUERY_LENGTH: 10000, // Maximum query length
    DOCUMENT_CHUNKS: 1000, // Maximum chunks per ingestion
    RETRIEVAL_COUNT: 100, // Maximum retrieval count
    // BubbleLab limits
    BUBBLE_NAME_LENGTH: 255, // Maximum bubble name length
    WORKFLOW_STEPS: 100, // Maximum workflow steps
    DEPENDENCY_CHAIN: 50, // Maximum dependency depth
    // VectorDB limits
    VECTOR_DIMENSION: 10000, // Maximum vector dimension
    VECTORS_PER_UPSERT: 1000, // Maximum vectors per upsert
    SEARCH_TOP_K: 100, // Maximum search results
    // Graphiti limits
    EPISODE_CONTENT_LENGTH: 100000, // Maximum episode content
    ENTITY_ATTRIBUTES: 100, // Maximum entity attributes
    COMMUNITY_SIZE: 10000, // Maximum community members
    // KarateClub limits
    GRAPH_NODES: 1000000, // Maximum nodes in a graph
    GRAPH_EDGES: 10000000, // Maximum edges in a graph
    EMBEDDING_DIMENSION: 1024, // Maximum embedding dimension
    // RESE limits
    TACIT_ASSUMPTIONS: 1000, // Maximum tacit assumptions
    CONTRADICTIONS: 500, // Maximum contradictions
    HYPOTHESES: 10000, // Maximum hypotheses
    SEARCH_TREE_NODES: 100000, // Maximum search tree nodes
    CROSS_DOMAIN_PATTERNS: 500, // Maximum cross-domain patterns
    PARADIGM_SHIFTS: 50, // Maximum paradigm shifts
    SYNTHESIZED_KNOWLEDGE: 1000, // Maximum knowledge items
    // PES limits
    PROBLEM_DESCRIPTION_LENGTH: 100000, // Maximum problem description (100KB)
    EXECUTION_STEPS: 1000, // Maximum execution steps in a plan
    PLAN_DEPENDENCY_DEPTH: 50, // Maximum dependency chain depth
    LOG_ENTRIES: 10000, // Maximum log entries per result
    ARTIFACTS: 100, // Maximum artifacts per result
    INSIGHTS: 100, // Maximum insights per summary
    RECOMMENDATIONS: 50, // Maximum recommendations per summary
    // LoongFlow limits
    LOONGFLOW_ISLANDS: 100, // Maximum islands in island model
    LOONGFLOW_ITERATIONS: 10000, // Maximum iterations
    LOONGFLOW_SAMPLE_SIZE: 10000, // Maximum Boltzmann sample size
    LOONGFLOW_SOLUTIONS: 100000, // Maximum solutions in population
    LOONGFLOW_CHECKPOINTS: 1000, // Maximum checkpoints
    // Hybrid PES-Evolution limits
    HYBRID_ADAPTIVE_TRIGGERS: 50, // Maximum adaptive triggers
    EVOLUTIONARY_KNOWLEDGE: 10000, // Maximum knowledge items
    EVOLUTION_GENERATIONS: 100000, // Maximum evolution generations
    POPULATION_SIZE: 100000, // Maximum population size
    KNOWLEDGE_TRANSFERS: 1000, // Maximum knowledge transfers
};
/**
 * Error codes for common validation failures
 * Use these for consistent error messaging across adapters.
 */
exports.VALIDATION_ERRORS = {
    MISSING_FIELD: 'MISSING_FIELD',
    INVALID_TYPE: 'INVALID_TYPE',
    OUT_OF_RANGE: 'OUT_OF_RANGE',
    TOO_LONG: 'TOO_LONG',
    TOO_SHORT: 'TOO_SHORT',
    INVALID_FORMAT: 'INVALID_FORMAT',
    TIMEOUT_EXCEEDED: 'TIMEOUT_EXCEEDED',
    SIZE_LIMIT_EXCEEDED: 'SIZE_LIMIT_EXCEEDED',
};
/**
 * Utility function to create a correlation ID
 * Uses UUID v4 format.
 */
function createCorrelationId() {
    // Simple UUID v4 generator
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}
/**
 * Utility function to create a UTC timestamp
 * Follows the Law of UTC.
 */
function createUTCTimestamp() {
    return new Date().toISOString();
}
/**
 * Utility function to format validation errors
 * Converts Zod errors to a human-readable format.
 */
function formatValidationErrors(errors) {
    return errors
        .map((e) => {
        const path = e.path.length > 0 ? e.path.join('.') : 'root';
        return `${path}: ${e.message}`;
    })
        .join('\n');
}
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
//# sourceMappingURL=index.js.map