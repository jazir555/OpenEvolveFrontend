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

// Export Z3 schemas
export {
  SolverRequest,
  SolverResponse,
  KnowledgeGraphResponse,
  Entity,
  Relation,
  Z3ResultType,
  type Z3ResultType as Z3ResultTypeEnum,
  type SolverRequest as Z3SolverRequest,
  type SolverResponse as Z3SolverResponse,
  type KnowledgeGraphResponse as Z3KnowledgeGraphResponse,
  type Entity as Z3Entity,
  type Relation as Z3Relation,
  transformZ3ResponseToCanonical,
  transformCanonicalToZ3Request,
  validateSolverRequest,
  validateSolverResponse,
  validateKnowledgeGraphResponse,
  Z3Examples,
} from './z3-canonical';

// Export LeanAide schemas
export {
  ProofVerificationRequest,
  ProofVerificationResponse,
  LeanCompilationRequest,
  LeanCompilationResponse,
  LeanMessage,
  LeanTactic,
  LeanSeverity,
  type LeanTactic as LeanTacticEnum,
  type LeanSeverity as LeanSeverityEnum,
  type ProofVerificationRequest as LeanAideProofVerificationRequest,
  type ProofVerificationResponse as LeanAideProofVerificationResponse,
  type LeanCompilationRequest as LeanAideCompilationRequest,
  type LeanCompilationResponse as LeanAideCompilationResponse,
  type LeanMessage as LeanCompilerMessage,
  transformLeanAideResponseToCanonical,
  transformCanonicalToLeanAideRequest,
  transformCompilationResponseToCanonical,
  validateProofVerificationRequest,
  validateProofVerificationResponse,
  validateLeanCompilationRequest,
  validateLeanCompilationResponse,
  LeanAideExamples,
} from './leanaide-canonical';

// Export RAGbits schemas
export {
  RAGRequest,
  RAGResponse,
  DocumentChunk,
  DocumentIngestionRequest,
  DocumentIngestionResponse,
  RAGError,
  type RAGRequest as RAGBitsRequest,
  type RAGResponse as RAGBitsResponse,
  type DocumentChunk as RAGBitsDocumentChunk,
  type DocumentIngestionRequest as RAGBitsIngestionRequest,
  type DocumentIngestionResponse as RAGBitsIngestionResponse,
  transformRAGResponseToCanonical,
  transformCanonicalToRAGRequest,
  validateRAGRequest,
  validateRAGResponse,
  validateDocumentChunk,
  isRAGRequest,
  isRAGResponse,
  RAGExamples,
} from './ragbits-canonical';

// Export BubbleLab schemas
export {
  BubbleRequest,
  BubbleResponse,
  WorkflowRequest,
  WorkflowResponse,
  BubbleStatusRequest,
  BubbleStatusResponse,
  BubbleType,
  BubbleStatus,
  BubbleLabError,
  type BubbleType as BubbleTypeEnum,
  type BubbleStatus as BubbleStatusEnum,
  type BubbleRequest as BubbleLabRequest,
  type BubbleResponse as BubbleLabResponse,
  type WorkflowRequest as BubbleLabWorkflowRequest,
  type WorkflowResponse as BubbleLabWorkflowResponse,
  transformBubbleResponseToCanonical,
  transformCanonicalToBubbleRequest,
  transformWorkflowResponseToCanonical,
  transformCanonicalToWorkflowRequest,
  validateBubbleRequest,
  validateBubbleResponse,
  validateWorkflowRequest,
  validateWorkflowResponse,
  isBubbleRequest,
  isWorkflowRequest,
  BubbleLabExamples,
} from './bubblelab-canonical';

// Export VectorDB schemas
export {
  VectorData,
  VectorMetadata,
  CollectionInfo,
  VectorUpsertRequest,
  VectorUpsertResponse,
  VectorSearchRequest,
  VectorSearchResponse,
  VectorSearchResult,
  VectorDeleteRequest,
  VectorDeleteResponse,
  CollectionCreateRequest,
  CollectionCreateResponse,
  VectorDBError,
  type VectorData as VectorDBVectorData,
  type CollectionInfo as VectorDBCollectionInfo,
  transformUpsertResponseToCanonical,
  transformCanonicalToUpsertRequest,
  transformSearchResponseToCanonical,
  transformCanonicalToSearchRequest,
  validateVectorUpsertRequest,
  validateVectorSearchRequest,
  validateVectorSearchResponse,
  validateCollectionInfo,
  isVectorSearchRequest,
  isVectorUpsertRequest,
  isCollectionInfo,
  VectorDBExamples,
} from './vectordb-canonical';

// Export Graphiti schemas
export {
  CanonicalEntitySchema,
  CanonicalEntityEdgeSchema,
  CanonicalEpisodeSchema,
  CanonicalCommunitySchema,
  CanonicalSearchQuerySchema,
  CanonicalSearchResultSchema,
  AddEpisodeOperationSchema,
  AddEpisodeResultSchema,
  AddTripletOperationSchema,
  AddTripletResultSchema,
  GraphStatisticsSchema,
  EpisodeTypeEnum,
  TemporalFilterEnum,
  type CanonicalEntity,
  type CanonicalEntityEdge,
  type CanonicalEpisode,
  type CanonicalCommunity,
  type CanonicalSearchQuery,
  type CanonicalSearchResult,
  type AddEpisodeOperation,
  type AddEpisodeResult,
  type AddTripletOperation,
  type AddTripletResult,
  type GraphStatistics,
  type EpisodeType,
  type TemporalFilter,
  validateCanonical,
} from './graphiti-canonical';

// Export KarateClub schemas
export {
  AlgorithmCategory,
  NodeEmbeddingAlgorithm,
  CommunityAlgorithm,
  GraphEmbeddingAlgorithm,
  GraphStructure,
  NodeEmbeddingRequest,
  NodeEmbeddingResponse,
  CommunityDetectionRequest,
  CommunityDetectionResponse,
  GraphEmbeddingRequest,
  GraphEmbeddingResponse,
  GraphAnalysisRequest,
  GraphAnalysisResponse,
  type AlgorithmCategory as KarateClubAlgorithmCategory,
  type NodeEmbeddingAlgorithm,
  type CommunityAlgorithm,
  type GraphEmbeddingAlgorithm,
  type GraphStructure,
  type NodeEmbeddingRequest,
  type NodeEmbeddingResponse,
  type CommunityDetectionRequest,
  type CommunityDetectionResponse,
  type GraphEmbeddingRequest,
  type GraphEmbeddingResponse,
  type GraphAnalysisRequest,
  type GraphAnalysisResponse,
  validateNodeEmbeddingRequest,
  validateCommunityDetectionRequest,
  validateGraphEmbeddingRequest,
  validateGraphAnalysisRequest,
} from './karateclub-canonical';

// Export RESE schemas
export {
  RESEPhase,
  ConstraintCategory,
  LogicalFallacy,
  TacitAssumption,
  ContradictionDetection,
  FalsificationResult,
  EpistemicAuditResult,
  FunctionalDependencyGraph,
  CrossDomainPattern,
  InvertedConstraint,
  IsomorphicMapping,
  SearchTreeNode,
  Hypothesis,
  ValidationMetrics,
  MCTSSearchResult,
  ParadigmShift,
  SynthesizedKnowledge,
  ArchitectureAssembly,
  type RESEPhase as RESEPhaseEnum,
  type ConstraintCategory as ConstraintCategoryEnum,
  type LogicalFallacy as LogicalFallacyEnum,
  type TacitAssumption as RESETacitAssumption,
  type ContradictionDetection as RESEContradictionDetection,
  type FalsificationResult as RESEFalsificationResult,
  type EpistemicAuditResult as RESEEpistemicAuditResult,
  type FunctionalDependencyGraph as RESEFunctionalDependencyGraph,
  type CrossDomainPattern as RESECrossDomainPattern,
  type InvertedConstraint as RESEInvertedConstraint,
  type IsomorphicMapping as RESEIsomorphicMapping,
  type SearchTreeNode as RESESearchTreeNode,
  type Hypothesis as RESEHypothesis,
  type ValidationMetrics as RESEValidationMetrics,
  type MCTSSearchResult as RESEMCTSSearchResult,
  type ParadigmShift as RESEParadigmShift,
  type SynthesizedKnowledge as RESESynthesizedKnowledge,
  type ArchitectureAssembly as RESEArchitectureAssembly,
  transformEpistemicAuditToCanonical,
  transformIsomorphicMappingToCanonical,
  transformMCTSSearchToCanonical,
  transformArchitectureAssemblyToCanonical,
  validateEpistemicAuditResult,
  validateIsomorphicMapping,
  validateMCTSSearchResult,
  validateArchitectureAssembly,
  createUTCTimestamp as createRESEUTCTimestamp,
  createCorrelationId as createRESECorrelationId,
  RESEExamples,
} from './rese-canonical';

/**
 * Schema Registry
 *
 * Central registry of all available canonical schemas.
 * Useful for introspection and documentation generation.
 */
export const SchemaRegistry = {
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
} as const;

/**
 * Validation Utilities
 *
 * Common validation helpers for all schemas.
 */

/**
 * Validate any data against a schema
 * This is a type-safe wrapper that works with any of our canonical schemas.
 */
export function validateSchema<T>(
  schema: { safeParse: (data: unknown) => { success: boolean; data?: T; error?: any } },
  data: unknown
): {
  success: boolean;
  data?: T;
  errors?: string[];
} {
  const result = schema.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map((e: any) => `${e.path.join('.')}: ${e.message}`),
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
export function isZ3SolverRequest(data: unknown): data is import('./z3-canonical').SolverRequest {
  // This is a runtime check - in practice you'd use the schema
  return typeof data === 'object' && data !== null && 'problem' in data && 'timeout_ms' in data;
}

/**
 * Check if data is a valid LeanAide ProofVerificationRequest
 */
export function isLeanAideProofVerificationRequest(
  data: unknown
): data is import('./leanaide-canonical').ProofVerificationRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    'proof_code' in data &&
    'theorem' in data &&
    'timeout_ms' in data
  );
}

/**
 * Check if data is a valid RAGBits RAGRequest
 */
export function isRAGBitsRequest(
  data: unknown
): data is import('./ragbits-canonical').RAGRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    'query' in data &&
    'retrieval_count' in data &&
    'timeout_ms' in data
  );
}

/**
 * Check if data is a valid BubbleLab BubbleRequest
 */
export function isBubbleLabRequest(
  data: unknown
): data is import('./bubblelab-canonical').BubbleRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    'workspace_id' in data &&
    'bubble_type' in data &&
    'name' in data
  );
}

/**
 * Check if data is a valid VectorDB VectorSearchRequest
 */
export function isVectorDBSearchRequest(
  data: unknown
): data is import('./vectordb-canonical').VectorSearchRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    'collection_name' in data &&
    'query_vector' in data &&
    'top_k' in data
  );
}

/**
 * Check if data is a valid Graphiti Episode
 */
export function isGraphitiEpisode(
  data: unknown
): data is import('./graphiti-canonical').CanonicalEpisode {
  return (
    typeof data === 'object' &&
    data !== null &&
    'id' in data &&
    'name' in data &&
    'content' in data &&
    'valid_at' in data
  );
}

/**
 * Check if data is a valid KarateClub NodeEmbeddingRequest
 */
export function isKarateClubNodeEmbeddingRequest(
  data: unknown
): data is import('./karateclub-canonical').NodeEmbeddingRequest {
  return (
    typeof data === 'object' &&
    data !== null &&
    'algorithm' in data &&
    'graph' in data &&
    'timeout_ms' in data
  );
}

/**
 * Check if data is a valid RESE EpistemicAuditResult
 */
export function isRESEEpistemicAuditResult(
  data: unknown
): data is import('./rese-canonical').EpistemicAuditResult {
  return (
    typeof data === 'object' &&
    data !== null &&
    'phase' in data &&
    'audit_id' in data &&
    'problem_description' in data &&
    'timestamp' in data
  );
}

/**
 * Constants for Common Use Cases
 */

/**
 * Default timeout values (in milliseconds)
 * These are the recommended defaults based on experience.
 */
export const DEFAULT_TIMEOUTS = {
  QUICK: 5000,      // 5 seconds - for simple queries
  NORMAL: 15000,    // 15 seconds - for average complexity
  LONG: 60000,      // 1 minute - for complex proofs
  EXTENDED: 300000, // 5 minutes - maximum allowed timeout
} as const;

/**
 * Maximum sizes for various fields
 * Prevents memory issues and abuse.
 */
export const MAX_SIZES = {
  PROBLEM_LENGTH: 100000,      // 100KB for problem statements
  PROOF_CODE_LENGTH: 500000,   // 500KB for proof code
  IMPORTS_COUNT: 100,          // Maximum number of imports
  TACTICS_COUNT: 1000,         // Maximum number of tactics
  ENTITIES_COUNT: 10000,       // Maximum entities in knowledge graph
  RELATIONS_COUNT: 50000,      // Maximum relations in knowledge graph
  MESSAGES_COUNT: 1000,        // Maximum compiler messages

  // RAGBits limits
  RAG_QUERY_LENGTH: 10000,     // Maximum query length
  DOCUMENT_CHUNKS: 1000,       // Maximum chunks per ingestion
  RETRIEVAL_COUNT: 100,        // Maximum retrieval count

  // BubbleLab limits
  BUBBLE_NAME_LENGTH: 255,     // Maximum bubble name length
  WORKFLOW_STEPS: 100,         // Maximum workflow steps
  DEPENDENCY_CHAIN: 50,        // Maximum dependency depth

  // VectorDB limits
  VECTOR_DIMENSION: 10000,     // Maximum vector dimension
  VECTORS_PER_UPSERT: 1000,    // Maximum vectors per upsert
  SEARCH_TOP_K: 100,           // Maximum search results

  // Graphiti limits
  EPISODE_CONTENT_LENGTH: 100000,  // Maximum episode content
  ENTITY_ATTRIBUTES: 100,           // Maximum entity attributes
  COMMUNITY_SIZE: 10000,            // Maximum community members

  // KarateClub limits
  GRAPH_NODES: 1000000,        // Maximum nodes in a graph
  GRAPH_EDGES: 10000000,       // Maximum edges in a graph
  EMBEDDING_DIMENSION: 1024,   // Maximum embedding dimension

  // RESE limits
  TACIT_ASSUMPTIONS: 1000,         // Maximum tacit assumptions
  CONTRADICTIONS: 500,             // Maximum contradictions
  HYPOTHESES: 10000,               // Maximum hypotheses
  SEARCH_TREE_NODES: 100000,       // Maximum search tree nodes
  CROSS_DOMAIN_PATTERNS: 500,      // Maximum cross-domain patterns
  PARADIGM_SHIFTS: 50,             // Maximum paradigm shifts
  SYNTHESIZED_KNOWLEDGE: 1000,     // Maximum knowledge items
} as const;

/**
 * Error codes for common validation failures
 * Use these for consistent error messaging across adapters.
 */
export const VALIDATION_ERRORS = {
  MISSING_FIELD: 'MISSING_FIELD',
  INVALID_TYPE: 'INVALID_TYPE',
  OUT_OF_RANGE: 'OUT_OF_RANGE',
  TOO_LONG: 'TOO_LONG',
  TOO_SHORT: 'TOO_SHORT',
  INVALID_FORMAT: 'INVALID_FORMAT',
  TIMEOUT_EXCEEDED: 'TIMEOUT_EXCEEDED',
  SIZE_LIMIT_EXCEEDED: 'SIZE_LIMIT_EXCEEDED',
} as const;

/**
 * Utility function to create a correlation ID
 * Uses UUID v4 format.
 */
export function createCorrelationId(): string {
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
export function createUTCTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Utility function to format validation errors
 * Converts Zod errors to a human-readable format.
 */
export function formatValidationErrors(errors: any[]): string {
  return errors
    .map((e) => {
      const path = e.path.length > 0 ? e.path.join('.') : 'root';
      return `${path}: ${e.message}`;
    })
    .join('\n');
}

/**
 * Re-export types for convenience
 * These can be used directly in adapter code.
 */
export type {
  // Z3 Types
  SolverRequest as Z3SolverRequestType,
  SolverResponse as Z3SolverResponseType,
  KnowledgeGraphResponse as Z3KnowledgeGraphResponseType,
  Entity as Z3EntityType,
  Relation as Z3RelationType,

  // LeanAide Types
  ProofVerificationRequest as LeanAideProofVerificationRequestType,
  ProofVerificationResponse as LeanAideProofVerificationResponseType,
  LeanCompilationRequest as LeanAideCompilationRequestType,
  LeanCompilationResponse as LeanAideCompilationResponseType,
  LeanMessage as LeanMessageType,

  // RAGBits Types
  RAGRequest as RAGBitsRequestType,
  RAGResponse as RAGBitsResponseType,
  DocumentChunk as RAGBitsDocumentChunkType,
  DocumentIngestionRequest as RAGBitsIngestionRequestType,
  DocumentIngestionResponse as RAGBitsIngestionResponseType,
  RAGError as RAGBitsErrorType,

  // BubbleLab Types
  BubbleRequest as BubbleLabRequestType,
  BubbleResponse as BubbleLabResponseType,
  WorkflowRequest as BubbleLabWorkflowRequestType,
  WorkflowResponse as BubbleLabWorkflowResponseType,
  BubbleStatusRequest as BubbleLabStatusRequestType,
  BubbleStatusResponse as BubbleLabStatusResponseType,
  BubbleLabError as BubbleLabErrorType,

  // VectorDB Types
  VectorData as VectorDBVectorDataType,
  CollectionInfo as VectorDBCollectionInfoType,
  VectorUpsertRequest as VectorDBUpsertRequestType,
  VectorUpsertResponse as VectorDBUpsertResponseType,
  VectorSearchRequest as VectorDBSearchRequestType,
  VectorSearchResponse as VectorDBSearchResponseType,
  VectorDeleteRequest as VectorDBDeleteRequestType,
  VectorDeleteResponse as VectorDBDeleteResponseType,
  CollectionCreateRequest as VectorDBCollectionCreateRequestType,
  CollectionCreateResponse as VectorDBCollectionCreateResponseType,
  VectorDBError as VectorDBErrorType,

  // Graphiti Types
  CanonicalEntity as GraphitiEntityType,
  CanonicalEntityEdge as GraphitiEntityEdgeType,
  CanonicalEpisode as GraphitiEpisodeType,
  CanonicalCommunity as GraphitiCommunityType,
  CanonicalSearchQuery as GraphitiSearchQueryType,
  CanonicalSearchResult as GraphitiSearchResultType,
  AddEpisodeOperation as GraphitiAddEpisodeOperationType,
  AddEpisodeResult as GraphitiAddEpisodeResultType,
  AddTripletOperation as GraphitiAddTripletOperationType,
  AddTripletResult as GraphitiAddTripletResultType,
  GraphStatistics as GraphitiStatisticsType,

  // KarateClub Types
  GraphStructure as KarateClubGraphStructureType,
  NodeEmbeddingRequest as KarateClubNodeEmbeddingRequestType,
  NodeEmbeddingResponse as KarateClubNodeEmbeddingResponseType,
  CommunityDetectionRequest as KarateClubCommunityDetectionRequestType,
  CommunityDetectionResponse as KarateClubCommunityDetectionResponseType,
  GraphEmbeddingRequest as KarateClubGraphEmbeddingRequestType,
  GraphEmbeddingResponse as KarateClubGraphEmbeddingResponseType,
  GraphAnalysisRequest as KarateClubGraphAnalysisRequestType,
  GraphAnalysisResponse as KarateClubGraphAnalysisResponseType,

  // RESE Types
  RESEPhase as RESEPhaseType,
  ConstraintCategory as RESEConstraintCategoryType,
  LogicalFallacy as RESELogicalFallacyType,
  TacitAssumption as RESETacitAssumptionType,
  ContradictionDetection as RESEContradictionDetectionType,
  FalsificationResult as RESEFalsificationResultType,
  EpistemicAuditResult as RESEEpistemicAuditResultType,
  FunctionalDependencyGraph as RESEFunctionalDependencyGraphType,
  CrossDomainPattern as RESECrossDomainPatternType,
  InvertedConstraint as RESEInvertedConstraintType,
  IsomorphicMapping as RESEIsomorphicMappingType,
  SearchTreeNode as RESESearchTreeNodeType,
  Hypothesis as RESEHypothesisType,
  ValidationMetrics as RESEValidationMetricsType,
  MCTSSearchResult as RESEMCTSSearchResultType,
  ParadigmShift as RESEParadigmShiftType,
  SynthesizedKnowledge as RESESynthesizedKnowledgeType,
  ArchitectureAssembly as RESEArchitectureAssemblyType,
} from './index';

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
