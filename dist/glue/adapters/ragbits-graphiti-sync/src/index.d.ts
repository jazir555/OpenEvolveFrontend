/**
 * RAGBits-Graphiti Bidirectional Sync Adapter
 *
 * Main entry point for the synchronization adapter
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Configuration Explicitness: All config via env vars
 * - Failure Management: Circuit breakers and retries
 */
import SyncManager from './sync-manager';
import RAGBitsToGraphitiSync from './ragbits-to-graphiti';
import GraphitiToRAGBitsSync from './graphiti-to-ragbits';
import ConflictDetector from './conflict-detector';
export * from './canonical';
export { SyncManager, RAGBitsToGraphitiSync, GraphitiToRAGBitsSync, ConflictDetector };
export type { Document, SyncStrategy, SyncManagerConfig, SyncOperationResult, } from './sync-manager';
export type { DocumentChunkData, GraphEpisode, Entity, Relationship, RAGBitsToGraphitiConfig, } from './ragbits-to-graphiti';
export type { GraphitiEntity, GraphitiEpisode, TemporalEntity, GraphitiToRAGBitsConfig, } from './graphiti-to-ragbits';
export type { RAGBitsData, GraphitiData, DocumentChunkData as ConflictDocChunk, GraphitiEpisodeData, GraphitiEntityData, GraphitiRelationshipData, ConflictDetectorConfig, } from './conflict-detector';
export { SyncManager as default };
//# sourceMappingURL=index.d.ts.map