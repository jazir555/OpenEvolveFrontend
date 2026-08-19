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

import SyncManager from './sync-manager.js';
import RAGBitsToGraphitiSync from './ragbits-to-graphiti.js';
import GraphitiToRAGBitsSync from './graphiti-to-ragbits.js';
import ConflictDetector from './conflict-detector.js';

// Export canonical schemas
export * from './canonical.js';

// Export main classes
export { SyncManager, RAGBitsToGraphitiSync, GraphitiToRAGBitsSync, ConflictDetector };

// Export types
export type {
  Document,
  SyncStrategy,
  SyncManagerConfig,
  SyncOperationResult,
} from './sync-manager.js';

export type {
  DocumentChunkData,
  GraphEpisode,
  Entity,
  Relationship,
  RAGBitsToGraphitiConfig,
} from './ragbits-to-graphiti.js';

export type {
  GraphitiEntity,
  GraphitiEpisode,
  TemporalEntity,
  GraphitiToRAGBitsConfig,
} from './graphiti-to-ragbits.js';

export type {
  RAGBitsData,
  GraphitiData,
  DocumentChunkData as ConflictDocChunk,
  GraphitiEpisodeData,
  GraphitiEntityData,
  GraphitiRelationshipData,
  ConflictDetectorConfig,
} from './conflict-detector.js';

// Default export
export { SyncManager as default };
