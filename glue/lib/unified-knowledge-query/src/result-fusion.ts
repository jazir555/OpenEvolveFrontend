/**
 * Result Fusion
 *
 * Merges and normalizes results from multiple knowledge systems.
 *
 * Federation Constitution Compliance:
 * - Anti-Corruption Layer: All results normalized to canonical format
 * - Conflict Resolution: Detects and resolves data conflicts
 * - Idempotency: Merge operations are deterministic
 */

import { Logger } from '@openevolve/glue-lib';
import {
  KnowledgeItem,
  SystemSource,
  UnifiedQueryResult,
  SourceMetadata,
  ConflictReport,
} from './canonical';

/**
 * System Result with metadata
 */
interface SystemResult {
  system: SystemSource;
  items: KnowledgeItem[];
  queryTimeMs: number;
  success: boolean;
  error?: string;
}

/**
 * Normalized result for fusion
 */
interface NormalizedResult extends KnowledgeItem {
  normalizedScore: number;
  sources: SystemSource[];
}

/**
 * Conflict detection result
 */
interface Conflict {
  field: string;
  sources: SystemSource[];
  values: any[];
  resolution?: string;
}

/**
 * Result Fusion Class
 */
export class ResultFusion {
  private logger: Logger;

  constructor() {
    this.logger = new Logger('result-fusion');
  }

  /**
   * Merge results from multiple systems
   */
  async merge(
    results: SystemResult[],
    query: string,
    correlationId: string
  ): Promise<UnifiedQueryResult> {
    const startTime = Date.now();

    this.logger.info('Merging results', {
      correlation_id: correlationId,
      result_count: results.length,
      systems: results.map(r => r.system),
    });

    // Step 1: Normalize scores
    const normalized = this.normalizeScores(results);

    // Step 2: Detect conflicts
    const conflicts = this.detectConflicts(normalized);

    // Step 3: Resolve conflicts
    const resolved = this.resolveConflicts(normalized, conflicts);

    // Step 4: Rank results
    const ranked = this.rankResults(resolved);

    // Step 5: Build source metadata
    const sources = this.buildSourceMetadata(results);

    // Step 6: Calculate overall confidence
    const confidence = this.calculateOverallConfidence(ranked);

    const mergedResult: UnifiedQueryResult = {
      query,
      results: ranked,
      sources,
      confidence,
      executionTimeMs: Date.now() - startTime,
      conflicts,
      correlationId,
    };

    this.logger.info('Results merged', {
      correlation_id: correlationId,
      total_results: ranked.length,
      confidence,
      conflicts: conflicts.hasConflicts,
    });

    return mergedResult;
  }

  /**
   * Normalize scores across systems
   * Uses min-max normalization per system
   */
  private normalizeScores(results: SystemResult[]): NormalizedResult[] {
    const normalized: NormalizedResult[] = [];

    for (const result of results) {
      if (!result.success || result.items.length === 0) {
        continue;
      }

      // Find min and max relevance for this system
      const relevances = result.items.map(item => item.relevance);
      const minRel = Math.min(...relevances);
      const maxRel = Math.max(...relevances);
      const range = maxRel - minRel || 1; // Avoid division by zero

      // Normalize each item
      for (const item of result.items) {
        const normalizedScore = (item.relevance - minRel) / range;

        normalized.push({
          ...item,
          normalizedScore,
          sources: [result.system],
        });
      }
    }

    this.logger.debug('Scores normalized', {
      total_items: normalized.length,
    });

    return normalized;
  }

  /**
   * Detect conflicts between results
   * Currently checks for duplicate IDs with different content
   */
  private detectConflicts(results: NormalizedResult[]): ConflictReport {
    const conflicts: Conflict[] = [];
    const idMap = new Map<string, NormalizedResult[]>();

    // Group by ID
    for (const result of results) {
      if (!idMap.has(result.id)) {
        idMap.set(result.id, []);
      }
      idMap.get(result.id)!.push(result);
    }

    // Check for conflicts
    for (const [id, items] of idMap.entries()) {
      if (items.length > 1) {
        // Multiple systems returned same ID
        const contents = new Set(items.map(i => i.content));

        if (contents.size > 1) {
          // Content conflict detected
          conflicts.push({
            field: 'content',
            sources: items.map(i => i.source),
            values: Array.from(contents),
            resolution: 'highest_confidence',
          });
        }
      }
    }

    const report: ConflictReport = {
      hasConflicts: conflicts.length > 0,
      conflicts,
    };

    if (conflicts.length > 0) {
      this.logger.warn('Conflicts detected', {
        conflict_count: conflicts.length,
      });
    }

    return report;
  }

  /**
   * Resolve conflicts using highest confidence strategy
   */
  private resolveConflicts(
    results: NormalizedResult[],
    conflicts: ConflictReport
  ): NormalizedResult[] {
    if (!conflicts.hasConflicts) {
      return results;
    }

    const resolved: NormalizedResult[] = [];
    const processedIds = new Set<string>();

    for (const result of results) {
      // Skip if already processed
      if (processedIds.has(result.id)) {
        continue;
      }

      // Find all results with same ID
      const duplicates = results.filter(r => r.id === result.id);

      if (duplicates.length === 1) {
        // No conflict
        resolved.push(result);
        processedIds.add(result.id);
      } else {
        // Conflict - resolve by highest confidence
        const sorted = duplicates.sort((a, b) => b.confidence - a.confidence);
        const winner = sorted[0];

        // Merge sources
        winner.sources = Array.from(new Set(
          duplicates.flatMap(d => d.sources)
        ));

        resolved.push(winner);
        processedIds.add(result.id);
      }
    }

    this.logger.debug('Conflicts resolved', {
      resolved_count: resolved.length,
    });

    return resolved;
  }

  /**
   * Rank results by combined score
   * Combines normalized score and confidence
   */
  private rankResults(results: NormalizedResult[]): KnowledgeItem[] {
    // Calculate combined score
    const withScores = results.map(item => ({
      item,
      combinedScore: (item.normalizedScore * 0.6) + (item.confidence * 0.4),
    }));

    // Sort by combined score descending
    withScores.sort((a, b) => b.combinedScore - a.combinedScore);

    // Remove internal fields and return
    const ranked = withScores.map(ws => {
      const { normalizedScore, sources, ...item } = ws.item;
      return item;
    });

    this.logger.debug('Results ranked', {
      total_results: ranked.length,
      top_score: ranked[0]?.relevance || 0,
    });

    return ranked;
  }

  /**
   * Build source metadata from results
   */
  private buildSourceMetadata(results: SystemResult[]): SourceMetadata[] {
    return results.map(r => ({
      system: r.system,
      queryTimeMs: r.queryTimeMs,
      resultCount: r.items.length,
      success: r.success,
      error: r.error,
    }));
  }

  /**
   * Calculate overall confidence from all results
   */
  private calculateOverallConfidence(results: KnowledgeItem[]): number {
    if (results.length === 0) {
      return 0;
    }

    // Weighted average by relevance
    let totalWeight = 0;
    let weightedSum = 0;

    for (const result of results) {
      const weight = result.relevance;
      weightedSum += result.confidence * weight;
      totalWeight += weight;
    }

    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }

  /**
   * Deduplicate results by ID
   * Keeps highest confidence version
   */
  deduplicateById(results: KnowledgeItem[]): KnowledgeItem[] {
    const idMap = new Map<string, KnowledgeItem>();

    for (const result of results) {
      const existing = idMap.get(result.id);

      if (!existing || result.confidence > existing.confidence) {
        idMap.set(result.id, result);
      }
    }

    const deduped = Array.from(idMap.values());

    this.logger.debug('Results deduplicated', {
      original_count: results.length,
      deduplicated_count: deduped.length,
    });

    return deduped;
  }

  /**
   * Filter results by minimum confidence
   */
  filterByConfidence(results: KnowledgeItem[], minConfidence: number): KnowledgeItem[] {
    const filtered = results.filter(r => r.confidence >= minConfidence);

    this.logger.debug('Results filtered by confidence', {
      original_count: results.length,
      filtered_count: filtered.length,
      min_confidence: minConfidence,
    });

    return filtered;
  }

  /**
   * Filter results by knowledge type
   */
  filterByType(results: KnowledgeItem[], types: string[]): KnowledgeItem[] {
    if (types.includes('all')) {
      return results;
    }

    const filtered = results.filter(r => types.includes(r.type));

    this.logger.debug('Results filtered by type', {
      original_count: results.length,
      filtered_count: filtered.length,
      types,
    });

    return filtered;
  }

  /**
   * Limit results to top N
   */
  limitResults(results: KnowledgeItem[], limit: number): KnowledgeItem[] {
    return results.slice(0, limit);
  }
}

/**
 * Default fusion instance
 */
export const resultFusion = new ResultFusion();
