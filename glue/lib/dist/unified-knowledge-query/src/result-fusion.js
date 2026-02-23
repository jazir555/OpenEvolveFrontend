"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.resultFusion = exports.ResultFusion = void 0;
const glue_lib_1 = require("@openevolve/glue-lib");
/**
 * Result Fusion Class
 */
class ResultFusion {
    constructor() {
        this.logger = new glue_lib_1.Logger('result-fusion');
    }
    /**
     * Merge results from multiple systems
     */
    async merge(results, query, correlationId) {
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
        const mergedResult = {
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
    normalizeScores(results) {
        const normalized = [];
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
    detectConflicts(results) {
        const conflicts = [];
        const idMap = new Map();
        // Group by ID
        for (const result of results) {
            if (!idMap.has(result.id)) {
                idMap.set(result.id, []);
            }
            idMap.get(result.id).push(result);
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
        const report = {
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
    resolveConflicts(results, conflicts) {
        if (!conflicts.hasConflicts) {
            return results;
        }
        const resolved = [];
        const processedIds = new Set();
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
            }
            else {
                // Conflict - resolve by highest confidence
                const sorted = duplicates.sort((a, b) => b.confidence - a.confidence);
                const winner = sorted[0];
                // Merge sources
                winner.sources = Array.from(new Set(duplicates.flatMap(d => d.sources)));
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
    rankResults(results) {
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
    buildSourceMetadata(results) {
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
    calculateOverallConfidence(results) {
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
    deduplicateById(results) {
        const idMap = new Map();
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
    filterByConfidence(results, minConfidence) {
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
    filterByType(results, types) {
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
    limitResults(results, limit) {
        return results.slice(0, limit);
    }
}
exports.ResultFusion = ResultFusion;
/**
 * Default fusion instance
 */
exports.resultFusion = new ResultFusion();
//# sourceMappingURL=result-fusion.js.map