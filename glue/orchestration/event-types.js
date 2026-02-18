"use strict";
/**
 * Event Type Definitions
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: All events have unique IDs for deduplication
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - Canonical schema for all orchestration events
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateEvent = exports.isEventType = exports.createBaseEvent = void 0;
const uuid_1 = require("uuid");
/**
 * Create a base event with common fields
 */
function createBaseEvent(type, sourceService, correlationId, data) {
    return {
        id: (0, uuid_1.v4)(),
        type,
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
        source_service: sourceService,
        data,
        metadata: {}
    };
}
exports.createBaseEvent = createBaseEvent;
/**
 * Type guard to check if event is of specific type
 */
function isEventType(event, type) {
    return event.type === type;
}
exports.isEventType = isEventType;
/**
 * Validate event structure
 */
function validateEvent(event) {
    const errors = [];
    if (!event.id || typeof event.id !== 'string') {
        errors.push('Missing or invalid id field');
    }
    if (!event.type || typeof event.type !== 'string') {
        errors.push('Missing or invalid type field');
    }
    if (!event.timestamp || typeof event.timestamp !== 'string') {
        errors.push('Missing or invalid timestamp field');
    }
    if (!event.correlation_id || typeof event.correlation_id !== 'string') {
        errors.push('Missing or invalid correlation_id field');
    }
    if (!event.source_service || typeof event.source_service !== 'string') {
        errors.push('Missing or invalid source_service field');
    }
    if (!event.data || typeof event.data !== 'object') {
        errors.push('Missing or invalid data field');
    }
    return {
        valid: errors.length === 0,
        errors
    };
}
exports.validateEvent = validateEvent;
/**
 * Example usage:
 *
 * ```typescript
 * import { createBaseEvent, isEventType } from './event-types';
 *
 * // Create a KnowledgeExtracted event
 * const event = createBaseEvent(
 *   'KnowledgeExtracted',
 *   'ragbits-adapter',
 *   'corr-123',
 *   {
 *     document_id: 'doc-456',
 *     chunk_count: 10,
 *     chunks: [...],
 *     extraction_method: 'recursive'
 *   }
 * );
 *
 * // Check event type
 * if (isEventType(event, 'KnowledgeExtracted')) {
 *   console.log(`Extracted ${event.data.chunk_count} chunks`);
 * }
 *
 * // Validate event
 * const validation = validateEvent(event);
 * if (!validation.valid) {
 *   console.error('Invalid event:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=event-types.js.map