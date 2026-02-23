"use strict";
/**
 * BubbleLab Canonical Schema
 *
 * Purpose: Define the canonical data model for BubbleLab integration
 * Compliance: Anti-Corruption Layer - normalize BubbleLab data to canonical form
 *
 * This schema maps BubbleLab's bubble/workflow concepts to the OpenEvolve canonical model
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CanonicalCredentialMappingSchema = exports.CanonicalBubbleLabEventSchema = exports.CanonicalExecutionResultSchema = exports.CanonicalBubbleFlowSchema = exports.CanonicalBubbleSchema = exports.ExecutionStatus = exports.EventType = exports.CredentialType = exports.BubbleType = void 0;
exports.mapToCanonicalBubbleFlow = mapToCanonicalBubbleFlow;
exports.mapToCanonicalExecutionResult = mapToCanonicalExecutionResult;
exports.mapFromCanonicalBubbleFlow = mapFromCanonicalBubbleFlow;
exports.mapFromCanonicalCredentials = mapFromCanonicalCredentials;
exports.validateCanonicalBubbleFlow = validateCanonicalBubbleFlow;
exports.validateCanonicalExecutionResult = validateCanonicalExecutionResult;
exports.validateCanonicalBubbleLabEvent = validateCanonicalBubbleLabEvent;
exports.generateCorrelationId = generateCorrelationId;
exports.toUTCISOString = toUTCISOString;
exports.fromUTCISOString = fromUTCISOString;
const zod_1 = require("zod");
// =============================================================================
// BubbleLab-Specific Canonical Types
// =============================================================================
/**
 * Bubble Type Enumeration
 * Maps to different bubble types in BubbleLab (PostgreSQL, Slack, AI Agent, etc.)
 */
var BubbleType;
(function (BubbleType) {
    BubbleType["POSTGRESQL"] = "postgresql";
    BubbleType["SLACK"] = "slack";
    BubbleType["AI_AGENT"] = "ai_agent";
    BubbleType["DATABASE_ANALYZER"] = "database_analyzer";
    BubbleType["SLACK_NOTIFIER"] = "slack_notifier";
    BubbleType["WEBHOOK"] = "webhook";
    BubbleType["CUSTOM"] = "custom";
})(BubbleType || (exports.BubbleType = BubbleType = {}));
/**
 * Credential Type Enumeration
 * Maps to BubbleLab credential types
 */
var CredentialType;
(function (CredentialType) {
    CredentialType["DATABASE_CRED"] = "DATABASE_CRED";
    CredentialType["SLACK_CRED"] = "SLACK_CRED";
    CredentialType["FIRECRAWL_API_KEY"] = "FIRECRAWL_API_KEY";
    CredentialType["OPENAI_CRED"] = "OPENAI_CRED";
    CredentialType["ANTHROPIC_CRED"] = "ANTHROPIC_CRED";
    CredentialType["GOOGLE_GEMINI_CRED"] = "GOOGLE_GEMINI_CRED";
})(CredentialType || (exports.CredentialType = CredentialType = {}));
/**
 * Event Type Enumeration
 * Maps to BubbleLab trigger event types
 */
var EventType;
(function (EventType) {
    EventType["WEBHOOK_HTTP"] = "webhook/http";
    EventType["SCHEDULE"] = "schedule";
    EventType["MANUAL"] = "manual";
})(EventType || (exports.EventType = EventType = {}));
/**
 * Workflow Execution Status
 */
var ExecutionStatus;
(function (ExecutionStatus) {
    ExecutionStatus["PENDING"] = "pending";
    ExecutionStatus["RUNNING"] = "running";
    ExecutionStatus["SUCCESS"] = "success";
    ExecutionStatus["FAILED"] = "failed";
    ExecutionStatus["TIMEOUT"] = "timeout";
})(ExecutionStatus || (exports.ExecutionStatus = ExecutionStatus = {}));
// =============================================================================
// Canonical Schemas (Zod)
// =============================================================================
/**
 * Canonical Bubble Definition
 * Represents a single bubble in a workflow
 */
exports.CanonicalBubbleSchema = zod_1.z.object({
    id: zod_1.z.string().optional(),
    name: zod_1.z.string(),
    type: zod_1.z.nativeEnum(BubbleType),
    config: zod_1.z.record(zod_1.z.any()),
    required_credentials: zod_1.z.array(zod_1.z.nativeEnum(CredentialType)).optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Canonical BubbleFlow (Workflow) Definition
 * Represents a complete BubbleLab workflow
 */
exports.CanonicalBubbleFlowSchema = zod_1.z.object({
    id: zod_1.z.string().optional(),
    name: zod_1.z.string().min(1, 'Workflow name is required'),
    description: zod_1.z.string().optional(),
    event_type: zod_1.z.nativeEnum(EventType),
    code: zod_1.z.string().optional(),
    bubbles: zod_1.z.array(exports.CanonicalBubbleSchema).optional(),
    required_credentials: zod_1.z.record(zod_1.z.string(), zod_1.z.array(zod_1.z.nativeEnum(CredentialType))).optional(),
    webhook_active: zod_1.z.boolean().default(false),
    webhook_url: zod_1.z.string().url().optional(),
    created_at: zod_1.z.string().datetime().optional(),
    updated_at: zod_1.z.string().datetime().optional(),
});
/**
 * Canonical Workflow Execution Result
 */
exports.CanonicalExecutionResultSchema = zod_1.z.object({
    execution_id: zod_1.z.string().optional(),
    flow_id: zod_1.z.string(),
    status: zod_1.z.nativeEnum(ExecutionStatus),
    output: zod_1.z.any().optional(),
    error: zod_1.z.string().optional(),
    started_at: zod_1.z.string().datetime(),
    completed_at: zod_1.z.string().datetime().optional(),
    duration_ms: zod_1.z.number().optional(),
    logs: zod_1.z.array(zod_1.z.object({
        timestamp: zod_1.z.string().datetime(),
        level: zod_1.z.string(),
        message: zod_1.z.string(),
    })).optional(),
});
/**
 * Canonical BubbleLab Event
 * Represents events from BubbleLab to be processed by the orchestration layer
 */
exports.CanonicalBubbleLabEventSchema = zod_1.z.object({
    event_id: zod_1.z.string().uuid(),
    event_type: zod_1.z.enum([
        'workflow.created',
        'workflow.updated',
        'workflow.deleted',
        'workflow.executed',
        'workflow.execution_failed',
        'bubble.created',
        'bubble.updated',
    ]),
    flow_id: zod_1.z.string().optional(),
    execution_id: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
    data: zod_1.z.any(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Canonical Credential Mapping
 * Maps credential types to their IDs
 */
exports.CanonicalCredentialMappingSchema = zod_1.z.record(zod_1.z.nativeEnum(CredentialType), zod_1.z.number() // credential ID
);
// =============================================================================
// Mapping Functions: BubbleLab Native -> Canonical
// =============================================================================
/**
 * Map BubbleLab API response to Canonical BubbleFlow
 *
 * @param apiResponse - Raw API response from BubbleLab
 * @returns Canonical BubbleFlow
 */
function mapToCanonicalBubbleFlow(apiResponse) {
    return {
        id: apiResponse.id?.toString(),
        name: apiResponse.name || 'Unnamed Flow',
        description: apiResponse.description,
        event_type: mapEventType(apiResponse.eventType),
        code: apiResponse.code,
        bubbles: apiResponse.bubbles || [],
        required_credentials: apiResponse.requiredCredentials || {},
        webhook_active: apiResponse.webhookActive || false,
        webhook_url: apiResponse.webhookUrl,
        created_at: apiResponse.createdAt
            ? new Date(apiResponse.createdAt).toISOString()
            : undefined,
        updated_at: apiResponse.updatedAt
            ? new Date(apiResponse.updatedAt).toISOString()
            : undefined,
    };
}
/**
 * Map BubbleLab event type string to canonical enum
 */
function mapEventType(eventType) {
    const mapping = {
        'webhook/http': EventType.WEBHOOK_HTTP,
        'schedule': EventType.SCHEDULE,
        'manual': EventType.MANUAL,
    };
    return mapping[eventType] || EventType.MANUAL;
}
/**
 * Map BubbleLab execution result to canonical form
 */
function mapToCanonicalExecutionResult(apiResponse, flowId) {
    const startedAt = apiResponse.startedAt
        ? new Date(apiResponse.startedAt).toISOString()
        : new Date().toISOString();
    const completedAt = apiResponse.completedAt
        ? new Date(apiResponse.completedAt).toISOString()
        : undefined;
    return {
        execution_id: apiResponse.id?.toString(),
        flow_id: flowId,
        status: mapExecutionStatus(apiResponse.status),
        output: apiResponse.output,
        error: apiResponse.error,
        started_at: startedAt,
        completed_at: completedAt,
        duration_ms: completedAt && startedAt
            ? new Date(completedAt).getTime() - new Date(startedAt).getTime()
            : undefined,
        logs: apiResponse.logs || [],
    };
}
/**
 * Map execution status string to canonical enum
 */
function mapExecutionStatus(status) {
    const mapping = {
        'pending': ExecutionStatus.PENDING,
        'running': ExecutionStatus.RUNNING,
        'success': ExecutionStatus.SUCCESS,
        'failed': ExecutionStatus.FAILED,
        'timeout': ExecutionStatus.TIMEOUT,
    };
    return mapping[status?.toLowerCase()] || ExecutionStatus.PENDING;
}
// =============================================================================
// Mapping Functions: Canonical -> BubbleLab Native
// =============================================================================
/**
 * Map Canonical BubbleFlow to BubbleLab API request format
 */
function mapFromCanonicalBubbleFlow(canonical) {
    return {
        name: canonical.name,
        description: canonical.description,
        code: canonical.code,
        eventType: canonical.event_type,
        webhookActive: canonical.webhook_active,
    };
}
/**
 * Map Canonical Credential Mapping to BubbleLab format
 */
function mapFromCanonicalCredentials(canonical) {
    // Convert enum keys to string keys
    const result = {};
    for (const [key, value] of Object.entries(canonical)) {
        result[key] = Number(value);
    }
    return result;
}
// =============================================================================
// Validation Functions
// =============================================================================
/**
 * Validate and parse a CanonicalBubbleFlow
 */
function validateCanonicalBubbleFlow(data) {
    return exports.CanonicalBubbleFlowSchema.parse(data);
}
/**
 * Validate and parse a CanonicalExecutionResult
 */
function validateCanonicalExecutionResult(data) {
    return exports.CanonicalExecutionResultSchema.parse(data);
}
/**
 * Validate and parse a CanonicalBubbleLabEvent
 */
function validateCanonicalBubbleLabEvent(data) {
    return exports.CanonicalBubbleLabEventSchema.parse(data);
}
// =============================================================================
// Utility Functions
// =============================================================================
/**
 * Generate a unique correlation ID for tracking
 */
function generateCorrelationId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
/**
 * Convert UTC Date to ISO-8601 string (Law of UTC)
 */
function toUTCISOString(date) {
    return date.toISOString();
}
/**
 * Parse ISO-8601 string to UTC Date
 */
function fromUTCISOString(isoString) {
    return new Date(isoString);
}
//# sourceMappingURL=bubblelab-canonical.js.map