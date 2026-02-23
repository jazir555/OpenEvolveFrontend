"use strict";
/**
 * AI Council Framework Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AI Council Framework
 * (multi-agent deliberation and decision-making) interactions.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AiCouncilExamples = exports.AiCouncilError = exports.AiCouncilResponse = exports.AiCouncilRequest = exports.CouncilVote = exports.CouncilProposal = exports.CouncilMember = exports.DecisionMethod = exports.CouncilState = exports.CouncilRole = void 0;
exports.validateAiCouncilRequest = validateAiCouncilRequest;
exports.isAiCouncilRequest = isAiCouncilRequest;
const zod_1 = require("zod");
/**
 * Council Role Enum
 */
exports.CouncilRole = zod_1.z.enum([
    'facilitator',
    'expert',
    'critic',
    'advocate',
    'observer',
    'voter',
]);
/**
 * Council State Enum
 */
exports.CouncilState = zod_1.z.enum([
    'forming',
    'deliberating',
    'voting',
    'consensus',
    'dissolved',
]);
/**
 * Decision Method Enum
 */
exports.DecisionMethod = zod_1.z.enum([
    'consensus',
    'majority_vote',
    'weighted_vote',
    'expert_weighted',
    'facilitator_decides',
]);
/**
 * Council Member Schema
 */
exports.CouncilMember = zod_1.z.object({
    member_id: zod_1.z.string().describe("Unique member identifier"),
    agent_id: zod_1.z.string().optional().describe("Underlying agent ID"),
    role: exports.CouncilRole.describe("Member role in council"),
    expertise: zod_1.z.array(zod_1.z.string()).optional().describe("Areas of expertise"),
    weight: zod_1.z.number().min(0).max(1).optional().describe("Voting weight"),
    config: zod_1.z.object({
        participation_required: zod_1.z.boolean().optional(),
        can_vote: zod_1.z.boolean().optional(),
        can_propose: zod_1.z.boolean().optional(),
    }).optional().describe("Member configuration"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Council Proposal Schema
 */
exports.CouncilProposal = zod_1.z.object({
    proposal_id: zod_1.z.string().uuid().optional().describe("Proposal identifier"),
    proposal_type: zod_1.z.enum([
        'decision',
        'action',
        'policy',
        'recommendation',
    ]).describe("Type of proposal"),
    title: zod_1.z.string().describe("Proposal title"),
    description: zod_1.z.string().optional().describe("Detailed description"),
    proposer_id: zod_1.z.string().describe("Member proposing"),
    content: zod_1.z.record(zod_1.z.any()).describe("Proposal content/data"),
    options: zod_1.z.array(zod_1.z.object({
        option_id: zod_1.z.string(),
        description: zod_1.z.string(),
        data: zod_1.z.record(zod_1.z.any()).optional(),
    })).optional().describe("Multiple choice options"),
    deadline: zod_1.z.string().datetime().optional().describe("Voting deadline"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Council Vote Schema
 */
exports.CouncilVote = zod_1.z.object({
    vote_id: zod_1.z.string().uuid().optional().describe("Vote identifier"),
    proposal_id: zod_1.z.string().describe("Proposal being voted on"),
    member_id: zod_1.z.string().describe("Voting member"),
    decision: zod_1.z.union([
        zod_1.z.boolean(),
        zod_1.z.string(),
        zod_1.z.number(),
    ]).describe("Vote decision"),
    rationale: zod_1.z.string().optional().describe("Reasoning for vote"),
    confidence: zod_1.z.number().min(0).max(1).optional().describe("Confidence level"),
    timestamp: zod_1.z.string().datetime().optional().describe("Vote timestamp"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Council Session Request Schema
 */
exports.AiCouncilRequest = zod_1.z.object({
    council_id: zod_1.z.string().optional().describe("Council identifier (optional for creation)"),
    session_id: zod_1.z.string().optional().describe("Session identifier"),
    action: zod_1.z.enum([
        'create_council',
        'add_member',
        'remove_member',
        'propose',
        'vote',
        'deliberate',
        'decide',
        'query_state',
    ]).describe("Action to perform"),
    council_config: zod_1.z.object({
        name: zod_1.z.string().optional(),
        description: zod_1.z.string().optional(),
        decision_method: exports.DecisionMethod.optional(),
        quorum: zod_1.z.number().min(0).max(1).optional().describe("Minimum participation"),
        max_deliberation_time_ms: zod_1.z.number().int().positive().optional(),
        voting_timeout_ms: zod_1.z.number().int().positive().optional(),
    }).optional().describe("Council configuration"),
    members: zod_1.z.array(exports.CouncilMember).optional().describe("Council members"),
    proposal: exports.CouncilProposal.optional().describe("Proposal to submit"),
    vote: exports.CouncilVote.optional().describe("Vote to cast"),
    deliberation_config: zod_1.z.object({
        topic: zod_1.z.string().optional().describe("Topic to deliberate"),
        context: zod_1.z.record(zod_1.z.any()).optional().describe("Deliberation context"),
        max_rounds: zod_1.z.number().int().positive().optional().describe("Max deliberation rounds"),
    }).optional().describe("Deliberation configuration"),
    timeout_ms: zod_1.z.number()
        .int().positive().max(300000)
        .describe("Request timeout (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Council Session Response Schema
 */
exports.AiCouncilResponse = zod_1.z.object({
    council_id: zod_1.z.string().describe("Council identifier"),
    session_id: zod_1.z.string().optional().describe("Session identifier"),
    action: zod_1.z.enum([
        'create_council',
        'add_member',
        'remove_member',
        'propose',
        'vote',
        'deliberate',
        'decide',
        'query_state',
    ]).describe("Action performed"),
    status: zod_1.z.enum([
        'success',
        'failed',
        'timeout',
        'pending',
    ]).describe("Action status"),
    council_state: exports.CouncilState.optional().describe("Current council state"),
    result: zod_1.z.object({
        members: zod_1.z.array(exports.CouncilMember).optional().describe("Council members"),
        proposal: exports.CouncilProposal.optional().describe("Active proposal"),
        votes: zod_1.z.array(exports.CouncilVote).optional().describe("Cast votes"),
        decision: zod_1.z.record(zod_1.z.any()).optional().describe("Final decision"),
        consensus: zod_1.z.boolean().optional().describe("Whether consensus reached"),
        deliberation_summary: zod_1.z.string().optional().describe("Summary of deliberations"),
        vote_summary: zod_1.z.object({
            total_votes: zod_1.z.number().optional(),
            in_favor: zod_1.z.number().optional(),
            against: zod_1.z.number().optional(),
            abstained: zod_1.z.number().optional(),
        }).optional().describe("Vote summary"),
    }).optional().describe("Action result"),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }).optional(),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional(),
        updated_at: zod_1.z.string().datetime().optional(),
        processing_time_ms: zod_1.z.number().optional(),
    }).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Error Model
 */
exports.AiCouncilError = zod_1.z.object({
    code: zod_1.z.enum([
        'COUNCIL_NOT_FOUND',
        'MEMBER_NOT_FOUND',
        'PROPOSAL_NOT_FOUND',
        'QUORUM_NOT_MET',
        'DELIBERATION_TIMEOUT',
        'INVALID_VOTE',
        'MEMBER_ALREADY_EXISTS',
        'COUNCIL_DISSOLVED',
        'VALIDATION_ERROR',
        'UNKNOWN_ERROR',
    ]),
    message: zod_1.z.string(),
    details: zod_1.z.record(zod_1.z.any()).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Validation Functions
 */
function validateAiCouncilRequest(data) {
    const result = exports.AiCouncilRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function isAiCouncilRequest(data) {
    return typeof data === 'object' && data !== null &&
        'action' in data;
}
/**
 * Examples
 */
exports.AiCouncilExamples = {
    validCreateCouncil: {
        council_config: {
            name: "Security Council",
            decision_method: "consensus",
            quorum: 0.67,
            max_deliberation_time_ms: 300000,
        },
        members: [
            {
                member_id: "member_001",
                role: "facilitator",
                expertise: ["security", "policy"],
            },
            {
                member_id: "member_002",
                role: "expert",
                expertise: ["threat_detection"],
            },
        ],
        action: "create_council",
        timeout_ms: 10000,
    },
    validProposal: {
        council_id: "council_001",
        action: "propose",
        proposal: {
            proposal_id: "550e8400-e29b-41d4-a716-446655440000",
            proposal_type: "decision",
            title: "Deploy Security Patch",
            description: "Deploy critical security patch to production",
            proposer_id: "member_001",
            content: { patch_id: "PATCH-1234", priority: "critical" },
            options: [
                { option_id: "deploy", description: "Deploy immediately" },
                { option_id: "defer", description: "Defer to maintenance window" },
            ],
        },
        timeout_ms: 5000,
    },
};
//# sourceMappingURL=ai-council-framework-canonical.js.map