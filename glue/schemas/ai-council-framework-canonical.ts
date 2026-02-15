/**
 * AI Council Framework Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AI Council Framework
 * (multi-agent deliberation and decision-making) interactions.
 */

import { z } from 'zod';

/**
 * Council Role Enum
 */
export const CouncilRole = z.enum([
  'facilitator',
  'expert',
  'critic',
  'advocate',
  'observer',
  'voter',
]);

export type CouncilRole = z.infer<typeof CouncilRole>;

/**
 * Council State Enum
 */
export const CouncilState = z.enum([
  'forming',
  'deliberating',
  'voting',
  'consensus',
  'dissolved',
]);

export type CouncilState = z.infer<typeof CouncilState>;

/**
 * Decision Method Enum
 */
export const DecisionMethod = z.enum([
  'consensus',
  'majority_vote',
  'weighted_vote',
  'expert_weighted',
  'facilitator_decides',
]);

export type DecisionMethod = z.infer<typeof DecisionMethod>;

/**
 * Council Member Schema
 */
export const CouncilMember = z.object({
  member_id: z.string().describe("Unique member identifier"),

  agent_id: z.string().optional().describe("Underlying agent ID"),

  role: CouncilRole.describe("Member role in council"),

  expertise: z.array(z.string()).optional().describe("Areas of expertise"),

  weight: z.number().min(0).max(1).optional().describe("Voting weight"),

  config: z.object({
    participation_required: z.boolean().optional(),
    can_vote: z.boolean().optional(),
    can_propose: z.boolean().optional(),
  }).optional().describe("Member configuration"),

  metadata: z.record(z.any()).optional(),
});

export type CouncilMember = z.infer<typeof CouncilMember>;

/**
 * Council Proposal Schema
 */
export const CouncilProposal = z.object({
  proposal_id: z.string().uuid().optional().describe("Proposal identifier"),

  proposal_type: z.enum([
    'decision',
    'action',
    'policy',
    'recommendation',
  ]).describe("Type of proposal"),

  title: z.string().describe("Proposal title"),

  description: z.string().optional().describe("Detailed description"),

  proposer_id: z.string().describe("Member proposing"),

  content: z.record(z.any()).describe("Proposal content/data"),

  options: z.array(z.object({
    option_id: z.string(),
    description: z.string(),
    data: z.record(z.any()).optional(),
  })).optional().describe("Multiple choice options"),

  deadline: z.string().datetime().optional().describe("Voting deadline"),

  metadata: z.record(z.any()).optional(),
});

export type CouncilProposal = z.infer<typeof CouncilProposal>;

/**
 * Council Vote Schema
 */
export const CouncilVote = z.object({
  vote_id: z.string().uuid().optional().describe("Vote identifier"),

  proposal_id: z.string().describe("Proposal being voted on"),

  member_id: z.string().describe("Voting member"),

  decision: z.union([
    z.boolean(),
    z.string(),
    z.number(),
  ]).describe("Vote decision"),

  rationale: z.string().optional().describe("Reasoning for vote"),

  confidence: z.number().min(0).max(1).optional().describe("Confidence level"),

  timestamp: z.string().datetime().optional().describe("Vote timestamp"),

  metadata: z.record(z.any()).optional(),
});

export type CouncilVote = z.infer<typeof CouncilVote>;

/**
 * Council Session Request Schema
 */
export const AiCouncilRequest = z.object({
  council_id: z.string().optional().describe("Council identifier (optional for creation)"),

  session_id: z.string().optional().describe("Session identifier"),

  action: z.enum([
    'create_council',
    'add_member',
    'remove_member',
    'propose',
    'vote',
    'deliberate',
    'decide',
    'query_state',
  ]).describe("Action to perform"),

  council_config: z.object({
    name: z.string().optional(),
    description: z.string().optional(),
    decision_method: DecisionMethod.optional(),
    quorum: z.number().min(0).max(1).optional().describe("Minimum participation"),
    max_deliberation_time_ms: z.number().int().positive().optional(),
    voting_timeout_ms: z.number().int().positive().optional(),
  }).optional().describe("Council configuration"),

  members: z.array(CouncilMember).optional().describe("Council members"),

  proposal: CouncilProposal.optional().describe("Proposal to submit"),

  vote: CouncilVote.optional().describe("Vote to cast"),

  deliberation_config: z.object({
    topic: z.string().optional().describe("Topic to deliberate"),
    context: z.record(z.any()).optional().describe("Deliberation context"),
    max_rounds: z.number().int().positive().optional().describe("Max deliberation rounds"),
  }).optional().describe("Deliberation configuration"),

  timeout_ms: z.number()
    .int().positive().max(300000)
    .describe("Request timeout (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),

  metadata: z.record(z.any()).optional(),
});

export type AiCouncilRequest = z.infer<typeof AiCouncilRequest>;

/**
 * Council Session Response Schema
 */
export const AiCouncilResponse = z.object({
  council_id: z.string().describe("Council identifier"),

  session_id: z.string().optional().describe("Session identifier"),

  action: z.enum([
    'create_council',
    'add_member',
    'remove_member',
    'propose',
    'vote',
    'deliberate',
    'decide',
    'query_state',
  ]).describe("Action performed"),

  status: z.enum([
    'success',
    'failed',
    'timeout',
    'pending',
  ]).describe("Action status"),

  council_state: CouncilState.optional().describe("Current council state"),

  result: z.object({
    members: z.array(CouncilMember).optional().describe("Council members"),
    proposal: CouncilProposal.optional().describe("Active proposal"),
    votes: z.array(CouncilVote).optional().describe("Cast votes"),
    decision: z.record(z.any()).optional().describe("Final decision"),
    consensus: z.boolean().optional().describe("Whether consensus reached"),
    deliberation_summary: z.string().optional().describe("Summary of deliberations"),
    vote_summary: z.object({
      total_votes: z.number().optional(),
      in_favor: z.number().optional(),
      against: z.number().optional(),
      abstained: z.number().optional(),
    }).optional().describe("Vote summary"),
  }).optional().describe("Action result"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional(),

  metadata: z.object({
    created_at: z.string().datetime().optional(),
    updated_at: z.string().datetime().optional(),
    processing_time_ms: z.number().optional(),
  }).optional(),

  correlation_id: z.string().uuid().optional(),

  timestamp: z.string().datetime(),
});

export type AiCouncilResponse = z.infer<typeof AiCouncilResponse>;

/**
 * Error Model
 */
export const AiCouncilError = z.object({
  code: z.enum([
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
  message: z.string(),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AiCouncilError = z.infer<typeof AiCouncilError>;

/**
 * Validation Functions
 */
export function validateAiCouncilRequest(data: unknown): {
  success: boolean;
  data?: AiCouncilRequest;
  errors?: string[];
} {
  const result = AiCouncilRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function isAiCouncilRequest(data: unknown): data is AiCouncilRequest {
  return typeof data === 'object' && data !== null &&
    'action' in data;
}

/**
 * Examples
 */
export const AiCouncilExamples = {
  validCreateCouncil: {
    council_config: {
      name: "Security Council",
      decision_method: "consensus" as const,
      quorum: 0.67,
      max_deliberation_time_ms: 300000,
    },
    members: [
      {
        member_id: "member_001",
        role: "facilitator" as const,
        expertise: ["security", "policy"],
      },
      {
        member_id: "member_002",
        role: "expert" as const,
        expertise: ["threat_detection"],
      },
    ],
    action: "create_council" as const,
    timeout_ms: 10000,
  } as AiCouncilRequest,

  validProposal: {
    council_id: "council_001",
    action: "propose" as const,
    proposal: {
      proposal_id: "550e8400-e29b-41d4-a716-446655440000",
      proposal_type: "decision" as const,
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
  } as AiCouncilRequest,
};
