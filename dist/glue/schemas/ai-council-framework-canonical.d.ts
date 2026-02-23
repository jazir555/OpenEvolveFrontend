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
export declare const CouncilRole: z.ZodEnum<["facilitator", "expert", "critic", "advocate", "observer", "voter"]>;
export type CouncilRole = z.infer<typeof CouncilRole>;
/**
 * Council State Enum
 */
export declare const CouncilState: z.ZodEnum<["forming", "deliberating", "voting", "consensus", "dissolved"]>;
export type CouncilState = z.infer<typeof CouncilState>;
/**
 * Decision Method Enum
 */
export declare const DecisionMethod: z.ZodEnum<["consensus", "majority_vote", "weighted_vote", "expert_weighted", "facilitator_decides"]>;
export type DecisionMethod = z.infer<typeof DecisionMethod>;
/**
 * Council Member Schema
 */
export declare const CouncilMember: z.ZodObject<{
    member_id: z.ZodString;
    agent_id: z.ZodOptional<z.ZodString>;
    role: z.ZodEnum<["facilitator", "expert", "critic", "advocate", "observer", "voter"]>;
    expertise: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    weight: z.ZodOptional<z.ZodNumber>;
    config: z.ZodOptional<z.ZodObject<{
        participation_required: z.ZodOptional<z.ZodBoolean>;
        can_vote: z.ZodOptional<z.ZodBoolean>;
        can_propose: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        participation_required?: boolean | undefined;
        can_vote?: boolean | undefined;
        can_propose?: boolean | undefined;
    }, {
        participation_required?: boolean | undefined;
        can_vote?: boolean | undefined;
        can_propose?: boolean | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
    member_id: string;
    config?: {
        participation_required?: boolean | undefined;
        can_vote?: boolean | undefined;
        can_propose?: boolean | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    weight?: number | undefined;
    agent_id?: string | undefined;
    expertise?: string[] | undefined;
}, {
    role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
    member_id: string;
    config?: {
        participation_required?: boolean | undefined;
        can_vote?: boolean | undefined;
        can_propose?: boolean | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    weight?: number | undefined;
    agent_id?: string | undefined;
    expertise?: string[] | undefined;
}>;
export type CouncilMember = z.infer<typeof CouncilMember>;
/**
 * Council Proposal Schema
 */
export declare const CouncilProposal: z.ZodObject<{
    proposal_id: z.ZodOptional<z.ZodString>;
    proposal_type: z.ZodEnum<["decision", "action", "policy", "recommendation"]>;
    title: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    proposer_id: z.ZodString;
    content: z.ZodRecord<z.ZodString, z.ZodAny>;
    options: z.ZodOptional<z.ZodArray<z.ZodObject<{
        option_id: z.ZodString;
        description: z.ZodString;
        data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        description: string;
        option_id: string;
        data?: Record<string, any> | undefined;
    }, {
        description: string;
        option_id: string;
        data?: Record<string, any> | undefined;
    }>, "many">>;
    deadline: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    content: Record<string, any>;
    title: string;
    proposal_type: "action" | "decision" | "policy" | "recommendation";
    proposer_id: string;
    metadata?: Record<string, any> | undefined;
    description?: string | undefined;
    options?: {
        description: string;
        option_id: string;
        data?: Record<string, any> | undefined;
    }[] | undefined;
    proposal_id?: string | undefined;
    deadline?: string | undefined;
}, {
    content: Record<string, any>;
    title: string;
    proposal_type: "action" | "decision" | "policy" | "recommendation";
    proposer_id: string;
    metadata?: Record<string, any> | undefined;
    description?: string | undefined;
    options?: {
        description: string;
        option_id: string;
        data?: Record<string, any> | undefined;
    }[] | undefined;
    proposal_id?: string | undefined;
    deadline?: string | undefined;
}>;
export type CouncilProposal = z.infer<typeof CouncilProposal>;
/**
 * Council Vote Schema
 */
export declare const CouncilVote: z.ZodObject<{
    vote_id: z.ZodOptional<z.ZodString>;
    proposal_id: z.ZodString;
    member_id: z.ZodString;
    decision: z.ZodUnion<[z.ZodBoolean, z.ZodString, z.ZodNumber]>;
    rationale: z.ZodOptional<z.ZodString>;
    confidence: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    member_id: string;
    proposal_id: string;
    decision: string | number | boolean;
    timestamp?: string | undefined;
    metadata?: Record<string, any> | undefined;
    confidence?: number | undefined;
    vote_id?: string | undefined;
    rationale?: string | undefined;
}, {
    member_id: string;
    proposal_id: string;
    decision: string | number | boolean;
    timestamp?: string | undefined;
    metadata?: Record<string, any> | undefined;
    confidence?: number | undefined;
    vote_id?: string | undefined;
    rationale?: string | undefined;
}>;
export type CouncilVote = z.infer<typeof CouncilVote>;
/**
 * Council Session Request Schema
 */
export declare const AiCouncilRequest: z.ZodObject<{
    council_id: z.ZodOptional<z.ZodString>;
    session_id: z.ZodOptional<z.ZodString>;
    action: z.ZodEnum<["create_council", "add_member", "remove_member", "propose", "vote", "deliberate", "decide", "query_state"]>;
    council_config: z.ZodOptional<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        decision_method: z.ZodOptional<z.ZodEnum<["consensus", "majority_vote", "weighted_vote", "expert_weighted", "facilitator_decides"]>>;
        quorum: z.ZodOptional<z.ZodNumber>;
        max_deliberation_time_ms: z.ZodOptional<z.ZodNumber>;
        voting_timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        description?: string | undefined;
        decision_method?: "consensus" | "majority_vote" | "weighted_vote" | "expert_weighted" | "facilitator_decides" | undefined;
        quorum?: number | undefined;
        max_deliberation_time_ms?: number | undefined;
        voting_timeout_ms?: number | undefined;
    }, {
        name?: string | undefined;
        description?: string | undefined;
        decision_method?: "consensus" | "majority_vote" | "weighted_vote" | "expert_weighted" | "facilitator_decides" | undefined;
        quorum?: number | undefined;
        max_deliberation_time_ms?: number | undefined;
        voting_timeout_ms?: number | undefined;
    }>>;
    members: z.ZodOptional<z.ZodArray<z.ZodObject<{
        member_id: z.ZodString;
        agent_id: z.ZodOptional<z.ZodString>;
        role: z.ZodEnum<["facilitator", "expert", "critic", "advocate", "observer", "voter"]>;
        expertise: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        weight: z.ZodOptional<z.ZodNumber>;
        config: z.ZodOptional<z.ZodObject<{
            participation_required: z.ZodOptional<z.ZodBoolean>;
            can_vote: z.ZodOptional<z.ZodBoolean>;
            can_propose: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            participation_required?: boolean | undefined;
            can_vote?: boolean | undefined;
            can_propose?: boolean | undefined;
        }, {
            participation_required?: boolean | undefined;
            can_vote?: boolean | undefined;
            can_propose?: boolean | undefined;
        }>>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
        member_id: string;
        config?: {
            participation_required?: boolean | undefined;
            can_vote?: boolean | undefined;
            can_propose?: boolean | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        weight?: number | undefined;
        agent_id?: string | undefined;
        expertise?: string[] | undefined;
    }, {
        role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
        member_id: string;
        config?: {
            participation_required?: boolean | undefined;
            can_vote?: boolean | undefined;
            can_propose?: boolean | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        weight?: number | undefined;
        agent_id?: string | undefined;
        expertise?: string[] | undefined;
    }>, "many">>;
    proposal: z.ZodOptional<z.ZodObject<{
        proposal_id: z.ZodOptional<z.ZodString>;
        proposal_type: z.ZodEnum<["decision", "action", "policy", "recommendation"]>;
        title: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        proposer_id: z.ZodString;
        content: z.ZodRecord<z.ZodString, z.ZodAny>;
        options: z.ZodOptional<z.ZodArray<z.ZodObject<{
            option_id: z.ZodString;
            description: z.ZodString;
            data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            description: string;
            option_id: string;
            data?: Record<string, any> | undefined;
        }, {
            description: string;
            option_id: string;
            data?: Record<string, any> | undefined;
        }>, "many">>;
        deadline: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        content: Record<string, any>;
        title: string;
        proposal_type: "action" | "decision" | "policy" | "recommendation";
        proposer_id: string;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        options?: {
            description: string;
            option_id: string;
            data?: Record<string, any> | undefined;
        }[] | undefined;
        proposal_id?: string | undefined;
        deadline?: string | undefined;
    }, {
        content: Record<string, any>;
        title: string;
        proposal_type: "action" | "decision" | "policy" | "recommendation";
        proposer_id: string;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        options?: {
            description: string;
            option_id: string;
            data?: Record<string, any> | undefined;
        }[] | undefined;
        proposal_id?: string | undefined;
        deadline?: string | undefined;
    }>>;
    vote: z.ZodOptional<z.ZodObject<{
        vote_id: z.ZodOptional<z.ZodString>;
        proposal_id: z.ZodString;
        member_id: z.ZodString;
        decision: z.ZodUnion<[z.ZodBoolean, z.ZodString, z.ZodNumber]>;
        rationale: z.ZodOptional<z.ZodString>;
        confidence: z.ZodOptional<z.ZodNumber>;
        timestamp: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        member_id: string;
        proposal_id: string;
        decision: string | number | boolean;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        confidence?: number | undefined;
        vote_id?: string | undefined;
        rationale?: string | undefined;
    }, {
        member_id: string;
        proposal_id: string;
        decision: string | number | boolean;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        confidence?: number | undefined;
        vote_id?: string | undefined;
        rationale?: string | undefined;
    }>>;
    deliberation_config: z.ZodOptional<z.ZodObject<{
        topic: z.ZodOptional<z.ZodString>;
        context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        max_rounds: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        context?: Record<string, any> | undefined;
        topic?: string | undefined;
        max_rounds?: number | undefined;
    }, {
        context?: Record<string, any> | undefined;
        topic?: string | undefined;
        max_rounds?: number | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    action: "query_state" | "create_council" | "add_member" | "remove_member" | "propose" | "vote" | "deliberate" | "decide";
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    members?: {
        role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
        member_id: string;
        config?: {
            participation_required?: boolean | undefined;
            can_vote?: boolean | undefined;
            can_propose?: boolean | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        weight?: number | undefined;
        agent_id?: string | undefined;
        expertise?: string[] | undefined;
    }[] | undefined;
    session_id?: string | undefined;
    council_id?: string | undefined;
    vote?: {
        member_id: string;
        proposal_id: string;
        decision: string | number | boolean;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        confidence?: number | undefined;
        vote_id?: string | undefined;
        rationale?: string | undefined;
    } | undefined;
    council_config?: {
        name?: string | undefined;
        description?: string | undefined;
        decision_method?: "consensus" | "majority_vote" | "weighted_vote" | "expert_weighted" | "facilitator_decides" | undefined;
        quorum?: number | undefined;
        max_deliberation_time_ms?: number | undefined;
        voting_timeout_ms?: number | undefined;
    } | undefined;
    proposal?: {
        content: Record<string, any>;
        title: string;
        proposal_type: "action" | "decision" | "policy" | "recommendation";
        proposer_id: string;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        options?: {
            description: string;
            option_id: string;
            data?: Record<string, any> | undefined;
        }[] | undefined;
        proposal_id?: string | undefined;
        deadline?: string | undefined;
    } | undefined;
    deliberation_config?: {
        context?: Record<string, any> | undefined;
        topic?: string | undefined;
        max_rounds?: number | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    action: "query_state" | "create_council" | "add_member" | "remove_member" | "propose" | "vote" | "deliberate" | "decide";
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    members?: {
        role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
        member_id: string;
        config?: {
            participation_required?: boolean | undefined;
            can_vote?: boolean | undefined;
            can_propose?: boolean | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        weight?: number | undefined;
        agent_id?: string | undefined;
        expertise?: string[] | undefined;
    }[] | undefined;
    session_id?: string | undefined;
    council_id?: string | undefined;
    vote?: {
        member_id: string;
        proposal_id: string;
        decision: string | number | boolean;
        timestamp?: string | undefined;
        metadata?: Record<string, any> | undefined;
        confidence?: number | undefined;
        vote_id?: string | undefined;
        rationale?: string | undefined;
    } | undefined;
    council_config?: {
        name?: string | undefined;
        description?: string | undefined;
        decision_method?: "consensus" | "majority_vote" | "weighted_vote" | "expert_weighted" | "facilitator_decides" | undefined;
        quorum?: number | undefined;
        max_deliberation_time_ms?: number | undefined;
        voting_timeout_ms?: number | undefined;
    } | undefined;
    proposal?: {
        content: Record<string, any>;
        title: string;
        proposal_type: "action" | "decision" | "policy" | "recommendation";
        proposer_id: string;
        metadata?: Record<string, any> | undefined;
        description?: string | undefined;
        options?: {
            description: string;
            option_id: string;
            data?: Record<string, any> | undefined;
        }[] | undefined;
        proposal_id?: string | undefined;
        deadline?: string | undefined;
    } | undefined;
    deliberation_config?: {
        context?: Record<string, any> | undefined;
        topic?: string | undefined;
        max_rounds?: number | undefined;
    } | undefined;
}>;
export type AiCouncilRequest = z.infer<typeof AiCouncilRequest>;
/**
 * Council Session Response Schema
 */
export declare const AiCouncilResponse: z.ZodObject<{
    council_id: z.ZodString;
    session_id: z.ZodOptional<z.ZodString>;
    action: z.ZodEnum<["create_council", "add_member", "remove_member", "propose", "vote", "deliberate", "decide", "query_state"]>;
    status: z.ZodEnum<["success", "failed", "timeout", "pending"]>;
    council_state: z.ZodOptional<z.ZodEnum<["forming", "deliberating", "voting", "consensus", "dissolved"]>>;
    result: z.ZodOptional<z.ZodObject<{
        members: z.ZodOptional<z.ZodArray<z.ZodObject<{
            member_id: z.ZodString;
            agent_id: z.ZodOptional<z.ZodString>;
            role: z.ZodEnum<["facilitator", "expert", "critic", "advocate", "observer", "voter"]>;
            expertise: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            weight: z.ZodOptional<z.ZodNumber>;
            config: z.ZodOptional<z.ZodObject<{
                participation_required: z.ZodOptional<z.ZodBoolean>;
                can_vote: z.ZodOptional<z.ZodBoolean>;
                can_propose: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            }, {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            }>>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
            member_id: string;
            config?: {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            } | undefined;
            metadata?: Record<string, any> | undefined;
            weight?: number | undefined;
            agent_id?: string | undefined;
            expertise?: string[] | undefined;
        }, {
            role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
            member_id: string;
            config?: {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            } | undefined;
            metadata?: Record<string, any> | undefined;
            weight?: number | undefined;
            agent_id?: string | undefined;
            expertise?: string[] | undefined;
        }>, "many">>;
        proposal: z.ZodOptional<z.ZodObject<{
            proposal_id: z.ZodOptional<z.ZodString>;
            proposal_type: z.ZodEnum<["decision", "action", "policy", "recommendation"]>;
            title: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            proposer_id: z.ZodString;
            content: z.ZodRecord<z.ZodString, z.ZodAny>;
            options: z.ZodOptional<z.ZodArray<z.ZodObject<{
                option_id: z.ZodString;
                description: z.ZodString;
                data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            }, "strip", z.ZodTypeAny, {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }, {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }>, "many">>;
            deadline: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            content: Record<string, any>;
            title: string;
            proposal_type: "action" | "decision" | "policy" | "recommendation";
            proposer_id: string;
            metadata?: Record<string, any> | undefined;
            description?: string | undefined;
            options?: {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }[] | undefined;
            proposal_id?: string | undefined;
            deadline?: string | undefined;
        }, {
            content: Record<string, any>;
            title: string;
            proposal_type: "action" | "decision" | "policy" | "recommendation";
            proposer_id: string;
            metadata?: Record<string, any> | undefined;
            description?: string | undefined;
            options?: {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }[] | undefined;
            proposal_id?: string | undefined;
            deadline?: string | undefined;
        }>>;
        votes: z.ZodOptional<z.ZodArray<z.ZodObject<{
            vote_id: z.ZodOptional<z.ZodString>;
            proposal_id: z.ZodString;
            member_id: z.ZodString;
            decision: z.ZodUnion<[z.ZodBoolean, z.ZodString, z.ZodNumber]>;
            rationale: z.ZodOptional<z.ZodString>;
            confidence: z.ZodOptional<z.ZodNumber>;
            timestamp: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            member_id: string;
            proposal_id: string;
            decision: string | number | boolean;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            confidence?: number | undefined;
            vote_id?: string | undefined;
            rationale?: string | undefined;
        }, {
            member_id: string;
            proposal_id: string;
            decision: string | number | boolean;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            confidence?: number | undefined;
            vote_id?: string | undefined;
            rationale?: string | undefined;
        }>, "many">>;
        decision: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        consensus: z.ZodOptional<z.ZodBoolean>;
        deliberation_summary: z.ZodOptional<z.ZodString>;
        vote_summary: z.ZodOptional<z.ZodObject<{
            total_votes: z.ZodOptional<z.ZodNumber>;
            in_favor: z.ZodOptional<z.ZodNumber>;
            against: z.ZodOptional<z.ZodNumber>;
            abstained: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            total_votes?: number | undefined;
            in_favor?: number | undefined;
            against?: number | undefined;
            abstained?: number | undefined;
        }, {
            total_votes?: number | undefined;
            in_favor?: number | undefined;
            against?: number | undefined;
            abstained?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        members?: {
            role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
            member_id: string;
            config?: {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            } | undefined;
            metadata?: Record<string, any> | undefined;
            weight?: number | undefined;
            agent_id?: string | undefined;
            expertise?: string[] | undefined;
        }[] | undefined;
        consensus?: boolean | undefined;
        decision?: Record<string, any> | undefined;
        proposal?: {
            content: Record<string, any>;
            title: string;
            proposal_type: "action" | "decision" | "policy" | "recommendation";
            proposer_id: string;
            metadata?: Record<string, any> | undefined;
            description?: string | undefined;
            options?: {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }[] | undefined;
            proposal_id?: string | undefined;
            deadline?: string | undefined;
        } | undefined;
        votes?: {
            member_id: string;
            proposal_id: string;
            decision: string | number | boolean;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            confidence?: number | undefined;
            vote_id?: string | undefined;
            rationale?: string | undefined;
        }[] | undefined;
        deliberation_summary?: string | undefined;
        vote_summary?: {
            total_votes?: number | undefined;
            in_favor?: number | undefined;
            against?: number | undefined;
            abstained?: number | undefined;
        } | undefined;
    }, {
        members?: {
            role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
            member_id: string;
            config?: {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            } | undefined;
            metadata?: Record<string, any> | undefined;
            weight?: number | undefined;
            agent_id?: string | undefined;
            expertise?: string[] | undefined;
        }[] | undefined;
        consensus?: boolean | undefined;
        decision?: Record<string, any> | undefined;
        proposal?: {
            content: Record<string, any>;
            title: string;
            proposal_type: "action" | "decision" | "policy" | "recommendation";
            proposer_id: string;
            metadata?: Record<string, any> | undefined;
            description?: string | undefined;
            options?: {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }[] | undefined;
            proposal_id?: string | undefined;
            deadline?: string | undefined;
        } | undefined;
        votes?: {
            member_id: string;
            proposal_id: string;
            decision: string | number | boolean;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            confidence?: number | undefined;
            vote_id?: string | undefined;
            rationale?: string | undefined;
        }[] | undefined;
        deliberation_summary?: string | undefined;
        vote_summary?: {
            total_votes?: number | undefined;
            in_favor?: number | undefined;
            against?: number | undefined;
            abstained?: number | undefined;
        } | undefined;
    }>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    }, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "success" | "failed" | "pending" | "timeout";
    action: "query_state" | "create_council" | "add_member" | "remove_member" | "propose" | "vote" | "deliberate" | "decide";
    council_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        members?: {
            role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
            member_id: string;
            config?: {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            } | undefined;
            metadata?: Record<string, any> | undefined;
            weight?: number | undefined;
            agent_id?: string | undefined;
            expertise?: string[] | undefined;
        }[] | undefined;
        consensus?: boolean | undefined;
        decision?: Record<string, any> | undefined;
        proposal?: {
            content: Record<string, any>;
            title: string;
            proposal_type: "action" | "decision" | "policy" | "recommendation";
            proposer_id: string;
            metadata?: Record<string, any> | undefined;
            description?: string | undefined;
            options?: {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }[] | undefined;
            proposal_id?: string | undefined;
            deadline?: string | undefined;
        } | undefined;
        votes?: {
            member_id: string;
            proposal_id: string;
            decision: string | number | boolean;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            confidence?: number | undefined;
            vote_id?: string | undefined;
            rationale?: string | undefined;
        }[] | undefined;
        deliberation_summary?: string | undefined;
        vote_summary?: {
            total_votes?: number | undefined;
            in_favor?: number | undefined;
            against?: number | undefined;
            abstained?: number | undefined;
        } | undefined;
    } | undefined;
    session_id?: string | undefined;
    council_state?: "forming" | "deliberating" | "voting" | "consensus" | "dissolved" | undefined;
}, {
    timestamp: string;
    status: "success" | "failed" | "pending" | "timeout";
    action: "query_state" | "create_council" | "add_member" | "remove_member" | "propose" | "vote" | "deliberate" | "decide";
    council_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        members?: {
            role: "expert" | "facilitator" | "critic" | "advocate" | "observer" | "voter";
            member_id: string;
            config?: {
                participation_required?: boolean | undefined;
                can_vote?: boolean | undefined;
                can_propose?: boolean | undefined;
            } | undefined;
            metadata?: Record<string, any> | undefined;
            weight?: number | undefined;
            agent_id?: string | undefined;
            expertise?: string[] | undefined;
        }[] | undefined;
        consensus?: boolean | undefined;
        decision?: Record<string, any> | undefined;
        proposal?: {
            content: Record<string, any>;
            title: string;
            proposal_type: "action" | "decision" | "policy" | "recommendation";
            proposer_id: string;
            metadata?: Record<string, any> | undefined;
            description?: string | undefined;
            options?: {
                description: string;
                option_id: string;
                data?: Record<string, any> | undefined;
            }[] | undefined;
            proposal_id?: string | undefined;
            deadline?: string | undefined;
        } | undefined;
        votes?: {
            member_id: string;
            proposal_id: string;
            decision: string | number | boolean;
            timestamp?: string | undefined;
            metadata?: Record<string, any> | undefined;
            confidence?: number | undefined;
            vote_id?: string | undefined;
            rationale?: string | undefined;
        }[] | undefined;
        deliberation_summary?: string | undefined;
        vote_summary?: {
            total_votes?: number | undefined;
            in_favor?: number | undefined;
            against?: number | undefined;
            abstained?: number | undefined;
        } | undefined;
    } | undefined;
    session_id?: string | undefined;
    council_state?: "forming" | "deliberating" | "voting" | "consensus" | "dissolved" | undefined;
}>;
export type AiCouncilResponse = z.infer<typeof AiCouncilResponse>;
/**
 * Error Model
 */
export declare const AiCouncilError: z.ZodObject<{
    code: z.ZodEnum<["COUNCIL_NOT_FOUND", "MEMBER_NOT_FOUND", "PROPOSAL_NOT_FOUND", "QUORUM_NOT_MET", "DELIBERATION_TIMEOUT", "INVALID_VOTE", "MEMBER_ALREADY_EXISTS", "COUNCIL_DISSOLVED", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "COUNCIL_NOT_FOUND" | "MEMBER_NOT_FOUND" | "PROPOSAL_NOT_FOUND" | "QUORUM_NOT_MET" | "DELIBERATION_TIMEOUT" | "INVALID_VOTE" | "MEMBER_ALREADY_EXISTS" | "COUNCIL_DISSOLVED";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "COUNCIL_NOT_FOUND" | "MEMBER_NOT_FOUND" | "PROPOSAL_NOT_FOUND" | "QUORUM_NOT_MET" | "DELIBERATION_TIMEOUT" | "INVALID_VOTE" | "MEMBER_ALREADY_EXISTS" | "COUNCIL_DISSOLVED";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type AiCouncilError = z.infer<typeof AiCouncilError>;
/**
 * Validation Functions
 */
export declare function validateAiCouncilRequest(data: unknown): {
    success: boolean;
    data?: AiCouncilRequest;
    errors?: string[];
};
export declare function isAiCouncilRequest(data: unknown): data is AiCouncilRequest;
/**
 * Examples
 */
export declare const AiCouncilExamples: {
    validCreateCouncil: AiCouncilRequest;
    validProposal: AiCouncilRequest;
};
//# sourceMappingURL=ai-council-framework-canonical.d.ts.map