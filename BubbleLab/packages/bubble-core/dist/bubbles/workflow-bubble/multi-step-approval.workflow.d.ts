/**
 * MULTI-STEP APPROVAL WORKFLOW
 *
 * Manage multi-step approval processes with routing and notifications.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const ApprovalStatusSchema: z.ZodEnum<["pending", "approved", "rejected", "cancelled"]>;
declare const MultiStepApprovalParamsSchema: z.ZodObject<{
    action: z.ZodEnum<["submit", "approve", "reject", "cancel", "resubmit"]>;
    workflowId: z.ZodOptional<z.ZodString>;
    title: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    requester: z.ZodString;
    approvalSteps: z.ZodArray<z.ZodObject<{
        stepName: z.ZodString;
        approvers: z.ZodArray<z.ZodObject<{
            userId: z.ZodString;
            name: z.ZodString;
            email: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            userId: string;
            email?: string | undefined;
        }, {
            name: string;
            userId: string;
            email?: string | undefined;
        }>, "many">;
        approvalType: z.ZodDefault<z.ZodEnum<["any", "all", "sequence"]>>;
        timeout: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        stepName: string;
        approvers: {
            name: string;
            userId: string;
            email?: string | undefined;
        }[];
        approvalType: "all" | "any" | "sequence";
        timeout?: number | undefined;
    }, {
        stepName: string;
        approvers: {
            name: string;
            userId: string;
            email?: string | undefined;
        }[];
        timeout?: number | undefined;
        approvalType?: "all" | "any" | "sequence" | undefined;
    }>, "many">;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    notifyOnComplete: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    requester: string;
    action: "submit" | "approve" | "reject" | "cancel" | "resubmit";
    approvalSteps: {
        stepName: string;
        approvers: {
            name: string;
            userId: string;
            email?: string | undefined;
        }[];
        approvalType: "all" | "any" | "sequence";
        timeout?: number | undefined;
    }[];
    notifyOnComplete: boolean;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, unknown> | undefined;
    workflowId?: string | undefined;
}, {
    title: string;
    requester: string;
    action: "submit" | "approve" | "reject" | "cancel" | "resubmit";
    approvalSteps: {
        stepName: string;
        approvers: {
            name: string;
            userId: string;
            email?: string | undefined;
        }[];
        timeout?: number | undefined;
        approvalType?: "all" | "any" | "sequence" | undefined;
    }[];
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, unknown> | undefined;
    workflowId?: string | undefined;
    notifyOnComplete?: boolean | undefined;
}>;
type MultiStepApprovalParams = z.input<typeof MultiStepApprovalParamsSchema>;
declare const MultiStepApprovalResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    workflowId: z.ZodString;
    status: z.ZodEnum<["pending", "approved", "rejected", "cancelled"]>;
    currentStep: z.ZodOptional<z.ZodNumber>;
    completedSteps: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    nextStep: z.ZodOptional<z.ZodString>;
    approvals: z.ZodOptional<z.ZodArray<z.ZodObject<{
        stepName: z.ZodString;
        approver: z.ZodString;
        status: z.ZodEnum<["pending", "approved", "rejected", "cancelled"]>;
        timestamp: z.ZodOptional<z.ZodDate>;
        comments: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        status: "cancelled" | "pending" | "approved" | "rejected";
        stepName: string;
        approver: string;
        timestamp?: Date | undefined;
        comments?: string | undefined;
    }, {
        status: "cancelled" | "pending" | "approved" | "rejected";
        stepName: string;
        approver: string;
        timestamp?: Date | undefined;
        comments?: string | undefined;
    }>, "many">>;
    finalDecision: z.ZodOptional<z.ZodEnum<["approved", "rejected", "pending"]>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    status: "cancelled" | "pending" | "approved" | "rejected";
    success: boolean;
    workflowId: string;
    currentStep?: number | undefined;
    completedSteps?: string[] | undefined;
    nextStep?: string | undefined;
    approvals?: {
        status: "cancelled" | "pending" | "approved" | "rejected";
        stepName: string;
        approver: string;
        timestamp?: Date | undefined;
        comments?: string | undefined;
    }[] | undefined;
    finalDecision?: "pending" | "approved" | "rejected" | undefined;
}, {
    error: string;
    status: "cancelled" | "pending" | "approved" | "rejected";
    success: boolean;
    workflowId: string;
    currentStep?: number | undefined;
    completedSteps?: string[] | undefined;
    nextStep?: string | undefined;
    approvals?: {
        status: "cancelled" | "pending" | "approved" | "rejected";
        stepName: string;
        approver: string;
        timestamp?: Date | undefined;
        comments?: string | undefined;
    }[] | undefined;
    finalDecision?: "pending" | "approved" | "rejected" | undefined;
}>;
export declare class MultiStepApprovalWorkflow extends WorkflowBubble<MultiStepApprovalParams, z.infer<typeof MultiStepApprovalResultSchema>> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        action: z.ZodEnum<["submit", "approve", "reject", "cancel", "resubmit"]>;
        workflowId: z.ZodOptional<z.ZodString>;
        title: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        requester: z.ZodString;
        approvalSteps: z.ZodArray<z.ZodObject<{
            stepName: z.ZodString;
            approvers: z.ZodArray<z.ZodObject<{
                userId: z.ZodString;
                name: z.ZodString;
                email: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name: string;
                userId: string;
                email?: string | undefined;
            }, {
                name: string;
                userId: string;
                email?: string | undefined;
            }>, "many">;
            approvalType: z.ZodDefault<z.ZodEnum<["any", "all", "sequence"]>>;
            timeout: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            stepName: string;
            approvers: {
                name: string;
                userId: string;
                email?: string | undefined;
            }[];
            approvalType: "all" | "any" | "sequence";
            timeout?: number | undefined;
        }, {
            stepName: string;
            approvers: {
                name: string;
                userId: string;
                email?: string | undefined;
            }[];
            timeout?: number | undefined;
            approvalType?: "all" | "any" | "sequence" | undefined;
        }>, "many">;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        notifyOnComplete: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        requester: string;
        action: "submit" | "approve" | "reject" | "cancel" | "resubmit";
        approvalSteps: {
            stepName: string;
            approvers: {
                name: string;
                userId: string;
                email?: string | undefined;
            }[];
            approvalType: "all" | "any" | "sequence";
            timeout?: number | undefined;
        }[];
        notifyOnComplete: boolean;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, unknown> | undefined;
        workflowId?: string | undefined;
    }, {
        title: string;
        requester: string;
        action: "submit" | "approve" | "reject" | "cancel" | "resubmit";
        approvalSteps: {
            stepName: string;
            approvers: {
                name: string;
                userId: string;
                email?: string | undefined;
            }[];
            timeout?: number | undefined;
            approvalType?: "all" | "any" | "sequence" | undefined;
        }[];
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, unknown> | undefined;
        workflowId?: string | undefined;
        notifyOnComplete?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        workflowId: z.ZodString;
        status: z.ZodEnum<["pending", "approved", "rejected", "cancelled"]>;
        currentStep: z.ZodOptional<z.ZodNumber>;
        completedSteps: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        nextStep: z.ZodOptional<z.ZodString>;
        approvals: z.ZodOptional<z.ZodArray<z.ZodObject<{
            stepName: z.ZodString;
            approver: z.ZodString;
            status: z.ZodEnum<["pending", "approved", "rejected", "cancelled"]>;
            timestamp: z.ZodOptional<z.ZodDate>;
            comments: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "cancelled" | "pending" | "approved" | "rejected";
            stepName: string;
            approver: string;
            timestamp?: Date | undefined;
            comments?: string | undefined;
        }, {
            status: "cancelled" | "pending" | "approved" | "rejected";
            stepName: string;
            approver: string;
            timestamp?: Date | undefined;
            comments?: string | undefined;
        }>, "many">>;
        finalDecision: z.ZodOptional<z.ZodEnum<["approved", "rejected", "pending"]>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: "cancelled" | "pending" | "approved" | "rejected";
        success: boolean;
        workflowId: string;
        currentStep?: number | undefined;
        completedSteps?: string[] | undefined;
        nextStep?: string | undefined;
        approvals?: {
            status: "cancelled" | "pending" | "approved" | "rejected";
            stepName: string;
            approver: string;
            timestamp?: Date | undefined;
            comments?: string | undefined;
        }[] | undefined;
        finalDecision?: "pending" | "approved" | "rejected" | undefined;
    }, {
        error: string;
        status: "cancelled" | "pending" | "approved" | "rejected";
        success: boolean;
        workflowId: string;
        currentStep?: number | undefined;
        completedSteps?: string[] | undefined;
        nextStep?: string | undefined;
        approvals?: {
            status: "cancelled" | "pending" | "approved" | "rejected";
            stepName: string;
            approver: string;
            timestamp?: Date | undefined;
            comments?: string | undefined;
        }[] | undefined;
        finalDecision?: "pending" | "approved" | "rejected" | undefined;
    }>;
    static readonly shortDescription = "Multi-step approval workflow with routing";
    static readonly longDescription = "\n    Comprehensive multi-step approval workflow with configurable routing, notifications, and timeout handling.\n\n    Features:\n    - Multi-step approval with sequential or parallel approvers\n    - Flexible approval types (any, all, sequence)\n    - Automatic routing between steps\n    - Slack/email notifications\n    - Approval timeout handling\n    - Approval history tracking\n    - Request cancellation and resubmission\n\n    Use cases:\n    - Purchase order approvals\n    - Document review workflows\n    - Access request approvals\n    - Budget approvals\n    - Contract reviews\n  ";
    static readonly alias = "approval-workflow";
    static workflows: Map<string, {
        title: string;
        description?: string;
        requester: string;
        approvalSteps: z.infer<typeof MultiStepApprovalParamsSchema>['approvalSteps'];
        metadata?: Record<string, unknown>;
        currentStep: number;
        status: z.infer<typeof ApprovalStatusSchema>;
        approvals: Array<{
            stepName: string;
            approver: string;
            status: z.infer<typeof ApprovalStatusSchema>;
            timestamp?: Date;
            comments?: string;
        }>;
        createdAt: Date;
    }>;
    constructor(params: MultiStepApprovalParams, context?: BubbleContext);
    protected performAction(): Promise<z.infer<typeof MultiStepApprovalResultSchema>>;
    /**
     * Submit new approval request
     */
    private submitApproval;
    /**
     * Process approval/rejection
     */
    private processApproval;
    /**
     * Cancel approval workflow
     */
    private cancelApproval;
    /**
     * Resubmit approval workflow
     */
    private resubmitApproval;
    /**
     * Initiate approval step (send notifications)
     */
    private initiateApprovalStep;
    /**
     * Notify requester of completion
     */
    private notifyCompletion;
    /**
     * Format approval message
     */
    private formatApprovalMessage;
    /**
     * Generate workflow ID
     */
    private generateWorkflowId;
    /**
     * Get workflow status (utility method)
     */
    static getWorkflowStatus(workflowId: string): ReturnType<typeof MultiStepApprovalWorkflow['workflows']['get']> | undefined;
}
export {};
//# sourceMappingURL=multi-step-approval.workflow.d.ts.map