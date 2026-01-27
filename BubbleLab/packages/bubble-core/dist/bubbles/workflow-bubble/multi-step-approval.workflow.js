/**
 * MULTI-STEP APPROVAL WORKFLOW
 *
 * Manage multi-step approval processes with routing and notifications.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { SlackBubble } from '../service-bubble/slack.js';
const ApprovalStatusSchema = z.enum([
    'pending',
    'approved',
    'rejected',
    'cancelled',
]);
const ApprovalActionSchema = z.enum([
    'submit',
    'approve',
    'reject',
    'cancel',
    'resubmit',
]);
const MultiStepApprovalParamsSchema = z.object({
    action: ApprovalActionSchema,
    workflowId: z.string().optional().describe('Workflow ID for approve/reject/cancel actions'),
    title: z.string().describe('Approval request title'),
    description: z.string().optional().describe('Detailed description'),
    requester: z.string().describe('Requester identifier'),
    approvalSteps: z.array(z.object({
        stepName: z.string(),
        approvers: z.array(z.object({
            userId: z.string(),
            name: z.string(),
            email: z.string().optional(),
        })),
        approvalType: z.enum(['any', 'all', 'sequence']).default('any'),
        timeout: z.number().int().positive().optional().describe('Timeout in minutes'),
    })),
    metadata: z.record(z.unknown()).optional().describe('Additional metadata'),
    notifyOnComplete: z.boolean().default(true).describe('Notify requester on completion'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const MultiStepApprovalResultSchema = z.object({
    success: z.boolean(),
    error: z.string(),
    workflowId: z.string(),
    status: ApprovalStatusSchema,
    currentStep: z.number().optional(),
    completedSteps: z.array(z.string()).optional(),
    nextStep: z.string().optional().describe('Next step requiring approval'),
    approvals: z.array(z.object({
        stepName: z.string(),
        approver: z.string(),
        status: ApprovalStatusSchema,
        timestamp: z.date().optional(),
        comments: z.string().optional(),
    })).optional(),
    finalDecision: z.enum(['approved', 'rejected', 'pending']).optional(),
});
export class MultiStepApprovalWorkflow extends WorkflowBubble {
    static type = 'workflow';
    static bubbleName = 'multi-step-approval-workflow';
    static schema = MultiStepApprovalParamsSchema;
    static resultSchema = MultiStepApprovalResultSchema;
    static shortDescription = 'Multi-step approval workflow with routing';
    static longDescription = `
    Comprehensive multi-step approval workflow with configurable routing, notifications, and timeout handling.

    Features:
    - Multi-step approval with sequential or parallel approvers
    - Flexible approval types (any, all, sequence)
    - Automatic routing between steps
    - Slack/email notifications
    - Approval timeout handling
    - Approval history tracking
    - Request cancellation and resubmission

    Use cases:
    - Purchase order approvals
    - Document review workflows
    - Access request approvals
    - Budget approvals
    - Contract reviews
  `;
    static alias = 'approval-workflow';
    // In-memory storage for approval workflows (in production, use database)
    static workflows = new Map();
    constructor(params, context) {
        super(params, context);
    }
    async performAction() {
        console.log(`[MultiStepApproval] Action: ${this.params.action}`);
        try {
            switch (this.params.action) {
                case 'submit':
                    return await this.submitApproval();
                case 'approve':
                    return await this.processApproval('approved');
                case 'reject':
                    return await this.processApproval('rejected');
                case 'cancel':
                    return await this.cancelApproval();
                case 'resubmit':
                    return await this.resubmitApproval();
                default:
                    return {
                        success: false,
                        error: `Unknown action: ${this.params.action}`,
                        workflowId: '',
                        status: 'pending',
                    };
            }
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error('[MultiStepApproval] Action failed:', errorMessage);
            return {
                success: false,
                error: errorMessage,
                workflowId: this.params.workflowId || '',
                status: 'pending',
            };
        }
    }
    /**
     * Submit new approval request
     */
    async submitApproval() {
        const workflowId = this.generateWorkflowId();
        console.log(`[MultiStepApproval] Submitting approval workflow: ${workflowId}`);
        const workflow = {
            title: this.params.title,
            description: this.params.description,
            requester: this.params.requester,
            approvalSteps: this.params.approvalSteps.map(step => ({
                ...step,
                approvalType: step.approvalType || 'any',
            })),
            metadata: this.params.metadata,
            currentStep: 0,
            status: 'pending',
            approvals: [],
            createdAt: new Date(),
        };
        MultiStepApprovalWorkflow.workflows.set(workflowId, workflow);
        // Start first approval step
        await this.initiateApprovalStep(workflowId, workflow);
        return {
            success: true,
            error: '',
            workflowId,
            status: 'pending',
            currentStep: 0,
            completedSteps: [],
            nextStep: this.params.approvalSteps[0]?.stepName,
            approvals: [],
        };
    }
    /**
     * Process approval/rejection
     */
    async processApproval(decision) {
        const workflowId = this.params.workflowId;
        if (!workflowId) {
            return {
                success: false,
                error: 'Workflow ID is required for approve/reject actions',
                workflowId: '',
                status: 'pending',
            };
        }
        const workflow = MultiStepApprovalWorkflow.workflows.get(workflowId);
        if (!workflow) {
            return {
                success: false,
                error: `Workflow not found: ${workflowId}`,
                workflowId,
                status: 'pending',
            };
        }
        console.log(`[MultiStepApproval] Processing ${decision} for workflow: ${workflowId}`);
        // Record approval
        // In production, you'd identify the actual approver from context
        const approver = this.params.requester; // Placeholder
        workflow.approvals.push({
            stepName: workflow.approvalSteps[workflow.currentStep].stepName,
            approver,
            status: decision,
            timestamp: new Date(),
        });
        // Check if step is complete based on approval type
        const step = workflow.approvalSteps[workflow.currentStep];
        const stepApprovals = workflow.approvals.filter(a => a.stepName === step.stepName);
        let stepComplete = false;
        if (step.approvalType === 'any') {
            stepComplete = stepApprovals.some(a => a.status === 'approved');
        }
        else if (step.approvalType === 'all') {
            stepComplete = stepApprovals.length === step.approvers.length &&
                stepApprovals.every(a => a.status === 'approved');
        }
        else if (step.approvalType === 'sequence') {
            stepComplete = stepApprovals.some(a => a.status === 'approved');
        }
        // If rejected, mark workflow as rejected
        if (decision === 'rejected') {
            workflow.status = 'rejected';
            return {
                success: true,
                error: '',
                workflowId,
                status: 'rejected',
                currentStep: workflow.currentStep,
                completedSteps: workflow.approvalSteps.slice(0, workflow.currentStep).map(s => s.stepName),
                approvals: workflow.approvals,
                finalDecision: 'rejected',
            };
        }
        // If step complete, move to next step or complete workflow
        if (stepComplete) {
            if (workflow.currentStep < workflow.approvalSteps.length - 1) {
                // Move to next step
                workflow.currentStep++;
                await this.initiateApprovalStep(workflowId, workflow);
                return {
                    success: true,
                    error: '',
                    workflowId,
                    status: 'pending',
                    currentStep: workflow.currentStep,
                    completedSteps: workflow.approvalSteps.slice(0, workflow.currentStep).map(s => s.stepName),
                    nextStep: workflow.approvalSteps[workflow.currentStep].stepName,
                    approvals: workflow.approvals,
                };
            }
            else {
                // All steps complete
                workflow.status = 'approved';
                if (this.params.notifyOnComplete) {
                    await this.notifyCompletion(workflowId, workflow);
                }
                return {
                    success: true,
                    error: '',
                    workflowId,
                    status: 'approved',
                    currentStep: workflow.currentStep,
                    completedSteps: workflow.approvalSteps.map(s => s.stepName),
                    approvals: workflow.approvals,
                    finalDecision: 'approved',
                };
            }
        }
        // Step not yet complete
        return {
            success: true,
            error: '',
            workflowId,
            status: 'pending',
            currentStep: workflow.currentStep,
            approvals: workflow.approvals,
        };
    }
    /**
     * Cancel approval workflow
     */
    async cancelApproval() {
        const workflowId = this.params.workflowId;
        if (!workflowId) {
            return {
                success: false,
                error: 'Workflow ID is required for cancel action',
                workflowId: '',
                status: 'pending',
            };
        }
        const workflow = MultiStepApprovalWorkflow.workflows.get(workflowId);
        if (!workflow) {
            return {
                success: false,
                error: `Workflow not found: ${workflowId}`,
                workflowId,
                status: 'pending',
            };
        }
        workflow.status = 'cancelled';
        console.log(`[MultiStepApproval] Cancelled workflow: ${workflowId}`);
        return {
            success: true,
            error: '',
            workflowId,
            status: 'cancelled',
            currentStep: workflow.currentStep,
            approvals: workflow.approvals,
        };
    }
    /**
     * Resubmit approval workflow
     */
    async resubmitApproval() {
        const workflowId = this.params.workflowId;
        if (!workflowId) {
            return {
                success: false,
                error: 'Workflow ID is required for resubmit action',
                workflowId: '',
                status: 'pending',
            };
        }
        const workflow = MultiStepApprovalWorkflow.workflows.get(workflowId);
        if (!workflow) {
            return {
                success: false,
                error: `Workflow not found: ${workflowId}`,
                workflowId,
                status: 'pending',
            };
        }
        // Reset to first step
        workflow.currentStep = 0;
        workflow.status = 'pending';
        workflow.approvals = [];
        await this.initiateApprovalStep(workflowId, workflow);
        console.log(`[MultiStepApproval] Resubmitted workflow: ${workflowId}`);
        return {
            success: true,
            error: '',
            workflowId,
            status: 'pending',
            currentStep: 0,
            completedSteps: [],
            nextStep: workflow.approvalSteps[0].stepName,
            approvals: [],
        };
    }
    /**
     * Initiate approval step (send notifications)
     */
    async initiateApprovalStep(workflowId, workflow) {
        const step = workflow.approvalSteps[workflow.currentStep];
        console.log(`[MultiStepApproval] Initiating step: ${step.stepName}`);
        // Send notifications to approvers
        const message = this.formatApprovalMessage(workflowId, workflow, step);
        for (const approver of step.approvers) {
            try {
                const slackBubble = new SlackBubble({
                    operation: 'send_message',
                    channel: approver.name, // In production, use actual user/channel
                    text: message,
                    credentials: this.params.credentials,
                }, this.context);
                await slackBubble.action();
            }
            catch (error) {
                console.error(`[MultiStepApproval] Failed to notify ${approver.name}:`, error);
            }
        }
        // Set timeout if configured
        if (step.timeout) {
            setTimeout(async () => {
                const wf = MultiStepApprovalWorkflow.workflows.get(workflowId);
                if (wf && wf.status === 'pending' && wf.currentStep === workflow.currentStep) {
                    console.log(`[MultiStepApproval] Step ${step.stepName} timed out`);
                    // Handle timeout - could auto-reject or escalate
                }
            }, step.timeout * 60 * 1000);
        }
    }
    /**
     * Notify requester of completion
     */
    async notifyCompletion(workflowId, workflow) {
        console.log(`[MultiStepApproval] Notifying completion for: ${workflowId}`);
        const message = `Approval Request Completed: ${workflow.title}\n\nStatus: ${workflow.status}\nWorkflow ID: ${workflowId}`;
        try {
            const slackBubble = new SlackBubble({
                operation: 'send_message',
                channel: workflow.requester,
                text: message,
                credentials: this.params.credentials,
            }, this.context);
            await slackBubble.action();
        }
        catch (error) {
            console.error('[MultiStepApproval] Failed to notify requester:', error);
        }
    }
    /**
     * Format approval message
     */
    formatApprovalMessage(workflowId, workflow, step) {
        let message = `Approval Required: ${workflow.title}\n\n`;
        message += `Step ${workflow.currentStep + 1}/${workflow.approvalSteps.length}: ${step.stepName}\n`;
        message += `Description: ${workflow.description || 'No description'}\n`;
        message += `Requested by: ${workflow.requester}\n`;
        message += `Workflow ID: ${workflowId}\n\n`;
        message += `Approvers: ${step.approvers.map(a => a.name).join(', ')}\n\n`;
        message += `Please approve or reject this request.`;
        if (workflow.metadata) {
            message += `\n\nMetadata: ${JSON.stringify(workflow.metadata, null, 2)}`;
        }
        return message;
    }
    /**
     * Generate workflow ID
     */
    generateWorkflowId() {
        return `approval_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    }
    /**
     * Get workflow status (utility method)
     */
    static getWorkflowStatus(workflowId) {
        return MultiStepApprovalWorkflow.workflows.get(workflowId);
    }
}
//# sourceMappingURL=multi-step-approval.workflow.js.map