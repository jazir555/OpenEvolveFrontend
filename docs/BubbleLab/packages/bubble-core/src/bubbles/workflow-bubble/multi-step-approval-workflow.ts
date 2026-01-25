import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * MultiStepApprovalWorkflow - Multi-level approval workflow
 */
export class MultiStepApprovalWorkflow extends WorkflowBubble<MultiStepApprovalParams, MultiStepApprovalResult> {
  bubbleName = 'multi-step-approval';
  type = 'workflow';
  alias = 'multi-step-approval';

  params = {
    timeout: z.number().int().positive().default(300000)
  };

  async execute(input: any): Promise<MultiStepApprovalResult> {
    const steps = [];

    try {
      // Step 1: Submit
      const step1Result = await this.submit(input);
      steps.push({
        step: 1,
        name: 'submit',
        status: 'completed',
        result: step1Result
      });

      // Step 2: Approve
      const step2Result = await this.approve({ ...input, submission: step1Result });
      steps.push({
        step: 2,
        name: 'approve',
        status: 'completed',
        result: step2Result
      });

      // Step 3: Reject
      const step3Result = await this.reject({ ...input, approval: step2Result });
      steps.push({
        step: 3,
        name: 'reject',
        status: 'completed',
        result: step3Result
      });

      // Step 4: Notify
      const step4Result = await this.notify({ ...input, rejection: step3Result });
      steps.push({
        step: 4,
        name: 'notify',
        status: 'completed',
        result: step4Result
      });

      return { success: true, steps };
    } catch (error: any) {
      return { success: false, error: error.message, steps };
    }
  }

  async submit(params: { request: any; requester: string }): Promise<MultiStepApprovalResult> {
    try {
      const submission = {
        id: `req_${Date.now()}`,
        request: params.request,
        requester: params.requester,
        status: 'pending',
        submittedAt: new Date().toISOString(),
        currentLevel: 1
      };
      return { success: true, submission };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async approve(params: { submission: any; approver: string }): Promise<MultiStepApprovalResult> {
    try {
      const approval = {
        requestId: params.submission.id,
        approver: params.approver,
        level: params.submission.currentLevel,
        status: 'approved',
        approvedAt: new Date().toISOString(),
        comments: 'Approved at current level'
      };
      return { success: true, approval };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async reject(params: { approval: any; reason: string }): Promise<MultiStepApprovalResult> {
    try {
      const rejection = {
        requestId: params.approval.requestId,
        rejectedBy: params.approval.approver,
        reason: params.reason,
        rejectedAt: new Date().toISOString(),
        status: 'rejected'
      };
      return { success: true, rejection };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async notify(params: { rejection: any }): Promise<MultiStepApprovalResult> {
    try {
      const notification = {
        requestId: params.rejection.requestId,
        notified: [params.rejection.rejectedBy, 'requester'],
        message: `Request ${params.rejection.requestId} was rejected: ${params.rejection.reason}`,
        notifiedAt: new Date().toISOString()
      };
      return { success: true, notification };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface MultiStepApprovalParams {
  timeout?: number;
}

export interface MultiStepApprovalResult {
  success: boolean;
  submission?: any;
  approval?: any;
  rejection?: any;
  notification?: any;
  steps?: any[];
  error?: string;
}
