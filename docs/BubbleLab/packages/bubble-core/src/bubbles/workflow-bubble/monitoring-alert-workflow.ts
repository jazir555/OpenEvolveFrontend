import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * MonitoringAlertWorkflow - System monitoring and alert escalation
 */
export class MonitoringAlertWorkflow extends WorkflowBubble<MonitoringAlertParams, MonitoringAlertResult> {
  bubbleName = 'monitoring-alert';
  type = 'workflow';
  alias = 'monitoring-alert';

  params = {
    timeout: z.number().int().positive().default(300000)
  };

  async execute(input: any): Promise<MonitoringAlertResult> {
    const steps = [];

    try {
      // Step 1: Monitor
      const step1Result = await this.monitor(input);
      steps.push({
        step: 1,
        name: 'monitor',
        status: 'completed',
        result: step1Result
      });

      // Step 2: Alert
      const step2Result = await this.alert({ ...input, metrics: step1Result });
      steps.push({
        step: 2,
        name: 'alert',
        status: 'completed',
        result: step2Result
      });

      // Step 3: Escalate
      const step3Result = await this.escalate({ ...input, alert: step2Result });
      steps.push({
        step: 3,
        name: 'escalate',
        status: 'completed',
        result: step3Result
      });

      return { success: true, steps };
    } catch (error: any) {
      return { success: false, error: error.message, steps };
    }
  }

  async monitor(params: { targets: string[] }): Promise<MonitoringAlertResult> {
    try {
      const metrics = params.targets.map(target => ({
        target,
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        status: Math.random() > 0.8 ? 'warning' : 'healthy'
      }));
      return { success: true, metrics };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async alert(params: { metrics: any[]; threshold?: number }): Promise<MonitoringAlertResult> {
    try {
      const threshold = params.threshold || 80;
      const alerts = params.metrics.filter(m => m.cpu > threshold || m.memory > threshold);
      const alert = {
        triggered: alerts.length > 0,
        count: alerts.length,
        alerts: alerts.map(a => ({ target: a.target, reason: `CPU: ${a.cpu.toFixed(1)}%` }))
      };
      return { success: true, alert };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async escalate(params: { alert: any; level?: 'info' | 'warning' | 'critical' }): Promise<MonitoringAlertResult> {
    try {
      const escalated = {
        level: params.level || (params.alert.count > 5 ? 'critical' : 'warning'),
        notified: ['ops-team'],
        timestamp: new Date().toISOString(),
        resolved: false
      };
      return { success: true, escalated };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface MonitoringAlertParams {
  timeout?: number;
}

export interface MonitoringAlertResult {
  success: boolean;
  metrics?: any[];
  alert?: any;
  escalated?: any;
  steps?: any[];
  error?: string;
}
