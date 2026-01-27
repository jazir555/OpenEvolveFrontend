import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
const ScheduleTypeSchema = z.enum(['cron', 'interval', 'once']);
const ScheduledTaskParamsSchema = z.object({
    taskName: z.string(),
    schedule: z.object({
        type: ScheduleTypeSchema,
        expression: z.string().optional().describe('Cron expression or interval'),
        runAt: z.date().optional().describe('Specific run time for "once" type'),
    }),
    action: z.object({
        type: z.enum(['http', 'workflow', 'function']),
        config: z.record(z.unknown()),
    }),
    timeout: z.number().int().positive().default(300000),
    retryOnFailure: z.boolean().default(false),
    maxRetries: z.number().int().default(3),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const ScheduledTaskResultSchema = z.object({
    success: z.boolean(),
    error: z.string(),
    taskId: z.string(),
    status: z.enum(['scheduled', 'running', 'completed', 'failed', 'cancelled']),
    nextRun: z.date().optional(),
    result: z.unknown().optional(),
});
export class ScheduledTaskWorkflow extends WorkflowBubble {
    static type = 'workflow';
    static bubbleName = 'scheduled-task-workflow';
    static schema = ScheduledTaskParamsSchema;
    static resultSchema = ScheduledTaskResultSchema;
    static shortDescription = 'Run tasks on schedule with cron/interval support';
    static longDescription = 'Schedule and execute tasks using cron expressions, intervals, or specific times with retry support.';
    static alias = 'scheduled-task';
    static scheduledTasks = new Map();
    constructor(params, context) {
        super(params, context);
    }
    async performAction() {
        const taskId = this.generateTaskId();
        console.log(`[ScheduledTask] Scheduling task: ${taskId}`);
        try {
            const { schedule, action } = this.params;
            let nextRun;
            if (schedule.type === 'once' && schedule.runAt) {
                nextRun = schedule.runAt;
                const delay = schedule.runAt.getTime() - Date.now();
                if (delay > 0) {
                    const timeout = setTimeout(() => this.executeTask(taskId, action), delay);
                    ScheduledTaskWorkflow.scheduledTasks.set(taskId, timeout);
                }
                else {
                    await this.executeTask(taskId, action);
                }
            }
            else if (schedule.type === 'interval' && schedule.expression) {
                const interval = parseInt(schedule.expression);
                const intervalObj = setInterval(() => this.executeTask(taskId, action), interval);
                ScheduledTaskWorkflow.scheduledTasks.set(taskId, intervalObj);
                nextRun = new Date(Date.now() + interval);
            }
            else if (schedule.type === 'cron') {
                console.log('[ScheduledTask] Cron scheduling:', schedule.expression);
                nextRun = new Date();
            }
            return {
                success: true,
                error: '',
                taskId,
                status: 'scheduled',
                nextRun,
            };
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                taskId,
                status: 'failed',
                nextRun: undefined,
            };
        }
    }
    async executeTask(taskId, action) {
        console.log(`[ScheduledTask] Executing task: ${taskId}`);
        try {
            let result;
            if (action.type === 'http') {
                const config = action.config;
                const httpBubble = new HttpBubble({
                    url: config.url,
                    method: config.method || 'GET',
                    credentials: this.params.credentials,
                }, this.context);
                const httpResult = await httpBubble.action();
                result = httpResult;
            }
            return {
                success: true,
                error: '',
                taskId,
                status: 'completed',
                result,
            };
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                taskId,
                status: 'failed',
                nextRun: undefined,
            };
        }
    }
    generateTaskId() {
        return `task_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    }
    static cancelTask(taskId) {
        const timeout = ScheduledTaskWorkflow.scheduledTasks.get(taskId);
        if (timeout) {
            clearTimeout(timeout);
            ScheduledTaskWorkflow.scheduledTasks.delete(taskId);
            return true;
        }
        return false;
    }
}
//# sourceMappingURL=scheduled-task.workflow.js.map