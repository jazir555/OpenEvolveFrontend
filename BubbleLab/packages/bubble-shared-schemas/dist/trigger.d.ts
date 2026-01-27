export interface BubbleTriggerEventRegistry {
    'slack/bot_mentioned': SlackMentionEvent;
    'schedule/cron': CronEvent;
    'webhook/http': WebhookEvent;
}
export declare const BUBBLE_TRIGGER_EVENTS: {
    readonly 'slack/bot_mentioned': true;
    readonly 'schedule/cron': true;
    readonly 'webhook/http': true;
};
export declare function isValidBubbleTriggerEvent(eventType: string): eventType is keyof BubbleTriggerEventRegistry;
export interface BubbleTriggerEvent {
    type: keyof BubbleTriggerEventRegistry;
    timestamp: string;
    executionId: string;
    path: string;
    [key: string]: unknown;
}
/**
 * Cron event payload structure
 *
 * The 'cron' field contains the cron expression in standard 5-part cron format:
 *
 * ┌───────────── minute (0 - 59)
 * │ ┌───────────── hour (0 - 23)
 * │ │ ┌───────────── day of month (1 - 31)
 * │ │ │ ┌───────────── month (1 - 12)
 * │ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
 * │ │ │ │ │
 * * * * * *
 *
 * @example
 * ```typescript
 * // Daily at midnight
 * { cron: '0 0 * * *' }
 *
 * // Every weekday at 9am
 * { cron: '0 9 * * 1-5' }
 *
 * // Every 15 minutes
 * { cron: '*\/15 * * * *' }
 *
 * // First day of every month at midnight
 * { cron: '0 0 1 * *' }
 * ```
 */
export interface CronEvent extends BubbleTriggerEvent {
    /** The cron expression defining when this event triggers */
    cron: string;
    body?: Record<string, unknown>;
}
export interface WebhookEvent extends BubbleTriggerEvent {
    body?: Record<string, unknown>;
}
export interface BubbleTrigger {
    type: keyof BubbleTriggerEventRegistry;
    cronSchedule?: string;
    name?: string;
    description?: string;
    timeout?: number;
    retries?: number;
}
export interface SlackEventWrapper {
    token: string;
    team_id: string;
    api_app_id: string;
    event: SlackAppMentionEvent | SlackMessageEvent;
    type: 'event_callback';
    authorizations: Array<{
        enterprise_id?: string;
        team_id: string;
        user_id: string;
        is_bot: boolean;
    }>;
    event_context: string;
    event_id: string;
    event_time: number;
}
export interface SlackAppMentionEvent {
    type: 'app_mention';
    user: string;
    text: string;
    ts: string;
    channel: string;
    event_ts: string;
    thread_ts?: string;
}
export interface SlackMessageEvent {
    type: 'message';
    user: string;
    text: string;
    ts: string;
    channel: string;
    event_ts: string;
    channel_type: 'channel' | 'group' | 'im' | 'mpim';
    subtype?: string;
}
export interface SlackMentionEvent extends BubbleTriggerEvent {
    slack_event: SlackEventWrapper;
    channel: string;
    user: string;
    text: string;
    thread_ts?: string;
}
export interface SlackMessageReceivedEvent extends BubbleTriggerEvent {
    slack_event: SlackEventWrapper;
    channel: string;
    user: string;
    text: string;
    channel_type: 'channel' | 'group' | 'im' | 'mpim';
    subtype?: string;
}
//# sourceMappingURL=trigger.d.ts.map