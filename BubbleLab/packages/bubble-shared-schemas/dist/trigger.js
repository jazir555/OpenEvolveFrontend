// Runtime object that mirrors the interface keys
// This allows us to validate event types at runtime
export const BUBBLE_TRIGGER_EVENTS = {
    'slack/bot_mentioned': true,
    'schedule/cron': true,
    'webhook/http': true,
};
// Helper function to check if an event type is valid
export function isValidBubbleTriggerEvent(eventType) {
    return eventType in BUBBLE_TRIGGER_EVENTS;
}
//# sourceMappingURL=trigger.js.map