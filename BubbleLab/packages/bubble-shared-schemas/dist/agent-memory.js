import { z } from 'zod';
export const TOOL_CALL_TO_DISCARD = ['get-bubble-details-tool'];
export const ConversationMessageSchema = z.object({
    role: z
        .enum(['user', 'assistant', 'tool'])
        .describe('The role of the message sender'),
    content: z.string().describe('The message content'),
    toolCallId: z.string().optional().describe('Tool call ID for tool messages'),
    name: z.string().optional().describe('Tool name for tool messages'),
});
//# sourceMappingURL=agent-memory.js.map