import type { BubbleName } from './types.js';
import { z } from 'zod';
export declare const TOOL_CALL_TO_DISCARD: BubbleName[];
export declare const ConversationMessageSchema: z.ZodObject<{
    role: z.ZodEnum<["user", "assistant", "tool"]>;
    content: z.ZodString;
    toolCallId: z.ZodOptional<z.ZodString>;
    name: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    role: "user" | "assistant" | "tool";
    content: string;
    toolCallId?: string | undefined;
    name?: string | undefined;
}, {
    role: "user" | "assistant" | "tool";
    content: string;
    toolCallId?: string | undefined;
    name?: string | undefined;
}>;
export type ConversationMessage = z.infer<typeof ConversationMessageSchema>;
//# sourceMappingURL=agent-memory.d.ts.map