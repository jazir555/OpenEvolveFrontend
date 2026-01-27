import { z } from '@hono/zod-openapi';
export declare const joinWaitlistSchema: z.ZodObject<{
    name: z.ZodString;
    email: z.ZodString;
    database: z.ZodString;
    otherDatabase: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    email: string;
    database: string;
    otherDatabase?: string | undefined;
}, {
    name: string;
    email: string;
    database: string;
    otherDatabase?: string | undefined;
}>;
export declare const joinWaitlistResponseSchema: z.ZodObject<{
    success: z.ZodBoolean;
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
    success: boolean;
}, {
    message: string;
    success: boolean;
}>;
export type JoinWaitlistRequest = z.infer<typeof joinWaitlistSchema>;
export type JoinWaitlistResponse = z.infer<typeof joinWaitlistResponseSchema>;
//# sourceMappingURL=waitlist-schema.d.ts.map