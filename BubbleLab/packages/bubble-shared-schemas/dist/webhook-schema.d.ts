import { z } from '@hono/zod-openapi';
export declare const webhookExecutionResponseSchema: z.ZodObject<{
    executionId: z.ZodNumber;
    success: z.ZodBoolean;
    data: z.ZodOptional<z.ZodUnknown>;
    error: z.ZodOptional<z.ZodString>;
    webhook: z.ZodObject<{
        userId: z.ZodString;
        path: z.ZodString;
        triggeredAt: z.ZodString;
        method: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    }, {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    }>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    executionId: number;
    webhook: {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    };
    error?: string | undefined;
    data?: unknown;
}, {
    success: boolean;
    executionId: number;
    webhook: {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    };
    error?: string | undefined;
    data?: unknown;
}>;
export declare const webhookResponseSchema: z.ZodObject<{
    challenge: z.ZodOptional<z.ZodString>;
    executionId: z.ZodOptional<z.ZodNumber>;
    success: z.ZodOptional<z.ZodBoolean>;
    data: z.ZodOptional<z.ZodUnion<[z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodUndefined]>>;
    error: z.ZodOptional<z.ZodString>;
    webhook: z.ZodOptional<z.ZodObject<{
        userId: z.ZodString;
        path: z.ZodString;
        triggeredAt: z.ZodString;
        method: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    }, {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    }>>;
}, "strip", z.ZodTypeAny, {
    error?: string | undefined;
    challenge?: string | undefined;
    success?: boolean | undefined;
    executionId?: number | undefined;
    data?: Record<string, unknown> | undefined;
    webhook?: {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    } | undefined;
}, {
    error?: string | undefined;
    challenge?: string | undefined;
    success?: boolean | undefined;
    executionId?: number | undefined;
    data?: Record<string, unknown> | undefined;
    webhook?: {
        path: string;
        userId: string;
        triggeredAt: string;
        method: string;
    } | undefined;
}>;
export type WebhookResponse = z.infer<typeof webhookResponseSchema>;
export type WebhookExecutionResponse = z.infer<typeof webhookExecutionResponseSchema>;
//# sourceMappingURL=webhook-schema.d.ts.map