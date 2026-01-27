import { z } from '@hono/zod-openapi';
export declare const errorResponseSchema: z.ZodObject<{
    error: z.ZodString;
    details: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error: string;
    details?: string | undefined;
}, {
    error: string;
    details?: string | undefined;
}>;
export type ErrorResponse = z.infer<typeof errorResponseSchema>;
export interface HealthCheckResponse {
    message: string;
    timestamp: string;
}
export declare const slackUrlVerificationSchema: z.ZodObject<{
    token: z.ZodString;
    challenge: z.ZodString;
    type: z.ZodLiteral<"url_verification">;
}, "strip", z.ZodTypeAny, {
    type: "url_verification";
    token: string;
    challenge: string;
}, {
    type: "url_verification";
    token: string;
    challenge: string;
}>;
export declare const slackUrlVerificationResponseSchema: z.ZodObject<{
    challenge: z.ZodString;
}, "strip", z.ZodTypeAny, {
    challenge: string;
}, {
    challenge: string;
}>;
export type SlackUrlVerificationResponse = z.infer<typeof slackUrlVerificationResponseSchema>;
//# sourceMappingURL=api-schema.d.ts.map