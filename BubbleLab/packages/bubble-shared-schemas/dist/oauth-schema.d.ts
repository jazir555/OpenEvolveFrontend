import { z } from '@hono/zod-openapi';
import { CredentialType } from './types';
export declare const oauthInitiateRequestSchema: z.ZodObject<{
    credentialType: z.ZodNativeEnum<typeof CredentialType>;
    name: z.ZodOptional<z.ZodString>;
    scopes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    credentialType: CredentialType;
    name?: string | undefined;
    scopes?: string[] | undefined;
}, {
    credentialType: CredentialType;
    name?: string | undefined;
    scopes?: string[] | undefined;
}>;
export declare const oauthInitiateResponseSchema: z.ZodObject<{
    authUrl: z.ZodString;
    state: z.ZodString;
}, "strip", z.ZodTypeAny, {
    authUrl: string;
    state: string;
}, {
    authUrl: string;
    state: string;
}>;
export declare const oauthCallbackRequestSchema: z.ZodObject<{
    code: z.ZodString;
    state: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    code: string;
    state: string;
    description?: string | undefined;
}, {
    name: string;
    code: string;
    state: string;
    description?: string | undefined;
}>;
export declare const oauthTokenRefreshResponseSchema: z.ZodObject<{
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
}, {
    message: string;
}>;
export declare const oauthRevokeResponseSchema: z.ZodObject<{
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
}, {
    message: string;
}>;
export type OAuthInitiateRequest = z.infer<typeof oauthInitiateRequestSchema>;
export type OAuthInitiateResponse = z.infer<typeof oauthInitiateResponseSchema>;
export type OAuthCallbackRequest = z.infer<typeof oauthCallbackRequestSchema>;
export type OAuthTokenRefreshResponse = z.infer<typeof oauthTokenRefreshResponseSchema>;
export type OAuthRevokeResponse = z.infer<typeof oauthRevokeResponseSchema>;
//# sourceMappingURL=oauth-schema.d.ts.map