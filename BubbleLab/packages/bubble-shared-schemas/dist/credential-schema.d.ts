import { BubbleName, CredentialType } from './types.js';
import { z } from '@hono/zod-openapi';
/**
 * Maps credential types to their environment variable names (for backend only!!!!)
 */
export declare const CREDENTIAL_ENV_MAP: Record<CredentialType, string>;
/** Used by bubblelab studio */
export declare const SYSTEM_CREDENTIALS: Set<CredentialType>;
/**
 * Credentials that are optional (not required) for their associated bubbles.
 * These will not show as "missing" in the UI when not selected.
 */
export declare const OPTIONAL_CREDENTIALS: Set<CredentialType>;
/**
 * OAuth provider names - type-safe provider identifiers
 */
export type OAuthProvider = 'google' | 'followupboss' | 'notion';
/**
 * Scope description mapping - maps OAuth scope URLs to human-readable descriptions
 */
export interface ScopeDescription {
    scope: string;
    description: string;
    defaultEnabled: boolean;
}
/**
 * OAuth credential type configuration for a specific service under a provider
 */
export interface OAuthCredentialConfig {
    displayName: string;
    defaultScopes: string[];
    description: string;
    scopeDescriptions?: ScopeDescription[];
}
/**
 * OAuth provider configuration shared between frontend and backend
 */
export interface OAuthProviderConfig {
    name: OAuthProvider;
    displayName: string;
    credentialTypes: Partial<Record<CredentialType, OAuthCredentialConfig>>;
    authorizationParams?: Record<string, string>;
}
/**
 * OAuth provider configurations - single source of truth for OAuth providers
 * Contains all information needed by frontend and backend
 */
export declare const OAUTH_PROVIDERS: Record<OAuthProvider, OAuthProviderConfig>;
/**
 * Get the OAuth provider for a specific credential type
 * Safely maps credential types to their OAuth providers
 */
export declare function getOAuthProvider(credentialType: CredentialType): OAuthProvider | null;
/**
 * Check if a credential type is OAuth-based
 */
export declare function isOAuthCredential(credentialType: CredentialType): boolean;
/**
 * Get scope descriptions for a specific credential type
 * Returns an array of scope descriptions that will be requested during OAuth
 */
export declare function getScopeDescriptions(credentialType: CredentialType): ScopeDescription[];
/**
 * Maps bubble names to their accepted credential types
 */
export type CredentialOptions = Partial<Record<CredentialType, string>>;
/**
 * Collection of credential options for all bubbles
 */
export declare const BUBBLE_CREDENTIAL_OPTIONS: Record<BubbleName, CredentialType[]>;
export declare const createCredentialSchema: z.ZodObject<{
    credentialType: z.ZodNativeEnum<typeof CredentialType>;
    value: z.ZodString;
    name: z.ZodOptional<z.ZodString>;
    skipValidation: z.ZodOptional<z.ZodBoolean>;
    isDefault: z.ZodOptional<z.ZodBoolean>;
    credentialConfigurations: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    metadata: z.ZodOptional<z.ZodObject<{
        tables: z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>;
        tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        databaseName: z.ZodOptional<z.ZodString>;
        databaseType: z.ZodOptional<z.ZodEnum<["postgresql", "mysql", "sqlite", "mssql", "oracle"]>>;
        rules: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            text: z.ZodString;
            enabled: z.ZodBoolean;
            createdAt: z.ZodString;
            updatedAt: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }, {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }>, "many">>;
        notes: z.ZodOptional<z.ZodString>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    }, {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    value: string;
    credentialType: CredentialType;
    name?: string | undefined;
    skipValidation?: boolean | undefined;
    isDefault?: boolean | undefined;
    credentialConfigurations?: Record<string, unknown> | undefined;
    metadata?: {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    } | undefined;
}, {
    value: string;
    credentialType: CredentialType;
    name?: string | undefined;
    skipValidation?: boolean | undefined;
    isDefault?: boolean | undefined;
    credentialConfigurations?: Record<string, unknown> | undefined;
    metadata?: {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    } | undefined;
}>;
export declare const updateCredentialSchema: z.ZodObject<{
    value: z.ZodOptional<z.ZodString>;
    name: z.ZodOptional<z.ZodString>;
    skipValidation: z.ZodOptional<z.ZodBoolean>;
    isDefault: z.ZodOptional<z.ZodBoolean>;
    credentialConfigurations: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    metadata: z.ZodOptional<z.ZodObject<{
        tables: z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>;
        tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        databaseName: z.ZodOptional<z.ZodString>;
        databaseType: z.ZodOptional<z.ZodEnum<["postgresql", "mysql", "sqlite", "mssql", "oracle"]>>;
        rules: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            text: z.ZodString;
            enabled: z.ZodBoolean;
            createdAt: z.ZodString;
            updatedAt: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }, {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }>, "many">>;
        notes: z.ZodOptional<z.ZodString>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    }, {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    value?: string | undefined;
    name?: string | undefined;
    skipValidation?: boolean | undefined;
    isDefault?: boolean | undefined;
    credentialConfigurations?: Record<string, unknown> | undefined;
    metadata?: {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    } | undefined;
}, {
    value?: string | undefined;
    name?: string | undefined;
    skipValidation?: boolean | undefined;
    isDefault?: boolean | undefined;
    credentialConfigurations?: Record<string, unknown> | undefined;
    metadata?: {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    } | undefined;
}>;
export declare const credentialResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    credentialType: z.ZodString;
    name: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        tables: z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>;
        tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        databaseName: z.ZodOptional<z.ZodString>;
        databaseType: z.ZodOptional<z.ZodEnum<["postgresql", "mysql", "sqlite", "mssql", "oracle"]>>;
        rules: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            text: z.ZodString;
            enabled: z.ZodBoolean;
            createdAt: z.ZodString;
            updatedAt: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }, {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }>, "many">>;
        notes: z.ZodOptional<z.ZodString>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    }, {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    }>>;
    createdAt: z.ZodString;
    isDefault: z.ZodOptional<z.ZodBoolean>;
    isOauth: z.ZodOptional<z.ZodBoolean>;
    oauthProvider: z.ZodOptional<z.ZodString>;
    oauthExpiresAt: z.ZodOptional<z.ZodString>;
    oauthScopes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    oauthStatus: z.ZodOptional<z.ZodEnum<["active", "expired", "needs_refresh"]>>;
}, "strip", z.ZodTypeAny, {
    credentialType: string;
    id: number;
    createdAt: string;
    name?: string | undefined;
    isDefault?: boolean | undefined;
    metadata?: {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    } | undefined;
    isOauth?: boolean | undefined;
    oauthProvider?: string | undefined;
    oauthExpiresAt?: string | undefined;
    oauthScopes?: string[] | undefined;
    oauthStatus?: "active" | "expired" | "needs_refresh" | undefined;
}, {
    credentialType: string;
    id: number;
    createdAt: string;
    name?: string | undefined;
    isDefault?: boolean | undefined;
    metadata?: {
        tables: Record<string, Record<string, string>>;
        tableNotes?: Record<string, string> | undefined;
        databaseName?: string | undefined;
        databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
        rules?: {
            id: string;
            text: string;
            enabled: boolean;
            createdAt: string;
            updatedAt: string;
        }[] | undefined;
        notes?: string | undefined;
        tags?: string[] | undefined;
    } | undefined;
    isOauth?: boolean | undefined;
    oauthProvider?: string | undefined;
    oauthExpiresAt?: string | undefined;
    oauthScopes?: string[] | undefined;
    oauthStatus?: "active" | "expired" | "needs_refresh" | undefined;
}>;
export declare const createCredentialResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
    id: number;
}, {
    message: string;
    id: number;
}>;
export declare const updateCredentialResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
    id: number;
}, {
    message: string;
    id: number;
}>;
export declare const successMessageResponseSchema: z.ZodObject<{
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
}, {
    message: string;
}>;
export type CreateCredentialRequest = z.infer<typeof createCredentialSchema>;
export type UpdateCredentialRequest = z.infer<typeof updateCredentialSchema>;
export type CredentialResponse = z.infer<typeof credentialResponseSchema>;
export type CreateCredentialResponse = z.infer<typeof createCredentialResponseSchema>;
export type UpdateCredentialResponse = z.infer<typeof updateCredentialResponseSchema>;
//# sourceMappingURL=credential-schema.d.ts.map