import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
export declare const SqlOperations: z.ZodEnum<["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "WITH", "EXPLAIN"]>;
declare const InsForgeDbParamsSchema: z.ZodObject<{
    query: z.ZodString;
    allowedOperations: z.ZodDefault<z.ZodArray<z.ZodEnum<["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "WITH", "EXPLAIN"]>, "many">>;
    parameters: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>>;
    timeout: z.ZodDefault<z.ZodNumber>;
    maxRows: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    query: string;
    allowedOperations: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "CREATE")[];
    parameters: unknown[];
    maxRows: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    query: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    allowedOperations?: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "CREATE")[] | undefined;
    parameters?: unknown[] | undefined;
    maxRows?: number | undefined;
}>;
type InsForgeDbParamsInput = z.input<typeof InsForgeDbParamsSchema>;
type InsForgeDbParams = z.output<typeof InsForgeDbParamsSchema>;
declare const InsForgeDbResultSchema: z.ZodObject<{
    rows: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
    rowCount: z.ZodNullable<z.ZodNumber>;
    command: z.ZodString;
    executionTime: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
    cleanedJSONString: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    rows: Record<string, unknown>[];
    rowCount: number | null;
    command: string;
    executionTime: number;
    cleanedJSONString: string;
}, {
    error: string;
    success: boolean;
    rows: Record<string, unknown>[];
    rowCount: number | null;
    command: string;
    executionTime: number;
    cleanedJSONString: string;
}>;
type InsForgeDbResult = z.output<typeof InsForgeDbResultSchema>;
export type { InsForgeDbParamsInput };
/**
 * InsForge Database Bubble
 *
 * Execute SQL queries against an InsForge backend database.
 * Works similarly to the PostgreSQL bubble but uses InsForge's REST API.
 *
 * @example
 * ```typescript
 * const result = await new InsForgeDbBubble({
 *   query: 'SELECT * FROM users WHERE active = $1',
 *   parameters: [true],
 *   allowedOperations: ['SELECT'],
 *   maxRows: 100,
 * }).action();
 *
 * console.log(result.data.rows);
 * ```
 */
export declare class InsForgeDbBubble extends ServiceBubble<InsForgeDbParams, InsForgeDbResult> {
    static readonly type: "service";
    static readonly service = "insforge";
    static readonly authType: "apikey";
    static readonly bubbleName = "insforge-db";
    static readonly schema: z.ZodObject<{
        query: z.ZodString;
        allowedOperations: z.ZodDefault<z.ZodArray<z.ZodEnum<["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "WITH", "EXPLAIN"]>, "many">>;
        parameters: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>>;
        timeout: z.ZodDefault<z.ZodNumber>;
        maxRows: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        query: string;
        allowedOperations: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "CREATE")[];
        parameters: unknown[];
        maxRows: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        query: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        allowedOperations?: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "CREATE")[] | undefined;
        parameters?: unknown[] | undefined;
        maxRows?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        rows: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
        rowCount: z.ZodNullable<z.ZodNumber>;
        command: z.ZodString;
        executionTime: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
        cleanedJSONString: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        rows: Record<string, unknown>[];
        rowCount: number | null;
        command: string;
        executionTime: number;
        cleanedJSONString: string;
    }, {
        error: string;
        success: boolean;
        rows: Record<string, unknown>[];
        rowCount: number | null;
        command: string;
        executionTime: number;
        cleanedJSONString: string;
    }>;
    static readonly shortDescription = "InsForge is the backend built for AI-assisted development. Connect InsForge with any agent. Add authentication, database, storage, functions, and AI integrations to your app in seconds.";
    static readonly longDescription = "\n    Authentication - Complete user management system\n    Database - Flexible data storage and retrieval\n    Storage - File management and organization\n    AI Integration - Chat completions and image generation (OpenAI-compatible)\n    Serverless Functions - Scalable compute power\n    Site Deployment (coming soon) - Easy application deployment\n  ";
    static readonly alias = "insforge";
    constructor(params?: InsForgeDbParamsInput, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    private getCredentials;
    /**
     * Validate that the SQL query operation is allowed
     */
    private validateSqlOperation;
    protected performAction(context?: BubbleContext): Promise<InsForgeDbResult>;
}
//# sourceMappingURL=insforge-db.d.ts.map