import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type DatabaseMetadata } from '@bubblelab/shared-schemas';
export declare const SqlOperations: z.ZodEnum<["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "EXPLAIN", "ANALYZE", "SHOW", "DESCRIBE", "DESC", "CREATE"]>;
declare const PostgreSQLParamsSchema: z.ZodObject<{
    ignoreSSL: z.ZodDefault<z.ZodBoolean>;
    query: z.ZodEffects<z.ZodString, string, string>;
    allowedOperations: z.ZodDefault<z.ZodArray<z.ZodEnum<["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "EXPLAIN", "ANALYZE", "SHOW", "DESCRIBE", "DESC", "CREATE"]>, "many">>;
    parameters: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>>;
    timeout: z.ZodDefault<z.ZodNumber>;
    maxRows: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    ignoreSSL: boolean;
    query: string;
    allowedOperations: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "ANALYZE" | "SHOW" | "DESCRIBE" | "DESC" | "CREATE")[];
    parameters: unknown[];
    maxRows: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    query: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ignoreSSL?: boolean | undefined;
    allowedOperations?: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "ANALYZE" | "SHOW" | "DESCRIBE" | "DESC" | "CREATE")[] | undefined;
    parameters?: unknown[] | undefined;
    maxRows?: number | undefined;
}>;
type PostgreSQLParamsInput = z.input<typeof PostgreSQLParamsSchema>;
type PostgreSQLParams = z.output<typeof PostgreSQLParamsSchema>;
declare const PostgreSQLResultSchema: z.ZodObject<{
    rows: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
    rowCount: z.ZodNullable<z.ZodNumber>;
    command: z.ZodString;
    fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        dataTypeID: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        name: string;
        dataTypeID: number;
    }, {
        name: string;
        dataTypeID: number;
    }>, "many">>;
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
    fields?: {
        name: string;
        dataTypeID: number;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    rows: Record<string, unknown>[];
    rowCount: number | null;
    command: string;
    executionTime: number;
    cleanedJSONString: string;
    fields?: {
        name: string;
        dataTypeID: number;
    }[] | undefined;
}>;
type PostgreSQLResult = z.output<typeof PostgreSQLResultSchema>;
/**
 * PostgreSQL Bubble
 *
 * Executes SQL queries against PostgreSQL databases with security controls and validation.
 *
 * @class
 * @extends ServiceBubble<PostgreSQLParams, PostgreSQLResult>
 *
 * @example
 * ```typescript
 * const pgBubble = new PostgreSQLBubble({
 *   query: 'SELECT * FROM users WHERE active = $1',
 *   parameters: [true],
 *   allowedOperations: ['SELECT'],
 *   maxRows: 100,
 *   timeout: 30000
 * });
 *
 * const result = await pgBubble.action();
 * console.log(result.rows);
 * ```
 *
 * @see {@link https://www.postgresql.org/docs/|PostgreSQL Documentation}
 */
export declare class PostgreSQLBubble extends ServiceBubble<PostgreSQLParams, PostgreSQLResult> {
    private circuitBreaker;
    /**
     * Test the PostgreSQL connection credentials
     *
     * @returns Promise that resolves to true if credentials are valid
     *
     * @example
     * ```typescript
     * const isValid = await pgBubble.testCredential();
     * if (!isValid) {
     *   console.error('Invalid PostgreSQL credentials');
     * }
     * ```
     */
    testCredential(): Promise<boolean>;
    static readonly type: "service";
    static readonly service = "postgresql";
    static readonly authType: "connection-string";
    static readonly bubbleName = "postgresql";
    static readonly schema: z.ZodObject<{
        ignoreSSL: z.ZodDefault<z.ZodBoolean>;
        query: z.ZodEffects<z.ZodString, string, string>;
        allowedOperations: z.ZodDefault<z.ZodArray<z.ZodEnum<["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "EXPLAIN", "ANALYZE", "SHOW", "DESCRIBE", "DESC", "CREATE"]>, "many">>;
        parameters: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>>;
        timeout: z.ZodDefault<z.ZodNumber>;
        maxRows: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        ignoreSSL: boolean;
        query: string;
        allowedOperations: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "ANALYZE" | "SHOW" | "DESCRIBE" | "DESC" | "CREATE")[];
        parameters: unknown[];
        maxRows: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        query: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ignoreSSL?: boolean | undefined;
        allowedOperations?: ("SELECT" | "INSERT" | "UPDATE" | "DELETE" | "WITH" | "EXPLAIN" | "ANALYZE" | "SHOW" | "DESCRIBE" | "DESC" | "CREATE")[] | undefined;
        parameters?: unknown[] | undefined;
        maxRows?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        rows: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
        rowCount: z.ZodNullable<z.ZodNumber>;
        command: z.ZodString;
        fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            dataTypeID: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            name: string;
            dataTypeID: number;
        }, {
            name: string;
            dataTypeID: number;
        }>, "many">>;
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
        fields?: {
            name: string;
            dataTypeID: number;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        rows: Record<string, unknown>[];
        rowCount: number | null;
        command: string;
        executionTime: number;
        cleanedJSONString: string;
        fields?: {
            name: string;
            dataTypeID: number;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Execute PostgreSQL queries with operation validation";
    static readonly longDescription = "\n    Execute SQL queries against PostgreSQL databases with proper validation and security controls.\n    Use cases:\n    - Data retrieval with SELECT queries\n    - Data manipulation with INSERT, UPDATE, DELETE (when explicitly allowed)\n    - Database reporting and analytics\n    - Data migration and synchronization tasks\n    - JSON string output for integration with other systems\n    \n    Security Features:\n    - Operation whitelist (defaults to SELECT only)\n    - Parameterized queries to prevent SQL injection\n    - Connection timeout controls\n    - Result sanitization for JSON output\n  ";
    static readonly alias = "pg";
    constructor(params?: PostgreSQLParamsInput, context?: BubbleContext);
    protected performAction(context?: BubbleContext): Promise<PostgreSQLResult>;
    /**
     * Execute the actual database query
     */
    private executeQuery;
    getCredentialMetadata(): Promise<DatabaseMetadata | undefined>;
    /**
     * Validate that the SQL query operation is allowed
     */
    private validateSqlOperation;
    /**
     * Validate parameter usage to encourage parameterized queries
     */
    private validateParameterUsage;
    /**
     * Validate parentheses and quotes are balanced
     */
    private validateParenthesesBalance;
    /**
     * Clean and format query results as a JSON string
     */
    private cleanJSONString;
    /**
     * Clean an object by handling special values and preventing circular references
     */
    private cleanObject;
    protected chooseCredential(): string | undefined;
}
export {};
//# sourceMappingURL=postgresql.d.ts.map