import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const SQLQueryToolParamsSchema: z.ZodObject<{
    query: z.ZodString;
    reasoning: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    reasoning: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
}, {
    query: string;
    reasoning: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
}>;
type SQLQueryToolParamsInput = z.input<typeof SQLQueryToolParamsSchema>;
type SQLQueryToolParams = z.output<typeof SQLQueryToolParamsSchema>;
type SQLQueryToolResult = z.output<typeof SQLQueryToolResultSchema>;
declare const SQLQueryToolResultSchema: z.ZodObject<{
    rows: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    rowCount: z.ZodNumber;
    executionTime: z.ZodNumber;
    fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        dataTypeID: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        dataTypeID?: number | undefined;
    }, {
        name: string;
        dataTypeID?: number | undefined;
    }>, "many">>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    rowCount: number;
    executionTime: number;
    rows?: Record<string, unknown>[] | undefined;
    fields?: {
        name: string;
        dataTypeID?: number | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    rowCount: number;
    executionTime: number;
    rows?: Record<string, unknown>[] | undefined;
    fields?: {
        name: string;
        dataTypeID?: number | undefined;
    }[] | undefined;
}>;
/**
 * SQLQueryTool - Execute SQL queries against PostgreSQL databases
 *
 * This tool bubble provides a safe, read-only interface for AI agents to query
 * PostgreSQL databases. It's designed for data analysis, exploration, and
 * business intelligence tasks.
 */
export declare class SQLQueryTool extends ToolBubble<SQLQueryToolParams, SQLQueryToolResult> {
    static readonly type: "tool";
    static readonly bubbleName = "sql-query-tool";
    static readonly schema: z.ZodObject<{
        query: z.ZodString;
        reasoning: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        reasoning: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }, {
        query: string;
        reasoning: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        rows: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        rowCount: z.ZodNumber;
        executionTime: z.ZodNumber;
        fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            dataTypeID: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            dataTypeID?: number | undefined;
        }, {
            name: string;
            dataTypeID?: number | undefined;
        }>, "many">>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        rowCount: number;
        executionTime: number;
        rows?: Record<string, unknown>[] | undefined;
        fields?: {
            name: string;
            dataTypeID?: number | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        rowCount: number;
        executionTime: number;
        rows?: Record<string, unknown>[] | undefined;
        fields?: {
            name: string;
            dataTypeID?: number | undefined;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Execute read-only SQL queries against PostgreSQL databases for data analysis";
    static readonly longDescription = "\n    A tool bubble that provides safe, read-only SQL query execution against PostgreSQL databases.\n    \n    Features:\n    - Execute SELECT, WITH, EXPLAIN, ANALYZE, SHOW, and DESCRIBE queries\n    - Automatic query timeout and row limit enforcement (30s timeout, 1000 rows max)\n    - Clean JSON formatting of results for AI consumption\n    - Detailed execution metadata including timing and row counts\n    \n    Security:\n    - Read-only operations enforced\n    - Query timeout protection (30 seconds)\n    - Row limit protection (1000 rows max)\n    \n    Use cases:\n    - AI agents performing iterative database exploration\n    - Data analysis and business intelligence queries\n    - Schema discovery and table introspection\n    - Performance analysis with EXPLAIN queries\n    - Automated reporting and data extraction\n  ";
    static readonly alias = "sql";
    constructor(params: SQLQueryToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<SQLQueryToolResult>;
    /**
     * Validates SQL query before execution
     * Uses comprehensive security utilities to prevent SQL injection and other attacks
     * Prevents dangerous operations and enforces read-only access
     */
    private validateQuery;
    /**
     * Enhances query result with additional metadata and analysis
     */
    private enhanceResult;
    /**
     * Calculates statistics about query results
     */
    private calculateResultStats;
    /**
     * Formats query results as CSV string
     */
    private formatAsCSV;
    /**
     * Formats query results as markdown table
     */
    private formatAsMarkdown;
    /**
     * Extracts sample queries for common database exploration tasks
     */
    static getSampleQueries(): Record<string, string>;
}
export {};
//# sourceMappingURL=sql-query-tool.d.ts.map