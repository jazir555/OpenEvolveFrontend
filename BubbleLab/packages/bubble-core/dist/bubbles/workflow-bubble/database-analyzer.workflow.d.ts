import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const DatabaseAnalyzerParamsSchema: z.ZodObject<{
    /**
     * The data source service to analyze (currently supports 'postgresql')
     */
    dataSourceType: z.ZodEnum<["postgresql"]>;
    /**
     * Whether to ignore SSL certificate errors when connecting
     */
    ignoreSSLErrors: z.ZodDefault<z.ZodBoolean>;
    /**
     * Include additional metadata like enum values and constraints
     */
    includeMetadata: z.ZodDefault<z.ZodBoolean>;
    /**
     * Injected credentials from the system
     */
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    /**
     * Injected metadata from user credentials (database schema notes and rules)
     */
    injectedMetadata: z.ZodOptional<z.ZodObject<{
        tables: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>>;
        tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        rules: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    }, {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    dataSourceType: "postgresql";
    ignoreSSLErrors: boolean;
    includeMetadata: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    injectedMetadata?: {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    } | undefined;
}, {
    dataSourceType: "postgresql";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ignoreSSLErrors?: boolean | undefined;
    includeMetadata?: boolean | undefined;
    injectedMetadata?: {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    } | undefined;
}>;
declare const DatabaseAnalyzerResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    /**
     * The analyzed database schema in structured format
     */
    databaseSchema: z.ZodOptional<z.ZodObject<{
        /**
         * Raw schema data as returned by the database
         */
        rawData: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        /**
         * Cleaned JSON string representation of the schema
         */
        cleanedJSON: z.ZodOptional<z.ZodString>;
        /**
         * Number of tables found
         */
        tableCount: z.ZodOptional<z.ZodNumber>;
        /**
         * List of table names
         */
        tableNames: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        rawData?: Record<string, unknown>[] | undefined;
        cleanedJSON?: string | undefined;
        tableCount?: number | undefined;
        tableNames?: string[] | undefined;
    }, {
        rawData?: Record<string, unknown>[] | undefined;
        cleanedJSON?: string | undefined;
        tableCount?: number | undefined;
        tableNames?: string[] | undefined;
    }>>;
    /**
     * Summary of the analysis operation
     */
    analysisSummary: z.ZodOptional<z.ZodObject<{
        dataSourceType: z.ZodString;
        connectionSuccessful: z.ZodBoolean;
        analysisTimestamp: z.ZodDate;
    }, "strip", z.ZodTypeAny, {
        dataSourceType: string;
        connectionSuccessful: boolean;
        analysisTimestamp: Date;
    }, {
        dataSourceType: string;
        connectionSuccessful: boolean;
        analysisTimestamp: Date;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    databaseSchema?: {
        rawData?: Record<string, unknown>[] | undefined;
        cleanedJSON?: string | undefined;
        tableCount?: number | undefined;
        tableNames?: string[] | undefined;
    } | undefined;
    analysisSummary?: {
        dataSourceType: string;
        connectionSuccessful: boolean;
        analysisTimestamp: Date;
    } | undefined;
}, {
    error: string;
    success: boolean;
    databaseSchema?: {
        rawData?: Record<string, unknown>[] | undefined;
        cleanedJSON?: string | undefined;
        tableCount?: number | undefined;
        tableNames?: string[] | undefined;
    } | undefined;
    analysisSummary?: {
        dataSourceType: string;
        connectionSuccessful: boolean;
        analysisTimestamp: Date;
    } | undefined;
}>;
type DatabaseAnalyzerResult = z.output<typeof DatabaseAnalyzerResultSchema>;
type DatabaseAnalyzerParamsInput = z.input<typeof DatabaseAnalyzerParamsSchema>;
type DatabaseAnalyzerParams = z.output<typeof DatabaseAnalyzerParamsSchema>;
/**
 * DatabaseAnalyzerWorkflowBubble - Analyzes database schema structure
 *
 * This workflow bubble simplifies database schema analysis by:
 * 1. Connecting to the specified database
 * 2. Querying the information schema to get table/column structure
 * 3. Extracting metadata like enum values and constraints
 * 4. Returning structured schema information ready for AI analysis
 *
 * Common use cases:
 * - Data discovery and cataloging
 * - Generating queries based on schema structure
 * - Database documentation generation
 * - AI-powered data analysis preparation
 */
export declare class DatabaseAnalyzerWorkflowBubble extends WorkflowBubble<DatabaseAnalyzerParams, DatabaseAnalyzerResult> {
    static readonly type: "workflow";
    static readonly bubbleName = "database-analyzer";
    static readonly schema: z.ZodObject<{
        /**
         * The data source service to analyze (currently supports 'postgresql')
         */
        dataSourceType: z.ZodEnum<["postgresql"]>;
        /**
         * Whether to ignore SSL certificate errors when connecting
         */
        ignoreSSLErrors: z.ZodDefault<z.ZodBoolean>;
        /**
         * Include additional metadata like enum values and constraints
         */
        includeMetadata: z.ZodDefault<z.ZodBoolean>;
        /**
         * Injected credentials from the system
         */
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        /**
         * Injected metadata from user credentials (database schema notes and rules)
         */
        injectedMetadata: z.ZodOptional<z.ZodObject<{
            tables: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>>;
            tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
            rules: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        }, {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        dataSourceType: "postgresql";
        ignoreSSLErrors: boolean;
        includeMetadata: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        injectedMetadata?: {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        } | undefined;
    }, {
        dataSourceType: "postgresql";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ignoreSSLErrors?: boolean | undefined;
        includeMetadata?: boolean | undefined;
        injectedMetadata?: {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        /**
         * The analyzed database schema in structured format
         */
        databaseSchema: z.ZodOptional<z.ZodObject<{
            /**
             * Raw schema data as returned by the database
             */
            rawData: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
            /**
             * Cleaned JSON string representation of the schema
             */
            cleanedJSON: z.ZodOptional<z.ZodString>;
            /**
             * Number of tables found
             */
            tableCount: z.ZodOptional<z.ZodNumber>;
            /**
             * List of table names
             */
            tableNames: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            rawData?: Record<string, unknown>[] | undefined;
            cleanedJSON?: string | undefined;
            tableCount?: number | undefined;
            tableNames?: string[] | undefined;
        }, {
            rawData?: Record<string, unknown>[] | undefined;
            cleanedJSON?: string | undefined;
            tableCount?: number | undefined;
            tableNames?: string[] | undefined;
        }>>;
        /**
         * Summary of the analysis operation
         */
        analysisSummary: z.ZodOptional<z.ZodObject<{
            dataSourceType: z.ZodString;
            connectionSuccessful: z.ZodBoolean;
            analysisTimestamp: z.ZodDate;
        }, "strip", z.ZodTypeAny, {
            dataSourceType: string;
            connectionSuccessful: boolean;
            analysisTimestamp: Date;
        }, {
            dataSourceType: string;
            connectionSuccessful: boolean;
            analysisTimestamp: Date;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        databaseSchema?: {
            rawData?: Record<string, unknown>[] | undefined;
            cleanedJSON?: string | undefined;
            tableCount?: number | undefined;
            tableNames?: string[] | undefined;
        } | undefined;
        analysisSummary?: {
            dataSourceType: string;
            connectionSuccessful: boolean;
            analysisTimestamp: Date;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        databaseSchema?: {
            rawData?: Record<string, unknown>[] | undefined;
            cleanedJSON?: string | undefined;
            tableCount?: number | undefined;
            tableNames?: string[] | undefined;
        } | undefined;
        analysisSummary?: {
            dataSourceType: string;
            connectionSuccessful: boolean;
            analysisTimestamp: Date;
        } | undefined;
    }>;
    static readonly shortDescription = "Analyzes database schema structure and metadata";
    static readonly longDescription = "Connects to a database and extracts comprehensive schema information including tables, columns, data types, constraints, and enum values. Perfect for AI-powered data analysis, query generation, and database documentation. Currently supports PostgreSQL with plans for additional database types.";
    static readonly alias = "analyze-db";
    constructor(params: DatabaseAnalyzerParamsInput, context?: BubbleContext);
    protected performAction(_context?: BubbleContext): Promise<DatabaseAnalyzerResult>;
    /**
     * Builds the PostgreSQL schema query based on metadata requirements
     */
    private buildPostgreSQLSchemaQuery;
}
export {};
//# sourceMappingURL=database-analyzer.workflow.d.ts.map