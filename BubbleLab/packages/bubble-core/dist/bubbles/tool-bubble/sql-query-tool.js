import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { PostgreSQLBubble } from '../service-bubble/postgresql.js';
import { sanitizeSQLQuery } from '../../utils/security-utils.js';
// Define the parameters schema
const SQLQueryToolParamsSchema = z.object({
    query: z
        .string()
        .describe('SQL query to execute (SELECT, WITH, EXPLAIN, ANALYZE, SHOW, DESCRIBE only)'),
    reasoning: z
        .string()
        .describe("Explain why you're running this specific query and what you hope to learn from it"),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Database credentials (injected at runtime)'),
    config: z
        .record(z.string(), z.unknown())
        .optional()
        .describe('Configuration for the tool bubble'),
});
// Result schema for validation
const SQLQueryToolResultSchema = z.object({
    // Query results
    rows: z
        .array(z.record(z.unknown()))
        .optional()
        .describe('Array of query result rows as objects'),
    rowCount: z.number().describe('Number of rows returned by the query'),
    // Metadata
    executionTime: z.number().describe('Query execution time in milliseconds'),
    fields: z
        .array(z.object({
        name: z.string().describe('Name of the column'),
        dataTypeID: z
            .number()
            .optional()
            .describe('PostgreSQL data type ID for the column'),
    }))
        .optional()
        .describe('Array of column metadata from the query result'),
    // Standard result fields
    success: z.boolean().describe('Whether the query execution was successful'),
    error: z.string().describe('Error message if query execution failed'),
});
/**
 * SQLQueryTool - Execute SQL queries against PostgreSQL databases
 *
 * This tool bubble provides a safe, read-only interface for AI agents to query
 * PostgreSQL databases. It's designed for data analysis, exploration, and
 * business intelligence tasks.
 */
export class SQLQueryTool extends ToolBubble {
    static type = 'tool';
    static bubbleName = 'sql-query-tool';
    static schema = SQLQueryToolParamsSchema;
    static resultSchema = SQLQueryToolResultSchema;
    static shortDescription = 'Execute read-only SQL queries against PostgreSQL databases for data analysis';
    static longDescription = `
    A tool bubble that provides safe, read-only SQL query execution against PostgreSQL databases.
    
    Features:
    - Execute SELECT, WITH, EXPLAIN, ANALYZE, SHOW, and DESCRIBE queries
    - Automatic query timeout and row limit enforcement (30s timeout, 1000 rows max)
    - Clean JSON formatting of results for AI consumption
    - Detailed execution metadata including timing and row counts
    
    Security:
    - Read-only operations enforced
    - Query timeout protection (30 seconds)
    - Row limit protection (1000 rows max)
    
    Use cases:
    - AI agents performing iterative database exploration
    - Data analysis and business intelligence queries
    - Schema discovery and table introspection
    - Performance analysis with EXPLAIN queries
    - Automated reporting and data extraction
  `;
    static alias = 'sql';
    constructor(params, context) {
        super(params, context);
    }
    async performAction(context) {
        void context; // Context available but not currently used
        const startTime = Date.now();
        try {
            // Validate query before execution
            const validation = this.validateQuery(this.params.query);
            if (!validation.valid) {
                console.log(`❌ [SQLQueryTool] Query validation failed: ${validation.error}`);
                return {
                    rowCount: 0,
                    executionTime: Date.now() - startTime,
                    success: false,
                    error: validation.error || 'Query validation failed',
                };
            }
            // Log query execution
            console.debug(`\n🔍 [SQLQueryTool] Executing SQL query...`);
            console.debug(`💭 [SQLQueryTool] Reasoning: ${this.params.reasoning}`);
            console.debug(`📝 [SQLQueryTool] Query: ${this.params.query.substring(0, 200)}${this.params.query.length > 200 ? '...' : ''}`);
            // Create PostgreSQL bubble with default settings
            const pgBubble = new PostgreSQLBubble({
                query: this.params.query,
                allowedOperations: [
                    'SELECT',
                    'WITH',
                    'EXPLAIN',
                    'ANALYZE',
                    'SHOW',
                    'DESCRIBE',
                    'DESC',
                ],
                timeout: 30000, // 30 seconds
                maxRows: 1000, // Reasonable limit for analysis
                credentials: this.params.credentials,
                ...(this.params.config || {}),
            }, this.context);
            // Execute the query
            const result = await pgBubble.action();
            const executionTime = Date.now() - startTime;
            if (!result.success) {
                console.log(`❌ [SQLQueryTool] Query failed: ${result.error}`);
                return {
                    rowCount: 0,
                    executionTime,
                    success: false,
                    error: result.error,
                };
            }
            const rowCount = result.data?.rowCount || result.data?.rows?.length || 0;
            console.log(`✅ [SQLQueryTool] Query successful:`);
            console.log(`📊 [SQLQueryTool] Rows returned: ${rowCount}`);
            console.log(`⏱️  [SQLQueryTool] Execution time: ${executionTime}ms`);
            // Enhance result with metadata
            const enhancedResult = this.enhanceResult(result, executionTime);
            return enhancedResult;
        }
        catch (error) {
            const executionTime = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            console.log(`💥 [SQLQueryTool] Query error: ${errorMessage}`);
            return {
                rowCount: 0,
                executionTime,
                success: false,
                error: errorMessage,
            };
        }
    }
    /**
     * Validates SQL query before execution
     * Uses comprehensive security utilities to prevent SQL injection and other attacks
     * Prevents dangerous operations and enforces read-only access
     */
    validateQuery(query) {
        // Use the centralized security utility for SQL sanitization
        const sanitizationResult = sanitizeSQLQuery(query);
        if (!sanitizationResult.isSafe) {
            console.log(`❌ [SQLQueryTool] Query validation failed: ${sanitizationResult.reason}`);
            return {
                valid: false,
                error: sanitizationResult.reason || 'Query validation failed',
            };
        }
        // Additional check for empty query
        if (!query.trim()) {
            return {
                valid: false,
                error: 'Query cannot be empty',
            };
        }
        // Check query length to prevent DoS attacks
        const maxQueryLength = 10000; // 10KB max query size
        if (query.length > maxQueryLength) {
            return {
                valid: false,
                error: `Query too large (max ${maxQueryLength} characters)`,
            };
        }
        return { valid: true };
    }
    /**
     * Enhances query result with additional metadata and analysis
     */
    enhanceResult(result, executionTime) {
        const rows = result.data?.rows || [];
        const rowCount = result.data?.rowCount || rows.length;
        // Calculate statistics about the result
        const stats = this.calculateResultStats(rows);
        return {
            rows: result.data?.rows,
            rowCount,
            executionTime,
            fields: result.data?.fields,
            success: true,
            error: '',
        };
    }
    /**
     * Calculates statistics about query results
     */
    calculateResultStats(rows) {
        if (rows.length === 0) {
            return {
                hasData: false,
                columnCount: 0,
            };
        }
        const columns = Object.keys(rows[0]);
        return {
            hasData: true,
            columnCount: columns.length,
            columns,
            sampleRow: rows[0],
        };
    }
    /**
     * Formats query results as CSV string
     */
    formatAsCSV(rows) {
        if (rows.length === 0)
            return '';
        const headers = Object.keys(rows[0]);
        const csvRows = [headers.join(',')];
        rows.forEach((row) => {
            const values = headers.map((header) => {
                const value = row[header];
                // Escape values containing commas or quotes
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value ?? '';
            });
            csvRows.push(values.join(','));
        });
        return csvRows.join('\n');
    }
    /**
     * Formats query results as markdown table
     */
    formatAsMarkdown(rows) {
        if (rows.length === 0)
            return 'No results';
        const headers = Object.keys(rows[0]);
        const separator = headers.map(() => '---');
        const tableRows = [
            `| ${headers.join(' | ')} |`,
            `| ${separator.join(' | ')} |`,
        ];
        rows.forEach((row) => {
            const values = headers.map((header) => String(row[header] ?? ''));
            tableRows.push(`| ${values.join(' | ')} |`);
        });
        return tableRows.join('\n');
    }
    /**
     * Extracts sample queries for common database exploration tasks
     */
    static getSampleQueries() {
        return {
            listTables: 'SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\'',
            tableSchema: 'SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = $1 ORDER BY ordinal_position',
            tableSize: 'SELECT pg_size_pretty(pg_total_relation_size($1::regclass)) AS size',
            tableCount: 'SELECT COUNT(*) FROM table_name',
            topN: 'SELECT * FROM table_name ORDER BY created_at DESC LIMIT 10',
            indexes: 'SELECT indexname, indexdef FROM pg_indexes WHERE tablename = $1',
            databaseSize: 'SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size',
        };
    }
}
//# sourceMappingURL=sql-query-tool.js.map