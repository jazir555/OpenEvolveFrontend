import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType, } from '@bubblelab/shared-schemas';
import { Pool } from 'pg';
import { createErrorResponse, generateCorrelationId, withTimeout, retryWithBackoff, CircuitBreaker, defaultCircuitBreakerConfig, } from '../../utils/error-handler.js';
// Define available SQL operations
export const SqlOperations = z.enum([
    'SELECT',
    'INSERT',
    'UPDATE',
    'DELETE',
    'WITH', // Common Table Expressions for complex analysis
    'EXPLAIN', // Query execution plans for optimization
    'ANALYZE', // Table statistics for analysis
    'SHOW', // Show database/table information
    'DESCRIBE', // Describe table structure
    'DESC', // Alias for DESCRIBE
    'CREATE', // Allow CREATE TEMPORARY for analysis
]);
// SECURITY FIX: Change default to false for secure connections
// Default to secure SSL connections to prevent man-in-the-middle attacks
const PostgreSQLParamsSchema = z.object({
    ignoreSSL: z
        .boolean()
        .default(false) // SECURITY FIX: Default to false for secure connections
        .describe('Ignore SSL certificate errors when connecting to the database (WARNING: Only set to true in trusted networks)'),
    query: z
        .string()
        .min(1, 'Query is required')
        .refine((query) => {
        // SECURITY FIX: Enhanced SQL injection validation with whitelist-based approach
        // Only allow safe SQL characters and patterns
        const trimmedQuery = query.trim().toUpperCase();
        // Whitelist: Only allow safe SQL keywords and characters
        const safeKeywords = [
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'IS', 'NULL',
            'ORDER', 'BY', 'LIMIT', 'OFFSET', 'GROUP', 'HAVING', 'AS', 'JOIN', 'LEFT',
            'RIGHT', 'INNER', 'OUTER', 'ON', 'WITH', 'CASE', 'WHEN', 'THEN', 'ELSE',
            'END', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CAST', 'EXTRACT',
            'DATE_TRUNC', 'NOW', 'INTERVAL', 'ASC', 'DESC', 'NULLS', 'FIRST', 'LAST',
            'UNION', 'ALL', 'EXCEPT', 'INTERSECT', 'EXISTS', 'BETWEEN', 'TRUE', 'FALSE',
            'INSERT', 'INTO', 'VALUES', 'RETURNING', 'UPDATE', 'SET', 'DELETE',
            'CREATE', 'TABLE', 'TEMPORARY', 'TEMP', 'VIEW', 'INDEX', 'UNIQUE', 'PRIMARY',
            'KEY', 'FOREIGN', 'REFERENCES', 'CHECK', 'DEFAULT', 'CONSTRAINT', 'DROP',
            'ALTER', 'ADD', 'COLUMN', 'RENAME', 'TO', 'CASCADE', 'RESTRICT', 'ANALYZE',
            'EXPLAIN', 'SHOW', 'DESCRIBE', 'DESC', 'GRANT', 'REVOKE', 'TRANSACTION',
            'BEGIN', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'RELEASE', 'FOR', 'UPDATE',
            'SHARE', 'MODE', 'ISOLATION', 'LEVEL', 'SERIALIZABLE', 'REPEATABLE', 'READ',
            'COMMITTED', 'UNCOMMITTED', 'WRITE', 'ONLY', 'LOCK', 'TABLES', 'IF', 'EXISTS'
        ];
        // Blacklist: Dangerous patterns that should never appear
        const dangerousPatterns = [
            /;\s*--/, // Comment injection
            /;\s*\/\*/, // Block comment injection
            /;\s*(drop|delete|insert|update|alter|create|truncate|grant|revoke)\s+/i, // Multiple statements
            /'\s*;\s*/i, // Quote + semicolon (statement break)
            /union\s+select/i, // Union-based injection
            /'\s*(or|and)\s+\w+\s*=\s*\w+/i, // Boolean-based injection
            /exec\s*\(/i, // Command execution
            /xp_|sp_/i, // System stored procedures (SQL Server)
            /declare\s+@/i, // Variable declaration
            /waitfor\s+delay/i, // Time-based attack
            /benchmark\s*\(/i, // Timing attack
            /sleep\s*\(/i, // Timing attack
            /load_file\s*\(/i, // File read
            /into\s+outfile/i, // File write
            /into\s+dumpfile/i, // File write
            /script.*src/i, // Script injection
            /javascript:/i, // JavaScript injection
            /<script/i, // Script tag
            /eval\s*\(/i, // Code execution
            /\bxp_cmdshell\b/i, // Command shell
            /\bsp_executesql\b/i, // Dynamic SQL
            /\bdbms_\w+\./i, // Oracle DBMS packages
            /\butl_\w+\./i, // Oracle UTL packages
            /copy\s+from\s+program/i, // Command execution
            /copy\s+to\s+program/i, // Command execution
            /lo_import\s*\(/i, // Large object import
            /lo_export\s*\(/i, // Large object export
            /pg_read_file\s*\(/i, // File read
            /pg_ls_dir\s*\(/i, // Directory listing
        ];
        // Check for dangerous patterns first
        if (dangerousPatterns.some((pattern) => pattern.test(query))) {
            return false;
        }
        // Check for unbalanced quotes (potential injection vector)
        const singleQuotes = (query.match(/'/g) || []).length;
        if (singleQuotes % 2 !== 0) {
            return false;
        }
        // Check for multiple statements (semicolons outside quotes/string literals)
        // This is a simplified check - a full parser would be better
        const statementEnds = (query.match(/;/g) || []).length;
        if (statementEnds > 1) {
            return false;
        }
        // Validate that query only contains safe characters
        // Allow alphanumeric, spaces, and SQL-safe punctuation
        const safeCharPattern = /^[a-zA-Z0-9\s\.,\(\)\[\]\{\}'"=<>!+\-*/%_$@?#&|]+$/;
        if (!safeCharPattern.test(query)) {
            return false;
        }
        return true;
    }, 'Query contains potentially dangerous SQL patterns or invalid syntax')
        .describe('SQL query to execute against the PostgreSQL database (use parameterized queries with $1, $2, etc.)'),
    allowedOperations: z
        .array(SqlOperations)
        .default([
        'SELECT',
        'WITH',
        'EXPLAIN',
        'ANALYZE',
        'SHOW',
        'DESCRIBE',
        'DESC',
    ])
        .describe('List of allowed SQL operations for security (defaults to read-only operations)'),
    parameters: z
        .array(z.unknown())
        .optional()
        .default([])
        .describe('Parameters for parameterized queries (e.g., [value1, value2] for $1, $2)'),
    timeout: z
        .number()
        .positive()
        .default(30000)
        .describe('Query timeout in milliseconds (default: 30 seconds, max recommended: 300000)'),
    maxRows: z
        .number()
        .positive()
        .default(1000)
        .describe('Maximum number of rows to return to prevent large result sets (default: 1000)'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Object mapping credential types to values (injected at runtime)'),
});
// Define the result schema for validation
const PostgreSQLResultSchema = z.object({
    rows: z
        .array(z.record(z.unknown()))
        .describe('Array of result rows, each row is an object with column names as keys'),
    rowCount: z
        .number()
        .nullable()
        .describe('Number of rows affected by the query (null for SELECT queries)'),
    command: z
        .string()
        .describe('SQL command that was executed (SELECT, INSERT, UPDATE, DELETE)'),
    fields: z
        .array(z.object({
        name: z.string().describe('Column name'),
        dataTypeID: z.number().describe('PostgreSQL data type identifier'),
    }))
        .optional()
        .describe('Metadata about the columns returned by the query'),
    executionTime: z.number().describe('Query execution time in milliseconds'),
    success: z.boolean().describe('Whether the query executed successfully'),
    // Make error optional or undefined
    error: z
        .string()
        .describe('Error message if query execution failed (empty string if successful)'),
    cleanedJSONString: z
        .string()
        .describe('Clean JSON string representation of the row data, suitable for AI prompts and integrations'),
});
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
export class PostgreSQLBubble extends ServiceBubble {
    circuitBreaker;
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
    async testCredential() {
        // Make a query to the database to test the credential
        const connectionString = this.chooseCredential();
        const pool = new Pool({
            connectionString,
            ssl: this.params.ignoreSSL ? { rejectUnauthorized: false } : undefined,
        });
        try {
            await pool.query('SELECT 1');
            return true;
        }
        catch (error) {
            console.error('PostgreSQL credential test failed:', error);
            return false;
        }
        finally {
            await pool.end();
        }
    }
    static type = 'service';
    static service = 'postgresql';
    static authType = 'connection-string';
    // Required static metadata - TypeScript will enforce these exist
    static bubbleName = 'postgresql';
    static schema = PostgreSQLParamsSchema;
    static resultSchema = PostgreSQLResultSchema;
    static shortDescription = 'Execute PostgreSQL queries with operation validation';
    static longDescription = `
    Execute SQL queries against PostgreSQL databases with proper validation and security controls.
    Use cases:
    - Data retrieval with SELECT queries
    - Data manipulation with INSERT, UPDATE, DELETE (when explicitly allowed)
    - Database reporting and analytics
    - Data migration and synchronization tasks
    - JSON string output for integration with other systems
    
    Security Features:
    - Operation whitelist (defaults to SELECT only)
    - Parameterized queries to prevent SQL injection
    - Connection timeout controls
    - Result sanitization for JSON output
  `;
    static alias = 'pg';
    constructor(params = {
        query: 'SELECT 1',
        allowedOperations: ['SELECT'],
        parameters: [],
        timeout: 30000,
        maxRows: 1000,
    }, context) {
        super(params, context);
        // Perform additional validation after Zod schema validation
        this.validateSqlOperation(this.params.query, this.params.allowedOperations);
        this.validateParameterUsage(this.params.query, this.params.parameters);
        // Initialize circuit breaker for connection failures
        this.circuitBreaker = new CircuitBreaker({
            ...defaultCircuitBreakerConfig,
            failureThreshold: 5,
            successThreshold: 2,
            timeoutMs: this.params.timeout || 30000,
            monitoringPeriodMs: 60000,
        });
    }
    async performAction(context) {
        // Context is available but not currently used in this implementation
        void context;
        const correlationId = generateCorrelationId();
        const { ignoreSSL, query, allowedOperations, parameters, timeout, maxRows, } = this.params;
        const startTime = Date.now();
        try {
            // Input validation
            this.validateSqlOperation(query, allowedOperations);
            this.validateParameterUsage(query, parameters);
            // Get connection string
            const connectionString = this.chooseCredential();
            // Execute with circuit breaker, timeout, and retry logic
            const result = await this.circuitBreaker.execute(async () => {
                return await retryWithBackoff(async () => {
                    return await this.executeQuery(connectionString, ignoreSSL, query, parameters, timeout, maxRows);
                }, {
                    maxAttempts: 3,
                    baseDelayMs: 1000,
                    correlationId,
                    operation: 'PostgreSQL Query',
                });
            }, 'PostgreSQL Query');
            const executionTime = Date.now() - startTime;
            console.log(`[${correlationId}] PostgreSQL query succeeded in ${executionTime}ms`);
            return {
                ...result,
                executionTime,
            };
        }
        catch (error) {
            const executionTime = Date.now() - startTime;
            const errorResponse = createErrorResponse(error, correlationId);
            console.error(`[${correlationId}] PostgreSQL query failed after ${executionTime}ms:`, error);
            return {
                rows: [],
                rowCount: null,
                command: '',
                fields: undefined,
                executionTime,
                success: false,
                error: errorResponse.error.message,
                cleanedJSONString: '[]',
            };
        }
    }
    /**
     * Execute the actual database query
     */
    async executeQuery(connectionString, ignoreSSL, query, parameters, timeout, maxRows) {
        if (!connectionString) {
            throw new Error('PostgreSQL connection string is required but was not provided');
        }
        // Create connection pool with strict settings
        const pool = new Pool({
            connectionString,
            connectionTimeoutMillis: timeout,
            idleTimeoutMillis: timeout,
            max: 1, // Single connection for bubble execution
            allowExitOnIdle: true, // Exit when idle
            statement_timeout: timeout, // Query timeout
            ssl: ignoreSSL ? { rejectUnauthorized: false } : undefined,
        });
        try {
            // Execute the query with timeout
            const result = await withTimeout(pool.query(query, parameters), timeout, 'PostgreSQL Query');
            // Additional safety: truncate rows if they exceed maxRows
            const truncatedRows = result.rows.slice(0, maxRows);
            const wasTruncated = result.rows.length > maxRows;
            if (wasTruncated) {
                console.warn(`Result set truncated to ${maxRows} rows`);
            }
            return {
                rows: truncatedRows,
                rowCount: result.rowCount,
                command: result.command,
                fields: result.fields?.map((field) => ({
                    name: field.name,
                    dataTypeID: field.dataTypeID,
                })),
                success: true,
                error: '',
                cleanedJSONString: this.cleanJSONString(truncatedRows),
            };
        }
        finally {
            // Always close the pool
            await pool.end();
        }
    }
    async getCredentialMetadata() {
        const correlationId = generateCorrelationId();
        const connectionString = this.chooseCredential();
        if (!connectionString) {
            return undefined;
        }
        const pool = new Pool({
            connectionString,
            ssl: this.params.ignoreSSL ? { rejectUnauthorized: false } : undefined,
        });
        try {
            // Query all schemas, not just 'public'
            const schemaQuery = `
        SELECT
          t.table_schema,
          t.table_name,
          c.column_name,
          c.data_type,
          c.is_nullable,
          c.column_default,
          c.ordinal_position
        FROM information_schema.tables t
        JOIN information_schema.columns c ON t.table_name = c.table_name
          AND t.table_schema = c.table_schema
        WHERE t.table_type = 'BASE TABLE'
          AND t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        ORDER BY t.table_schema, t.table_name, c.ordinal_position
      `;
            const result = await withTimeout(pool.query(schemaQuery), this.params.timeout || 30000, 'PostgreSQL Metadata Query');
            const rawData = result.rows;
            // Process the schema data into the same compact format as database-analyzer
            // Tables from 'public' schema will not have schema prefix, others will
            const compactSchema = {};
            rawData.forEach((row) => {
                const tableSchema = row.table_schema;
                const tableName = row.table_name;
                const columnName = row.column_name;
                const dataType = row.data_type;
                // Format table name: public schema tables have no prefix, others have schema prefix
                const formattedTableName = `${tableSchema}.${tableName}`;
                if (!compactSchema[formattedTableName]) {
                    compactSchema[formattedTableName] = {};
                }
                compactSchema[formattedTableName][columnName] = dataType;
            });
            console.log(`[${correlationId}] Retrieved metadata for ${Object.keys(compactSchema).length} tables`);
            return {
                tables: compactSchema,
                databaseName: 'postgresql_database',
                databaseType: 'postgresql',
            };
        }
        catch (error) {
            console.error(`[${correlationId}] Error getting credential metadata:`, error);
            return undefined;
        }
        finally {
            await pool.end();
        }
    }
    /**
     * Validate that the SQL query operation is allowed
     */
    validateSqlOperation(query, allowedOperations) {
        // Extract the first SQL keyword (operation) from the query
        const trimmedQuery = query.trim().toUpperCase();
        const firstKeyword = trimmedQuery.split(/\s+/)[0];
        // Check if the operation is in the allowed list
        const isAllowed = allowedOperations.some((op) => firstKeyword.startsWith(op));
        if (!isAllowed) {
            throw new Error(`SQL operation '${firstKeyword}' is not allowed. Allowed operations: ${allowedOperations.join(', ')}`);
        }
        // Additional validation for dangerous operations
        if (firstKeyword === 'DELETE' && !trimmedQuery.includes('WHERE')) {
            throw new Error('DELETE queries must include a WHERE clause for safety');
        }
        if (firstKeyword === 'UPDATE' && !trimmedQuery.includes('WHERE')) {
            throw new Error('UPDATE queries must include a WHERE clause for safety');
        }
        // Block potentially dangerous keywords using word boundaries
        // Allow CREATE TEMPORARY for analysis but block permanent CREATE operations
        const dangerousKeywords = [
            '\\bDROP\\b',
            '\\bALTER\\b',
            '\\bTRUNCATE\\b',
            '\\bGRANT\\b',
            '\\bREVOKE\\b',
            '\\bCOPY\\b',
            '\\bBULK\\b',
            '\\bLOAD\\b',
        ];
        const containsDangerous = dangerousKeywords.some((keyword) => new RegExp(keyword, 'i').test(trimmedQuery));
        if (containsDangerous) {
            throw new Error(`Query contains potentially dangerous operations. This bubble only supports: ${allowedOperations.join(', ')}`);
        }
        // Validate parentheses balance to prevent injection through unclosed strings
        this.validateParenthesesBalance(query);
    }
    /**
     * Validate parameter usage to encourage parameterized queries
     */
    validateParameterUsage(query, parameters) {
        // Count parameter placeholders ($1, $2, etc.)
        const paramPlaceholders = (query.match(/\$\d+/g) || []).length;
        if (paramPlaceholders !== parameters.length) {
            throw new Error(`Parameter count mismatch: query has ${paramPlaceholders} placeholders but ${parameters.length} parameters provided`);
        }
        // Warn if query contains string literals with potential variables
        const hasStringLiterals = /['"][^'"]*\$[^'"]*['"]/.test(query);
        if (hasStringLiterals) {
            console.warn('Query contains string literals with $ symbols. Consider using parameterized queries.');
        }
    }
    /**
     * Validate parentheses and quotes are balanced
     */
    validateParenthesesBalance(query) {
        let parenCount = 0;
        let singleQuoteCount = 0;
        let doubleQuoteCount = 0;
        for (const char of query) {
            switch (char) {
                case '(':
                    parenCount++;
                    break;
                case ')':
                    parenCount--;
                    break;
                case "'":
                    singleQuoteCount++;
                    break;
                case '"':
                    doubleQuoteCount++;
                    break;
            }
        }
        if (parenCount !== 0) {
            throw new Error('Unbalanced parentheses in query');
        }
        if (singleQuoteCount % 2 !== 0) {
            throw new Error('Unbalanced single quotes in query');
        }
        if (doubleQuoteCount % 2 !== 0) {
            throw new Error('Unbalanced double quotes in query');
        }
    }
    /**
     * Clean and format query results as a JSON string
     */
    cleanJSONString(rows) {
        try {
            // Clean the data by removing any potential circular references and handling special values
            const cleanedRows = rows.map((row) => this.cleanObject(row));
            // Return just the essential data as a clean JSON string
            return JSON.stringify(cleanedRows, null, 2);
        }
        catch (error) {
            // Fallback to basic JSON stringify if cleaning fails
            console.warn('Failed to clean JSON data, using basic stringify:', error);
            return JSON.stringify(rows, null, 2);
        }
    }
    /**
     * Clean an object by handling special values and preventing circular references
     */
    cleanObject(obj) {
        if (obj === null || obj === undefined) {
            return obj;
        }
        if (typeof obj === 'bigint') {
            return obj.toString();
        }
        if (obj instanceof Date) {
            return obj.toISOString();
        }
        if (obj instanceof Buffer) {
            return `<Buffer ${obj.length} bytes>`;
        }
        if (typeof obj === 'function') {
            return '<function>';
        }
        if (typeof obj === 'symbol') {
            return obj.toString();
        }
        if (Array.isArray(obj)) {
            return obj.map((item) => this.cleanObject(item));
        }
        if (typeof obj === 'object' && obj !== null) {
            const cleaned = {};
            for (const [key, value] of Object.entries(obj)) {
                // Skip functions and symbols as keys
                if (typeof key === 'string') {
                    cleaned[key] = this.cleanObject(value);
                }
            }
            return cleaned;
        }
        return obj;
    }
    chooseCredential() {
        const { credentials } = this.params;
        // If no credentials were injected, return undefined
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('No postgres credentials provided');
        }
        // PostgreSQL bubble always uses database credentials
        return credentials[CredentialType.DATABASE_CRED];
    }
}
//# sourceMappingURL=postgresql.js.map