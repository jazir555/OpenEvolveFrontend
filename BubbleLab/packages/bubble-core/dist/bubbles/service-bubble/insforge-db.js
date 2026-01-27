import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// Define available SQL operations (same as PostgreSQL for consistency)
export const SqlOperations = z.enum([
    'SELECT',
    'INSERT',
    'UPDATE',
    'DELETE',
    'CREATE',
    'WITH',
    'EXPLAIN',
]);
// Define the parameters schema for the InsForge DB bubble
const InsForgeDbParamsSchema = z.object({
    query: z
        .string()
        .min(1, 'Query is required')
        .describe('SQL query to execute against the InsForge database'),
    allowedOperations: z
        .array(SqlOperations)
        .default(['SELECT', 'WITH'])
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
        .describe('Query timeout in milliseconds (default: 30 seconds)'),
    maxRows: z
        .number()
        .positive()
        .default(1000)
        .describe('Maximum number of rows to return (default: 1000)'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Object mapping credential types to values (injected at runtime)'),
});
// Define the result schema
const InsForgeDbResultSchema = z.object({
    rows: z.array(z.record(z.unknown())).describe('Array of result rows'),
    rowCount: z
        .number()
        .nullable()
        .describe('Number of rows affected by the query'),
    command: z.string().describe('SQL command that was executed'),
    executionTime: z.number().describe('Query execution time in milliseconds'),
    success: z.boolean().describe('Whether the query executed successfully'),
    error: z.string().describe('Error message if query execution failed'),
    cleanedJSONString: z
        .string()
        .describe('Clean JSON string representation of the row data'),
});
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
export class InsForgeDbBubble extends ServiceBubble {
    static type = 'service';
    static service = 'insforge';
    static authType = 'apikey';
    static bubbleName = 'insforge-db';
    static schema = InsForgeDbParamsSchema;
    static resultSchema = InsForgeDbResultSchema;
    static shortDescription = 'InsForge is the backend built for AI-assisted development. Connect InsForge with any agent. Add authentication, database, storage, functions, and AI integrations to your app in seconds.';
    static longDescription = `
    Authentication - Complete user management system
    Database - Flexible data storage and retrieval
    Storage - File management and organization
    AI Integration - Chat completions and image generation (OpenAI-compatible)
    Serverless Functions - Scalable compute power
    Site Deployment (coming soon) - Easy application deployment
  `;
    static alias = 'insforge';
    constructor(params = {
        query: 'SELECT 1',
        allowedOperations: ['SELECT'],
        parameters: [],
        timeout: 30000,
        maxRows: 1000,
    }, context) {
        super(params, context);
        // Validate SQL operation
        this.validateSqlOperation(this.params.query, this.params.allowedOperations);
    }
    async testCredential() {
        const { credentials } = this.params;
        if (!credentials || typeof credentials !== 'object') {
            return false;
        }
        const baseUrl = credentials[CredentialType.INSFORGE_BASE_URL]?.replace(/\/$/, '');
        const apiKey = credentials[CredentialType.INSFORGE_API_KEY];
        // If only base URL provided, check if server is reachable
        if (baseUrl && !apiKey) {
            try {
                const response = await fetch(`${baseUrl}/api/health`, {
                    method: 'GET',
                });
                return response.ok;
            }
            catch {
                return false;
            }
        }
        // If only API key provided, can't validate without URL - return true
        if (apiKey && !baseUrl) {
            // Can't validate API key without base URL, assume valid
            return true;
        }
        // If both provided, do full validation
        if (baseUrl && apiKey) {
            try {
                const response = await fetch(`${baseUrl}/api/database/advance/rawsql/unrestricted`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        Authorization: `Bearer ${apiKey}`,
                    },
                    body: JSON.stringify({
                        query: 'SELECT 1 as test',
                        params: [],
                    }),
                });
                return response.ok;
            }
            catch {
                return false;
            }
        }
        return false;
    }
    chooseCredential() {
        // InsForge uses multiple credentials, handled in getCredentials()
        return undefined;
    }
    getCredentials() {
        const { credentials } = this.params;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('No InsForge credentials provided');
        }
        const baseUrl = credentials[CredentialType.INSFORGE_BASE_URL];
        const apiKey = credentials[CredentialType.INSFORGE_API_KEY];
        if (!baseUrl) {
            throw new Error('InsForge base URL not provided');
        }
        if (!apiKey) {
            throw new Error('InsForge API key not provided');
        }
        // Remove trailing slash from base URL
        return {
            baseUrl: baseUrl.replace(/\/$/, ''),
            apiKey,
        };
    }
    /**
     * Validate that the SQL query operation is allowed
     */
    validateSqlOperation(query, allowedOperations) {
        const trimmedQuery = query.trim().toUpperCase();
        const firstKeyword = trimmedQuery.split(/\s+/)[0];
        const isAllowed = allowedOperations.some((op) => firstKeyword.startsWith(op));
        if (!isAllowed) {
            throw new Error(`SQL operation '${firstKeyword}' is not allowed. Allowed operations: ${allowedOperations.join(', ')}`);
        }
        // Safety checks for dangerous operations
        if (firstKeyword === 'DELETE' && !trimmedQuery.includes('WHERE')) {
            throw new Error('DELETE queries must include a WHERE clause for safety');
        }
        if (firstKeyword === 'UPDATE' && !trimmedQuery.includes('WHERE')) {
            throw new Error('UPDATE queries must include a WHERE clause for safety');
        }
        // Block dangerous keywords
        const dangerousKeywords = [
            '\\bDROP\\b',
            '\\bALTER\\b',
            '\\bTRUNCATE\\b',
            '\\bGRANT\\b',
            '\\bREVOKE\\b',
        ];
        const containsDangerous = dangerousKeywords.some((keyword) => new RegExp(keyword, 'i').test(trimmedQuery));
        if (containsDangerous) {
            throw new Error(`Query contains potentially dangerous operations. Only allowed: ${allowedOperations.join(', ')}`);
        }
    }
    async performAction(context) {
        void context;
        const { query, parameters, maxRows } = this.params;
        const { baseUrl, apiKey } = this.getCredentials();
        const startTime = Date.now();
        try {
            // Call InsForge raw SQL endpoint
            const response = await fetch(`${baseUrl}/api/database/advance/rawsql/unrestricted`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${apiKey}`,
                },
                body: JSON.stringify({
                    query,
                    params: parameters,
                }),
            });
            if (!response.ok) {
                const errorBody = await response.text();
                throw new Error(`InsForge query failed: ${response.status} - ${errorBody}`);
            }
            const data = (await response.json());
            const executionTime = Date.now() - startTime;
            // Handle response - InsForge returns array of rows or object with rows property
            const rows = Array.isArray(data)
                ? data
                : data.rows || [];
            const truncatedRows = rows.slice(0, maxRows);
            // Extract command from query
            const command = query.trim().split(/\s+/)[0].toUpperCase();
            return {
                rows: truncatedRows,
                rowCount: rows.length,
                command,
                executionTime,
                success: true,
                error: '',
                cleanedJSONString: JSON.stringify(truncatedRows, null, 2),
            };
        }
        catch (error) {
            const executionTime = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            return {
                rows: [],
                rowCount: null,
                command: query.trim().split(/\s+/)[0].toUpperCase(),
                executionTime,
                success: false,
                error: errorMessage,
                cleanedJSONString: '[]',
            };
        }
    }
}
//# sourceMappingURL=insforge-db.js.map