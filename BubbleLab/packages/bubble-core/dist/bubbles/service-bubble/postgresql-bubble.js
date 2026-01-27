import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { Pool } from 'pg';
/**
 * PostgreSQL Bubble - Relational Database Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. executeQuery - Execute a SQL query
 * 2. executeBatch - Execute multiple queries in a transaction
 * 3. insertRow - Insert a single row
 * 4. updateRows - Update rows matching conditions
 * 5. deleteRows - Delete rows matching conditions
 * 6. selectRows - Select rows with optional filtering
 * 7. createTable - Create a new table
 * 8. dropTable - Drop a table
 * 9. tableExists - Check if a table exists
 * 10. tableInfo - Get table structure information
 */
// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================
const ExecuteQueryParamsSchema = z.object({
    operation: z.literal('executeQuery'),
    query: z.string().min(1, 'SQL query is required'),
    params: z.array(z.any()).optional().describe('Query parameters for prepared statements'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const ExecuteBatchParamsSchema = z.object({
    operation: z.literal('executeBatch'),
    queries: z.array(z.object({
        query: z.string().min(1, 'SQL query is required'),
        params: z.array(z.any()).optional(),
    })).min(1, 'At least one query is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const InsertRowParamsSchema = z.object({
    operation: z.literal('insertRow'),
    table: z.string().min(1, 'Table name is required'),
    data: z.record(z.any()).describe('Column-value pairs to insert'),
    returning: z.array(z.string()).optional().describe('Columns to return after insert'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const UpdateRowsParamsSchema = z.object({
    operation: z.literal('updateRows'),
    table: z.string().min(1, 'Table name is required'),
    data: z.record(z.any()).describe('Column-value pairs to update'),
    where: z.record(z.any()).describe('Filter conditions for rows to update'),
    returning: z.array(z.string()).optional().describe('Columns to return after update'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const DeleteRowsParamsSchema = z.object({
    operation: z.literal('deleteRows'),
    table: z.string().min(1, 'Table name is required'),
    where: z.record(z.any()).describe('Filter conditions for rows to delete'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const SelectRowsParamsSchema = z.object({
    operation: z.literal('selectRows'),
    table: z.string().min(1, 'Table name is required'),
    columns: z.array(z.string()).optional().default(['*']).describe('Columns to select'),
    where: z.record(z.any()).optional().describe('Filter conditions'),
    orderBy: z.array(z.string()).optional().describe('Order by columns'),
    limit: z.number().int().positive().optional().describe('Maximum rows to return'),
    offset: z.number().int().nonnegative().optional().default(0),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const CreateTableParamsSchema = z.object({
    operation: z.literal('createTable'),
    table: z.string().min(1, 'Table name is required'),
    columns: z.record(z.object({
        type: z.string().describe('PostgreSQL column type (e.g., VARCHAR(255), INTEGER)'),
        constraints: z.array(z.string()).optional().describe('Constraints like NOT NULL, UNIQUE'),
        default: z.any().optional(),
    })).describe('Column definitions'),
    primaryKey: z.array(z.string()).optional().describe('Primary key columns'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const DropTableParamsSchema = z.object({
    operation: z.literal('dropTable'),
    table: z.string().min(1, 'Table name is required'),
    ifExists: z.boolean().optional().default(true),
    cascade: z.boolean().optional().default(false),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const TableExistsParamsSchema = z.object({
    operation: z.literal('tableExists'),
    table: z.string().min(1, 'Table name is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const TableInfoParamsSchema = z.object({
    operation: z.literal('tableInfo'),
    table: z.string().min(1, 'Table name is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
// Union of all parameter schemas
const PostgresqlBubbleParamsSchema = z.discriminatedUnion('operation', [
    ExecuteQueryParamsSchema,
    ExecuteBatchParamsSchema,
    InsertRowParamsSchema,
    UpdateRowsParamsSchema,
    DeleteRowsParamsSchema,
    SelectRowsParamsSchema,
    CreateTableParamsSchema,
    DropTableParamsSchema,
    TableExistsParamsSchema,
    TableInfoParamsSchema,
]);
// Result schema
const PostgresqlBubbleResultSchema = z.object({
    success: z.boolean(),
    data: z.unknown().describe('Operation result data'),
    error: z.string(),
    meta: z.object({
        operation: z.string(),
        table: z.string().optional(),
        rowsAffected: z.number().optional(),
    }),
});
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class PostgresqlBubble extends ServiceBubble {
    static service = 'postgresql';
    static authType = 'password';
    static bubbleName = 'postgresql';
    static type = 'service';
    static schema = PostgresqlBubbleParamsSchema;
    static resultSchema = PostgresqlBubbleResultSchema;
    static shortDescription = 'Advanced open source relational database';
    static longDescription = `
    PostgreSQL Bubble for relational data management.

    Features:
    - ACID compliant transactions
    - Complex queries with JOINs and subqueries
    - Support for JSON/JSONB data types
    - Full-text search capabilities
    - Extensible with custom functions
    - Advanced indexing options

    Use cases:
    - Primary application database
    - Complex data relationships
    - Financial transactions
    - Analytics and reporting
    - Geospatial data with PostGIS
  `;
    static alias = 'postgres';
    pool = null;
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    getCredentialType() {
        return CredentialType.POSTGRESQL_CRED;
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('PostgreSQL credentials are required');
        }
        return credentials[CredentialType.POSTGRESQL_CRED];
    }
    async testCredential() {
        try {
            const pool = this.getPool();
            const client = await pool.connect();
            await client.query('SELECT 1');
            client.release();
            return true;
        }
        catch (error) {
            console.error('[PostgreSQL] Credential test failed:', error);
            return false;
        }
    }
    getPool() {
        if (!this.pool) {
            const credential = this.chooseCredential();
            if (!credential) {
                throw new Error('PostgreSQL credentials not found');
            }
            // Parse credential (expected format: JSON string with connection details)
            let config;
            try {
                config = typeof credential === 'string' ? JSON.parse(credential) : credential;
            }
            catch {
                throw new Error('Invalid PostgreSQL credentials format. Expected JSON string.');
            }
            if (!config.host && !config.url) {
                throw new Error('PostgreSQL host or URL is required');
            }
            // Build pool configuration
            const poolConfig = {
                max: config.maxConnections || 20,
                idleTimeoutMillis: config.idleTimeout || 30000,
                connectionTimeoutMillis: config.connectionTimeout || 2000,
            };
            // Use URL if provided, otherwise build from components
            if (config.url) {
                poolConfig.connectionString = config.url;
            }
            else {
                poolConfig.host = config.host || 'localhost';
                poolConfig.port = config.port || 5432;
                poolConfig.database = config.database || 'postgres';
                poolConfig.user = config.user || 'postgres';
                poolConfig.password = config.password;
            }
            // Add SSL configuration if provided
            if (config.ssl) {
                poolConfig.ssl = config.ssl;
            }
            this.pool = new Pool(poolConfig);
            this.pool.on('error', (err) => {
                console.error('[PostgreSQL] Unexpected error on idle client', err);
            });
            console.log('[PostgreSQL] Pool initialized successfully');
        }
        return this.pool;
    }
    async performAction(context) {
        void context;
        try {
            const pool = this.getPool();
            const operation = this.params.operation;
            let result;
            let rowsAffected = 0;
            console.log(`[PostgreSQL] Executing operation: ${operation}`);
            switch (operation) {
                case 'executeQuery':
                    result = await this.executeQuery(pool);
                    rowsAffected = result.rowCount || 0;
                    break;
                case 'executeBatch':
                    result = await this.executeBatch(pool);
                    rowsAffected = result.totalRows || 0;
                    break;
                case 'insertRow':
                    result = await this.insertRow(pool);
                    rowsAffected = result.rowCount || 0;
                    break;
                case 'updateRows':
                    result = await this.updateRows(pool);
                    rowsAffected = result.rowCount || 0;
                    break;
                case 'deleteRows':
                    result = await this.deleteRows(pool);
                    rowsAffected = result.rowCount || 0;
                    break;
                case 'selectRows':
                    result = await this.selectRows(pool);
                    rowsAffected = result.rowCount || 0;
                    break;
                case 'createTable':
                    result = await this.createTable(pool);
                    break;
                case 'dropTable':
                    result = await this.dropTable(pool);
                    break;
                case 'tableExists':
                    result = await this.tableExists(pool);
                    break;
                case 'tableInfo':
                    result = await this.tableInfo(pool);
                    break;
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
            return {
                success: true,
                data: result,
                error: '', // Empty string for successful operations,
                meta: {
                    operation,
                    table: this.extractTableName(),
                    rowsAffected,
                },
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[PostgreSQL] Operation failed:`, errorMessage);
            return {
                success: false,
                data: null,
                error: errorMessage,
                meta: {
                    operation: this.params.operation,
                    table: this.extractTableName(),
                },
            };
        }
    }
    async executeQuery(pool) {
        const params = this.params;
        console.log(`[PostgreSQL] Executing query: ${params.query.substring(0, 100)}...`);
        const result = await pool.query(params.query, params.params);
        console.log(`[PostgreSQL] Query completed, ${result.rows.length} rows returned`);
        return {
            rows: result.rows,
            rowCount: result.rowCount,
            fields: result.fields,
        };
    }
    async executeBatch(pool) {
        const params = this.params;
        console.log(`[PostgreSQL] Executing batch of ${params.queries.length} queries`);
        const client = await pool.connect();
        let totalRows = 0;
        const results = [];
        try {
            await client.query('BEGIN');
            for (const queryDef of params.queries) {
                const result = await client.query(queryDef.query, queryDef.params);
                results.push({
                    query: queryDef.query,
                    rows: result.rowCount || 0,
                });
                totalRows += result.rowCount || 0;
            }
            await client.query('COMMIT');
            console.log(`[PostgreSQL] Batch completed, ${totalRows} total rows affected`);
        }
        catch (error) {
            await client.query('ROLLBACK');
            throw error;
        }
        finally {
            client.release();
        }
        return {
            results: results,
            totalRows: totalRows,
            count: params.queries.length,
        };
    }
    async insertRow(pool) {
        const params = this.params;
        const columns = Object.keys(params.data);
        const values = Object.values(params.data);
        const placeholders = values.map((_, i) => `$${i + 1}`).join(', ');
        let query = `INSERT INTO ${params.table} (${columns.join(', ')}) VALUES (${placeholders})`;
        if (params.returning && params.returning.length > 0) {
            query += ` RETURNING ${params.returning.join(', ')}`;
        }
        const result = await pool.query(query, values);
        console.log(`[PostgreSQL] Inserted row into ${params.table}`);
        return {
            table: params.table,
            rowCount: result.rowCount,
            returning: result.rows,
        };
    }
    async updateRows(pool) {
        const params = this.params;
        const setClause = Object.keys(params.data)
            .map((col, i) => `${col} = $${i + 1}`)
            .join(', ');
        const values = [...Object.values(params.data)];
        const whereClauses = [];
        for (const [key, value] of Object.entries(params.where)) {
            whereClauses.push(`${key} = $${values.length + 1}`);
            values.push(value);
        }
        let query = `UPDATE ${params.table} SET ${setClause} WHERE ${whereClauses.join(' AND ')}`;
        if (params.returning && params.returning.length > 0) {
            query += ` RETURNING ${params.returning.join(', ')}`;
        }
        const result = await pool.query(query, values);
        console.log(`[PostgreSQL] Updated ${result.rowCount} rows in ${params.table}`);
        return {
            table: params.table,
            rowCount: result.rowCount,
            returning: result.rows,
        };
    }
    async deleteRows(pool) {
        const params = this.params;
        const values = [];
        const whereClauses = Object.keys(params.where).map((key) => {
            values.push(params.where[key]);
            return `${key} = $${values.length}`;
        });
        const query = `DELETE FROM ${params.table} WHERE ${whereClauses.join(' AND ')}`;
        const result = await pool.query(query, values);
        console.log(`[PostgreSQL] Deleted ${result.rowCount} rows from ${params.table}`);
        return {
            table: params.table,
            rowCount: result.rowCount,
        };
    }
    async selectRows(pool) {
        const params = this.params;
        const columns = params.columns.join(', ');
        let query = `SELECT ${columns} FROM ${params.table}`;
        const values = [];
        if (params.where && Object.keys(params.where).length > 0) {
            const whereClauses = Object.keys(params.where).map((key) => {
                values.push(params.where[key]);
                return `${key} = $${values.length}`;
            });
            query += ` WHERE ${whereClauses.join(' AND ')}`;
        }
        if (params.orderBy && params.orderBy.length > 0) {
            query += ` ORDER BY ${params.orderBy.join(', ')}`;
        }
        if (params.limit) {
            query += ` LIMIT ${params.limit}`;
        }
        if (params.offset) {
            query += ` OFFSET ${params.offset}`;
        }
        const result = await pool.query(query, values);
        console.log(`[PostgreSQL] Selected ${result.rows.length} rows from ${params.table}`);
        return {
            table: params.table,
            rows: result.rows,
            rowCount: result.rowCount,
        };
    }
    async createTable(pool) {
        const params = this.params;
        const columnDefs = Object.entries(params.columns).map(([name, def]) => {
            let colDef = `${name} ${def.type}`;
            if (def.constraints && def.constraints.length > 0) {
                colDef += ' ' + def.constraints.join(' ');
            }
            if (def.default !== undefined) {
                colDef += ` DEFAULT ${typeof def.default === 'string' ? `'${def.default}'` : def.default}`;
            }
            return colDef;
        });
        if (params.primaryKey && params.primaryKey.length > 0) {
            columnDefs.push(`PRIMARY KEY (${params.primaryKey.join(', ')})`);
        }
        const query = `CREATE TABLE ${params.table} (${columnDefs.join(', ')})`;
        await pool.query(query);
        console.log(`[PostgreSQL] Created table: ${params.table}`);
        return {
            table: params.table,
            columns: Object.keys(params.columns),
            status: 'created',
        };
    }
    async dropTable(pool) {
        const params = this.params;
        let query = 'DROP TABLE';
        if (params.ifExists) {
            query += ' IF EXISTS';
        }
        query += ` ${params.table}`;
        if (params.cascade) {
            query += ' CASCADE';
        }
        await pool.query(query);
        console.log(`[PostgreSQL] Dropped table: ${params.table}`);
        return {
            table: params.table,
            status: 'dropped',
        };
    }
    async tableExists(pool) {
        const params = this.params;
        const query = `
      SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = $1
      )
    `;
        const result = await pool.query(query, [params.table]);
        const exists = result.rows[0].exists;
        console.log(`[PostgreSQL] Table ${params.table} ${exists ? 'exists' : 'does not exist'}`);
        return {
            table: params.table,
            exists: exists,
        };
    }
    async tableInfo(pool) {
        const params = this.params;
        const query = `
      SELECT
        column_name,
        data_type,
        is_nullable,
        column_default,
        character_maximum_length
      FROM information_schema.columns
      WHERE table_name = $1
      ORDER BY ordinal_position
    `;
        const result = await pool.query(query, [params.table]);
        console.log(`[PostgreSQL] Retrieved info for table: ${params.table}, ${result.rows.length} columns`);
        return {
            table: params.table,
            columns: result.rows,
            columnCount: result.rows.length,
        };
    }
    extractTableName() {
        const params = this.params;
        return params.table;
    }
}
//# sourceMappingURL=postgresql-bubble.js.map