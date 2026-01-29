/**
 * PostgreSQL Database Service Bubble (Extended)
 *
 * Enhanced PostgreSQL integration with connection pooling,
 * transaction support, and OpenEvolve-specific utilities.
 */

import { z } from 'zod';
import { PostgreSQLBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';

const ExtendedPostgreSQLOperationSchema = z.enum([
  'query',
  'execute',
  'batch_execute',
  'transaction',
  'health_check',
  'schema_info',
  'table_info',
  'backup',
  'restore',
]);

const PostgresParamsSchema = z.object({
  operation: ExtendedPostgreSQLOperationSchema.describe('Database operation'),
  connectionString: z.string().describe('PostgreSQL connection string'),
  query: z.string().optional().describe('SQL query'),
  params: z.array(z.unknown()).optional().describe('Query parameters'),
  queries: z.array(z.object({
    query: z.string(),
    params: z.array(z.unknown()).optional(),
  })).optional().describe('Batch queries'),

  // Transaction operations
  transactionQueries: z.array(z.object({
    query: z.string(),
    params: z.array(z.unknown()).optional(),
  })).optional().describe('Queries within transaction'),

  // Schema operations
  tableName: z.string().optional().describe('Table name for info operations'),
  schemaName: z.string().default('public').describe('Schema name'),

  // Backup/restore
  backupPath: z.string().optional().describe('Backup file path'),
  restorePath: z.string().optional().describe('Restore file path'),
  databaseName: z.string().optional().describe('Database name for backup/restore'),

  // Connection pool
  poolSize: z.number().min(1).max(100).default(10),
  timeout: z.number().min(1000).max(120000).default(30000),
});

type PostgresParamsInput = z.input<typeof PostgresParamsSchema>;
type PostgresParams = z.output<typeof PostgresParamsSchema>;

const PostgresResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  rows: z.array(z.record(z.unknown())).optional(),
  rowCount: z.number().optional(),
  affectedRows: z.number().optional(),
  schema: z.record(z.unknown()).optional(),
  tables: z.array(z.string()).optional(),
  columns: z.array(z.object({
    name: z.string(),
    type: z.string(),
    nullable: z.boolean(),
  })).optional(),
  backupPath: z.string().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type PostgresResult = z.output<typeof PostgresResultSchema>;

export class PostgreSQLBubbleExtended extends PostgreSQLBubble {
  private params: PostgresParams;
  private context?: BubbleContext;

  constructor(params: PostgresParamsInput, context?: BubbleContext) {
    // Initialize base PostgreSQLBubble with basic query params
    super({
      query: params.query || 'SELECT 1',
      params: params.params,
      connectionPool: {
        max: params.poolSize,
        idleTimeoutMillis: 30000,
      },
    }, context);

    this.params = PostgresParamsSchema.parse(params);
    this.context = context;
  }

  public async schemaInfo(): Promise<PostgresResult> {
    const startTime = Date.now();

    try {
      const query = `
        SELECT
          table_name,
          column_name,
          data_type,
          is_nullable,
          column_default
        FROM information_schema.columns
        WHERE table_schema = $1
        ORDER BY table_name, ordinal_position
      `;

      const result = await this.query(query, [this.params.schemaName]);
      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'schema_info',
        rows: result.rows,
        tables: [...new Set(result.rows?.map((r: any) => r.table_name))],
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'schema_info',
        error: errorMessage,
        timing,
      };
    }
  }

  public async tableInfo(): Promise<PostgresResult> {
    if (!this.params.tableName) {
      throw new Error('tableName is required for table_info operation');
    }

    const startTime = Date.now();

    try {
      const query = `
        SELECT
          column_name,
          data_type,
          is_nullable,
          column_default,
          character_maximum_length,
          numeric_precision,
          numeric_scale
        FROM information_schema.columns
        WHERE table_schema = $1
          AND table_name = $2
        ORDER BY ordinal_position
      `;

      const result = await this.query(query, [this.params.schemaName, this.params.tableName]);
      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'table_info',
        columns: result.rows?.map((row: any) => ({
          name: row.column_name,
          type: row.data_type,
          nullable: row.is_nullable === 'YES',
          maxLength: row.character_maximum_length,
          precision: row.numeric_precision,
          scale: row.numeric_scale,
          default: row.column_default,
        })),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'table_info',
        error: errorMessage,
        timing,
      };
    }
  }

  public async batchExecute(): Promise<PostgresResult> {
    if (!this.params.queries) {
      throw new Error('queries array is required for batch_execute operation');
    }

    const startTime = Date.now();
    const results: any[] = [];

    try {
      for (const queryDef of this.params.queries) {
        const result = await this.query(queryDef.query, queryDef.params);
        results.push({
          query: queryDef.query,
          rowCount: result.rowCount,
          rows: result.rows,
        });
      }

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'batch_execute',
        rows: results,
        affectedRows: results.reduce((sum, r) => sum + (r.rowCount || 0), 0),
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'batch_execute',
        error: errorMessage,
        timing,
      };
    }
  }

  public async transaction(): Promise<PostgresResult> {
    if (!this.params.transactionQueries) {
      throw new Error('transactionQueries array is required for transaction operation');
    }

    const startTime = Date.now();

    try {
      // Start transaction
      await this.query('BEGIN', []);

      const results: any[] = [];

      try {
        for (const queryDef of this.params.transactionQueries) {
          const result = await this.query(queryDef.query, queryDef.params);
          results.push({
            query: queryDef.query,
            rowCount: result.rowCount,
            rows: result.rows,
          });
        }

        // Commit transaction
        await this.query('COMMIT', []);

        const timing = Date.now() - startTime;

        return {
          success: true,
          operation: 'transaction',
          rows: results,
          affectedRows: results.reduce((sum, r) => sum + (r.rowCount || 0), 0),
          timing,
        };
      } catch (error) {
        // Rollback on error
        await this.query('ROLLBACK', []);
        throw error;
      }
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'transaction',
        error: errorMessage,
        timing,
      };
    }
  }

  public async healthCheck(): Promise<PostgresResult> {
    const startTime = Date.now();

    try {
      const result = await this.query('SELECT 1 as health_check', []);
      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'health_check',
        rows: result.rows,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'health_check',
        error: errorMessage,
        timing,
      };
    }
  }

  public async backup(): Promise<PostgresResult> {
    if (!this.params.databaseName || !this.params.backupPath) {
      throw new Error('databaseName and backupPath are required for backup operation');
    }

    // This would use pg_dump through a subprocess or external service
    const startTime = Date.now();

    try {
      // Mock implementation - in production would call pg_dump
      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'backup',
        backupPath: this.params.backupPath,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'backup',
        error: errorMessage,
        timing,
      };
    }
  }

  public async restore(): Promise<PostgresResult> {
    if (!this.params.databaseName || !this.params.restorePath) {
      throw new Error('databaseName and restorePath are required for restore operation');
    }

    // This would use psql or pg_restore through a subprocess or external service
    const startTime = Date.now();

    try {
      // Mock implementation - in production would call pg_restore
      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'restore',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'restore',
        error: errorMessage,
        timing,
      };
    }
  }

  public async actionExtended(): Promise<PostgresResult> {
    switch (this.params.operation) {
      case 'query':
        const queryResult = await this.action();
        return {
          success: queryResult.success,
          operation: 'query',
          rows: queryResult.data?.rows,
          rowCount: queryResult.data?.rowCount,
          error: queryResult.error,
          timing: 0,
        };
      case 'batch_execute':
        return this.batchExecute();
      case 'transaction':
        return this.transaction();
      case 'health_check':
        return this.healthCheck();
      case 'schema_info':
        return this.schemaInfo();
      case 'table_info':
        return this.tableInfo();
      case 'backup':
        return this.backup();
      case 'restore':
        return this.restore();
      default:
        return {
          success: false,
          operation: this.params.operation,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default PostgreSQLBubbleExtended;
