import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * PostgreSQLBubble - PostgreSQL database operations
 */
export class PostgreSQLBubble extends ServiceBubble<PostgreSQLParams, PostgreSQLResult> {
  bubbleName = 'postgresql';
  type = 'service';
  alias = 'PostgreSQL';
  credentialType = 'postgresql_api_key';

  params = {
    connectionString: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private pool: any = null;

  async connect() {
    const { Pool } = await import('pg');
    this.pool = new Pool({
      connectionString: this.params.connectionString,
      connectionTimeoutMillis: this.params.timeout
    });
  }

  async query(params: { sql: string; values?: any[] }): Promise<PostgreSQLResult> {
    try {
      const result = await this.pool.query(params.sql, params.values || []);
      return { success: true, rows: result.rows, rowCount: result.rowCount };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async execute(params: { sql: string; values?: any[] }): Promise<PostgreSQLResult> {
    try {
      const result = await this.pool.query(params.sql, params.values || []);
      return { success: true, rowCount: result.rowCount };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async transaction(params: { queries: Array<{ sql: string; values?: any[] }> }): Promise<PostgreSQLResult> {
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      const results = [];
      for (const q of params.queries) {
        const result = await client.query(q.sql, q.values || []);
        results.push(result.rows);
      }
      await client.query('COMMIT');
      return { success: true, results };
    } catch (error: any) {
      await client.query('ROLLBACK');
      return { success: false, error: error.message };
    } finally {
      client.release();
    }
  }

  async batchExecute(params: { queries: Array<{ sql: string; values?: any[] }> }): Promise<PostgreSQLResult> {
    try {
      const results = [];
      for (const q of params.queries) {
        const result = await this.pool.query(q.sql, q.values || []);
        results.push({ rows: result.rows, rowCount: result.rowCount });
      }
      return { success: true, results };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async schemaInfo(params: { schemaName?: string }): Promise<PostgreSQLResult> {
    try {
      const schema = params.schemaName || 'public';
      const result = await this.pool.query(`
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = $1
        ORDER BY table_name, ordinal_position
      `, [schema]);
      return { success: true, schema: result.rows };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async tableInfo(params: { tableName: string }): Promise<PostgreSQLResult> {
    try {
      const result = await this.pool.query(`
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
      `, [params.tableName]);
      return { success: true, columns: result.rows };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getTableList(params: { schemaName?: string }): Promise<PostgreSQLResult> {
    try {
      const schema = params.schemaName || 'public';
      const result = await this.pool.query(`
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = $1 AND table_type = 'BASE TABLE'
        ORDER BY table_name
      `, [schema]);
      return { success: true, tables: result.rows.map((r: any) => r.table_name) };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getColumnList(params: { tableName: string }): Promise<PostgreSQLResult> {
    try {
      const result = await this.pool.query(`
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
      `, [params.tableName]);
      return { success: true, columns: result.rows };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface PostgreSQLParams {
  connectionString: string;
  timeout?: number;
}

export interface PostgreSQLResult {
  success: boolean;
  rows?: any[];
  rowCount?: number;
  results?: any[];
  schema?: any[];
  columns?: any[];
  tables?: string[];
  error?: string;
}
