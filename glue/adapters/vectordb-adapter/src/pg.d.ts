declare module 'pg' {
  export interface PoolConfig {
    connectionString?: string;
    max?: number;
    idleTimeoutMillis?: number;
    connectionTimeoutMillis?: number;
    [key: string]: unknown;
  }

  export interface QueryResult {
    rows: any[];
    rowCount: number | null;
    [key: string]: unknown;
  }

  export class Pool {
    constructor(config?: PoolConfig);
    query(text: string, params?: any[]): Promise<QueryResult>;
    end(): Promise<void>;
  }
}
