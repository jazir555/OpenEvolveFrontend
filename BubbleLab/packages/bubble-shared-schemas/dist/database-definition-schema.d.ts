/**
 * Database Definition Schema
 *
 * This schema is designed to store database table definitions and metadata
 */
import { z } from '@hono/zod-openapi';
export interface DatabaseConnection {
    id: string;
    name: string;
    type: 'postgresql' | 'mysql' | 'mongodb' | 'bigquery' | 'sqlite';
    host: string;
    port: number;
    database: string;
    username?: string;
    status: 'connected' | 'disconnected' | 'error';
    createdAt: string;
    lastUsed: string;
    description?: string;
}
export type DatabaseStatus = 'connected' | 'disconnected' | 'error';
export type DatabaseType = 'postgresql' | 'mysql' | 'mongodb' | 'bigquery' | 'sqlite';
export interface DatabaseColumn {
    name: string;
    type: string;
    isNullable: boolean;
    defaultValue?: string;
    constraints?: string[];
}
export interface DatabaseTable {
    name: string;
    schema: string;
    columns: DatabaseColumn[];
    rowCount?: number;
    size?: string;
}
export interface DatabaseSchema {
    tables: DatabaseTable[];
    totalTables: number;
    totalSize?: string;
}
export declare const databaseMetadataSchema: z.ZodObject<{
    tables: z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>;
    tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    databaseName: z.ZodOptional<z.ZodString>;
    databaseType: z.ZodOptional<z.ZodEnum<["postgresql", "mysql", "sqlite", "mssql", "oracle"]>>;
    rules: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        text: z.ZodString;
        enabled: z.ZodBoolean;
        createdAt: z.ZodString;
        updatedAt: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        id: string;
        text: string;
        enabled: boolean;
        createdAt: string;
        updatedAt: string;
    }, {
        id: string;
        text: string;
        enabled: boolean;
        createdAt: string;
        updatedAt: string;
    }>, "many">>;
    notes: z.ZodOptional<z.ZodString>;
    tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    tables: Record<string, Record<string, string>>;
    tableNotes?: Record<string, string> | undefined;
    databaseName?: string | undefined;
    databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
    rules?: {
        id: string;
        text: string;
        enabled: boolean;
        createdAt: string;
        updatedAt: string;
    }[] | undefined;
    notes?: string | undefined;
    tags?: string[] | undefined;
}, {
    tables: Record<string, Record<string, string>>;
    tableNotes?: Record<string, string> | undefined;
    databaseName?: string | undefined;
    databaseType?: "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle" | undefined;
    rules?: {
        id: string;
        text: string;
        enabled: boolean;
        createdAt: string;
        updatedAt: string;
    }[] | undefined;
    notes?: string | undefined;
    tags?: string[] | undefined;
}>;
export type DatabaseMetadata = z.infer<typeof databaseMetadataSchema>;
//# sourceMappingURL=database-definition-schema.d.ts.map