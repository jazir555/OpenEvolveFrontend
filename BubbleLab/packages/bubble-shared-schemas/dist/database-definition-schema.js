/**
 * Database Definition Schema
 *
 * This schema is designed to store database table definitions and metadata
 */
import { z } from '@hono/zod-openapi';
// Schema for database metadata that can be stored in credentials
export const databaseMetadataSchema = z.object({
    // Core database definition - mapping of table names to column definitions
    // Format: { [tableName]: { [columnName]: columnType } }
    tables: z.record(z.string(), // table name
    z.record(z.string(), // column name
    z.string() // notes about it
    )),
    // Table-level notes - mapping of table names to notes about the entire table
    tableNotes: z.record(z.string(), z.string()).optional(),
    // Optional metadata
    databaseName: z.string().optional(),
    databaseType: z
        .enum(['postgresql', 'mysql', 'sqlite', 'mssql', 'oracle'])
        .optional(),
    // Rules and constraints - simplified to match frontend
    rules: z
        .array(z.object({
        id: z.string(),
        text: z.string(),
        enabled: z.boolean(),
        createdAt: z.string(), // ISO string
        updatedAt: z.string(), // ISO string
    }))
        .optional(),
    // Additional context
    notes: z.string().optional(),
    tags: z.array(z.string()).optional(),
});
//# sourceMappingURL=database-definition-schema.js.map