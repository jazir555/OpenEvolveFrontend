import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG, } from '../../__mocks__/resilience.js';
/**
 * Airtable Bubble - Complete Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. listRecords - List records from a table with pagination
 * 2. getRecord - Get a specific record by ID
 * 3. createRecord - Create a new record
 * 4. updateRecord - Update an existing record
 * 5. deleteRecord - Delete a record
 * 6. batchCreate - Create multiple records
 * 7. batchUpdate - Update multiple records
 * 8. batchDelete - Delete multiple records
 * 9. queryRecords - Query records with formula filter
 * 10. getTable - Get table schema and information
 */
// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================
const ListRecordsParamsSchema = z.object({
    operation: z.literal('listRecords'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    maxRecords: z.number().int().positive().optional().default(100),
    offset: z.string().optional(),
    fields: z.array(z.string()).optional().describe('Fields to return'),
    sort: z.array(z.object({
        field: z.string(),
        direction: z.enum(['asc', 'desc']),
    })).optional(),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetRecordParamsSchema = z.object({
    operation: z.literal('getRecord'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    recordId: z.string().min(1, 'Record ID is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const CreateRecordParamsSchema = z.object({
    operation: z.literal('createRecord'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    fields: z.record(z.any()).describe('Record fields and values'),
    typecast: z.boolean().optional().default(false).describe('Automatically convert field types'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const UpdateRecordParamsSchema = z.object({
    operation: z.literal('updateRecord'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    recordId: z.string().min(1, 'Record ID is required'),
    fields: z.record(z.any()).describe('Fields to update'),
    typecast: z.boolean().optional().default(false),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const DeleteRecordParamsSchema = z.object({
    operation: z.literal('deleteRecord'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    recordId: z.string().min(1, 'Record ID is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const BatchCreateParamsSchema = z.object({
    operation: z.literal('batchCreate'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    records: z.array(z.object({
        fields: z.record(z.any()),
    })).min(1).max(10).describe('Records to create (max 10)'),
    typecast: z.boolean().optional().default(false),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const BatchUpdateParamsSchema = z.object({
    operation: z.literal('batchUpdate'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    records: z.array(z.object({
        id: z.string(),
        fields: z.record(z.any()),
    })).min(1).max(10).describe('Records to update (max 10)'),
    typecast: z.boolean().optional().default(false),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const BatchDeleteParamsSchema = z.object({
    operation: z.literal('batchDelete'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    recordIds: z.array(z.string()).min(1).max(10).describe('Record IDs to delete (max 10)'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const QueryRecordsParamsSchema = z.object({
    operation: z.literal('queryRecords'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    filterByFormula: z.string().describe('Airtable formula to filter records'),
    maxRecords: z.number().int().positive().optional().default(100),
    fields: z.array(z.string()).optional(),
    sort: z.array(z.object({
        field: z.string(),
        direction: z.enum(['asc', 'desc']),
    })).optional(),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetTableParamsSchema = z.object({
    operation: z.literal('getTable'),
    baseId: z.string().min(1, 'Base ID is required'),
    tableId: z.string().min(1, 'Table ID or name is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const AirtableBubbleParamsSchema = z.discriminatedUnion('operation', [
    ListRecordsParamsSchema,
    GetRecordParamsSchema,
    CreateRecordParamsSchema,
    UpdateRecordParamsSchema,
    DeleteRecordParamsSchema,
    BatchCreateParamsSchema,
    BatchUpdateParamsSchema,
    BatchDeleteParamsSchema,
    QueryRecordsParamsSchema,
    GetTableParamsSchema,
]);
// ============================================================================
// RESULT SCHEMAS
// ============================================================================
const RecordsSchema = z.object({
    records: z.array(z.object({
        id: z.string(),
        createdTime: z.string(),
        fields: z.record(z.any()),
    })),
    offset: z.string().optional(),
    count: z.number(),
    success: z.boolean(),
    error: z.string(),
});
const RecordSchema = z.object({
    id: z.string(),
    createdTime: z.string(),
    fields: z.record(z.any()),
    success: z.boolean(),
    error: z.string(),
});
const BatchResultSchema = z.object({
    records: z.array(z.object({
        id: z.string(),
        createdTime: z.string().optional(),
        fields: z.record(z.any()),
    })),
    count: z.number(),
    success: z.boolean(),
    error: z.string(),
});
const DeleteResultSchema = z.object({
    deleted: z.boolean(),
    recordId: z.string(),
    success: z.boolean(),
    error: z.string(),
});
const BatchDeleteResultSchema = z.object({
    deleted: z.boolean(),
    count: z.number(),
    recordIds: z.array(z.string()),
    success: z.boolean(),
    error: z.string(),
});
const TableInfoSchema = z.object({
    tableId: z.string(),
    name: z.string(),
    description: z.string().optional(),
    fields: z.array(z.object({
        id: z.string(),
        name: z.string(),
        type: z.string(),
        description: z.string().optional(),
        options: z.any().optional(),
    })),
    success: z.boolean(),
    error: z.string(),
});
const AirtableBubbleResultSchema = z.discriminatedUnion('operation', [
    z.object({
        operation: z.literal('listRecords'),
        result: RecordsSchema,
    }),
    z.object({
        operation: z.literal('getRecord'),
        result: RecordSchema,
    }),
    z.object({
        operation: z.literal('createRecord'),
        result: RecordSchema,
    }),
    z.object({
        operation: z.literal('updateRecord'),
        result: RecordSchema,
    }),
    z.object({
        operation: z.literal('deleteRecord'),
        result: DeleteResultSchema,
    }),
    z.object({
        operation: z.literal('batchCreate'),
        result: BatchResultSchema,
    }),
    z.object({
        operation: z.literal('batchUpdate'),
        result: BatchResultSchema,
    }),
    z.object({
        operation: z.literal('batchDelete'),
        result: BatchDeleteResultSchema,
    }),
    z.object({
        operation: z.literal('queryRecords'),
        result: RecordsSchema,
    }),
    z.object({
        operation: z.literal('getTable'),
        result: TableInfoSchema,
    }),
]);
// ============================================================================
// AIRTABLE API CLIENT
// ============================================================================
class AirtableClient {
    baseUrl;
    headers;
    constructor(apiKey) {
        this.baseUrl = 'https://api.airtable.com/v0';
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
        };
    }
    async get(baseId, endpoint, params) {
        const url = new URL(`${this.baseUrl}/${baseId}/${endpoint}`);
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) {
                    url.searchParams.append(key, value);
                }
            });
        }
        const response = await fetch(url.toString(), {
            method: 'GET',
            headers: this.headers,
            signal: AbortSignal.timeout(30000),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Airtable API error: ${response.status} - ${error}`);
        }
        return response.json();
    }
    async post(baseId, endpoint, body) {
        const url = `${this.baseUrl}/${baseId}/${endpoint}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(60000),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Airtable API error: ${response.status} - ${error}`);
        }
        return response.json();
    }
    async patch(baseId, endpoint, body) {
        const url = `${this.baseUrl}/${baseId}/${endpoint}`;
        const response = await fetch(url, {
            method: 'PATCH',
            headers: this.headers,
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(60000),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Airtable API error: ${response.status} - ${error}`);
        }
        return response.json();
    }
    async delete(baseId, endpoint) {
        const url = `${this.baseUrl}/${baseId}/${endpoint}`;
        const response = await fetch(url, {
            method: 'DELETE',
            headers: this.headers,
            signal: AbortSignal.timeout(60000),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Airtable API error: ${response.status} - ${error}`);
        }
        return response.json();
    }
}
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class AirtableBubble extends ServiceBubble {
    static type = 'service';
    static service = 'airtable';
    static authType = 'apikey';
    static bubbleName = 'airtable';
    static schema = AirtableBubbleParamsSchema;
    static resultSchema = AirtableBubbleResultSchema;
    static shortDescription = 'Complete Airtable integration for database operations';
    static longDescription = `
    Comprehensive Airtable service bubble for all database operations.

    Operations:
    1. listRecords - List records with pagination and sorting
    2. getRecord - Get a specific record by ID
    3. createRecord - Create a new record
    4. updateRecord - Update an existing record
    5. deleteRecord - Delete a record
    6. batchCreate - Create multiple records (up to 10)
    7. batchUpdate - Update multiple records (up to 10)
    8. batchDelete - Delete multiple records (up to 10)
    9. queryRecords - Query with formula filters
    10. getTable - Get table schema and field definitions

    Features:
    - Full CRUD operations
    - Batch operations for efficiency
    - Formula-based querying
    - Field type conversion
    - Pagination support
    - Sorting capabilities
    - Resilience patterns
  `;
    static alias = 'airtable';
    client = null;
    resilience;
    constructor(params, context) {
        super(params, context);
        this.resilience = new ResilienceWrapper(DEFAULT_RESILIENCE_CONFIG);
    }
    async testCredential() {
        const apiKey = this.chooseCredential();
        if (!apiKey) {
            return false;
        }
        try {
            // Test with a simple base info call
            const client = new AirtableClient(apiKey);
            await client.get('', 'meta/bases');
            return true;
        }
        catch {
            return false;
        }
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('Airtable API credentials are required');
        }
        return credentials[CredentialType.AIRTABLE_CRED];
    }
    async performAction(context) {
        void context;
        const apiKey = this.chooseCredential();
        if (!apiKey) {
            return this.errorResult('Airtable API key is required');
        }
        this.client = new AirtableClient(apiKey);
        const { operation } = this.params;
        try {
            const result = await this.resilience.execute(`airtable-${operation}-${Date.now()}`, async () => {
                switch (operation) {
                    case 'listRecords':
                        return await this.listRecords(this.params);
                    case 'getRecord':
                        return await this.getRecord(this.params);
                    case 'createRecord':
                        return await this.createRecord(this.params);
                    case 'updateRecord':
                        return await this.updateRecord(this.params);
                    case 'deleteRecord':
                        return await this.deleteRecord(this.params);
                    case 'batchCreate':
                        return await this.batchCreate(this.params);
                    case 'batchUpdate':
                        return await this.batchUpdate(this.params);
                    case 'batchDelete':
                        return await this.batchDelete(this.params);
                    case 'queryRecords':
                        return await this.queryRecords(this.params);
                    case 'getTable':
                        return await this.getTable(this.params);
                    default:
                        throw new Error(`Unsupported operation: ${operation}`);
                }
            });
            return {
                operation,
                result,
            };
        }
        catch (error) {
            return {
                operation,
                result: {
                    success: false,
                    error: error instanceof Error ? error.message : 'Unknown error',
                },
            };
        }
    }
    // ========================================================================
    // OPERATION 1: LIST RECORDS
    // ========================================================================
    async listRecords(params) {
        const { baseId, tableId, maxRecords, offset, fields, sort } = params;
        try {
            const queryParams = {
                max_records: maxRecords.toString(),
            };
            if (offset) {
                queryParams.offset = offset;
            }
            if (fields && fields.length > 0) {
                queryParams.fields = fields.join(',');
            }
            if (sort && sort.length > 0) {
                sort.forEach((s, index) => {
                    queryParams[`sort[${index}][field]`] = s.field;
                    queryParams[`sort[${index}][direction]`] = s.direction;
                });
            }
            const response = await this.client.get(baseId, encodeURIComponent(tableId), queryParams);
            return {
                records: response.records || [],
                offset: response.offset,
                count: (response.records || []).length,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                records: [],
                count: 0,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to list records',
            };
        }
    }
    // ========================================================================
    // OPERATION 2: GET RECORD
    // ========================================================================
    async getRecord(params) {
        const { baseId, tableId, recordId } = params;
        try {
            const response = await this.client.get(baseId, `${encodeURIComponent(tableId)}/${recordId}`);
            return {
                id: response.id,
                createdTime: response.createdTime,
                fields: response.fields,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                id: recordId,
                createdTime: '',
                fields: {},
                success: false,
                error: error instanceof Error ? error.message : 'Failed to get record',
            };
        }
    }
    // ========================================================================
    // OPERATION 3: CREATE RECORD
    // ========================================================================
    async createRecord(params) {
        const { baseId, tableId, fields, typecast } = params;
        try {
            const body = {
                fields,
            };
            if (typecast) {
                body.typecast = true;
            }
            const response = await this.client.post(baseId, encodeURIComponent(tableId), body);
            return {
                id: response.id,
                createdTime: response.createdTime,
                fields: response.fields,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                id: '',
                createdTime: '',
                fields,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to create record',
            };
        }
    }
    // ========================================================================
    // OPERATION 4: UPDATE RECORD
    // ========================================================================
    async updateRecord(params) {
        const { baseId, tableId, recordId, fields, typecast } = params;
        try {
            const body = {
                fields,
            };
            if (typecast) {
                body.typecast = true;
            }
            const response = await this.client.patch(baseId, `${encodeURIComponent(tableId)}/${recordId}`, body);
            return {
                id: response.id,
                createdTime: response.createdTime,
                fields: response.fields,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                id: recordId,
                createdTime: '',
                fields: {},
                success: false,
                error: error instanceof Error ? error.message : 'Failed to update record',
            };
        }
    }
    // ========================================================================
    // OPERATION 5: DELETE RECORD
    // ========================================================================
    async deleteRecord(params) {
        const { baseId, tableId, recordId } = params;
        try {
            await this.client.delete(baseId, `${encodeURIComponent(tableId)}/${recordId}`);
            return {
                deleted: true,
                recordId,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                deleted: false,
                recordId,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to delete record',
            };
        }
    }
    // ========================================================================
    // OPERATION 6: BATCH CREATE
    // ========================================================================
    async batchCreate(params) {
        const { baseId, tableId, records, typecast } = params;
        try {
            const body = {
                records,
            };
            if (typecast) {
                body.typecast = true;
            }
            const response = await this.client.post(baseId, encodeURIComponent(tableId), body);
            return {
                records: response.records || [],
                count: (response.records || []).length,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                records: [],
                count: 0,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to batch create records',
            };
        }
    }
    // ========================================================================
    // OPERATION 7: BATCH UPDATE
    // ========================================================================
    async batchUpdate(params) {
        const { baseId, tableId, records, typecast } = params;
        try {
            const body = {
                records,
            };
            if (typecast) {
                body.typecast = true;
            }
            const response = await this.client.patch(baseId, encodeURIComponent(tableId), body);
            return {
                records: response.records || [],
                count: (response.records || []).length,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                records: [],
                count: 0,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to batch update records',
            };
        }
    }
    // ========================================================================
    // OPERATION 8: BATCH DELETE
    // ========================================================================
    async batchDelete(params) {
        const { baseId, tableId, recordIds } = params;
        try {
            // Airtable requires deleting records one at a time or using batch endpoint
            const records = recordIds.map(id => ({ id }));
            await this.client.post(baseId, `${encodeURIComponent(tableId)}`, {
                records,
                method: 'delete',
            });
            return {
                deleted: true,
                count: recordIds.length,
                recordIds,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                deleted: false,
                count: 0,
                recordIds,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to batch delete records',
            };
        }
    }
    // ========================================================================
    // OPERATION 9: QUERY RECORDS
    // ========================================================================
    async queryRecords(params) {
        const { baseId, tableId, filterByFormula, maxRecords, fields, sort } = params;
        try {
            const queryParams = {
                filterByFormula,
                max_records: maxRecords.toString(),
            };
            if (fields && fields.length > 0) {
                queryParams.fields = fields.join(',');
            }
            if (sort && sort.length > 0) {
                sort.forEach((s, index) => {
                    queryParams[`sort[${index}][field]`] = s.field;
                    queryParams[`sort[${index}][direction]`] = s.direction;
                });
            }
            const response = await this.client.get(baseId, encodeURIComponent(tableId), queryParams);
            return {
                records: response.records || [],
                offset: response.offset,
                count: (response.records || []).length,
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                records: [],
                count: 0,
                success: false,
                error: error instanceof Error ? error.message : 'Failed to query records',
            };
        }
    }
    // ========================================================================
    // OPERATION 10: GET TABLE
    // ========================================================================
    async getTable(params) {
        const { baseId, tableId } = params;
        try {
            const response = await this.client.get(baseId, `meta/tables`);
            const table = response.tables.find((t) => t.id === tableId || t.name === tableId);
            if (!table) {
                return {
                    tableId,
                    name: '',
                    fields: [],
                    success: false,
                    error: 'Table not found',
                };
            }
            return {
                tableId: table.id,
                name: table.name,
                description: table.description,
                fields: table.fields || [],
                success: true,
                error: '',
            };
        }
        catch (error) {
            return {
                tableId,
                name: '',
                fields: [],
                success: false,
                error: error instanceof Error ? error.message : 'Failed to get table info',
            };
        }
    }
    // ========================================================================
    // HELPER METHODS
    // ========================================================================
    errorResult(error) {
        return {
            operation: this.params.operation,
            result: {
                success: false,
                error,
            },
        };
    }
}
//# sourceMappingURL=airtable-bubble.js.map