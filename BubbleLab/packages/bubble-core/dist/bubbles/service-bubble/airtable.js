import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// Airtable API base URL
const AIRTABLE_API_BASE = 'https://api.airtable.com/v0';
// Define Airtable field value schema (supports multiple types)
const AirtableFieldValueSchema = z
    .union([
    z.string(),
    z.number(),
    z.boolean(),
    z.array(z.unknown()),
    z.record(z.unknown()),
    z.null(),
])
    .describe('Value for an Airtable field (string, number, boolean, array, object, or null)');
// Define Airtable record schema
const AirtableRecordSchema = z
    .object({
    id: z.string().describe('Unique record identifier (starts with rec)'),
    createdTime: z
        .string()
        .datetime()
        .describe('ISO 8601 datetime when record was created'),
    fields: z
        .record(z.string(), AirtableFieldValueSchema)
        .describe('Record field values as key-value pairs'),
})
    .describe('Airtable record with ID, creation time, and field data');
// Define sort direction
const SortDirectionSchema = z
    .enum(['asc', 'desc'])
    .describe('Sort direction: ascending or descending');
// Define sort specification
const SortSpecSchema = z
    .object({
    field: z.string().describe('Field name to sort by'),
    direction: SortDirectionSchema.optional()
        .default('asc')
        .describe('Sort direction (asc or desc)'),
})
    .describe('Sort specification for ordering records');
const AirtableFieldTypeEnum = z.enum([
    'singleLineText',
    'multilineText',
    'richText',
    'email',
    'url',
    'phoneNumber',
    'number',
    'percent',
    'currency',
    'rating',
    'duration',
    'singleSelect',
    'multipleSelects',
    'singleCollaborator',
    'multipleCollaborators',
    'date',
    'dateTime',
    'checkbox',
    'multipleRecordLinks',
    'multipleAttachments',
    'barcode',
    'button',
    'formula',
    'createdTime',
    'lastModifiedTime',
    'createdBy',
    'lastModifiedBy',
    'autoNumber',
    'externalSyncSource',
    'count',
    'lookup',
    'rollup',
]);
// Define the parameters schema for different Airtable operations
const AirtableParamsSchema = z.discriminatedUnion('operation', [
    // List records operation
    z.object({
        operation: z
            .literal('list_records')
            .describe('List records from an Airtable table with filtering and sorting'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        fields: z
            .array(z.string())
            .optional()
            .describe('Array of field names to include in results (returns all fields if not specified)'),
        filterByFormula: z
            .string()
            .optional()
            .describe('Airtable formula to filter records (e.g., "{Status} = \'Done\'")'),
        maxRecords: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .describe('Maximum number of records to return (1-100, returns all if not specified)'),
        pageSize: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(100)
            .describe('Number of records per page for pagination (1-100)'),
        sort: z
            .array(SortSpecSchema)
            .optional()
            .describe('Array of sort specifications to order records'),
        view: z
            .string()
            .optional()
            .describe("View name or ID to use (includes view's filters and sorts)"),
        cellFormat: z
            .enum(['json', 'string'])
            .optional()
            .default('json')
            .describe('Format for cell values: json (structured) or string (formatted)'),
        timeZone: z
            .string()
            .optional()
            .describe('Time zone for date/time fields (e.g., "America/Los_Angeles")'),
        userLocale: z
            .string()
            .optional()
            .describe('Locale for formatting (e.g., "en-US")'),
        offset: z
            .string()
            .optional()
            .describe('Pagination offset from previous response'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get record operation
    z.object({
        operation: z
            .literal('get_record')
            .describe('Retrieve a single record by its ID'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        recordId: z
            .string()
            .min(1, 'Record ID is required')
            .describe('Record ID to retrieve (starts with rec)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Create records operation
    z.object({
        operation: z
            .literal('create_records')
            .describe('Create one or more new records in an Airtable table'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        records: z
            .array(z
            .object({
            fields: z
                .record(z.string(), AirtableFieldValueSchema)
                .describe('Field values for the new record'),
        })
            .describe('Record data to create'))
            .min(1, 'At least one record is required')
            .max(10, 'Maximum 10 records can be created at once')
            .describe('Array of records to create (max 10 per request)'),
        typecast: z
            .boolean()
            .optional()
            .default(false)
            .describe('Automatically convert field values to the appropriate type'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Update records operation
    z.object({
        operation: z
            .literal('update_records')
            .describe('Update existing records in an Airtable table'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        records: z
            .array(z
            .object({
            id: z
                .string()
                .min(1, 'Record ID is required')
                .describe('Record ID to update (starts with rec)'),
            fields: z
                .record(z.string(), AirtableFieldValueSchema)
                .describe('Field values to update (only specified fields will be updated)'),
        })
            .describe('Record data to update'))
            .min(1, 'At least one record is required')
            .max(10, 'Maximum 10 records can be updated at once')
            .describe('Array of records to update (max 10 per request)'),
        typecast: z
            .boolean()
            .optional()
            .default(false)
            .describe('Automatically convert field values to the appropriate type'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Delete records operation
    z.object({
        operation: z
            .literal('delete_records')
            .describe('Delete one or more records from an Airtable table'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        recordIds: z
            .array(z.string())
            .min(1, 'At least one record ID is required')
            .max(10, 'Maximum 10 records can be deleted at once')
            .describe('Array of record IDs to delete (max 10 per request)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // List bases operation
    z.object({
        operation: z
            .literal('list_bases')
            .describe('List all bases accessible with the current API key'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get base schema operation
    z.object({
        operation: z
            .literal('get_base_schema')
            .describe('Get the schema for a specific base including all tables and fields'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Create table operation
    z.object({
        operation: z
            .literal('create_table')
            .describe('Create a new table in an Airtable base'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        name: z
            .string()
            .min(1, 'Table name is required')
            .describe('Name for the new table'),
        description: z
            .string()
            .optional()
            .describe('Optional description for the table'),
        fields: z
            .array(z.object({
            name: z.string().describe('Field name'),
            type: AirtableFieldTypeEnum.describe('Field type'),
            description: z.string().optional().describe('Field description'),
            options: z.record(z.unknown()).optional().describe('Field options'),
        }))
            .describe('Array of field definitions for the table'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Update table operation
    z.object({
        operation: z
            .literal('update_table')
            .describe('Update table properties like name and description'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        name: z.string().optional().describe('New name for the table'),
        description: z
            .string()
            .optional()
            .describe('New description for the table'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Create field operation
    z.object({
        operation: z
            .literal('create_field')
            .describe('Create a new field in an Airtable table'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        name: z
            .string()
            .min(1, 'Field name is required')
            .describe('Name for the new field'),
        type: AirtableFieldTypeEnum.describe('Field type'),
        description: z.string().optional().describe('Field description'),
        options: z
            .record(z.unknown())
            .optional()
            .describe('Field-specific options'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Update field operation
    z.object({
        operation: z
            .literal('update_field')
            .describe('Update field properties like name, type, or description'),
        baseId: z
            .string()
            .min(1, 'Base ID is required')
            .describe('Airtable base ID (e.g., appXXXXXXXXXXXXXX)'),
        tableIdOrName: z
            .string()
            .min(1, 'Table ID or name is required')
            .describe('Table ID (e.g., tblXXXXXXXXXXXXXX) or table name'),
        fieldIdOrName: z
            .string()
            .min(1, 'Field ID or name is required')
            .describe('Field ID (e.g., fldXXXXXXXXXXXXXX) or field name'),
        name: z.string().optional().describe('New name for the field'),
        description: z
            .string()
            .optional()
            .describe('New description for the field'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
]);
// Define result schemas for different operations
const AirtableResultSchema = z.discriminatedUnion('operation', [
    z.object({
        operation: z
            .literal('list_records')
            .describe('List records from an Airtable table with filtering and sorting'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        records: z
            .array(AirtableRecordSchema)
            .optional()
            .describe('Array of record objects'),
        offset: z
            .string()
            .optional()
            .describe('Pagination offset for retrieving next page of results'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('get_record')
            .describe('Retrieve a single record by its ID'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        record: AirtableRecordSchema.optional().describe('Record object'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('create_records')
            .describe('Create one or more new records in an Airtable table'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        records: z
            .array(AirtableRecordSchema)
            .optional()
            .describe('Array of created record objects'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('update_records')
            .describe('Update existing records in an Airtable table'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        records: z
            .array(AirtableRecordSchema)
            .optional()
            .describe('Array of updated record objects'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('delete_records')
            .describe('Delete one or more records from an Airtable table'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        records: z
            .array(z
            .object({
            id: z.string().describe('ID of deleted record'),
            deleted: z.boolean().describe('Whether the record was deleted'),
        })
            .describe('Deletion confirmation object'))
            .optional()
            .describe('Array of deletion confirmation objects'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    // List bases result
    z.object({
        operation: z
            .literal('list_bases')
            .describe('List all bases accessible with the current API key'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        bases: z
            .array(z.object({
            id: z.string().describe('Base ID'),
            name: z.string().describe('Base name'),
            permissionLevel: z
                .string()
                .describe('Permission level for this base'),
        }))
            .optional()
            .describe('Array of base objects'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    // Get base schema result
    z.object({
        operation: z
            .literal('get_base_schema')
            .describe('Get the schema for a specific base including all tables and fields'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        tables: z
            .array(z.object({
            id: z.string().describe('Table ID'),
            name: z.string().describe('Table name'),
            description: z.string().optional().describe('Table description'),
            primaryFieldId: z.string().describe('ID of the primary field'),
            fields: z
                .array(z.object({
                id: z.string().describe('Field ID'),
                name: z.string().describe('Field name'),
                type: z.string().describe('Field type'),
                description: z
                    .string()
                    .optional()
                    .describe('Field description'),
                options: z
                    .record(z.unknown())
                    .optional()
                    .describe('Field options'),
            }))
                .describe('Array of field definitions'),
            views: z
                .array(z.object({
                id: z.string().describe('View ID'),
                name: z.string().describe('View name'),
                type: z.string().describe('View type'),
            }))
                .optional()
                .describe('Array of view definitions'),
        }))
            .optional()
            .describe('Array of table schemas'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    // Create table result
    z.object({
        operation: z
            .literal('create_table')
            .describe('Create a new table in an Airtable base'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        table: z
            .object({
            id: z.string().describe('Table ID'),
            name: z.string().describe('Table name'),
            description: z.string().optional().describe('Table description'),
            primaryFieldId: z.string().describe('ID of the primary field'),
            fields: z
                .array(z.object({
                id: z.string().describe('Field ID'),
                name: z.string().describe('Field name'),
                type: AirtableFieldTypeEnum.describe('Field type'),
            }))
                .describe('Array of field definitions'),
        })
            .optional()
            .describe('Created table object'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    // Update table result
    z.object({
        operation: z
            .literal('update_table')
            .describe('Update table properties like name and description'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        table: z
            .object({
            id: z.string().describe('Table ID'),
            name: z.string().describe('Table name'),
            description: z.string().optional().describe('Table description'),
        })
            .optional()
            .describe('Updated table object'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    // Create field result
    z.object({
        operation: z
            .literal('create_field')
            .describe('Create a new field in an Airtable table'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        field: z
            .object({
            id: z.string().describe('Field ID'),
            name: z.string().describe('Field name'),
            type: z.string().describe('Field type'),
            description: z.string().optional().describe('Field description'),
        })
            .optional()
            .describe('Created field object'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    // Update field result
    z.object({
        operation: z
            .literal('update_field')
            .describe('Update field properties like name, type, or description'),
        ok: z.boolean().describe('Whether the Airtable API call was successful'),
        field: z
            .object({
            id: z.string().describe('Field ID'),
            name: z.string().describe('Field name'),
            type: z.string().describe('Field type'),
            description: z.string().optional().describe('Field description'),
        })
            .optional()
            .describe('Updated field object'),
        error: z.string().default('').describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
]);
export class AirtableBubble extends ServiceBubble {
    async testCredential() {
        // Test credential by checking the Authorization header format
        // Note: We cannot test actual API access without knowing which base/table the user wants to access
        // and what scopes their PAT has. Airtable PATs can have varying scopes and base restrictions.
        //
        // The best we can do is verify the token format is valid.
        // Actual access will be validated when the user makes their first API call.
        try {
            const credential = this.chooseCredential();
            if (!credential) {
                return false;
            }
            // Verify the token format looks like an Airtable PAT
            // Format: patXXXXXXXXXX.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            if (!credential.startsWith('pat') || credential.length < 50) {
                return false;
            }
            return true;
        }
        catch {
            return false;
        }
    }
    static type = 'service';
    static service = 'airtable';
    static authType = 'apikey';
    static bubbleName = 'airtable';
    static schema = AirtableParamsSchema;
    static resultSchema = AirtableResultSchema;
    static shortDescription = 'Airtable integration for managing records in bases and tables';
    static longDescription = `
    Comprehensive Airtable integration bubble for managing bases, tables, fields, and records.
    Use cases:
    - List records with filtering, sorting, and pagination
    - Retrieve individual records by ID
    - Create, update, and delete records
    - List all accessible bases
    - Get base schema with all tables and fields
    - Create and update tables
    - Create and update fields
    - Support for all Airtable field types (text, number, attachments, links, etc.)
    
    Security Features:
    - Personal Access Token authentication
    - Parameter validation and sanitization
    - Rate limiting awareness (5 requests per second per base)
    - Comprehensive error handling
  `;
    static alias = 'airtable';
    constructor(params = {
        operation: 'list_records',
        baseId: '',
        tableIdOrName: '',
    }, context, instanceId) {
        super(params, context, instanceId);
    }
    async performAction(context) {
        void context;
        const { operation } = this.params;
        try {
            const result = await (async () => {
                switch (operation) {
                    case 'list_records':
                        return await this.listRecords(this.params);
                    case 'get_record':
                        return await this.getRecord(this.params);
                    case 'create_records':
                        return await this.createRecords(this.params);
                    case 'update_records':
                        return await this.updateRecords(this.params);
                    case 'delete_records':
                        return await this.deleteRecords(this.params);
                    case 'list_bases':
                        return await this.listBases(this.params);
                    case 'get_base_schema':
                        return await this.getBaseSchema(this.params);
                    case 'create_table':
                        return await this.createTable(this.params);
                    case 'update_table':
                        return await this.updateTable(this.params);
                    case 'create_field':
                        return await this.createField(this.params);
                    case 'update_field':
                        return await this.updateField(this.params);
                    default:
                        throw new Error(`Unsupported operation: ${operation}`);
                }
            })();
            return result;
        }
        catch (error) {
            const failedOperation = this.params.operation;
            return {
                success: false,
                ok: false,
                operation: failedOperation,
                error: error instanceof Error
                    ? error.message
                    : 'Unknown error occurred in AirtableBubble',
            };
        }
    }
    async listRecords(params) {
        // Parse params to apply defaults
        const parsed = AirtableParamsSchema.parse(params);
        const { baseId, tableIdOrName, fields, filterByFormula, maxRecords, pageSize, sort, view, cellFormat, timeZone, userLocale, offset, } = parsed;
        const queryParams = new URLSearchParams();
        if (fields && fields.length > 0) {
            fields.forEach((field) => queryParams.append('fields[]', field));
        }
        if (filterByFormula)
            queryParams.append('filterByFormula', filterByFormula);
        if (maxRecords)
            queryParams.append('maxRecords', maxRecords.toString());
        if (pageSize)
            queryParams.append('pageSize', pageSize.toString());
        if (sort && sort.length > 0) {
            sort.forEach((s, index) => {
                queryParams.append(`sort[${index}][field]`, s.field);
                queryParams.append(`sort[${index}][direction]`, s.direction);
            });
        }
        if (view)
            queryParams.append('view', view);
        if (cellFormat)
            queryParams.append('cellFormat', cellFormat);
        if (timeZone)
            queryParams.append('timeZone', timeZone);
        if (userLocale)
            queryParams.append('userLocale', userLocale);
        if (offset)
            queryParams.append('offset', offset);
        const response = await this.makeAirtableApiCall(`${baseId}/${encodeURIComponent(tableIdOrName)}?${queryParams.toString()}`, 'GET');
        if ('error' in response) {
            return {
                operation: 'list_records',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'list_records',
            ok: true,
            records: response.records
                ? z.array(AirtableRecordSchema).parse(response.records)
                : undefined,
            offset: response.offset,
            error: '',
            success: true,
        };
    }
    async getRecord(params) {
        const { baseId, tableIdOrName, recordId } = params;
        const response = await this.makeAirtableApiCall(`${baseId}/${encodeURIComponent(tableIdOrName)}/${recordId}`, 'GET');
        if ('error' in response) {
            return {
                operation: 'get_record',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'get_record',
            ok: true,
            record: AirtableRecordSchema.parse(response),
            error: '',
            success: true,
        };
    }
    async createRecords(params) {
        // Parse params to apply defaults
        const parsed = AirtableParamsSchema.parse(params);
        const { baseId, tableIdOrName, records, typecast } = parsed;
        const body = {
            records,
            typecast,
        };
        const response = await this.makeAirtableApiCall(`${baseId}/${encodeURIComponent(tableIdOrName)}`, 'POST', body);
        if ('error' in response) {
            return {
                operation: 'create_records',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'create_records',
            ok: true,
            records: response.records
                ? z.array(AirtableRecordSchema).parse(response.records)
                : undefined,
            error: '',
            success: true,
        };
    }
    async updateRecords(params) {
        // Parse params to apply defaults
        const parsed = AirtableParamsSchema.parse(params);
        const { baseId, tableIdOrName, records, typecast } = parsed;
        const body = {
            records,
            typecast,
        };
        const response = await this.makeAirtableApiCall(`${baseId}/${encodeURIComponent(tableIdOrName)}`, 'PATCH', body);
        if ('error' in response) {
            return {
                operation: 'update_records',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'update_records',
            ok: true,
            records: response.records
                ? z.array(AirtableRecordSchema).parse(response.records)
                : undefined,
            error: '',
            success: true,
        };
    }
    async deleteRecords(params) {
        const { baseId, tableIdOrName, recordIds } = params;
        // Airtable expects record IDs as query parameters for DELETE
        const queryParams = new URLSearchParams();
        recordIds.forEach((id) => queryParams.append('records[]', id));
        const response = await this.makeAirtableApiCall(`${baseId}/${encodeURIComponent(tableIdOrName)}?${queryParams.toString()}`, 'DELETE');
        if ('error' in response) {
            return {
                operation: 'delete_records',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        // For delete, response.records has a different structure
        const deleteRecords = response.records;
        return {
            operation: 'delete_records',
            ok: true,
            records: deleteRecords,
            error: '',
            success: true,
        };
    }
    async listBases(params) {
        void params;
        const response = await this.makeAirtableApiCall('meta/bases', 'GET');
        if ('error' in response) {
            return {
                operation: 'list_bases',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'list_bases',
            ok: true,
            bases: response.bases,
            error: '',
            success: true,
        };
    }
    async getBaseSchema(params) {
        const { baseId } = params;
        const response = await this.makeAirtableApiCall(`meta/bases/${baseId}/tables`, 'GET');
        if ('error' in response) {
            return {
                operation: 'get_base_schema',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'get_base_schema',
            ok: true,
            tables: response.tables,
            error: '',
            success: true,
        };
    }
    /**
     * Normalizes field definitions by adding required default options for field types that need them.
     * This provides a better UX by auto-fixing common configuration issues.
     */
    normalizeFieldOptions(fields) {
        const typeRequiresOptions = new Set([
            'multipleRecordLinks',
            'lookup',
            'rollup',
            'count',
            'button',
            'externalSyncSource',
            'formula',
        ]);
        return fields.map((field) => {
            const normalizedField = { ...field };
            // Add default options for field types that require them
            switch (field.type) {
                case 'date':
                    // Date fields require a dateFormat option
                    if (!field.options || Object.keys(field.options).length === 0) {
                        normalizedField.options = {
                            dateFormat: {
                                name: 'local',
                                format: 'l',
                            },
                        };
                    }
                    break;
                case 'dateTime':
                    // DateTime fields require dateFormat, timeFormat, and timeZone
                    if (!field.options || Object.keys(field.options).length === 0) {
                        normalizedField.options = {
                            dateFormat: {
                                name: 'local',
                                format: 'l',
                            },
                            timeFormat: {
                                name: '12hour',
                                format: 'h:mma',
                            },
                            timeZone: 'utc',
                        };
                    }
                    break;
                case 'number':
                    // Number fields should have precision
                    if (!field.options || !('precision' in field.options)) {
                        normalizedField.options = {
                            ...field.options,
                            precision: 0,
                        };
                    }
                    break;
                case 'currency':
                    // Currency fields need precision and symbol
                    if (!field.options || !('precision' in field.options)) {
                        normalizedField.options = {
                            ...field.options,
                            precision: 2,
                            symbol: '$',
                        };
                    }
                    break;
                case 'percent':
                    // Percent fields need precision
                    if (!field.options || !('precision' in field.options)) {
                        normalizedField.options = {
                            ...field.options,
                            precision: 0,
                        };
                    }
                    break;
                case 'rating':
                    // Rating fields need max, icon, and color
                    if (!field.options || !('max' in field.options)) {
                        normalizedField.options = {
                            ...field.options,
                            max: 5,
                            icon: 'star',
                            color: 'yellowBright',
                        };
                    }
                    break;
                case 'duration':
                    // Duration fields need durationFormat
                    if (!field.options || !('durationFormat' in field.options)) {
                        normalizedField.options = {
                            ...field.options,
                            durationFormat: 'h:mm',
                        };
                    }
                    break;
                case 'checkbox':
                    // Checkbox fields support icon/color options; provide sensible defaults
                    if (!field.options || Object.keys(field.options).length === 0) {
                        normalizedField.options = {
                            icon: 'check',
                            color: 'greenBright',
                        };
                    }
                    break;
                case 'singleSelect':
                case 'multipleSelects':
                    // Select fields require an options object with choices.
                    // Provide an empty choices array if none are supplied so Airtable's schema validation passes.
                    if (!field.options ||
                        typeof field.options !== 'object' ||
                        !('choices' in field.options)) {
                        normalizedField.options = {
                            ...field.options,
                            choices: [],
                        };
                    }
                    break;
                // singleSelect and multipleSelects MUST have choices - don't auto-fix these
                // as we can't guess what the choices should be. Let it fail with Airtable's error.
            }
            if (typeRequiresOptions.has(field.type) &&
                (!normalizedField.options ||
                    (typeof normalizedField.options === 'object' &&
                        Object.keys(normalizedField.options).length === 0))) {
                throw new Error(`Airtable field "${field.name}" of type "${field.type}" requires an options object`);
            }
            return normalizedField;
        });
    }
    async createTable(params) {
        const { baseId, name, description, fields } = params;
        // Normalize field options to add sensible defaults where needed
        const normalizedFields = this.normalizeFieldOptions(fields);
        const body = {
            name,
            fields: normalizedFields,
        };
        if (description) {
            body.description = description;
        }
        const response = await this.makeAirtableApiCall(`meta/bases/${baseId}/tables`, 'POST', body);
        console.log('response', response);
        if ('error' in response) {
            return {
                operation: 'create_table',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'create_table',
            ok: true,
            table: response,
            error: '',
            success: true,
        };
    }
    async updateTable(params) {
        const { baseId, tableIdOrName, name, description } = params;
        const body = {};
        if (name) {
            body.name = name;
        }
        if (description !== undefined) {
            body.description = description;
        }
        const response = await this.makeAirtableApiCall(`meta/bases/${baseId}/tables/${encodeURIComponent(tableIdOrName)}`, 'PATCH', body);
        if ('error' in response) {
            return {
                operation: 'update_table',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'update_table',
            ok: true,
            table: response,
            error: '',
            success: true,
        };
    }
    async createField(params) {
        const { baseId, tableIdOrName, name, type, description, options } = params;
        // Normalize the field to add default options if needed
        const normalizedField = this.normalizeFieldOptions([
            { name, type, description, options },
        ])[0];
        const body = {
            name: normalizedField.name,
            type: normalizedField.type,
        };
        if (normalizedField.description) {
            body.description = normalizedField.description;
        }
        if (normalizedField.options) {
            body.options = normalizedField.options;
        }
        const response = await this.makeAirtableApiCall(`meta/bases/${baseId}/tables/${encodeURIComponent(tableIdOrName)}/fields`, 'POST', body);
        if ('error' in response) {
            return {
                operation: 'create_field',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'create_field',
            ok: true,
            field: response,
            error: '',
            success: true,
        };
    }
    async updateField(params) {
        const { baseId, tableIdOrName, fieldIdOrName, name, description } = params;
        const body = {};
        if (name) {
            body.name = name;
        }
        if (description !== undefined) {
            body.description = description;
        }
        const response = await this.makeAirtableApiCall(`meta/bases/${baseId}/tables/${encodeURIComponent(tableIdOrName)}/fields/${encodeURIComponent(fieldIdOrName)}`, 'PATCH', body);
        if ('error' in response) {
            return {
                operation: 'update_field',
                ok: false,
                error: this.formatAirtableError(response),
                success: false,
            };
        }
        return {
            operation: 'update_field',
            ok: true,
            field: response,
            error: '',
            success: true,
        };
    }
    formatAirtableError(errorResponse) {
        const topLevelMessage = 'message' in errorResponse ? errorResponse.message : undefined;
        const { error } = errorResponse;
        if (typeof error === 'string') {
            return topLevelMessage ?? error;
        }
        if (error && typeof error === 'object' && !Array.isArray(error)) {
            const { message, type } = error;
            return topLevelMessage ?? message ?? type ?? JSON.stringify(error);
        }
        return topLevelMessage ?? 'Unknown Airtable API error';
    }
    chooseCredential() {
        const { credentials } = this.params;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('No Airtable credentials provided');
        }
        return credentials[CredentialType.AIRTABLE_CRED];
    }
    async makeAirtableApiCall(endpoint, method = 'GET', body) {
        const url = `${AIRTABLE_API_BASE}/${endpoint}`;
        const authToken = this.chooseCredential();
        if (!authToken) {
            throw new Error('Airtable authentication token is required but was not provided');
        }
        const headers = {
            Authorization: `Bearer ${authToken}`,
            'Content-Type': 'application/json',
        };
        const fetchConfig = {
            method,
            headers,
        };
        if (body && (method === 'POST' || method === 'PATCH')) {
            fetchConfig.body = JSON.stringify(body);
        }
        const response = await fetch(url, fetchConfig);
        const data = (await response.json());
        if (!response.ok) {
            return data;
        }
        return data;
    }
}
//# sourceMappingURL=airtable.js.map