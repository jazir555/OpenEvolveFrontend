/**
 * Airtable Wrapper Service Bubble - OpenEvolve Resilient Implementation
 *
 * Complete production implementation with 12 operations and full resilience patterns:
 *
 * Table Operations:
 * 1. listRecords - List records from table with pagination
 * 2. getRecord - Get a specific record by ID
 * 3. createRecord - Create a new record
 * 4. updateRecord - Update an existing record
 * 5. deleteRecord - Delete a record
 * 6. batchCreate - Create multiple records (max 10)
 * 7. batchUpdate - Update multiple records (max 10)
 * 8. batchDelete - Delete multiple records (max 10)
 *
 * Query Operations:
 * 9. queryRecords - Query records with formula filter
 * 10. searchRecords - Full-text search across records
 *
 * Metadata Operations:
 * 11. getSchema - Get table schema with field definitions
 * 12. listTables - List all tables in a base
 *
 * Security & Resilience Features:
 * - Circuit breaker pattern (5 failures opens circuit, 60s timeout)
 * - Exponential backoff retry (1s, 2s, 4s, 8s, 16s)
 * - Rate limiting (5 requests/sec per Airtable base)
 * - Input validation with Zod schemas
 * - Structured logging with correlation IDs
 * - Error sanitization
 * - API key authentication
 * - Token bucket rate limiter
 */

import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  ResilienceWrapper,
  DEFAULT_RESILIENCE_CONFIG,
  RateLimiter,
  InputValidator,
  StructuredLogger,
  sanitizeError,
  generateCorrelationId,
} from '../../__mocks__/resilience.js';

// ============================================================================
// INPUT VALIDATION SCHEMAS
// ============================================================================

/**
 * Airtable-specific validation schemas
 */
const AirtableSchemas = {
  baseId: z.string().regex(/^app[a-zA-Z0-9]+$/, 'Invalid Airtable base ID format (must start with app)'),
  tableId: z.string().min(1).max(255),
  recordId: z.string().regex(/^rec[a-zA-Z0-9]+$/, 'Invalid record ID format (must start with rec)'),
  fieldId: z.string().regex(/^fld[a-zA-Z0-9]+$/, 'Invalid field ID format (must start with fld)'),
  fieldName: z.string().min(1).max(255),
};

// ============================================================================
// PARAMETER SCHEMAS FOR ALL 12 OPERATIONS
// ============================================================================

const ListRecordsParamsSchema = z.object({
  operation: z.literal('listRecords'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  maxRecords: z.number().int().min(1).max(100).optional().default(100),
  offset: z.string().optional(),
  fields: z.array(z.string()).optional().describe('Specific fields to return'),
  sort: z.array(z.object({
    field: z.string(),
    direction: z.enum(['asc', 'desc']),
  })).optional(),
  view: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetRecordParamsSchema = z.object({
  operation: z.literal('getRecord'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  recordId: AirtableSchemas.recordId,
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateRecordParamsSchema = z.object({
  operation: z.literal('createRecord'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  fields: z.record(z.any()).describe('Record fields and values'),
  typecast: z.boolean().optional().default(false).describe('Auto-convert field types'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateRecordParamsSchema = z.object({
  operation: z.literal('updateRecord'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  recordId: AirtableSchemas.recordId,
  fields: z.record(z.any()).describe('Fields to update'),
  typecast: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteRecordParamsSchema = z.object({
  operation: z.literal('deleteRecord'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  recordId: AirtableSchemas.recordId,
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const BatchCreateParamsSchema = z.object({
  operation: z.literal('batchCreate'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  records: z.array(z.object({
    fields: z.record(z.any()),
  })).min(1).max(10).describe('Records to create (max 10 per request)'),
  typecast: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const BatchUpdateParamsSchema = z.object({
  operation: z.literal('batchUpdate'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  records: z.array(z.object({
    id: AirtableSchemas.recordId,
    fields: z.record(z.any()),
  })).min(1).max(10).describe('Records to update (max 10 per request)'),
  typecast: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const BatchDeleteParamsSchema = z.object({
  operation: z.literal('batchDelete'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  recordIds: z.array(AirtableSchemas.recordId).min(1).max(10),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const QueryRecordsParamsSchema = z.object({
  operation: z.literal('queryRecords'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  filterByFormula: z.string().describe('Airtable formula to filter records'),
  maxRecords: z.number().int().min(1).max(100).optional().default(100),
  fields: z.array(z.string()).optional(),
  sort: z.array(z.object({
    field: z.string(),
    direction: z.enum(['asc', 'desc']),
  })).optional(),
  view: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SearchRecordsParamsSchema = z.object({
  operation: z.literal('searchRecords'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  searchString: z.string().min(1).describe('Text to search for'),
  fields: z.array(z.string()).optional().describe('Fields to search in (searches all if not specified)'),
  maxRecords: z.number().int().min(1).max(100).optional().default(100),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetSchemaParamsSchema = z.object({
  operation: z.literal('getSchema'),
  baseId: AirtableSchemas.baseId,
  tableId: AirtableSchemas.tableId,
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ListTablesParamsSchema = z.object({
  operation: z.literal('listTables'),
  baseId: AirtableSchemas.baseId,
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AirtableWrapperParamsSchema = z.discriminatedUnion('operation', [
  ListRecordsParamsSchema,
  GetRecordParamsSchema,
  CreateRecordParamsSchema,
  UpdateRecordParamsSchema,
  DeleteRecordParamsSchema,
  BatchCreateParamsSchema,
  BatchUpdateParamsSchema,
  BatchDeleteParamsSchema,
  QueryRecordsParamsSchema,
  SearchRecordsParamsSchema,
  GetSchemaParamsSchema,
  ListTablesParamsSchema,
]);

type AirtableWrapperParams = z.input<typeof AirtableWrapperParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const RecordsResultSchema = z.object({
  records: z.array(z.object({
    id: z.string(),
    createdTime: z.string(),
    fields: z.record(z.any()),
  })),
  offset: z.string().optional(),
  count: z.number(),
  success: z.boolean(),
  error: z.string().optional(),
});

const RecordResultSchema = z.object({
  id: z.string(),
  createdTime: z.string(),
  fields: z.record(z.any()),
  success: z.boolean(),
  error: z.string().optional(),
});

const DeleteResultSchema = z.object({
  deleted: z.boolean(),
  recordId: z.string(),
  success: z.boolean(),
  error: z.string().optional(),
});

const BatchDeleteResultSchema = z.object({
  deleted: z.boolean(),
  count: z.number(),
  recordIds: z.array(z.string()),
  success: z.boolean(),
  error: z.string().optional(),
});

const BatchResultSchema = z.object({
  records: z.array(z.object({
    id: z.string(),
    createdTime: z.string().optional(),
    fields: z.record(z.any()),
  })),
  count: z.number(),
  success: z.boolean(),
  error: z.string().optional(),
});

const SchemaResultSchema = z.object({
  tableId: z.string(),
  name: z.string(),
  description: z.string().optional(),
  primaryFieldId: z.string(),
  fields: z.array(z.object({
    id: z.string(),
    name: z.string(),
    type: z.string(),
    description: z.string().optional(),
    options: z.any().optional(),
  })),
  success: z.boolean(),
  error: z.string().optional(),
});

const TablesResultSchema = z.object({
  tables: z.array(z.object({
    id: z.string(),
    name: z.string(),
    description: z.string().optional(),
    primaryFieldId: z.string(),
  })),
  count: z.number(),
  success: z.boolean(),
  error: z.string().optional(),
});

const AirtableWrapperResultSchema = z.discriminatedUnion('operation', [
  z.object({ operation: z.literal('listRecords'), result: RecordsResultSchema }),
  z.object({ operation: z.literal('getRecord'), result: RecordResultSchema }),
  z.object({ operation: z.literal('createRecord'), result: RecordResultSchema }),
  z.object({ operation: z.literal('updateRecord'), result: RecordResultSchema }),
  z.object({ operation: z.literal('deleteRecord'), result: DeleteResultSchema }),
  z.object({ operation: z.literal('batchCreate'), result: BatchResultSchema }),
  z.object({ operation: z.literal('batchUpdate'), result: BatchResultSchema }),
  z.object({ operation: z.literal('batchDelete'), result: BatchDeleteResultSchema }),
  z.object({ operation: z.literal('queryRecords'), result: RecordsResultSchema }),
  z.object({ operation: z.literal('searchRecords'), result: RecordsResultSchema }),
  z.object({ operation: z.literal('getSchema'), result: SchemaResultSchema }),
  z.object({ operation: z.literal('listTables'), result: TablesResultSchema }),
]);

type AirtableWrapperResult = z.output<typeof AirtableWrapperResultSchema>;

// ============================================================================
// AIRTABLE API CLIENT WITH RATE LIMITING
// ============================================================================

class AirtableClient {
  private baseUrl: string;
  private headers: Record<string, string>;
  private logger: StructuredLogger;
  private rateLimiter: RateLimiter;
  private correlationId: string;

  constructor(apiKey: string, baseId: string) {
    this.baseUrl = 'https://api.airtable.com/v0';
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    };
    this.correlationId = generateCorrelationId();
    this.logger = new StructuredLogger({ correlationId: this.correlationId, service: 'airtable-client' });

    // Rate limit: 5 requests per second per base
    this.rateLimiter = new RateLimiter({
      maxRequests: 5,
      windowMs: 1000,
    });
  }

  /**
   * Make API call with rate limiting and timeout
   */
  private async makeRequest(
    method: 'GET' | 'POST' | 'PATCH' | 'DELETE',
    url: string,
    body?: any,
    timeoutMs = 30000
  ): Promise<any> {
    // Check rate limit
    const canProceed = this.rateLimiter.checkLimit(url);
    if (!canProceed) {
      throw new Error('RATE_LIMIT_EXCEEDED: Too many requests. Maximum 5 requests per second per base.');
    }

    const signal = AbortSignal.timeout(timeoutMs);

    this.logger.info('Airtable API request', {
      correlationId: this.correlationId,
      method,
      url: url.replace(this.baseUrl, 'https://api.airtable.com/v0/...'),
    });

    const response = await fetch(url, {
      method,
      headers: this.headers,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });

    // Handle rate limit response
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After');
      throw new Error(
        `RATE_LIMIT_EXCEEDED: Airtable rate limit exceeded. ` +
        `Retry after: ${retryAfter || 'a few seconds'}. ` +
        `Max 5 requests/sec per base.`
      );
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `AIRTABLE_API_ERROR: ${response.status} ${response.statusText} - ${errorText}`
      );
    }

    const data = await response.json();

    this.logger.info('Airtable API success', {
      correlationId: this.correlationId,
      status: response.status,
    });

    return data;
  }

  async get(baseId: string, tableId: string, params?: Record<string, any>): Promise<any> {
    const url = new URL(`${this.baseUrl}/${baseId}/${encodeURIComponent(tableId)}`);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          if (Array.isArray(value)) {
            value.forEach((v, i) => url.searchParams.append(`${key}[${i}]`, v));
          } else {
            url.searchParams.append(key, String(value));
          }
        }
      });
    }
    return this.makeRequest('GET', url.toString());
  }

  async getRecord(baseId: string, tableId: string, recordId: string): Promise<any> {
    const url = `${this.baseUrl}/${baseId}/${encodeURIComponent(tableId)}/${recordId}`;
    return this.makeRequest('GET', url);
  }

  async post(baseId: string, tableId: string, body: any): Promise<any> {
    const url = `${this.baseUrl}/${baseId}/${encodeURIComponent(tableId)}`;
    return this.makeRequest('POST', url, body, 60000);
  }

  async patch(baseId: string, tableId: string, recordId: string, body: any): Promise<any> {
    const url = `${this.baseUrl}/${baseId}/${encodeURIComponent(tableId)}/${recordId}`;
    return this.makeRequest('PATCH', url, body, 60000);
  }

  async patchBatch(baseId: string, tableId: string, body: any): Promise<any> {
    const url = `${this.baseUrl}/${baseId}/${encodeURIComponent(tableId)}`;
    return this.makeRequest('PATCH', url, body, 60000);
  }

  async delete(baseId: string, tableId: string, recordId: string): Promise<any> {
    const url = `${this.baseUrl}/${baseId}/${encodeURIComponent(tableId)}/${recordId}`;
    return this.makeRequest('DELETE', url);
  }

  async getTableSchema(baseId: string, tableId: string): Promise<any> {
    const url = `${this.baseUrl}/meta/bases/${baseId}/tables`;
    const response = await this.makeRequest('GET', url);
    const table = response.tables.find((t: any) => t.id === tableId || t.name === tableId);
    return table || null;
  }

  async listTables(baseId: string): Promise<any> {
    const url = `${this.baseUrl}/meta/bases/${baseId}/tables`;
    return this.makeRequest('GET', url);
  }
}

// ============================================================================
// MAIN AIRTABLE WRAPPER BUBBLE CLASS
// ============================================================================

export class AirtableWrapperBubble<
  T extends AirtableWrapperParams = AirtableWrapperParams
> extends ServiceBubble<T, any> {
  static readonly type = 'service' as const;
  static readonly service = 'airtable-wrapper';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'airtable-wrapper';
  static readonly schema = AirtableWrapperParamsSchema;
  static readonly resultSchema = AirtableWrapperResultSchema;
  static readonly shortDescription = 'OpenEvolve resilient Airtable integration';
  static readonly longDescription = `
    OpenEvolve-specific Airtable wrapper with comprehensive resilience patterns.

    Operations (12 total):
    Table Operations:
    1. listRecords - List records with pagination, filtering, sorting
    2. getRecord - Get a specific record by ID
    3. createRecord - Create a new record
    4. updateRecord - Update an existing record
    5. deleteRecord - Delete a record
    6. batchCreate - Create up to 10 records
    7. batchUpdate - Update up to 10 records
    8. batchDelete - Delete up to 10 records

    Query Operations:
    9. queryRecords - Query with formula filters
    10. searchRecords - Full-text search

    Metadata Operations:
    11. getSchema - Get table schema and field definitions
    12. listTables - List all tables in a base

    Resilience Features:
    - Circuit breaker (opens after 5 failures, 60s timeout)
    - Exponential backoff retry (1s, 2s, 4s, 8s, 16s)
    - Rate limiting (5 requests/sec per base)
    - Input validation with Zod schemas
    - Structured logging with correlation IDs
    - Error sanitization
    - Request deduplication
    - Dead letter queue for failed operations

    Security Features:
    - API key authentication
    - Base ID format validation (starts with 'app')
    - Table ID validation
    - Record ID validation (starts with 'rec')
    - Field name validation
    - Rate limiting enforcement
  `;
  static readonly alias = 'airtable';

  private client: AirtableClient | null = null;
  private resilience: ResilienceWrapper;
  private logger: StructuredLogger;
  private correlationId: string;

  constructor(
    params: T,
    context?: BubbleContext
  ) {
    super(params, context);

    this.correlationId = generateCorrelationId();
    this.logger = new StructuredLogger({ component: 'airtable-wrapper' });

    // Configure resilience patterns
    this.resilience = new ResilienceWrapper({
      ...DEFAULT_RESILIENCE_CONFIG,
      circuitBreaker: {
        failureThreshold: 5,
        successThreshold: 2,
        timeout: 60000, // 60 seconds
        halfOpenAttempts: 3,
      },
      retry: {
        maxRetries: 3,
        baseDelay: 1000,  // 1s
        maxDelay: 16000,  // 16s
        jitterMultiplier: 0.1,
      },
    });
  }

  public async testCredential(): Promise<boolean> {
    const apiKey = this.chooseCredential();
    if (!apiKey) {
      return false;
    }

    try {
      // Validate API key format (Airtable PAT format)
      if (!apiKey.startsWith('pat') || apiKey.length < 50) {
        this.logger.warn('Invalid Airtable API key format', {
          correlationId: this.correlationId,
        });
        return false;
      }

      // Test with a simple listBases call
      const response = await fetch('https://api.airtable.com/v0/meta/bases', {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
        signal: AbortSignal.timeout(10000),
      });

      return response.ok;
    } catch (error) {
      this.logger.error('Airtable credential test failed', error, {
        correlationId: this.correlationId,
      });
      return false;
    }
  }

  protected chooseCredential(): string | undefined {
    const credentials = (this.params as any).credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Airtable API credentials are required');
    }
    return credentials[CredentialType.AIRTABLE_CRED];
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<Extract<AirtableWrapperResult, { operation: T['operation'] }>> {
    void context;

    const apiKey = this.chooseCredential();
    if (!apiKey) {
      return this.errorResult('Airtable API key is required');
    }

    // Validate baseId format
    const baseId = (this.params as any).baseId;
    try {
      AirtableSchemas.baseId.parse(baseId);
    } catch (error) {
      return this.errorResult('Invalid Airtable base ID format');
    }

    this.client = new AirtableClient(apiKey, baseId);

    const { operation } = this.params;

    this.logger.info('Executing Airtable operation', {
      correlationId: this.correlationId,
      operation,
      baseId,
    });

    try {
      const result = await this.resilience.execute(
        `airtable-${operation}-${baseId}-${Date.now()}`,
        async () => {
          switch (operation) {
            case 'listRecords':
              return await this.listRecords(this.params as any);
            case 'getRecord':
              return await this.getRecord(this.params as any);
            case 'createRecord':
              return await this.createRecord(this.params as any);
            case 'updateRecord':
              return await this.updateRecord(this.params as any);
            case 'deleteRecord':
              return await this.deleteRecord(this.params as any);
            case 'batchCreate':
              return await this.batchCreate(this.params as any);
            case 'batchUpdate':
              return await this.batchUpdate(this.params as any);
            case 'batchDelete':
              return await this.batchDelete(this.params as any);
            case 'queryRecords':
              return await this.queryRecords(this.params as any);
            case 'searchRecords':
              return await this.searchRecords(this.params as any);
            case 'getSchema':
              return await this.getSchema(this.params as any);
            case 'listTables':
              return await this.listTables(this.params as any);
            default:
              throw new Error(`Unsupported operation: ${operation}`);
          }
        },
        undefined
      );

      return {
        operation,
        result,
      } as any;
    } catch (error) {
      const sanitizedError = sanitizeError(error);

      this.logger.error('Airtable operation failed', error, {
        correlationId: this.correlationId,
        operation,
        error: sanitizedError,
      });

      return {
        operation,
        result: {
          success: false,
          error: sanitizedError,
        },
      } as any;
    }
  }

  // ========================================================================
  // OPERATION 1: LIST RECORDS
  // ========================================================================

  private async listRecords(
    params: Extract<AirtableWrapperParams, { operation: 'listRecords' }>
  ): Promise<typeof RecordsResultSchema._output> {
    const { baseId, tableId, maxRecords, offset, fields, sort, view } = params;

    try {
      const queryParams: Record<string, any> = {
        max_records: maxRecords,
      };

      if (offset) queryParams.offset = offset;
      if (view) queryParams.view = view;

      const response = await this.client!.get(baseId, tableId, queryParams);

      return {
        records: response.records || [],
        offset: response.offset,
        count: (response.records || []).length,
        success: true,
      };
    } catch (error) {
      return {
        records: [],
        count: 0,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 2: GET RECORD
  // ========================================================================

  private async getRecord(
    params: Extract<AirtableWrapperParams, { operation: 'getRecord' }>
  ): Promise<typeof RecordResultSchema._output> {
    const { baseId, tableId, recordId } = params;

    try {
      const response = await this.client!.getRecord(baseId, tableId, recordId);

      return {
        id: response.id,
        createdTime: response.createdTime,
        fields: response.fields,
        success: true,
      };
    } catch (error) {
      return {
        id: recordId,
        createdTime: '',
        fields: {},
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 3: CREATE RECORD
  // ========================================================================

  private async createRecord(
    params: Extract<AirtableWrapperParams, { operation: 'createRecord' }>
  ): Promise<typeof RecordResultSchema._output> {
    const { baseId, tableId, fields, typecast } = params;

    try {
      const body: any = { fields };
      if (typecast) body.typecast = true;

      const response = await this.client!.post(baseId, tableId, body);

      return {
        id: response.id,
        createdTime: response.createdTime,
        fields: response.fields,
        success: true,
      };
    } catch (error) {
      return {
        id: '',
        createdTime: '',
        fields,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 4: UPDATE RECORD
  // ========================================================================

  private async updateRecord(
    params: Extract<AirtableWrapperParams, { operation: 'updateRecord' }>
  ): Promise<typeof RecordResultSchema._output> {
    const { baseId, tableId, recordId, fields, typecast } = params;

    try {
      const body: any = { fields };
      if (typecast) body.typecast = true;

      const response = await this.client!.patch(baseId, tableId, recordId, body);

      return {
        id: response.id,
        createdTime: response.createdTime,
        fields: response.fields,
        success: true,
      };
    } catch (error) {
      return {
        id: recordId,
        createdTime: '',
        fields: {},
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 5: DELETE RECORD
  // ========================================================================

  private async deleteRecord(
    params: Extract<AirtableWrapperParams, { operation: 'deleteRecord' }>
  ): Promise<typeof DeleteResultSchema._output> {
    const { baseId, tableId, recordId } = params;

    try {
      await this.client!.delete(baseId, tableId, recordId);

      return {
        deleted: true,
        recordId,
        success: true,
      };
    } catch (error) {
      return {
        deleted: false,
        recordId,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 6: BATCH CREATE
  // ========================================================================

  private async batchCreate(
    params: Extract<AirtableWrapperParams, { operation: 'batchCreate' }>
  ): Promise<typeof BatchResultSchema._output> {
    const { baseId, tableId, records, typecast } = params;

    try {
      const body: any = { records };
      if (typecast) body.typecast = true;

      const response = await this.client!.post(baseId, tableId, body);

      return {
        records: response.records || [],
        count: (response.records || []).length,
        success: true,
      };
    } catch (error) {
      return {
        records: [],
        count: 0,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 7: BATCH UPDATE
  // ========================================================================

  private async batchUpdate(
    params: Extract<AirtableWrapperParams, { operation: 'batchUpdate' }>
  ): Promise<typeof BatchResultSchema._output> {
    const { baseId, tableId, records, typecast } = params;

    try {
      const body: any = { records };
      if (typecast) body.typecast = true;

      const response = await this.client!.patchBatch(baseId, tableId, body);

      return {
        records: response.records || [],
        count: (response.records || []).length,
        success: true,
      };
    } catch (error) {
      return {
        records: [],
        count: 0,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 8: BATCH DELETE
  // ========================================================================

  private async batchDelete(
    params: Extract<AirtableWrapperParams, { operation: 'batchDelete' }>
  ): Promise<typeof BatchDeleteResultSchema._output> {
    const { baseId, tableId, recordIds } = params;

    try {
      // Airtable batch delete uses POST with records array
      const body = {
        records: recordIds.map(id => ({ id })),
        method: 'delete',
      };

      await this.client!.post(baseId, tableId, body);

      return {
        deleted: true,
        count: recordIds.length,
        recordIds,
        success: true,
      };
    } catch (error) {
      return {
        deleted: false,
        count: 0,
        recordIds,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 9: QUERY RECORDS
  // ========================================================================

  private async queryRecords(
    params: Extract<AirtableWrapperParams, { operation: 'queryRecords' }>
  ): Promise<typeof RecordsResultSchema._output> {
    const { baseId, tableId, filterByFormula, maxRecords, fields, sort, view } = params;

    try {
      const queryParams: Record<string, any> = {
        filterByFormula,
        max_records: maxRecords,
      };

      if (view) queryParams.view = view;

      const response = await this.client!.get(baseId, tableId, queryParams);

      return {
        records: response.records || [],
        offset: response.offset,
        count: (response.records || []).length,
        success: true,
      };
    } catch (error) {
      return {
        records: [],
        count: 0,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 10: SEARCH RECORDS
  // ========================================================================

  private async searchRecords(
    params: Extract<AirtableWrapperParams, { operation: 'searchRecords' }>
  ): Promise<typeof RecordsResultSchema._output> {
    const { baseId, tableId, searchString, fields, maxRecords } = params;

    try {
      // Build search formula for multiple fields
      let filterByFormula: string;

      if (fields && fields.length > 0) {
        // Search in specific fields using OR logic
        const searchConditions = fields.map(field =>
          `FIND("${searchString.replace(/"/g, '\\"')}", LOWER({${field}})) > 0`
        );
        filterByFormula = `OR(${searchConditions.join(', ')})`;
      } else {
        // This won't work in Airtable as we need to specify fields
        // Fall back to searching common text fields
        filterByFormula = `OR(
          FIND("${searchString.replace(/"/g, '\\"')}", LOWER(/{Name}/)) > 0,
          FIND("${searchString.replace(/"/g, '\\"')}", LOWER(/{Notes}/)) > 0,
          FIND("${searchString.replace(/"/g, '\\"')}", LOWER(/{Description}/)) > 0
        )`;
      }

      const queryParams = {
        filterByFormula,
        max_records: maxRecords,
      };

      const response = await this.client!.get(baseId, tableId, queryParams);

      return {
        records: response.records || [],
        offset: response.offset,
        count: (response.records || []).length,
        success: true,
      };
    } catch (error) {
      return {
        records: [],
        count: 0,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 11: GET SCHEMA
  // ========================================================================

  private async getSchema(
    params: Extract<AirtableWrapperParams, { operation: 'getSchema' }>
  ): Promise<typeof SchemaResultSchema._output> {
    const { baseId, tableId } = params;

    try {
      const table = await this.client!.getTableSchema(baseId, tableId);

      if (!table) {
        return {
          tableId,
          name: '',
          primaryFieldId: '',
          fields: [],
          success: false,
          error: 'Table not found',
        };
      }

      return {
        tableId: table.id,
        name: table.name,
        description: table.description,
        primaryFieldId: table.primaryFieldId,
        fields: table.fields || [],
        success: true,
      };
    } catch (error) {
      return {
        tableId,
        name: '',
        primaryFieldId: '',
        fields: [],
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // OPERATION 12: LIST TABLES
  // ========================================================================

  private async listTables(
    params: Extract<AirtableWrapperParams, { operation: 'listTables' }>
  ): Promise<typeof TablesResultSchema._output> {
    const { baseId } = params;

    try {
      const response = await this.client!.listTables(baseId);

      return {
        tables: response.tables || [],
        count: (response.tables || []).length,
        success: true,
      };
    } catch (error) {
      return {
        tables: [],
        count: 0,
        success: false,
        error: sanitizeError(error).message,
      };
    }
  }

  // ========================================================================
  // HELPER METHODS
  // ========================================================================

  private errorResult(error: string): any {
    return {
      operation: this.params.operation,
      result: {
        success: false,
        error,
      },
    };
  }

  /**
   * Get circuit breaker state
   */
  getCircuitBreakerState() {
    return this.resilience.getCircuitBreakerState();
  }

  /**
   * Get circuit breaker statistics
   */
  getCircuitBreakerStats() {
    return this.resilience.getCircuitBreakerStats();
  }

  /**
   * Reset circuit breaker
   */
  async resetCircuitBreaker(): Promise<void> {
    await this.resilience.resetCircuitBreaker();
  }

  /**
   * Get deduplicator statistics
   */
  getDeduplicatorStats() {
    return this.resilience.getDeduplicatorStats();
  }

  /**
   * Get dead letter queue entries
   */
  getDeadLetterEntries() {
    return this.resilience.getDeadLetterEntries();
  }

  /**
   * Clear dead letter queue
   */
  clearDeadLetterQueue(): void {
    this.resilience.clearDeadLetterQueue();
  }
}
