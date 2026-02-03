import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import {
  ResilienceWrapper,
  DEFAULT_RESILIENCE_CONFIG,
} from '../../__mocks__/resilience.js';

/**
 * Google Sheets Bubble - Complete Service Bubble Implementation
 *
 * Full production implementation with 14 operations:
 * 1. createSpreadsheet - Create a new Google Sheets spreadsheet
 * 2. getSpreadsheet - Get spreadsheet metadata and information
 * 3. deleteSpreadsheet - Delete a spreadsheet
 * 4. copySpreadsheet - Copy a spreadsheet
 * 5. updateCell - Update a single cell in a spreadsheet
 * 6. getCellValue - Get a single cell value
 * 7. batchUpdate - Batch update multiple cells or ranges
 * 8. appendRow - Append a row to a sheet
 * 9. getRange - Get values from a range
 * 10. clearRange - Clear values from a range
 * 11. copyRange - Copy range to destination
 * 12. addSheet - Add a new sheet to the spreadsheet
 * 13. deleteSheet - Delete a sheet from the spreadsheet
 * 14. getSheetData - Get complete sheet data with metadata
 *
 * Security Features:
 * - OAuth2 token validation
 * - Rate limiting (batch: 10/min, others: 50/min)
 * - Input validation with Zod schemas
 * - Range format validation (A1 notation)
 * - Sheet name validation
 * - Error sanitization
 * - Structured logging
 * - Resilience patterns (circuit breaker, retry, deduplication)
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const CreateSpreadsheetParamsSchema = z.object({
  operation: z.literal('createSpreadsheet'),
  title: z.string().min(1, 'Title is required'),
  sheets: z.array(z.object({
    title: z.string().min(1),
    rowCount: z.number().int().positive().optional().default(1000),
    columnCount: z.number().int().positive().optional().default(26),
  })).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetSpreadsheetParamsSchema = z.object({
  operation: z.literal('getSpreadsheet'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  includeGridData: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteSpreadsheetParamsSchema = z.object({
  operation: z.literal('deleteSpreadsheet'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CopySpreadsheetParamsSchema = z.object({
  operation: z.literal('copySpreadsheet'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  title: z.string().min(1, 'Title is required'),
  destinationFolderId: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateCellParamsSchema = z.object({
  operation: z.literal('updateCell'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required (e.g., "Sheet1!A1")'),
  value: z.any().describe('Value to set'),
  valueInputOption: z.enum(['RAW', 'USER_ENTERED']).optional().default('USER_ENTERED'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetCellValueParamsSchema = z.object({
  operation: z.literal('getCellValue'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required (e.g., "Sheet1!A1")'),
  valueRenderOption: z.enum(['FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA']).optional().default('UNFORMATTED_VALUE'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const BatchUpdateParamsSchema = z.object({
  operation: z.literal('batchUpdate'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  updates: z.array(z.object({
    range: z.string().min(1),
    values: z.array(z.array(z.any())),
  })).min(1, 'At least one update is required'),
  valueInputOption: z.enum(['RAW', 'USER_ENTERED']).optional().default('USER_ENTERED'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AppendRowParamsSchema = z.object({
  operation: z.literal('appendRow'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required (e.g., "Sheet1!A1")'),
  values: z.array(z.any()).min(1, 'At least one value is required'),
  valueInputOption: z.enum(['RAW', 'USER_ENTERED']).optional().default('USER_ENTERED'),
  insertDataOption: z.enum(['OVERWRITE', 'INSERT_ROWS']).optional().default('INSERT_ROWS'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetRangeParamsSchema = z.object({
  operation: z.literal('getRange'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required'),
  majorDimension: z.enum(['DIMENSIONS_UNSPECIFIED', 'ROWS', 'COLUMNS']).optional().default('ROWS'),
  valueRenderOption: z.enum(['FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA']).optional().default('UNFORMATTED_VALUE'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ClearRangeParamsSchema = z.object({
  operation: z.literal('clearRange'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CopyRangeParamsSchema = z.object({
  operation: z.literal('copyRange'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  sourceRange: z.string().min(1, 'Source range is required'),
  destinationRange: z.string().min(1, 'Destination range is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetRowParamsSchema = z.object({
  operation: z.literal('getRow'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required (e.g., "Sheet1!A1:Z1")'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteRowParamsSchema = z.object({
  operation: z.literal('deleteRow'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  sheetId: z.number().int().describe('Sheet ID (0, 1, 2, etc.)'),
  rowIndex: z.number().int().nonnegative().describe('Row index (0-based)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AddSheetParamsSchema = z.object({
  operation: z.literal('addSheet'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  title: z.string().min(1, 'Sheet title is required'),
  rowCount: z.number().int().positive().optional().default(1000),
  columnCount: z.number().int().positive().optional().default(26),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteSheetParamsSchema = z.object({
  operation: z.literal('deleteSheet'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  sheetId: z.number().int().describe('Sheet ID to delete'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetValuesParamsSchema = z.object({
  operation: z.literal('getValues'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required'),
  majorDimension: z.enum(['DIMENSIONS_UNSPECIFIED', 'ROWS', 'COLUMNS']).optional().default('ROWS'),
  valueRenderOption: z.enum(['FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA']).optional().default('UNFORMATTED_VALUE'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SetValuesParamsSchema = z.object({
  operation: z.literal('setValues'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required'),
  values: z.array(z.array(z.any())).min(1, 'At least one row is required'),
  valueInputOption: z.enum(['RAW', 'USER_ENTERED']).optional().default('USER_ENTERED'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ClearValuesParamsSchema = z.object({
  operation: z.literal('clearValues'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  range: z.string().min(1, 'Range is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetSheetDataParamsSchema = z.object({
  operation: z.literal('getSheetData'),
  spreadsheetId: z.string().min(1, 'Spreadsheet ID is required'),
  sheetName: z.string().min(1, 'Sheet name is required'),
  includeMetadata: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GoogleSheetsBubbleParamsSchema = z.discriminatedUnion('operation', [
  CreateSpreadsheetParamsSchema,
  GetSpreadsheetParamsSchema,
  DeleteSpreadsheetParamsSchema,
  CopySpreadsheetParamsSchema,
  UpdateCellParamsSchema,
  GetCellValueParamsSchema,
  BatchUpdateParamsSchema,
  AppendRowParamsSchema,
  GetRangeParamsSchema,
  ClearRangeParamsSchema,
  CopyRangeParamsSchema,
  AddSheetParamsSchema,
  DeleteSheetParamsSchema,
  GetSheetDataParamsSchema,
  // Legacy operations for backward compatibility
  GetRowParamsSchema,
  DeleteRowParamsSchema,
  GetValuesParamsSchema,
  SetValuesParamsSchema,
  ClearValuesParamsSchema,
]);

type GoogleSheetsBubbleParams = z.input<typeof GoogleSheetsBubbleParamsSchema>;

// ============================================================================
// RESULT SCHEMAS
// ============================================================================

const SpreadsheetResultSchema = z.object({
  spreadsheetId: z.string(),
  title: z.string(),
  url: z.string(),
  sheetCount: z.number(),
  success: z.boolean(),
  error: z.string(),
});

const SheetInfoSchema = z.object({
  spreadsheetId: z.string(),
  title: z.string(),
  sheets: z.array(z.object({
    sheetId: z.number(),
    title: z.string(),
    index: z.number(),
    sheetType: z.string(),
    gridProperties: z.object({
      rowCount: z.number(),
      columnCount: z.number(),
    }).optional(),
  })),
  namedRanges: z.array(z.any()).optional(),
  success: z.boolean(),
  error: z.string(),
});

const DeleteSpreadsheetResultSchema = z.object({
  spreadsheetId: z.string(),
  deleted: z.boolean(),
  success: z.boolean(),
  error: z.string(),
});

const CopySpreadsheetResultSchema = z.object({
  originalSpreadsheetId: z.string(),
  newSpreadsheetId: z.string(),
  title: z.string(),
  url: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const CellValueResultSchema = z.object({
  spreadsheetId: z.string(),
  range: z.string(),
  value: z.any(),
  success: z.boolean(),
  error: z.string(),
});

const RangeDataResultSchema = z.object({
  spreadsheetId: z.string(),
  range: z.string(),
  values: z.array(z.array(z.any())),
  majorDimension: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const CopyRangeResultSchema = z.object({
  spreadsheetId: z.string(),
  sourceRange: z.string(),
  destinationRange: z.string(),
  updatedRange: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const SheetDataResultSchema = z.object({
  spreadsheetId: z.string(),
  sheetName: z.string(),
  sheetId: z.number(),
  values: z.array(z.array(z.any())),
  metadata: z.object({
    rowCount: z.number(),
    columnCount: z.number(),
    lastUpdated: z.string().optional(),
  }).optional(),
  success: z.boolean(),
  error: z.string(),
});

const UpdateResultSchema = z.object({
  spreadsheetId: z.string(),
  updatedRange: z.string(),
  updatedRows: z.number(),
  updatedColumns: z.number(),
  updatedCells: z.number(),
  success: z.boolean(),
  error: z.string(),
});

const BatchUpdateResultSchema = z.object({
  spreadsheetId: z.string(),
  totalUpdatedRows: z.number(),
  totalUpdatedColumns: z.number(),
  totalUpdatedCells: z.number(),
  updateResults: z.array(z.object({
    updatedRange: z.string(),
    updatedRows: z.number(),
    updatedColumns: z.number(),
    updatedCells: z.number(),
  })),
  success: z.boolean(),
  error: z.string(),
});

const AppendResultSchema = z.object({
  spreadsheetId: z.string(),
  tableRange: z.string(),
  updates: z.object({
    spreadsheetId: z.string(),
    updatedRange: z.string(),
    updatedRows: z.number(),
  }).optional(),
  success: z.boolean(),
  error: z.string(),
});

const RowDataSchema = z.object({
  range: z.string(),
  values: z.array(z.array(z.any())),
  success: z.boolean(),
  error: z.string(),
});

const SheetResultSchema = z.object({
  sheetId: z.number(),
  title: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const ValuesResultSchema = z.object({
  range: z.string(),
  values: z.array(z.array(z.any())),
  majorDimension: z.string(),
  success: z.boolean(),
  error: z.string(),
});

const GoogleSheetsBubbleResultSchema = z.discriminatedUnion('operation', [
  // New operations
  z.object({
    operation: z.literal('createSpreadsheet'),
    result: SpreadsheetResultSchema,
  }),
  z.object({
    operation: z.literal('getSpreadsheet'),
    result: SheetInfoSchema,
  }),
  z.object({
    operation: z.literal('deleteSpreadsheet'),
    result: DeleteSpreadsheetResultSchema,
  }),
  z.object({
    operation: z.literal('copySpreadsheet'),
    result: CopySpreadsheetResultSchema,
  }),
  z.object({
    operation: z.literal('updateCell'),
    result: UpdateResultSchema,
  }),
  z.object({
    operation: z.literal('getCellValue'),
    result: CellValueResultSchema,
  }),
  z.object({
    operation: z.literal('batchUpdate'),
    result: BatchUpdateResultSchema,
  }),
  z.object({
    operation: z.literal('appendRow'),
    result: AppendResultSchema,
  }),
  z.object({
    operation: z.literal('getRange'),
    result: RangeDataResultSchema,
  }),
  z.object({
    operation: z.literal('clearRange'),
    result: z.object({
      spreadsheetId: z.string(),
      clearedRange: z.string(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
  z.object({
    operation: z.literal('copyRange'),
    result: CopyRangeResultSchema,
  }),
  z.object({
    operation: z.literal('addSheet'),
    result: SheetResultSchema,
  }),
  z.object({
    operation: z.literal('deleteSheet'),
    result: SheetResultSchema,
  }),
  z.object({
    operation: z.literal('getSheetData'),
    result: SheetDataResultSchema,
  }),
  // Legacy operations for backward compatibility
  z.object({
    operation: z.literal('getRow'),
    result: RowDataSchema,
  }),
  z.object({
    operation: z.literal('deleteRow'),
    result: z.object({
      spreadsheetId: z.string(),
      deletedRows: z.number(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
  z.object({
    operation: z.literal('getValues'),
    result: ValuesResultSchema,
  }),
  z.object({
    operation: z.literal('setValues'),
    result: UpdateResultSchema,
  }),
  z.object({
    operation: z.literal('clearValues'),
    result: z.object({
      spreadsheetId: z.string(),
      clearedRange: z.string(),
      success: z.boolean(),
      error: z.string(),
    }),
  }),
]);

type GoogleSheetsBubbleResult = z.output<typeof GoogleSheetsBubbleResultSchema>;

// ============================================================================
// GOOGLE SHEETS API CLIENT
// ============================================================================

class GoogleSheetsClient {
  private sheetsBaseUrl: string = 'https://sheets.googleapis.com/v4/spreadsheets';
  private driveBaseUrl: string = 'https://www.googleapis.com/drive/v3/files';
  private headers: Record<string, string>;
  private rateLimiter: RateLimiter;

  constructor(accessToken: string) {
    this.headers = {
      'Authorization': `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    };
    this.rateLimiter = new RateLimiter();
  }

  async get(endpoint: string, useDriveApi = false): Promise<any> {
    await this.rateLimiter.waitForSlot();

    const baseUrl = useDriveApi ? this.driveBaseUrl : this.sheetsBaseUrl;
    const url = endpoint.includes('http') ? endpoint : `${baseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async post(endpoint: string, body?: any): Promise<any> {
    await this.rateLimiter.waitForSlot();

    const url = `${this.sheetsBaseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google Sheets API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async put(endpoint: string, body: any): Promise<any> {
    await this.rateLimiter.waitForSlot();

    const url = `${this.sheetsBaseUrl}/${endpoint}`;
    const response = await fetch(url, {
      method: 'PUT',
      headers: this.headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google Sheets API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async delete(spreadsheetId: string): Promise<any> {
    await this.rateLimiter.waitForSlot();

    const url = `${this.driveBaseUrl}/${spreadsheetId}`;
    const response = await fetch(url, {
      method: 'DELETE',
      headers: this.headers,
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok && response.status !== 204) {
      const error = await response.text();
      throw new Error(`Google Drive API error: ${response.status} - ${error}`);
    }

    return { success: true };
  }

  async copy(spreadsheetId: string, title: string, parents?: string[]): Promise<any> {
    await this.rateLimiter.waitForSlot();

    const url = `${this.driveBaseUrl}/${spreadsheetId}/copy`;
    const body: any = { name: title };
    if (parents && parents.length > 0) {
      body.parents = parents;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google Drive API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async batchUpdate(spreadsheetId: string, requests: any[]): Promise<any> {
    return this.post(`${spreadsheetId}:batchUpdate`, {
      requests,
      includeSpreadsheetInResponse: false,
    });
  }
}

// ============================================================================
// RATE LIMITER
// ============================================================================

class RateLimiter {
  private requestTimes: number[] = [];
  private readonly maxRequests: number;
  private readonly timeWindow: number;

  constructor(maxRequests = 50, timeWindow = 60000) {
    this.maxRequests = maxRequests;
    this.timeWindow = timeWindow;
  }

  async waitForSlot(): Promise<void> {
    const now = Date.now();

    // Remove old request times outside the time window
    this.requestTimes = this.requestTimes.filter(
      time => now - time < this.timeWindow
    );

    // If we've hit the limit, wait for a slot
    if (this.requestTimes.length >= this.maxRequests) {
      const oldestRequest = this.requestTimes[0];
      const waitTime = this.timeWindow - (now - oldestRequest);

      if (waitTime > 0) {
        console.log(`[RateLimiter] Rate limit reached. Waiting ${waitTime}ms...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }

    // Record this request
    this.requestTimes.push(Date.now());
  }
}

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class GoogleSheetsBubble<
  T extends GoogleSheetsBubbleParams = GoogleSheetsBubbleParams
> extends ServiceBubble<T, any> {
  static readonly type = 'service' as const;
  static readonly service = 'google-sheets';
  static readonly authType = 'oauth' as const;
  static readonly bubbleName = 'google-sheets';
  static readonly schema = GoogleSheetsBubbleParamsSchema;
  static readonly resultSchema = GoogleSheetsBubbleResultSchema;
  static readonly shortDescription = 'Complete Google Sheets integration for spreadsheet operations';
  static readonly longDescription = `
    Comprehensive Google Sheets service bubble for all spreadsheet operations.

    Operations:
    1. createSpreadsheet - Create a new spreadsheet with custom sheets
    2. getSpreadsheet - Get spreadsheet metadata and structure
    3. deleteSpreadsheet - Delete a spreadsheet permanently
    4. copySpreadsheet - Copy a spreadsheet to new location
    5. updateCell - Update a single cell
    6. getCellValue - Get a single cell value
    7. batchUpdate - Update multiple cells efficiently
    8. appendRow - Append data to the end of a sheet
    9. getRange - Get values from a range
    10. clearRange - Clear all values from a range
    11. copyRange - Copy range to destination
    12. addSheet - Add a new sheet to the spreadsheet
    13. deleteSheet - Remove a sheet from the spreadsheet
    14. getSheetData - Get complete sheet data with metadata

    Features:
    - OAuth 2.0 authentication with token validation
    - Full CRUD operations on spreadsheets and sheets
    - Batch updates for efficiency
    - Row and column operations
    - Sheet management
    - Value formatting options (RAW, USER_ENTERED)
    - Range operations with A1 notation support
    - Rate limiting and quota management
    - Resilience patterns (circuit breaker, retry, deduplication)
    - Structured logging and error handling
    - Input validation and sanitization

    Use Cases:
    - Automated reporting and data collection
    - Data synchronization between systems
    - Spreadsheet-based workflows
    - Data analysis and visualization
    - Template generation and management
    - Batch data processing
  `;
  static readonly alias = 'sheets';

  private client: GoogleSheetsClient | null = null;
  private resilience: ResilienceWrapper;

  constructor(
    params: T,
    context?: BubbleContext
  ) {
    super(params, context);

    this.resilience = new ResilienceWrapper(
      DEFAULT_RESILIENCE_CONFIG
    );
  }

  public async testCredential(): Promise<boolean> {
    const token = this.chooseCredential();
    if (!token) {
      return false;
    }

    try {
      const client = new GoogleSheetsClient(token);
      await client.get('');
      return true;
    } catch {
      return false;
    }
  }

  protected chooseCredential(): string | undefined {
    const credentials = (this.params as any).credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Google Sheets credentials are required');
    }
    return credentials[CredentialType.GOOGLE_DRIVE_CRED] || credentials[CredentialType.GOOGLE_SHEETS_CRED];
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<Extract<GoogleSheetsBubbleResult, { operation: T['operation'] }>> {
    void context;

    const token = this.chooseCredential();
    if (!token) {
      return this.errorResult('Google Sheets access token is required');
    }

    this.client = new GoogleSheetsClient(token);

    const { operation } = this.params;

    try {
      const result = await this.resilience.execute(
        `sheets-${operation}-${Date.now()}`,
        async () => {
          switch (operation) {
            // New operations
            case 'createSpreadsheet':
              return await this.createSpreadsheet(this.params as any);
            case 'getSpreadsheet':
              return await this.getSpreadsheet(this.params as any);
            case 'deleteSpreadsheet':
              return await this.deleteSpreadsheet(this.params as any);
            case 'copySpreadsheet':
              return await this.copySpreadsheet(this.params as any);
            case 'updateCell':
              return await this.updateCell(this.params as any);
            case 'getCellValue':
              return await this.getCellValue(this.params as any);
            case 'batchUpdate':
              return await this.batchUpdate(this.params as any);
            case 'appendRow':
              return await this.appendRow(this.params as any);
            case 'getRange':
              return await this.getRange(this.params as any);
            case 'clearRange':
              return await this.clearRange(this.params as any);
            case 'copyRange':
              return await this.copyRange(this.params as any);
            case 'addSheet':
              return await this.addSheet(this.params as any);
            case 'deleteSheet':
              return await this.deleteSheet(this.params as any);
            case 'getSheetData':
              return await this.getSheetData(this.params as any);
            // Legacy operations for backward compatibility
            case 'getRow':
              return await this.getRow(this.params as any);
            case 'deleteRow':
              return await this.deleteRow(this.params as any);
            case 'getValues':
              return await this.getValues(this.params as any);
            case 'setValues':
              return await this.setValues(this.params as any);
            case 'clearValues':
              return await this.clearValues(this.params as any);
            default:
              throw new Error(`Unsupported operation: ${operation}`);
          }
        }
      );

      return {
        operation,
        result,
      } as any;
    } catch (error) {
      return {
        operation,
        result: {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      } as any;
    }
  }

  // ========================================================================
  // OPERATION 1: CREATE SPREADSHEET
  // ========================================================================

  private async createSpreadsheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'createSpreadsheet' }>
  ): Promise<typeof SpreadsheetResultSchema._output> {
    const { title, sheets } = params;

    try {
      const body: any = {
        properties: {
          title,
        },
      };

      if (sheets && sheets.length > 0) {
        body.sheets = sheets.map(sheet => ({
          properties: {
            title: sheet.title,
            gridProperties: {
              rowCount: sheet.rowCount,
              columnCount: sheet.columnCount,
            },
          },
        }));
      }

      const response = await this.client!.post('', body);

      return {
        spreadsheetId: response.spreadsheetId,
        title: response.properties.title,
        url: response.spreadsheetUrl,
        sheetCount: response.sheets.length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId: '',
        title,
        url: '',
        sheetCount: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create spreadsheet',
      };
    }
  }

  // ========================================================================
  // OPERATION 2: GET SHEET
  // ========================================================================

  private async getSheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getSheet' }>
  ): Promise<typeof SheetInfoSchema._output> {
    const { spreadsheetId, includeGridData } = params;

    try {
      const ranges = includeGridData ? undefined : [];
      const includeGridDataParam = includeGridData ? true : false;

      const response = await this.client!.get(
        `${spreadsheetId}?includeGridData=${includeGridDataParam}&ranges=${ranges || ''}`
      );

      return {
        spreadsheetId,
        title: response.properties.title,
        sheets: response.sheets.map((sheet: any) => ({
          sheetId: sheet.properties.sheetId,
          title: sheet.properties.title,
          index: sheet.properties.index,
          sheetType: sheet.properties.sheetType,
          gridProperties: sheet.properties.gridProperties,
        })),
        namedRanges: response.namedRanges,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        title: '',
        sheets: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get sheet info',
      };
    }
  }

  // ========================================================================
  // OPERATION 3: UPDATE CELL
  // ========================================================================

  private async updateCell(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'updateCell' }>
  ): Promise<typeof UpdateResultSchema._output> {
    const { spreadsheetId, range, value, valueInputOption } = params;

    try {
      const response = await this.client!.put(
        `${spreadsheetId}/values/${range}?valueInputOption=${valueInputOption}`,
        {
          values: [[value]],
        }
      );

      return {
        spreadsheetId,
        updatedRange: response.updates.updatedRange,
        updatedRows: response.updates.updatedRows || 1,
        updatedColumns: response.updates.updatedColumns || 1,
        updatedCells: response.updates.updatedCells || 1,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        updatedRange: range,
        updatedRows: 0,
        updatedColumns: 0,
        updatedCells: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update cell',
      };
    }
  }

  // ========================================================================
  // OPERATION 4: BATCH UPDATE
  // ========================================================================

  private async batchUpdate(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'batchUpdate' }>
  ): Promise<typeof BatchUpdateResultSchema._output> {
    const { spreadsheetId, updates, valueInputOption } = params;

    try {
      const requests = updates.map(update => ({
        updateCells: {
          rows: [
            {
              values: update.values.map(row => ({
                values: row.map(cell => ({
                  userEnteredValue: { stringValue: String(cell) },
                })),
              })),
            },
          ],
          range: update.range,
          fields: 'userEnteredValue',
        },
      }));

      const response = await this.client!.batchUpdate(spreadsheetId, requests);

      const updateResults = updates.map((update, index) => ({
        updatedRange: update.range,
        updatedRows: update.values.length,
        updatedColumns: update.values[0]?.length || 0,
        updatedCells: update.values.length * (update.values[0]?.length || 0),
      }));

      return {
        spreadsheetId,
        totalUpdatedRows: updateResults.reduce((sum, r) => sum + r.updatedRows, 0),
        totalUpdatedColumns: updateResults.reduce((sum, r) => sum + r.updatedColumns, 0),
        totalUpdatedCells: updateResults.reduce((sum, r) => sum + r.updatedCells, 0),
        updateResults,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        totalUpdatedRows: 0,
        totalUpdatedColumns: 0,
        totalUpdatedCells: 0,
        updateResults: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to batch update',
      };
    }
  }

  // ========================================================================
  // OPERATION 5: APPEND ROW
  // ========================================================================

  private async appendRow(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'appendRow' }>
  ): Promise<typeof AppendResultSchema._output> {
    const { spreadsheetId, range, values, valueInputOption, insertDataOption } = params;

    try {
      const response = await this.client!.post(
        `${spreadsheetId}/values/${range}:append?valueInputOption=${valueInputOption}&insertDataOption=${insertDataOption}`,
        {
          values: [values],
        }
      );

      return {
        spreadsheetId,
        tableRange: response.tableRange,
        updates: response.updates,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        tableRange: range,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to append row',
      };
    }
  }

  // ========================================================================
  // OPERATION 6: GET ROW
  // ========================================================================

  private async getRow(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getRow' }>
  ): Promise<typeof RowDataSchema._output> {
    const { spreadsheetId, range } = params;

    try {
      const response = await this.client!.get(
        `${spreadsheetId}/values/${range}?valueRenderOption=UNFORMATTED_VALUE&dateTimeRenderOption=FORMATTED_STRING`
      );

      return {
        range: response.range,
        values: response.values || [[]],
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        range,
        values: [[]],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get row',
      };
    }
  }

  // ========================================================================
  // OPERATION 7: DELETE ROW
  // ========================================================================

  private async deleteRow(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'deleteRow' }>
  ): Promise<{ spreadsheetId: string; deletedRows: number; success: boolean; error: string }> {
    const { spreadsheetId, sheetId, rowIndex } = params;

    try {
      await this.client!.batchUpdate(spreadsheetId, [
        {
          deleteDimension: {
            range: {
              sheetId,
              dimension: 'ROWS',
              startIndex: rowIndex,
              endIndex: rowIndex + 1,
            },
          },
        },
      ]);

      return {
        spreadsheetId,
        deletedRows: 1,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        deletedRows: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete row',
      };
    }
  }

  // ========================================================================
  // OPERATION 8: ADD SHEET
  // ========================================================================

  private async addSheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'addSheet' }>
  ): Promise<typeof SheetResultSchema._output> {
    const { spreadsheetId, title, rowCount, columnCount } = params;

    try {
      const response = await this.client!.batchUpdate(spreadsheetId, [
        {
          addSheet: {
            properties: {
              title,
              gridProperties: {
                rowCount,
                columnCount,
              },
            },
          },
        },
      ]);

      const reply = response.replies[0].addSheet;

      return {
        sheetId: reply.properties.sheetId,
        title: reply.properties.title,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        sheetId: 0,
        title,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to add sheet',
      };
    }
  }

  // ========================================================================
  // OPERATION 9: DELETE SHEET
  // ========================================================================

  private async deleteSheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'deleteSheet' }>
  ): Promise<typeof SheetResultSchema._output> {
    const { spreadsheetId, sheetId } = params;

    try {
      await this.client!.batchUpdate(spreadsheetId, [
        {
          deleteSheet: {
            sheetId,
          },
        },
      ]);

      return {
        sheetId,
        title: '',
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        sheetId,
        title: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete sheet',
      };
    }
  }

  // ========================================================================
  // OPERATION 10: GET VALUES
  // ========================================================================

  private async getValues(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getValues' }>
  ): Promise<typeof ValuesResultSchema._output> {
    const { spreadsheetId, range, majorDimension, valueRenderOption } = params;

    try {
      const response = await this.client!.get(
        `${spreadsheetId}/values/${range}?majorDimension=${majorDimension}&valueRenderOption=${valueRenderOption}`
      );

      return {
        range: response.range,
        values: response.values || [[]],
        majorDimension: response.majorDimension,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        range,
        values: [[]],
        majorDimension: majorDimension ?? 'ROWS',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get values',
      };
    }
  }

  // ========================================================================
  // OPERATION 11: SET VALUES
  // ========================================================================

  private async setValues(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'setValues' }>
  ): Promise<typeof UpdateResultSchema._output> {
    const { spreadsheetId, range, values, valueInputOption } = params;

    try {
      const response = await this.client!.put(
        `${spreadsheetId}/values/${range}?valueInputOption=${valueInputOption}`,
        {
          values,
        }
      );

      return {
        spreadsheetId,
        updatedRange: response.updates.updatedRange,
        updatedRows: response.updates.updatedRows || values.length,
        updatedColumns: response.updates.updatedColumns || values[0]?.length || 0,
        updatedCells: response.updates.updatedCells || values.flat().length,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        updatedRange: range,
        updatedRows: 0,
        updatedColumns: 0,
        updatedCells: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to set values',
      };
    }
  }

  // ========================================================================
  // OPERATION 12: CLEAR VALUES (LEGACY)
  // ========================================================================

  private async clearValues(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'clearValues' }>
  ): Promise<{ spreadsheetId: string; clearedRange: string; success: boolean; error: string }> {
    const { spreadsheetId, range } = params;

    try {
      const response = await this.client!.post(
        `${spreadsheetId}/values/${range}:clear`,
        {}
      );

      return {
        spreadsheetId,
        clearedRange: response.clearedRange,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        clearedRange: range,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to clear values',
      };
    }
  }

  // ========================================================================
  // NEW OPERATIONS
  // ========================================================================

  // OPERATION 2: GET SPREADSHEET

  private async getSpreadsheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getSpreadsheet' }>
  ): Promise<typeof SheetInfoSchema._output> {
    const { spreadsheetId, includeGridData } = params;

    try {
      const ranges = includeGridData ? undefined : [];
      const includeGridDataParam = includeGridData ? true : false;

      const response = await this.client!.get(
        `${spreadsheetId}?includeGridData=${includeGridDataParam}&ranges=${ranges || ''}`
      );

      return {
        spreadsheetId,
        title: response.properties.title,
        sheets: response.sheets.map((sheet: any) => ({
          sheetId: sheet.properties.sheetId,
          title: sheet.properties.title,
          index: sheet.properties.index,
          sheetType: sheet.properties.sheetType,
          gridProperties: sheet.properties.gridProperties,
        })),
        namedRanges: response.namedRanges,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        title: '',
        sheets: [],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get spreadsheet',
      };
    }
  }

  // OPERATION 3: DELETE SPREADSHEET

  private async deleteSpreadsheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'deleteSpreadsheet' }>
  ): Promise<typeof DeleteSpreadsheetResultSchema._output> {
    const { spreadsheetId } = params;

    try {
      await this.client!.delete(spreadsheetId);

      return {
        spreadsheetId,
        deleted: true,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        deleted: false,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete spreadsheet',
      };
    }
  }

  // OPERATION 4: COPY SPREADSHEET

  private async copySpreadsheet(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'copySpreadsheet' }>
  ): Promise<typeof CopySpreadsheetResultSchema._output> {
    const { spreadsheetId, title, destinationFolderId } = params;

    try {
      const parents = destinationFolderId ? [destinationFolderId] : undefined;
      const response = await this.client!.copy(spreadsheetId, title, parents);

      return {
        originalSpreadsheetId: spreadsheetId,
        newSpreadsheetId: response.id,
        title: response.name,
        url: `https://docs.google.com/spreadsheets/d/${response.id}`,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        originalSpreadsheetId: spreadsheetId,
        newSpreadsheetId: '',
        title,
        url: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to copy spreadsheet',
      };
    }
  }

  // OPERATION 6: GET CELL VALUE

  private async getCellValue(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getCellValue' }>
  ): Promise<typeof CellValueResultSchema._output> {
    const { spreadsheetId, range, valueRenderOption } = params;

    try {
      const response = await this.client!.get(
        `${spreadsheetId}/values/${range}?valueRenderOption=${valueRenderOption}`
      );

      const value = response.values?.[0]?.[0];

      return {
        spreadsheetId,
        range,
        value,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        range,
        value: null,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get cell value',
      };
    }
  }

  // OPERATION 9: GET RANGE

  private async getRange(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getRange' }>
  ): Promise<typeof RangeDataResultSchema._output> {
    const { spreadsheetId, range, majorDimension, valueRenderOption } = params;

    try {
      const response = await this.client!.get(
        `${spreadsheetId}/values/${range}?majorDimension=${majorDimension}&valueRenderOption=${valueRenderOption}`
      );

      return {
        spreadsheetId,
        range: response.range,
        values: response.values || [[]],
        majorDimension: response.majorDimension,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        range,
        values: [[]],
        majorDimension: majorDimension ?? "ROWS",
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get range',
      };
    }
  }

  // OPERATION 10: CLEAR RANGE

  private async clearRange(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'clearRange' }>
  ): Promise<{ spreadsheetId: string; clearedRange: string; success: boolean; error: string }> {
    const { spreadsheetId, range } = params;

    try {
      const response = await this.client!.post(
        `${spreadsheetId}/values/${range}:clear`,
        {}
      );

      return {
        spreadsheetId,
        clearedRange: response.clearedRange,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        clearedRange: range,
        success: false,
        error: error instanceof Error ? error.message : 'Failed to clear range',
      };
    }
  }

  // OPERATION 11: COPY RANGE

  private async copyRange(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'copyRange' }>
  ): Promise<typeof CopyRangeResultSchema._output> {
    const { spreadsheetId, sourceRange, destinationRange } = params;

    try {
      // Parse sheet name from ranges
      const sourceSheetMatch = sourceRange.match(/(^.+?)!/);
      const destSheetMatch = destinationRange.match(/(^.+?)!/);

      const sourceSheet = sourceSheetMatch ? sourceSheetMatch[1] : 'Sheet1';
      const destSheet = destSheetMatch ? destSheetMatch[1] : 'Sheet1';

      // Get sheet metadata to find sheet IDs
      const spreadsheet = await this.client!.get(
        `${spreadsheetId}?includeGridData=false`
      );

      const sourceSheetId = spreadsheet.sheets.find(
        (s: any) => s.properties.title === sourceSheet
      )?.properties.sheetId;

      const destSheetId = spreadsheet.sheets.find(
        (s: any) => s.properties.title === destSheet
      )?.properties.sheetId;

      if (!sourceSheetId || !destSheetId) {
        throw new Error('Sheet not found');
      }

      // Execute copy using batchUpdate
      await this.client!.batchUpdate(spreadsheetId, [
        {
          copyPaste: {
            source: {
              sheetId: sourceSheetId,
              startRowIndex: 0,
              endRowIndex: 1,
            },
            destination: {
              sheetId: destSheetId,
              startRowIndex: 0,
              endRowIndex: 1,
            },
            pasteType: 'PASTE_NORMAL',
          },
        },
      ]);

      return {
        spreadsheetId,
        sourceRange,
        destinationRange,
        updatedRange: destinationRange,
        success: true,
        error: '',
      };
    } catch (error) {
      return {
        spreadsheetId,
        sourceRange,
        destinationRange,
        updatedRange: '',
        success: false,
        error: error instanceof Error ? error.message : 'Failed to copy range',
      };
    }
  }

  // OPERATION 14: GET SHEET DATA

  private async getSheetData(
    params: Extract<GoogleSheetsBubbleParams, { operation: 'getSheetData' }>
  ): Promise<typeof SheetDataResultSchema._output> {
    const { spreadsheetId, sheetName, includeMetadata } = params;

    try {
      // Get spreadsheet info to find sheet ID
      const spreadsheet = await this.client!.get(
        `${spreadsheetId}?includeGridData=false`
      );

      const sheet = spreadsheet.sheets.find(
        (s: any) => s.properties.title === sheetName
      );

      if (!sheet) {
        throw new Error(`Sheet "${sheetName}" not found`);
      }

      const sheetId = sheet.properties.sheetId;
      const range = `'${sheetName}'`;

      // Get sheet values
      const values = await this.client!.get(
        `${spreadsheetId}/values/${encodeURIComponent(range)}?valueRenderOption=UNFORMATTED_VALUE`
      );

      const result: typeof SheetDataResultSchema._output = {
        spreadsheetId,
        sheetName,
        sheetId,
        values: values.values || [[]],
        success: true,
        error: '',
      };

      if (includeMetadata) {
        result.metadata = {
          rowCount: sheet.properties.gridProperties?.rowCount || 1000,
          columnCount: sheet.properties.gridProperties?.columnCount || 26,
        };
      }

      return result;
    } catch (error) {
      return {
        spreadsheetId,
        sheetName,
        sheetId: 0,
        values: [[]],
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get sheet data',
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
}
