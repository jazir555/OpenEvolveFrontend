import { ServiceBubble } from '../../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { GoogleSheetsParamsSchema, GoogleSheetsResultSchema, } from './google-sheets.schema.js';
import { enhanceErrorMessage } from './google-sheets.utils.js';
import { AuthenticationError } from '../../common/error-handlers.js';
/**
 * Google Sheets Service Bubble
 *
 * Comprehensive Google Sheets integration for spreadsheet data management.
 *
 * Features:
 * - Automatic range normalization (quotes sheet names with spaces)
 * - Automatic value sanitization (converts null/undefined to empty strings)
 * - Enhanced error messages with helpful hints
 * - Support for all major Google Sheets operations
 *
 * Use cases:
 * - Read and write spreadsheet data with flexible ranges
 * - Batch operations for efficient data processing
 * - Create and manage spreadsheets and sheets
 * - Clear and append data with various formatting options
 * - Handle formulas, formatted values, and raw data
 *
 * Security Features:
 * - OAuth 2.0 authentication with Google
 * - Scoped access permissions for Google Sheets
 * - Secure data validation and sanitization
 * - User-controlled access to spreadsheet data
 *
 * @template T - Google Sheets bubble parameters type
 */
export class GoogleSheetsBubble extends ServiceBubble {
    static type = 'service';
    static service = 'google-sheets';
    static authType = 'oauth';
    static bubbleName = 'google-sheets';
    static schema = GoogleSheetsParamsSchema;
    static resultSchema = GoogleSheetsResultSchema;
    static shortDescription = 'Google Sheets integration for spreadsheet operations';
    static longDescription = `
    Google Sheets service integration for comprehensive spreadsheet data management.

    Features:
    - Automatic range normalization (sheet names with spaces are automatically quoted)
    - Automatic value sanitization (null/undefined converted to empty strings)
    - Enhanced error messages with helpful hints
    - Support for all major Google Sheets operations

    Use cases:
    - Read and write spreadsheet data with flexible ranges
    - Batch operations for efficient data processing
    - Create and manage spreadsheets and sheets
    - Clear and append data with various formatting options
    - Handle formulas, formatted values, and raw data

    Security Features:
    - OAuth 2.0 authentication with Google
    - Scoped access permissions for Google Sheets
    - Secure data validation and sanitization
    - User-controlled access to spreadsheet data
  `;
    static alias = 'sheets';
    /**
     * Create a new Google Sheets Bubble instance
     * @param params - Operation parameters
     * @param context - Bubble execution context
     */
    constructor(params = {
        operation: 'read_values',
        spreadsheet_id: '',
        range: 'Sheet1!A1:B10',
    }, context) {
        super(params, context);
    }
    /**
     * Test the validity of the Google Sheets credentials
     * @returns Promise that resolves to true if credentials are valid, false otherwise
     * @throws AuthenticationError if credentials are missing
     */
    async testCredential() {
        const credential = this.chooseCredential();
        if (!credential) {
            throw new AuthenticationError('Google Sheets credentials are required');
        }
        try {
            // Test the credentials by validating the OAuth access token using Google's tokeninfo endpoint
            const response = await fetch(`https://www.googleapis.com/oauth2/v3/tokeninfo?access_token=${encodeURIComponent(credential)}`);
            // A successful response indicates that the access token is valid
            return response.ok;
        }
        catch {
            return false;
        }
    }
    async makeSheetsApiRequest(endpoint, method = 'GET', body, headers = {}, spreadsheetId, range) {
        const url = endpoint.startsWith('https://')
            ? endpoint
            : `https://sheets.googleapis.com/v4${endpoint}`;
        const requestHeaders = {
            Authorization: `Bearer ${this.chooseCredential()}`,
            'Content-Type': 'application/json',
            ...headers,
        };
        const requestInit = {
            method,
            headers: requestHeaders,
        };
        if (body && method !== 'GET') {
            requestInit.body = JSON.stringify(body);
        }
        const response = await fetch(url, requestInit);
        if (!response.ok) {
            const errorText = await response.text();
            // Extract spreadsheet ID from endpoint if not provided
            const extractedSpreadsheetId = spreadsheetId ||
                endpoint.match(/\/spreadsheets\/([^/]+)/)?.[1] ||
                undefined;
            const enhancedError = enhanceErrorMessage(errorText, response.status, response.statusText, extractedSpreadsheetId, range);
            throw new Error(enhancedError);
        }
        // Handle empty responses
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }
        else {
            return await response.text();
        }
    }
    async performAction(context) {
        void context;
        const { operation } = this.params;
        try {
            const result = await (async () => {
                // Cast to output type since base class already parsed input through Zod
                const parsedParams = this.params;
                switch (operation) {
                    case 'read_values':
                        return await this.readValues(parsedParams);
                    case 'write_values':
                        return await this.writeValues(parsedParams);
                    case 'update_values':
                        return await this.updateValues(parsedParams);
                    case 'append_values':
                        return await this.appendValues(parsedParams);
                    case 'clear_values':
                        return await this.clearValues(parsedParams);
                    case 'batch_read_values':
                        return await this.batchReadValues(parsedParams);
                    case 'batch_update_values':
                        return await this.batchUpdateValues(parsedParams);
                    case 'get_spreadsheet_info':
                        return await this.getSpreadsheetInfo(parsedParams);
                    case 'create_spreadsheet':
                        return await this.createSpreadsheet(parsedParams);
                    case 'add_sheet':
                        return await this.addSheet(parsedParams);
                    case 'delete_sheet':
                        return await this.deleteSheet(parsedParams);
                    default:
                        throw new Error(`Unsupported operation: ${operation}`);
                }
            })();
            return result;
        }
        catch (error) {
            return {
                operation,
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async readValues(params) {
        const { spreadsheet_id, range, major_dimension, value_render_option, date_time_render_option, } = params;
        const queryParams = new URLSearchParams({
            majorDimension: major_dimension || 'ROWS',
            valueRenderOption: value_render_option || 'FORMATTED_VALUE',
            dateTimeRenderOption: date_time_render_option || 'SERIAL_NUMBER',
        });
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values/${encodeURIComponent(range)}?${queryParams.toString()}`, 'GET', undefined, {}, undefined, range);
        return {
            operation: 'read_values',
            success: true,
            range: response.range,
            values: response.values || [],
            major_dimension: response.majorDimension,
            error: '',
        };
    }
    async writeValues(params) {
        const { spreadsheet_id, range, values, major_dimension, value_input_option, include_values_in_response, } = params;
        const queryParams = new URLSearchParams({
            valueInputOption: value_input_option || 'USER_ENTERED',
            includeValuesInResponse: include_values_in_response?.toString() || 'false',
        });
        const body = {
            range,
            majorDimension: major_dimension,
            values,
        };
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values/${encodeURIComponent(range)}?${queryParams.toString()}`, 'PUT', body, {}, undefined, range);
        return {
            operation: 'write_values',
            success: true,
            updated_range: response.updatedRange,
            updated_rows: response.updatedRows,
            updated_columns: response.updatedColumns,
            updated_cells: response.updatedCells,
            updated_data: response.updatedData,
            error: '',
        };
    }
    async updateValues(params) {
        const { spreadsheet_id, range, values, major_dimension, value_input_option, include_values_in_response, } = params;
        const queryParams = new URLSearchParams({
            valueInputOption: value_input_option || 'USER_ENTERED',
            includeValuesInResponse: include_values_in_response?.toString() || 'false',
        });
        const body = {
            range,
            majorDimension: major_dimension,
            values,
        };
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values/${encodeURIComponent(range)}?${queryParams.toString()}`, 'PUT', body, {}, undefined, range);
        return {
            operation: 'update_values',
            success: true,
            updated_range: response.updatedRange,
            updated_rows: response.updatedRows,
            updated_columns: response.updatedColumns,
            updated_cells: response.updatedCells,
            updated_data: response.updatedData,
            error: '',
        };
    }
    async appendValues(params) {
        const { spreadsheet_id, range, values, major_dimension, value_input_option, insert_data_option, include_values_in_response, } = params;
        const queryParams = new URLSearchParams({
            valueInputOption: value_input_option || 'USER_ENTERED',
            insertDataOption: insert_data_option || 'INSERT_ROWS',
            includeValuesInResponse: include_values_in_response?.toString() || 'false',
        });
        const body = {
            range,
            majorDimension: major_dimension,
            values,
        };
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values/${encodeURIComponent(range)}:append?${queryParams.toString()}`, 'POST', body, {}, undefined, range);
        return {
            operation: 'append_values',
            success: true,
            table_range: response.tableRange,
            updated_range: response.updates?.updatedRange,
            updated_rows: response.updates?.updatedRows,
            updated_columns: response.updates?.updatedColumns,
            updated_cells: response.updates?.updatedCells,
            error: '',
        };
    }
    async clearValues(params) {
        const { spreadsheet_id, range } = params;
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values/${encodeURIComponent(range)}:clear`, 'POST', {}, {}, undefined, range);
        return {
            operation: 'clear_values',
            success: true,
            cleared_range: response.clearedRange,
            error: '',
        };
    }
    async batchReadValues(params) {
        const { spreadsheet_id, ranges, major_dimension, value_render_option, date_time_render_option, } = params;
        const queryParams = new URLSearchParams({
            majorDimension: major_dimension || 'ROWS',
            valueRenderOption: value_render_option || 'FORMATTED_VALUE',
            dateTimeRenderOption: date_time_render_option || 'SERIAL_NUMBER',
        });
        // Add multiple ranges
        ranges.forEach((range) => queryParams.append('ranges', range));
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values:batchGet?${queryParams.toString()}`);
        return {
            operation: 'batch_read_values',
            success: true,
            value_ranges: response.valueRanges || [],
            error: '',
        };
    }
    async batchUpdateValues(params) {
        const { spreadsheet_id, value_ranges, value_input_option, include_values_in_response, } = params;
        const body = {
            valueInputOption: value_input_option,
            includeValuesInResponse: include_values_in_response,
            data: value_ranges.map((vr) => ({
                range: vr.range,
                majorDimension: vr.major_dimension,
                values: vr.values,
            })),
        };
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}/values:batchUpdate`, 'POST', body);
        return {
            operation: 'batch_update_values',
            success: true,
            total_updated_rows: response.totalUpdatedRows,
            total_updated_columns: response.totalUpdatedColumns,
            total_updated_cells: response.totalUpdatedCells,
            total_updated_sheets: response.totalUpdatedSheets,
            responses: response.responses?.map((r) => ({
                updated_range: r.updatedRange,
                updated_rows: r.updatedRows,
                updated_columns: r.updatedColumns,
                updated_cells: r.updatedCells,
            })),
            error: '',
        };
    }
    async getSpreadsheetInfo(params) {
        const { spreadsheet_id, include_grid_data } = params;
        const queryParams = new URLSearchParams();
        if (include_grid_data) {
            queryParams.set('includeGridData', 'true');
        }
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}?${queryParams.toString()}`);
        return {
            operation: 'get_spreadsheet_info',
            success: true,
            spreadsheet: response,
            error: '',
        };
    }
    async createSpreadsheet(params) {
        const { title, sheet_titles } = params;
        // sheet_titles has a default value of ['Sheet1'] from schema, so this is a safety check
        const sheets = sheet_titles ?? ['Sheet1'];
        const body = {
            properties: {
                title,
            },
            sheets: sheets.map((sheetTitle, index) => ({
                properties: {
                    title: sheetTitle,
                    index,
                    sheetType: 'GRID',
                    gridProperties: {
                        rowCount: 1000,
                        columnCount: 26,
                    },
                },
            })),
        };
        const response = await this.makeSheetsApiRequest('/spreadsheets', 'POST', body);
        return {
            operation: 'create_spreadsheet',
            success: true,
            spreadsheet: response,
            error: '',
        };
    }
    async addSheet(params) {
        const { spreadsheet_id, sheet_title, row_count, column_count } = params;
        const body = {
            requests: [
                {
                    addSheet: {
                        properties: {
                            title: sheet_title,
                            sheetType: 'GRID',
                            gridProperties: {
                                rowCount: row_count,
                                columnCount: column_count,
                            },
                        },
                    },
                },
            ],
        };
        const response = await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}:batchUpdate`, 'POST', body);
        const addSheetResponse = response.replies?.[0]?.addSheet;
        return {
            operation: 'add_sheet',
            success: true,
            sheet_id: addSheetResponse?.properties?.sheetId,
            sheet_title: addSheetResponse?.properties?.title,
            error: '',
        };
    }
    async deleteSheet(params) {
        const { spreadsheet_id, sheet_id } = params;
        const body = {
            requests: [
                {
                    deleteSheet: {
                        sheetId: sheet_id,
                    },
                },
            ],
        };
        await this.makeSheetsApiRequest(`/spreadsheets/${spreadsheet_id}:batchUpdate`, 'POST', body);
        return {
            operation: 'delete_sheet',
            success: true,
            deleted_sheet_id: sheet_id,
            error: '',
        };
    }
    chooseCredential() {
        const { credentials } = this.params;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('No Google Sheets credentials provided');
        }
        // Google Sheets bubble uses GOOGLE_SHEETS_CRED credentials
        return credentials[CredentialType.GOOGLE_SHEETS_CRED];
    }
}
//# sourceMappingURL=google-sheets.js.map