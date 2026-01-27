import { ServiceBubble } from '../../../types/service-bubble-class.js';
import type { BubbleContext } from '../../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { type GoogleSheetsParamsInput, type GoogleSheetsResult } from './google-sheets.schema.js';
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
export declare class GoogleSheetsBubble<T extends GoogleSheetsParamsInput = GoogleSheetsParamsInput> extends ServiceBubble<T, Extract<GoogleSheetsResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "google-sheets";
    static readonly authType: "oauth";
    static readonly bubbleName = "google-sheets";
    static readonly schema: import("zod").ZodDiscriminatedUnion<"operation", [import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"read_values">;
        spreadsheet_id: import("zod").ZodString;
        range: import("zod").ZodEffects<import("zod").ZodString, string, string>;
        major_dimension: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>>;
        value_render_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
        date_time_render_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["SERIAL_NUMBER", "FORMATTED_STRING"]>>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "read_values";
        spreadsheet_id: string;
        range: string;
        major_dimension: "ROWS" | "COLUMNS";
        value_render_option: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
        date_time_render_option: "SERIAL_NUMBER" | "FORMATTED_STRING";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "read_values";
        spreadsheet_id: string;
        range: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        major_dimension?: "ROWS" | "COLUMNS" | undefined;
        value_render_option?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
        date_time_render_option?: "SERIAL_NUMBER" | "FORMATTED_STRING" | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"write_values">;
        spreadsheet_id: import("zod").ZodString;
        range: import("zod").ZodEffects<import("zod").ZodString, string, string>;
        values: import("zod").ZodEffects<import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
        major_dimension: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>>;
        value_input_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["RAW", "USER_ENTERED"]>>>;
        include_values_in_response: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        values: (string | number | boolean)[][];
        operation: "write_values";
        spreadsheet_id: string;
        range: string;
        major_dimension: "ROWS" | "COLUMNS";
        value_input_option: "RAW" | "USER_ENTERED";
        include_values_in_response: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        values: unknown[][];
        operation: "write_values";
        spreadsheet_id: string;
        range: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        major_dimension?: "ROWS" | "COLUMNS" | undefined;
        value_input_option?: "RAW" | "USER_ENTERED" | undefined;
        include_values_in_response?: boolean | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"update_values">;
        spreadsheet_id: import("zod").ZodString;
        range: import("zod").ZodEffects<import("zod").ZodString, string, string>;
        values: import("zod").ZodEffects<import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
        major_dimension: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>>;
        value_input_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["RAW", "USER_ENTERED"]>>>;
        include_values_in_response: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        values: (string | number | boolean)[][];
        operation: "update_values";
        spreadsheet_id: string;
        range: string;
        major_dimension: "ROWS" | "COLUMNS";
        value_input_option: "RAW" | "USER_ENTERED";
        include_values_in_response: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        values: unknown[][];
        operation: "update_values";
        spreadsheet_id: string;
        range: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        major_dimension?: "ROWS" | "COLUMNS" | undefined;
        value_input_option?: "RAW" | "USER_ENTERED" | undefined;
        include_values_in_response?: boolean | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"append_values">;
        spreadsheet_id: import("zod").ZodString;
        range: import("zod").ZodEffects<import("zod").ZodString, string, string>;
        values: import("zod").ZodEffects<import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
        major_dimension: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>>;
        value_input_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["RAW", "USER_ENTERED"]>>>;
        insert_data_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["OVERWRITE", "INSERT_ROWS"]>>>;
        include_values_in_response: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        values: (string | number | boolean)[][];
        operation: "append_values";
        spreadsheet_id: string;
        range: string;
        major_dimension: "ROWS" | "COLUMNS";
        value_input_option: "RAW" | "USER_ENTERED";
        include_values_in_response: boolean;
        insert_data_option: "OVERWRITE" | "INSERT_ROWS";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        values: unknown[][];
        operation: "append_values";
        spreadsheet_id: string;
        range: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        major_dimension?: "ROWS" | "COLUMNS" | undefined;
        value_input_option?: "RAW" | "USER_ENTERED" | undefined;
        include_values_in_response?: boolean | undefined;
        insert_data_option?: "OVERWRITE" | "INSERT_ROWS" | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"clear_values">;
        spreadsheet_id: import("zod").ZodString;
        range: import("zod").ZodEffects<import("zod").ZodString, string, string>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "clear_values";
        spreadsheet_id: string;
        range: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "clear_values";
        spreadsheet_id: string;
        range: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"batch_read_values">;
        spreadsheet_id: import("zod").ZodString;
        ranges: import("zod").ZodEffects<import("zod").ZodArray<import("zod").ZodString, "many">, string[], string[]>;
        major_dimension: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>>;
        value_render_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
        date_time_render_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["SERIAL_NUMBER", "FORMATTED_STRING"]>>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "batch_read_values";
        spreadsheet_id: string;
        major_dimension: "ROWS" | "COLUMNS";
        value_render_option: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
        date_time_render_option: "SERIAL_NUMBER" | "FORMATTED_STRING";
        ranges: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "batch_read_values";
        spreadsheet_id: string;
        ranges: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        major_dimension?: "ROWS" | "COLUMNS" | undefined;
        value_render_option?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
        date_time_render_option?: "SERIAL_NUMBER" | "FORMATTED_STRING" | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"batch_update_values">;
        spreadsheet_id: import("zod").ZodString;
        value_ranges: import("zod").ZodArray<import("zod").ZodObject<{
            range: import("zod").ZodEffects<import("zod").ZodString, string, string>;
            values: import("zod").ZodEffects<import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
            major_dimension: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>>;
        }, "strip", import("zod").ZodTypeAny, {
            values: (string | number | boolean)[][];
            range: string;
            major_dimension: "ROWS" | "COLUMNS";
        }, {
            values: unknown[][];
            range: string;
            major_dimension?: "ROWS" | "COLUMNS" | undefined;
        }>, "many">;
        value_input_option: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["RAW", "USER_ENTERED"]>>>;
        include_values_in_response: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "batch_update_values";
        spreadsheet_id: string;
        value_input_option: "RAW" | "USER_ENTERED";
        include_values_in_response: boolean;
        value_ranges: {
            values: (string | number | boolean)[][];
            range: string;
            major_dimension: "ROWS" | "COLUMNS";
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "batch_update_values";
        spreadsheet_id: string;
        value_ranges: {
            values: unknown[][];
            range: string;
            major_dimension?: "ROWS" | "COLUMNS" | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        value_input_option?: "RAW" | "USER_ENTERED" | undefined;
        include_values_in_response?: boolean | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"get_spreadsheet_info">;
        spreadsheet_id: import("zod").ZodString;
        include_grid_data: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "get_spreadsheet_info";
        spreadsheet_id: string;
        include_grid_data: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_spreadsheet_info";
        spreadsheet_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        include_grid_data?: boolean | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"create_spreadsheet">;
        title: import("zod").ZodString;
        sheet_titles: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        title: string;
        operation: "create_spreadsheet";
        sheet_titles: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        title: string;
        operation: "create_spreadsheet";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        sheet_titles?: string[] | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"add_sheet">;
        spreadsheet_id: import("zod").ZodString;
        sheet_title: import("zod").ZodString;
        row_count: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodNumber>>;
        column_count: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodNumber>>;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "add_sheet";
        spreadsheet_id: string;
        sheet_title: string;
        row_count: number;
        column_count: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "add_sheet";
        spreadsheet_id: string;
        sheet_title: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        row_count?: number | undefined;
        column_count?: number | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"delete_sheet">;
        spreadsheet_id: import("zod").ZodString;
        sheet_id: import("zod").ZodNumber;
        credentials: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodNativeEnum<typeof CredentialType>, import("zod").ZodString>>;
    }, "strip", import("zod").ZodTypeAny, {
        operation: "delete_sheet";
        spreadsheet_id: string;
        sheet_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_sheet";
        spreadsheet_id: string;
        sheet_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: import("zod").ZodDiscriminatedUnion<"operation", [import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"read_values">;
        success: import("zod").ZodBoolean;
        range: import("zod").ZodOptional<import("zod").ZodString>;
        values: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnion<[import("zod").ZodString, import("zod").ZodNumber, import("zod").ZodBoolean]>, "many">, "many">>;
        major_dimension: import("zod").ZodOptional<import("zod").ZodString>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "read_values";
        values?: (string | number | boolean)[][] | undefined;
        range?: string | undefined;
        major_dimension?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "read_values";
        values?: (string | number | boolean)[][] | undefined;
        range?: string | undefined;
        major_dimension?: string | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"write_values">;
        success: import("zod").ZodBoolean;
        updated_range: import("zod").ZodOptional<import("zod").ZodString>;
        updated_rows: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_columns: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_cells: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_data: import("zod").ZodOptional<import("zod").ZodObject<{
            range: import("zod").ZodString;
            majorDimension: import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>;
            values: import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnion<[import("zod").ZodString, import("zod").ZodNumber, import("zod").ZodBoolean]>, "many">, "many">;
        }, "strip", import("zod").ZodTypeAny, {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }, {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }>>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "write_values";
        updated_range?: string | undefined;
        updated_rows?: number | undefined;
        updated_columns?: number | undefined;
        updated_cells?: number | undefined;
        updated_data?: {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "write_values";
        updated_range?: string | undefined;
        updated_rows?: number | undefined;
        updated_columns?: number | undefined;
        updated_cells?: number | undefined;
        updated_data?: {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        } | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"update_values">;
        success: import("zod").ZodBoolean;
        updated_range: import("zod").ZodOptional<import("zod").ZodString>;
        updated_rows: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_columns: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_cells: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_data: import("zod").ZodOptional<import("zod").ZodObject<{
            range: import("zod").ZodString;
            majorDimension: import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>;
            values: import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnion<[import("zod").ZodString, import("zod").ZodNumber, import("zod").ZodBoolean]>, "many">, "many">;
        }, "strip", import("zod").ZodTypeAny, {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }, {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }>>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_values";
        updated_range?: string | undefined;
        updated_rows?: number | undefined;
        updated_columns?: number | undefined;
        updated_cells?: number | undefined;
        updated_data?: {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_values";
        updated_range?: string | undefined;
        updated_rows?: number | undefined;
        updated_columns?: number | undefined;
        updated_cells?: number | undefined;
        updated_data?: {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        } | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"append_values">;
        success: import("zod").ZodBoolean;
        table_range: import("zod").ZodOptional<import("zod").ZodString>;
        updated_range: import("zod").ZodOptional<import("zod").ZodString>;
        updated_rows: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_columns: import("zod").ZodOptional<import("zod").ZodNumber>;
        updated_cells: import("zod").ZodOptional<import("zod").ZodNumber>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "append_values";
        updated_range?: string | undefined;
        updated_rows?: number | undefined;
        updated_columns?: number | undefined;
        updated_cells?: number | undefined;
        table_range?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "append_values";
        updated_range?: string | undefined;
        updated_rows?: number | undefined;
        updated_columns?: number | undefined;
        updated_cells?: number | undefined;
        table_range?: string | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"clear_values">;
        success: import("zod").ZodBoolean;
        cleared_range: import("zod").ZodOptional<import("zod").ZodString>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "clear_values";
        cleared_range?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "clear_values";
        cleared_range?: string | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"batch_read_values">;
        success: import("zod").ZodBoolean;
        value_ranges: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
            range: import("zod").ZodString;
            majorDimension: import("zod").ZodOptional<import("zod").ZodEnum<["ROWS", "COLUMNS"]>>;
            values: import("zod").ZodArray<import("zod").ZodArray<import("zod").ZodUnion<[import("zod").ZodString, import("zod").ZodNumber, import("zod").ZodBoolean]>, "many">, "many">;
        }, "strip", import("zod").ZodTypeAny, {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }, {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }>, "many">>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "batch_read_values";
        value_ranges?: {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "batch_read_values";
        value_ranges?: {
            values: (string | number | boolean)[][];
            range: string;
            majorDimension?: "ROWS" | "COLUMNS" | undefined;
        }[] | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"batch_update_values">;
        success: import("zod").ZodBoolean;
        total_updated_rows: import("zod").ZodOptional<import("zod").ZodNumber>;
        total_updated_columns: import("zod").ZodOptional<import("zod").ZodNumber>;
        total_updated_cells: import("zod").ZodOptional<import("zod").ZodNumber>;
        total_updated_sheets: import("zod").ZodOptional<import("zod").ZodNumber>;
        responses: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
            updated_range: import("zod").ZodOptional<import("zod").ZodString>;
            updated_rows: import("zod").ZodOptional<import("zod").ZodNumber>;
            updated_columns: import("zod").ZodOptional<import("zod").ZodNumber>;
            updated_cells: import("zod").ZodOptional<import("zod").ZodNumber>;
        }, "strip", import("zod").ZodTypeAny, {
            updated_range?: string | undefined;
            updated_rows?: number | undefined;
            updated_columns?: number | undefined;
            updated_cells?: number | undefined;
        }, {
            updated_range?: string | undefined;
            updated_rows?: number | undefined;
            updated_columns?: number | undefined;
            updated_cells?: number | undefined;
        }>, "many">>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "batch_update_values";
        total_updated_rows?: number | undefined;
        total_updated_columns?: number | undefined;
        total_updated_cells?: number | undefined;
        total_updated_sheets?: number | undefined;
        responses?: {
            updated_range?: string | undefined;
            updated_rows?: number | undefined;
            updated_columns?: number | undefined;
            updated_cells?: number | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "batch_update_values";
        total_updated_rows?: number | undefined;
        total_updated_columns?: number | undefined;
        total_updated_cells?: number | undefined;
        total_updated_sheets?: number | undefined;
        responses?: {
            updated_range?: string | undefined;
            updated_rows?: number | undefined;
            updated_columns?: number | undefined;
            updated_cells?: number | undefined;
        }[] | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"get_spreadsheet_info">;
        success: import("zod").ZodBoolean;
        spreadsheet: import("zod").ZodOptional<import("zod").ZodObject<{
            spreadsheetId: import("zod").ZodString;
            properties: import("zod").ZodOptional<import("zod").ZodObject<{
                title: import("zod").ZodString;
                locale: import("zod").ZodOptional<import("zod").ZodString>;
                autoRecalc: import("zod").ZodOptional<import("zod").ZodString>;
                timeZone: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            }, {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            }>>;
            sheets: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                properties: import("zod").ZodObject<{
                    sheetId: import("zod").ZodNumber;
                    title: import("zod").ZodString;
                    index: import("zod").ZodNumber;
                    sheetType: import("zod").ZodOptional<import("zod").ZodString>;
                    gridProperties: import("zod").ZodOptional<import("zod").ZodObject<{
                        rowCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                        columnCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                    }, "strip", import("zod").ZodTypeAny, {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    }, {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    }>>;
                }, "strip", import("zod").ZodTypeAny, {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                }, {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                }>;
            }, "strip", import("zod").ZodTypeAny, {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }, {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }>, "many">>;
            spreadsheetUrl: import("zod").ZodOptional<import("zod").ZodString>;
        }, "strip", import("zod").ZodTypeAny, {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        }, {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        }>>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_spreadsheet_info";
        spreadsheet?: {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_spreadsheet_info";
        spreadsheet?: {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        } | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"create_spreadsheet">;
        success: import("zod").ZodBoolean;
        spreadsheet: import("zod").ZodOptional<import("zod").ZodObject<{
            spreadsheetId: import("zod").ZodString;
            properties: import("zod").ZodOptional<import("zod").ZodObject<{
                title: import("zod").ZodString;
                locale: import("zod").ZodOptional<import("zod").ZodString>;
                autoRecalc: import("zod").ZodOptional<import("zod").ZodString>;
                timeZone: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            }, {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            }>>;
            sheets: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                properties: import("zod").ZodObject<{
                    sheetId: import("zod").ZodNumber;
                    title: import("zod").ZodString;
                    index: import("zod").ZodNumber;
                    sheetType: import("zod").ZodOptional<import("zod").ZodString>;
                    gridProperties: import("zod").ZodOptional<import("zod").ZodObject<{
                        rowCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                        columnCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                    }, "strip", import("zod").ZodTypeAny, {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    }, {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    }>>;
                }, "strip", import("zod").ZodTypeAny, {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                }, {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                }>;
            }, "strip", import("zod").ZodTypeAny, {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }, {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }>, "many">>;
            spreadsheetUrl: import("zod").ZodOptional<import("zod").ZodString>;
        }, "strip", import("zod").ZodTypeAny, {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        }, {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        }>>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_spreadsheet";
        spreadsheet?: {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_spreadsheet";
        spreadsheet?: {
            spreadsheetId: string;
            properties?: {
                title: string;
                locale?: string | undefined;
                autoRecalc?: string | undefined;
                timeZone?: string | undefined;
            } | undefined;
            sheets?: {
                properties: {
                    title: string;
                    sheetId: number;
                    index: number;
                    sheetType?: string | undefined;
                    gridProperties?: {
                        rowCount?: number | undefined;
                        columnCount?: number | undefined;
                    } | undefined;
                };
            }[] | undefined;
            spreadsheetUrl?: string | undefined;
        } | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"add_sheet">;
        success: import("zod").ZodBoolean;
        sheet_id: import("zod").ZodOptional<import("zod").ZodNumber>;
        sheet_title: import("zod").ZodOptional<import("zod").ZodString>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "add_sheet";
        sheet_title?: string | undefined;
        sheet_id?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "add_sheet";
        sheet_title?: string | undefined;
        sheet_id?: number | undefined;
    }>, import("zod").ZodObject<{
        operation: import("zod").ZodLiteral<"delete_sheet">;
        success: import("zod").ZodBoolean;
        deleted_sheet_id: import("zod").ZodOptional<import("zod").ZodNumber>;
        error: import("zod").ZodString;
    }, "strip", import("zod").ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_sheet";
        deleted_sheet_id?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_sheet";
        deleted_sheet_id?: number | undefined;
    }>]>;
    static readonly shortDescription = "Google Sheets integration for spreadsheet operations";
    static readonly longDescription = "\n    Google Sheets service integration for comprehensive spreadsheet data management.\n\n    Features:\n    - Automatic range normalization (sheet names with spaces are automatically quoted)\n    - Automatic value sanitization (null/undefined converted to empty strings)\n    - Enhanced error messages with helpful hints\n    - Support for all major Google Sheets operations\n\n    Use cases:\n    - Read and write spreadsheet data with flexible ranges\n    - Batch operations for efficient data processing\n    - Create and manage spreadsheets and sheets\n    - Clear and append data with various formatting options\n    - Handle formulas, formatted values, and raw data\n\n    Security Features:\n    - OAuth 2.0 authentication with Google\n    - Scoped access permissions for Google Sheets\n    - Secure data validation and sanitization\n    - User-controlled access to spreadsheet data\n  ";
    static readonly alias = "sheets";
    /**
     * Create a new Google Sheets Bubble instance
     * @param params - Operation parameters
     * @param context - Bubble execution context
     */
    constructor(params?: T, context?: BubbleContext);
    /**
     * Test the validity of the Google Sheets credentials
     * @returns Promise that resolves to true if credentials are valid, false otherwise
     * @throws AuthenticationError if credentials are missing
     */
    testCredential(): Promise<boolean>;
    private makeSheetsApiRequest;
    protected performAction(context?: BubbleContext): Promise<Extract<GoogleSheetsResult, {
        operation: T['operation'];
    }>>;
    private readValues;
    private writeValues;
    private updateValues;
    private appendValues;
    private clearValues;
    private batchReadValues;
    private batchUpdateValues;
    private getSpreadsheetInfo;
    private createSpreadsheet;
    private addSheet;
    private deleteSheet;
    protected chooseCredential(): string | undefined;
}
//# sourceMappingURL=google-sheets.d.ts.map