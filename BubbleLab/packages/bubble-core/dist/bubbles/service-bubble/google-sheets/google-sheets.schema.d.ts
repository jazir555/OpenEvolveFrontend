import { z } from 'zod';
import { CredentialType } from '@bubblelab/shared-schemas';
export declare const ValueRangeSchema: z.ZodObject<{
    range: z.ZodString;
    majorDimension: z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>;
    values: z.ZodArray<z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>, "many">, "many">;
}, "strip", z.ZodTypeAny, {
    values: (string | number | boolean)[][];
    range: string;
    majorDimension?: "ROWS" | "COLUMNS" | undefined;
}, {
    values: (string | number | boolean)[][];
    range: string;
    majorDimension?: "ROWS" | "COLUMNS" | undefined;
}>;
export declare const SpreadsheetInfoSchema: z.ZodObject<{
    spreadsheetId: z.ZodString;
    properties: z.ZodOptional<z.ZodObject<{
        title: z.ZodString;
        locale: z.ZodOptional<z.ZodString>;
        autoRecalc: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
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
    sheets: z.ZodOptional<z.ZodArray<z.ZodObject<{
        properties: z.ZodObject<{
            sheetId: z.ZodNumber;
            title: z.ZodString;
            index: z.ZodNumber;
            sheetType: z.ZodOptional<z.ZodString>;
            gridProperties: z.ZodOptional<z.ZodObject<{
                rowCount: z.ZodOptional<z.ZodNumber>;
                columnCount: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                rowCount?: number | undefined;
                columnCount?: number | undefined;
            }, {
                rowCount?: number | undefined;
                columnCount?: number | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
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
    }, "strip", z.ZodTypeAny, {
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
    spreadsheetUrl: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
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
}>;
export declare const GoogleSheetsParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"read_values">;
    spreadsheet_id: z.ZodString;
    range: z.ZodEffects<z.ZodString, string, string>;
    major_dimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>>;
    value_render_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
    date_time_render_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["SERIAL_NUMBER", "FORMATTED_STRING"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"write_values">;
    spreadsheet_id: z.ZodString;
    range: z.ZodEffects<z.ZodString, string, string>;
    values: z.ZodEffects<z.ZodArray<z.ZodArray<z.ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
    major_dimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>>;
    value_input_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    include_values_in_response: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_values">;
    spreadsheet_id: z.ZodString;
    range: z.ZodEffects<z.ZodString, string, string>;
    values: z.ZodEffects<z.ZodArray<z.ZodArray<z.ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
    major_dimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>>;
    value_input_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    include_values_in_response: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"append_values">;
    spreadsheet_id: z.ZodString;
    range: z.ZodEffects<z.ZodString, string, string>;
    values: z.ZodEffects<z.ZodArray<z.ZodArray<z.ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
    major_dimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>>;
    value_input_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    insert_data_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["OVERWRITE", "INSERT_ROWS"]>>>;
    include_values_in_response: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"clear_values">;
    spreadsheet_id: z.ZodString;
    range: z.ZodEffects<z.ZodString, string, string>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "clear_values";
    spreadsheet_id: string;
    range: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "clear_values";
    spreadsheet_id: string;
    range: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batch_read_values">;
    spreadsheet_id: z.ZodString;
    ranges: z.ZodEffects<z.ZodArray<z.ZodString, "many">, string[], string[]>;
    major_dimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>>;
    value_render_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
    date_time_render_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["SERIAL_NUMBER", "FORMATTED_STRING"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batch_update_values">;
    spreadsheet_id: z.ZodString;
    value_ranges: z.ZodArray<z.ZodObject<{
        range: z.ZodEffects<z.ZodString, string, string>;
        values: z.ZodEffects<z.ZodArray<z.ZodArray<z.ZodUnknown, "many">, "many">, (string | number | boolean)[][], unknown[][]>;
        major_dimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>>;
    }, "strip", z.ZodTypeAny, {
        values: (string | number | boolean)[][];
        range: string;
        major_dimension: "ROWS" | "COLUMNS";
    }, {
        values: unknown[][];
        range: string;
        major_dimension?: "ROWS" | "COLUMNS" | undefined;
    }>, "many">;
    value_input_option: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    include_values_in_response: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_spreadsheet_info">;
    spreadsheet_id: z.ZodString;
    include_grid_data: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_spreadsheet_info";
    spreadsheet_id: string;
    include_grid_data: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_spreadsheet_info";
    spreadsheet_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    include_grid_data?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_spreadsheet">;
    title: z.ZodString;
    sheet_titles: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodString, "many">>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "create_spreadsheet";
    sheet_titles: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    title: string;
    operation: "create_spreadsheet";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    sheet_titles?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"add_sheet">;
    spreadsheet_id: z.ZodString;
    sheet_title: z.ZodString;
    row_count: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    column_count: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_sheet">;
    spreadsheet_id: z.ZodString;
    sheet_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
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
export declare const GoogleSheetsResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"read_values">;
    success: z.ZodBoolean;
    range: z.ZodOptional<z.ZodString>;
    values: z.ZodOptional<z.ZodArray<z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>, "many">, "many">>;
    major_dimension: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"write_values">;
    success: z.ZodBoolean;
    updated_range: z.ZodOptional<z.ZodString>;
    updated_rows: z.ZodOptional<z.ZodNumber>;
    updated_columns: z.ZodOptional<z.ZodNumber>;
    updated_cells: z.ZodOptional<z.ZodNumber>;
    updated_data: z.ZodOptional<z.ZodObject<{
        range: z.ZodString;
        majorDimension: z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>;
        values: z.ZodArray<z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        values: (string | number | boolean)[][];
        range: string;
        majorDimension?: "ROWS" | "COLUMNS" | undefined;
    }, {
        values: (string | number | boolean)[][];
        range: string;
        majorDimension?: "ROWS" | "COLUMNS" | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_values">;
    success: z.ZodBoolean;
    updated_range: z.ZodOptional<z.ZodString>;
    updated_rows: z.ZodOptional<z.ZodNumber>;
    updated_columns: z.ZodOptional<z.ZodNumber>;
    updated_cells: z.ZodOptional<z.ZodNumber>;
    updated_data: z.ZodOptional<z.ZodObject<{
        range: z.ZodString;
        majorDimension: z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>;
        values: z.ZodArray<z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        values: (string | number | boolean)[][];
        range: string;
        majorDimension?: "ROWS" | "COLUMNS" | undefined;
    }, {
        values: (string | number | boolean)[][];
        range: string;
        majorDimension?: "ROWS" | "COLUMNS" | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"append_values">;
    success: z.ZodBoolean;
    table_range: z.ZodOptional<z.ZodString>;
    updated_range: z.ZodOptional<z.ZodString>;
    updated_rows: z.ZodOptional<z.ZodNumber>;
    updated_columns: z.ZodOptional<z.ZodNumber>;
    updated_cells: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"clear_values">;
    success: z.ZodBoolean;
    cleared_range: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "clear_values";
    cleared_range?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "clear_values";
    cleared_range?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batch_read_values">;
    success: z.ZodBoolean;
    value_ranges: z.ZodOptional<z.ZodArray<z.ZodObject<{
        range: z.ZodString;
        majorDimension: z.ZodOptional<z.ZodEnum<["ROWS", "COLUMNS"]>>;
        values: z.ZodArray<z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean]>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        values: (string | number | boolean)[][];
        range: string;
        majorDimension?: "ROWS" | "COLUMNS" | undefined;
    }, {
        values: (string | number | boolean)[][];
        range: string;
        majorDimension?: "ROWS" | "COLUMNS" | undefined;
    }>, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batch_update_values">;
    success: z.ZodBoolean;
    total_updated_rows: z.ZodOptional<z.ZodNumber>;
    total_updated_columns: z.ZodOptional<z.ZodNumber>;
    total_updated_cells: z.ZodOptional<z.ZodNumber>;
    total_updated_sheets: z.ZodOptional<z.ZodNumber>;
    responses: z.ZodOptional<z.ZodArray<z.ZodObject<{
        updated_range: z.ZodOptional<z.ZodString>;
        updated_rows: z.ZodOptional<z.ZodNumber>;
        updated_columns: z.ZodOptional<z.ZodNumber>;
        updated_cells: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
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
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_spreadsheet_info">;
    success: z.ZodBoolean;
    spreadsheet: z.ZodOptional<z.ZodObject<{
        spreadsheetId: z.ZodString;
        properties: z.ZodOptional<z.ZodObject<{
            title: z.ZodString;
            locale: z.ZodOptional<z.ZodString>;
            autoRecalc: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
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
        sheets: z.ZodOptional<z.ZodArray<z.ZodObject<{
            properties: z.ZodObject<{
                sheetId: z.ZodNumber;
                title: z.ZodString;
                index: z.ZodNumber;
                sheetType: z.ZodOptional<z.ZodString>;
                gridProperties: z.ZodOptional<z.ZodObject<{
                    rowCount: z.ZodOptional<z.ZodNumber>;
                    columnCount: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    rowCount?: number | undefined;
                    columnCount?: number | undefined;
                }, {
                    rowCount?: number | undefined;
                    columnCount?: number | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
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
        }, "strip", z.ZodTypeAny, {
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
        spreadsheetUrl: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
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
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_spreadsheet">;
    success: z.ZodBoolean;
    spreadsheet: z.ZodOptional<z.ZodObject<{
        spreadsheetId: z.ZodString;
        properties: z.ZodOptional<z.ZodObject<{
            title: z.ZodString;
            locale: z.ZodOptional<z.ZodString>;
            autoRecalc: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
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
        sheets: z.ZodOptional<z.ZodArray<z.ZodObject<{
            properties: z.ZodObject<{
                sheetId: z.ZodNumber;
                title: z.ZodString;
                index: z.ZodNumber;
                sheetType: z.ZodOptional<z.ZodString>;
                gridProperties: z.ZodOptional<z.ZodObject<{
                    rowCount: z.ZodOptional<z.ZodNumber>;
                    columnCount: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    rowCount?: number | undefined;
                    columnCount?: number | undefined;
                }, {
                    rowCount?: number | undefined;
                    columnCount?: number | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
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
        }, "strip", z.ZodTypeAny, {
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
        spreadsheetUrl: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
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
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"add_sheet">;
    success: z.ZodBoolean;
    sheet_id: z.ZodOptional<z.ZodNumber>;
    sheet_title: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_sheet">;
    success: z.ZodBoolean;
    deleted_sheet_id: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
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
export type GoogleSheetsResult = z.output<typeof GoogleSheetsResultSchema>;
export type GoogleSheetsParams = z.output<typeof GoogleSheetsParamsSchema>;
export type GoogleSheetsParamsInput = z.input<typeof GoogleSheetsParamsSchema>;
//# sourceMappingURL=google-sheets.schema.d.ts.map