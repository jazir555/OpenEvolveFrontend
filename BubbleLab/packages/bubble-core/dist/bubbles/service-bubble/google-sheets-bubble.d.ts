import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const GoogleSheetsBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createSpreadsheet">;
    title: z.ZodString;
    sheets: z.ZodOptional<z.ZodArray<z.ZodObject<{
        title: z.ZodString;
        rowCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        columnCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        rowCount: number;
        columnCount: number;
    }, {
        title: string;
        rowCount?: number | undefined;
        columnCount?: number | undefined;
    }>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "createSpreadsheet";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    sheets?: {
        title: string;
        rowCount: number;
        columnCount: number;
    }[] | undefined;
}, {
    title: string;
    operation: "createSpreadsheet";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    sheets?: {
        title: string;
        rowCount?: number | undefined;
        columnCount?: number | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getSpreadsheet">;
    spreadsheetId: z.ZodString;
    includeGridData: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getSpreadsheet";
    spreadsheetId: string;
    includeGridData: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getSpreadsheet";
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    includeGridData?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteSpreadsheet">;
    spreadsheetId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteSpreadsheet";
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteSpreadsheet";
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"copySpreadsheet">;
    spreadsheetId: z.ZodString;
    title: z.ZodString;
    destinationFolderId: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "copySpreadsheet";
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    destinationFolderId?: string | undefined;
}, {
    title: string;
    operation: "copySpreadsheet";
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    destinationFolderId?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateCell">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    value: z.ZodAny;
    valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateCell";
    range: string;
    spreadsheetId: string;
    valueInputOption: "RAW" | "USER_ENTERED";
    value?: any;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "updateCell";
    range: string;
    spreadsheetId: string;
    value?: any;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getCellValue">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    valueRenderOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getCellValue";
    range: string;
    spreadsheetId: string;
    valueRenderOption: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getCellValue";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    valueRenderOption?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchUpdate">;
    spreadsheetId: z.ZodString;
    updates: z.ZodArray<z.ZodObject<{
        range: z.ZodString;
        values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        values: any[][];
        range: string;
    }, {
        values: any[][];
        range: string;
    }>, "many">;
    valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "batchUpdate";
    spreadsheetId: string;
    updates: {
        values: any[][];
        range: string;
    }[];
    valueInputOption: "RAW" | "USER_ENTERED";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "batchUpdate";
    spreadsheetId: string;
    updates: {
        values: any[][];
        range: string;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"appendRow">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    values: z.ZodArray<z.ZodAny, "many">;
    valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    insertDataOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["OVERWRITE", "INSERT_ROWS"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    values: any[];
    operation: "appendRow";
    range: string;
    spreadsheetId: string;
    valueInputOption: "RAW" | "USER_ENTERED";
    insertDataOption: "OVERWRITE" | "INSERT_ROWS";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    values: any[];
    operation: "appendRow";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
    insertDataOption?: "OVERWRITE" | "INSERT_ROWS" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRange">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    majorDimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["DIMENSIONS_UNSPECIFIED", "ROWS", "COLUMNS"]>>>;
    valueRenderOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getRange";
    range: string;
    majorDimension: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED";
    spreadsheetId: string;
    valueRenderOption: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getRange";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    majorDimension?: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED" | undefined;
    valueRenderOption?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"clearRange">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "clearRange";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "clearRange";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"copyRange">;
    spreadsheetId: z.ZodString;
    sourceRange: z.ZodString;
    destinationRange: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "copyRange";
    spreadsheetId: string;
    sourceRange: string;
    destinationRange: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "copyRange";
    spreadsheetId: string;
    sourceRange: string;
    destinationRange: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"addSheet">;
    spreadsheetId: z.ZodString;
    title: z.ZodString;
    rowCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    columnCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    rowCount: number;
    operation: "addSheet";
    spreadsheetId: string;
    columnCount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    title: string;
    operation: "addSheet";
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    rowCount?: number | undefined;
    columnCount?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteSheet">;
    spreadsheetId: z.ZodString;
    sheetId: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteSheet";
    spreadsheetId: string;
    sheetId: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteSheet";
    spreadsheetId: string;
    sheetId: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getSheetData">;
    spreadsheetId: z.ZodString;
    sheetName: z.ZodString;
    includeMetadata: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getSheetData";
    spreadsheetId: string;
    includeMetadata: boolean;
    sheetName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getSheetData";
    spreadsheetId: string;
    sheetName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    includeMetadata?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRow">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getRow";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getRow";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteRow">;
    spreadsheetId: z.ZodString;
    sheetId: z.ZodNumber;
    rowIndex: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteRow";
    spreadsheetId: string;
    sheetId: number;
    rowIndex: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteRow";
    spreadsheetId: string;
    sheetId: number;
    rowIndex: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getValues">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    majorDimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["DIMENSIONS_UNSPECIFIED", "ROWS", "COLUMNS"]>>>;
    valueRenderOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getValues";
    range: string;
    majorDimension: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED";
    spreadsheetId: string;
    valueRenderOption: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getValues";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    majorDimension?: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED" | undefined;
    valueRenderOption?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"setValues">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
    valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    values: any[][];
    operation: "setValues";
    range: string;
    spreadsheetId: string;
    valueInputOption: "RAW" | "USER_ENTERED";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    values: any[][];
    operation: "setValues";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"clearValues">;
    spreadsheetId: z.ZodString;
    range: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "clearValues";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "clearValues";
    range: string;
    spreadsheetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type GoogleSheetsBubbleParams = z.input<typeof GoogleSheetsBubbleParamsSchema>;
declare const GoogleSheetsBubbleResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createSpreadsheet">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        title: z.ZodString;
        url: z.ZodString;
        sheetCount: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        title: string;
        url: string;
        success: boolean;
        spreadsheetId: string;
        sheetCount: number;
    }, {
        error: string;
        title: string;
        url: string;
        success: boolean;
        spreadsheetId: string;
        sheetCount: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createSpreadsheet";
    result: {
        error: string;
        title: string;
        url: string;
        success: boolean;
        spreadsheetId: string;
        sheetCount: number;
    };
}, {
    operation: "createSpreadsheet";
    result: {
        error: string;
        title: string;
        url: string;
        success: boolean;
        spreadsheetId: string;
        sheetCount: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getSpreadsheet">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        title: z.ZodString;
        sheets: z.ZodArray<z.ZodObject<{
            sheetId: z.ZodNumber;
            title: z.ZodString;
            index: z.ZodNumber;
            sheetType: z.ZodString;
            gridProperties: z.ZodOptional<z.ZodObject<{
                rowCount: z.ZodNumber;
                columnCount: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                rowCount: number;
                columnCount: number;
            }, {
                rowCount: number;
                columnCount: number;
            }>>;
        }, "strip", z.ZodTypeAny, {
            title: string;
            sheetId: number;
            index: number;
            sheetType: string;
            gridProperties?: {
                rowCount: number;
                columnCount: number;
            } | undefined;
        }, {
            title: string;
            sheetId: number;
            index: number;
            sheetType: string;
            gridProperties?: {
                rowCount: number;
                columnCount: number;
            } | undefined;
        }>, "many">;
        namedRanges: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        title: string;
        success: boolean;
        spreadsheetId: string;
        sheets: {
            title: string;
            sheetId: number;
            index: number;
            sheetType: string;
            gridProperties?: {
                rowCount: number;
                columnCount: number;
            } | undefined;
        }[];
        namedRanges?: any[] | undefined;
    }, {
        error: string;
        title: string;
        success: boolean;
        spreadsheetId: string;
        sheets: {
            title: string;
            sheetId: number;
            index: number;
            sheetType: string;
            gridProperties?: {
                rowCount: number;
                columnCount: number;
            } | undefined;
        }[];
        namedRanges?: any[] | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getSpreadsheet";
    result: {
        error: string;
        title: string;
        success: boolean;
        spreadsheetId: string;
        sheets: {
            title: string;
            sheetId: number;
            index: number;
            sheetType: string;
            gridProperties?: {
                rowCount: number;
                columnCount: number;
            } | undefined;
        }[];
        namedRanges?: any[] | undefined;
    };
}, {
    operation: "getSpreadsheet";
    result: {
        error: string;
        title: string;
        success: boolean;
        spreadsheetId: string;
        sheets: {
            title: string;
            sheetId: number;
            index: number;
            sheetType: string;
            gridProperties?: {
                rowCount: number;
                columnCount: number;
            } | undefined;
        }[];
        namedRanges?: any[] | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteSpreadsheet">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        deleted: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        deleted: boolean;
        spreadsheetId: string;
    }, {
        error: string;
        success: boolean;
        deleted: boolean;
        spreadsheetId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteSpreadsheet";
    result: {
        error: string;
        success: boolean;
        deleted: boolean;
        spreadsheetId: string;
    };
}, {
    operation: "deleteSpreadsheet";
    result: {
        error: string;
        success: boolean;
        deleted: boolean;
        spreadsheetId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"copySpreadsheet">;
    result: z.ZodObject<{
        originalSpreadsheetId: z.ZodString;
        newSpreadsheetId: z.ZodString;
        title: z.ZodString;
        url: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        title: string;
        url: string;
        success: boolean;
        originalSpreadsheetId: string;
        newSpreadsheetId: string;
    }, {
        error: string;
        title: string;
        url: string;
        success: boolean;
        originalSpreadsheetId: string;
        newSpreadsheetId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "copySpreadsheet";
    result: {
        error: string;
        title: string;
        url: string;
        success: boolean;
        originalSpreadsheetId: string;
        newSpreadsheetId: string;
    };
}, {
    operation: "copySpreadsheet";
    result: {
        error: string;
        title: string;
        url: string;
        success: boolean;
        originalSpreadsheetId: string;
        newSpreadsheetId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateCell">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        updatedRange: z.ZodString;
        updatedRows: z.ZodNumber;
        updatedColumns: z.ZodNumber;
        updatedCells: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updateCell";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    };
}, {
    operation: "updateCell";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getCellValue">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        value: z.ZodAny;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        range: string;
        spreadsheetId: string;
        value?: any;
    }, {
        error: string;
        success: boolean;
        range: string;
        spreadsheetId: string;
        value?: any;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getCellValue";
    result: {
        error: string;
        success: boolean;
        range: string;
        spreadsheetId: string;
        value?: any;
    };
}, {
    operation: "getCellValue";
    result: {
        error: string;
        success: boolean;
        range: string;
        spreadsheetId: string;
        value?: any;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchUpdate">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        totalUpdatedRows: z.ZodNumber;
        totalUpdatedColumns: z.ZodNumber;
        totalUpdatedCells: z.ZodNumber;
        updateResults: z.ZodArray<z.ZodObject<{
            updatedRange: z.ZodString;
            updatedRows: z.ZodNumber;
            updatedColumns: z.ZodNumber;
            updatedCells: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }, {
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }>, "many">;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        totalUpdatedRows: number;
        totalUpdatedColumns: number;
        totalUpdatedCells: number;
        updateResults: {
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }[];
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        totalUpdatedRows: number;
        totalUpdatedColumns: number;
        totalUpdatedCells: number;
        updateResults: {
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchUpdate";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        totalUpdatedRows: number;
        totalUpdatedColumns: number;
        totalUpdatedCells: number;
        updateResults: {
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }[];
    };
}, {
    operation: "batchUpdate";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        totalUpdatedRows: number;
        totalUpdatedColumns: number;
        totalUpdatedCells: number;
        updateResults: {
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }[];
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"appendRow">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        tableRange: z.ZodString;
        updates: z.ZodOptional<z.ZodObject<{
            spreadsheetId: z.ZodString;
            updatedRange: z.ZodString;
            updatedRows: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
        }, {
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
        }>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        tableRange: string;
        updates?: {
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        tableRange: string;
        updates?: {
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
        } | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "appendRow";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        tableRange: string;
        updates?: {
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
        } | undefined;
    };
}, {
    operation: "appendRow";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        tableRange: string;
        updates?: {
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
        } | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRange">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
        majorDimension: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
        spreadsheetId: string;
    }, {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
        spreadsheetId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getRange";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
        spreadsheetId: string;
    };
}, {
    operation: "getRange";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
        spreadsheetId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"clearRange">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        clearedRange: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "clearRange";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    };
}, {
    operation: "clearRange";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"copyRange">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        sourceRange: z.ZodString;
        destinationRange: z.ZodString;
        updatedRange: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        sourceRange: string;
        destinationRange: string;
        updatedRange: string;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        sourceRange: string;
        destinationRange: string;
        updatedRange: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "copyRange";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        sourceRange: string;
        destinationRange: string;
        updatedRange: string;
    };
}, {
    operation: "copyRange";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        sourceRange: string;
        destinationRange: string;
        updatedRange: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"addSheet">;
    result: z.ZodObject<{
        sheetId: z.ZodNumber;
        title: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    }, {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "addSheet";
    result: {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    };
}, {
    operation: "addSheet";
    result: {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteSheet">;
    result: z.ZodObject<{
        sheetId: z.ZodNumber;
        title: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    }, {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteSheet";
    result: {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    };
}, {
    operation: "deleteSheet";
    result: {
        error: string;
        title: string;
        success: boolean;
        sheetId: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getSheetData">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        sheetName: z.ZodString;
        sheetId: z.ZodNumber;
        values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
        metadata: z.ZodOptional<z.ZodObject<{
            rowCount: z.ZodNumber;
            columnCount: z.ZodNumber;
            lastUpdated: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            rowCount: number;
            columnCount: number;
            lastUpdated?: string | undefined;
        }, {
            rowCount: number;
            columnCount: number;
            lastUpdated?: string | undefined;
        }>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        values: any[][];
        success: boolean;
        spreadsheetId: string;
        sheetId: number;
        sheetName: string;
        metadata?: {
            rowCount: number;
            columnCount: number;
            lastUpdated?: string | undefined;
        } | undefined;
    }, {
        error: string;
        values: any[][];
        success: boolean;
        spreadsheetId: string;
        sheetId: number;
        sheetName: string;
        metadata?: {
            rowCount: number;
            columnCount: number;
            lastUpdated?: string | undefined;
        } | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getSheetData";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        spreadsheetId: string;
        sheetId: number;
        sheetName: string;
        metadata?: {
            rowCount: number;
            columnCount: number;
            lastUpdated?: string | undefined;
        } | undefined;
    };
}, {
    operation: "getSheetData";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        spreadsheetId: string;
        sheetId: number;
        sheetName: string;
        metadata?: {
            rowCount: number;
            columnCount: number;
            lastUpdated?: string | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRow">;
    result: z.ZodObject<{
        range: z.ZodString;
        values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
    }, {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getRow";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
    };
}, {
    operation: "getRow";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteRow">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        deletedRows: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        deletedRows: number;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        deletedRows: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteRow";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        deletedRows: number;
    };
}, {
    operation: "deleteRow";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        deletedRows: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getValues">;
    result: z.ZodObject<{
        range: z.ZodString;
        values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
        majorDimension: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
    }, {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getValues";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
    };
}, {
    operation: "getValues";
    result: {
        error: string;
        values: any[][];
        success: boolean;
        range: string;
        majorDimension: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"setValues">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        updatedRange: z.ZodString;
        updatedRows: z.ZodNumber;
        updatedColumns: z.ZodNumber;
        updatedCells: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "setValues";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    };
}, {
    operation: "setValues";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        updatedRange: string;
        updatedRows: number;
        updatedColumns: number;
        updatedCells: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"clearValues">;
    result: z.ZodObject<{
        spreadsheetId: z.ZodString;
        clearedRange: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    }, {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "clearValues";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    };
}, {
    operation: "clearValues";
    result: {
        error: string;
        success: boolean;
        spreadsheetId: string;
        clearedRange: string;
    };
}>]>;
type GoogleSheetsBubbleResult = z.output<typeof GoogleSheetsBubbleResultSchema>;
export declare class GoogleSheetsBubble<T extends GoogleSheetsBubbleParams = GoogleSheetsBubbleParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "google-sheets";
    static readonly authType: "oauth";
    static readonly bubbleName = "google-sheets";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createSpreadsheet">;
        title: z.ZodString;
        sheets: z.ZodOptional<z.ZodArray<z.ZodObject<{
            title: z.ZodString;
            rowCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
            columnCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            title: string;
            rowCount: number;
            columnCount: number;
        }, {
            title: string;
            rowCount?: number | undefined;
            columnCount?: number | undefined;
        }>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        operation: "createSpreadsheet";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        sheets?: {
            title: string;
            rowCount: number;
            columnCount: number;
        }[] | undefined;
    }, {
        title: string;
        operation: "createSpreadsheet";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        sheets?: {
            title: string;
            rowCount?: number | undefined;
            columnCount?: number | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getSpreadsheet">;
        spreadsheetId: z.ZodString;
        includeGridData: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getSpreadsheet";
        spreadsheetId: string;
        includeGridData: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getSpreadsheet";
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        includeGridData?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteSpreadsheet">;
        spreadsheetId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteSpreadsheet";
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteSpreadsheet";
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"copySpreadsheet">;
        spreadsheetId: z.ZodString;
        title: z.ZodString;
        destinationFolderId: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        operation: "copySpreadsheet";
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        destinationFolderId?: string | undefined;
    }, {
        title: string;
        operation: "copySpreadsheet";
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        destinationFolderId?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateCell">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        value: z.ZodAny;
        valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateCell";
        range: string;
        spreadsheetId: string;
        valueInputOption: "RAW" | "USER_ENTERED";
        value?: any;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "updateCell";
        range: string;
        spreadsheetId: string;
        value?: any;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getCellValue">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        valueRenderOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getCellValue";
        range: string;
        spreadsheetId: string;
        valueRenderOption: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getCellValue";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        valueRenderOption?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchUpdate">;
        spreadsheetId: z.ZodString;
        updates: z.ZodArray<z.ZodObject<{
            range: z.ZodString;
            values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
        }, "strip", z.ZodTypeAny, {
            values: any[][];
            range: string;
        }, {
            values: any[][];
            range: string;
        }>, "many">;
        valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchUpdate";
        spreadsheetId: string;
        updates: {
            values: any[][];
            range: string;
        }[];
        valueInputOption: "RAW" | "USER_ENTERED";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "batchUpdate";
        spreadsheetId: string;
        updates: {
            values: any[][];
            range: string;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"appendRow">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        values: z.ZodArray<z.ZodAny, "many">;
        valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
        insertDataOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["OVERWRITE", "INSERT_ROWS"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        values: any[];
        operation: "appendRow";
        range: string;
        spreadsheetId: string;
        valueInputOption: "RAW" | "USER_ENTERED";
        insertDataOption: "OVERWRITE" | "INSERT_ROWS";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        values: any[];
        operation: "appendRow";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
        insertDataOption?: "OVERWRITE" | "INSERT_ROWS" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRange">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        majorDimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["DIMENSIONS_UNSPECIFIED", "ROWS", "COLUMNS"]>>>;
        valueRenderOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRange";
        range: string;
        majorDimension: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED";
        spreadsheetId: string;
        valueRenderOption: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getRange";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        majorDimension?: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED" | undefined;
        valueRenderOption?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"clearRange">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "clearRange";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "clearRange";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"copyRange">;
        spreadsheetId: z.ZodString;
        sourceRange: z.ZodString;
        destinationRange: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "copyRange";
        spreadsheetId: string;
        sourceRange: string;
        destinationRange: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "copyRange";
        spreadsheetId: string;
        sourceRange: string;
        destinationRange: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"addSheet">;
        spreadsheetId: z.ZodString;
        title: z.ZodString;
        rowCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        columnCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        rowCount: number;
        operation: "addSheet";
        spreadsheetId: string;
        columnCount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        title: string;
        operation: "addSheet";
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        rowCount?: number | undefined;
        columnCount?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteSheet">;
        spreadsheetId: z.ZodString;
        sheetId: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteSheet";
        spreadsheetId: string;
        sheetId: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteSheet";
        spreadsheetId: string;
        sheetId: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getSheetData">;
        spreadsheetId: z.ZodString;
        sheetName: z.ZodString;
        includeMetadata: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getSheetData";
        spreadsheetId: string;
        includeMetadata: boolean;
        sheetName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getSheetData";
        spreadsheetId: string;
        sheetName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        includeMetadata?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRow">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRow";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getRow";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteRow">;
        spreadsheetId: z.ZodString;
        sheetId: z.ZodNumber;
        rowIndex: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteRow";
        spreadsheetId: string;
        sheetId: number;
        rowIndex: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteRow";
        spreadsheetId: string;
        sheetId: number;
        rowIndex: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getValues">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        majorDimension: z.ZodDefault<z.ZodOptional<z.ZodEnum<["DIMENSIONS_UNSPECIFIED", "ROWS", "COLUMNS"]>>>;
        valueRenderOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getValues";
        range: string;
        majorDimension: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED";
        spreadsheetId: string;
        valueRenderOption: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getValues";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        majorDimension?: "ROWS" | "COLUMNS" | "DIMENSIONS_UNSPECIFIED" | undefined;
        valueRenderOption?: "FORMATTED_VALUE" | "UNFORMATTED_VALUE" | "FORMULA" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"setValues">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
        valueInputOption: z.ZodDefault<z.ZodOptional<z.ZodEnum<["RAW", "USER_ENTERED"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        values: any[][];
        operation: "setValues";
        range: string;
        spreadsheetId: string;
        valueInputOption: "RAW" | "USER_ENTERED";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        values: any[][];
        operation: "setValues";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        valueInputOption?: "RAW" | "USER_ENTERED" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"clearValues">;
        spreadsheetId: z.ZodString;
        range: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "clearValues";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "clearValues";
        range: string;
        spreadsheetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createSpreadsheet">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            title: z.ZodString;
            url: z.ZodString;
            sheetCount: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            title: string;
            url: string;
            success: boolean;
            spreadsheetId: string;
            sheetCount: number;
        }, {
            error: string;
            title: string;
            url: string;
            success: boolean;
            spreadsheetId: string;
            sheetCount: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createSpreadsheet";
        result: {
            error: string;
            title: string;
            url: string;
            success: boolean;
            spreadsheetId: string;
            sheetCount: number;
        };
    }, {
        operation: "createSpreadsheet";
        result: {
            error: string;
            title: string;
            url: string;
            success: boolean;
            spreadsheetId: string;
            sheetCount: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getSpreadsheet">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            title: z.ZodString;
            sheets: z.ZodArray<z.ZodObject<{
                sheetId: z.ZodNumber;
                title: z.ZodString;
                index: z.ZodNumber;
                sheetType: z.ZodString;
                gridProperties: z.ZodOptional<z.ZodObject<{
                    rowCount: z.ZodNumber;
                    columnCount: z.ZodNumber;
                }, "strip", z.ZodTypeAny, {
                    rowCount: number;
                    columnCount: number;
                }, {
                    rowCount: number;
                    columnCount: number;
                }>>;
            }, "strip", z.ZodTypeAny, {
                title: string;
                sheetId: number;
                index: number;
                sheetType: string;
                gridProperties?: {
                    rowCount: number;
                    columnCount: number;
                } | undefined;
            }, {
                title: string;
                sheetId: number;
                index: number;
                sheetType: string;
                gridProperties?: {
                    rowCount: number;
                    columnCount: number;
                } | undefined;
            }>, "many">;
            namedRanges: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            title: string;
            success: boolean;
            spreadsheetId: string;
            sheets: {
                title: string;
                sheetId: number;
                index: number;
                sheetType: string;
                gridProperties?: {
                    rowCount: number;
                    columnCount: number;
                } | undefined;
            }[];
            namedRanges?: any[] | undefined;
        }, {
            error: string;
            title: string;
            success: boolean;
            spreadsheetId: string;
            sheets: {
                title: string;
                sheetId: number;
                index: number;
                sheetType: string;
                gridProperties?: {
                    rowCount: number;
                    columnCount: number;
                } | undefined;
            }[];
            namedRanges?: any[] | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getSpreadsheet";
        result: {
            error: string;
            title: string;
            success: boolean;
            spreadsheetId: string;
            sheets: {
                title: string;
                sheetId: number;
                index: number;
                sheetType: string;
                gridProperties?: {
                    rowCount: number;
                    columnCount: number;
                } | undefined;
            }[];
            namedRanges?: any[] | undefined;
        };
    }, {
        operation: "getSpreadsheet";
        result: {
            error: string;
            title: string;
            success: boolean;
            spreadsheetId: string;
            sheets: {
                title: string;
                sheetId: number;
                index: number;
                sheetType: string;
                gridProperties?: {
                    rowCount: number;
                    columnCount: number;
                } | undefined;
            }[];
            namedRanges?: any[] | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteSpreadsheet">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            deleted: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            deleted: boolean;
            spreadsheetId: string;
        }, {
            error: string;
            success: boolean;
            deleted: boolean;
            spreadsheetId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteSpreadsheet";
        result: {
            error: string;
            success: boolean;
            deleted: boolean;
            spreadsheetId: string;
        };
    }, {
        operation: "deleteSpreadsheet";
        result: {
            error: string;
            success: boolean;
            deleted: boolean;
            spreadsheetId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"copySpreadsheet">;
        result: z.ZodObject<{
            originalSpreadsheetId: z.ZodString;
            newSpreadsheetId: z.ZodString;
            title: z.ZodString;
            url: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            title: string;
            url: string;
            success: boolean;
            originalSpreadsheetId: string;
            newSpreadsheetId: string;
        }, {
            error: string;
            title: string;
            url: string;
            success: boolean;
            originalSpreadsheetId: string;
            newSpreadsheetId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "copySpreadsheet";
        result: {
            error: string;
            title: string;
            url: string;
            success: boolean;
            originalSpreadsheetId: string;
            newSpreadsheetId: string;
        };
    }, {
        operation: "copySpreadsheet";
        result: {
            error: string;
            title: string;
            url: string;
            success: boolean;
            originalSpreadsheetId: string;
            newSpreadsheetId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateCell">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            updatedRange: z.ZodString;
            updatedRows: z.ZodNumber;
            updatedColumns: z.ZodNumber;
            updatedCells: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateCell";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        };
    }, {
        operation: "updateCell";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getCellValue">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            range: z.ZodString;
            value: z.ZodAny;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            range: string;
            spreadsheetId: string;
            value?: any;
        }, {
            error: string;
            success: boolean;
            range: string;
            spreadsheetId: string;
            value?: any;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getCellValue";
        result: {
            error: string;
            success: boolean;
            range: string;
            spreadsheetId: string;
            value?: any;
        };
    }, {
        operation: "getCellValue";
        result: {
            error: string;
            success: boolean;
            range: string;
            spreadsheetId: string;
            value?: any;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchUpdate">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            totalUpdatedRows: z.ZodNumber;
            totalUpdatedColumns: z.ZodNumber;
            totalUpdatedCells: z.ZodNumber;
            updateResults: z.ZodArray<z.ZodObject<{
                updatedRange: z.ZodString;
                updatedRows: z.ZodNumber;
                updatedColumns: z.ZodNumber;
                updatedCells: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                updatedRange: string;
                updatedRows: number;
                updatedColumns: number;
                updatedCells: number;
            }, {
                updatedRange: string;
                updatedRows: number;
                updatedColumns: number;
                updatedCells: number;
            }>, "many">;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            totalUpdatedRows: number;
            totalUpdatedColumns: number;
            totalUpdatedCells: number;
            updateResults: {
                updatedRange: string;
                updatedRows: number;
                updatedColumns: number;
                updatedCells: number;
            }[];
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            totalUpdatedRows: number;
            totalUpdatedColumns: number;
            totalUpdatedCells: number;
            updateResults: {
                updatedRange: string;
                updatedRows: number;
                updatedColumns: number;
                updatedCells: number;
            }[];
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchUpdate";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            totalUpdatedRows: number;
            totalUpdatedColumns: number;
            totalUpdatedCells: number;
            updateResults: {
                updatedRange: string;
                updatedRows: number;
                updatedColumns: number;
                updatedCells: number;
            }[];
        };
    }, {
        operation: "batchUpdate";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            totalUpdatedRows: number;
            totalUpdatedColumns: number;
            totalUpdatedCells: number;
            updateResults: {
                updatedRange: string;
                updatedRows: number;
                updatedColumns: number;
                updatedCells: number;
            }[];
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"appendRow">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            tableRange: z.ZodString;
            updates: z.ZodOptional<z.ZodObject<{
                spreadsheetId: z.ZodString;
                updatedRange: z.ZodString;
                updatedRows: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                spreadsheetId: string;
                updatedRange: string;
                updatedRows: number;
            }, {
                spreadsheetId: string;
                updatedRange: string;
                updatedRows: number;
            }>>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            tableRange: string;
            updates?: {
                spreadsheetId: string;
                updatedRange: string;
                updatedRows: number;
            } | undefined;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            tableRange: string;
            updates?: {
                spreadsheetId: string;
                updatedRange: string;
                updatedRows: number;
            } | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "appendRow";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            tableRange: string;
            updates?: {
                spreadsheetId: string;
                updatedRange: string;
                updatedRows: number;
            } | undefined;
        };
    }, {
        operation: "appendRow";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            tableRange: string;
            updates?: {
                spreadsheetId: string;
                updatedRange: string;
                updatedRows: number;
            } | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRange">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            range: z.ZodString;
            values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
            majorDimension: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
            spreadsheetId: string;
        }, {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
            spreadsheetId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRange";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
            spreadsheetId: string;
        };
    }, {
        operation: "getRange";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
            spreadsheetId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"clearRange">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            clearedRange: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "clearRange";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        };
    }, {
        operation: "clearRange";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"copyRange">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            sourceRange: z.ZodString;
            destinationRange: z.ZodString;
            updatedRange: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            sourceRange: string;
            destinationRange: string;
            updatedRange: string;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            sourceRange: string;
            destinationRange: string;
            updatedRange: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "copyRange";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            sourceRange: string;
            destinationRange: string;
            updatedRange: string;
        };
    }, {
        operation: "copyRange";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            sourceRange: string;
            destinationRange: string;
            updatedRange: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"addSheet">;
        result: z.ZodObject<{
            sheetId: z.ZodNumber;
            title: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        }, {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "addSheet";
        result: {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        };
    }, {
        operation: "addSheet";
        result: {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteSheet">;
        result: z.ZodObject<{
            sheetId: z.ZodNumber;
            title: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        }, {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteSheet";
        result: {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        };
    }, {
        operation: "deleteSheet";
        result: {
            error: string;
            title: string;
            success: boolean;
            sheetId: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getSheetData">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            sheetName: z.ZodString;
            sheetId: z.ZodNumber;
            values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
            metadata: z.ZodOptional<z.ZodObject<{
                rowCount: z.ZodNumber;
                columnCount: z.ZodNumber;
                lastUpdated: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                rowCount: number;
                columnCount: number;
                lastUpdated?: string | undefined;
            }, {
                rowCount: number;
                columnCount: number;
                lastUpdated?: string | undefined;
            }>>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            values: any[][];
            success: boolean;
            spreadsheetId: string;
            sheetId: number;
            sheetName: string;
            metadata?: {
                rowCount: number;
                columnCount: number;
                lastUpdated?: string | undefined;
            } | undefined;
        }, {
            error: string;
            values: any[][];
            success: boolean;
            spreadsheetId: string;
            sheetId: number;
            sheetName: string;
            metadata?: {
                rowCount: number;
                columnCount: number;
                lastUpdated?: string | undefined;
            } | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getSheetData";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            spreadsheetId: string;
            sheetId: number;
            sheetName: string;
            metadata?: {
                rowCount: number;
                columnCount: number;
                lastUpdated?: string | undefined;
            } | undefined;
        };
    }, {
        operation: "getSheetData";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            spreadsheetId: string;
            sheetId: number;
            sheetName: string;
            metadata?: {
                rowCount: number;
                columnCount: number;
                lastUpdated?: string | undefined;
            } | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRow">;
        result: z.ZodObject<{
            range: z.ZodString;
            values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
        }, {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRow";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
        };
    }, {
        operation: "getRow";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteRow">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            deletedRows: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            deletedRows: number;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            deletedRows: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteRow";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            deletedRows: number;
        };
    }, {
        operation: "deleteRow";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            deletedRows: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getValues">;
        result: z.ZodObject<{
            range: z.ZodString;
            values: z.ZodArray<z.ZodArray<z.ZodAny, "many">, "many">;
            majorDimension: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
        }, {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getValues";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
        };
    }, {
        operation: "getValues";
        result: {
            error: string;
            values: any[][];
            success: boolean;
            range: string;
            majorDimension: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"setValues">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            updatedRange: z.ZodString;
            updatedRows: z.ZodNumber;
            updatedColumns: z.ZodNumber;
            updatedCells: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "setValues";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        };
    }, {
        operation: "setValues";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            updatedRange: string;
            updatedRows: number;
            updatedColumns: number;
            updatedCells: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"clearValues">;
        result: z.ZodObject<{
            spreadsheetId: z.ZodString;
            clearedRange: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        }, {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "clearValues";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        };
    }, {
        operation: "clearValues";
        result: {
            error: string;
            success: boolean;
            spreadsheetId: string;
            clearedRange: string;
        };
    }>]>;
    static readonly shortDescription = "Complete Google Sheets integration for spreadsheet operations";
    static readonly longDescription = "\n    Comprehensive Google Sheets service bubble for all spreadsheet operations.\n\n    Operations:\n    1. createSpreadsheet - Create a new spreadsheet with custom sheets\n    2. getSpreadsheet - Get spreadsheet metadata and structure\n    3. deleteSpreadsheet - Delete a spreadsheet permanently\n    4. copySpreadsheet - Copy a spreadsheet to new location\n    5. updateCell - Update a single cell\n    6. getCellValue - Get a single cell value\n    7. batchUpdate - Update multiple cells efficiently\n    8. appendRow - Append data to the end of a sheet\n    9. getRange - Get values from a range\n    10. clearRange - Clear all values from a range\n    11. copyRange - Copy range to destination\n    12. addSheet - Add a new sheet to the spreadsheet\n    13. deleteSheet - Remove a sheet from the spreadsheet\n    14. getSheetData - Get complete sheet data with metadata\n\n    Features:\n    - OAuth 2.0 authentication with token validation\n    - Full CRUD operations on spreadsheets and sheets\n    - Batch updates for efficiency\n    - Row and column operations\n    - Sheet management\n    - Value formatting options (RAW, USER_ENTERED)\n    - Range operations with A1 notation support\n    - Rate limiting and quota management\n    - Resilience patterns (circuit breaker, retry, deduplication)\n    - Structured logging and error handling\n    - Input validation and sanitization\n\n    Use Cases:\n    - Automated reporting and data collection\n    - Data synchronization between systems\n    - Spreadsheet-based workflows\n    - Data analysis and visualization\n    - Template generation and management\n    - Batch data processing\n  ";
    static readonly alias = "sheets";
    private client;
    private resilience;
    constructor(params: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<GoogleSheetsBubbleResult, {
        operation: T['operation'];
    }>>;
    private createSpreadsheet;
    private getSheet;
    private updateCell;
    private batchUpdate;
    private appendRow;
    private getRow;
    private deleteRow;
    private addSheet;
    private deleteSheet;
    private getValues;
    private setValues;
    private clearValues;
    private getSpreadsheet;
    private deleteSpreadsheet;
    private copySpreadsheet;
    private getCellValue;
    private getRange;
    private clearRange;
    private copyRange;
    private getSheetData;
    private errorResult;
}
export {};
//# sourceMappingURL=google-sheets-bubble.d.ts.map