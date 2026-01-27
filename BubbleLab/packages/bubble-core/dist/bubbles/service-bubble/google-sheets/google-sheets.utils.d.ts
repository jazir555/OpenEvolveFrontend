/**
 * Utility functions for Google Sheets bubble
 * Handles range normalization, value sanitization, and error handling
 */
/**
 * Normalizes A1 notation range to ensure sheet names with spaces/special chars are quoted
 *
 * Examples:
 *   "Sheet1!A1:B10" -> "Sheet1!A1:B10"
 *   "Kaus Mode Landing zone!A:G" -> "'Kaus Mode Landing zone'!A:G"
 *   "'Sheet Name'!A1" -> "'Sheet Name'!A1" (already quoted, no change)
 *   "Sheet1!A1" -> "Sheet1!A1" (no sheet name change needed)
 *
 * @param range - A1 notation range string
 * @returns Normalized range with proper quoting
 */
export declare function normalizeRange(range: string): string;
/**
 * Validates A1 notation range format
 * Returns normalized range or throws descriptive error
 *
 * @param range - A1 notation range string
 * @returns Normalized and validated range
 * @throws Error if range format is invalid
 */
export declare function validateAndNormalizeRange(range: string): string;
/**
 * Sanitizes values array by converting null/undefined to empty strings
 * This ensures Google Sheets API compatibility (only accepts string | number | boolean)
 *
 * Also handles Date objects by converting them to ISO strings
 *
 * @param values - Values array (may contain null/undefined/Date objects)
 * @returns Sanitized array with only string | number | boolean values
 */
export declare function sanitizeValues(values: unknown): Array<Array<string | number | boolean>>;
export declare function enhanceErrorMessage(errorText: string, status: number, statusText: string, spreadsheetId?: string, range?: string): string;
//# sourceMappingURL=google-sheets.utils.d.ts.map