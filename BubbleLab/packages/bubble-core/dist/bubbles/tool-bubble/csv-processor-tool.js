/**
 * CSV PROCESSOR TOOL
 *
 * A tool bubble for processing CSV files with comprehensive validation,
 * transformation, and analysis capabilities.
 *
 * Features:
 * - Parse CSV files with flexible delimiters
 * - Validate CSV structure and data types
 * - Transform CSV data (filter, map, reduce operations)
 * - Export data to CSV format
 * - Handle large files with streaming
 * - Support for custom delimiters and quote characters
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { evaluate } from 'mathjs';
/**
 * CSV operation types
 */
export var CSVOperationType;
(function (CSVOperationType) {
    CSVOperationType["PARSE"] = "parse";
    CSVOperationType["VALIDATE"] = "validate";
    CSVOperationType["TRANSFORM"] = "transform";
    CSVOperationType["FILTER"] = "filter";
    CSVOperationType["EXPORT"] = "export";
    CSVOperationType["MERGE"] = "merge";
    CSVOperationType["AGGREGATE"] = "aggregate";
})(CSVOperationType || (CSVOperationType = {}));
/**
 * CSV delimiter options
 */
export var CSVDelimiter;
(function (CSVDelimiter) {
    CSVDelimiter["COMMA"] = ",";
    CSVDelimiter["SEMICOLON"] = ";";
    CSVDelimiter["TAB"] = "\t";
    CSVDelimiter["PIPE"] = "|";
    CSVDelimiter["COLON"] = ":";
})(CSVDelimiter || (CSVDelimiter = {}));
/**
 * CSV processor parameters schema
 */
const CSVProcessorToolParamsSchema = z.object({
    // Operation specification
    operation: z
        .nativeEnum(CSVOperationType)
        .describe('Type of CSV operation to perform'),
    // Input data
    csvData: z
        .string()
        .optional()
        .describe('CSV data as string (for parse/validate operations)'),
    csvFilePath: z
        .string()
        .optional()
        .describe('Path to CSV file (for file-based operations)'),
    // Output format
    delimiter: z
        .nativeEnum(CSVDelimiter)
        .default(CSVDelimiter.COMMA)
        .describe('CSV delimiter character'),
    quoteChar: z
        .string()
        .length(1)
        .default('"')
        .describe('Character used for quoting fields'),
    escapeChar: z
        .string()
        .length(1)
        .default('"')
        .describe('Character used for escaping quotes'),
    hasHeader: z
        .boolean()
        .default(true)
        .describe('Whether CSV has a header row'),
    skipEmptyLines: z
        .boolean()
        .default(true)
        .describe('Whether to skip empty lines'),
    trimWhitespace: z
        .boolean()
        .default(true)
        .describe('Whether to trim whitespace from fields'),
    // Validation options
    validateSchema: z
        .record(z.string(), z.union([z.string(), z.boolean(), z.number()]))
        .optional()
        .describe('Schema for validation (column -> type mapping)'),
    maxRows: z
        .number()
        .int()
        .positive()
        .optional()
        .describe('Maximum number of rows to process'),
    // Transformation options
    transformRules: z
        .array(z.object({
        column: z.string().describe('Column name to apply transformation to'),
        operation: z
            .enum(['upper', 'lower', 'trim', 'replace', 'calculate', 'format'])
            .describe('Transformation operation'),
        value: z
            .string()
            .optional()
            .describe('Value for replace operation'),
        expression: z
            .string()
            .optional()
            .describe('Expression for calculate operation'),
    }))
        .optional()
        .describe('Transformation rules to apply'),
    // Filter options
    filterRules: z
        .array(z.object({
        column: z.string().describe('Column name to filter on'),
        operator: z
            .enum(['equals', 'contains', 'startsWith', 'endsWith', 'gt', 'lt', 'gte', 'lte'])
            .describe('Comparison operator'),
        value: z.union([z.string(), z.number()]).describe('Value to compare against'),
    }))
        .optional()
        .describe('Filter rules to apply'),
    // Aggregation options
    groupBy: z
        .array(z.string())
        .optional()
        .describe('Columns to group by for aggregation'),
    aggregations: z
        .array(z.object({
        column: z.string().describe('Column to aggregate'),
        operation: z
            .enum(['sum', 'avg', 'min', 'max', 'count', 'concat'])
            .describe('Aggregation operation'),
        alias: z.string().optional().describe('Alias for aggregated column'),
    }))
        .optional()
        .describe('Aggregation operations'),
    // Export data (for export operation)
    exportData: z
        .array(z.record(z.unknown()))
        .optional()
        .describe('Data to export to CSV'),
    // Credentials (for cloud storage access)
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Credentials for cloud storage'),
});
/**
 * CSV processor result schema
 */
const CSVProcessorToolResultSchema = z.object({
    // Parsed data
    data: z
        .array(z.record(z.unknown()))
        .optional()
        .describe('Processed CSV data as array of objects'),
    // Metadata
    rowCount: z.number().describe('Number of rows processed'),
    columnCount: z.number().describe('Number of columns'),
    headers: z.array(z.string()).optional().describe('CSV headers'),
    // Validation results
    validationErrors: z
        .array(z.object({
        row: z.number().describe('Row number with error'),
        column: z.string().describe('Column name with error'),
        error: z.string().describe('Error message'),
        value: z.unknown().describe('Invalid value'),
    }))
        .optional()
        .describe('Validation errors if any'),
    // Export result
    csvOutput: z
        .string()
        .optional()
        .describe('Generated CSV string (for export operation)'),
    // Statistics
    statistics: z
        .object({
        totalRows: z.number(),
        validRows: z.number(),
        invalidRows: z.number(),
        processingTime: z.number(),
    })
        .optional()
        .describe('Processing statistics'),
    success: z.boolean().describe('Whether the operation was successful'),
    error: z.string().describe('Error message if operation failed'),
});
/**
 * CSV Processor Tool
 * Comprehensive CSV file processing with validation and transformation
 */
export class CSVProcessorTool extends ToolBubble {
    /**
     * REQUIRED STATIC METADATA
     */
    static type = 'tool';
    static bubbleName = 'csv-processor-tool';
    static schema = CSVProcessorToolParamsSchema;
    static resultSchema = CSVProcessorToolResultSchema;
    static shortDescription = 'Process, validate, transform, and analyze CSV files';
    static longDescription = `
    A comprehensive tool for processing CSV files with support for parsing,
    validation, transformation, filtering, aggregation, and export operations.

    Features:
    - Parse CSV files with flexible delimiters (comma, semicolon, tab, pipe)
    - Validate CSV structure and data types against schemas
    - Transform data with operations (uppercase, lowercase, trim, replace, calculate)
    - Filter rows based on conditions
    - Aggregate data with group by operations (sum, avg, min, max, count)
    - Export data arrays to CSV format
    - Handle large files efficiently
    - Detailed error reporting for validation issues

    Operations:
    - PARSE: Convert CSV string to array of objects
    - VALIDATE: Validate CSV data against schema
    - TRANSFORM: Apply transformations to columns
    - FILTER: Filter rows based on conditions
    - EXPORT: Convert array of objects to CSV string
    - MERGE: Merge multiple CSV files
    - AGGREGATE: Group and aggregate data

    Use cases:
    - Data preprocessing for analysis
    - ETL (Extract, Transform, Load) pipelines
    - Data validation and cleaning
    - Report generation from data
    - Data format conversion
    - Batch data processing
  `;
    static alias = 'csv';
    constructor(params, context) {
        super(params, context);
    }
    /**
     * Main action method - performs CSV operation
     */
    async performAction(context) {
        void context; // Context available but not currently used
        const startTime = Date.now();
        try {
            console.log(`[CSVProcessorTool] Executing operation: ${this.params.operation}`);
            let result;
            switch (this.params.operation) {
                case CSVOperationType.PARSE:
                    result = await this.parseCSV();
                    break;
                case CSVOperationType.VALIDATE:
                    result = await this.validateCSV();
                    break;
                case CSVOperationType.TRANSFORM:
                    result = await this.transformCSV();
                    break;
                case CSVOperationType.FILTER:
                    result = await this.filterCSV();
                    break;
                case CSVOperationType.EXPORT:
                    result = await this.exportCSV();
                    break;
                case CSVOperationType.AGGREGATE:
                    result = await this.aggregateCSV();
                    break;
                default:
                    throw new Error(`Unsupported operation: ${this.params.operation}`);
            }
            const processingTime = Date.now() - startTime;
            if (result.statistics) {
                result.statistics.processingTime = processingTime;
            }
            return result;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[CSVProcessorTool] Operation failed: ${errorMessage}`);
            return {
                rowCount: 0,
                columnCount: 0,
                success: false,
                error: errorMessage,
            };
        }
    }
    /**
     * Parse CSV line with delimiter, handling quotes and escapes properly
     * Implements RFC 4180 CSV parsing with extended features
     */
    parseLine(line, delimiter) {
        const { quoteChar, escapeChar } = this.params;
        const values = [];
        let current = '';
        let inQuotes = false;
        let i = 0;
        while (i < line.length) {
            const char = line[i];
            const nextChar = line[i + 1];
            // Handle escaped quote character
            if (char === escapeChar && nextChar === quoteChar && inQuotes) {
                current += quoteChar;
                i += 2; // Skip both escape and quote
                continue;
            }
            // Handle quote character
            if (char === quoteChar) {
                // Check if it's a doubled quote (escaped quote without escape char)
                if (nextChar === quoteChar && inQuotes) {
                    current += quoteChar;
                    i += 2;
                    continue;
                }
                inQuotes = !inQuotes;
                i++;
                continue;
            }
            // Handle delimiter (only when not in quotes)
            if (char === delimiter && !inQuotes) {
                values.push(current);
                current = '';
                i++;
                continue;
            }
            // Handle newline characters within quoted fields
            if (char === '\r' && nextChar === '\n' && inQuotes) {
                current += '\n';
                i += 2;
                continue;
            }
            if (char === '\n' && inQuotes) {
                current += '\n';
                i++;
                continue;
            }
            // Regular character
            current += char;
            i++;
        }
        // Add the last value
        values.push(current);
        return values;
    }
    /**
     * Infer data type from a string value
     * Attempts to parse as boolean, number, or date before falling back to string
     */
    inferDataType(value) {
        const trimmed = value.trim();
        // Empty string
        if (trimmed === '') {
            return '';
        }
        // Boolean
        if (trimmed.toLowerCase() === 'true') {
            return true;
        }
        if (trimmed.toLowerCase() === 'false') {
            return false;
        }
        // Number (integer and float)
        if (/^-?\d+\.?\d*$/.test(trimmed)) {
            const num = parseFloat(trimmed);
            if (!isNaN(num)) {
                return num;
            }
        }
        // Date (ISO 8601 format)
        const date = new Date(trimmed);
        if (!isNaN(date.getTime()) && /^\d{4}-\d{2}-\d{2}/.test(trimmed)) {
            return date;
        }
        // Default to string
        return trimmed;
    }
    /**
     * Parse CSV string to array of objects with enhanced error handling
     */
    async parseCSV() {
        const { csvData, delimiter, hasHeader, skipEmptyLines, trimWhitespace } = this.params;
        if (!csvData) {
            throw new Error('csvData is required for parse operation');
        }
        // Handle different line endings (CRLF, LF, CR)
        const normalizedData = csvData.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        const lines = normalizedData.split('\n');
        const data = [];
        let headers = [];
        const validationErrors = [];
        // Process header row if present
        let startIndex = 0;
        if (hasHeader) {
            try {
                headers = this.parseLine(lines[0] || '', delimiter);
                // Trim headers if configured
                if (trimWhitespace) {
                    headers = headers.map((h) => h.trim());
                }
                // Validate headers are unique
                const uniqueHeaders = new Set(headers);
                if (uniqueHeaders.size !== headers.length) {
                    const duplicates = headers.filter((item, index) => headers.indexOf(item) !== index);
                    validationErrors.push({
                        row: 1,
                        column: 'headers',
                        error: `Duplicate header names found: ${duplicates.join(', ')}`,
                        value: duplicates,
                    });
                }
                startIndex = 1;
            }
            catch (error) {
                throw new Error(`Failed to parse header row: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
        }
        else {
            // Generate column names
            const firstRow = this.parseLine(lines[0] || '', delimiter);
            headers = firstRow.map((_, i) => `column_${i}`);
        }
        // Parse data rows
        let rowCount = 0;
        for (let i = startIndex; i < lines.length; i++) {
            const line = lines[i];
            // Skip empty lines if configured
            if (skipEmptyLines && line.trim() === '') {
                continue;
            }
            try {
                const values = this.parseLine(line, delimiter);
                // Trim whitespace if configured
                const processedValues = trimWhitespace
                    ? values.map((v) => v.trim())
                    : values;
                // Validate row length matches header length
                if (processedValues.length !== headers.length) {
                    validationErrors.push({
                        row: i + 1,
                        column: 'row_length',
                        error: `Row has ${processedValues.length} values, expected ${headers.length}`,
                        value: processedValues,
                    });
                    // Pad or trim to match headers
                    while (processedValues.length < headers.length) {
                        processedValues.push('');
                    }
                    if (processedValues.length > headers.length) {
                        processedValues.length = headers.length;
                    }
                }
                // Create row object with type inference
                const row = {};
                headers.forEach((header, index) => {
                    const value = processedValues[index] || '';
                    // Infer data type for better handling
                    row[header] = this.inferDataType(value);
                });
                data.push(row);
                rowCount++;
                // Check max rows limit
                if (this.params.maxRows && data.length >= this.params.maxRows) {
                    console.warn(`[CSVProcessorTool] Reached max rows limit (${this.params.maxRows}), stopping parse`);
                    break;
                }
            }
            catch (error) {
                validationErrors.push({
                    row: i + 1,
                    column: 'parse_error',
                    error: error instanceof Error ? error.message : 'Unknown parse error',
                    value: line,
                });
            }
        }
        return {
            data,
            rowCount,
            columnCount: headers.length,
            headers,
            validationErrors: validationErrors.length > 0 ? validationErrors : undefined,
            statistics: {
                totalRows: rowCount,
                validRows: rowCount - validationErrors.length,
                invalidRows: validationErrors.length,
                processingTime: 0,
            },
            success: true,
            error: '',
        };
    }
    /**
     * Validate CSV data against schema
     */
    async validateCSV() {
        const parseResult = await this.parseCSV();
        if (!parseResult.data || !this.params.validateSchema) {
            return parseResult;
        }
        const validationErrors = [];
        let validRows = 0;
        parseResult.data.forEach((row, rowIndex) => {
            let rowValid = true;
            Object.entries(this.params.validateSchema).forEach(([column, expectedType]) => {
                const value = row[column];
                if (expectedType === 'string' && typeof value !== 'string') {
                    validationErrors.push({
                        row: rowIndex + 1,
                        column,
                        error: `Expected string, got ${typeof value}`,
                        value,
                    });
                    rowValid = false;
                }
                else if (expectedType === 'number' &&
                    typeof value !== 'number' &&
                    isNaN(Number(value))) {
                    validationErrors.push({
                        row: rowIndex + 1,
                        column,
                        error: `Expected number, got ${typeof value}`,
                        value,
                    });
                    rowValid = false;
                }
                else if (expectedType === 'boolean' &&
                    typeof value !== 'boolean' &&
                    value !== 'true' &&
                    value !== 'false') {
                    validationErrors.push({
                        row: rowIndex + 1,
                        column,
                        error: `Expected boolean, got ${typeof value}`,
                        value,
                    });
                    rowValid = false;
                }
            });
            if (rowValid) {
                validRows++;
            }
        });
        return {
            ...parseResult,
            validationErrors,
            statistics: {
                totalRows: parseResult.rowCount,
                validRows,
                invalidRows: parseResult.rowCount - validRows,
                processingTime: 0,
            },
            success: true,
            error: '',
        };
    }
    /**
     * Transform CSV data
     */
    async transformCSV() {
        const parseResult = await this.parseCSV();
        if (!parseResult.data || !this.params.transformRules) {
            return parseResult;
        }
        const transformedData = parseResult.data.map((row) => {
            const transformedRow = { ...row };
            const currentRow = row; // Alias for clarity
            this.params.transformRules.forEach((rule) => {
                const { column, operation, value, expression } = rule;
                // For calculate operations, allow creating new columns
                // For other operations, only process existing columns
                if (!(column in transformedRow) && operation !== 'calculate') {
                    return;
                }
                const currentValue = transformedRow[column];
                switch (operation) {
                    case 'upper':
                        transformedRow[column] = String(currentValue).toUpperCase();
                        break;
                    case 'lower':
                        transformedRow[column] = String(currentValue).toLowerCase();
                        break;
                    case 'trim':
                        transformedRow[column] = String(currentValue).trim();
                        break;
                    case 'replace':
                        transformedRow[column] = String(currentValue).replace(new RegExp(value, 'g'), value);
                        break;
                    case 'calculate':
                        // SECURE: Use mathjs library for safe expression evaluation
                        // This prevents code injection vulnerabilities
                        try {
                            if (!expression || expression.trim().length === 0) {
                                throw new Error('Expression cannot be empty');
                            }
                            // Validate expression length to prevent DoS attacks
                            if (expression.length > 1000) {
                                throw new Error('Expression too long (max 1000 characters)');
                            }
                            // Replace column placeholders with actual values
                            // Support both {column} and column formats
                            let parsedExpression = expression;
                            const scope = {};
                            // Add all columns from current row to scope
                            Object.keys(currentRow).forEach(key => {
                                // SECURITY FIX: Prevent prototype pollution attacks
                                // Block dangerous property names that could pollute Object.prototype
                                if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
                                    console.warn(`[CSVProcessorTool] Skipping dangerous column name: ${key}`);
                                    return;
                                }
                                // Convert to number, use 0 if conversion fails
                                const numValue = typeof currentRow[key] === 'number'
                                    ? currentRow[key]
                                    : Number(currentRow[key]) || 0;
                                scope[key] = numValue;
                                // Replace {key} placeholders in expression with actual numeric values
                                parsedExpression = parsedExpression.replace(new RegExp(`\\{${key}\\}`, 'g'), String(numValue));
                            });
                            // Also add the current column being calculated (if it exists)
                            if (column in currentRow) {
                                scope[column] = typeof currentValue === 'number'
                                    ? currentValue
                                    : Number(currentValue) || 0;
                            }
                            // Use mathjs evaluate for secure math expression evaluation
                            // Only allows mathematical operations, no code execution
                            const result = evaluate(parsedExpression, scope);
                            // Validate result is a number and finite
                            if (typeof result === 'number' && !isNaN(result) && isFinite(result)) {
                                transformedRow[column] = result;
                            }
                            else {
                                console.warn(`[CSVProcessorTool] Expression result is not a valid number: ${result}`);
                                // For new columns, don't add them if calculation failed
                                // For existing columns, keep the original value
                                if (column in currentRow) {
                                    transformedRow[column] = currentValue;
                                }
                            }
                        }
                        catch (e) {
                            const errorMsg = e instanceof Error ? e.message : 'Unknown error';
                            console.error(`[CSVProcessorTool] Failed to evaluate expression "${expression}": ${errorMsg}`);
                            // For new columns, don't add them if calculation failed
                            // For existing columns, keep the original value
                            if (column in currentRow) {
                                transformedRow[column] = currentValue;
                            }
                        }
                        break;
                    case 'format':
                        // Apply custom formatting
                        transformedRow[column] = value.replace('{value}', String(currentValue));
                        break;
                }
            });
            return transformedRow;
        });
        return {
            ...parseResult,
            data: transformedData,
            success: true,
            error: '',
        };
    }
    /**
     * Filter CSV data
     */
    async filterCSV() {
        const parseResult = await this.parseCSV();
        if (!parseResult.data || !this.params.filterRules) {
            return parseResult;
        }
        const filteredData = parseResult.data.filter((row) => {
            return this.params.filterRules.every((rule) => {
                const { column, operator, value } = rule;
                const rowValue = row[column];
                switch (operator) {
                    case 'equals':
                        return rowValue === value;
                    case 'contains':
                        return String(rowValue).includes(String(value));
                    case 'startsWith':
                        return String(rowValue).startsWith(String(value));
                    case 'endsWith':
                        return String(rowValue).endsWith(String(value));
                    case 'gt':
                        return Number(rowValue) > Number(value);
                    case 'lt':
                        return Number(rowValue) < Number(value);
                    case 'gte':
                        return Number(rowValue) >= Number(value);
                    case 'lte':
                        return Number(rowValue) <= Number(value);
                    default:
                        return true;
                }
            });
        });
        return {
            ...parseResult,
            data: filteredData,
            rowCount: filteredData.length,
            statistics: {
                totalRows: parseResult.rowCount,
                validRows: filteredData.length,
                invalidRows: parseResult.rowCount - filteredData.length,
                processingTime: 0,
            },
            success: true,
            error: '',
        };
    }
    /**
     * Export data to CSV
     */
    async exportCSV() {
        const { exportData, delimiter, hasHeader } = this.params;
        if (!exportData || exportData.length === 0) {
            throw new Error('exportData is required for export operation');
        }
        const headers = Object.keys(exportData[0]);
        const lines = [];
        // Add header row
        if (hasHeader) {
            lines.push(headers.join(delimiter));
        }
        // Add data rows
        exportData.forEach((row) => {
            const values = headers.map((header) => {
                const value = row[header];
                return typeof value === 'string' &&
                    (value.includes(delimiter) || value.includes('"') || value.includes('\n'))
                    ? `"${value.replace(/"/g, '""')}"`
                    : String(value ?? '');
            });
            lines.push(values.join(delimiter));
        });
        const csvOutput = lines.join('\n');
        return {
            data: exportData,
            rowCount: exportData.length,
            columnCount: headers.length,
            headers,
            csvOutput,
            success: true,
            error: '',
        };
    }
    /**
     * Aggregate CSV data
     */
    async aggregateCSV() {
        const parseResult = await this.parseCSV();
        if (!parseResult.data || !this.params.groupBy || !this.params.aggregations) {
            throw new Error('groupBy and aggregations are required for aggregate operation');
        }
        const groups = new Map();
        // Group data
        parseResult.data.forEach((row) => {
            const key = this.params.groupBy.map((col) => String(row[col])).join('|');
            if (!groups.has(key)) {
                groups.set(key, []);
            }
            groups.get(key).push(row);
        });
        // Apply aggregations to each group
        const aggregatedData = [];
        groups.forEach((groupRows, key) => {
            const aggregatedRow = {};
            // Add group by columns
            this.params.groupBy.forEach((col, index) => {
                aggregatedRow[col] = key.split('|')[index];
            });
            // Apply aggregations
            this.params.aggregations.forEach((agg) => {
                const { column, operation, alias } = agg;
                const values = groupRows.map((row) => row[column]).filter((v) => v != null);
                let result;
                switch (operation) {
                    case 'sum':
                        result = values.reduce((sum, v) => sum + Number(v), 0);
                        break;
                    case 'avg':
                        result =
                            values.reduce((sum, v) => sum + Number(v), 0) / values.length;
                        break;
                    case 'min':
                        result = Math.min(...values.map((v) => Number(v)));
                        break;
                    case 'max':
                        result = Math.max(...values.map((v) => Number(v)));
                        break;
                    case 'count':
                        result = values.length;
                        break;
                    case 'concat':
                        result = values.join(', ');
                        break;
                }
                aggregatedRow[alias || `${column}_${operation}`] = result;
            });
            aggregatedData.push(aggregatedRow);
        });
        return {
            ...parseResult,
            data: aggregatedData,
            rowCount: aggregatedData.length,
            success: true,
            error: '',
        };
    }
}
//# sourceMappingURL=csv-processor-tool.js.map