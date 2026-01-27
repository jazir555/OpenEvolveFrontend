/**
 * CODE FORMATTER TOOL
 *
 * A tool bubble for formatting and beautifying code in various programming languages.
 * Supports multiple languages with customizable formatting options.
 *
 * Features:
 * - Format code in multiple languages
 * - Configurable indentation
 * - Add/remove line breaks
 * - Sort imports alphabetically
 * - Remove trailing whitespace
 * - Enforce consistent style
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Supported languages for code formatting
 */
export declare enum CodeLanguage {
    JAVASCRIPT = "javascript",
    TYPESCRIPT = "typescript",
    PYTHON = "python",
    JAVA = "java",
    CSHARP = "csharp",
    CPP = "cpp",
    GO = "go",
    RUST = "rust",
    HTML = "html",
    CSS = "css",
    JSON = "json",
    XML = "xml",
    YAML = "yaml",
    SQL = "sql",
    MARKDOWN = "markdown"
}
/**
 * Code formatter parameters schema
 */
declare const CodeFormatterToolParamsSchema: z.ZodObject<{
    code: z.ZodString;
    language: z.ZodNativeEnum<typeof CodeLanguage>;
    indentSize: z.ZodDefault<z.ZodNumber>;
    indentType: z.ZodDefault<z.ZodEnum<["spaces", "tabs"]>>;
    maxLineLength: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    trailingComma: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    semicolons: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    quotes: z.ZodOptional<z.ZodDefault<z.ZodEnum<["single", "double", "auto"]>>>;
    sortImports: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    removeUnusedImports: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    trimTrailingWhitespace: z.ZodDefault<z.ZodBoolean>;
    insertFinalNewline: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    language: CodeLanguage;
    indentSize: number;
    indentType: "spaces" | "tabs";
    trimTrailingWhitespace: boolean;
    insertFinalNewline: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxLineLength?: number | undefined;
    trailingComma?: boolean | undefined;
    semicolons?: boolean | undefined;
    quotes?: "double" | "auto" | "single" | undefined;
    sortImports?: boolean | undefined;
    removeUnusedImports?: boolean | undefined;
}, {
    code: string;
    language: CodeLanguage;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    indentSize?: number | undefined;
    indentType?: "spaces" | "tabs" | undefined;
    maxLineLength?: number | undefined;
    trailingComma?: boolean | undefined;
    semicolons?: boolean | undefined;
    quotes?: "double" | "auto" | "single" | undefined;
    sortImports?: boolean | undefined;
    removeUnusedImports?: boolean | undefined;
    trimTrailingWhitespace?: boolean | undefined;
    insertFinalNewline?: boolean | undefined;
}>;
/**
 * Code formatter result schema
 */
declare const CodeFormatterToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    formattedCode: z.ZodString;
    changes: z.ZodObject<{
        linesAdded: z.ZodNumber;
        linesRemoved: z.ZodNumber;
        indentationsFixed: z.ZodNumber;
        whitespaceRemoved: z.ZodNumber;
        importsSorted: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        linesAdded: number;
        linesRemoved: number;
        indentationsFixed: number;
        whitespaceRemoved: number;
        importsSorted: boolean;
    }, {
        linesAdded: number;
        linesRemoved: number;
        indentationsFixed: number;
        whitespaceRemoved: number;
        importsSorted: boolean;
    }>;
    stats: z.ZodObject<{
        originalLines: z.ZodNumber;
        formattedLines: z.ZodNumber;
        originalLength: z.ZodNumber;
        formattedLength: z.ZodNumber;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        originalLength: number;
        processingTime: number;
        originalLines: number;
        formattedLines: number;
        formattedLength: number;
    }, {
        originalLength: number;
        processingTime: number;
        originalLines: number;
        formattedLines: number;
        formattedLength: number;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        originalLength: number;
        processingTime: number;
        originalLines: number;
        formattedLines: number;
        formattedLength: number;
    };
    formattedCode: string;
    changes: {
        linesAdded: number;
        linesRemoved: number;
        indentationsFixed: number;
        whitespaceRemoved: number;
        importsSorted: boolean;
    };
}, {
    error: string;
    success: boolean;
    stats: {
        originalLength: number;
        processingTime: number;
        originalLines: number;
        formattedLines: number;
        formattedLength: number;
    };
    formattedCode: string;
    changes: {
        linesAdded: number;
        linesRemoved: number;
        indentationsFixed: number;
        whitespaceRemoved: number;
        importsSorted: boolean;
    };
}>;
type CodeFormatterToolParams = z.output<typeof CodeFormatterToolParamsSchema>;
type CodeFormatterToolResult = z.output<typeof CodeFormatterToolResultSchema>;
type CodeFormatterToolParamsInput = z.input<typeof CodeFormatterToolParamsSchema>;
/**
 * Code Formatter Tool
 * Format and beautify code in multiple languages
 */
export declare class CodeFormatterTool extends ToolBubble<CodeFormatterToolParams, CodeFormatterToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        code: z.ZodString;
        language: z.ZodNativeEnum<typeof CodeLanguage>;
        indentSize: z.ZodDefault<z.ZodNumber>;
        indentType: z.ZodDefault<z.ZodEnum<["spaces", "tabs"]>>;
        maxLineLength: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        trailingComma: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        semicolons: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        quotes: z.ZodOptional<z.ZodDefault<z.ZodEnum<["single", "double", "auto"]>>>;
        sortImports: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        removeUnusedImports: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        trimTrailingWhitespace: z.ZodDefault<z.ZodBoolean>;
        insertFinalNewline: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        language: CodeLanguage;
        indentSize: number;
        indentType: "spaces" | "tabs";
        trimTrailingWhitespace: boolean;
        insertFinalNewline: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxLineLength?: number | undefined;
        trailingComma?: boolean | undefined;
        semicolons?: boolean | undefined;
        quotes?: "double" | "auto" | "single" | undefined;
        sortImports?: boolean | undefined;
        removeUnusedImports?: boolean | undefined;
    }, {
        code: string;
        language: CodeLanguage;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        indentSize?: number | undefined;
        indentType?: "spaces" | "tabs" | undefined;
        maxLineLength?: number | undefined;
        trailingComma?: boolean | undefined;
        semicolons?: boolean | undefined;
        quotes?: "double" | "auto" | "single" | undefined;
        sortImports?: boolean | undefined;
        removeUnusedImports?: boolean | undefined;
        trimTrailingWhitespace?: boolean | undefined;
        insertFinalNewline?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        formattedCode: z.ZodString;
        changes: z.ZodObject<{
            linesAdded: z.ZodNumber;
            linesRemoved: z.ZodNumber;
            indentationsFixed: z.ZodNumber;
            whitespaceRemoved: z.ZodNumber;
            importsSorted: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            linesAdded: number;
            linesRemoved: number;
            indentationsFixed: number;
            whitespaceRemoved: number;
            importsSorted: boolean;
        }, {
            linesAdded: number;
            linesRemoved: number;
            indentationsFixed: number;
            whitespaceRemoved: number;
            importsSorted: boolean;
        }>;
        stats: z.ZodObject<{
            originalLines: z.ZodNumber;
            formattedLines: z.ZodNumber;
            originalLength: z.ZodNumber;
            formattedLength: z.ZodNumber;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            originalLength: number;
            processingTime: number;
            originalLines: number;
            formattedLines: number;
            formattedLength: number;
        }, {
            originalLength: number;
            processingTime: number;
            originalLines: number;
            formattedLines: number;
            formattedLength: number;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            originalLength: number;
            processingTime: number;
            originalLines: number;
            formattedLines: number;
            formattedLength: number;
        };
        formattedCode: string;
        changes: {
            linesAdded: number;
            linesRemoved: number;
            indentationsFixed: number;
            whitespaceRemoved: number;
            importsSorted: boolean;
        };
    }, {
        error: string;
        success: boolean;
        stats: {
            originalLength: number;
            processingTime: number;
            originalLines: number;
            formattedLines: number;
            formattedLength: number;
        };
        formattedCode: string;
        changes: {
            linesAdded: number;
            linesRemoved: number;
            indentationsFixed: number;
            whitespaceRemoved: number;
            importsSorted: boolean;
        };
    }>;
    static readonly shortDescription = "Format and beautify code in multiple programming languages";
    static readonly longDescription = "\n    A code formatting tool supporting multiple programming languages with\n    customizable formatting options.\n\n    Features:\n    - Format code in 15+ programming languages\n    - Configurable indentation (spaces or tabs)\n    - Line length enforcement\n    - Import sorting\n    - Trailing whitespace removal\n    - Consistent quote style\n    - Semicolon insertion (JS/TS)\n    - Final newline insertion\n\n    Supported Languages:\n    - JavaScript, TypeScript\n    - Python, Java, C#, C++\n    - Go, Rust\n    - HTML, CSS\n    - JSON, XML, YAML\n    - SQL, Markdown\n\n    Formatting Options:\n    - Indent size: 1-8 characters\n    - Indent type: spaces or tabs\n    - Max line length: 40+ characters\n    - Trailing comma: add or remove\n    - Semicolons: add or remove\n    - Quotes: single, double, or auto\n    - Sort imports: alphabetical order\n    - Remove unused imports\n\n    Use cases:\n    - Code style enforcement\n    - Pre-commit formatting\n    - Code review preparation\n    - Educational code formatting\n    - IDE integration\n    - Automated code quality checks\n\n    Note: This is a basic formatter implementation.\n    For production use, consider using dedicated formatters like:\n    - Prettier (JavaScript/TypeScript)\n    - Black (Python)\n    - clang-format (C/C++)\n    - gofmt (Go)\n    - rustfmt (Rust)\n  ";
    static readonly alias = "format-code";
    constructor(params: CodeFormatterToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs code formatting
     */
    performAction(context?: BubbleContext): Promise<CodeFormatterToolResult>;
    /**
     * Format code based on language
     */
    private formatCode;
    /**
     * Format JavaScript/TypeScript code
     */
    private formatJavaScript;
    /**
     * Format Python code
     */
    private formatPython;
    /**
     * Format JSON code
     */
    private formatJSON;
    /**
     * Format XML/HTML code
     */
    private formatXML;
    /**
     * Basic indentation for generic code
     */
    private basicIndentation;
    /**
     * Sort imports in code
     */
    private sortImports;
    /**
     * Count indentation changes
     */
    private countIndentationChanges;
    /**
     * Count whitespace removed
     */
    private countWhitespaceRemoved;
}
export {};
//# sourceMappingURL=code-formatter-tool.d.ts.map