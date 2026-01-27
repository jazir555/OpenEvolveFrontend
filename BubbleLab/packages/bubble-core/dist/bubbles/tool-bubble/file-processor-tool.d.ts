/**
 * FILE PROCESSOR TOOL
 *
 * A tool bubble for comprehensive file operations including reading,
 * writing, validation, and manipulation of various file types.
 *
 * Features:
 * - Read and write files
 * - File validation (existence, size, type)
 * - Batch file operations
 * - File metadata extraction
 * - Secure file handling with path validation
 * - Support for multiple file types
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * File operation types
 */
export declare enum FileOperationType {
    READ = "read",
    WRITE = "write",
    EXISTS = "exists",
    DELETE = "delete",
    LIST = "list",
    METADATA = "metadata",
    COPY = "copy",
    MOVE = "move",
    MKDIR = "mkdir",
    WATCH = "watch",
    BATCH = "batch"
}
/**
 * File processor parameters schema
 */
declare const FileProcessorToolParamsSchema: z.ZodObject<{
    operation: z.ZodNativeEnum<typeof FileOperationType>;
    filePath: z.ZodOptional<z.ZodString>;
    targetPath: z.ZodOptional<z.ZodString>;
    directoryPath: z.ZodOptional<z.ZodString>;
    content: z.ZodOptional<z.ZodString>;
    encoding: z.ZodDefault<z.ZodEnum<["utf8", "ascii", "base64", "hex", "auto"]>>;
    recursive: z.ZodDefault<z.ZodBoolean>;
    overwrite: z.ZodDefault<z.ZodBoolean>;
    createDirectory: z.ZodDefault<z.ZodBoolean>;
    maxFileSize: z.ZodDefault<z.ZodNumber>;
    allowedExtensions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    watchDuration: z.ZodOptional<z.ZodNumber>;
    batchOperations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        operation: z.ZodNativeEnum<typeof FileOperationType>;
        filePath: z.ZodString;
        targetPath: z.ZodOptional<z.ZodString>;
        content: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        filePath: string;
        operation: FileOperationType;
        content?: string | undefined;
        targetPath?: string | undefined;
    }, {
        filePath: string;
        operation: FileOperationType;
        content?: string | undefined;
        targetPath?: string | undefined;
    }>, "many">>;
    allowPaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    denyPaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    encoding: "ascii" | "utf8" | "base64" | "hex" | "auto";
    recursive: boolean;
    operation: FileOperationType;
    overwrite: boolean;
    createDirectory: boolean;
    maxFileSize: number;
    filePath?: string | undefined;
    content?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    targetPath?: string | undefined;
    directoryPath?: string | undefined;
    allowedExtensions?: string[] | undefined;
    watchDuration?: number | undefined;
    batchOperations?: {
        filePath: string;
        operation: FileOperationType;
        content?: string | undefined;
        targetPath?: string | undefined;
    }[] | undefined;
    allowPaths?: string[] | undefined;
    denyPaths?: string[] | undefined;
}, {
    operation: FileOperationType;
    filePath?: string | undefined;
    encoding?: "ascii" | "utf8" | "base64" | "hex" | "auto" | undefined;
    recursive?: boolean | undefined;
    content?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    targetPath?: string | undefined;
    directoryPath?: string | undefined;
    overwrite?: boolean | undefined;
    createDirectory?: boolean | undefined;
    maxFileSize?: number | undefined;
    allowedExtensions?: string[] | undefined;
    watchDuration?: number | undefined;
    batchOperations?: {
        filePath: string;
        operation: FileOperationType;
        content?: string | undefined;
        targetPath?: string | undefined;
    }[] | undefined;
    allowPaths?: string[] | undefined;
    denyPaths?: string[] | undefined;
}>;
/**
 * File processor result schema
 */
declare const FileProcessorToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    content: z.ZodOptional<z.ZodString>;
    files: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodObject<{
        name: z.ZodString;
        path: z.ZodString;
        size: z.ZodNumber;
        extension: z.ZodString;
        created: z.ZodString;
        modified: z.ZodString;
        isDirectory: z.ZodBoolean;
        isFile: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        path: string;
        name: string;
        created: string;
        size: number;
        extension: string;
        modified: string;
        isDirectory: boolean;
        isFile: boolean;
    }, {
        path: string;
        name: string;
        created: string;
        size: number;
        extension: string;
        modified: string;
        isDirectory: boolean;
        isFile: boolean;
    }>>;
    stats: z.ZodObject<{
        fileSize: z.ZodOptional<z.ZodNumber>;
        filesProcessed: z.ZodOptional<z.ZodNumber>;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        fileSize?: number | undefined;
        filesProcessed?: number | undefined;
    }, {
        processingTime: number;
        fileSize?: number | undefined;
        filesProcessed?: number | undefined;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        fileSize?: number | undefined;
        filesProcessed?: number | undefined;
    };
    content?: string | undefined;
    files?: string[] | undefined;
    metadata?: {
        path: string;
        name: string;
        created: string;
        size: number;
        extension: string;
        modified: string;
        isDirectory: boolean;
        isFile: boolean;
    } | undefined;
}, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        fileSize?: number | undefined;
        filesProcessed?: number | undefined;
    };
    content?: string | undefined;
    files?: string[] | undefined;
    metadata?: {
        path: string;
        name: string;
        created: string;
        size: number;
        extension: string;
        modified: string;
        isDirectory: boolean;
        isFile: boolean;
    } | undefined;
}>;
type FileProcessorToolParams = z.output<typeof FileProcessorToolParamsSchema>;
type FileProcessorToolResult = z.output<typeof FileProcessorToolResultSchema>;
type FileProcessorToolParamsInput = z.input<typeof FileProcessorToolParamsSchema>;
/**
 * File Processor Tool
 * Comprehensive file operations with security validation
 */
export declare class FileProcessorTool extends ToolBubble<FileProcessorToolParams, FileProcessorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodNativeEnum<typeof FileOperationType>;
        filePath: z.ZodOptional<z.ZodString>;
        targetPath: z.ZodOptional<z.ZodString>;
        directoryPath: z.ZodOptional<z.ZodString>;
        content: z.ZodOptional<z.ZodString>;
        encoding: z.ZodDefault<z.ZodEnum<["utf8", "ascii", "base64", "hex", "auto"]>>;
        recursive: z.ZodDefault<z.ZodBoolean>;
        overwrite: z.ZodDefault<z.ZodBoolean>;
        createDirectory: z.ZodDefault<z.ZodBoolean>;
        maxFileSize: z.ZodDefault<z.ZodNumber>;
        allowedExtensions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        watchDuration: z.ZodOptional<z.ZodNumber>;
        batchOperations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            operation: z.ZodNativeEnum<typeof FileOperationType>;
            filePath: z.ZodString;
            targetPath: z.ZodOptional<z.ZodString>;
            content: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            filePath: string;
            operation: FileOperationType;
            content?: string | undefined;
            targetPath?: string | undefined;
        }, {
            filePath: string;
            operation: FileOperationType;
            content?: string | undefined;
            targetPath?: string | undefined;
        }>, "many">>;
        allowPaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        denyPaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        encoding: "ascii" | "utf8" | "base64" | "hex" | "auto";
        recursive: boolean;
        operation: FileOperationType;
        overwrite: boolean;
        createDirectory: boolean;
        maxFileSize: number;
        filePath?: string | undefined;
        content?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        targetPath?: string | undefined;
        directoryPath?: string | undefined;
        allowedExtensions?: string[] | undefined;
        watchDuration?: number | undefined;
        batchOperations?: {
            filePath: string;
            operation: FileOperationType;
            content?: string | undefined;
            targetPath?: string | undefined;
        }[] | undefined;
        allowPaths?: string[] | undefined;
        denyPaths?: string[] | undefined;
    }, {
        operation: FileOperationType;
        filePath?: string | undefined;
        encoding?: "ascii" | "utf8" | "base64" | "hex" | "auto" | undefined;
        recursive?: boolean | undefined;
        content?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        targetPath?: string | undefined;
        directoryPath?: string | undefined;
        overwrite?: boolean | undefined;
        createDirectory?: boolean | undefined;
        maxFileSize?: number | undefined;
        allowedExtensions?: string[] | undefined;
        watchDuration?: number | undefined;
        batchOperations?: {
            filePath: string;
            operation: FileOperationType;
            content?: string | undefined;
            targetPath?: string | undefined;
        }[] | undefined;
        allowPaths?: string[] | undefined;
        denyPaths?: string[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodOptional<z.ZodString>;
        files: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodObject<{
            name: z.ZodString;
            path: z.ZodString;
            size: z.ZodNumber;
            extension: z.ZodString;
            created: z.ZodString;
            modified: z.ZodString;
            isDirectory: z.ZodBoolean;
            isFile: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            path: string;
            name: string;
            created: string;
            size: number;
            extension: string;
            modified: string;
            isDirectory: boolean;
            isFile: boolean;
        }, {
            path: string;
            name: string;
            created: string;
            size: number;
            extension: string;
            modified: string;
            isDirectory: boolean;
            isFile: boolean;
        }>>;
        stats: z.ZodObject<{
            fileSize: z.ZodOptional<z.ZodNumber>;
            filesProcessed: z.ZodOptional<z.ZodNumber>;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            fileSize?: number | undefined;
            filesProcessed?: number | undefined;
        }, {
            processingTime: number;
            fileSize?: number | undefined;
            filesProcessed?: number | undefined;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            fileSize?: number | undefined;
            filesProcessed?: number | undefined;
        };
        content?: string | undefined;
        files?: string[] | undefined;
        metadata?: {
            path: string;
            name: string;
            created: string;
            size: number;
            extension: string;
            modified: string;
            isDirectory: boolean;
            isFile: boolean;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            fileSize?: number | undefined;
            filesProcessed?: number | undefined;
        };
        content?: string | undefined;
        files?: string[] | undefined;
        metadata?: {
            path: string;
            name: string;
            created: string;
            size: number;
            extension: string;
            modified: string;
            isDirectory: boolean;
            isFile: boolean;
        } | undefined;
    }>;
    static readonly shortDescription = "Read, write, and manipulate files with security validation";
    static readonly longDescription = "\n    A comprehensive file processing tool for secure file operations.\n\n    Features:\n    - READ: Read file contents with encoding support\n    - WRITE: Write content to files with overwrite options\n    - EXISTS: Check if file or directory exists\n    - DELETE: Delete files or directories\n    - LIST: List files in directory (with recursive option)\n    - METADATA: Extract file metadata (size, dates, type, encoding, MIME)\n    - COPY: Copy files to target location\n    - MOVE: Move/rename files\n    - MKDIR: Create directories\n    - WATCH: Watch directory for changes\n    - BATCH: Execute multiple file operations\n\n    Security Features:\n    - Path validation against whitelist/blacklist\n    - File size limits to prevent memory issues\n    - Extension validation for file type checking\n    - Safe path resolution to prevent directory traversal\n    - Encoding validation for read/write operations\n    - MIME type detection for file type identification\n\n    Use cases:\n    - Configuration file reading/writing\n    - Log file processing\n    - Batch file operations\n    - Data export/import\n    - File system monitoring\n    - Secure file handling in workflows\n\n    Supported Encodings:\n    - utf8: Standard text encoding (default)\n    - ascii: ASCII text encoding\n    - base64: Base64 binary encoding\n    - hex: Hexadecimal encoding\n    - auto: Auto-detect encoding\n\n    Security Notes:\n    - Always validate paths against allowed paths\n    - Use maxFileSize to prevent memory issues\n    - Use allowedExtensions to restrict file types\n    - Enable createDirectory for safe directory creation\n  ";
    static readonly alias = "file";
    private static fileWatcher;
    private static quotaManager;
    private static virusScanner;
    private static sizeLimiter;
    private static rateLimiter;
    constructor(params: FileProcessorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs file operation
     */
    performAction(context?: BubbleContext): Promise<FileProcessorToolResult>;
    /**
     * Validate path for security
     * Uses centralized security utilities for comprehensive validation
     */
    private validatePath;
    /**
     * Read file
     */
    private readFile;
    /**
     * Write file with virus scanning and quota management
     */
    private writeFile;
    /**
     * Check if file exists
     */
    private checkExists;
    /**
     * Delete file or directory with comprehensive error handling
     */
    private deleteFile;
    /**
     * List files in directory
     */
    private listFiles;
    /**
     * Get file metadata
     */
    private getMetadata;
    /**
     * Copy file
     */
    private copyFile;
    /**
     * Move file with atomic operation and rollback support
     */
    private moveFile;
    /**
     * Create directory
     */
    private makeDirectory;
    /**
     * Watch directory for changes
     */
    private watchDirectory;
    /**
     * Execute batch operations
     */
    private executeBatch;
}
export {};
//# sourceMappingURL=file-processor-tool.d.ts.map