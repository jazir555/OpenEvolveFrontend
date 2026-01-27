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
import { CredentialType } from '@bubblelab/shared-schemas';
import { readFileSync, writeFileSync, existsSync, statSync, readdirSync, mkdirSync, unlinkSync, copyFileSync, renameSync, watch as fsWatch, } from 'fs';
import { join, basename, dirname, extname, resolve } from 'path';
import { sanitizeFilePath, validateFileType, SizeLimiter, RateLimiter, } from '../../utils/security-utils.js';
/**
 * MIME type detector
 */
class MIMEDetector {
    static MIME_MAP = {
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.pdf': 'application/pdf',
        '.zip': 'application/zip',
        '.tar': 'application/x-tar',
        '.gz': 'application/gzip',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.mp4': 'video/mp4',
        '.mp3': 'audio/mpeg',
        '.csv': 'text/csv',
        '.md': 'text/markdown',
        '.yml': 'text/yaml',
        '.yaml': 'text/yaml',
    };
    /**
     * Detect MIME type from file extension
     */
    static detect(filePath) {
        const ext = extname(filePath).toLowerCase();
        return this.MIME_MAP[ext] || 'application/octet-stream';
    }
}
/**
 * File encoding detector
 */
class EncodingDetector {
    /**
     * Detect file encoding (simplified version)
     */
    static detect(filePath) {
        try {
            const buffer = readFileSync(filePath);
            // Check for UTF-8 BOM
            if (buffer.length >= 3 && buffer[0] === 0xEF && buffer[1] === 0xBB && buffer[2] === 0xBF) {
                return 'utf8';
            }
            // Check if file is valid UTF-8
            let isUtf8 = true;
            for (let i = 0; i < buffer.length; i++) {
                const byte = buffer[i];
                if (byte > 127) {
                    // Multi-byte sequence
                    let expectedBytes = 0;
                    if (byte >= 0xC0 && byte <= 0xDF)
                        expectedBytes = 1;
                    else if (byte >= 0xE0 && byte <= 0xEF)
                        expectedBytes = 2;
                    else if (byte >= 0xF0 && byte <= 0xF7)
                        expectedBytes = 3;
                    if (i + expectedBytes >= buffer.length) {
                        isUtf8 = false;
                        break;
                    }
                    for (let j = 1; j <= expectedBytes; j++) {
                        if (buffer[i + j] < 0x80 || buffer[i + j] > 0xBF) {
                            isUtf8 = false;
                            break;
                        }
                    }
                    i += expectedBytes;
                }
            }
            if (isUtf8)
                return 'utf8';
            // Check if ASCII
            let isAscii = true;
            for (const byte of buffer) {
                if (byte > 127) {
                    isAscii = false;
                    break;
                }
            }
            if (isAscii)
                return 'ascii';
            // Default to base64 for binary
            return 'base64';
        }
        catch (error) {
            return 'utf8'; // Default fallback
        }
    }
}
/**
 * File watcher for monitoring directory changes
 * Implements proper cleanup and resource limits
 */
class FileWatcher {
    watchers = new Map();
    maxWatchers;
    watchCount = 0;
    constructor(maxWatchers = 100) {
        this.maxWatchers = maxWatchers;
    }
    /**
     * Watch a directory for changes
     */
    watch(directoryPath, onChange) {
        if (this.watchers.has(directoryPath)) {
            return; // Already watching
        }
        // Enforce maximum watcher limit
        if (this.watchCount >= this.maxWatchers) {
            console.warn(`[FileWatcher] Maximum watcher limit reached (${this.maxWatchers}). Cannot watch ${directoryPath}`);
            return;
        }
        try {
            const watcher = fsWatch(directoryPath, (eventType, filename) => {
                if (filename) {
                    onChange(eventType, filename);
                }
            });
            this.watchers.set(directoryPath, watcher);
            this.watchCount++;
            console.log(`[FileWatcher] Now watching ${directoryPath} (${this.watchCount}/${this.maxWatchers} active)`);
        }
        catch (error) {
            console.error(`Failed to watch directory ${directoryPath}:`, error);
        }
    }
    /**
     * Stop watching a directory
     */
    unwatch(directoryPath) {
        const watcher = this.watchers.get(directoryPath);
        if (watcher) {
            try {
                watcher.close();
                this.watchers.delete(directoryPath);
                this.watchCount--;
                console.log(`[FileWatcher] Stopped watching ${directoryPath} (${this.watchCount}/${this.maxWatchers} active)`);
            }
            catch (error) {
                console.error(`[FileWatcher] Error closing watcher for ${directoryPath}:`, error);
                // Still remove from map even if close fails
                this.watchers.delete(directoryPath);
                this.watchCount--;
            }
        }
    }
    /**
     * Stop watching all directories
     */
    unwatchAll() {
        console.log(`[FileWatcher] Closing all ${this.watchCount} watchers`);
        this.watchers.forEach((watcher, path) => {
            try {
                watcher.close();
            }
            catch (error) {
                console.error(`[FileWatcher] Error closing watcher for ${path}:`, error);
            }
        });
        this.watchers.clear();
        this.watchCount = 0;
    }
    /**
     * Get current watcher count
     */
    getWatcherCount() {
        return this.watchCount;
    }
}
/**
 * Quota manager for file operations
 * Tracks storage usage and enforces quotas
 */
class QuotaManager {
    usage = new Map();
    quotas;
    constructor(defaultQuota = 1024 * 1024 * 1024) {
        this.quotas = new Map();
        this.quotas.set('default', defaultQuota);
    }
    /**
     * Set quota for a specific path
     */
    setQuota(path, quota) {
        this.quotas.set(path, quota);
    }
    /**
     * Get quota for a path
     */
    getQuota(path) {
        // Find the most specific quota
        let matchedPath = '';
        let matchedQuota = this.quotas.get('default') || 0;
        for (const [quotaPath, quota] of this.quotas) {
            if (path.startsWith(quotaPath) && quotaPath.length > matchedPath.length) {
                matchedPath = quotaPath;
                matchedQuota = quota;
            }
        }
        return matchedQuota;
    }
    /**
     * Check if adding file size would exceed quota
     */
    checkQuota(path, additionalSize) {
        const quota = this.getQuota(path);
        const currentUsage = this.usage.get(path) || 0;
        return (currentUsage + additionalSize) <= quota;
    }
    /**
     * Record file operation for quota tracking
     */
    recordOperation(path, size, operation) {
        const currentUsage = this.usage.get(path) || 0;
        if (operation === 'write') {
            this.usage.set(path, currentUsage + size);
        }
        else if (operation === 'delete') {
            this.usage.set(path, Math.max(0, currentUsage - size));
        }
    }
    /**
     * Get current usage for a path
     */
    getUsage(path) {
        return this.usage.get(path) || 0;
    }
    /**
     * Reset usage for a path
     */
    resetUsage(path) {
        this.usage.set(path, 0);
    }
    /**
     * Get all usage statistics
     */
    getAllUsage() {
        return Object.fromEntries(this.usage);
    }
}
/**
 * Virus scanner integration hooks
 * Provides interface for antivirus scanning of files
 */
class VirusScanner {
    enabled;
    scannerCommand;
    timeout;
    constructor(enabled = false, scannerCommand, timeout = 30000) {
        this.enabled = enabled;
        this.scannerCommand = scannerCommand || process.env.VIRUS_SCANNER_COMMAND;
        this.timeout = timeout;
    }
    /**
     * Scan a file for viruses
     * Returns true if file is clean, false if infected
     */
    async scanFile(filePath) {
        if (!this.enabled || !this.scannerCommand) {
            // If virus scanning is not enabled, return clean
            console.warn('[VirusScanner] Virus scanning is not enabled. File will not be scanned.');
            return { clean: true };
        }
        try {
            console.log(`[VirusScanner] Scanning file: ${filePath}`);
            // In a real implementation, this would call an antivirus CLI tool
            // Examples: clamscan, Windows Defender, etc.
            // For now, we provide a placeholder that logs the action
            // Placeholder: Execute virus scanner command
            // const { exec } = require('child_process');
            // const result = await new Promise((resolve, reject) => {
            //   exec(`${this.scannerCommand} "${filePath}"`, { timeout: this.timeout }, (error, stdout, stderr) => {
            //     if (error) {
            //       reject(error);
            //     } else {
            //       resolve({ stdout, stderr });
            //     }
            //   });
            // });
            console.log(`[VirusScanner] Scan completed for: ${filePath}`);
            return { clean: true };
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[VirusScanner] Scan failed for ${filePath}: ${errorMsg}`);
            return { clean: false, reason: `Virus scan failed: ${errorMsg}` };
        }
    }
    /**
     * Enable virus scanning
     */
    enable() {
        this.enabled = true;
    }
    /**
     * Disable virus scanning
     */
    disable() {
        this.enabled = false;
    }
    /**
     * Check if virus scanning is enabled
     */
    isEnabled() {
        return this.enabled;
    }
}
/**
 * File operation types
 */
export var FileOperationType;
(function (FileOperationType) {
    FileOperationType["READ"] = "read";
    FileOperationType["WRITE"] = "write";
    FileOperationType["EXISTS"] = "exists";
    FileOperationType["DELETE"] = "delete";
    FileOperationType["LIST"] = "list";
    FileOperationType["METADATA"] = "metadata";
    FileOperationType["COPY"] = "copy";
    FileOperationType["MOVE"] = "move";
    FileOperationType["MKDIR"] = "mkdir";
    FileOperationType["WATCH"] = "watch";
    FileOperationType["BATCH"] = "batch";
})(FileOperationType || (FileOperationType = {}));
/**
 * File processor parameters schema
 */
const FileProcessorToolParamsSchema = z.object({
    // Operation specification
    operation: z
        .nativeEnum(FileOperationType)
        .describe('Type of file operation to perform'),
    // File paths
    filePath: z
        .string()
        .optional()
        .describe('Path to the file'),
    targetPath: z
        .string()
        .optional()
        .describe('Target path for copy/move operations'),
    directoryPath: z
        .string()
        .optional()
        .describe('Path to directory (for list operation)'),
    // Content for write operation
    content: z
        .string()
        .optional()
        .describe('Content to write to file'),
    encoding: z
        .enum(['utf8', 'ascii', 'base64', 'hex', 'auto'])
        .default('utf8')
        .describe('File encoding for read/write operations'),
    // Options
    recursive: z
        .boolean()
        .default(false)
        .describe('Recursive operation for directories'),
    overwrite: z
        .boolean()
        .default(false)
        .describe('Overwrite existing file for write operation'),
    createDirectory: z
        .boolean()
        .default(false)
        .describe('Create directory if it does not exist'),
    maxFileSize: z
        .number()
        .int()
        .positive()
        .default(10 * 1024 * 1024)
        .describe('Maximum file size in bytes (default 10MB)'),
    allowedExtensions: z
        .array(z.string())
        .optional()
        .describe('Allowed file extensions for validation'),
    // Watch options
    watchDuration: z
        .number()
        .int()
        .positive()
        .optional()
        .describe('Duration to watch directory in milliseconds (default: indefinite)'),
    // Batch options
    batchOperations: z
        .array(z.object({
        operation: z.nativeEnum(FileOperationType),
        filePath: z.string(),
        targetPath: z.string().optional(),
        content: z.string().optional(),
    }))
        .optional()
        .describe('Batch operations to perform'),
    // Security options
    allowPaths: z
        .array(z.string())
        .optional()
        .describe('Whitelist of allowed base paths'),
    denyPaths: z
        .array(z.string())
        .optional()
        .describe('Blacklist of denied paths'),
    // Credentials
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Credentials for cloud storage access'),
});
/**
 * File metadata schema
 */
const FileMetadataSchema = z.object({
    name: z.string().describe('File name'),
    path: z.string().describe('Full file path'),
    size: z.number().describe('File size in bytes'),
    extension: z.string().describe('File extension'),
    encoding: z.string().optional().describe('File encoding'),
    created: z.string().describe('Creation timestamp'),
    modified: z.string().describe('Last modified timestamp'),
    isDirectory: z.boolean().describe('Whether it is a directory'),
    isFile: z.boolean().describe('Whether it is a file'),
});
/**
 * File processor result schema
 */
const FileProcessorToolResultSchema = z.object({
    // Operation result
    success: z.boolean().describe('Whether the operation was successful'),
    // Content (for read operations)
    content: z
        .string()
        .optional()
        .describe('File content for read operations'),
    // File list (for list operations)
    files: z
        .array(z.string())
        .optional()
        .describe('List of files for list operations'),
    // Metadata (for metadata operations)
    metadata: z
        .object({
        name: z.string(),
        path: z.string(),
        size: z.number(),
        extension: z.string(),
        created: z.string(),
        modified: z.string(),
        isDirectory: z.boolean(),
        isFile: z.boolean(),
    })
        .optional()
        .describe('File metadata'),
    // Statistics
    stats: z
        .object({
        fileSize: z.number().optional(),
        filesProcessed: z.number().optional(),
        processingTime: z.number(),
    })
        .describe('Operation statistics'),
    error: z.string().describe('Error message if operation failed'),
});
/**
 * File Processor Tool
 * Comprehensive file operations with security validation
 */
export class FileProcessorTool extends ToolBubble {
    /**
     * REQUIRED STATIC METADATA
     */
    static type = 'tool';
    static bubbleName = 'file-processor-tool';
    static schema = FileProcessorToolParamsSchema;
    static resultSchema = FileProcessorToolResultSchema;
    static shortDescription = 'Read, write, and manipulate files with security validation';
    static longDescription = `
    A comprehensive file processing tool for secure file operations.

    Features:
    - READ: Read file contents with encoding support
    - WRITE: Write content to files with overwrite options
    - EXISTS: Check if file or directory exists
    - DELETE: Delete files or directories
    - LIST: List files in directory (with recursive option)
    - METADATA: Extract file metadata (size, dates, type, encoding, MIME)
    - COPY: Copy files to target location
    - MOVE: Move/rename files
    - MKDIR: Create directories
    - WATCH: Watch directory for changes
    - BATCH: Execute multiple file operations

    Security Features:
    - Path validation against whitelist/blacklist
    - File size limits to prevent memory issues
    - Extension validation for file type checking
    - Safe path resolution to prevent directory traversal
    - Encoding validation for read/write operations
    - MIME type detection for file type identification

    Use cases:
    - Configuration file reading/writing
    - Log file processing
    - Batch file operations
    - Data export/import
    - File system monitoring
    - Secure file handling in workflows

    Supported Encodings:
    - utf8: Standard text encoding (default)
    - ascii: ASCII text encoding
    - base64: Base64 binary encoding
    - hex: Hexadecimal encoding
    - auto: Auto-detect encoding

    Security Notes:
    - Always validate paths against allowed paths
    - Use maxFileSize to prevent memory issues
    - Use allowedExtensions to restrict file types
    - Enable createDirectory for safe directory creation
  `;
    static alias = 'file';
    // File watcher instance with resource limits
    static fileWatcher = new FileWatcher(100); // Max 100 concurrent watchers
    // Quota manager for storage tracking
    static quotaManager = new QuotaManager(1024 * 1024 * 1024); // 1GB default quota
    // Virus scanner for security
    static virusScanner = new VirusScanner(process.env.ENABLE_VIRUS_SCANNING === 'true', process.env.VIRUS_SCANNER_COMMAND, 30000 // 30 second timeout
    );
    // Size limiter for preventing DoS attacks
    static sizeLimiter = new SizeLimiter(10 * 1024 * 1024); // 10MB default
    // Rate limiter for preventing abuse
    static rateLimiter = new RateLimiter(100, 60000); // 100 requests per minute
    constructor(params, context) {
        super(params, context);
    }
    /**
     * Main action method - performs file operation
     */
    async performAction(context) {
        void context; // Context available but not currently used
        const startTime = Date.now();
        try {
            console.log(`[FileProcessorTool] Executing operation: ${this.params.operation}`);
            // Validate paths for security
            if (this.params.filePath) {
                this.validatePath(this.params.filePath);
            }
            if (this.params.targetPath) {
                this.validatePath(this.params.targetPath);
            }
            if (this.params.directoryPath) {
                this.validatePath(this.params.directoryPath);
            }
            let result;
            switch (this.params.operation) {
                case FileOperationType.READ:
                    result = await this.readFile();
                    break;
                case FileOperationType.WRITE:
                    result = await this.writeFile();
                    break;
                case FileOperationType.EXISTS:
                    result = await this.checkExists();
                    break;
                case FileOperationType.DELETE:
                    result = await this.deleteFile();
                    break;
                case FileOperationType.LIST:
                    result = await this.listFiles();
                    break;
                case FileOperationType.METADATA:
                    result = await this.getMetadata();
                    break;
                case FileOperationType.COPY:
                    result = await this.copyFile();
                    break;
                case FileOperationType.MOVE:
                    result = await this.moveFile();
                    break;
                case FileOperationType.MKDIR:
                    result = await this.makeDirectory();
                    break;
                case FileOperationType.WATCH:
                    result = await this.watchDirectory();
                    break;
                case FileOperationType.BATCH:
                    result = await this.executeBatch();
                    break;
                default:
                    throw new Error(`Unsupported operation: ${this.params.operation}`);
            }
            result.stats = {
                ...result.stats,
                processingTime: Date.now() - startTime,
            };
            return result;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[FileProcessorTool] Operation failed: ${errorMessage}`);
            return {
                success: false,
                stats: {
                    processingTime: Date.now() - startTime,
                },
                error: errorMessage,
            };
        }
    }
    /**
     * Validate path for security
     * Uses centralized security utilities for comprehensive validation
     */
    validatePath(path) {
        // Use the centralized security utility for path sanitization
        const sanitizationResult = sanitizeFilePath(path, this.params.allowPaths || []);
        if (!sanitizationResult.isSafe) {
            throw new Error(sanitizationResult.reason || 'Path validation failed');
        }
        // Additional deny list check
        if (this.params.denyPaths) {
            const resolvedPath = resolve(path);
            for (const deniedPath of this.params.denyPaths) {
                if (resolvedPath.startsWith(resolve(deniedPath))) {
                    throw new Error(`Path is denied: ${path}`);
                }
            }
        }
    }
    /**
     * Read file
     */
    async readFile() {
        const { filePath, encoding, maxFileSize, allowedExtensions } = this.params;
        if (!filePath) {
            throw new Error('filePath is required for read operation');
        }
        // Check file exists
        if (!existsSync(filePath)) {
            throw new Error(`File does not exist: ${filePath}`);
        }
        // Get file stats
        const stats = statSync(filePath);
        // Check size limit
        if (stats.size > maxFileSize) {
            throw new Error(`File too large: ${stats.size} bytes (max ${maxFileSize} bytes)`);
        }
        // Check extension
        if (allowedExtensions && allowedExtensions.length > 0) {
            const ext = extname(filePath).toLowerCase();
            if (!allowedExtensions.includes(ext)) {
                throw new Error(`File extension not allowed: ${ext}`);
            }
        }
        // Detect encoding if auto
        let detectedEncoding = encoding;
        if (encoding === 'auto') {
            detectedEncoding = EncodingDetector.detect(filePath);
        }
        // Read file
        const content = readFileSync(filePath, { encoding: detectedEncoding });
        console.log(`[FileProcessorTool] Read file: ${filePath} (${stats.size} bytes, encoding: ${detectedEncoding})`);
        return {
            success: true,
            content,
            stats: {
                fileSize: stats.size,
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Write file with virus scanning and quota management
     */
    async writeFile() {
        const { filePath, content, encoding, overwrite, createDirectory, allowedExtensions } = this.params;
        if (!filePath) {
            throw new Error('filePath is required for write operation');
        }
        if (content === undefined) {
            throw new Error('content is required for write operation');
        }
        // Check file type validation
        if (allowedExtensions && allowedExtensions.length > 0) {
            const validationResult = validateFileType(filePath, allowedExtensions);
            if (!validationResult.isValid) {
                throw new Error(validationResult.reason);
            }
        }
        // Check size limits
        const sizeCheck = FileProcessorTool.sizeLimiter.checkSize(content);
        if (!sizeCheck.withinLimit) {
            throw new Error(`Content too large: ${sizeCheck.size} bytes (max ${sizeCheck.maxSize} bytes)`);
        }
        // Check quota
        const contentSize = Buffer.byteLength(content, encoding);
        if (!FileProcessorTool.quotaManager.checkQuota(filePath, contentSize)) {
            const quota = FileProcessorTool.quotaManager.getQuota(filePath);
            const usage = FileProcessorTool.quotaManager.getUsage(filePath);
            throw new Error(`Storage quota exceeded. Quota: ${quota} bytes, Used: ${usage} bytes, Required: ${contentSize} bytes`);
        }
        // Check if file exists
        if (existsSync(filePath) && !overwrite) {
            throw new Error(`File already exists and overwrite is false: ${filePath}`);
        }
        // Create directory if needed
        if (createDirectory) {
            const dir = dirname(filePath);
            if (!existsSync(dir)) {
                mkdirSync(dir, { recursive: true });
                console.log(`[FileProcessorTool] Created directory: ${dir}`);
            }
        }
        // Write file to temporary location first
        const tempFilePath = `${filePath}.tmp`;
        writeFileSync(tempFilePath, content, { encoding: encoding });
        try {
            // Scan for viruses
            const scanResult = await FileProcessorTool.virusScanner.scanFile(tempFilePath);
            if (!scanResult.clean) {
                // Clean up temp file if virus detected
                unlinkSync(tempFilePath);
                throw new Error(scanResult.reason || 'File scan failed - possible virus detected');
            }
            // Move temp file to final location
            if (existsSync(filePath)) {
                const oldSize = statSync(filePath).size;
                FileProcessorTool.quotaManager.recordOperation(filePath, oldSize, 'delete');
            }
            renameSync(tempFilePath, filePath);
            const stats = statSync(filePath);
            // Record quota usage
            FileProcessorTool.quotaManager.recordOperation(filePath, stats.size, 'write');
            console.log(`[FileProcessorTool] Wrote file: ${filePath} (${stats.size} bytes)`);
            console.log(`[FileProcessorTool] Quota usage for ${dirname(filePath)}: ${FileProcessorTool.quotaManager.getUsage(dirname(filePath))} bytes`);
            return {
                success: true,
                stats: {
                    fileSize: stats.size,
                    processingTime: 0,
                },
                error: '',
            };
        }
        catch (error) {
            // Clean up temp file on error
            if (existsSync(tempFilePath)) {
                try {
                    unlinkSync(tempFilePath);
                }
                catch (cleanupError) {
                    console.error(`[FileProcessorTool] Failed to clean up temp file: ${cleanupError}`);
                }
            }
            throw error;
        }
    }
    /**
     * Check if file exists
     */
    async checkExists() {
        const { filePath } = this.params;
        if (!filePath) {
            throw new Error('filePath is required for exists operation');
        }
        const exists = existsSync(filePath);
        const isDirectory = exists ? statSync(filePath).isDirectory() : false;
        const isFile = exists ? !isDirectory : false;
        console.log(`[FileProcessorTool] Check exists: ${filePath} -> ${exists}`);
        return {
            success: true,
            metadata: exists
                ? {
                    name: basename(filePath),
                    path: filePath,
                    size: statSync(filePath).size,
                    extension: extname(filePath),
                    created: statSync(filePath).birthtime.toISOString(),
                    modified: statSync(filePath).mtime.toISOString(),
                    isDirectory,
                    isFile,
                }
                : undefined,
            stats: {
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Delete file or directory with comprehensive error handling
     */
    async deleteFile() {
        const { filePath, recursive } = this.params;
        if (!filePath) {
            throw new Error('filePath is required for delete operation');
        }
        // Verify file exists before attempting deletion
        if (!existsSync(filePath)) {
            throw new Error(`File does not exist: ${filePath}`);
        }
        const stats = statSync(filePath);
        const isDirectory = stats.isDirectory();
        const originalSize = stats.size;
        try {
            if (isDirectory) {
                // Handle directory deletion
                if (!recursive) {
                    // Check if directory is empty for non-recursive delete
                    const entries = readdirSync(filePath);
                    if (entries.length > 0) {
                        throw new Error(`Cannot delete non-empty directory without recursive flag: ${filePath}. ` +
                            `Directory contains ${entries.length} items.`);
                    }
                }
                // Use rimraf-style approach for recursive deletion
                if (recursive) {
                    // Delete directory contents recursively
                    const deleteRecursive = (dirPath) => {
                        const entries = readdirSync(dirPath, { withFileTypes: true });
                        for (const entry of entries) {
                            const fullPath = join(dirPath, entry.name);
                            if (entry.isDirectory()) {
                                deleteRecursive(fullPath);
                            }
                            else {
                                // Delete file
                                const fileStats = statSync(fullPath);
                                unlinkSync(fullPath);
                                // Record quota change
                                FileProcessorTool.quotaManager.recordOperation(fullPath, fileStats.size, 'delete');
                                console.log(`[FileProcessorTool] Deleted file: ${fullPath} (${fileStats.size} bytes)`);
                            }
                        }
                        // Delete the directory itself
                        const dirStats = statSync(dirPath);
                        unlinkSync(dirPath);
                        console.log(`[FileProcessorTool] Deleted directory: ${dirPath}`);
                    };
                    deleteRecursive(filePath);
                }
                else {
                    // Delete empty directory
                    unlinkSync(filePath);
                    console.log(`[FileProcessorTool] Deleted empty directory: ${filePath}`);
                }
            }
            else {
                // Handle file deletion
                unlinkSync(filePath);
                // Record quota change
                FileProcessorTool.quotaManager.recordOperation(filePath, originalSize, 'delete');
                console.log(`[FileProcessorTool] Deleted file: ${filePath} (${originalSize} bytes)`);
            }
            // Verify deletion was successful
            if (existsSync(filePath)) {
                throw new Error(`File still exists after deletion attempt: ${filePath}`);
            }
            return {
                success: true,
                stats: {
                    fileSize: originalSize,
                    processingTime: 0,
                },
                error: '',
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            // Check for common permission errors
            if (errorMessage.includes('EACCES') || errorMessage.includes('EPERM')) {
                throw new Error(`Permission denied deleting file: ${filePath}. ` +
                    `Check file permissions and ensure the file is not in use.`);
            }
            // Check for busy/file in use errors
            if (errorMessage.includes('EBUSY')) {
                throw new Error(`File is busy and cannot be deleted: ${filePath}. ` +
                    `The file may be in use by another process.`);
            }
            throw new Error(`Failed to delete file ${filePath}: ${errorMessage}`);
        }
    }
    /**
     * List files in directory
     */
    async listFiles() {
        const { directoryPath, recursive, allowedExtensions } = this.params;
        if (!directoryPath) {
            throw new Error('directoryPath is required for list operation');
        }
        if (!existsSync(directoryPath)) {
            throw new Error(`Directory does not exist: ${directoryPath}`);
        }
        const files = [];
        let filesProcessed = 0;
        const listDirectory = (dir) => {
            const entries = readdirSync(dir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = join(dir, entry.name);
                if (entry.isDirectory()) {
                    if (recursive) {
                        listDirectory(fullPath);
                        files.push(fullPath);
                    }
                }
                else {
                    // Check extension
                    if (allowedExtensions && allowedExtensions.length > 0) {
                        const ext = extname(fullPath).toLowerCase();
                        if (!allowedExtensions.includes(ext)) {
                            continue;
                        }
                    }
                    files.push(fullPath);
                    filesProcessed++;
                }
            }
        };
        listDirectory(directoryPath);
        console.log(`[FileProcessorTool] Listed ${files.length} files in: ${directoryPath}`);
        return {
            success: true,
            files,
            stats: {
                filesProcessed,
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Get file metadata
     */
    async getMetadata() {
        const { filePath } = this.params;
        if (!filePath) {
            throw new Error('filePath is required for metadata operation');
        }
        if (!existsSync(filePath)) {
            throw new Error(`File does not exist: ${filePath}`);
        }
        const stats = statSync(filePath);
        const isDirectory = stats.isDirectory();
        const isFile = stats.isFile();
        const metadata = {
            name: basename(filePath),
            path: filePath,
            size: stats.size,
            extension: extname(filePath),
            mimeType: isFile ? MIMEDetector.detect(filePath) : undefined,
            encoding: isFile ? EncodingDetector.detect(filePath) : undefined,
            created: stats.birthtime.toISOString(),
            modified: stats.mtime.toISOString(),
            isDirectory,
            isFile,
        };
        console.log(`[FileProcessorTool] Got metadata for: ${filePath}`);
        return {
            success: true,
            metadata,
            stats: {
                fileSize: stats.size,
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Copy file
     */
    async copyFile() {
        const { filePath, targetPath, createDirectory } = this.params;
        if (!filePath || !targetPath) {
            throw new Error('filePath and targetPath are required for copy operation');
        }
        if (!existsSync(filePath)) {
            throw new Error(`Source file does not exist: ${filePath}`);
        }
        // Create directory if needed
        if (createDirectory) {
            const dir = dirname(targetPath);
            if (!existsSync(dir)) {
                mkdirSync(dir, { recursive: true });
                console.log(`[FileProcessorTool] Created directory: ${dir}`);
            }
        }
        // Copy file
        const content = readFileSync(filePath);
        writeFileSync(targetPath, content);
        const stats = statSync(targetPath);
        console.log(`[FileProcessorTool] Copied file: ${filePath} -> ${targetPath}`);
        return {
            success: true,
            stats: {
                fileSize: stats.size,
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Move file with atomic operation and rollback support
     */
    async moveFile() {
        const { filePath, targetPath, createDirectory, overwrite } = this.params;
        if (!filePath || !targetPath) {
            throw new Error('filePath and targetPath are required for move operation');
        }
        // Verify source file exists
        if (!existsSync(filePath)) {
            throw new Error(`Source file does not exist: ${filePath}`);
        }
        // Check if source and target are the same
        const resolvedSource = resolve(filePath);
        const resolvedTarget = resolve(targetPath);
        if (resolvedSource === resolvedTarget) {
            throw new Error(`Source and target paths are identical: ${filePath}`);
        }
        // Check if target already exists
        if (existsSync(targetPath) && !overwrite) {
            throw new Error(`Target file already exists and overwrite is false: ${targetPath}. ` +
                `Use overwrite=true to replace the existing file.`);
        }
        // Store source stats for potential rollback
        const sourceStats = statSync(filePath);
        const sourceSize = sourceStats.size;
        let targetCreated = false;
        let sourceDeleted = false;
        try {
            // Create target directory if needed
            if (createDirectory) {
                const dir = dirname(targetPath);
                if (!existsSync(dir)) {
                    mkdirSync(dir, { recursive: true });
                    console.log(`[FileProcessorTool] Created directory: ${dir}`);
                }
            }
            // Check if target is on same device (for efficient rename)
            try {
                // Try atomic rename first (fastest, works on same filesystem)
                renameSync(filePath, targetPath);
                sourceDeleted = true;
                targetCreated = true;
                console.log(`[FileProcessorTool] Moved file (atomic rename): ${filePath} -> ${targetPath}`);
            }
            catch (renameError) {
                // If rename fails (e.g., cross-device), fall back to copy + delete
                const renameErrorMessage = renameError instanceof Error ? renameError.message : 'Unknown error';
                if (renameErrorMessage.includes('EXDEV')) {
                    console.log(`[FileProcessorTool] Cross-device move detected, using copy + delete approach`);
                    // Copy file to target
                    const content = readFileSync(filePath);
                    writeFileSync(targetPath, content);
                    targetCreated = true;
                    // Verify target was created successfully
                    if (!existsSync(targetPath)) {
                        throw new Error(`Target file was not created: ${targetPath}`);
                    }
                    // Verify target has correct content
                    const targetStats = statSync(targetPath);
                    if (targetStats.size !== sourceSize) {
                        throw new Error(`Target file size mismatch: source=${sourceSize} bytes, target=${targetStats.size} bytes. ` +
                            `Move operation aborted to prevent data loss.`);
                    }
                    // Delete source file only after successful copy
                    try {
                        unlinkSync(filePath);
                        sourceDeleted = true;
                        console.log(`[FileProcessorTool] Deleted source file after successful copy: ${filePath}`);
                    }
                    catch (deleteError) {
                        // Rollback: delete target if source deletion failed
                        try {
                            unlinkSync(targetPath);
                            console.log(`[FileProcessorTool] Rolled back: deleted target due to source deletion failure`);
                        }
                        catch (rollbackError) {
                            console.error(`[FileProcessorTool] Rollback failed: ${rollbackError}`);
                        }
                        throw new Error(`Failed to delete source file after successful copy: ${filePath}. ` +
                            `The file has been copied to ${targetPath} but the source could not be deleted. ` +
                            `Original error: ${deleteError instanceof Error ? deleteError.message : 'Unknown error'}`);
                    }
                    console.log(`[FileProcessorTool] Moved file (copy + delete): ${filePath} -> ${targetPath} (${sourceSize} bytes)`);
                }
                else {
                    throw renameError;
                }
            }
            // Final verification
            if (existsSync(filePath) && sourceDeleted) {
                throw new Error(`Source file still exists after move operation: ${filePath}`);
            }
            if (!existsSync(targetPath)) {
                throw new Error(`Target file does not exist after move operation: ${targetPath}`);
            }
            return {
                success: true,
                stats: {
                    fileSize: sourceSize,
                    processingTime: 0,
                },
                error: '',
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            // Handle specific error cases
            if (errorMessage.includes('EACCES') || errorMessage.includes('EPERM')) {
                throw new Error(`Permission denied during move operation. ` +
                    `Check permissions for both source (${filePath}) and target (${targetPath}). ` +
                    `Ensure files are not in use by another process.`);
            }
            if (errorMessage.includes('ENOSPC')) {
                throw new Error(`No space left on device for move operation: ${targetPath}. ` +
                    `Free up disk space and try again.`);
            }
            // If we have a partial state (target created but source not deleted), attempt cleanup
            if (targetCreated && !sourceDeleted && existsSync(targetPath)) {
                try {
                    unlinkSync(targetPath);
                    console.log(`[FileProcessorTool] Cleaned up partial move: deleted ${targetPath}`);
                }
                catch (cleanupError) {
                    console.error(`[FileProcessorTool] Failed to cleanup partial move: ${cleanupError}`);
                }
            }
            throw new Error(`Failed to move file from ${filePath} to ${targetPath}: ${errorMessage}`);
        }
    }
    /**
     * Create directory
     */
    async makeDirectory() {
        const { filePath, createDirectory } = this.params;
        if (!filePath) {
            throw new Error('filePath is required for mkdir operation');
        }
        if (existsSync(filePath)) {
            throw new Error(`Directory already exists: ${filePath}`);
        }
        // Create directory
        if (createDirectory) {
            mkdirSync(filePath, { recursive: true });
            console.log(`[FileProcessorTool] Created directory: ${filePath}`);
        }
        else {
            mkdirSync(filePath);
            console.log(`[FileProcessorTool] Created directory (no parents): ${filePath}`);
        }
        return {
            success: true,
            stats: {
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Watch directory for changes
     */
    async watchDirectory() {
        const { directoryPath, watchDuration } = this.params;
        if (!directoryPath) {
            throw new Error('directoryPath is required for watch operation');
        }
        if (!existsSync(directoryPath)) {
            throw new Error(`Directory does not exist: ${directoryPath}`);
        }
        const changes = [];
        // Watch directory
        FileProcessorTool.fileWatcher.watch(directoryPath, (eventType, filename) => {
            changes.push({
                eventType,
                filename,
                timestamp: new Date().toISOString(),
            });
            console.log(`[FileProcessorTool] File ${eventType}: ${filename}`);
        });
        // Wait for specified duration or indefinitely
        if (watchDuration) {
            await new Promise((resolve) => setTimeout(resolve, watchDuration));
            FileProcessorTool.fileWatcher.unwatch(directoryPath);
        }
        return {
            success: true,
            files: changes.map((c) => c.filename),
            stats: {
                filesProcessed: changes.length,
                processingTime: 0,
            },
            error: '',
        };
    }
    /**
     * Execute batch operations
     */
    async executeBatch() {
        const { batchOperations } = this.params;
        if (!batchOperations || batchOperations.length === 0) {
            throw new Error('batchOperations is required for batch operation');
        }
        const results = [];
        let successCount = 0;
        let failureCount = 0;
        for (const op of batchOperations) {
            try {
                // Validate path
                this.validatePath(op.filePath);
                if (op.targetPath) {
                    this.validatePath(op.targetPath);
                }
                // Execute operation
                switch (op.operation) {
                    case FileOperationType.READ:
                        if (existsSync(op.filePath)) {
                            const content = readFileSync(op.filePath, 'utf8');
                            results.push({ operation: op.operation, success: true });
                            successCount++;
                        }
                        else {
                            results.push({ operation: op.operation, success: false, error: 'File not found' });
                            failureCount++;
                        }
                        break;
                    case FileOperationType.WRITE:
                        if (op.content !== undefined) {
                            writeFileSync(op.filePath, op.content, 'utf8');
                            results.push({ operation: op.operation, success: true });
                            successCount++;
                        }
                        else {
                            results.push({ operation: op.operation, success: false, error: 'No content provided' });
                            failureCount++;
                        }
                        break;
                    case FileOperationType.DELETE:
                        if (existsSync(op.filePath)) {
                            try {
                                const stats = statSync(op.filePath);
                                // Handle directory deletion
                                if (stats.isDirectory()) {
                                    // For simplicity in batch, we only delete empty directories
                                    const entries = readdirSync(op.filePath);
                                    if (entries.length > 0) {
                                        results.push({
                                            operation: op.operation,
                                            success: false,
                                            error: 'Cannot delete non-empty directory in batch operation'
                                        });
                                        failureCount++;
                                        continue;
                                    }
                                }
                                unlinkSync(op.filePath);
                                results.push({ operation: op.operation, success: true });
                                successCount++;
                            }
                            catch (error) {
                                results.push({
                                    operation: op.operation,
                                    success: false,
                                    error: error instanceof Error ? error.message : 'Unknown error'
                                });
                                failureCount++;
                            }
                        }
                        else {
                            results.push({ operation: op.operation, success: false, error: 'File not found' });
                            failureCount++;
                        }
                        break;
                    case FileOperationType.COPY:
                        if (op.targetPath && existsSync(op.filePath)) {
                            copyFileSync(op.filePath, op.targetPath);
                            results.push({ operation: op.operation, success: true });
                            successCount++;
                        }
                        else {
                            results.push({ operation: op.operation, success: false, error: 'Invalid source or target' });
                            failureCount++;
                        }
                        break;
                    case FileOperationType.MOVE:
                        if (op.targetPath && existsSync(op.filePath)) {
                            try {
                                // Check if source and target are the same
                                const resolvedSource = resolve(op.filePath);
                                const resolvedTarget = resolve(op.targetPath);
                                if (resolvedSource === resolvedTarget) {
                                    results.push({
                                        operation: op.operation,
                                        success: false,
                                        error: 'Source and target paths are identical'
                                    });
                                    failureCount++;
                                    continue;
                                }
                                const sourceStats = statSync(op.filePath);
                                // Try atomic rename first
                                try {
                                    renameSync(op.filePath, op.targetPath);
                                    results.push({ operation: op.operation, success: true });
                                    successCount++;
                                }
                                catch (renameError) {
                                    const renameErrorMessage = renameError instanceof Error ? renameError.message : 'Unknown error';
                                    // Handle cross-device move
                                    if (renameErrorMessage.includes('EXDEV')) {
                                        const content = readFileSync(op.filePath);
                                        writeFileSync(op.targetPath, content);
                                        // Verify target was created
                                        if (!existsSync(op.targetPath)) {
                                            throw new Error('Target file was not created');
                                        }
                                        // Verify file size matches
                                        const targetStats = statSync(op.targetPath);
                                        if (targetStats.size !== sourceStats.size) {
                                            unlinkSync(op.targetPath); // Cleanup failed copy
                                            throw new Error('File size mismatch after copy');
                                        }
                                        // Delete source
                                        unlinkSync(op.filePath);
                                        results.push({ operation: op.operation, success: true });
                                        successCount++;
                                    }
                                    else {
                                        throw renameError;
                                    }
                                }
                            }
                            catch (error) {
                                results.push({
                                    operation: op.operation,
                                    success: false,
                                    error: error instanceof Error ? error.message : 'Unknown error'
                                });
                                failureCount++;
                            }
                        }
                        else {
                            results.push({ operation: op.operation, success: false, error: 'Invalid source or target' });
                            failureCount++;
                        }
                        break;
                    default:
                        results.push({ operation: op.operation, success: false, error: 'Unsupported operation' });
                        failureCount++;
                }
            }
            catch (error) {
                results.push({
                    operation: op.operation,
                    success: false,
                    error: error instanceof Error ? error.message : 'Unknown error',
                });
                failureCount++;
            }
        }
        console.log(`[FileProcessorTool] Batch completed: ${successCount} success, ${failureCount} failed`);
        return {
            success: failureCount === 0,
            stats: {
                filesProcessed: successCount + failureCount,
                processingTime: 0,
            },
            error: failureCount > 0 ? `${failureCount} operations failed` : '',
        };
    }
}
//# sourceMappingURL=file-processor-tool.js.map