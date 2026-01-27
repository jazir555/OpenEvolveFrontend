import { z } from 'zod';
import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, HeadObjectCommand, } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// SECURITY FIX: Add userId validation to prevent path traversal attacks
// This validates userId across all operations that use it
const validateUserId = (userId) => {
    if (!userId) {
        return true; // userId is optional, allow undefined
    }
    // SECURITY: Whitelist-based validation for userId
    // Only allow alphanumeric characters, hyphens, underscores, and dots
    // This prevents path traversal attacks via userId
    const validUserIdPattern = /^[a-zA-Z0-9._-]+$/;
    if (!validUserIdPattern.test(userId)) {
        return false;
    }
    // Block path traversal attempts
    if (userId.includes('..') || userId.includes('./') || userId.includes('.\\')) {
        return false;
    }
    // Block absolute paths
    if (userId.startsWith('/') || userId.startsWith('\\')) {
        return false;
    }
    // Limit userId length to prevent DoS
    if (userId.length > 256) {
        return false;
    }
    return true;
};
// Base credentials schema that all operations share
const BaseCredentialsSchema = z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Object mapping credential types to values (injected at runtime)');
// Define the parameters schema for storage operations using discriminated union
const StorageParamsSchema = z.discriminatedUnion('operation', [
    // Get upload URL operation
    z.object({
        operation: z
            .literal('getUploadUrl')
            .describe('Generate presigned upload URL'),
        bucketName: z
            .string()
            .min(1, 'Bucket name is required')
            .describe('Name of the R2 bucket'),
        fileName: z
            .string()
            .min(1, 'File name is required')
            .describe('Original filename for the upload'),
        accountId: z
            .string()
            .optional()
            .describe('Cloudflare Account ID - can be provided via credentials'),
        region: z
            .string()
            .optional()
            .default('auto')
            .describe('AWS region for R2 storage (defaults to auto)'),
        expirationMinutes: z
            .number()
            .optional()
            .default(60)
            .describe('URL expiration time in minutes'),
        contentType: z.string().optional().describe('Content type for uploads'),
        userId: z
            .string()
            .optional()
            .describe('User ID for secure file isolation')
            .refine((userId) => validateUserId(userId), 'Invalid userId format. Only alphanumeric characters, hyphens, underscores, and dots are allowed.'),
        credentials: BaseCredentialsSchema,
    }),
    // Get file operation
    z.object({
        operation: z.literal('getFile').describe('Generate presigned download URL'),
        bucketName: z
            .string()
            .min(1, 'Bucket name is required')
            .describe('Name of the R2 bucket'),
        fileName: z
            .string()
            .min(1, 'File name is required')
            .describe('Name of the file to retrieve'),
        accountId: z
            .string()
            .optional()
            .describe('Cloudflare Account ID - can be provided via credentials'),
        region: z
            .string()
            .optional()
            .default('auto')
            .describe('AWS region for R2 storage (defaults to auto)'),
        expirationMinutes: z
            .number()
            .optional()
            .default(60)
            .describe('URL expiration time in minutes'),
        userId: z
            .string()
            .optional()
            .describe('User ID for secure file isolation')
            .refine((userId) => validateUserId(userId), 'Invalid userId format. Only alphanumeric characters, hyphens, underscores, and dots are allowed.'),
        credentials: BaseCredentialsSchema,
    }),
    // Delete file operation
    z.object({
        operation: z.literal('deleteFile').describe('Delete file from bucket'),
        bucketName: z
            .string()
            .min(1, 'Bucket name is required')
            .describe('Name of the R2 bucket'),
        fileName: z
            .string()
            .min(1, 'File name is required')
            .describe('Name of the file to delete'),
        accountId: z
            .string()
            .optional()
            .describe('Cloudflare Account ID - can be provided via credentials'),
        region: z
            .string()
            .optional()
            .default('auto')
            .describe('AWS region for R2 storage (defaults to auto)'),
        credentials: BaseCredentialsSchema,
    }),
    // Update file operation
    z.object({
        operation: z.literal('updateFile').describe('Update/replace file content'),
        bucketName: z
            .string()
            .min(1, 'Bucket name is required')
            .default('bubble-lab-bucket'),
        fileName: z
            .string()
            .min(1, 'File name is required')
            .describe('Name of the file to update'),
        accountId: z
            .string()
            .optional()
            .describe('Cloudflare Account ID - can be provided via credentials'),
        region: z
            .string()
            .optional()
            .default('auto')
            .describe('AWS region for R2 storage (defaults to auto)'),
        contentType: z.string().optional().describe('Content type for uploads'),
        fileContent: z
            .string()
            .min(1, 'File content is required for updates')
            .describe('Base64 encoded file content or raw text content'),
        credentials: BaseCredentialsSchema,
    }),
    // Get multiple upload URLs operation
    z.object({
        operation: z
            .literal('getMultipleUploadUrls')
            .describe('Generate multiple presigned upload URLs for PDF + page images'),
        bucketName: z
            .string()
            .min(1, 'Bucket name is required')
            .describe('Name of the R2 bucket'),
        pdfFileName: z
            .string()
            .min(1, 'PDF file name is required')
            .describe('Original filename for the PDF'),
        pageCount: z
            .number()
            .min(1, 'Page count must be at least 1')
            .describe('Number of pages to generate upload URLs for'),
        accountId: z
            .string()
            .optional()
            .describe('Cloudflare Account ID - can be provided via credentials'),
        region: z
            .string()
            .optional()
            .default('auto')
            .describe('AWS region for R2 storage (defaults to auto)'),
        expirationMinutes: z
            .number()
            .optional()
            .default(60)
            .describe('URL expiration time in minutes'),
        userId: z
            .string()
            .optional()
            .describe('User ID for secure file isolation')
            .refine((userId) => validateUserId(userId), 'Invalid userId format. Only alphanumeric characters, hyphens, underscores, and dots are allowed.'),
        credentials: BaseCredentialsSchema,
    }),
]);
// Define result schemas for different operations using discriminated union
const StorageResultSchema = z.discriminatedUnion('operation', [
    // Get upload URL result
    z.object({
        operation: z
            .literal('getUploadUrl')
            .describe('Generate presigned upload URL'),
        success: z.boolean().describe('Whether the operation was successful'),
        uploadUrl: z.string().optional().describe('Presigned upload URL'),
        fileName: z
            .string()
            .optional()
            .describe('Secure filename generated for the upload'),
        contentType: z.string().optional().describe('Content type of the file'),
        error: z.string().describe('Error message if operation failed'),
    }),
    // Get file result
    z.object({
        operation: z.literal('getFile').describe('Generate presigned download URL'),
        success: z.boolean().describe('Whether the operation was successful'),
        downloadUrl: z.string().optional().describe('Presigned download URL'),
        fileUrl: z.string().optional().describe('Direct file access URL'),
        fileName: z.string().optional().describe('Name of the file'),
        fileSize: z.number().optional().describe('File size in bytes'),
        contentType: z.string().optional().describe('Content type of the file'),
        lastModified: z
            .string()
            .optional()
            .describe('Last modified timestamp in ISO format'),
        error: z.string().describe('Error message if operation failed'),
    }),
    // Delete file result
    z.object({
        operation: z.literal('deleteFile').describe('Delete file from bucket'),
        success: z.boolean().describe('Whether the operation was successful'),
        fileName: z.string().optional().describe('Name of the deleted file'),
        deleted: z
            .boolean()
            .optional()
            .describe('Whether the file was successfully deleted'),
        error: z.string().describe('Error message if operation failed'),
    }),
    // Update file result
    z.object({
        operation: z
            .literal('updateFile')
            .describe('Update/replace file content and generate a new secure filename for the file'),
        success: z.boolean().describe('Whether the operation was successful'),
        fileName: z
            .string()
            .optional()
            .describe('Secure filename for the updated file (different from the original filename)'),
        updated: z
            .boolean()
            .optional()
            .describe('Whether the file was successfully updated'),
        contentType: z
            .string()
            .optional()
            .describe('Content type of the updated file'),
        error: z.string().describe('Error message if operation failed'),
    }),
    // Get multiple upload URLs result
    z.object({
        operation: z
            .literal('getMultipleUploadUrls')
            .describe('Generate multiple presigned upload URLs for PDF + page images'),
        success: z.boolean().describe('Whether the operation was successful'),
        pdfUploadUrl: z
            .string()
            .optional()
            .describe('Presigned upload URL for PDF'),
        pdfFileName: z.string().optional().describe('Secure filename for PDF'),
        pageUploadUrls: z
            .array(z.object({
            pageNumber: z.number().describe('Page number (1-indexed)'),
            uploadUrl: z.string().describe('Presigned upload URL for this page'),
            fileName: z.string().describe('Secure filename for this page image'),
        }))
            .optional()
            .describe('Array of upload URLs for page images'),
        error: z.string().describe('Error message if operation failed'),
    }),
]);
export class StorageBubble extends ServiceBubble {
    static service = 'cloudflare-r2';
    static authType = 'apikey';
    static bubbleName = 'storage';
    static type = 'service';
    static schema = StorageParamsSchema;
    static resultSchema = StorageResultSchema;
    static shortDescription = 'Cloudflare R2 storage operations for file management';
    static longDescription = `
    A comprehensive storage bubble for Cloudflare R2 operations.
    Use cases:
    - Generate presigned upload URLs for client-side file uploads
    - Get secure download URLs for file retrieval with authentication  
    - Delete files from R2 buckets
    - Update/replace files in R2 buckets (supports base64 encoded content for binary files like images)
    - Manage file access with time-limited URLs
  `;
    static alias = 'r2';
    s3Client = null;
    constructor(params = {
        operation: 'getUploadUrl',
        bucketName: 'my-bucket',
        fileName: 'example.txt',
    }, context) {
        super(params, context);
    }
    chooseCredential() {
        // Look for R2 API key in credentials
        const credentials = this.params.credentials;
        if (credentials && credentials[CredentialType.CLOUDFLARE_R2_ACCESS_KEY]) {
            return credentials[CredentialType.CLOUDFLARE_R2_ACCESS_KEY];
        }
        return undefined;
    }
    initializeS3Client() {
        const accessKeyId = this.chooseCredential();
        const secretAccessKey = this.params.credentials?.[CredentialType.CLOUDFLARE_R2_SECRET_KEY];
        if (!accessKeyId || !secretAccessKey) {
            throw new Error('R2 credentials not found. Need CLOUDFLARE_R2_ACCESS_KEY and CLOUDFLARE_R2_SECRET_KEY.');
        }
        // Get accountId from params or credentials
        const accountId = this.params.accountId ||
            this.params.credentials?.[CredentialType.CLOUDFLARE_R2_ACCOUNT_ID];
        if (!accountId) {
            throw new Error('Cloudflare Account ID is required. Provide via accountId parameter or CLOUDFLARE_R2_ACCOUNT_ID credential.');
        }
        const endpoint = `https://${accountId}.r2.cloudflarestorage.com`;
        this.s3Client = new S3Client({
            region: this.params.region,
            endpoint,
            credentials: {
                accessKeyId,
                secretAccessKey,
            },
        });
    }
    async testCredential() {
        //TODO: Implement credential addition for multiple credentials
        return true;
    }
    async performAction(context) {
        void context;
        const { operation } = this.params;
        try {
            this.initializeS3Client();
            if (!this.s3Client) {
                throw new Error('Failed to initialize S3 client');
            }
            const result = await (async () => {
                switch (operation) {
                    case 'getUploadUrl':
                        return await this.getUploadUrl(this.params);
                    case 'getFile':
                        return await this.getFile(this.params);
                    case 'deleteFile':
                        return await this.deleteFile(this.params);
                    case 'updateFile':
                        return await this.updateFile(this.params);
                    case 'getMultipleUploadUrls':
                        return await this.getMultipleUploadUrls(this.params);
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
    async getUploadUrl(params) {
        if (!this.s3Client) {
            console.error('[StorageBubble] S3 client not initialized');
            throw new Error('S3 client not initialized');
        }
        // Generate secure filename with timestamp and optional userId for isolation
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileExtension = params.fileName.split('.').pop() || 'bin';
        const baseName = params.fileName.replace(/\.[^/.]+$/, ''); // Remove extension
        const sanitizedBaseName = baseName.replace(/[^a-zA-Z0-9-_]/g, '_');
        // Include userId in path if available (from payload)
        const userPrefix = params.userId ? `${params.userId}/` : '';
        const secureFileName = `${userPrefix}${timestamp}-${sanitizedBaseName}.${fileExtension}`;
        console.log('[StorageBubble] Generated secure filename:', {
            original: params.fileName,
            secure: secureFileName,
        });
        const command = new PutObjectCommand({
            Bucket: params.bucketName,
            Key: secureFileName, // Use secure filename as the S3 key
            ContentType: params.contentType,
        });
        console.log('[StorageBubble] Generating signed URL...');
        try {
            const uploadUrl = await getSignedUrl(this.s3Client, command, {
                expiresIn: params.expirationMinutes * 60, // Convert minutes to seconds
            });
            console.log('[StorageBubble] Upload URL generated successfully:', {
                uploadUrl: uploadUrl.substring(0, 100) + '...',
                secureFileName: secureFileName,
            });
            return {
                operation: 'getUploadUrl',
                success: true,
                uploadUrl,
                fileName: secureFileName, // Return the secure filename
                contentType: params.contentType,
                error: '',
            };
        }
        catch (error) {
            console.error('[StorageBubble] Failed to generate upload URL:', error);
            throw error;
        }
    }
    async getFile(params) {
        if (!this.s3Client)
            throw new Error('S3 client not initialized');
        const command = new GetObjectCommand({
            Bucket: params.bucketName,
            Key: params.fileName,
        });
        const downloadUrl = await getSignedUrl(this.s3Client, command, {
            expiresIn: params.expirationMinutes * 60,
        });
        // Also get file metadata
        try {
            const headCommand = new HeadObjectCommand({
                Bucket: params.bucketName,
                Key: params.fileName,
            });
            const metadata = await this.s3Client.send(headCommand);
            return {
                operation: 'getFile',
                success: true,
                downloadUrl,
                fileUrl: downloadUrl,
                fileName: params.fileName,
                fileSize: metadata.ContentLength,
                contentType: metadata.ContentType,
                lastModified: metadata.LastModified?.toISOString(),
                error: '',
            };
        }
        catch {
            // If metadata fetch fails, still return the download URL
            return {
                operation: 'getFile',
                success: true,
                downloadUrl,
                fileUrl: downloadUrl,
                fileName: params.fileName,
                error: '',
            };
        }
    }
    async deleteFile(params) {
        if (!this.s3Client)
            throw new Error('S3 client not initialized');
        const command = new DeleteObjectCommand({
            Bucket: params.bucketName,
            Key: params.fileName,
        });
        await this.s3Client.send(command);
        return {
            operation: 'deleteFile',
            success: true,
            fileName: params.fileName,
            deleted: true,
            error: '',
        };
    }
    async updateFile(params) {
        if (!this.s3Client)
            throw new Error('S3 client not initialized');
        console.log('[StorageBubble] updateFile called with params:', {
            bucketName: params.bucketName,
            fileName: params.fileName,
            contentType: params.contentType,
        });
        // Generate secure filename with timestamp and optional userId for isolation
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileExtension = params.fileName.split('.').pop() || 'bin';
        const baseName = params.fileName.replace(/\.[^/.]+$/, ''); // Remove extension
        const sanitizedBaseName = baseName.replace(/[^a-zA-Z0-9-_]/g, '_');
        const secureFileName = `${timestamp}-${crypto.randomUUID()}-${sanitizedBaseName}.${fileExtension}`;
        // Handle base64 encoded content
        let bodyContent;
        const isBase64 = this.isBase64(params.fileContent);
        if (isBase64) {
            // Extract base64 content (remove data URL prefix if present)
            const base64Data = params.fileContent.replace(/^data:[^;]+;base64,/, '');
            bodyContent = Buffer.from(base64Data, 'base64');
            console.log('[StorageBubble] Decoded base64 content to Buffer');
        }
        else {
            // Treat as plain text
            bodyContent = params.fileContent;
            console.log('[StorageBubble] Treating content as plain text');
        }
        // For R2/S3, update is the same as upload - it replaces the entire object
        const command = new PutObjectCommand({
            Bucket: params.bucketName,
            Key: secureFileName,
            ContentType: params.contentType,
            Body: bodyContent,
        });
        await this.s3Client.send(command);
        return {
            operation: 'updateFile',
            success: true,
            fileName: secureFileName,
            updated: true,
            contentType: params.contentType,
            error: '',
        };
    }
    /**
     * Helper method to detect if a string is base64 encoded
     */
    isBase64(str) {
        try {
            // Check if it's a data URL (e.g., "data:image/png;base64,...")
            if (str.startsWith('data:') && str.includes('base64,')) {
                return true;
            }
            // Check if it's pure base64 (valid base64 characters, proper length)
            const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
            if (base64Regex.test(str) && str.length > 0) {
                // Try to decode and re-encode to verify it's valid base64
                try {
                    const decoded = Buffer.from(str, 'base64').toString('base64');
                    return decoded === str;
                }
                catch {
                    return false;
                }
            }
            return false;
        }
        catch {
            return false;
        }
    }
    async getMultipleUploadUrls(params) {
        if (!this.s3Client)
            throw new Error('S3 client not initialized');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const userPrefix = params.userId ? `${params.userId}/` : '';
        // Generate secure PDF filename
        const pdfExtension = params.pdfFileName.split('.').pop() || 'pdf';
        const pdfBaseName = params.pdfFileName
            .replace(/\\.[^/.]+$/, '')
            .replace(/[^a-zA-Z0-9-_]/g, '_');
        const securePdfFileName = `${userPrefix}${timestamp}-${pdfBaseName}.${pdfExtension}`;
        // Generate PDF upload URL
        const pdfCommand = new PutObjectCommand({
            Bucket: params.bucketName,
            Key: securePdfFileName,
            ContentType: 'application/pdf',
        });
        const pdfUploadUrl = await getSignedUrl(this.s3Client, pdfCommand, {
            expiresIn: params.expirationMinutes * 60,
        });
        // Generate page image upload URLs
        const pageUploadUrls = [];
        for (let pageNum = 1; pageNum <= params.pageCount; pageNum++) {
            const pageFileName = `${userPrefix}${timestamp}-${pdfBaseName}_page${pageNum}.jpeg`;
            const pageCommand = new PutObjectCommand({
                Bucket: params.bucketName,
                Key: pageFileName,
                ContentType: 'image/jpeg',
            });
            const pageUploadUrl = await getSignedUrl(this.s3Client, pageCommand, {
                expiresIn: params.expirationMinutes * 60,
            });
            pageUploadUrls.push({
                pageNumber: pageNum,
                uploadUrl: pageUploadUrl,
                fileName: pageFileName,
            });
        }
        return {
            operation: 'getMultipleUploadUrls',
            success: true,
            pdfUploadUrl,
            pdfFileName: securePdfFileName,
            pageUploadUrls,
            error: '',
        };
    }
}
//# sourceMappingURL=storage.js.map