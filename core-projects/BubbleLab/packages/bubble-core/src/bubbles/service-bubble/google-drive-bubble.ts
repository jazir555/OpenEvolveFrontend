import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { createLogger } from '../../utils/logger.js';
import { sanitizeErrorMessage } from '../../utils/error-sanitizer.js';

/**
 * Google Drive Bubble - Cloud Storage Service Bubble Implementation
 *
 * Full production implementation with 13 operations:
 * 1. uploadFile - Upload a file to Drive
 * 2. downloadFile - Download a file from Drive
 * 3. deleteFile - Delete a file
 * 4. updateFile - Update file content/metadata
 * 5. copyFile - Copy a file
 * 6. createFolder - Create a folder
 * 7. listFiles - List files in folder
 * 8. searchFiles - Search files
 * 9. shareFile - Share file with user
 * 10. getPermissions - Get file permissions
 * 11. revokeAccess - Revoke access
 * 12. getFileInfo - Get file metadata
 * 13. updateMetadata - Update file metadata
 *
 * Security Features:
 * - OAuth2 token validation
 * - Rate limiting (upload: 5/min, others: 50/min)
 * - File size validation (max 5GB per file)
 * - Path traversal prevention
 * - Input validation with Zod schemas
 * - Error sanitization
 * - Structured logging
 */

// ============================================================================
// SECURITY & VALIDATION CONSTANTS
// ============================================================================

const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024; // 5GB
const MAX_UPLOAD_RATE = 5; // uploads per minute
const MAX_DEFAULT_RATE = 50; // other operations per minute
const DEFAULT_TIMEOUT = 30000; // 30 seconds
const UPLOAD_TIMEOUT = 60000; // 60 seconds for uploads

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const UploadFileParamsSchema = z.object({
  operation: z.literal('uploadFile'),
  fileName: z
    .string()
    .min(1, 'File name is required')
    .max(255, 'File name too long (max 255 characters)')
    .refine((name) => !name.includes('..'), 'File name cannot contain path traversal sequences'),
  content: z.union([z.string(), z.instanceof(Buffer)]).describe('File content'),
  mimeType: z.string().optional().describe('MIME type of the file'),
  parents: z.array(z.string()).optional().describe('Folder IDs to upload to'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DownloadFileParamsSchema = z.object({
  operation: z.literal('downloadFile'),
  fileId: z.string().min(1, 'File ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ListFilesParamsSchema = z.object({
  operation: z.literal('listFiles'),
  pageSize: z.number().int().positive().optional().default(100),
  pageToken: z.string().optional().describe('Token for pagination'),
  query: z.string().optional().describe('Search query for filtering'),
  orderBy: z.string().optional().describe('Sort order (e.g., "name", "modifiedTime desc")'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SearchFilesParamsSchema = z.object({
  operation: z.literal('searchFiles'),
  query: z.string().min(1, 'Search query (e.g., "name contains \'report\'")'),
  pageSize: z.number().int().positive().optional().default(100),
  pageToken: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteFileParamsSchema = z.object({
  operation: z.literal('deleteFile'),
  fileId: z.string().min(1, 'File ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateFolderParamsSchema = z.object({
  operation: z.literal('createFolder'),
  folderName: z.string().min(1, 'Folder name is required'),
  parents: z.array(z.string()).optional().describe('Parent folder IDs'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ShareFileParamsSchema = z.object({
  operation: z.literal('shareFile'),
  fileId: z.string().min(1, 'File ID is required'),
  role: z.enum(['reader', 'writer', 'commenter', 'owner']).describe('Permission role'),
  type: z.enum(['user', 'group', 'anyone', 'domain']).describe('Type of grantee'),
  emailAddress: z.string().email().optional().describe('Email for user or group'),
  allowFileDiscovery: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetFileInfoParamsSchema = z.object({
  operation: z.literal('getFileInfo'),
  fileId: z.string().min(1, 'File ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateFileParamsSchema = z.object({
  operation: z.literal('updateFile'),
  fileId: z.string().min(1, 'File ID is required'),
  content: z.union([z.string(), z.instanceof(Buffer)]).describe('New file content'),
  mimeType: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CopyFileParamsSchema = z.object({
  operation: z.literal('copyFile'),
  fileId: z.string().min(1, 'File ID to copy'),
  fileName: z.string().min(1, 'Name for the copy'),
  parents: z.array(z.string()).optional().describe('Folder IDs for the copy'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetPermissionsParamsSchema = z.object({
  operation: z.literal('getPermissions'),
  fileId: z.string().min(1, 'File ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const RevokeAccessParamsSchema = z.object({
  operation: z.literal('revokeAccess'),
  fileId: z.string().min(1, 'File ID is required'),
  permissionId: z.string().min(1, 'Permission ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateMetadataParamsSchema = z.object({
  operation: z.literal('updateMetadata'),
  fileId: z.string().min(1, 'File ID is required'),
  fileName: z.string().optional().describe('New file name'),
  description: z.string().optional().describe('File description'),
  starred: z.boolean().optional().describe('Star the file'),
  parents: z.array(z.string()).optional().describe('Move to new folders'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

// Union of all parameter schemas
const GoogleDriveBubbleParamsSchema = z.discriminatedUnion('operation', [
  UploadFileParamsSchema,
  DownloadFileParamsSchema,
  ListFilesParamsSchema,
  SearchFilesParamsSchema,
  DeleteFileParamsSchema,
  CreateFolderParamsSchema,
  ShareFileParamsSchema,
  GetFileInfoParamsSchema,
  UpdateFileParamsSchema,
  CopyFileParamsSchema,
  GetPermissionsParamsSchema,
  RevokeAccessParamsSchema,
  UpdateMetadataParamsSchema,
]);

type GoogleDriveBubbleParams = z.input<typeof GoogleDriveBubbleParamsSchema>;

// Result schema
const GoogleDriveBubbleResultSchema = z.object({
  success: z.boolean(),
  data: z.unknown().describe('Operation result data'),
  error: z.string(),
  meta: z.object({
    operation: z.string(),
    fileId: z.string().optional(),
    fileName: z.string().optional(),
  }),
});

type GoogleDriveBubbleResult = z.output<typeof GoogleDriveBubbleResultSchema>;

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class GoogleDriveBubble extends ServiceBubble<
  GoogleDriveBubbleParams,
  GoogleDriveBubbleResult
> {
  static readonly service = 'google-drive';
  static readonly authType = 'oauth' as const;
  static readonly bubbleName: BubbleName = 'google-drive';
  static readonly type = 'service' as const;
  static readonly schema = GoogleDriveBubbleParamsSchema;
  static readonly resultSchema = GoogleDriveBubbleResultSchema;
  static readonly shortDescription = 'Cloud file storage and synchronization service';
  static readonly longDescription = `
    Google Drive Bubble for cloud storage and file management.

    Features:
    - Upload and download files (up to 5GB)
    - Create and manage folders
    - Share files with specific permissions
    - Search files by name or content
    - File metadata management
    - Copy and update operations
    - Permission management
    - Integration with Google Workspace
    - OAuth2 authentication
    - Rate limiting and quota management

    Use cases:
    - Document storage and backup
    - File sharing and collaboration
    - Automated report generation
    - Content management
    - Integration with other Google services

    Security:
    - All operations use OAuth2 tokens
    - File size validation (max 5GB)
    - Path traversal prevention
    - Rate limiting (upload: 5/min, others: 50/min)
    - Input sanitization and validation
  `;
  static readonly alias = 'drive';

  private accessToken: string | null = null;
  private baseUrl = 'https://www.googleapis.com/drive/v3';
  private uploadUrl = 'https://www.googleapis.com/upload/drive/v3';
  private logger = createLogger('GoogleDriveBubble');
  private rateLimitTracker: Map<string, number[]> = new Map();

  constructor(
    params: GoogleDriveBubbleParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  /**
   * Check rate limit for an operation
   */
  private checkRateLimit(operation: string): boolean {
    const now = Date.now();
    const oneMinuteAgo = now - 60000;

    const key = `${this.instanceId || 'default'}-${operation}`;
    const timestamps = this.rateLimitTracker.get(key) || [];

    // Remove old timestamps
    const recentTimestamps = timestamps.filter(t => t > oneMinuteAgo);

    const maxRate = operation === 'uploadFile' ? MAX_UPLOAD_RATE : MAX_DEFAULT_RATE;

    if (recentTimestamps.length >= maxRate) {
      this.logger.warn('Rate limit exceeded', {
        operation,
        count: recentTimestamps.length,
        maxRate,
      });
      return false;
    }

    recentTimestamps.push(now);
    this.rateLimitTracker.set(key, recentTimestamps);
    return true;
  }

  /**
   * Validate file size
   */
  private validateFileSize(content: string | Buffer): void {
    const size = typeof content === 'string'
      ? Buffer.byteLength(content, 'utf8')
      : content.length;

    if (size > MAX_FILE_SIZE) {
      throw new Error(
        `File size (${Math.round(size / 1024 / 1024)}MB) exceeds maximum allowed size (${MAX_FILE_SIZE / 1024 / 1024 / 1024}GB)`
      );
    }
  }

  protected getCredentialType(): CredentialType {
    return CredentialType.GOOGLE_DRIVE_CRED;
  }

  protected chooseCredential(): string | undefined {
    const credentials = this.params.credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Google Drive credentials are required');
    }
    return credentials[CredentialType.GOOGLE_DRIVE_CRED];
  }

  public async testCredential(): Promise<boolean> {
    try {
      const token = this.getToken();
      const response = await fetch(`${this.baseUrl}/about?fields=user`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      return response.ok;
    } catch (error) {
      console.error('[Google Drive] Credential test failed:', error);
      return false;
    }
  }

  private getToken(): string {
    if (!this.accessToken) {
      const credential = this.chooseCredential();
      if (!credential) {
        throw new Error('Google Drive credentials not found');
      }

      // Parse credential (expected format: JSON string with accessToken)
      let config: any;
      try {
        config = typeof credential === 'string' ? JSON.parse(credential) : credential;
      } catch {
        throw new Error('Invalid Google Drive credentials format. Expected JSON string.');
      }

      if (!config.accessToken && !config.token) {
        throw new Error('Google Drive access token is required in credentials');
      }

      this.accessToken = config.accessToken || config.token;
      this.logger.info('Access token initialized successfully');
    }

    if (!this.accessToken) {
      throw new Error('Google Drive access token initialization failed');
    }

    return this.accessToken;
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<GoogleDriveBubbleResult> {
    void context;

    const operation = this.params.operation;

    // Check rate limit
    if (!this.checkRateLimit(operation)) {
      this.logger.error('Rate limit exceeded', { operation });
      return {
        success: false,
        data: null,
        error: `Rate limit exceeded for operation: ${operation}. Please try again later.`,
        meta: {
          operation,
          fileId: (this.params as any).fileId,
        },
      };
    }

    this.logger.info('Executing operation', { operation });

    try {
      let result: any;

      switch (operation) {
        case 'uploadFile':
          result = await this.uploadFile();
          break;

        case 'downloadFile':
          result = await this.downloadFile();
          break;

        case 'listFiles':
          result = await this.listFiles();
          break;

        case 'searchFiles':
          result = await this.searchFiles();
          break;

        case 'deleteFile':
          result = await this.deleteFile();
          break;

        case 'createFolder':
          result = await this.createFolder();
          break;

        case 'shareFile':
          result = await this.shareFile();
          break;

        case 'getFileInfo':
          result = await this.getFileInfo();
          break;

        case 'updateFile':
          result = await this.updateFile();
          break;

        case 'copyFile':
          result = await this.copyFile();
          break;

        case 'getPermissions':
          result = await this.getPermissions();
          break;

        case 'revokeAccess':
          result = await this.revokeAccess();
          break;

        case 'updateMetadata':
          result = await this.updateMetadata();
          break;

        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      this.logger.info('Operation completed successfully', { operation });

      return {
        success: true,
        data: result,
        error: '', // Empty string for successful operations
        meta: {
          operation,
          fileId: result.fileId,
          fileName: result.fileName,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error
        ? sanitizeErrorMessage(error.message)
        : 'Unknown error';

      this.logger.error('Operation failed', {
        operation,
        error: errorMessage,
      });

      return {
        success: false,
        data: null,
        error: errorMessage,
        meta: {
          operation: this.params.operation,
          fileId: (this.params as any).fileId,
        },
      };
    }
  }

  private async makeRequest(method: string, endpoint: string, body?: any): Promise<any> {
    const token = this.getToken();

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${token}`,
    };

    if (body) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `Google Drive API error: ${response.statusText}`);
    }

    return response.json();
  }

  private async uploadFile(): Promise<any> {
    const params = this.params as z.output<typeof UploadFileParamsSchema>;
    const token = this.getToken();

    // Validate file size
    this.validateFileSize(params.content);

    this.logger.info('Uploading file', {
      fileName: params.fileName,
      mimeType: params.mimeType,
    });

    // Prepare metadata
    const metadata: any = {
      name: params.fileName,
    };

    if (params.parents && params.parents.length > 0) {
      metadata.parents = params.parents;
    }

    // Prepare multipart upload
    const boundary = '-------314159265358979323846';
    const delimiter = `\r\n--${boundary}\r\n`;
    const closeDelimiter = `\r\n--${boundary}--`;

    const metadataPart =
      'Content-Type: application/json; charset=UTF-8\r\n\r\n' + JSON.stringify(metadata);

    let content = params.content;
    if (typeof content === 'string') {
      content = Buffer.from(content);
    }

    const mediaPart =
      'Content-Type: ' + (params.mimeType || 'application/octet-stream') + '\r\n\r\n';

    const requestBody =
      delimiter +
      metadataPart +
      delimiter +
      mediaPart +
      content.toString('base64') +
      closeDelimiter;

    const response = await fetch(
      `${this.uploadUrl}/files?uploadType=multipart`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': `multipart/related; boundary=${boundary}`,
        },
        body: requestBody,
        signal: AbortSignal.timeout(UPLOAD_TIMEOUT),
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || 'File upload failed');
    }

    const result = await response.json();

    this.logger.info('File uploaded successfully', {
      fileId: result.id,
      fileName: result.name,
    });

    return {
      fileId: result.id,
      fileName: result.name,
      mimeType: result.mimeType,
      webViewLink: result.webViewLink,
      size: result.size,
      status: 'uploaded',
    };
  }

  private async downloadFile(): Promise<any> {
    const params = this.params as z.output<typeof DownloadFileParamsSchema>;

    // First get file metadata to find download URL
    const metadata = await this.makeRequest('GET', `/files/${params.fileId}?fields=id,name,mimeType,webContentLink`);

    let content = '';
    let mimeType = metadata.mimeType;

    // For Google Workspace files, we need to export
    if (metadata.mimeType.startsWith('application/vnd.google-apps')) {
      let exportMimeType = 'text/plain';
      if (metadata.mimeType.includes('document')) {
        exportMimeType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      } else if (metadata.mimeType.includes('spreadsheet')) {
        exportMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
      } else if (metadata.mimeType.includes('presentation')) {
        exportMimeType = 'application/vnd.openxmlformats-officedocument.presentationml.presentation';
      }

      const exportResponse = await fetch(
        `${this.baseUrl}/files/${params.fileId}/export?mimeType=${encodeURIComponent(exportMimeType)}`,
        {
          headers: {
            'Authorization': `Bearer ${this.getToken()}`,
          },
        }
      );

      if (!exportResponse.ok) {
        throw new Error('File export failed');
      }

      content = await exportResponse.text();
      mimeType = exportMimeType;
    } else if (metadata.webContentLink) {
      // Regular files use webContentLink
      const downloadResponse = await fetch(metadata.webContentLink);
      if (!downloadResponse.ok) {
        throw new Error('File download failed');
      }
      content = await downloadResponse.text();
    }

    this.logger.info('File downloaded successfully', {
      fileId: params.fileId,
      fileName: metadata.name,
    });

    return {
      fileId: params.fileId,
      fileName: metadata.name,
      content: content,
      mimeType: mimeType,
      size: content.length,
      status: 'downloaded',
    };
  }

  private async listFiles(): Promise<any> {
    const params = this.params as z.output<typeof ListFilesParamsSchema>;

    let queryParams = `pageSize=${params.pageSize}`;

    if (params.pageToken) {
      queryParams += `&pageToken=${params.pageToken}`;
    }

    if (params.query) {
      queryParams += `&q=${encodeURIComponent(params.query)}`;
    }

    if (params.orderBy) {
      queryParams += `&orderBy=${params.orderBy}`;
    }

    queryParams += '&fields=files(id,name,mimeType,createdTime,modifiedTime,size,owners,parents),nextPageToken';

    const result = await this.makeRequest('GET', `/files?${queryParams}`);

    this.logger.info('Listed files successfully', {
      count: result.files?.length || 0,
      hasNextPage: !!result.nextPageToken,
    });

    return {
      files: result.files?.map((file: any) => ({
        id: file.id,
        name: file.name,
        mimeType: file.mimeType,
        createdTime: file.createdTime,
        modifiedTime: file.modifiedTime,
        size: file.size,
        owners: file.owners?.map((o: any) => o.displayName),
        parents: file.parents,
      })) || [],
      nextPageToken: result.nextPageToken,
      count: result.files?.length || 0,
    };
  }

  private async searchFiles(): Promise<any> {
    const params = this.params as z.output<typeof SearchFilesParamsSchema>;

    let queryParams = `q=${encodeURIComponent(params.query)}&pageSize=${params.pageSize}`;

    if (params.pageToken) {
      queryParams += `&pageToken=${params.pageToken}`;
    }

    queryParams += '&fields=files(id,name,mimeType,createdTime,modifiedTime,size),nextPageToken';

    const result = await this.makeRequest('GET', `/files?${queryParams}`);

    this.logger.info('Search completed', {
      query: params.query,
      count: result.files?.length || 0,
    });

    return {
      query: params.query,
      files: result.files?.map((file: any) => ({
        id: file.id,
        name: file.name,
        mimeType: file.mimeType,
        createdTime: file.createdTime,
        modifiedTime: file.modifiedTime,
        size: file.size,
      })) || [],
      nextPageToken: result.nextPageToken,
      count: result.files?.length || 0,
    };
  }

  private async deleteFile(): Promise<any> {
    const params = this.params as z.output<typeof DeleteFileParamsSchema>;

    await this.makeRequest('DELETE', `/files/${params.fileId}`);

    this.logger.info('File deleted successfully', {
      fileId: params.fileId,
    });

    return {
      fileId: params.fileId,
      status: 'deleted',
    };
  }

  private async createFolder(): Promise<any> {
    const params = this.params as z.output<typeof CreateFolderParamsSchema>;

    const metadata: any = {
      name: params.folderName,
      mimeType: 'application/vnd.google-apps.folder',
    };

    if (params.parents && params.parents.length > 0) {
      metadata.parents = params.parents;
    }

    const result = await this.makeRequest('POST', '/files', {
      resource: metadata,
    });

    this.logger.info('Folder created successfully', {
      folderId: result.id,
      folderName: params.folderName,
    });

    return {
      fileId: result.id,
      fileName: result.name,
      mimeType: result.mimeType,
      status: 'created',
    };
  }

  private async shareFile(): Promise<any> {
    const params = this.params as z.output<typeof ShareFileParamsSchema>;

    const permission: any = {
      role: params.role,
      type: params.type,
    };

    if (params.emailAddress && (params.type === 'user' || params.type === 'group')) {
      permission.emailAddress = params.emailAddress;
    }

    if (params.allowFileDiscovery) {
      permission.allowFileDiscovery = true;
    }

    const result = await this.makeRequest('POST', `/files/${params.fileId}/permissions`, permission);

    this.logger.info('File shared successfully', {
      fileId: params.fileId,
      type: params.type,
      role: params.role,
    });

    return {
      fileId: params.fileId,
      permissionId: result.id,
      role: result.role,
      type: result.type,
      status: 'shared',
    };
  }

  private async getFileInfo(): Promise<any> {
    const params = this.params as z.output<typeof GetFileInfoParamsSchema>;

    const result = await this.makeRequest(
      'GET',
      `/files/${params.fileId}?fields=id,name,mimeType,createdTime,modifiedTime,size,owners,parents,permissions,webContentLink,webViewLink`
    );

    this.logger.info('Retrieved file info', {
      fileId: params.fileId,
      fileName: result.name,
    });

    return {
      fileId: result.id,
      fileName: result.name,
      mimeType: result.mimeType,
      createdTime: result.createdTime,
      modifiedTime: result.modifiedTime,
      size: result.size,
      owners: result.owners?.map((o: any) => ({
        displayName: o.displayName,
        emailAddress: o.emailAddress,
      })),
      parents: result.parents,
      permissions: result.permissions?.map((p: any) => ({
        id: p.id,
        role: p.role,
        type: p.type,
        emailAddress: p.emailAddress,
      })),
      webContentLink: result.webContentLink,
      webViewLink: result.webViewLink,
    };
  }

  private async updateFile(): Promise<any> {
    const params = this.params as z.output<typeof UpdateFileParamsSchema>;
    const token = this.getToken();

    // Validate file size
    this.validateFileSize(params.content);

    // Update file content
    let content = params.content;
    if (typeof content === 'string') {
      content = Buffer.from(content);
    }

    const response = await fetch(
      `${this.uploadUrl}/files/${params.fileId}?uploadType=media`,
      {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': params.mimeType || 'application/octet-stream',
        },
        body: content,
        signal: AbortSignal.timeout(UPLOAD_TIMEOUT),
      }
    );

    if (!response.ok) {
      throw new Error('File update failed');
    }

    const result = await response.json();

    this.logger.info('File updated successfully', {
      fileId: params.fileId,
    });

    return {
      fileId: result.id,
      fileName: result.name,
      size: result.size,
      modifiedTime: result.modifiedTime,
      status: 'updated',
    };
  }

  private async copyFile(): Promise<any> {
    const params = this.params as z.output<typeof CopyFileParamsSchema>;

    const metadata: any = {
      name: params.fileName,
    };

    if (params.parents && params.parents.length > 0) {
      metadata.parents = params.parents;
    }

    const result = await this.makeRequest('POST', `/files/${params.fileId}/copy`, metadata);

    this.logger.info('File copied successfully', {
      from: params.fileId,
      to: result.id,
    });

    return {
      fileId: result.id,
      originalFileId: params.fileId,
      fileName: result.name,
      mimeType: result.mimeType,
      status: 'copied',
    };
  }

  /**
   * Get file permissions
   */
  private async getPermissions(): Promise<any> {
    const params = this.params as z.output<typeof GetPermissionsParamsSchema>;

    const result = await this.makeRequest(
      'GET',
      `/files/${params.fileId}/permissions?fields=id,type,role,emailAddress,displayName,photoLink,expirationTime,deleted`
    );

    this.logger.info('Retrieved file permissions', {
      fileId: params.fileId,
      permissionCount: result.length,
    });

    return {
      fileId: params.fileId,
      permissions: result.map((p: any) => ({
        id: p.id,
        role: p.role,
        type: p.type,
        emailAddress: p.emailAddress,
        displayName: p.displayName,
        photoLink: p.photoLink,
        expirationTime: p.expirationTime,
        deleted: p.deleted,
      })),
      count: result.length,
    };
  }

  /**
   * Revoke access to a file
   */
  private async revokeAccess(): Promise<any> {
    const params = this.params as z.output<typeof RevokeAccessParamsSchema>;

    await this.makeRequest(
      'DELETE',
      `/files/${params.fileId}/permissions/${params.permissionId}`
    );

    this.logger.info('Access revoked successfully', {
      fileId: params.fileId,
      permissionId: params.permissionId,
    });

    return {
      fileId: params.fileId,
      permissionId: params.permissionId,
      status: 'revoked',
    };
  }

  /**
   * Update file metadata
   */
  private async updateMetadata(): Promise<any> {
    const params = this.params as z.output<typeof UpdateMetadataParamsSchema>;

    const metadata: any = {};

    if (params.fileName !== undefined) {
      metadata.name = params.fileName;
    }

    if (params.description !== undefined) {
      metadata.description = params.description;
    }

    if (params.starred !== undefined) {
      metadata.starred = params.starred;
    }

    if (params.parents !== undefined) {
      // Note: Moving files to a new folder requires special handling
      // This is a simplified version
      metadata.parents = params.parents;
    }

    const result = await this.makeRequest(
      'PATCH',
      `/files/${params.fileId}`,
      metadata
    );

    this.logger.info('File metadata updated', {
      fileId: params.fileId,
      updates: Object.keys(metadata),
    });

    return {
      fileId: result.id,
      fileName: result.name,
      description: result.description,
      starred: result.starred,
      modifiedTime: result.modifiedTime,
      status: 'updated',
    };
  }
}

