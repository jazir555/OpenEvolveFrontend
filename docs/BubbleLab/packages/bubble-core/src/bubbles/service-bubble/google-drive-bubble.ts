import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * GoogleDriveBubble - Google Drive file operations
 */
export class GoogleDriveBubble extends ServiceBubble<GoogleDriveParams, GoogleDriveResult> {
  bubbleName = 'google-drive';
  type = 'service';
  alias = 'GoogleDrive';
  credentialType = 'google_drive_api_key';

  params = {
    credentials: z.any(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { drive } = await import('@googleapis/drive');
    const auth = await import('google-auth-library');
    const authClient = auth.JWT.fromJSON(this.params.credentials);
    this.client = drive({ version: 'v3', auth: authClient });
  }

  async uploadFile(params: { name: string; content: string; mimeType?: string; parentId?: string }): Promise<GoogleDriveResult> {
    try {
      const media = {
        mimeType: params.mimeType || 'text/plain',
        body: params.content
      };
      const result = await this.client.files.create({
        requestBody: {
          name: params.name,
          parents: params.parentId ? [params.parentId] : undefined
        },
        media: media
      });
      return { success: true, file: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async downloadFile(params: { fileId: string }): Promise<GoogleDriveResult> {
    try {
      const result = await this.client.files.get({
        fileId: params.fileId,
        alt: 'media'
      });
      return { success: true, content: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listFiles(params: { pageSize?: number; folderId?: string; query?: string }): Promise<GoogleDriveResult> {
    try {
      const query = params.query || (params.folderId ? `'${params.folderId}' in parents` : undefined);
      const result = await this.client.files.list({
        pageSize: params.pageSize || 10,
        q: query
      });
      return { success: true, files: result.data.files || [] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async searchFiles(params: { query: string; pageSize?: number }): Promise<GoogleDriveResult> {
    try {
      const result = await this.client.files.list({
        q: params.query,
        pageSize: params.pageSize || 10
      });
      return { success: true, files: result.data.files || [] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createFolder(params: { name: string; parentId?: string }): Promise<GoogleDriveResult> {
    try {
      const result = await this.client.files.create({
        requestBody: {
          name: params.name,
          mimeType: 'application/vnd.google-apps.folder',
          parents: params.parentId ? [params.parentId] : undefined
        }
      });
      return { success: true, folder: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async shareFile(params: { fileId: string; role: 'reader' | 'writer' | 'commenter'; emailAddress?: string }): Promise<GoogleDriveResult> {
    try {
      const result = await this.client.permissions.create({
        fileId: params.fileId,
        requestBody: {
          role: params.role,
          type: params.emailAddress ? 'user' : 'anyone',
          emailAddress: params.emailAddress
        }
      });
      return { success: true, permission: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteFile(params: { fileId: string }): Promise<GoogleDriveResult> {
    try {
      await this.client.files.delete({ fileId: params.fileId });
      return { success: true, deleted: params.fileId };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async updateFile(params: { fileId: string; content: string; mimeType?: string }): Promise<GoogleDriveResult> {
    try {
      const result = await this.client.files.update({
        fileId: params.fileId,
        media: {
          mimeType: params.mimeType || 'text/plain',
          body: params.content
        }
      });
      return { success: true, file: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface GoogleDriveParams {
  credentials: any;
  timeout?: number;
}

export interface GoogleDriveResult {
  success: boolean;
  file?: any;
  files?: any[];
  folder?: any;
  content?: string;
  permission?: any;
  deleted?: string;
  error?: string;
}
