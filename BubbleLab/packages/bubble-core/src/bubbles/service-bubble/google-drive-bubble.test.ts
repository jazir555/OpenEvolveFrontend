/**
 * Comprehensive tests for Google Drive Bubble
 *
 * Tests all 13 operations:
 * 1. uploadFile
 * 2. downloadFile
 * 3. deleteFile
 * 4. updateFile
 * 5. copyFile
 * 6. createFolder
 * 7. listFiles
 * 8. searchFiles
 * 9. shareFile
 * 10. getPermissions
 * 11. revokeAccess
 * 12. getFileInfo
 * 13. updateMetadata
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleDriveBubble } from './google-drive-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('GoogleDriveBubble', () => {
  let driveBubble: GoogleDriveBubble;
  const mockCredentials = {
    [CredentialType.GOOGLE_DRIVE_CRED]: JSON.stringify({
      accessToken: 'ya_test_mock_token',
    }),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  describe('Operation 1: uploadFile', () => {
    it('should upload a file successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'test.txt',
          mimeType: 'text/plain',
          webViewLink: 'https://drive.google.com/file/d/file_123',
          size: 1024,
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'Hello, World!',
        mimeType: 'text/plain',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('file_123');
      expect(result.data.fileName).toBe('test.txt');
      expect(result.data.status).toBe('uploaded');
    });

    it('should validate file size limits', async () => {
      const largeContent = 'x'.repeat(6 * 1024 * 1024 * 1024); // 6GB

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'large.txt',
        content: largeContent,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('exceeds maximum allowed size');
    });

    it('should prevent path traversal attacks', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: '../../../etc/passwd',
        content: 'malicious',
        credentials: mockCredentials,
      });

      await expect(driveBubble.performAction()).rejects.toThrow();
    });

    it('should handle upload rate limiting', async () => {
      // Make 5 successful uploads
      for (let i = 0; i < 5; i++) {
        vi.mocked(global.fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: `file_${i}`, name: `test${i}.txt` }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: `test${i}.txt`,
          content: 'content',
          credentials: mockCredentials,
        });

        await driveBubble.performAction();
      }

      // 6th upload should fail due to rate limit
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test6.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limit exceeded');
    });
  });

  describe('Operation 2: downloadFile', () => {
    it('should download a file successfully', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'test.txt',
            mimeType: 'text/plain',
            webContentLink: 'https://drive.google.com/uc?id=file_123',
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          text: async () => 'Hello, World!',
        } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'downloadFile',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('file_123');
      expect(result.data.content).toBe('Hello, World!');
      expect(result.data.status).toBe('downloaded');
    });

    it('should export Google Docs format', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'document',
            mimeType: 'application/vnd.google-apps.document',
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          text: async () => 'exported content',
        } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'downloadFile',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.mimeType).toContain('openxmlformats');
    });
  });

  describe('Operation 3: deleteFile', () => {
    it('should delete a file successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 204,
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'deleteFile',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('file_123');
      expect(result.data.status).toBe('deleted');
    });
  });

  describe('Operation 4: updateFile', () => {
    it('should update file content successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'test.txt',
          size: 2048,
          modifiedTime: '2024-01-01T12:00:00Z',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'updateFile',
        fileId: 'file_123',
        content: 'Updated content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.status).toBe('updated');
      expect(result.data.size).toBe(2048);
    });
  });

  describe('Operation 5: copyFile', () => {
    it('should copy a file successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_456',
          name: 'copy_of_test.txt',
          mimeType: 'text/plain',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'copyFile',
        fileId: 'file_123',
        fileName: 'copy_of_test.txt',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('file_456');
      expect(result.data.originalFileId).toBe('file_123');
      expect(result.data.status).toBe('copied');
    });
  });

  describe('Operation 6: createFolder', () => {
    it('should create a folder successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'folder_123',
          name: 'My Folder',
          mimeType: 'application/vnd.google-apps.folder',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'createFolder',
        folderName: 'My Folder',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('folder_123');
      expect(result.data.mimeType).toBe('application/vnd.google-apps.folder');
      expect(result.data.status).toBe('created');
    });

    it('should create folder with parent', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'folder_456',
          name: 'Subfolder',
          parents: ['folder_123'],
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'createFolder',
        folderName: 'Subfolder',
        parents: ['folder_123'],
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('folder_456');
    });
  });

  describe('Operation 7: listFiles', () => {
    it('should list files successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          files: [
            {
              id: 'file_1',
              name: 'test1.txt',
              mimeType: 'text/plain',
              createdTime: '2024-01-01T10:00:00Z',
              modifiedTime: '2024-01-01T11:00:00Z',
              size: '1024',
              owners: [{ displayName: 'User' }],
            },
            {
              id: 'file_2',
              name: 'test2.txt',
              mimeType: 'text/plain',
              createdTime: '2024-01-01T10:00:00Z',
              modifiedTime: '2024-01-01T11:00:00Z',
              size: '2048',
              owners: [{ displayName: 'User' }],
            },
          ],
          nextPageToken: 'token_123',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'listFiles',
        pageSize: 100,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.files).toHaveLength(2);
      expect(result.data.count).toBe(2);
      expect(result.data.nextPageToken).toBe('token_123');
    });

    it('should handle pagination', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          files: [
            {
              id: 'file_3',
              name: 'test3.txt',
              mimeType: 'text/plain',
            },
          ],
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'listFiles',
        pageSize: 100,
        pageToken: 'token_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.files).toHaveLength(1);
    });
  });

  describe('Operation 8: searchFiles', () => {
    it('should search files successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          files: [
            {
              id: 'file_123',
              name: 'report.pdf',
              mimeType: 'application/pdf',
            },
          ],
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'searchFiles',
        query: "name contains 'report'",
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.files).toHaveLength(1);
      expect(result.data.files[0].name).toBe('report.pdf');
    });
  });

  describe('Operation 9: shareFile', () => {
    it('should share file with user successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'perm_123',
          role: 'writer',
          type: 'user',
          emailAddress: 'user@example.com',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'shareFile',
        fileId: 'file_123',
        role: 'writer',
        type: 'user',
        emailAddress: 'user@example.com',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.permissionId).toBe('perm_123');
      expect(result.data.role).toBe('writer');
      expect(result.data.status).toBe('shared');
    });

    it('should validate email address', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'shareFile',
        fileId: 'file_123',
        role: 'writer',
        type: 'user',
        emailAddress: 'invalid-email',
        credentials: mockCredentials,
      });

      await expect(driveBubble.performAction()).rejects.toThrow();
    });
  });

  describe('Operation 10: getPermissions', () => {
    it('should get file permissions successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => [
          {
            id: 'perm_1',
            role: 'owner',
            type: 'user',
            emailAddress: 'owner@example.com',
          },
          {
            id: 'perm_2',
            role: 'writer',
            type: 'user',
            emailAddress: 'writer@example.com',
          },
        ],
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getPermissions',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.permissions).toHaveLength(2);
      expect(result.data.count).toBe(2);
    });
  });

  describe('Operation 11: revokeAccess', () => {
    it('should revoke access successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 204,
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'revokeAccess',
        fileId: 'file_123',
        permissionId: 'perm_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.permissionId).toBe('perm_123');
      expect(result.data.status).toBe('revoked');
    });
  });

  describe('Operation 12: getFileInfo', () => {
    it('should get file info successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'test.txt',
          mimeType: 'text/plain',
          createdTime: '2024-01-01T10:00:00Z',
          modifiedTime: '2024-01-01T11:00:00Z',
          size: '1024',
          owners: [{ displayName: 'Owner', emailAddress: 'owner@example.com' }],
          permissions: [
            {
              id: 'perm_1',
              role: 'owner',
              type: 'user',
              emailAddress: 'owner@example.com',
            },
          ],
          webContentLink: 'https://drive.google.com/uc?id=file_123',
          webViewLink: 'https://drive.google.com/file/d/file_123',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBe('file_123');
      expect(result.data.fileName).toBe('test.txt');
      expect(result.data.owners).toHaveLength(1);
      expect(result.data.permissions).toHaveLength(1);
    });
  });

  describe('Operation 13: updateMetadata', () => {
    it('should update file metadata successfully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'updated_name.txt',
          description: 'Updated description',
          starred: true,
          modifiedTime: '2024-01-01T12:00:00Z',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'updateMetadata',
        fileId: 'file_123',
        fileName: 'updated_name.txt',
        description: 'Updated description',
        starred: true,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileName).toBe('updated_name.txt');
      expect(result.data.description).toBe('Updated description');
      expect(result.data.starred).toBe(true);
      expect(result.data.status).toBe('updated');
    });
  });

  describe('Error Handling', () => {
    it('should handle authentication errors', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ error: { message: 'Invalid credentials' } }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid credentials');
    });

    it('should handle file not found', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ error: { message: 'File not found' } }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'nonexistent',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('File not found');
    });

    it('should handle network errors', async () => {
      vi.mocked(global.fetch).mockRejectedValueOnce(new Error('Network error'));

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });
  });

  describe('Credential Testing', () => {
    it('should test valid credentials', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ user: { displayName: 'Test User' } }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'test',
        credentials: mockCredentials,
      });

      const isValid = await driveBubble.testCredential();

      expect(isValid).toBe(true);
    });

    it('should test invalid credentials', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'test',
        credentials: { [CredentialType.GOOGLE_DRIVE_CRED]: 'invalid' },
      });

      const isValid = await driveBubble.testCredential();

      expect(isValid).toBe(false);
    });
  });
});
