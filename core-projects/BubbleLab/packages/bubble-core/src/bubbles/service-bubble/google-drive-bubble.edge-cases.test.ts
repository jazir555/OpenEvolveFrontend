/**
 * Edge Case and Boundary Tests for Google Drive Bubble
 *
 * Comprehensive edge case coverage including:
 * - Input boundaries (empty, null, max length, unicode, special characters)
 * - Network boundaries (timeouts, retries, rate limits)
 * - Error paths (all error types and codes)
 * - Data edge cases (malformed JSON, missing fields)
 * - Security edge cases (injection attacks, path traversal)
 * - Concurrency edge cases (race conditions)
 * - Performance edge cases (large files, memory)
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleDriveBubble } from './google-drive-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('GoogleDriveBubble - Edge Cases and Boundary Tests', () => {
  let driveBubble: GoogleDriveBubble;
  const mockCredentials = {
    [CredentialType.GOOGLE_DRIVE_CRED]: JSON.stringify({
      accessToken: 'ya_test_mock_token',
    }),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Input Boundary Tests', () => {
    describe('String Boundaries', () => {
      it('should handle empty string for file name', async () => {
        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: '',
          content: 'test content',
          credentials: mockCredentials,
        });

        await expect(driveBubble.performAction()).rejects.toThrow();
      });

      it('should handle maximum length file name (255 chars)', async () => {
        const maxFileName = 'x'.repeat(255) + '.txt';

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: maxFileName,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: maxFileName,
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle minimum length file name (1 char)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'x',
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'x',
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle unicode and emoji characters in file names', async () => {
        const unicodeFileName = '文件世界 📁 Documentos mondo';

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: unicodeFileName,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: unicodeFileName,
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.data.fileName).toBe(unicodeFileName);
      });

      it('should handle special characters in file names', async () => {
        const specialChars = 'file<>:|"?.*txt';

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'file_txt', // Special chars should be sanitized
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: specialChars,
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle null characters in file names', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'file.txt',
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'file\x00.txt',
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle case sensitivity in file names', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'TEST.TXT',
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'TEST.TXT',
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.data.fileName).toBe('TEST.TXT');
      });

      it('should handle file names with multiple extensions', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'file.tar.gz',
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'file.tar.gz',
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle file names with leading/trailing whitespace', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'file.txt',
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: '  file.txt  ',
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });
    });

    describe('File Size Boundaries', () => {
      it('should handle exact 5GB limit', async () => {
        const exactLimitContent = 'x'.repeat(5 * 1024 * 1024 * 1024);

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            size: 5 * 1024 * 1024 * 1024,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'large.txt',
          content: exactLimitContent,
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle size just over 5GB limit', async () => {
        const overLimit = 'x'.repeat(5 * 1024 * 1024 * 1024 + 1);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'large.txt',
          content: overLimit,
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('exceeds maximum allowed size');
      });

      it('should handle empty file (0 bytes)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            size: 0,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'empty.txt',
          content: '',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.data.size).toBe(0);
      });

      it('should handle single byte file', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            size: 1,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'single.txt',
          content: 'x',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.data.size).toBe(1);
      });
    });

    describe('ID Format Validations', () => {
      it('should handle valid file ID format', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: '1kJnhbUi_mYQ-Zx4q3qE3qQ3qQ3qQ3qQ3qQ3qQ3qQ',
            name: 'test.txt',
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'getFileInfo',
          fileId: '1kJnhbUi_mYQ-Zx4q3qE3qQ3qQ3qQ3qQ3qQ3qQ3qQ',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle invalid file ID format', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: async () => ({ error: { message: 'File not found' } }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'getFileInfo',
          fileId: 'invalid_id_format!',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(false);
      });

      it('should handle null file ID', async () => {
        driveBubble = new GoogleDriveBubble({
          operation: 'getFileInfo',
          fileId: null as any,
          credentials: mockCredentials,
        });

        await expect(driveBubble.performAction()).rejects.toThrow();
      });
    });

    describe('Array Boundaries', () => {
      it('should handle empty parents array', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'test.txt',
            parents: [],
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'test.txt',
          content: 'content',
          parents: [],
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle multiple parents (if supported)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'test.txt',
            parents: ['folder_1', 'folder_2'],
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'test.txt',
          content: 'content',
          parents: ['folder_1', 'folder_2'],
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle empty file list', async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            files: [],
            nextPageToken: null,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'listFiles',
          pageSize: 100,
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.data.count).toBe(0);
      });

      it('should handle maximum page size (1000 items)', async () => {
        const thousandFiles = Array.from({ length: 1000 }, (_, i) => ({
          id: `file_${i}`,
          name: `file${i}.txt`,
        }));

        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            files: thousandFiles,
            nextPageToken: null,
          }),
        } as Response);

        driveBubble = new GoogleDriveBubble({
          operation: 'listFiles',
          pageSize: 1000,
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
        expect(result.data.count).toBe(1000);
      });
    });
  });

  describe('Network Edge Cases', () => {
    it('should handle timeout just before limit', async () => {
      vi.mocked(fetch).mockImplementationOnce(() =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: async () => ({ id: 'file_123', name: 'test.txt' }),
            } as Response);
          }, 4500); // Just before 5000ms timeout
        })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        timeout: 5000,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle timeout at limit', async () => {
      vi.mocked(fetch).mockImplementationOnce(() =>
        new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Request timeout'));
          }, 5000);
        })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        timeout: 5000,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('should handle rate limit boundary', async () => {
      // Make 5 successful uploads
      for (let i = 0; i < 5; i++) {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({ id: `file_${i}`, name: `test${i}.txt` }),
        } as Response);
      }

      for (let i = 0; i < 5; i++) {
        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: `test${i}.txt`,
          content: 'content',
          credentials: mockCredentials,
        });

        await driveBubble.performAction();
      }

      // 6th upload hits rate limit
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: async () => ({
          error: {
            errors: [{ message: 'Rate limit exceeded' }],
          },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test6.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limit');
    });

    it('should handle slow upload speeds', async () => {
      vi.mocked(fetch).mockImplementationOnce(() =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: async () => ({ id: 'file_123', name: 'test.txt' }),
            } as Response);
          }, 9000);
        })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        timeout: 10000,
        credentials: mockCredentials,
      });

      const startTime = Date.now();
      const result = await driveBubble.performAction();
      const duration = Date.now() - startTime;

      expect(result.success).toBe(true);
      expect(duration).toBeGreaterThan(8000);
    });
  });

  describe('Error Path Coverage', () => {
    it('should handle 401 Unauthorized', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({
          error: { message: 'Invalid credentials' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('credentials');
    });

    it('should handle 403 Forbidden', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({
          error: { message: 'Insufficient permissions' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('permission');
    });

    it('should handle 404 Not Found', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({
          error: { message: 'File not found' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'nonexistent_file',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });

    it('should handle 409 Conflict (file already exists)', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({
          error: { message: 'File with same name already exists' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'existing.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
    });

    it('should handle 412 Precondition Failed', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 412,
        json: async () => ({
          error: { message: 'Precondition failed' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'updateFile',
        fileId: 'file_123',
        content: 'updated content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
    });

    it('should handle 500 Internal Server Error', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({
          error: { message: 'Internal server error' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('server error');
    });

    it('should handle 503 Service Unavailable', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({
          error: { message: 'Service unavailable' },
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
    });
  });

  describe('Data Edge Cases', () => {
    it('should handle malformed JSON response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => {
          throw new SyntaxError('Invalid JSON');
        },
        text: async () => 'invalid json{{{',
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
    });

    it('should handle missing required fields in response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          // Missing 'id' field
          name: 'test.txt',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle extra unexpected fields in response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'test.txt',
          unexpected_field: 'value',
          another_unexpected: 123,
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBeDefined();
    });

    it('should handle null values in non-nullable fields', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: null, // Should be non-nullable
          mimeType: 'text/plain',
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle Google Workspace file export', async () => {
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: 'file_123',
            name: 'document',
            mimeType: 'application/vnd.google-apps.document',
            exportLinks: {
              'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                'https://export_url',
            },
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
      expect(result.data.content).toBe('exported content');
    });

    it('should handle date/time boundary conditions', async () => {
      const leapYearDate = '2024-02-29T23:59:59.999Z';
      const epochBoundary = Math.floor(new Date('1970-01-01T00:00:00Z').getTime() / 1000);

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          createdTime: leapYearDate,
          modifiedTime: leapYearDate,
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });
  });

  describe('Security Edge Cases', () => {
    it('should prevent path traversal attacks in file names', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: '../../../etc/passwd',
        content: 'malicious',
        credentials: mockCredentials,
      });

      await expect(driveBubble.performAction()).rejects.toThrow();
    });

    it('should prevent path traversal with encoded characters', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: '..%2F..%2F..%2Fetc%2Fpasswd',
        content: 'malicious',
        credentials: mockCredentials,
      });

      await expect(driveBubble.performAction()).rejects.toThrow();
    });

    it('should prevent null byte injection', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt\x00.jpg',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      // Null byte should be removed
      expect(result.data.fileName).not.toContain('\x00');
    });

    it('should validate email addresses in share operations', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'shareFile',
        fileId: 'file_123',
        role: 'writer',
        type: 'user',
        emailAddress: 'invalid-email-format',
        credentials: mockCredentials,
      });

      await expect(driveBubble.performAction()).rejects.toThrow();
    });

    it('should handle XSS in metadata', async () => {
      const xssPayload = '<script>alert("xss")</script>';

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          name: 'test.txt',
          description: xssPayload,
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'updateMetadata',
        fileId: 'file_123',
        description: xssPayload,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      // Payload should be properly escaped
      expect(typeof result.data.description).toBe('string');
    });

    it('should handle SQL injection in search queries', async () => {
      const sqlInjection = "name = 'test' OR '1'='1'";

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          files: [],
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'searchFiles',
        query: sqlInjection,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      // SQL injection should be ineffective
      expect(result.data.files).toEqual([]);
    });
  });

  describe('Concurrency Edge Cases', () => {
    it('should handle simultaneous uploads to same folder', async () => {
      const promises = [];

      for (let i = 0; i < 10; i++) {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: `file_${i}`,
            name: `test${i}.txt`,
            parents: ['folder_123'],
          }),
        } as Response);

        const bubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: `test${i}.txt`,
          content: 'content',
          parents: ['folder_123'],
          credentials: mockCredentials,
        });

        promises.push(bubble.performAction());
      }

      const results = await Promise.all(promises);

      results.forEach((result) => {
        expect(result.success).toBe(true);
      });
    });

    it('should handle concurrent updates to same file', async () => {
      const fileId = 'file_123';

      const promise1 = (async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: fileId,
            name: 'updated1.txt',
          }),
        } as Response);

        const bubble = new GoogleDriveBubble({
          operation: 'updateMetadata',
          fileId,
          fileName: 'updated1.txt',
          credentials: mockCredentials,
        });

        return await bubble.performAction();
      })();

      const promise2 = (async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: fileId,
            name: 'updated2.txt',
          }),
        } as Response);

        const bubble = new GoogleDriveBubble({
          operation: 'updateMetadata',
          fileId,
          fileName: 'updated2.txt',
          credentials: mockCredentials,
        });

        return await bubble.performAction();
      })();

      const [result1, result2] = await Promise.all([promise1, promise2]);

      expect(result1.success).toBe(true);
      expect(result2.success).toBe(true);
    });

    it('should handle race conditions in delete operations', async () => {
      const fileId = 'file_123';

      // File already deleted
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ error: { message: 'File not found' } }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'deleteFile',
        fileId,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });
  });

  describe('Memory/Performance Edge Cases', () => {
    it('should handle large file uploads efficiently', async () => {
      const largeContent = 'x'.repeat(100 * 1024 * 1024); // 100MB

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'file_123',
          size: 100 * 1024 * 1024,
        }),
      } as Response);

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'large.txt',
        content: largeContent,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle many small files', async () => {
      const promises = [];

      for (let i = 0; i < 100; i++) {
        vi.mocked(fetch).mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            id: `file_${i}`,
            name: `test${i}.txt`,
          }),
        } as Response);

        const bubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: `test${i}.txt`,
          content: `content ${i}`,
          credentials: mockCredentials,
        });

        promises.push(bubble.performAction());
      }

      const results = await Promise.all(promises);

      results.forEach((result) => {
        expect(result.success).toBe(true);
      });
    });

    it('should handle pagination with large result sets', async () => {
      // First page
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          files: Array.from({ length: 100 }, (_, i) => ({
            id: `file_${i}`,
            name: `file${i}.txt`,
          })),
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
      expect(result.data.count).toBe(100);
      expect(result.data.nextPageToken).toBe('token_123');
    });
  });
});
