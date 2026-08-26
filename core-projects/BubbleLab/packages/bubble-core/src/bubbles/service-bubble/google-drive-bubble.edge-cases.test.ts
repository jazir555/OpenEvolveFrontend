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
 *
 * NOTE ON THE BUBBLE CONTRACT:
 *  - The Bubble base constructor does NOT throw on invalid params; it records a
 *    `validationError` and `action()` returns `{ success: false, error }`.
 *    Invalid-input cases therefore assert on `await bubble.action()`.
 *  - `performAction()` returns `{ success, data, error, meta }` for valid input.
 *  - Multi-gigabyte payloads are represented by a Buffer-like stub with a
 *    reported `length`, because V8 caps strings at ~512MB (so
 *    `'x'.repeat(5 * 1024 ** 3)` throws RangeError rather than testing anything).
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleDriveBubble } from './google-drive-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

/** Build a mock `Response`. `json` is always present: makeRequest() always calls it. */
function mockResponse(
  body: unknown,
  init?: { ok?: boolean; status?: number; text?: string }
) {
  return {
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    statusText: init?.ok === false ? 'Error' : 'OK',
    json: async () => body,
    text: async () => init?.text ?? JSON.stringify(body),
  } as unknown as Response;
}

/** Buffer-like stub with an arbitrary reported length (no real allocation). */
function fakeBufferOfSize(size: number): Buffer {
  const stub = Object.create(Buffer.prototype) as Buffer;
  Object.defineProperty(stub, 'length', { value: size });
  Object.defineProperty(stub, 'toString', { value: () => 'stub-content' });
  return stub;
}

const FIVE_GB = 5 * 1024 * 1024 * 1024;

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
      it('should reject an empty file name', async () => {
        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: '',
          content: 'test content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('File name is required');
        expect(global.fetch).not.toHaveBeenCalled();
      });

      it('should handle maximum length file name (255 chars)', async () => {
        // The schema caps fileName at 255 characters inclusive.
        const maxFileName = 'x'.repeat(251) + '.txt';
        expect(maxFileName).toHaveLength(255);

        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: maxFileName })
        );

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: maxFileName,
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should reject a file name over 255 chars', async () => {
        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'x'.repeat(256),
          content: 'content',
          credentials: mockCredentials,
        });

        const result = await driveBubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('File name too long');
      });

      it('should handle minimum length file name (1 char)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'x' })
        );

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

        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: unicodeFileName })
        );

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
        const specialChars = 'file<>:|"?*.txt';

        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'file_txt' })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'file.txt' })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'TEST.TXT' })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'file.tar.gz' })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'file.txt' })
        );

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
      it('should accept exactly the 5GB limit', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', size: FIVE_GB })
        );

        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'large.txt',
          content: fakeBufferOfSize(FIVE_GB),
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should reject a size just over the 5GB limit', async () => {
        driveBubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: 'large.txt',
          content: fakeBufferOfSize(FIVE_GB + 1),
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(false);
        expect(result.error).toContain('exceeds maximum allowed size');
        expect(global.fetch).not.toHaveBeenCalled();
      });

      it('should handle empty file (0 bytes)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', size: 0 })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', size: 1 })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({
            id: '1kJnhbUi_mYQ-Zx4q3qE3qQ3qQ3qQ3qQ3qQ3qQ3qQ',
            name: 'test.txt',
          })
        );

        driveBubble = new GoogleDriveBubble({
          operation: 'getFileInfo',
          fileId: '1kJnhbUi_mYQ-Zx4q3qE3qQ3qQ3qQ3qQ3qQ3qQ3qQ',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(true);
      });

      it('should handle invalid file ID format', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse(
            { error: { message: 'File not found' } },
            { ok: false, status: 404 }
          )
        );

        driveBubble = new GoogleDriveBubble({
          operation: 'getFileInfo',
          fileId: 'invalid_id_format!',
          credentials: mockCredentials,
        });

        const result = await driveBubble.performAction();

        expect(result.success).toBe(false);
      });

      it('should reject a null file ID', async () => {
        driveBubble = new GoogleDriveBubble({
          operation: 'getFileInfo',
          fileId: null as unknown as string,
          credentials: mockCredentials,
        });

        const result = await driveBubble.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('fileId');
        expect(global.fetch).not.toHaveBeenCalled();
      });
    });

    describe('Array Boundaries', () => {
      it('should handle empty parents array', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: 'file_123', name: 'test.txt', parents: [] })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({
            id: 'file_123',
            name: 'test.txt',
            parents: ['folder_1', 'folder_2'],
          })
        );

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
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ files: [], nextPageToken: null })
        );

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

        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ files: thousandFiles, nextPageToken: null })
        );

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
    it('should succeed when the response arrives before the timeout', async () => {
      vi.mocked(fetch).mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () => resolve(mockResponse({ id: 'file_123', name: 'test.txt' })),
              20
            );
          })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should surface a request timeout as a controlled failure', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Request timeout'));

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('should surface an API-level rate limit (HTTP 429)', async () => {
      // Five successful uploads on distinct instances (rate limiting is
      // per-instance, so this exercises the API error path, not the local guard).
      for (let i = 0; i < 5; i++) {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: `file_${i}`, name: `test${i}.txt` })
        );

        const bubble = new GoogleDriveBubble({
          operation: 'uploadFile',
          fileName: `test${i}.txt`,
          content: 'content',
          credentials: mockCredentials,
        });

        const ok = await bubble.performAction();
        expect(ok.success).toBe(true);
      }

      // 6th upload is rejected by the API with a rate-limit error.
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'Rate limit exceeded' } },
          { ok: false, status: 429 }
        )
      );

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

    it('should trip the local per-instance upload rate limiter', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      for (let i = 0; i < 5; i++) {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: `file_${i}`, name: 'test.txt' })
        );
        const ok = await driveBubble.performAction();
        expect(ok.success).toBe(true);
      }

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Rate limit exceeded');
      expect(global.fetch).toHaveBeenCalledTimes(5);
    });

    it('should handle slow upload speeds', async () => {
      vi.mocked(fetch).mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () => resolve(mockResponse({ id: 'file_123', name: 'test.txt' })),
              60
            );
          })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const startTime = Date.now();
      const result = await driveBubble.performAction();
      const duration = Date.now() - startTime;

      expect(result.success).toBe(true);
      expect(duration).toBeGreaterThanOrEqual(50);
    });
  });

  describe('Error Path Coverage', () => {
    it('should handle 401 Unauthorized', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'Invalid credentials' } },
          { ok: false, status: 401 }
        )
      );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'Insufficient permissions' } },
          { ok: false, status: 403 }
        )
      );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'File not found' } },
          { ok: false, status: 404 }
        )
      );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'File with same name already exists' } },
          { ok: false, status: 409 }
        )
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'existing.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('already exists');
    });

    it('should handle 412 Precondition Failed', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'Precondition failed' } },
          { ok: false, status: 412 }
        )
      );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'Internal server error' } },
          { ok: false, status: 500 }
        )
      );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'Service unavailable' } },
          { ok: false, status: 503 }
        )
      );

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
        status: 200,
        json: async () => {
          throw new SyntaxError('Invalid JSON');
        },
        text: async () => 'invalid json{{{',
      } as unknown as Response);

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
      vi.mocked(fetch).mockResolvedValueOnce(
        // Missing 'id' field
        mockResponse({ name: 'test.txt' })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.fileId).toBeUndefined();
    });

    it('should handle extra unexpected fields in response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({
          id: 'file_123',
          name: 'test.txt',
          unexpected_field: 'value',
          another_unexpected: 123,
        })
      );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({
          id: 'file_123',
          name: null,
          mimeType: 'text/plain',
        })
      );

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
        .mockResolvedValueOnce(
          mockResponse({
            id: 'file_123',
            name: 'document',
            mimeType: 'application/vnd.google-apps.document',
            exportLinks: {
              'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                'https://export_url',
            },
          })
        )
        .mockResolvedValueOnce(mockResponse({}, { text: 'exported content' }));

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

      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({
          id: 'file_123',
          createdTime: leapYearDate,
          modifiedTime: leapYearDate,
        })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'getFileInfo',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.createdTime).toBe(leapYearDate);
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

      const result = await driveBubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('path traversal');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('should prevent path traversal with encoded characters', async () => {
      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: '..%2F..%2F..%2Fetc%2Fpasswd',
        content: 'malicious',
        credentials: mockCredentials,
      });

      const result = await driveBubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('path traversal');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('should handle null byte injection', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({ id: 'file_123', name: 'test.txt.jpg' })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'test.txt\x00.jpg',
        content: 'content',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      // Name reported by Drive carries no null byte.
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

      const result = await driveBubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('emailAddress');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('should handle XSS in metadata', async () => {
      const xssPayload = '<script>alert("xss")</script>';

      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({
          id: 'file_123',
          name: 'test.txt',
          description: xssPayload,
        })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'updateMetadata',
        fileId: 'file_123',
        description: xssPayload,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      // Payload is carried as an opaque string, never executed/interpolated.
      expect(typeof result.data.description).toBe('string');
    });

    it('should handle SQL-injection-shaped search queries', async () => {
      const sqlInjection = "name = 'test' OR '1'='1'";

      vi.mocked(fetch).mockResolvedValueOnce(mockResponse({ files: [] }));

      driveBubble = new GoogleDriveBubble({
        operation: 'searchFiles',
        query: sqlInjection,
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.data.files).toEqual([]);
      // Query is URL-encoded before being sent.
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain(
        encodeURIComponent(sqlInjection)
      );
    });
  });

  describe('Concurrency Edge Cases', () => {
    it('should handle simultaneous uploads to same folder', async () => {
      const promises = [];

      for (let i = 0; i < 10; i++) {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({
            id: `file_${i}`,
            name: `test${i}.txt`,
            parents: ['folder_123'],
          })
        );

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

      vi.mocked(fetch)
        .mockResolvedValueOnce(mockResponse({ id: fileId, name: 'updated1.txt' }))
        .mockResolvedValueOnce(
          mockResponse({ id: fileId, name: 'updated2.txt' })
        );

      const bubble1 = new GoogleDriveBubble({
        operation: 'updateMetadata',
        fileId,
        fileName: 'updated1.txt',
        credentials: mockCredentials,
      });

      const bubble2 = new GoogleDriveBubble({
        operation: 'updateMetadata',
        fileId,
        fileName: 'updated2.txt',
        credentials: mockCredentials,
      });

      const [result1, result2] = await Promise.all([
        bubble1.performAction(),
        bubble2.performAction(),
      ]);

      expect(result1.success).toBe(true);
      expect(result2.success).toBe(true);
    });

    it('should handle race conditions in delete operations', async () => {
      // File already deleted
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse(
          { error: { message: 'File not found' } },
          { ok: false, status: 404 }
        )
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'deleteFile',
        fileId: 'file_123',
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });
  });

  describe('Memory/Performance Edge Cases', () => {
    it('should handle large file uploads efficiently', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({ id: 'file_123', size: 100 * 1024 * 1024 })
      );

      driveBubble = new GoogleDriveBubble({
        operation: 'uploadFile',
        fileName: 'large.txt',
        content: fakeBufferOfSize(100 * 1024 * 1024), // 100MB reported
        credentials: mockCredentials,
      });

      const result = await driveBubble.performAction();

      expect(result.success).toBe(true);
    });

    it('should handle many small files', async () => {
      const promises = [];

      for (let i = 0; i < 100; i++) {
        vi.mocked(fetch).mockResolvedValueOnce(
          mockResponse({ id: `file_${i}`, name: `test${i}.txt` })
        );

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
      vi.mocked(fetch).mockResolvedValueOnce(
        mockResponse({
          files: Array.from({ length: 100 }, (_, i) => ({
            id: `file_${i}`,
            name: `file${i}.txt`,
          })),
          nextPageToken: 'token_123',
        })
      );

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
