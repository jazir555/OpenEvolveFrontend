/**
 * Test Suite for GoogleDriveBubble (google-drive.ts — the Drive + Docs bubble)
 *
 * The previous contents of this file were a generated placeholder that tested a
 * non-existent API (`instance.authenticate()`, `instance.execute()`, an `env`
 * context) and did not even parse. It has been rewritten against the real
 * bubble:
 *  - discriminated-union params keyed on `operation`
 *    (upload_file / download_file / list_files / create_folder / delete_file /
 *     get_file_info / share_file / move_file / get_doc / replace_text / copy_doc)
 *  - `action()` returns a BubbleResult wrapper `{ success, data, error }`
 *  - invalid params are captured at construction (NOT thrown) and surfaced as a
 *    controlled error from `action()`
 *  - all network access goes through the global `fetch`, mocked here
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleDriveBubble } from './google-drive.js';
import { CredentialType } from '@bubblelab/shared-schemas';

const mockCredentials = {
  [CredentialType.GOOGLE_DRIVE_CRED]: 'ya29.test-mock-token',
};

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
    text: async () => JSON.stringify(body),
    arrayBuffer: async () => new TextEncoder().encode(JSON.stringify(body))
      .buffer as ArrayBuffer,
  } as unknown as Response;
}

function binaryResponse(text: string, contentType = 'text/plain') {
  const bytes = new TextEncoder().encode(text);
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: new Headers({ 'content-type': contentType }),
    json: async () => JSON.parse(text),
    text: async () => text,
    arrayBuffer: async () =>
      bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength
      ) as ArrayBuffer,
  } as unknown as Response;
}

function errorResponse(status: number, message: string) {
  return {
    ok: false,
    status,
    statusText: message,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => ({ error: { message } }),
    text: async () => JSON.stringify({ error: { message } }),
    arrayBuffer: async () => new ArrayBuffer(0),
  } as unknown as Response;
}

/** Read the `q` search param from a recorded fetch call (URLSearchParams encodes spaces as `+`). */
function queryParamOfCall(callIndex: number, name: string): string | null {
  const url = new URL(vi.mocked(global.fetch).mock.calls[callIndex][0] as string);
  return url.searchParams.get(name);
}

describe('GoogleDriveBubble (google-drive)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ========================================
  // METADATA
  // ========================================
  describe('Bubble metadata', () => {
    it('exposes the expected static metadata', () => {
      expect(GoogleDriveBubble.bubbleName).toBe('google-drive');
      expect(GoogleDriveBubble.type).toBe('service');
      expect(GoogleDriveBubble.service).toBe('google-drive');
      expect(GoogleDriveBubble.authType).toBe('oauth');
      expect(GoogleDriveBubble.alias).toBe('gdrive');
      expect(GoogleDriveBubble.schema).toBeDefined();
      expect(GoogleDriveBubble.resultSchema).toBeDefined();
    });

    it('defaults to a list_files operation when constructed with no params', () => {
      const bubble = new GoogleDriveBubble();
      expect(bubble.currentParams.operation).toBe('list_files');
    });

    it('excludes credentials from currentParams', () => {
      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });
      expect(
        (bubble.currentParams as Record<string, unknown>).credentials
      ).toBeUndefined();
    });

    it('applies schema defaults for list_files', () => {
      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });
      const params = bubble.currentParams as Record<string, unknown>;
      expect(params.max_results).toBe(100);
      expect(params.include_folders).toBe(true);
      expect(params.order_by).toBe('modifiedTime desc');
    });
  });

  // ========================================
  // INPUT VALIDATION
  // ========================================
  // The base constructor records validation failures rather than throwing, so
  // these assert on the controlled error returned by action().
  describe('Input Validation', () => {
    it('returns a controlled error for an unknown operation', async () => {
      const bubble = new GoogleDriveBubble({
        // @ts-expect-error deliberately invalid operation
        operation: 'teleport_file',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('returns a controlled error when download_file is missing file_id', async () => {
      const bubble = new GoogleDriveBubble({
        // @ts-expect-error file_id is required
        operation: 'download_file',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('file_id');
    });

    it('returns a controlled error for an empty file_id', async () => {
      const bubble = new GoogleDriveBubble({
        operation: 'get_file_info',
        file_id: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('File ID is required');
    });

    it('returns a controlled error for an empty folder name', async () => {
      const bubble = new GoogleDriveBubble({
        operation: 'create_folder',
        name: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Folder name is required');
    });

    it('rejects an invalid email address for share_file', async () => {
      const bubble = new GoogleDriveBubble({
        operation: 'share_file',
        file_id: 'file_1',
        email_address: 'not-an-email',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('email_address');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('rejects max_results above the documented ceiling', async () => {
      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        max_results: 5000,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('max_results');
    });
  });

  // ========================================
  // list_files
  // ========================================
  describe('list_files', () => {
    it('returns the file list on success', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          files: [
            { id: 'f1', name: 'a.txt', mimeType: 'text/plain' },
            { id: 'f2', name: 'b.txt', mimeType: 'text/plain' },
          ],
          nextPageToken: 'page-2',
        })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        max_results: 10,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.operation).toBe('list_files');
      expect(result.data?.files).toHaveLength(2);
      expect(result.data?.total_count).toBe(2);
      expect(result.data?.next_page_token).toBe('page-2');
    });

    it('sends the bearer token and always filters out trashed files', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ files: [] })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      await bubble.action();

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toContain('/files?');
      expect(queryParamOfCall(0, 'q')).toContain('trashed = false');
      expect((init.headers as Record<string, string>).Authorization).toBe(
        'Bearer ya29.test-mock-token'
      );
    });

    it('scopes the query to a folder when folder_id is provided', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ files: [] })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        folder_id: 'folder_abc',
        credentials: mockCredentials,
      });

      await bubble.action();

      const url = vi.mocked(global.fetch).mock.calls[0][0] as string;
      expect(url).toContain('/files?');
      expect(queryParamOfCall(0, 'q')).toContain("'folder_abc' in parents");
    });

    it('excludes folders when include_folders is false', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ files: [] })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        include_folders: false,
        credentials: mockCredentials,
      });

      await bubble.action();

      expect(queryParamOfCall(0, 'q')).toContain(
        "mimeType != 'application/vnd.google-apps.folder'"
      );
    });

    it('returns an empty list when Drive returns no files', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse({}));

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.files).toEqual([]);
      expect(result.data?.total_count).toBe(0);
    });
  });

  // ========================================
  // upload_file
  // ========================================
  describe('upload_file', () => {
    it('uploads plain text content', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'file_new', name: 'notes.txt', mimeType: 'text/plain' })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'upload_file',
        name: 'notes.txt',
        content: 'hello world',
        mimeType: 'text/plain',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.file?.id).toBe('file_new');

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toContain('/upload/drive/v3/files');
      expect(url).toContain('uploadType=multipart');
      expect(init.method).toBe('POST');
      expect(
        (init.headers as Record<string, string>)['Content-Type']
      ).toContain('multipart/related');
    });

    it('places the file in a parent folder when requested', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'file_new', name: 'notes.txt', mimeType: 'text/plain' })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'upload_file',
        name: 'notes.txt',
        content: 'hello world',
        parent_folder_id: 'folder_abc',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      const body = vi.mocked(global.fetch).mock.calls[0][1]?.body as Buffer;
      expect(body.toString('utf8')).toContain('folder_abc');
    });

    it('returns a controlled failure for empty content', async () => {
      const bubble = new GoogleDriveBubble({
        operation: 'upload_file',
        name: 'notes.txt',
        content: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('File content is required');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('maps a 401 upload failure to an authentication message', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'Unauthorized')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'upload_file',
        name: 'notes.txt',
        content: 'hello',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Authentication failed');
    });

    it('maps a 403 upload failure to a permission message', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(403, 'Forbidden')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'upload_file',
        name: 'notes.txt',
        content: 'hello',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Permission denied');
    });
  });

  // ========================================
  // download_file
  // ========================================
  describe('download_file', () => {
    it('downloads a regular text file as plain text', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(
          jsonResponse({ name: 'notes.txt', mimeType: 'text/plain' })
        )
        .mockResolvedValueOnce(binaryResponse('file contents here'));

      const bubble = new GoogleDriveBubble({
        operation: 'download_file',
        file_id: 'file_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.content).toBe('file contents here');
      expect(result.data?.filename).toBe('notes.txt');
      expect(result.data?.mimeType).toBe('text/plain');
    });

    it('requires an export format for Google Workspace files', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          name: 'doc',
          mimeType: 'application/vnd.google-apps.document',
        })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'download_file',
        file_id: 'file_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Export format is required');
    });

    it('exports a Google Workspace file with the requested format', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(
          jsonResponse({
            name: 'doc',
            mimeType: 'application/vnd.google-apps.document',
          })
        )
        .mockResolvedValueOnce(binaryResponse('exported text'));

      const bubble = new GoogleDriveBubble({
        operation: 'download_file',
        file_id: 'file_1',
        export_format: 'text/plain',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.content).toBe('exported text');
      expect(result.data?.mimeType).toBe('text/plain');

      const exportUrl = vi.mocked(global.fetch).mock.calls[1][0] as string;
      expect(exportUrl).toContain('/export?');
      expect(exportUrl).toContain(encodeURIComponent('text/plain'));
    });

    it('base64-encodes binary downloads', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(
          jsonResponse({ name: 'pic.png', mimeType: 'image/png' })
        )
        .mockResolvedValueOnce(binaryResponse('PNGDATA', 'image/png'));

      const bubble = new GoogleDriveBubble({
        operation: 'download_file',
        file_id: 'file_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.content).toBe(
        Buffer.from('PNGDATA', 'utf8').toString('base64')
      );
    });
  });

  // ========================================
  // create_folder / delete_file / get_file_info
  // ========================================
  describe('create_folder', () => {
    it('creates a folder at the Drive root', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'folder_1', name: 'Reports' })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'create_folder',
        name: 'Reports',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.folder?.id).toBe('folder_1');

      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body.mimeType).toBe('application/vnd.google-apps.folder');
      expect(body.parents).toBeUndefined();
    });

    it('creates a nested folder when parent_folder_id is provided', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'folder_2', name: 'Q1', parents: ['folder_1'] })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'create_folder',
        name: 'Q1',
        parent_folder_id: 'folder_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body.parents).toEqual(['folder_1']);
    });
  });

  describe('delete_file', () => {
    it('trashes a file by default', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse({}));

      const bubble = new GoogleDriveBubble({
        operation: 'delete_file',
        file_id: 'file_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.deleted_file_id).toBe('file_1');

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('PATCH');
      expect(JSON.parse(init.body as string)).toEqual({ trashed: true });
    });

    it('permanently deletes when permanent is true', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse({}));

      const bubble = new GoogleDriveBubble({
        operation: 'delete_file',
        file_id: 'file_1',
        permanent: true,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('DELETE');
    });
  });

  describe('get_file_info', () => {
    it('returns file metadata without permissions by default', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'file_1', name: 'notes.txt', mimeType: 'text/plain' })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'get_file_info',
        file_id: 'file_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.file?.id).toBe('file_1');
      expect(result.data?.permissions).toBeUndefined();
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('fetches permissions when include_permissions is true', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(
          jsonResponse({ id: 'file_1', name: 'notes.txt', mimeType: 'text/plain' })
        )
        .mockResolvedValueOnce(
          jsonResponse({
            permissions: [
              {
                id: 'perm_1',
                type: 'user',
                role: 'owner',
                emailAddress: 'me@example.com',
              },
            ],
          })
        );

      const bubble = new GoogleDriveBubble({
        operation: 'get_file_info',
        file_id: 'file_1',
        include_permissions: true,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.permissions).toHaveLength(1);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });

  // ========================================
  // share_file
  // ========================================
  describe('share_file', () => {
    it('creates a permission and returns the share link', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(jsonResponse({ id: 'perm_1' }))
        .mockResolvedValueOnce(
          jsonResponse({ webViewLink: 'https://drive.google.com/file/d/file_1' })
        );

      const bubble = new GoogleDriveBubble({
        operation: 'share_file',
        file_id: 'file_1',
        email_address: 'friend@example.com',
        role: 'writer',
        type: 'user',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.permission_id).toBe('perm_1');
      expect(result.data?.share_link).toBe(
        'https://drive.google.com/file/d/file_1'
      );

      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body).toMatchObject({
        role: 'writer',
        type: 'user',
        emailAddress: 'friend@example.com',
      });
    });

    it('omits emailAddress for anyone-type shares', async () => {
      vi.mocked(global.fetch)
        .mockResolvedValueOnce(jsonResponse({ id: 'perm_2' }))
        .mockResolvedValueOnce(jsonResponse({ webViewLink: 'https://link' }));

      const bubble = new GoogleDriveBubble({
        operation: 'share_file',
        file_id: 'file_1',
        type: 'anyone',
        email_address: 'friend@example.com',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      const body = JSON.parse(
        vi.mocked(global.fetch).mock.calls[0][1]?.body as string
      );
      expect(body.emailAddress).toBeUndefined();
    });
  });

  // ========================================
  // ERROR HANDLING
  // ========================================
  describe('Error Handling', () => {
    it('surfaces a 500 API error as a controlled failure', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(500, 'Internal Server Error')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Google Drive API error');
      expect(result.error).toContain('500');
    });

    it('surfaces a 404 API error as a controlled failure', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(404, 'Not Found')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'get_file_info',
        file_id: 'missing',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('404');
    });

    it('surfaces network failures as a controlled failure', async () => {
      vi.mocked(global.fetch).mockRejectedValueOnce(
        new Error('Network error: ECONNRESET')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });

    it('does not leak the bearer token in error messages', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(403, 'Forbidden')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).not.toContain('ya29.test-mock-token');
    });

    it('reports missing credentials as a controlled failure', async () => {
      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('credentials');
      expect(global.fetch).not.toHaveBeenCalled();
    });
  });

  // ========================================
  // CREDENTIALS
  // ========================================
  describe('testCredential', () => {
    it('resolves true when the about probe succeeds', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ user: { displayName: 'Test User' } })
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).resolves.toBe(true);
    });

    it('rejects when the about probe fails', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'Unauthorized')
      );

      const bubble = new GoogleDriveBubble({
        operation: 'list_files',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).rejects.toThrow(
        'Google Drive API error'
      );
    });

    it('rejects when no credentials are supplied', async () => {
      const bubble = new GoogleDriveBubble({ operation: 'list_files' });

      await expect(bubble.testCredential()).rejects.toThrow(/credentials/i);
    });
  });

  // ========================================
  // CONCURRENCY
  // ========================================
  describe('Concurrency', () => {
    it('handles concurrent list_files calls independently', async () => {
      vi.mocked(global.fetch).mockImplementation(async () =>
        jsonResponse({
          files: [{ id: 'f1', name: 'a.txt', mimeType: 'text/plain' }],
        })
      );

      const bubbles = Array.from(
        { length: 5 },
        () =>
          new GoogleDriveBubble({
            operation: 'list_files',
            credentials: mockCredentials,
          })
      );

      const results = await Promise.all(bubbles.map((b) => b.action()));

      expect(results).toHaveLength(5);
      expect(results.every((r) => r.success)).toBe(true);
      expect(global.fetch).toHaveBeenCalledTimes(5);
    });
  });
});
