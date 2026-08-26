/**
 * Test Suite for GoogleCalendarBubble
 *
 * Rewritten to target the actual GoogleCalendarBubble API:
 *  - discriminated-union params keyed on `operation`
 *  - `action()` returns a BubbleResult wrapper `{ success, data, error }`
 *  - invalid params are captured at construction (NOT thrown) and surfaced
 *    as a controlled error from `action()`
 *  - all network access goes through the global `fetch`, which is mocked here
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleCalendarBubble } from './google-calendar.js';
import { CredentialType } from '@bubblelab/shared-schemas';

const mockCredentials = {
  [CredentialType.GOOGLE_CALENDAR_CRED]: 'ya29.test-mock-token',
};

/** Build a minimal JSON `Response`-like object for the mocked fetch. */
function jsonResponse(body: unknown, init?: { ok?: boolean; status?: number }) {
  return {
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    statusText: init?.ok === false ? 'Error' : 'OK',
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
    text: async () => JSON.stringify(body),
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
  } as unknown as Response;
}

describe('GoogleCalendarBubble', () => {
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
      expect(GoogleCalendarBubble.bubbleName).toBe('google-calendar');
      expect(GoogleCalendarBubble.type).toBe('service');
      expect(GoogleCalendarBubble.service).toBe('google-calendar');
      expect(GoogleCalendarBubble.authType).toBe('oauth');
      expect(GoogleCalendarBubble.alias).toBe('gcal');
      expect(GoogleCalendarBubble.schema).toBeDefined();
      expect(GoogleCalendarBubble.resultSchema).toBeDefined();
    });

    it('defaults to a list_events operation when constructed with no params', () => {
      const bubble = new GoogleCalendarBubble();
      expect(bubble.currentParams.operation).toBe('list_events');
    });

    it('excludes credentials from currentParams', () => {
      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });
      expect(
        (bubble.currentParams as Record<string, unknown>).credentials
      ).toBeUndefined();
    });
  });

  // ========================================
  // INPUT VALIDATION
  // ========================================
  // NOTE: the Bubble base class captures schema validation failures instead of
  // throwing from the constructor, so we construct successfully and assert the
  // controlled error surfaced by action().
  describe('Input Validation', () => {
    it('returns a controlled error for an unknown operation', async () => {
      const bubble = new GoogleCalendarBubble({
        // @ts-expect-error deliberately invalid operation
        operation: 'not_a_real_operation',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('returns a controlled error when get_event is missing event_id', async () => {
      const bubble = new GoogleCalendarBubble({
        // @ts-expect-error event_id is required for get_event
        operation: 'get_event',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
      expect(result.error).toContain('event_id');
    });

    it('returns a controlled error when get_event has an empty event_id', async () => {
      const bubble = new GoogleCalendarBubble({
        operation: 'get_event',
        event_id: '',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Event ID is required');
    });

    it('returns a controlled error when create_event is missing a summary', async () => {
      const bubble = new GoogleCalendarBubble({
        // @ts-expect-error summary/start/end are required for create_event
        operation: 'create_event',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Input Schema validation failed');
    });

    it('rejects max_results above the documented ceiling', async () => {
      const bubble = new GoogleCalendarBubble({
        operation: 'list_events',
        max_results: 5000,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('max_results');
    });

    it('applies schema defaults for optional fields', () => {
      const bubble = new GoogleCalendarBubble({
        operation: 'list_events',
        credentials: mockCredentials,
      });

      const params = bubble.currentParams as Record<string, unknown>;
      expect(params.calendar_id).toBe('primary');
      expect(params.single_events).toBe(true);
      expect(params.order_by).toBe('startTime');
      expect(params.max_results).toBe(50);
    });
  });

  // ========================================
  // list_calendars
  // ========================================
  describe('list_calendars', () => {
    it('returns the calendar list on success', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          items: [
            { id: 'primary', summary: 'Me', accessRole: 'owner' },
            { id: 'team@group.calendar.google.com', summary: 'Team' },
          ],
          nextPageToken: 'next-token',
        })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        max_results: 10,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.operation).toBe('list_calendars');
      expect(result.data?.calendars).toHaveLength(2);
      expect(result.data?.next_page_token).toBe('next-token');
    });

    it('sends the bearer token and maxResults query param', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ items: [] })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        max_results: 7,
        credentials: mockCredentials,
      });

      await bubble.action();

      const [url, init] = vi.mocked(global.fetch).mock.calls[0] as [
        string,
        RequestInit,
      ];
      expect(url).toContain('/users/me/calendarList');
      expect(url).toContain('maxResults=7');
      expect((init.headers as Record<string, string>).Authorization).toBe(
        'Bearer ya29.test-mock-token'
      );
    });

    it('tolerates a response with no items array', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(jsonResponse({}));

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.calendars).toEqual([]);
    });
  });

  // ========================================
  // list_events
  // ========================================
  describe('list_events', () => {
    it('returns events and extracts Drive attachment file IDs', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({
          items: [
            {
              id: 'evt_1',
              summary: 'Standup',
              attachments: [
                { fileId: 'drive_file_1', title: 'notes' },
                { title: 'no-file-id' },
              ],
            },
            { id: 'evt_2', summary: 'Retro' },
          ],
          nextPageToken: 'page-2',
        })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_events',
        calendar_id: 'primary',
        max_results: 25,
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.events).toHaveLength(2);
      expect(result.data?.events?.[0].driveAttachmentFileIds).toEqual([
        'drive_file_1',
      ]);
      expect(result.data?.events?.[1].driveAttachmentFileIds).toEqual([]);
      expect(result.data?.next_page_token).toBe('page-2');
    });

    it('returns an empty list when the calendar has no events', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ items: [] })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_events',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.events).toEqual([]);
    });

    it('forwards optional filters to the API', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ items: [] })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_events',
        calendar_id: 'work@example.com',
        time_min: '2025-01-01T00:00:00Z',
        time_max: '2025-01-31T00:00:00Z',
        q: 'sprint',
        credentials: mockCredentials,
      });

      await bubble.action();

      const url = vi.mocked(global.fetch).mock.calls[0][0] as string;
      expect(url).toContain(encodeURIComponent('work@example.com'));
      expect(url).toContain('timeMin=');
      expect(url).toContain('timeMax=');
      expect(url).toContain('q=sprint');
    });
  });

  // ========================================
  // get_event / create_event / update_event / delete_event
  // ========================================
  describe('get_event', () => {
    it('returns a single event', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'evt_1', summary: 'Standup' })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'get_event',
        event_id: 'evt_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.event?.id).toBe('evt_1');
      expect(result.data?.event?.driveAttachmentFileIds).toEqual([]);
    });
  });

  describe('create_event', () => {
    it('creates an event with attendees', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'evt_new', summary: 'Planning' })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'create_event',
        summary: 'Planning',
        start: { dateTime: '2025-05-01T10:00:00Z' },
        end: { dateTime: '2025-05-01T11:00:00Z' },
        attendees: [{ email: 'a@example.com' }],
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.operation).toBe('create_event');
      expect(result.data?.event?.id).toBe('evt_new');

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('POST');
      expect(JSON.parse(init.body as string).summary).toBe('Planning');
    });
  });

  describe('update_event', () => {
    it('updates an existing event', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ id: 'evt_1', summary: 'Renamed' })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'update_event',
        event_id: 'evt_1',
        summary: 'Renamed',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.event?.summary).toBe('Renamed');
    });
  });

  describe('delete_event', () => {
    it('deletes an event', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        status: 204,
        statusText: 'No Content',
        headers: new Headers(),
        text: async () => '',
      } as unknown as Response);

      const bubble = new GoogleCalendarBubble({
        operation: 'delete_event',
        event_id: 'evt_1',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.deleted).toBe(true);

      const init = vi.mocked(global.fetch).mock.calls[0][1] as RequestInit;
      expect(init.method).toBe('DELETE');
    });
  });

  // ========================================
  // ERROR HANDLING
  // ========================================
  describe('Error Handling', () => {
    it('surfaces API errors as a controlled failure', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(500, 'Internal Server Error')
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Google Calendar API error');
      expect(result.error).toContain('500');
    });

    it('surfaces 401 Unauthorized as a controlled failure', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'Unauthorized')
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_events',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('401');
    });

    it('surfaces 404 Not Found for a missing event', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(404, 'Not Found')
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'get_event',
        event_id: 'missing',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('404');
    });

    it('surfaces network failures as a controlled failure', async () => {
      vi.mocked(global.fetch).mockRejectedValueOnce(
        new Error('Network error: connection reset')
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });

    it('does not leak the bearer token in the error message', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(403, 'Forbidden')
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).not.toContain('ya29.test-mock-token');
    });
  });

  // ========================================
  // CREDENTIALS
  // ========================================
  describe('testCredential', () => {
    it('resolves true when the calendarList probe succeeds', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        jsonResponse({ items: [] })
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).resolves.toBe(true);
    });

    it('rejects when the calendarList probe fails', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce(
        errorResponse(401, 'Unauthorized')
      );

      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
        credentials: mockCredentials,
      });

      await expect(bubble.testCredential()).rejects.toThrow(
        'Google Calendar API error'
      );
    });

    it('rejects when no credential is available', async () => {
      const bubble = new GoogleCalendarBubble({
        operation: 'list_calendars',
      });

      // chooseCredential() throws first when the credentials map is absent.
      await expect(bubble.testCredential()).rejects.toThrow(
        /credentials/i
      );
    });
  });

  // ========================================
  // CONCURRENCY
  // ========================================
  describe('Concurrency', () => {
    it('handles concurrent list_events calls independently', async () => {
      vi.mocked(global.fetch).mockImplementation(async () =>
        jsonResponse({ items: [{ id: 'evt_x' }] })
      );

      const bubbles = Array.from(
        { length: 5 },
        () =>
          new GoogleCalendarBubble({
            operation: 'list_events',
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
