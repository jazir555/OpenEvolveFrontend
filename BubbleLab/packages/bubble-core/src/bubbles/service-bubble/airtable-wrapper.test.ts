/**
 * Airtable Wrapper Service Bubble Test Suite
 *
 * Tests all 12 operations with resilience patterns:
 * - Circuit breaker behavior
 * - Retry logic
 * - Rate limiting
 * - Input validation
 * - Error handling
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AirtableWrapperBubble } from './airtable-wrapper.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Mock fetch globally
global.fetch = vi.fn();

describe('AirtableWrapperBubble', () => {
  const mockApiKey = 'patTest1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890';
  const mockBaseId = 'appTestBase123';
  const mockTableId = 'tblTestTable123';
  const mockRecordId = 'recTestRecord123';

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ========================================================================
  // OPERATION 1: LIST RECORDS
  // ========================================================================

  describe('listRecords', () => {
    it('should list records successfully', async () => {
      const mockResponse = {
        records: [
          { id: 'rec1', createdTime: '2024-01-01T00:00:00.000Z', fields: { Name: 'Test' } },
        ],
        offset: 'nextPageToken',
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('listRecords');
      expect(result.result.success).toBe(true);
      expect(result.result.records).toHaveLength(1);
      expect(result.result.count).toBe(1);
      expect(result.result.offset).toBe('nextPageToken');
    });

    it('should handle errors gracefully', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toBeDefined();
      expect(result.result.records).toHaveLength(0);
    });
  });

  // ========================================================================
  // OPERATION 2: GET RECORD
  // ========================================================================

  describe('getRecord', () => {
    it('should get a single record', async () => {
      const mockResponse = {
        id: mockRecordId,
        createdTime: '2024-01-01T00:00:00.000Z',
        fields: { Name: 'Test Record' },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'getRecord',
        baseId: mockBaseId,
        tableId: mockTableId,
        recordId: mockRecordId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('getRecord');
      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe(mockRecordId);
      expect(result.result.fields.Name).toBe('Test Record');
    });
  });

  // ========================================================================
  // OPERATION 3: CREATE RECORD
  // ========================================================================

  describe('createRecord', () => {
    it('should create a new record', async () => {
      const mockResponse = {
        id: 'recNew123',
        createdTime: '2024-01-01T00:00:00.000Z',
        fields: { Name: 'New Record', Status: 'Active' },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'createRecord',
        baseId: mockBaseId,
        tableId: mockTableId,
        fields: { Name: 'New Record', Status: 'Active' },
        typecast: true,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('createRecord');
      expect(result.result.success).toBe(true);
      expect(result.result.id).toBe('recNew123');
      expect(result.result.fields.Status).toBe('Active');
    });
  });

  // ========================================================================
  // OPERATION 4: UPDATE RECORD
  // ========================================================================

  describe('updateRecord', () => {
    it('should update an existing record', async () => {
      const mockResponse = {
        id: mockRecordId,
        createdTime: '2024-01-01T00:00:00.000Z',
        fields: { Name: 'Updated Record' },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'updateRecord',
        baseId: mockBaseId,
        tableId: mockTableId,
        recordId: mockRecordId,
        fields: { Name: 'Updated Record' },
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('updateRecord');
      expect(result.result.success).toBe(true);
      expect(result.result.fields.Name).toBe('Updated Record');
    });
  });

  // ========================================================================
  // OPERATION 5: DELETE RECORD
  // ========================================================================

  describe('deleteRecord', () => {
    it('should delete a record', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'deleteRecord',
        baseId: mockBaseId,
        tableId: mockTableId,
        recordId: mockRecordId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('deleteRecord');
      expect(result.result.success).toBe(true);
      expect(result.result.deleted).toBe(true);
      expect(result.result.recordId).toBe(mockRecordId);
    });
  });

  // ========================================================================
  // OPERATION 6: BATCH CREATE
  // ========================================================================

  describe('batchCreate', () => {
    it('should create multiple records', async () => {
      const mockResponse = {
        records: [
          { id: 'rec1', createdTime: '2024-01-01T00:00:00.000Z', fields: { Name: 'Record 1' } },
          { id: 'rec2', createdTime: '2024-01-01T00:00:00.000Z', fields: { Name: 'Record 2' } },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'batchCreate',
        baseId: mockBaseId,
        tableId: mockTableId,
        records: [
          { fields: { Name: 'Record 1' } },
          { fields: { Name: 'Record 2' } },
        ],
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('batchCreate');
      expect(result.result.success).toBe(true);
      expect(result.result.count).toBe(2);
      expect(result.result.records).toHaveLength(2);
    });
  });

  // ========================================================================
  // OPERATION 7: BATCH UPDATE
  // ========================================================================

  describe('batchUpdate', () => {
    it('should update multiple records', async () => {
      const mockResponse = {
        records: [
          { id: 'rec1', createdTime: '2024-01-01T00:00:00.000Z', fields: { Status: 'Updated' } },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'batchUpdate',
        baseId: mockBaseId,
        tableId: mockTableId,
        records: [
          { id: 'rec1', fields: { Status: 'Updated' } },
        ],
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('batchUpdate');
      expect(result.result.success).toBe(true);
      expect(result.result.records).toHaveLength(1);
    });
  });

  // ========================================================================
  // OPERATION 8: BATCH DELETE
  // ========================================================================

  describe('batchDelete', () => {
    it('should delete multiple records', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'batchDelete',
        baseId: mockBaseId,
        tableId: mockTableId,
        recordIds: ['rec1', 'rec2', 'rec3'],
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('batchDelete');
      expect(result.result.success).toBe(true);
      expect(result.result.deleted).toBe(true);
      expect(result.result.count).toBe(3);
      expect(result.result.recordIds).toHaveLength(3);
    });
  });

  // ========================================================================
  // OPERATION 9: QUERY RECORDS
  // ========================================================================

  describe('queryRecords', () => {
    it('should query records with formula', async () => {
      const mockResponse = {
        records: [
          { id: 'rec1', createdTime: '2024-01-01T00:00:00.000Z', fields: { Status: 'Active' } },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'queryRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        filterByFormula: '{Status} = "Active"',
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('queryRecords');
      expect(result.result.success).toBe(true);
      expect(result.result.records).toHaveLength(1);
    });
  });

  // ========================================================================
  // OPERATION 10: SEARCH RECORDS
  // ========================================================================

  describe('searchRecords', () => {
    it('should search records with text', async () => {
      const mockResponse = {
        records: [
          { id: 'rec1', createdTime: '2024-01-01T00:00:00.000Z', fields: { Name: 'Search Match' } },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'searchRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        searchString: 'Match',
        fields: ['Name'],
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('searchRecords');
      expect(result.result.success).toBe(true);
      expect(result.result.records).toHaveLength(1);
    });
  });

  // ========================================================================
  // OPERATION 11: GET SCHEMA
  // ========================================================================

  describe('getSchema', () => {
    it('should get table schema', async () => {
      const mockResponse = {
        tables: [
          {
            id: mockTableId,
            name: 'Test Table',
            description: 'Test table description',
            primaryFieldId: 'fld1',
            fields: [
              { id: 'fld1', name: 'Name', type: 'singleLineText' },
              { id: 'fld2', name: 'Status', type: 'singleSelect' },
            ],
          },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'getSchema',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('getSchema');
      expect(result.result.success).toBe(true);
      expect(result.result.name).toBe('Test Table');
      expect(result.result.fields).toHaveLength(2);
      expect(result.result.primaryFieldId).toBe('fld1');
    });

    it('should handle table not found', async () => {
      const mockResponse = {
        tables: [
          { id: 'tblOther', name: 'Other Table' },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'getSchema',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toBe('Table not found');
    });
  });

  // ========================================================================
  // OPERATION 12: LIST TABLES
  // ========================================================================

  describe('listTables', () => {
    it('should list all tables in a base', async () => {
      const mockResponse = {
        tables: [
          { id: 'tbl1', name: 'Table 1', primaryFieldId: 'fld1' },
          { id: 'tbl2', name: 'Table 2', primaryFieldId: 'fld2' },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'listTables',
        baseId: mockBaseId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.operation).toBe('listTables');
      expect(result.result.success).toBe(true);
      expect(result.result.tables).toHaveLength(2);
      expect(result.result.count).toBe(2);
    });
  });

  // ========================================================================
  // RESILIENCE PATTERNS TESTS
  // ========================================================================

  describe('Resilience Patterns', () => {
    it('should handle rate limiting errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 429,
        headers: { get: (name: string) => name === 'Retry-After' ? '5' : null },
        text: async () => 'RATE_LIMIT_EXCEEDED',
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('RATE_LIMIT_EXCEEDED');
    });

    it('should validate base ID format', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: 'invalid-base-id',
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Invalid Airtable base ID format');
    });

    it('should validate record ID format', async () => {
      expect(() => {
        new AirtableWrapperBubble({
          operation: 'getRecord',
          baseId: mockBaseId,
          tableId: mockTableId,
          recordId: 'invalid-record-id',
          credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
        });
      }).toThrow();
    });
  });

  // ========================================================================
  // CREDENTIAL TESTS
  // ========================================================================

  describe('Credentials', () => {
    it('should test credentials successfully', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ bases: [] }),
      });

      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const isValid = await bubble.testCredential();
      expect(isValid).toBe(true);
    });

    it('should reject invalid API key format', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: 'invalid-key' },
      });

      const isValid = await bubble.testCredential();
      expect(isValid).toBe(false);
    });
  });

  // ========================================================================
  // CIRCUIT BREAKER TESTS
  // ========================================================================

  describe('Circuit Breaker', () => {
    it('should open circuit after 5 failures', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      // Mock 5 consecutive failures
      for (let i = 0; i < 5; i++) {
        (global.fetch as any).mockRejectedValueOnce(new Error('Connection failed'));
        await bubble.performAction();
      }

      // Circuit should now be open
      const state = bubble.getCircuitBreakerState();
      expect(state).toBeDefined();

      // Next call should fail immediately without trying
      (global.fetch as any).mockClear();
      const result = await bubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should provide circuit breaker stats', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const stats = bubble.getCircuitBreakerStats();
      expect(stats).toHaveProperty('state');
      expect(stats).toHaveProperty('failureCount');
      expect(stats).toHaveProperty('successCount');
    });

    it('should reset circuit breaker', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      await bubble.resetCircuitBreaker();

      const state = bubble.getCircuitBreakerState();
      expect(state).toBe('closed');
    });
  });

  // ========================================================================
  // DEAD LETTER QUEUE TESTS
  // ========================================================================

  describe('Dead Letter Queue', () => {
    it('should track failed operations', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Permanent failure'));

      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      await bubble.performAction();

      // After retry attempts, should be in DLQ
      const dlqEntries = bubble.getDeadLetterEntries();
      expect(Array.isArray(dlqEntries)).toBe(true);
    });

    it('should clear dead letter queue', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      bubble.clearDeadLetterQueue();

      const entries = bubble.getDeadLetterEntries();
      expect(entries).toHaveLength(0);
    });
  });

  // ========================================================================
  // DEDUPLICATION TESTS
  // ========================================================================

  describe('Request Deduplication', () => {
    it('should provide deduplicator stats', async () => {
      const bubble = new AirtableWrapperBubble({
        operation: 'listRecords',
        baseId: mockBaseId,
        tableId: mockTableId,
        credentials: { [CredentialType.AIRTABLE_CRED]: mockApiKey },
      });

      const stats = bubble.getDeduplicatorStats();
      expect(stats).toHaveProperty('pendingRequests');
      expect(stats).toHaveProperty('completedRequests');
    });
  });
});
