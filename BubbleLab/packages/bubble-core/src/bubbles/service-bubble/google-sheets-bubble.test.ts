/**
 * Comprehensive tests for Google Sheets Bubble
 *
 * Tests all 14 operations:
 * 1. createSpreadsheet
 * 2. getSpreadsheet
 * 3. deleteSpreadsheet
 * 4. copySpreadsheet
 * 5. updateCell
 * 6. getCellValue
 * 7. batchUpdate
 * 8. appendRow
 * 9. getRange
 * 10. clearRange
 * 11. copyRange
 * 12. addSheet
 * 13. deleteSheet
 * 14. getSheetData
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleSheetsBubble } from './google-sheets-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('GoogleSheetsBubble', () => {
  let sheetsBubble: GoogleSheetsBubble;
  const mockCredentials = {
    [CredentialType.GOOGLE_DRIVE_CRED]: JSON.stringify({
      accessToken: 'ya_test_mock_token',
      refreshToken: 'test_refresh_token',
    }),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Operation 1: createSpreadsheet', () => {
    it('should create a spreadsheet successfully', async () => {
      const mockResponse = {
        spreadsheetId: 'sheet_123',
        properties: {
          title: 'Test Spreadsheet',
          locale: 'en_US',
        },
        sheets: [
          {
            properties: {
              sheetId: 0,
              title: 'Sheet1',
              index: 0,
              sheetType: 'GRID',
              gridProperties: {
                rowCount: 1000,
                columnCount: 26,
              },
            },
          },
        ],
        spreadsheetUrl: 'https://docs.google.com/spreadsheets/d/sheet_123',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'createSpreadsheet',
        title: 'Test Spreadsheet',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.spreadsheetId).toBe('sheet_123');
      expect(result.result.title).toBe('Test Spreadsheet');
      expect(result.result.sheetCount).toBe(1);
    });

    it('should create spreadsheet with custom sheets', async () => {
      const mockResponse = {
        spreadsheetId: 'sheet_456',
        properties: { title: 'Custom Sheets' },
        sheets: [
          { properties: { sheetId: 0, title: 'Data', index: 0 } },
          { properties: { sheetId: 1, title: 'Summary', index: 1 } },
        ],
        spreadsheetUrl: 'https://docs.google.com/spreadsheets/d/sheet_456',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'createSpreadsheet',
        title: 'Custom Sheets',
        sheets: [
          { title: 'Data', rowCount: 1000, columnCount: 26 },
          { title: 'Summary', rowCount: 500, columnCount: 10 },
        ],
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.sheetCount).toBe(2);
    });

    it('should handle authentication errors', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'createSpreadsheet',
        title: 'Test',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('401');
    });

    it('should validate title is required', async () => {
      expect(() => {
        new GoogleSheetsBubble({
          operation: 'createSpreadsheet',
          title: '',
          credentials: mockCredentials,
        });
      }).toThrow();
    });
  });

  describe('Operation 2: getSpreadsheet', () => {
    it('should retrieve spreadsheet metadata successfully', async () => {
      const mockResponse = {
        spreadsheetId: 'sheet_123',
        properties: {
          title: 'Test Spreadsheet',
          locale: 'en_US',
        },
        sheets: [
          {
            properties: {
              sheetId: 0,
              title: 'Sheet1',
              index: 0,
              sheetType: 'GRID',
              gridProperties: {
                rowCount: 1000,
                columnCount: 26,
              },
            },
          },
        ],
        namedRanges: [
          {
            name: 'TestRange',
            range: 'Sheet1!A1:B10',
          },
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        includeGridData: false,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.spreadsheetId).toBe('sheet_123');
      expect(result.result.title).toBe('Test Spreadsheet');
      expect(result.result.sheets).toHaveLength(1);
      expect(result.result.sheets[0].title).toBe('Sheet1');
    });

    it('should include grid data when requested', async () => {
      const mockResponse = {
        spreadsheetId: 'sheet_123',
        properties: { title: 'Test' },
        sheets: [
          {
            properties: { sheetId: 0, title: 'Sheet1' },
            data: [
              {
                rowData: [
                  {
                    values: [
                      { userEnteredValue: { stringValue: 'Hello' } },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        includeGridData: true,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle spreadsheet not found', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => 'Not Found',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'nonexistent',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Not Found');
    });
  });

  describe('Operation 3: deleteSpreadsheet', () => {
    it('should delete a spreadsheet successfully', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        status: 204,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'deleteSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.deleted).toBe(true);
      expect(result.result.spreadsheetId).toBe('sheet_123');
    });

    it('should handle deletion of non-existent spreadsheet', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => 'Not Found',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'deleteSpreadsheet',
        spreadsheetId: 'nonexistent',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.deleted).toBe(false);
    });
  });

  describe('Operation 4: copySpreadsheet', () => {
    it('should copy a spreadsheet successfully', async () => {
      const mockResponse = {
        id: 'sheet_456',
        name: 'Copy of Test',
        kind: 'drive#file',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'copySpreadsheet',
        spreadsheetId: 'sheet_123',
        title: 'Copy of Test',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.originalSpreadsheetId).toBe('sheet_123');
      expect(result.result.newSpreadsheetId).toBe('sheet_456');
      expect(result.result.title).toBe('Copy of Test');
      expect(result.result.url).toContain('sheet_456');
    });

    it('should copy to specific folder', async () => {
      const mockResponse = {
        id: 'sheet_789',
        name: 'Copy in Folder',
        parents: ['folder_123'],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'copySpreadsheet',
        spreadsheetId: 'sheet_123',
        title: 'Copy in Folder',
        destinationFolderId: 'folder_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.newSpreadsheetId).toBe('sheet_789');
    });
  });

  describe('Operation 5: updateCell', () => {
    it('should update a single cell successfully', async () => {
      const mockResponse = {
        updates: {
          updatedRange: 'Sheet1!A1',
          updatedRows: 1,
          updatedColumns: 1,
          updatedCells: 1,
        },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        value: 'Hello World',
        valueInputOption: 'USER_ENTERED',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.updatedRange).toBe('Sheet1!A1');
      expect(result.result.updatedCells).toBe(1);
    });

    it('should handle invalid range format', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Invalid range',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'InvalidRange',
        value: 'test',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should update cell with RAW input option', async () => {
      const mockResponse = {
        updates: {
          updatedRange: 'Sheet1!B2',
          updatedCells: 1,
        },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!B2',
        value: 12345,
        valueInputOption: 'RAW',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });
  });

  describe('Operation 6: getCellValue', () => {
    it('should get a single cell value successfully', async () => {
      const mockResponse = {
        range: 'Sheet1!A1',
        values: [['Test Value']],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getCellValue',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        valueRenderOption: 'UNFORMATTED_VALUE',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.value).toBe('Test Value');
      expect(result.result.range).toBe('Sheet1!A1');
    });

    it('should return null for empty cell', async () => {
      const mockResponse = {
        range: 'Sheet1!A1',
        values: [['']],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getCellValue',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.value).toBe('');
    });

    it('should get formatted value', async () => {
      const mockResponse = {
        range: 'Sheet1!C5',
        values: [['$1,234.56']],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getCellValue',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!C5',
        valueRenderOption: 'FORMATTED_VALUE',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.value).toBe('$1,234.56');
    });
  });

  describe('Operation 7: batchUpdate', () => {
    it('should update multiple ranges successfully', async () => {
      const mockResponse = {
        replies: [
          { updateCells: { updatedRange: 'Sheet1!A1:A2' } },
          { updateCells: { updatedRange: 'Sheet1!B1:B2' } },
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'batchUpdate',
        spreadsheetId: 'sheet_123',
        updates: [
          { range: 'Sheet1!A1:A2', values: [['A'], ['B']] },
          { range: 'Sheet1!B1:B2', values: [['C'], ['D']] },
        ],
        valueInputOption: 'USER_ENTERED',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.totalUpdatedCells).toBeGreaterThan(0);
      expect(result.result.updateResults).toHaveLength(2);
    });

    it('should handle large batch updates', async () => {
      const updates = Array.from({ length: 100 }, (_, i) => ({
        range: `Sheet1!A${i + 1}:B${i + 1}`,
        values: [[`Value${i}`, `Data${i}`]],
      }));

      const mockResponse = {
        replies: updates.map(() => ({ updateCells: {} })),
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'batchUpdate',
        spreadsheetId: 'sheet_123',
        updates,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.updateResults.length).toBe(100);
    });

    it('should validate updates array is not empty', async () => {
      expect(() => {
        new GoogleSheetsBubble({
          operation: 'batchUpdate',
          spreadsheetId: 'sheet_123',
          updates: [],
          credentials: mockCredentials,
        });
      }).toThrow();
    });
  });

  describe('Operation 8: appendRow', () => {
    it('should append a row successfully', async () => {
      const mockResponse = {
        tableRange: 'Sheet1!A1:Z1000',
        updates: {
          updatedRange: 'Sheet1!A1001:E1001',
          updatedRows: 1,
          updatedColumns: 5,
          updatedCells: 5,
        },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'appendRow',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        values: ['Name', 'Age', 'City', 'Country', 'Email'],
        valueInputOption: 'USER_ENTERED',
        insertDataOption: 'INSERT_ROWS',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.tableRange).toBeDefined();
      expect(result.result.updates?.updatedRows).toBe(1);
    });

    it('should append row with overwrite option', async () => {
      const mockResponse = {
        tableRange: 'Sheet1!A1:Z100',
        updates: {
          updatedRange: 'Sheet1!A1:E1',
          updatedRows: 1,
        },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'appendRow',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        values: ['Test1', 'Test2', 'Test3'],
        insertDataOption: 'OVERWRITE',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should validate values array is not empty', async () => {
      expect(() => {
        new GoogleSheetsBubble({
          operation: 'appendRow',
          spreadsheetId: 'sheet_123',
          range: 'Sheet1!A1',
          values: [],
          credentials: mockCredentials,
        });
      }).toThrow();
    });
  });

  describe('Operation 9: getRange', () => {
    it('should get a range of values successfully', async () => {
      const mockResponse = {
        range: 'Sheet1!A1:C3',
        majorDimension: 'ROWS',
        values: [
          ['A1', 'B1', 'C1'],
          ['A2', 'B2', 'C2'],
          ['A3', 'B3', 'C3'],
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getRange',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1:C3',
        majorDimension: 'ROWS',
        valueRenderOption: 'UNFORMATTED_VALUE',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.values).toHaveLength(3);
      expect(result.result.values[0]).toHaveLength(3);
      expect(result.result.majorDimension).toBe('ROWS');
    });

    it('should get range by columns', async () => {
      const mockResponse = {
        range: 'Sheet1!A1:C1',
        majorDimension: 'COLUMNS',
        values: [['A1', 'B1', 'C1']],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getRange',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1:C1',
        majorDimension: 'COLUMNS',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.majorDimension).toBe('COLUMNS');
    });

    it('should handle empty range', async () => {
      const mockResponse = {
        range: 'Sheet1!A1:B2',
        values: [],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getRange',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1:B2',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.values).toHaveLength(0);
    });
  });

  describe('Operation 10: clearRange', () => {
    it('should clear a range successfully', async () => {
      const mockResponse = {
        clearedRange: 'Sheet1!A1:C10',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'clearRange',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1:C10',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.clearedRange).toBe('Sheet1!A1:C10');
    });

    it('should clear single cell', async () => {
      const mockResponse = {
        clearedRange: 'Sheet1!A1',
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'clearRange',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle invalid range for clear', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Invalid range',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'clearRange',
        spreadsheetId: 'sheet_123',
        range: 'InvalidRange',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Operation 11: copyRange', () => {
    it('should copy range successfully', async () => {
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            sheets: [
              { properties: { sheetId: 0, title: 'Sheet1' } },
              { properties: { sheetId: 1, title: 'Sheet2' } },
            ],
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({}),
        } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'copyRange',
        spreadsheetId: 'sheet_123',
        sourceRange: 'Sheet1!A1:B10',
        destinationRange: 'Sheet2!A1:B10',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.sourceRange).toBe('Sheet1!A1:B10');
      expect(result.result.destinationRange).toBe('Sheet2!A1:B10');
    });

    it('should handle sheet not found for copy', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          sheets: [
            { properties: { sheetId: 0, title: 'Sheet1' } },
          ],
        }),
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'copyRange',
        spreadsheetId: 'sheet_123',
        sourceRange: 'Sheet1!A1:B10',
        destinationRange: 'NonExistent!A1:B10',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('Sheet not found');
    });
  });

  describe('Operation 12: addSheet', () => {
    it('should add a new sheet successfully', async () => {
      const mockResponse = {
        replies: [
          {
            addSheet: {
              properties: {
                sheetId: 1,
                title: 'NewSheet',
                index: 1,
                gridProperties: {
                  rowCount: 1000,
                  columnCount: 26,
                },
              },
            },
          },
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'addSheet',
        spreadsheetId: 'sheet_123',
        title: 'NewSheet',
        rowCount: 1000,
        columnCount: 26,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.sheetId).toBe(1);
      expect(result.result.title).toBe('NewSheet');
    });

    it('should add sheet with custom dimensions', async () => {
      const mockResponse = {
        replies: [
          {
            addSheet: {
              properties: {
                sheetId: 2,
                title: 'LargeSheet',
                gridProperties: { rowCount: 5000, columnCount: 50 },
              },
            },
          },
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'addSheet',
        spreadsheetId: 'sheet_123',
        title: 'LargeSheet',
        rowCount: 5000,
        columnCount: 50,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should validate sheet title is required', async () => {
      expect(() => {
        new GoogleSheetsBubble({
          operation: 'addSheet',
          spreadsheetId: 'sheet_123',
          title: '',
          credentials: mockCredentials,
        });
      }).toThrow();
    });
  });

  describe('Operation 13: deleteSheet', () => {
    it('should delete a sheet successfully', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'deleteSheet',
        spreadsheetId: 'sheet_123',
        sheetId: 1,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.sheetId).toBe(1);
    });

    it('should handle deleting non-existent sheet', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'deleteSheet',
        spreadsheetId: 'sheet_123',
        sheetId: 999,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      // Sheets API returns success even if sheet doesn't exist
      expect(result.result.success).toBe(true);
    });
  });

  describe('Operation 14: getSheetData', () => {
    it('should get complete sheet data successfully', async () => {
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            sheets: [
              {
                properties: {
                  sheetId: 0,
                  title: 'Data',
                  gridProperties: { rowCount: 1000, columnCount: 26 },
                },
              },
            ],
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            values: [
              ['Name', 'Age', 'City'],
              ['John', '30', 'NYC'],
              ['Jane', '25', 'LA'],
            ],
          }),
        } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSheetData',
        spreadsheetId: 'sheet_123',
        sheetName: 'Data',
        includeMetadata: true,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.sheetName).toBe('Data');
      expect(result.result.values).toHaveLength(3);
      expect(result.result.metadata).toBeDefined();
      expect(result.result.metadata?.rowCount).toBe(1000);
    });

    it('should get sheet data without metadata', async () => {
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            sheets: [
              { properties: { sheetId: 0, title: 'Simple' } },
            ],
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            values: [['A', 'B'], ['C', 'D']],
          }),
        } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSheetData',
        spreadsheetId: 'sheet_123',
        sheetName: 'Simple',
        includeMetadata: false,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(result.result.metadata).toBeUndefined();
    });

    it('should handle sheet not found', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          sheets: [
            { properties: { sheetId: 0, title: 'ExistingSheet' } },
          ],
        }),
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSheetData',
        spreadsheetId: 'sheet_123',
        sheetName: 'NonExistent',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('not found');
    });
  });

  describe('Error Handling', () => {
    it('should handle network timeouts', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Request timeout'));

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('timeout');
    });

    it('should handle rate limiting', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 429,
        text: async () => 'Rate limit exceeded',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('429');
    });

    it('should handle quota exceeded errors', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 403,
        text: async () => 'Quota exceeded',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        value: 'test',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).toContain('403');
    });
  });

  describe('Credential Testing', () => {
    it('should test valid credentials', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ spreadsheetId: 'test' }),
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const isValid = await sheetsBubble.testCredential();

      expect(isValid).toBe(true);
    });

    it('should test invalid credentials', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: { [CredentialType.GOOGLE_DRIVE_CRED]: 'invalid' },
      });

      const isValid = await sheetsBubble.testCredential();

      expect(isValid).toBe(false);
    });

    it('should handle missing credentials', async () => {
      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
      });

      const isValid = await sheetsBubble.testCredential();

      expect(isValid).toBe(false);
    });
  });

  describe('Input Validation', () => {
    it('should validate spreadsheetId is required', async () => {
      expect(() => {
        new GoogleSheetsBubble({
          operation: 'getSpreadsheet',
          spreadsheetId: '',
          credentials: mockCredentials,
        });
      }).toThrow();
    });

    it('should validate range format', async () => {
      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'InvalidRangeFormat!',
        value: 'test',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should validate memory limits', async () => {
      const largeValue = 'x'.repeat(50 * 1024 * 1024); // 50MB

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        value: largeValue,
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Security Tests', () => {
    it('should sanitize error messages', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(
        new Error('Failed with Bearer ya_test_secret_token')
      );

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
      expect(result.result.error).not.toContain('ya_test_secret_token');
    });

    it('should handle malicious input in range', async () => {
      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: '../../../etc/passwd',
        value: 'malicious',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });

    it('should validate URL in operations', async () => {
      // Test that operations validate spreadsheet URLs
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Invalid request',
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'javascript:alert(1)',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Retry Logic', () => {
    it('should retry on transient errors', async () => {
      vi.mocked(fetch)
        .mockRejectedValueOnce(new Error('ECONNRESET'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ spreadsheetId: 'sheet_123' }),
        } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);
    });

    it('should respect max retry limit', async () => {
      for (let i = 0; i < 10; i++) {
        vi.mocked(fetch).mockRejectedValueOnce(new Error('ECONNRESET'));
      }

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getSpreadsheet',
        spreadsheetId: 'sheet_123',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(false);
    });
  });

  describe('Edge Cases', () => {
    it('should handle very large ranges', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          range: 'Sheet1!A1:Z100000',
          values: Array.from({ length: 100000 }, () => Array(26).fill('')),
        }),
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'getRange',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1:Z100000',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle special characters in values', async () => {
      const mockResponse = {
        updates: { updatedRange: 'Sheet1!A1', updatedCells: 1 },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        value: 'Special: \n\t\r"\'<>',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });

    it('should handle unicode characters', async () => {
      const mockResponse = {
        updates: { updatedRange: 'Sheet1!A1', updatedCells: 1 },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      sheetsBubble = new GoogleSheetsBubble({
        operation: 'updateCell',
        spreadsheetId: 'sheet_123',
        range: 'Sheet1!A1',
        value: 'Hello 世界 🌍 Привет',
        credentials: mockCredentials,
      });

      const result = await sheetsBubble.performAction();

      expect(result.result.success).toBe(true);
    });
  });
});
