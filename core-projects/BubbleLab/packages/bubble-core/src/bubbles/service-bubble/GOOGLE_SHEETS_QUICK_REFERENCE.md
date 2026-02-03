# Google Sheets Service Bubble - Quick Reference

**Location:** `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/google-sheets-bubble.ts`
**Total Lines:** 1,615 lines
**Operations:** 14 core + 5 legacy
**Status:** ✅ Production Ready

---

## 🚀 QUICK START

```typescript
import { GoogleSheetsBubble } from '@bubblelab/bubble-core';
import { CredentialType } from '@bubblelab/shared-schemas';

// Create a spreadsheet
const bubble = new GoogleSheetsBubble({
  operation: 'createSpreadsheet',
  title: 'My Spreadsheet',
  sheets: [
    { title: 'Data', rowCount: 1000, columnCount: 26 }
  ],
  credentials: {
    [CredentialType.GOOGLE_SHEETS_CRED]: 'your-oauth-token'
  }
});

const result = await bubble.execute();
console.log(result.data); // { spreadsheetId, title, url, sheetCount, success, error }
```

---

## 📋 ALL OPERATIONS

### Spreadsheet Operations

| Operation | Description | Input | Output |
|-----------|-------------|-------|--------|
| **createSpreadsheet** | Create new spreadsheet | title, sheets[] | spreadsheetId, url |
| **getSpreadsheet** | Get spreadsheet metadata | spreadsheetId | sheets[], namedRanges[] |
| **deleteSpreadsheet** | Delete spreadsheet | spreadsheetId | deleted (bool) |
| **copySpreadsheet** | Copy spreadsheet | spreadsheetId, title, folderId | newSpreadsheetId, url |

### Cell Operations

| Operation | Description | Input | Output |
|-----------|-------------|-------|--------|
| **updateCell** | Update single cell | spreadsheetId, range, value | updatedRange |
| **getCellValue** | Get cell value | spreadsheetId, range | value |
| **batchUpdate** | Update multiple cells | spreadsheetId, updates[] | totalUpdatedCells |
| **appendRow** | Append row to sheet | spreadsheetId, range, values[] | tableRange |

### Range Operations

| Operation | Description | Input | Output |
|-----------|-------------|-------|--------|
| **getRange** | Get range values | spreadsheetId, range | values[][] |
| **clearRange** | Clear range | spreadsheetId, range | clearedRange |
| **copyRange** | Copy range | spreadsheetId, source, dest | updatedRange |

### Sheet Operations

| Operation | Description | Input | Output |
|-----------|-------------|-------|--------|
| **addSheet** | Add new sheet | spreadsheetId, title, size | sheetId, title |
| **deleteSheet** | Delete sheet | spreadsheetId, sheetId | success |
| **getSheetData** | Get sheet with metadata | spreadsheetId, sheetName | values, metadata |

---

## 💡 COMMON PATTERNS

### Pattern 1: Create and Populate
```typescript
// 1. Create spreadsheet
const createResult = await new GoogleSheetsBubble({
  operation: 'createSpreadsheet',
  title: 'Sales Report',
  sheets: [{ title: 'Q1', rowCount: 1000, columnCount: 26 }],
  credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: token }
}).execute();

const spreadsheetId = createResult.data.spreadsheetId;

// 2. Add headers
await new GoogleSheetsBubble({
  operation: 'updateCell',
  spreadsheetId,
  range: 'Q1!A1',
  value: 'Date',
  credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: token }
}).execute();

// 3. Append data
await new GoogleSheetsBubble({
  operation: 'appendRow',
  spreadsheetId,
  range: 'Q1!A2',
  values: ['2026-01-18', '$1,234', '567'],
  credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: token }
}).execute();
```

### Pattern 2: Batch Update
```typescript
const bubble = new GoogleSheetsBubble({
  operation: 'batchUpdate',
  spreadsheetId: '1BxiMVs0XRA5nFMdK...',
  updates: [
    { range: 'Sheet1!A1', values: [['Name', 'Email']] },
    { range: 'Sheet1!A2', values: [['John', 'john@example.com']] },
    { range: 'Sheet1!A3', values: [['Jane', 'jane@example.com']] }
  ],
  valueInputOption: 'USER_ENTERED',
  credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: token }
});

const result = await bubble.execute();
// result.data.totalUpdatedCells === 6
```

### Pattern 3: Read and Process
```typescript
// Get sheet data
const bubble = new GoogleSheetsBubble({
  operation: 'getSheetData',
  spreadsheetId: '1BxiMVs0XRA5nFMdK...',
  sheetName: 'Data',
  includeMetadata: true,
  credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: token }
});

const result = await bubble.execute();

// Access data
const values = result.data.values; // 2D array
const rowCount = result.data.metadata.rowCount;

// Process rows
values.forEach(row => {
  console.log(row[0], row[1]); // First two columns
});
```

### Pattern 4: Copy Template
```typescript
// Copy template spreadsheet
const copyResult = await new GoogleSheetsBubble({
  operation: 'copySpreadsheet',
  spreadsheetId: 'template-id',
  title: 'Monthly Report - January',
  destinationFolderId: 'folder-id', // optional
  credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: token }
}).execute();

const newSpreadsheetUrl = copyResult.data.url;
```

---

## 🔧 PARAMETERS REFERENCE

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `spreadsheetId` | string | Spreadsheet ID from URL |
| `range` | string | A1 notation (e.g., "Sheet1!A1:B10") |
| `valueInputOption` | enum | RAW or USER_ENTERED |
| `valueRenderOption` | enum | FORMATTED_VALUE, UNFORMATTED_VALUE, FORMULA |
| `majorDimension` | enum | ROWS (default) or COLUMNS |

### Value Input Options

- **RAW:** Values are parsed exactly as entered (no formula parsing)
- **USER_ENTERED:** Values are parsed like user input (formulas, dates, numbers)

### Value Render Options

- **FORMATTED_VALUE:** Values formatted as displayed in Sheets
- **UNFORMATTED_VALUE:** Raw values (no formatting)
- **FORMULA:** Return formulas instead of computed values

---

## ⚠️ ERROR HANDLING

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| **401** | Invalid/expired token | Refresh OAuth token |
| **403** | Insufficient permissions | Check spreadsheet sharing |
| **404** | Spreadsheet not found | Verify spreadsheetId |
| **400** | Invalid range | Check A1 notation format |
| **429** | Rate limit exceeded | Wait and retry |

### Error Response Format
```typescript
{
  operation: 'updateCell',
  result: {
    success: false,
    error: 'Invalid range: Sheet1!A1',
    spreadsheetId: '...',
    updatedRange: '',
    updatedRows: 0,
    updatedColumns: 0,
    updatedCells: 0
  }
}
```

---

## 🔒 SECURITY CHECKLIST

- ✅ Use OAuth2 tokens (never hardcode)
- ✅ Validate all input parameters
- ✅ Sanitize error messages (no tokens in logs)
- ✅ Implement rate limiting
- ✅ Use environment variables for credentials
- ✅ Check spreadsheet permissions before operations
- ✅ Log all operations for audit trail

---

## 📊 RATE LIMITS

### Default Limits
- **Standard operations:** 50 requests/minute
- **Batch operations:** 10 requests/minute
- **Time window:** 60 seconds

### Handling Rate Limits
```typescript
// Automatic retry with exponential backoff
// Circuit breaker opens after 5 failures
// Request deduplication prevents duplicate calls
```

---

## 🧪 TESTING

### Unit Test Example
```typescript
describe('GoogleSheetsBubble', () => {
  it('should create spreadsheet', async () => {
    const bubble = new GoogleSheetsBubble({
      operation: 'createSpreadsheet',
      title: 'Test',
      credentials: { [CredentialType.GOOGLE_SHEETS_CRED]: 'token' }
    });

    const result = await bubble.execute();
    expect(result.data.success).toBe(true);
    expect(result.data.spreadsheetId).toBeDefined();
  });
});
```

---

## 📚 ADDITIONAL RESOURCES

- **Full Documentation:** `GOOGLE_SHEETS_IMPLEMENTATION_SUMMARY.md`
- **API Reference:** `google-sheets-bubble.ts` (1,615 lines)
- **Google Sheets API:** https://developers.google.com/sheets/api
- **Google Drive API:** https://developers.google.com/drive/api/v3/reference

---

## 🎯 BEST PRACTICES

### DO ✅
- Use batch operations for multiple updates
- Validate spreadsheetId before operations
- Handle errors gracefully
- Use USER_ENTERED for formula support
- Check permissions before write operations
- Implement retry logic for transient failures

### DON'T ❌
- Don't hardcode credentials
- Don't ignore rate limits
- Don't use huge ranges (split into chunks)
- Don't assume sheets exist (validate first)
- Don't delete without confirmation
- Don't share tokens in logs

---

## 🔄 MIGRATION GUIDE

### From Legacy Operations

| Legacy | New | Migration |
|--------|-----|-----------|
| `getSheet` | `getSpreadsheet` | Direct replacement |
| `getValues` | `getRange` | Add dimension parameter |
| `setValues` | `batchUpdate` | Use updates array |
| `clearValues` | `clearRange` | Direct replacement |
| `getRow` | `getRange` | Specify row range |

---

## 💬 GETTING HELP

1. **Check logs:** Structured JSON logs with correlation IDs
2. **Verify credentials:** Test OAuth token separately
3. **Check permissions:** Ensure spreadsheet is shared
4. **Validate ranges:** Use A1 notation correctly
5. **Monitor rate limits:** Check API usage in Google Console

---

**Status:** ✅ Production Ready
**Version:** 1.0.0
**Last Updated:** January 18, 2026
