# Google Drive Bubble - Quick Reference

## File Location
`BubbleLab/packages/bubble-core/src/bubbles/service-bubble/google-drive-bubble.ts`

## Quick Stats
- **Total Operations:** 13
- **Lines of Code:** 987
- **Security Features:** ✅ Complete
- **Status:** Production Ready

---

## All 13 Operations

### File Operations
```typescript
// 1. Upload File
{ operation: 'uploadFile', fileName, content, mimeType?, parents? }

// 2. Download File
{ operation: 'downloadFile', fileId }

// 3. Delete File
{ operation: 'deleteFile', fileId }

// 4. Update File Content
{ operation: 'updateFile', fileId, content, mimeType? }

// 5. Copy File
{ operation: 'copyFile', fileId, fileName, parents? }
```

### Folder Operations
```typescript
// 6. Create Folder
{ operation: 'createFolder', folderName, parents? }

// 7. List Files
{ operation: 'listFiles', pageSize?, pageToken?, query?, orderBy? }

// 8. Search Files
{ operation: 'searchFiles', query, pageSize?, pageToken? }
```

### Sharing Operations
```typescript
// 9. Share File
{
  operation: 'shareFile',
  fileId,
  role: 'reader' | 'writer' | 'commenter' | 'owner',
  type: 'user' | 'group' | 'anyone' | 'domain',
  emailAddress?,
  allowFileDiscovery?
}

// 10. Get Permissions (NEW)
{ operation: 'getPermissions', fileId }

// 11. Revoke Access (NEW)
{ operation: 'revokeAccess', fileId, permissionId }
```

### Metadata Operations
```typescript
// 12. Get File Info
{ operation: 'getFileInfo', fileId }

// 13. Update Metadata (NEW)
{
  operation: 'updateMetadata',
  fileId,
  fileName?,
  description?,
  starred?,
  parents?
}
```

---

## Security Limits

| Feature | Limit |
|---------|-------|
| Max File Size | 5GB |
| Upload Rate | 5/minute |
| Other Operations | 50/minute |
| File Name Length | 255 chars |
| Default Timeout | 30 seconds |
| Upload Timeout | 60 seconds |

---

## Common Patterns

### Upload with Folder
```typescript
const bubble = new GoogleDriveBubble({
  operation: 'uploadFile',
  fileName: 'report.pdf',
  content: fileBuffer,
  mimeType: 'application/pdf',
  parents: ['folder_id_123'],
  credentials: { [CredentialType.GOOGLE_DRIVE_CRED]: token },
});
```

### Share and Manage Permissions
```typescript
// Share
await new GoogleDriveBubble({
  operation: 'shareFile',
  fileId: 'abc123',
  role: 'writer',
  type: 'user',
  emailAddress: 'user@example.com',
  credentials: { ... },
}).execute();

// Get Permissions
const { data } = await new GoogleDriveBubble({
  operation: 'getPermissions',
  fileId: 'abc123',
  credentials: { ... },
}).execute();

// Revoke
await new GoogleDriveBubble({
  operation: 'revokeAccess',
  fileId: 'abc123',
  permissionId: data.permissions[0].id,
  credentials: { ... },
}).execute();
```

### Update Metadata
```typescript
await new GoogleDriveBubble({
  operation: 'updateMetadata',
  fileId: 'abc123',
  fileName: 'New Name.pdf',
  description: 'Updated description',
  starred: true,
  credentials: { ... },
}).execute();
```

---

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| Rate limit exceeded | Too many requests | Wait 1 minute |
| File size exceeds 5GB | File too large | Compress or split |
| Path traversal sequences | ".." in fileName | Remove ".." from name |
| Access token required | Missing credentials | Provide OAuth token |
| File not found | Invalid fileId | Check file ID |

---

## OAuth2 Scopes

**Default (Recommended):**
```
https://www.googleapis.com/auth/drive.file
```

**Full Access (Shows Warning):**
```
https://www.googleapis.com/auth/drive
```

**Additional Scopes:**
- Documents: `https://www.googleapis.com/auth/documents`
- Spreadsheets: `https://www.googleapis.com/auth/spreadsheets`

---

## Testing

```bash
# Run tests
cd BubbleLab
npm test -- google-drive-bubble.test.ts

# Check TypeScript
npx tsc --noEmit packages/bubble-core/src/bubbles/service-bubble/google-drive-bubble.ts
```

---

## Response Format

All operations return:
```typescript
{
  success: boolean,
  data: {
    fileId?: string,
    fileName?: string,
    // ... operation-specific fields
    status: string
  },
  error?: string,
  meta: {
    operation: string,
    fileId?: string,
    fileName?: string
  }
}
```

---

## Need Help?

- **Implementation Summary:** `GOOGLE_DRIVE_BUBBLE_IMPLEMENTATION_SUMMARY.md`
- **Source Code:** `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/google-drive-bubble.ts`
- **Tests:** `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/google-drive-bubble.test.ts`
