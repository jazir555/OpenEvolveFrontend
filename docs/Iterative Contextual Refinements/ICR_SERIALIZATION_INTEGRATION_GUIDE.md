# ICR StateSerializer Integration Guide

**Date:** 2026-02-17  
**Status:** Ready for Implementation  
**Priority:** HIGH

---

## Overview

This document provides step-by-step instructions for integrating the StateSerializer framework into the local ICR-enhanced version while preserving all custom features (MathSolver, GenerativeUI, React mode, auto-refine, ICR configuration).

---

## Files to Update

### 1. Core/ConfigManager.ts

**Current State:**
- Has local exportConfiguration() using JSON
- Has local handleImportConfiguration()
- Supports all 8 modes (5 upstream + 3 local)
- Has ICR configuration export/import
- Has auto-refine configuration

**Required Changes:**

#### Add StateSerializer Imports

```typescript
// Add at top of file, after existing imports
import {
    serialize,
    deserialize,
    sanitizeState,
    downloadBlob,
    getFileExtension,
    formatBytes,
} from './StateSerializer';
import { convertLegacyToVersioned, migrateToLatest } from './StateSerializer';
```

#### Update exportConfiguration()

Replace the JSON export with StateSerializer:

```typescript
export async function exportConfiguration() {
    // ... existing config building code ...
    
    // NEW: Use StateSerializer instead of JSON
    const versionedConfig = {
        _version: 1,
        _exportedAt: new Date().toISOString(),
        _mode: globalState.currentMode,
        data: config  // Existing config object
    };
    
    // Serialize with MessagePack + compression
    const blob = await serialize(versionedConfig, {
        format: 'msgpack',
        compress: true,
        onProgress: (percent) => {
            console.log(`Export progress: ${percent}%`);
        }
    });
    
    // Download with appropriate extension
    const timestamp = new Date().toISOString().replace(/[:]/g, '-').split('.')[0];
    const filename = `iterative-studio-config-${timestamp}.msgpack.gz`;
    downloadBlob(blob, filename);
    
    console.log(`Exported configuration: ${formatBytes(blob.size)}`);
}
```

#### Update handleImportConfiguration()

Add StateSerializer import support:

```typescript
export async function handleImportConfiguration(event: Event) {
    // ... existing validation code ...
    
    const file = fileInputTarget.files[0];
    
    try {
        // Detect format and deserialize
        const imported = await deserialize<any>(file, (percent) => {
            console.log(`Import progress: ${percent}%`);
        });
        
        // Check if versioned or legacy format
        let configToUse;
        if (imported._version) {
            // New versioned format - migrate to latest
            configToUse = migrateToLatest(imported);
        } else {
            // Legacy JSON format - convert
            configToUse = convertLegacyToVersioned(imported);
        }
        
        // Sanitize the state (reset processing states)
        const sanitized = sanitizeState(configToUse.data);
        
        // Restore configuration using existing logic
        await restoreConfiguration(sanitized);
        
        console.log(`Successfully imported configuration from ${file.name}`);
        
    } catch (error) {
        console.error('Import failed:', error);
        alert(`Import failed: ${error.message}. The file may be corrupted or in an unsupported format.`);
    }
}
```

#### Add restoreConfiguration() Helper

```typescript
async function restoreConfiguration(config: any) {
    // Existing restoration logic from handleImportConfiguration
    // This extracts the restoration code into a reusable function
    
    globalState.currentMode = config.currentMode;
    // ... rest of existing restoration code ...
}
```

---

### 2. Core/App.ts

**Current State:**
- Initializes all 8 modes
- Has local export/import button handlers
- Has ICR event bridge initialization

**Required Changes:**

#### Add StateSerializer Initialization

```typescript
// Add after existing imports
import { initializeIcrIntegration } from '../glue/adapters/icr-adapter';

// In App.init():
public static init() {
    this.initializeGlobalFunctions();
    this.initializeUI();
    this.initializeEventListeners();
    startIcrEventBridge();
    
    // NEW: Initialize ICR integration if available
    try {
        initializeIcrIntegration();
        console.log('[App] ICR integration initialized');
    } catch (error) {
        console.warn('[App] ICR integration not available:', error);
    }
    
    LayoutController.initialize();
    GlobalModals.initialize();
}
```

---

### 3. Core/Types.ts

**Current State:**
- Has ApplicationMode type with 'mathsolver', 'generativeui', 'react'
- Has ExportedConfig interface

**Required Changes:**

#### Update ExportedConfig Interface

```typescript
export interface ExportedConfig {
    // ... existing fields ...
    
    // NEW: Add fields for StateSerializer compatibility
    _version?: number;  // Optional for backward compatibility
    _exportedAt?: string;
    _mode?: ApplicationMode;
    
    // Ensure all custom mode states are included
    activeMathSolverState?: any | null;
    activeGenerativeUIState?: any | null;
    activeReactState?: any | null;
    
    // Custom prompts for all modes including local
    customPromptsMathSolver?: any;
    customPromptsGenerativeUI?: any;
    customPromptsReact?: any;
}
```

---

## Testing Checklist

After integration:

### Export Testing
- [ ] Export configuration in Website mode
- [ ] Export configuration in Deepthink mode
- [ ] Export configuration in Agentic mode
- [ ] Export configuration in Contextual mode
- [ ] Export configuration in Adaptive Deepthink mode
- [ ] Export configuration in MathSolver mode ⚠️ CRITICAL
- [ ] Export configuration in GenerativeUI mode ⚠️ CRITICAL
- [ ] Export configuration in React mode ⚠️ CRITICAL
- [ ] Verify file size is smaller (compression working)
- [ ] Verify file extension is .msgpack.gz

### Import Testing
- [ ] Import legacy JSON export
- [ ] Import new MessagePack export
- [ ] Import in different mode than exported
- [ ] Verify MathSolver state restored correctly ⚠️ CRITICAL
- [ ] Verify GenerativeUI state restored correctly ⚠️ CRITICAL
- [ ] Verify React state restored correctly ⚠️ CRITICAL
- [ ] Verify ICR configuration restored
- [ ] Verify auto-refine configuration restored
- [ ] Verify custom prompts restored

### State Sanitization Testing
- [ ] Export during processing
- [ ] Import and verify processing states reset
- [ ] Verify isGenerating flags reset
- [ ] Verify abort controllers not restored

---

## Rollback Plan

If integration fails:

1. **Keep Backup:** Original ConfigManager.ts and App.ts
2. **Revert:** Replace with backup files
3. **Test:** Verify export/import still works with JSON
4. **Debug:** Review error logs, fix issues
5. **Retry:** Apply fixes and try again

---

## Migration Timeline

| Task | Estimated Time | Status |
|------|---------------|--------|
| Update ConfigManager.ts | 2-3 hours | ⏳ Pending |
| Update App.ts | 1 hour | ⏳ Pending |
| Update Types.ts | 30 minutes | ⏳ Pending |
| Testing - Export | 2 hours | ⏳ Pending |
| Testing - Import | 2 hours | ⏳ Pending |
| Testing - All modes | 4 hours | ⏳ Pending |
| **TOTAL** | **11-12 hours** | ⏳ Pending |

---

## Success Criteria

- ✅ All 8 modes can export state
- ✅ All 8 modes can import state
- ✅ File sizes reduced by 50-70% (compression)
- ✅ Import/export completes in <5 seconds
- ✅ State sanitization working (processing states reset)
- ✅ Legacy JSON imports still work
- ✅ MathSolver state preserved ⚠️ CRITICAL
- ✅ GenerativeUI state preserved ⚠️ CRITICAL
- ✅ React state preserved ⚠️ CRITICAL
- ✅ ICR configuration preserved ⚠️ CRITICAL
- ✅ Auto-refine configuration preserved ⚠️ CRITICAL

---

**Document Version:** 1.0  
**Created:** 2026-02-17  
**Ready for Implementation:** YES
