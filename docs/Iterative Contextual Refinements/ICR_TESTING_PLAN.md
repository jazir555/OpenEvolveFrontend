# ICR Migration - Comprehensive Testing Plan

**Date:** 2026-02-17  
**Status:** Ready for Testing  
**Priority:** CRITICAL  
**Estimated Time:** 1-2 days

---

## Overview

This document provides a comprehensive testing plan for the ICR upstream migration. All 8 modes must be tested for export/import functionality with the new StateSerializer.

---

## Test Environment Setup

### Prerequisites

- [ ] Backup current `Iterative-Contextual-Refinements/` directory
- [ ] Ensure Node.js 18+ is installed
- [ ] Clear browser cache before testing
- [ ] Have test data ready for each mode

### Test Files to Prepare

1. **Legacy JSON export** - From before migration
2. **New MessagePack export** - After migration
3. **Large state file** - 500KB+ for performance testing
4. **Corrupted file** - For error handling testing

---

## Phase 1: Export Testing

### 1.1 Website Mode (Refine Mode)

**Test Steps:**
1. Open application
2. Select "Refine (Website)" mode
3. Enter test idea: "Test website mode export"
4. Click Export Configuration button
5. Save file as `test-website-export.msgpack.gz`

**Expected Results:**
- ✅ File downloaded successfully
- ✅ File extension is `.msgpack.gz`
- ✅ File size < 50 KB (compressed)
- ✅ Console shows: `[ConfigManager] Exported configuration: XX KB (compressed)`

**Verification:**
```javascript
// Check file size
const fileSize = file.size;
console.assert(fileSize < 50000, 'File should be < 50KB compressed');

// Check file type
console.assert(file.name.endsWith('.msgpack.gz'), 'Should be .msgpack.gz');
```

---

### 1.2 Deepthink Mode

**Test Steps:**
1. Select "Deepthink" mode
2. Enter complex problem: "Design a distributed system for..."
3. Configure strategies: 3, sub-strategies: 5, hypotheses: 10
4. Enable iterative corrections
5. Export configuration

**Expected Results:**
- ✅ File downloaded
- ✅ File size < 100 KB
- ✅ Solution pool versions included
- ✅ Custom prompts preserved

**Critical Checks:**
- [ ] Deepthink pipeline state exported
- [ ] Solution pool versions exported
- [ ] Active tab ID preserved
- [ ] Custom prompts preserved

---

### 1.3 Agentic Mode

**Test Steps:**
1. Select "Agentic" mode
2. Configure tools (enable diff, file, web search)
3. Enter test prompt
4. Export configuration

**Expected Results:**
- ✅ Agentic state exported
- ✅ Tool configurations preserved
- ✅ Conversation history (if any) preserved

---

### 1.4 Contextual Mode

**Test Steps:**
1. Select "Contextual" mode
2. Enable memory agent
3. Enter test prompt
4. Export configuration

**Expected Results:**
- ✅ Contextual state exported
- ✅ Memory agent settings preserved
- ✅ Agent interaction history (if any) preserved

---

### 1.5 Adaptive Deepthink Mode

**Test Steps:**
1. Select "Adaptive Deepthink" mode
2. Configure conversation settings
3. Enter test prompt
4. Export configuration

**Expected Results:**
- ✅ Adaptive state exported
- ✅ Conversation ID preserved
- ✅ Streaming settings preserved

---

### 1.6 MathSolver Mode ⚠️ CRITICAL

**Test Steps:**
1. Select "MathSolver" mode
2. Enter math problem: "Prove that for all n ∈ ℕ, n² ≥ n"
3. Configure Z3/Lean settings
4. Export configuration

**Expected Results:**
- ✅ MathSolver state exported
- ✅ Problem statement preserved
- ✅ Solver settings preserved
- ✅ Custom prompts preserved
- ✅ ICR configuration preserved

**Critical Checks:**
```javascript
// Verify MathSolver-specific fields
console.assert(config.activeMathSolverState !== undefined, 'MathSolver state must be exported');
console.assert(config.customPromptsMathSolver !== undefined, 'MathSolver prompts must be exported');
```

---

### 1.7 GenerativeUI Mode ⚠️ CRITICAL

**Test Steps:**
1. Select "GenerativeUI" mode
2. Enter UI description: "Create a login form with..."
3. Enable interaction capture
4. Configure quality threshold: 0.8
5. Export configuration

**Expected Results:**
- ✅ GenerativeUI state exported
- ✅ Interaction history (if any) preserved
- ✅ Heatmap data (if any) preserved
- ✅ Quality threshold preserved
- ✅ Custom prompts preserved

**Critical Checks:**
```javascript
// Verify GenerativeUI-specific fields
console.assert(config.activeGenerativeUIState !== undefined, 'GenerativeUI state must be exported');
console.assert(config.activeGenerativeUIState.interactionHistory !== undefined, 'Interaction history should be exported');
```

---

### 1.8 React Mode ⚠️ CRITICAL

**Test Steps:**
1. Select "React" mode
2. Enter app description: "Create a React dashboard with..."
3. Configure worker count: 5
4. Enable preview
5. Export configuration

**Expected Results:**
- ✅ React state exported
- ✅ Build artifacts (if any) preserved
- ✅ Worker states preserved
- ✅ Custom prompts preserved

**Critical Checks:**
```javascript
// Verify React-specific fields
console.assert(config.activeReactPipeline !== undefined, 'React pipeline must be exported');
console.assert(config.embeddedAgenticState !== undefined, 'Embedded agentic state must be exported');
```

---

## Phase 2: Import Testing

### 2.1 Legacy JSON Import

**Test Steps:**
1. Create export BEFORE migration (JSON format)
2. Click Import Configuration
3. Select legacy JSON file
4. Verify import successful

**Expected Results:**
- ✅ Import successful
- ✅ Console shows: `[ConfigManager] Imported legacy JSON config`
- ✅ All state restored correctly
- ✅ Mode switched to exported mode
- ✅ Custom prompts restored

**Verification:**
```javascript
// Verify critical fields restored
console.assert(globalState.currentMode === imported.currentMode, 'Mode should be restored');
console.assert(initialIdeaInput.value === imported.initialIdea, 'Initial idea should be restored');
```

---

### 2.2 New MessagePack Import

**Test Steps:**
1. Create export AFTER migration (MessagePack format)
2. Click Import Configuration
3. Select .msgpack.gz file
4. Verify import successful

**Expected Results:**
- ✅ Import successful
- ✅ Console shows: `[ConfigManager] Imported versioned config (v1)`
- ✅ Progress indicator shown during import
- ✅ State sanitized (processing flags reset)
- ✅ All state restored correctly

**Critical Checks:**
- [ ] Processing states reset (isGenerating = false)
- [ ] Running states reset (isRunning = false)
- [ ] Abort controllers not restored
- [ ] DOM elements not restored

---

### 2.3 Cross-Mode Import

**Test Steps:**
1. Export in Deepthink mode
2. Switch to Website mode
3. Import Deepthink export
4. Verify mode switches to Deepthink

**Expected Results:**
- ✅ Mode switches to exported mode (Deepthink)
- ✅ Radio button updated
- ✅ UI updated for Deepthink mode
- ✅ Deepthink pipeline restored

---

### 2.4 MathSolver Import ⚠️ CRITICAL

**Test Steps:**
1. Export MathSolver configuration
2. Switch to Website mode
3. Import MathSolver export
4. Verify MathSolver state restored

**Expected Results:**
- ✅ Mode switches to MathSolver
- ✅ MathSolver state restored
- ✅ Problem statement restored
- ✅ Solver settings restored
- ✅ Custom prompts restored

**Verification:**
```javascript
// Verify MathSolver state after import
console.assert(globalState.currentMode === 'mathsolver', 'Mode should be mathsolver');
console.assert(globalState.activeMathSolverState !== null, 'MathSolver state should be restored');
```

---

### 2.5 GenerativeUI Import ⚠️ CRITICAL

**Test Steps:**
1. Export GenerativeUI configuration
2. Switch to Website mode
3. Import GenerativeUI export
4. Verify GenerativeUI state restored

**Expected Results:**
- ✅ Mode switches to GenerativeUI
- ✅ GenerativeUI state restored
- ✅ Interaction history restored
- ✅ Heatmap data restored
- ✅ Custom prompts restored

---

### 2.6 React Mode Import ⚠️ CRITICAL

**Test Steps:**
1. Export React configuration
2. Switch to Website mode
3. Import React export
4. Verify React state restored

**Expected Results:**
- ✅ Mode switches to React
- ✅ React pipeline restored
- ✅ Build artifacts restored
- ✅ Embedded agentic state restored
- ✅ Custom prompts restored

---

## Phase 3: Performance Testing

### 3.1 Large State Export

**Test Steps:**
1. Create large state (multiple pipelines, solutions)
2. Export configuration
3. Measure export time
4. Measure file size

**Expected Results:**
- ✅ Export completes in < 5 seconds
- ✅ File size < 1 MB (compressed)
- ✅ Progress indicator shown
- ✅ No memory leaks

**Performance Metrics:**
```
Target Performance:
┌──────────────┬──────────┬──────────────┐
│ State Size   │ JSON     │ MessagePack  │
├──────────────┼──────────┼──────────────┤
│ 100 KB       │ < 200ms  │ < 150ms      │
│ 500 KB       │ < 500ms  │ < 400ms      │
│ 1 MB         │ < 1000ms │ < 800ms      │
└──────────────┴──────────┴──────────────┘
```

---

### 3.2 Large State Import

**Test Steps:**
1. Import large state file
2. Measure import time
3. Verify state restored correctly
4. Check memory usage

**Expected Results:**
- ✅ Import completes in < 5 seconds
- ✅ State sanitized correctly
- ✅ No memory leaks
- ✅ UI responsive after import

---

### 3.3 Compression Ratio Testing

**Test Steps:**
1. Export same state in JSON and MessagePack
2. Compare file sizes
3. Calculate compression ratio

**Expected Results:**
- ✅ MessagePack 30% smaller than JSON
- ✅ Compressed 70% smaller than JSON
- ✅ No data loss from compression

**Compression Metrics:**
```
Expected Compression:
┌──────────────┬──────────┬──────────────┬────────────┐
│ Mode         │ JSON     │ MessagePack  │ Compressed │
├──────────────┼──────────┼──────────────┼────────────┤
│ Website      │ 50 KB    │ 35 KB        │ 15 KB      │
│ Deepthink    │ 200 KB   │ 140 KB       │ 60 KB      │
│ MathSolver   │ 150 KB   │ 105 KB       │ 45 KB      │
│ GenerativeUI │ 300 KB   │ 210 KB       │ 90 KB      │
│ React        │ 250 KB   │ 175 KB       │ 75 KB      │
└──────────────┴──────────┴──────────────┴────────────┘
```

---

## Phase 4: Error Handling Testing

### 4.1 Corrupted File Import

**Test Steps:**
1. Create corrupted file (truncate binary data)
2. Attempt import
3. Verify error handling

**Expected Results:**
- ✅ Error message shown to user
- ✅ Application doesn't crash
- ✅ Console shows error details
- ✅ State unchanged

**Expected Error:**
```
[ConfigManager] Import failed: Failed to decompress data. The file may be corrupted.
Alert: "Import failed: Failed to decompress data. The file may be corrupted."
```

---

### 4.2 Invalid JSON Import

**Test Steps:**
1. Create invalid JSON file
2. Attempt import
3. Verify error handling

**Expected Results:**
- ✅ Error message shown
- ✅ Application doesn't crash
- ✅ State unchanged

---

### 4.3 Missing Fields Import

**Test Steps:**
1. Create export with missing critical fields
2. Attempt import
3. Verify validation

**Expected Results:**
- ✅ Validation error shown
- ✅ Missing fields detected
- ✅ State unchanged

---

## Phase 5: Integration Testing

### 5.1 ICR Configuration Preservation

**Test Steps:**
1. Configure ICR settings (enable prediction, learning)
2. Export configuration
3. Import configuration
4. Verify ICR settings preserved

**Expected Results:**
- ✅ ICR enabled/disabled preserved
- ✅ Prediction settings preserved
- ✅ Learning settings preserved
- ✅ Pattern storage configuration preserved

---

### 5.2 Auto-Refine Preservation

**Test Steps:**
1. Enable auto-refine
2. Configure auto-refine settings
3. Export configuration
4. Import configuration
5. Verify auto-refine preserved

**Expected Results:**
- ✅ Auto-refine enabled/disabled preserved
- ✅ Auto-refine settings preserved
- ✅ Model parameters preserved

---

### 5.3 Custom Prompts Preservation

**Test Steps:**
1. Configure custom prompts for all modes
2. Export configuration
3. Import configuration
4. Verify all custom prompts restored

**Expected Results:**
- ✅ Website custom prompts restored
- ✅ Deepthink custom prompts restored
- ✅ Agentic custom prompts restored
- ✅ Contextual custom prompts restored
- ✅ Adaptive Deepthink custom prompts restored
- ✅ MathSolver custom prompts restored ⚠️
- ✅ GenerativeUI custom prompts restored ⚠️
- ✅ React custom prompts restored ⚠️

---

## Test Results Template

### Export Test Results

| Mode | File Size | Time | Status | Notes |
|------|-----------|------|--------|-------|
| Website | | | ⏳ | |
| Deepthink | | | ⏳ | |
| Agentic | | | ⏳ | |
| Contextual | | | ⏳ | |
| Adaptive Deepthink | | | ⏳ | |
| MathSolver | | | ⏳ | |
| GenerativeUI | | | ⏳ | |
| React | | | ⏳ | |

### Import Test Results

| Test Type | Status | Time | Notes |
|-----------|--------|------|-------|
| Legacy JSON | ⏳ | | |
| MessagePack | ⏳ | | |
| Cross-Mode | ⏳ | | |
| MathSolver | ⏳ | | |
| GenerativeUI | ⏳ | | |
| React | ⏳ | | |

### Performance Test Results

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Export (100KB) | < 200ms | | ⏳ |
| Import (100KB) | < 250ms | | ⏳ |
| Export (1MB) | < 1000ms | | ⏳ |
| Import (1MB) | < 1200ms | | ⏳ |
| Compression Ratio | > 60% | | ⏳ |

---

## Sign-Off Checklist

### Critical Tests (Must Pass)

- [ ] MathSolver export/import working
- [ ] GenerativeUI export/import working
- [ ] React export/import working
- [ ] ICR configuration preserved
- [ ] Auto-refine preserved
- [ ] Custom prompts preserved (all 8 modes)
- [ ] Legacy JSON imports working
- [ ] State sanitization working

### Performance Tests (Should Pass)

- [ ] Export < 5 seconds for all modes
- [ ] Import < 5 seconds for all modes
- [ ] Compression ratio > 60%
- [ ] No memory leaks

### Error Handling (Must Pass)

- [ ] Corrupted file handled gracefully
- [ ] Invalid JSON handled gracefully
- [ ] Missing fields detected
- [ ] User-friendly error messages

---

**Testing Status:** ⏳ Ready to Start  
**Estimated Duration:** 1-2 days  
**Priority:** CRITICAL  

---

**Document Created:** 2026-02-17  
**Version:** 1.0  
**Approved By:** [Pending]
