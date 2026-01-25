# TypeScript Error Examples

This document shows specific examples of each major error category to help with fixing.

---

## 1. Duplicate Type Definitions (TS2484)

### Error
```
src/types/nodes.ts(1404,3): error TS2484: Export declaration conflicts with exported declaration of 'OpenEvolveNodeData'.
src/core/types/nodes.ts(1394,3): error TS2484: Export declaration conflicts with exported declaration of 'OpenEvolveNodeData'.
```

### Root Cause
Both `src/types/nodes.ts` and `src/core/types/nodes.ts` export `OpenEvolveNodeData`.

### Fix
**Option A:** Remove `src/core/types/` directory
```bash
# Remove duplicate
rm -rf src/core/types/
# Update all imports from @/core/types/... to @/types/...
```

**Option B:** Remove `src/types/` directory
```bash
# Remove duplicate
rm -rf src/types/
# Update all imports from @/types/... to @/core/types/...
```

---

## 2. Missing Type Properties (TS2339)

### Error Example 1: IntegrationConfiguration
```typescript
// In: src/utils/createEnhancedOpenEvancedPlugin.ts:197
error TS2339: Property 'rest_api' does not exist on type 'IntegrationConfiguration'.
```

### Current Type
```typescript
// src/types/enhanced-plugin-types.ts
interface IntegrationConfiguration {
  enabled: boolean;
  timeout: number;
  retry_policy: RetryPolicy;
  // MISSING: rest_api, graphql, websocket
}
```

### Fix
```typescript
interface IntegrationConfiguration {
  enabled: boolean;
  timeout: number;
  retry_policy: RetryPolicy;

  // Add these properties
  rest_api?: {
    enabled: boolean;
    base_url: string;
    timeout: number;
  };
  graphql?: {
    enabled: boolean;
    endpoint: string;
    timeout: number;
  };
  websocket?: {
    enabled: boolean;
    url: string;
    reconnect_interval: number;
  };
}
```

### Error Example 2: EnhancedOpenEvolvePluginState
```typescript
// In: src/utils/createEnhancedOpenEvolvePlugin.ts:287
error TS2339: Property 'performanceProfiles' does not exist on type 'EnhancedOpenEvolvePluginState'.
```

### Current Type
```typescript
// src/types/enhanced-plugin-types.ts
interface EnhancedOpenEvolvePluginState {
  config: EnhancedConfig;
  status: PluginStatus;
  statistics: Statistics;
  // MISSING: performanceProfiles, securityProfiles, executionStatistics, errorStatistics, validationHistory
}
```

### Fix
```typescript
interface EnhancedOpenEvolvePluginState {
  config: EnhancedConfig;
  status: PluginStatus;
  statistics: Statistics;

  // Add these properties
  performanceProfiles?: PerformanceProfile[];
  securityProfiles?: SecurityProfile[];
  executionStatistics?: ExecutionStatistics;
  errorStatistics?: ErrorStatistics;
  validationHistory?: ValidationRecord[];
}
```

---

## 3. Missing Exported Members (TS1205, TS2724)

### Error
```typescript
// In: src/utils/index.ts:23
error TS2724: '"./createOpenEvolvePlugin"' has no exported member named 'getOpenEvolvePlugin'. Did you mean 'OpenEvolvePlugin'?
```

### Problem
```typescript
// src/utils/index.ts (line 23)
export { getOpenEvolvePlugin } from './createOpenEvolvePlugin'; // ❌ doesn't exist

// src/utils/createOpenEvolvePlugin.ts
export { createOpenEvolvePlugin, OpenEvolvePlugin }; // ❌ getOpenEvolvePlugin not exported
```

### Fix
**Option A:** Export the function
```typescript
// src/utils/createOpenEvolvePlugin.ts
export function getOpenEvolvePlugin() {
  return openevolvePlugin;
}

export { createOpenEvolvePlugin, OpenEvolvePlugin, getOpenEvolvePlugin };
```

**Option B:** Remove the import
```typescript
// src/utils/index.ts
// Remove line 23 or comment it out
// export { getOpenEvolvePlugin } from './createOpenEvolvePlugin';
```

---

## 4. Property Access on Unknown Types (TS2339)

### Error
```typescript
// In: src/components/nodes/DecompositionNodeComponent.tsx:60
error TS2339: Property 'subProblems' does not exist on type 'unknown'.
```

### Problem
```typescript
// src/components/nodes/DecompositionNodeComponent.tsx
export const DecompositionNodeComponent = memo((props: NodeProps<DecompositionNodeData>) => {
  const { data } = props;
  if (!data.subProblems) return null; // ❌ 'data' is 'unknown'
```

### Root Cause
`NodeProps` generic constraint is wrong:
```typescript
// src/types/nodes.ts
interface DecompositionNodeData {
  subProblems: SubProblem[];
  qualityScore: number;
  // ...
}

// But NodeProps expects Node<> type, not just data interface
export const DecompositionNodeComponent = memo((props: NodeProps<DecompositionNodeData>) => {
  // ❌ DecompositionNodeData doesn't satisfy constraint Node<...>
```

### Fix
**Option A:** Extract data properly
```typescript
export const DecompositionNodeComponent = memo((props: NodeProps) => {
  const data = props.data as DecompositionNodeData; // Type assertion
  if (!data.subProblems) return null; // ✅ Now it works
```

**Option B:** Fix the type constraint
```typescript
// Create proper node data type that satisfies React Flow
interface NodeDataType {
  subProblems?: SubProblem[];
  qualityScore?: number;
  // ... other properties
}

export const DecompositionNodeComponent = memo((props: NodeProps<NodeDataType>) => {
  const { data } = props;
  if (!data.subProblems) return null; // ✅ Now properly typed
```

---

## 5. Wrong Import Paths (TS2307)

### Error
```typescript
// In: src/components/config/EnhancedOpenEvolveConfigPanel.tsx:2
error TS2307: Cannot find module '../types/enhanced-plugin-types' or its corresponding type declarations.
```

### Problem
```typescript
// src/components/config/EnhancedOpenEvolveConfigPanel.tsx
import { EnhancedOpenEvolvePluginState } from '../types/enhanced-plugin-types'; // ❌ Wrong path
```

### File Structure
```
src/
├── components/
│   └── config/
│       └── EnhancedOpenEvolveConfigPanel.tsx
└── types/
    └── enhanced-plugin-types.ts
```

From `src/components/config/`, going up two levels (`../`) gets you to `src/components/`, not `src/`.

### Fix
**Use correct relative path:**
```typescript
// From src/components/config/ to src/types/ needs ../../types/
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
```

**Or use absolute imports (recommended):**
```typescript
// In tsconfig.json, ensure paths are configured:
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@/types/*": ["src/types/*"],
      "@/utils/*": ["src/utils/*"]
    }
  }
}

// Then use:
import { EnhancedOpenEvolvePluginState } from '@/types/enhanced-plugin-types';
```

---

## 6. Export Conflicts (TS2323, TS2484)

### Error
```typescript
// In: src/utils/enhancedErrorHandling.ts:13
error TS2323: Cannot redeclare exported variable 'AdvancedErrorClassifier'.

// In: src/utils/enhancedErrorHandling.ts:1038
error TS2484: Export declaration conflicts with exported declaration of 'AdvancedErrorClassifier'.
```

### Problem
```typescript
// src/utils/enhancedErrorHandling.ts
export class AdvancedErrorClassifier { // ❌ First declaration
  // ... implementation ...
}

// ... later in the same file ...

export { AdvancedErrorClassifier }; // ❌ Second export (conflict)
```

### Fix
**Remove the duplicate export:**
```typescript
// Keep only the class export
export class AdvancedErrorClassifier {
  // ... implementation ...
}

// Remove this line:
// export { AdvancedErrorClassifier };
```

---

## 7. Shorthand Property Issues (TS18004)

### Error
```typescript
// In: src/utils/index.ts:373
error TS18004: No value exists in scope for the shorthand property 'createEnhancedOpenEvolvePlugin'.
```

### Problem
```typescript
// src/utils/index.ts
import { SomethingElse } from './createEnhancedOpenEvolvePlugin';

export {
  createEnhancedOpenEvolvePlugin, // ❌ Not imported
  getEnhancedOpenEvolvePlugin,    // ❌ Not imported
  resetEnhancedOpenEvolvePlugin   // ❌ Not imported
};
```

### Fix
**Import what you export:**
```typescript
// Import the functions
import {
  createEnhancedOpenEvolvePlugin,
  getEnhancedOpenEvolvePlugin,
  resetEnhancedOpenEvolvePlugin
} from './createEnhancedOpenEvolvePlugin';

// Now you can export them
export {
  createEnhancedOpenEvancePlugin,
  getEnhancedOpenEvolvePlugin,
  resetEnhancedOpenEvolvePlugin
};
```

**Or use re-export (better):**
```typescript
export {
  createEnhancedOpenEvolvePlugin,
  getEnhancedOpenEvolvePlugin,
  resetEnhancedOpenEvolvePlugin
} from './createEnhancedOpenEvolvePlugin';
```

---

## 8. Type Assignability Issues (TS2322)

### Error
```typescript
// In: src/components/config/EnhancedOpenEvolveConfigPanel.tsx:380
error TS2322: Type '{ children: string; jsx: true; }' is not assignable to type 'DetailedHTMLProps<StyleHTMLAttributes<HTMLStyleElement>, HTMLStyleElement>'.
  Property 'jsx' does not exist on type 'DetailedHTMLProps<StyleHTMLAttributes<HTMLStyleElement>, HTMLStyleElement>'.
```

### Problem
```typescript
<style jsx>{`
  // ... css ...
`}</style>
```

The `jsx` prop is specific to styled-jsx, but TypeScript doesn't recognize it.

### Fix
**Install styled-jsx types:**
```bash
npm install --save-dev @types/styled-jsx
```

**Or disable the error:**
```typescript
<style jsx global>{`/* @jsx jsx */`}</style>
```

**Or use regular style tag:**
```typescript
<style>{`
  // ... css ...
`}</style>
```

---

## Summary Table

| Error Type | Count | Fix Complexity | Files Affected |
|------------|-------|----------------|----------------|
| Duplicate exports | 222 | Medium | 2 |
| Missing properties | 200+ | Low-Medium | 10+ |
| Property on unknown | 366 | High | 20+ |
| Wrong import paths | 53 | Low | 30+ |
| Missing exports | 50 | Low | 15+ |
| Export conflicts | 72 | Low | 5 |
| Shorthand properties | 38 | Low | 2 |

---

## Recommended Fix Order

1. **Start with duplicates** (222 errors fixed)
   - Remove `src/core/types/` or `src/types/`
   - Update all imports

2. **Fix imports** (53 errors fixed)
   - Correct all import paths
   - Use absolute imports

3. **Add missing exports** (50 errors fixed)
   - Export all referenced functions
   - Clean up index files

4. **Add missing properties** (200+ errors fixed)
   - Update all interface definitions
   - Add all referenced properties

5. **Fix node types** (366 errors fixed)
   - Use proper type assertions
   - Fix NodeProps constraints

6. **Clean up conflicts** (72 errors fixed)
   - Remove duplicate exports
   - Fix re-exports

---

**Total estimated fixes:** 1,073 errors
**Time estimate:** 10-14 hours for complete resolution
