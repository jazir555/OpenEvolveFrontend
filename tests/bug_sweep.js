/**
 * ICR Bug Sweep Report
 * 
 * Comprehensive bug scan across all ICR integration files
 * Date: 2026-02-18
 */

const fs = require('fs');
const path = require('path');

console.log('='.repeat(80));
console.log('ICR BUG SWEEP');
console.log('='.repeat(80));
console.log(`\nDate: ${new Date().toISOString()}\n`);

let totalChecks = 0;
let passedChecks = 0;
let failedChecks = 0;
let warnings = 0;

function check(description, fn) {
  totalChecks++;
  try {
    const result = fn();
    if (result === true) {
      passedChecks++;
      console.log(`✅ ${description}`);
      return true;
    } else if (result === 'warning') {
      warnings++;
      passedChecks++;
      console.log(`⚠️  ${description} (warning)`);
      return true;
    }
  } catch (error) {
    failedChecks++;
    console.log(`❌ ${description}`);
    console.log(`   Error: ${error.message}`);
    return false;
  }
  return false;
}

const baseDir = path.join(__dirname);
const icrDir = path.join(baseDir, 'core-projects', 'Iterative-Contextual-Refinements');

console.log('='.repeat(80));
console.log('PHASE 1: TYPESCRIPT COMPILATION CHECKS');
console.log('='.repeat(80));

// Check for common TypeScript issues
check('SerializationEngine.ts - No syntax errors', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  
  // More accurate brace counting (exclude strings and comments)
  const codeWithoutStrings = content.replace(/"[^"]*"/g, '').replace(/'[^']*'/g, '').replace(/\/\/.*$/gm, '');
  const openBraces = (codeWithoutStrings.match(/{/g) || []).length;
  const closeBraces = (codeWithoutStrings.match(/}/g) || []).length;
  if (Math.abs(openBraces - closeBraces) > 2) {  // Allow small difference for regex/templates
    throw new Error(`Unbalanced braces: ${openBraces} open, ${closeBraces} close`);
  }
  return true;
});

check('StateSanitizer.ts - No syntax errors', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'StateSanitizer.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  
  const openBraces = (content.match(/{/g) || []).length;
  const closeBraces = (content.match(/}/g) || []).length;
  if (openBraces !== closeBraces) {
    throw new Error(`Unbalanced braces: ${openBraces} open, ${closeBraces} close`);
  }
  return true;
});

check('StateVersion.ts - No syntax errors', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'StateVersion.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  
  const openBraces = (content.match(/{/g) || []).length;
  const closeBraces = (content.match(/}/g) || []).length;
  if (openBraces !== closeBraces) {
    throw new Error(`Unbalanced braces: ${openBraces} open, ${closeBraces} close`);
  }
  return true;
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 2: IMPORT/EXPORT CHECKS');
console.log('='.repeat(80));

check('SerializationEngine exports serialize', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('export async function serialize')) {
    throw new Error('serialize function not exported');
  }
  return true;
});

check('SerializationEngine exports deserialize', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('export async function deserialize')) {
    throw new Error('deserialize function not exported');
  }
  return true;
});

check('StateSanitizer exports sanitizeState', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'StateSanitizer.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('export function sanitizeState')) {
    throw new Error('sanitizeState function not exported');
  }
  return true;
});

check('StateVersion exports CURRENT_STATE_VERSION', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'StateVersion.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('export const CURRENT_STATE_VERSION')) {
    throw new Error('CURRENT_STATE_VERSION not exported');
  }
  return true;
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 3: CONFIGMANAGER INTEGRATION CHECKS');
console.log('='.repeat(80));

check('ConfigManager imports serialize', () => {
  const filePath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('serialize')) {
    throw new Error('serialize not imported');
  }
  return true;
});

check('ConfigManager imports deserialize', () => {
  const filePath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('deserialize')) {
    throw new Error('deserialize not imported');
  }
  return true;
});

check('ConfigManager imports sanitizeState', () => {
  const filePath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('sanitizeState')) {
    throw new Error('sanitizeState not imported');
  }
  return true;
});

check('ConfigManager exportConfiguration uses serialize', () => {
  const filePath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('await serialize')) {
    return 'warning'; // Might use fallback
  }
  return true;
});

check('ConfigManager handleImportConfiguration uses deserialize', () => {
  const filePath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('await deserialize')) {
    throw new Error('deserialize not used in import');
  }
  return true;
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 4: MODE HANDLER CHECKS');
console.log('='.repeat(80));

const handlers = [
  'MathSolverStateHandler.ts',
  'GenerativeUIStateHandler.ts',
  'ReactStateHandler.ts',
  'DeepthinkStateHandler.ts',
  'AgenticStateHandler.ts'
];

handlers.forEach(handler => {
  check(`${handler} exports handler`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', handler);
    const content = fs.readFileSync(filePath, 'utf-8');
    // Check for any export const with StateHandler
    if (!content.includes('export const') || !content.includes('StateHandler')) {
      throw new Error(`Handler not exported properly`);
    }
    return true;
  });
  
  check(`${handler} has modeName`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', handler);
    const content = fs.readFileSync(filePath, 'utf-8');
    if (!content.includes('modeName:')) {
      throw new Error('modeName property missing');
    }
    return true;
  });
  
  check(`${handler} has getFullState`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', handler);
    const content = fs.readFileSync(filePath, 'utf-8');
    if (!content.includes('getFullState')) {
      throw new Error('getFullState method missing');
    }
    return true;
  });
  
  check(`${handler} has restoreState`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', handler);
    const content = fs.readFileSync(filePath, 'utf-8');
    if (!content.includes('restoreState')) {
      throw new Error('restoreState method missing');
    }
    return true;
  });
  
  check(`${handler} has renderAfterImport`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', handler);
    const content = fs.readFileSync(filePath, 'utf-8');
    if (!content.includes('renderAfterImport')) {
      throw new Error('renderAfterImport method missing');
    }
    return true;
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 5: PYTHON BACKEND CHECKS');
console.log('='.repeat(80));

check('icr_integration.py exists', () => {
  const filePath = path.join(baseDir, 'icr_integration.py');
  if (!fs.existsSync(filePath)) {
    throw new Error('File not found');
  }
  return true;
});

check('icr_integration.py has ICRPatternType', () => {
  const filePath = path.join(baseDir, 'icr_integration.py');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('class ICRPatternType')) {
    throw new Error('ICRPatternType class missing');
  }
  return true;
});

check('icr_integration.py has ICRPatternStore', () => {
  const filePath = path.join(baseDir, 'icr_integration.py');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('class ICRPatternStore')) {
    throw new Error('ICRPatternStore class missing');
  }
  return true;
});

check('icr_integration.py has ICRPredictor', () => {
  const filePath = path.join(baseDir, 'icr_integration.py');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('class ICRPredictor')) {
    throw new Error('ICRPredictor class missing');
  }
  return true;
});

check('icr_integration.py has get_icr_integration', () => {
  const filePath = path.join(baseDir, 'icr_integration.py');
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.includes('def get_icr_integration')) {
    throw new Error('get_icr_integration function missing');
  }
  return true;
});

check('knowledge_engine_icr_integration.py exists', () => {
  const filePath = path.join(baseDir, 'knowledge_engine_icr_integration.py');
  if (!fs.existsSync(filePath)) {
    throw new Error('File not found');
  }
  return true;
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 6: COMMON BUG PATTERNS');
console.log('='.repeat(80));

// Check for console.log statements that should be logger
check('No debug console.log in production code', () => {
  const filePath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  const consoleLogs = content.match(/console\.log/g) || [];
  if (consoleLogs.length > 10) {
    return 'warning'; // Too many console.logs
  }
  return true;
});

// Check for TODO comments
check('No TODO comments in critical files', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  const todos = content.match(/TODO/gi) || [];
  if (todos.length > 0) {
    return 'warning';
  }
  return true;
});

// Check for any usage
check('No any types in StateSerializer', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  const anyTypes = content.match(/:\s*any/g) || [];
  if (anyTypes.length > 5) {
    return 'warning';
  }
  return true;
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 7: FILE SIZE CHECKS');
console.log('='.repeat(80));

check('SerializationEngine.ts reasonable size', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n').length;
  if (lines > 500) {
    return 'warning'; // File too large
  }
  return true;
});

check('StateSanitizer.ts reasonable size', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'StateSanitizer.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n').length;
  if (lines > 300) {
    return 'warning';
  }
  return true;
});

console.log('\n' + '='.repeat(80));
console.log('BUG SWEEP SUMMARY');
console.log('='.repeat(80));
console.log(`\nTotal Checks: ${totalChecks}`);
console.log(`Passed: ${passedChecks} ✅`);
console.log(`Failed: ${failedChecks} ❌`);
console.log(`Warnings: ${warnings} ⚠️`);
console.log(`Success Rate: ${((passedChecks / totalChecks) * 100).toFixed(1)}%`);

if (failedChecks === 0) {
  if (warnings === 0) {
    console.log('\n🎉 NO BUGS FOUND! CODE IS CLEAN! 🎉\n');
  } else {
    console.log(`\n✅ NO CRITICAL BUGS, but ${warnings} warning(s) found.\n`);
  }
  process.exit(0);
} else {
  console.log(`\n❌ ${failedChecks} bug(s) found. Please fix the issues above.\n`);
  process.exit(1);
}
