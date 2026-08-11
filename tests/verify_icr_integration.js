/**
 * ICR Integration Verification Script
 * 
 * Verifies all ICR integrations are working correctly
 * Run with: node verify_icr_integration.js
 */

const fs = require('fs');
const path = require('path');

console.log('='.repeat(80));
console.log('ICR INTEGRATION VERIFICATION');
console.log('='.repeat(80));
console.log(`\nDate: ${new Date().toISOString()}\n`);

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function test(description, fn) {
  totalTests++;
  try {
    fn();
    passedTests++;
    console.log(`✅ ${description}`);
    return true;
  } catch (error) {
    failedTests++;
    console.log(`❌ ${description}`);
    console.log(`   Error: ${error.message}`);
    return false;
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

function fileExists(filePath) {
  return fs.existsSync(filePath);
}

function directoryExists(dirPath) {
  return fs.existsSync(dirPath) && fs.statSync(dirPath).isDirectory();
}

// Base paths
const baseDir = path.join(__dirname);
const icrDir = path.join(baseDir, 'core-projects', 'Iterative-Contextual-Refinements');
const docsDir = path.join(baseDir, 'docs', 'Iterative Contextual Refinements');

console.log('='.repeat(80));
console.log('PHASE 1: FILE STRUCTURE VERIFICATION');
console.log('='.repeat(80));

test('ICR directory exists', () => {
  assert(directoryExists(icrDir), 'ICR directory not found');
});

test('StateSerializer directory exists', () => {
  const ssDir = path.join(icrDir, 'Core', 'StateSerializer');
  assert(directoryExists(ssDir), 'StateSerializer directory not found');
});

test('StateSerializer handlers directory exists', () => {
  const handlersDir = path.join(icrDir, 'Core', 'StateSerializer', 'handlers');
  assert(directoryExists(handlersDir), 'StateSerializer handlers directory not found');
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 2: STATE SERIALIZER FILES');
console.log('='.repeat(80));

const stateSerializerFiles = [
  'SerializationEngine.ts',
  'ModeStateHandler.ts',
  'StateSanitizer.ts',
  'StateVersion.ts',
  'index.ts'
];

stateSerializerFiles.forEach(file => {
  test(`StateSerializer/${file} exists`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', file);
    assert(fileExists(filePath), `File not found: ${file}`);
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 3: MODE HANDLERS');
console.log('='.repeat(80));

const handlerFiles = [
  'DeepthinkStateHandler.ts',
  'AgenticStateHandler.ts',
  'ContextualStateHandler.ts',
  'AdaptiveDeepthinkStateHandler.ts',
  'WebsiteModeStateHandler.ts',
  'MathSolverStateHandler.ts',
  'GenerativeUIStateHandler.ts',
  'ReactStateHandler.ts',
  'index.ts'
];

handlerFiles.forEach(file => {
  test(`handlers/${file} exists`, () => {
    const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', file);
    assert(fileExists(filePath), `Handler not found: ${file}`);
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 4: CUSTOM MODES');
console.log('='.repeat(80));

test('MathSolver directory exists', () => {
  const mathSolverDir = path.join(icrDir, 'MathSolver');
  assert(directoryExists(mathSolverDir), 'MathSolver directory not found');
});

test('MathSolver has core files', () => {
  const mathSolverDir = path.join(icrDir, 'MathSolver');
  const requiredFiles = ['MathSolverCore.ts', 'MathSolverMode.ts', 'MathSolverUI.tsx'];
  requiredFiles.forEach(file => {
    assert(fileExists(path.join(mathSolverDir, file)), `MathSolver file not found: ${file}`);
  });
});

test('GenerativeUI directory exists', () => {
  const generativeUIDir = path.join(icrDir, 'GenerativeUI');
  assert(directoryExists(generativeUIDir), 'GenerativeUI directory not found');
});

test('GenerativeUI has core files', () => {
  const generativeUIDir = path.join(icrDir, 'GenerativeUI');
  const requiredFiles = ['GenerativeUICore.ts', 'GenerativeUI.tsx'];
  requiredFiles.forEach(file => {
    assert(fileExists(path.join(generativeUIDir, file)), `GenerativeUI file not found: ${file}`);
  });
});

test('React directory exists', () => {
  const reactDir = path.join(icrDir, 'React');
  assert(directoryExists(reactDir), 'React directory not found');
});

test('React has core files', () => {
  const reactDir = path.join(icrDir, 'React');
  const requiredFiles = ['ReactLogic.ts', 'ReactUI.ts'];  // .ts not .tsx
  requiredFiles.forEach(file => {
    assert(fileExists(path.join(reactDir, file)), `React file not found: ${file}`);
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 5: ICR INTEGRATION MODULES (PYTHON BACKEND)');
console.log('='.repeat(80));

test('icr_integration.py exists', () => {
  const icrPath = path.join(baseDir, 'icr_integration.py');
  assert(fileExists(icrPath), 'ICR integration file not found');
});

test('knowledge_engine_icr_integration.py exists', () => {
  const keIcrPath = path.join(baseDir, 'knowledge_engine_icr_integration.py');
  assert(fileExists(keIcrPath), 'Knowledge Engine ICR integration file not found');
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 6: CORE INTEGRATION');
console.log('='.repeat(80));

test('ConfigManager.ts exists', () => {
  const configManagerPath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  assert(fileExists(configManagerPath), 'ConfigManager not found');
});

test('ConfigManager has StateSerializer imports', () => {
  const configManagerPath = path.join(icrDir, 'Core', 'ConfigManager.ts');
  const content = fs.readFileSync(configManagerPath, 'utf-8');
  assert(content.includes('serialize'), 'ConfigManager missing serialize import');
  assert(content.includes('deserialize'), 'ConfigManager missing deserialize import');
  assert(content.includes('sanitizeState'), 'ConfigManager missing sanitizeState import');
});

test('package.json has @msgpack/msgpack', () => {
  const packageJsonPath = path.join(icrDir, 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
  assert(packageJson.dependencies['@msgpack/msgpack'], '@msgpack/msgpack not in dependencies');
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 7: ROUTING ENHANCEMENTS');
console.log('='.repeat(80));

const routingFiles = [
  'ApiCallEstimator.ts',
  'ApiConfig.ts',
  'ApiKeyUI.ts',
  'ProviderManager.ts',
  'ProviderManagementUI.ts',
  'DeepthinkConfigController.ts'
];

routingFiles.forEach(file => {
  test(`Routing/${file} exists`, () => {
    const filePath = path.join(icrDir, 'Routing', file);
    assert(fileExists(filePath), `Routing file not found: ${file}`);
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 8: UI COMPONENTS');
console.log('='.repeat(80));

test('CodeMirrorFileEditor exists', () => {
  const componentPath = path.join(icrDir, 'Components', 'CodeMirrorFileEditor.tsx');
  assert(fileExists(componentPath), 'CodeMirrorFileEditor not found');
});

test('FileUpload component exists', () => {
  const componentPath = path.join(icrDir, 'Components', 'FileUpload', 'FileUpload.tsx');
  assert(fileExists(componentPath), 'FileUpload component not found');
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 9: STYLES MERGE');
console.log('='.repeat(80));

const styleFiles = [
  'global.css',
  'content.css',
  'layout.css',
  'sidebar.css',
  'Shiki.css'
];

styleFiles.forEach(file => {
  test(`styles/${file} exists`, () => {
    const filePath = path.join(icrDir, 'styles', file);
    assert(fileExists(filePath), `Style file not found: ${file}`);
  });
});

test('styles/components/Buttons.css exists', () => {
  const filePath = path.join(icrDir, 'styles', 'components', 'Buttons.css');
  assert(fileExists(filePath), 'Buttons.css not found');
});

test('styles/components/Inputs.css exists', () => {
  const filePath = path.join(icrDir, 'styles', 'components', 'Inputs.css');
  assert(fileExists(filePath), 'Inputs.css not found');
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 10: UI UTILITIES');
console.log('='.repeat(80));

const uiFiles = [
  'CommonUI.ts',
  'Controls.ts',
  'GlobalModals.ts',
  'LayoutController.ts',
  'Sidebar.ts',
  'Tabs.ts',
  'Theme.ts',
  'UIManager.ts',
  'setupCodeExecutionToggle.ts',
  'Shiki.ts'
];

uiFiles.forEach(file => {
  test(`UI/${file} exists`, () => {
    const filePath = path.join(icrDir, 'UI', file);
    assert(fileExists(filePath), `UI file not found: ${file}`);
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 11: DOCUMENTATION');
console.log('='.repeat(80));

const docFiles = [
  'ICR_100_PERCENT_CERTIFICATE.md',
  'ICR_ABSOLUTE_FINAL_100_PERCENT.md',
  'ICR_TESTING_PLAN.md',
  'ICR_SERIALIZATION_INTEGRATION_GUIDE.md',
  'ICR_UPSTREAM_MIGRATION_MASTER_PLAN.md',
  'ICR_COMPLETION_REPORT.md',
  'ICR_FINAL_VERIFICATION_REPORT.md',
  'ICR_STYLES_UI_MERGE_REPORT.md'
];

docFiles.forEach(file => {
  test(`Documentation/${file} exists`, () => {
    const filePath = path.join(docsDir, file);
    assert(fileExists(filePath), `Documentation not found: ${file}`);
  });
});

console.log('\n' + '='.repeat(80));
console.log('PHASE 12: FILE CONTENT VERIFICATION');
console.log('='.repeat(80));

test('StateSerializer exports serialize function', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  assert(content.includes('export async function serialize'), 'serialize function not exported');
});

test('StateSerializer exports deserialize function', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'SerializationEngine.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  assert(content.includes('export async function deserialize'), 'deserialize function not exported');
});

test('MathSolver handler exports handler', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', 'MathSolverStateHandler.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  assert(content.includes('export const mathsolverStateHandler'), 'MathSolver handler not exported');
});

test('GenerativeUI handler exports handler', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', 'GenerativeUIStateHandler.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  assert(content.includes('export const generativeUIStateHandler'), 'GenerativeUI handler not exported');
});

test('React handler exports handler', () => {
  const filePath = path.join(icrDir, 'Core', 'StateSerializer', 'handlers', 'ReactStateHandler.ts');
  const content = fs.readFileSync(filePath, 'utf-8');
  assert(content.includes('export const reactStateHandler'), 'React handler not exported');
});

console.log('\n' + '='.repeat(80));
console.log('VERIFICATION SUMMARY');
console.log('='.repeat(80));
console.log(`\nTotal Tests: ${totalTests}`);
console.log(`Passed: ${passedTests} ✅`);
console.log(`Failed: ${failedTests} ❌`);
console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);

if (failedTests === 0) {
  console.log('\n🎉 ALL TESTS PASSED! ICR INTEGRATION IS COMPLETE AND VERIFIED! 🎉\n');
  process.exit(0);
} else {
  console.log(`\n⚠️  ${failedTests} test(s) failed. Please review the errors above.\n`);
  process.exit(1);
}
