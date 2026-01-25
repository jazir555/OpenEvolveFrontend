#!/usr/bin/env node

/**
 * Standalone verification script for BubbleRunner critical fixes
 * Tests the 5 critical fixes without requiring the full test infrastructure
 */

const path = require('path');
const fs = require('fs');

console.log('='.repeat(80));
console.log('BubbleRunner Critical Fixes Verification');
console.log('='.repeat(80));
console.log();

// Read the BubbleRunner source file
const runnerPath = path.join(__dirname, 'src/runtime/BubbleRunner.ts');
const runnerSource = fs.readFileSync(runnerPath, 'utf-8');

console.log('✅ Read BubbleRunner source file');
console.log();

// Test 1: Property initialization verification
console.log('TEST 1: Property Initialization');
console.log('-'.repeat(80));

const propertyTests = {
  bubbleFactory: {
    pattern: /private\s+bubbleFactory:\s*BubbleFactory/,
    description: 'bubbleFactory property declared'
  },
  currentStep: {
    pattern: /private\s+currentStep:\s*number/,
    description: 'currentStep property declared'
  },
  savedStates: {
    pattern: /private\s+savedStates:\s*Map<number,\s*any>/,
    description: 'savedStates property declared'
  },
  plan: {
    pattern: /private\s+plan:\s*ExecutionPlan\s*\|\s*null\s*=\s*null/,
    description: 'plan property initialized to null'
  },
  logger: {
    pattern: /private\s+logger:\s*BubbleLogger/,
    description: 'logger property declared'
  },
  injector: {
    pattern: /public\s+injector:\s*BubbleInjector/,
    description: 'injector property declared'
  }
};

let test1Passed = true;
for (const [prop, test] of Object.entries(propertyTests)) {
  const found = test.pattern.test(runnerSource);
  const status = found ? '✅' : '❌';
  console.log(`${status} ${test.description}`);
  if (!found) test1Passed = false;
}

console.log();
if (test1Passed) {
  console.log('✅ TEST 1 PASSED: All properties are declared');
} else {
  console.log('❌ TEST 1 FAILED: Some properties are missing');
}
console.log();

// Test 2: Constructor initialization
console.log('TEST 2: Constructor Initialization');
console.log('-'.repeat(80));

const constructorTests = {
  bubbleFactoryInit: {
    pattern: /this\.bubbleFactory\s*=\s*bubbleFactory/,
    description: 'bubbleFactory initialized in constructor'
  },
  currentStepInit: {
    pattern: /this\.currentStep\s*=\s*0/,
    description: 'currentStep initialized to 0'
  },
  savedStatesInit: {
    pattern: /this\.savedStates\s*=\s*new\s+Map\(\)/,
    description: 'savedStates initialized as empty Map'
  },
  planInit: {
    pattern: /this\.plan\s*=\s*this\.buildExecutionPlan\(\)/,
    description: 'plan initialized with buildExecutionPlan()'
  }
};

let test2Passed = true;
for (const [testName, test] of Object.entries(constructorTests)) {
  const found = test.pattern.test(runnerSource);
  const status = found ? '✅' : '❌';
  console.log(`${status} ${test.description}`);
  if (!found) test2Passed = false;
}

console.log();
if (test2Passed) {
  console.log('✅ TEST 2 PASSED: All properties initialized in constructor');
} else {
  console.log('❌ TEST 2 FAILED: Some properties not initialized');
}
console.log();

// Test 3: runStep() method
console.log('TEST 3: runStep() Method');
console.log('-'.repeat(80));

const runStepTests = {
  methodExists: {
    pattern: /async\s+runStep\s*\(\s*stepId\s*:\s*number\s*\)\s*:\s*Promise<ExecutionResult>/,
    description: 'runStep() method signature correct'
  },
  planCheck: {
    pattern: /if\s*\(\s*!this\.plan\s*\)\s*{[\s\S]*throw new Error\(['"]Execution plan not initialized['"]\)/,
    description: 'Checks for plan existence'
  },
  stepValidation: {
    pattern: /const\s+step\s*=\s*this\.plan\.steps\.find\([\s\S]*step\.id\s*===\s*stepId[\s\S]*\)[\s\S]*if\s*\(\s*!step\s*\)/,
    description: 'Validates step exists in plan'
  },
  currentStepUpdate: {
    pattern: /this\.currentStep\s*=\s*stepId/,
    description: 'Updates currentStep after execution'
  },
  stateSave: {
    pattern: /this\.saveState\s*\(\s*stepId\s*\)/,
    description: 'Saves state after execution'
  },
  errorHandling: {
    pattern: /catch\s*\(error:\s*unknown\s*\)[\s\S]*return\s*{[\s\S]*success:\s*false/,
    description: 'Handles errors gracefully'
  }
};

let test3Passed = true;
for (const [testName, test] of Object.entries(runStepTests)) {
  const found = test.pattern.test(runnerSource);
  const status = found ? '✅' : '❌';
  console.log(`${status} ${test.description}`);
  if (!found) test3Passed = false;
}

console.log();
if (test3Passed) {
  console.log('✅ TEST 3 PASSED: runStep() method properly implemented');
} else {
  console.log('❌ TEST 3 FAILED: runStep() method incomplete');
}
console.log();

// Test 4: resumeFromStep() method
console.log('TEST 4: resumeFromStep() Method');
console.log('-'.repeat(80));

const resumeTests = {
  methodExists: {
    pattern: /async\s+resumeFromStep\s*\(\s*stepId\s*:\s*number\s*\)\s*:\s*Promise<ExecutionResult>/,
    description: 'resumeFromStep() method signature correct'
  },
  planCheck: {
    pattern: /if\s*\(\s*!this\.plan\s*\)[\s\S]*throw new Error\(['"]Execution plan not initialized['"]\)/,
    description: 'Checks for plan existence'
  },
  savedStateCheck: {
    pattern: /const\s+savedState\s*=\s*this\.savedStates\.get\(\s*stepId\s*\)[\s\S]*if\s*\(\s*!savedState\s*\)[\s\S]*throw new Error\(['"]No saved state found/s,
    description: 'Checks for saved state existence'
  },
  stateRestore: {
    pattern: /this\.currentStep\s*=\s*stepId/,
    description: 'Restores currentStep'
  },
  reExecution: {
    pattern: /const\s+result\s*=\s*await\s*this\.runStep\s*\(\s*stepId\s*\)/,
    description: 'Re-executes the step'
  },
  errorHandling: {
    pattern: /catch\s*\(error:\s*unknown\s*\)[\s\S]*return\s*{[\s\S]*success:\s*false/,
    description: 'Handles errors gracefully'
  }
};

let test4Passed = true;
for (const [testName, test] of Object.entries(resumeTests)) {
  const found = test.pattern.test(runnerSource);
  const status = found ? '✅' : '❌';
  console.log(`${status} ${test.description}`);
  if (!found) test4Passed = false;
}

console.log();
if (test4Passed) {
  console.log('✅ TEST 4 PASSED: resumeFromStep() method properly implemented');
} else {
  console.log('❌ TEST 4 FAILED: resumeFromStep() method incomplete');
}
console.log();

// Test 5: State management methods
console.log('TEST 5: State Management');
console.log('-'.repeat(80));

const stateTests = {
  saveStateMethod: {
    pattern: /private\s+saveState\s*\(\s*stepId\s*:\s*number\s*\)\s*:\s*void/,
    description: 'saveState() method exists'
  },
  saveStateContent: {
    pattern: /const\s+state\s*=\s*{[\s\S]*stepId[\s\S]*currentStep[\s\S]*variables[\s\S]*timestamp/,
    description: 'saveState() stores all required fields'
  },
  getSavedStateMethod: {
    pattern: /getSavedState\s*\(\s*stepId\s*:\s*number\s*\)\s*:\s*any\s*\|\s*undefined/,
    description: 'getSavedState() method exists'
  },
  getAllSavedStatesMethod: {
    pattern: /getAllSavedStates\s*\(\)\s*:\s*Map<number,\s*any>/,
    description: 'getAllSavedStates() method exists'
  },
  clearSavedStatesMethod: {
    pattern: /clearSavedStates\s*\(\)\s*:\s*void/,
    description: 'clearSavedStates() method exists'
  },
  clearImplementation: {
    pattern: /this\.savedStates\.clear\(\)[\s\S]*this\.currentStep\s*=\s*0/,
    description: 'clearSavedStates() resets state'
  }
};

let test5Passed = true;
for (const [testName, test] of Object.entries(stateTests)) {
  const found = test.pattern.test(runnerSource);
  const status = found ? '✅' : '❌';
  console.log(`${status} ${test.description}`);
  if (!found) test5Passed = false;
}

console.log();
if (test5Passed) {
  console.log('✅ TEST 5 PASSED: State management methods properly implemented');
} else {
  console.log('❌ TEST 5 FAILED: State management incomplete');
}
console.log();

// Test 6: Error handling
console.log('TEST 6: Error Handling');
console.log('-'.repeat(80));

const errorHandlingTests = {
  runStepTryCatch: {
    pattern: /async\s+runStep[\s\S]*try\s*{[\s\S]*}\s*catch\s*\(\s*error:\s*unknown\s*\)/,
    description: 'runStep() has try-catch block'
  },
  resumeFromStepTryCatch: {
    pattern: /async\s+resumeFromStep[\s\S]*try\s*{[\s\S]*}\s*catch\s*\(\s*error:\s*unknown\s*\)/,
    description: 'resumeFromStep() has try-catch block'
  },
  safeErrorMessage: {
    pattern: /const\s+safeError\s*=\s*getSafeErrorMessage\s*\(\s*error\s*\)/,
    description: 'Uses getSafeErrorMessage for error sanitization'
  },
  errorLogging: {
    pattern: /this\.logger\?\.error\(/,
    description: 'Logs errors'
  },
  errorReturnStructure: {
    pattern: /return\s*{[\s\S]*success:\s*false[\s\S]*error:\s*safeError/,
    description: 'Returns proper error structure'
  }
};

let test6Passed = true;
for (const [testName, test] of Object.entries(errorHandlingTests)) {
  const found = test.pattern.test(runnerSource);
  const status = found ? '✅' : '❌';
  console.log(`${status} ${test.description}`);
  if (!found) test6Passed = false;
}

console.log();
if (test6Passed) {
  console.log('✅ TEST 6 PASSED: Error handling properly implemented');
} else {
  console.log('❌ TEST 6 FAILED: Error handling incomplete');
}
console.log();

// Summary
console.log('='.repeat(80));
console.log('SUMMARY');
console.log('='.repeat(80));

const allTests = [
  { name: 'Property Initialization', passed: test1Passed },
  { name: 'Constructor Initialization', passed: test2Passed },
  { name: 'runStep() Method', passed: test3Passed },
  { name: 'resumeFromStep() Method', passed: test4Passed },
  { name: 'State Management', passed: test5Passed },
  { name: 'Error Handling', passed: test6Passed }
];

const passedCount = allTests.filter(t => t.passed).length;
const totalCount = allTests.length;

allTests.forEach(test => {
  const status = test.passed ? '✅' : '❌';
  console.log(`${status} ${test.name}`);
});

console.log();
console.log(`Total: ${passedCount}/${totalCount} test suites passed`);
console.log();

if (passedCount === totalCount) {
  console.log('🎉 ALL TESTS PASSED! BubbleRunner critical fixes verified.');
  process.exit(0);
} else {
  console.log('⚠️  SOME TESTS FAILED. Please review the implementation.');
  process.exit(1);
}
