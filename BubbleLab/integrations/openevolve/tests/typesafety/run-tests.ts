/**
 * Test Runner for Type Safety Tests
 * Runs all type safety tests and generates coverage report
 */

import { execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';

const testDir = import.meta.dir;
const rootDir = join(testDir, '..', '..', '..', '..');

console.log('🧪 Running Type Safety Tests...\n');
console.log(`Test Directory: ${testDir}`);
console.log(`Root Directory: ${rootDir}\n`);

// Run tests using Jest
try {
  console.log('Running tests with Jest...\n');

  const testCommand = `cd "${rootDir}" && npx jest "${testDir}" --config="${join(testDir, 'jest.config.js')}" --verbose`;

  console.log(`Command: ${testCommand}\n`);

  execSync(testCommand, {
    stdio: 'inherit',
    cwd: rootDir,
  });

  console.log('\n✅ All tests passed!\n');
} catch (error) {
  console.error('\n❌ Tests failed!\n');
  process.exit(1);
}
