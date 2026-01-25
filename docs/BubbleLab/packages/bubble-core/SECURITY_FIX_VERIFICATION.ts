#!/usr/bin/env ts-node

/**
 * SECURITY FIX VERIFICATION SCRIPT
 *
 * This script performs automated verification of the security fixes
 * implemented in backup-restore-workflow.ts
 *
 * Run: npx ts-node SECURITY_FIX_VERIFICATION.ts
 */

import { BackupRestoreWorkflow } from './src/bubbles/workflow-bubble/backup-restore-workflow';

console.log('='.repeat(80));
console.log('SECURITY FIX VERIFICATION - backup-restore-workflow.ts');
console.log('='.repeat(80));

const workflow = new BackupRestoreWorkflow();
let passed = 0;
let failed = 0;

async function test(name: string, testFn: () => Promise<void>) {
  try {
    await testFn();
    console.log(`✅ PASS: ${name}`);
    passed++;
  } catch (error: any) {
    console.log(`❌ FAIL: ${name}`);
    console.log(`   Error: ${error.message}`);
    failed++;
  }
}

console.log('\n🔒 TESTING: Command Injection Prevention\n');

await test('Blocks hostname with semicolon injection', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'localhost; rm -rf /; #',
      port: 5432,
      username: 'validuser',
      database: 'validdb'
    }
  });
  if (result.success) throw new Error('Should have rejected malicious hostname');
});

await test('Blocks hostname with pipe injection', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'localhost | evil',
      port: 5432,
      username: 'validuser',
      database: 'validdb'
    }
  });
  if (result.success) throw new Error('Should have rejected pipe injection');
});

await test('Blocks hostname with command substitution', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'localhost$(whoami)',
      port: 5432,
      username: 'validuser',
      database: 'validdb'
    }
  });
  if (result.success) throw new Error('Should have rejected command substitution');
});

await test('Blocks username with command injection', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'localhost',
      port: 5432,
      username: 'user; DROP TABLE users; --',
      database: 'validdb'
    }
  });
  if (result.success) throw new Error('Should have rejected malicious username');
});

await test('Blocks database name with injection', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'localhost',
      port: 5432,
      username: 'validuser',
      database: 'mydb && evil'
    }
  });
  if (result.success) throw new Error('Should have rejected malicious database name');
});

await test('Blocks port with injection', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'localhost',
      port: '5432; evil' as any,
      username: 'validuser',
      database: 'validdb'
    }
  });
  if (result.success) throw new Error('Should have rejected malicious port');
});

await test('Accepts valid database configuration', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'postgresql',
      host: 'db.example.com',
      port: 5432,
      username: 'dbuser',
      database: 'production_db'
    }
  });
  if (!result.success) throw new Error('Should have accepted valid config');
});

console.log('\n🔒 TESTING: Path Traversal Prevention\n');

await test('Blocks localPath with double-dot traversal', async () => {
  try {
    await workflow['saveToLocal']({
      localPath: '../../../etc',
      backup: { type: 'database', path: 'backup.sql' }
    }, 'backup.sql');
    throw new Error('Should have rejected path traversal');
  } catch (error: any) {
    if (!error.message.includes('Path traversal')) {
      throw new Error('Wrong error: ' + error.message);
    }
  }
});

await test('Blocks localPath with null bytes', async () => {
  try {
    await workflow['saveToLocal']({
      localPath: '/tmp/backups\0malicious',
      backup: { type: 'database', path: 'backup.sql' }
    }, 'backup.sql');
    throw new Error('Should have rejected null bytes');
  } catch (error: any) {
    if (!error.message.includes('null bytes')) {
      throw new Error('Wrong error: ' + error.message);
    }
  }
});

await test('Blocks relative paths', async () => {
  try {
    await workflow['saveToLocal']({
      localPath: 'relative/path',
      backup: { type: 'database', path: 'backup.sql' }
    }, 'backup.sql');
    throw new Error('Should have rejected relative path');
  } catch (error: any) {
    if (!error.message.includes('absolute')) {
      throw new Error('Wrong error: ' + error.message);
    }
  }
});

await test('Blocks filePath with traversal characters', async () => {
  try {
    await workflow['saveToLocal']({
      localPath: '/tmp/backups',
      backup: { type: 'database', path: 'backup.sql' }
    }, '../../../etc/passwd');
    throw new Error('Should have rejected traversal in filename');
  } catch (error: any) {
    if (!error.message.includes('path traversal')) {
      throw new Error('Wrong error: ' + error.message);
    }
  }
});

await test('Blocks filePath exceeding maximum length', async () => {
  try {
    await workflow['saveToLocal']({
      localPath: '/tmp/backups',
      backup: { type: 'database', path: 'backup.sql' }
    }, 'a'.repeat(256));
    throw new Error('Should have rejected long filename');
  } catch (error: any) {
    if (!error.message.includes('exceeds maximum length')) {
      throw new Error('Wrong error: ' + error.message);
    }
  }
});

await test('Accepts valid local path', async () => {
  const result = await workflow['saveToLocal']({
    localPath: '/tmp/backups',
    backup: { type: 'database', path: 'backup.sql' }
  }, 'backup.sql');
  if (result.provider !== 'local') throw new Error('Should have accepted valid path');
});

console.log('\n🔒 TESTING: DoS Prevention\n');

await test('Blocks backup size exceeding maximum', async () => {
  const maxSize = 1024 * 1024 * 1024 * 1024; // 1TB
  const result = await workflow.validateSource({
    source: '/path/to/file',
    sourceSize: maxSize + 1
  });
  if (result.success) throw new Error('Should have rejected oversized backup');
});

await test('Accepts backup size within limits', async () => {
  const result = await workflow.validateSource({
    source: '/path/to/file',
    sourceSize: 1024 * 1024 * 100 // 100MB
  });
  if (!result.success) throw new Error('Should have accepted valid size');
});

console.log('\n🔒 TESTING: SQLite Path Validation\n');

await test('Blocks SQLite path with traversal', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'sqlite',
      path: '../../../etc/passwd'
    }
  });
  if (result.success) throw new Error('Should have rejected traversal');
});

await test('Blocks absolute SQLite paths', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'sqlite',
      path: '/etc/passwd'
    }
  });
  if (result.success) throw new Error('Should have rejected absolute path');
});

await test('Accepts valid relative SQLite path', async () => {
  const result = await workflow.validateSource({
    database: {
      type: 'sqlite',
      path: 'data/database.db'
    }
  });
  if (!result.success) throw new Error('Should have accepted valid path');
});

console.log('\n🔒 TESTING: Credential Sanitization\n');

await test('Sanitizes MySQL password in command output', async () => {
  const result = await workflow.createBackup({
    database: {
      type: 'mysql',
      host: 'localhost',
      port: 3306,
      username: 'root',
      password: 'supersecret123',
      database: 'production'
    },
    sourceSize: 1024
  });
  if (result.success && result.backup) {
    if (!result.backup.command.includes('-p****')) {
      throw new Error('Password not sanitized in command');
    }
    if (result.backup.command.includes('supersecret123')) {
      throw new Error('Password exposed in command');
    }
  } else {
    throw new Error('Backup creation failed');
  }
});

console.log('\n' + '='.repeat(80));
console.log('VERIFICATION COMPLETE');
console.log('='.repeat(80));
console.log(`✅ Passed: ${passed}`);
console.log(`❌ Failed: ${failed}`);
console.log(`📊 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);
console.log('='.repeat(80));

if (failed > 0) {
  console.log('\n⚠️  SECURITY FIXES NEED ATTENTION - SOME TESTS FAILED');
  process.exit(1);
} else {
  console.log('\n✅ ALL SECURITY FIXES VERIFIED - SYSTEM IS SECURE');
  process.exit(0);
}
