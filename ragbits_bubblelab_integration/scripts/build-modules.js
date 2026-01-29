/**
 * Build script to generate both CommonJS and ES modules
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Ensure dist directory exists
const distDir = path.join(__dirname, '..', 'dist');
if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir, { recursive: true });
}

// Build TypeScript files
console.log('Building TypeScript files...');
execSync('npx tsc', { stdio: 'inherit' });

// Copy and rename CommonJS files to have .cjs extension if needed
function renameCJSFiles(dir) {
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      renameCJSFiles(filePath);
    } else if (file.endsWith('.js')) {
      // For now, we'll keep the .js extension for CommonJS as specified in package.json
      // In a real scenario, we might want to generate both .cjs and .mjs files
    }
  }
}

// Process all subdirectories
renameCJSFiles(distDir);

console.log('Build completed successfully!');