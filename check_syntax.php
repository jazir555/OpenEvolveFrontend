#!/usr/bin/env php
<?php
/**
 * PHP Syntax Error Checker
 * Scans all PHP files in the codebase for syntax errors
 */

$rootDir = __DIR__;
$filesWithErrors = [];
$totalFiles = 0;

/**
 * Recursively find all PHP files
 */
function findPhpFiles($dir) {
    $files = [];
    $iterator = new RecursiveIteratorIterator(
        new RecursiveDirectoryIterator($dir, RecursiveDirectoryIterator::SKIP_DOTS)
    );

    foreach ($iterator as $file) {
        if ($file->isFile() && $file->getExtension() === 'php') {
            $files[] = $file->getPathname();
        }
    }

    return $files;
}

// Get all PHP files
$phpFiles = findPhpFiles($rootDir);
$totalFiles = count($phpFiles);

echo "Checking $totalFiles PHP files for syntax errors...\n\n";

// Check each file
foreach ($phpFiles as $file) {
    $relativePath = str_replace($rootDir . '/', '', $file);
    $output = [];
    $returnCode = 0;

    exec('php -l ' . escapeshellarg($file) . ' 2>&1', $output, $returnCode);

    if ($returnCode !== 0) {
        $filesWithErrors[$relativePath] = implode("\n", $output);
        echo "❌ ERROR: $relativePath\n";
        echo "   " . implode("\n   ", $output) . "\n\n";
    }
}

// Summary
echo "\n" . str_repeat("=", 80) . "\n";
echo "SYNTAX CHECK SUMMARY\n";
echo str_repeat("=", 80) . "\n";
echo "Total files checked: $totalFiles\n";
echo "Files with errors: " . count($filesWithErrors) . "\n";

if (count($filesWithErrors) > 0) {
    echo "\nFILES WITH SYNTAX ERRORS:\n";
    echo str_repeat("-", 80) . "\n";
    foreach ($filesWithErrors as $file => $error) {
        echo "\n📁 $file\n";
        echo "   $error\n";
    }
    exit(1);
} else {
    echo "\n✅ All PHP files are syntactically correct!\n";
    exit(0);
}
