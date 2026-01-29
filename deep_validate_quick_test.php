<?php
/**
 * Quick Test for Deep Validation Tool
 * Tests on a few sample files to demonstrate functionality
 */

declare(strict_types=1);

namespace LHA\Tools;

// prevent direct access
if (php_sapi_name() !== 'cli') {
    die("This script can only be run from CLI.\n");
}

// Include the main validator
require_once __DIR__ . '/deep_validate.php';

echo "Quick Test - Validating Sample Files\n";
echo str_repeat('=', 80) . "\n\n";

// Test on a few sample files
$sampleFiles = [
    __DIR__ . '/TaskHelpers/TaskValidationHelper.php',
    __DIR__ . '/DatabaseHelpers/DatabaseCacheHelper.php',
    __DIR__ . '/interfaces/TaskProcessorInterface.php',
];

$validator = new DeepValidator(__DIR__);

// Manually test a few files
foreach ($sampleFiles as $file) {
    if (file_exists($file)) {
        echo "Testing: " . basename($file) . "\n";
        echo str_repeat('-', 80) . "\n";

        $relativePath = str_replace(__DIR__ . '/', '', $file);
        $validator->validateFile($relativePath);

        echo "\n";
    } else {
        echo "File not found: {$file}\n\n";
    }
}

// Print summary report
$validator->printReport();
