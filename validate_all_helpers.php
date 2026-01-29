<?php
/**
 * Comprehensive Helper File Validator
 * Validates all 160 helper files across all 12 helper directories
 */

$directories = [
    'AjaxHelpers',
    'AssetDataHelpers',
    'AssetOrderHelpers',
    'CleanupHelpers',
    'DatabaseHelpers',
    'ExtractHelpers',
    'LoggingHelpers',
    'ProcessHelpers',
    'RetryHelpers',
    'SanitizeHelpers',
    'SettingsHelpers',
    'TaskHelpers',
];

$total_files = 0;
$valid_files = 0;
$invalid_files = [];
$errors_by_type = [];

echo "================================================================\n";
echo "COMPREHENSIVE HELPER FILE VALIDATION\n";
echo "================================================================\n\n";

foreach ($directories as $dir) {
    $files = [];
    $iterator = new RecursiveIteratorIterator(
        new RecursiveDirectoryIterator(__DIR__ . '/' . $dir)
    );

    foreach ($iterator as $file) {
        if ($file->isFile() && $file->getExtension() === 'php') {
            $files[] = $file->getPathname();
        }
    }

    $dir_valid = 0;
    $dir_invalid = 0;

    echo "--------------------------------------------------------------\n";
    echo "Directory: $dir (" . count($files) . " files)\n";
    echo "--------------------------------------------------------------\n";

    foreach ($files as $file) {
        $total_files++;
        $filename = basename($file);
        $output = shell_exec("php -l " . escapeshellarg($file) . " 2>&1");

        if (strpos($output, 'Errors parsing') !== false || strpos($output, 'Parse error') !== false) {
            $invalid_files[] = $file;
            $dir_invalid++;

            // Extract error type
            if (preg_match('/PHP (?:Fatal )?[Pp]arse error:\s*(.+)/', $output, $matches)) {
                $error_type = trim($matches[1]);
                if (!isset($errors_by_type[$error_type])) {
                    $errors_by_type[$error_type] = [];
                }
                $errors_by_type[$error_type][] = "$dir/$filename";
            }

            echo "  ✗ $filename\n";
            echo "    $output\n";
        } else {
            $valid_files++;
            $dir_valid++;
            echo "  ✓ $filename\n";
        }
    }

    echo "\n";
    echo "  Valid: $dir_valid | Invalid: $dir_invalid\n\n";
}

echo "================================================================\n";
echo "SUMMARY\n";
echo "================================================================\n";
echo "Total Files:     $total_files\n";
echo "Valid Files:     $valid_files\n";
echo "Invalid Files:   " . count($invalid_files) . "\n";
echo "Success Rate:    " . round(($valid_files / $total_files) * 100, 2) . "%\n";

if (!empty($invalid_files)) {
    echo "\n================================================================\n";
    echo "INVALID FILES\n";
    echo "================================================================\n";
    foreach ($invalid_files as $file) {
        echo "  - $file\n";
    }
}

if (!empty($errors_by_type)) {
    echo "\n================================================================\n";
    echo "ERRORS BY TYPE\n";
    echo "================================================================\n";
    foreach ($errors_by_type as $error => $files) {
        echo "\n$error (" . count($files) . " files):\n";
        foreach ($files as $file) {
            echo "  - $file\n";
        }
    }
}

echo "\n================================================================\n";
echo count($invalid_files) > 0 ? "❌ VALIDATION FAILED\n" : "✅ ALL FILES VALID\n";
echo "================================================================\n";
