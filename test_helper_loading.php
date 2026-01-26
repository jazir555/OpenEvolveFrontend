<?php
declare(strict_types=1);

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

echo "Testing helper file loading...\n\n";
$success = 0;
$failed = 0;
$failed_files = [];

foreach ($directories as $dir) {
    $files = glob(__DIR__ . '/' . $dir . '/*.php');
    foreach ($files as $file) {
        $filename = basename($file);
        // Try to load the file
        $output = [];
        $return_var = 0;
        exec('php -l ' . escapeshellarg($file) . ' 2>&1', $output, $return_var);
        
        if ($return_var === 0) {
            $success++;
        } else {
            $failed++;
            $failed_files[] = $file;
        }
    }
}

echo "Success: $success\n";
echo "Failed: $failed\n";

if (!empty($failed_files)) {
    echo "\nFailed files:\n";
    foreach ($failed_files as $file) {
        echo "  - $file\n";
    }
}
