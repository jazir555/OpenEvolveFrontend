<?php

declare(strict_types=1);

/**
 * Fix missing return type declarations in helper files
 */

$fixes = [
    'AssetOrderHelpers/AssetOrderStaticHelper.php' => [
        786 => 'private function execute_timed_query',
    ],
    'CleanupHelpers/CleanupFileOperator.php' => [
        453 => 'private function get_plugin_upload_dir_info',
    ],
    'CleanupHelpers/CleanupHelper.php' => [
        133 => 'private function get_plugin_upload_dir_info',
    ],
    'RetryHelpers/RetryOperationHelper.php' => [
        75 => 'public function add_to_retry_queue',
        239 => 'public function retry_failed_job',
    ],
    'RetryHelpers/RetryOperationHelperRefactored.php' => [
        37 => 'public function enqueue_retry',
        166 => 'public function retry_failed_job',
    ],
    'RetryHelpers/RetryQueue.php' => [
        37 => 'public function store_retry_job',
    ],
    'RetryHelpers/RetryScheduleHelper.php' => [
        84 => 'public function cancel_job',
    ],
    'RetryHelpers/RetryUtilityHelper.php' => [
        241 => 'public function store_retry_job',
    ],
    'LoggingHelpers/LoggingConfig.php' => [
        73 => 'public function get_cached_option',
        189 => 'public function prepare_email_headers',
    ],
    'LoggingHelpers/LoggingErrorHandler.php' => [
        464 => 'private function prepare_email_headers',
    ],
    'LoggingHelpers/LoggingNotifier.php' => [
        248 => 'private function prepare_email_headers',
    ],
    'SanitizeHelpers/SanitizeInputHelper.php' => [
        812 => 'private function sanitize_content_dispatcher',
    ],
    'SanitizeHelpers/SanitizeSvgHelper.php' => [
        144 => 'protected function extract_svg_dimensions',
    ],
    'SanitizeHelpers/SanitizeValidationHelper.php' => [
        284 => 'public function is_valid_password',
    ],
];

// Map function signatures to return types based on analysis
$return_types = [
    'execute_timed_query' => 'array', // Returns associative array
    'get_plugin_upload_dir_info' => 'array|false', // Returns array or false
    'add_to_retry_queue' => 'int|false', // Returns job ID or false
    'retry_failed_job' => 'bool', // Returns success boolean
    'enqueue_retry' => 'int|false', // Returns job ID or false
    'store_retry_job' => 'int|false', // Returns job ID or false
    'cancel_job' => 'bool', // Returns success boolean
    'get_cached_option' => 'mixed', // Returns option value
    'prepare_email_headers' => 'array', // Returns headers array
    'sanitize_content_dispatcher' => 'string', // Returns sanitized string
    'extract_svg_dimensions' => 'array|false', // Returns dimensions or false
    'is_valid_password' => 'bool', // Returns boolean
];

$base_dir = __DIR__;
$fixed_count = 0;
$failed = [];

foreach ($fixes as $file => $methods) {
    $filepath = $base_dir . '/' . $file;
    if (!file_exists($filepath)) {
        echo "WARNING: File not found: {$file}\n";
        continue;
    }

    $content = file_get_contents($filepath);
    $lines = explode("\n", $content);
    $modified = false;

    foreach ($methods as $line_num => $signature) {
        $line_index = $line_num - 1;
        if (!isset($lines[$line_index])) {
            echo "WARNING: Line {$line_num} not found in {$file}\n";
            continue;
        }

        $line = trim($lines[$line_index]);

        // Check if already has return type
        if (preg_match('/\)\s*:\s*\S/', $line)) {
            echo "SKIP: Line {$line_num} in {$file} already has return type\n";
            continue;
        }

        // Find method name in signature
        $method_name = null;
        foreach ($return_types as $method => $return_type) {
            if (strpos($signature, $method) !== false) {
                $method_name = $method;
                break;
            }
        }

        if (!$method_name || !isset($return_types[$method_name])) {
            echo "WARNING: Unknown return type for: {$signature}\n";
            continue;
        }

        $return_type = $return_types[$method_name];

        // Add return type
        $new_line = rtrim($lines[$line_index]) . ': ' . $return_type;
        $lines[$line_index] = $new_line;
        $modified = true;
        $fixed_count++;
        echo "FIXED: Line {$line_num} in {$file} - Added return type: {$return_type}\n";
    }

    if ($modified) {
        file_put_contents($filepath, implode("\n", $lines));
        // Validate syntax
        $output = [];
        $return_code = 0;
        exec('php -l ' . escapeshellarg($filepath) . ' 2>&1', $output, $return_code);
        if ($return_code !== 0) {
            echo "ERROR: Syntax error in {$file} after fix!\n";
            echo implode("\n", $output) . "\n";
            $failed[] = $file;
            // Revert changes
            file_put_contents($filepath, $content);
        } else {
            echo "✓ Validated: {$file}\n";
        }
    }
}

echo "\n\n=== SUMMARY ===\n";
echo "Fixed {$fixed_count} methods\n";
if (!empty($failed)) {
    echo "Failed files: " . implode(', ', $failed) . "\n";
} else {
    echo "All fixes applied successfully!\n";
}
