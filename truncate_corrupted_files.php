<?php
/**
 * Truncate corrupted helper files at last valid method
 */

$fixes = [
    'RetryHelpers/RetryQueryHelper.php' => [
        'last_valid_line' => 348, // End of get_pending_retries method
        'add_braces' => 2
    ],
    'RetryHelpers/RetryOperationHelper.php' => [
        'last_valid_line' => 152, // End of cleanup_old_retries method
        'add_braces' => 1
    ],
    'RetryHelpers/RetryScheduleHelper.php' => [
        'last_valid_line' => 80, // Before SQL corruption starts
        'add_braces' => 1
    ],
];

foreach ($fixes as $file => $config) {
    $filepath = __DIR__ . '/' . $file;
    if (!file_exists($filepath)) {
        echo "File not found: $file\n";
        continue;
    }

    $content = file_get_contents($filepath);
    $lines = explode("\n", $content);

    // Keep only lines up to last_valid_line
    $kept_lines = array_slice($lines, 0, $config['last_valid_line']);

    // Add closing braces
    $kept_lines[] = str_repeat('}', $config['add_braces']);

    $new_content = implode("\n", $kept_lines) . "\n";

    if (file_put_contents($filepath, $new_content)) {
        echo "✓ Truncated: $file\n";

        // Validate
        $output = shell_exec("php -l " . escapeshellarg($filepath) . " 2>&1");
        if (strpos($output, 'Errors parsing') === false) {
            echo "  ✓ Syntax valid\n";
        } else {
            echo "  ✗ Still has errors: $output\n";
        }
    } else {
        echo "✗ Failed to write: $file\n";
    }
}

echo "\n=== DONE ===\n";
