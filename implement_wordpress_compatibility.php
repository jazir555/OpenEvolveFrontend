<?php
/**
 * WordPress Compatibility Implementation Script
 *
 * Scans all helper files and adds WordPressCompatibility.php require statement
 */

$wp_functions = [
    'current_time',
    'wp_date',
    'esc_html',
    'esc_html__',
    'esc_sql',
    'esc_url',
    '__',
    '_e',
    'wp_schedule_event',
    'wp_schedule_single_event',
    'wp_next_scheduled',
    'wp_clear_scheduled_hook',
    'wp_cache_get',
    'wp_cache_set',
    'wp_cache_delete',
    'wp_cache_add',
    'get_transient',
    'set_transient',
    'delete_transient',
    'get_option',
    'update_option',
    'do_action',
    'apply_filters',
    '_get_cron_array',
    'absint',
    'maybe_serialize',
    'wp_get_schedules',
    'is_wp_error'
];

$directories = [
    'TaskHelpers',
    'SettingsHelpers',
    'DatabaseHelpers',
    'AjaxHelpers',
    'AssetDataHelpers',
    'AssetOrderHelpers',
    'ExtractHelpers',
    'LoggingHelpers',
    'CleanupHelpers',
    'RetryHelpers',
    'SanitizeHelpers',
    'ProcessHelpers'
];

$base_dir = __DIR__;
$results = [];
$total_files_updated = 0;
$total_function_calls = 0;

foreach ($directories as $dir) {
    $dir_path = $base_dir . '/' . $dir;
    if (!is_dir($dir_path)) {
        continue;
    }

    $files = glob($dir_path . '/*.php');
    $results[$dir] = [
        'files_scanned' => 0,
        'files_updated' => 0,
        'function_calls' => 0,
        'files' => []
    ];

    foreach ($files as $file) {
        $content = file_get_contents($file);
        $results[$dir]['files_scanned']++;

        // Check if file already has WordPressCompatibility require
        if (strpos($content, 'WordPressCompatibility.php') !== false) {
            continue; // Skip if already included
        }

        // Scan for WordPress function calls
        $file_functions = [];
        foreach ($wp_functions as $func) {
            // Look for function calls with word boundaries
            $pattern = '/\b' . preg_quote($func, '/') . '\s*\(/';
            if (preg_match_all($pattern, $content, $matches)) {
                $count = count($matches[0]);
                if ($count > 0) {
                    $file_functions[$func] = $count;
                    $results[$dir]['function_calls'] += $count;
                }
            }
        }

        if (empty($file_functions)) {
            continue; // No WordPress functions found
        }

        // Find namespace declaration
        $namespace_pattern = '/namespace\s+[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*(\\\\[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)*;/';
        if (!preg_match($namespace_pattern, $content, $matches, PREG_OFFSET_CAPTURE)) {
            echo "Warning: No namespace found in $file\n";
            continue;
        }

        $namespace_end = $matches[0][1] + strlen($matches[0][0]);

        // Build require statement
        $require_line = "\nrequire_once __DIR__ . '/../WordPressCompatibility.php';\n";

        // Insert after namespace
        $new_content = substr_replace($content, $require_line, $namespace_end, 0);

        // Write back
        file_put_contents($file, $new_content);

        $results[$dir]['files_updated']++;
        $total_files_updated++;
        $results[$dir]['files'][] = [
            'name' => basename($file),
            'functions' => $file_functions
        ];

        echo "Updated: $file\n";
        foreach ($file_functions as $func => $count) {
            echo "  - $func: $count call(s)\n";
        }
    }

    $total_function_calls += $results[$dir]['function_calls'];
}

echo "\n=== IMPLEMENTATION COMPLETE ===\n\n";
echo "Total files updated: $total_files_updated\n";
echo "Total WordPress function calls covered: $total_function_calls\n\n";

foreach ($results as $dir => $stats) {
    if ($stats['files_scanned'] > 0) {
        echo "$dir:\n";
        echo "  Files scanned: {$stats['files_scanned']}\n";
        echo "  Files updated: {$stats['files_updated']}\n";
        echo "  Function calls: {$stats['function_calls']}\n";
        if (!empty($stats['files'])) {
            echo "  Updated files:\n";
            foreach ($stats['files'] as $file) {
                echo "    - {$file['name']}: " . implode(', ', array_keys($file['functions'])) . "\n";
            }
        }
        echo "\n";
    }
}

echo "\nNow validating all updated files...\n\n";

$validation_errors = 0;
foreach ($results as $dir => $stats) {
    foreach ($stats['files'] as $file_info) {
        $file = $base_dir . '/' . $dir . '/' . $file_info['name'];
        $output = [];
        $return_var = 0;
        exec("php -l " . escapeshellarg($file) . " 2>&1", $output, $return_var);

        if ($return_var !== 0) {
            echo "SYNTAX ERROR in $file:\n";
            echo implode("\n", $output) . "\n";
            $validation_errors++;
        } else {
            echo "✓ Valid: $file\n";
        }
    }
}

echo "\n=== VALIDATION COMPLETE ===\n";
echo "Errors found: $validation_errors\n";

if ($validation_errors === 0) {
    echo "\n✓ All files updated successfully and validated!\n";
} else {
    echo "\n✗ Some files have syntax errors that need to be fixed.\n";
}

// Save detailed report
$report_file = $base_dir . '/WORDPRESS_COMPATIBILITY_IMPLEMENTATION_REPORT.md';
$report = "# WordPress Compatibility Implementation Report\n\n";
$report .= "Generated: " . date('Y-m-d H:i:s') . "\n\n";
$report .= "## Summary\n\n";
$report .= "- **Total files updated:** $total_files_updated\n";
$report .= "- **Total WordPress function calls covered:** $total_function_calls\n";
$report .= "- **Validation errors:** $validation_errors\n\n";

$report .= "## Results by Directory\n\n";
foreach ($results as $dir => $stats) {
    if ($stats['files_scanned'] > 0) {
        $report .= "### $dir\n\n";
        $report .= "- Files scanned: {$stats['files_scanned']}\n";
        $report .= "- Files updated: {$stats['files_updated']}\n";
        $report .= "- WordPress function calls: {$stats['function_calls']}\n\n";

        if (!empty($stats['files'])) {
            $report .= "#### Updated Files\n\n";
            foreach ($stats['files'] as $file) {
                $report .= "**{$file['name']}**\n\n";
                $report .= "Functions used:\n";
                foreach ($file['functions'] as $func => $count) {
                    $report .= "- `$func()` - $count call(s)\n";
                }
                $report .= "\n";
            }
        }
    }
}

file_put_contents($report_file, $report);
echo "\nDetailed report saved to: $report_file\n";
