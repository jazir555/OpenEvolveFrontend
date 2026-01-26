<?php

declare(strict_types=1);

/**
 * Scan helper directories for missing return type declarations
 */

$directories = [
    'AjaxHelpers',
    'DatabaseHelpers',
    'ProcessHelpers',
    'TaskHelpers',
    'SettingsHelpers',
    'AssetOrderHelpers',
    'CleanupHelpers',
    'RetryHelpers',
    'ExtractHelpers',
    'LoggingHelpers',
    'SanitizeHelpers',
];

$results = [];

foreach ($directories as $dir) {
    $files = glob(__DIR__ . "/{$dir}/*.php");
    foreach ($files as $file) {
        $content = file_get_contents($file);
        $lines = explode("\n", $content);
        $missing = [];

        $in_class = false;
        $in_method = false;
        $method_line = 0;
        $brace_count = 0;
        $in_interface = false;

        for ($i = 0; $i < count($lines); $i++) {
            $line = $lines[$i];

            // Skip interfaces
            if (preg_match('/^\s*interface\s+\w+/', $line)) {
                $in_interface = true;
            }
            if ($in_interface && preg_match('/^\s*\}/', $line)) {
                $in_interface = false;
            }
            if ($in_interface) {
                continue;
            }

            // Check for class
            if (preg_match('/^\s*(abstract\s+)?class\s+/', $line)) {
                $in_class = true;
            }

            if (!$in_class) {
                continue;
            }

            // Check for method declaration
            if (preg_match('/^\s*(public|private|protected)\s+function\s+(\w+)\s*\((.*)\)\s*(?=:)/', $line, $matches)) {
                $visibility = $matches[1];
                $method_name = $matches[2];

                // Skip constructor and destructor
                if ($method_name === '__construct' || $method_name === '__destruct') {
                    continue;
                }

                // Look ahead to see if return type is declared
                $has_return_type = false;
                $next_line = isset($lines[$i + 1]) ? trim($lines[$i + 1]) : '';
                $current_line_trimmed = trim($line);

                // Check for return type on same line
                if (preg_match('/:\s*(void|bool|int|float|string|array|callable|object|mixed|\?[\w]+|[\w\|]+)\s*(\{|$)/', $current_line_trimmed)) {
                    $has_return_type = true;
                }

                // Check for return type on next line (multi-line declaration)
                if (!$has_return_type && !empty($next_line) && preg_match('/^:\s*(void|bool|int|float|string|array|callable|object|mixed|\?[\w]+|[\w\|]+)\s*(\{|$)/', $next_line)) {
                    $has_return_type = true;
                }

                if (!$has_return_type) {
                    $relative_path = str_replace(__DIR__ . DIRECTORY_SEPARATOR, '', $file);
                    $missing[] = [
                        'line' => $i + 1,
                        'method' => $method_name,
                        'visibility' => $visibility
                    ];
                }
            }
        }

        if (!empty($missing)) {
            $results[$dir][basename($file)] = $missing;
        }
    }
}

// Output results
echo "SCAN RESULTS: Missing Return Type Declarations\n";
echo "==============================================\n\n";

$total_missing = 0;
$total_files = 0;

foreach ($results as $dir => $files) {
    if (!empty($files)) {
        echo "## {$dir}/\n";
        foreach ($files as $file => $methods) {
            $count = count($methods);
            $total_missing += $count;
            $total_files++;
            echo "  - {$file}: {$count} method(s)\n";
            foreach ($methods as $method) {
                echo "    Line {$method['line']}: {$method['visibility']} function {$method['method']}()\n";
            }
        }
        echo "\n";
    }
}

echo "\nSummary:\n";
echo "  Total files with issues: {$total_files}\n";
echo "  Total methods missing return types: {$total_missing}\n";

if ($total_missing === 0) {
    echo "\n✅ All helper files have complete return type declarations!\n";
} else {
    echo "\n⚠️  Found {$total_missing} methods that need return type declarations.\n";
}
