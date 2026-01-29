<?php

declare(strict_types=1);

/**
 * Scan helper directories for missing return type declarations
 * More accurate version
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
$total_files_scanned = 0;

foreach ($directories as $dir) {
    $files = glob(__DIR__ . "/{$dir}/*.php");
    foreach ($files as $file) {
        $total_files_scanned++;
        $content = file_get_contents($file);
        $lines = explode("\n", $content);
        $missing = [];

        $in_class = false;
        $in_interface = false;

        for ($i = 0; $i < count($lines); $i++) {
            $line = $lines[$i];
            $trimmed = trim($line);

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
            if (preg_match('/^\s*(abstract\s+)?(class|trait)\s+/', $line)) {
                $in_class = true;
            }

            if (!$in_class) {
                continue;
            }

            // Check for method declaration with function keyword
            // Match: visibility function name(params)
            if (preg_match('/^\s*(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*\(/', $line, $matches)) {
                $visibility = $matches[1];
                $method_name = $matches[3];

                // Skip magic methods
                if (in_array($method_name, ['__construct', '__destruct', '__call', '__callStatic', '__get', '__set', '__isset', '__unset', '__sleep', '__wakeup', '__serialize', '__unserialize', '__toString', '__invoke', '__set_state', '__clone', '__debugInfo'])) {
                    continue;
                }

                // Check if return type is declared on the same line after closing paren
                // Look for pattern: function name(...) : return_type
                if (preg_match('/\)\s*:\s*\S/', $line)) {
                    // Has return type on same line
                    continue;
                }

                // Check multi-line declaration - look at next few lines
                $has_return_type = false;
                for ($j = $i + 1; $j < min($i + 5, count($lines)); $j++) {
                    $next_line = trim($lines[$j]);
                    if (preg_match('/^:\s*(void|bool|int|float|string|array|callable|object|mixed|\??\w+|[\w\|]+\??)/', $next_line)) {
                        $has_return_type = true;
                        break;
                    }
                    // If we hit an opening brace, declaration is over
                    if (strpos($next_line, '{') !== false) {
                        break;
                    }
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
echo "Scanned {$total_files_scanned} PHP files\n\n";

$total_missing = 0;
$total_files = 0;

if (empty($results)) {
    echo "✅ NO ISSUES FOUND!\n";
    echo "All methods have complete return type declarations.\n";
} else {
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
    echo "\n⚠️  Found {$total_missing} methods that need return type declarations.\n";
}
