#!/usr/bin/env php
<?php
/**
 * Comprehensive Type Hints Fixer for All Helper Files
 *
 * This script:
 * 1. Scans all helper files for missing return types
 * 2. Fixes corrupted catch blocks
 * 3. Adds missing return types based on naming conventions
 * 4. Validates all files with PHP lint
 * 5. Generates comprehensive report
 */

declare(strict_types=1);

echo "=== Comprehensive Type Hints Fixer ===\n";
echo "Scanning and fixing all helper files...\n\n";

$baseDir = __DIR__;
$helperDirs = [
    'AjaxHelpers',
    'ExtractHelpers',
    'LoggingHelpers',
    'RetryHelpers',
    'SanitizeHelpers',
    'SettingsHelpers',
    'AssetOrderHelpers',
    'AssetDataHelpers',
    'CleanupHelpers',
    'ProcessHelpers',
    'DatabaseHelpers',
    'TaskHelpers',
];

$stats = [
    'total_files' => 0,
    'files_with_corruption' => 0,
    'corruption_fixed' => 0,
    'files_modified' => 0,
    'methods_typed' => 0,
    'validation_errors' => 0,
    'files_validated' => 0,
];

$details = [];

// Function to fix corrupted catch blocks
function fixCorruptedCatchBlocks(?string $content): string {
    if ($content === null) {
        return '';
    }

    // Fix duplicate catch blocks - simpler pattern
    $content = preg_replace('/\} catch \(\\InvalidArgumentException \$e\) \{[^\}]*return \$this->getFallbackValue\(\);\s*\} catch \(\\RuntimeException \$e\) \{\s*throw \$e;\s*\}(\s*\})+/s', '}', $content);

    return $content ?: '';
}

// Function to add return type to method
function addReturnType(string $methodName, string $params): ?string {
    // Void patterns
    if (preg_match('/^(set_|delete_|clear_|remove_|add_|update_|save_|load_|handle_|process_|execute_|run_|invalidate_|log_|write_|enqueue_|dequeue_|trigger_|start_|stop_|begin_|end_|init_|ajax_|wp_|action_)/i', $methodName)) {
        return 'void';
    }

    // Bool patterns
    if (preg_match('/^(is_|has_|can_|should_|will_|validate_|check_|verify_|exists_|contains_)/i', $methodName)) {
        return 'bool';
    }

    // Array patterns
    if (preg_match('/^(get_|fetch_|retrieve_|list_|all_|find_|search_|query_)/i', $methodName)) {
        return 'array';
    }

    // Int patterns
    if (preg_match('/^(count_|total_|sum_|calculate_|compute_|size_|length)/i', $methodName)) {
        return 'int';
    }

    // String patterns
    if (preg_match('/^(render_|format_|escape_|sanitize_|prepare_|build_|create_)/i', $methodName)) {
        return 'string';
    }

    return null; // Can't determine automatically
}

// Process each directory
foreach ($helperDirs as $dir) {
    $fullDir = $baseDir . '/' . $dir;
    if (!is_dir($fullDir)) {
        continue;
    }

    echo "Processing $dir...\n";
    $files = glob($fullDir . '/*.php');

    foreach ($files as $file) {
        if (strpos($file, 'Interface') !== false) {
            continue; // Skip interface files
        }

        $stats['total_files']++;
        $filename = basename($file);
        $content = file_get_contents($file);

        if ($content === false) {
            $details[] = "ERROR: Could not read $dir/$filename";
            $stats['validation_errors']++;
            continue;
        }

        $original = $content;
        $changes = 0;
        $hasCorruption = false;

        // Fix corrupted catch blocks
        $fixed = fixCorruptedCatchBlocks($content);
        if ($fixed !== $content) {
            $content = $fixed;
            $stats['corruption_fixed']++;
            $changes++;
            $hasCorruption = true;
        }

        // Add missing return types
        $content = preg_replace_callback(
            '/^\s*(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?!:\s*(?:void|int|float|string|bool|array|object|\?|null))\s*\{/m',
            function ($matches) use (&$changes, $dir, $filename) {
                $visibility = $matches[1];
                $static = $matches[2] ?? '';
                $methodName = $matches[3];
                $params = $matches[4];

                $returnType = addReturnType($methodName, $params);

                if ($returnType) {
                    $changes++;
                    return "\t" . $visibility . ' ' . $static . "function $methodName($params): $returnType {";
                }

                return $matches[0];
            },
            $content
        );

        // Write changes
        if ($content !== $original) {
            if (file_put_contents($file, $content)) {
                $stats['files_modified']++;
                $stats['methods_typed'] += $changes;

                if ($hasCorruption) {
                    $stats['files_with_corruption']++;
                    $details[] = "FIXED CORRUPTION: $dir/$filename ($changes changes)";
                } else {
                    $details[] = "FIXED: $dir/$filename ($changes methods)";
                }
            } else {
                $details[] = "ERROR: Could not write to $dir/$filename";
                $stats['validation_errors']++;
            }
        }

        // Validate syntax
        exec("php -l " . escapeshellarg($file) . " 2>&1", $output, $returnCode);
        $stats['files_validated']++;

        if ($returnCode !== 0) {
            $details[] = "SYNTAX ERROR in $dir/$filename: " . implode("\n", $output);
            $stats['validation_errors']++;
        }
    }
}

// Print results
echo "\n\n=== FINAL REPORT ===\n\n";
echo "Statistics:\n";
echo "  Total files scanned: {$stats['total_files']}\n";
echo "  Files with corruption found: {$stats['files_with_corruption']}\n";
echo "  Corruption issues fixed: {$stats['corruption_fixed']}\n";
echo "  Total files modified: {$stats['files_modified']}\n";
echo "  Methods typed: {$stats['methods_typed']}\n";
echo "  Files validated: {$stats['files_validated']}\n";
echo "  Validation errors: {$stats['validation_errors']}\n";

if ($stats['total_files'] > 0) {
    $completion = (($stats['total_files'] - $stats['validation_errors']) / $stats['total_files']) * 100;
    echo "\n  Completion rate: " . number_format($completion, 2) . "%\n";
}

echo "\nDetailed Changes:\n";
foreach (array_slice($details, 0, 50) as $detail) {
    echo "  - $detail\n";
}

if (count($details) > 50) {
    echo "  ... and " . (count($details) - 50) . " more changes\n";
}

echo "\n=== DONE ===\n";
