<?php
/**
 * Automated Type Hints Fixer
 *
 * This script adds return type declarations to all methods in helper files
 * that are missing them.
 */

error_reporting(E_ALL);
ini_set('display_errors', '1');

// Base directory
$baseDir = __DIR__;
$helperDirs = glob($baseDir . '/*Helpers', GLOB_ONLYDIR);

// Statistics
$stats = [
    'files_processed' => 0,
    'return_types_added' => 0,
    'parameter_types_added' => 0,
    'loose_comparisons_fixed' => 0,
    'errors' => []
];

// Common return type mappings based on return statements
$returnTypeMappings = [
    'return [];' => ': array',
    'return []; }' => ': array',
    'return false;' => ': false',
    'return false; }' => ': false',
    'return true;' => ': bool',
    'return true; }' => ': bool',
    'return null;' => ': void',
    'return null; }' => ': void',
    "return '';" => ': string',
    "return ''; }" => ': string',
    'return "";' => ': string',
    'return ""; }' => ': string',
    'return 0;' => ': int',
    'return 0; }' => ': int',
    'return $wpdb;' => ': ?wpdb',
];

// Process each helper directory
foreach ($helperDirs as $dir) {
    $files = glob($dir . '/*.php');
    foreach ($files as $file) {
        processFile($file, $stats);
    }
}

// Output results
echo "\n=== TYPE HINTS FIXING REPORT ===\n";
echo "Files Processed: {$stats['files_processed']}\n";
echo "Return Types Added: {$stats['return_types_added']}\n";
echo "Parameter Types Added: {$stats['parameter_types_added']}\n";
echo "Loose Comparisons Fixed: {$stats['loose_comparisons_fixed']}\n";

if (!empty($stats['errors'])) {
    echo "\nErrors encountered:\n";
    foreach ($stats['errors'] as $error) {
        echo "  - $error\n";
    }
}

echo "\nTotal issues fixed: " . ($stats['return_types_added'] + $stats['parameter_types_added'] + $stats['loose_comparisons_fixed']) . "\n";

/**
 * Process a single file
 */
function processFile($file, &$stats) {
    $content = file_get_contents($file);
    if ($content === false) {
        $stats['errors'][] = "Could not read file: $file";
        return;
    }

    $originalContent = $content;
    $changes = 0;

    // 1. Fix missing return types
    $content = fixReturnTypes($content, $changes);

    // 2. Fix missing parameter types (constructors and simple cases)
    $content = fixParameterTypes($content, $changes);

    // 3. Fix loose comparisons
    $content = fixLooseComparisons($content, $changes);

    // Only write if changes were made
    if ($content !== $originalContent) {
        $result = file_put_contents($file, $content);
        if ($result === false) {
            $stats['errors'][] = "Could not write file: $file";
        } else {
            // Validate syntax
            $output = [];
            $returnVar = 0;
            exec("php -l " . escapeshellarg($file) . " 2>&1", $output, $returnVar);

            if ($returnVar !== 0) {
                $stats['errors'][] = "Syntax error in $file: " . implode("\n", $output);
                // Revert changes
                file_put_contents($file, $originalContent);
            } else {
                $stats['files_processed']++;
                $stats['return_types_added'] += $changes['return_types'] ?? 0;
                $stats['parameter_types_added'] += $changes['parameter_types'] ?? 0;
                $stats['loose_comparisons_fixed'] += $changes['loose_comparisons'] ?? 0;

                echo "Fixed: " . basename($file) . " (+" . ($changes['return_types'] ?? 0) . " return types, +" .
                     ($changes['parameter_types'] ?? 0) . " param types, +" .
                     ($changes['loose_comparisons'] ?? 0) . " comparisons)\n";
            }
        }
    }
}

/**
 * Fix missing return types
 */
function fixReturnTypes($content, &$changes) {
    $changes['return_types'] = 0;

    // Pattern: public function methodName(params) {  (no return type)
    $pattern = '/(public|private|protected)\s+function\s+(\w+)\s*\(([^)]*)\)\s*\{/';

    $content = preg_replace_callback($pattern, function($matches) {
        $visibility = $matches[1];
        $methodName = $matches[2];
        $params = $matches[3];

        // Look ahead to find the return statement
        $returnType = null;

        // Simple heuristic: check if it's a getter or common pattern
        if (strpos($methodName, 'get_') === 0 || strpos($methodName, 'has_') === 0 || strpos($methodName, 'is_') === 0) {
            $returnType = ': mixed'; // Default for getters
        } elseif (strpos($methodName, 'set_') === 0) {
            $returnType = ': void'; // Setters don't return
        } elseif (in_array($methodName, ['__construct', '__destruct', '__clone'])) {
            $returnType = ''; // Constructors/destructors don't have return types
        } else {
            // Try to infer from method name
            if (strpos($methodName, 'validate') !== false || strpos($methodName, 'check') !== false) {
                $returnType = ': bool';
            } elseif (strpos($methodName, 'find') !== false || strpos($methodName, 'query') !== false) {
                $returnType = ': array';
            } else {
                $returnType = ': mixed';
            }
        }

        // Add return type if not empty
        if (!empty($returnType)) {
            $changes['return_types']++;
            return "{$visibility} function {$methodName}({$params}){$returnType} {";
        }

        return $matches[0]; // No change
    }, $content);

    return $content;
}

/**
 * Fix missing parameter types
 */
function fixParameterTypes($content, &$changes) {
    $changes['parameter_types'] = 0;

    // Fix constructor parameters without types: __construct($param = null)
    $content = preg_replace_callback(
        '/public\s+function\s+__construct\s*\(([^)]*)\)/s',
        function($matches) use (&$changes) {
            $params = $matches[1];

            // Skip if already has types
            if (preg_match('/\$\w+\s+\?\s*\w+/', $params) || preg_match('/\$\w+\s+\w+\s+\$/', $params)) {
                return $matches[0];
            }

            // Add types to untyped parameters
            $fixedParams = preg_replace_callback(
                '/\$([a-z_][a-z0-9_]*)\s*=\s*null/i',
                function($m) use (&$changes) {
                    $changes['parameter_types']++;
                    return "?{$m[1]} = null";
                },
                $params
            );

            return "public function __construct({$fixedParams})";
        },
        $content
    );

    return $content;
}

/**
 * Fix loose comparisons
 */
function fixLooseComparisons($content, &$changes) {
    $changes['loose_comparisons'] = 0;

    // Fix == to === (but be careful with strings)
    $replacements = [
        '/(\s+)\$([a-z_][a-z0-9_]*)\s+==\s+false/i' => '\1$$2 === false',
        '/(\s+)\$([a-z_][a-z0-9_]*)\s+!=\s+false/i' => '\1$$2 !== false',
        '/(\s+)\$([a-z_][a-z0-9_]*)\s+==\s+true/i' => '\1$$2 === true',
        '/(\s+)\$([a-z_][a-z0-9_]*)\s+!=\s+true/i' => '\1$$2 !== true',
        '/(\s+)\$([a-z_][a-z0-9_]*)\s+==\s+null/i' => '\1$$2 === null',
        '/(\s+)\$([a-z_][a-z0-9_]*)\s+!=\s+null/i' => '\1$$2 !== null',
        '/\(\s*\$([a-z_][a-z0-9_]*)\s+==\s+([^)]+)\s*\)/ie' => '($1 === $2)',
    ];

    foreach ($replacements as $pattern => $replacement) {
        $count = 0;
        $content = preg_replace($pattern, $replacement, $content, -1, $count);
        $changes['loose_comparisons'] += $count;
    }

    return $content;
}
