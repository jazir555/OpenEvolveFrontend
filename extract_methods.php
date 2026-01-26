<?php
/**
 * Script to extract and categorize methods from Sanitize.php.backup
 */

$backupFile = 'C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\Sanitize.php.backup';
$content = file_get_contents($backupFile);

// Extract all method definitions with their full bodies
preg_match_all('/^\s+(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*\([^)]*\)\s*(?::\s*[\w|?]+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/ms', $content, $matches, PREG_OFFSET_CAPTURE);

// This regex won't capture nested braces properly, so let's use a better approach
$lines = explode("\n", $content);
$methods = [];
$currentMethod = null;
$braceCount = 0;
$inMethod = false;
$startLine = 0;

foreach ($lines as $lineNum => $line) {
    // Check for method declaration
    if (preg_match('/^\s+(public|private|protected)\s+(static\s+)?function\s+(\w+)/', $line, $matches)) {
        if ($inMethod && $braceCount === 0) {
            // Save previous method
            $currentMethod['endLine'] = $lineNum - 1;
            $methods[] = $currentMethod;
        }

        $currentMethod = [
            'visibility' => $matches[1],
            'static' => !empty($matches[2]),
            'name' => $matches[3],
            'startLine' => $lineNum,
            'content' => $line . "\n"
        ];
        $inMethod = true;
        $braceCount = 0;
        // Count opening braces in this line
        $braceCount += substr_count($line, '{');
        $braceCount -= substr_count($line, '}');
    } elseif ($inMethod) {
        $currentMethod['content'] .= $line . "\n";
        $braceCount += substr_count($line, '{');
        $braceCount -= substr_count($line, '}');

        if ($braceCount === 0 && strpos($line, '}') !== false) {
            $currentMethod['endLine'] = $lineNum;
            $methods[] = $currentMethod;
            $inMethod = false;
            $currentMethod = null;
        }
    }
}

// Add last method if file doesn't end with newline
if ($inMethod && $currentMethod) {
    $currentMethod['endLine'] = count($lines) - 1;
    $methods[] = $currentMethod;
}

// Categorize methods
$categories = [
    'input' => [],
    'security' => [],
    'svg' => [],
    'utility' => [],
    'validation' => []
];

foreach ($methods as $method) {
    $name = $method['name'];
    $content = $method['content'];

    // Categorize based on method name and content
    if (strpos($name, 'sanitize') !== false ||
        in_array($name, ['basic_sanitize_text', 'sanitize_text_field', 'sanitize_key', 'sanitize_url', 'sanitize_file_path', 'sanitize_file_name'])) {
        // Further categorize
        if (strpos($name, 'svg') !== false || strpos($content, 'SVG') !== false) {
            $categories['svg'][] = $method;
        } elseif (strpos($name, 'validate') !== false || strpos($name, 'check') !== false) {
            $categories['validation'][] = $method;
        } elseif (strpos($name, 'ajax') !== false || strpos($name, 'security') !== false ||
                  strpos($name, 'permission') !== false || strpos($name, 'rest_') !== false) {
            $categories['security'][] = $method;
        } elseif (in_array($name, ['getConfig', 'functionExists', 'has_circular_reference', 'extract_svg_dimensions', 'parse_svg_dimension'])) {
            $categories['utility'][] = $method;
        } else {
            $categories['input'][] = $method;
        }
    } elseif (strpos($name, 'validate') !== false || strpos($name, 'is_valid') !== false || strpos($name, 'is_allowed') !== false) {
        $categories['validation'][] = $method;
    } elseif (strpos($name, 'ajax') !== false || strpos($name, 'security') !== false ||
              strpos($name, 'permission') !== false || strpos($name, 'rest_') !== false ||
              strpos($name, 'fail_ajax') !== false || strpos($name, 'can_perform') !== false) {
        $categories['security'][] = $method;
    } elseif (in_array($name, ['getConfig', 'functionExists', 'has_circular_reference', 'extract_svg_dimensions', 'parse_svg_dimension', 'log_svg_sanitization'])) {
        $categories['utility'][] = $method;
    } else {
        // Default to input for general sanitization methods
        $categories['input'][] = $method;
    }
}

// Output categorized methods
echo "=== METHOD CATEGORIZATION ===\n\n";
foreach ($categories as $cat => $methods) {
    echo strtoupper($cat) . " (" . count($methods) . " methods):\n";
    foreach ($methods as $m) {
        echo "  - {$m['name']} (line {$m['startLine']})\n";
    }
    echo "\n";
}
