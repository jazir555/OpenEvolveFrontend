<?php
/**
 * Script to extract methods from Sanitize.php.backup and write to helper files
 */

$backupFile = 'C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\Sanitize.php.backup';
$content = file_get_contents($backupFile);
$lines = explode("\n", $content);

// Extract all methods with their full content
function extractMethods($lines) {
    $methods = [];
    $currentMethod = null;
    $braceCount = 0;
    $inMethod = false;

    foreach ($lines as $lineNum => $line) {
        $lineNum++; // 1-indexed

        // Check for method declaration
        if (preg_match('/^\s+(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*\([^)]*\)\s*(?::\s*[\w|? \]+)?\s*\{/', $line, $matches)) {
            if ($inMethod && $braceCount === 0) {
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
            $braceCount = substr_count($line, '{') - substr_count($line, '}');
        } elseif ($inMethod) {
            $currentMethod['content'] .= $line . "\n";
            $braceCount += substr_count($line, '{') - substr_count($line, '}');

            if ($braceCount === 0 && strpos($line, '}') !== false) {
                $methods[] = $currentMethod;
                $inMethod = false;
                $currentMethod = null;
            }
        }
    }

    return $methods;
}

// Categorize a single method
function categorizeMethod($method) {
    $name = $method['name'];
    $content = $method['content'];

    // SVG methods
    if (strpos($name, 'svg') !== false || stripos($content, 'SVG') !== false) {
        if (in_array($name, ['extract_svg_dimensions', 'parse_svg_dimension', 'log_svg_sanitization'])) {
            return 'utility'; // These are helpers, not main SVG sanitization
        }
        return 'svg';
    }

    // Validation methods
    if (strpos($name, 'validate') !== false ||
        strpos($name, 'is_valid') !== false ||
        strpos($name, 'is_allowed') !== false ||
        strpos($name, 'check') !== false && strpos($name, 'ajax') === false) {
        return 'validation';
    }

    // Security methods
    if (strpos($name, 'ajax') !== false ||
        strpos($name, 'security') !== false ||
        strpos($name, 'permission') !== false ||
        strpos($name, 'rest_') !== false ||
        strpos($name, 'can_perform') !== false ||
        strpos($name, 'fail_ajax') !== false) {
        return 'security';
    }

    // Utility methods (helpers, config, logging)
    if (in_array($name, ['getConfig', 'functionExists', 'has_circular_reference',
                        'extract_svg_dimensions', 'parse_svg_dimension', 'log_svg_sanitization',
                        'get_sanitized_headers', 'sanitize_header', 'get_client_ip',
                        'log_validation_error'])) {
        return 'utility';
    }

    // Default to input sanitization
    return 'input';
}

$methods = extractMethods($lines);

// Categorize all methods
$categories = [
    'input' => [],
    'security' => [],
    'svg' => [],
    'utility' => [],
    'validation' => []
];

foreach ($methods as $method) {
    $category = categorizeMethod($method);
    $categories[$category][] = $method;
}

// Output the methods for each category
foreach ($categories as $cat => $methods) {
    $outputDir = "C:\\Users\\mmeadow\\Documents\\locallyhostassetsbackup\\classes\\SanitizeHelpers";
    $className = ucfirst($cat) . 'Helper';
    $fileName = "Sanitize" . ucfirst($cat) . "Helper.php";
    $filePath = $outputDir . "\\" . $fileName;

    $output = "<?php\n\nnamespace LHA\\SanitizeHelpers;\n\n/**\n * " . $className . "\n * Extracted from Sanitize.php.backup\n *\n";
    $output .= " * This file contains " . ucfirst($cat) . "-related sanitization methods.\n";
    $output .= " * Auto-generated from Sanitize.php.backup\n */\n";
    $output .= "class " . $className . " {\n\n";

    // Add static methods
    foreach ($methods as $method) {
        $output .= $method['content'] . "\n";
    }

    $output .= "}\n";

    // Write to file
    file_put_contents($filePath, $output);
    echo "Wrote $fileName with " . count($methods) . " methods\n";
}

echo "\nDone!\n";
