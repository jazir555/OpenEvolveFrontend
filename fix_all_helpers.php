<?php
/**
 * Automated Helper File Fixer
 *
 * Fixes common syntax errors in helper files:
 * - Missing closing braces
 * - Missing PHPDoc comment openers
 * - Property declaration issues
 */

$classes_dir = __DIR__;
$fix_count = 0;
$errors = [];

// Get all helper PHP files
$files = [];
$directories = [
    'RetryHelpers',
    'SettingsHelpers',
    'SanitizeHelpers',
];

foreach ($directories as $dir) {
    $path = $classes_dir . '/' . $dir;
    if (is_dir($path)) {
        $iterator = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($path)
        );

        foreach ($iterator as $file) {
            if ($file->isFile() && $file->getExtension() === 'php') {
                $files[] = $file->getPathname();
            }
        }
    }
}

echo "Scanning " . count($files) . " helper files...\n\n";

foreach ($files as $file) {
    $content = file_get_contents($file);
    $original = $content;
    $filename = basename($file);

    // Fix 1: Check brace balance
    $open_count = substr_count($content, '{');
    $close_count = substr_count($content, '}');
    $diff = $open_count - $close_count;

    if ($diff > 0) {
        echo "[$filename] Missing $diff closing braces\n";
        // Add missing closing braces at end
        $content = rtrim($content);
        // Remove existing closing brace if present
        if (substr($content, -1) === '}') {
            $content = substr($content, 0, -1);
        }
        // Add the correct number of closing braces
        $content .= str_repeat('}', $diff + 1) . "\n";
    }

    // Fix 2: Check for incomplete PHPDoc
    if (preg_match('/\n\s*\*\s*@return/m', $content)) {
        // Find lines with * but no opening /**
        $lines = explode("\n", $content);
        $in_phpdoc = false;
        $fixed_lines = [];

        foreach ($lines as $i => $line) {
            // Check if this looks like a PHPDoc comment without opener
            if (preg_match('/^\s*\*\s+@(?:since|return|param|throws)/', $line) && !$in_phpdoc) {
                // Insert /** before this line
                $prev_line = isset($lines[$i-1]) ? $lines[$i-1] : '';
                if (!preg_match('/\/\*\*$/', $prev_line)) {
                    $fixed_lines[] = '/**';
                }
            }

            $fixed_lines[] = $line;

            // Track if we're in a PHPDoc block
            if (preg_match('/\/\*\*/', $line)) {
                $in_phpdoc = false;
            } elseif (preg_match('/\/\*\*/', $line) === false && preg_match('/^\s*\*/', $line)) {
                $in_phpdoc = true;
            }
        }

        $content = implode("\n", $fixed_lines);
    }

    // Fix 3: Common syntax patterns
    // Fix incomplete property declarations like "private Type \;"
    $content = preg_replace('/(private|public|protected)\s+(\S+\s*(?:\\\\)?\\\\)\s*;/', '$1 $2 $variable_name;', $content);

    // Only write if content changed
    if ($content !== $original) {
        if (file_put_contents($file, $content)) {
            $fix_count++;
            echo "✓ Fixed: $filename\n";

            // Validate the fix
            $output = shell_exec("php -l " . escapeshellarg($file) . " 2>&1");
            if (strpos($output, 'Errors parsing') !== false) {
                echo "  ✗ Still has errors: $output\n";
                $errors[] = $filename;
            } else {
                echo "  ✓ Syntax valid\n";
            }
        } else {
            echo "✗ Failed to write: $filename\n";
            $errors[] = $filename;
        }
    }
}

echo "\n=== SUMMARY ===\n";
echo "Files scanned: " . count($files) . "\n";
echo "Files fixed: $fix_count\n";
echo "Files with errors: " . count($errors) . "\n";

if (!empty($errors)) {
    echo "\nStill needs manual fix:\n";
    foreach ($errors as $err) {
        echo "  - $err\n";
    }
}
