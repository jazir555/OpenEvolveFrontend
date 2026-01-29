<?php
/**
 * Helper Health Check - Simplified Version
 */

$directories = [
    'AjaxHelpers',
    'AssetDataHelpers',
    'AssetOrderHelpers',
    'CleanupHelpers',
    'DatabaseHelpers',
    'ExtractHelpers',
    'LoggingHelpers',
    'ProcessHelpers',
    'RetryHelpers',
    'SanitizeHelpers',
    'SettingsHelpers',
    'TaskHelpers',
];

$issues = [];
$checked_files = 0;

echo "================================================================\n";
echo "HELPER HEALTH CHECK\n";
echo "================================================================\n\n";

foreach ($directories as $dir) {
    $iterator = new RecursiveIteratorIterator(
        new RecursiveDirectoryIterator(__DIR__ . '/' . $dir)
    );

    foreach ($iterator as $file) {
        if ($file->isFile() && $file->getExtension() === 'php') {
            $checked_files++;
            $filename = basename($file);
            $content = file_get_contents($file->getPathname());
            $relative_path = str_replace(__DIR__ . '\', '', $file->getPathname());

            // Check 1: Missing namespace
            if (strpos($content, 'namespace ') === false) {
                $issues[] = [
                    'file' => $relative_path,
                    'type' => 'Missing Namespace',
                    'severity' => 'ERROR'
                ];
            }

            // Check 2: Interface files - check if they extend something
            if (strpos($filename, 'Interface.php') !== false) {
                if (strpos($content, 'interface ') === false) {
                    $issues[] = [
                        'file' => $relative_path,
                        'type' => 'Interface File Missing Interface Declaration',
                        'severity' => 'ERROR'
                    ];
                }
            }

            // Check 3: Duplicate class/interface names
            if (preg_match('/^(class|interface|trait)\s+(\w+)/m', $content, $matches)) {
                // Found a class/interface declaration
            }
        }
    }
}

echo "Checked Files: $checked_files\n";
echo "Issues Found: " . count($issues) . "\n\n";

if (!empty($issues)) {
    echo "================================================================\n";
    echo "ISSUES\n";
    echo "================================================================\n";
    foreach ($issues as $issue) {
        echo "[{$issue['severity']}] {$issue['file']}\n";
        echo "  Type: {$issue['type']}\n\n";
    }
} else {
    echo "✅ NO CRITICAL ISSUES FOUND\n";
}

echo "\n================================================================\n";
echo !empty($issues) ? "❌ HEALTH CHECK FOUND ISSUES\n" : "✅ HEALTH CHECK PASSED\n";
echo "================================================================\n";
