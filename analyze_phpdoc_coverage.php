<?php
/**
 * PHPDoc Coverage Analyzer for Helper Files
 *
 * Analyzes all helper files to determine which public methods need PHPDoc documentation.
 */

function analyzeHelperFile(string $filePath): array {
    $content = file_get_contents($filePath);
    if ($content === false) {
        return ['error' => 'Could not read file'];
    }

    $stats = [
        'file' => $filePath,
        'public_methods' => 0,
        'documented_methods' => 0,
        'undocumented_methods' => [],
        'coverage_percentage' => 0
    ];

    // Find all public methods
    preg_match_all('/^\s*public\s+function\s+(\w+)\s*\(/m', $content, $matches, PREG_OFFSET_CAPTURE);

    foreach ($matches[1] as $index => $match) {
        $methodName = $match[0];
        $methodOffset = $match[1];

        $stats['public_methods']++;

        // Check if there's a PHPDoc block before this method
        $precedingText = substr($content, 0, $methodOffset);
        $hasPHPDoc = preg_match('/\/\*\*[\s\S]*?\*\)\s*$/s', $precedingText);

        if ($hasPHPDoc) {
            $stats['documented_methods']++;
        } else {
            $stats['undocumented_methods'][] = $methodName;
        }
    }

    if ($stats['public_methods'] > 0) {
        $stats['coverage_percentage'] = round(($stats['documented_methods'] / $stats['public_methods']) * 100, 2);
    }

    return $stats;
}

function scanHelperDirectory(string $baseDir): array {
    $results = [];
    $totalPublicMethods = 0;
    $totalDocumentedMethods = 0;

    // Get all helper directories
    $helperDirs = glob($baseDir . '/*Helpers', GLOB_ONLYDIR);

    foreach ($helperDirs as $helperDir) {
        $helperType = basename($helperDir);
        $results[$helperType] = [];

        // Get all PHP files (excluding interfaces and subdirectories)
        $files = glob($helperDir . '/*.php');

        foreach ($files as $file) {
            // Skip interface files
            if (strpos(basename($file), 'Interface.php') !== false) {
                continue;
            }

            $stats = analyzeHelperFile($file);
            if (isset($stats['error'])) {
                continue;
            }

            // Only include files that have public methods
            if ($stats['public_methods'] > 0) {
                $results[$helperType][basename($file)] = $stats;
                $totalPublicMethods += $stats['public_methods'];
                $totalDocumentedMethods += $stats['documented_methods'];
            }
        }
    }

    $overallCoverage = $totalPublicMethods > 0
        ? round(($totalDocumentedMethods / $totalPublicMethods) * 100, 2)
        : 0;

    return [
        'by_helper_type' => $results,
        'summary' => [
            'total_public_methods' => $totalPublicMethods,
            'total_documented_methods' => $totalDocumentedMethods,
            'overall_coverage_percentage' => $overallCoverage
        ]
    ];
}

// Run the analysis
$baseDir = __DIR__;
$analysis = scanHelperDirectory($baseDir);

// Output results
echo "=== PHPDoc Coverage Analysis Report ===\n\n";

echo "Summary:\n";
echo "  Total Public Methods: " . $analysis['summary']['total_public_methods'] . "\n";
echo "  Total Documented: " . $analysis['summary']['total_documented_methods'] . "\n";
echo "  Overall Coverage: " . $analysis['summary']['overall_coverage_percentage'] . "%\n\n";

echo "Files Needing Attention (< 100% coverage):\n\n";

foreach ($analysis['by_helper_type'] as $helperType => $files) {
    echo "=== $helperType ===\n";

    $needsWork = false;
    foreach ($files as $filename => $stats) {
        if ($stats['coverage_percentage'] < 100) {
            $needsWork = true;
            echo "  - $filename\n";
            echo "    Public Methods: {$stats['public_methods']}\n";
            echo "    Documented: {$stats['documented_methods']}\n";
            echo "    Coverage: {$stats['coverage_percentage']}%\n";

            if (!empty($stats['undocumented_methods'])) {
                echo "    Undocumented: " . implode(', ', $stats['undocumented_methods']) . "\n";
            }
            echo "\n";
        }
    }

    if (!$needsWork) {
        echo "  All files fully documented!\n\n";
    }
}

echo "\n=== Detailed CSV Export ===\n";
echo "Helper Type,Filename,Public Methods,Documented,Coverage%,Undocumented Methods\n";

foreach ($analysis['by_helper_type'] as $helperType => $files) {
    foreach ($files as $filename => $stats) {
        $undocumented = !empty($stats['undocumented_methods'])
            ? '"' . implode('; ', $stats['undocumented_methods']) . '"'
            : '';
        echo "$helperType,$filename,{$stats['public_methods']},{$stats['documented_methods']},{$stats['coverage_percentage']}%,$undocumented\n";
    }
}
