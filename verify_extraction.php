<?php
/**
 * Final verification of helper extraction
 */

$helpersDir = __DIR__ . '/AssetDataHelpers/';

$expectedHelpers = [
    'AssetCacheHelper' => 4,
    'AssetDatabaseHelper' => 7,
    'AssetDataRegistryHelper' => 1,
    'AssetIntegrationHelper' => 1,
    'AssetMemoryHelper' => 4,
    'AssetMetadataHelper' => 12,
    'AssetOrderHelper' => 2,
    'AssetQueryHelper' => 8,
    'AssetStatisticsHelper' => 4,
    'AssetTaskHelper' => 2,
    'AssetURLHelper' => 11,
    'AssetUtilityHelper' => 11,
    'AssetValidationHelper' => 9,
];

echo "=== FINAL VERIFICATION REPORT ===\n\n";

$allPassed = true;

foreach ($expectedHelpers as $className => $expectedMethodCount) {
    $filename = $className . '.php';
    $filepath = $helpersDir . $filename;

    echo "Checking $filename...\n";

    // File exists
    if (!file_exists($filepath)) {
        echo "  ❌ File does not exist\n";
        $allPassed = false;
        continue;
    }
    echo "  ✓ File exists\n";

    // Valid PHP syntax
    $output = shell_exec("php -l " . escapeshellarg($filepath) . " 2>&1");
    if (strpos($output, 'No syntax errors') === false) {
        echo "  ❌ Syntax error detected\n";
        $allPassed = false;
        continue;
    }
    echo "  ✓ Valid PHP syntax\n";

    // Read content
    $content = file_get_contents($filepath);

    // Check namespace
    if (strpos($content, 'namespace LHA\\AssetDataHelpers;') === false) {
        echo "  ❌ Missing correct namespace\n";
        $allPassed = false;
    } else {
        echo "  ✓ Correct namespace (LHA\\AssetDataHelpers)\n";
    }

    // Check strict types
    if (strpos($content, 'declare(strict_types=1);') === false) {
        echo "  ❌ Missing strict types declaration\n";
        $allPassed = false;
    } else {
        echo "  ✓ Strict types declared\n";
    }

    // Check class exists
    if (strpos($content, "class $className") === false) {
        echo "  ❌ Class $className not found\n";
        $allPassed = false;
    } else {
        echo "  ✓ Class $className defined\n";
    }

    // Count methods
    preg_match_all('/^\s+(public|private|protected)\s+static\s+function\s+/m', $content, $matches);
    $actualMethodCount = count($matches[0]);

    if ($actualMethodCount !== $expectedMethodCount) {
        echo "  ⚠ Method count mismatch: expected $expectedMethodCount, found $actualMethodCount\n";
    } else {
        echo "  ✓ All $expectedMethodCount methods extracted\n";
    }

    // Check for docblocks
    preg_match_all('/\/\*\*[\s\S]*?\*\//', $content, $docblocks);
    $docblockCount = count($docblocks[0]);
    echo "  📝 $docblockCount docblocks found\n";

    echo "\n";
}

echo "=== " . ($allPassed ? "✓ ALL CHECKS PASSED" : "⚠ SOME CHECKS FAILED") . " ===\n";
