<?php
/**
 * Fix helper files to wrap methods in proper class declarations
 */

$helpersDir = __DIR__ . '/AssetDataHelpers/';

$helperClasses = [
    'AssetCacheHelper.php' => 'AssetCacheHelper',
    'AssetDatabaseHelper.php' => 'AssetDatabaseHelper',
    'AssetDataRegistryHelper.php' => 'AssetDataRegistryHelper',
    'AssetIntegrationHelper.php' => 'AssetIntegrationHelper',
    'AssetMemoryHelper.php' => 'AssetMemoryHelper',
    'AssetMetadataHelper.php' => 'AssetMetadataHelper',
    'AssetOrderHelper.php' => 'AssetOrderHelper',
    'AssetQueryHelper.php' => 'AssetQueryHelper',
    'AssetStatisticsHelper.php' => 'AssetStatisticsHelper',
    'AssetTaskHelper.php' => 'AssetTaskHelper',
    'AssetURLHelper.php' => 'AssetURLHelper',
    'AssetUtilityHelper.php' => 'AssetUtilityHelper',
    'AssetValidationHelper.php' => 'AssetValidationHelper',
];

foreach ($helperClasses as $file => $className) {
    $filepath = $helpersDir . $file;

    if (!file_exists($filepath)) {
        echo "Skipping $file - does not exist\n";
        continue;
    }

    $content = file_get_contents($filepath);

    // Remove the old header
    $content = preg_replace('/<\?php\s*\ndeclare\(strict_types=1\);\s*\nnamespace LHA\\\\AssetDataHelpers;\s*/', '', $content);

    // Add docblock and class
    $newContent = "<?php\n\n";
    $newContent .= "declare(strict_types=1);\n\n";
    $newContent .= "namespace LHA\\AssetDataHelpers;\n\n";
    $newContent .= "/**\n";
    $newContent .= " * $className\n";
    $newContent .= " *\n";
    $newContent .= " * Helper class containing extracted methods from AssetData.\n";
    $newContent .= " * All methods are static for backward compatibility.\n";
    $newContent .= " */\n";
    $newContent .= "class $className\n";
    $newContent .= "{\n";
    $newContent .= "    " . trim($content) . "\n";
    $newContent .= "}\n";

    // Fix indentation - add 4 spaces to each line
    $lines = explode("\n", $newContent);
    $fixedLines = [];
    $inClass = false;

    foreach ($lines as $i => $line) {
        if (strpos($line, 'class ') === 0) {
            $inClass = true;
            $fixedLines[] = $line;
            continue;
        }

        if ($inClass && (strpos($line, 'public static function') === 0 ||
                         strpos($line, 'private static function') === 0 ||
                         strpos($line, 'protected static function') === 0)) {
            $fixedLines[] = '    ' . $line;
        } elseif ($inClass && trim($line) !== '' && trim($line) !== '}' && !preg_match('/^    \}/', $line)) {
            // Add indentation for content lines
            if (strpos($line, '/') === 0 || strpos($line, '*') === 0) {
                // Comment
                $fixedLines[] = '    ' . $line;
            } elseif (preg_match('/^\s{4}/', $line)) {
                // Already indented
                $fixedLines[] = $line;
            } elseif (trim($line) !== '') {
                $fixedLines[] = '    ' . $line;
            } else {
                $fixedLines[] = $line;
            }
        } else {
            $fixedLines[] = $line;
        }
    }

    file_put_contents($filepath, implode("\n", $fixedLines));
    echo "Fixed $file\n";
}

echo "\nAll helper files have been fixed!\n";
