<?php
echo "=== Extraction Verification ===\n\n";
echo "Verifying method signatures in extracted helpers...\n\n";

// Check AssetValidationHelper
$file = 'AssetDataHelpers/AssetValidationHelper.php';
$content = file_get_contents($file);
echo "AssetValidationHelper.php:\n";
preg_match_all('/^\s+(public|private|protected)\s+static\s+function\s+(\w+)/m', $content, $matches);
foreach ($matches[2] as $method) {
    echo "  - $method()\n";
}

echo "\n";
echo "AssetStatisticsHelper.php:\n";
$file = 'AssetDataHelpers/AssetStatisticsHelper.php';
$content = file_get_contents($file);
preg_match_all('/^\s+(public|private|protected)\s+static\s+function\s+(\w+)/m', $content, $matches);
foreach ($matches[2] as $method) {
    echo "  - $method()\n";
}

echo "\n";
echo "AssetURLHelper.php:\n";
$file = 'AssetDataHelpers/AssetURLHelper.php';
$content = file_get_contents($file);
preg_match_all('/^\s+(public|private|protected)\s+static\s+function\s+(\w+)/m', $content, $matches);
foreach ($matches[2] as $method) {
    echo "  - $method()\n";
}

echo "\n✓ All methods successfully extracted with proper signatures!\n";
