<?php
declare(strict_types=1);

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
$stats = [
    'files' => 0,
    'classes' => 0,
    'methods' => 0,
    'no_return_type' => 0,
    'no_param_type' => 0,
    'missing_namespace' => 0,
];

echo "\n=================================================================\n";
echo "           THOROUGH HELPER VALIDATION\n";
echo "=================================================================\n\n";

foreach ($directories as $dir) {
    $files = glob(__DIR__ . '/' . $dir . '/*.php');
    
    foreach ($files as $file) {
        $stats['files']++;
        $content = file_get_contents($file);
        $filename = basename($file);
        $relative = "$dir/$filename";
        
        // Check namespace
        if (!preg_match('/namespace\s+LHA\\/', $content)) {
            $issues[] = ['ERROR', 'Missing Namespace', $relative, 'No LHA namespace found'];
            $stats['missing_namespace']++;
        }
        
        // Find class
        if (preg_match('/class\s+(\w+)/', $content, $match)) {
            $stats['classes']++;
            
            // Find all methods
            preg_match_all('/^\s*(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*\(([^)]*)\)(\s*:\s*([\w\|\\]+))?/m', $content, $methods, PREG_SET_ORDER);
            
            foreach ($methods as $m) {
                $stats['methods']++;
                $methodName = $m[3];
                $hasReturn = !empty($m[6]);
                $params = $m[4];
                
                // Skip magic methods
                if (strpos($methodName, '__') === 0) continue;
                
                // Check return type
                if (!$hasReturn && strpos($content, 'declare(strict_types=1)') !== false) {
                    $stats['no_return_type']++;
                    $issues[] = ['WARNING', 'Missing Return Type', $relative, "Method $methodName() missing return type"];
                }
                
                // Check parameter types
                if (!empty(trim($params))) {
                    $paramList = explode(',', $params);
                    foreach ($paramList as $p) {
                        $p = trim($p);
                        if (preg_match('/\$(\w+)/', $p, $pm)) {
                            $paramName = $pm[1];
                            // Check if has type hint
                            if (!preg_match('/^[^$]+\s+\$/', $p)) {
                                $stats['no_param_type']++;
                            }
                        }
                    }
                }
            }
            
            // Check constants
            preg_match_all('/self::([A-Z_][A-Z0-9_]*)/', $content, $refs);
            $defined = [];
            preg_match_all('/(private|public|protected|const)\s+const\s+([A-Z_][A-Z0-9_]*)/', $content, $consts);
            foreach ($consts[2] ?? [] as $c) {
                $defined[$c] = true;
            }
            
            foreach ($refs[1] ?? [] as $ref) {
                if (!isset($defined[$ref]) && !in_array($ref, ['PHP_VERSION', 'PHP_MAJOR_VERSION', 'PHP_MINOR_VERSION', 'PHP_RELEASE_VERSION'])) {
                    $issues[] = ['WARNING', 'Undefined Constant', $relative, "References self::$ref which may not be defined"];
                }
            }
        }
    }
}

echo "Statistics:\n";
echo "-----------------------------------------------------------------\n";
echo "Files scanned: {$stats['files']}\n";
echo "Classes found: {$stats['classes']}\n";
echo "Methods found: {$stats['methods']}\n";
echo "Missing return types: {$stats['no_return_type']}\n";
echo "Missing namespaces: {$stats['missing_namespace']}\n";
echo "\n";

// Group by severity
$bySev = ['ERROR' => [], 'WARNING' => []];
foreach ($issues as $i) {
    $bySev[$i[0]][] = $i;
}

echo "Issues Summary:\n";
echo "-----------------------------------------------------------------\n";
echo "Errors: " . count($bySev['ERROR']) . "\n";
echo "Warnings: " . count($bySev['WARNING']) . "\n";
echo "Total: " . count($issues) . "\n\n";

if (!empty($bySev['ERROR'])) {
    echo "ERRORS:\n";
    echo "-----------------------------------------------------------------\n";
    foreach ($bySev['ERROR'] as $i) {
        echo "[{$i[2]}] {$i[1]}: {$i[3]}\n";
    }
    echo "\n";
}

if (!empty($bySev['WARNING'])) {
    echo "WARNINGS (showing first 30):\n";
    echo "-----------------------------------------------------------------\n";
    $shown = 0;
    foreach ($bySev['WARNING'] as $i) {
        if ($shown++ >= 30) break;
        echo "[{$i[2]}] {$i[1]}: {$i[3]}\n";
    }
    if (count($bySev['WARNING']) > 30) {
        echo "... and " . (count($bySev['WARNING']) - 30) . " more warnings\n";
    }
    echo "\n";
}

echo "=================================================================\n";
echo count($bySev['ERROR']) > 0 ? "❌ VALIDATION FAILED\n" : "✅ VALIDATION PASSED\n";
echo "=================================================================\n";
