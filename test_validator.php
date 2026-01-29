<?php
/**
 * Quick standalone test of validator
 */

require_once __DIR__ . '/deep_validate.php';

use LHA\Tools\DeepValidator;

echo "Testing Deep Validator on Sample File\n";
echo str_repeat('=', 80) . "\n\n";

$validator = new DeepValidator(__DIR__);

// Test one known file
$testFile = __DIR__ . '/TaskHelpers/TaskValidationHelper.php';

if (!file_exists($testFile)) {
    echo "Test file not found: {$testFile}\n";
    exit(1);
}

echo "Validating: TaskHelpers/TaskValidationHelper.php\n\n";

$content = file_get_contents($testFile);
$relativePath = 'TaskHelpers/TaskValidationHelper.php';

// Manual validation call
$tokens = @token_get_all($content);
$structure = $validator->extractStructure($tokens, $content);

echo "Extracted Structure:\n";
echo "- Namespace: " . ($structure['namespace'] ?: 'MISSING') . "\n";
echo "- Classes: " . count($structure['classes']) . "\n";
echo "- Interfaces: " . count($structure['interfaces']) . "\n";

if (!empty($structure['classes'])) {
    foreach ($structure['classes'] as $class) {
        echo "\nClass: {$class['name']}\n";
        echo "  Methods: " . count($class['methods']) . "\n";
        echo "  Properties: " . count($class['properties']) . "\n";
        echo "  Constants: " . count($class['constants']) . "\n";

        if (!empty($class['methods'])) {
            echo "\n  Method Details:\n";
            foreach ($class['methods'] as $method) {
                $returnType = $method['return_type'] ?: 'MISSING';
                $params = [];
                foreach ($method['parameters'] as $param) {
                    $type = $param['type'] ?: 'no type';
                    $params[] = "{$type} {$param['name']}";
                }
                $paramList = implode(', ', $params);
                echo "    - {$method['name']}({$paramList}): {$returnType}\n";
            }
        }
    }
}

echo "\n" . str_repeat('=', 80) . "\n";
echo "Test Complete!\n";
