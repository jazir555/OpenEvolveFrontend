<?php
declare(strict_types=1);

/**
 * Scan all helper files and categorize by constructor dependencies
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

$helpers = [
    'instance_with_deps' => [],
    'instance_simple' => [],
    'static' => [],
];

foreach ($directories as $dir) {
    $files = glob(__DIR__ . '/' . $dir . '/*.php');

    foreach ($files as $file) {
        if (strpos($file, 'Interface.php') !== false) {
            continue; // Skip interface files
        }

        $content = file_get_contents($file);

        // Skip if just a namespace file without class
        if (!preg_match('/class\s+(\w+)/', $content, $classMatch)) {
            continue;
        }

        $className = $classMatch[1];
        $namespace = '';
        if (preg_match('/namespace\s+([\w\\\\]+);/', $content, $nsMatch)) {
            $namespace = $nsMatch[1];
        }
        $fullClassName = $namespace . '\\' . $className;

        // Check for constructor
        if (preg_match('/public\s+function\s+__construct\s*\(([^)]*)\)/s', $content, $constructorMatch)) {
            $params = trim($constructorMatch[1]);

            if (empty($params) || $params === '') {
                // Constructor with no parameters
                $helpers['instance_simple'][] = [
                    'class' => $fullClassName,
                    'file' => str_replace(__DIR__ . '/', '', $file),
                ];
            } else {
                // Constructor with parameters
                $helpers['instance_with_deps'][] = [
                    'class' => $fullClassName,
                    'file' => str_replace(__DIR__ . '/', '', $file),
                    'params' => $params,
                ];
            }
        } else {
            // No constructor - static class
            $helpers['static'][] = [
                'class' => $fullClassName,
                'file' => str_replace(__DIR__ . '/', '', $file),
            ];
        }
    }
}

echo "\n";
echo "====================================================================\n";
echo "              HELPER DEPENDENCY ANALYSIS                            \n";
echo "====================================================================\n\n";

echo "Instance-based helpers with dependencies: " . count($helpers['instance_with_deps']) . "\n";
echo "Instance-based helpers without dependencies: " . count($helpers['instance_simple']) . "\n";
echo "Static helpers: " . count($helpers['static']) . "\n\n";

// Show instance helpers with dependencies
if (!empty($helpers['instance_with_deps'])) {
    echo "====================================================================\n";
    echo "         INSTANCE HELPERS WITH DEPENDENCIES                        \n";
    echo "====================================================================\n\n";

    foreach ($helpers['instance_with_deps'] as $helper) {
        echo "{$helper['class']}\n";
        echo "  File: {$helper['file']}\n";
        echo "  Constructor params: {$helper['params']}\n\n";
    }
}

// Generate service registrations for instance helpers with deps
echo "\n";
echo "====================================================================\n";
echo "         SERVICE CONTAINER REGISTRATIONS                            \n";
echo "====================================================================\n\n";

foreach ($helpers['instance_with_deps'] as $helper) {
    echo "// {$helper['class']}\n";
    echo "{$helper['class']}::class => function (\$container) {\n";
    echo "    return new {$helper['class']}(\n";
    echo "        // TODO: Add dependencies\n";
    echo "    );\n";
    echo "},\n\n";
}

echo "\nTotal instance helpers with dependencies: " . count($helpers['instance_with_deps']) . "\n";
