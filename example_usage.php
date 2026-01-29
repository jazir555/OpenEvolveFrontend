<?php
/**
 * Example Usage of Deep Validation Tool
 *
 * This file demonstrates how to use the deep_validate.php tool
 */

// ============================================================================
// COMMAND LINE USAGE
// ============================================================================

/*
BASIC USAGE:

1. Show help:
   php deep_validate.php --help

2. Validate all helper files in current directory:
   php deep_validate.php

3. Validate specific directory:
   php deep_validate.php --dir /path/to/classes

4. Generate JSON report:
   php deep_validate.php --format json --output report.json

5. Generate Markdown report:
   php deep_validate.php --format markdown --output report.md

6. Validate and fail on issues (for CI/CD):
   php deep_validate.php || exit 1

*/

// ============================================================================
// PROGRAMMATIC USAGE
// ============================================================================

require_once __DIR__ . '/deep_validate.php';

use LHA\Tools\DeepValidator;

// Example 1: Basic validation
echo "Example 1: Basic Validation\n";
echo str_repeat('=', 80) . "\n";

$validator = new DeepValidator(__DIR__);
$issues = $validator->run();

// Example 2: Validate specific directory
echo "\nExample 2: Validate Specific Directory\n";
echo str_repeat('=', 80) . "\n";

$validator2 = new DeepValidator(__DIR__ . '/TaskHelpers');
// Note: Run the validator as needed

// Example 3: Process issues programmatically
echo "\nExample 3: Process Issues Programmatically\n";
echo str_repeat('=', 80) . "\n";

if (!empty($issues)) {
    $totalIssues = 0;
    $byType = [];

    foreach ($issues as $file => $fileIssues) {
        foreach ($fileIssues as $type => $messages) {
            if (!isset($byType[$type])) {
                $byType[$type] = 0;
            }
            $byType[$type] += count($messages);
            $totalIssues += count($messages);
        }
    }

    echo "Total Issues: {$totalIssues}\n\n";

    echo "Issues by Type:\n";
    foreach ($byType as $type => $count) {
        echo "  - {$type}: {$count}\n";
    }

    // Generate CSV report
    $csvFile = __DIR__ . '/validation_report.csv';
    $fp = fopen($csvFile, 'w');

    fputcsv($fp, ['File', 'Issue Type', 'Message']);

    foreach ($issues as $file => $fileIssues) {
        foreach ($fileIssues as $type => $messages) {
            foreach ($messages as $message) {
                fputcsv($fp, [$file, $type, $message]);
            }
        }
    }

    fclose($fp);
    echo "\nCSV report saved to: {$csvFile}\n";
} else {
    echo "No issues found!\n";
}

// ============================================================================
// AUTOMATED FIX GENERATION
// ============================================================================

echo "\n" . str_repeat('=', 80) . "\n";
echo "Example 4: Generate Fix Suggestions\n";
echo str_repeat('=', 80) . "\n";

function generateFixSuggestions(array $issues): array {
    $fixes = [];

    foreach ($issues as $file => $fileIssues) {
        foreach ($fileIssues as $type => $messages) {
            foreach ($messages as $message) {
                $fixes[] = [
                    'file' => $file,
                    'type' => $type,
                    'message' => $message,
                    'suggestion' => generateSuggestion($type, $message),
                ];
            }
        }
    }

    return $fixes;
}

function generateSuggestion(string $type, string $message): string {
    $suggestions = [
        'missing_namespace' => 'Add namespace declaration at top of file: namespace LHA\\Helpers;',
        'missing_return_type' => 'Add return type declaration to method signature',
        'missing_param_type' => 'Add type hint to parameter declaration',
        'missing_wp_guard' => 'Wrap WordPress function calls in function_exists() check',
        'property_missing_type' => 'Add type declaration to property',
        'undefined_constant' => 'Define the constant or check if it exists with defined()',
    ];

    return $suggestions[$type] ?? 'Review and fix manually';
}

if (!empty($issues)) {
    $fixSuggestions = generateFixSuggestions($issues);

    echo "\nFix Suggestions:\n\n";
    foreach (array_slice($fixSuggestions, 0, 5) as $fix) {
        echo "File: {$fix['file']}\n";
        echo "Issue: {$fix['message']}\n";
        echo "Suggestion: {$fix['suggestion']}\n";
        echo str_repeat('-', 80) . "\n\n";
    }

    if (count($fixSuggestions) > 5) {
        echo "... and " . (count($fixSuggestions) - 5) . " more issues\n";
    }
}

// ============================================================================
// CI/CD INTEGRATION EXAMPLES
// ============================================================================

echo "\n" . str_repeat('=', 80) . "\n";
echo "Example 5: CI/CD Integration Patterns\n";
echo str_repeat('=', 80) . "\n";

$ciExamples = [
    'GitHub Actions' => [
        'name' => 'PHP Validation',
        'on' => ['push', 'pull_request'],
        'jobs' => [
            'validate' => [
                'runs-on' => 'ubuntu-latest',
                'steps' => [
                    ['uses' => 'actions/checkout@v2'],
                    [
                        'name' => 'Run Deep Validation',
                        'run' => 'php classes/deep_validate.php --format json --output report.json'
                    ],
                    [
                        'name' => 'Upload Report',
                        'uses' => 'actions/upload-artifact@v2',
                        'with' => [
                            'name' => 'validation-report',
                            'path' => 'report.json'
                        ]
                    ],
                    [
                        'name' => 'Fail on Issues',
                        'run' => 'php classes/deep_validate.php || exit 1'
                    ]
                ]
            ]
        ]
    ],
    'GitLab CI' => [
        'validate' => [
            'stage' => 'test',
            'image' => 'php:8.1',
            'script' => [
                'php classes/deep_validate.php --format markdown --output report.md',
            ],
            'artifacts' => [
                'paths' => ['report.md']
            ],
            'allow_failure' => false
        ]
    ]
];

echo "\nGitHub Actions Example:\n";
echo "---\n";
echo "name: PHP Validation\n";
echo "on: [push, pull_request]\n";
echo "jobs:\n";
echo "  validate:\n";
echo "    runs-on: ubuntu-latest\n";
echo "    steps:\n";
echo "      - uses: actions/checkout@v2\n";
echo "      - name: Run Deep Validation\n";
echo "        run: php classes/deep_validate.php --format json --output report.json\n";
echo "      - name: Fail on Issues\n";
echo "        run: php classes/deep_validate.php || exit 1\n";
echo "---\n\n";

echo "GitLab CI Example:\n";
echo "---\n";
echo "validate:\n";
echo "  stage: test\n";
echo "  image: php:8.1\n";
echo "  script:\n";
echo "    - php classes/deep_validate.php\n";
echo "  allow_failure: false\n";
echo "---\n\n";

// ============================================================================
// REPORTING EXAMPLES
// ============================================================================

echo str_repeat('=', 80) . "\n";
echo "Example 6: Custom Reporting\n";
echo str_repeat('=', 80) . "\n\n";

function generateHtmlReport(array $issues, array $stats): string {
    $html = '<!DOCTYPE html>
<html>
<head>
    <title>Deep Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .stats { background: #f4f4f4; padding: 15px; border-radius: 5px; }
        .issue { margin: 10px 0; padding: 10px; border-left: 3px solid #0073aa; }
        .file { font-weight: bold; color: #0073aa; }
        .error { color: #d63638; }
        .warning { color: #dba617; }
    </style>
</head>
<body>
    <h1>Deep Validation Report</h1>
    <div class="stats">
        <h2>Statistics</h2>
        <ul>';

    foreach ($stats as $key => $value) {
        $label = ucwords(str_replace('_', ' ', $key));
        $html .= "<li>{$label}: {$value}</li>";
    }

    $html .= '</ul></div><div class="issues"><h2>Issues</h2>';

    foreach ($issues as $file => $fileIssues) {
        $html .= "<div class='issue'>";
        $html .= "<div class='file'>" . htmlspecialchars($file) . "</div>";

        foreach ($fileIssues as $type => $messages) {
            $html .= "<div class='{$type}'>";
            $html .= "<strong>" . htmlspecialchars($type) . "</strong>";
            $html .= "<ul>";

            foreach ($messages as $message) {
                $html .= "<li>" . htmlspecialchars($message) . "</li>";
            }

            $html .= "</ul></div>";
        }

        $html .= "</div>";
    }

    $html .= '</div></body></html>';

    return $html;
}

// Generate HTML report if issues exist
if (!empty($issues)) {
    $htmlReport = generateHtmlReport($issues, ['total_issues' => count($issues, COUNT_RECURSIVE)]);
    $htmlFile = __DIR__ . '/validation_report.html';

    file_put_contents($htmlFile, $htmlReport);
    echo "HTML report saved to: {$htmlFile}\n";
}

echo "\n" . str_repeat('=', 80) . "\n";
echo "Examples Complete!\n";
echo str_repeat('=', 80) . "\n";

// ============================================================================
// QUICK REFERENCE
// ============================================================================

echo "\nQuick Reference:\n";
echo str_repeat('-', 80) . "\n";
echo "Command Line:\n";
echo "  php deep_validate.php --help\n";
echo "  php deep_validate.php --format json --output report.json\n";
echo "  php deep_validate.php --format markdown --output report.md\n";
echo "\nProgrammatic:\n";
echo "  \$validator = new DeepValidator(__DIR__);\n";
echo "  \$issues = \$validator->run();\n";
echo "\nOutput Formats:\n";
echo "  - text (default, colored)\n";
echo "  - json (for automated processing)\n";
echo "  - markdown (for documentation)\n";
echo str_repeat('-', 80) . "\n";
