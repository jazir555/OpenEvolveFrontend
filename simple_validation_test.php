<?php
/**
 * Simple validation test - validates just TaskHelpers directory
 */

declare(strict_types=1);

namespace LHA\Tools;

if (php_sapi_name() !== 'cli') {
    die("This script can only be run from CLI.\n");
}

class SimpleDeepValidator {
    private string $baseDir;
    private array $issues = [];
    private array $stats = [
        'files_scanned' => 0,
        'total_issues' => 0,
    ];

    private array $wordpressFunctions = [
        'wp_cache_get', 'wp_cache_set', 'wp_cache_delete',
        'get_option', 'update_option', 'delete_option',
        'esc_sql', 'esc_html', 'esc_attr', 'esc_url',
    ];

    public function __construct(string $baseDir) {
        $this->baseDir = rtrim($baseDir, '/\\');
    }

    public function run(): array {
        echo "Deep Validation Tool - Simple Test\n";
        echo str_repeat('=', 80) . "\n\n";

        // Test TaskHelpers directory
        $dir = $this->baseDir . '/TaskHelpers';
        if (!is_dir($dir)) {
            echo "TaskHelpers directory not found!\n";
            return $this->issues;
        }

        $files = glob($dir . '/*Helper.php');
        echo "Found " . count($files) . " helper files in TaskHelpers\n\n";

        foreach ($files as $file) {
            $this->validateFile($file);
        }

        $this->printReport();
        return $this->issues;
    }

    private function validateFile(string $filepath): void {
        $this->stats['files_scanned']++;
        $relativePath = str_replace($this->baseDir . '/', '', $filepath);
        echo "Validating: {$relativePath}\n";

        $content = file_get_contents($filepath);
        if ($content === false) {
            return;
        }

        // Check namespace
        if (!preg_match('/namespace\s+[\w\\\\]+;/i', $content)) {
            $this->addIssue($relativePath, 'missing_namespace', 'File is missing namespace declaration');
        }

        // Check declare strict_types
        if (!str_contains($content, 'declare(strict_types=1)')) {
            $this->addIssue($relativePath, 'missing_strict_types', 'File is missing declare(strict_types=1)');
        }

        // Check for WordPress functions without guards
        $hasWpFunctions = false;
        $hasGuard = false;

        foreach ($this->wordpressFunctions as $func) {
            if (preg_match('/\b' . $func . '\s*\(/i', $content)) {
                $hasWpFunctions = true;
            }
        }

        if ($hasWpFunctions) {
            if (preg_match('/function_exists\s*\(/i', $content) ||
                preg_match('/defined\s*\(\s*["\']ABSPATH["\']\s*\)/i', $content)) {
                $hasGuard = true;
            }

            if (!$hasGuard) {
                $this->addIssue($relativePath, 'missing_wp_guard', 'WordPress functions used without function_exists guard');
            }
        }

        // Check for missing return types
        $tokens = @token_get_all($content);
        $inFunction = false;
        $functionName = '';
        $parenCount = 0;
        $foundReturnType = false;

        for ($i = 0; $i < count($tokens); $i++) {
            if (!is_array($tokens[$i])) {
                if ($tokens[$i] === '{' && $inFunction) {
                    $parenCount++;
                } elseif ($tokens[$i] === '}' && $inFunction) {
                    $parenCount--;
                    if ($parenCount === 0) {
                        // Function ended
                        if (!empty($functionName) && !$foundReturnType && !str_starts_with($functionName, '__')) {
                            $this->addIssue($relativePath, 'missing_return_type',
                                "Function {$functionName}() is missing return type");
                        }
                        $inFunction = false;
                        $functionName = '';
                        $foundReturnType = false;
                    }
                }
                continue;
            }

            if ($tokens[$i][0] === T_FUNCTION) {
                $inFunction = true;
                $foundReturnType = false;

                // Get function name
                $j = $i + 1;
                while ($j < count($tokens)) {
                    if (is_array($tokens[$j])) {
                        if ($tokens[$j][0] === T_STRING) {
                            $functionName = $tokens[$j][1];
                            break;
                        }
                    } elseif ($tokens[$j] === '{') {
                        break;
                    }
                    $j++;
                }
            } elseif ($inFunction && in_array($tokens[$i][0], [T_STRING, T_NAME_QUALIFIED])) {
                // Check if this is a return type
                $j = $i - 1;
                while ($j >= 0 && $j >= $i - 5) {
                    if (!is_array($tokens[$j]) && $tokens[$j] === ')') {
                        // Found closing paren, this might be return type
                        $k = $i + 1;
                        while ($k < count($tokens) && $k < $i + 3) {
                            if (!is_array($tokens[$k]) && ($tokens[$k] === '{' || $tokens[$k] === ';')) {
                                $foundReturnType = true;
                                break;
                            }
                            $k++;
                        }
                        break;
                    }
                    $j--;
                }
            }
        }
    }

    private function addIssue(string $file, string $type, string $message): void {
        if (!isset($this->issues[$file])) {
            $this->issues[$file] = [];
        }
        if (!isset($this->issues[$file][$type])) {
            $this->issues[$file][$type] = [];
        }
        $this->issues[$file][$type][] = $message;
        $this->stats['total_issues']++;
        echo "  ⚠ {$message}\n";
    }

    private function printReport(): void {
        echo "\n" . str_repeat('=', 80) . "\n";
        echo "SUMMARY\n";
        echo str_repeat('=', 80) . "\n\n";

        echo "Files Scanned: {$this->stats['files_scanned']}\n";
        echo "Total Issues: {$this->stats['total_issues']}\n\n";

        if (!empty($this->issues)) {
            foreach ($this->issues as $file => $issues) {
                echo "\n{$file}:\n";
                foreach ($issues as $type => $messages) {
                    foreach ($messages as $msg) {
                        echo "  [{$type}] {$msg}\n";
                    }
                }
            }
        } else {
            echo "No issues found!\n";
        }

        echo "\n" . str_repeat('=', 80) . "\n";
    }
}

// Run the test
$validator = new SimpleDeepValidator(__DIR__);
$validator->run();
