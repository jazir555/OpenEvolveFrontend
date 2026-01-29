<?php
/**
 * Automated Type Hints Fixer for Helper Files
 *
 * This script scans all helper files and adds missing return types based on:
 * 1. Method naming conventions
 * 2. Existing return statements
 * 3. Common patterns
 */

declare(strict_types=1);

class HelperTypesFixer {
    private array $stats = [
        'files_scanned' => 0,
        'files_modified' => 0,
        'methods_fixed' => 0,
        'errors' => [],
    ];

    private array $patterns = [
        'void' => [
            '/^(set_|delete_|clear_|remove_|add_|update_|save_|load_|handle_|process_|execute_|run_|invalidate_|log_|write_|enqueue_|dequeue_|trigger_|start_|stop_|begin_|end_|init_|ajax_|wp_|action_)/i',
            '/^(bulk_|batch_|mass_)/i',
        ],
        'bool' => [
            '/^(is_|has_|can_|should_|will_|validate_|check_|verify_|exists_|contains_|enabled_|disabled_|active_|inactive_|valid_|invalid_)/i',
        ],
        'array' => [
            '/^(get_|fetch_|retrieve_|list_|all_|find_|search_|query_|load_)/i',
        ],
        'int' => [
            '/^(count_|total_|sum_|calculate_|compute_|size_|length|id_|key_|index_)/i',
        ],
        'string' => [
            '/^(render_|format_|escape_|sanitize_|prepare_|build_|create_|generate_|produce_|make_)/i',
        ],
    ];

    public function fixDirectory(string $dir): void {
        if (!is_dir($dir)) {
            return;
        }

        $files = glob($dir . '/*.php');
        foreach ($files as $file) {
            $this->fixFile($file);
        }
    }

    public function fixFile(string $file): void {
        $this->stats['files_scanned']++;
        $content = file_get_contents($file);

        if ($content === false) {
            $this->stats['errors'][] = "Could not read: $file";
            return;
        }

        // Skip if already strict types declared
        if (strpos($content, 'declare(strict_types=1)') === false) {
            return;
        }

        $original = $content;
        $fixes = 0;

        // Fix corrupted catch blocks (duplicate catch blocks)
        $content = $this->fixCorruptedCatchBlocks($content);

        // Find methods without return types
        $content = preg_replace_callback(
            '/^\s*(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?!:\s*(?:void|int|float|string|bool|array|object|\?[^{]+))\s*\{/m',
            function ($matches) use (&$fixes) {
                $visibility = $matches[1];
                $static = $matches[2] ?? '';
                $methodName = $matches[3];
                $params = $matches[4];

                $returnType = $this->determineReturnType($methodName, $params);

                if ($returnType) {
                    $fixes++;
                    return "\t" . $visibility . ' ' . $static . "function $methodName($params): $returnType {";
                }

                return $matches[0];
            },
            $content
        );

        if ($content !== $original) {
            if (file_put_contents($file, $content)) {
                $this->stats['files_modified']++;
                $this->stats['methods_fixed'] += $fixes;
                echo "Fixed: " . basename($file) . " ($fixes methods)\n";
            } else {
                $this->stats['errors'][] = "Could not write: $file";
            }
        }
    }

    private function determineReturnType(string $methodName, string $params): ?string {
        // Check patterns
        foreach ($this->patterns as $type => $patterns) {
            foreach ($patterns as $pattern) {
                if (preg_match($pattern, $methodName)) {
                    return $type;
                }
            }
        }

        // Check for return statements in the method (heuristic)
        // Methods returning 'void' typically don't have meaningful return statements
        // This is a simple heuristic - not perfect

        return null; // Can't determine
    }

    private function fixCorruptedCatchBlocks(string $content): string {
        // Fix duplicate catch blocks like:
        // } catch (\InvalidArgumentException $e) {
        //     return $this->getFallbackValue();
        // } catch (\RuntimeException $e) {
        //     throw $e;
        // } catch (\InvalidArgumentException $e) {
        //     return $this->getFallbackValue();
        // }

        // Remove the duplicate blocks
        $pattern = '/\} catch \(\InvalidArgumentException \$e\) \{\s*return \$this->getFallbackValue\(\);\s*\} catch \(\RuntimeException \$e\) \{\s*throw \$e;\s*\} \}/s';

        return preg_replace($pattern, '}', $content);
    }

    public function getStats(): array {
        return $this->stats;
    }

    public function printStats(): void {
        echo "\n=== Statistics ===\n";
        echo "Files scanned: {$this->stats['files_scanned']}\n";
        echo "Files modified: {$this->stats['files_modified']}\n";
        echo "Methods fixed: {$this->stats['methods_fixed']}\n";
        echo "Errors: " . count($this->stats['errors']) . "\n";

        if (!empty($this->stats['errors'])) {
            echo "\nErrors:\n";
            foreach ($this->stats['errors'] as $error) {
                echo "  - $error\n";
            }
        }
    }
}

// Main execution
$fixer = new HelperTypesFixer();

$directories = [
    __DIR__ . '/classes/AjaxHelpers',
    __DIR__ . '/classes/ExtractHelpers',
    __DIR__ . '/classes/LoggingHelpers',
    __DIR__ . '/classes/RetryHelpers',
    __DIR__ . '/classes/SanitizeHelpers',
    __DIR__ . '/classes/SettingsHelpers',
    __DIR__ . '/classes/AssetOrderHelpers',
    __DIR__ . '/classes/AssetDataHelpers',
    __DIR__ . '/classes/CleanupHelpers',
    __DIR__ . '/classes/ProcessHelpers',
    __DIR__ . '/classes/DatabaseHelpers',
    __DIR__ . '/classes/TaskHelpers',
];

echo "=== Fixing Type Hints in Helper Files ===\n\n";

foreach ($directories as $dir) {
    if (is_dir($dir)) {
        echo "\nProcessing " . basename($dir) . "...\n";
        $fixer->fixDirectory($dir);
    }
}

$fixer->printStats();
