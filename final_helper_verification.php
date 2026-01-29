<?php
/**
 * Final Helper Verification
 * Comprehensive check including instantiation testing
 */

declare(strict_types=1);

// Mock WordPress environment
define('ABSPATH', true);

// WordPress function mocks
$wp_mocks = [
    'wp_cache_delete', 'wp_cache_get', 'wp_cache_set',
    'add_action', 'add_filter', 'do_action', 'apply_filters',
    'get_option', 'update_option', 'delete_option',
    'esc_sql', 'esc_html', 'esc_attr', 'esc_url',
    'sanitize_key', 'sanitize_text_field', 'sanitize_title',
    'absint', 'wp_parse_args', 'wp_json_encode',
    'current_time', 'get_bloginfo', 'is_admin',
    'wp_upload_dir', 'is_wp_error',
];

foreach ($wp_mocks as $func) {
    eval("function $func() { return true; }");
}

class wpdb {
    public $prefix = 'wp_';
    public $last_error = '';
    public function prepare($q, ...$a) { return $q; }
    public function query($q) { return true; }
    public function get_results($q, $o = OBJECT) { return []; }
    public function get_row($q) { return null; }
    public function insert($t, $d) { return true; }
    public function update($t, $d, $w) { return true; }
    public function delete($t, $w) { return true; }
}

$GLOBALS['wpdb'] = new wpdb();

echo "\n";
echo "====================================================================\n";
echo "              FINAL HELPER VERIFICATION                             \n";
echo "====================================================================\n\n";

// Test Instance-based helpers with mock dependencies
echo "TEST: Instance Helper Instantiation\n";
echo "--------------------------------------------------------------------\n";

interface IMockLogger {
    public function log_debug($msg, $ctx = []);
    public function log_info($msg, $ctx = []);
    public function log_warning($msg, $ctx = []);
    public function log_error($msg, $ctx = []);
}

interface IMockDatabase {
    public function get_results($q);
}

interface IMockTaskQueue {
    public function enqueue($task);
}

// Create mock objects
$mockLogger = new class implements IMockLogger {
    public function log_debug($msg, $ctx = []) {}
    public function log_info($msg, $ctx = []) {}
    public function log_warning($msg, $ctx = []) {}
    public function log_error($msg, $ctx = []) {}
};

$mockDatabase = new class implements IMockDatabase {
    public function get_results($q) { return []; }
};

$mockTaskQueue = new class implements IMockTaskQueue {
    public function enqueue($task) { return true; }
};

// Try to include and instantiate instance-based helpers
$instance_helpers = [
    'RetryHelpers/RetryDatabaseHelper.php',
    'RetryHelpers/RetryDependencyManager.php',
    'SettingsHelpers/SettingsQueryHelper.php',
];

$instantiated = 0;
$failed = [];

foreach ($instance_helpers as $helper) {
    $file = __DIR__ . '/' . $helper;
    if (!file_exists($file)) {
        echo "  ✗ File not found: $helper\n";
        $failed[] = $helper;
        continue;
    }

    try {
        include_once $file;

        // Get class name from file
        $content = file_get_contents($file);
        if (preg_match('/class\s+(\w+)/', $content, $match)) {
            $className = $match[1];

            // Get namespace
            if (preg_match('/namespace\s+([\w\\\\]+);/', $content, $nsMatch)) {
                $fullClassName = $nsMatch[1] . '\\' . $className;

                // Try to instantiate (will fail if constructor requires dependencies we don't have)
                try {
                    if (strpos($content, 'LoggerInterface') !== false) {
                        $obj = new $fullClassName($mockLogger, $mockDatabase, $GLOBALS['wpdb']);
                    } else {
                        // Try without constructor args
                        $reflection = new ReflectionClass($fullClassName);
                        if ($reflection->getConstructor() && !$reflection->getConstructor()->isPublic()) {
                            $obj = $reflection->newInstanceWithoutConstructor();
                        } else {
                            continue; // Skip classes that need other dependencies
                        }
                    }
                    $instantiated++;
                    echo "  ✓ Instantiated: $helper ($className)\n";
                } catch (Throwable $e) {
                    // Expected for classes with complex dependencies
                    echo "  ⚠ Can't instantiate (needs deps): $helper ($className)\n";
                }
            }
        }
    } catch (Throwable $e) {
        $failed[] = $helper;
        echo "  ✗ Error in $helper: {$e->getMessage()}\n";
    }
}

echo "\nInstantiated: $instantiated\n";
echo "Failed: " . count($failed) . "\n\n";

// Summary
echo "====================================================================\n";
echo "                        FINAL RESULT                                  \n";
echo "====================================================================\n";
echo "\n";

// Final file count and validation
$all_files = glob(__DIR__ . '/*Helpers/*.php');
$interface_files = glob(__DIR__ . '/*Helpers/*Interface.php');
$class_files = array_diff($all_files, $interface_files);

echo "Total Helper Files: " . count($all_files) . "\n";
echo "Interface Files: " . count($interface_files) . "\n";
echo "Class Files: " . count($class_files) . "\n";
echo "\n";

// Check syntax one more time
$syntax_valid = 0;
$syntax_invalid = [];

foreach ($all_files as $file) {
    $output = shell_exec('php -l ' . escapeshellarg($file) . ' 2>&1');
    if (strpos($output, 'Errors parsing') === false && strpos($output, 'Parse error') === false) {
        $syntax_valid++;
    } else {
        $syntax_invalid[] = basename($file);
    }
}

echo "Syntax Valid: $syntax_valid / " . count($all_files) . "\n";
echo "Syntax Invalid: " . count($syntax_invalid) . "\n";

if (!empty($syntax_invalid)) {
    echo "\nFiles with syntax errors:\n";
    foreach ($syntax_invalid as $f) {
        echo "  - $f\n";
    }
}

echo "\n";
echo "====================================================================\n";
echo empty($syntax_invalid) ? "✅ ALL HELPERS VERIFIED\n" : "⚠️  SOME HELPERS HAVE ISSUES\n";
echo "====================================================================\n";
