<?php
/**
 * Deep Validation Tool for Helper Classes
 *
 * A standalone PHP CLI tool that scans all helper files and validates:
 * - Missing namespaces
 * - Missing return types
 * - Missing parameter type hints
 * - Undefined constant references
 * - Missing interface methods
 * - WordPress function guards
 *
 * Usage: php deep_validate.php [--output=file] [--format=text|json|markdown]
 */

declare(strict_types=1);

namespace LHA\Tools;

// prevent direct access
if (php_sapi_name() !== 'cli') {
    die("This script can only be run from CLI.\n");
}

class DeepValidator {
    private string $baseDir;
    private array $issues = [];
    private array $stats = [
        'files_scanned' => 0,
        'classes_found' => 0,
        'interfaces_found' => 0,
        'methods_found' => 0,
        'total_issues' => 0,
    ];
    private array $wordpressFunctions = [
        'wp_cache_get', 'wp_cache_set', 'wp_cache_delete',
        'get_option', 'update_option', 'delete_option',
        'get_site_option', 'update_site_option', 'delete_site_option',
        'add_action', 'add_filter', 'do_action', 'apply_filters',
        'esc_sql', 'esc_html', 'esc_attr', 'esc_url',
        'sanitize_text_field', 'sanitize_key', 'sanitize_title',
        'wpdb', 'wp_normalize_path',
        'wp_is_numeric', 'wp_json_encode',
        'wp_mail', 'wp_get_current_user',
        'current_time', 'mysql2date',
        'get_bloginfo', 'get_site_url', 'home_url',
        'admin_url', 'includes_url', 'content_url',
        'plugins_url', 'rest_url',
        'is_admin', 'is_multisite', 'is_user_logged_in',
        'current_user_can', 'user_can',
        'check_ajax_referer', 'wp_verify_nonce',
        'wp_create_nonce',
    ];

    private array $typeIssueCategories = [
        'missing_namespace' => 'Missing Namespace',
        'missing_return_type' => 'Missing Return Type',
        'missing_param_type' => 'Missing Parameter Type',
        'undefined_constant' => 'Undefined Constant',
        'missing_interface_method' => 'Missing Interface Method',
        'missing_wp_guard' => 'Missing WordPress Function Guard',
        'property_missing_type' => 'Property Missing Type',
        'constant_missing_type' => 'Constant Missing Type',
    ];

    public function __construct(string $baseDir) {
        $this->baseDir = rtrim($baseDir, '/\\');
    }

    /**
     * Main validation runner
     */
    public function run(): array {
        $this->printHeader();

        // Find all helper files
        $helperFiles = $this->findHelperFiles();

        if (empty($helperFiles)) {
            $this->output("No helper files found in: {$this->baseDir}", 'warning');
            return $this->issues;
        }

        $this->output("Found " . count($helperFiles) . " helper files to validate\n", 'info');

        // Scan each file
        foreach ($helperFiles as $file) {
            $this->validateFile($file);
        }

        // Generate report
        $this->printReport();

        return $this->issues;
    }

    /**
     * Find all helper files recursively
     */
    private function findHelperFiles(): array {
        $files = [];
        $pattern = $this->baseDir . '/**/*Helper.php';

        // Use RecursiveDirectoryIterator for better performance
        $directory = new \RecursiveDirectoryIterator($this->baseDir, \RecursiveDirectoryIterator::SKIP_DOTS);
        $iterator = new \RecursiveIteratorIterator($directory);

        foreach ($iterator as $file) {
            if ($file->isFile() && $file->getExtension() === 'php') {
                $filename = $file->getFilename();
                // Match helper files (ending in Helper.php or containing Helper)
                if (str_ends_with($filename, 'Helper.php') ||
                    str_contains($filename, 'Helper')) {
                    $files[] = $file->getPathname();
                }
            }
        }

        sort($files);
        return $files;
    }

    /**
     * Validate a single PHP file
     */
    private function validateFile(string $filepath): void {
        $this->stats['files_scanned']++;
        $relativePath = str_replace($this->baseDir . '/', '', $filepath);

        $content = file_get_contents($filepath);
        if ($content === false) {
            $this->addIssue($relativePath, 'file_read_error', 'Could not read file');
            return;
        }

        // Parse the file
        $tokens = @token_get_all($content);
        if (empty($tokens)) {
            $this->addIssue($relativePath, 'parse_error', 'Could not parse PHP file');
            return;
        }

        // Extract structure
        $structure = $this->extractStructure($tokens, $content);

        // Store structure for interface checking
        $this->currentStructure = $structure;
        $this->currentContent = $content;

        // Validate namespace
        if (empty($structure['namespace'])) {
            $this->addIssue($relativePath, 'missing_namespace', 'File is missing namespace declaration');
        }

        // Check each class/interface
        foreach ($structure['classes'] as $class) {
            $this->stats['classes_found']++;
            $this->validateClass($relativePath, $class, $content);
        }

        foreach ($structure['interfaces'] as $interface) {
            $this->stats['interfaces_found']++;
            $this->validateInterface($relativePath, $interface);
        }

        // Check for WordPress function usage without guards
        $this->validateWordPressGuards($relativePath, $content, $structure['namespace']);
    }

    /**
     * Extract class/interface structure from tokens
     */
    private function extractStructure(array $tokens, string $content): array {
        $structure = [
            'namespace' => '',
            'uses' => [],
            'classes' => [],
            'interfaces' => [],
            'constants' => [],
        ];

        $count = count($tokens);
        $i = 0;

        while ($i < $count) {
            $token = $tokens[$i];

            if (is_array($token)) {
                switch ($token[0]) {
                    case T_NAMESPACE:
                        $namespace = '';
                        $j = $i + 1;
                        while ($j < $count) {
                            if (is_array($tokens[$j])) {
                                if ($tokens[$j][0] === T_STRING || $tokens[$j][0] === T_NAME_QUALIFIED) {
                                    $namespace .= $tokens[$j][1];
                                } elseif ($tokens[$j][0] === T_NS_SEPARATOR) {
                                    $namespace .= '\\';
                                } elseif ($tokens[$j][0] === T_WHITESPACE) {
                                    $namespace .= ' ';
                                } elseif ($tokens[$j][0] === T_SEMICOLON) {
                                    break;
                                }
                            } else {
                                if ($tokens[$j] === '{') {
                                    // Namespace with braces
                                    $j++;
                                    while ($j < $count && (is_array($tokens[$j]) || $tokens[$j] !== '}')) {
                                        $j++;
                                    }
                                }
                                break;
                            }
                            $j++;
                        }
                        $structure['namespace'] = trim($namespace);
                        $i = $j;
                        break;

                    case T_USE:
                        $useStatement = '';
                        $j = $i + 1;
                        while ($j < $count) {
                            if (is_array($tokens[$j])) {
                                if ($tokens[$j][0] === T_STRING || $tokens[$j][0] === T_NAME_QUALIFIED) {
                                    $useStatement .= $tokens[$j][1];
                                } elseif ($tokens[$j][0] === T_NS_SEPARATOR) {
                                    $useStatement .= '\\';
                                } elseif ($tokens[$j][0] === T_AS) {
                                    // Handle aliases
                                    $j++;
                                    if ($j < $count && is_array($tokens[$j]) && $tokens[$j][0] === T_STRING) {
                                        $alias = $tokens[$j][1];
                                        $structure['uses'][$alias] = trim($useStatement);
                                        $useStatement = '';
                                    }
                                } elseif ($tokens[$j][0] === T_WHITESPACE) {
                                    $useStatement .= ' ';
                                } elseif ($tokens[$j][0] === T_SEMICOLON) {
                                    if (!empty($useStatement)) {
                                        $parts = explode('\\', trim($useStatement));
                                        $alias = end($parts);
                                        $structure['uses'][$alias] = trim($useStatement);
                                    }
                                    break;
                                }
                            } else {
                                if ($tokens[$j] === ',') {
                                    if (!empty($useStatement)) {
                                        $parts = explode('\\', trim($useStatement));
                                        $alias = end($parts);
                                        $structure['uses'][$alias] = trim($useStatement);
                                    }
                                    $useStatement = '';
                                } elseif ($tokens[$j] === ';') {
                                    if (!empty($useStatement)) {
                                        $parts = explode('\\', trim($useStatement));
                                        $alias = end($parts);
                                        $structure['uses'][$alias] = trim($useStatement);
                                    }
                                    break;
                                }
                            }
                            $j++;
                        }
                        $i = $j;
                        break;

                    case T_CLASS:
                    case T_INTERFACE:
                        $type = $token[0] === T_CLASS ? 'class' : 'interface';
                        $i++;

                        // Skip abstract and final keywords
                        while ($i < $count && is_array($tokens[$i])) {
                            if ($tokens[$i][0] === T_ABSTRACT || $tokens[$i][0] === T_FINAL) {
                                $i++;
                            } else {
                                break;
                            }
                        }

                        // Get class/interface name
                        $name = '';
                        while ($i < $count && is_array($tokens[$i])) {
                            if ($tokens[$i][0] === T_STRING) {
                                $name = $tokens[$i][1];
                                $i++;
                                break;
                            }
                            $i++;
                        }

                        // Check for extends/implements
                        $extends = '';
                        $implements = [];
                        while ($i < $count) {
                            if (is_array($tokens[$i])) {
                                if ($tokens[$i][0] === T_EXTENDS) {
                                    $i++;
                                    $extends = $this->extractQualifiedName($tokens, $i);
                                } elseif ($tokens[$i][0] === T_IMPLEMENTS) {
                                    $i++;
                                    while ($i < $count) {
                                        if (is_array($tokens[$i])) {
                                            if ($tokens[$i][0] === T_STRING || $tokens[$i][0] === T_NAME_QUALIFIED) {
                                                $implements[] = $tokens[$i][1];
                                            } elseif ($tokens[$i][0] === T_WHITESPACE) {
                                                // skip
                                            } else {
                                                break;
                                            }
                                        } else {
                                            if ($tokens[$i] === ',') {
                                                // continue
                                            } elseif ($tokens[$i] === '{') {
                                                $i--;
                                                break;
                                            } else {
                                                break;
                                            }
                                        }
                                        $i++;
                                    }
                                } elseif ($tokens[$i][0] === T_WHITESPACE) {
                                    // skip
                                } else {
                                    break;
                                }
                            } else {
                                if ($tokens[$i] === '{') {
                                    break;
                                }
                                $i++;
                            }
                        }

                        // Extract body
                        $body = '';
                        $braceCount = 0;
                        $inBody = false;

                        while ($i < $count) {
                            if (!is_array($tokens[$i])) {
                                if ($tokens[$i] === '{') {
                                    $inBody = true;
                                    $braceCount++;
                                } elseif ($tokens[$i] === '}') {
                                    $braceCount--;
                                    if ($braceCount === 0 && $inBody) {
                                        break;
                                    }
                                }
                            }

                            if ($inBody) {
                                $body .= is_array($tokens[$i]) ? $tokens[$i][1] : $tokens[$i];
                            }
                            $i++;
                        }

                        $item = [
                            'name' => $name,
                            'type' => $type,
                            'extends' => $extends,
                            'implements' => $implements,
                            'methods' => $this->extractMethods($body),
                            'properties' => $this->extractProperties($body),
                            'constants' => $this->extractClassConstants($body),
                        ];

                        if ($type === 'class') {
                            $structure['classes'][] = $item;
                        } else {
                            $structure['interfaces'][] = $item;
                        }

                        break;

                    default:
                        $i++;
                        break;
                }
            } else {
                $i++;
            }
        }

        return $structure;
    }

    /**
     * Extract qualified class name from tokens
     */
    private function extractQualifiedName(array $tokens, int &$i): string {
        $name = '';
        while ($i < count($tokens)) {
            if (is_array($tokens[$i])) {
                if ($tokens[$i][0] === T_STRING || $tokens[$i][0] === T_NAME_QUALIFIED) {
                    $name .= $tokens[$i][1];
                } elseif ($tokens[$i][0] === T_NS_SEPARATOR) {
                    $name .= '\\';
                } elseif ($tokens[$i][0] === T_WHITESPACE) {
                    // skip
                } else {
                    break;
                }
            } else {
                if ($tokens[$i] === '{') {
                    $i--;
                    break;
                }
                break;
            }
            $i++;
        }
        return $name;
    }

    /**
     * Extract methods from class body
     */
    private function extractMethods(string $body): array {
        $methods = [];
        $tokens = @token_get_all('<?php ' . $body);

        $count = count($tokens);
        $i = 0; // Skip <?php

        while ($i < $count) {
            if (is_array($tokens[$i])) {
                // Look for function keyword
                if ($tokens[$i][0] === T_FUNCTION) {
                    $i++;

                    $method = [
                        'name' => '',
                        'visibility' => 'public',
                        'static' => false,
                        'abstract' => false,
                        'final' => false,
                        'return_type' => '',
                        'parameters' => [],
                    ];

                    // Check for final/abstract (before function)
                    // Already handled by T_FUNCTION position

                    // Check for reference return
                    if ($i < $count && !is_array($tokens[$i]) && $tokens[$i] === '&') {
                        $method['by_ref'] = true;
                        $i++;
                    }

                    // Get method name
                    while ($i < $count && is_array($tokens[$i])) {
                        if ($tokens[$i][0] === T_STRING) {
                            $method['name'] = $tokens[$i][1];
                            $i++;
                            break;
                        }
                        $i++;
                    }

                    // Skip to parameters
                    while ($i < $count && (!is_array($tokens[$i]) || $tokens[$i][0] !== T_WHITESPACE)) {
                        $i++;
                    }

                    // Get parameters
                    if ($i < $count && !is_array($tokens[$i]) && $tokens[$i] === '(') {
                        $method['parameters'] = $this->extractParameters($tokens, $i);
                    }

                    // Get return type
                    while ($i < $count) {
                        if (is_array($tokens[$i])) {
                            if ($tokens[$i][0] === T_WHITESPACE) {
                                // skip
                            } elseif ($tokens[$i][0] === T_STRING || $tokens[$i][0] === T_NAME_QUALIFIED) {
                                $method['return_type'] = $tokens[$i][1];
                            } elseif (in_array($tokens[$i][0], [T_PUBLIC, T_PRIVATE, T_PROTECTED, T_STATIC, T_ABSTRACT, T_FINAL])) {
                                break; // Next method
                            } elseif ($tokens[$i][0] === T_FUNCTION) {
                                break; // Next method
                            } elseif ($tokens[$i][0] === T_VARIABLE) {
                                break; // Property
                            } elseif ($tokens[$i][0] === T_CONST) {
                                break; // Constant
                            } elseif ($tokens[$i][0] === T_USE) {
                                break; // Trait use
                            } else {
                                // Found something else
                            }
                        } else {
                            if ($tokens[$i] === '{' || $tokens[$i] === ';') {
                                break;
                            }
                        }
                        $i++;
                    }

                    if (!empty($method['name'])) {
                        $methods[] = $method;
                        $this->stats['methods_found']++;
                    }
                } else {
                    $i++;
                }
            } else {
                $i++;
            }
        }

        return $methods;
    }

    /**
     * Extract parameters from function declaration
     */
    private function extractParameters(array $tokens, int &$i): array {
        $params = [];
        $parenCount = 1;
        $i++; // Skip opening paren

        while ($i < count($tokens) && $parenCount > 0) {
            if (!is_array($tokens[$i])) {
                if ($tokens[$i] === '(') {
                    $parenCount++;
                } elseif ($tokens[$i] === ')') {
                    $parenCount--;
                    if ($parenCount === 0) {
                        $i++;
                        break;
                    }
                } elseif ($tokens[$i] === ',') {
                    // Next parameter
                }
                $i++;
                continue;
            }

            $param = [
                'name' => '',
                'type' => '',
                'default' => '',
                'variadic' => false,
                'by_ref' => false,
            ];

            // Check for variadic or by reference
            if ($tokens[$i][0] === T_ELLIPSIS) {
                $param['variadic'] = true;
                $i++;
            } elseif (!is_array($tokens[$i]) && $tokens[$i] === '&') {
                $param['by_ref'] = true;
                $i++;
            }

            // Get type
            while ($i < count($tokens) && is_array($tokens[$i])) {
                if (in_array($tokens[$i][0], [T_STRING, T_NAME_QUALIFIED, T_ARRAY, T_CALLABLE])) {
                    $param['type'] .= $tokens[$i][1];
                } elseif ($tokens[$i][0] === T_NS_SEPARATOR) {
                    $param['type'] .= '\\';
                } elseif ($tokens[$i][0] === T_VARIABLE) {
                    $param['name'] = $tokens[$i][1];
                    $i++;
                    break;
                } else {
                    break;
                }
                $i++;
            }

            // Skip to next parameter or end
            while ($i < count($tokens)) {
                if (!is_array($tokens[$i])) {
                    if ($tokens[$i] === ',') {
                        $params[] = $param;
                        $i++;
                        break;
                    } elseif ($tokens[$i] === ')') {
                        $params[] = $param;
                        $parenCount--;
                        if ($parenCount === 0) {
                            $i++;
                            break 2;
                        }
                    } elseif ($tokens[$i] === '=') {
                        // Skip default value
                        $i++;
                        while ($i < count($tokens)) {
                            if (!is_array($tokens[$i])) {
                                if ($tokens[$i] === ',' || $tokens[$i] === ')') {
                                    break;
                                }
                            }
                            $i++;
                        }
                    } else {
                        $i++;
                    }
                } else {
                    $i++;
                }
            }
        }

        return $params;
    }

    /**
     * Extract properties from class body
     */
    private function extractProperties(string $body): array {
        $properties = [];
        $tokens = @token_get_all('<?php ' . $body);

        $count = count($tokens);
        $i = 0; // Skip <?php

        while ($i < $count) {
            if (is_array($tokens[$i])) {
                if (in_array($tokens[$i][0], [T_PUBLIC, T_PRIVATE, T_PROTECTED, T_VAR])) {
                    $visibility = $tokens[$i][1];
                    $i++;

                    // Check for static
                    $static = false;
                    while ($i < $count && is_array($tokens[$i])) {
                        if ($tokens[$i][0] === T_STATIC) {
                            $static = true;
                            $i++;
                        } else {
                            break;
                        }
                    }

                    // Check for readonly (PHP 8.2+)
                    $readonly = false;
                    while ($i < $count && is_array($tokens[$i])) {
                        if ($tokens[$i][0] === T_READONLY) {
                            $readonly = true;
                            $i++;
                        } else {
                            break;
                        }
                    }

                    // Get type
                    $type = '';
                    while ($i < $count && is_array($tokens[$i])) {
                        if (in_array($tokens[$i][0], [T_STRING, T_NAME_QUALIFIED, T_ARRAY, T_CALLABLE])) {
                            $type .= $tokens[$i][1];
                        } elseif ($tokens[$i][0] === T_NS_SEPARATOR) {
                            $type .= '\\';
                        } elseif ($tokens[$i][0] === T_VARIABLE) {
                            $prop = [
                                'name' => $tokens[$i][1],
                                'visibility' => $visibility,
                                'static' => $static,
                                'readonly' => $readonly,
                                'type' => $type,
                            ];
                            $properties[] = $prop;
                            $i++;
                            break;
                        } else {
                            break;
                        }
                        $i++;
                    }
                } else {
                    $i++;
                }
            } else {
                $i++;
            }
        }

        return $properties;
    }

    /**
     * Extract class constants
     */
    private function extractClassConstants(string $body): array {
        $constants = [];
        $tokens = @token_get_all('<?php ' . $body);

        $count = count($tokens);
        $i = 0; // Skip <?php

        while ($i < $count) {
            if (is_array($tokens[$i])) {
                if ($tokens[$i][0] === T_CONST) {
                    $i++;

                    // Get constant name
                    $name = '';
                    while ($i < $count && is_array($tokens[$i])) {
                        if ($tokens[$i][0] === T_STRING) {
                            $name = $tokens[$i][1];
                            $i++;
                            break;
                        }
                        $i++;
                    }

                    // Skip value
                    while ($i < $count) {
                        if (!is_array($tokens[$i]) && $tokens[$i] === ';') {
                            $i++;
                            break;
                        }
                        $i++;
                    }

                    if (!empty($name)) {
                        $constants[] = ['name' => $name];
                    }
                } else {
                    $i++;
                }
            } else {
                $i++;
            }
        }

        return $constants;
    }

    /**
     * Validate a class
     */
    private function validateClass(string $filepath, array $class, string $content): void {
        // Check for missing property types
        foreach ($class['properties'] as $prop) {
            if (empty($prop['type']) && !$prop['readonly']) {
                $this->addIssue($filepath, 'property_missing_type',
                    "Property {$class['name']}::\${$prop['name']} is missing type declaration");
            }
        }

        // Check methods
        foreach ($class['methods'] as $method) {
            // Check return type (skip __construct, __destruct, __clone, etc.)
            if (empty($method['return_type']) && !str_starts_with($method['name'], '__')) {
                $this->addIssue($filepath, 'missing_return_type',
                    "Method {$class['name']}::{$method['name']}() is missing return type declaration");
            }

            // Check parameter types
            foreach ($method['parameters'] as $param) {
                if (empty($param['type']) && !$param['variadic']) {
                    $this->addIssue($filepath, 'missing_param_type',
                        "Parameter \${$param['name']} in {$class['name']}::{$method['name']}() is missing type hint");
                }
            }
        }

        // Check interface implementation
        if (!empty($class['implements'])) {
            foreach ($class['implements'] as $interfaceName) {
                $this->checkInterfaceImplementation($filepath, $class, $interfaceName);
            }
        }

        // Check for undefined constant references
        $this->checkUndefinedConstants($filepath, $content);
    }

    /**
     * Validate an interface
     */
    private function validateInterface(string $filepath, array $interface): void {
        // Interface methods should always have return types
        foreach ($interface['methods'] as $method) {
            if (empty($method['return_type']) && !str_starts_with($method['name'], '__')) {
                $this->addIssue($filepath, 'missing_return_type',
                    "Interface method {$interface['name']}::{$method['name']}() is missing return type declaration");
            }

            // Check parameter types
            foreach ($method['parameters'] as $param) {
                if (empty($param['type']) && !$param['variadic']) {
                    $this->addIssue($filepath, 'missing_param_type',
                        "Parameter \${$param['name']} in {$interface['name']}::{$method['name']}() is missing type hint");
                }
            }
        }
    }

    /**
     * Check if class properly implements interface
     */
    private function checkInterfaceImplementation(string $filepath, array $class, string $interfaceName): void {
        // Find the interface file
        $interfaceFile = $this->findInterfaceFile($interfaceName);

        if (!$interfaceFile) {
            $this->addIssue($filepath, 'missing_interface_method',
                "Could not find interface file for {$interfaceName}");
            return;
        }

        // Load interface methods
        $interfaceMethods = $this->getInterfaceMethods($interfaceFile);

        if (empty($interfaceMethods)) {
            return;
        }

        // Check if all interface methods are implemented
        $classMethodNames = array_column($class['methods'], 'name');

        foreach ($interfaceMethods as $method) {
            if (!in_array($method['name'], $classMethodNames)) {
                $this->addIssue($filepath, 'missing_interface_method',
                    "Class {$class['name']} is missing interface method {$interfaceName}::{$method['name']}()");
            }
        }
    }

    /**
     * Find interface file by name
     */
    private function findInterfaceFile(string $interfaceName): ?string {
        // Check if it's a fully qualified name
        if (str_contains($interfaceName, '\\')) {
            $parts = explode('\\', $interfaceName);
            $name = end($parts);
        } else {
            $name = $interfaceName;
        }

        // Search in interfaces directory
        $interfacePath = $this->baseDir . '/interfaces/' . $name . '.php';
        if (file_exists($interfacePath)) {
            return $interfacePath;
        }

        // Search in entire directory
        $pattern = $this->baseDir . '/**/' . $name . '.php';
        $files = glob($pattern);

        if (!empty($files)) {
            return $files[0];
        }

        return null;
    }

    /**
     * Get interface methods from interface file
     */
    private function getInterfaceMethods(string $interfaceFile): array {
        $content = file_get_contents($interfaceFile);
        if ($content === false) {
            return [];
        }

        $tokens = @token_get_all($content);
        if (empty($tokens)) {
            return [];
        }

        $structure = $this->extractStructure($tokens, $content);

        foreach ($structure['interfaces'] as $interface) {
            return $interface['methods'];
        }

        return [];
    }

    /**
     * Check for undefined constants
     */
    private function checkUndefinedConstants(string $filepath, string $content): void {
        // Common PHP constants that don't need definition
        $phpConstants = [
            'PHP_VERSION', 'PHP_OS', 'PHP_SAPI',
            'true', 'false', 'null',
            'DIRECTORY_SEPARATOR', 'PATH_SEPARATOR',
            'E_ERROR', 'E_WARNING', 'E_NOTICE', 'E_DEPRECATED',
            'STDIN', 'STDOUT', 'STDERR',
        ];

        // WordPress constants
        $wpConstants = [
            'WP_DEBUG', 'WP_CONTENT_DIR', 'WP_PLUGIN_DIR',
            'ABSPATH', 'WPINC', 'WP_CONTENT_URL',
        ];

        $tokens = @token_get_all($content);

        foreach ($tokens as $token) {
            if (is_array($token) && $token[0] === T_STRING) {
                $name = $token[1];

                // Skip if it's a PHP or WP constant
                if (in_array(strtoupper($name), array_map('strtoupper', $phpConstants)) ||
                    in_array($name, $wpConstants)) {
                    continue;
                }

                // Check if it looks like a constant (all uppercase with underscores)
                if (preg_match('/^[A-Z][A-Z0-9_]+$/', $name) &&
                    !defined($name) &&
                    !in_array($name, $phpConstants) &&
                    !in_array($name, $wpConstants)) {

                    // Only flag if it's not a class name (heuristic)
                    if (!str_ends_with($name, 'Helper') &&
                        !str_ends_with($name, 'Exception')) {
                        $this->addIssue($filepath, 'undefined_constant',
                            "Possible undefined constant: {$name}");
                    }
                }
            }
        }
    }

    /**
     * Validate WordPress function guards
     */
    private function validateWordPressGuards(string $filepath, string $content, ?string $namespace): void {
        // If already in WordPress namespace or has guards, skip
        if ($namespace && (str_contains($namespace, 'WordPress') || str_contains($namespace, 'WP'))) {
            return;
        }

        $tokens = @token_get_all($content);
        $foundWpFunctions = [];

        // First pass: collect WordPress function calls
        for ($i = 0; $i < count($tokens); $i++) {
            if (is_array($tokens[$i]) && $tokens[$i][0] === T_STRING) {
                $functionName = $tokens[$i][1];

                if (in_array($functionName, $this->wordpressFunctions)) {
                    // Check if it's a function call (followed by parentheses)
                    $j = $i + 1;
                    while ($j < count($tokens) && is_array($tokens[$j]) && $tokens[$j][0] === T_WHITESPACE) {
                        $j++;
                    }

                    if ($j < count($tokens) && !is_array($tokens[$j]) && $tokens[$j] === '(') {
                        $foundWpFunctions[] = [
                            'function' => $functionName,
                            'line' => $tokens[$i][2] ?? 0,
                        ];
                    }
                }
            }
        }

        if (empty($foundWpFunctions)) {
            return;
        }

        // Check for function_exists guards
        $hasGuard = false;
        $guardPattern = '/function_exists\s*\(\s*["\'](' . implode('|', $this->wordpressFunctions) . ')["\']\s*\)/i';

        if (preg_match($guardPattern, $content)) {
            $hasGuard = true;
        }

        // Check for WordPress availability checks
        $wpChecks = [
            '/defined\s*\(\s*["\']ABSPATH["\']\s*\)/',
            '/function_exists\s*\(\s*["\']wp_get_current_user["\']\s*\)/',
        ];

        foreach ($wpChecks as $pattern) {
            if (preg_match($pattern, $content)) {
                $hasGuard = true;
                break;
            }
        }

        // If using WordPress functions without guards
        if (!$hasGuard) {
            $functions = array_unique(array_column($foundWpFunctions, 'function'));
            $this->addIssue($filepath, 'missing_wp_guard',
                "WordPress functions used without guard: " . implode(', ', $functions));
        }
    }

    /**
     * Add an issue to the report
     */
    private function addIssue(string $file, string $type, string $message): void {
        if (!isset($this->issues[$file])) {
            $this->issues[$file] = [];
        }

        if (!isset($this->issues[$file][$type])) {
            $this->issues[$file][$type] = [];
        }

        $this->issues[$file][$type][] = $message;
        $this->stats['total_issues']++;
    }

    /**
     * Print header
     */
    private function printHeader(): void {
        $this->output(str_repeat('=', 80), 'header');
        $this->output('DEEP VALIDATION TOOL FOR HELPER CLASSES', 'header');
        $this->output(str_repeat('=', 80), 'header');
        $this->output('');
    }

    /**
     * Print validation report
     */
    private function printReport(): void {
        $this->output(str_repeat('-', 80), 'header');
        $this->output('VALIDATION REPORT', 'header');
        $this->output(str_repeat('-', 80), 'header');
        $this->output('');

        // Print statistics
        $this->output("FILES SCANNED: {$this->stats['files_scanned']}", 'info');
        $this->output("CLASSES FOUND: {$this->stats['classes_found']}", 'info');
        $this->output("INTERFACES FOUND: {$this->stats['interfaces_found']}", 'info');
        $this->output("METHODS FOUND: {$this->stats['methods_found']}", 'info');
        $this->output("TOTAL ISSUES: {$this->stats['total_issues']}", 'error');
        $this->output('');

        // Group issues by category
        $byCategory = [];
        foreach ($this->issues as $file => $fileIssues) {
            foreach ($fileIssues as $type => $messages) {
                if (!isset($byCategory[$type])) {
                    $byCategory[$type] = [];
                }
                $byCategory[$type][$file] = $messages;
            }
        }

        // Print issues by category
        if (!empty($byCategory)) {
            foreach ($byCategory as $type => $files) {
                $categoryName = $this->typeIssueCategories[$type] ?? $type;
                $count = 0;
                foreach ($files as $fileIssues) {
                    $count += count($fileIssues);
                }

                $this->output(str_repeat('-', 80), 'section');
                $this->output("{$categoryName}: {$count} issues", 'section');
                $this->output(str_repeat('-', 80), 'section');

                foreach ($files as $file => $messages) {
                    $this->output("\n{$file}:", 'file');
                    foreach ($messages as $message) {
                        $this->output("  - {$message}", 'issue');
                    }
                }
                $this->output('');
            }
        } else {
            $this->output("No issues found! All files are properly structured.", 'success');
        }

        $this->output(str_repeat('=', 80), 'header');
        $this->output('VALIDATION COMPLETE', 'header');
        $this->output(str_repeat('=', 80), 'header');
    }

    /**
     * Output colored message
     */
    private function output(string $message, string $type = 'info'): void {
        $colors = [
            'header' => "\033[1;36m", // Cyan bold
            'section' => "\033[1;35m", // Magenta bold
            'info' => "\033[0;36m", // Cyan
            'success' => "\033[0;32m", // Green
            'warning' => "\033[0;33m", // Yellow
            'error' => "\033[0;31m", // Red
            'file' => "\033[1;33m", // Yellow bold
            'issue' => "\033[0;37m", // White
        ];

        $reset = "\033[0m";
        $color = $colors[$type] ?? '';

        echo $color . $message . $reset . PHP_EOL;
    }
}

// CLI Entry Point
function main(array $argv): int {
    $baseDir = __DIR__;
    $outputFormat = 'text';
    $outputFile = null;

    // Parse arguments
    for ($i = 1; $i < count($argv); $i++) {
        if ($argv[$i] === '--help' || $argv[$i] === '-h') {
            printHelp();
            return 0;
        } elseif ($argv[$i] === '--format' && isset($argv[$i + 1])) {
            $outputFormat = $argv[++$i];
        } elseif ($argv[$i] === '--output' && isset($argv[$i + 1])) {
            $outputFile = $argv[++$i];
        } elseif ($argv[$i] === '--dir' && isset($argv[$i + 1])) {
            $baseDir = $argv[++$i];
        }
    }

    // Validate format
    if (!in_array($outputFormat, ['text', 'json', 'markdown'])) {
        fprintf(STDERR, "Invalid format: {$outputFormat}\n");
        return 1;
    }

    // Check directory
    if (!is_dir($baseDir)) {
        fprintf(STDERR, "Directory not found: {$baseDir}\n");
        return 1;
    }

    // Run validator
    $validator = new DeepValidator($baseDir);
    $issues = $validator->run();

    // Output results
    if ($outputFormat === 'json') {
        $json = json_encode([
            'stats' => $validator->stats,
            'issues' => $issues,
        ], JSON_PRETTY_PRINT);

        if ($outputFile) {
            file_put_contents($outputFile, $json);
            echo "JSON report saved to: {$outputFile}\n";
        } else {
            echo $json . PHP_EOL;
        }
    } elseif ($outputFormat === 'markdown') {
        $markdown = generateMarkdownReport($validator->stats, $issues);

        if ($outputFile) {
            file_put_contents($outputFile, $markdown);
            echo "Markdown report saved to: {$outputFile}\n";
        } else {
            echo $markdown . PHP_EOL;
        }
    }

    // Return exit code based on issues
    return $validator->stats['total_issues'] > 0 ? 1 : 0;
}

/**
 * Print help message
 */
function printHelp(): void {
    echo <<<HELP
Deep Validation Tool for Helper Classes
========================================

Usage: php deep_validate.php [options]

Options:
  --help, -h          Show this help message
  --dir <path>        Directory to scan (default: current directory)
  --format <type>     Output format: text, json, markdown (default: text)
  --output <file>     Save report to file

Examples:
  php deep_validate.php
  php deep_validate.php --format json --output report.json
  php deep_validate.php --dir ../classes --format markdown --output report.md

HELP;
}

/**
 * Generate markdown report
 */
function generateMarkdownReport(array $stats, array $issues): string {
    $md = "# Deep Validation Report\n\n";

    // Statistics
    $md .= "## Statistics\n\n";
    $md .= "- **Files Scanned:** {$stats['files_scanned']}\n";
    $md .= "- **Classes Found:** {$stats['classes_found']}\n";
    $md .= "- **Interfaces Found:** {$stats['interfaces_found']}\n";
    $md .= "- **Methods Found:** {$stats['methods_found']}\n";
    $md .= "- **Total Issues:** {$stats['total_issues']}\n\n";

    // Issues
    if (!empty($issues)) {
        foreach ($issues as $file => $fileIssues) {
            $md .= "## {$file}\n\n";

            foreach ($fileIssues as $type => $messages) {
                $md .= "### {$type}\n\n";
                foreach ($messages as $message) {
                    $md .= "- {$message}\n";
                }
                $md .= "\n";
            }
        }
    } else {
        $md .= "## No Issues Found\n\n";
        $md .= "All files are properly structured!\n";
    }

    return $md;
}

// Run if called directly
if (isset($argv) && realpath($argv[0]) === realpath(__FILE__)) {
    exit(main($argv));
}
