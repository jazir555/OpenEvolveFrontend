<?php
/**
 * Comprehensive Code Quality Analyzer
 *
 * Analyzes PHP code for:
 * 1. Missing type hints on function parameters
 * 2. Missing return type declarations
 * 3. Inconsistent type usage
 * 4. PHPDoc issues (missing or incorrect)
 * 5. Dead code or unused functions
 * 6. Code duplication
 * 7. Functions that are too long or complex
 * 8. Classes with too many responsibilities (god classes)
 * 9. Missing or incorrect interface implementations
 * 10. Naming convention violations
 */

class CodeQualityAnalyzer {
    private $basePath;
    private $issues = [];
    private $stats = [
        'files_analyzed' => 0,
        'total_functions' => 0,
        'functions_with_return_types' => 0,
        'functions_with_param_types' => 0,
        'functions_with_phpdoc' => 0,
        'total_classes' => 0,
        'total_interfaces' => 0,
    ];

    public function __construct(string $basePath) {
        $this->basePath = rtrim($basePath, '/\\');
    }

    public function analyze(): array {
        $this->scanDirectory($this->basePath);
        $this->generateReport();
        return $this->issues;
    }

    private function scanDirectory(string $path): void {
        $items = scandir($path);
        foreach ($items as $item) {
            if ($item === '.' || $item === '..') {
                continue;
            }

            $fullPath = $path . DIRECTORY_SEPARATOR . $item;

            // Skip vendor and tests directories
            if (strpos($fullPath, DIRECTORY_SEPARATOR . 'vendor') !== false ||
                strpos($fullPath, DIRECTORY_SEPARATOR . 'tests') !== false ||
                strpos($fullPath, DIRECTORY_SEPARATOR . '.git') !== false) {
                continue;
            }

            if (is_dir($fullPath)) {
                $this->scanDirectory($fullPath);
            } elseif (str_ends_with($item, '.php')) {
                $this->analyzeFile($fullPath);
            }
        }
    }

    private function analyzeFile(string $filePath): void {
        $this->stats['files_analyzed']++;

        $content = file_get_contents($filePath);
        if ($content === false) {
            return;
        }

        $relativePath = str_replace($this->basePath . DIRECTORY_SEPARATOR, '', $filePath);

        // Tokenize the file
        $tokens = token_get_all($content);

        $currentNamespace = '';
        $classes = [];
        $interfaces = [];
        $functions = [];
        $currentClass = null;
        $currentInterface = null;

        for ($i = 0; $i < count($tokens); $i++) {
            $token = $tokens[$i];

            if (!is_array($token)) {
                continue;
            }

            switch ($token[0]) {
                case T_NAMESPACE:
                    $currentNamespace = $this->extractNamespace($tokens, $i);
                    break;

                case T_CLASS:
                    if ($currentInterface === null) {
                        $classInfo = $this->extractClassInfo($tokens, $i, $filePath, $currentNamespace);
                        if ($classInfo) {
                            $classes[] = $classInfo;
                            $currentClass = $classInfo['name'];
                            $this->stats['total_classes']++;

                            // Check for god classes
                            $this->checkGodClass($classInfo, $relativePath);
                        }
                    }
                    break;

                case T_INTERFACE:
                    $interfaceInfo = $this->extractInterfaceInfo($tokens, $i, $filePath, $currentNamespace);
                    if ($interfaceInfo) {
                        $interfaces[] = $interfaceInfo;
                        $currentInterface = $interfaceInfo['name'];
                        $this->stats['total_interfaces']++;
                    }
                    break;

                case T_FUNCTION:
                    $funcInfo = $this->extractFunctionInfo($tokens, $i, $filePath, $currentNamespace, $currentClass);
                    if ($funcInfo) {
                        $functions[] = $funcInfo;
                        $this->stats['total_functions']++;

                        // Analyze function
                        $this->checkFunctionTypeHints($funcInfo, $relativePath);
                        $this->checkFunctionPHPDoc($funcInfo, $relativePath);
                        $this->checkFunctionComplexity($funcInfo, $relativePath);
                        $this->checkNamingConventions($funcInfo, $relativePath);
                    }
                    break;
            }
        }

        // Check interface implementations
        $this->checkInterfaceImplementations($classes, $interfaces, $relativePath);

        // Check for code duplication
        $this->checkCodeDuplication($functions, $relativePath);
    }

    private function extractNamespace(array $tokens, int &$index): string {
        $namespace = '';
        $index++;

        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if (is_array($token) && in_array($token[0], [T_STRING, T_NS_SEPARATOR])) {
                $namespace .= $token[1];
            } elseif ($token === ';') {
                break;
            }

            $index++;
        }

        return $namespace;
    }

    private function extractClassInfo(array $tokens, int &$index, string $filePath, string $namespace): ?array {
        $index++;
        $className = '';
        $extends = '';
        $implements = [];
        $methods = [];
        $properties = [];
        $startLine = $tokens[$index - 2][2] ?? 1;

        // Get class name
        while ($index < count($tokens)) {
            $token = $tokens[$index];
            if (is_array($token) && $token[0] === T_STRING) {
                $className = $token[1];
                $index++;
                break;
            }
            $index++;
        }

        // Check for extends and implements
        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if (is_array($token)) {
                if ($token[0] === T_EXTENDS) {
                    $extends = $this->extractExtends($tokens, $index);
                } elseif ($token[0] === T_IMPLEMENTS) {
                    $implements = $this->extractImplements($tokens, $index);
                }
            } elseif ($token === '{') {
                $index++;
                break;
            }

            $index++;
        }

        // Count methods and properties
        $braceCount = 1;
        while ($index < count($tokens) && $braceCount > 0) {
            $token = $tokens[$index];

            if ($token === '{') {
                $braceCount++;
            } elseif ($token === '}') {
                $braceCount--;
            } elseif (is_array($token)) {
                if ($token[0] === T_FUNCTION) {
                    $methods[] = $this->extractFunctionInfo($tokens, $index, $filePath, $namespace, $className);
                } elseif ($token[0] === T_PUBLIC || $token[0] === T_PRIVATE || $token[0] === T_PROTECTED || $token[0] === T_VAR) {
                    // Check if it's a property
                    if ($this->isProperty($tokens, $index)) {
                        $properties[] = $this->extractPropertyInfo($tokens, $index);
                    }
                }
            }

            $index++;
        }

        return [
            'name' => $className,
            'namespace' => $namespace,
            'extends' => $extends,
            'implements' => $implements,
            'methods' => array_filter($methods),
            'properties' => $properties,
            'start_line' => $startLine,
            'file' => $filePath,
        ];
    }

    private function extractInterfaceInfo(array $tokens, int &$index, string $filePath, string $namespace): ?array {
        $index++;
        $interfaceName = '';
        $methods = [];
        $startLine = $tokens[$index - 2][2] ?? 1;

        // Get interface name
        while ($index < count($tokens)) {
            $token = $tokens[$index];
            if (is_array($token) && $token[0] === T_STRING) {
                $interfaceName = $token[1];
                $index++;
                break;
            }
            $index++;
        }

        // Get methods
        $braceCount = 0;
        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if ($token === '{') {
                $braceCount++;
            } elseif ($token === '}') {
                $braceCount--;
                if ($braceCount === 0) {
                    break;
                }
            } elseif (is_array($token) && $token[0] === T_FUNCTION) {
                $methods[] = $this->extractFunctionInfo($tokens, $index, $filePath, $namespace, $interfaceName);
            }

            $index++;
        }

        return [
            'name' => $interfaceName,
            'namespace' => $namespace,
            'methods' => array_filter($methods),
            'start_line' => $startLine,
            'file' => $filePath,
        ];
    }

    private function extractFunctionInfo(array $tokens, int &$index, string $filePath, string $namespace, ?string $class): ?array {
        $index++;
        $functionName = '';
        $params = [];
        $returnType = '';
        $phpdoc = '';
        $startLine = $tokens[$index - 2][2] ?? 1;
        $endLine = $startLine;
        $isStatic = false;
        $visibility = 'public';

        // Check for static and visibility
        for ($j = $index - 5; $j < $index; $j++) {
            if (isset($tokens[$j]) && is_array($tokens[$j])) {
                if ($tokens[$j][0] === T_STATIC) {
                    $isStatic = true;
                }
                if ($tokens[$j][0] === T_PUBLIC) {
                    $visibility = 'public';
                } elseif ($tokens[$j][0] === T_PRIVATE) {
                    $visibility = 'private';
                } elseif ($tokens[$j][0] === T_PROTECTED) {
                    $visibility = 'protected';
                }
            }
        }

        // Check for PHPDoc before function
        for ($j = $index - 10; $j < $index; $j++) {
            if (isset($tokens[$j]) && is_array($tokens[$j]) && $tokens[$j][0] === T_DOC_COMMENT) {
                $phpdoc = $tokens[$j][1];
                break;
            }
        }

        // Skip anonymous functions
        while ($index < count($tokens)) {
            $token = $tokens[$index];
            if (is_array($token) && $token[0] === T_STRING) {
                $functionName = $token[1];
                $index++;
                break;
            } elseif ($token === '(') {
                // Anonymous function
                return null;
            }
            $index++;
        }

        // Get parameters
        if ($index < count($tokens) && $tokens[$index] === '(') {
            $index++;
            $paramInfo = $this->extractParameters($tokens, $index);
            $params = $paramInfo['params'];
            $index = $paramInfo['index'];
        }

        // Get return type
        if ($index < count($tokens) && $tokens[$index] === ':') {
            $index++;
            $returnType = $this->extractReturnType($tokens, $index);
            if ($returnType) {
                $this->stats['functions_with_return_types']++;
            }
        }

        // Find end line
        $braceCount = 0;
        $foundBrace = false;
        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if ($token === '{') {
                $braceCount++;
                $foundBrace = true;
            } elseif ($token === '}') {
                $braceCount--;
                if ($braceCount === 0 && $foundBrace) {
                    $endLine = $token[2] ?? $endLine;
                    break;
                }
            }

            $index++;
        }

        return [
            'name' => $functionName,
            'namespace' => $namespace,
            'class' => $class,
            'params' => $params,
            'return_type' => $returnType,
            'phpdoc' => $phpdoc,
            'start_line' => $startLine,
            'end_line' => $endLine,
            'file' => $filePath,
            'is_static' => $isStatic,
            'visibility' => $visibility,
            'lines_of_code' => $endLine - $startLine + 1,
        ];
    }

    private function extractParameters(array $tokens, int &$index): array {
        $params = [];
        $paramCount = 0;

        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if ($token === ')') {
                $index++;
                break;
            } elseif ($token === ',') {
                $paramCount++;
                $index++;
                continue;
            } elseif (is_array($token)) {
                $paramName = '';
                $paramType = '';

                // Check for type hint
                    $validTokens = [T_STRING, T_NS_SEPARATOR, T_ARRAY, T_CALLABLE];
                    if (defined('T_ITERABLE')) {
                        $validTokens[] = T_ITERABLE;
                    }
                    if (defined('T_NAME_QUALIFIED')) {
                        $validTokens[] = T_NAME_QUALIFIED;
                    }
                    if (defined('T_NAME_FULLY_QUALIFIED')) {
                        $validTokens[] = T_NAME_FULLY_QUALIFIED;
                    }

                    if (in_array($token[0], $validTokens)) {
                    $paramType = $token[1];
                    $index++;

                    // Check for union types (PHP 8.0+)
                    if ($index < count($tokens) && $tokens[$index] === '|') {
                        $unionTokens = [T_STRING, T_NS_SEPARATOR, T_ARRAY, T_CALLABLE];
                        if (defined('T_ITERABLE')) {
                            $unionTokens[] = T_ITERABLE;
                        }

                        while ($index < count($tokens)) {
                            $index++;
                            if (is_array($tokens[$index]) && in_array($tokens[$index][0], $unionTokens)) {
                                $paramType .= '|' . $tokens[$index][1];
                                $index++;
                            } elseif ($tokens[$index] === '$') {
                                break;
                            }
                        }
                    }
                }

                // Get parameter name
                if ($index < count($tokens) && is_array($tokens[$index]) && $tokens[$index][0] === T_VARIABLE) {
                    $paramName = $tokens[$index][1];
                    $index++;
                }

                if ($paramName) {
                    $params[] = [
                        'name' => $paramName,
                        'type' => $paramType,
                        'has_type' => !empty($paramType),
                    ];
                }
            }

            $index++;
        }

        if (!empty($params)) {
            $this->stats['functions_with_param_types'] += count(array_filter($params, fn($p) => $p['has_type']));
        }

        return ['params' => $params, 'index' => $index];
    }

    private function extractReturnType(array $tokens, int &$index): string {
        $returnType = '';

        $validTokens = [T_STRING, T_NS_SEPARATOR, T_ARRAY, T_CALLABLE];
        if (defined('T_ITERABLE')) {
            $validTokens[] = T_ITERABLE;
        }
        if (defined('T_NAME_QUALIFIED')) {
            $validTokens[] = T_NAME_QUALIFIED;
        }
        if (defined('T_NAME_FULLY_QUALIFIED')) {
            $validTokens[] = T_NAME_FULLY_QUALIFIED;
        }

        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if (is_array($token) && in_array($token[0], $validTokens)) {
                $returnType .= $token[1];
            } elseif ($token === '|') {
                $returnType .= '|';
            } elseif ($token === '{' || $token === ';') {
                break;
            }

            $index++;
        }

        return $returnType;
    }

    private function isProperty(array $tokens, int $index): bool {
        // Look ahead to see if there's a variable declaration
        for ($i = $index; $i < min($index + 5, count($tokens)); $i++) {
            if (is_array($tokens[$i]) && $tokens[$i][0] === T_FUNCTION) {
                return false;
            }
            if (is_array($tokens[$i]) && $tokens[$i][0] === T_VARIABLE) {
                return true;
            }
        }
        return false;
    }

    private function extractPropertyInfo(array $tokens, int &$index): array {
        $propertyName = '';
        $propertyType = '';
        $defaultValue = null;

        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if (is_array($token)) {
                if ($token[0] === T_VARIABLE) {
                    $propertyName = $token[1];
                } elseif ($token[0] === T_STRING && empty($propertyName)) {
                    $propertyType = $token[1];
                }
            } elseif ($token === ';') {
                $index++;
                break;
            }

            $index++;
        }

        return [
            'name' => $propertyName,
            'type' => $propertyType,
            'default' => $defaultValue,
        ];
    }

    private function extractExtends(array $tokens, int &$index): string {
        $extends = '';
        $index++;

        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if (is_array($token) && in_array($token[0], [T_STRING, T_NS_SEPARATOR])) {
                $extends .= $token[1];
            } elseif ($token === '{' || $token === T_IMPLEMENTS) {
                break;
            }

            $index++;
        }

        return $extends;
    }

    private function extractImplements(array $tokens, int &$index): array {
        $implements = [];
        $currentInterface = '';

        $index++;

        while ($index < count($tokens)) {
            $token = $tokens[$index];

            if (is_array($token) && in_array($token[0], [T_STRING, T_NS_SEPARATOR])) {
                $currentInterface .= $token[1];
            } elseif ($token === ',') {
                if (!empty($currentInterface)) {
                    $implements[] = $currentInterface;
                    $currentInterface = '';
                }
            } elseif ($token === '{') {
                if (!empty($currentInterface)) {
                    $implements[] = $currentInterface;
                }
                break;
            }

            $index++;
        }

        return $implements;
    }

    private function checkFunctionTypeHints(array $func, string $relativePath): void {
        $location = "{$relativePath}:{$func['start_line']} - {$func['name']}()";

        // Check parameters without type hints
        foreach ($func['params'] as $param) {
            if (!$param['has_type']) {
                $this->addIssue('missing_param_type', $location, "Parameter '{$param['name']}' is missing type hint");
            }
        }

        // Check return type
        if (empty($func['return_type']) && $func['name'] !== '__construct' && $func['name'] !== '__destruct') {
            $this->addIssue('missing_return_type', $location, "Function is missing return type declaration");
        }
    }

    private function checkFunctionPHPDoc(array $func, string $relativePath): void {
        $location = "{$relativePath}:{$func['start_line']} - {$func['name']}()";

        if (empty($func['phpdoc'])) {
            // Public methods should have PHPDoc
            if ($func['visibility'] === 'public' && $func['name'] !== '__construct' && $func['name'] !== '__destruct') {
                $this->addIssue('missing_phpdoc', $location, "Public function is missing PHPDoc comment");
            }
        } else {
            $this->stats['functions_with_phpdoc']++;

            // Check if PHPDoc matches actual signatures
            $phpdocParams = $this->parsePHPDocParams($func['phpdoc']);
            $phpdocReturn = $this->parsePHPDocReturn($func['phpdoc']);

            // Check for mismatched parameters
            $actualParams = array_map(fn($p) => ltrim($p['name'], '$'), $func['params']);
            $phpdocParamNames = array_keys($phpdocParams);

            foreach ($actualParams as $param) {
                if (!in_array($param, $phpdocParamNames)) {
                    $this->addIssue('phpdoc_mismatch', $location, "PHPDoc missing @param tag for parameter '{$param}'");
                }
            }

            // Check for return type mismatch
            if (!empty($func['return_type']) && !empty($phpdocReturn)) {
                $normalizedActual = $this->normalizeType($func['return_type']);
                $normalizedDoc = $this->normalizeType($phpdocReturn);

                if ($normalizedActual !== $normalizedDoc && !$this->isTypeCompatible($normalizedActual, $normalizedDoc)) {
                    $this->addIssue('phpdoc_mismatch', $location, "PHPDoc @return type '{$phpdocReturn}' doesn't match actual return type '{$func['return_type']}'");
                }
            }
        }
    }

    private function parsePHPDocParams(string $phpdoc): array {
        $params = [];
        if (preg_match_all('/@param\s+(\S+)\s+\$(\w+)/', $phpdoc, $matches, PREG_SET_ORDER)) {
            foreach ($matches as $match) {
                $params[$match[2]] = $match[1];
            }
        }
        return $params;
    }

    private function parsePHPDocReturn(string $phpdoc): ?string {
        if (preg_match('/@return\s+(\S+)/', $phpdoc, $match)) {
            return $match[1];
        }
        return null;
    }

    private function normalizeType(string $type): string {
        $type = str_replace(['[]', 'array<', 'mixed'], ['', 'array', 'mixed'], $type);
        return strtolower(trim($type));
    }

    private function isTypeCompatible(string $actual, string $doc): bool {
        // Allow some common type aliases
        $aliases = [
            'boolean' => 'bool',
            'integer' => 'int',
            'double' => 'float',
        ];

        $actual = $aliases[$actual] ?? $actual;
        $doc = $aliases[$doc] ?? $doc;

        return $actual === $doc;
    }

    private function checkFunctionComplexity(array $func, string $relativePath): void {
        $location = "{$relativePath}:{$func['start_line']} - {$func['name']}()";

        // Check lines of code
        if ($func['lines_of_code'] > 50) {
            $this->addIssue('complex_function', $location, "Function is too long ({$func['lines_of_code']} lines), consider breaking it down");
        }

        // Check parameter count
        if (count($func['params']) > 5) {
            $this->addIssue('too_many_params', $location, "Function has too many parameters (" . count($func['params']) . "), consider using an array or object");
        }
    }

    private function checkNamingConventions(array $func, string $relativePath): void {
        $location = "{$relativePath}:{$func['start_line']} - {$func['name']}()";

        // Check for snake_case function names (should be camelCase for methods)
        if (strpos($func['name'], '_') !== false && $func['class'] !== null) {
            $this->addIssue('naming_convention', $location, "Method name should be camelCase, not snake_case");
        }

        // Check for constants (all uppercase) that aren't actually constants
        if (strtoupper($func['name']) === $func['name'] && strtolower($func['name']) !== $func['name']) {
            if (!$func['is_static']) {
                $this->addIssue('naming_convention', $location, "Method name looks like a constant but isn't static");
            }
        }
    }

    private function checkGodClass(array $class, string $relativePath): void {
        $location = "{$relativePath}:{$class['start_line']} - {$class['name']}";

        $methodCount = count($class['methods']);
        $propertyCount = count($class['properties']);

        // God class: too many methods or properties
        if ($methodCount > 20) {
            $this->addIssue('god_class', $location, "Class has too many methods ({$methodCount}), consider splitting into smaller classes");
        }

        if ($propertyCount > 15) {
            $this->addIssue('god_class', $location, "Class has too many properties ({$propertyCount}), consider splitting into smaller classes");
        }

        // Check for multiple responsibilities (different method prefixes)
        $prefixes = [];
        foreach ($class['methods'] as $method) {
            if (preg_match('/^([a-z]+)[A-Z]/', $method['name'], $match)) {
                $prefixes[] = $match[1];
            }
        }

        $uniquePrefixes = array_unique($prefixes);
        if (count($uniquePrefixes) > 5) {
            $this->addIssue('multiple_responsibilities', $location, "Class seems to have multiple responsibilities (detected prefixes: " . implode(', ', $uniquePrefixes) . ")");
        }
    }

    private function checkInterfaceImplementations(array $classes, array $interfaces, string $relativePath): void {
        $interfaceMap = [];
        foreach ($interfaces as $interface) {
            $interfaceMap[$interface['name']] = $interface['methods'];
        }

        foreach ($classes as $class) {
            foreach ($class['implements'] as $implementedInterface) {
                $interfaceName = basename(str_replace('\\', '/', $implementedInterface));

                if (isset($interfaceMap[$interfaceName])) {
                    $interfaceMethods = $interfaceMap[$interfaceName];
                    $classMethodNames = array_map(fn($m) => $m['name'], $class['methods']);

                    foreach ($interfaceMethods as $interfaceMethod) {
                        if ($interfaceMethod && !in_array($interfaceMethod['name'], $classMethodNames)) {
                            $location = "{$relativePath}:{$class['start_line']} - {$class['name']}";
                            $this->addIssue('missing_interface_method', $location, "Class '{$class['name']}' implements '{$implementedInterface}' but is missing method '{$interfaceMethod['name']}'");
                        }
                    }
                }
            }
        }
    }

    private function checkCodeDuplication(array $functions, string $relativePath): void {
        // This is a simplified check - in reality you'd want more sophisticated similarity detection
        $signatures = [];
        foreach ($functions as $func) {
            $signature = md5(serialize($func['params']) . $func['return_type']);
            if (isset($signatures[$signature])) {
                $location = "{$relativePath}:{$func['start_line']} - {$func['name']}()";
                $this->addIssue('potential_duplication', $location, "Function has same signature as '{$signatures[$signature]}', consider if this is duplicated code");
            }
            $signatures[$signature] = $func['name'];
        }
    }

    private function addIssue(string $type, string $location, string $message): void {
        if (!isset($this->issues[$type])) {
            $this->issues[$type] = [];
        }
        $this->issues[$type][] = [
            'location' => $location,
            'message' => $message,
        ];
    }

    private function generateReport(): void {
        echo "\n" . str_repeat('=', 80) . "\n";
        echo "CODE QUALITY ANALYSIS REPORT\n";
        echo str_repeat('=', 80) . "\n\n";

        echo "STATISTICS:\n";
        echo str_repeat('-', 80) . "\n";
        echo "Files analyzed: {$this->stats['files_analyzed']}\n";
        echo "Total classes: {$this->stats['total_classes']}\n";
        echo "Total interfaces: {$this->stats['total_interfaces']}\n";
        echo "Total functions: {$this->stats['total_functions']}\n";
        echo "Functions with return types: {$this->stats['functions_with_return_types']} (" . round($this->stats['functions_with_return_types'] / max($this->stats['total_functions'], 1) * 100, 1) . "%)\n";
        echo "Functions with param types: {$this->stats['functions_with_param_types']} (parameters)\n";
        echo "Functions with PHPDoc: {$this->stats['functions_with_phpdoc']} (" . round($this->stats['functions_with_phpdoc'] / max($this->stats['total_functions'], 1) * 100, 1) . "%)\n";
        echo "\n";

        $issueLabels = [
            'missing_param_type' => 'Missing Parameter Type Hints',
            'missing_return_type' => 'Missing Return Type Declarations',
            'missing_phpdoc' => 'Missing PHPDoc Comments',
            'phpdoc_mismatch' => 'PHPDoc Mismatches',
            'complex_function' => 'Complex Functions (Too Long)',
            'too_many_params' => 'Functions with Too Many Parameters',
            'god_class' => 'God Classes (Too Large)',
            'multiple_responsibilities' => 'Classes with Multiple Responsibilities',
            'missing_interface_method' => 'Missing Interface Methods',
            'naming_convention' => 'Naming Convention Violations',
            'potential_duplication' => 'Potential Code Duplication',
        ];

        foreach ($this->issues as $type => $issues) {
            $label = $issueLabels[$type] ?? $type;
            echo strtoupper($label) . "\n";
            echo str_repeat('=', 80) . "\n";
            echo "Count: " . count($issues) . "\n\n";

            // Show first 20 examples
            $examples = array_slice($issues, 0, 20);
            foreach ($examples as $issue) {
                echo "  - {$issue['location']}\n";
                echo "    {$issue['message']}\n\n";
            }

            if (count($issues) > 20) {
                echo "  ... and " . (count($issues) - 20) . " more\n\n";
            }

            echo "\n";
        }
    }
}

// Run the analyzer
if (php_sapi_name() === 'cli') {
    $basePath = $argv[1] ?? __DIR__;
    $analyzer = new CodeQualityAnalyzer($basePath);
    $analyzer->analyze();
}
