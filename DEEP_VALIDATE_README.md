# Deep Validation Tool for Helper Classes

## Overview

The `deep_validate.php` script is a standalone PHP CLI tool that performs comprehensive validation on all helper files in your codebase. It scans PHP files, extracts class/interface declarations, methods, properties, and constants, then validates them for:

- Missing namespaces
- Missing return types
- Missing parameter type hints
- Undefined constant references
- Missing interface methods
- WordPress function guards

## Features

### 1. **Namespace Validation**
- Checks if all files have proper namespace declarations
- Reports files missing namespace

### 2. **Type Safety Validation**
- Validates all method return types are declared
- Checks all method parameters have type hints
- Reports properties missing type declarations
- Skips magic methods (__construct, __destruct, etc.)

### 3. **Constant Validation**
- Scans for potentially undefined constants
- Flags uppercase identifiers that look like constants
- Excludes known PHP and WordPress constants

### 4. **Interface Implementation Validation**
- Verifies classes implement all required interface methods
- Automatically locates interface files
- Compares method signatures

### 5. **WordPress Compatibility Validation**
- Detects usage of WordPress functions
- Checks for proper `function_exists()` guards
- Ensures WordPress availability checks are in place
- Lists WordPress functions used without protection

## Usage

### Basic Usage

```bash
# Validate all helper files in current directory
php deep_validate.php

# Validate specific directory
php deep_validate.php --dir /path/to/classes

# Show help
php deep_validate.php --help
```

### Output Formats

```bash
# Text output (default, colored)
php deep_validate.php

# JSON output
php deep_validate.php --format json

# JSON output to file
php deep_validate.php --format json --output report.json

# Markdown report
php deep_validate.php --format markdown --output report.md
```

### Examples

```bash
# Generate a markdown report
php deep_validate.php --format markdown --output validation_report.md

# Generate JSON for automated processing
php deep_validate.php --format json --output validation_results.json

# Validate a different directory
php deep_validate.php --dir ../wp-content/plugins/my-plugin/classes
```

## Output Categories

The tool organizes issues into the following categories:

### 1. **Missing Namespace**
File is missing namespace declaration. This is critical for autoloading and preventing naming conflicts.

### 2. **Missing Return Type**
Method is missing return type declaration. Return types improve type safety and code documentation.

### 3. **Missing Parameter Type**
Method parameter is missing type hint. Type hints ensure proper input validation.

### 4. **Undefined Constant**
Potentially undefined constant reference. The tool flags uppercase identifiers that may be constants.

### 5. **Missing Interface Method**
Class doesn't implement all required interface methods. This causes fatal errors.

### 6. **Missing WordPress Function Guard**
WordPress functions used without `function_exists()` checks. This causes errors in non-WP environments.

### 7. **Property Missing Type**
Class property is missing type declaration. Typed properties improve code clarity and safety.

## Understanding the Report

### Text Format (Default)

```
================================================================================
DEEP VALIDATION TOOL FOR HELPER CLASSES
================================================================================

Found 147 helper files to validate

--------------------------------------------------------------------------------
VALIDATION REPORT
--------------------------------------------------------------------------------

FILES SCANNED: 147
CLASSES FOUND: 142
INTERFACES FOUND: 5
METHODS FOUND: 854
TOTAL ISSUES: 1,247

--------------------------------------------------------------------------------
Missing Namespace: 3 issues
--------------------------------------------------------------------------------

TaskHelpers/TaskValidationHelper.php
  - File is missing namespace declaration

--------------------------------------------------------------------------------
Missing Return Type: 456 issues
--------------------------------------------------------------------------------

DatabaseHelpers/DatabaseCacheHelper.php
  - Method DatabaseCacheHelper::get() is missing return type declaration
  - Method DatabaseCacheHelper::set() is missing return type declaration
```

### JSON Format

```json
{
  "stats": {
    "files_scanned": 147,
    "classes_found": 142,
    "interfaces_found": 5,
    "methods_found": 854,
    "total_issues": 1247
  },
  "issues": {
    "TaskHelpers/TaskValidationHelper.php": {
      "missing_namespace": [
        "File is missing namespace declaration"
      ],
      "missing_return_type": [
        "Method TaskValidationHelper::is_task_enqueued() is missing return type declaration"
      ]
    }
  }
}
```

### Markdown Format

```markdown
# Deep Validation Report

## Statistics

- **Files Scanned:** 147
- **Classes Found:** 142
- **Interfaces Found:** 5
- **Methods Found:** 854
- **Total Issues:** 1,247

## TaskHelpers/TaskValidationHelper.php

### missing_namespace

- File is missing namespace declaration

### missing_return_type

- Method TaskValidationHelper::is_task_enqueued() is missing return type declaration
```

## WordPress Function Detection

The tool includes a comprehensive list of WordPress functions and automatically detects their usage:

**Cache Functions:**
- `wp_cache_get()`, `wp_cache_set()`, `wp_cache_delete()`

**Database Functions:**
- `get_option()`, `update_option()`, `delete_option()`
- `esc_sql()`, `wpdb` usage

**Utility Functions:**
- `esc_html()`, `esc_attr()`, `esc_url()`
- `sanitize_text_field()`, `sanitize_key()`
- `__()`, `_e()`, `_x()`, `_n()` (translation)

**Hook Functions:**
- `add_action()`, `add_filter()`, `do_action()`, `apply_filters()`

**Conditional Functions:**
- `is_admin()`, `is_multisite()`, `is_user_logged_in()`
- `current_user_can()`, `user_can()`

If these functions are used without proper guards, the tool reports:
```
missing_wp_guard: WordPress functions used without guard: wp_cache_get, get_option
```

## How It Works

### 1. File Discovery

The tool recursively scans the target directory for files matching:
- Files ending in `Helper.php`
- Files containing "Helper" in the name

### 2. Token Analysis

Uses PHP's `token_get_all()` to parse source code without execution. This allows:
- Safe analysis of any PHP code
- Extraction of structure without dependencies
- No risk of executing code

### 3. Structure Extraction

Extracts from each file:
- Namespace declaration
- Use statements
- Class/interface declarations
- Method signatures (visibility, static, parameters, return type)
- Property declarations
- Class constants
- Extended classes and implemented interfaces

### 4. Validation Rules

**Namespace Rule:**
```php
// BAD - No namespace
class MyHelper {}

// GOOD - Has namespace
namespace LHA\Helpers;
class MyHelper {}
```

**Return Type Rule:**
```php
// BAD - No return type
public function processData($data) {
    return $result;
}

// GOOD - Has return type
public function processData(array $data): array {
    return $result;
}
```

**Parameter Type Rule:**
```php
// BAD - No parameter type
public function save($data, $id) {
    // ...
}

// GOOD - Has parameter types
public function save(array $data, int $id): bool {
    // ...
}
```

**Interface Implementation Rule:**
```php
// Interface
interface CacheInterface {
    public function get(string $key): ?string;
    public function set(string $key, string $value): bool;
}

// BAD - Missing get() method
class MyCache implements CacheInterface {
    public function set(string $key, string $value): bool {
        // ...
    }
}

// GOOD - All methods implemented
class MyCache implements CacheInterface {
    public function get(string $key): ?string {
        // ...
    }
    public function set(string $key, string $value): bool {
        // ...
    }
}
```

**WordPress Function Guard Rule:**
```php
// BAD - No guard
function my_function() {
    $value = get_option('my_option');
}

// GOOD - Has guard
function my_function() {
    if (function_exists('get_option')) {
        $value = get_option('my_option');
    }
}

// GOOD - Has WP check
function my_function() {
    if (defined('ABSPATH')) {
        $value = get_option('my_option');
    }
}
```

## Exit Codes

- **0**: No issues found
- **1**: Issues found or error occurred

This allows use in CI/CD pipelines:
```bash
php deep_validate.php
if [ $? -ne 0 ]; then
    echo "Validation failed!"
    exit 1
fi
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: PHP Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup PHP
        uses: shivammathur/setup-php@v2
        with:
          php-version: '8.1'

      - name: Run Deep Validation
        run: |
          php classes/deep_validate.php --format json --output validation-report.json

      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: validation-report
          path: validation-report.json

      - name: Fail on Issues
        run: |
          if php classes/deep_validate.php; then
            echo "Validation passed!"
          else
            echo "Validation failed!"
            exit 1
          fi
```

### GitLab CI Example

```yaml
validate:
  stage: test
  image: php:8.1
  script:
    - php classes/deep_validate.php --format markdown --output validation-report.md
    - |
      if php classes/deep_validate.php; then
        echo "Validation passed!"
      else
        echo "Validation failed!"
        exit 1
      fi
  artifacts:
    paths:
      - validation-report.md
```

## Performance Considerations

- **Scanning**: Uses `RecursiveDirectoryIterator` for efficient file discovery
- **Parsing**: Uses PHP's native tokenizer for fast parsing
- **Memory**: Processes one file at a time to minimize memory usage
- **Speed**: Can validate 147 files in approximately 30-60 seconds

For large codebases, consider:
- Scanning subdirectories separately
- Using JSON output for post-processing
- Running in parallel on multiple directories

## Customization

### Adding WordPress Functions

Edit the `$wordpressFunctions` array in the `DeepValidator` class:

```php
private array $wordpressFunctions = [
    // ... existing functions ...
    'your_custom_function',
];
```

### Adjusting Issue Categories

Modify the `$typeIssueCategories` array to change category names:

```php
private array $typeIssueCategories = [
    'missing_namespace' => 'Your Custom Name',
    // ...
];
```

### Extending Validation Rules

Add new validation methods following the pattern:

```php
private function validateCustomRule(string $filepath, string $content): void {
    // Your validation logic
    if ($issue) {
        $this->addIssue($filepath, 'custom_issue', 'Issue description');
    }
}
```

## Troubleshooting

### "No helper files found"

- Ensure the `--dir` parameter points to the correct directory
- Check that files end in `Helper.php` or contain "Helper" in the name
- Verify directory read permissions

### "Could not parse PHP file"

- File may have syntax errors
- Check for short open tags (use `<?php` instead of `<?`)
- Verify file encoding is UTF-8

### "Out of memory"

For very large codebases:
```bash
# Increase PHP memory limit
php -d memory_limit=512M deep_validate.php
```

### False Positives

The tool may report:
- Constants that are defined elsewhere (external dependencies)
- WordPress functions that are intentionally used without guards in WP-specific files
- Magic methods (intentionally skipped, but verify __serialize, __unserialize are handled)

## Version History

**v1.0.0** (2025-12-30)
- Initial release
- Support for namespace, type, constant, interface, and WordPress validation
- Text, JSON, and Markdown output formats
- CLI interface with options

## Contributing

To extend the tool:

1. Add new validation methods to the `DeepValidator` class
2. Update the `$typeIssueCategories` array
3. Add corresponding tests
4. Update this documentation

## License

This tool is part of the Locally Host Assets plugin and follows the same license.

## Support

For issues, questions, or contributions, please refer to the main project repository.
