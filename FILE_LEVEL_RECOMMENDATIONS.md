# File-Level Code Quality Recommendations

**Generated:** 2025-01-01
**Total Files Reviewed:** 495 PHP files

This document provides specific recommendations for individual files and directories that need immediate attention.

---

## Priority 1: Critical Files (Immediate Action Required)

### 1. ActionSchedulerHelper.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\ActionSchedulerHelper.php`

**Issues:**
- Missing return types on 6+ methods
- Missing parameter type hints on 10+ parameters
- Snake_case method names (PSR violation)
- PHPDoc mismatches

**Required Actions:**
```php
// BEFORE:
public function get_version() {
    // ...
}

// AFTER:
public function getVersion(): string {
    // ...
}

// BEFORE:
public function get_recent_actions($limit, $status) {
    // ...
}

// AFTER:
public function getRecentActions(int $limit, string $status): array {
    // ...
}
```

**Estimated Effort:** 2-3 hours

---

### 2. ActionSchedulerTaskProcessor.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\ActionSchedulerTaskProcessor.php`

**Issues:**
- Missing @param tags in PHPDoc
- Return type mismatches
- Snake_case method names
- 4+ functions with missing return types

**Required Actions:**
1. Add @param tags for all parameters
2. Fix return type declarations (line 821, 902)
3. Rename methods to camelCase
4. Add parameter type hints

**Estimated Effort:** 3-4 hours

---

### 3. Ajax.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\Ajax.php`

**Issues:**
- Missing return types on multiple methods
- Constructor has 15+ parameters (violates best practices)

**Required Actions:**
```php
// BEFORE:
public function __construct(
    LoggerInterface $logger,
    AssetOrderInterface $assetOrder,
    TaskQueueInterface $taskQueue,
    RetryInterface $retry,
    // ... 11 more parameters
) { }

// AFTER: Use configuration object
public function __construct(AjaxConfiguration $config) {
    $this->logger = $config->logger;
    $this->assetOrder = $config->assetOrder;
    // ...
}

// Or use builder pattern
$ajax = AjaxBuilder::create()
    ->withLogger($logger)
    ->withAssetOrder($assetOrder)
    ->build();
```

**Estimated Effort:** 4-6 hours (requires refactoring calling code)

---

### 4. AssetData.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\AssetData.php`

**Issues:**
- Many stub methods that just delegate to __call()
- Could be simplified with magic method alone

**Required Actions:**
```php
// Current: Stub for every interface method
public function get_memory_usage(bool $real_usage = true): int {
    return $this->__call('get_memory_usage', func_get_args());
}

// Improved: Remove stubs, rely on __call() and interface
// Just implement __call() to handle all methods dynamically
public function __call(string $method, array $args) {
    // Delegate to appropriate helper
    $helper = $this->getHelperForMethod($method);
    return $helper->$method(...$args);
}
```

**Estimated Effort:** 2-3 hours

---

## Priority 2: High-Impact Files

### 5. Admin/AssetStatusPage.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\Admin\AssetStatusPage.php`

**Issues:**
- Line 612: `get_total_count()` has 11 parameters
- Line 1449: `is_asset_download_enabled()` has 9 parameters
- Missing return types
- Missing PHPDoc

**Required Actions:**
```php
// BEFORE:
public function get_total_count(
    $type, $status, $search, $orderby,
    $order, $paged, $per_page,
    $asset_type, $date_from, $date_to, $user_id
) {
    // ...
}

// AFTER: Use query object
public function getTotalCount(AssetStatusQuery $query): int {
    // ...
}

class AssetStatusQuery {
    public function __construct(
        public string $type,
        public string $status,
        public ?string $search = null,
        public ?string $orderby = null,
        public ?string $order = null,
        public int $paged = 1,
        public int $per_page = 20,
        public ?string $asset_type = null,
        public ?\DateTime $date_from = null,
        public ?\DateTime $date_to = null,
        public ?int $user_id = null
    ) {}
}
```

**Estimated Effort:** 6-8 hours (requires creating query objects)

---

### 6. Admin/ExternalAssetsLogPage.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\Admin\ExternalAssetsLogPage.php`

**Issues:**
- Line 505: `truncate_url()` has 18 parameters (EXTREME)
- Line 310: `get_assets()` has 14 parameters
- Line 398: `get_assets_count()` has 12 parameters

**Required Actions:**
1. Create `UrlTruncationConfig` class for truncate_url parameters
2. Create `AssetListQuery` class for get_assets parameters
3. Reduce complexity by splitting into smaller methods

**Estimated Effort:** 8-10 hours

---

### 7. AssetActionHandler.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\AssetActionHandler.php`

**Issues:**
- Line 152: `handle_redirect()` has 7 parameters
- Line 1195: `get_asset_form_data()` has 6 parameters
- Line 1536: `get_asset_processing_status()` has 6 parameters
- Missing return types on multiple methods

**Required Actions:**
1. Create parameter objects for complex methods
2. Add return type declarations
3. Simplify method logic

**Estimated Effort:** 5-7 hours

---

## Priority 3: Helper Classes

### 8. AjaxHelpers/AssetManagementAjaxHelper.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\AjaxHelpers\AssetManagementAjaxHelper.php`

**Issues:**
- Constructor has 15+ dependencies
- Some methods missing return types (line 725, 894)
- Missing PHPDoc on public methods

**Positive:**
- Good use of `declare(strict_types=1);`
- Good type hints on constructor
- Good use of interfaces

**Required Actions:**
1. Refactor constructor to use configuration object
2. Add return types to all methods
3. Add PHPDoc for public methods
4. Reduce number of dependencies

**Estimated Effort:** 4-5 hours

---

### 9. AssetDataHelpers/AssetCacheHelper.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\AssetDataHelpers\AssetCacheHelper.php`

**Issues:**
- All methods are static (prevents dependency injection)
- Direct WordPress function calls (wp_cache_delete, etc.)
- Good type hints on parameters and return types ✓

**Required Actions:**
```php
// BEFORE:
class AssetCacheHelper {
    public static function invalidate_asset_cache(string $url, string $type): void {
        wp_cache_delete($key, 'lha_asset_data');
    }
}

// AFTER:
class AssetCacheHelper {
    private CacheInterface $cache;

    public function __construct(CacheInterface $cache) {
        $this->cache = $cache;
    }

    public function invalidateAssetCache(string $url, string $type): void {
        $this->cache->delete($key, 'lha_asset_data');
    }
}
```

**Estimated Effort:** 3-4 hours (requires updating all call sites)

---

### 10. DatabaseHelpers/DatabaseQueryHelper.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\DatabaseHelpers\DatabaseQueryHelper.php`

**Positive:**
- Good use of `declare(strict_types=1);`
- Good PHPDoc documentation ✓
- Good parameter and return types ✓
- Good use of traits ✓

**Issues:**
- Direct \wpdb dependency (consider abstracting)
- Query statistics tracking could be extracted

**Required Actions:**
1. Consider creating DatabaseConnection interface
2. Extract QueryStatistics class
3. Continue good practices!

**Estimated Effort:** 2-3 hours (optional improvements)

---

### 11. TaskHelpers/TaskValidationHelper.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\TaskHelpers\TaskValidationHelper.php`

**Issues:**
- Uses `$this->logger` but doesn't declare property or inject in constructor
- Uses `$this->urlProcessor` without declaration
- Uses `$this->wpdb` without declaration
- Good type hints ✓
- Good return types ✓

**Required Actions:**
```php
// BEFORE:
class TaskValidationHelper {
    public function is_task_enqueued(string $url, string $type): bool {
        $this->logger->log_warning(...); // Property not declared!
        // ...
    }
}

// AFTER:
class TaskValidationHelper {
    private LoggerInterface $logger;
    private UrlProcessorInterface $urlProcessor;
    private \wpdb $wpdb;

    public function __construct(
        LoggerInterface $logger,
        UrlProcessorInterface $urlProcessor,
        \wpdb $wpdb
    ) {
        $this->logger = $logger;
        $this->urlProcessor = $urlProcessor;
        $this->wpdb = $wpdb;
    }

    public function isTaskEnqueued(string $url, string $type): bool {
        $this->logger->warning(...);
        // ...
    }
}
```

**Estimated Effort:** 1-2 hours

---

## Priority 4: Test Files

### 12. unit/CleanupComprehensiveTest.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\unit\CleanupComprehensiveTest.php`

**Issues:**
- **206 test methods** in one class (MAJOR)
- God class anti-pattern

**Required Actions:**
```php
// Split into multiple test classes:
// - CleanupBasicTest.php (basic cleanup operations)
// - CcleanupBatchTest.php (batch operations)
// - CleanupCacheTest.php (cache-related tests)
// - CleanupDatabaseTest.php (database operations)
// - CleanupEdgeCasesTest.php (edge case scenarios)
// - CleanupSecurityTest.php (security tests)
// - etc.

class CleanupBasicTest extends TestCase {
    // 20-30 related test methods
}

class CleanupBatchTest extends TestCase {
    // 20-30 related test methods
}
```

**Estimated Effort:** 4-6 hours

---

### 13. unit/ExtractTest.php
**Location:** `C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\unit\ExtractTest.php`

**Issues:**
- **203 test methods** in one class (MAJOR)
- Should be split by feature

**Required Actions:**
Split into:
- ExtractCssTest.php
- ExtractJsTest.php
- ExtractSvgTest.php
- ExtractUrlsTest.php
- ExtractValidationTest.php
- ExtractCacheTest.php

**Estimated Effort:** 4-6 hours

---

## Directory-Level Recommendations

### AjaxHelpers/ Directory
**Files:** 10 helper classes + 8 interface files

**Overall Assessment:** GOOD

**Strengths:**
- Clear separation of concerns
- Each helper has a specific domain
- Good use of interfaces
- Most have type hints

**Improvements Needed:**
1. Remove static methods
2. Add dependency injection
3. Add PHPDoc to all public methods
4. Consider consolidating similar helpers

**Estimated Effort:** 10-15 hours for entire directory

---

### AssetDataHelpers/ Directory
**Files:** 12 helper classes + 12 interface files

**Overall Assessment:** FAIR

**Strengths:**
- Good separation of concerns
- Most have good type hints
- Clear naming

**Improvements Needed:**
1. Convert static methods to instance methods (CRITICAL)
2. Abstract WordPress dependencies
3. Add proper constructors
4. Add PHPDoc where missing

**Estimated Effort:** 15-20 hours for entire directory

---

### DatabaseHelpers/ Directory
**Files:** 12 helper classes

**Overall Assessment:** GOOD

**Strengths:**
- Excellent PHPDoc
- Good type hints
- Good use of traits
- Clear responsibilities

**Improvements Needed:**
1. Consider abstracting \wpdb
2. Extract statistics tracking
3. Add interfaces for all helpers

**Estimated Effort:** 8-10 hours for entire directory

---

### TaskHelpers/ Directory
**Files:** 10 helper classes

**Overall Assessment:** NEEDS WORK

**Issues:**
- Some helpers use undeclared properties ($this->logger, etc.)
- Inconsistent constructor declarations
- Good type hints on methods

**Improvements Needed:**
1. Add proper constructors to all helpers
2. Declare all properties
3. Add interfaces
4. Add PHPDoc

**Estimated Effort:** 12-15 hours for entire directory

---

## Specific Code Patterns to Fix

### Pattern 1: Snake_case Method Names

**Files Affected:** 1764 methods across multiple files

**Example Fix:**
```php
// BEFORE:
public function get_admin_url() { }
public function cancel_all_pending() { }
public function has_admin_page() { }

// AFTER:
public function getAdminUrl() { }
public function cancelAllPending() { }
public function hasAdminPage() { }
```

**Batch Rename Strategy:**
1. Use IDE refactoring tools (PHPStorm, VS Code)
2. Create alias methods for backward compatibility
3. Deprecate old names in major version

**Estimated Effort:** 20-30 hours (entire codebase)

---

### Pattern 2: Missing Return Types

**Files Affected:** 3367 functions

**Example Fix:**
```php
// BEFORE:
public function enqueue_admin_assets($plugin_url) {
    // ...
}

// AFTER:
public function enqueue_admin_assets(string $plugin_url): void {
    // ...
}
```

**Batch Fix Strategy:**
1. Start with public APIs
2. Use PHPStan to identify issues
3. Add :void for methods that don't return
4. Add specific types (string, int, array, bool)

**Estimated Effort:** 40-60 hours (entire codebase)

---

### Pattern 3: Functions with Too Many Parameters

**Files Affected:** 439 functions

**Example Fix:**
```php
// BEFORE:
public function render_field(
    $id, $name, $label, $type,
    $value, $placeholder, $required, $disabled
) {
    // ...
}

// AFTER:
public function renderField(FieldConfig $config): string {
    // ...
}

class FieldConfig {
    public function __construct(
        public string $id,
        public string $name,
        public string $label,
        public string $type,
        public mixed $value = null,
        public ?string $placeholder = null,
        public bool $required = false,
        public bool $disabled = false
    ) {}
}
```

**Estimated Effort:** 30-40 hours (most complex refactoring)

---

## Automated Fixes

### Tools to Use

1. **PHP-CS-Fixer** - Can automatically fix:
   - Missing spaces
   - Line endings
   - Some formatting issues
   - Some type hints

2. **Rector** - Can automatically:
   - Add return types
   - Add parameter types
   - Rename methods
   - Upgrade PHP syntax

3. **PHPStan** - Can detect:
   - Missing types
   - Type mismatches
   - Undefined variables
   - Dead code

### Example Rector Configuration

```php
// rector.php
<?php

declare(strict_types=1);

use Rector\Config\RectorConfig;
use Rector\Php81\Rector\FunctionLike\AddNeverReturnTypeRector;
use Rector\TypeDeclaration\Rector\ClassMethod\AddReturnTypeDeclarationRector;

return RectorConfig::configure()
    ->withPaths([
        __DIR__ . '/AjaxHelpers',
        __DIR__ . '/AssetDataHelpers',
        __DIR__ . '/DatabaseHelpers',
        __DIR__ . '/TaskHelpers',
    ])
    ->withRules([
        AddReturnTypeDeclarationRector::class,
    ])
    ->withPhpSets(php81: true);
```

---

## Quick Wins (< 1 hour each)

1. **Add `: void` to all methods that don't return values**
   - Run automated tool
   - Review and commit
   - Time: 30 minutes

2. **Add missing @param tags to PHPDoc**
   - Use PHPDoc template in IDE
   - Fill in parameter descriptions
   - Time: 1-2 hours

3. **Remove unused use statements**
   - Run PHPStan or IDE cleanup
   - Time: 15 minutes

4. **Format code with PHP-CS-Fixer**
   - Run: `php-cs-fixer fix .`
   - Review changes
   - Time: 15 minutes

5. **Add `declare(strict_types=1);` to files missing it**
   - Global find/replace
   - Test for issues
   - Time: 30 minutes

---

## Estimated Total Effort

| Priority | Files | Hours | Impact |
|----------|-------|-------|--------|
| Priority 1 (Critical) | 4 | 11-16 hours | High |
| Priority 2 (High) | 4 | 25-31 hours | High |
| Priority 3 (Helpers) | 4 | 22-28 hours | Medium |
| Priority 4 (Tests) | 2 | 8-12 hours | Medium |
| Quick Wins | - | 2-4 hours | Low |
| **TOTAL** | **14** | **68-91 hours** | - |

**Note:** This is just for the files listed. The entire codebase would require 200-300 hours of work to address all issues comprehensively.

---

## Implementation Strategy

### Week 1-2: Critical Files
- Focus on Priority 1 files
- Add return types and parameter types
- Fix PHPDoc mismatches

### Week 3-4: High-Impact Files
- Refactor functions with too many parameters
- Break down complex methods
- Add comprehensive documentation

### Week 5-6: Helper Classes
- Remove static methods
- Add dependency injection
- Improve interfaces

### Week 7-8: Test Files & Polish
- Split test classes
- Add final documentation
- Run automated tools

### Ongoing: Automation
- Set up PHPStan in CI/CD
- Add pre-commit hooks
- Regular code reviews

---

## Success Metrics

Track progress with these metrics:

1. **Type Coverage:**
   - Week 1: 57.6% → 65%
   - Week 4: 65% → 80%
   - Week 8: 80% → 95%

2. **Documentation Coverage:**
   - Week 1: 35.6% → 50%
   - Week 4: 50% → 70%
   - Week 8: 70% → 85%

3. **Code Quality Score:**
   - Use PHPStan baseline score
   - Goal: Reduce errors from thousands to <100

4. **Technical Debt:**
   - Track god classes (target: 0)
   - Track functions with >5 params (target: <50)

---

*For detailed analysis of all 495 files, see COMPREHENSIVE_CODE_QUALITY_REPORT.md*
